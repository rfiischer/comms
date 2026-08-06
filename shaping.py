import re
import torch
import torch.nn.functional as F
import numpy as np
import logging


def pgcs_1(
        encoder, decoder, bit_map,
        encoder_grace,
        channel,
        reinforce_memory_length,
        proximal_lambda,
        n_epochs,
        bit_wise,
        n_mean=50, n_logging=500, beta_mean=0.9
    ):

    # Initialize optimizers
    encoder_opt = torch.optim.Adam(encoder.parameters())
    decoder_opt = torch.optim.Adam(decoder.parameters())
    
    # Get optimizer parameters for proximal gradient descent
    lr = decoder_opt.param_groups[0]['lr']
    eps = decoder_opt.param_groups[0]['eps']
    beta_2 = decoder_opt.param_groups[0]['betas'][1]

    # Main training loop
    avg_ber = torch.tensor(0.0)
    avg_ser = torch.tensor(0.0)
    avg_gmi = torch.tensor(0.0)
    for e_idx in range(n_epochs):
        decoder_opt.zero_grad()
        encoder_opt.zero_grad()

        if e_idx < encoder_grace:
            with torch.no_grad():
                symbols, idxs = encoder()
        else:
            symbols, idxs = encoder()

        rx = channel(symbols)
        logits = decoder(rx)

        symbol_probabilities = encoder.symbol_probabilities
        if bit_wise:
            term1 = torch.sum(bit_map[idxs] * logits.permute(0, 2, 1), dim=-1, keepdim=True).permute(0, 2, 1)
            term2 = torch.matmul(bit_map, logits)
            metric = term1 - term2
            
        else:
            metric = logits - torch.gather(logits, 1, idxs[:, None, :])
            
        gmi = torch.logsumexp(
            metric + torch.log(symbol_probabilities[None, :, None]), 
            dim=1, 
        ) / np.log(2)
        gmi_loss = torch.mean(gmi)
        
        correction = torch.sum(
            F.pad(
                torch.log(symbol_probabilities[idxs]),
                (reinforce_memory_length // 2, reinforce_memory_length // 2), mode='circular'
            ).unfold(dimension=1, size=reinforce_memory_length, step=1),
            dim=2
        )
        
        loss_reinforce = torch.mean((gmi - gmi.mean()).detach() * correction)
        
        total_loss = gmi_loss + loss_reinforce
        total_loss.backward()
            
        decoder_opt.step()
        encoder_opt.step()

        # Proximal gradient descent
        if proximal_lambda > 0:
            with torch.no_grad():
                for name, param in decoder.named_parameters():
                    if re.match(r'.*weight.*', name):
                        state = decoder_opt.state[param]
                        v_t = state['exp_avg_sq'] / (1 - beta_2 ** (e_idx + 1))
                        step_size = lr / (torch.sqrt(v_t) + eps)
                    
                        # Apply Soft-thresholding
                        param.copy_(torch.sign(param) * torch.relu(torch.abs(param) - step_size * proximal_lambda))

        # Training metrics
        if e_idx % n_mean == 0:
            with torch.no_grad():
                decision_metric = term2
                current_ber = torch.mean(((logits < 0) != bit_map[idxs, :]).float())
                current_ser = torch.mean((torch.argmax(decision_metric, dim=1) != idxs).float())
                avg_ber = beta_mean * avg_ber + (1 - beta_mean) * current_ber
                avg_ser = beta_mean * avg_ser + (1 - beta_mean) * current_ser
                avg_gmi = beta_mean * avg_gmi + (1 - beta_mean) * gmi_loss
                if e_idx % n_logging == 0:
                    logging.info(f"Epoch: {e_idx}")
                    logging.info(f"SER estimate: {avg_ser.item()}")
                    logging.info(f"BER estimate: {avg_ber.item()}")
                    logging.info(f"GMI estimate: {avg_gmi.item()}")

    return avg_ber.item(), avg_ser.item(), avg_gmi.item()


def pgcs_2(
        encoder, decoder, bit_map,
        encoder_grace,
        channel,
        reinforce_memory_length,
        proximal_lambda,
        n_epochs,
        bit_wise,
        use_teacher_forcing=True,
        tau_start=1.0, tau_min=0.1, tau_decay=0.01,
        n_mean=50, n_logging=500, beta_mean=0.9
    ):

    # Initialize optimizers
    encoder_opt = torch.optim.Adam(encoder.parameters())
    decoder_opt = torch.optim.Adam(decoder.parameters())
    
    # Get optimizer parameters for proximal gradient descent
    lr = decoder_opt.param_groups[0]['lr']
    eps = decoder_opt.param_groups[0]['eps']
    beta_2 = decoder_opt.param_groups[0]['betas'][1]

    # Main training loop
    avg_ber = torch.tensor(0.0)
    avg_ser = torch.tensor(0.0)
    avg_gmi = torch.tensor(0.0)
    for e_idx in range(n_epochs):
        decoder_opt.zero_grad()
        encoder_opt.zero_grad()

        if e_idx < encoder_grace:
            tau = tau_start
        else:
            tau = max(tau_min, tau_start * np.exp(-tau_decay * (e_idx - encoder_grace)))

        symbols, idxs = encoder()

        # 1. Map integer symbol indices to ground-truth binary bit sequences
        # bit_map shape: (2^m, m) | idxs shape: (B, N_symbols) -> true_B: (B, m, N_symbols)
        true_B = bit_map[idxs].permute(0, 2, 1).float()
        # TODO: this internally

        # 2. Transmit through channel
        rx = channel(symbols)

        # 3. Non-Linear Equalizer forward pass
        # Returns LLRs of shape (B, m, N_symbols) and CNN partition logits
        # In NonLinearEqualizer.forward(), return (torch.stack(llr_list, dim=1), cnn_logits_list)
        LLRs, expanded_llr_list, one_hot_list = decoder(
            rx, 
            true_B=true_B, 
            use_true_B=use_teacher_forcing,  # True for teacher forcing, False for scheduled autoregression[cite: 1]
            tau=tau  # Softmax relaxation temperature[cite: 1]
        )

        one_hot_list = [t for l in one_hot_list for t in l]
        prob_grid = one_hot_list[0]
        for p in one_hot_list[1:]:
            prob_grid = torch.einsum('b...l, bql -> b...ql', prob_grid, p)

        # 4. GMI Grid Broadcasting and Loss Calculation
        B, N_sym = LLRs.shape[0], LLRs.shape[-1]
        # a. Determine the full grid dimensions from the decoder
        grid_dims = []
        for N_i, Q_i in zip(decoder.N_partitions, decoder.Q_partitions):
            grid_dims.extend([Q_i] * N_i)
        
        K = len(grid_dims) # Total number of partitions
        target_shape = [B] + grid_dims + [N_sym]
        
        # b. Reshape and broadcast each level's LLRs to match the full grid
        expanded_grid_llrs = []
        start_idx = 0
        for i in range(decoder.m):
            N_i = decoder.N_partitions[i]
            curr_llr = expanded_llr_list[i]
            
            if i == 0:
                # Level 0 shape: (Q_0, ..., Q_0) -> N_0 dims. Doesn't have B or N_sym.
                view_shape = [1] + grid_dims[:N_i] + [1] * (K - N_i) + [1]
            else:
                # Level i > 0 shape: (B, Q_i, ..., Q_i, N_sym) -> 2 + N_i dims.
                view_shape = [B] + [1] * start_idx + grid_dims[start_idx:start_idx+N_i] + [1] * (K - start_idx - N_i) + [N_sym]
                
            # View to insert dummy dimensions, then expand to the full target shape
            expanded_grid_llrs.append(curr_llr.view(view_shape).expand(target_shape))
            start_idx += N_i
            
        # Stack along the bit dimension 'm'
        # grid_logits shape: (B, m, q_1, q_2, ..., q_K, N_sym)
        grid_logits = torch.stack(expanded_grid_llrs, dim=1)

        symbol_probabilities = encoder.symbol_probabilities
        true_B_grid = true_B.view(B, decoder.m, *([1]*K), N_sym)

        # c. Compute the metric over the entire grid
        if bit_wise:
            # term1: (B, q_1, ..., q_K, N_sym)
            term1 = torch.sum(true_B_grid * grid_logits, dim=1)
            
            # term2: Use einsum to multiply bit_map (c, m) with grid_logits (B, m, ..., N_sym)
            # Result: (B, 2^m, q_1, ..., q_K, N_sym)
            term2 = torch.einsum('cm, bm...s -> bc...s', bit_map.float(), grid_logits)
            
            metric = term1.unsqueeze(1) - term2
        else:
            # Symbol logits over the grid: (B, 2^m, q_1, ..., q_K, N_sym)
            symbol_logits = torch.einsum('cm, bm...s -> bc...s', bit_map.float(), grid_logits)
            
            idxs_grid = idxs.view(B, 1, *([1]*K), N_sym).expand(B, 1, *grid_dims, N_sym)
            true_symbol_logits = torch.gather(symbol_logits, 1, idxs_grid)
            metric = symbol_logits - true_symbol_logits

        # d. Calculate GMI and expectation
        log_sym_probs = torch.log(symbol_probabilities + 1e-12).view(1, -1, *([1]*K), 1)
        
        # GMI for every possible path in the grid: (B, q_1, ..., q_K, N_sym)
        gmi_grid = -torch.logsumexp(metric + log_sym_probs, dim=1) / np.log(2)
        
        # Expected GMI per symbol (weighting by prob_grid and summing over all Q dimensions)
        dim_to_sum = list(range(1, K + 1))
        gmi_per_symbol = torch.sum(prob_grid * gmi_grid, dim=dim_to_sum) # Shape: (B, N_sym)
        
        gmi_loss = gmi_per_symbol.mean()

        # 5. REINFORCE correction for encoder symbol probabilities
        correction = torch.sum(
            F.pad(
                torch.log(symbol_probabilities[idxs] + 1e-12),
                (reinforce_memory_length // 2, reinforce_memory_length // 2), mode='circular'
            ).unfold(dimension=1, size=reinforce_memory_length, step=1),
            dim=2
        )
        
        # Detach baseline subtraction using the exact per-symbol GMI tensor
        loss_reinforce = torch.mean((gmi_per_symbol.detach() - gmi_per_symbol.detach().mean()) * correction)

        # 6. Backpropagation and optimization step
        total_loss = gmi_loss + loss_reinforce
        # ... [remaining backward pass code] ...
        total_loss.backward()

        decoder_opt.step()
        if e_idx >= encoder_grace:
            encoder_opt.step()

        # Proximal gradient descent
        if proximal_lambda > 0:
            with torch.no_grad():
                for name, param in decoder.named_parameters():
                    if re.match(r'.*weight.*', name):
                        state = decoder_opt.state[param]
                        v_t = state['exp_avg_sq'] / (1 - beta_2 ** (e_idx + 1))
                        step_size = lr / (torch.sqrt(v_t) + eps)
                    
                        # Apply Soft-thresholding
                        param.copy_(torch.sign(param) * torch.relu(torch.abs(param) - step_size * proximal_lambda))

        # Training metrics
        if e_idx % n_mean == 0:
            with torch.no_grad():
                decision_metric = torch.matmul(bit_map, LLRs)
                current_ber = torch.mean(((LLRs > 0) != bit_map[idxs, :].permute(0, 2, 1)).float())
                current_ser = torch.mean((torch.argmax(decision_metric, dim=1) != idxs).float())
                avg_ber = beta_mean * avg_ber + (1 - beta_mean) * current_ber
                avg_ser = beta_mean * avg_ser + (1 - beta_mean) * current_ser
                avg_gmi = beta_mean * avg_gmi + (1 - beta_mean) * gmi_loss
                if e_idx % n_logging == 0:
                    logging.info(f"Epoch: {e_idx}")
                    logging.info(f"SER estimate: {avg_ser.item()}")
                    logging.info(f"BER estimate: {avg_ber.item()}")
                    logging.info(f"GMI estimate: {avg_gmi.item()}")

    return avg_ber.item(), avg_ser.item(), avg_gmi.item()
