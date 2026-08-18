import re
import torch
import torch.nn.functional as F
import numpy as np
import logging


def evaluation(
    encoder, decoder, channel, bit_map,
    batch_size,
    n_batches,
    hard_factor=1.0
):
    with torch.no_grad():
        total_ber = 0.0
        total_ser = 0.0
        total_gmi = 0.0

        for b_idx in range(n_batches):
            symbols, idxs = encoder(batch_size) # Override the training batch size
            rx = channel(symbols)
            logits = hard_factor * decoder(rx)

            symbol_probabilities = encoder.symbol_probabilities
            term1 = torch.sum(bit_map[idxs] * logits.permute(0, 2, 1), dim=-1, keepdim=True).permute(0, 2, 1)
            term2 = torch.matmul(bit_map, logits)
            metric = term1 - term2
                
            gmi = -torch.mean(torch.logsumexp(
                metric + torch.log(symbol_probabilities[None, :, None]), 
                dim=1, 
            )) / np.log(2)

            decision_metric = term2
            total_ber += torch.mean(((logits < 0) != bit_map[idxs, :].permute(0, 2, 1)).float())
            total_ser += torch.mean((torch.argmin(decision_metric, dim=1) != idxs).float())
            total_gmi += gmi

        return total_ber.item() / n_batches, total_ser.item() / n_batches, total_gmi.item() / n_batches


def pgcs_1(
        encoder, decoder, bit_map,
        encoder_grace,
        channel,
        reinforce_memory_length,
        proximal_lambda,
        n_epochs,
        bit_wise,
        hard_factor=1.0,
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
    avg_ber = 0.0
    avg_ser = 0.0
    avg_gmi = 0.0
    for e_idx in range(n_epochs):
        decoder_opt.zero_grad()
        encoder_opt.zero_grad()

        if e_idx < encoder_grace:
            with torch.no_grad():
                symbols, idxs = encoder()
        else:
            symbols, idxs = encoder()

        rx = channel(symbols)
        logits = hard_factor * decoder(rx)

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
        
        if reinforce_memory_length > 0 and symbol_probabilities.requires_grad:
            correction = torch.sum(
                F.pad(
                    torch.log(symbol_probabilities[idxs]),
                    (reinforce_memory_length // 2, reinforce_memory_length // 2), mode='circular'
                ).unfold(dimension=1, size=reinforce_memory_length, step=1),
                dim=2
            )
            
            loss_reinforce = torch.mean((gmi - gmi.mean()).detach() * correction)

        else:
            loss_reinforce = 0
        
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
                current_ber = torch.mean(((logits < 0) != bit_map[idxs, :].permute(0, 2, 1)).float())
                current_ser = torch.mean((torch.argmin(decision_metric, dim=1) != idxs).float())
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
        decoding_order=(0, 1, 2),
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
    avg_ber = 0.0
    avg_ser = 0.0
    avg_gmi = 0.0
    avg_gmi_bits = [0.0 for _ in range(bit_map.shape[1])]
    for e_idx in range(n_epochs):
        decoder_opt.zero_grad()
        encoder_opt.zero_grad()

        if e_idx < encoder_grace:
            tau = tau_start
        else:
            tau = max(tau_min, tau_start * np.exp(-tau_decay * (e_idx - encoder_grace)))

        symbols, idxs = encoder()
        true_B = bit_map[idxs].permute(0, 2, 1)
        true_B = true_B[:, decoding_order, :]

        # 2. Transmit through channel
        rx = channel(symbols)

        # 3. Decoder forward pass
        logits, expanded_llr_list, one_hot_list = decoder(rx, true_B=true_B, use_true_B=use_teacher_forcing, tau=tau)
        B, _, N_sym = logits.shape
        symbol_probabilities = encoder.symbol_probabilities

        # Build prob_grid (Q_0, ..., Q_K, B, N_sym)
        # We permute each one-hot partition from (B, Q, N) -> (Q, B, N)
        one_hot_list_flat = [p.permute(1, 0, 2) for l in one_hot_list for p in l]
        prob_grid = one_hot_list_flat[0]
        for p in one_hot_list_flat[1:]:
            prob_grid = torch.einsum('...bs, qbs -> q...bs', prob_grid, p)

        term1_list = [true_B_slice * expanded_llr for true_B_slice, expanded_llr in zip(torch.unbind(true_B, dim=1), expanded_llr_list)]

        term1 = sum(term1_list)
        term2 = sum(bit_map_slice * expanded_llr[..., None] for bit_map_slice, expanded_llr in zip(torch.unbind(bit_map, dim=1), expanded_llr_list)) # Sums out 'm'
        metric = term1[..., None] - term2

        # GMI expectation 
        log_sym_probs = torch.log(symbol_probabilities + 1e-12)
        gmi_grid = torch.logsumexp(metric + log_sym_probs, dim=-1) / np.log(2)
        
        # Multiply by prob_grid and sum out all 'Q' dimensions (dim 0 to K-1)
        gmi_per_symbol = torch.sum((prob_grid * gmi_grid).flatten(0, -3), dim=0) # Leaves (B, N_sym)
        gmi_loss = gmi_per_symbol.mean()

        # 5. REINFORCE correction for encoder symbol probabilities
        if reinforce_memory_length > 0 and symbol_probabilities.requires_grad:
            correction = torch.sum(
                F.pad(
                    torch.log(symbol_probabilities[idxs] + 1e-12),
                    (reinforce_memory_length // 2, reinforce_memory_length // 2), mode='circular'
                ).unfold(dimension=1, size=reinforce_memory_length, step=1),
                dim=2
            )
            
            # Detach baseline subtraction using the exact per-symbol GMI tensor
            loss_reinforce = torch.mean((gmi_per_symbol.detach() - gmi_per_symbol.detach().mean()) * correction)

        else:
            loss_reinforce = 0

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
                term2_bit = [torch.tensor([0.0, 1.0]) * expanded_llr[..., None] for expanded_llr in expanded_llr_list]
                gmi_bits = []
                for i, term1 in enumerate(term1_list):
                    metric = term1[..., None] - term2_bit[i]
                    zero_prob = torch.sum(symbol_probabilities[bit_map[:, i] == 0])
                    bit_probabilities = torch.tensor([zero_prob, 1 - zero_prob], device=metric.device)
                    log_bit_probs = torch.log(bit_probabilities + 1e-12)
                    gmi_grid = torch.logsumexp(metric + log_bit_probs, dim=-1) / np.log(2)
                    gmi_per_symbol = torch.sum((prob_grid * gmi_grid).flatten(0, -3), dim=0) # Leaves (B, N_sym)
                    gmi_bits.append(gmi_per_symbol.mean())

                for i, gmi_bit in enumerate(gmi_bits):
                    avg_gmi_bits[i] = beta_mean * avg_gmi_bits[i] + (1 - beta_mean) * gmi_bit

                decision_metric = torch.matmul(bit_map, logits)
                current_ber = torch.mean(((logits < 0) != true_B).float())
                current_ser = torch.mean((torch.argmin(decision_metric, dim=1) != idxs).float())
                avg_ber = beta_mean * avg_ber + (1 - beta_mean) * current_ber
                avg_ser = beta_mean * avg_ser + (1 - beta_mean) * current_ser
                avg_gmi = beta_mean * avg_gmi + (1 - beta_mean) * gmi_loss
                if e_idx % n_logging == 0:
                    logging.info(f"Epoch: {e_idx}")
                    logging.info(f"SER estimate: {avg_ser.item()}")
                    logging.info(f"BER estimate: {avg_ber.item()}")
                    logging.info(f"GMI estimate: {avg_gmi.item()}")
                    for i, gmi_bit in enumerate(avg_gmi_bits):
                        logging.info(f"GMI bit {i}: {gmi_bit.item()}")

    return avg_ber.item(), avg_ser.item(), avg_gmi.item()
