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
                current_ber = torch.mean((bit_map[torch.argmax(metric, dim=1), :] != bit_map[idxs, :]).float())
                current_ser = torch.mean((torch.argmax(metric, dim=1) != idxs).float())
                avg_ber = beta_mean * avg_ber + (1 - beta_mean) * current_ber
                avg_ser = beta_mean * avg_ser + (1 - beta_mean) * current_ser
                avg_gmi = beta_mean * avg_gmi + (1 - beta_mean) * gmi_loss
                if e_idx % n_logging == 0:
                    logging.info(f"Epoch: {e_idx}")
                    logging.info(f"SER estimate: {avg_ser.item()}")
                    logging.info(f"BER estimate: {avg_ber.item()}")
                    logging.info(f"GMI estimate: {avg_gmi.item()}")

    return avg_ber.item(), avg_ser.item(), avg_gmi.item()
