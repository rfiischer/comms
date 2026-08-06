import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

import torch_utils as tu


def get_cost(module):
    if hasattr(module, 'get_cost'):
        return module.get_cost()
    
    else:
        return 0


class SequentialConv(tu.SequentialConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_cost(self):
        cost = 0
        for layer in self.children():
            own_cost = get_cost(layer)
            cost = tu.get_stride(layer)[0] * cost + own_cost

        return cost


class ParallelConv(tu.ParallelConv):
    def __init__(self, *args, normalize_cost=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_cost = normalize_cost

    def get_cost(self):
        cost = 0
        for layer in self.children():
            cost += get_cost(layer)    
                
        if self.normalize_cost:
            return cost / len(self)
        
        else:
            return cost


class SurrogateConv(tu.SurrogateConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_cost(self):
        return get_cost(self.main)


class ConvVolterra(tu.ConvVolterra):
    def get_cost(self):
        return self.num_features * (self.order - 1)


class ConvGRU(tu.ConvGRU):
    def get_cost(self):
        mult_cost = 0
        for k in range(self.num_layers):
            mult_cost += torch.sum(getattr(self, f'weight_ih_l{k}') != 0)
            mult_cost += torch.sum(getattr(self, f'weight_hh_l{k}') != 0)

        mult_cost += 3 * self.num_layers * self.hidden_size

        return mult_cost.item()


class Conv1d(nn.Conv1d):
    def get_cost(self):
        return torch.sum(self.weight != 0).item()


class BSpline(tu.BSpline):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels

    def get_cost(self):
        return self.in_channels * max(self.degree - 1, 0)


class ConvEQ(SequentialConv):
    def __init__(
        self,
        model,
        channels,
        kernel_sizes,
        strides,
        activation,
        activation_kwargs=None,
    ):
        layers = OrderedDict()
        n_layers = len(channels) - 1
        for layer_idx in range(0, n_layers):
            if layer_idx > 0:
                layers[f'layer_{layer_idx}_act'] = {
                    'model': activation,
                    'kwargs': activation_kwargs if activation_kwargs is not None else {},
                }
            layers[f'layer_{layer_idx + 1}'] = {
                'model': model,
                'kwargs': {
                    'in_channels': channels[layer_idx],
                    'out_channels': channels[layer_idx + 1],
                    'kernel_size': kernel_sizes[layer_idx],
                    'stride': strides[layer_idx],
                }
            }
        super().__init__(layers)
        
    def forward(self, x):
        return super().forward(F.pad(x, self.padding, mode='circular'))


class SequentialTensorTrain(nn.Module):
    def __init__(self, option_sizes: list[int], rank: int):
        super().__init__()
        self.num_cores = len(option_sizes)
        self.cores = nn.ParameterList()
        
        r_prev = 1
        for k, num_options in enumerate(option_sizes):
            r_next = 1 if (k == self.num_cores - 1) else rank
            
            core = nn.Parameter(torch.empty(num_options, r_prev, r_next))
            
            nn.init.normal_(core, mean=0.0, std=1.0 / math.sqrt(r_prev * r_next))
            self.cores.append(core)
            r_prev = r_next

    def expand(self, transition_matrix: torch.Tensor = None, start: int = 0) -> torch.Tensor:
        state = transition_matrix
        for core in self.cores[start:]:
            if state is None:
                # First core: (Q, r_prev, r_next). Drop r_prev=1 safely.
                state = core[:, 0, :] 
            else:
                state = torch.einsum('...r, qrn -> ...qn', state, core)
                
        # Drop the final r_next dimension (which is always 1) safely without squeeze
        return state[..., 0]

    def forward(self, inputs: list[torch.Tensor], transition_matrix: torch.Tensor = None, start: int = 0) -> torch.Tensor:
        state = transition_matrix
        for i in range(len(inputs)):
            core_slice = self.cores[i + start][inputs[i]]  # (B_eff, r_i, r_{i+1})
            if state is None:
                state = core_slice
            else:
                state = torch.bmm(state, core_slice)
        return state


class TTEQ(nn.Module):
    def __init__(
        self,
        model,
        channels,
        kernel_sizes,
        strides,
        activation,
        N_partitions: list[int],
        Q_partitions: list[int],
        N_b: list[int],
        tt_rank: int,
        activation_kwargs=None,
    ):
        super().__init__()
        self.N_partitions = N_partitions
        self.Q_partitions = Q_partitions
        self.N_b = N_b
        self.m = len(N_partitions)

        out_channels = sum([N_partitions[i] * (Q_partitions[i] - 1) for i in range(len(N_partitions))])
        kernel_sizes.append(1)
        strides.append(1)
        channels.append(out_channels)
        self.decision_regions = ConvEQ(model, channels, kernel_sizes, strides, activation, activation_kwargs)

        self.tt_engines = nn.ModuleList()
        for i in range(self.m):
            option_sizes = [Q_partitions[i]] * N_partitions[i]
            
            for j in range(i):
                window_len = N_b[j] + 1  # L_{b,j} = N_{b,j} + 1
                option_sizes.extend([2] * window_len)
                
            self.tt_engines.append(SequentialTensorTrain(option_sizes, rank=tt_rank))

    def _extract_bit_windows(self, B_j: torch.Tensor, N_b_j: int) -> torch.Tensor:
        L_b = N_b_j + 1
        pad_left = N_b_j // 2
        pad_right = N_b_j - pad_left
        
        padded_B = F.pad(B_j, (pad_left, pad_right), mode='circular')
        
        return padded_B.unfold(-1, L_b, 1)

    def forward(
        self,
        Y: torch.Tensor,
        true_B: torch.Tensor | None = None,
        use_true_B: bool = True,
        tau=1.0,
    ) -> torch.Tensor:
        B = Y.shape[0]
        regions_logits = self.decision_regions(Y)
        N_sym = regions_logits.shape[-1]
        one_hot_list = []
        C_i_list = []
        for i in range(self.m):
            N_i, Q_i = self.N_partitions[i], self.Q_partitions[i]
            logits_i = regions_logits[:, :N_i * (Q_i - 1), :]
            regions_logits = regions_logits[:, N_i * (Q_i - 1):, :]
            
            logits_i = logits_i.view(Y.shape[0], N_i, Q_i - 1, N_sym)
            logits_i = F.softmax(torch.cat([logits_i, torch.zeros(Y.shape[0], N_i, 1, N_sym, device=Y.device)], dim=2) / tau, dim=2)

            one_hot_list.append(list(torch.unbind(logits_i, dim=1)))
            C_i_list.append(list(torch.unbind(torch.argmax(logits_i, dim=2), dim=1)))
        
        decoded_bits_list = []
        llr_list = []
        expanded_llr_list = []
        bit_windows = []

        for i in range(self.m):
            if i > 0:
                if use_true_B and true_B is not None:
                    src_B = true_B[:, i - 1, :]
                else:
                    src_B = decoded_bits_list[-1]

                b_win = self._extract_bit_windows(src_B, self.N_b[i - 1])
                bit_windows.append(torch.unbind(b_win, dim=2))

            tt_inputs = [t.flatten().long() for b in bit_windows for t in b]
            bit_priors = self.tt_engines[i](tt_inputs)
            expanded_llr_i = self.tt_engines[i].expand(bit_priors, start=len(tt_inputs))
            llr_i = self.tt_engines[i]([t.flatten() for t in C_i_list[i]], bit_priors, start=len(tt_inputs)).squeeze()

            # Reshape back to temporal symbol format (B, N_symbols)
            llr_i = llr_i.view(B, N_sym)
            llr_list.append(llr_i)
            if i == 0:
                expanded_llr_list.append(expanded_llr_i)
            else:
                expanded_llr_list.append(expanded_llr_i.reshape(B, N_sym, *expanded_llr_i.shape[1:]))

            # Hard decision thresholding
            decided_bit = (llr_i < 0)
            decoded_bits_list.append(decided_bit)

        # Stack LLRs across all bit levels: (B, m, N_symbols)
        return torch.stack(llr_list, dim=1), expanded_llr_list, one_hot_list
