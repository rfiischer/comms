import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import re

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


class ConvDecisionRegion(nn.Module):
    def __init__(self, n_intervals, kernel_size, sps, hidden_dim=8, device='cpu'):
        super().__init__()
        # n_intervals is now the TOTAL number of regions (N_I), e.g., 32 or 64.
        self.N_I = n_intervals 
        self.kernel_size = kernel_size
        self.sps = sps
        
        # 1. Feature Extractor (Sliding Window)
        # stride=sps natively downsamples the oversampled signal to the symbol rate
        self.conv1 = nn.Conv1d(
            in_channels=1, 
            out_channels=hidden_dim, 
            kernel_size=kernel_size, 
            stride=sps
        ).to(device)
        
        # 2. Region Projector (1x1 Convolution)
        # Acts as a pointwise linear combination to map features to regions
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=self.N_I, 
            kernel_size=1
        ).to(device)

    def forward(self, rx):
        """
        rx: Expected shape (B, 1, N_samp) or (B, N_samp)
        """
        if rx.dim() == 2:
            rx = rx.unsqueeze(1)
            
        # Pad exactly like the previous logic to maintain time alignment
        padded_rx = F.pad(rx, (self.kernel_size // 2, self.kernel_size // 2))
        
        # Extract features: (B, 1, padded_L) -> (B, hidden_dim, block_size)
        x = F.relu(self.conv1(padded_rx))
        
        # Map to regions: (B, hidden_dim, block_size) -> (B, N_I, block_size)
        logits = self.conv2(x)
        
        # Softmax creates valid probabilistic partitions across the N_I regions
        probs = F.softmax(logits, dim=1)
        
        # Permute to match the estimator's expected shape: (N_I, B, block_size)
        return probs.permute(1, 0, 2)