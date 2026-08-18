import numpy as np
import torch
import torch.nn as nn
import math
from itertools import product


def gray(m, n_dim, inverse=False):
    """Gray code generation."""
    base = np.arange(m)
    base = np.bitwise_xor(base, np.right_shift(base, 1))
    out = np.zeros(m**n_dim, dtype=int)
    n_bits = int(np.log2(m))

    for o_idx, elements in enumerate(product(base, repeat=n_dim)):
        if inverse:
            out[sum(2 ** (idx * n_bits) * elements[idx] for idx in range(n_dim))] = (
                o_idx
            )
        else:
            out[o_idx] = sum(
                2 ** (idx * n_bits) * elements[idx] for idx in range(n_dim)
            )

    return out


def get_bit_map(m):
    """Get bit mapping for constellation mapping."""
    n_bits = int(np.log2(m))
    rows = np.arange(m, dtype=np.uint16)
    bits = np.unpackbits(rows[:, None].view(np.uint8)[:, ::-1], axis=1)
    return bits[:, -n_bits:]


def get_bits(idxs, bit_map):
    """Retrieve bits for specific indices."""
    return bit_map[idxs, :].transpose(-1, -2)


def normalize(alphabet, probs):
    """Normalize the constellation alphabet power."""
    power = torch.sum(probs[:, None, None] * alphabet**2, dim=(0, 2), keepdim=True)
    alphabet = alphabet / torch.sqrt(power)
    return alphabet


def sample(size, alphabet, probs):
    """Sample from the constellation distribution."""
    n_samples = math.prod(size)
    out_idxs = torch.multinomial(probs, n_samples, replacement=True).view(size)
    out_symbols = alphabet[out_idxs, ...]

    # Flatten time dimension and get real and imaginary parts
    out_symbols = out_symbols.transpose(1, 2)
    out_symbols = out_symbols[..., 0] + 1j * out_symbols[..., 1]

    return out_symbols, out_idxs


class Sampler(nn.Module):
    """Constellation Sampler module."""

    def __init__(
        self,
        alphabet,
        train_alphabet=True,
        train_probabilities=True,
        real_only=False,
        size=None,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        self.real_only = real_only
        self.log_probs = nn.Parameter(
            torch.zeros(alphabet.shape[0], dtype=dtype, device=device),
            requires_grad=train_probabilities,
        )
        self.alphabet = nn.Parameter(
            torch.tensor(alphabet, dtype=dtype, device=device),
            requires_grad=train_alphabet,
        )
        self.real_mask = torch.zeros_like(self.alphabet)
        self.real_mask[:, :, 0] = 1
        self.default_size = size

    @property
    def symbol_probabilities(self):
        return nn.functional.softmax(self.log_probs, dim=0)

    def get_alphabet(self):
        if self.real_only:
            alphabet = self.alphabet * self.real_mask
        else:
            alphabet = self.alphabet
        return normalize(alphabet, self.symbol_probabilities)

    def forward(self, size=None):
        if size is None:
            size = self.default_size
        return sample(size, self.get_alphabet(), self.symbol_probabilities)


class BMOCZEncoder(nn.Module):
    def __init__(self, K: int, L: int, batch_size: int, train_alphabet: bool = False, train_probabilities: bool = False, device: torch.device = 'cpu', dtype=torch.float32):
        super().__init__()
        self.K = K
        self.L = L
        self.N = self.K + self.L
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        self.bit_logits = nn.Parameter(torch.zeros(self.K, device=device, dtype=dtype), requires_grad=train_probabilities)
        
        self.R = nn.Parameter(
            torch.sqrt(1.0 + torch.sin(torch.tensor(torch.pi / self.K, device=device, dtype=dtype))),
            requires_grad=train_alphabet,
        )

        self.register_buffer("rotations", torch.exp(2j * torch.pi * (torch.arange(0, self.K, device=device, dtype=dtype)) / self.K))

    @property
    def symbol_probabilities(self) -> torch.Tensor:
        aux_logits = torch.zeros(self.K, 2, device=self.device, dtype=self.dtype)
        aux_logits[:, 0] = self.bit_logits
        symbol_logits = aux_logits[0, :]
        for i in range(1, self.K):
            symbol_logits = symbol_logits[..., None] + aux_logits[i, :]

        return torch.softmax(symbol_logits.flatten(), dim=0)

    def get_alphabet(self) -> torch.Tensor:
        alpha_0 = 1 / self.R * self.rotations
        alpha_1 = self.R * self.rotations
        alphabet = torch.stack([alpha_0[[-1]], alpha_1[[-1]]], dim=0)
        for i in range(self.K - 2, -1, -1):
            n = alphabet.shape[0]
            alphabet_0 = torch.cat([alpha_0[None, [i]].expand(n, -1), alphabet], dim=1)
            alphabet_1 = torch.cat([alpha_1[None, [i]].expand(n, -1), alphabet], dim=1)
            alphabet = torch.cat([alphabet_0, alphabet_1], dim=0)

        alphabet = self.encode_selected_zeros(alphabet)

        return torch.stack([alphabet.real, alphabet.imag], dim=2)

    @staticmethod
    def encode_selected_zeros(alpha: torch.Tensor) -> torch.Tensor:
        B, K = alpha.shape
        x = torch.stack(
            [
                -alpha[:, 0],
                torch.ones(B, device=alpha.device, dtype=alpha.dtype),
            ],
            dim=1,
        )

        for q in range(1, K):
            a = alpha[:, q : q + 1]
            x = torch.cat(
                [torch.zeros(B, 1, device=alpha.device, dtype=alpha.dtype), x],
                dim=1,
            ) - torch.cat(
                [a * x, torch.zeros(B, 1, device=alpha.device, dtype=alpha.dtype)],
                dim=1,
            )

        return x / torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=1, keepdim=True))

    def forward(
        self,
        size=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if size is None:
            B = self.batch_size
        else:
            B = size

        return sample((B, 1), self.get_alphabet() * np.sqrt(self.N), self.symbol_probabilities)
