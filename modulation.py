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
