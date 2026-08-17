import numpy as np
import torch
import torch.nn.functional as F

def abs2(x):
    """Efficiently compute absolute value squared."""
    if torch.is_complex(x):
        return x.real ** 2 + x.imag ** 2
    return x ** 2


def fd_cd(wavelength, speed_of_light, dispersion_coeff, fiber_len, symbol_rate, n_os, fft_size, dtype=torch.float32):
    """Chromatic dispersion on the frequency domain."""
    # cd_coeff calculation
    cd_coeff = np.pi * wavelength ** 2 / (speed_of_light * 1e9 / 1e12) * dispersion_coeff * fiber_len * (symbol_rate / 1e12) ** 2
    freqs = torch.fft.fftfreq(fft_size).to(dtype=dtype)
    H_cd = np.exp(1j * cd_coeff * n_os ** 2 * freqs ** 2)
    return H_cd


def fd_rrc(roll_off, n_os, fft_size):
    """Root-raised cosine filter on the frequency domain."""
    cutoff_freq = 1 / (2 * n_os)
    freqs = torch.fft.fftfreq(fft_size)
    H_rrc = torch.zeros_like(freqs)
    for i in range(fft_size):
        f = abs(freqs[i])
        if f < cutoff_freq * (1 - roll_off):
            H_rrc[i] = 1
        elif f < cutoff_freq * (1 + roll_off):
            H_rrc[i] = 0.5 * (1 + torch.cos(np.pi * n_os / roll_off * (f - cutoff_freq * (1 - roll_off))))
    
    # Normalize RRC filter in frequency domain
    H_rrc = H_rrc * torch.sqrt(fft_size / torch.sum(H_rrc ** 2))
    return H_rrc


class CD_DD:
    def __init__(
            self,
            wavelength,
            speed_of_light,
            dispersion_coeff,
            fiber_len,
            symbol_rate,
            n_os,
            block_size,
            roll_off,
            nu_1_dB,
            nu_2_dB,
            sps,
            dtype=torch.float32,
            device="cpu"
        ):

        self.fft_size = n_os * block_size
        self.H_rrc = fd_rrc(roll_off, n_os, self.fft_size).to(dtype=dtype, device=device)
        self.H_cd = fd_cd(wavelength, speed_of_light, dispersion_coeff, fiber_len, symbol_rate, n_os, self.fft_size, dtype).to(device=device)
        self.pd_width = int((1 + roll_off) / 2 * block_size)
        self.nu_1 = 0. if nu_1_dB == 'inf' else 10 ** (-nu_1_dB / 10)
        self.nu_2 = 0. if nu_2_dB == 'inf' else 10 ** (-nu_2_dB / 10)
        self.n_os = n_os
        self.sps = sps

    def __call__(self, symbols):
        """FFT-based chromatic dispersion with direct detection."""
        # Upsample symbols
        symbols_upsampled = torch.zeros(symbols.shape[:-1] + (self.fft_size,), dtype=symbols.dtype, device=symbols.device)
        symbols_upsampled[..., ::self.n_os] = symbols

        # Apply RRC and CD in frequency domain
        tx_freq = torch.fft.fft(symbols_upsampled, dim=-1)
        rx_freq = tx_freq * self.H_rrc * self.H_cd
        
        # Add optical noise
        rx_noise = rx_freq + np.sqrt(self.fft_size * self.nu_1) * torch.randn_like(rx_freq)
        
        # Photodiode optical filter
        rx_noise[..., self.pd_width:-self.pd_width] = 0

        # Time domain signal after photodiode
        rx_time = torch.fft.ifft(rx_noise, dim=-1)
        
        # Normalize power before non-linearity
        rx_time = rx_time * np.sqrt(self.n_os)
        rx_time = abs2(rx_time).real

        # Add electrical noise
        rx_time = rx_time + np.sqrt(self.nu_2 * self.n_os) * torch.randn_like(rx_time)

        # ADC filter
        rx_freq = torch.fft.fft(rx_time, dim=-1)
        rx_freq[..., 2 * self.pd_width:-2 * self.pd_width] = 0
        
        # Downsample
        rx_time = torch.fft.ifft(rx_freq, dim=-1)
        rx_time = rx_time[..., ::self.n_os // self.sps].real

        return rx_time


class PDPAWGNChannel:
    def __init__(
        self,
        K: int,
        L: int,
        min_p: float,
        max_p: float,
        min_snr_db: float,
        max_snr_db: float,
        dtype=torch.float32,
        device="cpu"
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.K = K
        self.L = L
        self.N = self.K + self.L
        self.rho = self.K / self.N

        self.min_p = float(min_p)
        self.max_p = float(max_p)
        self.min_snr_db = float(min_snr_db)
        self.max_snr_db = float(max_snr_db)

    def sample_p(
        self,
        B: int,
    ):
        return torch.rand((B, 1), dtype=self.dtype, device=self.device) * (self.max_p - self.min_p) + self.min_p

    def sample_snr_db(
        self,
        B: int,
    ) -> torch.Tensor:
        return torch.rand((B, 1), dtype=self.dtype, device=self.device) * (self.max_snr_db - self.min_snr_db) + self.min_snr_db

    def generate_h(
        self,
        B: int,
        p,
    ) -> torch.Tensor:
        complex_noise = torch.randn(size=(B, self.L), dtype=self.dtype, device=self.device) + 1j * torch.randn(size=(B, self.L), dtype=self.dtype, device=self.device)
        power_scaling = torch.sqrt((p ** torch.arange(self.L, dtype=self.dtype, device=self.device)) / 2)

        return power_scaling * complex_noise

    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x[..., 0]   # Last dimension should be a single time sample
        B = x.shape[0]

        p = self.sample_p(B) 
        snr_db = self.sample_snr_db(B)
        h = self.generate_h(B, p)

        y = torch.conv1d(F.pad(x[None, :, :], (h.shape[-1] - 1, h.shape[-1] - 1)), h[:, None, :], groups=B)[0, ...]

        is_one = torch.isclose(p, torch.tensor(1.0, dtype=self.dtype, device=self.device))
        p_safe = torch.where(is_one, torch.tensor(0.0, dtype=self.dtype, device=self.device), p)
        avg_fading = (1.0 - p_safe ** self.L) / (1.0 - p_safe)
        avg_fading = torch.where(is_one, torch.tensor(self.L, dtype=self.dtype, device=self.device), avg_fading)
        ebn0_linear = 10.0 ** (snr_db / 10.0)
        noise_variance = avg_fading / (self.rho * ebn0_linear) # N_0
        noise_std = torch.sqrt(noise_variance / 2.0)    # sqrt(N_0 / 2)

        return y + noise_std * (torch.randn_like(y) + 1j * torch.randn_like(y))
