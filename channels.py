import numpy as np
import torch

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
