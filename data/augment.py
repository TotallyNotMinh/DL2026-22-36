import random
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as AF


class FrequencyMasking(nn.Module):
    """
    Applies frequency masking to a spectrogram of shape (..., F, T).
    """
    def __init__(self, freq_mask_param=24, num_masks=1, iid_masks=True):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        if freq_mask_param > 0:
            self.masker = T.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=iid_masks)
        else:
            self.masker = None

    def forward(self, x):
        if self.masker is None or self.num_masks <= 0:
            return x
        for _ in range(self.num_masks):
            x = self.masker(x)
        return x


class TimeMasking(nn.Module):
    """
    Applies time masking to a spectrogram of shape (..., F, T).
    """
    def __init__(self, time_mask_param=48, num_masks=1, iid_masks=True):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
        if time_mask_param > 0:
            self.masker = T.TimeMasking(time_mask_param=time_mask_param, iid_masks=iid_masks)
        else:
            self.masker = None

    def forward(self, x):
        if self.masker is None or self.num_masks <= 0:
            return x
        for _ in range(self.num_masks):
            x = self.masker(x)
        return x


class RandomTimeShift(nn.Module):
    """
    Random time shift augmentation along time axis.
    Paper: +/- 10 frames for FSD50K.
    """
    def __init__(self, max_shift=10):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x):
        if self.max_shift <= 0:
            return x
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)


class RandomNoise(nn.Module):
    """
    Uniform additive random noise on spectrogram.
    Paper: U(0, 0.05) on spectrogram for FSD50K.
    """
    def __init__(self, max_noise=0.05):
        super().__init__()
        self.max_noise = max_noise

    def forward(self, x):
        if self.max_noise <= 0:
            return x
        noise = torch.rand_like(x) * self.max_noise
        return x + noise


class SpecAugment(nn.Module):
    """
    SpecAugment module combining frequency and time masking for audio spectrograms.
    Defaults matching AST / CMKD: freq_mask_param=48, time_mask_param=192.
    """
    def __init__(
        self,
        freq_mask_param=48,
        time_mask_param=192,
        num_freq_masks=2,
        num_time_masks=2,
        iid_masks=True,
    ):
        super().__init__()
        self.freq_mask = FrequencyMasking(
            freq_mask_param=freq_mask_param,
            num_masks=num_freq_masks,
            iid_masks=iid_masks,
        )
        self.time_mask = TimeMasking(
            time_mask_param=time_mask_param,
            num_masks=num_time_masks,
            iid_masks=iid_masks,
        )

    def forward(self, x):
        return self.time_mask(self.freq_mask(x))


def mixup_samples(x1, y1, x2, y2, alpha=0.5):
    """
    Mixup between two samples (or pairs of tensors).
    Works for:
      - Multi-label classification: x=spectrogram, y=binary/multi-hot vector
      - Denoising: x=noisy_spectrogram, y=clean_spectrogram
    Returns:
      mixed_x, mixed_y
    """
    if alpha <= 0.0:
        return x1, y1
    lam = float(np.random.beta(alpha, alpha))
    mixed_x = lam * x1 + (1.0 - lam) * x2
    mixed_y = lam * y1 + (1.0 - lam) * y2
    return mixed_x, mixed_y


def mixup_batch(x, y, alpha=0.5):
    """
    Mixup applied across a mini-batch along dimension 0.
    x: Tensor (B, ...)
    y: Tensor (B, ...)
    Returns:
      mixed_x, mixed_y, lam
    """
    if alpha <= 0.0 or x.size(0) <= 1:
        return x, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    mixed_y = lam * y + (1.0 - lam) * y[perm]
    return mixed_x, mixed_y, lam


class Mixup:
    """
    Mixup wrapper supporting both sample-level and batch-level invocation.
    """
    def __init__(self, alpha=0.5, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, x1, y1, x2=None, y2=None):
        if random.random() > self.p:
            if x2 is None:
                return x1, y1
            return x1, y1
        if x2 is None:
            mixed_x, mixed_y, _ = mixup_batch(x1, y1, alpha=self.alpha)
            return mixed_x, mixed_y
        return mixup_samples(x1, y1, x2, y2, alpha=self.alpha)


class RandomColoredNoise(nn.Module):
    """
    Applies synthetic colored noise (white, pink, brown/red, band-pass) to waveform
    at randomized SNR levels without requiring external noise datasets.
    """
    def __init__(
        self,
        p=0.5,
        min_snr_db=-5.0,
        max_snr_db=20.0,
        min_alpha=0.0,
        max_alpha=2.0,
        bandpass_prob=0.3,
        sample_rate=16000,
    ):
        super().__init__()
        self.p = p
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.bandpass_prob = bandpass_prob
        self.sample_rate = sample_rate

    def forward(self, x):
        """
        x: Tensor of shape (..., L), e.g. (1, target_len)
        """
        if self.p <= 0.0 or random.random() > self.p:
            return x

        rms_sig = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
        if torch.all(rms_sig < 1e-5):
            return x

        L = x.shape[-1]
        device = x.device
        dtype = x.dtype

        # Generate white Gaussian noise
        white = torch.randn_like(x)
        W = torch.fft.rfft(white, n=L)
        freqs = torch.fft.rfftfreq(L, d=1.0 / self.sample_rate, device=device)

        # Spectral exponent alpha: 0=white, 1=pink (1/f), 2=brown (1/f^2)
        alpha = random.uniform(self.min_alpha, self.max_alpha)
        f_clamp = torch.clamp(freqs, min=10.0)
        scale = 1.0 / (f_clamp ** (alpha / 2.0))
        scale[..., 0] = 0.0

        # Optional bandpass filtering
        if random.random() < self.bandpass_prob:
            f_low = random.uniform(0.0, self.sample_rate * 0.25)
            f_high = random.uniform(f_low + 500.0, self.sample_rate * 0.5)
            band_mask = ((freqs >= f_low) & (freqs <= f_high)).to(dtype)
            scale = scale * band_mask

        noise = torch.fft.irfft(W * scale, n=L)
        rms_noise = torch.sqrt(torch.mean(noise ** 2, dim=-1, keepdim=True) + 1e-8)
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        target_rms = rms_sig * (10.0 ** (-snr_db / 20.0))
        noise = noise * (target_rms / (rms_noise + 1e-8))

        return x + noise


class RandomReverberation(nn.Module):
    """
    Applies synthetic reverberation to waveform using an exponentially decaying
    room impulse response (Schroeder late reverberation model).
    """
    def __init__(
        self,
        p=0.3,
        min_t60=0.15,
        max_t60=0.6,
        min_wet=0.1,
        max_wet=0.4,
        sample_rate=16000,
    ):
        super().__init__()
        self.p = p
        self.min_t60 = min_t60
        self.max_t60 = max_t60
        self.min_wet = min_wet
        self.max_wet = max_wet
        self.sample_rate = sample_rate

    def forward(self, x):
        """
        x: Tensor of shape (..., L), e.g. (1, target_len)
        """
        if self.p <= 0.0 or random.random() > self.p:
            return x

        rms_in = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
        if torch.all(rms_in < 1e-5):
            return x

        t60 = random.uniform(self.min_t60, self.max_t60)
        dur = min(t60, 0.4)
        n_rir = max(16, int(dur * self.sample_rate))

        t = torch.linspace(0, dur, n_rir, device=x.device, dtype=x.dtype)
        decay = torch.exp(-6.9078 * t / t60)
        rir_shape = [1] * (x.ndim - 1) + [n_rir]
        rir = torch.randn(rir_shape, device=x.device, dtype=x.dtype) * decay
        rir[..., 0] = 1.0  # Direct sound impulse
        rir = rir / torch.sqrt(torch.sum(rir ** 2) + 1e-8)

        reverbed = AF.fftconvolve(x, rir, mode="full")[..., :x.shape[-1]]
        wet = random.uniform(self.min_wet, self.max_wet)
        out = (1.0 - wet) * x + wet * reverbed

        rms_out = torch.sqrt(torch.mean(out ** 2, dim=-1, keepdim=True) + 1e-8)
        out = out * (rms_in / (rms_out + 1e-8))

        return out

