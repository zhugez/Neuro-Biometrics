"""
Spectrogram Dataset Module for Multimodal EEG Biometrics.

Provides STFT spectrogram computation, SpectrogramDataset wrapper with cached
spectrograms, and integration with EEGDatasetBuilder for bimodal data loading.
"""

import torch
from torch.utils.data import TensorDataset


# ==============================================================================
# SPECTROGRAM COMPUTATION
# ==============================================================================
def compute_spectrogram(eeg: torch.Tensor, n_fft: int = 128,
                        hop_length: int = 64) -> torch.Tensor:
    """
    Compute STFT magnitude spectrogram from flattened EEG.

    Input:  (B*C, T) flattened EEG tensor (e.g., from EEGDatasetBuilder)
            or (B, C, T) shaped EEG tensor.
    Output: (B, C, F, T_spec) magnitude spectrogram.

    With n_fft=128 and hop_length=64 on 800 samples at 200 Hz:
        F = n_fft // 2 + 1 = 65
        T_spec = ceil((800 + 128 - 1) / 64) = 13

    Args:
        eeg:    EEG tensor, either (B*C, T) or (B, C, T).
        n_fft:  FFT window size (default: 128).
        hop_length: Hop between windows (default: 64).

    Returns:
        torch.Tensor of shape (B, C, F, T_spec) with magnitude spectrograms.
    """
    # Handle shape: (B, C, T) or (B*C, T)
    if eeg.dim() == 3:
        B, C, T = eeg.shape
        # Flatten to (B*C, T) for torch.stft along last dimension
        eeg_flat = eeg.view(B * C, T)
    else:
        BC, T = eeg.shape
        C = 1  # will reshape back assuming B=BC if dim was 2
        eeg_flat = eeg
        B = BC // C if 'C' in dir() and C > 1 else BC

    window = torch.hann_window(n_fft, device=eeg.device, dtype=eeg.dtype)

    # torch.stft expects (..., time), applies along last dim
    spec = torch.stft(
        eeg_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        return_complex=True,
    )

    # Magnitude
    spec = torch.abs(spec)

    # spec shape: (B*C, F, T_spec)
    F = spec.shape[1]
    T_spec = spec.shape[2]

    # Reshape to (B, C, F, T_spec)
    if eeg.dim() == 3:
        spec = spec.view(B, C, F, T_spec)
    else:
        spec = spec.view(BC, F, T_spec)

    return spec


# ==============================================================================
# SPECTROGRAM DATASET
# ==============================================================================
class SpectrogramDataset(TensorDataset):
    """
    Wraps an existing TensorDataset (noisy, clean, y) and precomputes
    STFT spectrograms from noisy EEG, caching them for fast access.

    Returns tuple: (noisy_EEG, clean_EEG, labels, spectrogram) where
        noisy_EEG:     (4, 800) float32
        clean_EEG:     (4, 800) float32
        labels:        scalar long
        spectrogram:   (4, 65, 13) float32, per-sample z-score normalized
    """

    def __init__(self, noisy: torch.Tensor, clean: torch.Tensor,
                 labels: torch.Tensor, n_fft: int = 128,
                 hop_length: int = 64):
        """
        Args:
            noisy:     (N, C, T) tensor of noisy EEG.
            clean:     (N, C, T) tensor of clean EEG.
            labels:    (N,) long tensor of subject labels.
            n_fft:     FFT window size (default: 128).
            hop_length: Hop between windows (default: 64).
        """
        super().__init__(noisy, clean, labels)
        self.noisy = noisy
        self.clean = clean
        self.labels = labels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Precompute and cache spectrograms for all samples
        self._compute_cached_spectrograms()

    def _compute_cached_spectrograms(self):
        """Compute spectrograms for all noisy EEG samples and cache."""
        # Compute spectrograms: (N, C, F, T_spec)
        specs = compute_spectrogram(self.noisy, self.n_fft, self.hop_length)

        # Per-sample z-score normalization
        # spec shape: (C, F, T_spec) per sample; flatten for mean/std
        normalized_specs = []
        for i in range(specs.shape[0]):
            spec = specs[i]  # (C, F, T_spec)
            mean = spec.mean()
            std = spec.std() + 1e-8
            normed = (spec - mean) / std
            normalized_specs.append(normed)

        self.spectrograms = torch.stack(normalized_specs, dim=0)  # (N, C, F, T_spec)

    def __getitem__(self, index):
        """Return (noisy, clean, labels, spectrogram) tuple."""
        noisy = self.noisy[index]
        clean = self.clean[index]
        labels = self.labels[index]
        spectrogram = self.spectrograms[index]
        return noisy, clean, labels, spectrogram

    def __len__(self) -> int:
        return len(self.labels)


# ==============================================================================
# EEG DATASET BUILDER INTEGRATION
# ==============================================================================
def build_dataset_with_spectrogram(self, clean_df, noise_type: str = "gaussian",
                                   n_fft: int = 128,
                                   hop_length: int = 64):
    """
    Build dataset with spectrograms precomputed alongside raw EEG.

    Returns:
        (X_n, X_c, y, X_spec, n_cls) tuple where:
            X_n:    (N, C, T) noisy EEG tensor
            X_c:    (N, C, T) clean EEG tensor
            y:      (N,) subject labels tensor
            X_spec: (N, C, F, T_spec) per-sample z-score normalized spectrograms
            n_cls:  number of known classes
    """
    # Reuse existing build logic to get raw tensors
    (X_n, X_c, y, n_cls), _ = self.build_dataset_with_novelty(clean_df, noise_type)

    # X_n is already a torch.Tensor from build_dataset_with_novelty
    # Compute spectrograms for all noisy EEG samples
    specs = compute_spectrogram(X_n, n_fft=n_fft, hop_length=hop_length)

    # Per-sample z-score normalization
    normalized_specs = []
    for i in range(specs.shape[0]):
        spec = specs[i]
        mean = spec.mean()
        std = spec.std() + 1e-8
        normed = (spec - mean) / std
        normalized_specs.append(normed)

    X_spec = torch.stack(normalized_specs, dim=0)

    return X_n, X_c, y, X_spec, n_cls


# Monkey-patch EEGDatasetBuilder with the new method
# Call this function once after importing both modules:
#   from .datapreprocessor import EEGDatasetBuilder
#   from .dataset_spectrogram import patch_builder_with_spectrogram
#   patch_builder_with_spectrogram()
# Or simply import dataset_spectrogram after datapreprocessor to auto-patch.
def _compute_spectrograms_batch(eeg: torch.Tensor, n_fft: int = 128,
                                 hop_length: int = 64) -> torch.Tensor:
    """Compute STFT spectrograms for all EEG channels, returns (N, C, F, T_spec)."""
    B, C, T = eeg.shape
    all_specs = []
    for ch in range(C):
        ch_data = eeg[:, ch, :]
        stft = torch.stft(
            ch_data, n_fft=n_fft, hop_length=hop_length,
            window=torch.hann_window(n_fft, device=eeg.device),
            return_complex=True, center=True, normalized=False,
        )
        all_specs.append(stft.abs())
    return torch.stack(all_specs, dim=1)  # (N, C, F, T_spec)


def _normalize_spectrograms(specs: torch.Tensor) -> torch.Tensor:
    """Per-sample z-score normalization of spectrograms."""
    normalized = []
    for i in range(specs.shape[0]):
        spec = specs[i]
        mean = spec.mean()
        std = spec.std() + 1e-8
        normalized.append((spec - mean) / std)
    return torch.stack(normalized, dim=0)


def build_dataset_with_novelty_bimodal(self, clean_df, noise_type: str = "gaussian",
                                        n_fft: int = 128,
                                        hop_length: int = 64):
    """
    Build bimodal dataset with known/unknown split for novelty detection.

    Extends build_dataset_with_novelty to also return precomputed spectrograms
    for both known (train) and unknown (holdout) subject data.

    Returns:
        (known_tuple, unknown_tuple) where:
            known_tuple: (X_n, X_c, y, X_spec, n_cls) — known subject data with spectrograms
            unknown_tuple: (X_n_unk, X_c_unk, y_unk, X_spec_unk) — holdout subject data
    """
    # Get known + unknown from parent
    (X_n, X_c, y, n_cls), (X_n_unk, X_c_unk, y_unk) = \
        self.build_dataset_with_novelty(clean_df, noise_type)

    # Compute spectrograms for known samples
    specs = _compute_spectrograms_batch(X_n, n_fft=n_fft, hop_length=hop_length)
    specs = _normalize_spectrograms(specs)

    # Compute spectrograms for unknown (holdout) samples
    specs_unk = _compute_spectrograms_batch(X_n_unk, n_fft=n_fft, hop_length=hop_length)
    specs_unk = _normalize_spectrograms(specs_unk)

    return (X_n, X_c, y, specs, n_cls), (X_n_unk, X_c_unk, y_unk, specs_unk)


def patch_builder_with_spectrogram():
    """Patch EEGDatasetBuilder with spectrogram methods."""
    from .datapreprocessor import EEGDatasetBuilder
    EEGDatasetBuilder.build_dataset_with_spectrogram = build_dataset_with_spectrogram
    EEGDatasetBuilder.build_dataset_with_novelty_bimodal = build_dataset_with_novelty_bimodal


EEGDatasetBuilder = None  # placeholder until patch_builder_with_spectrogram() is called
