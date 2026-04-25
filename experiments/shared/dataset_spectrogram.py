import torch
from torch.utils.data import TensorDataset


def compute_spectrogram(eeg: torch.Tensor, n_fft: int = 128, hop_length: int = 64) -> torch.Tensor:
    if eeg.dim() != 3:
        raise ValueError(f"Expected EEG tensor with shape (B, C, T), got {tuple(eeg.shape)}")
    batch_size, channels, time_steps = eeg.shape
    eeg_flat = eeg.reshape(batch_size * channels, time_steps)
    window = torch.hann_window(n_fft, device=eeg.device, dtype=eeg.dtype)
    spec = torch.stft(
        eeg_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        return_complex=True,
    ).abs()
    spec = torch.log1p(spec)
    freq_bins, spec_steps = spec.shape[1], spec.shape[2]
    return spec.reshape(batch_size, channels, freq_bins, spec_steps)


def normalize_spectrograms(specs: torch.Tensor) -> torch.Tensor:
    mean = specs.mean(dim=(2, 3), keepdim=True)
    std = specs.std(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    return (specs - mean) / std


_normalize_spectrograms = normalize_spectrograms


class SpectrogramDataset(TensorDataset):
    def __init__(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        labels: torch.Tensor,
        n_fft: int = 128,
        hop_length: int = 64,
    ):
        super().__init__(noisy, clean, labels)
        self.noisy = noisy
        self.clean = clean
        self.labels = labels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectrograms = _normalize_spectrograms(
            compute_spectrogram(self.noisy, self.n_fft, self.hop_length)
        )

    def __getitem__(self, index):
        noisy = self.noisy[index]
        clean = self.clean[index]
        labels = self.labels[index]
        spectrogram = self.spectrograms[index]
        return noisy, clean, labels, spectrogram

    def __len__(self) -> int:
        return len(self.labels)


def build_dataset_with_spectrogram(
    self,
    clean_df,
    noise_type: str = "gaussian",
    n_fft: int = 128,
    hop_length: int = 64,
):
    (X_n, X_c, y, n_cls), _ = self.build_dataset_with_novelty(clean_df, noise_type)
    X_spec = _normalize_spectrograms(
        compute_spectrogram(X_n, n_fft=n_fft, hop_length=hop_length)
    )
    return X_n, X_c, y, X_spec, n_cls


def _compute_spectrograms_batch(
    eeg: torch.Tensor,
    n_fft: int = 128,
    hop_length: int = 64,
) -> torch.Tensor:
    return compute_spectrogram(eeg, n_fft=n_fft, hop_length=hop_length)


def build_dataset_with_novelty_bimodal(
    self,
    clean_df,
    noise_type: str = "gaussian",
    n_fft: int = 128,
    hop_length: int = 64,
):
    (X_n, X_c, y, n_cls), (X_n_unk, X_c_unk, y_unk) = self.build_dataset_with_novelty(
        clean_df,
        noise_type,
    )
    specs = _normalize_spectrograms(
        _compute_spectrograms_batch(X_n, n_fft=n_fft, hop_length=hop_length)
    )
    specs_unk = _normalize_spectrograms(
        _compute_spectrograms_batch(X_n_unk, n_fft=n_fft, hop_length=hop_length)
    )
    return (X_n, X_c, y, specs, n_cls), (X_n_unk, X_c_unk, y_unk, specs_unk)


def patch_builder_with_spectrogram():
    from .datapreprocessor import EEGDatasetBuilder

    EEGDatasetBuilder.build_dataset_with_spectrogram = build_dataset_with_spectrogram
    EEGDatasetBuilder.build_dataset_with_novelty_bimodal = build_dataset_with_novelty_bimodal
