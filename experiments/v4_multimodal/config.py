from dataclasses import dataclass, field
from typing import List


@dataclass
class V4Config:
    """Configuration for V4 multimodal experiments.

    Inherits all fields from shared.datapreprocessor.Config. Add multimodal-specific:
    """
    # Spectrogram parameters
    spectrogram_n_fft: int = 128
    spectrogram_hop_length: int = 64
    spectrogram_source: str = "noisy"  # "noisy" or "denoised"
    spec_embed_dim: int = 256  # embedding dim for spectrogram branch

    # Fusion parameters
    fusion_num_heads: int = 4

    # Training
    stage1_epochs: int = 30
    stage2_epochs: int = 30
    fusion_lr: float = 2e-4
    spec_embedder_lr: float = 1e-4

    # Early stopping on AUROC
    early_stop_metric: str = "auroc"

    # Rest inherited from shared Config:
    # embed_dim, batch_size, epochs, learning_rate, weight_decay, patience,
    # arcface_margin, arcface_scale, data_path, electrodes, etc.
