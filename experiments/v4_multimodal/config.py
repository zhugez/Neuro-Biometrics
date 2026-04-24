from dataclasses import dataclass

from shared.datapreprocessor import Config


@dataclass
class V4Config(Config):
    spectrogram_n_fft: int = 128
    spectrogram_hop_length: int = 64
    spectrogram_source: str = "denoised"
    spec_embed_dim: int | None = None
    fusion_num_heads: int = 4
    stage1_epochs: int = 30
    stage2_epochs: int = 30
    fusion_lr: float = 2e-4
    spec_embedder_lr: float = 1e-4
    early_stop_metric: str = "auroc"
