import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .model import WaveNetDenoiser, ResNetMetricEmbedder
from .model_spectrogram import SpectrogramMambaBranch
from .fusion import CrossAttentionFusion


class MultimodalEEGMetricModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet34",
        n_channels: int = 4,
        embed_dim: int = 128,
        pretrained: bool = True,
        use_mamba: bool = True,
        spec_embed_dim: int | None = None,
    ):
        super().__init__()
        if spec_embed_dim is None:
            spec_embed_dim = embed_dim
        self.embed_dim = embed_dim
        self.spec_embed_dim = spec_embed_dim
        self.denoiser = WaveNetDenoiser(channels=n_channels, use_mamba=use_mamba)
        self.embedder = ResNetMetricEmbedder(
            backbone=backbone,
            in_chans=n_channels,
            embed_dim=embed_dim,
            pretrained=pretrained,
        )
        self.spec_embedder = SpectrogramMambaBranch(
            in_chans=n_channels,
            embed_dim=spec_embed_dim,
            pretrained=pretrained,
            use_mamba=use_mamba,
        )
        self.spec_projection = (
            nn.Identity()
            if spec_embed_dim == embed_dim
            else nn.Linear(spec_embed_dim, embed_dim)
        )
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim)

    def forward(
        self,
        raw_eeg: torch.Tensor,
        spectrogram: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        denoised = self.denoiser(raw_eeg)
        emb1 = self.embedder(denoised)
        emb2 = self.spec_embedder(spectrogram)
        emb2 = self.spec_projection(emb2)
        if self.spec_embed_dim != self.embed_dim:
            emb2 = F.normalize(emb2, p=2, dim=1)
        fused = self.fusion(emb1, emb2)
        return denoised, fused


def create_multimodal_model(
    backbone: str = "resnet34",
    n_channels: int = 4,
    embed_dim: int = 128,
    pretrained: bool = True,
    use_mamba: bool = True,
    spec_embed_dim: int | None = None,
) -> MultimodalEEGMetricModel:
    return MultimodalEEGMetricModel(
        backbone=backbone,
        n_channels=n_channels,
        embed_dim=embed_dim,
        pretrained=pretrained,
        use_mamba=use_mamba,
        spec_embed_dim=spec_embed_dim,
    )
