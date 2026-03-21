"""
Multimodal EEG biometric model: WaveNet+Mamba + SpectrogramMambaBranch + CrossAttentionFusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .model import WaveNetDenoiser, ResNetMetricEmbedder, create_metric_model
from .model_spectrogram import SpectrogramMambaBranch
from .fusion import CrossAttentionFusion


class MultimodalEEGMetricModel(nn.Module):
    """
    Three-branch multimodal model:
    - Branch 1: WaveNet+Mamba denoiser -> ResNet embedder -> Embed1 (L2-normed)
    - Branch 2: SpectrogramMambaBranch -> Embed2 (L2-normed)
    - Fusion: CrossAttentionFusion(Embed1, Embed2) -> Fused embedding (NOT normed)
    """
    def __init__(self, backbone="resnet34", n_channels=4, embed_dim=128,
                 pretrained=True, use_mamba=True, spec_embed_dim=None):
        super().__init__()
        if spec_embed_dim is None:
            spec_embed_dim = embed_dim
        self.embed_dim = embed_dim
        self.spec_embed_dim = spec_embed_dim

        # Branch 1: EEG denoiser + embedder
        self.denoiser = WaveNetDenoiser(channels=n_channels, use_mamba=use_mamba)
        self.embedder = ResNetMetricEmbedder(
            backbone=backbone, in_chans=n_channels,
            embed_dim=embed_dim, pretrained=pretrained,
        )

        # Branch 2: Spectrogram embedder
        self.spec_embedder = SpectrogramMambaBranch(
            in_chans=n_channels,
            embed_dim=spec_embed_dim,
            pretrained=pretrained,
        )

        # Fusion module (requires matching embed_dim and spec_embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim)

    def forward(self, raw_eeg: torch.Tensor,
                spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            raw_eeg: (B, 4, 800) -- noisy EEG input
            spectrogram: (B, 4, 65, 13) -- precomputed STFT spectrogram
        Returns:
            denoised: (B, 4, 800) -- WaveNet denoiser output
            fused: (B, embed_dim) -- cross-attention fused embedding, NOT L2-normed
        """
        # Branch 1: denoise + embed EEG
        denoised = self.denoiser(raw_eeg)      # (B, 4, 800)
        emb1 = self.embedder(denoised)          # (B, embed_dim), L2-normed

        # Branch 2: embed spectrogram
        emb2 = self.spec_embedder(spectrogram)  # (B, spec_embed_dim), L2-normed

        # Fusion: cross-attention combine
        fused = self.fusion(emb1, emb2)         # (B, embed_dim), NOT normed

        return denoised, fused


def create_multimodal_model(backbone="resnet34", n_channels=4, embed_dim=128,
                            pretrained=True, use_mamba=True) -> MultimodalEEGMetricModel:
    """
    Factory function to create the full multimodal model.

    Args:
        backbone: ResNet variant for EEG branch ("resnet18", "resnet34", "resnet50")
        n_channels: Number of EEG channels (default 4: T7, F8, Cz, P4)
        embed_dim: Embedding dimension for EEG branch and fusion output (default 128)
        pretrained: Use ImageNet-pretrained ResNet weights (default True)
        use_mamba: Insert Mamba SSM block in WaveNet denoiser (default True)

    Returns:
        MultimodalEEGMetricModel with embed_dim==spec_embed_dim
    """
    return MultimodalEEGMetricModel(
        backbone=backbone,
        n_channels=n_channels,
        embed_dim=embed_dim,
        pretrained=pretrained,
        use_mamba=use_mamba,
        spec_embed_dim=embed_dim,  # Must match for CrossAttentionFusion
    )
