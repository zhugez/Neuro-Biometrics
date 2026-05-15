"""
Model architectures for EEG denoising + metric learning.

- WaveNetDenoiser: Dilated Conv1D denoiser with optional Mamba SSM block
- ResNetMetricEmbedder: ResNet-based metric learning embedder  
- EEGMetricModel: Two-stage composite model (denoiser → embedder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchvision.models import resnet18, resnet34, resnet50

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

BACKBONES = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


class MambaBlock(nn.Module):
    """
    Mamba SSM block with pre-norm, dropout, and residual connection.
    Bridges WaveNet's (B, C, L) layout with Mamba's expected (B, L, D).
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError(
                "mamba-ssm is required for MambaBlock. "
                "Install: pip install mamba-ssm>=1.2.0 causal-conv1d>=1.1.0"
            )
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) from Conv1d flow."""
        residual = x
        h = x.transpose(1, 2)           # (B, L, C)
        h = self.norm(h)
        h = self.mamba(h)
        h = self.dropout(h)
        return h.transpose(1, 2) + residual  # (B, C, L)


class WaveNetBlock(nn.Module):
    """Single dilated-conv gated activation block."""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels * 2, 3,
                                 padding=dilation, dilation=dilation)
        self.conv1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dilated(x)
        h1, h2 = h.chunk(2, dim=1)
        h = torch.tanh(h1) * torch.sigmoid(h2)
        return x + self.conv1x1(h)


class WaveNetDenoiser(nn.Module):
    """
    WaveNet-style 1D denoiser with optional Mamba block at midpoint.

    Args:
        channels: Number of input EEG channels.
        hidden: Hidden dimension for Conv1d layers.
        blocks: Number of WaveNet block groups.
        layers_per_block: Dilated layers per block group (dilation = 2^i).
        use_mamba: If True, insert a MambaBlock at the midpoint.
    """
    def __init__(self, channels: int = 4, hidden: int = 64, blocks: int = 3,
                 layers_per_block: int = 4, use_mamba: bool = False):
        super().__init__()
        self.input_conv = nn.Conv1d(channels, hidden, 1)

        self.blocks = nn.ModuleList()
        total_layers = blocks * layers_per_block
        mamba_pos = total_layers // 2

        current_layer = 0
        for _b in range(blocks):
            for i in range(layers_per_block):
                self.blocks.append(WaveNetBlock(hidden, 2 ** i))
                current_layer += 1
                if use_mamba and current_layer == mamba_pos:
                    self.blocks.append(MambaBlock(hidden))

        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(x)
        for block in self.blocks:
            h = block(h)
        return self.output_conv(h)


class ResNetMetricEmbedder(nn.Module):
    """
    ResNet-based metric learning embedder for EEG signals.

    Reshapes 1D EEG (B, C, T) → 2D (B, C, H, W) for spatial convolutions.
    Uses smaller conv1 (3×3 stride=1) and no maxpool to preserve spatial info
    on small inputs.
    """
    def __init__(self, backbone: str = "resnet18", in_chans: int = 4,
                 embed_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        base = BACKBONES[backbone](weights=weights)
        base.conv1 = nn.Conv2d(in_chans, 64, kernel_size=3, stride=1,
                                padding=1, bias=False)
        base.maxpool = nn.Identity()
        nfeat = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(nfeat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    @staticmethod
    def _find_2d_shape(n: int) -> Tuple[int, int]:
        """Find H, W close to sqrt for proper 2D structure. E.g. 800→(25,32)."""
        sqrt_n = int(n ** 0.5)
        for h in range(sqrt_n, 0, -1):
            if n % h == 0:
                return h, n // h
        return 1, n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        H, W = self._find_2d_shape(T)
        x = x.view(B, C, H, W)
        feat = self.backbone(x)
        emb = self.head(feat)
        return F.normalize(emb, p=2, dim=1)


class EEGMetricModel(nn.Module):
    """Two-stage composite: denoiser → embedder."""
    def __init__(self, filter_model: WaveNetDenoiser,
                 embedder_model: ResNetMetricEmbedder):
        super().__init__()
        self.denoiser = filter_model
        self.embedder = embedder_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        denoised = self.denoiser(x)
        emb = self.embedder(denoised)
        return denoised, emb


def create_metric_model(backbone: str = "resnet18", n_channels: int = 4,
                        embed_dim: int = 128, pretrained: bool = True,
                        use_mamba: bool = False,
                        embedder_type: str = "resnet",
                        use_denoiser: bool = True) -> EEGMetricModel:
    """Factory function to create the full two-stage model.

    ``embedder_type`` selects which encoder follows the denoiser:

    * ``"resnet"`` (default) — ``ResNetMetricEmbedder`` with ``backbone``.
    * ``"mindid"`` — ``MindIDEmbedder`` (LSTM + attention).
    * ``"brainnet"`` — ``BrainNetEmbedder`` (1D CNN tower).

    The MindID and BrainNet embedders are re-implementations of prior-work
    architectures for the cross-protocol comparison; they ignore the
    ``backbone`` and ``pretrained`` arguments.

    ``use_denoiser`` controls whether Stage I is active:

    * ``True`` (default) — WaveNet denoiser is built and trained on SI-SNR.
    * ``False`` — denoiser is replaced by ``nn.Identity()`` so the embedder
      receives the raw noisy signal directly. Used for pure prior-work
      baselines that must not benefit from our Stage-I contribution.
    """
    if use_denoiser:
        denoiser = WaveNetDenoiser(channels=n_channels, use_mamba=use_mamba)
    else:
        denoiser = nn.Identity()
    if embedder_type == "resnet":
        embedder = ResNetMetricEmbedder(
            backbone=backbone, in_chans=n_channels,
            embed_dim=embed_dim, pretrained=pretrained,
        )
    elif embedder_type == "mindid":
        from .baseline_embedders import MindIDEmbedder
        embedder = MindIDEmbedder(in_chans=n_channels, embed_dim=embed_dim)
    elif embedder_type == "brainnet":
        from .baseline_embedders import BrainNetEmbedder
        embedder = BrainNetEmbedder(in_chans=n_channels, embed_dim=embed_dim)
    else:
        raise ValueError(
            f"Unknown embedder_type={embedder_type!r}. "
            "Expected one of: 'resnet', 'mindid', 'brainnet'."
        )
    return EEGMetricModel(denoiser, embedder)
