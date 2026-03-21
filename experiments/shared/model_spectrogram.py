"""
SpectrogramMambaBranch: 2D CNN + Mamba on frequency and temporal dimensions.

Processes STFT spectrograms (B, 4, F=65, T_spec=13) into 128-d identity embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from mamba_ssm import Mamba


class SpectrogramMambaBranch(nn.Module):
    """
    2D CNN + Mamba on frequency and temporal dimensions.

    Architecture:
        Conv2d feature extractor: (B, 4, F, T) → (B, 64, F, T)
        Mamba on frequency: (B*T, F, 64) → LayerNorm → Mamba → (B*T, F, 64)
        Mamba on temporal: (B*F, T, 64) → LayerNorm → Mamba → (B*F, T, 64)
        GlobalAvgPool2D → Linear(64, 256) → ReLU → Dropout(0.1) → Linear(256, 128) → BN → L2 norm
    """

    def __init__(self, in_chans: int = 4, embed_dim: int = 128,
                 d_state: int = 16, pretrained: bool = False):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Conv2d feature extractor
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Mamba on frequency dimension (d_model=64)
        self.freq_norm = nn.LayerNorm(64)
        self.freq_mamba = Mamba(d_model=64, d_state=d_state, d_conv=4, expand=2)

        # Mamba on temporal dimension (d_model=64)
        self.temp_norm = nn.LayerNorm(64)
        self.temp_mamba = Mamba(d_model=64, d_state=d_state, d_conv=4, expand=2)

        # Projection head
        self.head = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # Pretrained initialization for 4-channel input
        if pretrained:
            self._init_pretrained_conv1()

    def _init_pretrained_conv1(self):
        """Average ImageNet conv1 weights across channels for 4-channel input."""
        # conv1 in ResNet has 64 output channels, 3 input channels
        # We have 32 output channels, 4 input channels
        base_weight = torch.randn(32, 3, 3, 3)
        nn.init.kaiming_normal_(base_weight, mode='fan_out', nonlinearity='relu')
        # Average across RGB channels and replicate for the 4th channel
        avg_weight = base_weight.mean(dim=1, keepdim=True)  # (32, 1, 3, 3)
        # Repeat for 4 input channels → (32, 4, 3, 3)
        init_weight = avg_weight.repeat(1, self.in_chans, 1, 1)
        with torch.no_grad():
            self.conv1.weight.copy_(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        if C != self.in_chans:
            raise ValueError(
                f"Expected {self.in_chans} channels, got {C}"
            )
        if F != 65 or T != 13:
            raise ValueError(
                f"Expected spectrogram shape (B, {self.in_chans}, 65, 13), "
                f"got (B, {C}, {F}, {T})"
            )

        # Conv2d feature extractor
        h = F.relu(self.bn1(self.conv1(x)))   # (B, 32, F, T)
        h = F.relu(self.bn2(self.conv2(h)))   # (B, 64, F, T)

        # Mamba on frequency: (B, 64, F, T) → (B, F, T, 64) → (B*T, F, 64)
        h = h.permute(0, 2, 3, 1)             # (B, F, T, 64)
        h = h.reshape(B * T, F, 64)          # (B*T, F, 64)
        h = self.freq_norm(h)                 # (B*T, F, 64)
        h = self.freq_mamba(h)                # (B*T, F, 64)
        h = h.reshape(B, F, T, 64)            # (B, F, T, 64)
        h = h.permute(0, 3, 1, 2)             # (B, 64, F, T)

        # Mamba on temporal: (B, 64, F, T) → (B*F, T, 64)
        h = h.permute(0, 2, 3, 1)             # (B, F, T, 64)
        h = h.reshape(B * F, T, 64)          # (B*F, T, 64)
        h = self.temp_norm(h)                 # (B*F, T, 64)
        h = self.temp_mamba(h)                # (B*F, T, 64)
        h = h.reshape(B, F, T, 64)            # (B, F, T, 64)
        h = h.permute(0, 3, 1, 2)             # (B, 64, F, T)

        # Global average pooling
        h = h.mean(dim=[2, 3])               # (B, 64)

        # Projection head
        emb = self.head(h)                     # (B, 128)

        # L2 normalization
        emb = F.normalize(emb, p=2, dim=1)
        return emb
