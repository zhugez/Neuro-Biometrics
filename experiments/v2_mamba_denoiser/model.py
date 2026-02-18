import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchvision.models import resnet18, resnet34, resnet50
from mamba_ssm import Mamba

BACKBONES = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


class MambaBlock(nn.Module):
    """
    Mamba Block using official mamba-ssm CUDA kernels.
    Wraps mamba_ssm.Mamba to bridge WaveNet's (B, C, L) layout
    with Mamba's expected (B, L, D) layout.
    Includes LayerNorm pre-normalization, dropout, and residual connection.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, C, L) from Conv1d flow."""
        # (B, C, L) -> (B, L, C) for Mamba
        residual = x
        h = x.transpose(1, 2)
        h = self.norm(h)
        h = self.mamba(h)
        h = self.dropout(h)
        # (B, L, C) -> (B, C, L) + residual
        return h.transpose(1, 2) + residual


class WaveNetBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels*2, 3, padding=dilation, dilation=dilation)
        self.conv1x1 = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x):
        h = self.dilated(x)
        h1, h2 = h.chunk(2, dim=1)
        h = torch.tanh(h1) * torch.sigmoid(h2)
        return x + self.conv1x1(h)


class WaveNetDenoiser(nn.Module):
    def __init__(self, channels=4, hidden=64, blocks=3, layers_per_block=4, use_mamba=False):
        super().__init__()
        self.input_conv = nn.Conv1d(channels, hidden, 1)
        
        self.blocks = nn.ModuleList()
        total_layers = blocks * layers_per_block
        mamba_pos = total_layers // 2  # Insert in the middle
        
        current_layer = 0
        for b in range(blocks):
            for i in range(layers_per_block):
                self.blocks.append(WaveNetBlock(hidden, 2**i))
                current_layer += 1
                
                # Check if we should insert Mamba here
                if use_mamba and current_layer == mamba_pos:
                    self.blocks.append(MambaBlock(hidden))
                    print(f"  [Model] Inserted MambaBlock after layer {current_layer}")
                    
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, 1)
        )
    
    def forward(self, x):
        h = self.input_conv(x)
        for block in self.blocks:
            h = block(h)
        return self.output_conv(h)

class ResNetMetricEmbedder(nn.Module):
    def __init__(self, backbone="resnet18", in_chans=4, embed_dim=128, pretrained=True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        base = BACKBONES[backbone](weights=weights)
        base.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nfeat = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(nfeat, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.head(feat)
        return F.normalize(emb, p=2, dim=1)

class EEGMetricModel(nn.Module):
    def __init__(self, filter_model, embedder_model):
        super().__init__()
        self.filter_model = filter_model
        self.embedder_model = embedder_model
        self.denoiser = filter_model
        self.embedder = embedder_model

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        denoised = self.filter_model(x)
        img_like = denoised.unsqueeze(-1)
        emb = self.embedder_model(img_like)
        return denoised, emb

def create_metric_model(backbone="resnet18", n_channels=4, embed_dim=128, pretrained=True, use_mamba=False):
    filter_model = WaveNetDenoiser(channels=n_channels, use_mamba=use_mamba)
    embedder_model = ResNetMetricEmbedder(
        backbone=backbone, 
        in_chans=n_channels, 
        embed_dim=embed_dim, 
        pretrained=pretrained
    )
    return EEGMetricModel(filter_model, embedder_model)
