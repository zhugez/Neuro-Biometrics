import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchvision.models import resnet18, resnet34, resnet50

BACKBONES = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}

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
    def __init__(self, channels=4, hidden=64, blocks=3, layers_per_block=4):
        super().__init__()
        self.input_conv = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.ModuleList([
            WaveNetBlock(hidden, 2**i) 
            for _ in range(blocks) 
            for i in range(layers_per_block)
        ])
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
        # FIX: smaller conv1 (3x3 stride=1) + remove maxpool for small 2D EEG inputs
        base.conv1 = nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        nfeat = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        # Deeper projection head for better embedding quality
        self.head = nn.Sequential(
            nn.Linear(nfeat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

    @staticmethod
    def _find_2d_shape(n):
        """Find H, W close to square root for proper 2D spatial structure.
        E.g. 800 -> (25, 32), 64 -> (8, 8)"""
        sqrt_n = int(n ** 0.5)
        for h in range(sqrt_n, 0, -1):
            if n % h == 0:
                return h, n // h
        return 1, n

    def forward(self, x):
        # FIX: reshape (B, C, T) -> (B, C, H, W) for proper 2D spatial convolutions
        # Old code used unsqueeze(-1) giving (B,C,T,1) with width=1 -> ResNet can't learn
        B, C, T = x.shape
        H, W = self._find_2d_shape(T)
        x = x.view(B, C, H, W)
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
        emb = self.embedder_model(denoised)  # embedder now handles (B,C,T)->2D internally
        return denoised, emb

def create_metric_model(backbone="resnet18", n_channels=4, embed_dim=128, pretrained=True):
    filter_model = WaveNetDenoiser(channels=n_channels)
    embedder_model = ResNetMetricEmbedder(
        backbone=backbone, 
        in_chans=n_channels, 
        embed_dim=embed_dim, 
        pretrained=pretrained
    )
    return EEGMetricModel(filter_model, embedder_model)
