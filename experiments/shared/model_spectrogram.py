import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class SpectrogramMambaBranch(nn.Module):
    def __init__(
        self,
        in_chans: int = 4,
        embed_dim: int = 128,
        d_state: int = 16,
        pretrained: bool = False,
        use_mamba: bool = True,
    ):
        super().__init__()
        if use_mamba and not HAS_MAMBA:
            raise ImportError(
                "mamba-ssm is required for SpectrogramMambaBranch when use_mamba=True"
            )
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.spec_embed_dim = embed_dim
        self.use_mamba = use_mamba
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.freq_norm = nn.LayerNorm(64) if use_mamba else nn.Identity()
        self.freq_mamba = (
            Mamba(d_model=64, d_state=d_state, d_conv=4, expand=2)
            if use_mamba
            else nn.Identity()
        )
        self.temp_norm = nn.LayerNorm(64) if use_mamba else nn.Identity()
        self.temp_mamba = (
            Mamba(d_model=64, d_state=d_state, d_conv=4, expand=2)
            if use_mamba
            else nn.Identity()
        )
        self.head = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        if pretrained:
            self._init_pretrained_conv1()

    def _init_pretrained_conv1(self):
        base_weight = torch.randn(32, 3, 3, 3)
        nn.init.kaiming_normal_(base_weight, mode="fan_out", nonlinearity="relu")
        avg_weight = base_weight.mean(dim=1, keepdim=True)
        init_weight = avg_weight.repeat(1, self.in_chans, 1, 1)
        with torch.no_grad():
            self.conv1.weight.copy_(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected a 4D spectrogram tensor, got shape {tuple(x.shape)}")
        batch_size, channels, freq_bins, time_bins = x.shape
        if channels != self.in_chans:
            raise ValueError(f"Expected {self.in_chans} channels, got {channels}")
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = h.permute(0, 2, 3, 1).reshape(batch_size * time_bins, freq_bins, 64)
        h = self.freq_norm(h)
        h = self.freq_mamba(h)
        h = h.reshape(batch_size, freq_bins, time_bins, 64).permute(0, 3, 1, 2)
        h = h.permute(0, 2, 3, 1).reshape(batch_size * freq_bins, time_bins, 64)
        h = self.temp_norm(h)
        h = self.temp_mamba(h)
        h = h.reshape(batch_size, freq_bins, time_bins, 64).permute(0, 3, 1, 2)
        h = h.mean(dim=(2, 3))
        emb = self.head(h)
        return F.normalize(emb, p=2, dim=1)
