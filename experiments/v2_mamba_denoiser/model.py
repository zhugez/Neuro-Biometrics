import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from torchvision.models import resnet18, resnet34, resnet50

BACKBONES = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}

class MambaBlock(nn.Module):
    """
    Simplified Pure PyTorch Mamba Block (Selective SSM)
    Uses a standard (slow) sequential scan for compatibility without CUDA kernels.
    Enhanced with LayerNorm and optional dropout for training stability.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = Variable(torch.ceil(torch.tensor(d_model / 16))).int().item()
        self.d_state = d_state
        
        # Layer normalization for training stability
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    @torch.jit.export
    def selective_scan(self, x_branch, dA, dB, C_ssm, d_inner: int, d_state: int):
        """
        JIT-compiled scan loop for speedup.
        """
        batch, seq_len, _ = x_branch.shape
        h = torch.zeros(batch, d_inner, d_state, device=x_branch.device)
        ys = torch.jit.annotate(List[torch.Tensor], [])
        
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x_branch[:, t].unsqueeze(-1)
            # Equivalent to einsum('bls,bds->bd') but JIT-friendly
            # h: (B, D, N), C: (B, N) -> (B, 1, N)
            # y = sum(h * C, dim=-1)
            y = (h * C_ssm[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y)
            
        return torch.stack(ys, dim=1)

    def forward(self, x):
        """
        x: (B, C, L) or (B, L, C). We assume (B, C, L) from Conv1d flows.
        """
        # Squeeze inputs: (B, C, L) -> (B, L, C)
        x_in = x.transpose(1, 2)
        batch, seq_len, _ = x_in.shape
        
        # Apply layer normalization
        x_normed = self.norm(x_in)

        xz = self.in_proj(x_normed) 
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # Conv1d expects (B, C, L)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len] # Causal padding
        x_branch = F.silu(x_branch)
        x_branch = x_branch.transpose(1, 2) # Back to (B, L, C)

        # Selective Scan (Simplified Sequential)
        # x_dbl: (B, L, dt_rank + 2*d_state)
        x_dbl = self.x_proj(x_branch)  
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)) # (B, L, d_inner)
        
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        # Discretize A -> dA (B, L, d_inner, d_state)
        # We need to broadcast A to (B, L, d_inner, d_state)
        # dA = exp(dt * A)
        dA = torch.exp(torch.einsum('bld,ds->blds', dt, A))
        
        # Discretize B -> dB (B, L, d_inner, d_state)
        # dB = dt * B
        # B_ssm is (B, L, d_state). Broadcast to d_inner.
        dB = torch.einsum('bld,bls->blds', dt, B_ssm)
        
        # Scan (JIT Optimized)
        y_ssm = self.selective_scan(x_branch, dA, dB, C_ssm, self.d_inner, self.d_state)
        
        # Add D residual
        y_ssm = y_ssm + (x_branch * self.D)
        
        # Gating
        y_out = y_ssm * F.silu(z_branch)
        
        out = self.out_proj(y_out)
        
        # Apply dropout for regularization
        out = self.dropout(out)
        
        return out.transpose(1, 2) + x # Residual + (B, C, L) output


from torch.autograd import Variable


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
