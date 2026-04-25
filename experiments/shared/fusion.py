import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.token_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([e1, e2], dim=1)
        attn_out, _ = self.cross_attn(
            self.token_norm(tokens),
            self.token_norm(tokens),
            self.token_norm(tokens),
            need_weights=False,
        )
        eeg_ctx, spec_ctx = attn_out[:, 0], attn_out[:, 1]
        gate = self.gate(torch.cat([e1, e2], dim=1))
        mixed = gate * eeg_ctx + (1.0 - gate) * spec_ctx
        fused = self.out(mixed + 0.5 * (e1 + e2))
        return F.normalize(fused, p=2, dim=1)
