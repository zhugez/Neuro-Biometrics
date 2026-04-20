import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_1to2 = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.attn_2to1 = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        q1 = e1.unsqueeze(1)
        q2 = e2.unsqueeze(1)
        attn1, _ = self.attn_1to2(q1, q2, q2)
        attn2, _ = self.attn_2to1(q2, q1, q1)
        concat = torch.cat([e1, attn1.squeeze(1), attn2.squeeze(1)], dim=1)
        return self.proj(concat)
