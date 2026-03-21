"""
CrossAttentionFusion: Bidirectional cross-attention fusion of two 128-d embeddings.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion of two embeddings.

    Computes:
        attn1 = MultiHeadAttention(Q=e1, K=e2, V=e2)  # e1 queries e2
        attn2 = MultiHeadAttention(Q=e2, K=e1, V=e1)  # e2 queries e1
        concat = [e1, attn1.squeeze(1), attn2.squeeze(1)]  # (B, 384)
        return proj(concat)  # (B, 128), NOT L2 normalized (ArcFace expects unnormalized)
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e1: Embedding from EEG branch, shape (B, 128)
            e2: Embedding from Spectrogram branch, shape (B, 128)
        Returns:
            Fused embedding, shape (B, 128), NOT L2 normalized
        """
        # Add sequence dimension: (B, 128) → (B, 1, 128)
        q1 = e1.unsqueeze(1)
        q2 = e2.unsqueeze(1)

        # Bidirectional cross-attention
        # attn1: e1 queries e2 (Q=e1, K=e2, V=e2)
        attn1, _ = self.attn(q1, q2, q2)
        # attn2: e2 queries e1 (Q=e2, K=e1, V=e1)
        attn2, _ = self.attn(q2, q1, q1)

        # Concatenate: [e1, attn1, attn2]
        concat = torch.cat([e1, attn1.squeeze(1), attn2.squeeze(1)], dim=1)  # (B, 384)

        # Project to embed_dim
        fused = self.proj(concat)  # (B, 128)
        return fused
