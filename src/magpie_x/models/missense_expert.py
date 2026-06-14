"""Lightweight fusion for precomputed missense foundation/expert scores."""

from __future__ import annotations

import torch
from torch import nn


class MissenseExpertFusion(nn.Module):
    """Project precomputed missense expert scores and fuse them with embeddings."""

    def __init__(self, base_dim: int, expert_dim: int, projection_dim: int = 24, dropout: float = 0.1) -> None:
        super().__init__()
        self.expert_dim = expert_dim
        self.project = nn.Sequential(
            nn.Linear(expert_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(base_dim + projection_dim, base_dim),
            nn.ReLU(),
        )

    def forward(self, base_embedding: torch.Tensor, expert_features: torch.Tensor, missense_mask: torch.Tensor | None = None) -> torch.Tensor:
        expert = torch.nan_to_num(expert_features, nan=0.0)
        if missense_mask is not None:
            expert = expert * missense_mask.float().view(-1, 1)
        projected = self.project(expert)
        return self.fuse(torch.cat([base_embedding, projected], dim=-1))

