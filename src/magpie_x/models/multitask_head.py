"""Binary and optional phenotype heads."""

from __future__ import annotations

import torch
from torch import nn


class MultiTaskHead(nn.Module):
    def __init__(self, input_dim: int, phenotype_groups: int | None = None) -> None:
        super().__init__()
        self.binary = nn.Linear(input_dim, 1)
        self.phenotype = nn.Linear(input_dim, phenotype_groups) if phenotype_groups else None

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {"logits": self.binary(embedding).squeeze(-1)}
        if self.phenotype is not None:
            out["phenotype_logits"] = self.phenotype(embedding)
        return out

