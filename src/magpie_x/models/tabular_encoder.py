"""Tabular neural baseline for variant pathogenicity."""

from __future__ import annotations

import torch
from torch import nn


class TabularEncoder(nn.Module):
    """MLP encoder for standardized genomic annotations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (96, 64),
        dropout: float = 0.15,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden
        self.network = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TabularClassifier(nn.Module):
    """Binary pathogenicity classifier using only tabular annotations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (96, 64),
        dropout: float = 0.15,
        batch_norm: bool = True,
        phenotype_groups: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = TabularEncoder(input_dim, hidden_dims, dropout, batch_norm)
        self.binary_head = nn.Linear(self.encoder.output_dim, 1)
        self.phenotype_head = nn.Linear(self.encoder.output_dim, phenotype_groups) if phenotype_groups else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encoder(x)
        out = {"logits": self.binary_head(embedding).squeeze(-1), "embedding": embedding}
        if self.phenotype_head is not None:
            out["phenotype_logits"] = self.phenotype_head(embedding)
        return out

