"""Loss helpers for single-task and optional phenotype multitask training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def binary_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum().clamp_min(1.0)
    negatives = (labels.numel() - labels.sum()).clamp_min(1.0)
    return negatives / positives


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    phenotype_labels: torch.Tensor | None = None,
    phenotype_weight: float = 0.2,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(outputs["logits"], labels.float(), pos_weight=pos_weight)
    if phenotype_labels is not None and "phenotype_logits" in outputs:
        pheno = F.cross_entropy(outputs["phenotype_logits"], phenotype_labels.long())
        return bce + phenotype_weight * pheno
    return bce

