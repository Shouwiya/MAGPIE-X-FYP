"""Small training callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from magpie_x.utils.io import ensure_dir


@dataclass
class EarlyStopping:
    patience: int = 6
    min_delta: float = 1e-5
    best_loss: float = float("inf")
    epochs_without_improvement: int = 0

    def step(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


class CheckpointManager:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.best_path = self.output_dir / "best_model.pt"

    def save_best(self, model: torch.nn.Module, metadata: dict) -> None:
        torch.save({"model_state": model.state_dict(), "metadata": metadata}, self.best_path)

    def load_best(self, model: torch.nn.Module) -> dict:
        checkpoint = torch.load(self.best_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        return checkpoint.get("metadata", {})

