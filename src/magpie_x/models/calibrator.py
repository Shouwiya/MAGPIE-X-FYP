"""Post-hoc calibration models for pathogenicity probabilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from magpie_x.utils.io import ensure_dir


class TemperatureScaler:
    """Single-parameter logit temperature scaling."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        self.temperature = float(initial_temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> "TemperatureScaler":
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.float32)
        log_temp = torch.nn.Parameter(torch.tensor(np.log(max(self.temperature, 1e-3)), dtype=torch.float32))
        optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            temp = torch.exp(log_temp).clamp(1e-3, 100.0)
            loss = loss_fn(logits_t / temp, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temp).detach().clamp(1e-3, 100.0))
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=float) / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.transform_logits(logits)
        return 1.0 / (1.0 + np.exp(-scaled))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        with path.open("wb") as handle:
            pickle.dump({"temperature": self.temperature}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        with Path(path).open("rb") as handle:
            data = pickle.load(handle)
        return cls(initial_temperature=float(data["temperature"]))


class IsotonicCalibrator:
    """Isotonic regression on validation probabilities."""

    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=int))
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(probabilities, dtype=float))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        with path.open("wb") as handle:
            pickle.dump(self.model, handle)

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        obj = cls()
        with Path(path).open("rb") as handle:
            obj.model = pickle.load(handle)
        return obj

