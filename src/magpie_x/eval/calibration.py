"""Calibration metrics and report helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from magpie_x.eval.metrics import compute_binary_metrics


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - confidence)
    return float(ece)


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"mean_predicted_probability": mean_pred, "fraction_positives": frac_pos})


def calibration_report(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, threshold: float = 0.5) -> dict[str, float]:
    report = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    report["ece"] = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    return report

