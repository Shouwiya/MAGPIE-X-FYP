"""Pathogenicity prediction metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


def _safe_metric(fn, *args, default: float = float("nan"), **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "auroc": _safe_metric(roc_auc_score, y_true, y_prob),
        "auprc": _safe_metric(average_precision_score, y_true, y_prob),
        "mcc": _safe_metric(matthews_corrcoef, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "f1": _safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "brier": _safe_metric(brier_score_loss, y_true, y_prob),
        "threshold": float(threshold),
        "n": float(len(y_true)),
    }


def threshold_table(y_true: np.ndarray, y_prob: np.ndarray, thresholds: list[float] | np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_prob, threshold=float(threshold))
        rows.append(metrics)
    return pd.DataFrame(rows)


def pr_operating_points(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    padded = np.r_[thresholds, np.nan]
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": padded})

