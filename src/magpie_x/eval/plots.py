"""Publication-oriented evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from magpie_x.eval.calibration import reliability_curve
from magpie_x.utils.io import ensure_dir


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str | Path, filename: str = "roc_curve.png") -> Path:
    output_dir = ensure_dir(output_dir)
    path = output_dir / filename
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str | Path, filename: str = "pr_curve.png") -> Path:
    output_dir = ensure_dir(output_dir)
    path = output_dir / filename
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("Precision-recall curve")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str | Path,
    filename: str = "calibration_curve.png",
    n_bins: int = 10,
) -> Path:
    output_dir = ensure_dir(output_dir)
    path = output_dir / filename
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Perfect calibration")
    ax.plot(curve["mean_predicted_probability"], curve["fraction_positives"], marker="o", label="MAGPIE_X")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction pathogenic")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str | Path,
    filename: str = "confusion_matrix.png",
    threshold: float = 0.5,
) -> Path:
    output_dir = ensure_dir(output_dir)
    path = output_dir / filename
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=160)
    ConfusionMatrixDisplay(cm, display_labels=["benign", "pathogenic"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix @ {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    output_dir: str | Path,
    filename: str = "feature_importance.csv",
    top_k: int = 40,
) -> Path:
    output_dir = ensure_dir(output_dir)
    path = output_dir / filename
    table = pd.DataFrame({"feature_name": feature_names, "importance": np.asarray(importances, dtype=float)})
    table = table.sort_values("importance", ascending=False).head(top_k)
    table.to_csv(path, index=False)
    return path

