from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_residual"
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "residual_predictions.csv"


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i == 0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob > lower) & (y_prob <= upper)
        if not np.any(mask):
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - confidence)
    return ece


def safe_metric(metric: str, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        if metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        if metric == "f1":
            return float(f1_score(y_true, y_pred, zero_division=0))
        if metric == "auroc":
            return float(roc_auc_score(y_true, y_prob))
        if metric == "auprc":
            return float(average_precision_score(y_true, y_prob))
        if metric == "mcc":
            return float(matthews_corrcoef(y_true, y_pred))
        if metric == "brier":
            return float(brier_score_loss(y_true, y_prob))
        if metric == "ece":
            return float(expected_calibration_error(y_true, y_prob))
    except Exception:
        return float("nan")
    raise ValueError(metric)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        metric: safe_metric(metric, y_true, y_prob)
        for metric in ["accuracy", "balanced_accuracy", "f1", "auroc", "auprc", "mcc", "brier", "ece"]
    }


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for base_model, base_group in predictions.groupby("base_model_name", sort=True):
        if "variant_id" in base_group.columns:
            baseline_group = base_group.drop_duplicates(subset=["variant_id", "base_model_name"])
        else:
            baseline_group = base_group.drop_duplicates()
        y_true = baseline_group["label"].astype(int).to_numpy()
        baseline_metrics = compute_metrics(y_true, baseline_group["p_tabular"].astype(float).to_numpy())
        rows.append(
            {
                "model": str(base_model),
                "comparison": "tabular_baseline_only",
                "correction_model": "none",
                "n": int(len(baseline_group)),
                **baseline_metrics,
            }
        )

    group_cols = ["base_model_name", "correction_model"]
    for (base_model, correction_model), group in predictions.groupby(group_cols, sort=True):
        y_true = group["label"].astype(int).to_numpy()
        residual_metrics = compute_metrics(y_true, group["p_final"].astype(float).to_numpy())
        rows.append(
            {
                "model": str(base_model),
                "comparison": "residual_correction",
                "correction_model": str(correction_model),
                "n": int(len(group)),
                **residual_metrics,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No rows available."]
    columns = df.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}" if math.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(output_dir: Path, predictions_path: Path, metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    expert_states = sorted(predictions.get("expert_feature_state", pd.Series(dtype=str)).dropna().unique().tolist())
    lines = [
        "# MAGPIE-X Residual Correction Report",
        "",
        "This report compares a strong MAGPIE-style tabular baseline against the first residual-correction stage.",
        "",
        "This is not simple voting. The correction model learns an additive logit-space correction, represented as `z_final = logit_tabular + delta`, and reports `p_final = sigmoid(z_final)`.",
        "",
        "This stage does not train a GNN. VEGN-style graph context is Phase 2 after the residual framework works.",
        "",
        "## Overclaiming Guardrail",
        "",
        "- We do not claim to invent MAGPIE.",
        "- We do not claim to invent AlphaMissense or ESM.",
        "- We do not claim to invent GNNs.",
        "- We do not claim this exact residual fusion is already proven.",
        "- The contribution is evaluating whether independent expert evidence can correct remaining errors of a strong MAGPIE-style baseline.",
        "",
        "## Inputs",
        "",
        f"- Residual predictions: `{predictions_path}`",
        f"- Expert feature state: {', '.join(expert_states) if expert_states else 'not recorded'}",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(markdown_table(metrics))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        "A useful residual model should improve held-out discrimination or error-sensitive metrics without worsening calibration. If no AlphaMissense matches are available, this run is only an uncertainty-plus-variant-type and AF-bin correction baseline and should not be presented as protein-expert fusion.",
            "",
            "The next robustness-focused stages are gene-held-out testing, rare-variant evaluation, imbalanced clinical-style testing, calibration, and later graph/context augmentation.",
        ]
    )
    (output_dir / "RESIDUAL_CORRECTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.predictions.exists():
        raise FileNotFoundError(
            f"Residual predictions not found: {args.predictions}. "
            "Run scripts/train_residual_correction_model.py first."
        )
    predictions = pd.read_csv(args.predictions, low_memory=False)
    required = {"label", "p_tabular", "p_final", "base_model_name", "correction_model"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Residual predictions are missing required columns: {missing}")

    metrics = summarize_predictions(predictions)
    metrics_path = args.output_dir / "residual_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    write_report(args.output_dir, args.predictions, metrics, predictions)
    print(f"Saved residual metrics to {metrics_path}")
    print(f"Saved residual report to {args.output_dir / 'RESIDUAL_CORRECTION_REPORT.md'}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MAGPIE-style tabular baseline probabilities against residual correction.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
