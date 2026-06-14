from __future__ import annotations

import argparse
import math
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
DEFAULT_RESIDUAL_DIR = ROOT / "outputs" / "real" / "magpie_x_residual"
DEFAULT_GNN_DIR = ROOT / "outputs" / "real" / "magpie_x_gnn"
DEFAULT_OOF = DEFAULT_RESIDUAL_DIR / "oof_tabular_predictions.csv"
DEFAULT_ALPHA_RESIDUAL = DEFAULT_RESIDUAL_DIR / "residual_predictions.csv"
DEFAULT_GNN = DEFAULT_GNN_DIR / "gnn_predictions.csv"


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        mask = ((y_prob >= lower) if i == 0 else (y_prob > lower)) & (y_prob <= upper)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metric_fns = {
        "accuracy": lambda: accuracy_score(y_true, y_pred),
        "balanced_accuracy": lambda: balanced_accuracy_score(y_true, y_pred),
        "f1": lambda: f1_score(y_true, y_pred, zero_division=0),
        "auroc": lambda: roc_auc_score(y_true, y_prob),
        "auprc": lambda: average_precision_score(y_true, y_prob),
        "mcc": lambda: matthews_corrcoef(y_true, y_pred),
        "brier": lambda: brier_score_loss(y_true, y_prob),
        "ece": lambda: expected_calibration_error(y_true, y_prob),
    }
    out: dict[str, float] = {}
    for name, fn in metric_fns.items():
        try:
            out[name] = float(fn())
        except Exception:
            out[name] = float("nan")
    return out


def add_result(rows: list[dict[str, object]], name: str, df: pd.DataFrame, prob_col: str, split: str = "all") -> None:
    if df.empty or prob_col not in df.columns or "label" not in df.columns:
        return
    work = df.copy()
    work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce")
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    work = work[work["label"].isin([0, 1]) & work[prob_col].notna()]
    if work.empty:
        return
    rows.append(
        {
            "model": name,
            "split": split,
            "n": int(len(work)),
            **compute_metrics(work["label"].to_numpy(dtype=int), work[prob_col].to_numpy(dtype=float)),
        }
    )


def load_tabular(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    oof = pd.read_csv(path, low_memory=False)
    if "variant_id" in oof.columns:
        oof = oof.drop_duplicates(subset=["variant_id", "model_name"])
    return oof


def load_alpha_residual(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_gnn(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No comparable prediction files were found yet."]
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


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    tabular = load_tabular(args.oof_predictions)
    add_result(rows, "tabular_baseline", tabular, "p_tabular")

    alpha_residual = load_alpha_residual(args.alpha_residual_predictions)
    add_result(rows, "tabular_alpha_residual", alpha_residual, "p_final")

    gnn = load_gnn(args.gnn_predictions)
    if not gnn.empty:
        for split_name, group in gnn.groupby("split", dropna=False):
            split = str(split_name)
            model_mode = str(group["mode"].iloc[0]) if "mode" in group.columns and not group.empty else "unknown"
            if model_mode == "gnn_only":
                add_result(rows, "gnn_only", group, "p_final", split=split)
            elif model_mode == "residual":
                add_result(rows, "tabular_gnn_residual", group, "p_final", split=split)
                if "alphamissense_available" in group.columns and group["alphamissense_available"].fillna(False).any():
                    add_result(rows, "tabular_alpha_gnn_residual", group, "p_final", split=split)

    metrics = pd.DataFrame(rows)
    metrics_path = args.output_dir / "tabular_alpha_gnn_comparison_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    report = [
        "# Tabular Alpha GNN Residual Comparison",
        "",
        "This report compares whichever prediction files already exist. It does not train models.",
        "",
        f"- Tabular OOF: `{args.oof_predictions}`",
        f"- Alpha residual: `{args.alpha_residual_predictions}`",
        f"- GNN predictions: `{args.gnn_predictions}`",
        "",
        "## Metrics",
        "",
        *markdown_table(metrics),
        "",
        "## Notes",
        "",
        "- Use OOF tabular logits for residual models.",
        "- AlphaMissense is auxiliary missense-only evidence, not a ground-truth label.",
        "- GNN results should first be generated only on the reference 5k graph.",
    ]
    report_path = args.output_dir / "TABULAR_ALPHA_GNN_COMPARISON_REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved comparison metrics to {metrics_path}")
    print(f"Saved comparison report to {report_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tabular, AlphaMissense residual, GNN-only, and GNN residual predictions when files exist."
    )
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--alpha-residual-predictions", type=Path, default=DEFAULT_ALPHA_RESIDUAL)
    parser.add_argument("--gnn-predictions", type=Path, default=DEFAULT_GNN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GNN_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
