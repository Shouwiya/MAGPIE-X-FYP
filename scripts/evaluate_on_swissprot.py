from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_sample_size_benchmark import (
    DEFAULT_INPUT as DEFAULT_CLINVAR_FEATURE_TABLE,
    MODEL_CHOICES,
    balanced_sample,
    build_model,
    build_preprocessor,
    infer_features,
    load_feature_table,
    normalize_missing_markers,
    predict_probability,
)

DEFAULT_SWISSPROT_FEATURE_TABLE = ROOT / "outputs" / "real" / "swissprot" / "swissprot_feature_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "swissprot"
METRIC_COLUMNS = [
    "accuracy",
    "f1",
    "precision",
    "recall_sensitivity",
    "specificity",
    "mcc",
    "auroc",
    "auprc",
    "brier",
    "ece",
]
METRIC_TUPLE = "(Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, MCC, AUROC, AUPRC, Brier, ECE)"


def variant_key_frame(df: pd.DataFrame) -> pd.Series:
    return (
        df["chrom"].astype(str).str.replace("^chr", "", regex=True)
        + ":"
        + pd.to_numeric(df["pos"], errors="coerce").astype("Int64").astype(str)
        + ":"
        + df["ref"].astype(str)
        + ">"
        + df["alt"].astype(str)
    )


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left = bins[idx]
        right = bins[idx + 1]
        mask = (y_prob >= left) & (y_prob <= right if idx == n_bins - 1 else y_prob < right)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(y_true[mask])) - float(np.mean(y_prob[mask])))
    return float(ece)


def safe_metric(fn, *args, default: float = float("nan"), **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> dict[str, float | int]:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "mcc": safe_metric(matthews_corrcoef, y_true, y_pred),
        "auroc": safe_metric(roc_auc_score, y_true, y_prob),
        "auprc": safe_metric(average_precision_score, y_true, y_prob),
        "brier": safe_metric(brier_score_loss, y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}" if math.isfinite(float(value)) else "-"
    return str(value)


def result_tuple(metrics: dict[str, object]) -> str:
    return "(" + ", ".join(fmt(metrics.get(col)) for col in METRIC_COLUMNS) + ")"


def load_pickle_model(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "label" not in df.columns:
        return {}
    labels = pd.to_numeric(df["label"], errors="coerce")
    return {str(int(k)): int(v) for k, v in labels.value_counts().sort_index().items()}


def build_subsets(swissprot: pd.DataFrame, rare_af_threshold: float) -> dict[str, pd.DataFrame]:
    subsets: dict[str, pd.DataFrame] = {"all_joined": swissprot}
    if "exact_clinvar_feature_overlap" in swissprot.columns:
        exact = swissprot["exact_clinvar_feature_overlap"].astype(str).str.lower().isin({"true", "1", "yes"})
        subsets["no_exact_clinvar_overlap"] = swissprot.loc[~exact].copy()
    else:
        subsets["no_exact_clinvar_overlap"] = swissprot.iloc[0:0].copy()

    if "AF" in swissprot.columns:
        af = pd.to_numeric(swissprot["AF"], errors="coerce")
        subsets[f"rare_af_le_{rare_af_threshold:g}"] = swissprot.loc[af.notna() & (af <= rare_af_threshold)].copy()
    else:
        subsets[f"rare_af_le_{rare_af_threshold:g}"] = swissprot.iloc[0:0].copy()

    gene_overlap_col = None
    for candidate in ("gene_seen_in_clinvar_feature_table", "gene_seen_in_clinvar_feature_table_audit"):
        if candidate in swissprot.columns:
            gene_overlap_col = candidate
            break
    if gene_overlap_col:
        gene_overlap = swissprot[gene_overlap_col].astype(str).str.lower().isin({"true", "1", "yes"})
        subsets["no_gene_overlap"] = swissprot.loc[~gene_overlap].copy()
    else:
        subsets["no_gene_overlap"] = swissprot.iloc[0:0].copy()
    return subsets


def train_reference_models(
    swissprot: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, Pipeline], list[str], dict[str, object]]:
    clinvar = load_feature_table(args.clinvar_features)
    excluded_exact = 0
    if args.exclude_swissprot_exact_from_training and {"chrom", "pos", "ref", "alt"}.issubset(swissprot.columns):
        swiss_keys = set(variant_key_frame(swissprot).dropna().astype(str))
        clinvar_keys = variant_key_frame(clinvar)
        keep_mask = ~clinvar_keys.isin(swiss_keys)
        excluded_exact = int((~keep_mask).sum())
        clinvar = clinvar.loc[keep_mask].reset_index(drop=True)

    train_df = balanced_sample(clinvar, sample_size=args.clinvar_train_size, seed=args.seed + args.clinvar_train_size)
    spec = infer_features(train_df, args.max_categorical_cardinality)
    x_train = train_df[spec.feature_columns]
    y_train = train_df["label"].to_numpy(dtype=int)
    models: dict[str, Pipeline] = {}
    for model_name in args.models:
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(spec)),
                ("model", build_model(model_name, args.seed)),
            ]
        )
        pipeline.fit(x_train, y_train)
        models[model_name] = pipeline
    metadata = {
        "training_mode": "clinvar_reference_balanced_sample",
        "clinvar_features": str(args.clinvar_features),
        "clinvar_train_size": int(args.clinvar_train_size),
        "models": args.models,
        "seed": int(args.seed),
        "excluded_clinvar_training_rows_with_exact_swissprot_keys": excluded_exact,
        "feature_columns": spec.feature_columns,
        "synthetic_or_toy_data_used": False,
    }
    return models, spec.feature_columns, metadata


def evaluate_subset(
    model_name: str,
    model,
    subset_name: str,
    subset: pd.DataFrame,
    feature_columns: list[str],
    ece_bins: int,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    row: dict[str, object] = {
        "dataset": "SwissProt/HUMSAVAR",
        "subset": subset_name,
        "model": model_name,
        "n": int(len(subset)),
        "label_counts": json.dumps(label_counts(subset), sort_keys=True),
    }
    if subset.empty:
        row["status"] = "skipped_empty_subset"
        row["result_tuple"] = result_tuple(row)
        return row, None
    if subset["label"].nunique() < 2:
        row["status"] = "evaluated_single_class_metrics_limited"
    else:
        row["status"] = "evaluated"

    x_eval = subset.copy()
    for column in feature_columns:
        if column not in x_eval.columns:
            x_eval[column] = np.nan
    y_true = subset["label"].to_numpy(dtype=int)
    y_prob = predict_probability(model, x_eval[feature_columns])
    metrics = compute_metrics(y_true, y_prob, n_bins=ece_bins)
    row.update(metrics)
    row["result_tuple"] = result_tuple(row)

    prediction_cols = [col for col in ["variant_id", "chrom", "pos", "ref", "alt", "gene", "swissprot_gene", "label"] if col in subset.columns]
    predictions = subset[prediction_cols].copy()
    predictions["subset"] = subset_name
    predictions["model"] = model_name
    predictions["y_prob_pathogenic"] = y_prob
    predictions["y_pred"] = (y_prob >= 0.5).astype(int)
    return row, predictions


def write_report(path: Path, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    lines = [
        "# SwissProt/HUMSAVAR Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Status: completed reference-model external validation on feasible feature-reuse subsets.",
        "",
        f"Metric tuple: `{METRIC_TUPLE}`",
        "",
        "SwissProt/HUMSAVAR external validation is separate from random ClinVar split and is used to test generalization.",
        "",
        "## Training Scope",
        "",
        f"- Training mode: `{metadata.get('training_mode')}`",
        f"- ClinVar train size: {metadata.get('clinvar_train_size')}",
        f"- Exact SwissProt-key rows excluded from ClinVar training pool: {metadata.get('excluded_clinvar_training_rows_with_exact_swissprot_keys')}",
        "- SwissProt labels were not used for training.",
        "",
        "## Metric Summary",
        "",
        "| subset | model | n | status | result_tuple |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['subset']} | {row['model']} | {row['n']} | {row['status']} | {row['result_tuple']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- The current SwissProt feature table is built by feature reuse from exact ClinVar feature-table overlaps.",
            "- Therefore, `all_joined` is useful as a pipeline/external-label check but is not a fully independent no-overlap validation.",
            "- `no_exact_clinvar_overlap` is expected to be empty until real annotation is run for mapped variants that did not join to the ClinVar feature table.",
            "- Final claims still require circularity ablation, gene-held-out validation, temporal validation, and a stronger external validation table.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "SWISSPROT_EVALUATION_REPORT.md"
    metrics_path = args.output_dir / "swissprot_metrics.csv"

    if not args.swissprot_features.exists():
        raise FileNotFoundError(f"SwissProt feature table not found: {args.swissprot_features}")

    swissprot = pd.read_csv(args.swissprot_features, dtype={"chrom": str}, low_memory=False)
    swissprot = normalize_missing_markers(swissprot)
    if "label" not in swissprot.columns:
        raise ValueError("SwissProt feature table must contain a label column.")
    swissprot["label"] = pd.to_numeric(swissprot["label"], errors="coerce")
    swissprot = swissprot[swissprot["label"].isin([0, 1])].copy()
    swissprot["label"] = swissprot["label"].astype(int)

    if args.model_path:
        model = load_pickle_model(args.model_path)
        models = {Path(args.model_path).stem: model}
        if hasattr(model, "feature_names_in_"):
            feature_columns = list(model.feature_names_in_)
        else:
            feature_columns = [col for col in swissprot.columns if col != "label"]
        metadata = {
            "training_mode": "loaded_model",
            "model_path": str(args.model_path),
            "feature_columns": feature_columns,
            "synthetic_or_toy_data_used": False,
        }
    elif args.train_reference:
        models, feature_columns, metadata = train_reference_models(swissprot, args)
    else:
        raise ValueError("No compatible model supplied. Use --model-path or explicitly approve small reference training with --train-reference.")

    subsets = build_subsets(swissprot, args.rare_af_threshold)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for subset_name, subset in subsets.items():
        for model_name, model in models.items():
            row, predictions = evaluate_subset(model_name, model, subset_name, subset, feature_columns, args.ece_bins)
            metric_rows.append(row)
            if predictions is not None:
                prediction_frames.append(predictions)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(metrics_path, index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(args.output_dir / "swissprot_predictions.csv", index=False)

    metadata.update(
        {
            "swissprot_features": str(args.swissprot_features),
            "output_metrics": str(metrics_path),
            "rare_af_threshold": float(args.rare_af_threshold),
            "ece_definition": "Positive-class calibration ECE over y_prob_pathogenic.",
            "metric_tuple_definition": METRIC_TUPLE,
        }
    )
    (args.output_dir / "swissprot_evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    write_report(report_path, metric_rows, metadata)
    print(f"Wrote {metrics_path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate tabular ClinVar reference models on SwissProt/HUMSAVAR feature rows.")
    parser.add_argument("--swissprot-features", type=Path, default=DEFAULT_SWISSPROT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--train-reference", action="store_true", help="Explicitly train small ClinVar reference models before external validation.")
    parser.add_argument("--clinvar-features", type=Path, default=DEFAULT_CLINVAR_FEATURE_TABLE)
    parser.add_argument("--clinvar-train-size", type=int, default=5_000)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--rare-af-threshold", type=float, default=0.01)
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    parser.add_argument("--exclude-swissprot-exact-from-training", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
