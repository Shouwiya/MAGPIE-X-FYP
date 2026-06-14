from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "sample_size_benchmark"

FORBIDDEN_FALLBACK_FILES = {
    (ROOT / "outputs" / "real" / "processed" / "annovar.csv").resolve(),
    (ROOT / "outputs" / "real" / "processed" / "dbnsfp.csv").resolve(),
}

IDENTIFIER_OR_LEAKAGE_COLUMNS = {
    "label",
    "variant_id",
    "clin_sig_raw",
    "clinical_significance",
    "review_status",
    "source_file",
    "assembly",
    "chrom",
    "pos",
    "end",
    "ref",
    "alt",
    "gene",
    "gene_annovar",
    "GeneDetail.refGene",
    "GTEx_V8_gene",
    "aa_change_refgene",
    "Interpro_domain",
    "consequence",
}

NUMERIC_PATTERNS = (
    "AF",
    "score",
    "rankscore",
    "CADD",
    "DANN",
    "Eigen",
    "GenoCanyon",
    "fitCons",
    "confidence",
    "LINSIGHT",
    "GERP",
    "phyloP",
    "phastCons",
    "SiPhy",
    "length",
    "change",
)

MISSING_MARKERS = {"", "-", ".", "NA", "N/A", "nan", "None"}
MISSING_MARKERS_LOWER = {marker.lower() for marker in MISSING_MARKERS}
NUMERIC_CONVERTIBLE_THRESHOLD = 0.80

MODEL_CHOICES = ("xgboost", "lightgbm", "random_forest")
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
METRIC_TUPLE_DEFINITION = (
    "(Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, "
    "MCC, AUROC, AUPRC, Brier, ECE)"
)


@dataclass(frozen=True)
class FeatureSpec:
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    dropped_columns: dict[str, str]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def check_input_path(path: Path) -> None:
    resolved = path.resolve()
    if resolved in FORBIDDEN_FALLBACK_FILES:
        raise ValueError(f"Refusing to use fallback processed file as input: {path}")
    if "toy" in {part.lower() for part in resolved.parts}:
        raise ValueError(f"Refusing to use toy/synthetic path as input: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Input feature table not found: {path}")
    if path.name != "basic_feature_table.csv":
        raise ValueError("This benchmark expects the real basic_feature_table.csv feature table.")


def normalize_missing_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in MISSING_MARKERS or stripped.lower() in MISSING_MARKERS_LOWER:
            return np.nan
        return stripped
    return value


def normalize_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_object_dtype(out[column]) or pd.api.types.is_string_dtype(out[column]):
            out[column] = out[column].map(normalize_missing_value)
    return out


def load_feature_table(path: Path) -> pd.DataFrame:
    check_input_path(path)
    df = pd.read_csv(path, low_memory=False, na_values=[".", "NA", "nan", ""])
    df = normalize_missing_markers(df)
    if "label" not in df.columns:
        raise ValueError("Input feature table must contain a binary label column.")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    if df.empty:
        raise ValueError("No valid binary labels were found.")
    return df.reset_index(drop=True)


def is_prediction_label_column(column: str) -> bool:
    lower = column.lower()
    return lower.endswith("_pred") or lower.endswith("-pred")


def is_numeric_feature_name(column: str) -> bool:
    if column.startswith("consequence_") or column.startswith("is_"):
        return True
    if is_prediction_label_column(column):
        return False
    return any(pattern.lower() in column.lower() for pattern in NUMERIC_PATTERNS)


def numeric_conversion_fraction(series: pd.Series) -> tuple[pd.Series, float, int]:
    non_null = series.notna()
    n_non_null = int(non_null.sum())
    converted = pd.to_numeric(series, errors="coerce")
    if n_non_null == 0:
        return converted, 0.0, 0
    fraction = float(converted[non_null].notna().mean())
    return converted, fraction, n_non_null


def maybe_add_categorical(
    column: str,
    series: pd.Series,
    categorical: list[str],
    dropped: dict[str, str],
    max_categorical_cardinality: int,
    *,
    required_categorical: bool = False,
) -> None:
    cardinality = int(series.nunique(dropna=True))
    if required_categorical and cardinality == 0:
        dropped[column] = "all_missing_prediction_label"
        return
    if 1 <= cardinality <= max_categorical_cardinality:
        categorical.append(column)
        return
    if cardinality == 0:
        dropped[column] = "all_missing"
    else:
        dropped[column] = f"high_cardinality_or_unusable_text_cardinality_{cardinality}"


def infer_features(df: pd.DataFrame, max_categorical_cardinality: int) -> FeatureSpec:
    dropped: dict[str, str] = {}
    numeric: list[str] = []
    categorical: list[str] = []

    for column in df.columns:
        if column in IDENTIFIER_OR_LEAKAGE_COLUMNS:
            dropped[column] = "identifier_or_direct_leakage"
            continue
        if column == "origin":
            dropped[column] = "clinvar_metadata_not_model_feature"
            continue

        if is_prediction_label_column(column):
            maybe_add_categorical(
                column,
                df[column],
                categorical,
                dropped,
                max_categorical_cardinality,
                required_categorical=True,
            )
            continue

        converted, conversion_fraction, n_non_null = numeric_conversion_fraction(df[column])
        if is_numeric_feature_name(column):
            if conversion_fraction >= NUMERIC_CONVERTIBLE_THRESHOLD:
                df[column] = converted
                numeric.append(column)
            elif n_non_null == 0:
                dropped[column] = "all_missing_numeric_candidate"
            else:
                dropped[column] = f"numeric_name_but_only_{conversion_fraction:.2f}_convertible"
            continue

        if conversion_fraction >= NUMERIC_CONVERTIBLE_THRESHOLD:
            df[column] = converted
            numeric.append(column)
            continue

        maybe_add_categorical(column, df[column], categorical, dropped, max_categorical_cardinality)

    feature_columns = [*numeric, *categorical]
    if not feature_columns:
        raise ValueError("No usable feature columns were inferred.")
    return FeatureSpec(
        feature_columns=feature_columns,
        numeric_columns=numeric,
        categorical_columns=categorical,
        dropped_columns=dropped,
    )


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), spec.numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                spec.categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def require_package(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(f"{module_name} is required for the selected model. Install it with: {install_hint}") from exc


def build_model(model_name: str, seed: int):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "xgboost":
        xgboost = require_package("xgboost", "pip install xgboost")
        return xgboost.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "lightgbm":
        lightgbm = require_package("lightgbm", "pip install lightgbm")
        return lightgbm.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def balanced_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size % 2 != 0:
        raise ValueError("Sample size must be a positive even integer.")
    counts = df["label"].value_counts()
    if 0 not in counts or 1 not in counts:
        raise ValueError("Balanced sampling requires both benign and pathogenic classes.")
    per_class = sample_size // 2
    available_per_class = int(min(counts.loc[0], counts.loc[1]))
    if per_class > available_per_class:
        raise ValueError(
            f"Requested {sample_size:,} rows, but only {available_per_class:,} rows per class are available."
        )
    sampled = pd.concat(
        [
            df[df["label"] == 0].sample(n=per_class, random_state=seed),
            df[df["label"] == 1].sample(n=per_class, random_state=seed + 1),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_splits(y: np.ndarray, seed: int, test_size: float, cv_folds: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    indices = np.arange(len(y))
    if cv_folds and cv_folds > 1:
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        return [(fold, train_idx, test_idx) for fold, (train_idx, test_idx) in enumerate(splitter.split(indices, y), start=1)]
    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=y, random_state=seed)
    return [(1, np.sort(train_idx), np.sort(test_idx))]


def predict_probability(pipeline: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x_test)[:, 1]
    decision = pipeline.decision_function(x_test)
    return 1.0 / (1.0 + np.exp(-decision))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        observed_positive_rate = float(np.mean(y_true[mask]))
        mean_predicted_probability = float(np.mean(y_prob[mask]))
        ece += float(np.mean(mask)) * abs(observed_positive_rate - mean_predicted_probability)
    return float(ece)


def confidence_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    correct = (y_pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return float(ece)


def safe_metric(fn, *args, default: float = float("nan"), **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
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
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "confidence_ece": confidence_expected_calibration_error(y_true, y_prob, n_bins),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def format_metric(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}" if math.isfinite(float(value)) else "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def result_tuple(metrics: dict[str, object] | pd.Series) -> str:
    values = [format_metric(metrics[col]) for col in METRIC_COLUMNS]
    return "(" + ", ".join(values) + ")"


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    grouped = metrics.groupby(["sample_size", "model"], as_index=False)[METRIC_COLUMNS].agg(["mean", "std"])
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
    summary = grouped.reset_index(drop=True)
    tuple_columns = [f"{metric}_mean" for metric in METRIC_COLUMNS]
    summary["mean_result_tuple"] = summary.apply(lambda row: "(" + ", ".join(format_metric(row[col]) for col in tuple_columns) + ")", axis=1)
    return summary


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> list[str]:
    if df.empty:
        return ["No rows available."]
    shown = df.head(max_rows)
    columns = shown.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(format_metric(row[col]) for col in columns) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... | {len(df) - max_rows} additional rows omitted from report preview |")
    return lines


def write_report(
    output_dir: Path,
    *,
    input_path: Path,
    df: pd.DataFrame,
    sample_sizes: list[int],
    models: list[str],
    seed: int,
    cv_folds: int,
    test_size: float,
    summary: pd.DataFrame | None,
    dry_run: bool,
    feasibility: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sample-Size Benchmark Report",
        "",
        f"Status: {'dry run only; no models were trained' if dry_run else 'completed run'}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Input feature table: `{input_path}`",
        "",
        "This benchmark uses real ClinVar-derived feature rows only. It does not create synthetic or toy results.",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
        "## Requested Configuration",
        "",
        f"- Sample sizes: {', '.join(format(size, ',') for size in sample_sizes)}",
        f"- Models: {', '.join(models)}",
        f"- Seed: {seed}",
        f"- Split: {'stratified cross-validation' if cv_folds and cv_folds > 1 else 'single stratified train/test split'}",
        f"- CV folds: {cv_folds}",
        f"- Test size: {test_size}",
        "",
        "## Feasibility",
        "",
    ]
    lines.extend(markdown_table(feasibility))
    lines.append("")

    if summary is not None and not summary.empty:
        lines.extend(["## Metric Summary", ""])
        preview_cols = ["sample_size", "model", "mean_result_tuple"]
        lines.extend(markdown_table(summary[preview_cols]))
        lines.append("")
        lines.extend(
            [
                "Full fold-level metrics are written to `sample_size_metrics.csv`.",
                "Future audited runs also write per-test-row predictions to `sample_size_predictions.csv`.",
                "",
            ]
        )

    counts = df["label"].value_counts().to_dict()
    lines.extend(
        [
            "## Data Counts",
            "",
            f"- Benign label 0 rows: {int(counts.get(0, 0)):,}",
            f"- Pathogenic label 1 rows: {int(counts.get(1, 0)):,}",
            "",
            "## Guardrails",
            "",
            "- Do not treat these results as final until circularity ablation, gene-held-out, temporal, rare-variant, calibration, and external validation are complete.",
            "- If a sample size is infeasible, report it as infeasible instead of substituting a smaller sample.",
            "- Use `-` for metrics that were not computed or cannot be computed.",
            "",
        ]
    )
    (output_dir / "SAMPLE_SIZE_BENCHMARK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def build_feasibility(df: pd.DataFrame, sample_sizes: list[int]) -> pd.DataFrame:
    counts = df["label"].value_counts()
    available_per_class = int(min(counts.get(0, 0), counts.get(1, 0)))
    rows = []
    for sample_size in sample_sizes:
        required_per_class = sample_size // 2 if sample_size % 2 == 0 else float("nan")
        feasible = sample_size > 0 and sample_size % 2 == 0 and required_per_class <= available_per_class
        rows.append(
            {
                "sample_size": sample_size,
                "required_per_class": required_per_class,
                "available_per_class": available_per_class,
                "feasible": "yes" if feasible else "no",
            }
        )
    return pd.DataFrame(rows)


def run_benchmark(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_feature_table(args.input)
    feasibility = build_feasibility(df, args.sample_sizes)

    if args.dry_run:
        write_report(
            output_dir,
            input_path=args.input,
            df=df,
            sample_sizes=args.sample_sizes,
            models=args.models,
            seed=args.seed,
            cv_folds=args.cv_folds,
            test_size=args.test_size,
            summary=None,
            dry_run=True,
            feasibility=feasibility,
        )
        print(f"Dry run complete. No models trained. Report written to {output_dir}")
        return

    spec = infer_features(df, args.max_categorical_cardinality)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for sample_size in args.sample_sizes:
        sample_df = balanced_sample(df, sample_size=sample_size, seed=args.seed + sample_size)
        y = sample_df["label"].to_numpy(dtype=int)
        x = sample_df[spec.feature_columns]
        splits = make_splits(y, seed=args.seed, test_size=args.test_size, cv_folds=args.cv_folds)

        for model_name in args.models:
            for fold, train_idx, test_idx in splits:
                pipeline = Pipeline(
                    steps=[
                        ("preprocess", build_preprocessor(spec)),
                        ("model", build_model(model_name, args.seed + fold)),
                    ]
                )
                pipeline.fit(x.iloc[train_idx], y[train_idx])
                y_prob = predict_probability(pipeline, x.iloc[test_idx])
                metrics = compute_metrics(y[test_idx], y_prob, args.ece_bins)
                y_pred = (y_prob >= 0.5).astype(int)
                row = {
                    "sample_size": int(sample_size),
                    "model": model_name,
                    "fold": int(fold),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    **metrics,
                }
                row["result_tuple"] = result_tuple(row)
                metric_rows.append(row)
                test_rows = sample_df.iloc[test_idx]
                for sample_row_index, source_row, true_label, probability, predicted_label in zip(
                    test_idx,
                    test_rows.to_dict(orient="records"),
                    y[test_idx],
                    y_prob,
                    y_pred,
                    strict=True,
                ):
                    prediction_row: dict[str, object] = {
                        "sample_size": int(sample_size),
                        "model": model_name,
                        "fold": int(fold),
                        "sample_row_index": int(sample_row_index),
                        "y_true": int(true_label),
                        "y_prob_pathogenic": float(probability),
                        "y_pred": int(predicted_label),
                    }
                    for identifier_column in ("variant_id", "gene", "chrom", "pos", "ref", "alt"):
                        if identifier_column in source_row:
                            prediction_row[identifier_column] = source_row[identifier_column]
                    prediction_rows.append(prediction_row)
                print(
                    f"sample_size={sample_size} model={model_name} fold={fold} "
                    f"accuracy={metrics['accuracy']:.4f} auroc={metrics['auroc']:.4f}"
                )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "sample_size_metrics.csv", index=False)
    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(output_dir / "sample_size_predictions.csv", index=False)
    summary = summarize_metrics(metrics_df)
    write_report(
        output_dir,
        input_path=args.input,
        df=df,
        sample_sizes=args.sample_sizes,
        models=args.models,
        seed=args.seed,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        summary=summary,
        dry_run=False,
        feasibility=feasibility,
    )
    write_json(
        {
            "input_path": str(args.input),
            "output_dir": str(output_dir),
            "sample_sizes": args.sample_sizes,
            "models": args.models,
            "seed": args.seed,
            "cv_folds": args.cv_folds,
            "test_size": args.test_size,
            "ece_bins": args.ece_bins,
            "ece_definition": "Positive-class calibration ECE: bins y_prob_pathogenic and compares observed positive rate with mean predicted pathogenic probability.",
            "confidence_ece_definition": "Supplemental classifier-confidence ECE: bins max(p, 1-p) and compares bin accuracy with mean confidence.",
            "metric_tuple_definition": METRIC_TUPLE_DEFINITION,
            "prediction_output": str(output_dir / "sample_size_predictions.csv"),
            "feature_columns": spec.feature_columns,
            "numeric_columns": spec.numeric_columns,
            "categorical_columns": spec.categorical_columns,
            "dropped_columns": spec.dropped_columns,
            "forbidden_fallback_files_not_used": [str(path) for path in FORBIDDEN_FALLBACK_FILES],
        },
        output_dir / "sample_size_benchmark_metadata.json",
    )
    print(f"Benchmark complete. Results written to {output_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tabular sample-size benchmarks on the real MAGPIE-X feature table."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=[5_000, 10_000, 20_000, 50_000, 100_000],
        help="Balanced total sample sizes to evaluate.",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cv-folds", type=int, default=1, help="Use 1 for one stratified split; use >1 for CV.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Check feasibility and write a plan report without training.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main(sys.argv[1:])
