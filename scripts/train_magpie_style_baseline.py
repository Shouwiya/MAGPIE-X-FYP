from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_style_baseline"

FORBIDDEN_FALLBACK_FILES = {
    (ROOT / "outputs" / "real" / "processed" / "annovar.csv").resolve(),
    (ROOT / "outputs" / "real" / "processed" / "dbnsfp.csv").resolve(),
}

REFERENCE_5K_TARGET = {"accuracy": 0.9630, "auroc": 0.9870}

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
        raise ValueError(f"Refusing to use synthetic fallback file as input: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Input feature table not found: {path}")


def load_feature_table(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    check_input_path(path)
    df = pd.read_csv(path, nrows=max_rows, low_memory=False, na_values=[".", "NA", "nan", ""])
    df = normalize_missing_markers(df)
    if "label" not in df.columns:
        raise ValueError("Input feature table must contain a binary label column.")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    if len(df) == 0:
        raise ValueError("No valid binary labels were found.")
    return df.reset_index(drop=True)


def normalize_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common ANNOVAR/dbNSFP missing markers before type inference."""

    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_object_dtype(out[column]) or pd.api.types.is_string_dtype(out[column]):
            out[column] = out[column].map(normalize_missing_value)
    return out


def normalize_missing_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in MISSING_MARKERS or stripped.lower() in MISSING_MARKERS_LOWER:
            return np.nan
        return stripped
    return value


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
            dropped[column] = "clinvar_metadata_not_magpie_feature"
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
        name_suggests_numeric = is_numeric_feature_name(column)
        if name_suggests_numeric:
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
    return FeatureSpec(feature_columns=feature_columns, numeric_columns=numeric, categorical_columns=categorical, dropped_columns=dropped)


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
        )
    raise ValueError(f"Unknown model: {model_name}")


def balanced_sample(df: pd.DataFrame, n_total: int | None, per_class_cap: int | None, seed: int) -> pd.DataFrame:
    counts = df["label"].value_counts()
    if 0 not in counts or 1 not in counts:
        raise ValueError("Balanced sampling requires both classes.")
    available_per_class = int(min(counts.loc[0], counts.loc[1]))
    if n_total is not None:
        per_class = min(available_per_class, n_total // 2)
    elif per_class_cap is not None and per_class_cap > 0:
        per_class = min(available_per_class, per_class_cap)
    else:
        per_class = available_per_class
    if per_class < 2:
        raise ValueError("Not enough rows per class for balanced sampling.")
    sampled = pd.concat(
        [
            df[df["label"] == 0].sample(n=per_class, random_state=seed),
            df[df["label"] == 1].sample(n=per_class, random_state=seed + 1),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def select_benchmarks(df: pd.DataFrame, mode: str, seed: int, large_per_class_cap: int) -> dict[str, pd.DataFrame]:
    benchmarks: dict[str, pd.DataFrame] = {}
    if mode in {"reference_5k", "all"}:
        benchmarks["balanced_5k_cv"] = balanced_sample(df, n_total=5000, per_class_cap=None, seed=seed)
    if mode in {"larger", "all"}:
        cap = None if large_per_class_cap <= 0 else large_per_class_cap
        benchmarks["larger_balanced"] = balanced_sample(df, n_total=None, per_class_cap=cap, seed=seed + 17)
    return benchmarks


def safe_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
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
    except Exception:
        return float("nan")
    raise ValueError(metric)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        metric: safe_metric(metric, y_true, y_pred, y_prob)
        for metric in ["accuracy", "balanced_accuracy", "f1", "auroc", "auprc", "mcc"]
    }


def predict_probability(pipeline: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x_test)[:, 1]
    decision = pipeline.decision_function(x_test)
    return 1.0 / (1.0 + np.exp(-decision))


def run_cv(
    benchmark_name: str,
    df: pd.DataFrame,
    spec: FeatureSpec,
    model_names: list[str],
    seed: int,
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["label"].to_numpy(dtype=int)
    x = df[spec.feature_columns]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for model_name in model_names:
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
            preprocessor = build_preprocessor(spec)
            model = build_model(model_name, seed + fold)
            pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
            pipeline.fit(x.iloc[train_idx], y[train_idx])
            y_prob = predict_probability(pipeline, x.iloc[test_idx])
            metrics = compute_metrics(y[test_idx], y_prob)
            metric_rows.append(
                {
                    "benchmark": benchmark_name,
                    "model": model_name,
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    **metrics,
                }
            )
            pred_cols = [c for c in ["variant_id", "gene", "consequence", "label"] if c in df.columns]
            pred = df.iloc[test_idx][pred_cols].copy()
            pred["benchmark"] = benchmark_name
            pred["model"] = model_name
            pred["fold"] = fold
            pred["probability_pathogenic"] = y_prob
            pred["predicted_label"] = (y_prob >= 0.5).astype(int)
            prediction_rows.append(pred)
            print(f"{benchmark_name} | {model_name} | fold {fold}/{folds} | accuracy={metrics['accuracy']:.4f} auroc={metrics['auroc']:.4f}")

    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "balanced_accuracy", "f1", "auroc", "auprc", "mcc"]
    grouped = metrics.groupby(["benchmark", "model"], as_index=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
    return grouped.reset_index(drop=True)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)


def markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No metrics available yet."]
    columns = df.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}" if math.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    output_dir: Path,
    *,
    input_path: Path,
    summary: pd.DataFrame | None = None,
    metadata: dict | None = None,
    pre_run: bool = False,
) -> None:
    lines = [
        "# MAGPIE-Style Baseline Report",
        "",
    ]
    if pre_run:
        lines.append("Status: script created and validation checks requested; full training has not been run yet.")
        lines.append("")
    lines.extend(
        [
            "This baseline is the first real MAGPIE-style feature baseline for MAGPIE-X.",
            "",
            f"Input feature table: `{input_path}`",
            "",
            "The input is expected to contain real ANNOVAR/dbNSFP/gnomAD-derived columns from `refGene`, `gnomad30_genome`, and `dbnsfp42a`.",
            "",
            "The script refuses to use the known synthetic fallback files:",
            "",
            "- `outputs/real/processed/annovar.csv`",
            "- `outputs/real/processed/dbnsfp.csv`",
            "",
            "## Reference Benchmark",
            "",
            "Prior MAGPIE-style experiments on balanced ClinVar subsets report approximately 96% accuracy and 0.987 ROC-AUC. This reference is used only to verify that the real ANNOVAR/gnomAD/dbNSFP feature pipeline is functioning correctly.",
            "",
            f"- Accuracy: {REFERENCE_5K_TARGET['accuracy']:.4f} / 96.30%",
            f"- ROC-AUC: {REFERENCE_5K_TARGET['auroc']:.4f}",
            "",
            "This script supports a balanced 5,000-variant MAGPIE-style benchmark with 5-fold stratified cross-validation.",
            "",
            "## Benchmark Modes",
            "",
            "- `balanced_5k_cv`: balanced 5,000 variants, 5-fold stratified CV. CLI compatibility: run this with `--mode reference_5k`.",
            "- `larger`: balanced sample capped by `--large-per-class-cap`, 5-fold stratified CV.",
            "- `all`: run both modes.",
            "",
            "## Interpretation",
            "",
            "This is not the final MAGPIE-X model. It is the first real annotated-feature baseline before robustness improvements such as gene-held-out evaluation, temporal evaluation, rare-variant testing, calibration, external validation, stacking, and graph augmentation.",
            "",
            "This baseline is not the final MAGPIE-X contribution. It confirms that the real annotation pipeline works. The main research contribution will come from robustness-focused evaluation and improvement: gene-held-out testing, rare-variant evaluation, imbalanced clinical-style testing, calibration, and later graph/context augmentation.",
            "",
        ]
    )
    if summary is not None:
        lines.extend(["## Results", ""])
        lines.extend(markdown_table(summary))
        lines.append("")
    if metadata is not None:
        dropped_columns = metadata.get("dropped_columns", {})
        unusable_drops = {
            column: reason
            for column, reason in dropped_columns.items()
            if reason not in {"identifier_or_direct_leakage", "clinvar_metadata_not_magpie_feature"}
        }
        lines.extend(
            [
                "## Preprocessing Summary",
                "",
                f"- Feature columns used: {metadata.get('n_feature_columns')}",
                f"- Numeric columns: {metadata.get('n_numeric_columns')}",
                f"- Categorical columns: {metadata.get('n_categorical_columns')}",
                f"- Dropped columns: {len(dropped_columns)}",
                f"- Dropped as unusable/non-feature text: {len(unusable_drops)}",
                "",
                "Common ANNOVAR/dbNSFP missing markers (`-`, `.`, `NA`, `N/A`, `nan`, `None`, and empty strings) are converted to missing values before feature inference. Numeric candidates are coerced with `pd.to_numeric(errors=\"coerce\")` and only kept as numeric when at least 80% of non-missing values convert successfully. `*_pred` columns are treated as categorical prediction labels.",
                "",
            ]
        )
        if unusable_drops:
            dropped_preview = pd.DataFrame(
                [{"column": column, "reason": reason} for column, reason in sorted(unusable_drops.items())]
            ).head(80)
            lines.extend(["### Dropped Unusable Columns", ""])
            lines.extend(markdown_table(dropped_preview))
            lines.append("")
        lines.extend(
            [
                "## Run Metadata",
                "",
                f"- Models: {', '.join(metadata.get('models', []))}",
                f"- Folds: {metadata.get('folds')}",
                f"- Seed: {metadata.get('seed')}",
                "",
            ]
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "MAGPIE_STYLE_BASELINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_feature_table(args.input, max_rows=args.max_rows)
    spec = infer_features(df, args.max_categorical_cardinality)
    benchmarks = select_benchmarks(df, args.mode, args.seed, args.large_per_class_cap)
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []

    for benchmark_name, benchmark_df in benchmarks.items():
        metrics, predictions = run_cv(benchmark_name, benchmark_df, spec, args.models, args.seed, args.folds)
        all_metrics.append(metrics)
        if args.save_predictions:
            all_predictions.append(predictions)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = summarize_metrics(metrics_df)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)

    metadata = {
        "input_path": str(args.input),
        "input_rows_loaded": int(len(df)),
        "mode": args.mode,
        "models": args.models,
        "folds": args.folds,
        "seed": args.seed,
        "large_per_class_cap": args.large_per_class_cap,
        "n_feature_columns": len(spec.feature_columns),
        "n_numeric_columns": len(spec.numeric_columns),
        "n_categorical_columns": len(spec.categorical_columns),
        "numeric_conversion_threshold": NUMERIC_CONVERTIBLE_THRESHOLD,
        "missing_markers_normalized": sorted(MISSING_MARKERS),
        "numeric_columns": spec.numeric_columns,
        "categorical_columns": spec.categorical_columns,
        "dropped_columns": spec.dropped_columns,
        "forbidden_fallback_files_not_used": [str(path) for path in FORBIDDEN_FALLBACK_FILES],
    }
    write_json(metadata, output_dir / "run_metadata.json")
    write_report(output_dir, input_path=args.input, summary=summary_df, metadata=metadata)
    print(f"Training complete. Results written to {output_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first real MAGPIE-style baseline on real ANNOVAR/dbNSFP/gnomAD features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["reference_5k", "larger", "all"], default="reference_5k")
    parser.add_argument("--models", nargs="+", choices=["xgboost", "lightgbm", "random_forest"], default=["xgboost", "lightgbm", "random_forest"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--large-per-class-cap", type=int, default=100_000, help="Per-class cap for larger mode. Use 0 to use all available balanced rows.")
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug row limit before sampling; omit for real runs.")
    parser.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    parser.set_defaults(save_predictions=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
