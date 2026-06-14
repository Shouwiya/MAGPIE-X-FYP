from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_magpie_style_baseline import (  # noqa: E402
    DEFAULT_INPUT,
    FeatureSpec,
    balanced_sample,
    build_model,
    build_preprocessor,
    infer_features,
    load_feature_table,
    predict_probability,
)

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_residual"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "oof_tabular_predictions.csv"


def safe_logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def select_benchmark(
    df: pd.DataFrame,
    benchmark: str,
    *,
    sample_size: int,
    large_per_class_cap: int,
    seed: int,
) -> pd.DataFrame:
    if benchmark in {"reference_5k", "balanced_5k_cv"}:
        return balanced_sample(df, n_total=sample_size, per_class_cap=None, seed=seed)
    if benchmark == "larger_balanced":
        cap = None if large_per_class_cap <= 0 else large_per_class_cap
        return balanced_sample(df, n_total=None, per_class_cap=cap, seed=seed + 17)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def ensure_variant_id(df: pd.DataFrame) -> pd.DataFrame:
    if "variant_id" in df.columns:
        return df
    required = {"chrom", "pos", "ref", "alt"}
    if required.issubset(df.columns):
        out = df.copy()
        out["variant_id"] = (
            out["chrom"].astype(str)
            + ":"
            + out["pos"].astype(str)
            + ":"
            + out["ref"].astype(str)
            + ">"
            + out["alt"].astype(str)
        )
        return out
    raise ValueError("Input table must contain variant_id or chrom/pos/ref/alt columns.")


def run_oof_for_model(
    df: pd.DataFrame,
    spec: FeatureSpec,
    *,
    model_name: str,
    folds: int,
    seed: int,
    benchmark: str,
) -> pd.DataFrame:
    y = df["label"].to_numpy(dtype=int)
    x = df[spec.feature_columns]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows: list[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        preprocessor = build_preprocessor(spec)
        model = build_model(model_name, seed + fold)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(x.iloc[train_idx], y[train_idx])
        p_tabular = predict_probability(pipeline, x.iloc[test_idx])
        fold_df = df.iloc[test_idx].copy()
        out_cols = [
            column
            for column in ["variant_id", "chrom", "pos", "ref", "alt", "label", "consequence", "gene", "AF"]
            if column in fold_df.columns
        ]
        out = fold_df[out_cols].copy()
        out["fold"] = fold
        out["p_tabular"] = p_tabular
        out["logit_tabular"] = safe_logit(p_tabular)
        out["uncertainty_tabular"] = np.abs(p_tabular - 0.5)
        out["model_name"] = model_name
        out["benchmark"] = benchmark
        rows.append(out)
        print(f"{benchmark} | {model_name} | fold {fold}/{folds} complete")

    return pd.concat(rows, ignore_index=True)


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_feature_table(args.input, max_rows=args.max_rows)
    df = ensure_variant_id(df)
    spec = infer_features(df, args.max_categorical_cardinality)
    benchmark_df = select_benchmark(
        df,
        args.benchmark,
        sample_size=args.sample_size,
        large_per_class_cap=args.large_per_class_cap,
        seed=args.seed,
    )
    if benchmark_df["label"].value_counts().min() < args.folds:
        raise ValueError("Each class must contain at least as many rows as the requested folds.")

    predictions = [
        run_oof_for_model(
            benchmark_df,
            spec,
            model_name=model_name,
            folds=args.folds,
            seed=args.seed,
            benchmark=args.benchmark,
        )
        for model_name in args.models
    ]
    result = pd.concat(predictions, ignore_index=True)
    output_path = args.output_dir / "oof_tabular_predictions.csv"
    result.to_csv(output_path, index=False)

    metadata = {
        "input": str(args.input),
        "output": str(output_path),
        "benchmark": args.benchmark,
        "models": args.models,
        "folds": args.folds,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "large_per_class_cap": args.large_per_class_cap,
        "n_rows_selected": int(len(benchmark_df)),
        "n_feature_columns": len(spec.feature_columns),
        "n_numeric_columns": len(spec.numeric_columns),
        "n_categorical_columns": len(spec.categorical_columns),
        "note": "Out-of-fold tabular predictions are produced from real feature-table input only; known fallback annotation files are refused by the shared loader.",
    }
    write_metadata(args.output_dir / "oof_tabular_predictions_metadata.json", metadata)
    print(f"Saved out-of-fold tabular predictions to {output_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create out-of-fold tabular probabilities for MAGPIE-X residual correction on the reference 5k benchmark."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--benchmark",
        choices=["reference_5k", "larger_balanced", "balanced_5k_cv"],
        default="reference_5k",
        help="Use reference_5k for the immediate 5,000-variant benchmark. balanced_5k_cv is kept as a backwards-compatible alias.",
    )
    parser.add_argument("--models", nargs="+", choices=["xgboost", "lightgbm"], default=["lightgbm"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sample-size", type=int, default=5000, help="Total balanced sample size for the reference 5k benchmark.")
    parser.add_argument(
        "--large-per-class-cap",
        type=int,
        default=100_000,
        help="Per-class cap for larger_balanced. Use 0 to use all available balanced rows.",
    )
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug row limit before sampling; omit for real runs.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
