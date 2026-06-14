from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.pipeline import Pipeline

from run_sample_size_benchmark import (
    DEFAULT_INPUT,
    FeatureSpec,
    METRIC_COLUMNS,
    METRIC_TUPLE_DEFINITION,
    MODEL_CHOICES,
    balanced_sample,
    build_feasibility,
    build_model,
    build_preprocessor,
    compute_metrics,
    format_metric,
    infer_features,
    load_feature_table,
    make_splits,
    markdown_table,
    predict_probability,
    result_tuple,
    summarize_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "circularity_ablation"

PRIOR_PREDICTOR_PATTERNS = (
    "sift",
    "sift4g",
    "polyphen",
    "revel",
    "cadd",
    "mutationtaster",
    "mutationassessor",
    "metasvm",
    "metalr",
    "alphamissense",
    "fathmm",
    "provean",
    "vest",
    "metarnn",
    "mcap",
    "mutpred",
    "mvp",
    "mpc",
    "primateai",
    "deogen",
    "bayesdel",
    "clinpred",
    "lists2",
    "lrt",
    "aloft",
    "dann",
    "eigen",
    "genocanyon",
    "linsight",
)

MODE_DESCRIPTIONS = {
    "magpie_parity": (
        "Removes prior predictor-score columns such as SIFT, PolyPhen, REVEL, CADD, "
        "MutationTaster, MetaSVM, MetaLR, AlphaMissense, and related dbNSFP predictor outputs."
    ),
    "magpie_x_enhanced": (
        "Keeps the full inferred feature set, including dbNSFP predictor scores where present. "
        "AlphaMissense and GNN evidence can be layered later when those columns or predictions are available."
    ),
}


def normalize_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column.lower())


def is_prior_predictor_column(column: str) -> bool:
    normalized = normalize_column_name(column)
    return any(pattern in normalized for pattern in PRIOR_PREDICTOR_PATTERNS)


def build_mode_specs(base_spec: FeatureSpec) -> tuple[dict[str, FeatureSpec], list[str]]:
    removed = [column for column in base_spec.feature_columns if is_prior_predictor_column(column)]
    removed_set = set(removed)

    parity_numeric = [column for column in base_spec.numeric_columns if column not in removed_set]
    parity_categorical = [column for column in base_spec.categorical_columns if column not in removed_set]
    parity_dropped = dict(base_spec.dropped_columns)
    for column in removed:
        parity_dropped[column] = "prior_predictor_score_removed_for_parity_mode"

    parity_spec = FeatureSpec(
        feature_columns=[*parity_numeric, *parity_categorical],
        numeric_columns=parity_numeric,
        categorical_columns=parity_categorical,
        dropped_columns=parity_dropped,
    )
    if not parity_spec.feature_columns:
        raise ValueError("MAGPIE-parity mode removed all feature columns; cannot run ablation.")
    return {"magpie_parity": parity_spec, "magpie_x_enhanced": base_spec}, removed


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_report(
    output_dir: Path,
    *,
    input_path: Path,
    sample_size: int,
    models: list[str],
    seed: int,
    cv_folds: int,
    test_size: float,
    summary: pd.DataFrame | None,
    dry_run: bool,
    removed_columns: list[str],
    mode_feature_counts: dict[str, int],
    feasibility: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Circularity Ablation Report",
        "",
        f"Status: {'dry run only; no models were trained' if dry_run else 'completed run'}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Input feature table: `{input_path}`",
        "",
        "Purpose: address the supervisor concern about unusually high accuracy by comparing a prior-predictor-free mode against the enhanced feature mode.",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
        "## Modes",
        "",
    ]
    for mode, description in MODE_DESCRIPTIONS.items():
        lines.append(f"- `{mode}`: {description}")
    lines.extend(
        [
            "",
            "## Requested Configuration",
            "",
            f"- Sample size: {sample_size:,}",
            f"- Models: {', '.join(models)}",
            f"- Seed: {seed}",
            f"- Split: {'stratified cross-validation' if cv_folds and cv_folds > 1 else 'single stratified train/test split'}",
            f"- CV folds: {cv_folds}",
            f"- Test size: {test_size}",
            "",
            "## Feasibility",
            "",
        ]
    )
    lines.extend(markdown_table(feasibility))
    lines.extend(["", "## Feature Counts", ""])
    feature_count_df = pd.DataFrame(
        [{"mode": mode, "feature_columns": count} for mode, count in mode_feature_counts.items()]
    )
    lines.extend(markdown_table(feature_count_df))
    lines.extend(
        [
            "",
            "## Prior Predictor Columns Removed In Parity Mode",
            "",
        ]
    )
    removed_preview = pd.DataFrame({"removed_column": removed_columns})
    lines.extend(markdown_table(removed_preview, max_rows=120))
    lines.append("")

    if summary is not None and not summary.empty:
        lines.extend(["## Metric Summary", ""])
        preview_cols = ["mode", "sample_size", "model", "mean_result_tuple"]
        lines.extend(markdown_table(summary[preview_cols]))
        lines.append("")
        lines.append("Full fold-level metrics are written to `circularity_metrics.csv`.")
        lines.append("")

    lines.extend(
        [
            "## Interpretation Guardrail",
            "",
            "If the enhanced mode performs much better than the parity mode, that is evidence that prior predictor columns may be driving part of the high accuracy. This does not invalidate the project, but it must be reported transparently and followed by gene-held-out, temporal, rare-variant, and external validation.",
            "",
        ]
    )
    (output_dir / "CIRCULARITY_ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_ablation(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(args.input)
    feasibility = build_feasibility(df, [args.sample_size])
    base_spec = infer_features(df, args.max_categorical_cardinality)
    mode_specs, removed_columns = build_mode_specs(base_spec)
    mode_feature_counts = {mode: len(spec.feature_columns) for mode, spec in mode_specs.items()}

    if args.dry_run:
        write_report(
            output_dir,
            input_path=args.input,
            sample_size=args.sample_size,
            models=args.models,
            seed=args.seed,
            cv_folds=args.cv_folds,
            test_size=args.test_size,
            summary=None,
            dry_run=True,
            removed_columns=removed_columns,
            mode_feature_counts=mode_feature_counts,
            feasibility=feasibility,
        )
        print(f"Dry run complete. No models trained. Report written to {output_dir}")
        return

    sample_df = balanced_sample(df, sample_size=args.sample_size, seed=args.seed + args.sample_size)
    metric_rows: list[dict[str, object]] = []

    for mode, spec in mode_specs.items():
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
                row = {
                    "mode": mode,
                    "sample_size": int(args.sample_size),
                    "model": model_name,
                    "fold": int(fold),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_features": int(len(spec.feature_columns)),
                    **metrics,
                }
                row["result_tuple"] = result_tuple(row)
                metric_rows.append(row)
                print(
                    f"mode={mode} model={model_name} fold={fold} "
                    f"accuracy={metrics['accuracy']:.4f} auroc={metrics['auroc']:.4f}"
                )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "circularity_metrics.csv", index=False)

    summary_parts = []
    for mode, mode_df in metrics_df.groupby("mode", sort=False):
        summary = summarize_metrics(mode_df.drop(columns=["mode"]))
        summary.insert(0, "mode", mode)
        summary_parts.append(summary)
    summary_df = pd.concat(summary_parts, ignore_index=True)

    write_report(
        output_dir,
        input_path=args.input,
        sample_size=args.sample_size,
        models=args.models,
        seed=args.seed,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        summary=summary_df,
        dry_run=False,
        removed_columns=removed_columns,
        mode_feature_counts=mode_feature_counts,
        feasibility=feasibility,
    )
    write_json(
        {
            "input_path": str(args.input),
            "output_dir": str(output_dir),
            "sample_size": args.sample_size,
            "models": args.models,
            "seed": args.seed,
            "cv_folds": args.cv_folds,
            "test_size": args.test_size,
            "ece_bins": args.ece_bins,
            "metric_tuple_definition": METRIC_TUPLE_DEFINITION,
            "prior_predictor_patterns": PRIOR_PREDICTOR_PATTERNS,
            "removed_prior_predictor_columns": removed_columns,
            "mode_feature_counts": mode_feature_counts,
            "metric_columns": METRIC_COLUMNS,
        },
        output_dir / "circularity_ablation_metadata.json",
    )
    print(f"Circularity ablation complete. Results written to {output_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MAGPIE-parity and MAGPIE-X-enhanced feature modes on the real feature table."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cv-folds", type=int, default=1, help="Use 1 for one stratified split; use >1 for CV.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Inspect removed columns and feasibility without training.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_ablation(args)


if __name__ == "__main__":
    main(sys.argv[1:])

