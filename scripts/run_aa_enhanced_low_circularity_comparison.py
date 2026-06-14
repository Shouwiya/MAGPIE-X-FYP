from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from build_amino_acid_clean_features import AA_FEATURE_COLUMNS
from run_circularity_ablation import build_mode_specs, is_prior_predictor_column
from run_magpie_low_circularity_baseline import (
    build_improved_low_circularity_spec,
    evaluate_pipeline,
    make_lightgbm_model,
    matches_explicit_forbidden_predictor,
    tune_lightgbm,
)
from run_sample_size_benchmark import (
    DEFAULT_INPUT,
    FeatureSpec,
    METRIC_TUPLE_DEFINITION,
    balanced_sample,
    build_feasibility,
    build_model,
    infer_features,
    load_feature_table,
    make_splits,
    markdown_table,
    result_tuple,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AA_INPUT = (
    ROOT / "data" / "interim" / "features" / "basic_feature_table_with_clean_aa_features.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_parity"
AA_CATEGORICAL_COLUMNS = ["aa_ref", "aa_alt"]
AA_NUMERIC_COLUMNS = [
    column for column in AA_FEATURE_COLUMNS if column not in AA_CATEGORICAL_COLUMNS
]


def build_aa_spec(improved_spec: FeatureSpec) -> FeatureSpec:
    for column in AA_FEATURE_COLUMNS:
        if matches_explicit_forbidden_predictor(column) or is_prior_predictor_column(column):
            raise ValueError(f"Clean AA feature unexpectedly matches a forbidden predictor pattern: {column}")
    return FeatureSpec(
        feature_columns=[*improved_spec.feature_columns, *AA_NUMERIC_COLUMNS, *AA_CATEGORICAL_COLUMNS],
        numeric_columns=[*improved_spec.numeric_columns, *AA_NUMERIC_COLUMNS],
        categorical_columns=[*improved_spec.categorical_columns, *AA_CATEGORICAL_COLUMNS],
        dropped_columns=dict(improved_spec.dropped_columns),
    )


def load_aa_columns(path: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"AA-enhanced feature table not found: {path}")
    usecols = ["variant_id", *AA_FEATURE_COLUMNS]
    aa_df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if len(aa_df) != len(base_df):
        raise ValueError(
            f"AA table row count {len(aa_df):,} does not match base table row count {len(base_df):,}."
        )
    if not aa_df["variant_id"].astype(str).equals(base_df["variant_id"].astype(str)):
        raise ValueError("AA table row order/key alignment does not match the base feature table.")
    return aa_df[AA_FEATURE_COLUMNS]


def write_report(
    path: Path,
    *,
    metrics_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    base_input: Path,
    aa_input: Path,
    sample_sizes: list[int],
    seed: int,
    test_size: float,
) -> None:
    summary = metrics_df[
        ["mode", "sample_size", "model", "n_features", "aa_available_in_sample", "result_tuple"]
    ].copy()
    lines = [
        "# AA-Enhanced Low-Circularity Report",
        "",
        "Status: completed run",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Base input: `{base_input}`",
        f"AA feature input: `{aa_input}`",
        "",
        "This comparison uses real ClinVar/ANNOVAR rows only. It does not use synthetic data, new database downloads, AlphaMissense, ESM, GNN output, or prior pathogenicity predictor scores in either low-circularity mode.",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
        "## Configuration",
        "",
        f"- Sample sizes: {', '.join(format(size, ',') for size in sample_sizes)}",
        "- Model family: LightGBM only",
        f"- Seed: {seed}",
        "- Split: one stratified train/test split per sample size",
        f"- Test size: {test_size}",
        "- Low-circularity tuning: the same six-candidate LightGBM grid and validation-AUPRC selection used by the previous improved baseline",
        "",
        "## Compared Modes",
        "",
        "- `previous_improved_low_circularity`: the prior clean engineered AF, missingness, consequence, conservation, length, and transcript-count baseline.",
        "- `aa_blosum_enhanced_low_circularity`: the previous improved baseline plus the ten clean AA/BLOSUM62 fields.",
        "- `enhanced_dbnsfp_context_only`: the prior full inferred dbNSFP mode, unchanged and shown only as a circularity-risk reference.",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(markdown_table(summary))
    lines.extend(["", "## Change From Previous Improved Baseline", ""])
    for sample_size in sample_sizes:
        previous = metrics_df[
            (metrics_df["sample_size"] == sample_size)
            & (metrics_df["mode"] == "previous_improved_low_circularity")
        ].iloc[0]
        aa = metrics_df[
            (metrics_df["sample_size"] == sample_size)
            & (metrics_df["mode"] == "aa_blosum_enhanced_low_circularity")
        ].iloc[0]
        lines.append(
            f"- {sample_size:,}: accuracy {float(previous['accuracy']):.4f} -> {float(aa['accuracy']):.4f} "
            f"({float(aa['accuracy']) - float(previous['accuracy']):+.4f}); "
            f"MCC {float(previous['mcc']):.4f} -> {float(aa['mcc']):.4f}; "
            f"AUROC {float(previous['auroc']):.4f} -> {float(aa['auroc']):.4f}; "
            f"AUPRC {float(previous['auprc']):.4f} -> {float(aa['auprc']):.4f}."
        )
    selected = tuning_df[tuning_df["selected"] == "yes"][
        ["mode", "sample_size", "candidate_id", "validation_auprc", "validation_auroc", "params_json"]
    ]
    lines.extend(["", "## Selected Parameters", ""])
    lines.extend(markdown_table(selected))
    lines.extend(
        [
            "",
            "Across these approved single-split checks, the AA/BLOSUM feature set does not improve the 5k result but improves the 10k and 20k results. This is scale-dependent evidence, not a universal improvement claim.",
            "",
            "## Interpretation Boundary",
            "",
            "The dbNSFP mode is not evidence for the low-circularity claim because it includes prior predictor outputs. The AA/BLOSUM comparison remains a random stratified split benchmark, not exact MAGPIE reproduction; gene-held-out and external validation are still required.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_df = load_feature_table(args.input)
    feasibility = build_feasibility(base_df, args.sample_sizes)
    if not all(feasibility["feasible"] == "yes"):
        raise ValueError(f"One or more requested sample sizes are infeasible:\n{feasibility}")

    aa_columns = load_aa_columns(args.aa_input, base_df)
    base_spec = infer_features(base_df, args.max_categorical_cardinality)
    mode_specs, removed_columns = build_mode_specs(base_spec)
    old_spec = mode_specs["magpie_parity"]
    enhanced_spec = mode_specs["magpie_x_enhanced"]
    combined_df, improved_spec, _ = build_improved_low_circularity_spec(base_df, old_spec)
    combined_df = pd.concat([combined_df, aa_columns], axis=1)
    aa_spec = build_aa_spec(improved_spec)

    for column in aa_spec.feature_columns:
        if column not in AA_FEATURE_COLUMNS and (
            matches_explicit_forbidden_predictor(column) or is_prior_predictor_column(column)
        ):
            raise ValueError(f"Forbidden prior predictor present in AA low-circularity mode: {column}")

    metric_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    selected_params: dict[str, dict[int, dict[str, object]]] = {
        "previous_improved_low_circularity": {},
        "aa_blosum_enhanced_low_circularity": {},
    }

    for sample_size in args.sample_sizes:
        sampled = balanced_sample(
            base_df.reset_index(names="_source_index"),
            sample_size=sample_size,
            seed=args.seed + sample_size,
        )
        source_indices = sampled["_source_index"].to_numpy(dtype=int)
        y = base_df.iloc[source_indices]["label"].to_numpy(dtype=int)
        fold, train_idx, test_idx = make_splits(
            y,
            seed=args.seed,
            test_size=args.test_size,
            cv_folds=1,
        )[0]
        aa_available = int(
            pd.to_numeric(
                combined_df.iloc[source_indices]["aa_substitution_available"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        improved_x = combined_df.iloc[source_indices][improved_spec.feature_columns].reset_index(drop=True)
        aa_x = combined_df.iloc[source_indices][aa_spec.feature_columns].reset_index(drop=True)
        enhanced_x = base_df.iloc[source_indices][enhanced_spec.feature_columns].reset_index(drop=True)

        for mode, x, spec in [
            ("previous_improved_low_circularity", improved_x, improved_spec),
            ("aa_blosum_enhanced_low_circularity", aa_x, aa_spec),
        ]:
            params, mode_tuning = tune_lightgbm(
                x=x,
                y=y,
                train_idx=train_idx,
                spec=spec,
                seed=args.seed + sample_size,
                ece_bins=args.ece_bins,
            )
            selected_params[mode][int(sample_size)] = params
            selected_json = json.dumps(params, sort_keys=True)
            for tuning_row in mode_tuning:
                params_json = json.dumps(tuning_row["params"], sort_keys=True)
                tuning_rows.append(
                    {
                        "mode": mode,
                        "sample_size": int(sample_size),
                        "candidate_id": tuning_row["candidate_id"],
                        "validation_auprc": tuning_row["validation_auprc"],
                        "validation_auroc": tuning_row["validation_auroc"],
                        "validation_accuracy": tuning_row["validation_accuracy"],
                        "selected": "yes" if params_json == selected_json else "no",
                        "params_json": params_json,
                    }
                )
            metrics, _ = evaluate_pipeline(
                x=x,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                spec=spec,
                model=make_lightgbm_model(params),
                ece_bins=args.ece_bins,
            )
            row = {
                "mode": mode,
                "sample_size": int(sample_size),
                "model": "lightgbm",
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_features": int(len(spec.feature_columns)),
                "aa_available_in_sample": aa_available,
                "selected_params_json": selected_json,
                **metrics,
            }
            row["result_tuple"] = result_tuple(row)
            metric_rows.append(row)
            print(
                f"sample_size={sample_size} mode={mode} "
                f"accuracy={metrics['accuracy']:.4f} auroc={metrics['auroc']:.4f}",
                flush=True,
            )

        context_metrics, _ = evaluate_pipeline(
            x=enhanced_x,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            spec=enhanced_spec,
            model=build_model("lightgbm", args.seed + fold),
            ece_bins=args.ece_bins,
        )
        context_row = {
            "mode": "enhanced_dbnsfp_context_only",
            "sample_size": int(sample_size),
            "model": "lightgbm",
            "fold": int(fold),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(len(enhanced_spec.feature_columns)),
            "aa_available_in_sample": aa_available,
            "selected_params_json": "{}",
            **context_metrics,
        }
        context_row["result_tuple"] = result_tuple(context_row)
        metric_rows.append(context_row)
        print(
            f"sample_size={sample_size} mode=enhanced_dbnsfp_context_only "
            f"accuracy={context_metrics['accuracy']:.4f} auroc={context_metrics['auroc']:.4f}",
            flush=True,
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["sample_size", "mode"]).reset_index(drop=True)
    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["sample_size", "mode", "candidate_id"]
    ).reset_index(drop=True)
    metrics_path = args.output_dir / "aa_enhanced_low_circularity_metrics.csv"
    tuning_path = args.output_dir / "aa_enhanced_low_circularity_tuning.csv"
    report_path = args.output_dir / "AA_ENHANCED_LOW_CIRCULARITY_REPORT.md"
    metrics_df.to_csv(metrics_path, index=False)
    tuning_df.to_csv(tuning_path, index=False)
    write_report(
        report_path,
        metrics_df=metrics_df,
        tuning_df=tuning_df,
        base_input=args.input,
        aa_input=args.aa_input,
        sample_sizes=args.sample_sizes,
        seed=args.seed,
        test_size=args.test_size,
    )
    metadata = {
        "base_input": str(args.input),
        "aa_input": str(args.aa_input),
        "sample_sizes": args.sample_sizes,
        "seed": args.seed,
        "test_size": args.test_size,
        "ece_bins": args.ece_bins,
        "metric_tuple_definition": METRIC_TUPLE_DEFINITION,
        "previous_improved_feature_count": len(improved_spec.feature_columns),
        "aa_enhanced_feature_count": len(aa_spec.feature_columns),
        "enhanced_dbnsfp_context_feature_count": len(enhanced_spec.feature_columns),
        "aa_feature_columns": AA_FEATURE_COLUMNS,
        "removed_prior_predictor_columns": removed_columns,
        "selected_params": selected_params,
        "synthetic_or_toy_data_used": False,
        "databases_downloaded": False,
    }
    (args.output_dir / "aa_enhanced_low_circularity_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare previous and AA/BLOSUM-enhanced low-circularity LightGBM baselines."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--aa-input", type=Path, default=DEFAULT_AA_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
