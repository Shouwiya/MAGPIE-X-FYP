#!/usr/bin/env python3
"""Run the controlled 5k MAGPIE-X enhanced tabular comparison."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from build_amino_acid_clean_features import AA_FEATURE_COLUMNS
from run_aa_enhanced_low_circularity_comparison import (
    AA_CATEGORICAL_COLUMNS,
    AA_NUMERIC_COLUMNS,
    build_aa_spec,
)
from run_circularity_ablation import build_mode_specs
from run_magpie_low_circularity_baseline import (
    build_improved_low_circularity_spec,
    evaluate_pipeline,
    make_lightgbm_model,
    tune_lightgbm,
)
from run_sample_size_benchmark import (
    FeatureSpec,
    METRIC_TUPLE_DEFINITION,
    infer_features,
    load_feature_table,
    make_splits,
    markdown_table,
    normalize_missing_markers,
    result_tuple,
)
from run_spliceai_5k_low_circularity_comparison import (
    AGGREGATED_SPLICEAI_COLUMNS,
    SPLICEAI_CATEGORICAL_COLUMNS,
    SPLICEAI_NUMERIC_COLUMNS,
    build_spliceai_spec,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_INPUT = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_5K_INPUT = (
    ROOT
    / "outputs"
    / "real"
    / "magpie_parity"
    / "reference_5k_aa_spliceai_feature_table.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_enhanced"
CONTROLLED_FEATURE_TABLE = "reference_5k_magpie_x_enhanced_feature_table.csv"

MODE_A = "aa_blosum_low_circularity"
MODE_B = "aa_blosum_spliceai_low_circularity"
MODE_C = "magpie_x_enhanced_tabular"
MODE_D = "magpie_x_enhanced_tabular_spliceai"


def extend_spec(
    base: FeatureSpec,
    *,
    numeric: Iterable[str] = (),
    categorical: Iterable[str] = (),
) -> FeatureSpec:
    numeric_columns = list(dict.fromkeys([*base.numeric_columns, *numeric]))
    categorical_columns = list(dict.fromkeys([*base.categorical_columns, *categorical]))
    categorical_set = set(categorical_columns)
    numeric_columns = [column for column in numeric_columns if column not in categorical_set]
    return FeatureSpec(
        feature_columns=[*numeric_columns, *categorical_columns],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        dropped_columns=dict(base.dropped_columns),
    )


def evaluate_mode(
    *,
    mode: str,
    combined: pd.DataFrame,
    spec: FeatureSpec,
    y,
    train_idx,
    test_idx,
    fold: int,
    args: argparse.Namespace,
    circularity_risk: str,
    includes_spliceai: bool,
) -> dict[str, object]:
    x = combined[spec.feature_columns].reset_index(drop=True)
    params, tuning = tune_lightgbm(
        x=x,
        y=y,
        train_idx=train_idx,
        spec=spec,
        seed=args.seed + args.sample_size,
        ece_bins=args.ece_bins,
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
    selected_json = json.dumps(params, sort_keys=True)
    selected_candidate = next(
        row["candidate_id"]
        for row in tuning
        if json.dumps(row["params"], sort_keys=True) == selected_json
    )
    output = {
        "mode": mode,
        "sample_size": args.sample_size,
        "model": "lightgbm",
        "fold": int(fold),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(len(spec.feature_columns)),
        "circularity_risk": circularity_risk,
        "includes_spliceai": int(includes_spliceai),
        "includes_alphamissense": 0,
        "includes_gnn": 0,
        "selected_candidate_id": int(selected_candidate),
        "selected_params_json": selected_json,
        **metrics,
    }
    output["result_tuple"] = result_tuple(output)
    return output


def write_feature_risk_audit(
    path: Path,
    *,
    aa_spec: FeatureSpec,
    spliceai_spec: FeatureSpec,
    risky_numeric: list[str],
    risky_categorical: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clean_aa = set(AA_FEATURE_COLUMNS)
    spliceai = set(AGGREGATED_SPLICEAI_COLUMNS)
    risky = set(risky_numeric) | set(risky_categorical)
    all_features = list(
        dict.fromkeys(
            [
                *aa_spec.feature_columns,
                *spliceai_spec.feature_columns,
                *risky_numeric,
                *risky_categorical,
            ]
        )
    )
    for feature in all_features:
        if feature in risky:
            group = "prior_pathogenicity_predictor"
            risk = "high"
        elif feature in spliceai:
            group = "spliceai_mechanistic_splicing"
            risk = "clean_parity_feature"
        elif feature in clean_aa:
            group = "clean_amino_acid_blosum"
            risk = "clean_parity_feature"
        else:
            group = "clean_parity_or_engineered"
            risk = "clean_parity_feature"
        rows.append(
            {
                "feature": feature,
                "feature_group": group,
                "circularity_risk": risk,
                "used_in_low_circularity": int(feature in spliceai_spec.feature_columns),
                "used_in_magpie_x_enhanced": 1,
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(path, index=False)
    return audit


def write_report(
    path: Path,
    *,
    metrics: pd.DataFrame,
    risky_columns: list[str],
    feature_table: Path,
    alpha_resource_exists: bool,
    gnn_alignment: dict[str, object],
) -> None:
    summary = metrics[
        ["mode", "n_features", "circularity_risk", "result_tuple"]
    ].copy()
    indexed = metrics.set_index("mode")
    best = metrics.sort_values(["accuracy", "auroc"], ascending=False).iloc[0]
    lines = [
        "# MAGPIE-X 5k Enhanced Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
        "## Controlled Design",
        "",
        "- Real 5k ClinVar-derived variants from the completed label-free SpliceAI pilot.",
        "- Identical ordered sample and identical stratified 4,000/1,000 train/test split for all four tabular modes.",
        "- LightGBM only, with the established six-candidate validation-AUPRC tuning grid.",
        "- MAGPIE-parity modes remain free of prior pathogenicity predictor outputs.",
        "- MAGPIE-X modes intentionally add prior dbNSFP predictor outputs and therefore have higher circularity risk.",
        f"- Temporary enhanced table: `outputs/real/magpie_x_enhanced/{feature_table.name}`.",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(markdown_table(summary))
    lines.extend(
        [
            "",
            "## Feature Composition",
            "",
            f"- A: 114 clean AA/BLOSUM low-circularity features.",
            f"- B: 114 clean features plus 12 aggregated SpliceAI fields.",
            f"- C: 114 clean features plus {len(risky_columns)} prior-predictor fields.",
            f"- D: 114 clean features plus 12 SpliceAI fields plus {len(risky_columns)} prior-predictor fields.",
            f"- AlphaMissense resource present: {'yes' if alpha_resource_exists else 'no'}; matched 5k AlphaMissense artifact present: no.",
            "- AlphaMissense was therefore not used or fabricated.",
            "",
            "The high-risk feature family includes SIFT, PolyPhen, REVEL, CADD, MutationTaster, "
            "MetaSVM, MetaLR, FATHMM, PROVEAN, M-CAP, and related dbNSFP predictor scores, "
            "ranks, and prediction labels. The complete list is stored in "
            "`magpie_x_5k_feature_risk_audit.csv`.",
            "",
            "## Paired Changes",
            "",
        ]
    )
    for baseline_mode, enhanced_mode in ((MODE_A, MODE_C), (MODE_B, MODE_D)):
        baseline = indexed.loc[baseline_mode]
        enhanced = indexed.loc[enhanced_mode]
        lines.append(
            f"- `{baseline_mode}` to `{enhanced_mode}`: accuracy "
            f"{baseline['accuracy']:.4f} -> {enhanced['accuracy']:.4f} "
            f"({enhanced['accuracy'] - baseline['accuracy']:+.4f}); AUROC "
            f"{baseline['auroc']:.4f} -> {enhanced['auroc']:.4f}; AUPRC "
            f"{baseline['auprc']:.4f} -> {enhanced['auprc']:.4f}; Brier "
            f"{baseline['brier']:.4f} -> {enhanced['brier']:.4f}; ECE "
            f"{baseline['ece']:.4f} -> {enhanced['ece']:.4f}."
        )

    lines.extend(
        [
            "",
            "## GNN and AlphaMissense Gate",
            "",
            f"- Existing GNN/query ID overlap: {gnn_alignment['overlap']:,}/5,000.",
            f"- Existing GNN test size: {gnn_alignment['gnn_test_size']:,}; parity test size: "
            f"{gnn_alignment['parity_test_size']:,}; test-ID overlap: "
            f"{gnn_alignment['test_overlap']:,}.",
            "- The existing GNN prediction is not fused into the controlled metrics because its sample differs by one ID and its split is not the parity split.",
            "- Evaluating its predictions on the parity test rows would mix GNN training/validation predictions into the test set and would be invalid.",
            "",
            "## Interpretation",
            "",
            f"The best controlled 5k tabular mode is `{best['mode']}` with accuracy "
            f"{best['accuracy']:.4f} and AUROC {best['auroc']:.4f}. Any enhanced gain must be "
            "interpreted as higher-performance, higher-circularity evidence rather than an "
            "improvement to the low-circularity MAGPIE-parity claim.",
            "",
            "No federated learning, full-490k training, new GNN training, or AlphaMissense matching was run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scale_plan(path: Path, metrics: pd.DataFrame) -> None:
    enhanced = metrics[metrics["mode"].isin([MODE_C, MODE_D])]
    high_performance = bool((enhanced["accuracy"] >= 0.95).any())
    lines = [
        "# MAGPIE-X 10k/20k Controlled Expansion Plan",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"5k high-performance gate (at least one enhanced accuracy >= 0.95): "
        f"**{'passed' if high_performance else 'not passed'}**.",
        "",
        "## Proposed Paired Runs",
        "",
        "For each of 10k and 20k, reuse the exact approved AA/BLOSUM sample order and split already "
        "used by the SpliceAI parity comparison. Run only:",
        "",
        "- AA/BLOSUM low-circularity.",
        "- AA/BLOSUM plus SpliceAI low-circularity.",
        "- MAGPIE-X enhanced tabular: clean parity features plus flagged dbNSFP predictors.",
        "- MAGPIE-X enhanced tabular plus SpliceAI.",
        "",
        "Do not add AlphaMissense until a coordinate-matched sidecar exists and its coverage is audited. "
        "Do not add GNN residuals until predictions are generated from the same sample and split without "
        "test leakage.",
        "",
        "## Runtime Decision",
        "",
        "The tabular runtime is reasonable based on the completed 5k and parity runs, but 10k/20k are "
        "not executed by this stage. Keeping a separate decision gate prevents an enhanced result from "
        "silently expanding the approved scope and preserves a clean audit trail.",
        "",
        "## Required Outputs When Approved",
        "",
        "- `magpie_x_10k_20k_metrics.csv`",
        "- `MAGPIE_X_10K_20K_ENHANCED_REPORT.md`",
        "- temporary 10k/20k enhanced feature tables",
        "- per-feature circularity-risk manifests",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_gnn_alignment(combined: pd.DataFrame, test_idx) -> dict[str, object]:
    gnn_path = (
        ROOT
        / "outputs"
        / "real"
        / "magpie_x_gnn"
        / "gnn_residual_run"
        / "gnn_predictions.csv"
    )
    if not gnn_path.exists():
        return {
            "available": False,
            "overlap": 0,
            "gnn_test_size": 0,
            "parity_test_size": int(len(test_idx)),
            "test_overlap": 0,
        }
    gnn = pd.read_csv(gnn_path, low_memory=False)
    sample_ids = set(combined["variant_id"].astype(str))
    gnn_ids = set(gnn["variant_id"].astype(str))
    parity_test_ids = set(combined.iloc[test_idx]["variant_id"].astype(str))
    gnn_test_ids = set(gnn.loc[gnn["split"].eq("test"), "variant_id"].astype(str))
    return {
        "available": True,
        "overlap": int(len(sample_ids & gnn_ids)),
        "gnn_test_size": int(len(gnn_test_ids)),
        "parity_test_size": int(len(parity_test_ids)),
        "test_overlap": int(len(parity_test_ids & gnn_test_ids)),
        "sample_set_equal": sample_ids == gnn_ids,
        "test_set_equal": parity_test_ids == gnn_test_ids,
        "path": str(gnn_path),
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = normalize_missing_markers(pd.read_csv(args.input_5k, low_memory=False))
    if len(source) != args.sample_size:
        raise ValueError(f"Expected {args.sample_size:,} rows, found {len(source):,}.")
    if source["variant_id"].duplicated().any():
        raise ValueError("The 5k input contains duplicate variant IDs.")

    base_df = load_feature_table(args.base_input)
    base_spec = infer_features(base_df, args.max_categorical_cardinality)
    mode_specs, removed_predictors = build_mode_specs(base_spec)
    enhanced_base_spec = mode_specs["magpie_x_enhanced"]
    del base_df

    combined, improved_spec, _ = build_improved_low_circularity_spec(
        source,
        mode_specs["magpie_parity"],
    )
    aa_spec = build_aa_spec(improved_spec)
    spliceai_spec = build_spliceai_spec(aa_spec)

    removed_set = set(removed_predictors)
    risky_numeric = [
        column for column in enhanced_base_spec.numeric_columns if column in removed_set
    ]
    risky_categorical = [
        column for column in enhanced_base_spec.categorical_columns if column in removed_set
    ]
    if len(risky_numeric) + len(risky_categorical) != len(removed_predictors):
        raise RuntimeError("Not all removed predictor columns were recovered for enhanced mode.")

    enhanced_aa_spec = extend_spec(
        aa_spec,
        numeric=risky_numeric,
        categorical=risky_categorical,
    )
    enhanced_spliceai_spec = extend_spec(
        enhanced_aa_spec,
        numeric=SPLICEAI_NUMERIC_COLUMNS,
        categorical=SPLICEAI_CATEGORICAL_COLUMNS,
    )
    expected_counts = {
        MODE_A: 114,
        MODE_B: 126,
        MODE_C: 114 + len(removed_predictors),
        MODE_D: 126 + len(removed_predictors),
    }
    actual_counts = {
        MODE_A: len(aa_spec.feature_columns),
        MODE_B: len(spliceai_spec.feature_columns),
        MODE_C: len(enhanced_aa_spec.feature_columns),
        MODE_D: len(enhanced_spliceai_spec.feature_columns),
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"Unexpected feature counts: {actual_counts} != {expected_counts}")

    y = pd.to_numeric(combined["label"], errors="raise").to_numpy(dtype=int)
    fold, train_idx, test_idx = make_splits(
        y,
        seed=args.seed,
        test_size=args.test_size,
        cv_folds=1,
    )[0]
    gnn_alignment = audit_gnn_alignment(combined, test_idx)

    controlled_table = args.output_dir / CONTROLLED_FEATURE_TABLE
    combined.to_csv(controlled_table, index=False)
    risk_audit = write_feature_risk_audit(
        args.output_dir / "magpie_x_5k_feature_risk_audit.csv",
        aa_spec=aa_spec,
        spliceai_spec=spliceai_spec,
        risky_numeric=risky_numeric,
        risky_categorical=risky_categorical,
    )

    metric_rows = []
    for mode, spec, risk, includes_spliceai in (
        (MODE_A, aa_spec, "low", False),
        (MODE_B, spliceai_spec, "low", True),
        (MODE_C, enhanced_aa_spec, "high", False),
        (MODE_D, enhanced_spliceai_spec, "high", True),
    ):
        row = evaluate_mode(
            mode=mode,
            combined=combined,
            spec=spec,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            fold=fold,
            args=args,
            circularity_risk=risk,
            includes_spliceai=includes_spliceai,
        )
        metric_rows.append(row)
        print(
            f"mode={mode} features={row['n_features']} "
            f"accuracy={row['accuracy']:.4f} auroc={row['auroc']:.4f}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "magpie_x_5k_metrics.csv", index=False)

    alpha_resource = (
        ROOT / "data" / "external" / "alphamissense" / "AlphaMissense_hg38.tsv.gz"
    )
    alpha_matched = (
        ROOT / "outputs" / "real" / "magpie_x_residual" / "alphamissense_features.csv"
    )
    if alpha_matched.exists():
        raise RuntimeError(
            "A matched AlphaMissense artifact appeared during the run; audit it before inclusion."
        )

    write_report(
        args.output_dir / "MAGPIE_X_5K_ENHANCED_REPORT.md",
        metrics=metrics,
        risky_columns=removed_predictors,
        feature_table=controlled_table,
        alpha_resource_exists=alpha_resource.exists(),
        gnn_alignment=gnn_alignment,
    )
    write_scale_plan(
        args.output_dir / "MAGPIE_X_10K_20K_PLAN.md",
        metrics,
    )

    metadata = {
        "base_input": str(args.base_input),
        "controlled_5k_input": str(args.input_5k),
        "controlled_feature_table": str(controlled_table),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "test_size": args.test_size,
        "feature_counts": actual_counts,
        "risky_predictor_count": len(removed_predictors),
        "risky_predictor_columns": removed_predictors,
        "feature_risk_rows": int(len(risk_audit)),
        "alphamissense_resource_present": alpha_resource.exists(),
        "alphamissense_matched_artifact_present": False,
        "gnn_alignment": gnn_alignment,
        "full_490k_trained": False,
        "ten_or_twenty_k_trained": False,
        "federated_learning_run": False,
        "main_feature_table_modified": False,
    }
    (args.output_dir / "magpie_x_5k_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled 5k LightGBM comparison separating low-circularity "
            "MAGPIE-parity modes from high-circularity MAGPIE-X enhanced modes."
        )
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument("--input-5k", type=Path, default=DEFAULT_5K_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
