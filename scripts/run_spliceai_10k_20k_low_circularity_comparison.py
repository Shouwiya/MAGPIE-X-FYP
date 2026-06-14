#!/usr/bin/env python3
"""Aggregate and evaluate paired 10k/20k SpliceAI benchmark samples."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from run_aa_enhanced_low_circularity_comparison import build_aa_spec
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
    result_tuple,
)
from run_spliceai_5k_low_circularity_comparison import (
    AGGREGATED_SPLICEAI_COLUMNS,
    build_spliceai_spec,
    extract_query_rows,
    forbidden_features,
    load_match_table,
    match_audit,
    read_query_vcf,
    aggregate_match_table,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_INPUT = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_AA_INPUT = (
    ROOT / "data" / "interim" / "features" / "basic_feature_table_with_clean_aa_features.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_parity"
DEFAULT_SPLICEAI_DIR = ROOT / "data" / "external" / "spliceai" / "v1.3"

BASELINE_MODE = "aa_blosum_low_circularity"
SPLICEAI_MODE = "aa_blosum_spliceai_low_circularity"


def sample_paths(output_dir: Path, sample_size: int) -> dict[str, Path]:
    short = f"{sample_size // 1000}k"
    return {
        "query": ROOT
        / "data"
        / "interim"
        / "spliceai"
        / f"clinvar_reference_{short}_for_spliceai.dryrun.vcf",
        "match": output_dir / f"spliceai_{short}_match.tsv",
        "aggregated": output_dir / f"spliceai_{short}_aggregated_features.csv",
        "temporary": output_dir / f"reference_{short}_aa_spliceai_feature_table.csv",
    }


def evaluate_mode(
    *,
    mode: str,
    x: pd.DataFrame,
    y,
    train_idx,
    test_idx,
    fold: int,
    spec: FeatureSpec,
    sample_size: int,
    seed: int,
    ece_bins: int,
    spliceai_available: int,
) -> dict[str, object]:
    params, tuning = tune_lightgbm(
        x=x,
        y=y,
        train_idx=train_idx,
        spec=spec,
        seed=seed + sample_size,
        ece_bins=ece_bins,
    )
    metrics, _ = evaluate_pipeline(
        x=x,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        spec=spec,
        model=make_lightgbm_model(params),
        ece_bins=ece_bins,
    )
    selected_json = json.dumps(params, sort_keys=True)
    selected_candidate = next(
        row["candidate_id"]
        for row in tuning
        if json.dumps(row["params"], sort_keys=True) == selected_json
    )
    output = {
        "mode": mode,
        "sample_size": sample_size,
        "model": "lightgbm",
        "fold": int(fold),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(len(spec.feature_columns)),
        "spliceai_available": spliceai_available,
        "selected_candidate_id": int(selected_candidate),
        "selected_params_json": selected_json,
        **metrics,
    }
    output["result_tuple"] = result_tuple(output)
    return output


def write_aggregation_report(
    path: Path,
    *,
    sample_sizes: list[int],
    audits: dict[int, dict[str, object]],
    aggregates: dict[int, pd.DataFrame],
) -> None:
    lines = [
        "# SpliceAI 10k/20k Aggregation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Official masked hg38 SpliceAI v1.3 matches were aggregated without labels. "
        "The long-form match TSVs were preserved unchanged.",
        "",
        "## Coverage",
        "",
        "| sample | query IDs | real-score queries | unavailable | long-form rows | multi-record queries | max DS > 0 | max DS >= 0.2 | max DS >= 0.5 | max DS >= 0.8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sample_size in sample_sizes:
        audit = audits[sample_size]
        lines.append(
            f"| {sample_size // 1000}k | {sample_size:,} | "
            f"{int(audit['matched_queries']):,} | {int(audit['unmatched_queries']):,} | "
            f"{int(audit['long_form_rows']):,} | {int(audit['multi_record_queries']):,} | "
            f"{int(audit['query_level_max_DS_gt_0']):,} | "
            f"{int(audit['query_level_max_DS_ge_0_2']):,} | "
            f"{int(audit['query_level_max_DS_ge_0_5']):,} | "
            f"{int(audit['query_level_max_DS_ge_0_8']):,} |"
        )

    lines.extend(
        [
            "",
            "## Matched Record Sources",
            "",
            "| sample | matched SNV records | matched indel records | unsupported query rows |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for sample_size in sample_sizes:
        audit = audits[sample_size]
        matched_sources = audit["matched_record_source_counts"]
        all_sources = audit["all_row_source_counts"]
        lines.append(
            f"| {sample_size // 1000}k | {int(matched_sources.get('snv', 0)):,} | "
            f"{int(matched_sources.get('indel', 0)):,} | "
            f"{int(all_sources.get('unsupported', 0)):,} |"
        )

    lines.extend(
        [
            "",
            "## Aggregation Rule",
            "",
            "- Compute `max_DS_AG`, `max_DS_AL`, `max_DS_DG`, and `max_DS_DL` independently per query.",
            "- Take each DP field from the deterministic first record supplying its corresponding maximum DS.",
            "- Set `max_spliceai_DS` to the maximum of the four aggregated DS values.",
            "- Set `spliceai_available=1` only when at least one real resource record matched.",
            "- Preserve real-record multiplicity in `spliceai_record_count` and resource types in `spliceai_source_types`.",
            "- Leave DS/DP missing when no exact resource record exists; do not fabricate zero scores.",
            "- Do not select a record using a clinical label.",
            "",
            "## Sidecars",
            "",
        ]
    )
    for sample_size in sample_sizes:
        paths = sample_paths(path.parent, sample_size)
        aggregate = aggregates[sample_size]
        lines.append(
            f"- `outputs/real/magpie_parity/{paths['aggregated'].name}`: {len(aggregate):,} rows, "
            f"{int(aggregate['spliceai_available'].sum()):,} available."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_report(path: Path, metrics: pd.DataFrame) -> None:
    summary = metrics[
        ["sample_size", "mode", "n_features", "spliceai_available", "result_tuple"]
    ].copy()
    lines = [
        "# SpliceAI 10k/20k Low-Circularity Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
        "## Configuration",
        "",
        "- Exact approved 10k and 20k balanced samples from the AA/BLOSUM benchmark.",
        "- Identical sample order, split seed, and test rows within each paired comparison.",
        "- LightGBM only, using the established six-candidate validation-AUPRC tuning grid.",
        "- AA/BLOSUM low-circularity baseline compared with the same baseline plus aggregated SpliceAI features.",
        "- Unavailable SpliceAI values remain missing with an explicit availability flag.",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(markdown_table(summary))
    lines.extend(["", "## Paired Changes", ""])
    for sample_size in sorted(metrics["sample_size"].unique()):
        indexed = metrics[metrics["sample_size"].eq(sample_size)].set_index("mode")
        baseline = indexed.loc[BASELINE_MODE]
        enhanced = indexed.loc[SPLICEAI_MODE]
        lines.append(
            f"- {sample_size // 1000}k: accuracy {baseline['accuracy']:.4f} -> "
            f"{enhanced['accuracy']:.4f} ({enhanced['accuracy'] - baseline['accuracy']:+.4f}); "
            f"MCC {baseline['mcc']:.4f} -> {enhanced['mcc']:.4f}; "
            f"AUROC {baseline['auroc']:.4f} -> {enhanced['auroc']:.4f}; "
            f"AUPRC {baseline['auprc']:.4f} -> {enhanced['auprc']:.4f}; "
            f"Brier {baseline['brier']:.4f} -> {enhanced['brier']:.4f}; "
            f"ECE {baseline['ece']:.4f} -> {enhanced['ece']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "Neither mode uses SIFT, PolyPhen, REVEL, CADD, MutationTaster, MetaSVM, MetaLR, "
            "FATHMM, PROVEAN, M-CAP, AlphaMissense, ESM, GNN outputs, or similar prior "
            "pathogenicity predictors. The protected full AA feature table was read only.",
            "",
            "SpliceAI improved the 5k pilot accuracy, but final benefit is evaluated at "
            "10k/20k and later under gene-held-out/external validation. The completed paired "
            "checks were accuracy-neutral at 10k and slightly worse at 20k, so they do not "
            "confirm a consistent benefit. These random-split results remain preliminary and "
            "are not a full MAGPIE reproduction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources = [
        args.spliceai_snv_vcf,
        Path(f"{args.spliceai_snv_vcf}.tbi"),
        args.spliceai_indel_vcf,
        Path(f"{args.spliceai_indel_vcf}.tbi"),
    ]
    for resource in resources:
        if not resource.is_file():
            raise FileNotFoundError(f"Required SpliceAI resource not found: {resource}")

    base_df = load_feature_table(args.base_input)
    base_spec = infer_features(base_df, args.max_categorical_cardinality)
    mode_specs, removed_predictors = build_mode_specs(base_spec)
    old_spec = mode_specs["magpie_parity"]
    del base_df

    audits: dict[int, dict[str, object]] = {}
    aggregates: dict[int, pd.DataFrame] = {}
    metric_rows: list[dict[str, object]] = []
    metadata_samples: dict[str, object] = {}

    for sample_size in args.sample_sizes:
        paths = sample_paths(args.output_dir, sample_size)
        query, vcf_metadata = read_query_vcf(paths["query"], expected_records=sample_size)
        match = load_match_table(paths["match"], set(query["query_id"]))
        audit = match_audit(match, query)
        aggregated = aggregate_match_table(match, query)
        aggregated.to_csv(paths["aggregated"], index=False)
        audits[sample_size] = audit
        aggregates[sample_size] = aggregated

        source_sample = extract_query_rows(args.aa_input, query, args.chunksize)
        temporary = source_sample.merge(
            aggregated,
            left_on="variant_id",
            right_on="query_id",
            how="left",
            validate="one_to_one",
        ).drop(columns="query_id")
        temporary.to_csv(paths["temporary"], index=False)

        expected_per_class = sample_size // 2
        expected_counts = {0: expected_per_class, 1: expected_per_class}
        if temporary["label"].value_counts().to_dict() != expected_counts:
            raise ValueError(
                f"Expected balanced labels for {sample_size:,}, got "
                f"{temporary['label'].value_counts().to_dict()}."
            )

        combined, improved_spec, _ = build_improved_low_circularity_spec(temporary, old_spec)
        aa_spec = build_aa_spec(improved_spec)
        spliceai_spec = build_spliceai_spec(aa_spec)
        forbidden_used = forbidden_features(spliceai_spec.feature_columns)
        if forbidden_used:
            raise ValueError(
                f"Forbidden predictor features entered low-circularity mode: {forbidden_used}"
            )
        if len(aa_spec.feature_columns) != 114:
            raise ValueError(
                f"Expected 114 AA/BLOSUM features, got {len(aa_spec.feature_columns)}."
            )

        y = pd.to_numeric(combined["label"], errors="raise").to_numpy(dtype=int)
        fold, train_idx, test_idx = make_splits(
            y,
            seed=args.seed,
            test_size=args.test_size,
            cv_folds=1,
        )[0]
        available = int(aggregated["spliceai_available"].sum())
        for mode, spec in (
            (BASELINE_MODE, aa_spec),
            (SPLICEAI_MODE, spliceai_spec),
        ):
            row = evaluate_mode(
                mode=mode,
                x=combined[spec.feature_columns].reset_index(drop=True),
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                fold=fold,
                spec=spec,
                sample_size=sample_size,
                seed=args.seed,
                ece_bins=args.ece_bins,
                spliceai_available=available,
            )
            metric_rows.append(row)
            print(
                f"sample_size={sample_size} mode={mode} "
                f"accuracy={row['accuracy']:.4f} auroc={row['auroc']:.4f}",
                flush=True,
            )

        metadata_samples[str(sample_size)] = {
            "query_vcf": str(paths["query"]),
            "match_input": str(paths["match"]),
            "aggregated_output": str(paths["aggregated"]),
            "temporary_feature_table": str(paths["temporary"]),
            "vcf_metadata": vcf_metadata,
            "audit": audit,
        }

    metrics = pd.DataFrame(metric_rows).sort_values(["sample_size", "mode"]).reset_index(drop=True)
    metrics.to_csv(
        args.output_dir / "spliceai_10k_20k_low_circularity_metrics.csv",
        index=False,
    )
    write_aggregation_report(
        args.output_dir / "SPLICEAI_10K_20K_AGGREGATION_REPORT.md",
        sample_sizes=args.sample_sizes,
        audits=audits,
        aggregates=aggregates,
    )
    write_model_report(
        args.output_dir / "SPLICEAI_10K_20K_LOW_CIRCULARITY_REPORT.md",
        metrics,
    )
    metadata = {
        "base_input": str(args.base_input),
        "aa_input": str(args.aa_input),
        "sample_sizes": args.sample_sizes,
        "seed": args.seed,
        "test_size": args.test_size,
        "removed_prior_predictor_columns": removed_predictors,
        "samples": metadata_samples,
        "full_490k_trained": False,
        "federated_learning_run": False,
        "magpie_x_enhanced_run": False,
        "main_feature_table_modified": False,
    }
    (args.output_dir / "spliceai_10k_20k_low_circularity_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate official precomputed SpliceAI matches and run paired "
            "10k/20k AA+BLOSUM versus AA+BLOSUM+SpliceAI LightGBM comparisons."
        )
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument("--aa-input", type=Path, default=DEFAULT_AA_INPUT)
    parser.add_argument(
        "--spliceai-snv-vcf",
        type=Path,
        default=DEFAULT_SPLICEAI_DIR / "spliceai_scores.masked.snv.hg38.vcf.gz",
    )
    parser.add_argument(
        "--spliceai-indel-vcf",
        type=Path,
        default=DEFAULT_SPLICEAI_DIR / "spliceai_scores.masked.indel.hg38.vcf.gz",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[10_000, 20_000])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    parser.add_argument("--chunksize", type=int, default=25_000)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
