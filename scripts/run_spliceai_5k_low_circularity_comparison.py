from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from run_aa_enhanced_low_circularity_comparison import build_aa_spec
from run_circularity_ablation import build_mode_specs, is_prior_predictor_column
from run_magpie_low_circularity_baseline import (
    build_improved_low_circularity_spec,
    evaluate_pipeline,
    make_lightgbm_model,
    matches_explicit_forbidden_predictor,
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_INPUT = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_AA_INPUT = (
    ROOT / "data" / "interim" / "features" / "basic_feature_table_with_clean_aa_features.csv"
)
DEFAULT_QUERY_VCF = (
    ROOT / "data" / "interim" / "spliceai" / "clinvar_reference_5k_for_spliceai.dryrun.vcf"
)
DEFAULT_MATCH_INPUT = ROOT / "outputs" / "real" / "magpie_parity" / "spliceai_5k_match.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_parity"

DS_COLUMNS = ["DS_AG", "DS_AL", "DS_DG", "DS_DL"]
DP_COLUMNS = ["DP_AG", "DP_AL", "DP_DG", "DP_DL"]
AGGREGATED_SPLICEAI_COLUMNS = [
    "max_DS_AG",
    "max_DS_AL",
    "max_DS_DG",
    "max_DS_DL",
    "max_spliceai_DS",
    "DP_AG",
    "DP_AL",
    "DP_DG",
    "DP_DL",
    "spliceai_available",
    "spliceai_record_count",
    "spliceai_source_types",
]
SPLICEAI_NUMERIC_COLUMNS = [
    column for column in AGGREGATED_SPLICEAI_COLUMNS if column != "spliceai_source_types"
]
SPLICEAI_CATEGORICAL_COLUMNS = ["spliceai_source_types"]
ADDITIONAL_FORBIDDEN_PATTERNS = ("gnn",)


def read_query_vcf(
    path: Path,
    expected_records: int | None = 5_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Query VCF not found: {path}")

    rows: list[dict[str, object]] = []
    header_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##"):
                header_lines.append(line.rstrip("\n"))
                continue
            if line.startswith("#CHROM"):
                continue
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                raise ValueError(f"Malformed VCF record with {len(fields)} fields: {line[:120]}")
            chrom, pos, variant_id, ref, alt, _, _, info = fields[:8]
            rows.append(
                {
                    "query_order": len(rows),
                    "query_chrom": chrom,
                    "query_pos": int(pos),
                    "query_id": variant_id,
                    "query_ref": ref,
                    "query_alt": alt,
                    "query_info": info,
                }
            )

    query = pd.DataFrame(rows)
    if expected_records is not None and len(query) != expected_records:
        raise ValueError(
            f"Expected {expected_records:,} query VCF records, found {len(query):,}."
        )
    if query["query_id"].duplicated().any():
        raise ValueError("Query VCF contains duplicate variant IDs.")

    lower_header = "\n".join(header_lines).lower()
    label_tokens = ("pathogenic", "benign", "clinical_significance", "clin_sig", "label=")
    labels_present = any(token in lower_header for token in label_tokens)
    labels_present = labels_present or query["query_info"].astype(str).str.lower().str.contains(
        "pathogenic|benign|clinical_significance|clin_sig|label=",
        regex=True,
    ).any()
    if labels_present:
        raise ValueError("Clinical labels or significance text were found in the query VCF.")

    metadata = {
        "records": int(len(query)),
        "unique_ids": int(query["query_id"].nunique()),
        "has_contig_headers": any(line.startswith("##contig=") for line in header_lines),
        "labels_present": False,
    }
    return query, metadata


def load_match_table(path: Path, query_ids: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"SpliceAI match table not found: {path}")
    match = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "query_id",
        "spliceai_source",
        "matched",
        "symbol",
        *DS_COLUMNS,
        *DP_COLUMNS,
    }
    missing = sorted(required - set(match.columns))
    if missing:
        raise ValueError(f"SpliceAI match table is missing columns: {missing}")
    if set(match["query_id"].astype(str)) != query_ids:
        missing_ids = query_ids - set(match["query_id"].astype(str))
        extra_ids = set(match["query_id"].astype(str)) - query_ids
        raise ValueError(
            f"Match/query ID mismatch: {len(missing_ids)} missing and {len(extra_ids)} extra."
        )
    match["query_id"] = match["query_id"].astype(str)
    match["matched"] = pd.to_numeric(match["matched"], errors="raise").astype(int)
    if not match["matched"].isin([0, 1]).all():
        raise ValueError("The matched column must contain only 0 and 1.")
    for column in [*DS_COLUMNS, *DP_COLUMNS]:
        match[column] = pd.to_numeric(match[column], errors="coerce")
    match["_match_row_order"] = np.arange(len(match), dtype=int)
    return match


def aggregate_one_query(group: pd.DataFrame) -> dict[str, object]:
    matched_records = group[group["matched"].eq(1)].sort_values("_match_row_order")
    output: dict[str, object] = {
        "query_id": str(group["query_id"].iloc[0]),
        "spliceai_available": int(not matched_records.empty),
        "spliceai_record_count": int(len(matched_records)),
        "spliceai_source_types": "|".join(
            sorted(matched_records["spliceai_source"].dropna().astype(str).unique())
        ),
    }

    all_ds_values: list[float] = []
    for ds_column, dp_column in zip(DS_COLUMNS, DP_COLUMNS):
        output_column = f"max_{ds_column}"
        valid = matched_records[matched_records[ds_column].notna()]
        if valid.empty:
            output[output_column] = np.nan
            output[dp_column] = np.nan
            continue
        max_value = float(valid[ds_column].max())
        winner = valid[valid[ds_column].eq(max_value)].iloc[0]
        output[output_column] = max_value
        output[dp_column] = winner[dp_column]
        all_ds_values.append(max_value)

    output["max_spliceai_DS"] = max(all_ds_values) if all_ds_values else np.nan
    return output


def aggregate_match_table(match: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    rows = [
        aggregate_one_query(group)
        for _, group in match.groupby("query_id", sort=False, dropna=False)
    ]
    aggregated = pd.DataFrame(rows)
    aggregated = query[["query_order", "query_id"]].merge(
        aggregated,
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    if aggregated[AGGREGATED_SPLICEAI_COLUMNS].isna().all(axis=1).any():
        raise ValueError("One or more query IDs lack an aggregated SpliceAI row.")
    aggregated = aggregated.sort_values("query_order").drop(columns="query_order").reset_index(drop=True)
    return aggregated[["query_id", *AGGREGATED_SPLICEAI_COLUMNS]]


def extract_query_rows(path: Path, query: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"AA feature table not found: {path}")
    query_ids = set(query["query_id"])
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        selected = chunk[chunk["variant_id"].astype(str).isin(query_ids)]
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError("No query variants were found in the AA feature table.")

    selected = pd.concat(pieces, ignore_index=True)
    selected["variant_id"] = selected["variant_id"].astype(str)
    if selected["variant_id"].duplicated().any():
        duplicates = selected.loc[selected["variant_id"].duplicated(False), "variant_id"].unique()
        raise ValueError(f"Duplicate query IDs in AA feature table: {duplicates[:5].tolist()}")
    missing = query_ids - set(selected["variant_id"])
    if missing:
        raise ValueError(f"{len(missing)} query variants are absent from the AA feature table.")

    order = query[["query_order", "query_id"]].rename(columns={"query_id": "variant_id"})
    selected = order.merge(selected, on="variant_id", how="left", validate="one_to_one")
    return selected.sort_values("query_order").drop(columns="query_order").reset_index(drop=True)


def build_spliceai_spec(aa_spec: FeatureSpec) -> FeatureSpec:
    for column in AGGREGATED_SPLICEAI_COLUMNS:
        if matches_explicit_forbidden_predictor(column) or is_prior_predictor_column(column):
            raise ValueError(f"SpliceAI feature matched a prior-predictor pattern: {column}")
    return FeatureSpec(
        feature_columns=[*aa_spec.feature_columns, *AGGREGATED_SPLICEAI_COLUMNS],
        numeric_columns=[*aa_spec.numeric_columns, *SPLICEAI_NUMERIC_COLUMNS],
        categorical_columns=[*aa_spec.categorical_columns, *SPLICEAI_CATEGORICAL_COLUMNS],
        dropped_columns=dict(aa_spec.dropped_columns),
    )


def forbidden_features(columns: Iterable[str]) -> list[str]:
    forbidden = []
    for column in columns:
        normalized = column.lower().replace("-", "").replace("_", "")
        if (
            matches_explicit_forbidden_predictor(column)
            or is_prior_predictor_column(column)
            or any(pattern in normalized for pattern in ADDITIONAL_FORBIDDEN_PATTERNS)
        ):
            forbidden.append(column)
    return forbidden


def match_audit(match: pd.DataFrame, query: pd.DataFrame) -> dict[str, object]:
    matched = match[match["matched"].eq(1)].copy()
    row_max = matched[DS_COLUMNS].max(axis=1, skipna=True)
    query_max = matched.groupby("query_id")[DS_COLUMNS].max().max(axis=1, skipna=True)
    all_query_max = query[["query_id"]].merge(
        query_max.rename("max_DS"),
        left_on="query_id",
        right_index=True,
        how="left",
    )["max_DS"]
    per_query_rows = match.groupby("query_id").size()
    per_query_symbols = match.groupby("query_id")["symbol"].nunique(dropna=True)
    unique_query_source = match.groupby("query_id")["spliceai_source"].first()
    return {
        "long_form_rows": int(len(match)),
        "query_ids_represented": int(match["query_id"].nunique()),
        "matched_score_records": int(len(matched)),
        "matched_queries": int(matched["query_id"].nunique()),
        "unmatched_queries": int(len(query) - matched["query_id"].nunique()),
        "all_row_source_counts": match["spliceai_source"].value_counts().to_dict(),
        "query_source_counts": unique_query_source.value_counts().to_dict(),
        "matched_record_source_counts": matched["spliceai_source"].value_counts().to_dict(),
        "unique_symbols": int(matched["symbol"].nunique(dropna=True)),
        "multi_record_queries": int((per_query_rows > 1).sum()),
        "multi_symbol_queries": int((per_query_symbols > 1).sum()),
        "record_level_max_DS_gt_0": int((row_max > 0).sum()),
        "record_level_max_DS_ge_0_2": int((row_max >= 0.2).sum()),
        "record_level_max_DS_ge_0_5": int((row_max >= 0.5).sum()),
        "record_level_max_DS_ge_0_8": int((row_max >= 0.8).sum()),
        "query_level_max_DS_gt_0": int((all_query_max > 0).sum()),
        "query_level_max_DS_ge_0_2": int((all_query_max >= 0.2).sum()),
        "query_level_max_DS_ge_0_5": int((all_query_max >= 0.5).sum()),
        "query_level_max_DS_ge_0_8": int((all_query_max >= 0.8).sum()),
    }


def write_match_audit(
    path: Path,
    *,
    args: argparse.Namespace,
    audit: dict[str, object],
    vcf_metadata: dict[str, object],
) -> None:
    query_count = int(vcf_metadata["records"])
    matched_queries = int(audit["matched_queries"])
    coverage = matched_queries / query_count
    lines = [
        "# SpliceAI 5k Match Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Files Used",
        "",
        f"- Query VCF: `{args.query_vcf}`",
        f"- Long-form match: `{args.match_input}`",
        f"- SNV resource: `{args.spliceai_snv_vcf}`",
        f"- SNV index: `{args.spliceai_snv_vcf}.tbi`",
        f"- Indel resource: `{args.spliceai_indel_vcf}`",
        f"- Indel index: `{args.spliceai_indel_vcf}.tbi`",
        "",
        "## Coverage",
        "",
        f"- Query VCF records: {query_count:,}",
        f"- Query IDs represented in the long-form output: {int(audit['query_ids_represented']):,} / {query_count:,} (100.00%).",
        f"- Queries with at least one real SpliceAI score record: {matched_queries:,} / {query_count:,} ({coverage:.2%}).",
        f"- Queries represented only by an unmatched placeholder row: {int(audit['unmatched_queries']):,}.",
        f"- Long-form rows: {int(audit['long_form_rows']):,}, comprising {int(audit['matched_score_records']):,} matched score records and {int(audit['unmatched_queries']):,} unmatched placeholders.",
        "",
        "A query ID appearing in the output is not by itself evidence that a SpliceAI score was available. The `matched` flag is required; modeling availability is therefore 4,582 rather than 5,000.",
        "",
        "## SNV and Indel Counts",
        "",
        f"- Long-form source routing: SNV {int(audit['all_row_source_counts'].get('snv', 0)):,}; indel {int(audit['all_row_source_counts'].get('indel', 0)):,}.",
        f"- Unique query source routing: SNV {int(audit['query_source_counts'].get('snv', 0)):,}; indel {int(audit['query_source_counts'].get('indel', 0)):,}.",
        f"- Matched score records: SNV {int(audit['matched_record_source_counts'].get('snv', 0)):,}; indel {int(audit['matched_record_source_counts'].get('indel', 0)):,}.",
        "",
        "## DS Thresholds",
        "",
        "| level | max DS > 0 | max DS >= 0.2 | max DS >= 0.5 | max DS >= 0.8 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| long-form score records | {int(audit['record_level_max_DS_gt_0']):,} | {int(audit['record_level_max_DS_ge_0_2']):,} | {int(audit['record_level_max_DS_ge_0_5']):,} | {int(audit['record_level_max_DS_ge_0_8']):,} |",
        f"| unique queries after max aggregation | {int(audit['query_level_max_DS_gt_0']):,} | {int(audit['query_level_max_DS_ge_0_2']):,} | {int(audit['query_level_max_DS_ge_0_5']):,} | {int(audit['query_level_max_DS_ge_0_8']):,} |",
        "",
        "The previously supplied 1,746 / 502 / 381 / 336 figures are record-level counts. The unique-query counts are used for the modeling sidecar.",
        "",
        "## Multiple Records",
        "",
        f"- Unique symbols among matched records: {int(audit['unique_symbols']):,}.",
        f"- Queries with multiple long-form records: {int(audit['multi_record_queries']):,}.",
        f"- Queries with multiple symbols: {int(audit['multi_symbol_queries']):,}.",
        "",
        "Multiple records are retained unchanged in the long-form TSV. The modeling sidecar uses per-score maxima and the DP value from the deterministic first record tied for that score maximum; it never selects by clinical label.",
        "",
        "## Contig Header Warning",
        "",
        "The query VCF has no `##contig=<...>` header declarations. Pysam/htslib can therefore emit warnings such as `Contig '1' is not defined in the header` while parsing records. Exact matching still completed, but future generated VCFs should include GRCh38 contig declarations to remove the warning and strengthen header validation.",
        "",
        "## Label Guardrail",
        "",
        "- The query VCF contains no labels, clinical significance, genes, or model features.",
        "- The long-form matching script reads VCF coordinates/alleles and precomputed SpliceAI INFO only.",
        "- Clinical labels were not used to select records, genes, DS values, or DP positions.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_aggregation_report(
    path: Path,
    *,
    aggregated: pd.DataFrame,
    audit: dict[str, object],
    args: argparse.Namespace,
) -> None:
    available = int(aggregated["spliceai_available"].sum())
    lines = [
        "# SpliceAI 5k Aggregation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Input long-form file: `{args.match_input}`",
        f"Output sidecar: `{args.output_dir / 'spliceai_5k_aggregated_features.csv'}`",
        "",
        "## Rule",
        "",
        "- Preserve the long-form match file unchanged.",
        "- Produce exactly one sidecar row per `query_id`.",
        "- Use the maximum value independently for `DS_AG`, `DS_AL`, `DS_DG`, and `DS_DL`.",
        "- Pair each DP field with the record supplying its corresponding maximum DS.",
        "- Resolve exact DS ties deterministically by original long-form row order.",
        "- Set `max_spliceai_DS` to the maximum of the four aggregated DS values.",
        "- Set `spliceai_available=1` only when at least one real matched score record exists; unmatched placeholder rows receive 0.",
        "- Count only real matched score records in `spliceai_record_count`.",
        "- Join distinct matched source types with `|`.",
        "- Do not use clinical labels.",
        "",
        "## Output Audit",
        "",
        f"- Aggregated rows: {len(aggregated):,}",
        f"- Unique query IDs: {aggregated['query_id'].nunique():,}",
        f"- SpliceAI available: {available:,}",
        f"- SpliceAI unavailable: {len(aggregated) - available:,}",
        f"- Queries with multiple real records: {int((aggregated['spliceai_record_count'] > 1).sum()):,}",
        f"- Maximum record count for one query: {int(aggregated['spliceai_record_count'].max()):,}",
        f"- Query-level `max_spliceai_DS > 0`: {int(audit['query_level_max_DS_gt_0']):,}",
        f"- Query-level `max_spliceai_DS >= 0.2`: {int(audit['query_level_max_DS_ge_0_2']):,}",
        f"- Query-level `max_spliceai_DS >= 0.5`: {int(audit['query_level_max_DS_ge_0_5']):,}",
        f"- Query-level `max_spliceai_DS >= 0.8`: {int(audit['query_level_max_DS_ge_0_8']):,}",
        "",
        "The availability definition deliberately distinguishes matched SpliceAI annotations from unmatched output placeholders. Treating all 5,000 represented IDs as available would fabricate 418 annotations.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_model_report(
    path: Path,
    *,
    metrics: pd.DataFrame,
    args: argparse.Namespace,
    aa_feature_count: int,
    spliceai_feature_count: int,
    forbidden_used: list[str],
) -> None:
    summary = metrics[
        ["mode", "sample_size", "model", "n_features", "spliceai_available", "result_tuple"]
    ].copy()
    baseline = metrics[metrics["mode"].eq("aa_blosum_low_circularity")].iloc[0]
    enhanced = metrics[metrics["mode"].eq("aa_blosum_spliceai_low_circularity")].iloc[0]
    lines = [
        "# SpliceAI 5k Low-Circularity Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
        "## Configuration",
        "",
        "- Real ClinVar-derived 5k query variants only.",
        "- Both modes use the identical ordered sample and identical stratified 4,000/1,000 train/test split.",
        f"- Split seed: {args.seed}.",
        "- LightGBM only.",
        "- Each mode uses the established six-candidate grid with validation-AUPRC selection inside the shared training partition.",
        f"- AA/BLOSUM baseline features: {aa_feature_count}.",
        f"- AA/BLOSUM plus SpliceAI features: {spliceai_feature_count}.",
        f"- Forbidden prior-predictor features used: {len(forbidden_used)}.",
        "",
        "The SpliceAI pilot query was sampled with seed 13 during VCF construction. The earlier AA/BLOSUM sample-size script used seed 5013 for its 5k sample, so its published 0.9210 result is not a directly paired baseline for this pilot. This report re-runs both modes on the exact same SpliceAI query sample.",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(markdown_table(summary))
    lines.extend(
        [
            "",
            "## Paired Change",
            "",
            f"- Accuracy: {float(baseline['accuracy']):.4f} -> {float(enhanced['accuracy']):.4f} ({float(enhanced['accuracy']) - float(baseline['accuracy']):+.4f}).",
            f"- F1: {float(baseline['f1']):.4f} -> {float(enhanced['f1']):.4f} ({float(enhanced['f1']) - float(baseline['f1']):+.4f}).",
            f"- MCC: {float(baseline['mcc']):.4f} -> {float(enhanced['mcc']):.4f} ({float(enhanced['mcc']) - float(baseline['mcc']):+.4f}).",
            f"- AUROC: {float(baseline['auroc']):.4f} -> {float(enhanced['auroc']):.4f} ({float(enhanced['auroc']) - float(baseline['auroc']):+.4f}).",
            f"- AUPRC: {float(baseline['auprc']):.4f} -> {float(enhanced['auprc']):.4f} ({float(enhanced['auprc']) - float(baseline['auprc']):+.4f}).",
            f"- Brier: {float(baseline['brier']):.4f} -> {float(enhanced['brier']):.4f} ({float(enhanced['brier']) - float(baseline['brier']):+.4f}; lower is better).",
            f"- ECE: {float(baseline['ece']):.4f} -> {float(enhanced['ece']):.4f} ({float(enhanced['ece']) - float(baseline['ece']):+.4f}; lower is better).",
            "",
            "## Circularity Guardrail",
            "",
            "Neither model uses SIFT, PolyPhen, REVEL, CADD, MutationTaster, MetaSVM, MetaLR, FATHMM, PROVEAN, M-CAP, AlphaMissense, ESM, GNN outputs, or related prior pathogenicity predictor fields. SpliceAI is evaluated as a mechanistic splicing feature group, separately from prior pathogenicity predictors.",
            "",
            "## Interpretation",
            "",
            "This is a preliminary random-split 5k pilot. It does not establish generalization or a universal gain. Results remain preliminary until 10k/20k and gene-held-out/external validation are run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for resource in (
        args.spliceai_snv_vcf,
        Path(f"{args.spliceai_snv_vcf}.tbi"),
        args.spliceai_indel_vcf,
        Path(f"{args.spliceai_indel_vcf}.tbi"),
    ):
        if not resource.is_file():
            raise FileNotFoundError(f"Required SpliceAI resource not found: {resource}")

    query, vcf_metadata = read_query_vcf(args.query_vcf)
    match = load_match_table(args.match_input, set(query["query_id"]))
    audit = match_audit(match, query)
    aggregated = aggregate_match_table(match, query)

    aggregated_path = args.output_dir / "spliceai_5k_aggregated_features.csv"
    aggregated.to_csv(aggregated_path, index=False)
    write_match_audit(
        args.output_dir / "SPLICEAI_5K_MATCH_AUDIT.md",
        args=args,
        audit=audit,
        vcf_metadata=vcf_metadata,
    )
    write_aggregation_report(
        args.output_dir / "SPLICEAI_5K_AGGREGATION_REPORT.md",
        aggregated=aggregated,
        audit=audit,
        args=args,
    )

    source_5k = extract_query_rows(args.aa_input, query, args.chunksize)
    temporary = source_5k.merge(
        aggregated,
        left_on="variant_id",
        right_on="query_id",
        how="left",
        validate="one_to_one",
    ).drop(columns="query_id")
    temporary_path = args.output_dir / "reference_5k_aa_spliceai_feature_table.csv"
    temporary.to_csv(temporary_path, index=False)

    if temporary["label"].value_counts().to_dict() != {0: 2_500, 1: 2_500}:
        raise ValueError(
            f"Expected balanced 2,500/2,500 labels, got {temporary['label'].value_counts().to_dict()}."
        )

    # Infer the established clean base specification from the full base table, but train only on 5k.
    base_df = load_feature_table(args.base_input)
    base_spec = infer_features(base_df, args.max_categorical_cardinality)
    mode_specs, removed_predictors = build_mode_specs(base_spec)
    old_spec = mode_specs["magpie_parity"]
    del base_df

    combined_5k, improved_spec, _ = build_improved_low_circularity_spec(temporary, old_spec)
    aa_spec = build_aa_spec(improved_spec)
    spliceai_spec = build_spliceai_spec(aa_spec)
    forbidden_used = forbidden_features(spliceai_spec.feature_columns)
    if forbidden_used:
        raise ValueError(f"Forbidden predictor features entered low-circularity mode: {forbidden_used}")
    if len(aa_spec.feature_columns) != 114:
        raise ValueError(
            f"Expected established AA/BLOSUM baseline to have 114 features, got {len(aa_spec.feature_columns)}."
        )

    y = pd.to_numeric(combined_5k["label"], errors="raise").to_numpy(dtype=int)
    fold, train_idx, test_idx = make_splits(
        y,
        seed=args.seed,
        test_size=args.test_size,
        cv_folds=1,
    )[0]
    test_ids = combined_5k.iloc[test_idx]["variant_id"].astype(str).tolist()

    modes = [
        (
            "aa_blosum_low_circularity",
            combined_5k[aa_spec.feature_columns].reset_index(drop=True),
            aa_spec,
        ),
        (
            "aa_blosum_spliceai_low_circularity",
            combined_5k[spliceai_spec.feature_columns].reset_index(drop=True),
            spliceai_spec,
        ),
    ]
    metric_rows: list[dict[str, object]] = []
    for mode, x, spec in modes:
        params, tuning = tune_lightgbm(
            x=x,
            y=y,
            train_idx=train_idx,
            spec=spec,
            seed=args.seed + 5_000,
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
            "sample_size": 5_000,
            "model": "lightgbm",
            "fold": int(fold),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(len(spec.feature_columns)),
            "spliceai_available": int(aggregated["spliceai_available"].sum()),
            "selected_candidate_id": int(selected_candidate),
            "selected_params_json": selected_json,
            **metrics,
        }
        output["result_tuple"] = result_tuple(output)
        metric_rows.append(output)
        print(
            f"mode={mode} accuracy={float(metrics['accuracy']):.4f} "
            f"auroc={float(metrics['auroc']):.4f}",
            flush=True,
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = args.output_dir / "spliceai_5k_low_circularity_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    write_model_report(
        args.output_dir / "SPLICEAI_5K_LOW_CIRCULARITY_REPORT.md",
        metrics=metrics_df,
        args=args,
        aa_feature_count=len(aa_spec.feature_columns),
        spliceai_feature_count=len(spliceai_spec.feature_columns),
        forbidden_used=forbidden_used,
    )

    metadata = {
        "base_input": str(args.base_input),
        "aa_input": str(args.aa_input),
        "query_vcf": str(args.query_vcf),
        "match_input": str(args.match_input),
        "temporary_feature_table": str(temporary_path),
        "sample_size": 5_000,
        "seed": args.seed,
        "test_size": args.test_size,
        "test_variant_ids_sha256": hashlib.sha256(
            "\n".join(test_ids).encode("utf-8")
        ).hexdigest(),
        "aa_feature_count": len(aa_spec.feature_columns),
        "spliceai_feature_count": len(spliceai_spec.feature_columns),
        "spliceai_feature_columns": AGGREGATED_SPLICEAI_COLUMNS,
        "removed_prior_predictor_columns": removed_predictors,
        "forbidden_features_used": forbidden_used,
        "audit": audit,
        "vcf_metadata": vcf_metadata,
        "full_490k_trained": False,
        "ten_or_twenty_k_trained": False,
        "main_feature_table_modified": False,
    }
    (args.output_dir / "spliceai_5k_low_circularity_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and aggregate the completed SpliceAI 5k match, create a temporary "
            "5k table, and compare AA/BLOSUM with AA/BLOSUM+SpliceAI LightGBM."
        )
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument("--aa-input", type=Path, default=DEFAULT_AA_INPUT)
    parser.add_argument("--query-vcf", type=Path, default=DEFAULT_QUERY_VCF)
    parser.add_argument("--match-input", type=Path, default=DEFAULT_MATCH_INPUT)
    parser.add_argument(
        "--spliceai-snv-vcf",
        type=Path,
        default=ROOT
        / "data"
        / "external"
        / "spliceai"
        / "v1.3"
        / "spliceai_scores.masked.snv.hg38.vcf.gz",
    )
    parser.add_argument(
        "--spliceai-indel-vcf",
        type=Path,
        default=ROOT
        / "data"
        / "external"
        / "spliceai"
        / "v1.3"
        / "spliceai_scores.masked.indel.hg38.vcf.gz",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
