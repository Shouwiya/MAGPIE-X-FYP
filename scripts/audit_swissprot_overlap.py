from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.utils.io import ensure_dir

DEFAULT_MAPPED = ROOT / "outputs" / "real" / "swissprot" / "humsavar_mapped_variants.csv"
DEFAULT_FEATURE_TABLE = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "swissprot"


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


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "label" not in df.columns:
        return {}
    labels = pd.to_numeric(df["label"], errors="coerce")
    return {str(int(k)): int(v) for k, v in labels.value_counts(dropna=False).sort_index().items() if pd.notna(k)}


def load_feature_overlap_reference(path: Path) -> tuple[set[str], set[str], dict[str, object]]:
    usecols = ["chrom", "pos", "ref", "alt", "gene", "label"]
    feature = pd.read_csv(path, usecols=usecols, dtype={"chrom": str}, low_memory=False)
    feature["_variant_key"] = variant_key_frame(feature)
    exact_keys = set(feature["_variant_key"].dropna().astype(str))
    genes = {gene.upper() for gene in feature["gene"].dropna().astype(str) if gene and gene != "UNKNOWN"}
    stats = {
        "feature_table_rows": int(len(feature)),
        "feature_table_unique_exact_variant_keys": int(len(exact_keys)),
        "feature_table_duplicate_exact_variant_keys": int(feature["_variant_key"].duplicated().sum()),
        "feature_table_gene_count": int(len(genes)),
        "feature_table_label_counts": label_counts(feature),
    }
    return exact_keys, genes, stats


def audit_overlap(mapped_csv: Path, feature_table: Path, output_dir: Path) -> dict[str, object]:
    if not mapped_csv.exists():
        raise FileNotFoundError(f"Mapped HUMSAVAR file not found: {mapped_csv}")
    if not feature_table.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table}")
    ensure_dir(output_dir)

    mapped = pd.read_csv(mapped_csv, dtype={"chrom": str}, low_memory=False)
    mapped["_variant_key"] = variant_key_frame(mapped)
    mapped["_gene_key"] = mapped["gene"].fillna("").astype(str).str.upper()

    exact_keys, feature_genes, feature_stats = load_feature_overlap_reference(feature_table)
    mapped["exact_clinvar_feature_overlap"] = mapped["_variant_key"].isin(exact_keys)
    mapped["gene_seen_in_clinvar_feature_table_audit"] = mapped["_gene_key"].isin(feature_genes)

    no_exact = mapped.loc[~mapped["exact_clinvar_feature_overlap"]].copy()
    no_gene = mapped.loc[~mapped["gene_seen_in_clinvar_feature_table_audit"]].copy()
    no_exact_no_gene = mapped.loc[
        ~mapped["exact_clinvar_feature_overlap"] & ~mapped["gene_seen_in_clinvar_feature_table_audit"]
    ].copy()

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "mapped_csv": str(mapped_csv),
        "feature_table": str(feature_table),
        "mapped_rows_before_filtering": int(len(mapped)),
        "mapped_unique_variant_keys": int(mapped["_variant_key"].nunique(dropna=True)),
        "mapped_duplicate_variant_keys": int(mapped["_variant_key"].duplicated().sum()),
        "exact_overlap_rows": int(mapped["exact_clinvar_feature_overlap"].sum()),
        "rows_after_exact_overlap_removal": int(len(no_exact)),
        "gene_overlap_rows": int(mapped["gene_seen_in_clinvar_feature_table_audit"].sum()),
        "rows_after_gene_overlap_removal": int(len(no_gene)),
        "rows_after_exact_and_gene_overlap_removal": int(len(no_exact_no_gene)),
        "label_distribution_before_filtering": label_counts(mapped),
        "label_distribution_after_exact_overlap_removal": label_counts(no_exact),
        "label_distribution_after_gene_overlap_removal": label_counts(no_gene),
        "label_distribution_after_exact_and_gene_overlap_removal": label_counts(no_exact_no_gene),
        "feature_reference": feature_stats,
        "note": "Overlap is audited against the current ClinVar-derived feature table, not the original MAGPIE ClinVarTrain split.",
        "synthetic_or_toy_data_used": False,
    }

    annotated_overlap = output_dir / "humsavar_mapped_overlap_flags.csv"
    mapped.drop(columns=["_gene_key"]).to_csv(annotated_overlap, index=False)
    summary["overlap_flagged_csv"] = str(annotated_overlap)

    manifest_path = output_dir / "swissprot_overlap_audit_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(output_dir / "SWISSPROT_OVERLAP_AUDIT.md", summary)
    return summary


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# SwissProt/HUMSAVAR Overlap Audit",
        "",
        f"Generated: {summary['generated']}",
        "",
        "Status: overlap audit only; no model training was run.",
        "",
        f"Mapped HUMSAVAR variants: `{summary['mapped_csv']}`",
        f"ClinVar-derived feature table: `{summary['feature_table']}`",
        "",
        "## Exact Variant Overlap",
        "",
        "| Item | Count | Label distribution |",
        "| --- | ---: | --- |",
        f"| Before filtering | {summary['mapped_rows_before_filtering']} | {format_counts(summary['label_distribution_before_filtering'])} |",
        f"| Exact overlap rows | {summary['exact_overlap_rows']} | - |",
        f"| After exact-overlap removal | {summary['rows_after_exact_overlap_removal']} | {format_counts(summary['label_distribution_after_exact_overlap_removal'])} |",
        "",
        "## Gene Overlap",
        "",
        "| Item | Count | Label distribution |",
        "| --- | ---: | --- |",
        f"| Gene overlap rows | {summary['gene_overlap_rows']} | - |",
        f"| After gene-overlap removal | {summary['rows_after_gene_overlap_removal']} | {format_counts(summary['label_distribution_after_gene_overlap_removal'])} |",
        f"| After exact + gene-overlap removal | {summary['rows_after_exact_and_gene_overlap_removal']} | {format_counts(summary['label_distribution_after_exact_and_gene_overlap_removal'])} |",
        "",
        "## Feature-Table Reference",
        "",
        "| Item | Count |",
        "| --- | ---: |",
        f"| Feature table rows | {summary['feature_reference']['feature_table_rows']} |",
        f"| Unique exact variant keys | {summary['feature_reference']['feature_table_unique_exact_variant_keys']} |",
        f"| Duplicate exact variant keys | {summary['feature_reference']['feature_table_duplicate_exact_variant_keys']} |",
        f"| Gene count | {summary['feature_reference']['feature_table_gene_count']} |",
        "",
        "## Interpretation",
        "",
        "This audit uses the current ClinVar-derived feature table as the overlap reference. It is stricter than pretending SwissProt is independent, but it is not identical to MAGPIE's original ClinVarTrain split because that exact split is not present here.",
        "",
        "If features are reused by exact variant join, rows in the reuse feature table are exact overlaps by construction. A true no-exact-overlap SwissProt evaluation requires fresh real annotation for mapped variants that are not present in the ClinVar-derived feature table.",
        "",
        f"Overlap-flagged output: `{summary['overlap_flagged_csv']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit exact/gene overlap between mapped HUMSAVAR and ClinVar feature table.")
    parser.add_argument("--mapped", type=Path, default=DEFAULT_MAPPED)
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    summary = audit_overlap(args.mapped, args.feature_table, args.output_dir)
    print(f"Exact overlap rows: {summary['exact_overlap_rows']:,}")
    print(f"Rows after exact-overlap removal: {summary['rows_after_exact_overlap_removal']:,}")
    print(f"Rows after gene-overlap removal: {summary['rows_after_gene_overlap_removal']:,}")


if __name__ == "__main__":
    main()
