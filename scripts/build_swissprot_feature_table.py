from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.data.real_pipeline import build_basic_feature_table, vcf_alleles_to_annovar
from magpie_x.utils.io import ensure_dir
from magpie_x.utils.schema import canonical_variant_id

DEFAULT_MAPPED = ROOT / "outputs" / "real" / "swissprot" / "humsavar_mapped_variants.csv"
DEFAULT_CLINVAR_FEATURE_TABLE = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
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
    return {str(int(k)): int(v) for k, v in labels.value_counts().sort_index().items()}


def write_avinput(mapped: pd.DataFrame, path: Path) -> dict[str, int]:
    conversion_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for _, row in mapped.iterrows():
            start, end, av_ref, av_alt, conversion = vcf_alleles_to_annovar(
                int(row["pos"]),
                str(row["ref"]),
                str(row["alt"]),
            )
            conversion_counts[conversion] += 1
            writer.writerow(
                [
                    row["chrom"],
                    start,
                    end,
                    av_ref,
                    av_alt,
                    row.get("variant_id", canonical_variant_id(row["chrom"], row["pos"], row["ref"], row["alt"])),
                    row.get("gene", "UNKNOWN"),
                    row.get("aa_change", ""),
                    row.get("label", ""),
                ]
            )
    return dict(conversion_counts)


def build_annovar_command(avinput: Path, out_prefix: Path, annovar_dir: str, db_dir: str) -> str:
    return " ".join(
        [
            "python",
            "scripts/run_annovar_wrapper.py",
            "--annovar-dir",
            annovar_dir,
            "--db-dir",
            db_dir,
            "--input-avinput",
            str(avinput),
            "--out-prefix",
            str(out_prefix),
            "--buildver",
            "hg38",
            "--protocol",
            "refGene,gnomad30_genome,dbnsfp42a",
            "--operation",
            "g,f,f",
            "--dry-run",
        ]
    )


def prepare_annotation_inputs(
    mapped: pd.DataFrame,
    output_dir: Path,
    *,
    annovar_dir: str,
    db_dir: str,
) -> dict[str, object]:
    avinput = output_dir / "swissprot_humsavar.avinput"
    out_prefix = output_dir / "swissprot_humsavar"
    command_file = output_dir / "SWISSPROT_ANNOVAR_COMMANDS.txt"
    conversion_counts = write_avinput(mapped, avinput)
    command = build_annovar_command(avinput, out_prefix, annovar_dir, db_dir)
    command_file.write_text(command + "\n", encoding="utf-8")
    return {
        "avinput": str(avinput),
        "annovar_command_file": str(command_file),
        "annovar_dry_run_command": command,
        "conversion_counts": conversion_counts,
    }


def build_feature_reuse_table(
    mapped: pd.DataFrame,
    clinvar_feature_table: Path,
    output_csv: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not clinvar_feature_table.exists():
        raise FileNotFoundError(f"ClinVar-derived feature table not found: {clinvar_feature_table}")

    mapped = mapped.copy()
    mapped["_variant_key"] = variant_key_frame(mapped)
    mapped_meta_cols = [
        "humsavar_line",
        "uniprot_accession",
        "gene",
        "ft_id",
        "aa_change",
        "variant_category",
        "classification",
        "dbsnp_rsid",
        "disease_name",
        "label",
        "mapping_confidence",
        "gene_seen_in_clinvar_feature_table",
        "_variant_key",
    ]
    mapped_meta_cols = [col for col in mapped_meta_cols if col in mapped.columns]
    mapped_meta = mapped[mapped_meta_cols].rename(
        columns={
            "gene": "swissprot_gene",
            "label": "swissprot_label",
            "classification": "swissprot_classification",
        }
    )

    feature = pd.read_csv(clinvar_feature_table, dtype={"chrom": str}, low_memory=False)
    feature["_variant_key"] = variant_key_frame(feature)
    duplicate_feature_keys = int(feature["_variant_key"].duplicated().sum())
    feature_unique = feature.drop_duplicates("_variant_key", keep="first").copy()

    joined = mapped_meta.merge(feature_unique, on="_variant_key", how="inner", suffixes=("_swissprot", ""))
    joined["clinvar_reuse_label"] = joined["label"]
    joined["label"] = pd.to_numeric(joined["swissprot_label"], errors="coerce").astype(int)
    joined["clin_sig_raw"] = joined["variant_category"]
    joined["origin"] = "SwissProt/HUMSAVAR feature reuse"
    joined["source_file"] = "data/external/humsavar/humsavar.txt"
    joined["exact_clinvar_feature_overlap"] = True
    joined["feature_reuse_source"] = str(clinvar_feature_table)
    joined["feature_reuse_note"] = "Features reused by exact chrom,pos,ref,alt join to current ClinVar-derived feature table."
    joined = joined.drop(columns=["_variant_key"])
    joined.to_csv(output_csv, index=False)

    mapped_keys = set(mapped["_variant_key"].dropna().astype(str))
    joined_keys = set(feature_unique["_variant_key"]).intersection(mapped_keys)
    stats = {
        "clinvar_feature_table": str(clinvar_feature_table),
        "mapped_rows": int(len(mapped)),
        "mapped_unique_variant_keys": int(mapped["_variant_key"].nunique(dropna=True)),
        "clinvar_feature_rows": int(len(feature)),
        "clinvar_feature_unique_variant_keys": int(feature_unique["_variant_key"].nunique(dropna=True)),
        "clinvar_feature_duplicate_variant_keys": duplicate_feature_keys,
        "joined_rows": int(len(joined)),
        "joined_unique_variant_keys": int(len(joined_keys)),
        "mapped_rows_not_joined_by_exact_key": int(len(mapped) - len(joined)),
        "joined_label_counts": label_counts(joined),
        "method": "feature_reuse_exact_chrom_pos_ref_alt_join",
    }
    return joined, stats


def build_from_real_annovar(
    mapped: pd.DataFrame,
    output_dir: Path,
    annovar_multianno: Path,
) -> dict[str, object]:
    variant_table = output_dir / "swissprot_variant_table.csv"
    rows = []
    for _, row in mapped.iterrows():
        rows.append(
            {
                "chrom": row["chrom"],
                "pos": int(row["pos"]),
                "ref": row["ref"],
                "alt": row["alt"],
                "variant_id": row["variant_id"],
                "clin_sig_raw": row["variant_category"],
                "label": int(row["label"]),
                "gene": row.get("gene", "UNKNOWN"),
                "consequence": row.get("consequence", "missense_variant") or "missense_variant",
                "review_status": "SwissProt/HUMSAVAR curated",
                "origin": "SwissProt/HUMSAVAR",
                "assembly": row.get("assembly", "GRCh38"),
                "source_file": row.get("source_file", "data/external/humsavar/humsavar.txt"),
            }
        )
    pd.DataFrame(rows).to_csv(variant_table, index=False)
    feature_table = output_dir / "swissprot_feature_table_from_annovar.csv"
    manifest_path = output_dir / "swissprot_feature_table_from_annovar_manifest.json"
    missing_sources_report = output_dir / "SWISSPROT_FEATURE_MISSING_SOURCES.md"
    manifest = build_basic_feature_table(
        variant_table,
        feature_table,
        manifest_path,
        missing_sources_report,
        annovar_multianno=annovar_multianno,
    )
    return {
        "variant_table": str(variant_table),
        "annovar_feature_table": str(feature_table),
        "annovar_feature_manifest": str(manifest_path),
        "annovar_build_manifest": manifest,
    }


def build_swissprot_features(
    mapped_csv: Path,
    output_dir: Path,
    *,
    clinvar_feature_table: Path,
    annovar_multianno: Path | None,
    annovar_dir: str,
    db_dir: str,
    min_joined_rows: int,
) -> dict[str, object]:
    if not mapped_csv.exists():
        raise FileNotFoundError(f"Mapped SwissProt variant file not found: {mapped_csv}")
    ensure_dir(output_dir)
    mapped = pd.read_csv(mapped_csv, dtype={"chrom": str}, low_memory=False)
    if mapped.empty:
        raise ValueError("No mapped SwissProt/HUMSAVAR variants are available for feature building.")

    annotation_inputs = prepare_annotation_inputs(mapped, output_dir, annovar_dir=annovar_dir, db_dir=db_dir)
    feature_table = output_dir / "swissprot_feature_table.csv"
    joined, reuse_stats = build_feature_reuse_table(mapped, clinvar_feature_table, feature_table)

    annovar_stats = None
    if annovar_multianno:
        annovar_stats = build_from_real_annovar(mapped, output_dir, annovar_multianno)

    status = "feature_reuse_table_written"
    if len(joined) < min_joined_rows:
        status = "feature_reuse_table_written_but_too_few_rows_for_main_evaluation"

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "mapped_csv": str(mapped_csv),
        "feature_table": str(feature_table),
        "feature_status": status,
        "feature_reuse": reuse_stats,
        "annotation_inputs": annotation_inputs,
        "annovar_multianno": str(annovar_multianno) if annovar_multianno else None,
        "annovar_annotation_build": annovar_stats,
        "synthetic_or_toy_data_used": False,
        "warning": "Feature reuse rows are exact overlaps with the current ClinVar-derived feature table by construction.",
    }
    manifest_path = output_dir / "swissprot_feature_table_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_feature_report(output_dir / "SWISSPROT_FEATURE_TABLE_REPORT.md", manifest)
    return manifest


def write_feature_report(path: Path, manifest: dict[str, object]) -> None:
    reuse = manifest["feature_reuse"]
    lines = [
        "# SwissProt/HUMSAVAR Feature Table Report",
        "",
        f"Generated: {manifest['generated']}",
        "",
        f"Status: `{manifest['feature_status']}`",
        "",
        "## Method",
        "",
        "First attempt: feature reuse by exact `chrom,pos,ref,alt` join from mapped HUMSAVAR variants to `data/interim/features/basic_feature_table.csv`.",
        "",
        "No features were fabricated for unmapped or unjoined variants. The ANNOVAR avinput and command file were prepared for future real annotation, but no annotation job was run automatically.",
        "",
        "## Feature Reuse Counts",
        "",
        "| Item | Count |",
        "| --- | ---: |",
        f"| Mapped HUMSAVAR rows | {reuse['mapped_rows']} |",
        f"| Mapped unique variant keys | {reuse['mapped_unique_variant_keys']} |",
        f"| ClinVar feature table rows | {reuse['clinvar_feature_rows']} |",
        f"| ClinVar feature unique variant keys | {reuse['clinvar_feature_unique_variant_keys']} |",
        f"| ClinVar feature duplicate variant keys | {reuse['clinvar_feature_duplicate_variant_keys']} |",
        f"| Joined feature rows | {reuse['joined_rows']} |",
        f"| Mapped rows not joined by exact key | {reuse['mapped_rows_not_joined_by_exact_key']} |",
        "",
        "## Joined Label Distribution",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for label, count in sorted(reuse["joined_label_counts"].items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Feature table: `{manifest['feature_table']}`",
            f"- ANNOVAR avinput: `{manifest['annotation_inputs']['avinput']}`",
            f"- ANNOVAR dry-run command: `{manifest['annotation_inputs']['annovar_command_file']}`",
            "",
            "## Warning",
            "",
            str(manifest["warning"]),
            "",
            "A true no-exact-overlap SwissProt feature table requires real annotation for mapped variants that did not join to the ClinVar-derived feature table.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build SwissProt/HUMSAVAR feature table by safe feature reuse and prepare annotation commands.")
    parser.add_argument("--mapped", type=Path, default=DEFAULT_MAPPED)
    parser.add_argument("--clinvar-feature-table", type=Path, default=DEFAULT_CLINVAR_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--annovar-multianno", type=Path, default=None)
    parser.add_argument("--annovar-dir", default="third_party/annovar")
    parser.add_argument("--db-dir", default="third_party/annovar/humandb")
    parser.add_argument("--min-joined-rows", type=int, default=1_000)
    args = parser.parse_args(argv)
    manifest = build_swissprot_features(
        args.mapped,
        args.output_dir,
        clinvar_feature_table=args.clinvar_feature_table,
        annovar_multianno=args.annovar_multianno,
        annovar_dir=args.annovar_dir,
        db_dir=args.db_dir,
        min_joined_rows=args.min_joined_rows,
    )
    print(f"Feature table status: {manifest['feature_status']}")
    print(f"Joined feature rows: {manifest['feature_reuse']['joined_rows']:,}")
    print(f"Wrote {manifest['feature_table']}")


if __name__ == "__main__":
    main()
