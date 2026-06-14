from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.utils.io import ensure_dir
from magpie_x.utils.schema import canonical_variant_id

DEFAULT_INPUT = ROOT / "outputs" / "real" / "swissprot" / "humsavar_parsed.csv"
DEFAULT_VARIANT_SUMMARY = ROOT / "data" / "raw" / "clinvar" / "latest" / "variant_summary.txt.gz"
DEFAULT_CLINVAR_FEATURE_TABLE = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "swissprot"


def normalize_rsid(value: object) -> str:
    text = str(value or "").strip()
    if not text or text == "-":
        return ""
    return text[2:] if text.lower().startswith("rs") else text


def clean_value(value: object) -> str:
    text = str(value or "").strip()
    return "" if text in {"", "-", ".", "nan", "None"} else text


def collect_clinvar_feature_genes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    genes: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["gene"], chunksize=250_000, low_memory=False):
        genes.update(g.upper() for g in chunk["gene"].dropna().astype(str) if g and g != "UNKNOWN")
    return genes


def iter_variant_summary_matches(path: Path, rsids: set[str], chunk_size: int) -> Iterable[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"ClinVar variant_summary not found: {path}")

    wanted_cols = [
        "#AlleleID",
        "Type",
        "Name",
        "GeneSymbol",
        "ClinicalSignificance",
        "RS# (dbSNP)",
        "Assembly",
        "Chromosome",
        "Start",
        "Stop",
        "ReferenceAllele",
        "AlternateAllele",
        "VariationID",
        "PositionVCF",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
    ]
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        batch: list[dict[str, str]] = []
        for row in reader:
            rsid = normalize_rsid(row.get("RS# (dbSNP)", ""))
            if not rsid or rsid not in rsids:
                continue
            if row.get("Assembly") != "GRCh38":
                continue
            batch.append({col: row.get(col, "") for col in wanted_cols})
            if len(batch) >= chunk_size:
                for item in batch:
                    yield item
                batch = []
        for item in batch:
            yield item


def variant_tuple_from_summary(row: dict[str, str]) -> tuple[str, str, str, str] | None:
    chrom = clean_value(row.get("Chromosome"))
    pos = clean_value(row.get("PositionVCF")) or clean_value(row.get("Start"))
    ref = clean_value(row.get("ReferenceAlleleVCF")) or clean_value(row.get("ReferenceAllele"))
    alt = clean_value(row.get("AlternateAlleleVCF")) or clean_value(row.get("AlternateAllele"))
    if not all([chrom, pos, ref, alt]):
        return None
    if chrom in {"Un", "na", "Not Available"}:
        return None
    return chrom, pos, ref, alt


def build_rsid_mapping(
    variant_summary: Path,
    rsids: set[str],
    *,
    chunk_size: int,
) -> tuple[dict[str, list[dict[str, object]]], Counter[str]]:
    grouped: dict[str, dict[tuple[str, str, str, str], dict[str, object]]] = defaultdict(dict)
    summary_counts: Counter[str] = Counter()
    for row in iter_variant_summary_matches(variant_summary, rsids, chunk_size):
        summary_counts["grch38_rsid_rows_seen"] += 1
        rsid = normalize_rsid(row.get("RS# (dbSNP)", ""))
        variant_tuple = variant_tuple_from_summary(row)
        if variant_tuple is None:
            summary_counts["rows_with_incomplete_coordinates"] += 1
            continue
        chrom, pos, ref, alt = variant_tuple
        key = (chrom, pos, ref, alt)
        if key not in grouped[rsid]:
            grouped[rsid][key] = {
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "clinvar_gene_symbols": set(),
                "clinvar_clinical_significance": set(),
                "clinvar_variation_ids": set(),
                "clinvar_variant_names": set(),
                "clinvar_types": set(),
            }
        grouped[rsid][key]["clinvar_gene_symbols"].add(clean_value(row.get("GeneSymbol")))
        grouped[rsid][key]["clinvar_clinical_significance"].add(clean_value(row.get("ClinicalSignificance")))
        grouped[rsid][key]["clinvar_variation_ids"].add(clean_value(row.get("VariationID")))
        grouped[rsid][key]["clinvar_variant_names"].add(clean_value(row.get("Name")))
        grouped[rsid][key]["clinvar_types"].add(clean_value(row.get("Type")))

    mapping: dict[str, list[dict[str, object]]] = {}
    for rsid, tuple_map in grouped.items():
        mapping[rsid] = []
        for item in tuple_map.values():
            converted = dict(item)
            for field in (
                "clinvar_gene_symbols",
                "clinvar_clinical_significance",
                "clinvar_variation_ids",
                "clinvar_variant_names",
                "clinvar_types",
            ):
                converted[field] = "|".join(sorted(x for x in converted[field] if x))
            mapping[rsid].append(converted)
    return mapping, summary_counts


def map_humsavar_to_variants(
    input_csv: Path,
    variant_summary: Path,
    output_dir: Path,
    *,
    clinvar_feature_table: Path | None,
    chunk_size: int,
) -> dict[str, object]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Parsed HUMSAVAR CSV not found: {input_csv}")
    ensure_dir(output_dir)

    humsavar = pd.read_csv(input_csv, dtype=str).fillna("")
    if "label" in humsavar.columns:
        humsavar["label"] = pd.to_numeric(humsavar["label"], errors="coerce").astype("Int64")
    humsavar["dbsnp_rsid_numeric"] = humsavar["dbsnp_rsid"].map(normalize_rsid)
    rsids = {rsid for rsid in humsavar["dbsnp_rsid_numeric"].astype(str) if rsid}

    clinvar_genes = collect_clinvar_feature_genes(clinvar_feature_table) if clinvar_feature_table else set()
    rsid_mapping, summary_counts = build_rsid_mapping(variant_summary, rsids, chunk_size=chunk_size)

    mapped_rows: list[dict[str, object]] = []
    unmapped_rows: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    for _, row in humsavar.iterrows():
        rsid = normalize_rsid(row.get("dbsnp_rsid", ""))
        base = row.to_dict()
        if not rsid:
            base["mapping_status"] = "unmapped"
            base["unmapped_reason"] = "missing_rsid"
            unmapped_rows.append(base)
            reason_counts["missing_rsid"] += 1
            continue
        candidates = rsid_mapping.get(rsid, [])
        if not candidates:
            base["mapping_status"] = "unmapped"
            base["unmapped_reason"] = "no_grch38_variant_summary_match_for_rsid"
            unmapped_rows.append(base)
            reason_counts["no_grch38_variant_summary_match_for_rsid"] += 1
            continue
        unique_tuples = {(c["chrom"], c["pos"], c["ref"], c["alt"]) for c in candidates}
        if len(unique_tuples) != 1:
            base["mapping_status"] = "unmapped"
            base["unmapped_reason"] = "ambiguous_multiple_grch38_variant_tuples_for_rsid"
            base["candidate_variant_tuples"] = ";".join(
                f"{chrom}:{pos}:{ref}>{alt}" for chrom, pos, ref, alt in sorted(unique_tuples)
            )
            unmapped_rows.append(base)
            reason_counts["ambiguous_multiple_grch38_variant_tuples_for_rsid"] += 1
            continue

        candidate = candidates[0]
        chrom = str(candidate["chrom"])
        pos = str(candidate["pos"])
        ref = str(candidate["ref"])
        alt = str(candidate["alt"])
        gene = str(row.get("gene", ""))
        gene_seen = gene.upper() in clinvar_genes if gene else False
        mapped = {
            **base,
            "chrom": chrom,
            "pos": int(pos) if pos.isdigit() else pos,
            "ref": ref,
            "alt": alt,
            "variant_id": canonical_variant_id(chrom, pos, ref, alt),
            "consequence": "missense_variant",
            "assembly": "GRCh38",
            "mapping_status": "mapped",
            "mapping_source": "ClinVar variant_summary rsID GRCh38 unique tuple",
            "mapping_confidence": "high_rsid_unique_grch38_tuple",
            "gene_seen_in_clinvar_feature_table": bool(gene_seen),
            **candidate,
        }
        mapped_rows.append(mapped)

    mapped = pd.DataFrame(mapped_rows)
    unmapped = pd.DataFrame(unmapped_rows)
    mapped_path = output_dir / "humsavar_mapped_variants.csv"
    unmapped_path = output_dir / "humsavar_unmapped_variants.csv"
    mapped.to_csv(mapped_path, index=False)
    unmapped.to_csv(unmapped_path, index=False)

    label_counts = mapped["label"].value_counts().sort_index().to_dict() if not mapped.empty else {}
    gene_seen_count = int(mapped["gene_seen_in_clinvar_feature_table"].sum()) if not mapped.empty else 0
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "variant_summary": str(variant_summary),
        "mapped_csv": str(mapped_path),
        "unmapped_csv": str(unmapped_path),
        "total_humsavar_supervised_rows": int(len(humsavar)),
        "rows_with_rsid": int(humsavar["dbsnp_rsid_numeric"].astype(str).ne("").sum()),
        "rows_without_rsid": int(humsavar["dbsnp_rsid_numeric"].astype(str).eq("").sum()),
        "mapped_rows": int(len(mapped)),
        "unmapped_rows": int(len(unmapped)),
        "unmapped_reason_counts": dict(reason_counts),
        "mapped_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "clinvar_variant_summary_counts": dict(summary_counts),
        "clinvar_feature_table_gene_reference": str(clinvar_feature_table) if clinvar_feature_table else None,
        "mapped_rows_in_genes_seen_in_clinvar_feature_table": gene_seen_count,
        "mapped_rows_after_excluding_genes_seen_in_clinvar_feature_table": int(len(mapped) - gene_seen_count),
        "mapping_rule": "Map only by dbSNP rsID to GRCh38 ClinVar variant_summary rows; keep only rsIDs with one unique chrom,pos,ref,alt tuple.",
        "synthetic_or_toy_data_used": False,
    }
    manifest_path = output_dir / "humsavar_mapping_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_mapping_report(output_dir / "SWISSPROT_MAPPING_REPORT.md", manifest)
    return manifest


def write_mapping_report(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# SwissProt/HUMSAVAR Mapping Report",
        "",
        f"Generated: {manifest['generated']}",
        "",
        "Status: parsing and mapping audit only; no model training was run.",
        "",
        f"HUMSAVAR parsed input: `{manifest['input_csv']}`",
        f"ClinVar variant summary used for rsID mapping: `{manifest['variant_summary']}`",
        "",
        "## Mapping Rule",
        "",
        str(manifest["mapping_rule"]),
        "",
        "If a HUMSAVAR variant could not be mapped confidently to `chrom,pos,ref,alt`, it was retained in the unmapped file. No coordinate guessing was performed.",
        "",
        "## Counts",
        "",
        "| Item | Count |",
        "| --- | ---: |",
        f"| Supervised HUMSAVAR rows | {manifest['total_humsavar_supervised_rows']} |",
        f"| Rows with rsID | {manifest['rows_with_rsid']} |",
        f"| Rows without rsID | {manifest['rows_without_rsid']} |",
        f"| Mapped rows | {manifest['mapped_rows']} |",
        f"| Unmapped rows | {manifest['unmapped_rows']} |",
        f"| Mapped rows in genes seen in ClinVar feature table | {manifest['mapped_rows_in_genes_seen_in_clinvar_feature_table']} |",
        f"| Mapped rows after excluding genes seen in ClinVar feature table | {manifest['mapped_rows_after_excluding_genes_seen_in_clinvar_feature_table']} |",
        "",
        "## Mapped Label Counts",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for label, count in sorted(dict(manifest["mapped_label_counts"]).items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Unmapped Reasons", "", "| Reason | Count |", "| --- | ---: |"])
    for reason, count in sorted(dict(manifest["unmapped_reason_counts"]).items()):
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{manifest['mapped_csv']}`",
            f"- `{manifest['unmapped_csv']}`",
            "",
            "## Next Step",
            "",
            "Build an ANNOVAR avinput from mapped variants and annotate only after confirming the mapping audit. SwissProt/HUMSAVAR external validation should stay separate from random ClinVar split results.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Map parsed HUMSAVAR variants to genomic coordinates by rsID.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--variant-summary", type=Path, default=DEFAULT_VARIANT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clinvar-feature-table", type=Path, default=DEFAULT_CLINVAR_FEATURE_TABLE)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args(argv)
    manifest = map_humsavar_to_variants(
        args.input,
        args.variant_summary,
        args.output_dir,
        clinvar_feature_table=args.clinvar_feature_table,
        chunk_size=args.chunk_size,
    )
    print(f"Mapped HUMSAVAR rows: {manifest['mapped_rows']:,}")
    print(f"Unmapped HUMSAVAR rows: {manifest['unmapped_rows']:,}")
    print(f"Wrote {manifest['mapped_csv']}")


if __name__ == "__main__":
    main()
