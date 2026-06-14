"""Real-data pipeline helpers for MAGPIE-X.

These helpers intentionally avoid the synthetic fallback ANNOVAR/dbNSFP files
created by the toy pipeline. They are designed for the Step 1 real annotation
workflow: ClinVar parsing, protein-altering filtering, ANNOVAR input writing,
and feature-table scaffolding from real available sources only.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

import numpy as np
import pandas as pd

from magpie_x.utils.io import ensure_dir, read_table, write_dataframe
from magpie_x.utils.schema import canonical_variant_id

LOGGER = logging.getLogger(__name__)

SUPERVISED_PATHOGENIC_TERMS = {"pathogenic", "likely_pathogenic", "pathogenic/likely_pathogenic"}
SUPERVISED_BENIGN_TERMS = {"benign", "likely_benign", "benign/likely_benign"}

EXCLUDED_LABEL_MARKERS = {
    "conflicting",
    "uncertain",
    "vus",
    "not_provided",
    "no_assertion",
    "drug_response",
    "risk_factor",
    "association",
    "protective",
    "affects",
    "other",
}

PROTEIN_ALTERING_CONSEQUENCES = {
    "missense_variant",
    "missense",
    "nonsynonymous_snv",
    "nonsynonymous snv",
    "stopgain",
    "stop_gained",
    "nonsense",
    "stoploss",
    "stop_lost",
    "startloss",
    "start_lost",
    "initiator_codon_variant",
    "frameshift",
    "frameshift_variant",
    "frameshift_deletion",
    "frameshift_insertion",
    "nonframeshift",
    "nonframeshift_deletion",
    "nonframeshift_insertion",
    "inframe_deletion",
    "inframe_insertion",
    "inframe_indel",
    "splice_donor_variant",
    "splice_acceptor_variant",
}

EXCLUDED_CONSEQUENCE_MARKERS = {
    "synonymous",
    "intron",
    "utr",
    "non_coding",
    "non-coding",
    "upstream",
    "downstream",
    "no_sequence_alteration",
    "intergenic",
}

CANONICAL_CLINVAR_COLUMNS = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "variant_id",
    "clin_sig_raw",
    "label",
    "gene",
    "consequence",
    "review_status",
    "origin",
    "assembly",
    "source_file",
]

FORBIDDEN_FALLBACK_FILENAMES = {
    str(Path("outputs/real/processed/annovar.csv").as_posix()).lower(),
    str(Path("outputs/real/processed/dbnsfp.csv").as_posix()).lower(),
}


def open_maybe_gzip(path: str | Path):
    path = Path(path)
    if "".join(path.suffixes).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def parse_info_field(info: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in str(info).split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            parsed[key] = unquote(value)
        else:
            parsed[item] = "true"
    return parsed


def normalize_token(value: object) -> str:
    text = unquote(str(value or "")).strip().lower()
    text = text.replace(" ", "_").replace("-", "_")
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_clinsig_terms(value: object) -> list[str]:
    text = normalize_token(value)
    if not text:
        return []
    raw_terms = re.split(r"[|,;]+", text)
    terms: list[str] = []
    for term in raw_terms:
        term = term.strip("_")
        if not term:
            continue
        if term in {"likely_pathogenic_low_penetrance", "pathogenic_low_penetrance"}:
            term = "likely_pathogenic" if term.startswith("likely") else "pathogenic"
        terms.append(term)
    if "/" in text and text in {"pathogenic/likely_pathogenic", "benign/likely_benign"}:
        terms.append(text)
    return terms


def label_from_clinsig(value: object) -> tuple[int | None, str]:
    terms = normalize_clinsig_terms(value)
    if not terms:
        return None, "missing"
    joined = "|".join(terms)
    if any(marker in joined for marker in EXCLUDED_LABEL_MARKERS):
        return None, "excluded_uncertain_or_conflicting"
    has_pathogenic = any(term in SUPERVISED_PATHOGENIC_TERMS or term in {"pathogenic", "likely_pathogenic"} for term in terms)
    has_benign = any(term in SUPERVISED_BENIGN_TERMS or term in {"benign", "likely_benign"} for term in terms)
    if has_pathogenic and has_benign:
        return None, "excluded_mixed_benign_pathogenic"
    if has_pathogenic:
        return 1, "pathogenic_or_likely_pathogenic"
    if has_benign:
        return 0, "benign_or_likely_benign"
    return None, "excluded_other_clinsig"


def split_multi_value(value: object) -> list[str]:
    text = unquote(str(value or ""))
    if not text or text == ".":
        return []
    return [part for part in re.split(r"[,;]+", text) if part]


def extract_gene(info_map: dict[str, str]) -> str:
    geneinfo = info_map.get("GENEINFO") or info_map.get("GENE") or ""
    if not geneinfo or geneinfo == ".":
        return "UNKNOWN"
    first = re.split(r"[|,;]+", geneinfo)[0]
    return first.split(":")[0] or "UNKNOWN"


def extract_consequence(info_map: dict[str, str]) -> str:
    candidates: list[str] = []
    for key in ("MC", "Consequence", "CLNVC"):
        for item in split_multi_value(info_map.get(key, "")):
            token = item.split("|")[-1]
            norm = normalize_token(token)
            if norm:
                candidates.append(norm)
    for candidate in candidates:
        if candidate in PROTEIN_ALTERING_CONSEQUENCES:
            return candidate
    return candidates[0] if candidates else "unknown"


def parse_clinvar_vcf(
    input_path: str | Path,
    output_csv: str | Path,
    manifest_path: str | Path,
    *,
    output_parquet: str | Path | None = None,
    assembly: str = "GRCh38",
    max_records: int | None = None,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    """Parse ClinVar VCF into a normalized supervised table.

    Only benign/likely benign and pathogenic/likely pathogenic labels are kept.
    Conflicting, uncertain, missing, and other clinical significance classes are
    excluded and counted in the manifest.
    """

    input_path = Path(input_path)
    output_csv = Path(output_csv)
    manifest_path = Path(manifest_path)
    ensure_dir(output_csv.parent)
    ensure_dir(manifest_path.parent)
    if output_csv.exists():
        output_csv.unlink()

    total_records = 0
    kept_records = 0
    label_counts: Counter[str] = Counter()
    excluded_label_reasons: Counter[str] = Counter()
    consequence_counts: Counter[str] = Counter()
    review_status_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal rows
        if not rows:
            return
        df = pd.DataFrame(rows, columns=CANONICAL_CLINVAR_COLUMNS)
        df.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)
        rows = []

    with open_maybe_gzip(input_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            total_records += 1
            if max_records is not None and total_records > max_records:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                excluded_label_reasons["malformed_vcf_line"] += 1
                continue
            chrom, pos, _record_id, ref, alt_field, _qual, _filter, info = fields[:8]
            info_map = parse_info_field(info)
            clin_sig_raw = info_map.get("CLNSIG", "")
            label, reason = label_from_clinsig(clin_sig_raw)
            if label is None:
                excluded_label_reasons[reason] += 1
                continue
            alts = [alt for alt in alt_field.split(",") if alt]
            if not alts:
                excluded_label_reasons["missing_alt"] += 1
                continue
            gene = extract_gene(info_map)
            consequence = extract_consequence(info_map)
            review_status = info_map.get("CLNREVSTAT", "unknown")
            origin = info_map.get("ORIGIN", "unknown")
            for alt in alts:
                row = {
                    "chrom": chrom,
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "variant_id": canonical_variant_id(chrom, pos, ref, alt),
                    "clin_sig_raw": clin_sig_raw,
                    "label": int(label),
                    "gene": gene,
                    "consequence": consequence,
                    "review_status": review_status,
                    "origin": origin,
                    "assembly": assembly,
                    "source_file": str(input_path),
                }
                rows.append(row)
                kept_records += 1
                label_counts[str(label)] += 1
                consequence_counts[consequence] += 1
                review_status_counts[review_status] += 1
            if len(rows) >= chunk_size:
                flush()
    flush()

    parquet_status = write_parquet_best_effort(output_csv, output_parquet) if output_parquet else {"requested": False}
    manifest = {
        "source_file": str(input_path),
        "output_csv": str(output_csv),
        "output_parquet": str(output_parquet) if output_parquet else None,
        "parquet_status": parquet_status,
        "assembly": assembly,
        "max_records": max_records,
        "total_vcf_records_seen": int(total_records if max_records is None else min(total_records, max_records)),
        "kept_supervised_rows": int(kept_records),
        "label_counts": dict(label_counts),
        "excluded_label_reasons": dict(excluded_label_reasons),
        "consequence_counts": dict(consequence_counts.most_common()),
        "review_status_counts": dict(review_status_counts.most_common(20)),
        "synthetic_fallback_features_used": False,
    }
    write_json(manifest, manifest_path)
    return manifest


def write_parquet_best_effort(csv_path: str | Path, parquet_path: str | Path | None) -> dict[str, Any]:
    if parquet_path is None:
        return {"requested": False}
    parquet_path = Path(parquet_path)
    ensure_dir(parquet_path.parent)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        return {"requested": True, "written": False, "reason": f"pyarrow unavailable: {exc}"}

    writer = None
    try:
        for chunk in pd.read_csv(csv_path, chunksize=100_000, low_memory=False):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema)
            writer.write_table(table)
    except Exception as exc:
        return {"requested": True, "written": False, "reason": str(exc)}
    finally:
        if writer is not None:
            writer.close()
    return {"requested": True, "written": True, "reason": None}


def normalize_consequence_series(df: pd.DataFrame) -> pd.Series:
    for col in ("consequence", "annovar_exonic_func", "ExonicFunc.refGene", "ExonicFunc"):
        if col in df.columns:
            values = df[col].fillna("").map(normalize_token)
            if values.ne("").any():
                return values
    return pd.Series(["unknown"] * len(df), index=df.index)


def filter_protein_altering_clinvar(
    input_csv: str | Path,
    output_csv: str | Path,
    manifest_path: str | Path,
    *,
    output_parquet: str | Path | None = None,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    manifest_path = Path(manifest_path)
    ensure_dir(output_csv.parent)
    if output_csv.exists():
        output_csv.unlink()

    rows_before = 0
    rows_after = 0
    label_before: Counter[str] = Counter()
    label_after: Counter[str] = Counter()
    kept_consequences: Counter[str] = Counter()
    excluded_consequences: Counter[str] = Counter()

    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, dtype={"chrom": str}, low_memory=False):
        rows_before += len(chunk)
        labels = pd.to_numeric(chunk["label"], errors="coerce")
        label_before.update(labels.dropna().astype(int).astype(str).tolist())
        consequence = normalize_consequence_series(chunk)
        keep_mask = consequence.isin(PROTEIN_ALTERING_CONSEQUENCES)
        filtered = chunk.loc[keep_mask].copy()
        filtered["consequence"] = consequence.loc[keep_mask].to_numpy()
        rows_after += len(filtered)
        label_after.update(pd.to_numeric(filtered["label"], errors="coerce").dropna().astype(int).astype(str).tolist())
        kept_consequences.update(filtered["consequence"].astype(str).tolist())
        excluded_consequences.update(consequence.loc[~keep_mask].astype(str).tolist())
        filtered.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)

    parquet_status = write_parquet_best_effort(output_csv, output_parquet) if output_parquet else {"requested": False}
    manifest = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_parquet": str(output_parquet) if output_parquet else None,
        "parquet_status": parquet_status,
        "allowed_consequences": sorted(PROTEIN_ALTERING_CONSEQUENCES),
        "rows_before_filtering": int(rows_before),
        "rows_after_filtering": int(rows_after),
        "label_distribution_before": dict(label_before),
        "label_distribution_after": dict(label_after),
        "kept_consequence_distribution": dict(kept_consequences.most_common()),
        "excluded_consequence_distribution": dict(excluded_consequences.most_common(50)),
        "synthetic_fallback_features_used": False,
    }
    write_json(manifest, manifest_path)
    return manifest


def vcf_alleles_to_annovar(pos: int, ref: str, alt: str) -> tuple[int, int, str, str, str]:
    """Convert VCF-style alleles to ANNOVAR avinput coordinates."""

    ref = str(ref)
    alt = str(alt)
    if len(ref) == 1 and len(alt) == 1:
        return pos, pos, ref, alt, "snv"
    if len(ref) == len(alt):
        return pos, pos + len(ref) - 1, ref, alt, "substitution"
    if len(ref) > len(alt) and ref.startswith(alt):
        deleted = ref[len(alt) :]
        start = pos + len(alt)
        end = pos + len(ref) - 1
        return start, end, deleted, "-", "deletion"
    if len(alt) > len(ref) and alt.startswith(ref):
        inserted = alt[len(ref) :]
        return pos, pos + 1, "-", inserted, "insertion"
    return pos, pos + len(ref) - 1, ref, alt, "complex"


def write_annovar_avinput(
    input_csv: str | Path,
    output_avinput: str | Path,
    manifest_path: str | Path,
    *,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    input_csv = Path(input_csv)
    output_avinput = Path(output_avinput)
    manifest_path = Path(manifest_path)
    ensure_dir(output_avinput.parent)
    if output_avinput.exists():
        output_avinput.unlink()

    rows_in = 0
    rows_written = 0
    conversion_counts: Counter[str] = Counter()
    columns = ["chrom", "start", "end", "ref", "alt", "variant_id", "gene", "consequence", "label"]
    with output_avinput.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for chunk in pd.read_csv(input_csv, chunksize=chunk_size, dtype={"chrom": str}, low_memory=False):
            rows_in += len(chunk)
            for _, row in chunk.iterrows():
                start, end, av_ref, av_alt, conversion = vcf_alleles_to_annovar(int(row["pos"]), row["ref"], row["alt"])
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
                        row.get("consequence", "unknown"),
                        row.get("label", ""),
                    ]
                )
                rows_written += 1

    manifest = {
        "input_csv": str(input_csv),
        "output_avinput": str(output_avinput),
        "rows_in": int(rows_in),
        "rows_written": int(rows_written),
        "columns": columns,
        "coordinate_note": "ANNOVAR avinput columns are chrom, start, end, ref, alt; additional columns preserve variant_id/gene/consequence/label.",
        "conversion_counts": dict(conversion_counts),
        "synthetic_fallback_features_used": False,
    }
    write_json(manifest, manifest_path)
    return manifest


def path_is_forbidden_fallback(path: str | Path) -> bool:
    path = Path(path)
    normalized = path.as_posix().lower()
    if normalized in FORBIDDEN_FALLBACK_FILENAMES:
        return True
    return normalized.endswith("/outputs/real/processed/annovar.csv") or normalized.endswith("/outputs/real/processed/dbnsfp.csv")


def read_real_annovar_multianno(path: str | Path) -> pd.DataFrame:
    if path_is_forbidden_fallback(path):
        raise ValueError(f"Refusing to read known synthetic fallback annotation file: {path}")
    df = read_table(path, na_values=[".", "NA", "nan", ""])
    sample_text = "\n".join(df.head(100).astype(str).to_numpy().ravel().tolist())
    if "NM_TOY" in sample_text:
        raise ValueError(f"Refusing to read annotation file containing NM_TOY fallback markers: {path}")
    rename = {
        "Chr": "chrom",
        "#Chr": "chrom",
        "Start": "pos",
        "End": "end",
        "Ref": "ref",
        "Alt": "alt",
        "Gene.refGene": "gene_annovar",
        "Func.refGene": "func_refgene",
        "ExonicFunc.refGene": "exonic_func_refgene",
        "AAChange.refGene": "aa_change_refgene",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "variant_id" not in df.columns and {"chrom", "pos", "ref", "alt"}.issubset(df.columns):
        df["variant_id"] = [
            canonical_variant_id(c, p, r, a)
            for c, p, r, a in zip(df["chrom"], df["pos"], df["ref"], df["alt"], strict=False)
        ]
    if "variant_id" not in df.columns:
        raise ValueError("ANNOVAR multianno file must contain variant_id or chrom/pos/ref/alt columns for merging.")
    return df


def build_basic_feature_table(
    clinvar_csv: str | Path,
    output_csv: str | Path,
    manifest_path: str | Path,
    missing_sources_report: str | Path,
    *,
    output_parquet: str | Path | None = None,
    annovar_multianno: str | Path | None = None,
) -> dict[str, Any]:
    clinvar_csv = Path(clinvar_csv)
    output_csv = Path(output_csv)
    manifest_path = Path(manifest_path)
    missing_sources_report = Path(missing_sources_report)
    ensure_dir(output_csv.parent)

    clinvar = pd.read_csv(clinvar_csv, dtype={"chrom": str}, low_memory=False)
    feature_table = clinvar.copy()
    used_sources = ["ClinVar"]
    missing_sources: list[str] = []
    warnings: list[str] = []

    if annovar_multianno:
        annovar = read_real_annovar_multianno(annovar_multianno)
        keep_cols = [c for c in annovar.columns if c == "variant_id" or c not in feature_table.columns]
        feature_table = feature_table.merge(annovar[keep_cols], on="variant_id", how="left")
        used_sources.append("real ANNOVAR multianno")
    else:
        missing_sources.append("real ANNOVAR multianno")

    missing_sources.extend(["real dbNSFP", "real SpliceAI", "real OMIM", "real ChromHMM"])

    for col in ("ref", "alt", "consequence", "gene"):
        if col in feature_table.columns:
            feature_table[col] = feature_table[col].fillna("UNKNOWN").astype(str)
    feature_table["ref_length"] = feature_table["ref"].astype(str).str.len()
    feature_table["alt_length"] = feature_table["alt"].astype(str).str.len()
    feature_table["variant_length_change"] = feature_table["alt_length"] - feature_table["ref_length"]
    feature_table["is_snv"] = ((feature_table["ref_length"] == 1) & (feature_table["alt_length"] == 1)).astype(int)
    feature_table["is_insertion"] = (feature_table["variant_length_change"] > 0).astype(int)
    feature_table["is_deletion"] = (feature_table["variant_length_change"] < 0).astype(int)
    consequence_dummies = pd.get_dummies(feature_table["consequence"], prefix="consequence", dtype=int)
    feature_table = pd.concat([feature_table, consequence_dummies], axis=1)

    write_dataframe(feature_table, output_csv)
    parquet_status = write_parquet_best_effort(output_csv, output_parquet) if output_parquet else {"requested": False}

    manifest = {
        "input_clinvar_csv": str(clinvar_csv),
        "output_csv": str(output_csv),
        "output_parquet": str(output_parquet) if output_parquet else None,
        "parquet_status": parquet_status,
        "rows": int(len(feature_table)),
        "columns": feature_table.columns.tolist(),
        "label_counts": {str(k): int(v) for k, v in feature_table["label"].value_counts().sort_index().items()},
        "used_sources": used_sources,
        "missing_sources": missing_sources,
        "synthetic_fallback_features_used": False,
        "known_forbidden_fallback_files_not_used": [
            "outputs/real/processed/annovar.csv",
            "outputs/real/processed/dbnsfp.csv",
        ],
        "warnings": warnings,
    }
    write_json(manifest, manifest_path)
    write_missing_sources_report(missing_sources_report, manifest)
    return manifest


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def write_missing_sources_report(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    lines = [
        "# Missing Sources Report",
        "",
        "This feature table was built without synthetic fallback annotations.",
        "",
        "## Used Sources",
        "",
    ]
    lines.extend(f"- {source}" for source in manifest.get("used_sources", []))
    lines.extend(["", "## Missing Real Sources", ""])
    lines.extend(f"- {source}" for source in manifest.get("missing_sources", []))
    lines.extend(
        [
            "",
            "## Forbidden Files Not Used",
            "",
            "- `outputs/real/processed/annovar.csv`",
            "- `outputs/real/processed/dbnsfp.csv`",
            "",
            "These files are synthetic fallback artifacts from an earlier pipeline and must not be used for MAGPIE-style claims.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

