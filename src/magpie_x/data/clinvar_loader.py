"""ClinVar-style loading and deterministic toy data generation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from magpie_x.utils.io import open_text, read_table, write_dataframe
from magpie_x.utils.schema import CANONICAL_VARIANT_COLUMNS, canonical_variant_id

PATHOGENIC_TERMS = {"pathogenic", "likely_pathogenic", "pathogenic/likely_pathogenic"}
BENIGN_TERMS = {"benign", "likely_benign", "benign/likely_benign"}


def normalize_clinsig(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace(" ", "_").replace("-", "_")
    text = text.replace("/", "/")
    if "pathogenic" in text and "benign" not in text:
        return "pathogenic" if "likely" not in text else "likely_pathogenic"
    if "benign" in text and "pathogenic" not in text:
        return "benign" if "likely" not in text else "likely_benign"
    return text or "unknown"


def label_from_clinsig(value: object) -> int | None:
    term = normalize_clinsig(value)
    if term in PATHOGENIC_TERMS or term in {"pathogenic", "likely_pathogenic"}:
        return 1
    if term in BENIGN_TERMS or term in {"benign", "likely_benign"}:
        return 0
    return None


def _parse_info_field(info: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in str(info).split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            parsed[key] = value
    return parsed


def load_clinvar(path: str | Path) -> pd.DataFrame:
    """Load a small ClinVar VCF or a normalized CSV/TSV table.

    Real projects should pin a ClinVar release date and keep the raw file
    under data/raw. The parser intentionally extracts only stable fields used
    by MAGPIE_X; additional INFO tags can be joined later as annotations.
    """

    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz")):
        df = read_table(path)
        return normalize_clinvar_table(df)

    rows: list[dict[str, object]] = []
    with open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            chrom, pos, _id, ref, alt, _qual, _filter, info = line.rstrip("\n").split("\t")[:8]
            info_map = _parse_info_field(info)
            clinsig = info_map.get("CLNSIG", "unknown")
            geneinfo = info_map.get("GENEINFO", "")
            gene = geneinfo.split(":")[0] if geneinfo else "UNKNOWN"
            condition = info_map.get("CLNDN", "not_provided").replace("_", " ")
            label = label_from_clinsig(clinsig)
            if label is None:
                continue
            rows.append(
                {
                    "variant_id": canonical_variant_id(chrom, pos, ref, alt),
                    "chrom": chrom,
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "gene": gene,
                    "consequence": info_map.get("MC", "missense").split("|")[-1].lower(),
                    "clinical_significance": normalize_clinsig(clinsig),
                    "label": label,
                    "submission_date": info_map.get("CLNREVSTAT", "2000-01-01"),
                    "condition_name": condition,
                    "review_status": info_map.get("CLNREVSTAT", "unknown"),
                }
            )
    return normalize_clinvar_table(pd.DataFrame(rows))


def normalize_clinvar_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize user-provided ClinVar-like tables to the MAGPIE_X schema."""

    df = df.copy()
    rename = {
        "#CHROM": "chrom",
        "CHROM": "chrom",
        "Chromosome": "chrom",
        "Start": "pos",
        "PositionVCF": "pos",
        "ReferenceAlleleVCF": "ref",
        "AlternateAlleleVCF": "alt",
        "GeneSymbol": "gene",
        "Gene.refGene": "gene",
        "ClinicalSignificance": "clinical_significance",
        "CLNSIG": "clinical_significance",
        "Condition": "condition_name",
        "CLNDN": "condition_name",
        "DateLastEvaluated": "submission_date",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "variant_id" not in df.columns:
        df["variant_id"] = [
            canonical_variant_id(c, p, r, a)
            for c, p, r, a in zip(df["chrom"], df["pos"], df["ref"], df["alt"], strict=False)
        ]
    if "label" not in df.columns and "clinical_significance" in df.columns:
        df["label"] = df["clinical_significance"].map(label_from_clinsig)
    if "consequence" not in df.columns:
        df["consequence"] = "missense"
    if "condition_name" not in df.columns:
        df["condition_name"] = "not provided"
    if "submission_date" not in df.columns:
        df["submission_date"] = "2000-01-01"
    if "clinical_significance" not in df.columns:
        df["clinical_significance"] = "unknown"
    df["clinical_significance"] = df["clinical_significance"].map(normalize_clinsig)
    df["submission_date"] = pd.to_datetime(df["submission_date"], errors="coerce").fillna(pd.Timestamp("2000-01-01"))
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    for col in CANONICAL_VARIANT_COLUMNS:
        if col not in df.columns:
            df[col] = "unknown"
    return df.sort_values(["submission_date", "variant_id"]).reset_index(drop=True)


def generate_toy_clinvar(
    n_variants: int = 96,
    n_genes: int = 18,
    pathogenic_fraction: float = 0.48,
    seed: int = 13,
) -> pd.DataFrame:
    """Create deterministic toy ClinVar-like exonic variants.

    The toy labels are generated from a hidden risk score so tests and dry runs
    exercise realistic signal without shipping restricted clinical data.
    """

    rng = np.random.default_rng(seed)
    genes = [f"GENE{i:03d}" for i in range(1, n_genes + 1)]
    consequences = np.array(["missense", "stopgain", "stoploss", "startloss", "frameshift", "nonframeshift"])
    condition_templates = [
        "developmental delay and seizures",
        "hypertrophic cardiomyopathy",
        "hereditary cancer syndrome",
        "inborn error of metabolism",
        "primary immunodeficiency",
        "skeletal dysplasia",
        "renal tubular disorder",
        "retinal dystrophy",
        "not provided",
    ]
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))

    rows: list[dict[str, object]] = []
    gene_risk = {gene: rng.normal(0, 0.9) for gene in genes}
    for i in range(n_variants):
        chrom = str(rng.choice(list(range(1, 23)) + ["X"]))
        pos = int(rng.integers(100_000, 240_000_000))
        ref = rng.choice(["A", "C", "G", "T"])
        alt = rng.choice([b for b in ["A", "C", "G", "T"] if b != ref])
        gene = str(rng.choice(genes))
        consequence = str(rng.choice(consequences, p=[0.62, 0.1, 0.04, 0.04, 0.12, 0.08]))
        missense_bonus = 0.2 if consequence == "missense" else 0.0
        lof_bonus = 1.5 if consequence in {"stopgain", "frameshift", "startloss"} else 0.0
        hidden_score = gene_risk[gene] + lof_bonus + missense_bonus + rng.normal(0, 0.9)
        threshold = float(np.quantile([rng.normal() for _ in range(2000)], 1 - pathogenic_fraction))
        label = int(hidden_score > threshold)
        clinsig = "pathogenic" if label else "benign"
        rows.append(
            {
                "variant_id": canonical_variant_id(chrom, pos, ref, alt),
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "gene": gene,
                "transcript": f"NM_{rng.integers(100000, 999999)}",
                "aa_ref": str(rng.choice(aa)),
                "aa_alt": str(rng.choice(aa)),
                "protein_pos": int(rng.integers(1, 2200)),
                "consequence": consequence,
                "clinical_significance": clinsig,
                "label": label,
                "submission_date": pd.Timestamp("2012-01-01") + pd.to_timedelta(int(i * 45 + rng.integers(0, 30)), unit="D"),
                "condition_name": condition_templates[i % len(condition_templates)],
                "review_status": str(rng.choice(["criteria provided", "reviewed by expert panel", "single submitter"])),
            }
        )
    return normalize_clinvar_table(pd.DataFrame(rows))


def save_clinvar(df: pd.DataFrame, path: str | Path) -> None:
    out = df.copy()
    if "submission_date" in out.columns:
        out["submission_date"] = pd.to_datetime(out["submission_date"]).dt.strftime("%Y-%m-%d")
    write_dataframe(out, path)
