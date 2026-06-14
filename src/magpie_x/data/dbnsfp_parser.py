"""dbNSFP-like parser and toy score generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from magpie_x.utils.io import read_table
from magpie_x.utils.schema import MISSENSE_EXPERT_COLUMNS, canonical_variant_id

DBNSFP_RENAME = {
    "#chr": "chrom",
    "chr": "chrom",
    "pos(1-based)": "pos",
    "pos": "pos",
    "ref": "ref",
    "alt": "alt",
    "genename": "gene",
    "AlphaMissense_score": "alphamissense_score",
    "ESM1b_score": "esm1b_score",
    "REVEL_score": "revel_score",
    "VARITY_R_score": "varity_score",
    "VARITY_ER_score": "varity_score",
    "MutPred_score": "mutpred_score",
    "CADD_phred": "cadd_phred",
    "SIFT_score": "sift_score",
    "Polyphen2_HDIV_score": "polyphen2_hdiv_score",
    "phyloP100way_vertebrate": "phylop100",
    "phastCons100way_vertebrate": "phastcons100",
    "gnomAD_exomes_AF": "gnomad_af",
}


def parse_dbnsfp(path: str | Path) -> pd.DataFrame:
    """Parse a dbNSFP-like table and normalize score names."""

    df = read_table(path, na_values=[".", "NA", "nan", ""])
    df = df.rename(columns={k: v for k, v in DBNSFP_RENAME.items() if k in df.columns})
    if "variant_id" not in df.columns and {"chrom", "pos", "ref", "alt"}.issubset(df.columns):
        df["variant_id"] = [
            canonical_variant_id(c, p, r, a)
            for c, p, r, a in zip(df["chrom"], df["pos"], df["ref"], df["alt"], strict=False)
        ]
    numeric_cols = [c for c in df.columns if c.endswith("_score") or c in {"cadd_phred", "phylop100", "phastcons100", "gnomad_af"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def generate_toy_dbnsfp(clinvar: pd.DataFrame, seed: int = 13) -> pd.DataFrame:
    """Generate toy dbNSFP scores correlated with pathogenicity."""

    rng = np.random.default_rng(seed + 23)
    rows: list[dict[str, object]] = []
    for _, row in clinvar.iterrows():
        y = int(row["label"])
        risk = np.clip(rng.normal(0.72 if y else 0.28, 0.18), 0, 1)
        is_missense = row["consequence"] == "missense"
        expert_values = {
            col: (np.clip(risk + rng.normal(0, 0.08), 0, 1) if is_missense else np.nan)
            for col in MISSENSE_EXPERT_COLUMNS
        }
        rows.append(
            {
                "variant_id": row["variant_id"],
                "gene": row["gene"],
                "gnomad_af": float(np.clip(rng.beta(0.4 if y else 1.2, 20 if y else 12), 0, 1)),
                "spliceai_delta_score": float(np.clip(rng.normal(0.55 if y and row["consequence"] != "missense" else 0.1, 0.16), 0, 1)),
                "phylop100": float(np.clip(rng.normal(4.0 if y else 1.4, 1.2), -8, 10)),
                "phastcons100": float(np.clip(rng.normal(0.82 if y else 0.35, 0.22), 0, 1)),
                "cadd_phred": float(np.clip(rng.normal(25 if y else 12, 5), 0, 45)),
                "sift_score": float(np.clip(rng.normal(0.08 if y else 0.45, 0.18), 0, 1)),
                "polyphen2_hdiv_score": float(np.clip(rng.normal(0.82 if y else 0.32, 0.22), 0, 1)),
                "loeuf": float(np.clip(rng.normal(0.25 if y else 0.75, 0.18), 0.02, 2.0)),
                "pli": float(np.clip(rng.normal(0.78 if y else 0.25, 0.2), 0, 1)),
                "recombination_rate": float(np.clip(rng.lognormal(-0.2, 0.5), 0, 8)),
                "chromatin_accessibility": float(np.clip(rng.normal(0.58 if y else 0.42, 0.18), 0, 1)),
                "h3k27ac_signal": float(np.clip(rng.normal(0.55 if y else 0.38, 0.2), 0, 1)),
                **expert_values,
            }
        )
    return pd.DataFrame(rows)

