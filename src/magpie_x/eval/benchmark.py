"""External benchmark loaders for SwissProt/ACMG-like and MAVE inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from magpie_x.utils.io import read_table


def load_external_benchmark(path: str | Path) -> pd.DataFrame:
    """Load an external benchmark with variant_id and label columns.

    TODO: Each benchmark has licensing and ascertainment caveats. Keep external
    gold sets out of training and document release dates in experiment metadata.
    """

    df = read_table(path)
    if "label" not in df.columns:
        candidates = [c for c in df.columns if c.lower() in {"pathogenic", "class", "target", "y"}]
        if not candidates:
            raise ValueError("Benchmark must contain a label column or a recognizable target column.")
        df = df.rename(columns={candidates[0]: "label"})
    if "variant_id" not in df.columns:
        raise ValueError("Benchmark must contain variant_id for joining predictions/features.")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    return df.dropna(subset=["label"]).copy()


def load_proteingym_mave(path: str | Path) -> pd.DataFrame:
    """Load a MAVE/ProteinGym-style missense score table if provided."""

    df = read_table(path)
    rename = {
        "mutant": "protein_variant",
        "DMS_score": "mave_score",
        "score": "mave_score",
        "target": "label",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "mave_score" not in df.columns:
        raise ValueError("MAVE table must contain DMS_score, score, or mave_score.")
    return df

