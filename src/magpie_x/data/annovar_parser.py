"""ANNOVAR-style parser and toy annotation builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from magpie_x.utils.io import read_table
from magpie_x.utils.schema import canonical_variant_id


def parse_annovar(path: str | Path) -> pd.DataFrame:
    """Parse a table resembling ANNOVAR multianno output."""

    df = read_table(path)
    rename = {
        "Chr": "chrom",
        "Start": "pos",
        "Ref": "ref",
        "Alt": "alt",
        "Gene.refGene": "gene",
        "ExonicFunc.refGene": "annovar_exonic_func",
        "Func.refGene": "annovar_func",
        "AAChange.refGene": "aa_change",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "variant_id" not in df.columns and {"chrom", "pos", "ref", "alt"}.issubset(df.columns):
        df["variant_id"] = [
            canonical_variant_id(c, p, r, a)
            for c, p, r, a in zip(df["chrom"], df["pos"], df["ref"], df["alt"], strict=False)
        ]
    return df


def generate_toy_annovar(clinvar: pd.DataFrame, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 11)
    rows: list[dict[str, object]] = []
    for _, row in clinvar.iterrows():
        rows.append(
            {
                "variant_id": row["variant_id"],
                "gene": row["gene"],
                "annovar_func": "exonic",
                "annovar_exonic_func": row["consequence"],
                "aa_change": f"{row.get('transcript', 'NM_TOY')}:exon{rng.integers(1, 25)}:"
                f"c.{rng.integers(1, 5000)}{row['ref']}>{row['alt']}:"
                f"p.{row.get('aa_ref', 'A')}{row.get('protein_pos', 1)}{row.get('aa_alt', 'V')}",
                "exon_number": int(rng.integers(1, 25)),
                "cds_position": int(rng.integers(1, 5000)),
            }
        )
    return pd.DataFrame(rows)

