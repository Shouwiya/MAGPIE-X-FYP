#!/usr/bin/env python3
"""Exact-match AlphaMissense hg38 scores to the controlled 5k sample."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "real"
    / "magpie_x_enhanced"
    / "reference_5k_magpie_x_enhanced_feature_table.csv"
)
DEFAULT_ALPHA = (
    ROOT / "data" / "external" / "alphamissense" / "AlphaMissense_hg38.tsv.gz"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_enhanced"
ALPHA_COLUMNS = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "genome",
    "uniprot_id",
    "transcript_id",
    "protein_variant",
    "am_pathogenicity",
    "am_class",
]


def normalize_chrom(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.replace(
        r"^chr", "", regex=True, case=False
    )
    return normalized.replace({"M": "MT", "m": "MT", "mt": "MT"})


def make_variant_id(
    chrom: pd.Series,
    pos: pd.Series,
    ref: pd.Series,
    alt: pd.Series,
) -> pd.Series:
    return (
        normalize_chrom(chrom)
        + ":"
        + pd.to_numeric(pos, errors="coerce").astype("Int64").astype(str)
        + ":"
        + ref.astype(str).str.upper()
        + ">"
        + alt.astype(str).str.upper()
    )


def load_query(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    allowed = {
        "variant_id",
        "chrom",
        "pos",
        "ref",
        "alt",
        "consequence",
        "exonic_func_refgene",
        "aa_substitution_available",
    }
    query = pd.read_csv(
        path,
        usecols=lambda column: column in allowed,
        low_memory=False,
    )
    required = {"variant_id", "chrom", "pos", "ref", "alt", "consequence"}
    missing = sorted(required - set(query.columns))
    if missing:
        raise ValueError(f"Controlled 5k table is missing columns: {missing}")
    if len(query) != 5_000 or query["variant_id"].duplicated().any():
        raise ValueError("Expected exactly 5,000 unique controlled query variants.")

    query["variant_id"] = query["variant_id"].astype(str)
    reconstructed = make_variant_id(
        query["chrom"], query["pos"], query["ref"], query["alt"]
    )
    if not reconstructed.equals(query["variant_id"]):
        mismatches = query.loc[
            ~reconstructed.eq(query["variant_id"]), "variant_id"
        ].head(5)
        raise ValueError(
            f"Coordinate-derived IDs do not match stored variant IDs: {mismatches.tolist()}"
        )
    query["is_missense"] = (
        query["consequence"].astype(str).str.lower().str.contains("missense")
    )
    return query


def scan_alpha(
    path: Path,
    wanted_ids: set[str],
    *,
    chunksize: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    matches: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=ALPHA_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        chunk["variant_id"] = make_variant_id(
            chunk["chrom"], chunk["pos"], chunk["ref"], chunk["alt"]
        )
        selected = chunk[chunk["variant_id"].isin(wanted_ids)].copy()
        if not selected.empty:
            selected["am_pathogenicity"] = pd.to_numeric(
                selected["am_pathogenicity"], errors="coerce"
            )
            matches.append(selected)
        if chunk_number % 20 == 0:
            print(f"Scanned {chunk_number * chunksize:,} AlphaMissense rows", flush=True)
    if not matches:
        return pd.DataFrame(columns=["variant_id", *ALPHA_COLUMNS])
    return pd.concat(matches, ignore_index=True)


def collapse_matches(query: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        collapsed = pd.DataFrame(
            columns=[
                "variant_id",
                "alphamissense_score",
                "alphamissense_record_count",
                "alphamissense_uniprot_id",
                "alphamissense_transcript_id",
                "alphamissense_protein_variant",
            ]
        )
    else:
        matches = matches.sort_values(
            ["variant_id", "am_pathogenicity", "transcript_id", "uniprot_id"],
            ascending=[True, False, True, True],
            na_position="last",
        )
        winners = matches.drop_duplicates("variant_id", keep="first").copy()
        counts = matches.groupby("variant_id").size().rename(
            "alphamissense_record_count"
        )
        winners = winners.join(counts, on="variant_id")
        collapsed = winners.rename(
            columns={
                "am_pathogenicity": "alphamissense_score",
                "uniprot_id": "alphamissense_uniprot_id",
                "transcript_id": "alphamissense_transcript_id",
                "protein_variant": "alphamissense_protein_variant",
            }
        )[
            [
                "variant_id",
                "alphamissense_score",
                "alphamissense_record_count",
                "alphamissense_uniprot_id",
                "alphamissense_transcript_id",
                "alphamissense_protein_variant",
            ]
        ]

    output = query[
        ["variant_id", "chrom", "pos", "ref", "alt", "is_missense"]
    ].merge(collapsed, on="variant_id", how="left", validate="one_to_one")
    output["alphamissense_available"] = (
        output["is_missense"] & output["alphamissense_score"].notna()
    ).astype(int)
    output.loc[
        output["alphamissense_available"].eq(0), "alphamissense_score"
    ] = np.nan
    output["alphamissense_record_count"] = (
        pd.to_numeric(output["alphamissense_record_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return output


def write_audit(
    path: Path,
    *,
    input_path: Path,
    alpha_path: Path,
    output_path: Path,
    features: pd.DataFrame,
) -> None:
    scores = features.loc[
        features["alphamissense_available"].eq(1), "alphamissense_score"
    ].astype(float)
    missense_count = int(features["is_missense"].sum())
    matched = int(features["alphamissense_available"].sum())
    quantiles = scores.quantile([0.0, 0.25, 0.5, 0.75, 1.0]) if not scores.empty else None
    lines = [
        "# AlphaMissense 5k Match Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Inputs",
        "",
        "- Exact controlled 5k sample: "
        "`outputs/real/magpie_x_enhanced/reference_5k_magpie_x_enhanced_feature_table.csv`",
        "- AlphaMissense hg38 resource: "
        "`data/external/alphamissense/AlphaMissense_hg38.tsv.gz`",
        "- Output sidecar: "
        "`outputs/real/magpie_x_enhanced/alphamissense_5k_features.csv`",
        "",
        "## Coverage",
        "",
        "- Resource genome build: **hg38**.",
        "- Query genome build: **hg38**.",
        "- Genome-build match: **Yes**.",
        "- Exact matching key: normalized `chrom`, integer `pos`, uppercase `ref`, "
        "and uppercase `alt`.",
        f"- Total controlled variants: {len(features):,}",
        f"- Missense-only eligible variants: {missense_count:,}",
        f"- Exact AlphaMissense matches: {matched:,}",
        f"- Eligible missense variants without a score: {missense_count - matched:,}",
        f"- Total variants without a score: {len(features) - matched:,}",
        f"- Variants with multiple exact AlphaMissense records: "
        f"{int((features['alphamissense_record_count'] > 1).sum()):,}",
        "",
        "AlphaMissense is a missense-only expert. Non-missense variants remain unavailable rather "
        "than receiving a fabricated neutral score. When multiple exact transcript records exist, "
        "the maximum numeric pathogenicity score is retained and the record count is preserved.",
        "",
        "## Score Distribution",
        "",
    ]
    if scores.empty:
        lines.append("No exact numeric AlphaMissense scores were matched.")
    else:
        lines.extend(
            [
                f"- Count: {len(scores):,}",
                f"- Mean: {scores.mean():.6f}",
                f"- Standard deviation: {scores.std(ddof=1):.6f}",
                f"- Minimum: {quantiles.loc[0.0]:.6f}",
                f"- 25th percentile: {quantiles.loc[0.25]:.6f}",
                f"- Median: {quantiles.loc[0.5]:.6f}",
                f"- 75th percentile: {quantiles.loc[0.75]:.6f}",
                f"- Maximum: {quantiles.loc[1.0]:.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Label Guardrail",
            "",
            "- Labels used for matching: **No**.",
            "- The matcher reads only variant coordinates, consequence, and amino-acid availability from the controlled table.",
            "- AlphaMissense `am_class` is not written to the modeling sidecar and is not used as a label.",
            "- Matching is exact on normalized chromosome, position, REF, and ALT; "
            "no clinical label participates in the key or tie-breaking.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query = load_query(args.input)
    wanted_ids = set(query.loc[query["is_missense"], "variant_id"])
    matches = scan_alpha(args.alphamissense, wanted_ids, chunksize=args.chunksize)
    features = collapse_matches(query, matches)
    output = args.output_dir / "alphamissense_5k_features.csv"
    features.to_csv(output, index=False)
    write_audit(
        args.output_dir / "alphamissense_5k_match_audit.md",
        input_path=args.input,
        alpha_path=args.alphamissense,
        output_path=output,
        features=features,
    )
    print(
        f"Matched {int(features['alphamissense_available'].sum()):,}/"
        f"{int(features['is_missense'].sum()):,} eligible missense variants",
        flush=True,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-match AlphaMissense hg38 to the controlled MAGPIE-X 5k sample."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--alphamissense", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
