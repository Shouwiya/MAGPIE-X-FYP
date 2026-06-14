#!/usr/bin/env python3
"""Exact-match AlphaMissense hg38 scores to the approved 10k/20k samples."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from match_alphamissense_5k import collapse_matches, make_variant_id, scan_alpha


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPHA = (
    ROOT / "data" / "external" / "alphamissense" / "AlphaMissense_hg38.tsv.gz"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_enhanced"


def input_path(sample_size: int) -> Path:
    return (
        ROOT
        / "outputs"
        / "real"
        / "magpie_parity"
        / f"reference_{sample_size // 1000}k_aa_spliceai_feature_table.csv"
    )


def load_query(path: Path, expected_rows: int) -> pd.DataFrame:
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
        raise ValueError(f"{path} is missing columns: {missing}")
    if len(query) != expected_rows or query["variant_id"].duplicated().any():
        raise ValueError(
            f"Expected {expected_rows:,} unique variants in {path}; "
            f"found {len(query):,} rows."
        )
    query["variant_id"] = query["variant_id"].astype(str)
    reconstructed = make_variant_id(
        query["chrom"], query["pos"], query["ref"], query["alt"]
    )
    mismatch = ~reconstructed.eq(query["variant_id"])
    if mismatch.any():
        examples = query.loc[mismatch, "variant_id"].head(5).tolist()
        raise ValueError(f"Coordinate-derived IDs disagree with stored IDs: {examples}")
    query["is_missense"] = (
        query["consequence"].astype(str).str.lower().str.contains("missense")
    )
    return query


def distribution(features: pd.DataFrame) -> dict[str, float | int]:
    scores = pd.to_numeric(
        features.loc[
            features["alphamissense_available"].eq(1), "alphamissense_score"
        ],
        errors="coerce",
    ).dropna()
    if scores.empty:
        return {"count": 0}
    return {
        "count": int(len(scores)),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "min": float(scores.min()),
        "q25": float(scores.quantile(0.25)),
        "median": float(scores.median()),
        "q75": float(scores.quantile(0.75)),
        "max": float(scores.max()),
    }


def write_audit(
    path: Path,
    *,
    alpha_path: Path,
    results: dict[int, pd.DataFrame],
) -> None:
    lines = [
        "# AlphaMissense 10k/20k Match Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "AlphaMissense hg38 was matched by normalized chromosome, integer position, "
        "uppercase REF, and uppercase ALT. The resource was scanned once for the union "
        "of missense variants in the two approved samples.",
        "",
        f"- Resource: `{alpha_path.relative_to(ROOT).as_posix()}`",
        "- Labels used for matching: **No**.",
        "- AlphaMissense class labels used: **No**.",
        "- Tie rule for multiple transcript records: maximum numeric pathogenicity "
        "score, with the record count preserved.",
        "- Non-missense and unmatched missense variants retain missing score plus "
        "`alphamissense_available=0`; no neutral score is fabricated.",
        "",
        "## Coverage",
        "",
        "| Sample | Total variants | Missense variants | Matched scores | Eligible missense unmatched | Total unmatched | Multiple records |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sample_size in sorted(results):
        features = results[sample_size]
        missense = int(features["is_missense"].sum())
        matched = int(features["alphamissense_available"].sum())
        lines.append(
            f"| {sample_size:,} | {len(features):,} | {missense:,} | "
            f"{matched:,} | {missense - matched:,} | {len(features) - matched:,} | "
            f"{int(features['alphamissense_record_count'].gt(1).sum()):,} |"
        )
    lines.extend(["", "## Score Distribution", ""])
    for sample_size in sorted(results):
        stats = distribution(results[sample_size])
        lines.append(f"### {sample_size // 1000}k")
        lines.append("")
        if stats["count"] == 0:
            lines.append("No exact numeric AlphaMissense scores were matched.")
        else:
            lines.extend(
                [
                    f"- Count: {stats['count']:,}",
                    f"- Mean: {stats['mean']:.6f}",
                    f"- Standard deviation: {stats['std']:.6f}",
                    f"- Minimum: {stats['min']:.6f}",
                    f"- 25th percentile: {stats['q25']:.6f}",
                    f"- Median: {stats['median']:.6f}",
                    f"- 75th percentile: {stats['q75']:.6f}",
                    f"- Maximum: {stats['max']:.6f}",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Output Sidecars",
            "",
            "- `outputs/real/magpie_x_enhanced/alphamissense_10k_features.csv`",
            "- `outputs/real/magpie_x_enhanced/alphamissense_20k_features.csv`",
            "",
            "The sidecars are aligned to the exact source-table order and contain no "
            "clinical label column.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queries = {
        sample_size: load_query(input_path(sample_size), sample_size)
        for sample_size in args.sample_sizes
    }
    wanted_ids: set[str] = set()
    for query in queries.values():
        wanted_ids.update(query.loc[query["is_missense"], "variant_id"])
    matches = scan_alpha(args.alphamissense, wanted_ids, chunksize=args.chunksize)

    results: dict[int, pd.DataFrame] = {}
    for sample_size, query in queries.items():
        sample_matches = matches[matches["variant_id"].isin(set(query["variant_id"]))]
        features = collapse_matches(query, sample_matches)
        if features["variant_id"].tolist() != query["variant_id"].tolist():
            raise RuntimeError(f"{sample_size:,} AlphaMissense sidecar order changed.")
        if "label" in features.columns:
            raise RuntimeError("AlphaMissense sidecar unexpectedly contains labels.")
        output = args.output_dir / f"alphamissense_{sample_size // 1000}k_features.csv"
        features.to_csv(output, index=False)
        results[sample_size] = features
        print(
            f"{sample_size // 1000}k: matched "
            f"{int(features['alphamissense_available'].sum()):,}/"
            f"{int(features['is_missense'].sum()):,} missense variants",
            flush=True,
        )

    write_audit(
        args.output_dir / "alphamissense_10k_20k_match_audit.md",
        alpha_path=args.alphamissense,
        results=results,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-match AlphaMissense hg38 to approved 10k/20k samples."
    )
    parser.add_argument("--alphamissense", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[10_000, 20_000]
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
