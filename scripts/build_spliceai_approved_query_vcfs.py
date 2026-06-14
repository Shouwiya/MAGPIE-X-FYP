#!/usr/bin/env python3
"""Build label-free SpliceAI VCFs for exact approved benchmark samples."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from run_sample_size_benchmark import DEFAULT_INPUT, balanced_sample, load_feature_table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "spliceai"
ALLELE_RE = re.compile(r"^[ACGTN]+$")


def natural_contig_key(contig: str) -> tuple[int, str]:
    normalized = contig.removeprefix("chr")
    if normalized.isdigit():
        return int(normalized), ""
    return {"X": 23, "Y": 24, "M": 25, "MT": 25}.get(normalized.upper(), 1000), normalized


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_query_vcf(sample, output: Path) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    contigs = sorted(sample["chrom"].astype(str).unique(), key=natural_contig_key)
    unsupported_alt = 0

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(f"##fileDate={datetime.now(timezone.utc):%Y%m%d}\n")
        handle.write("##source=FYP_MAGPIE_X_exact_approved_sample\n")
        handle.write("##reference=hg38\n")
        handle.write(
            "##INFO=<ID=UNSUPPORTED_ALLELE,Number=0,Type=Flag,"
            'Description="Allele is unavailable for exact precomputed matching; no score inferred">\n'
        )
        for contig in contigs:
            handle.write(f"##contig=<ID={contig}>\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        for row in sample.itertuples(index=False):
            chrom = str(row.chrom)
            pos = int(row.pos)
            variant_id = str(row.variant_id)
            ref = "." if pd.isna(row.ref) else str(row.ref).upper()
            alt = "." if pd.isna(row.alt) else str(row.alt).upper()

            if not ALLELE_RE.fullmatch(ref):
                raise ValueError(f"Unsupported REF for {variant_id}: {ref!r}")
            if alt == ".":
                info = "UNSUPPORTED_ALLELE"
                unsupported_alt += 1
            elif ALLELE_RE.fullmatch(alt):
                info = "."
            else:
                raise ValueError(f"Unsupported ALT for {variant_id}: {alt!r}")

            handle.write(
                f"{chrom}\t{pos}\t{variant_id}\t{ref}\t{alt}\t.\tPASS\t{info}\n"
            )

    return {"records": len(sample), "unsupported_alt": unsupported_alt}


def run(args: argparse.Namespace) -> None:
    base_df = load_feature_table(args.input)
    required = {"variant_id", "chrom", "pos", "ref", "alt", "label"}
    missing = sorted(required - set(base_df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    for sample_size in args.sample_sizes:
        sampled = balanced_sample(
            base_df.reset_index(names="_source_index"),
            sample_size=sample_size,
            seed=args.seed + sample_size,
        )
        output = (
            args.output_dir
            / f"clinvar_reference_{sample_size // 1000}k_for_spliceai.dryrun.vcf"
        )
        stats = write_query_vcf(sampled, output)
        print(
            f"{output}: records={stats['records']:,}, "
            f"unsupported_alt={stats['unsupported_alt']:,}, sha256={sha256(output)}"
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create label-free SpliceAI query VCFs using the exact balanced "
            "sampling logic of the AA/BLOSUM low-circularity benchmark."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[10_000, 20_000])
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
