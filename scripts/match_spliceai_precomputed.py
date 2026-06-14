from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Iterator


OUTPUT_COLUMNS = [
    "query_chrom",
    "query_pos",
    "query_id",
    "ref",
    "alt",
    "spliceai_source",
    "matched",
    "allele",
    "symbol",
    "DS_AG",
    "DS_AL",
    "DS_DG",
    "DS_DL",
    "DP_AG",
    "DP_AL",
    "DP_DG",
    "DP_DL",
    "raw_spliceai",
]


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{description} is not a file: {path}")


def require_index(path: Path, description: str) -> None:
    candidates = [
        Path(str(path) + ".tbi"),
        Path(str(path) + ".csi"),
        path.with_suffix(".tbi"),
        path.with_suffix(".csi"),
    ]
    if not any(candidate.exists() for candidate in candidates):
        expected = " or ".join(str(candidate) for candidate in candidates[:2])
        raise FileNotFoundError(
            f"{description} requires a tabix/CSI index. Expected {expected}"
        )


def require_pysam():
    try:
        import pysam
    except ImportError as exc:
        raise RuntimeError(
            "pysam is required for precomputed SpliceAI matching. "
            "Install it in a supported environment before running this script."
        ) from exc
    return pysam


def normalize_contig(source_contigs: set[str], query_contig: str) -> str | None:
    candidates = [query_contig]
    if query_contig.startswith("chr"):
        candidates.append(query_contig[3:])
    else:
        candidates.append(f"chr{query_contig}")
    if query_contig in {"M", "MT", "chrM", "chrMT"}:
        candidates.extend(["M", "MT", "chrM", "chrMT"])
    for candidate in candidates:
        if candidate in source_contigs:
            return candidate
    return None


def spliceai_values(record) -> list[str]:
    value = record.info.get("SpliceAI")
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def parse_spliceai_value(value: str) -> dict[str, str] | None:
    fields = value.split("|")
    if len(fields) != 10:
        return None
    return dict(zip(["allele", "symbol", *OUTPUT_COLUMNS[9:17]], fields))


def exact_matches(reader, chrom: str, pos: int, ref: str, alt: str) -> Iterator[dict[str, str]]:
    source_contig = normalize_contig(set(reader.header.contigs), chrom)
    if source_contig is None:
        return
    for record in reader.fetch(source_contig, pos - 1, pos):
        if record.pos != pos or record.ref.upper() != ref:
            continue
        source_alts = {str(value).upper() for value in (record.alts or ())}
        if alt not in source_alts:
            continue
        for raw_value in spliceai_values(record):
            parsed = parse_spliceai_value(raw_value)
            if parsed is None or parsed["allele"].upper() != alt:
                continue
            parsed["raw_spliceai"] = raw_value
            yield parsed


def unmatched_row(
    *,
    chrom: str,
    pos: int,
    query_id: str,
    ref: str,
    alt: str,
    source: str,
) -> dict[str, object]:
    return {
        "query_chrom": chrom,
        "query_pos": pos,
        "query_id": query_id,
        "ref": ref,
        "alt": alt,
        "spliceai_source": source,
        "matched": 0,
        "allele": "",
        "symbol": "",
        "DS_AG": "",
        "DS_AL": "",
        "DS_DG": "",
        "DS_DL": "",
        "DP_AG": "",
        "DP_AL": "",
        "DP_DG": "",
        "DP_DL": "",
        "raw_spliceai": "",
    }


def run(args: argparse.Namespace) -> None:
    require_file(args.query_vcf, "Query VCF")
    require_file(args.spliceai_snv_vcf, "Precomputed SpliceAI SNV VCF")
    require_file(args.spliceai_indel_vcf, "Precomputed SpliceAI indel VCF")
    require_index(args.spliceai_snv_vcf, "Precomputed SpliceAI SNV VCF")
    require_index(args.spliceai_indel_vcf, "Precomputed SpliceAI indel VCF")

    input_paths = {
        args.query_vcf.resolve(),
        args.spliceai_snv_vcf.resolve(),
        args.spliceai_indel_vcf.resolve(),
    }
    if args.output.resolve() in input_paths:
        raise ValueError("Output path must differ from all input paths.")
    if "basic_feature_table" in args.output.name.lower():
        raise ValueError(
            "Refusing to write directly to a main feature-table path. "
            "Use a separate SpliceAI match sidecar output."
        )

    pysam = require_pysam()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)

    query_reader = pysam.VariantFile(str(args.query_vcf))
    snv_reader = pysam.VariantFile(str(args.spliceai_snv_vcf))
    indel_reader = pysam.VariantFile(str(args.spliceai_indel_vcf))
    try:
        with temporary_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
            writer.writeheader()
            for query_record in query_reader:
                chrom = str(query_record.chrom)
                pos = int(query_record.pos)
                query_id = str(query_record.id or ".")
                ref = str(query_record.ref).upper()
                query_alts = query_record.alts or ()
                if not query_alts:
                    writer.writerow(
                        unmatched_row(
                            chrom=chrom,
                            pos=pos,
                            query_id=query_id,
                            ref=ref,
                            alt=".",
                            source="unsupported",
                        )
                    )
                    continue
                for alt_value in query_alts:
                    alt = str(alt_value).upper()
                    is_snv = len(ref) == 1 and len(alt) == 1
                    reader = snv_reader if is_snv else indel_reader
                    source = "snv" if is_snv else "indel"
                    matches = list(exact_matches(reader, chrom, pos, ref, alt))
                    if not matches:
                        writer.writerow(
                            unmatched_row(
                                chrom=chrom,
                                pos=pos,
                                query_id=query_id,
                                ref=ref,
                                alt=alt,
                                source=source,
                            )
                        )
                        continue
                    for match in matches:
                        writer.writerow(
                            {
                                "query_chrom": chrom,
                                "query_pos": pos,
                                "query_id": query_id,
                                "ref": ref,
                                "alt": alt,
                                "spliceai_source": source,
                                "matched": 1,
                                **match,
                            }
                        )
        temporary_output.replace(args.output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    finally:
        query_reader.close()
        snv_reader.close()
        indel_reader.close()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact-match a label-free query VCF against indexed precomputed "
            "SpliceAI SNV and indel VCFs. Writes a separate long-form TSV sidecar."
        )
    )
    parser.add_argument("--query-vcf", type=Path, required=True)
    parser.add_argument("--spliceai-snv-vcf", type=Path, required=True)
    parser.add_argument("--spliceai-indel-vcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    try:
        run(parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
