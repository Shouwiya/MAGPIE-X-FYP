from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_OUTPUT = (
    ROOT / "data" / "interim" / "features" / "basic_feature_table_with_clean_aa_features.csv"
)
DEFAULT_AUDIT = ROOT / "outputs" / "real" / "magpie_parity" / "AA_BLOSUM_FEATURE_AUDIT.md"
DEFAULT_REPORT = ROOT / "outputs" / "real" / "magpie_parity" / "AA_FEATURE_BUILD_REPORT.md"

AA_FEATURE_COLUMNS = [
    "aa_ref",
    "aa_alt",
    "aa_substitution_available",
    "molecular_weight_delta",
    "hydrophobicity_delta",
    "polarity_change",
    "charge_change",
    "aromatic_change",
    "acid_base_change",
    "BLOSUM62_score",
]

CANONICAL_AA = set("ARNDCQEGHILKMFPSTWYV")
THREE_TO_ONE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Ter": "*",
    "Stop": "*",
}

# Average amino-acid molecular weights, Kyte-Doolittle hydrophobicity, and
# Grantham polarity. Deltas are always alternate minus reference.
AA_PROPERTIES = {
    "A": {"molecular_weight": 89.09, "hydrophobicity": 1.8, "polarity": 8.1},
    "R": {"molecular_weight": 174.20, "hydrophobicity": -4.5, "polarity": 10.5},
    "N": {"molecular_weight": 132.12, "hydrophobicity": -3.5, "polarity": 11.6},
    "D": {"molecular_weight": 133.10, "hydrophobicity": -3.5, "polarity": 13.0},
    "C": {"molecular_weight": 121.16, "hydrophobicity": 2.5, "polarity": 5.5},
    "Q": {"molecular_weight": 146.15, "hydrophobicity": -3.5, "polarity": 10.5},
    "E": {"molecular_weight": 147.13, "hydrophobicity": -3.5, "polarity": 12.3},
    "G": {"molecular_weight": 75.07, "hydrophobicity": -0.4, "polarity": 9.0},
    "H": {"molecular_weight": 155.16, "hydrophobicity": -3.2, "polarity": 10.4},
    "I": {"molecular_weight": 131.17, "hydrophobicity": 4.5, "polarity": 5.2},
    "L": {"molecular_weight": 131.17, "hydrophobicity": 3.8, "polarity": 4.9},
    "K": {"molecular_weight": 146.19, "hydrophobicity": -3.9, "polarity": 11.3},
    "M": {"molecular_weight": 149.21, "hydrophobicity": 1.9, "polarity": 5.7},
    "F": {"molecular_weight": 165.19, "hydrophobicity": 2.8, "polarity": 5.2},
    "P": {"molecular_weight": 115.13, "hydrophobicity": -1.6, "polarity": 8.0},
    "S": {"molecular_weight": 105.09, "hydrophobicity": -0.8, "polarity": 9.2},
    "T": {"molecular_weight": 119.12, "hydrophobicity": -0.7, "polarity": 8.6},
    "W": {"molecular_weight": 204.23, "hydrophobicity": -0.9, "polarity": 5.4},
    "Y": {"molecular_weight": 181.19, "hydrophobicity": -1.3, "polarity": 6.2},
    "V": {"molecular_weight": 117.15, "hydrophobicity": 4.2, "polarity": 5.9},
}

CHARGE_CLASS = {aa: 0 for aa in CANONICAL_AA}
CHARGE_CLASS.update({"D": -1, "E": -1, "H": 1, "K": 1, "R": 1})
AROMATIC = {aa: int(aa in {"F", "W", "Y"}) for aa in CANONICAL_AA}

AA_TOKEN = r"(?:[A-Z][a-z]{2}|Stop|[A-Z*])"
SUBSTITUTION_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9])p\.\(?(?P<ref>{AA_TOKEN})(?P<position>[1-9][0-9]*)(?P<alt>{AA_TOKEN})\)?(?![A-Za-z])"
)

BLOSUM62_ORDER = list("ARNDCQEGHILKMFPSTWYV")
BLOSUM62_ROWS = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]
BLOSUM62 = {
    (ref, alt): BLOSUM62_ROWS[row_idx][col_idx]
    for row_idx, ref in enumerate(BLOSUM62_ORDER)
    for col_idx, alt in enumerate(BLOSUM62_ORDER)
}


def normalize_aa_token(token: str) -> str:
    return THREE_TO_ONE.get(token, token)


def parse_substitution(value: object) -> tuple[str | None, str | None, str]:
    if pd.isna(value):
        return None, None, "missing_aa_change"
    text = str(value).strip()
    if not text or text in {"-", ".", "UNKNOWN"}:
        return None, None, "missing_or_unknown_aa_change"

    matches = [
        (normalize_aa_token(match.group("ref")), normalize_aa_token(match.group("alt")))
        for match in SUBSTITUTION_PATTERN.finditer(text)
    ]
    if not matches:
        return None, None, "no_simple_substitution"
    if any(ref not in CANONICAL_AA or alt not in CANONICAL_AA for ref, alt in matches):
        return None, None, "noncanonical_or_stop_substitution"

    unique_pairs = set(matches)
    if len(unique_pairs) != 1:
        return None, None, "conflicting_transcript_substitutions"
    ref, alt = next(iter(unique_pairs))
    return ref, alt, "available"


def build_aa_feature_frame(series: pd.Series) -> tuple[pd.DataFrame, Counter]:
    rows: list[dict[str, object]] = []
    statuses: Counter = Counter()
    for value in series:
        ref, alt, status = parse_substitution(value)
        statuses[status] += 1
        if status != "available":
            rows.append(
                {
                    "aa_ref": np.nan,
                    "aa_alt": np.nan,
                    "aa_substitution_available": 0,
                    "molecular_weight_delta": np.nan,
                    "hydrophobicity_delta": np.nan,
                    "polarity_change": np.nan,
                    "charge_change": np.nan,
                    "aromatic_change": np.nan,
                    "acid_base_change": np.nan,
                    "BLOSUM62_score": np.nan,
                }
            )
            continue

        assert ref is not None and alt is not None
        ref_properties = AA_PROPERTIES[ref]
        alt_properties = AA_PROPERTIES[alt]
        ref_charge = CHARGE_CLASS[ref]
        alt_charge = CHARGE_CLASS[alt]
        rows.append(
            {
                "aa_ref": ref,
                "aa_alt": alt,
                "aa_substitution_available": 1,
                "molecular_weight_delta": alt_properties["molecular_weight"]
                - ref_properties["molecular_weight"],
                "hydrophobicity_delta": alt_properties["hydrophobicity"]
                - ref_properties["hydrophobicity"],
                "polarity_change": alt_properties["polarity"] - ref_properties["polarity"],
                "charge_change": alt_charge - ref_charge,
                "aromatic_change": AROMATIC[alt] - AROMATIC[ref],
                "acid_base_change": int(alt_charge != ref_charge),
                "BLOSUM62_score": BLOSUM62[(ref, alt)],
            }
        )
    return pd.DataFrame(rows, index=series.index, columns=AA_FEATURE_COLUMNS), statuses


def inspect_header(input_path: Path) -> tuple[list[str], list[str], list[str]]:
    columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    lower_columns = {column.lower(): column for column in columns}
    aa_change_columns = [
        column
        for column in columns
        if "aachange" in column.lower().replace("_", "")
        or "aa_change" in column.lower()
        or "amino_acid_change" in column.lower()
    ]
    existing_aa_features = [
        column
        for column in columns
        if column.lower()
        in {
            "aa_ref",
            "aa_alt",
            "aa_substitution_available",
            "molecular_weight_delta",
            "hydrophobicity_delta",
            "polarity_change",
            "charge_change",
            "aromatic_change",
            "acid_base_change",
        }
    ]
    existing_blosum = [
        original
        for lower, original in lower_columns.items()
        if "blosum" in lower or "pam_score" in lower
    ]
    return aa_change_columns, existing_aa_features, existing_blosum


def count_rows_and_statuses(input_path: Path, aa_column: str, chunk_size: int) -> tuple[int, Counter]:
    row_count = 0
    statuses: Counter = Counter()
    for chunk in pd.read_csv(
        input_path,
        usecols=[aa_column],
        chunksize=chunk_size,
        low_memory=False,
    ):
        _, chunk_statuses = build_aa_feature_frame(chunk[aa_column])
        row_count += len(chunk)
        statuses.update(chunk_statuses)
    return row_count, statuses


def write_audit(
    path: Path,
    *,
    input_path: Path,
    row_count: int,
    aa_change_columns: list[str],
    existing_aa_features: list[str],
    existing_blosum: list[str],
    statuses: Counter,
) -> None:
    available = int(statuses["available"])
    percent = 100.0 * available / row_count if row_count else 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AA and BLOSUM Feature Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Input: `{input_path}`",
        "",
        "## Answers",
        "",
        f"- **Can amino acid substitutions be parsed?** Yes. `aa_change_refgene` contains ANNOVAR AAChange strings with protein substitutions such as `p.E15G`. The parser accepts one-letter and three-letter canonical substitutions and requires all parsed transcript substitutions in a row to agree on the same reference/alternate residue pair.",
        f"- **How many rows have parseable substitutions?** {available:,} of {row_count:,} rows ({percent:.2f}%) have one unambiguous canonical amino-acid substitution.",
        f"- **Are AA property delta features already present?** {'Yes' if existing_aa_features else 'No'}. Existing matching columns: {', '.join(f'`{column}`' for column in existing_aa_features) if existing_aa_features else 'none'}.",
        f"- **Is a BLOSUM-like score already present?** {'Yes' if existing_blosum else 'No'}. Existing matching columns: {', '.join(f'`{column}`' for column in existing_blosum) if existing_blosum else 'none'}.",
        "",
        "## Source Columns",
        "",
        f"- AAChange-style columns found: {', '.join(f'`{column}`' for column in aa_change_columns) if aa_change_columns else 'none'}",
        "- The existing improved low-circularity feature set uses only `aa_change_transcript_count` and `aa_change_has_multiple_transcripts`; it does not contain residue identities, physicochemical deltas, or BLOSUM.",
        "",
        "## Parse Status",
        "",
        "| status | rows |",
        "| --- | ---: |",
    ]
    status_order = [
        "available",
        "conflicting_transcript_substitutions",
        "noncanonical_or_stop_substitution",
        "no_simple_substitution",
        "missing_or_unknown_aa_change",
        "missing_aa_change",
    ]
    for status in status_order:
        lines.append(f"| {status} | {int(statuses[status]):,} |")
    lines.extend(
        [
            "",
            "## Conservative Parsing Rule",
            "",
            "Property deltas and BLOSUM62 are emitted only for canonical missense substitutions. Stop gains, uncertain start-loss notation, frameshifts, in-frame indels, malformed/unknown values, and transcript-discordant residue pairs are left unavailable. No residue or score is imputed from the clinical label or from a pathogenicity predictor.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_build_report(
    path: Path,
    *,
    input_path: Path,
    output_path: Path,
    row_count: int,
    statuses: Counter,
    output_written: bool,
) -> None:
    available = int(statuses["available"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Clean Amino-Acid Feature Build Report",
        "",
        f"Status: {'completed' if output_written else 'audit only; feature table not written'}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Input: `{input_path}`",
        f"Output: `{output_path}`",
        "",
        "## Build Summary",
        "",
        f"- Input rows: {row_count:,}",
        f"- Output rows: {row_count:,}" if output_written else "- Output rows: not written",
        f"- Rows with clean AA features: {available:,}",
        "- Existing rows and columns were preserved; the ten requested clean AA/BLOSUM columns were appended.",
        "- No database was downloaded.",
        "- No synthetic or toy data was used.",
        "- No prior pathogenicity predictor output was used to derive these features.",
        "",
        "## Added Columns",
        "",
        "| column | definition |",
        "| --- | --- |",
        "| aa_ref | Canonical reference amino acid, one-letter code. |",
        "| aa_alt | Canonical alternate amino acid, one-letter code. |",
        "| aa_substitution_available | 1 only when transcript substitutions resolve to one canonical residue pair; otherwise 0. |",
        "| molecular_weight_delta | Alternate minus reference average molecular weight. |",
        "| hydrophobicity_delta | Alternate minus reference Kyte-Doolittle hydrophobicity. |",
        "| polarity_change | Alternate minus reference Grantham polarity. |",
        "| charge_change | Alternate minus reference nominal charge class: acidic=-1, neutral=0, basic=+1. |",
        "| aromatic_change | Alternate minus reference aromatic indicator, using F/W/Y as aromatic. |",
        "| acid_base_change | 1 when acidic/basic/neutral charge class changes, otherwise 0. |",
        "| BLOSUM62_score | Standard built-in BLOSUM62 substitution matrix score. |",
        "",
        "## Guardrails",
        "",
        "The build does not use SIFT, PolyPhen, REVEL, CADD, MutationTaster, MetaSVM, MetaLR, FATHMM, PROVEAN, M-CAP, AlphaMissense, ESM, GNN output, or any related pathogenicity predictor score.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_output(
    input_path: Path,
    output_path: Path,
    *,
    aa_column: str,
    chunk_size: int,
) -> tuple[int, Counter]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    row_count = 0
    statuses: Counter = Counter()
    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(input_path, chunksize=chunk_size, low_memory=False),
            start=1,
        ):
            features, chunk_statuses = build_aa_feature_frame(chunk[aa_column])
            combined = pd.concat([chunk, features], axis=1)
            combined.to_csv(
                temporary_path,
                mode="w" if chunk_number == 1 else "a",
                header=chunk_number == 1,
                index=False,
            )
            row_count += len(chunk)
            statuses.update(chunk_statuses)
            print(f"chunk={chunk_number} rows_written={row_count}", flush=True)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return row_count, statuses


def validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input feature table not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ.")
    if "toy" in {part.lower() for part in input_path.resolve().parts}:
        raise ValueError(f"Refusing toy/synthetic input path: {input_path}")


def run(args: argparse.Namespace) -> None:
    validate_paths(args.input, args.output)
    aa_change_columns, existing_aa_features, existing_blosum = inspect_header(args.input)
    if "aa_change_refgene" not in aa_change_columns:
        raise ValueError("Required `aa_change_refgene` column was not found.")
    if any(column in pd.read_csv(args.input, nrows=0).columns for column in AA_FEATURE_COLUMNS):
        raise ValueError("One or more requested AA feature columns already exist in the input.")

    if args.audit_only:
        row_count, statuses = count_rows_and_statuses(
            args.input,
            "aa_change_refgene",
            args.chunk_size,
        )
        output_written = False
    else:
        row_count, statuses = build_output(
            args.input,
            args.output,
            aa_column="aa_change_refgene",
            chunk_size=args.chunk_size,
        )
        output_written = True

    write_audit(
        args.audit_report,
        input_path=args.input,
        row_count=row_count,
        aa_change_columns=aa_change_columns,
        existing_aa_features=existing_aa_features,
        existing_blosum=existing_blosum,
        statuses=statuses,
    )
    write_build_report(
        args.build_report,
        input_path=args.input,
        output_path=args.output,
        row_count=row_count,
        statuses=statuses,
        output_written=output_written,
    )
    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": row_count,
        "parse_status_counts": dict(statuses),
        "added_columns": AA_FEATURE_COLUMNS,
        "output_written": output_written,
        "synthetic_or_toy_data_used": False,
        "databases_downloaded": False,
        "prior_pathogenicity_predictors_used": False,
    }
    metadata_path = args.build_report.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean amino-acid property deltas and BLOSUM62 features from ANNOVAR AAChange text."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--build-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
