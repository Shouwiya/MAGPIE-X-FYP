from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_TABLE = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_residual"
DEFAULT_OOF = DEFAULT_OUTPUT_DIR / "oof_tabular_predictions.csv"
FORBIDDEN_FALLBACK_FILES = {
    (ROOT / "outputs" / "real" / "processed" / "annovar.csv").resolve(),
    (ROOT / "outputs" / "real" / "processed" / "dbnsfp.csv").resolve(),
}


def check_not_forbidden(path: Path) -> None:
    if path.resolve() in FORBIDDEN_FALLBACK_FILES:
        raise ValueError(f"Refusing to use known synthetic fallback file: {path}")


def infer_separator(path: Path) -> str:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return "\t"
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return ","


def load_base_variants(oof_predictions: Path, feature_table: Path) -> pd.DataFrame:
    if oof_predictions.exists():
        base = pd.read_csv(oof_predictions, low_memory=False)
        source = oof_predictions
    else:
        check_not_forbidden(feature_table)
        if not feature_table.exists():
            raise FileNotFoundError(
                "No OOF predictions found and feature table is missing. "
                f"Expected one of: {oof_predictions}, {feature_table}"
            )
        base = pd.read_csv(feature_table, low_memory=False)
        source = feature_table
    if "variant_id" not in base.columns:
        raise ValueError(f"{source} must contain variant_id.")
    keep = [column for column in ["variant_id", "consequence", "gene"] if column in base.columns]
    return base[keep].drop_duplicates(subset=["variant_id"]).reset_index(drop=True)


def is_missense(consequence: object) -> bool:
    if pd.isna(consequence):
        return False
    return "missense" in str(consequence).lower()


def find_score_column(df: pd.DataFrame, explicit: str | None, candidates: list[str], source_name: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{source_name} score column not found: {explicit}")
        return explicit
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lower_lookup = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    raise ValueError(
        f"Could not infer {source_name} score column. "
        f"Provide --{source_name.lower()}-score-column. Available columns: {list(df.columns)[:30]}"
    )


def load_score_table(path: Path, score_column: str | None, candidates: list[str], source_name: str) -> pd.DataFrame:
    check_not_forbidden(path)
    if not path.exists():
        raise FileNotFoundError(path)
    sep = infer_separator(path)
    scores = pd.read_csv(path, sep=sep, low_memory=False)
    if "variant_id" not in scores.columns:
        raise ValueError(f"{source_name} file must contain variant_id.")
    selected_score_column = find_score_column(scores, score_column, candidates, source_name)
    out = scores[["variant_id", selected_score_column]].copy()
    out = out.rename(columns={selected_score_column: f"{source_name.lower()}_score"})
    out[f"{source_name.lower()}_score"] = pd.to_numeric(out[f"{source_name.lower()}_score"], errors="coerce")
    return out.drop_duplicates(subset=["variant_id"])


def attach_scores(
    base: pd.DataFrame,
    score_path: Path | None,
    score_column: str | None,
    *,
    source_name: str,
    candidates: list[str],
    report_lines: list[str],
) -> pd.DataFrame:
    score_field = f"{source_name.lower()}_score"
    available_field = f"{source_name.lower()}_available"
    if score_path is None:
        base[score_field] = np.nan
        base[available_field] = False
        report_lines.append(f"- {source_name}: missing. No file was provided, so no scores were fabricated.")
        return base

    scores = load_score_table(score_path, score_column, candidates, source_name)
    merged = base.merge(scores, on="variant_id", how="left")
    if "consequence" in merged.columns:
        missense_mask = merged["consequence"].map(is_missense)
    else:
        missense_mask = pd.Series(False, index=merged.index)
    merged.loc[~missense_mask, score_field] = np.nan
    merged[available_field] = merged[score_field].notna() & missense_mask
    n_available = int(merged[available_field].sum())
    report_lines.append(
        f"- {source_name}: loaded `{score_path}`. Available missense scores after merge: {n_available}."
    )
    return merged


def write_report(path: Path, lines: list[str], output_path: Path) -> None:
    content = [
        "# Missing Expert Features Report",
        "",
        "AlphaMissense and ESM/ESM1b are optional missense-only expert sources.",
        "This script never fabricates missing expert scores.",
        "",
        "## Source Status",
        "",
        *lines,
        "",
        "## Scope Rule",
        "",
        "Expert scores are kept only for missense variants. For frameshift, stopgain, stoploss, splice, and other non-missense variants, expert score fields are set to missing and availability flags are false.",
        "",
        f"Output feature file: `{output_path}`",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_base_variants(args.oof_predictions, args.feature_table)
    report_lines: list[str] = []

    base = attach_scores(
        base,
        args.alphamissense,
        args.alphamissense_score_column,
        source_name="alphamissense",
        candidates=["alphamissense_score", "AlphaMissense_score", "am_pathogenicity", "score"],
        report_lines=report_lines,
    )
    base = attach_scores(
        base,
        args.esm,
        args.esm_score_column,
        source_name="esm",
        candidates=["esm_score", "esm1b_score", "ESM1b_score", "esm1b_llr", "score"],
        report_lines=report_lines,
    )

    output_path = args.output_dir / "external_expert_features.csv"
    base.to_csv(output_path, index=False)
    write_report(args.output_dir / "missing_expert_features_report.md", report_lines, output_path)
    print(f"Saved external expert feature table to {output_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare optional missense-only AlphaMissense and ESM expert features without fabricating missing scores."
    )
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alphamissense", type=Path, default=None, help="Optional CSV/TSV with variant_id and AlphaMissense score.")
    parser.add_argument("--alphamissense-score-column", type=str, default=None)
    parser.add_argument("--esm", type=Path, default=None, help="Optional CSV/TSV with variant_id and ESM/ESM1b score.")
    parser.add_argument("--esm-score-column", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
