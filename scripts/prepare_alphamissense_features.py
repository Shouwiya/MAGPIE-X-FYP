from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_TABLE = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_residual"
DEFAULT_OOF = DEFAULT_OUTPUT_DIR / "oof_tabular_predictions.csv"
DEFAULT_ALPHA = ROOT / "data" / "external" / "alphamissense" / "AlphaMissense_hg38.tsv.gz"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "alphamissense_features.csv"

FORBIDDEN_FALLBACK_FILES = {
    (ROOT / "outputs" / "real" / "processed" / "annovar.csv").resolve(),
    (ROOT / "outputs" / "real" / "processed" / "dbnsfp.csv").resolve(),
}

ALPHAMISSENSE_COLUMNS = [
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


def check_not_forbidden(path: Path) -> None:
    if path.resolve() in FORBIDDEN_FALLBACK_FILES:
        raise ValueError(f"Refusing to use known fallback file: {path}")


def normalize_chrom(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    if text in {"M", "MT", "m", "mt"}:
        return "MT"
    return text.upper()


def make_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"chrom", "pos", "ref", "alt"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing coordinate columns required for AlphaMissense matching: {missing}")
    out = df.copy()
    out["match_chrom"] = out["chrom"].map(normalize_chrom)
    out["match_pos"] = pd.to_numeric(out["pos"], errors="coerce").astype("Int64")
    out["match_ref"] = out["ref"].astype(str).str.upper()
    out["match_alt"] = out["alt"].astype(str).str.upper()
    return out


def is_missense_value(value: object) -> bool:
    if pd.isna(value):
        return False
    return "missense" in str(value).lower()


def ensure_variant_id(df: pd.DataFrame) -> pd.DataFrame:
    if "variant_id" in df.columns:
        return df
    out = df.copy()
    out["variant_id"] = (
        out["chrom"].astype(str)
        + ":"
        + out["pos"].astype(str)
        + ":"
        + out["ref"].astype(str)
        + ">"
        + out["alt"].astype(str)
    )
    return out


def load_base_variants(oof_predictions: Path, feature_table: Path, max_rows: int | None) -> pd.DataFrame:
    check_not_forbidden(feature_table)
    if not feature_table.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table}")

    usecols = [column for column in ["variant_id", "chrom", "pos", "ref", "alt", "consequence", "gene"]]
    features = pd.read_csv(feature_table, usecols=lambda c: c in usecols, low_memory=False)
    features = ensure_variant_id(features)
    features = features.drop_duplicates(subset=["variant_id"])

    if oof_predictions.exists():
        oof = pd.read_csv(oof_predictions, usecols=lambda c: c in {"variant_id"}, low_memory=False)
        if "variant_id" not in oof.columns:
            raise ValueError(f"OOF predictions must contain variant_id: {oof_predictions}")
        selected = oof.drop_duplicates(subset=["variant_id"]).merge(features, on="variant_id", how="left")
    else:
        selected = features

    if max_rows is not None:
        selected = selected.head(max_rows)

    selected = make_key_frame(selected)
    selected["is_missense"] = selected.get("consequence", pd.Series(index=selected.index, dtype=object)).map(is_missense_value)
    return selected.reset_index(drop=True)


def load_matching_alpha_rows(alpha_path: Path, keys: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    if not alpha_path.exists():
        raise FileNotFoundError(f"AlphaMissense file not found: {alpha_path}")

    key_columns = ["match_chrom", "match_pos", "match_ref", "match_alt"]
    wanted = keys.loc[keys["is_missense"], key_columns].dropna().drop_duplicates()
    if wanted.empty:
        return pd.DataFrame(columns=[*key_columns, *ALPHAMISSENSE_COLUMNS])

    matches: list[pd.DataFrame] = []
    reader = pd.read_csv(
        alpha_path,
        sep="\t",
        comment="#",
        header=None,
        names=ALPHAMISSENSE_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        chunk = make_key_frame(chunk)
        merged = chunk.merge(wanted, on=key_columns, how="inner")
        if not merged.empty:
            matches.append(merged)
        if chunk_number % 25 == 0:
            print(f"Scanned {chunk_number * chunksize:,} AlphaMissense rows")

    if not matches:
        return pd.DataFrame(columns=[*key_columns, *ALPHAMISSENSE_COLUMNS])
    return pd.concat(matches, ignore_index=True)


def collapse_alpha_matches(matches: pd.DataFrame, duplicate_policy: str) -> pd.DataFrame:
    if matches.empty:
        return matches
    matches = matches.copy()
    matches["am_pathogenicity"] = pd.to_numeric(matches["am_pathogenicity"], errors="coerce")
    key_columns = ["match_chrom", "match_pos", "match_ref", "match_alt"]

    if duplicate_policy == "max":
        idx = matches.groupby(key_columns)["am_pathogenicity"].idxmax()
        return matches.loc[idx].reset_index(drop=True)
    if duplicate_policy == "mean":
        return (
            matches.groupby(key_columns, as_index=False)
            .agg(
                {
                    "am_pathogenicity": "mean",
                    "uniprot_id": "first",
                    "transcript_id": "first",
                    "protein_variant": "first",
                }
            )
            .reset_index(drop=True)
        )
    if duplicate_policy == "first":
        return matches.drop_duplicates(subset=key_columns, keep="first").reset_index(drop=True)
    raise ValueError(f"Unknown duplicate policy: {duplicate_policy}")


def prepare_features(args: argparse.Namespace) -> pd.DataFrame:
    base = load_base_variants(args.oof_predictions, args.feature_table, args.max_rows)
    matches = load_matching_alpha_rows(args.alphamissense, base, args.chunksize)
    collapsed = collapse_alpha_matches(matches, args.duplicate_policy)

    key_columns = ["match_chrom", "match_pos", "match_ref", "match_alt"]
    keep = [
        *key_columns,
        "am_pathogenicity",
        "uniprot_id",
        "transcript_id",
        "protein_variant",
    ]
    collapsed = collapsed[[column for column in keep if column in collapsed.columns]]
    merged = base.merge(collapsed, on=key_columns, how="left")

    merged["alphamissense_score"] = pd.to_numeric(merged["am_pathogenicity"], errors="coerce")
    merged.loc[~merged["is_missense"], "alphamissense_score"] = np.nan
    merged["alphamissense_available"] = merged["is_missense"] & merged["alphamissense_score"].notna()

    out = merged[
        [
            "variant_id",
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene",
            "consequence",
            "alphamissense_score",
            "alphamissense_available",
            "uniprot_id",
            "transcript_id",
            "protein_variant",
        ]
    ].copy()
    out = out.rename(
        columns={
            "uniprot_id": "alphamissense_uniprot_id",
            "transcript_id": "alphamissense_transcript_id",
            "protein_variant": "alphamissense_protein_variant",
        }
    )
    return out


def write_manifest(path: Path, args: argparse.Namespace, features: pd.DataFrame) -> None:
    payload = {
        "alphamissense": str(args.alphamissense),
        "feature_table": str(args.feature_table),
        "oof_predictions": str(args.oof_predictions),
        "output": str(args.output),
        "duplicate_policy": args.duplicate_policy,
        "rows": int(len(features)),
        "missense_rows": int(features["consequence"].map(is_missense_value).sum()) if "consequence" in features else None,
        "alphamissense_available_rows": int(features["alphamissense_available"].sum()),
        "note": "AlphaMissense is used only as a missense-variant score. Its provided class is not used as a ground-truth label.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features = prepare_features(args)
    features.to_csv(args.output, index=False)
    write_manifest(args.output.with_suffix(".manifest.json"), args, features)
    print(f"Saved AlphaMissense features to {args.output}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match AlphaMissense hg38 scores to missense variants by chrom, pos, ref, and alt."
    )
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--alphamissense", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--duplicate-policy", choices=["max", "mean", "first"], default="max")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug row limit; omit for real matching.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
