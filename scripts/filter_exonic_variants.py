from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.data.real_pipeline import filter_protein_altering_clinvar
from magpie_x.utils.logging import setup_logger

DEFAULT_INPUT = ROOT / "data" / "interim" / "clinvar" / "clinvar_parsed.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "clinvar"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter parsed ClinVar to protein-altering exonic variants.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logger = setup_logger("magpie_x.filter_exonic_variants", args.output_dir / "filter_exonic_variants.log")
    manifest = filter_protein_altering_clinvar(
        args.input,
        args.output_dir / "clinvar_exonic_protein_altering.csv",
        args.output_dir / "clinvar_exonic_filter_manifest.json",
        output_parquet=args.output_dir / "clinvar_exonic_protein_altering.parquet",
    )
    logger.info(
        "Filtered ClinVar rows: %s -> %s",
        manifest["rows_before_filtering"],
        manifest["rows_after_filtering"],
    )
    if not manifest["parquet_status"].get("written", False):
        logger.warning("Parquet not written: %s", manifest["parquet_status"].get("reason", "not requested"))


if __name__ == "__main__":
    main()

