from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.data.real_pipeline import build_basic_feature_table
from magpie_x.utils.logging import setup_logger

DEFAULT_CLINVAR = ROOT / "data" / "interim" / "clinvar" / "clinvar_exonic_protein_altering.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "features"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a basic real-source feature table from filtered ClinVar and optional real ANNOVAR multianno.")
    parser.add_argument("--clinvar", type=Path, default=DEFAULT_CLINVAR)
    parser.add_argument("--annovar-multianno", type=Path, default=None, help="Optional real ANNOVAR multianno file. Synthetic fallback files are rejected.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logger = setup_logger("magpie_x.build_basic_feature_table", args.output_dir / "build_basic_feature_table.log")
    manifest = build_basic_feature_table(
        args.clinvar,
        args.output_dir / "basic_feature_table.csv",
        args.output_dir / "basic_feature_manifest.json",
        args.output_dir / "missing_sources_report.md",
        output_parquet=args.output_dir / "basic_feature_table.parquet",
        annovar_multianno=args.annovar_multianno,
    )
    logger.info("Built basic feature table with %s rows and %s columns.", manifest["rows"], len(manifest["columns"]))
    if not manifest["parquet_status"].get("written", False):
        logger.warning("Parquet not written: %s", manifest["parquet_status"].get("reason", "not requested"))


if __name__ == "__main__":
    main()

