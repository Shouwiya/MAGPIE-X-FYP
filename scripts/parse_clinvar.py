from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.data.real_pipeline import parse_clinvar_vcf
from magpie_x.utils.logging import setup_logger

DEFAULT_INPUT = ROOT / "data" / "raw" / "clinvar" / "clinvar.vcf.gz"
LEGACY_INPUT = ROOT / "data" / "raw" / "clinvar.vcf.gz"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "clinvar"


def resolve_input(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_INPUT and LEGACY_INPUT.exists():
        return LEGACY_INPUT
    raise FileNotFoundError(f"ClinVar VCF not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse ClinVar VCF into a normalized supervised CSV/parquet table.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--assembly", default="GRCh38")
    parser.add_argument("--max-records", type=int, default=None, help="Optional smoke-test limit on VCF records read.")
    args = parser.parse_args()

    logger = setup_logger("magpie_x.parse_clinvar", args.output_dir / "parse_clinvar.log")
    input_path = resolve_input(args.input)
    manifest = parse_clinvar_vcf(
        input_path,
        args.output_dir / "clinvar_parsed.csv",
        args.output_dir / "clinvar_parse_manifest.json",
        output_parquet=args.output_dir / "clinvar_parsed.parquet",
        assembly=args.assembly,
        max_records=args.max_records,
    )
    logger.info("Parsed ClinVar rows: %s", manifest["kept_supervised_rows"])
    if not manifest["parquet_status"].get("written", False):
        logger.warning("Parquet not written: %s", manifest["parquet_status"].get("reason", "not requested"))


if __name__ == "__main__":
    main()

