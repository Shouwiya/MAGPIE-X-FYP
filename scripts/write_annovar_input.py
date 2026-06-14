from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.data.real_pipeline import write_annovar_avinput
from magpie_x.utils.logging import setup_logger

DEFAULT_INPUT = ROOT / "data" / "interim" / "clinvar" / "clinvar_exonic_protein_altering.csv"
DEFAULT_OUTPUT = ROOT / "data" / "interim" / "annovar_input" / "clinvar_exonic_protein_altering.avinput"
DEFAULT_MANIFEST = ROOT / "data" / "interim" / "annovar_input" / "annovar_input_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write filtered ClinVar variants to ANNOVAR avinput format.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    logger = setup_logger("magpie_x.write_annovar_input", args.output.parent / "write_annovar_input.log")
    manifest = write_annovar_avinput(args.input, args.output, args.manifest)
    logger.info("Wrote %s ANNOVAR avinput rows to %s", manifest["rows_written"], args.output)


if __name__ == "__main__":
    main()

