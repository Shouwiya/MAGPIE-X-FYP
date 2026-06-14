from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.utils.io import ensure_dir
from magpie_x.utils.logging import setup_logger

DEFAULT_INPUT = ROOT / "data" / "interim" / "annovar_input" / "clinvar_exonic_protein_altering.avinput"
DEFAULT_OUT_PREFIX = ROOT / "data" / "interim" / "features" / "clinvar_exonic_protein_altering"


def build_command(args: argparse.Namespace) -> list[str]:
    annovar_dir = Path(args.annovar_dir) if args.annovar_dir else None
    if annovar_dir is None:
        raise FileNotFoundError("ANNOVAR is not configured. Provide --annovar-dir pointing to the directory containing table_annovar.pl.")
    table_annovar = annovar_dir / "table_annovar.pl"
    if not table_annovar.exists():
        raise FileNotFoundError(f"Missing table_annovar.pl at {table_annovar}")
    db_dir = Path(args.db_dir)
    if not db_dir.exists():
        raise FileNotFoundError(f"ANNOVAR humandb directory not found: {db_dir}")
    input_avinput = Path(args.input_avinput)
    if not input_avinput.exists():
        raise FileNotFoundError(f"Input avinput not found: {input_avinput}")
    ensure_dir(Path(args.out_prefix).parent)
    return [
        "perl",
        str(table_annovar),
        str(input_avinput),
        str(db_dir),
        "-buildver",
        args.buildver,
        "-out",
        str(args.out_prefix),
        "-remove",
        "-protocol",
        args.protocol,
        "-operation",
        args.operation,
        "-nastring",
        ".",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run user-supplied ANNOVAR table_annovar.pl on MAGPIE-X avinput.")
    parser.add_argument("--annovar-dir", default=None, help="Directory containing table_annovar.pl. Required to actually run.")
    parser.add_argument("--buildver", default="hg38")
    parser.add_argument("--db-dir", default=str(ROOT / "third_party" / "annovar" / "humandb"))
    parser.add_argument("--protocol", default="refGene")
    parser.add_argument("--operation", default="g")
    parser.add_argument("--input-avinput", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--dry-run", action="store_true", help="Print the ANNOVAR command without running it.")
    args = parser.parse_args()

    logger = setup_logger("magpie_x.run_annovar_wrapper", ROOT / "outputs" / "logs" / "run_annovar_wrapper.log")
    command = build_command(args)
    logger.info("ANNOVAR command: %s", " ".join(command))
    if args.dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

