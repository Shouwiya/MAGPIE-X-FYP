from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.utils.io import ensure_dir

DEFAULT_INPUT = ROOT / "data" / "external" / "humsavar" / "humsavar.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "swissprot"

CATEGORY_TO_LABEL = {
    "LP/P": 1,
    "LB/B": 0,
}
EXCLUDED_CATEGORIES = {"US"}


def parse_humsavar_line(line: str, line_number: int, source_file: Path) -> dict[str, object] | None:
    parts = line.rstrip("\n").split(maxsplit=6)
    if len(parts) < 6:
        return None
    gene, accession, ft_id, aa_change, category, dbsnp = parts[:6]
    disease_name = parts[6].strip() if len(parts) > 6 else "-"
    if not ft_id.startswith("VAR_") or not aa_change.startswith("p."):
        return None
    if category not in CATEGORY_TO_LABEL and category not in EXCLUDED_CATEGORIES:
        return None

    rsid = "" if dbsnp == "-" else dbsnp
    rsid_numeric = rsid[2:] if rsid.lower().startswith("rs") else rsid
    label = CATEGORY_TO_LABEL.get(category)
    return {
        "humsavar_line": int(line_number),
        "uniprot_accession": accession,
        "gene": gene,
        "ft_id": ft_id,
        "aa_change": aa_change,
        "variant_category": category,
        "classification": "pathogenic_or_likely_pathogenic"
        if label == 1
        else "benign_or_likely_benign"
        if label == 0
        else "uncertain_significance",
        "dbsnp_rsid": rsid,
        "dbsnp_rsid_numeric": rsid_numeric,
        "disease_name": disease_name,
        "label": label,
        "include_for_supervised_external_validation": bool(label in {0, 1}),
        "source_file": str(source_file),
    }


def parse_humsavar(input_path: Path, output_dir: Path) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(f"HUMSAVAR file not found: {input_path}")
    if "toy" in {part.lower() for part in input_path.resolve().parts}:
        raise ValueError(f"Refusing to parse toy/synthetic HUMSAVAR path: {input_path}")

    ensure_dir(output_dir)
    parsed_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    data_lines_seen = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = parse_humsavar_line(line, line_number, input_path)
            if row is None:
                continue
            data_lines_seen += 1
            category = str(row["variant_category"])
            category_counts[category] += 1
            if row["include_for_supervised_external_validation"]:
                parsed_rows.append(row)
                label_counts[str(int(row["label"]))] += 1
            else:
                excluded_rows.append(row)

    parsed = pd.DataFrame(parsed_rows)
    excluded = pd.DataFrame(excluded_rows)
    parsed_path = output_dir / "humsavar_parsed.csv"
    excluded_path = output_dir / "humsavar_excluded_uncertain.csv"
    parsed.to_csv(parsed_path, index=False)
    excluded.to_csv(excluded_path, index=False)

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_csv": str(parsed_path),
        "excluded_uncertain_csv": str(excluded_path),
        "data_lines_seen": int(data_lines_seen),
        "supervised_rows_written": int(len(parsed)),
        "excluded_uncertain_rows": int(len(excluded)),
        "category_counts": dict(category_counts),
        "label_counts": dict(label_counts),
        "rows_with_rsid": int(parsed["dbsnp_rsid"].astype(str).ne("").sum()) if not parsed.empty else 0,
        "rows_without_rsid": int(parsed["dbsnp_rsid"].astype(str).eq("").sum()) if not parsed.empty else 0,
        "label_rule": {"LP/P": 1, "LB/B": 0, "US": "excluded"},
        "synthetic_or_toy_data_used": False,
    }
    manifest_path = output_dir / "humsavar_parse_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parse the real UniProt HUMSAVAR file for SwissProt validation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    manifest = parse_humsavar(args.input, args.output_dir)
    print(f"Parsed HUMSAVAR rows: {manifest['supervised_rows_written']:,}")
    print(f"Excluded uncertain rows: {manifest['excluded_uncertain_rows']:,}")
    print(f"Wrote {manifest['output_csv']}")


if __name__ == "__main__":
    main()
