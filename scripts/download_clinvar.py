from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.utils.io import ensure_dir
from magpie_x.utils.logging import setup_logger

CLINVAR_GRCH38_VCF_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
VARIANT_SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "clinvar"


def download(url: str, output_path: Path, *, force: bool = False, min_bytes: int = 1_000_000) -> dict[str, object]:
    ensure_dir(output_path.parent)
    if output_path.exists() and not force:
        size = output_path.stat().st_size
        return {"path": str(output_path), "downloaded": False, "bytes": size, "reason": "exists"}

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    started = time.time()
    urllib.request.urlretrieve(url, tmp_path)
    size = tmp_path.stat().st_size
    if size < min_bytes:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is unexpectedly small ({size} bytes): {url}")
    tmp_path.replace(output_path)
    return {
        "path": str(output_path),
        "downloaded": True,
        "bytes": size,
        "seconds": round(time.time() - started, 2),
        "reason": "downloaded",
    }


def run(output_dir: Path, *, variant_summary: bool = False, force: bool = False) -> list[dict[str, object]]:
    logger = setup_logger("magpie_x.download_clinvar")
    ensure_dir(output_dir)
    results = [
        download(CLINVAR_GRCH38_VCF_URL, output_dir / "clinvar.vcf.gz", force=force, min_bytes=10_000_000)
    ]
    if variant_summary:
        results.append(
            download(VARIANT_SUMMARY_URL, output_dir / "variant_summary.txt.gz", force=force, min_bytes=10_000_000)
        )
    for result in results:
        logger.info("%s | downloaded=%s | bytes=%s", result["path"], result["downloaded"], result["bytes"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public ClinVar GRCh38 data into the real-data raw folder.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant-summary", action="store_true", help="Also download variant_summary.txt.gz.")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist.")
    args = parser.parse_args()
    run(args.output_dir, variant_summary=args.variant_summary, force=args.force)


if __name__ == "__main__":
    main()

