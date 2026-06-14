from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.data.feature_builder import build_feature_bundle, save_feature_bundle
from magpie_x.utils.io import ensure_dir, load_yaml, read_table
from magpie_x.utils.logging import setup_logger


def run(
    config_path: str | Path = ROOT / "configs" / "data.yaml",
    processed_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    cfg = load_yaml(config_path)
    logger = setup_logger("magpie_x.build_features")
    paths = cfg.get("paths", {})
    processed = Path(processed_dir or paths.get("processed_dir", ROOT / "outputs" / "toy" / "processed"))
    out = ensure_dir(output_dir or paths.get("feature_dir", ROOT / "outputs" / "toy" / "features"))

    clinvar = read_table(processed / "clinvar.csv")
    annovar = read_table(processed / "annovar.csv")
    dbnsfp = read_table(processed / "dbnsfp.csv")
    bundle = build_feature_bundle(clinvar, annovar=annovar, dbnsfp=dbnsfp, config=cfg.get("features", {}))
    save_feature_bundle(bundle, out)
    logger.info("Built %d features for %d variants in %s", len(bundle.feature_names), len(bundle.labels), out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MAGPIE_X model features and feature manifest.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(args.config, args.processed_dir, args.output_dir)


if __name__ == "__main__":
    main()

