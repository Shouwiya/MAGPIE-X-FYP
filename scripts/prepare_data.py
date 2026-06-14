from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.data.annovar_parser import generate_toy_annovar, parse_annovar
from magpie_x.data.clinvar_loader import generate_toy_clinvar, load_clinvar, save_clinvar
from magpie_x.data.dbnsfp_parser import generate_toy_dbnsfp, parse_dbnsfp
from magpie_x.utils.io import ensure_dir, load_yaml, write_dataframe, write_json
from magpie_x.utils.logging import setup_logger
from magpie_x.utils.seed import set_seed


def run(config_path: str | Path = ROOT / "configs" / "data.yaml", output_dir: str | Path | None = None, dry_run: bool | None = None) -> Path:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 13))
    set_seed(seed)
    logger = setup_logger("magpie_x.prepare_data")
    paths = cfg.get("paths", {})
    out = ensure_dir(output_dir or paths.get("processed_dir", ROOT / "outputs" / "toy" / "processed"))
    use_toy = bool(cfg.get("dry_run", True) if dry_run is None else dry_run)

    if use_toy:
        toy_cfg = cfg.get("toy", {})
        clinvar = generate_toy_clinvar(
            n_variants=int(toy_cfg.get("n_variants", 96)),
            n_genes=int(toy_cfg.get("n_genes", 18)),
            pathogenic_fraction=float(toy_cfg.get("pathogenic_fraction", 0.48)),
            seed=seed,
        )
        annovar = generate_toy_annovar(clinvar, seed=seed)
        dbnsfp = generate_toy_dbnsfp(clinvar, seed=seed)
        logger.info("Generated deterministic toy ClinVar/ANNOVAR/dbNSFP tables.")
    else:
        clinvar_path = paths.get("clinvar")
        if not clinvar_path:
            raise ValueError("Real-data mode requires paths.clinvar in configs/data.yaml")
        clinvar = load_clinvar(ROOT / clinvar_path if not Path(clinvar_path).is_absolute() else clinvar_path)
        annovar = parse_annovar(ROOT / paths["annovar"]) if paths.get("annovar") else generate_toy_annovar(clinvar, seed=seed)
        dbnsfp = parse_dbnsfp(ROOT / paths["dbnsfp"]) if paths.get("dbnsfp") else generate_toy_dbnsfp(clinvar, seed=seed)
        logger.info("Loaded real ClinVar table and available annotations. TODO: verify licenses/releases.")

    save_clinvar(clinvar, out / "clinvar.csv")
    write_dataframe(annovar, out / "annovar.csv")
    write_dataframe(dbnsfp, out / "dbnsfp.csv")
    write_json(
        {
            "dry_run": use_toy,
            "n_clinvar": int(len(clinvar)),
            "n_annovar": int(len(annovar)),
            "n_dbnsfp": int(len(dbnsfp)),
            "seed": seed,
        },
        out / "data_manifest.json",
    )
    logger.info("Prepared data in %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MAGPIE_X ClinVar and annotation tables.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Force deterministic toy data generation.")
    parser.add_argument("--real-data", action="store_true", help="Use configured real input paths.")
    args = parser.parse_args()
    dry = True if args.dry_run else False if args.real_data else None
    run(args.config, args.output_dir, dry_run=dry)


if __name__ == "__main__":
    main()

