from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.data.feature_builder import load_feature_bundle
from magpie_x.data.splitters import make_split, save_split
from magpie_x.graph.similarity_graph import build_variant_similarity_graph, load_graph
from magpie_x.train.trainer import train_gat, train_tabular
from magpie_x.utils.io import ensure_dir, load_yaml
from magpie_x.utils.logging import setup_logger
from magpie_x.utils.seed import set_seed


def run(
    data_config_path: str | Path = ROOT / "configs" / "data.yaml",
    model_config_path: str | Path = ROOT / "configs" / "model.yaml",
    train_config_path: str | Path = ROOT / "configs" / "train.yaml",
    architecture: str | None = None,
    feature_dir: str | Path | None = None,
    graph_path: str | Path | None = None,
    output_dir: str | Path | None = None,
):
    data_cfg = load_yaml(data_config_path)
    model_cfg = load_yaml(model_config_path)
    train_cfg = load_yaml(train_config_path)
    seed = int(train_cfg.get("seed", data_cfg.get("seed", 13)))
    set_seed(seed)
    logger = setup_logger("magpie_x.train_model")
    feature_path = Path(feature_dir or data_cfg.get("paths", {}).get("feature_dir", ROOT / "outputs" / "toy" / "features"))
    bundle = load_feature_bundle(feature_path)
    split = make_split(bundle.variants.assign(gnomad_af=bundle.variants.get("gnomad_af", 0.0)), train_cfg.get("split", {}), seed=seed)

    arch = architecture or str(model_cfg.get("model", {}).get("architecture", "gat"))
    out = ensure_dir(output_dir or train_cfg.get("output_dir", ROOT / "outputs" / "toy" / "experiments" / arch))
    save_split(split, out / "splits")
    if arch == "tabular":
        result = train_tabular(bundle, split, model_cfg, train_cfg, out)
    elif arch == "gat":
        graph_file = Path(graph_path) if graph_path else Path(data_cfg.get("paths", {}).get("graph_dir", ROOT / "outputs" / "toy" / "graphs")) / "variant_similarity_graph.npz"
        if graph_file.exists():
            graph = load_graph(graph_file)
        else:
            graph = build_variant_similarity_graph(
                bundle.features,
                bundle.variants,
                feature_names=bundle.graph_feature_names,
                k=int(model_cfg.get("graph", {}).get("k_neighbors", 8)),
            )
        result = train_gat(bundle, split, graph, model_cfg, train_cfg, out)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    logger.info("Training complete. Predictions: %s", result.predictions_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAGPIE_X tabular or GAT model.")
    parser.add_argument("--data-config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model.yaml"))
    parser.add_argument("--train-config", default=str(ROOT / "configs" / "train.yaml"))
    parser.add_argument("--architecture", choices=["tabular", "gat"], default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(args.data_config, args.model_config, args.train_config, args.architecture, args.feature_dir, args.graph_path, args.output_dir)


if __name__ == "__main__":
    main()

