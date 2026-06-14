from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.data.feature_builder import load_feature_bundle
from magpie_x.graph.hetero_graph import build_variant_gene_hetero_graph
from magpie_x.graph.similarity_graph import build_variant_similarity_graph, save_graph
from magpie_x.utils.io import ensure_dir, load_yaml, write_json
from magpie_x.utils.logging import setup_logger


def run(
    data_config_path: str | Path = ROOT / "configs" / "data.yaml",
    model_config_path: str | Path = ROOT / "configs" / "model.yaml",
    feature_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    data_cfg = load_yaml(data_config_path)
    model_cfg = load_yaml(model_config_path)
    logger = setup_logger("magpie_x.build_graph")
    feature_path = Path(feature_dir or data_cfg.get("paths", {}).get("feature_dir", ROOT / "outputs" / "toy" / "features"))
    out = ensure_dir(output_dir or data_cfg.get("paths", {}).get("graph_dir", ROOT / "outputs" / "toy" / "graphs"))
    bundle = load_feature_bundle(feature_path)
    graph_cfg = model_cfg.get("graph", {})
    graph = build_variant_similarity_graph(
        bundle.features,
        bundle.variants,
        feature_names=bundle.graph_feature_names,
        k=int(graph_cfg.get("k_neighbors", 8)),
        metric=str(graph_cfg.get("metric", "cosine")),
        self_loops=bool(graph_cfg.get("self_loops", True)),
    )
    graph_path = save_graph(graph, out)
    if bool(graph_cfg.get("hetero_graph", False)):
        hetero = build_variant_gene_hetero_graph(bundle.variants)
        write_json(
            {
                "n_variant_nodes": len(hetero.variant_ids),
                "n_gene_nodes": len(hetero.gene_ids),
                "n_variant_gene_edges": int(hetero.variant_to_gene.shape[1]),
            },
            out / "hetero_graph_summary.json",
        )
    logger.info("Built graph with %d nodes and %d edges at %s", len(graph.node_ids), graph.edge_index.shape[1], graph_path)
    return graph_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MAGPIE_X variant similarity graph.")
    parser.add_argument("--data-config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model.yaml"))
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(args.data_config, args.model_config, args.feature_dir, args.output_dir)


if __name__ == "__main__":
    main()

