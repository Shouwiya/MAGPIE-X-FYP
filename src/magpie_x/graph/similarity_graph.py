"""Variant-variant similarity graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from magpie_x.utils.io import ensure_dir


@dataclass
class GraphData:
    edge_index: np.ndarray
    edge_weight: np.ndarray
    node_ids: list[str]
    graph_feature_names: list[str]


def build_variant_similarity_graph(
    features: pd.DataFrame,
    variants: pd.DataFrame,
    feature_names: list[str] | None = None,
    k: int = 8,
    metric: str = "cosine",
    self_loops: bool = True,
) -> GraphData:
    """Build a k-nearest-neighbor graph among variants.

    Edges are directed from neighbor to query node so message passing aggregates
    context into each variant.
    """

    node_ids = variants["variant_id"].astype(str).tolist() if "variant_id" in variants.columns else [str(i) for i in range(len(features))]
    if feature_names:
        available = [name for name in feature_names if name in features.columns]
    else:
        available = features.columns.tolist()
    if not available:
        available = features.columns.tolist()
    x = features[available].to_numpy(dtype=float)
    n_nodes = x.shape[0]
    if n_nodes == 0:
        raise ValueError("Cannot build a graph with zero variants.")
    if n_nodes == 1:
        return GraphData(edge_index=np.array([[0], [0]], dtype=np.int64), edge_weight=np.array([1.0]), node_ids=node_ids, graph_feature_names=available)

    n_neighbors = min(k + 1, n_nodes)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(x)
    distances, indices = nn.kneighbors(x)

    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    for target in range(n_nodes):
        for distance, source in zip(distances[target], indices[target], strict=False):
            if source == target and not self_loops:
                continue
            if source == target and self_loops:
                weight = 1.0
            else:
                weight = float(1.0 / (1.0 + distance))
            sources.append(int(source))
            targets.append(int(target))
            weights.append(weight)

    edge_index = np.vstack([np.array(sources, dtype=np.int64), np.array(targets, dtype=np.int64)])
    edge_weight = np.array(weights, dtype=np.float32)
    return GraphData(edge_index=edge_index, edge_weight=edge_weight, node_ids=node_ids, graph_feature_names=available)


def save_graph(graph: GraphData, output_dir: str | Path, filename: str = "variant_similarity_graph.npz") -> Path:
    output_dir = ensure_dir(output_dir)
    path = output_dir / filename
    np.savez_compressed(
        path,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        node_ids=np.array(graph.node_ids),
        graph_feature_names=np.array(graph.graph_feature_names),
    )
    return path


def load_graph(path: str | Path) -> GraphData:
    data = np.load(path, allow_pickle=True)
    return GraphData(
        edge_index=data["edge_index"].astype(np.int64),
        edge_weight=data["edge_weight"].astype(np.float32),
        node_ids=[str(x) for x in data["node_ids"].tolist()],
        graph_feature_names=[str(x) for x in data["graph_feature_names"].tolist()],
    )


def to_networkx(graph: GraphData) -> nx.DiGraph:
    g = nx.DiGraph()
    for i, node_id in enumerate(graph.node_ids):
        g.add_node(i, variant_id=node_id)
    for (source, target), weight in zip(graph.edge_index.T, graph.edge_weight, strict=False):
        g.add_edge(int(source), int(target), weight=float(weight))
    return g

