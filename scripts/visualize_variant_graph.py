from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_gnn"
DEFAULT_GRAPH = DEFAULT_OUTPUT_DIR / "variant_graph_reference_5k.npz"
DEFAULT_NODES = DEFAULT_OUTPUT_DIR / "variant_graph_nodes_reference_5k.csv"

EDGE_COLORS = {
    "same_gene": "#2b6cb0",
    "same_consequence": "#805ad5",
    "knn_similarity": "#718096",
}


def load_graph(graph_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    graph = np.load(graph_path, allow_pickle=True)
    edge_index = graph["edge_index"].astype(np.int64)
    edge_type = graph["edge_type"].astype(str) if "edge_type" in graph.files else np.array(["edge"] * edge_index.shape[1])
    edge_weight = graph["edge_weight"].astype(float) if "edge_weight" in graph.files else np.ones(edge_index.shape[1])
    return edge_index, edge_type, edge_weight


def choose_node_indices(edge_index: np.ndarray, n_nodes: int, sample_size: int, seed: int) -> list[int]:
    if sample_size >= n_nodes:
        return list(range(n_nodes))
    rng = np.random.default_rng(seed)
    adjacency: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
    for source, target in edge_index.T:
        source = int(source)
        target = int(target)
        adjacency[source].add(target)
        adjacency[target].add(source)

    start = int(rng.integers(0, n_nodes))
    selected: list[int] = [start]
    seen = {start}
    cursor = 0
    while cursor < len(selected) and len(selected) < sample_size:
        node = selected[cursor]
        cursor += 1
        neighbors = sorted(adjacency.get(node, []))
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                selected.append(neighbor)
                if len(selected) >= sample_size:
                    break

    if len(selected) < sample_size:
        remaining = [idx for idx in range(n_nodes) if idx not in seen]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])
    return sorted(selected)


def choose_nodes(nodes: pd.DataFrame, edge_index: np.ndarray, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size >= len(nodes):
        return nodes.copy()
    selected_indices = choose_node_indices(edge_index, len(nodes), sample_size, seed)
    return nodes[nodes["node_idx"].astype(int).isin(selected_indices)].sort_values("node_idx").copy()


def node_color_series(nodes: pd.DataFrame, color_by: str) -> list[str]:
    if color_by == "label" and "label" in nodes.columns:
        return ["#2f855a" if int(label) == 0 else "#c53030" for label in nodes["label"]]
    values = nodes.get("consequence", pd.Series(["unknown"] * len(nodes))).fillna("unknown").astype(str)
    palette = [
        "#2b6cb0",
        "#c53030",
        "#2f855a",
        "#b7791f",
        "#805ad5",
        "#319795",
        "#4a5568",
        "#d53f8c",
    ]
    lookup = {value: palette[i % len(palette)] for i, value in enumerate(sorted(values.unique()))}
    return [lookup[value] for value in values]


def build_networkx_sample(
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    edge_weight: np.ndarray,
    nodes: pd.DataFrame,
) -> nx.Graph:
    selected = set(nodes["node_idx"].astype(int).tolist())
    graph = nx.Graph()
    for _, row in nodes.iterrows():
        idx = int(row["node_idx"])
        graph.add_node(
            idx,
            label=str(row.get("variant_id", idx)),
            consequence=str(row.get("consequence", "unknown")),
            y=row.get("label", None),
        )
    for edge_id, (source, target) in enumerate(edge_index.T):
        source = int(source)
        target = int(target)
        if source not in selected or target not in selected:
            continue
        etype = str(edge_type[edge_id])
        weight = float(edge_weight[edge_id])
        if graph.has_edge(source, target):
            graph[source][target]["edge_types"].add(etype)
            graph[source][target]["weight"] = max(graph[source][target]["weight"], weight)
        else:
            graph.add_edge(source, target, edge_types={etype}, weight=weight)
    return graph


def draw_png(graph: nx.Graph, nodes: pd.DataFrame, output_png: Path, color_by: str, seed: int) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(graph, seed=seed, k=0.35, iterations=80)
    node_colors = node_color_series(nodes, color_by)
    nx.draw_networkx_nodes(graph, pos, node_size=46, node_color=node_colors, alpha=0.9, linewidths=0.2, edgecolors="#1a202c")
    for etype, color in EDGE_COLORS.items():
        edges = [(u, v) for u, v, data in graph.edges(data=True) if etype in data.get("edge_types", set())]
        if edges:
            nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=color, alpha=0.35, width=0.8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def draw_html(graph: nx.Graph, output_html: Path, color_by: str) -> bool:
    try:
        from pyvis.network import Network
    except ImportError:
        return False

    output_html.parent.mkdir(parents=True, exist_ok=True)
    net = Network(height="780px", width="100%", bgcolor="#ffffff", font_color="#1a202c")
    for node, data in graph.nodes(data=True):
        if color_by == "label":
            value = data.get("y")
            color = "#2f855a" if str(value) == "0" else "#c53030"
        else:
            color = "#2b6cb0"
        net.add_node(node, label=str(node), title=str(data.get("label", node)), color=color)
    for source, target, data in graph.edges(data=True):
        types = sorted(data.get("edge_types", {"edge"}))
        etype = types[0]
        net.add_edge(source, target, title=", ".join(types), color=EDGE_COLORS.get(etype, "#718096"), value=1)
    net.write_html(str(output_html))
    return True


def run(args: argparse.Namespace) -> None:
    if not args.nodes.exists():
        raise FileNotFoundError(f"Node CSV not found: {args.nodes}")
    edge_index, edge_type, edge_weight = load_graph(args.graph)
    nodes = pd.read_csv(args.nodes, low_memory=False)
    if "node_idx" not in nodes.columns:
        raise ValueError("Node CSV must contain node_idx.")

    selected_nodes = choose_nodes(nodes, edge_index, args.sample_size, args.seed)
    graph = build_networkx_sample(edge_index, edge_type, edge_weight, selected_nodes)
    draw_png(graph, selected_nodes, args.output_png, args.color_by, args.seed)
    print(f"Saved graph PNG to {args.output_png}")

    if args.output_html:
        if draw_html(graph, args.output_html, args.color_by):
            print(f"Saved graph HTML to {args.output_html}")
        else:
            print("pyvis is not installed; skipped HTML graph output.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create small sir-friendly visualizations of a MAGPIE-X variant graph.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_DIR / "graph_sample.png")
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_DIR / "graph_sample.html")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--color-by", choices=["label", "consequence"], default="label")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
