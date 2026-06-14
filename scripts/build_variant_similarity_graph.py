from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_TABLE = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_gnn"

FORBIDDEN_FALLBACK_FILES = {
    (ROOT / "outputs" / "real" / "processed" / "annovar.csv").resolve(),
    (ROOT / "outputs" / "real" / "processed" / "dbnsfp.csv").resolve(),
}

GRAPH_FEATURE_COLUMNS = [
    "AF",
    "CADD_phred",
    "REVEL_score",
    "SIFT_score",
    "Polyphen2_HDIV_score",
    "Polyphen2_HVAR_score",
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
]

BASE_COLUMNS = [
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "gene",
    "consequence",
]


def check_not_forbidden(path: Path) -> None:
    if path.resolve() in FORBIDDEN_FALLBACK_FILES:
        raise ValueError(f"Refusing to use known fallback file: {path}")


def ensure_variant_id(df: pd.DataFrame) -> pd.DataFrame:
    if "variant_id" in df.columns:
        return df
    required = {"chrom", "pos", "ref", "alt"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Cannot create variant_id; missing columns: {missing}")
    out = df.copy()
    out["variant_id"] = (
        out["chrom"].astype(str)
        + ":"
        + out["pos"].astype(str)
        + ":"
        + out["ref"].astype(str)
        + ">"
        + out["alt"].astype(str)
    )
    return out


def load_feature_table(path: Path) -> pd.DataFrame:
    check_not_forbidden(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")
    wanted = set(BASE_COLUMNS) | set(GRAPH_FEATURE_COLUMNS)
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False, na_values=["", ".", "-", "NA", "nan"])
    df = ensure_variant_id(df)
    if "label" not in df.columns:
        raise ValueError("Feature table must contain label for reference_5k sampling and later metrics.")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    return df.drop_duplicates(subset=["variant_id"]).reset_index(drop=True)


def select_nodes(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.mode == "reference_5k":
        counts = df["label"].value_counts()
        if 0 not in counts or 1 not in counts:
            raise ValueError("reference_5k requires both benign and pathogenic labels.")
        per_class = min(2500, int(counts.loc[0]), int(counts.loc[1]))
        selected = pd.concat(
            [
                df[df["label"] == 0].sample(n=per_class, random_state=args.seed),
                df[df["label"] == 1].sample(n=per_class, random_state=args.seed + 1),
            ],
            ignore_index=True,
        )
        return selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.mode == "sample":
        n = min(args.n, len(df))
        return df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    raise ValueError(f"Unknown mode: {args.mode}")


def add_group_edges(
    edge_set: set[tuple[int, int, str]],
    edge_weights: dict[tuple[int, int, str], float],
    groups: pd.Series,
    edge_type: str,
    cap_per_node: int,
) -> None:
    if cap_per_node <= 0:
        return
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, value in groups.fillna("UNKNOWN").astype(str).items():
        group_to_indices[value].append(int(idx))

    for indices in group_to_indices.values():
        if len(indices) < 2:
            continue
        sorted_indices = sorted(indices)
        n_group = len(sorted_indices)
        for local_pos, source in enumerate(sorted_indices):
            added = 0
            offset = 1
            while added < cap_per_node and offset < n_group:
                target = sorted_indices[(local_pos + offset) % n_group]
                if source != target:
                    key = (source, target, edge_type)
                    edge_set.add(key)
                    edge_weights[key] = 1.0
                    added += 1
                offset += 1


def build_knn_matrix(nodes: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    available = [column for column in GRAPH_FEATURE_COLUMNS if column in nodes.columns]
    if not available:
        raise ValueError("None of the configured graph numeric feature columns are present.")
    numeric = nodes[available].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(numeric)
    imputed = np.nan_to_num(imputed, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = StandardScaler().fit_transform(imputed)
    return scaled.astype(np.float32), available


def add_knn_edges(
    edge_set: set[tuple[int, int, str]],
    edge_weights: dict[tuple[int, int, str], float],
    x: np.ndarray,
    k: int,
    metric: str,
) -> None:
    if k <= 0 or len(x) < 2:
        return
    n_neighbors = min(k + 1, len(x))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(x)
    distances, indices = nn.kneighbors(x)
    for source, (source_distances, source_indices) in enumerate(zip(distances, indices, strict=False)):
        for distance, target in zip(source_distances, source_indices, strict=False):
            if int(target) == source:
                continue
            key = (source, int(target), "knn_similarity")
            edge_set.add(key)
            edge_weights[key] = float(1.0 / (1.0 + float(distance)))


def build_graph(nodes: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    edge_set: set[tuple[int, int, str]] = set()
    edge_weights: dict[tuple[int, int, str], float] = {}

    add_group_edges(edge_set, edge_weights, nodes["gene"], "same_gene", args.same_gene_cap)
    add_group_edges(edge_set, edge_weights, nodes["consequence"], "same_consequence", args.same_consequence_cap)
    x, feature_names = build_knn_matrix(nodes)
    add_knn_edges(edge_set, edge_weights, x, args.knn_k, args.knn_metric)

    if not edge_set:
        raise ValueError("Graph construction produced no edges. Lower caps or provide more nodes.")

    ordered_edges = sorted(edge_set, key=lambda item: (item[2], item[0], item[1]))
    edge_index = np.array([[source, target] for source, target, _ in ordered_edges], dtype=np.int64).T
    edge_type = np.array([edge_type for _, _, edge_type in ordered_edges])
    edge_weight = np.array([edge_weights[(source, target, edge_type)] for source, target, edge_type in ordered_edges], dtype=np.float32)
    return edge_index, edge_type, edge_weight, feature_names


def output_stem(mode: str, n: int) -> str:
    if mode == "reference_5k":
        return "reference_5k"
    return f"sample_n{n}"


def write_outputs(
    nodes: pd.DataFrame,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    edge_weight: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args.mode, len(nodes))
    graph_path = args.output_dir / f"variant_graph_{stem}.npz"
    nodes_path = args.output_dir / f"variant_graph_nodes_{stem}.csv"

    node_out = nodes.copy()
    node_out.insert(0, "node_idx", np.arange(len(node_out), dtype=np.int64))
    node_out.to_csv(nodes_path, index=False)

    np.savez_compressed(
        graph_path,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        node_ids=node_out["variant_id"].astype(str).to_numpy(),
        graph_feature_names=np.array(feature_names),
        mode=np.array([args.mode]),
    )

    edge_counts = Counter(edge_type.tolist())
    report = [
        "# Variant Graph Build Report",
        "",
        f"Input feature table: `{args.feature_table}`",
        f"Mode: `{args.mode}`",
        f"Nodes: {len(node_out)}",
        f"Edges: {edge_index.shape[1]}",
        "",
        "## Edge Counts",
        "",
    ]
    for name in sorted(edge_counts):
        report.append(f"- `{name}`: {edge_counts[name]}")
    report.extend(
        [
            "",
            "## Graph Feature Columns",
            "",
            *[f"- `{name}`" for name in feature_names],
            "",
            "## Safety Notes",
            "",
            "- Labels are saved for later evaluation/visualization but are not used as graph features.",
            "- Same-gene and same-consequence edges are capped per node.",
            "- This script builds a graph only; it does not train a GNN.",
            "",
            f"Graph file: `{graph_path}`",
            f"Node file: `{nodes_path}`",
        ]
    )
    (args.output_dir / "GRAPH_BUILD_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved graph to {graph_path}")
    print(f"Saved nodes to {nodes_path}")
    print(f"Saved report to {args.output_dir / 'GRAPH_BUILD_REPORT.md'}")


def run(args: argparse.Namespace) -> None:
    df = load_feature_table(args.feature_table)
    nodes = select_nodes(df, args)
    edge_index, edge_type, edge_weight, feature_names = build_graph(nodes, args)
    write_outputs(nodes, edge_index, edge_type, edge_weight, feature_names, args)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a capped variant similarity graph for MAGPIE-X GNN experiments."
    )
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["reference_5k", "sample"], default="reference_5k")
    parser.add_argument("--n", type=int, default=20_000, help="Number of rows for --mode sample.")
    parser.add_argument("--same-gene-cap", type=int, default=5)
    parser.add_argument("--same-consequence-cap", type=int, default=3)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--knn-metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
