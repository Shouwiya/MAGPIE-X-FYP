from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_TABLE = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "variant_gene_graph_scaffold"

FORBIDDEN_FALLBACK_FILES = {
    (ROOT / "outputs" / "real" / "processed" / "annovar.csv").resolve(),
    (ROOT / "outputs" / "real" / "processed" / "dbnsfp.csv").resolve(),
}


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


def load_variant_nodes(feature_table: Path, max_variants: int | None) -> pd.DataFrame:
    check_not_forbidden(feature_table)
    if not feature_table.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table}")
    wanted = {"variant_id", "chrom", "pos", "ref", "alt", "gene", "consequence", "AF", "CADD_phred", "REVEL_score"}
    df = pd.read_csv(feature_table, usecols=lambda c: c in wanted, low_memory=False)
    df = ensure_variant_id(df)
    if "gene" not in df.columns:
        raise ValueError("Feature table must contain a gene column for variant-gene graph construction.")
    if max_variants is not None:
        df = df.head(max_variants)
    return df.drop_duplicates(subset=["variant_id"]).reset_index(drop=True)


def build_variant_gene_edges(variants: pd.DataFrame) -> pd.DataFrame:
    edges = variants[["variant_id", "gene"]].copy()
    edges["gene"] = edges["gene"].fillna("UNKNOWN").astype(str)
    edges = edges.rename(columns={"variant_id": "source", "gene": "target"})
    edges["edge_type"] = "variant_to_gene"
    return edges[["source", "target", "edge_type"]]


def build_same_gene_variant_edges(variants: pd.DataFrame, max_edges_per_gene: int) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    grouped = variants[["variant_id", "gene"]].dropna().groupby("gene", sort=False)
    for gene, group in grouped:
        ids = group["variant_id"].astype(str).tolist()
        if len(ids) < 2:
            continue
        edge_count = 0
        for i, source in enumerate(ids):
            for target in ids[i + 1 :]:
                rows.append({"source": source, "target": target, "edge_type": "same_gene_variant"})
                rows.append({"source": target, "target": source, "edge_type": "same_gene_variant"})
                edge_count += 2
                if edge_count >= max_edges_per_gene:
                    break
            if edge_count >= max_edges_per_gene:
                break
    return pd.DataFrame(rows, columns=["source", "target", "edge_type"])


def write_schema_only(output_dir: Path, variants: pd.DataFrame, args: argparse.Namespace) -> None:
    genes = variants["gene"].fillna("UNKNOWN").astype(str).nunique()
    payload = {
        "status": "schema_only",
        "feature_table": str(args.feature_table),
        "variant_rows_seen": int(len(variants)),
        "gene_nodes_seen": int(genes),
        "planned_node_types": ["variant", "gene"],
        "planned_edge_types": ["variant_to_gene", "same_gene_variant", "future_gene_to_gene"],
        "phase": "Phase 2 scaffold only; no GNN training is performed by this script.",
        "future_sources": ["STRING or another trusted gene-gene source", "graph-context residual features"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "variant_gene_graph_schema.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = load_variant_nodes(args.feature_table, args.max_variants)

    if args.schema_only:
        write_schema_only(args.output_dir, variants, args)
        print(f"Saved graph scaffold schema to {args.output_dir / 'variant_gene_graph_schema.json'}")
        return

    variant_nodes = variants.copy()
    gene_nodes = pd.DataFrame({"gene": sorted(variant_nodes["gene"].fillna("UNKNOWN").astype(str).unique())})
    variant_gene_edges = build_variant_gene_edges(variant_nodes)
    same_gene_edges = build_same_gene_variant_edges(variant_nodes, args.max_edges_per_gene)
    edges = pd.concat([variant_gene_edges, same_gene_edges], ignore_index=True)

    variant_nodes.to_csv(args.output_dir / "variant_nodes.csv", index=False)
    gene_nodes.to_csv(args.output_dir / "gene_nodes.csv", index=False)
    edges.to_csv(args.output_dir / "edges.csv", index=False)
    manifest = {
        "status": "edge_lists_written",
        "feature_table": str(args.feature_table),
        "variant_nodes": int(len(variant_nodes)),
        "gene_nodes": int(len(gene_nodes)),
        "edges": int(len(edges)),
        "note": "This script builds graph input scaffolds only and does not train a GNN.",
    }
    (args.output_dir / "variant_gene_graph_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved graph scaffold edge lists to {args.output_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold a VEGN/GNN-MAP-inspired variant-gene graph. This does not train a GNN."
    )
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-variants", type=int, default=5000, help="Limit rows for scaffold edge-list creation.")
    parser.add_argument("--max-edges-per-gene", type=int, default=200)
    parser.add_argument(
        "--schema-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default: write only the graph schema/manifest. Use --no-schema-only to write small edge-list scaffolds.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
