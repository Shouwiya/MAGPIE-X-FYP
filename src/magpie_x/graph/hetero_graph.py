"""Optional heterogeneous variant-gene graph construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HeteroGraphData:
    variant_ids: list[str]
    gene_ids: list[str]
    variant_to_gene: np.ndarray
    gene_to_variant: np.ndarray


def build_variant_gene_hetero_graph(variants: pd.DataFrame) -> HeteroGraphData:
    """Build variant-gene bipartite edges.

    If torch-geometric is installed, downstream users can convert this dataclass
    to HeteroData. Keeping the core representation numpy-based makes dry runs
    and tests independent of optional graph packages.
    """

    variant_ids = variants["variant_id"].astype(str).tolist()
    gene_ids = sorted(variants["gene"].fillna("UNKNOWN").astype(str).unique().tolist())
    gene_to_idx = {gene: i for i, gene in enumerate(gene_ids)}
    sources: list[int] = []
    targets: list[int] = []
    for i, gene in enumerate(variants["gene"].fillna("UNKNOWN").astype(str)):
        sources.append(i)
        targets.append(gene_to_idx[gene])
    v2g = np.vstack([np.array(sources, dtype=np.int64), np.array(targets, dtype=np.int64)])
    g2v = np.vstack([v2g[1], v2g[0]])
    return HeteroGraphData(variant_ids=variant_ids, gene_ids=gene_ids, variant_to_gene=v2g, gene_to_variant=g2v)


def to_pyg_heterodata(hetero: HeteroGraphData):
    """Convert to torch_geometric.data.HeteroData when PyG is installed."""

    try:
        import torch
        from torch_geometric.data import HeteroData
    except Exception as exc:
        raise ImportError("torch_geometric is required for HeteroData conversion.") from exc

    data = HeteroData()
    data["variant"].num_nodes = len(hetero.variant_ids)
    data["gene"].num_nodes = len(hetero.gene_ids)
    data["variant", "maps_to", "gene"].edge_index = torch.as_tensor(hetero.variant_to_gene, dtype=torch.long)
    data["gene", "has_variant", "variant"].edge_index = torch.as_tensor(hetero.gene_to_variant, dtype=torch.long)
    return data

