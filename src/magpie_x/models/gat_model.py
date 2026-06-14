"""Residual GAT with jumping knowledge for variant similarity graphs."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    device = edge_index.device
    loops = torch.arange(num_nodes, device=device, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index.long(), loops], dim=1)


class SimpleGATLayer(nn.Module):
    """Small pure-PyTorch GAT layer used when PyG is unavailable.

    The implementation favors portability and testability. For million-node
    production graphs, install PyTorch Geometric and swap in its optimized GATConv.
    """

    def __init__(self, input_dim: int, output_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if output_dim % heads != 0:
            raise ValueError("output_dim must be divisible by heads")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.head_dim = output_dim // heads
        self.lin = nn.Linear(input_dim, output_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def _softmax_by_dst(self, scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        alphas = []
        for head in range(scores.shape[1]):
            s = scores[:, head]
            max_per_dst = torch.full((num_nodes,), -torch.inf, device=s.device, dtype=s.dtype)
            if hasattr(max_per_dst, "scatter_reduce_"):
                max_per_dst.scatter_reduce_(0, dst, s, reduce="amax", include_self=True)
            else:
                for i in range(dst.numel()):
                    max_per_dst[dst[i]] = torch.maximum(max_per_dst[dst[i]], s[i])
            exp_s = torch.exp(s - max_per_dst[dst])
            denom = torch.zeros((num_nodes,), device=s.device, dtype=s.dtype)
            denom.index_add_(0, dst, exp_s)
            alphas.append(exp_s / (denom[dst] + 1e-12))
        return torch.stack(alphas, dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        src = edge_index[0].long()
        dst = edge_index[1].long()
        h = self.lin(x).view(num_nodes, self.heads, self.head_dim)
        scores = (h[src] * self.att_src).sum(-1) + (h[dst] * self.att_dst).sum(-1)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        alpha = self.dropout(self._softmax_by_dst(scores, dst, num_nodes))
        out = torch.zeros((num_nodes, self.heads, self.head_dim), device=x.device, dtype=x.dtype)
        for head in range(self.heads):
            messages = h[src, head, :] * alpha[:, head].unsqueeze(-1)
            out[:, head, :].index_add_(0, dst, messages)
        return out.reshape(num_nodes, self.output_dim) + self.bias


class GATVariantModel(nn.Module):
    """Graph attention network for MAGPIE_X variant graphs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.15,
        residual: bool = True,
        jumping_knowledge: str = "concat",
        phenotype_groups: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            hidden_dim = int(math.ceil(hidden_dim / heads) * heads)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([SimpleGATLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.jumping_knowledge = jumping_knowledge
        final_dim = hidden_dim * num_layers if jumping_knowledge == "concat" else hidden_dim
        self.binary_head = nn.Linear(final_dim, 1)
        self.phenotype_head = nn.Linear(final_dim, phenotype_groups) if phenotype_groups else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        edge_index = add_self_loops(edge_index, x.shape[0])
        h = F.relu(self.input_proj(x))
        states: list[torch.Tensor] = []
        for layer, norm in zip(self.layers, self.norms, strict=True):
            updated = self.dropout(F.relu(layer(h, edge_index)))
            h = norm(h + updated) if self.residual else norm(updated)
            states.append(h)
        embedding = torch.cat(states, dim=-1) if self.jumping_knowledge == "concat" else states[-1]
        out = {"logits": self.binary_head(embedding).squeeze(-1), "embedding": embedding}
        if self.phenotype_head is not None:
            out["phenotype_logits"] = self.phenotype_head(embedding)
        return out

