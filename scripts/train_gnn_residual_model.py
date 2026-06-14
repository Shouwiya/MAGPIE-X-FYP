from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GNN_DIR = ROOT / "outputs" / "real" / "magpie_x_gnn"
DEFAULT_GRAPH = DEFAULT_GNN_DIR / "variant_graph_reference_5k.npz"
DEFAULT_NODES = DEFAULT_GNN_DIR / "variant_graph_nodes_reference_5k.csv"
DEFAULT_OOF = ROOT / "outputs" / "real" / "magpie_x_residual" / "oof_tabular_predictions.csv"
DEFAULT_ALPHA = ROOT / "outputs" / "real" / "magpie_x_residual" / "alphamissense_features.csv"

NUMERIC_FEATURES = [
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


def require_pyg():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch_geometric.nn import GATConv, SAGEConv
    except ImportError as exc:
        raise ImportError(
            "torch_geometric is required for GNN training. Install the PyG wheel that matches your "
            "PyTorch/CUDA setup, for example: pip install torch-geometric"
        ) from exc
    return torch, nn, functional, SAGEConv, GATConv


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def safe_logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        mask = ((y_prob >= lower) if i == 0 else (y_prob > lower)) & (y_prob <= upper)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    out: dict[str, float] = {}
    for name, fn in {
        "accuracy": lambda: accuracy_score(y_true, y_pred),
        "balanced_accuracy": lambda: balanced_accuracy_score(y_true, y_pred),
        "f1": lambda: f1_score(y_true, y_pred, zero_division=0),
        "auroc": lambda: roc_auc_score(y_true, y_prob),
        "auprc": lambda: average_precision_score(y_true, y_prob),
        "mcc": lambda: matthews_corrcoef(y_true, y_pred),
        "brier": lambda: brier_score_loss(y_true, y_prob),
        "ece": lambda: expected_calibration_error(y_true, y_prob),
    }.items():
        try:
            out[name] = float(fn())
        except Exception:
            out[name] = float("nan")
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_graph(graph_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    graph = np.load(graph_path, allow_pickle=True)
    return graph["edge_index"].astype(np.int64), graph["edge_weight"].astype(np.float32)


def merge_optional_inputs(nodes: pd.DataFrame, alpha_path: Path | None, oof_path: Path | None) -> pd.DataFrame:
    out = nodes.copy()
    if alpha_path and alpha_path.exists():
        alpha = pd.read_csv(alpha_path, low_memory=False)
        keep = [
            column
            for column in ["variant_id", "alphamissense_score", "alphamissense_available"]
            if column in alpha.columns
        ]
        if "variant_id" in keep:
            out = out.merge(alpha[keep].drop_duplicates(subset=["variant_id"]), on="variant_id", how="left")
    if "alphamissense_score" not in out.columns:
        out["alphamissense_score"] = np.nan
    if "alphamissense_available" not in out.columns:
        out["alphamissense_available"] = False
    out["alphamissense_score"] = pd.to_numeric(out["alphamissense_score"], errors="coerce")
    out["alphamissense_available"] = out["alphamissense_available"].fillna(False).astype(bool)

    if oof_path and oof_path.exists():
        oof = pd.read_csv(oof_path, low_memory=False)
        keep = [column for column in ["variant_id", "p_tabular", "logit_tabular"] if column in oof.columns]
        if "variant_id" in keep:
            out = out.merge(oof[keep].drop_duplicates(subset=["variant_id"]), on="variant_id", how="left")
    return out


def build_node_features(nodes: pd.DataFrame, include_tabular: bool) -> tuple[np.ndarray, list[str]]:
    features = [column for column in NUMERIC_FEATURES if column in nodes.columns]
    numeric = nodes[features].apply(pd.to_numeric, errors="coerce") if features else pd.DataFrame(index=nodes.index)
    numeric["alphamissense_score"] = pd.to_numeric(nodes.get("alphamissense_score"), errors="coerce")
    numeric["alphamissense_available"] = nodes.get("alphamissense_available", False).fillna(False).astype(int)
    if include_tabular:
        for column in ["p_tabular", "logit_tabular"]:
            if column in nodes.columns:
                numeric[column] = pd.to_numeric(nodes[column], errors="coerce")

    imputed = SimpleImputer(strategy="median").fit_transform(numeric)
    imputed = np.nan_to_num(imputed, nan=0.0, posinf=0.0, neginf=0.0)
    scaled_numeric = StandardScaler().fit_transform(imputed)
    consequence = pd.get_dummies(nodes.get("consequence", pd.Series(["unknown"] * len(nodes))).fillna("unknown"), prefix="consequence")
    x = np.hstack([scaled_numeric, consequence.to_numpy(dtype=np.float32)]).astype(np.float32)
    names = numeric.columns.tolist() + consequence.columns.tolist()
    return x, names


def split_masks(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed + 1, stratify=labels[temp_idx])
    return train_idx, val_idx, test_idx


def make_model(torch, nn, functional, SAGEConv, GATConv, conv: str, in_dim: int, hidden_dim: int, dropout: float):
    class GNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = dropout
            if conv == "gat":
                self.conv1 = GATConv(in_dim, hidden_dim, heads=2, concat=False)
                self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
            else:
                self.conv1 = SAGEConv(in_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            x = functional.relu(x)
            return self.head(x).squeeze(-1)

    return GNNModel()


def train(args: argparse.Namespace) -> None:
    torch, nn, functional, SAGEConv, GATConv = require_pyg()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.nodes.exists():
        raise FileNotFoundError(f"Node CSV not found: {args.nodes}")
    nodes = pd.read_csv(args.nodes, low_memory=False)
    edge_index_np, _ = load_graph(args.graph)
    nodes = merge_optional_inputs(nodes, args.alphamissense_features, args.oof_predictions)

    if "label" not in nodes.columns:
        raise ValueError("Node CSV must contain label for supervised GNN training.")
    labels = pd.to_numeric(nodes["label"], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("Labels must be binary 0/1.")
    if args.mode == "residual" and "logit_tabular" not in nodes.columns:
        raise ValueError("Residual mode requires OOF logit_tabular values. Run tabular OOF first.")

    include_tabular = args.mode == "residual"
    x_np, feature_names = build_node_features(nodes, include_tabular=include_tabular)
    y_np = labels.to_numpy(dtype=np.float32)
    train_idx, val_idx, test_idx = split_masks(y_np.astype(int), args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    train_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_tensor = torch.tensor(val_idx, dtype=torch.long, device=device)

    model = make_model(torch, nn, functional, SAGEConv, GATConv, args.conv, x.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    z_tabular = None
    if args.mode == "residual":
        z_values = pd.to_numeric(nodes["logit_tabular"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(z_values).any():
            if "p_tabular" not in nodes.columns:
                raise ValueError("Residual mode found missing logit_tabular values and no p_tabular fallback.")
            z_values = np.where(np.isnan(z_values), safe_logit(pd.to_numeric(nodes["p_tabular"], errors="coerce").to_numpy()), z_values)
        z_tabular = torch.tensor(z_values, dtype=torch.float32, device=device)

    best_state = None
    best_val_loss = float("inf")
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        raw = model(x, edge_index)
        logits = z_tabular + raw if z_tabular is not None else raw
        loss = loss_fn(logits[train_tensor], y[train_tensor])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            raw_val = model(x, edge_index)
            logits_val = z_tabular + raw_val if z_tabular is not None else raw_val
            val_loss = float(loss_fn(logits_val[val_tensor], y[val_tensor]).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw = model(x, edge_index)
        logits = z_tabular + raw if z_tabular is not None else raw
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        raw_np = raw.detach().cpu().numpy()

    split = np.array(["train"] * len(nodes), dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    predictions = pd.DataFrame(
        {
            "variant_id": nodes["variant_id"].astype(str),
            "label": y_np.astype(int),
            "split": split,
            "mode": args.mode,
            "gnn_raw_output": raw_np,
            "p_final": probs,
        }
    )
    if args.mode == "gnn_only":
        predictions["p_gnn"] = probs
    else:
        predictions["logit_tabular"] = pd.to_numeric(nodes["logit_tabular"], errors="coerce")
        predictions["delta_gnn"] = raw_np
        predictions["z_final"] = predictions["logit_tabular"] + predictions["delta_gnn"]
    predictions_path = args.output_dir / "gnn_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    metric_rows = []
    for split_name in ["train", "val", "test"]:
        mask = (predictions["split"] == split_name).to_numpy()
        metric_rows.append({"model": f"gnn_{args.mode}", "split": split_name, "n": int(mask.sum()), **compute_metrics(y_np[mask], probs[mask])})
    metrics = pd.DataFrame(metric_rows)
    metrics_path = args.output_dir / "gnn_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    report = [
        "# GNN Residual Report",
        "",
        f"Mode: `{args.mode}`",
        f"Graph: `{args.graph}`",
        f"Nodes: `{args.nodes}`",
        f"Predictions: `{predictions_path}`",
        f"Metrics: `{metrics_path}`",
        "",
        "This script trains a lightweight PyTorch Geometric GNN only when explicitly run. It does not replace missing PyG with a non-GNN model.",
        "",
        "## Node Features",
        "",
        *[f"- `{name}`" for name in feature_names],
    ]
    (args.output_dir / "GNN_RESIDUAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved GNN predictions to {predictions_path}")
    print(f"Saved GNN metrics to {metrics_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight PyTorch Geometric GNN on the reference 5k graph only after explicit approval."
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--alphamissense-features", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GNN_DIR)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gnn-only", dest="mode", action="store_const", const="gnn_only")
    mode_group.add_argument("--residual", dest="mode", action="store_const", const="residual")
    parser.set_defaults(mode="residual")
    parser.add_argument("--conv", choices=["sage", "gat"], default="sage")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main(sys.argv[1:])
