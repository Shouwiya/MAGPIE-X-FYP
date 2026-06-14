"""Training loops for tabular and graph MAGPIE_X models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from magpie_x.data.feature_builder import FeatureBundle
from magpie_x.data.splitters import SplitIndices
from magpie_x.graph.similarity_graph import GraphData
from magpie_x.models.gat_model import GATVariantModel
from magpie_x.models.tabular_encoder import TabularClassifier
from magpie_x.train.callbacks import CheckpointManager, EarlyStopping
from magpie_x.train.losses import binary_pos_weight, multitask_loss
from magpie_x.utils.io import ensure_dir, write_dataframe, write_json


@dataclass
class TrainResult:
    model_path: Path
    predictions_path: Path
    history_path: Path


def _device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _split_name_array(n: int, split: SplitIndices) -> np.ndarray:
    names = np.array(["unused"] * n, dtype=object)
    names[split.train] = "train"
    names[split.val] = "val"
    names[split.test] = "test"
    return names


def _predict(model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        outputs = model(x, edge_index) if edge_index is not None else model(x)
        logits = outputs["logits"].detach().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
    return logits, probs


def _write_predictions(
    bundle: FeatureBundle,
    split: SplitIndices,
    logits: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
) -> Path:
    predictions = pd.DataFrame(
        {
            "variant_id": bundle.variants["variant_id"].astype(str).to_numpy(),
            "gene": bundle.variants.get("gene", pd.Series(["UNKNOWN"] * len(bundle.labels))).astype(str).to_numpy(),
            "label": bundle.labels.to_numpy(dtype=int),
            "split": _split_name_array(len(bundle.labels), split),
            "logit": logits,
            "probability": probs,
        }
    )
    path = output_dir / "predictions.csv"
    write_dataframe(predictions, path)
    return path


def train_tabular(
    bundle: FeatureBundle,
    split: SplitIndices,
    model_config: dict,
    train_config: dict,
    output_dir: str | Path,
) -> TrainResult:
    output_dir = ensure_dir(output_dir)
    device = _device(str(train_config.get("device", "auto")))
    x = torch.as_tensor(bundle.features.to_numpy(dtype=np.float32), device=device)
    y = torch.as_tensor(bundle.labels.to_numpy(dtype=np.float32), device=device)
    phenotype_enabled = bool(model_config.get("phenotype_head", {}).get("enabled", False))
    phenotype_labels = None
    phenotype_groups = None
    if phenotype_enabled and "phenotype_group_id" in bundle.variants.columns:
        phenotype_labels = torch.as_tensor(bundle.variants["phenotype_group_id"].to_numpy(dtype=np.int64), device=device)
        phenotype_groups = int(model_config.get("phenotype_head", {}).get("num_groups", int(phenotype_labels.max().item() + 1)))

    tab_cfg = model_config.get("tabular", {})
    main_cfg = model_config.get("model", {})
    model = TabularClassifier(
        input_dim=x.shape[1],
        hidden_dims=tuple(tab_cfg.get("hidden_dims", [96, 64])),
        dropout=float(main_cfg.get("dropout", 0.15)),
        batch_norm=bool(tab_cfg.get("batch_norm", True)),
        phenotype_groups=phenotype_groups,
    ).to(device)
    opt_cfg = train_config.get("optimization", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )
    train_idx = torch.as_tensor(split.train, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(split.val, dtype=torch.long, device=device)
    pos_weight = binary_pos_weight(y[train_idx]).to(device)
    stopper = EarlyStopping(patience=int(opt_cfg.get("patience", 6)))
    ckpt = CheckpointManager(output_dir / "checkpoints")
    history: list[dict[str, float]] = []
    epochs = int(opt_cfg.get("epochs", 20))
    batch_size = int(opt_cfg.get("batch_size", 32))

    for epoch in range(1, epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        batch_losses: list[float] = []
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            outputs = model(x[idx])
            pheno = phenotype_labels[idx] if phenotype_labels is not None else None
            loss = multitask_loss(
                outputs,
                y[idx],
                phenotype_labels=pheno,
                phenotype_weight=float(train_config.get("loss", {}).get("phenotype_weight", 0.2)),
                pos_weight=pos_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg.get("gradient_clip_norm", 5.0)))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_outputs = model(x[val_idx])
            val_pheno = phenotype_labels[val_idx] if phenotype_labels is not None else None
            val_loss = multitask_loss(val_outputs, y[val_idx], val_pheno, pos_weight=pos_weight)
        record = {"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "val_loss": float(val_loss.detach().cpu())}
        history.append(record)
        if record["val_loss"] <= stopper.best_loss:
            ckpt.save_best(model, {"epoch": epoch, "val_loss": record["val_loss"], "architecture": "tabular"})
        if stopper.step(record["val_loss"]):
            break

    ckpt.load_best(model)
    logits, probs = _predict(model, x)
    predictions_path = _write_predictions(bundle, split, logits, probs, output_dir)
    history_path = output_dir / "history.csv"
    write_dataframe(pd.DataFrame(history), history_path)
    write_json({"architecture": "tabular", "n_features": int(x.shape[1])}, output_dir / "run_metadata.json")
    return TrainResult(model_path=ckpt.best_path, predictions_path=predictions_path, history_path=history_path)


def train_gat(
    bundle: FeatureBundle,
    split: SplitIndices,
    graph: GraphData,
    model_config: dict,
    train_config: dict,
    output_dir: str | Path,
) -> TrainResult:
    output_dir = ensure_dir(output_dir)
    device = _device(str(train_config.get("device", "auto")))
    x = torch.as_tensor(bundle.features.to_numpy(dtype=np.float32), device=device)
    y = torch.as_tensor(bundle.labels.to_numpy(dtype=np.float32), device=device)
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)

    phenotype_enabled = bool(model_config.get("phenotype_head", {}).get("enabled", False))
    phenotype_labels = None
    phenotype_groups = None
    if phenotype_enabled and "phenotype_group_id" in bundle.variants.columns:
        phenotype_labels = torch.as_tensor(bundle.variants["phenotype_group_id"].to_numpy(dtype=np.int64), device=device)
        phenotype_groups = int(model_config.get("phenotype_head", {}).get("num_groups", int(phenotype_labels.max().item() + 1)))

    main_cfg = model_config.get("model", {})
    graph_cfg = model_config.get("graph", {})
    model = GATVariantModel(
        input_dim=x.shape[1],
        hidden_dim=int(main_cfg.get("hidden_dim", 64)),
        num_layers=int(main_cfg.get("num_layers", 2)),
        heads=int(graph_cfg.get("gat_heads", 4)),
        dropout=float(main_cfg.get("dropout", 0.15)),
        residual=bool(graph_cfg.get("residual", True)),
        jumping_knowledge=str(graph_cfg.get("jumping_knowledge", "concat")),
        phenotype_groups=phenotype_groups,
    ).to(device)
    opt_cfg = train_config.get("optimization", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )
    train_idx = torch.as_tensor(split.train, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(split.val, dtype=torch.long, device=device)
    pos_weight = binary_pos_weight(y[train_idx]).to(device)
    stopper = EarlyStopping(patience=int(opt_cfg.get("patience", 6)))
    ckpt = CheckpointManager(output_dir / "checkpoints")
    history: list[dict[str, float]] = []

    for epoch in range(1, int(opt_cfg.get("epochs", 20)) + 1):
        model.train()
        optimizer.zero_grad()
        outputs_all = model(x, edge_index)
        outputs_train = {key: value[train_idx] for key, value in outputs_all.items() if key != "embedding"}
        pheno = phenotype_labels[train_idx] if phenotype_labels is not None else None
        loss = multitask_loss(
            outputs_train,
            y[train_idx],
            phenotype_labels=pheno,
            phenotype_weight=float(train_config.get("loss", {}).get("phenotype_weight", 0.2)),
            pos_weight=pos_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg.get("gradient_clip_norm", 5.0)))
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs_all = model(x, edge_index)
            val_outputs = {key: value[val_idx] for key, value in val_outputs_all.items() if key != "embedding"}
            val_pheno = phenotype_labels[val_idx] if phenotype_labels is not None else None
            val_loss = multitask_loss(val_outputs, y[val_idx], phenotype_labels=val_pheno, pos_weight=pos_weight)
        record = {"epoch": epoch, "train_loss": float(loss.detach().cpu()), "val_loss": float(val_loss.detach().cpu())}
        history.append(record)
        if record["val_loss"] <= stopper.best_loss:
            ckpt.save_best(model, {"epoch": epoch, "val_loss": record["val_loss"], "architecture": "gat"})
        if stopper.step(record["val_loss"]):
            break

    ckpt.load_best(model)
    logits, probs = _predict(model, x, edge_index=edge_index)
    predictions_path = _write_predictions(bundle, split, logits, probs, output_dir)
    history_path = output_dir / "history.csv"
    write_dataframe(pd.DataFrame(history), history_path)
    write_json(
        {"architecture": "gat", "n_features": int(x.shape[1]), "n_edges": int(graph.edge_index.shape[1])},
        output_dir / "run_metadata.json",
    )
    return TrainResult(model_path=ckpt.best_path, predictions_path=predictions_path, history_path=history_path)

