#!/usr/bin/env python3
"""Run simulated horizontal federated learning on real ClinVar-derived samples."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_aa_enhanced_low_circularity_comparison import build_aa_spec
from run_circularity_ablation import build_mode_specs
from run_magpie_low_circularity_baseline import build_improved_low_circularity_spec
from run_magpie_x_5k_enhanced_comparison import extend_spec
from run_magpie_x_5k_full_fusion import sha256_ids
from run_sample_size_benchmark import (
    FeatureSpec,
    METRIC_COLUMNS,
    METRIC_TUPLE_DEFINITION,
    balanced_sample,
    compute_metrics,
    infer_features,
    load_feature_table,
    make_splits,
    normalize_missing_markers,
    result_tuple,
)
from run_spliceai_5k_low_circularity_comparison import (
    SPLICEAI_CATEGORICAL_COLUMNS,
    SPLICEAI_NUMERIC_COLUMNS,
)


DEFAULT_BASE_INPUT = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_AA_INPUT = (
    ROOT
    / "data"
    / "interim"
    / "features"
    / "basic_feature_table_with_clean_aa_features.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "federated"

FEATURE_MODE_A = "FL-A"
FEATURE_MODE_B = "FL-B"
FEATURE_MODE_C = "FL-C"

FEATURE_MODE_NAMES = {
    FEATURE_MODE_A: (
        "MAGPIE-parity low-circularity: ANNOVAR/refGene + gnomAD AF + "
        "conservation + variant type/length + AA/BLOSUM62"
    ),
    FEATURE_MODE_B: (
        "MAGPIE-X circularity-risk: FL-A + 86 dbNSFP prior-predictor fields"
    ),
    FEATURE_MODE_C: (
        "MAGPIE-X circularity-risk: FL-B + 12 exact-matched SpliceAI DS/DP fields"
    ),
}

FEATURE_MODE_RISK = {
    FEATURE_MODE_A: "low-circularity",
    FEATURE_MODE_B: "high circularity risk",
    FEATURE_MODE_C: "high circularity risk",
}

SPLICEAI_SAMPLE_SIZES = {5_000, 10_000, 20_000}
SPLIT_TYPES = ("iid", "non_iid")


@dataclass(frozen=True)
class SampleBundle:
    frame: pd.DataFrame
    specs: dict[str, FeatureSpec]
    y: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    source: str
    ordered_hash: str
    train_hash: str
    test_hash: str


class FederatedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        second_dim = max(16, hidden_dim // 2)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_dim),
            nn.ReLU(),
            nn.Linear(second_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def make_dense_one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_neural_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                spec.numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="MISSING"),
                        ),
                        ("onehot", make_dense_one_hot()),
                    ]
                ),
                spec.categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def probability_from_model(
    model: nn.Module, x: np.ndarray, batch_size: int
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).float()
            output.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(output).astype(float)


def train_neural_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    global_state: dict[str, torch.Tensor] | None = None,
    proximal_mu: float = 0.0,
) -> nn.Module:
    seed_everything(seed)
    dataset = TensorDataset(
        torch.from_numpy(x[indices]).float(),
        torch.from_numpy(y[indices].astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    reference = (
        {key: value.detach().clone() for key, value in global_state.items()}
        if global_state is not None
        else None
    )
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            if reference is not None and proximal_mu > 0:
                proximal = torch.zeros((), dtype=torch.float32)
                for name, parameter in model.named_parameters():
                    proximal = proximal + torch.sum(
                        (parameter - reference[name]) ** 2
                    )
                loss = loss + 0.5 * proximal_mu * proximal
            loss.backward()
            optimizer.step()
    return model


def weighted_average_states(
    states: list[dict[str, torch.Tensor]], weights: list[int]
) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Federated averaging requires positive client weights.")
    averaged: dict[str, torch.Tensor] = {}
    for key in states[0]:
        averaged[key] = sum(
            state[key].detach().cpu() * (weight / total)
            for state, weight in zip(states, weights)
        )
    return averaged


def run_federated_neural(
    *,
    x: np.ndarray,
    y: np.ndarray,
    clients: list[np.ndarray],
    input_dim: int,
    args: argparse.Namespace,
    seed: int,
    proximal_mu: float,
) -> nn.Module:
    seed_everything(seed)
    global_model = FederatedMLP(input_dim, args.hidden_dim, args.dropout)
    for round_number in range(args.rounds):
        global_state = copy.deepcopy(global_model.state_dict())
        local_states: list[dict[str, torch.Tensor]] = []
        local_weights: list[int] = []
        for client_id, client_idx in enumerate(clients):
            local_model = FederatedMLP(input_dim, args.hidden_dim, args.dropout)
            local_model.load_state_dict(global_state)
            train_neural_model(
                local_model,
                x,
                y,
                client_idx,
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=seed + round_number * 100 + client_id,
                global_state=global_state if proximal_mu > 0 else None,
                proximal_mu=proximal_mu,
            )
            local_states.append(copy.deepcopy(local_model.state_dict()))
            local_weights.append(len(client_idx))
        global_model.load_state_dict(
            weighted_average_states(local_states, local_weights)
        )
    return global_model


def average_metric_dicts(rows: list[dict[str, float | int]]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRIC_COLUMNS
    }


def iid_clients(
    train_idx: np.ndarray, y: np.ndarray, client_count: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    clients: list[list[int]] = [[] for _ in range(client_count)]
    for label in (0, 1):
        label_idx = train_idx[y[train_idx] == label].copy()
        rng.shuffle(label_idx)
        for client_id, chunk in enumerate(np.array_split(label_idx, client_count)):
            clients[client_id].extend(chunk.tolist())
    return [np.asarray(sorted(client), dtype=int) for client in clients]


def non_iid_clients(
    train_idx: np.ndarray,
    y: np.ndarray,
    client_count: int,
    seed: int,
    alpha: float,
) -> list[np.ndarray]:
    for attempt in range(200):
        rng = np.random.default_rng(seed + attempt)
        clients: list[list[int]] = [[] for _ in range(client_count)]
        valid = True
        for label in (0, 1):
            label_idx = train_idx[y[train_idx] == label].copy()
            rng.shuffle(label_idx)
            proportions = rng.dirichlet(np.repeat(alpha, client_count))
            raw = proportions * len(label_idx)
            counts = np.floor(raw).astype(int)
            remainder = len(label_idx) - int(counts.sum())
            if remainder:
                order = np.argsort(raw - counts)[::-1]
                counts[order[:remainder]] += 1
            boundaries = np.cumsum(counts)[:-1]
            chunks = np.split(label_idx, boundaries)
            for client_id, chunk in enumerate(chunks):
                clients[client_id].extend(chunk.tolist())
        arrays = [np.asarray(sorted(client), dtype=int) for client in clients]
        min_total = max(20, len(train_idx) // (client_count * 20))
        min_class = max(2, len(train_idx) // (client_count * 500))
        for client in arrays:
            counts = np.bincount(y[client], minlength=2)
            if len(client) < min_total or counts.min() < min_class:
                valid = False
                break
        if valid and sum(len(client) for client in arrays) == len(train_idx):
            return arrays
    raise RuntimeError("Could not construct a stable non-IID client partition.")


def client_partitions(
    train_idx: np.ndarray,
    y: np.ndarray,
    *,
    client_count: int,
    split_type: str,
    seed: int,
    alpha: float,
) -> list[np.ndarray]:
    if split_type == "iid":
        return iid_clients(train_idx, y, client_count, seed)
    if split_type == "non_iid":
        return non_iid_clients(train_idx, y, client_count, seed, alpha)
    raise ValueError(f"Unknown client split type: {split_type}")


def build_sample_bundle(
    *,
    sample_size: int,
    full_frame: pd.DataFrame,
    parity_base_spec: FeatureSpec,
    full_improved_spec: FeatureSpec,
    risky_numeric: list[str],
    risky_categorical: list[str],
    seed: int,
    test_size: float,
) -> SampleBundle:
    reference_path = (
        ROOT
        / "outputs"
        / "real"
        / "magpie_parity"
        / f"reference_{sample_size // 1000}k_aa_spliceai_feature_table.csv"
    )
    if sample_size in SPLICEAI_SAMPLE_SIZES and reference_path.exists():
        source = normalize_missing_markers(
            pd.read_csv(reference_path, low_memory=False)
        )
        if len(source) != sample_size:
            raise ValueError(f"Unexpected row count in {reference_path}.")
        combined, improved_spec, _ = build_improved_low_circularity_spec(
            source, parity_base_spec
        )
        source_name = reference_path.relative_to(ROOT).as_posix()
    else:
        sampled = balanced_sample(
            full_frame,
            sample_size=sample_size,
            seed=seed + sample_size,
        )
        combined = sampled.reset_index(drop=True)
        improved_spec = full_improved_spec
        source_name = (
            "balanced sample from "
            "data/interim/features/basic_feature_table_with_clean_aa_features.csv"
        )

    aa_spec = build_aa_spec(improved_spec)
    enhanced_spec = extend_spec(
        aa_spec,
        numeric=risky_numeric,
        categorical=risky_categorical,
    )
    specs = {
        FEATURE_MODE_A: aa_spec,
        FEATURE_MODE_B: enhanced_spec,
    }
    if sample_size in SPLICEAI_SAMPLE_SIZES:
        specs[FEATURE_MODE_C] = extend_spec(
            enhanced_spec,
            numeric=SPLICEAI_NUMERIC_COLUMNS,
            categorical=SPLICEAI_CATEGORICAL_COLUMNS,
        )

    expected = {
        FEATURE_MODE_A: 114,
        FEATURE_MODE_B: 200,
        FEATURE_MODE_C: 212,
    }
    for mode, spec in specs.items():
        if len(spec.feature_columns) != expected[mode]:
            raise RuntimeError(
                f"{sample_size // 1000}k {mode} has "
                f"{len(spec.feature_columns)} features; expected {expected[mode]}."
            )

    y = pd.to_numeric(combined["label"], errors="raise").to_numpy(dtype=int)
    if len(combined) != sample_size or np.bincount(y, minlength=2).tolist() != [
        sample_size // 2,
        sample_size // 2,
    ]:
        raise ValueError(f"{sample_size:,} sample is not balanced and complete.")
    _, train_idx, test_idx = make_splits(
        y, seed=seed, test_size=test_size, cv_folds=1
    )[0]
    ids = combined["variant_id"].astype(str)
    return SampleBundle(
        frame=combined.reset_index(drop=True),
        specs=specs,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        source=source_name,
        ordered_hash=sha256_ids(ids),
        train_hash=sha256_ids(combined.iloc[train_idx]["variant_id"].astype(str)),
        test_hash=sha256_ids(combined.iloc[test_idx]["variant_id"].astype(str)),
    )


def metric_row(
    *,
    sample_size: int,
    feature_mode: str,
    split_type: str,
    method: str,
    algorithm: str,
    aggregation: str,
    metrics: dict[str, float | int],
    bundle: SampleBundle,
    spec: FeatureSpec,
    transformed_features: int,
    clients: list[np.ndarray],
    args: argparse.Namespace,
    runtime_seconds: float,
    is_true_federated: bool,
    best_local: dict[str, object] | None = None,
) -> dict[str, object]:
    sizes = [int(len(client)) for client in clients]
    distributions = [
        {
            "client_id": client_id + 1,
            "benign": int(np.sum(bundle.y[client] == 0)),
            "pathogenic": int(np.sum(bundle.y[client] == 1)),
        }
        for client_id, client in enumerate(clients)
    ]
    row: dict[str, object] = {
        "sample_size": sample_size,
        "feature_mode": feature_mode,
        "feature_mode_name": FEATURE_MODE_NAMES[feature_mode],
        "circularity_risk": FEATURE_MODE_RISK[feature_mode],
        "client_count": args.clients,
        "client_split_type": split_type,
        "method": method,
        "algorithm": algorithm,
        "aggregation_method": aggregation,
        "true_federated_optimization": "yes" if is_true_federated else "no",
        "federated_rounds": args.rounds if method in {"fedavg", "fedprox"} else 0,
        "local_epochs": args.local_epochs
        if method in {"fedavg", "fedprox"}
        else args.central_epochs,
        "fedprox_mu": args.fedprox_mu if method == "fedprox" else 0.0,
        "n_train": int(len(bundle.train_idx)),
        "global_test_size": int(len(bundle.test_idx)),
        "input_feature_count": int(len(spec.feature_columns)),
        "transformed_feature_count": int(transformed_features),
        "client_train_sizes": json.dumps(sizes),
        "client_label_distribution": json.dumps(distributions, sort_keys=True),
        "sample_source": bundle.source,
        "ordered_variant_id_sha256": bundle.ordered_hash,
        "train_variant_id_sha256": bundle.train_hash,
        "test_variant_id_sha256": bundle.test_hash,
        "swissprot_humsavar_used_in_training": "no",
        "simulation_scope": "single-machine simulated horizontal federated learning",
        "runtime_seconds": runtime_seconds,
        **metrics,
    }
    if best_local is not None:
        row.update(best_local)
    row["result_tuple"] = result_tuple(row)
    return row


def client_audit_rows(
    *,
    sample_size: int,
    feature_mode: str,
    split_type: str,
    clients: list[np.ndarray],
    bundle: SampleBundle,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for client_id, client in enumerate(clients, start=1):
        labels = bundle.y[client]
        genes = (
            bundle.frame.iloc[client]["gene"].fillna("UNKNOWN").astype(str)
            if "gene" in bundle.frame.columns
            else pd.Series(["UNKNOWN"] * len(client))
        )
        consequences = (
            bundle.frame.iloc[client]["consequence"]
            .fillna("UNKNOWN")
            .astype(str)
            if "consequence" in bundle.frame.columns
            else pd.Series(["UNKNOWN"] * len(client))
        )
        top_consequence = consequences.value_counts().index[0]
        top_count = int(consequences.value_counts().iloc[0])
        rows.append(
            {
                "sample_size": sample_size,
                "feature_mode": feature_mode,
                "client_split_type": split_type,
                "client_id": client_id,
                "client_train_size": int(len(client)),
                "benign_count": int(np.sum(labels == 0)),
                "pathogenic_count": int(np.sum(labels == 1)),
                "pathogenic_fraction": float(np.mean(labels)),
                "unique_gene_count": int(genes.nunique()),
                "top_consequence": top_consequence,
                "top_consequence_count": top_count,
                "top_consequence_fraction": top_count / len(client),
                "global_test_size": int(len(bundle.test_idx)),
                "labels_used_for_partitioning": (
                    "yes, only to simulate label-skewed non-IID clients"
                    if split_type == "non_iid"
                    else "yes, only to preserve approximate IID class balance"
                ),
                "test_labels_used_for_training": "no",
            }
        )
    return rows


def evaluate_configuration(
    *,
    sample_size: int,
    feature_mode: str,
    bundle: SampleBundle,
    split_type: str,
    args: argparse.Namespace,
    client_cache: dict[tuple[int, str], list[np.ndarray]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    spec = bundle.specs[feature_mode]
    x_frame = bundle.frame[spec.feature_columns].reset_index(drop=True)
    preprocessor = build_neural_preprocessor(spec)
    x_train = preprocessor.fit_transform(x_frame.iloc[bundle.train_idx])
    x_all = preprocessor.transform(x_frame)
    x_all = np.asarray(x_all, dtype=np.float32)
    if not np.isfinite(x_all).all():
        raise RuntimeError(f"Non-finite transformed values in {sample_size} {feature_mode}.")

    key = (sample_size, split_type)
    if key not in client_cache:
        client_cache[key] = client_partitions(
            bundle.train_idx,
            bundle.y,
            client_count=args.clients,
            split_type=split_type,
            seed=args.seed + sample_size + (0 if split_type == "iid" else 10_000),
            alpha=args.non_iid_alpha,
        )
    clients = client_cache[key]
    rows: list[dict[str, object]] = []
    input_dim = x_all.shape[1]
    common_seed = args.seed + sample_size + ord(feature_mode[-1])

    # Centralized pooled-data upper bound.
    started = time.perf_counter()
    seed_everything(common_seed)
    centralized = FederatedMLP(input_dim, args.hidden_dim, args.dropout)
    train_neural_model(
        centralized,
        x_all,
        bundle.y,
        bundle.train_idx,
        epochs=args.central_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=common_seed,
    )
    p_central = probability_from_model(centralized, x_all[bundle.test_idx], args.batch_size)
    central_metrics = compute_metrics(
        bundle.y[bundle.test_idx], p_central, args.ece_bins
    )
    rows.append(
        metric_row(
            sample_size=sample_size,
            feature_mode=feature_mode,
            split_type=split_type,
            method="centralized",
            algorithm="Centralized pooled-data MLP baseline",
            aggregation="none; pooled training data",
            metrics=central_metrics,
            bundle=bundle,
            spec=spec,
            transformed_features=input_dim,
            clients=clients,
            args=args,
            runtime_seconds=time.perf_counter() - started,
            is_true_federated=False,
        )
    )

    # Local-only client models.
    started = time.perf_counter()
    local_metrics: list[dict[str, float | int]] = []
    for client_id, client in enumerate(clients):
        seed_everything(common_seed + 100 + client_id)
        local_model = FederatedMLP(input_dim, args.hidden_dim, args.dropout)
        train_neural_model(
            local_model,
            x_all,
            bundle.y,
            client,
            epochs=args.central_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=common_seed + 100 + client_id,
        )
        local_prob = probability_from_model(
            local_model, x_all[bundle.test_idx], args.batch_size
        )
        local_metrics.append(
            compute_metrics(bundle.y[bundle.test_idx], local_prob, args.ece_bins)
        )
    local_average = average_metric_dicts(local_metrics)
    best_client_index = max(
        range(len(local_metrics)),
        key=lambda idx: (
            float(local_metrics[idx]["accuracy"]),
            float(local_metrics[idx]["auroc"]),
        ),
    )
    best_metrics = local_metrics[best_client_index]
    best_local = {
        "best_local_client_id": best_client_index + 1,
        "best_local_accuracy": float(best_metrics["accuracy"]),
        "best_local_result_tuple": result_tuple(best_metrics),
    }
    rows.append(
        metric_row(
            sample_size=sample_size,
            feature_mode=feature_mode,
            split_type=split_type,
            method="local_only_average",
            algorithm="Local-only client MLP average",
            aggregation="none; arithmetic mean of five client metric tuples",
            metrics=local_average,
            bundle=bundle,
            spec=spec,
            transformed_features=input_dim,
            clients=clients,
            args=args,
            runtime_seconds=time.perf_counter() - started,
            is_true_federated=False,
            best_local=best_local,
        )
    )

    # True FedAvg.
    started = time.perf_counter()
    fedavg = run_federated_neural(
        x=x_all,
        y=bundle.y,
        clients=clients,
        input_dim=input_dim,
        args=args,
        seed=common_seed + 200,
        proximal_mu=0.0,
    )
    p_fedavg = probability_from_model(
        fedavg, x_all[bundle.test_idx], args.batch_size
    )
    fedavg_metrics = compute_metrics(
        bundle.y[bundle.test_idx], p_fedavg, args.ece_bins
    )
    rows.append(
        metric_row(
            sample_size=sample_size,
            feature_mode=feature_mode,
            split_type=split_type,
            method="fedavg",
            algorithm="FedAvg MLP weighted model averaging",
            aggregation="client-sample-weighted neural-network parameter average",
            metrics=fedavg_metrics,
            bundle=bundle,
            spec=spec,
            transformed_features=input_dim,
            clients=clients,
            args=args,
            runtime_seconds=time.perf_counter() - started,
            is_true_federated=True,
        )
    )

    # True FedProx.
    started = time.perf_counter()
    fedprox = run_federated_neural(
        x=x_all,
        y=bundle.y,
        clients=clients,
        input_dim=input_dim,
        args=args,
        seed=common_seed + 300,
        proximal_mu=args.fedprox_mu,
    )
    p_fedprox = probability_from_model(
        fedprox, x_all[bundle.test_idx], args.batch_size
    )
    fedprox_metrics = compute_metrics(
        bundle.y[bundle.test_idx], p_fedprox, args.ece_bins
    )
    rows.append(
        metric_row(
            sample_size=sample_size,
            feature_mode=feature_mode,
            split_type=split_type,
            method="fedprox",
            algorithm="FedProx MLP proximal federated optimization",
            aggregation="client-sample-weighted neural-network parameter average",
            metrics=fedprox_metrics,
            bundle=bundle,
            spec=spec,
            transformed_features=input_dim,
            clients=clients,
            args=args,
            runtime_seconds=time.perf_counter() - started,
            is_true_federated=True,
        )
    )

    # Federated-style local tree probability ensemble.
    started = time.perf_counter()
    client_probabilities: list[np.ndarray] = []
    weights: list[int] = []
    for client_id, client in enumerate(clients):
        model = LGBMClassifier(
            objective="binary",
            n_estimators=args.ensemble_estimators,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            n_jobs=args.ensemble_jobs,
            random_state=common_seed + 400 + client_id,
            verbose=-1,
        )
        model.fit(x_all[client], bundle.y[client])
        client_probabilities.append(
            model.predict_proba(x_all[bundle.test_idx])[:, 1]
        )
        weights.append(len(client))
    p_ensemble = np.average(
        np.vstack(client_probabilities), axis=0, weights=np.asarray(weights)
    )
    ensemble_metrics = compute_metrics(
        bundle.y[bundle.test_idx], p_ensemble, args.ece_bins
    )
    rows.append(
        metric_row(
            sample_size=sample_size,
            feature_mode=feature_mode,
            split_type=split_type,
            method="federated_ensemble",
            algorithm="Federated ensemble / soft-voting probability aggregation",
            aggregation="client-sample-weighted average of local LightGBM probabilities",
            metrics=ensemble_metrics,
            bundle=bundle,
            spec=spec,
            transformed_features=input_dim,
            clients=clients,
            args=args,
            runtime_seconds=time.perf_counter() - started,
            is_true_federated=False,
        )
    )

    return rows, client_audit_rows(
        sample_size=sample_size,
        feature_mode=feature_mode,
        split_type=split_type,
        clients=clients,
        bundle=bundle,
    )


def markdown_table(frame: pd.DataFrame) -> list[str]:
    headers = frame.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|") for value in row)
            + " |"
        )
    return lines


def write_reports(
    output_dir: Path,
    metrics: pd.DataFrame,
    audit: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    federated = metrics[
        metrics["method"].isin(["fedavg", "fedprox", "federated_ensemble"])
    ].copy()
    best = (
        federated.sort_values(
            ["sample_size", "accuracy", "auroc"],
            ascending=[True, False, False],
        )
        .groupby("sample_size", as_index=False)
        .first()
    )
    report_lines = [
        "# Federated Learning Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
        "## Experimental Setup",
        "",
        "- Real balanced ClinVar-derived samples: 5k, 10k, 15k, 20k, and 25k.",
        "- One stratified 80/20 global train/test split per sample size.",
        "- Five simulated clients partition only the training portion.",
        "- IID clients preserve approximately equal class balance.",
        f"- Non-IID clients use reproducible Dirichlet label skew with alpha={args.non_iid_alpha}.",
        "- SwissProt/HUMSAVAR is not used in training and remains external validation.",
        "- This is single-machine simulated horizontal federated learning, not a privacy or security deployment.",
        "",
        "## Feature Modes",
        "",
        f"- FL-A: {FEATURE_MODE_NAMES[FEATURE_MODE_A]}.",
        f"- FL-B: {FEATURE_MODE_NAMES[FEATURE_MODE_B]}.",
        f"- FL-C: {FEATURE_MODE_NAMES[FEATURE_MODE_C]}; available only at 5k, 10k, and 20k because those exact sidecars already exist.",
        "",
        "## Best Federated Result Per Sample Size",
        "",
    ]
    preview = best[
        [
            "sample_size",
            "feature_mode",
            "client_split_type",
            "algorithm",
            "circularity_risk",
            "result_tuple",
        ]
    ]
    report_lines.extend(markdown_table(preview))
    report_lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- FedAvg and FedProx average neural-network parameters by client sample count.",
            "- The LightGBM method averages predicted probabilities and is not called FedAvg.",
            "- No GraphSAGE federated experiment is claimed.",
            "- No SwissProt/HUMSAVAR labels or variants are used for training.",
            "- Predictor-rich FL-B/FL-C results retain high circularity risk.",
        ]
    )
    (output_dir / "FEDERATED_LEARNING_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    comparison = metrics[
        [
            "sample_size",
            "feature_mode",
            "client_split_type",
            "algorithm",
            "aggregation_method",
            "true_federated_optimization",
            "result_tuple",
        ]
    ].sort_values(
        ["sample_size", "feature_mode", "client_split_type", "algorithm"]
    )
    comparison_lines = [
        "# Federated Algorithm Comparison",
        "",
        "Centralized and local-only rows are controls. FedAvg and FedProx are true "
        "federated neural optimization. The tree method is a federated-style "
        "soft-voting ensemble, not weight averaging.",
        "",
    ]
    comparison_lines.extend(markdown_table(comparison))
    (output_dir / "federated_algorithm_comparison.md").write_text(
        "\n".join(comparison_lines) + "\n", encoding="utf-8"
    )

    limitation_lines = [
        "# Federated Learning Limitations",
        "",
        "- These experiments are simulated federated learning on one machine using partitioned ClinVar data.",
        "- They demonstrate training and aggregation feasibility but do not prove deployment-level privacy, secure aggregation, encrypted transport, authentication, or resistance to model inversion.",
        "- Clients are synthetic partitions rather than real hospitals or laboratories.",
        "- Non-IID heterogeneity is label-skew simulation; it does not reproduce every institutional ascertainment pattern.",
        "- A shared preprocessing transform is fit on the global training partition so all simulated clients use one feature space; a deployment would need fixed encoders or federated aggregation of preprocessing statistics.",
        "- The global test set is a balanced random ClinVar-derived holdout, not temporal, gene-held-out, or external validation.",
        "- FL-B and FL-C include prior predictor fields and therefore have high circularity risk.",
        "- FL-C is limited to 5k, 10k, and 20k because exact SpliceAI sidecars were already available only for those samples.",
        "- SwissProt/HUMSAVAR remains external validation and is not used in client or server training.",
        "- Communication volume, client dropout, adversarial clients, differential privacy, and secure aggregation are not modeled.",
    ]
    (output_dir / "federated_limitations.md").write_text(
        "\n".join(limitation_lines) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.torch_threads))
    seed_everything(args.seed)

    base_df = load_feature_table(args.base_input)
    base_spec = infer_features(base_df, args.max_categorical_cardinality)
    mode_specs, removed_predictors = build_mode_specs(base_spec)
    enhanced_base_spec = mode_specs["magpie_x_enhanced"]
    removed_set = set(removed_predictors)
    risky_numeric = [
        column
        for column in enhanced_base_spec.numeric_columns
        if column in removed_set
    ]
    risky_categorical = [
        column
        for column in enhanced_base_spec.categorical_columns
        if column in removed_set
    ]
    if len(risky_numeric) + len(risky_categorical) != 86:
        raise RuntimeError("Expected exactly 86 prior-predictor fields.")
    parity_base_spec = mode_specs["magpie_parity"]
    del base_df

    full_source = normalize_missing_markers(
        pd.read_csv(args.aa_input, low_memory=False)
    )
    full_frame, full_improved_spec, _ = build_improved_low_circularity_spec(
        full_source, parity_base_spec
    )
    full_aa_spec = build_aa_spec(full_improved_spec)
    if len(full_aa_spec.feature_columns) != 114:
        raise RuntimeError("Full clean AA/BLOSUM feature specification is not 114.")
    del full_source

    metric_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    client_cache: dict[tuple[int, str], list[np.ndarray]] = {}

    for sample_size in args.sample_sizes:
        bundle = build_sample_bundle(
            sample_size=sample_size,
            full_frame=full_frame,
            parity_base_spec=parity_base_spec,
            full_improved_spec=full_improved_spec,
            risky_numeric=risky_numeric,
            risky_categorical=risky_categorical,
            seed=args.seed,
            test_size=args.test_size,
        )
        print(
            f"sample={sample_size:,} source={bundle.source} "
            f"train={len(bundle.train_idx):,} test={len(bundle.test_idx):,}",
            flush=True,
        )
        for feature_mode in bundle.specs:
            for split_type in args.split_types:
                started = time.perf_counter()
                rows, clients = evaluate_configuration(
                    sample_size=sample_size,
                    feature_mode=feature_mode,
                    bundle=bundle,
                    split_type=split_type,
                    args=args,
                    client_cache=client_cache,
                )
                metric_rows.extend(rows)
                audit_rows.extend(clients)
                best_federated = max(
                    (
                        row
                        for row in rows
                        if row["method"]
                        in {"fedavg", "fedprox", "federated_ensemble"}
                    ),
                    key=lambda row: (row["accuracy"], row["auroc"]),
                )
                print(
                    f"  {feature_mode} {split_type}: best="
                    f"{best_federated['method']} accuracy="
                    f"{best_federated['accuracy']:.4f} elapsed="
                    f"{(time.perf_counter() - started):.1f}s",
                    flush=True,
                )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["sample_size", "feature_mode", "client_split_type", "method"]
    )
    audit = pd.DataFrame(audit_rows).sort_values(
        ["sample_size", "feature_mode", "client_split_type", "client_id"]
    )
    metrics_path = (
        args.output_dir / "federated_5k_10k_15k_20k_25k_metrics.csv"
    )
    audit_path = args.output_dir / "federated_client_distribution_audit.csv"
    metrics.to_csv(metrics_path, index=False)
    audit.to_csv(audit_path, index=False)
    write_reports(args.output_dir, metrics, audit, args)

    metadata = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "sample_sizes": args.sample_sizes,
        "feature_modes": sorted(metrics["feature_mode"].unique().tolist()),
        "client_split_types": args.split_types,
        "clients": args.clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "central_epochs": args.central_epochs,
        "fedprox_mu": args.fedprox_mu,
        "metric_tuple": METRIC_TUPLE_DEFINITION,
        "swissprot_humsavar_used_in_training": False,
        "full_feature_tables_modified": False,
        "graphsage_federated_run": False,
        "privacy_or_security_claimed": False,
    }
    (args.output_dir / "federated_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simulated horizontal federated learning benchmarks."
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument("--aa-input", type=Path, default=DEFAULT_AA_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[5_000, 10_000, 15_000, 20_000, 25_000],
    )
    parser.add_argument(
        "--split-types",
        choices=list(SPLIT_TYPES),
        nargs="+",
        default=list(SPLIT_TYPES),
    )
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--central-epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--non-iid-alpha", type=float, default=0.3)
    parser.add_argument("--ensemble-estimators", type=int, default=180)
    parser.add_argument("--ensemble-jobs", type=int, default=2)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
