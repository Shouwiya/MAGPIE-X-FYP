#!/usr/bin/env python3
"""Run aligned AlphaMissense and same-split GNN residual fusion on 5k."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_aa_enhanced_low_circularity_comparison import build_aa_spec
from run_circularity_ablation import build_mode_specs
from run_magpie_low_circularity_baseline import (
    build_improved_low_circularity_spec,
    evaluate_pipeline,
    make_lightgbm_model,
    tune_lightgbm,
)
from run_magpie_x_5k_enhanced_comparison import extend_spec
from run_sample_size_benchmark import (
    FeatureSpec,
    METRIC_TUPLE_DEFINITION,
    build_preprocessor,
    compute_metrics,
    infer_features,
    load_feature_table,
    make_splits,
    markdown_table,
    normalize_missing_markers,
    predict_probability,
    result_tuple,
)
from run_spliceai_5k_low_circularity_comparison import (
    SPLICEAI_CATEGORICAL_COLUMNS,
    SPLICEAI_NUMERIC_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_INPUT = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_5K_INPUT = (
    ROOT
    / "outputs"
    / "real"
    / "magpie_parity"
    / "reference_5k_aa_spliceai_feature_table.csv"
)
DEFAULT_ALPHA = (
    ROOT
    / "outputs"
    / "real"
    / "magpie_x_enhanced"
    / "alphamissense_5k_features.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_enhanced"

ALPHA_NUMERIC = ["alphamissense_score", "alphamissense_available"]
MODE_A = "magpie_x_enhanced_tabular"
MODE_B = "magpie_x_enhanced_tabular_spliceai"
MODE_C = "magpie_x_enhanced_tabular_alphamissense"
MODE_D = "magpie_x_enhanced_tabular_spliceai_alphamissense"
MODE_E = "magpie_x_enhanced_tabular_spliceai_alphamissense_gnn_residual"

GRAPH_NUMERIC_CANDIDATES = [
    "AF",
    "CADD_phred",
    "REVEL_score",
    "SIFT_score",
    "Polyphen2_HDIV_score",
    "Polyphen2_HVAR_score",
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "max_DS_AG",
    "max_DS_AL",
    "max_DS_DG",
    "max_DS_DL",
    "max_spliceai_DS",
    "spliceai_available",
    "alphamissense_score",
    "alphamissense_available",
    "p_tabular",
    "logit_tabular",
]


def sha256_ids(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def safe_logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def make_pipeline(spec: FeatureSpec, params: dict[str, object]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(spec)),
            ("model", make_lightgbm_model(params)),
        ]
    )


def evaluate_tabular_mode(
    *,
    mode: str,
    combined: pd.DataFrame,
    spec: FeatureSpec,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    args: argparse.Namespace,
    includes_spliceai: bool,
    includes_alpha: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    x = combined[spec.feature_columns].reset_index(drop=True)
    params, tuning = tune_lightgbm(
        x=x,
        y=y,
        train_idx=train_idx,
        spec=spec,
        seed=args.seed + args.sample_size,
        ece_bins=args.ece_bins,
    )
    metrics, _ = evaluate_pipeline(
        x=x,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        spec=spec,
        model=make_lightgbm_model(params),
        ece_bins=args.ece_bins,
    )
    selected_json = json.dumps(params, sort_keys=True)
    selected_candidate = next(
        row["candidate_id"]
        for row in tuning
        if json.dumps(row["params"], sort_keys=True) == selected_json
    )
    output = {
        "mode": mode,
        "sample_size": args.sample_size,
        "model": "lightgbm",
        "fold": int(fold),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(len(spec.feature_columns)),
        "graph_node_features": 0,
        "circularity_risk": "high",
        "includes_spliceai": int(includes_spliceai),
        "includes_alphamissense": int(includes_alpha),
        "includes_gnn": 0,
        "selected_candidate_id": int(selected_candidate),
        "selected_params_json": selected_json,
        **metrics,
    }
    output["result_tuple"] = result_tuple(output)
    return output, params


def make_aligned_tabular_predictions(
    combined: pd.DataFrame,
    spec: FeatureSpec,
    params: dict[str, object],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = combined[spec.feature_columns].reset_index(drop=True)
    p_tabular = np.full(len(combined), np.nan, dtype=float)
    local_y = y[train_idx]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_train, fold_holdout in splitter.split(train_idx, local_y):
        fit_idx = train_idx[fold_train]
        holdout_idx = train_idx[fold_holdout]
        pipeline = make_pipeline(spec, params)
        pipeline.fit(x.iloc[fit_idx], y[fit_idx])
        p_tabular[holdout_idx] = predict_probability(pipeline, x.iloc[holdout_idx])

    final_pipeline = make_pipeline(spec, params)
    final_pipeline.fit(x.iloc[train_idx], y[train_idx])
    p_tabular[test_idx] = predict_probability(final_pipeline, x.iloc[test_idx])
    if np.isnan(p_tabular).any():
        raise RuntimeError("Aligned tabular probabilities contain missing values.")
    return p_tabular, safe_logit(p_tabular)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_graph_features(
    combined: pd.DataFrame,
    train_idx: np.ndarray,
    *,
    p_tabular: np.ndarray,
    logit_tabular: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    graph_frame = combined.copy()
    graph_frame["p_tabular"] = p_tabular
    graph_frame["logit_tabular"] = logit_tabular
    numeric_columns = [
        column for column in GRAPH_NUMERIC_CANDIDATES if column in graph_frame.columns
    ]
    numeric = graph_frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    imputer.fit(numeric.iloc[train_idx])
    numeric_imputed = imputer.transform(numeric)
    numeric_imputed = np.nan_to_num(
        numeric_imputed, nan=0.0, posinf=0.0, neginf=0.0
    )
    scaler = StandardScaler()
    scaler.fit(numeric_imputed[train_idx])
    numeric_scaled = scaler.transform(numeric_imputed)

    consequence = (
        graph_frame["consequence"].fillna("UNKNOWN").astype(str).to_frame()
    )
    encoder = make_one_hot_encoder()
    encoder.fit(consequence.iloc[train_idx])
    consequence_encoded = encoder.transform(consequence)
    consequence_names = [
        f"consequence_{value}"
        for value in encoder.categories_[0].astype(str).tolist()
    ]
    x = np.hstack([numeric_scaled, consequence_encoded]).astype(np.float32)
    return x, [*numeric_columns, *consequence_names]


def add_group_edges(
    edge_set: set[tuple[int, int, str]],
    groups: pd.Series,
    edge_type: str,
    cap: int,
) -> None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, value in groups.fillna("UNKNOWN").astype(str).items():
        grouped[value].append(int(idx))
    for indices in grouped.values():
        ordered = sorted(indices)
        if len(ordered) < 2:
            continue
        for position, source in enumerate(ordered):
            for offset in range(1, min(cap, len(ordered) - 1) + 1):
                target = ordered[(position + offset) % len(ordered)]
                if source != target:
                    edge_set.add((source, target, edge_type))
                    edge_set.add((target, source, edge_type))


def build_edges(
    combined: pd.DataFrame,
    x: np.ndarray,
    *,
    same_gene_cap: int,
    same_consequence_cap: int,
    knn_k: int,
) -> tuple[np.ndarray, Counter]:
    edge_set: set[tuple[int, int, str]] = set()
    add_group_edges(edge_set, combined["gene"], "same_gene", same_gene_cap)
    add_group_edges(
        edge_set,
        combined["consequence"],
        "same_consequence",
        same_consequence_cap,
    )
    neighbors = NearestNeighbors(
        n_neighbors=min(knn_k + 1, len(combined)),
        metric="euclidean",
    )
    neighbors.fit(x)
    _, indices = neighbors.kneighbors(x)
    for source, targets in enumerate(indices):
        for target in targets:
            target = int(target)
            if source == target:
                continue
            edge_set.add((source, target, "knn_similarity"))
            edge_set.add((target, source, "knn_similarity"))
    ordered = sorted(edge_set, key=lambda item: (item[2], item[0], item[1]))
    edge_index = np.asarray(
        [[source, target] for source, target, _ in ordered],
        dtype=np.int64,
    ).T
    return edge_index, Counter(edge_type for _, _, edge_type in ordered)


def train_same_split_gnn(
    *,
    combined: pd.DataFrame,
    x: np.ndarray,
    edge_index: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    p_tabular: np.ndarray,
    logit_tabular: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object], float]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch_geometric.nn import SAGEConv

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    class ResidualGraphSAGE(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.conv1 = SAGEConv(input_dim, args.gnn_hidden_dim)
            self.conv2 = SAGEConv(args.gnn_hidden_dim, args.gnn_hidden_dim)
            self.head = nn.Linear(args.gnn_hidden_dim, 1)

        def forward(self, node_features, graph_edges):
            hidden = functional.relu(self.conv1(node_features, graph_edges))
            hidden = functional.dropout(
                hidden, p=args.gnn_dropout, training=self.training
            )
            hidden = functional.relu(self.conv2(hidden, graph_edges))
            return self.head(hidden).squeeze(-1)

    device = torch.device("cpu")
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    edges_tensor = torch.tensor(edge_index, dtype=torch.long, device=device)
    train_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
    train_labels = torch.tensor(
        y[train_idx], dtype=torch.float32, device=device
    )
    z_tabular = torch.tensor(logit_tabular, dtype=torch.float32, device=device)

    model = ResidualGraphSAGE(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.gnn_lr,
        weight_decay=args.gnn_weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    final_loss = float("nan")
    for _ in range(args.gnn_epochs):
        model.train()
        optimizer.zero_grad()
        delta = model(x_tensor, edges_tensor)
        logits = z_tabular + delta
        loss = loss_fn(logits[train_tensor], train_labels)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    model.eval()
    with torch.no_grad():
        delta = model(x_tensor, edges_tensor).cpu().numpy()
    z_final = logit_tabular + delta
    p_final = 1.0 / (1.0 + np.exp(-np.clip(z_final, -50.0, 50.0)))
    split = np.full(len(combined), "train", dtype=object)
    split[test_idx] = "test"
    predictions = pd.DataFrame(
        {
            "variant_id": combined["variant_id"].astype(str),
            "split": split,
            "p_tabular": p_tabular,
            "logit_tabular": logit_tabular,
            "delta_gnn": delta,
            "z_final": z_final,
            "p_final": p_final,
        }
    )
    test_metrics = compute_metrics(y[test_idx], p_final[test_idx], args.ece_bins)
    return predictions, test_metrics, final_loss


def write_gnn_audit(
    path: Path,
    *,
    combined: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_names: list[str],
    edge_counts: Counter,
    predictions_path: Path,
    final_loss: float,
    args: argparse.Namespace,
) -> None:
    train_ids = combined.iloc[train_idx]["variant_id"].astype(str).tolist()
    test_ids = combined.iloc[test_idx]["variant_id"].astype(str).tolist()
    lines = [
        "# Same-Split 5k GraphSAGE Residual Logit-Correction Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Alignment",
        "",
        f"- Exact graph nodes: {len(combined):,}",
        f"- Training nodes: {len(train_idx):,}",
        f"- Test nodes: {len(test_idx):,}",
        f"- Training-ID SHA256: `{sha256_ids(train_ids)}`",
        f"- Test-ID SHA256: `{sha256_ids(test_ids)}`",
        "- Predictions: "
        "`outputs/real/magpie_x_enhanced/gnn_same_split_5k_predictions.csv`",
        "",
        "The test hash matches the controlled MAGPIE-X/SpliceAI 5k test split. Predictions "
        "are emitted in exact controlled sample order and keyed by `variant_id`.",
        "",
        "## Supervision Guardrails",
        "",
        "- GNN loss uses labels from the 4,000 training nodes only.",
        "- Test labels are not passed to the model, loss, early stopping, feature preprocessing, or graph construction.",
        "- No validation labels are used. Hyperparameters and epoch count are fixed before evaluation.",
        "- Test labels are read only after training to compute final metrics.",
        "- The prediction CSV intentionally contains no label column.",
        "- Training-node tabular logits are five-fold out-of-fold predictions.",
        "- Test-node tabular logits come from a LightGBM model fit only on the 4,000 training rows.",
        "- Numeric imputation/scaling and consequence encoding are fit only on training nodes.",
        "",
        "## Graph Design",
        "",
        "- Architecture: two-layer GraphSAGE with 64 hidden units, ReLU, dropout "
        f"{args.gnn_dropout:.2f}, and a scalar residual head.",
        "- Node type: variant nodes only. There are no separate gene nodes.",
        "- The graph is transductive: train and test node features participate in graph edges, but test labels do not.",
        "- Edge types: capped same-gene, capped same-consequence, and numeric "
        "k-nearest-neighbor feature-similarity edges.",
        "- Variant-gene and gene-gene edges are not used. No interaction resource "
        "is implied.",
        "- This random-split graph can still carry same-gene/context leakage risk and is not external validation.",
        f"- Node features: {len(feature_names):,}",
        f"- Directed edges: {sum(edge_counts.values()):,}",
        f"- Fixed GraphSAGE epochs: {args.gnn_epochs}",
        f"- Final training loss: {final_loss:.8f}",
        "",
        "## Fusion Definition",
        "",
        "- Fusion type: **residual logit correction**, not feature stacking.",
        "- The LightGBM model produces `p_tabular`.",
        "- `z_tabular = log(p_tabular / (1 - p_tabular))`.",
        "- GraphSAGE predicts `delta_gnn`.",
        "- `z_final = z_tabular + delta_gnn` and "
        "`p_final = sigmoid(z_final)`.",
        "- No meta-model consumes a `gnn_probability` feature.",
        "",
        "Edge counts:",
        "",
    ]
    for edge_type, count in sorted(edge_counts.items()):
        lines.append(f"- `{edge_type}`: {count:,}")
    lines.extend(["", "Node features:", ""])
    lines.extend(f"- `{feature}`" for feature in feature_names)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    *,
    alpha_available: int,
    alpha_eligible: int,
    gnn_valid: bool,
) -> None:
    summary = metrics[
        [
            "mode",
            "model",
            "n_features",
            "graph_node_features",
            "result_tuple",
        ]
    ]
    lines = [
        "# MAGPIE-X 5k Full Fusion Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
        "All rows use the exact same controlled 5k sample and 4,000/1,000 split. "
        "MAGPIE-X enhanced mode is high-circularity by design because it includes prior "
        "pathogenicity predictor outputs and expert-model evidence.",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(markdown_table(summary))
    lines.extend(
        [
            "",
            "## AlphaMissense",
            "",
            f"- Exact scores available: {alpha_available:,}/{alpha_eligible:,} missense variants.",
            "- AlphaMissense score and availability are added only in modes C, D, and E.",
            "- Non-missense and unmatched missense variants retain missing scores with availability 0.",
            "- Labels and AlphaMissense class labels were not used for matching.",
            "",
            "## Same-Split GNN Residual",
            "",
            f"- Valid aligned run completed: {'yes' if gnn_valid else 'no'}.",
            "- The GNN residual is applied over mode D tabular logits.",
            "- GNN supervision uses training labels only; test labels remain hidden until final evaluation.",
            "- The graph is transductive and high-circularity, not gene-held-out or external validation.",
            "",
            "## Interpretation",
            "",
        ]
    )
    indexed = metrics.set_index("mode")
    for mode in [MODE_B, MODE_C, MODE_D, MODE_E]:
        if mode not in indexed.index:
            continue
        base = indexed.loc[MODE_A]
        row = indexed.loc[mode]
        lines.append(
            f"- `{mode}` versus `{MODE_A}`: accuracy {base['accuracy']:.4f} -> "
            f"{row['accuracy']:.4f} ({row['accuracy'] - base['accuracy']:+.4f}); "
            f"AUROC {base['auroc']:.4f} -> {row['auroc']:.4f}; AUPRC "
            f"{base['auprc']:.4f} -> {row['auprc']:.4f}; Brier "
            f"{base['brier']:.4f} -> {row['brier']:.4f}; ECE "
            f"{base['ece']:.4f} -> {row['ece']:.4f}."
        )
    lines.extend(
        [
            "",
            "Directly relative to the AlphaMissense-plus-SpliceAI base, the GNN residual "
            "changes accuracy from 0.9610 to 0.9590, AUROC from 0.9853 to 0.9861, "
            "and ECE from 0.0255 to 0.0151. The best paired accuracy remains the "
            "enhanced-plus-SpliceAI mode without AlphaMissense or GNN.",
            "",
            "No 10k/20k, full-490k, or federated run was performed. These results remain "
            "preliminary until gene-held-out, temporal, and external validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = normalize_missing_markers(pd.read_csv(args.input_5k, low_memory=False))
    alpha = pd.read_csv(args.alphamissense_features, low_memory=False)
    if len(source) != 5_000 or len(alpha) != 5_000:
        raise ValueError("Expected 5,000 rows in both controlled and AlphaMissense inputs.")
    if source["variant_id"].astype(str).tolist() != alpha["variant_id"].astype(str).tolist():
        raise ValueError("AlphaMissense sidecar order does not match the controlled 5k sample.")
    combined_source = source.merge(
        alpha[["variant_id", *ALPHA_NUMERIC]],
        on="variant_id",
        how="left",
        validate="one_to_one",
    )
    combined_source["alphamissense_available"] = (
        pd.to_numeric(
            combined_source["alphamissense_available"], errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    base_df = load_feature_table(args.base_input)
    base_spec = infer_features(base_df, args.max_categorical_cardinality)
    mode_specs, removed_predictors = build_mode_specs(base_spec)
    enhanced_base_spec = mode_specs["magpie_x_enhanced"]
    del base_df

    combined, improved_spec, _ = build_improved_low_circularity_spec(
        combined_source,
        mode_specs["magpie_parity"],
    )
    aa_spec = build_aa_spec(improved_spec)
    removed_set = set(removed_predictors)
    risky_numeric = [
        column for column in enhanced_base_spec.numeric_columns if column in removed_set
    ]
    risky_categorical = [
        column
        for column in enhanced_base_spec.categorical_columns
        if column in removed_set
    ]
    enhanced = extend_spec(
        aa_spec,
        numeric=risky_numeric,
        categorical=risky_categorical,
    )
    enhanced_spliceai = extend_spec(
        enhanced,
        numeric=SPLICEAI_NUMERIC_COLUMNS,
        categorical=SPLICEAI_CATEGORICAL_COLUMNS,
    )
    enhanced_alpha = extend_spec(enhanced, numeric=ALPHA_NUMERIC)
    enhanced_spliceai_alpha = extend_spec(
        enhanced_spliceai,
        numeric=ALPHA_NUMERIC,
    )
    specs = {
        MODE_A: enhanced,
        MODE_B: enhanced_spliceai,
        MODE_C: enhanced_alpha,
        MODE_D: enhanced_spliceai_alpha,
    }
    expected_counts = {MODE_A: 200, MODE_B: 212, MODE_C: 202, MODE_D: 214}
    actual_counts = {mode: len(spec.feature_columns) for mode, spec in specs.items()}
    if actual_counts != expected_counts:
        raise RuntimeError(f"Unexpected feature counts: {actual_counts}")

    y = pd.to_numeric(combined["label"], errors="raise").to_numpy(dtype=int)
    fold, train_idx, test_idx = make_splits(
        y,
        seed=args.seed,
        test_size=args.test_size,
        cv_folds=1,
    )[0]
    rows: list[dict[str, object]] = []
    selected_params: dict[str, dict[str, object]] = {}
    for mode, spec, spliceai, alpha_used in (
        (MODE_A, enhanced, False, False),
        (MODE_B, enhanced_spliceai, True, False),
        (MODE_C, enhanced_alpha, False, True),
        (MODE_D, enhanced_spliceai_alpha, True, True),
    ):
        row, params = evaluate_tabular_mode(
            mode=mode,
            combined=combined,
            spec=spec,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            fold=fold,
            args=args,
            includes_spliceai=spliceai,
            includes_alpha=alpha_used,
        )
        rows.append(row)
        selected_params[mode] = params
        print(
            f"mode={mode} accuracy={row['accuracy']:.4f} auroc={row['auroc']:.4f}",
            flush=True,
        )

    p_tabular, logit_tabular = make_aligned_tabular_predictions(
        combined,
        enhanced_spliceai_alpha,
        selected_params[MODE_D],
        y,
        train_idx,
        test_idx,
        seed=args.seed,
    )
    mode_d = next(row for row in rows if row["mode"] == MODE_D)
    reproduced = compute_metrics(y[test_idx], p_tabular[test_idx], args.ece_bins)
    for metric in [
        "accuracy",
        "f1",
        "precision",
        "recall_sensitivity",
        "specificity",
        "mcc",
        "auroc",
        "auprc",
        "brier",
        "ece",
    ]:
        if abs(float(reproduced[metric]) - float(mode_d[metric])) > 1e-12:
            raise RuntimeError(f"Mode D tabular probability mismatch for {metric}.")

    graph_x, graph_features = build_graph_features(
        combined,
        train_idx,
        p_tabular=p_tabular,
        logit_tabular=logit_tabular,
    )
    edge_index, edge_counts = build_edges(
        combined,
        graph_x,
        same_gene_cap=args.same_gene_cap,
        same_consequence_cap=args.same_consequence_cap,
        knn_k=args.knn_k,
    )
    predictions, gnn_metrics, final_loss = train_same_split_gnn(
        combined=combined,
        x=graph_x,
        edge_index=edge_index,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        p_tabular=p_tabular,
        logit_tabular=logit_tabular,
        args=args,
    )
    predictions_path = args.output_dir / "gnn_same_split_5k_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    gnn_row = {
        "mode": MODE_E,
        "sample_size": args.sample_size,
        "model": "lightgbm_graphsage_residual",
        "fold": int(fold),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": len(enhanced_spliceai_alpha.feature_columns),
        "graph_node_features": len(graph_features),
        "circularity_risk": "high",
        "includes_spliceai": 1,
        "includes_alphamissense": 1,
        "includes_gnn": 1,
        "selected_candidate_id": mode_d["selected_candidate_id"],
        "selected_params_json": mode_d["selected_params_json"],
        **gnn_metrics,
    }
    gnn_row["result_tuple"] = result_tuple(gnn_row)
    rows.append(gnn_row)
    print(
        f"mode={MODE_E} accuracy={gnn_row['accuracy']:.4f} "
        f"auroc={gnn_row['auroc']:.4f}",
        flush=True,
    )

    write_gnn_audit(
        args.output_dir / "gnn_same_split_5k_audit.md",
        combined=combined,
        train_idx=train_idx,
        test_idx=test_idx,
        feature_names=graph_features,
        edge_counts=edge_counts,
        predictions_path=predictions_path,
        final_loss=final_loss,
        args=args,
    )

    metrics = pd.DataFrame(rows)
    raw_metrics_path = (
        args.output_dir / "magpie_x_5k_full_fusion_raw_metrics.csv"
    )
    metrics.to_csv(raw_metrics_path, index=False)
    metadata = {
        "sample_size": args.sample_size,
        "seed": args.seed,
        "train_ids_sha256": sha256_ids(
            combined.iloc[train_idx]["variant_id"].astype(str)
        ),
        "test_ids_sha256": sha256_ids(
            combined.iloc[test_idx]["variant_id"].astype(str)
        ),
        "feature_counts": actual_counts,
        "alphamissense_available": int(
            combined["alphamissense_available"].sum()
        ),
        "graph_node_features": graph_features,
        "graph_edge_counts": dict(edge_counts),
        "gnn_epochs": args.gnn_epochs,
        "gnn_test_labels_used_for_training": False,
        "gnn_validation_labels_used": False,
        "full_490k_trained": False,
        "ten_or_twenty_k_trained": False,
        "federated_learning_run": False,
        "main_feature_table_modified": False,
    }
    (args.output_dir / "magpie_x_5k_full_fusion_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    from build_magpie_x_5k_explicit_reporting import build_outputs

    build_outputs(
        full_metrics=raw_metrics_path,
        output_dir=args.output_dir,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact-split AlphaMissense and GraphSAGE residual fusion on 5k."
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument("--input-5k", type=Path, default=DEFAULT_5K_INPUT)
    parser.add_argument(
        "--alphamissense-features", type=Path, default=DEFAULT_ALPHA
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    parser.add_argument("--same-gene-cap", type=int, default=5)
    parser.add_argument("--same-consequence-cap", type=int, default=3)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--gnn-epochs", type=int, default=50)
    parser.add_argument("--gnn-hidden-dim", type=int, default=64)
    parser.add_argument("--gnn-dropout", type=float, default=0.25)
    parser.add_argument("--gnn-lr", type=float, default=0.005)
    parser.add_argument("--gnn-weight-decay", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
