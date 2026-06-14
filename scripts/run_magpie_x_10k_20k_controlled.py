#!/usr/bin/env python3
"""Run controlled MAGPIE-X C-F experiments on approved 10k/20k samples."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd

from run_aa_enhanced_low_circularity_comparison import build_aa_spec
from run_circularity_ablation import build_mode_specs
from run_magpie_low_circularity_baseline import build_improved_low_circularity_spec
from run_magpie_x_5k_enhanced_comparison import extend_spec
from run_magpie_x_5k_full_fusion import (
    ALPHA_NUMERIC,
    build_edges,
    build_graph_features,
    evaluate_tabular_mode,
    make_aligned_tabular_predictions,
    sha256_ids,
    train_same_split_gnn,
)
from run_sample_size_benchmark import (
    METRIC_TUPLE_DEFINITION,
    compute_metrics,
    infer_features,
    load_feature_table,
    make_splits,
    markdown_table,
    normalize_missing_markers,
    result_tuple,
)
from run_spliceai_5k_low_circularity_comparison import (
    SPLICEAI_CATEGORICAL_COLUMNS,
    SPLICEAI_NUMERIC_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_INPUT = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_enhanced"

MODE_C = "magpie_x_dbnsfp"
MODE_D = "magpie_x_dbnsfp_spliceai"
MODE_E = "magpie_x_dbnsfp_alphamissense"
MODE_F_BASE = "magpie_x_dbnsfp_spliceai_alphamissense_internal_graph_base"
MODE_F = "magpie_x_graphsage_residual"

MODE_LABELS = {
    MODE_C: "C",
    MODE_D: "D",
    MODE_E: "E",
    MODE_F: "F",
}

FEATURE_SOURCES = {
    MODE_C: (
        "ANNOVAR/refGene functional consequence; gnomAD allele frequency and "
        "derived rarity/missingness summaries; non-predictor conservation; variant "
        "type/length; amino-acid physicochemical deltas; BLOSUM62; 86 dbNSFP prior "
        "pathogenicity/damagingness score, rank, and prediction fields"
    ),
    MODE_D: (
        "Mode C feature sources; official masked SpliceAI v1.3 hg38 exact-match "
        "max DS_AG/AL/DG/DL, corresponding DP_AG/AL/DG/DL, max DS, availability, "
        "record count, and source type"
    ),
    MODE_E: (
        "Mode C feature sources; AlphaMissense hg38 exact-match pathogenicity score "
        "and availability flag; AlphaMissense class is excluded"
    ),
    MODE_F: (
        "Mode C feature sources; SpliceAI DS/DP sidecar; AlphaMissense score and "
        "availability; tabular p/logit; transductive variant graph with same-gene, "
        "same-consequence, and kNN feature-similarity edges"
    ),
}

PLUS_MEANING = {
    MODE_C: (
        "+ dbNSFP means 86 published prior-predictor fields are concatenated to the "
        "same clean variant rows; no new labels or dataset are added"
    ),
    MODE_D: (
        "+ SpliceAI means 12 exact-matched mechanistic splicing columns are added to "
        "mode C on the same variant IDs"
    ),
    MODE_E: (
        "+ AlphaMissense means an exact-matched expert missense score and availability "
        "flag are added to mode C; no SpliceAI or clinical labels are added"
    ),
    MODE_F: (
        "+ GraphSAGE means a graph-derived delta corrects the logit of the 214-feature "
        "mode C + SpliceAI + AlphaMissense LightGBM; it is residual correction, not stacking"
    ),
}

LIMITATIONS = {
    MODE_C: (
        "High circularity risk from predictor reuse and possible source-training "
        "overlap; random balanced split is not gene-held-out or external validation."
    ),
    MODE_D: (
        "Mode C circularity risk remains; SpliceAI coverage is incomplete and "
        "transcript/masking/normalization limitations apply."
    ),
    MODE_E: (
        "Mode C circularity risk remains; AlphaMissense is missense-only and may "
        "overlap external expert-model training evidence."
    ),
    MODE_F: (
        "High circularity plus transductive graph/same-gene context risk; test features "
        "participate in message passing although test labels remain hidden."
    ),
}


def source_path(sample_size: int) -> Path:
    return (
        ROOT
        / "outputs"
        / "real"
        / "magpie_parity"
        / f"reference_{sample_size // 1000}k_aa_spliceai_feature_table.csv"
    )


def alpha_path(sample_size: int, output_dir: Path) -> Path:
    return output_dir / f"alphamissense_{sample_size // 1000}k_features.csv"


def enrich_row(
    row: dict[str, object],
    *,
    experiment_id: str,
    sample_size: int,
    n_clean: int,
    n_risky: int,
    splice_available: int,
    alpha_available: int,
    ordered_hash: str,
    train_hash: str,
    test_hash: str,
) -> dict[str, object]:
    mode = str(row["mode"])
    row.update(
        {
            "experiment_id": experiment_id,
            "dataset": (
                f"Controlled ClinVar-derived balanced {sample_size // 1000}k "
                f"sample; {sample_size // 2:,} benign/likely benign and "
                f"{sample_size // 2:,} pathogenic/likely pathogenic"
            ),
            "split_definition": (
                f"Exact shared stratified 80/20 split, seed 13; "
                f"{int(row['n_train']):,} train/{int(row['n_test']):,} test"
            ),
            "ordered_variant_id_sha256": ordered_hash,
            "train_variant_id_sha256": train_hash,
            "test_variant_id_sha256": test_hash,
            "exact_feature_sources": FEATURE_SOURCES[mode],
            "number_clean_biological_features": n_clean,
            "number_risky_prior_predictor_fields": n_risky,
            "spliceai_available_rows": splice_available
            if int(row["includes_spliceai"])
            else 0,
            "alphamissense_matched_rows": alpha_available
            if int(row["includes_alphamissense"])
            else 0,
            "circularity_risk_level": "High circularity risk",
            "what_the_plus_means": PLUS_MEANING[mode],
            "limitation": LIMITATIONS[mode],
            "swissprot_humsavar_used_in_training": "No",
            "labels_used_for_external_matching": "No",
        }
    )
    return row


def load_combined(
    sample_size: int,
    *,
    output_dir: Path,
    parity_spec,
) -> tuple[pd.DataFrame, int, object]:
    source = normalize_missing_markers(
        pd.read_csv(source_path(sample_size), low_memory=False)
    )
    alpha = pd.read_csv(alpha_path(sample_size, output_dir), low_memory=False)
    if len(source) != sample_size or len(alpha) != sample_size:
        raise ValueError(f"Expected {sample_size:,} source and AlphaMissense rows.")
    if source["variant_id"].astype(str).tolist() != alpha["variant_id"].astype(str).tolist():
        raise ValueError(f"{sample_size // 1000}k AlphaMissense order mismatch.")
    if "label" in alpha.columns:
        raise ValueError("AlphaMissense sidecar must not contain labels.")
    merged = source.merge(
        alpha[["variant_id", *ALPHA_NUMERIC]],
        on="variant_id",
        how="left",
        validate="one_to_one",
    )
    merged["alphamissense_available"] = (
        pd.to_numeric(merged["alphamissense_available"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    combined, improved_spec, _ = build_improved_low_circularity_spec(
        merged, parity_spec
    )
    return combined, int(merged["alphamissense_available"].sum()), improved_spec


def graph_row(
    *,
    combined: pd.DataFrame,
    spec,
    base_params: dict[str, object],
    base_row: dict[str, object],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    sample_size: int,
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    started = time.perf_counter()
    p_tabular, logit_tabular = make_aligned_tabular_predictions(
        combined,
        spec,
        base_params,
        y,
        train_idx,
        test_idx,
        seed=args.seed,
    )
    reproduced = compute_metrics(y[test_idx], p_tabular[test_idx], args.ece_bins)
    for metric in (
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
    ):
        if abs(float(reproduced[metric]) - float(base_row[metric])) > 1e-12:
            raise RuntimeError(
                f"{sample_size // 1000}k graph-base probability mismatch: {metric}"
            )
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
    local_args = SimpleNamespace(**vars(args), sample_size=sample_size)
    predictions, metrics, final_loss = train_same_split_gnn(
        combined=combined,
        x=graph_x,
        edge_index=edge_index,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        p_tabular=p_tabular,
        logit_tabular=logit_tabular,
        args=local_args,
    )
    elapsed = time.perf_counter() - started
    row = {
        "mode": MODE_F,
        "sample_size": sample_size,
        "model": "lightgbm_graphsage_residual",
        "fold": int(base_row["fold"]),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(len(spec.feature_columns)),
        "graph_node_features": int(len(graph_features)),
        "circularity_risk": "high",
        "includes_spliceai": 1,
        "includes_alphamissense": 1,
        "includes_gnn": 1,
        "selected_candidate_id": base_row["selected_candidate_id"],
        "selected_params_json": base_row["selected_params_json"],
        **metrics,
    }
    row["result_tuple"] = result_tuple(row)
    metadata = {
        "sample_size": sample_size,
        "graph_nodes": int(len(combined)),
        "graph_directed_edges": int(edge_index.shape[1]),
        "edge_counts": dict(edge_counts),
        "graph_node_feature_count": int(len(graph_features)),
        "graph_node_features": graph_features,
        "architecture": "two-layer GraphSAGE",
        "hidden_units": args.gnn_hidden_dim,
        "dropout": args.gnn_dropout,
        "epochs": args.gnn_epochs,
        "final_training_loss": final_loss,
        "graph_runtime_seconds": elapsed,
        "same_gene_cap": args.same_gene_cap,
        "same_consequence_cap": args.same_consequence_cap,
        "knn_k": args.knn_k,
        "fusion_formula": (
            "z_final = logit(p_tabular) + delta_gnn; "
            "p_final = sigmoid(z_final)"
        ),
        "training_labels_visible_only_for_training_nodes": True,
        "validation_labels_used": False,
        "test_labels_used_during_training": False,
        "transductive_test_features_used": True,
    }
    return row, metadata, predictions


def write_gnn_audit(
    path: Path,
    *,
    graph_metadata: dict[int, dict[str, object]],
    skipped_20k_reason: str | None,
    split_metadata: dict[int, dict[str, str]],
) -> None:
    lines = [
        "# Same-Split GraphSAGE 10k/20k Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Alignment And Supervision",
        "",
        "- Samples are the exact approved A/B 10k and 20k ordered variant tables.",
        "- Splits use the same stratified seed-13 80/20 procedure as A/B.",
        "- Training labels are visible only for training nodes.",
        "- Validation and test labels are hidden during graph training.",
        "- Training-node tabular logits are five-fold out-of-fold predictions.",
        "- Test-node tabular logits come from a LightGBM fit only on training rows.",
        "- Graph preprocessing is fit on training nodes. Test features may participate "
        "in transductive graph message passing, but test labels do not.",
        "- Prediction files contain no label column.",
        "",
        "## Architecture And Fusion",
        "",
        "- Architecture: two-layer GraphSAGE, 64 hidden units, ReLU, dropout 0.25, "
        "50 fixed epochs.",
        "- Edge types: capped same-gene, capped same-consequence, and kNN "
        "feature-similarity.",
        "- Fusion is residual logit correction, not stacking:",
        "  `z_final = logit(p_tabular) + delta_gnn`; "
        "`p_final = sigmoid(z_final)`.",
        "- The tabular input is 214 features: 114 clean biological, 86 dbNSFP "
        "prior-predictor, 12 SpliceAI, and 2 AlphaMissense fields.",
        "",
        "## Runs",
        "",
    ]
    for sample_size in (10_000, 20_000):
        lines.append(f"### {sample_size // 1000}k")
        lines.append("")
        if sample_size not in graph_metadata:
            lines.append(f"- Run completed: **No**.")
            lines.append(f"- Reason: {skipped_20k_reason or 'not requested'}.")
            lines.append("")
            continue
        meta = graph_metadata[sample_size]
        split = split_metadata[sample_size]
        lines.extend(
            [
                "- Run completed: **Yes**.",
                f"- Graph nodes: {meta['graph_nodes']:,}",
                f"- Directed edges: {meta['graph_directed_edges']:,}",
                f"- Same-gene directed edges: {meta['edge_counts'].get('same_gene', 0):,}",
                f"- Same-consequence directed edges: {meta['edge_counts'].get('same_consequence', 0):,}",
                f"- kNN-similarity directed edges: {meta['edge_counts'].get('knn_similarity', 0):,}",
                f"- Graph node features: {meta['graph_node_feature_count']:,}",
                f"- Runtime: {meta['graph_runtime_seconds'] / 60.0:.2f} minutes",
                f"- Final training loss: {meta['final_training_loss']:.8f}",
                f"- Ordered-ID SHA256: `{split['ordered_hash']}`",
                f"- Train-ID SHA256: `{split['train_hash']}`",
                f"- Test-ID SHA256: `{split['test_hash']}`",
                f"- Predictions: `outputs/real/magpie_x_enhanced/gnn_same_split_{sample_size // 1000}k_predictions.csv`",
                "",
            ]
        )
    lines.extend(
        [
            "## Risk Boundary",
            "",
            "This is a high-circularity, transductive random-split experiment. The graph "
            "does not make the predictor-rich mode low-circularity, gene-held-out, or "
            "external validation evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    metrics: pd.DataFrame,
    graph_metadata: dict[int, dict[str, object]],
    skipped_20k_reason: str | None,
) -> None:
    summary = metrics[
        [
            "sample_size",
            "experiment_id",
            "model",
            "n_features",
            "number_risky_prior_predictor_fields",
            "spliceai_available_rows",
            "alphamissense_matched_rows",
            "result_tuple",
        ]
    ]
    lines = [
        "# MAGPIE-X Controlled 10k/20k Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
        "Rows C-F use each sample size's exact approved A/B variant order and "
        "stratified seed-13 split. SwissProt/HUMSAVAR is not used in training. "
        "All modes are reported as high circularity risk.",
        "",
        "## Verified Metrics",
        "",
    ]
    lines.extend(markdown_table(summary))
    lines.extend(["", "## Exact Definitions", ""])
    for mode in (MODE_C, MODE_D, MODE_E, MODE_F):
        subset = metrics[metrics["mode"].eq(mode)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.extend(
            [
                f"### {MODE_LABELS[mode]}. {mode}",
                "",
                f"- Feature sources: {FEATURE_SOURCES[mode]}.",
                f"- Total tabular features: {int(row['n_features'])}.",
                f"- Risky prior-predictor fields: {int(row['number_risky_prior_predictor_fields'])}.",
                f"- What `+` means: {PLUS_MEANING[mode]}.",
                f"- Circularity risk: **High**.",
                f"- Limitation: {LIMITATIONS[mode]}",
                "",
            ]
        )
    lines.extend(["## Scale Comparison", ""])
    for sample_size in sorted(metrics["sample_size"].unique()):
        subset = metrics[metrics["sample_size"].eq(sample_size)]
        best = subset.sort_values(["accuracy", "auroc"], ascending=False).iloc[0]
        lines.append(
            f"- {sample_size // 1000}k best verified row: "
            f"{best['experiment_id']} with accuracy {best['accuracy']:.4f}, "
            f"AUROC {best['auroc']:.4f}, and AUPRC {best['auprc']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Graph Runtime Gate",
            "",
            f"- 10k GraphSAGE completed: {'yes' if 10_000 in graph_metadata else 'no'}.",
            f"- 20k GraphSAGE completed: {'yes' if 20_000 in graph_metadata else 'no'}.",
        ]
    )
    if skipped_20k_reason:
        lines.append(f"- 20k skip reason: {skipped_20k_reason}.")
    lines.extend(
        [
            "",
            "No full-490k or federated training was run. These are random balanced "
            "ClinVar-derived comparisons and remain preliminary until gene-held-out, "
            "temporal, and external validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_df = load_feature_table(args.base_input)
    base_spec = infer_features(base_df, args.max_categorical_cardinality)
    mode_specs, removed_predictors = build_mode_specs(base_spec)
    enhanced_base_spec = mode_specs["magpie_x_enhanced"]
    del base_df
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
    n_risky = len(risky_numeric) + len(risky_categorical)
    if n_risky != 86:
        raise RuntimeError(f"Expected 86 risky predictor fields; found {n_risky}.")

    rows: list[dict[str, object]] = []
    graph_metadata: dict[int, dict[str, object]] = {}
    split_metadata: dict[int, dict[str, str]] = {}
    skipped_20k_reason: str | None = None
    run_20k_graph = False

    for sample_size in args.sample_sizes:
        combined, alpha_available, improved_spec = load_combined(
            sample_size,
            output_dir=args.output_dir,
            parity_spec=mode_specs["magpie_parity"],
        )
        aa_spec = build_aa_spec(improved_spec)
        enhanced = extend_spec(
            aa_spec,
            numeric=risky_numeric,
            categorical=risky_categorical,
        )
        enhanced_splice = extend_spec(
            enhanced,
            numeric=SPLICEAI_NUMERIC_COLUMNS,
            categorical=SPLICEAI_CATEGORICAL_COLUMNS,
        )
        enhanced_alpha = extend_spec(enhanced, numeric=ALPHA_NUMERIC)
        full_graph_base = extend_spec(enhanced_splice, numeric=ALPHA_NUMERIC)
        counts = {
            MODE_C: len(enhanced.feature_columns),
            MODE_D: len(enhanced_splice.feature_columns),
            MODE_E: len(enhanced_alpha.feature_columns),
            MODE_F_BASE: len(full_graph_base.feature_columns),
        }
        expected = {MODE_C: 200, MODE_D: 212, MODE_E: 202, MODE_F_BASE: 214}
        if counts != expected:
            raise RuntimeError(f"Unexpected {sample_size // 1000}k feature counts: {counts}")

        ids = combined["variant_id"].astype(str)
        if len(combined) != sample_size or ids.nunique() != sample_size:
            raise RuntimeError(f"{sample_size // 1000}k sample alignment failed.")
        y = pd.to_numeric(combined["label"], errors="raise").to_numpy(dtype=int)
        fold, train_idx, test_idx = make_splits(
            y, seed=args.seed, test_size=args.test_size, cv_folds=1
        )[0]
        split = {
            "ordered_hash": sha256_ids(ids),
            "train_hash": sha256_ids(combined.iloc[train_idx]["variant_id"].astype(str)),
            "test_hash": sha256_ids(combined.iloc[test_idx]["variant_id"].astype(str)),
        }
        split_metadata[sample_size] = split
        splice_available = int(
            pd.to_numeric(combined["spliceai_available"], errors="coerce")
            .fillna(0)
            .sum()
        )
        local_args = SimpleNamespace(**vars(args), sample_size=sample_size)
        specs = (
            (MODE_C, enhanced, False, False),
            (MODE_D, enhanced_splice, True, False),
            (MODE_E, enhanced_alpha, False, True),
        )
        for mode, spec, has_splice, has_alpha in specs:
            row, _ = evaluate_tabular_mode(
                mode=mode,
                combined=combined,
                spec=spec,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                fold=fold,
                args=local_args,
                includes_spliceai=has_splice,
                includes_alpha=has_alpha,
            )
            row = enrich_row(
                row,
                experiment_id=MODE_LABELS[mode],
                sample_size=sample_size,
                n_clean=len(aa_spec.feature_columns),
                n_risky=n_risky,
                splice_available=splice_available,
                alpha_available=alpha_available,
                ordered_hash=split["ordered_hash"],
                train_hash=split["train_hash"],
                test_hash=split["test_hash"],
            )
            rows.append(row)
            print(
                f"{sample_size // 1000}k {MODE_LABELS[mode]}: "
                f"accuracy={row['accuracy']:.4f} AUROC={row['auroc']:.4f}",
                flush=True,
            )

        should_graph = sample_size == 10_000 or (
            sample_size == 20_000 and run_20k_graph
        )
        if not should_graph:
            if sample_size == 20_000 and skipped_20k_reason is None:
                skipped_20k_reason = "10k graph runtime exceeded the configured gate"
            continue

        base_row, base_params = evaluate_tabular_mode(
            mode=MODE_F_BASE,
            combined=combined,
            spec=full_graph_base,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            fold=fold,
            args=local_args,
            includes_spliceai=True,
            includes_alpha=True,
        )
        gnn, graph_meta, predictions = graph_row(
            combined=combined,
            spec=full_graph_base,
            base_params=base_params,
            base_row=base_row,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            args=args,
            sample_size=sample_size,
        )
        gnn = enrich_row(
            gnn,
            experiment_id="F",
            sample_size=sample_size,
            n_clean=len(aa_spec.feature_columns),
            n_risky=n_risky,
            splice_available=splice_available,
            alpha_available=alpha_available,
            ordered_hash=split["ordered_hash"],
            train_hash=split["train_hash"],
            test_hash=split["test_hash"],
        )
        gnn.update(
            {
                "graph_nodes": graph_meta["graph_nodes"],
                "graph_directed_edges": graph_meta["graph_directed_edges"],
                "graph_edge_types": (
                    "same-gene; same-consequence; kNN feature-similarity"
                ),
                "graph_architecture": "two-layer GraphSAGE",
                "fusion_formula": graph_meta["fusion_formula"],
                "test_labels_hidden_during_graph_training": "Yes",
            }
        )
        rows.append(gnn)
        graph_metadata[sample_size] = graph_meta
        predictions.to_csv(
            args.output_dir
            / f"gnn_same_split_{sample_size // 1000}k_predictions.csv",
            index=False,
        )
        print(
            f"{sample_size // 1000}k F: accuracy={gnn['accuracy']:.4f} "
            f"AUROC={gnn['auroc']:.4f}; graph runtime="
            f"{graph_meta['graph_runtime_seconds'] / 60.0:.2f}m",
            flush=True,
        )
        if sample_size == 10_000:
            run_20k_graph = (
                graph_meta["graph_runtime_seconds"]
                <= args.max_10k_graph_minutes_for_20k * 60.0
            )
            if not run_20k_graph:
                skipped_20k_reason = (
                    f"10k graph runtime {graph_meta['graph_runtime_seconds'] / 60.0:.2f} "
                    f"minutes exceeded the {args.max_10k_graph_minutes_for_20k:.2f}-minute gate"
                )

    metrics = pd.DataFrame(rows).sort_values(
        ["sample_size", "experiment_id"]
    )
    metrics.to_csv(
        args.output_dir / "magpie_x_10k_20k_metrics.csv", index=False
    )
    write_report(
        args.output_dir / "MAGPIE_X_10K_20K_REPORT.md",
        metrics=metrics,
        graph_metadata=graph_metadata,
        skipped_20k_reason=skipped_20k_reason,
    )
    write_gnn_audit(
        args.output_dir / "gnn_same_split_10k_20k_audit.md",
        graph_metadata=graph_metadata,
        skipped_20k_reason=skipped_20k_reason,
        split_metadata=split_metadata,
    )
    metadata = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "sample_sizes": args.sample_sizes,
        "split_seed": args.seed,
        "split_metadata": split_metadata,
        "risky_prior_predictor_fields": n_risky,
        "graph_metadata": graph_metadata,
        "skipped_20k_reason": skipped_20k_reason,
        "full_490k_trained": False,
        "federated_learning_run": False,
        "swissprot_humsavar_used_in_training": False,
        "protected_full_feature_tables_modified": False,
    }
    (args.output_dir / "magpie_x_10k_20k_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact-split MAGPIE-X C-F experiments at 10k/20k."
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[10_000, 20_000]
    )
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
    parser.add_argument(
        "--max-10k-graph-minutes-for-20k", type=float, default=20.0
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
