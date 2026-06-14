#!/usr/bin/env python3
"""Build explicit, professor-readable reporting for the controlled 5k comparison."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARITY_METRICS = (
    ROOT / "outputs" / "real" / "magpie_x_enhanced" / "magpie_x_5k_metrics.csv"
)
DEFAULT_FULL_METRICS = (
    ROOT
    / "outputs"
    / "real"
    / "magpie_x_enhanced"
    / "magpie_x_5k_full_fusion_raw_metrics.csv"
)
LEGACY_FULL_METRICS = (
    ROOT
    / "outputs"
    / "real"
    / "magpie_x_enhanced"
    / "magpie_x_5k_full_fusion_metrics.csv"
)
DEFAULT_RISK_AUDIT = (
    ROOT
    / "outputs"
    / "real"
    / "magpie_x_enhanced"
    / "magpie_x_5k_feature_risk_audit.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_enhanced"

METRIC_COLUMNS = [
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
]
METRIC_TUPLE_DEFINITION = (
    "(Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, "
    "MCC, AUROC, AUPRC, Brier, ECE)"
)
DATASET = (
    "Controlled ClinVar-derived balanced 5k sample; 2,500 benign/likely benign "
    "and 2,500 pathogenic/likely pathogenic variants"
)
SPLIT = (
    "Exact shared stratified random 80/20 split: 4,000 train and 1,000 test "
    "(sample/split seed 13; identical variant IDs in modes A-G)"
)
CLEAN_SOURCES = (
    "ANNOVAR/refGene functional consequence; gnomAD allele-frequency fields and "
    "derived missingness/rarity summaries; non-predictor conservation fields; "
    "variant type/length fields; amino-acid physicochemical deltas; BLOSUM62"
)
SPLICEAI_SOURCES = (
    "Official masked SpliceAI v1.3 hg38 exact-match sidecar: max_DS_AG, "
    "max_DS_AL, max_DS_DG, max_DS_DL, DP_AG, DP_AL, DP_DG, DP_DL, "
    "max_spliceai_DS, spliceai_available, spliceai_record_count, "
    "spliceai_source_types"
)
ALPHA_SOURCES = (
    "AlphaMissense hg38 exact-match sidecar: alphamissense_score and "
    "alphamissense_available; AlphaMissense class was not used"
)
DBNSFP_SOURCES = (
    "86 explicitly flagged dbNSFP prior pathogenicity/damagingness score, rank, "
    "and prediction fields listed in this report"
)
GRAPH_SOURCES = (
    "Variant-only transductive graph with same-gene capped edges, "
    "same-consequence capped edges, k-nearest-neighbor feature-similarity edges, "
    "and tabular p/logit inputs"
)

MODE_CONFIG = OrderedDict(
    [
        (
            "A",
            {
                "technical_mode": "aa_blosum_low_circularity",
                "name": (
                    "MAGPIE-parity low-circularity LightGBM using ANNOVAR/refGene "
                    "+ gnomAD AF + conservation + variant type/length + AA/BLOSUM62"
                ),
                "model": "LightGBM binary classifier",
                "sources": CLEAN_SOURCES,
                "excluded": (
                    "All 86 dbNSFP prior predictor fields; SpliceAI; "
                    "AlphaMissense; graph-derived predictions"
                ),
                "total": 114,
                "clean": 114,
                "risky": 0,
                "splice": 0,
                "alpha": 0,
                "gnn_arch": "Not used",
                "nodes": 0,
                "edges": 0,
                "fusion": "No fusion; one clean tabular feature matrix",
                "risk": "Low-circularity",
                "plus": "No plus sign: clean biological annotations only",
                "purpose": (
                    "Reference estimate for the current partial MAGPIE-parity "
                    "feature set without prior pathogenicity predictors."
                ),
                "interpretation": (
                    "Provides the defensible low-circularity reference; it does "
                    "not reproduce every MAGPIE feature group or gene-held-out split."
                ),
                "limitation": (
                    "Random balanced ClinVar split; partial MAGPIE parity; no "
                    "gene-held-out, temporal, rare-variant, or external validation."
                ),
            },
        ),
        (
            "B",
            {
                "technical_mode": "aa_blosum_spliceai_low_circularity",
                "name": (
                    "MAGPIE-parity low-circularity LightGBM with SpliceAI DS/DP "
                    "feature fusion on the same ClinVar variants"
                ),
                "model": "LightGBM binary classifier",
                "sources": f"{CLEAN_SOURCES}; {SPLICEAI_SOURCES}",
                "excluded": (
                    "All 86 dbNSFP prior predictor fields; AlphaMissense; "
                    "graph-derived predictions"
                ),
                "total": 126,
                "clean": 126,
                "risky": 0,
                "splice": 4582,
                "alpha": 0,
                "gnn_arch": "Not used",
                "nodes": 0,
                "edges": 0,
                "fusion": (
                    "Tabular feature fusion: 12 SpliceAI columns concatenated to "
                    "the same 5k variant rows"
                ),
                "risk": "Low-circularity",
                "plus": (
                    "+ SpliceAI means mechanistic DS/DP columns were matched to "
                    "the same variants; no new labels or dataset were added"
                ),
                "purpose": (
                    "Tests whether a real MAGPIE-style splicing feature group adds "
                    "information beyond the clean AA/BLOSUM feature matrix."
                ),
                "interpretation": (
                    "Improves 5k accuracy and MCC over mode A, but lowers AUPRC "
                    "and worsens ECE; larger 10k/20k checks were not consistently positive."
                ),
                "limitation": (
                    "418 variants lacked real SpliceAI records; random balanced "
                    "split and transcript/masking/normalization limitations remain."
                ),
            },
        ),
        (
            "C",
            {
                "technical_mode": "magpie_x_enhanced_tabular",
                "name": (
                    "MAGPIE-X circularity-risk LightGBM using clean features + "
                    "86 dbNSFP prior predictor fields"
                ),
                "model": "LightGBM binary classifier",
                "sources": f"{CLEAN_SOURCES}; {DBNSFP_SOURCES}",
                "excluded": "SpliceAI; AlphaMissense; graph-derived predictions",
                "total": 200,
                "clean": 114,
                "risky": 86,
                "splice": 0,
                "alpha": 0,
                "gnn_arch": "Not used",
                "nodes": 0,
                "edges": 0,
                "fusion": (
                    "Tabular feature fusion: 86 prior predictor fields concatenated "
                    "to the same clean variant rows"
                ),
                "risk": "High circularity risk",
                "plus": (
                    "+ dbNSFP prior predictors means published pathogenicity/"
                    "damagingness outputs were added as input features"
                ),
                "purpose": (
                    "Quantifies the higher-performance track obtainable when "
                    "external expert predictors are intentionally admitted."
                ),
                "interpretation": (
                    "Substantially outperforms mode A on this random split, but "
                    "the gain cannot be interpreted as low-circularity evidence."
                ),
                "limitation": (
                    "Predictor reuse and training-data overlap may inflate "
                    "performance; separate gene-held-out/external validation is required."
                ),
            },
        ),
        (
            "D",
            {
                "technical_mode": "magpie_x_enhanced_tabular_spliceai",
                "name": (
                    "MAGPIE-X circularity-risk LightGBM using clean features + "
                    "86 dbNSFP prior predictor fields + SpliceAI DS/DP feature fusion"
                ),
                "model": "LightGBM binary classifier",
                "sources": f"{CLEAN_SOURCES}; {DBNSFP_SOURCES}; {SPLICEAI_SOURCES}",
                "excluded": "AlphaMissense; graph-derived predictions",
                "total": 212,
                "clean": 126,
                "risky": 86,
                "splice": 4582,
                "alpha": 0,
                "gnn_arch": "Not used",
                "nodes": 0,
                "edges": 0,
                "fusion": (
                    "Tabular feature fusion: dbNSFP predictor and SpliceAI columns "
                    "concatenated to the same 5k rows"
                ),
                "risk": "High circularity risk",
                "plus": (
                    "+ SpliceAI means DS/DP columns were added to mode C on the "
                    "same variants; it does not add labels"
                ),
                "purpose": (
                    "Tests the incremental value of mechanistic splice evidence "
                    "inside the circularity-risk predictor-rich model."
                ),
                "interpretation": (
                    "Achieves the highest controlled 5k accuracy (0.9620), while "
                    "AUROC/AUPRC and calibration do not uniformly improve over mode C."
                ),
                "limitation": (
                    "Best accuracy is random-split, predictor-rich evidence, not "
                    "a low-circularity or external-generalization claim."
                ),
            },
        ),
        (
            "E",
            {
                "technical_mode": "magpie_x_enhanced_tabular_alphamissense",
                "name": (
                    "MAGPIE-X circularity-risk LightGBM using clean features + "
                    "86 dbNSFP prior predictor fields + AlphaMissense feature fusion"
                ),
                "model": "LightGBM binary classifier",
                "sources": f"{CLEAN_SOURCES}; {DBNSFP_SOURCES}; {ALPHA_SOURCES}",
                "excluded": "SpliceAI; graph-derived predictions",
                "total": 202,
                "clean": 114,
                "risky": 86,
                "splice": 0,
                "alpha": 2713,
                "gnn_arch": "Not used",
                "nodes": 0,
                "edges": 0,
                "fusion": (
                    "Tabular feature fusion: AlphaMissense score/availability "
                    "concatenated to the same 5k rows"
                ),
                "risk": "High circularity risk",
                "plus": (
                    "+ AlphaMissense means an external expert missense score and "
                    "availability flag were exact-matched; no clinical label was added"
                ),
                "purpose": (
                    "Tests whether same-build, same-sample AlphaMissense evidence "
                    "adds value beyond the dbNSFP predictor-rich tabular model."
                ),
                "interpretation": (
                    "Does not improve accuracy over mode C, although AUPRC is "
                    "slightly higher; the expert feature is not universally beneficial."
                ),
                "limitation": (
                    "AlphaMissense is missense-only and available for 2,713/5,000 "
                    "rows; it is an expert predictor with circularity/overlap risk."
                ),
            },
        ),
        (
            "F",
            {
                "technical_mode": "magpie_x_enhanced_tabular_spliceai_alphamissense",
                "name": (
                    "MAGPIE-X circularity-risk LightGBM using clean features + "
                    "86 dbNSFP prior predictor fields + SpliceAI + AlphaMissense feature fusion"
                ),
                "model": "LightGBM binary classifier",
                "sources": (
                    f"{CLEAN_SOURCES}; {DBNSFP_SOURCES}; {SPLICEAI_SOURCES}; "
                    f"{ALPHA_SOURCES}"
                ),
                "excluded": "Graph-derived predictions",
                "total": 214,
                "clean": 126,
                "risky": 86,
                "splice": 4582,
                "alpha": 2713,
                "gnn_arch": "Not used",
                "nodes": 0,
                "edges": 0,
                "fusion": (
                    "Tabular feature fusion: dbNSFP, SpliceAI, and AlphaMissense "
                    "columns concatenated to the same 5k rows"
                ),
                "risk": "High circularity risk",
                "plus": (
                    "+ SpliceAI + AlphaMissense means both expert sidecars were "
                    "matched to the same variant IDs; no labels or new samples were added"
                ),
                "purpose": (
                    "Tests whether the two same-split expert sidecars complement "
                    "the predictor-rich LightGBM model."
                ),
                "interpretation": (
                    "Accuracy is below mode D, so adding AlphaMissense to the "
                    "SpliceAI model does not improve the primary controlled result."
                ),
                "limitation": (
                    "Combines multiple expert predictors with overlapping biological "
                    "scope; random-split performance may overstate generalization."
                ),
            },
        ),
        (
            "G",
            {
                "technical_mode": (
                    "magpie_x_enhanced_tabular_spliceai_alphamissense_gnn_residual"
                ),
                "name": (
                    "MAGPIE-X same-split two-layer GraphSAGE residual logit "
                    "correction over the mode-F LightGBM prediction"
                ),
                "model": (
                    "LightGBM tabular model + two-layer GraphSAGE residual "
                    "correction (64 hidden units, dropout 0.25, 50 fixed epochs)"
                ),
                "sources": (
                    f"{CLEAN_SOURCES}; {DBNSFP_SOURCES}; {SPLICEAI_SOURCES}; "
                    f"{ALPHA_SOURCES}; {GRAPH_SOURCES}"
                ),
                "excluded": (
                    "No ESM/GAT/heterogeneous variant-gene graph; no real "
                    "gene-gene interaction resource; no stacked meta-model"
                ),
                "total": 214,
                "clean": 126,
                "risky": 86,
                "splice": 4582,
                "alpha": 2713,
                "gnn_arch": "Two-layer GraphSAGE residual logit correction",
                "nodes": 5000,
                "edges": 100310,
                "fusion": (
                    "Residual logit correction: z_final = logit(p_tabular) + "
                    "delta_GraphSAGE; p_final = sigmoid(z_final)"
                ),
                "risk": "High circularity and transductive graph risk",
                "plus": (
                    "+ GraphSAGE means a graph-derived delta corrects the mode-F "
                    "tabular logit for the same IDs; it is not feature stacking"
                ),
                "purpose": (
                    "Tests whether same-split graph context can correct residual "
                    "tabular errors without exposing test labels."
                ),
                "interpretation": (
                    "Does not improve accuracy over mode F or the best mode D; "
                    "it improves ECE versus mode F, indicating a calibration effect."
                ),
                "limitation": (
                    "Transductive random-split graph permits test features and "
                    "same-gene neighborhoods in message passing; external and "
                    "gene-held-out validation are still required."
                ),
            },
        ),
    ]
)


def format_tuple(row: pd.Series) -> str:
    return "(" + ", ".join(f"{float(row[column]):.4f}" for column in METRIC_COLUMNS) + ")"


def predictor_family(feature: str) -> str:
    groups = [
        ("SIFT / SIFT4G", ("SIFT",)),
        ("PolyPhen-2", ("Polyphen2",)),
        ("LRT", ("LRT",)),
        ("MutationTaster", ("MutationTaster",)),
        ("MutationAssessor", ("MutationAssessor",)),
        ("FATHMM", ("FATHMM", "fathmm-")),
        ("PROVEAN", ("PROVEAN",)),
        ("VEST4", ("VEST4",)),
        ("MetaSVM / MetaLR / MetaRNN", ("MetaSVM", "MetaLR", "MetaRNN")),
        ("M-CAP", ("M-CAP",)),
        ("REVEL", ("REVEL",)),
        ("MutPred", ("MutPred",)),
        ("MVP / MPC", ("MVP", "MPC")),
        ("PrimateAI", ("PrimateAI",)),
        ("DEOGEN2", ("DEOGEN2",)),
        ("BayesDel", ("BayesDel",)),
        ("ClinPred", ("ClinPred",)),
        ("LIST-S2", ("LIST-S2",)),
        ("CADD", ("CADD",)),
        ("DANN", ("DANN",)),
        ("Eigen / Eigen-PC", ("Eigen",)),
        ("GenoCanyon", ("GenoCanyon",)),
        ("LINSIGHT", ("LINSIGHT",)),
    ]
    for family, prefixes in groups:
        if feature.startswith(prefixes):
            return family
    return "Other explicitly flagged predictor"


def load_metrics(parity_path: Path, full_path: Path) -> pd.DataFrame:
    parity = pd.read_csv(parity_path)
    if not full_path.exists():
        legacy = pd.read_csv(LEGACY_FULL_METRICS)
        legacy = legacy[
            legacy["mode"].astype(str).str.startswith("magpie_x_")
        ].copy()
        legacy.to_csv(full_path, index=False)
    full = pd.read_csv(full_path)
    combined = pd.concat([parity, full], ignore_index=True, sort=False)
    if combined["mode"].duplicated().any():
        combined = combined.drop_duplicates("mode", keep="last")
    return combined.set_index("mode", drop=False)


def build_explicit_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode_id, config in MODE_CONFIG.items():
        technical_mode = config["technical_mode"]
        if technical_mode not in metrics.index:
            raise ValueError(f"Missing verified metrics for {technical_mode}")
        source = metrics.loc[technical_mode]
        row = {
            "mode_id": mode_id,
            "technical_mode_id": technical_mode,
            "long_descriptive_mode_name": config["name"],
            "dataset_used": DATASET,
            "sample_size": 5000,
            "train_test_split": SPLIT,
            "n_train": 4000,
            "n_test": 1000,
            "ml_model": config["model"],
            "exact_feature_sources_used": config["sources"],
            "exact_feature_groups_excluded": config["excluded"],
            "number_total_features": (
                "214 tabular + 29 graph node inputs"
                if mode_id == "G"
                else config["total"]
            ),
            "number_total_tabular_features": config["total"],
            "number_clean_biological_features": config["clean"],
            "number_risky_prior_predictor_fields": config["risky"],
            "spliceai_available_rows": config["splice"],
            "alphamissense_matched_rows": config["alpha"],
            "gnn_architecture_if_used": config["gnn_arch"],
            "graph_nodes_if_used": config["nodes"],
            "graph_directed_edges_if_used": config["edges"],
            "fusion_type": config["fusion"],
            "circularity_risk_level": config["risk"],
            "what_the_plus_sign_means": config["plus"],
            "swissprot_humsavar_used_in_training": "No",
            "labels_used_for_spliceai_or_alphamissense_matching": "No",
            "test_labels_hidden_during_graph_training": (
                "Yes" if mode_id == "G" else "Not applicable"
            ),
            "scientific_purpose": config["purpose"],
        }
        for column in METRIC_COLUMNS:
            row[column] = float(source[column])
        row["metric_tuple"] = format_tuple(source)
        row["interpretation"] = config["interpretation"]
        row["limitation"] = config["limitation"]
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> list[str]:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cleaned = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(cleaned) + " |")
    return lines


def grouped_predictors(risk_audit: Path) -> OrderedDict[str, list[str]]:
    audit = pd.read_csv(risk_audit)
    risky = audit[
        audit["feature_group"].eq("prior_pathogenicity_predictor")
    ]["feature"].astype(str).tolist()
    if len(risky) != 86:
        raise ValueError(f"Expected 86 risky fields, found {len(risky)}")
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for feature in risky:
        grouped.setdefault(predictor_family(feature), []).append(feature)
    return grouped


def write_mode_table(path: Path, explicit: pd.DataFrame) -> None:
    lines = [
        "# MAGPIE-X Controlled 5k Mode Explanation Table",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Every row uses the same controlled ClinVar-derived balanced 5k sample and "
        "the same 4,000/1,000 train/test split. SwissProt/HUMSAVAR is not used in "
        "training; it remains a later external-generalization dataset.",
        "",
        "## Overview",
        "",
    ]
    lines.extend(
        markdown_table(
            explicit[
                [
                    "mode_id",
                    "long_descriptive_mode_name",
                    "ml_model",
                    "number_total_tabular_features",
                    "number_clean_biological_features",
                    "number_risky_prior_predictor_fields",
                    "fusion_type",
                    "circularity_risk_level",
                    "metric_tuple",
                ]
            ]
        )
    )
    field_labels = OrderedDict(
        [
            ("mode_id", "Mode ID"),
            ("long_descriptive_mode_name", "Long descriptive mode name"),
            ("dataset_used", "Dataset used"),
            ("sample_size", "Sample size"),
            ("train_test_split", "Train/test split"),
            ("ml_model", "ML model"),
            ("exact_feature_sources_used", "Exact feature sources used"),
            ("exact_feature_groups_excluded", "Exact feature groups excluded"),
            ("number_total_features", "Number of total features"),
            ("number_total_tabular_features", "Number of total tabular features"),
            (
                "number_clean_biological_features",
                "Number of clean biological features",
            ),
            (
                "number_risky_prior_predictor_fields",
                "Number of risky prior-predictor fields",
            ),
            ("spliceai_available_rows", "SpliceAI available rows"),
            ("alphamissense_matched_rows", "AlphaMissense matched rows"),
            ("gnn_architecture_if_used", "GNN architecture if used"),
            ("graph_nodes_if_used", "Graph nodes if used"),
            ("graph_directed_edges_if_used", "Graph directed edges if used"),
            ("fusion_type", "Fusion type"),
            ("circularity_risk_level", "Circularity risk level"),
            ("what_the_plus_sign_means", "What the plus sign means"),
            (
                "swissprot_humsavar_used_in_training",
                "SwissProt/HUMSAVAR used in training",
            ),
            (
                "labels_used_for_spliceai_or_alphamissense_matching",
                "Labels used for SpliceAI/AlphaMissense matching",
            ),
            (
                "test_labels_hidden_during_graph_training",
                "Test labels hidden during graph training",
            ),
            ("scientific_purpose", "Scientific purpose"),
            ("metric_tuple", "Metric tuple"),
            ("interpretation", "Interpretation"),
            ("limitation", "Limitation"),
        ]
    )
    for row in explicit.to_dict(orient="records"):
        lines.extend(
            [
                "",
                f"## Mode {row['mode_id']}",
                "",
                "| Field | Exact value |",
                "| --- | --- |",
            ]
        )
        for field, label in field_labels.items():
            value = str(row[field]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {label} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    explicit: pd.DataFrame,
    predictor_groups: OrderedDict[str, list[str]],
) -> None:
    compact = explicit[
        [
            "mode_id",
            "long_descriptive_mode_name",
            "ml_model",
            "number_total_tabular_features",
            "number_clean_biological_features",
            "number_risky_prior_predictor_fields",
            "fusion_type",
            "circularity_risk_level",
            "metric_tuple",
        ]
    ]
    lines = [
        "# Controlled MAGPIE-Parity versus MAGPIE-X 5k Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
        "## Controlled Dataset And Split",
        "",
        f"- Training/evaluation dataset: {DATASET}.",
        f"- Split: {SPLIT}.",
        "- SwissProt/HUMSAVAR used in this training: **No**. SwissProt/HUMSAVAR "
        "is external validation and will later test generalization of the "
        "ClinVar-trained model.",
        "- Labels used to match SpliceAI or AlphaMissense: **No**.",
        "- No mode adds new clinical labels or a new training dataset.",
        "",
        "## Results",
        "",
    ]
    lines.extend(markdown_table(compact))
    lines.extend(
        [
            "",
            "## What The Plus Sign Means",
            "",
            "- `+ SpliceAI`: the 12 DS/DP, maximum-score, availability, record-count, "
            "and source-type columns are exact-matched to the same ClinVar variant rows. "
            "This is tabular feature fusion, not a new dataset or label source.",
            "- `+ AlphaMissense`: `alphamissense_score` and "
            "`alphamissense_available` are exact-matched to the same hg38 variant rows. "
            "AlphaMissense is an external expert prediction, not a clinical label.",
            "- `+ dbNSFP prior predictors`: 86 published predictor score/rank/prediction "
            "fields are added to the same rows. This is intentionally high-circularity.",
            "- `+ GraphSAGE residual correction`: a graph model predicts "
            "`delta_gnn` for the same variant IDs and applies "
            "`z_final = logit(p_tabular) + delta_gnn`; this is residual logit "
            "correction, not feature stacking.",
            "- None of these additions mixes SwissProt/HUMSAVAR into the controlled "
            "5k training sample.",
            "",
            "## Feature Group Definitions",
            "",
            "### Clean MAGPIE-Parity Features",
            "",
            "The 114-feature clean matrix contains ANNOVAR/refGene functional "
            "consequence, gnomAD allele-frequency and rarity/missingness summaries, "
            "non-predictor conservation, variant type/length, amino-acid "
            "physicochemical deltas, and BLOSUM62. These are treated as "
            "low-circularity biological annotations.",
            "",
            "### SpliceAI Feature Fusion",
            "",
            f"`{SPLICEAI_SOURCES}`. Real scores are available for 4,582 of 5,000 "
            "variants; unmatched variants retain explicit unavailability rather than "
            "fabricated scores.",
            "",
            "### AlphaMissense Feature Fusion",
            "",
            f"`{ALPHA_SOURCES}`. Exact hg38 matching provides scores for 2,713 "
            "variants among 2,893 missense variants. Matching uses normalized "
            "chromosome, position, REF, and ALT only.",
            "",
            "### dbNSFP Prior-Predictor Fields",
            "",
            "The following 86 fields are useful for performance but carry "
            "circularity risk because they are already pathogenicity or damagingness "
            "predictors:",
            "",
        ]
    )
    for family, features in predictor_groups.items():
        lines.append(f"- **{family}:** " + ", ".join(f"`{feature}`" for feature in features))
    lines.extend(
        [
            "",
            "### Graph Module",
            "",
            "- Architecture: two-layer GraphSAGE, 64 hidden units, dropout 0.25, "
            "50 fixed epochs.",
            "- Nodes: 5,000 variant nodes only. No separate gene nodes are present.",
            "- Directed edges: 100,310 total: 14,442 same-gene capped edges, "
            "29,994 same-consequence capped edges, and 55,874 kNN "
            "feature-similarity edges.",
            "- No gene-gene interaction edge is used because no real interaction "
            "resource is part of this experiment.",
            "- Node inputs: selected clean and predictor-rich tabular values, SpliceAI, "
            "AlphaMissense, consequence one-hot indicators, `p_tabular`, and "
            "`logit_tabular`.",
            "- Training labels are visible only for the 4,000 training nodes. No "
            "validation labels are used, and test labels are hidden until final metrics.",
            "- The graph is transductive: test-node features and edges are visible, "
            "but test labels are not.",
            "",
            "## Scientific Interpretation",
            "",
        ]
    )
    for row in explicit.itertuples(index=False):
        lines.append(
            f"- **Mode {row.mode_id}:** {row.interpretation} "
            f"Limitation: {row.limitation}"
        )
    lines.extend(
        [
            "",
            "The best controlled 5k accuracy is mode D at 0.9620. That mode is "
            "not low-circularity: it includes 86 dbNSFP prior predictor fields. "
            "Mode G is a valid same-split residual correction experiment, but it "
            "does not improve accuracy over mode F or mode D; its clearest change "
            "is lower ECE relative to mode F.",
            "",
            "No 10k/20k fusion, full-490k training, federated learning, or new "
            "SwissProt/HUMSAVAR training was performed.",
            "",
            "`FYP_Literature_Results_Table_Omar.xlsx` was not modified because it "
            "is a literature-results workbook, whereas modes A--G are results from "
            "this project's controlled experiment. Mixing them would blur source "
            "provenance; the project results are documented in the controlled CSV, "
            "mode explanation table, consolidated matrix, and paper.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(
    *,
    parity_metrics: Path = DEFAULT_PARITY_METRICS,
    full_metrics: Path = DEFAULT_FULL_METRICS,
    risk_audit: Path = DEFAULT_RISK_AUDIT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(parity_metrics, full_metrics)
    explicit = build_explicit_rows(metrics)
    explicit.to_csv(output_dir / "magpie_x_5k_full_fusion_metrics.csv", index=False)
    predictor_groups = grouped_predictors(risk_audit)
    write_report(
        output_dir / "MAGPIE_X_5K_FULL_FUSION_REPORT.md",
        explicit,
        predictor_groups,
    )
    write_mode_table(
        output_dir / "MAGPIE_X_MODE_EXPLANATION_TABLE.md",
        explicit,
    )
    return explicit


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build explicit reporting for the controlled MAGPIE 5k modes A-G."
    )
    parser.add_argument("--parity-metrics", type=Path, default=DEFAULT_PARITY_METRICS)
    parser.add_argument("--full-metrics", type=Path, default=DEFAULT_FULL_METRICS)
    parser.add_argument("--risk-audit", type=Path, default=DEFAULT_RISK_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    explicit = build_outputs(
        parity_metrics=args.parity_metrics,
        full_metrics=args.full_metrics,
        risk_audit=args.risk_audit,
        output_dir=args.output_dir,
    )
    print(
        f"Wrote explicit controlled comparison for {len(explicit)} modes to "
        f"{args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
