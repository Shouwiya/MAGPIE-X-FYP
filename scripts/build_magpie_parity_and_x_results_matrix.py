#!/usr/bin/env python3
"""Build a precise MAGPIE-parity and MAGPIE-X results matrix."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from run_sample_size_benchmark import (
    METRIC_TUPLE_DEFINITION,
    compute_metrics,
    result_tuple,
)


ROOT = Path(__file__).resolve().parents[1]
PARITY_DIR = ROOT / "outputs" / "real" / "magpie_parity"
ENHANCED_DIR = ROOT / "outputs" / "real" / "magpie_x_enhanced"
GNN_PREDICTIONS = (
    ROOT
    / "outputs"
    / "real"
    / "magpie_x_gnn"
    / "gnn_residual_run"
    / "gnn_predictions.csv"
)
OUTPUT = (
    ROOT
    / "outputs"
    / "real"
    / "final_comparison"
    / "MAGPIE_PARITY_AND_X_RESULTS_MATRIX.md"
)


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def metric_tuple(row: pd.Series) -> str:
    if "metric_tuple" in row and pd.notna(row["metric_tuple"]):
        return str(row["metric_tuple"])
    return result_tuple(row)


def controlled_5k_lines(frame: pd.DataFrame) -> list[str]:
    lines = [
        "## Controlled ClinVar-Derived Balanced 5k Comparison",
        "",
        "All seven rows use the same 5,000 variant IDs and the same stratified "
        "4,000/1,000 train/test split. SwissProt/HUMSAVAR is not used in training. "
        "SpliceAI and AlphaMissense matching use no clinical labels.",
        "",
        "| ID | Full mode name | Model and fusion | Features | Coverage / graph | Risk | Metric tuple | Meaning and limitation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in frame.sort_values("mode_id").itertuples(index=False):
        features = (
            f"{row.number_total_tabular_features} tabular; "
            f"{row.number_clean_biological_features} clean biological; "
            f"{row.number_risky_prior_predictor_fields} risky prior-predictor"
        )
        coverage = (
            f"SpliceAI available rows={row.spliceai_available_rows}; "
            f"AlphaMissense matched rows={row.alphamissense_matched_rows}; "
            f"graph nodes={row.graph_nodes_if_used}; "
            f"directed edges={row.graph_directed_edges_if_used}"
        )
        meaning = (
            f"{row.what_the_plus_sign_means}. Interpretation: {row.interpretation} "
            f"Limitation: {row.limitation}"
        )
        lines.append(
            f"| {md(row.mode_id)} | {md(row.long_descriptive_mode_name)} | "
            f"{md(row.ml_model)}; {md(row.fusion_type)} | {md(features)} | "
            f"{md(coverage)} | {md(row.circularity_risk_level)} | "
            f"{md(row.metric_tuple)} | {md(meaning)} |"
        )
    return lines


def larger_low_circularity_lines() -> list[str]:
    metrics = pd.read_csv(
        PARITY_DIR / "spliceai_10k_20k_low_circularity_metrics.csv"
    )
    lines = [
        "",
        "## Approved 10k/20k Low-Circularity Context",
        "",
        "These rows use their own exact approved paired samples and splits. They are "
        "not MAGPIE-X fusion runs and are not mixed with the controlled 5k training.",
        "",
        "| Sample | Full mode name | Dataset / split | Model and features | Risk | Metric tuple | Interpretation |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    names = {
        "aa_blosum_low_circularity": (
            "MAGPIE-parity low-circularity LightGBM using ANNOVAR/refGene + "
            "gnomAD AF + conservation + variant type/length + AA/BLOSUM62"
        ),
        "aa_blosum_spliceai_low_circularity": (
            "MAGPIE-parity low-circularity LightGBM with SpliceAI DS/DP "
            "feature fusion on the same ClinVar variants"
        ),
    }
    for row in metrics.sort_values(["sample_size", "mode"]).itertuples(index=False):
        has_splice = row.mode == "aa_blosum_spliceai_low_circularity"
        interpretation = (
            "Adds 12 matched SpliceAI columns, no labels or new dataset."
            if has_splice
            else "Clean biological annotations only; prior predictors excluded."
        )
        lines.append(
            f"| {int(row.sample_size):,} | {names[row.mode]} | Controlled "
            f"ClinVar-derived balanced {int(row.sample_size):,}; shared paired "
            f"80/20 split within sample | LightGBM; {int(row.n_features)} tabular "
            f"features | Low-circularity | {result_tuple(pd.Series(row._asdict()))} | "
            f"{interpretation} Random split; gene-held-out/external validation pending. |"
        )
    return lines


def controlled_larger_magpie_x_lines() -> list[str]:
    metrics = pd.read_csv(
        ENHANCED_DIR / "magpie_x_10k_20k_metrics.csv"
    )
    expected = {
        (10_000, "C"),
        (10_000, "D"),
        (10_000, "E"),
        (10_000, "F"),
        (20_000, "C"),
        (20_000, "D"),
        (20_000, "E"),
        (20_000, "F"),
    }
    actual = set(
        zip(
            metrics["sample_size"].astype(int),
            metrics["experiment_id"].astype(str),
        )
    )
    if actual != expected:
        raise ValueError(
            f"Controlled larger MAGPIE-X rows do not match C-F x 10k/20k: {actual}"
        )
    lines = [
        "",
        "## Controlled 10k/20k MAGPIE-X Comparison",
        "",
        "Within each sample size, rows C-F use the exact approved A/B variant order "
        "and stratified seed-13 80/20 split. The Excel convention calls the "
        "GraphSAGE residual experiment F; this corresponds to technical mode G in "
        "the earlier seven-row 5k fusion report because that report also exposed "
        "the intermediate SpliceAI-plus-AlphaMissense tabular row.",
        "",
        "| Sample | Excel ID | Full mode definition | Features | Coverage / graph | Risk | Metric tuple | Limitation |",
        "| ---: | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    names = {
        "C": (
            "MAGPIE-X LightGBM: 114 clean biological features + 86 dbNSFP "
            "prior-predictor fields"
        ),
        "D": (
            "MAGPIE-X LightGBM: row C + 12 exact-matched SpliceAI DS/DP fields"
        ),
        "E": (
            "MAGPIE-X LightGBM: row C + AlphaMissense score and availability; "
            "no SpliceAI"
        ),
        "F": (
            "MAGPIE-X two-layer GraphSAGE residual correction over a 214-feature "
            "row C + SpliceAI + AlphaMissense LightGBM"
        ),
    }
    for row in metrics.sort_values(["sample_size", "experiment_id"]).itertuples(
        index=False
    ):
        coverage = (
            f"SpliceAI={int(row.spliceai_available_rows):,}; "
            f"AlphaMissense={int(row.alphamissense_matched_rows):,}"
        )
        if row.experiment_id == "F":
            coverage += (
                f"; graph nodes={int(row.graph_nodes):,}; "
                f"directed edges={int(row.graph_directed_edges):,}"
            )
        lines.append(
            f"| {int(row.sample_size):,} | {row.experiment_id} | "
            f"{md(names[row.experiment_id])} | {int(row.n_features)} tabular; "
            f"{int(row.number_risky_prior_predictor_fields)} risky | "
            f"{md(coverage)} | High circularity risk | {md(row.result_tuple)} | "
            f"{md(row.limitation)} |"
        )
    return lines


def historical_context_lines() -> list[str]:
    historical = pd.read_csv(
        PARITY_DIR / "aa_enhanced_low_circularity_metrics.csv"
    )
    historical = historical[
        historical["mode"].eq("enhanced_dbnsfp_context_only")
    ]
    lines = [
        "",
        "## Historical Circularity-Risk Context",
        "",
        "These earlier rows are not paired with the controlled A-G comparison. They "
        "are retained only to document the previous predictor-rich LightGBM context.",
        "",
        "| Sample | Full mode name | Dataset / split | Model / features | Risk | Metric tuple | Limitation |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in historical.sort_values("sample_size").iterrows():
        lines.append(
            f"| {int(row['sample_size']):,} | Historical MAGPIE-X circularity-risk "
            "LightGBM using clean annotations plus dbNSFP prior predictor fields | "
            f"Earlier balanced ClinVar-derived {int(row['sample_size']):,} sample; "
            f"{int(row['n_train']):,}/{int(row['n_test']):,} random split | "
            f"LightGBM; {int(row['n_features'])} tabular features | High circularity "
            f"risk | {result_tuple(row)} | Different sampling run; not paired with "
            "SpliceAI/AlphaMissense/GraphSAGE modes and not external validation. |"
        )

    gnn = pd.read_csv(GNN_PREDICTIONS, low_memory=False)
    gnn_test = gnn[gnn["split"].eq("test")]
    gnn_metrics = compute_metrics(
        gnn_test["label"].to_numpy(dtype=int),
        gnn_test["p_final"].to_numpy(dtype=float),
        n_bins=10,
    )
    gnn_row = pd.Series({"sample_size": 5000, **gnn_metrics})
    lines.extend(
        [
            "",
            "The earlier XGBoost-plus-GraphSAGE residual experiment used a different "
            f"750-row test split and produced `{result_tuple(gnn_row)}`. It is "
            "nonpaired context only; it must not be substituted for controlled mode G.",
        ]
    )
    return lines


def main() -> None:
    controlled = pd.read_csv(
        ENHANCED_DIR / "magpie_x_5k_full_fusion_metrics.csv"
    )
    if controlled["mode_id"].tolist() != list("ABCDEFG"):
        raise ValueError("Controlled metrics must contain ordered modes A-G.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MAGPIE-Parity and MAGPIE-X Results Matrix",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
    ]
    lines.extend(controlled_5k_lines(controlled))
    lines.extend(larger_low_circularity_lines())
    lines.extend(controlled_larger_magpie_x_lines())
    lines.extend(historical_context_lines())
    lines.extend(
        [
            "",
            "## Circularity Boundary",
            "",
            "- Modes A-B exclude prior pathogenicity predictors and are the "
            "low-circularity MAGPIE-parity comparison.",
            "- Modes C-G intentionally include 86 dbNSFP prior-predictor fields. "
            "Modes E-G also use AlphaMissense; modes B, D, F, and G use SpliceAI.",
            "- Mode G is two-layer GraphSAGE residual logit correction, not feature "
            "stacking. Its loss uses training labels only; test labels are hidden "
            "until final evaluation.",
            "- None of the controlled A-G rows uses SwissProt/HUMSAVAR for training. "
            "That resource remains external validation.",
            "- No row is full-490k, gene-held-out, temporal, or final external "
            "validation evidence.",
            "",
            "## Source Artifacts",
            "",
            "- `outputs/real/magpie_x_enhanced/magpie_x_5k_full_fusion_metrics.csv`",
            "- `outputs/real/magpie_x_enhanced/MAGPIE_X_MODE_EXPLANATION_TABLE.md`",
            "- `outputs/real/magpie_x_enhanced/gnn_same_split_5k_audit.md`",
            "- `outputs/real/magpie_x_enhanced/alphamissense_5k_match_audit.md`",
            "- `outputs/real/magpie_x_enhanced/magpie_x_10k_20k_metrics.csv`",
            "- `outputs/real/magpie_x_enhanced/MAGPIE_X_10K_20K_REPORT.md`",
            "- `outputs/real/magpie_x_enhanced/gnn_same_split_10k_20k_audit.md`",
            "- `outputs/real/magpie_x_enhanced/alphamissense_10k_20k_match_audit.md`",
            "- `outputs/real/magpie_parity/spliceai_10k_20k_low_circularity_metrics.csv`",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
