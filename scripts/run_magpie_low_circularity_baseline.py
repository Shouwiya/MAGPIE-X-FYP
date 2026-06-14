from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from run_circularity_ablation import (
    PRIOR_PREDICTOR_PATTERNS,
    build_mode_specs,
    is_prior_predictor_column,
)
from run_sample_size_benchmark import (
    DEFAULT_INPUT,
    FeatureSpec,
    METRIC_COLUMNS,
    METRIC_TUPLE_DEFINITION,
    balanced_sample,
    build_feasibility,
    build_model,
    build_preprocessor,
    compute_metrics,
    format_metric,
    infer_features,
    load_feature_table,
    make_splits,
    markdown_table,
    predict_probability,
    result_tuple,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_low_circularity"

EXPLICIT_FORBIDDEN_PREDICTORS = (
    "sift",
    "polyphen",
    "revel",
    "cadd",
    "mutationtaster",
    "metasvm",
    "metalr",
    "fathmm",
    "provean",
    "m-cap",
    "mcap",
    "alphamissense",
    "esm",
)

AF_COLUMNS = [
    "AF",
    "AF_raw",
    "AF_male",
    "AF_female",
    "AF_afr",
    "AF_ami",
    "AF_amr",
    "AF_asj",
    "AF_eas",
    "AF_fin",
    "AF_nfe",
    "AF_oth",
    "AF_sas",
]

RAW_CONSERVATION_COLUMNS = [
    "GERP++_NR",
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phyloP30way_mammalian",
    "phastCons100way_vertebrate",
    "phastCons30way_mammalian",
    "SiPhy_29way_logOdds",
]

RISKY_NON_MODEL_COLUMNS = {
    "label",
    "variant_id",
    "clin_sig_raw",
    "clinical_significance",
    "review_status",
    "source_file",
    "assembly",
    "chrom",
    "pos",
    "end",
    "ref",
    "alt",
    "gene",
    "gene_annovar",
    "GeneDetail.refGene",
    "GTEx_V8_gene",
    "aa_change_refgene",
    "Interpro_domain",
    "origin",
}

MISSING_MAGPIE_PARITY_FEATURES = [
    "Full splicing-context annotations beyond current consequence labels",
    "Regulatory/epigenomic context such as ChromHMM or enhancer/promoter state",
    "Protein-domain position and amino-acid physicochemical change features beyond raw InterPro text",
    "Gene-level constraint summaries not derived from ClinVar labels",
    "Transcript-aware canonical isoform and exon-position features",
    "External temporal/gene-held-out split identifiers matching the original MAGPIE protocol",
]


def normalize_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column.lower())


def matches_explicit_forbidden_predictor(column: str) -> bool:
    normalized = normalize_column_name(column)
    return any(normalize_column_name(pattern) in normalized for pattern in EXPLICIT_FORBIDDEN_PREDICTORS)


def require_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is required for this low-circularity tuning run.") from exc
    return lgb


def to_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return pd.DataFrame(index=df.index)
    return df[existing].apply(pd.to_numeric, errors="coerce")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def count_delimited_values(series: pd.Series) -> pd.Series:
    def count_one(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if not text or text in {"-", "."}:
            return np.nan
        parts = [part for part in re.split(r"[,;|]", text) if part.strip()]
        return float(len(parts)) if parts else np.nan

    return series.map(count_one)


def build_engineered_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    engineered = pd.DataFrame(index=df.index)
    feature_groups: dict[str, list[str]] = {
        "af_bins_and_flags": [],
        "missingness_indicators": [],
        "consequence_features": [],
        "conservation_summaries": [],
        "variant_length_features": [],
        "transcript_consequence_counts": [],
    }

    af = to_numeric_frame(df, AF_COLUMNS)
    if not af.empty:
        present_af = af.columns.tolist()
        engineered["af_nonmissing_count"] = af.notna().sum(axis=1)
        engineered["af_missing_count"] = af.isna().sum(axis=1)
        engineered["af_any_missing"] = af.isna().any(axis=1).astype(int)
        engineered["af_all_missing"] = af.isna().all(axis=1).astype(int)
        engineered["af_max"] = af.max(axis=1, skipna=True)
        engineered["af_min"] = af.min(axis=1, skipna=True)
        engineered["af_mean"] = af.mean(axis=1, skipna=True)
        engineered["af_range"] = engineered["af_max"] - engineered["af_min"]
        engineered["af_max_is_zero"] = (engineered["af_max"].fillna(-1) == 0).astype(int)
        engineered["af_max_le_1e_5"] = (engineered["af_max"] <= 1e-5).fillna(False).astype(int)
        engineered["af_max_le_1e_4"] = (engineered["af_max"] <= 1e-4).fillna(False).astype(int)
        engineered["af_max_le_1e_3"] = (engineered["af_max"] <= 1e-3).fillna(False).astype(int)
        engineered["af_max_le_1e_2"] = (engineered["af_max"] <= 1e-2).fillna(False).astype(int)
        engineered["af_max_ge_1e_2"] = (engineered["af_max"] >= 1e-2).fillna(False).astype(int)
        engineered["af_max_bin"] = pd.cut(
            engineered["af_max"],
            bins=[-np.inf, 0.0, 1e-5, 1e-4, 1e-3, 1e-2, np.inf],
            labels=["zero", "ultra_rare", "very_rare", "rare", "low_frequency", "common"],
        ).astype("object")
        engineered.loc[engineered["af_max"].isna(), "af_max_bin"] = "missing"
        feature_groups["af_bins_and_flags"].extend(
            [
                "af_nonmissing_count",
                "af_missing_count",
                "af_any_missing",
                "af_all_missing",
                "af_max",
                "af_min",
                "af_mean",
                "af_range",
                "af_max_is_zero",
                "af_max_le_1e_5",
                "af_max_le_1e_4",
                "af_max_le_1e_3",
                "af_max_le_1e_2",
                "af_max_ge_1e_2",
                "af_max_bin",
            ]
        )
        for column in present_af:
            name = f"{column}_missing_indicator"
            engineered[name] = af[column].isna().astype(int)
            feature_groups["missingness_indicators"].append(name)

    conservation = to_numeric_frame(df, RAW_CONSERVATION_COLUMNS)
    if not conservation.empty:
        engineered["conservation_nonmissing_count"] = conservation.notna().sum(axis=1)
        engineered["conservation_missing_count"] = conservation.isna().sum(axis=1)
        engineered["conservation_any_missing"] = conservation.isna().any(axis=1).astype(int)
        engineered["conservation_mean_raw"] = conservation.mean(axis=1, skipna=True)
        engineered["conservation_min_raw"] = conservation.min(axis=1, skipna=True)
        engineered["conservation_max_raw"] = conservation.max(axis=1, skipna=True)
        engineered["conservation_abs_max_raw"] = conservation.abs().max(axis=1, skipna=True)
        feature_groups["conservation_summaries"].extend(
            [
                "conservation_nonmissing_count",
                "conservation_missing_count",
                "conservation_any_missing",
                "conservation_mean_raw",
                "conservation_min_raw",
                "conservation_max_raw",
                "conservation_abs_max_raw",
            ]
        )
        for column in conservation.columns:
            name = f"{column}_missing_indicator"
            engineered[name] = conservation[column].isna().astype(int)
            feature_groups["missingness_indicators"].append(name)

    if {"ref_length", "alt_length"}.issubset(df.columns):
        ref_len = pd.to_numeric(df["ref_length"], errors="coerce")
        alt_len = pd.to_numeric(df["alt_length"], errors="coerce")
        delta = alt_len - ref_len
        engineered["abs_variant_length_change"] = delta.abs()
        engineered["ref_alt_length_ratio"] = safe_divide(ref_len, alt_len)
        engineered["alt_ref_length_ratio"] = safe_divide(alt_len, ref_len)
        engineered["is_length_preserving_substitution"] = ((ref_len == alt_len) & (ref_len > 1)).fillna(False).astype(int)
        engineered["is_indel_engineered"] = (delta.fillna(0).abs() > 0).astype(int)
        engineered["log1p_ref_length"] = np.log1p(ref_len.clip(lower=0))
        engineered["log1p_alt_length"] = np.log1p(alt_len.clip(lower=0))
        feature_groups["variant_length_features"].extend(
            [
                "abs_variant_length_change",
                "ref_alt_length_ratio",
                "alt_ref_length_ratio",
                "is_length_preserving_substitution",
                "is_indel_engineered",
                "log1p_ref_length",
                "log1p_alt_length",
            ]
        )

    consequence_columns = [column for column in df.columns if column.startswith("consequence_")]
    if consequence_columns:
        c = df[consequence_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        engineered["consequence_flag_count"] = c.sum(axis=1)
        engineered["is_splice_consequence"] = (
            c[[col for col in c.columns if "splice" in col]].sum(axis=1) > 0
        ).astype(int)
        engineered["is_indel_consequence"] = (
            c[[col for col in c.columns if "inframe" in col or "frameshift" in col]].sum(axis=1) > 0
        ).astype(int)
        engineered["is_stop_consequence"] = (
            c[[col for col in c.columns if "nonsense" in col or "stop" in col]].sum(axis=1) > 0
        ).astype(int)
        engineered["is_missense_consequence"] = (
            c[[col for col in c.columns if "missense" in col]].sum(axis=1) > 0
        ).astype(int)
        feature_groups["consequence_features"].extend(
            [
                "consequence_flag_count",
                "is_splice_consequence",
                "is_indel_consequence",
                "is_stop_consequence",
                "is_missense_consequence",
            ]
        )

    if "aa_change_refgene" in df.columns:
        engineered["aa_change_transcript_count"] = count_delimited_values(df["aa_change_refgene"])
        engineered["aa_change_has_multiple_transcripts"] = (engineered["aa_change_transcript_count"] > 1).fillna(False).astype(int)
        feature_groups["transcript_consequence_counts"].extend(
            ["aa_change_transcript_count", "aa_change_has_multiple_transcripts"]
        )

    if "exonic_func_refgene" in df.columns and "func_refgene" in df.columns:
        combo = (
            df["func_refgene"].fillna("missing").astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
            + "__"
            + df["exonic_func_refgene"].fillna("missing").astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
        )
        engineered["func_exonic_func_interaction"] = combo
        feature_groups["consequence_features"].append("func_exonic_func_interaction")

    feature_columns = engineered.columns.tolist()
    return engineered, feature_columns, feature_groups


def build_improved_low_circularity_spec(df: pd.DataFrame, old_spec: FeatureSpec) -> tuple[pd.DataFrame, FeatureSpec, dict[str, list[str]]]:
    engineered, engineered_columns, feature_groups = build_engineered_features(df)
    combined = pd.concat([df, engineered], axis=1)

    numeric = list(old_spec.numeric_columns)
    categorical = list(old_spec.categorical_columns)
    engineered_numeric: list[str] = []
    engineered_categorical: list[str] = []
    for column in engineered_columns:
        if matches_explicit_forbidden_predictor(column) or is_prior_predictor_column(column):
            raise ValueError(f"Engineered feature unexpectedly matches a forbidden predictor pattern: {column}")
        if pd.api.types.is_numeric_dtype(combined[column]):
            engineered_numeric.append(column)
        else:
            engineered_categorical.append(column)
    numeric.extend(engineered_numeric)
    categorical.extend(engineered_categorical)

    seen: set[str] = set()
    numeric = [column for column in numeric if not (column in seen or seen.add(column))]
    seen.clear()
    categorical = [column for column in categorical if not (column in seen or seen.add(column))]

    spec = FeatureSpec(
        feature_columns=[*numeric, *categorical],
        numeric_columns=numeric,
        categorical_columns=categorical,
        dropped_columns=dict(old_spec.dropped_columns),
    )
    return combined, spec, feature_groups


def lightgbm_candidates(seed: int) -> list[dict[str, object]]:
    return [
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": seed,
        },
        {
            "n_estimators": 700,
            "learning_rate": 0.04,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 0.5,
            "random_state": seed,
        },
        {
            "n_estimators": 800,
            "learning_rate": 0.035,
            "num_leaves": 63,
            "min_child_samples": 50,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "random_state": seed,
        },
        {
            "n_estimators": 900,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 80,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "random_state": seed,
        },
        {
            "n_estimators": 700,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 30,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": seed,
        },
        {
            "n_estimators": 1000,
            "learning_rate": 0.025,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 40,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.05,
            "reg_lambda": 1.5,
            "random_state": seed,
        },
    ]


def make_lightgbm_model(params: dict[str, object]):
    lgb = require_lightgbm()
    clean_params = {
        "objective": "binary",
        "class_weight": "balanced",
        "n_jobs": -1,
        "verbose": -1,
        **params,
    }
    return lgb.LGBMClassifier(**clean_params)


def evaluate_pipeline(
    *,
    x: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    spec: FeatureSpec,
    model,
    ece_bins: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(spec)),
            ("model", model),
        ]
    )
    pipeline.fit(x.iloc[train_idx], y[train_idx])
    y_prob = predict_probability(pipeline, x.iloc[test_idx])
    metrics = compute_metrics(y[test_idx], y_prob, ece_bins)
    return metrics, y_prob


def tune_lightgbm(
    *,
    x: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    spec: FeatureSpec,
    seed: int,
    ece_bins: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    train_sub_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        stratify=y[train_idx],
        random_state=seed,
    )
    tuning_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for candidate_id, params in enumerate(lightgbm_candidates(seed), start=1):
        metrics, _ = evaluate_pipeline(
            x=x,
            y=y,
            train_idx=np.sort(train_sub_idx),
            test_idx=np.sort(val_idx),
            spec=spec,
            model=make_lightgbm_model(params),
            ece_bins=ece_bins,
        )
        row = {
            "candidate_id": candidate_id,
            "selection_score": float(metrics["auprc"]) if math.isfinite(float(metrics["auprc"])) else -1.0,
            "validation_auprc": metrics["auprc"],
            "validation_auroc": metrics["auroc"],
            "validation_accuracy": metrics["accuracy"],
            "params": params,
        }
        tuning_rows.append(row)
        if best is None or float(row["selection_score"]) > float(best["selection_score"]):
            best = row
    assert best is not None
    return dict(best["params"]), tuning_rows


def summarize_metric_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics_df.iterrows():
        rows.append(
            {
                "mode": row["mode"],
                "sample_size": row["sample_size"],
                "model": row["model"],
                "n_features": row["n_features"],
                "result_tuple": result_tuple(row),
            }
        )
    return pd.DataFrame(rows)


def classify_columns(
    df: pd.DataFrame,
    base_spec: FeatureSpec,
    old_spec: FeatureSpec,
    improved_spec: FeatureSpec,
    removed_columns: list[str],
    engineered_groups: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    removed_set = set(removed_columns)
    old_set = set(old_spec.feature_columns)
    improved_set = set(improved_spec.feature_columns)
    base_feature_set = set(base_spec.feature_columns)

    kept_rows = []
    removed_rows = []
    risky_rows = []

    for column in df.columns:
        if column in removed_set or is_prior_predictor_column(column) or matches_explicit_forbidden_predictor(column):
            removed_rows.append(
                {
                    "column": column,
                    "reason": "prior_pathogenicity_predictor_or_related_dbNSFP_predictor_output",
                    "used_in_old_low_circularity": "no",
                    "used_in_improved_low_circularity": "no",
                }
            )
        elif column in old_set or column in improved_set:
            if column in AF_COLUMNS:
                group = "allele_frequency"
            elif column in RAW_CONSERVATION_COLUMNS:
                group = "raw_conservation"
            elif column.startswith("consequence_") or column in {"func_refgene", "exonic_func_refgene"}:
                group = "consequence_annotation"
            elif column in {"ref_length", "alt_length", "variant_length_change", "is_snv", "is_insertion", "is_deletion"}:
                group = "variant_length_type"
            elif column == "GTEx_V8_tissue":
                group = "tissue_expression_context"
            else:
                group = "other_clean_or_legacy_parity_feature"
            kept_rows.append(
                {
                    "column": column,
                    "feature_group": group,
                    "used_in_old_low_circularity": "yes" if column in old_set else "no",
                    "used_in_improved_low_circularity": "yes" if column in improved_set else "no",
                }
            )
        else:
            if column in RISKY_NON_MODEL_COLUMNS:
                reason = "identifier_label_metadata_or_same_gene_leakage_risk"
            elif column in base_feature_set:
                reason = "usable_by_base_inference_but_not_selected_for_low_circularity"
            else:
                reason = base_spec.dropped_columns.get(column, "not_selected_or_unusable")
            risky_rows.append({"column": column, "reason": reason})

    engineered_rows = []
    for group, columns in engineered_groups.items():
        for column in columns:
            engineered_rows.append(
                {
                    "column": column,
                    "feature_group": group,
                    "used_in_old_low_circularity": "no",
                    "used_in_improved_low_circularity": "yes",
                }
            )
    kept_df = pd.concat([pd.DataFrame(kept_rows), pd.DataFrame(engineered_rows)], ignore_index=True)
    missing_df = pd.DataFrame(
        {
            "missing_feature_group": MISSING_MAGPIE_PARITY_FEATURES,
            "status": "not present in current basic_feature_table.csv; not downloaded in this run",
        }
    )
    return {
        "kept_clean_features": kept_df.sort_values(["feature_group", "column"]).reset_index(drop=True),
        "removed_prior_predictor_score_features": pd.DataFrame(removed_rows).sort_values("column").reset_index(drop=True),
        "uncertain_risky_features": pd.DataFrame(risky_rows).sort_values("column").reset_index(drop=True),
        "missing_magpie_parity_features": missing_df,
    }


def write_feature_audit(
    output_dir: Path,
    *,
    input_path: Path,
    classifications: dict[str, pd.DataFrame],
    old_feature_count: int,
    improved_feature_count: int,
    enhanced_feature_count: int,
    engineered_groups: dict[str, list[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Low-Circularity Feature Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Input feature table: `{input_path}`",
        "",
        "Status: audit plus approved low-circularity feature engineering; no synthetic data and no optional database downloads.",
        "",
        "The improved low-circularity mode keeps prior pathogenicity predictor scores removed. It does not re-add SIFT, PolyPhen, REVEL, CADD, MutationTaster, MetaSVM, MetaLR, FATHMM, PROVEAN, M-CAP, AlphaMissense, ESM, or related predictor-output families removed by the circularity audit.",
        "",
        "## Feature Counts",
        "",
    ]
    counts = pd.DataFrame(
        [
            {"feature_set": "old_low_circularity", "feature_columns": old_feature_count},
            {"feature_set": "improved_low_circularity", "feature_columns": improved_feature_count},
            {"feature_set": "enhanced_dbnsfp_mode", "feature_columns": enhanced_feature_count},
        ]
    )
    lines.extend(markdown_table(counts))
    lines.extend(["", "## Engineered Clean Feature Groups", ""])
    engineered_summary = pd.DataFrame(
        [{"group": group, "engineered_columns": len(columns)} for group, columns in engineered_groups.items()]
    )
    lines.extend(markdown_table(engineered_summary))
    for title, key in [
        ("Kept Clean Features", "kept_clean_features"),
        ("Removed Prior Predictor-Score Features", "removed_prior_predictor_score_features"),
        ("Uncertain/Risky Features", "uncertain_risky_features"),
        ("Missing MAGPIE-Parity Features", "missing_magpie_parity_features"),
    ]:
        lines.extend(["", f"## {title}", ""])
        lines.extend(markdown_table(classifications[key], max_rows=180))
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Gene identity itself is not added to the improved feature set because random splits can leak same-gene signal.",
            "- Engineered features use only the existing real feature table.",
            "- Missing MAGPIE-parity feature groups are listed as future work; they were not downloaded or fabricated.",
            "",
        ]
    )
    (output_dir / "LOW_CIRCULARITY_FEATURE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(
    output_dir: Path,
    *,
    input_path: Path,
    metrics_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    sample_sizes: list[int],
    seed: int,
    test_size: float,
    old_feature_count: int,
    improved_feature_count: int,
    enhanced_feature_count: int,
) -> None:
    summary = summarize_metric_rows(metrics_df)
    lines = [
        "# Improved Low-Circularity Baseline Report",
        "",
        "Status: completed run",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Input feature table: `{input_path}`",
        "",
        "This run uses real ClinVar-derived feature rows only. It does not create synthetic or toy results.",
        "",
        f"Metric tuple: `{METRIC_TUPLE_DEFINITION}`",
        "",
        "## Configuration",
        "",
        f"- Sample sizes: {', '.join(format(size, ',') for size in sample_sizes)}",
        "- Model family: LightGBM",
        f"- Seed: {seed}",
        "- Split: single stratified train/test split",
        f"- Test size: {test_size}",
        "- Tuning: small validation split inside the training split; selection by validation AUPRC",
        "",
        "## Compared Modes",
        "",
        f"- `old_low_circularity`: previous MAGPIE-parity feature set without prior predictor scores ({old_feature_count} features).",
        f"- `improved_low_circularity_lightgbm`: old low-circularity features plus clean engineered AF, missingness, consequence, conservation, length, and transcript-count features ({improved_feature_count} features).",
        f"- `enhanced_dbnsfp_lightgbm`: full inferred feature set including prior predictor scores for context only ({enhanced_feature_count} features).",
        "",
        "## Metric Summary",
        "",
    ]
    lines.extend(markdown_table(summary))
    lines.extend(
        [
            "",
            "Full fold-level metrics are written to `improved_low_circularity_metrics.csv`.",
            "",
            "## Selected Improved-LightGBM Parameters",
            "",
        ]
    )
    if tuning_df.empty:
        lines.append("No tuning rows available.")
    else:
        selected = tuning_df[tuning_df["selected"] == "yes"][
            ["sample_size", "candidate_id", "validation_auprc", "validation_auroc", "validation_accuracy", "params_json"]
        ]
        lines.extend(markdown_table(selected, max_rows=20))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    pivot = metrics_df.pivot_table(index="sample_size", columns="mode", values="accuracy", aggfunc="first")
    for sample_size, row in pivot.iterrows():
        old = row.get("old_low_circularity")
        improved = row.get("improved_low_circularity_lightgbm")
        enhanced = row.get("enhanced_dbnsfp_lightgbm")
        if pd.notna(old) and pd.notna(improved):
            delta = float(improved) - float(old)
            direction = "improves" if delta > 0 else "does not improve"
            lines.append(
                f"- At {int(sample_size):,}, the improved low-circularity LightGBM {direction} accuracy versus old low-circularity by {delta:+.4f} ({float(old):.4f} to {float(improved):.4f})."
            )
        if pd.notna(enhanced):
            lines.append(
                f"  Enhanced dbNSFP-context accuracy at {int(sample_size):,} is {float(enhanced):.4f}; this remains a circularity-risk reference, not the low-circularity claim."
            )
    lines.extend(
        [
            "",
            "The improved low-circularity mode is still not a final MAGPIE reproduction. It strengthens the clean baseline using only existing biological annotations and engineered features while keeping predictor-score circularity removed.",
            "",
            "## Guardrails",
            "",
            "- Do not claim final MAGPIE reproduction from this run.",
            "- Do not treat enhanced dbNSFP mode as low-circularity evidence.",
            "- Final claims still require gene-held-out, temporal, SwissProt no-exact-overlap annotation, rare-variant, calibration, and external validation.",
            "",
        ]
    )
    (output_dir / "IMPROVED_LOW_CIRCULARITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(args.input)
    feasibility = build_feasibility(df, args.sample_sizes)
    if not all(feasibility["feasible"] == "yes"):
        raise ValueError(f"One or more requested sample sizes are infeasible:\n{feasibility}")

    base_spec = infer_features(df, args.max_categorical_cardinality)
    mode_specs, removed_columns = build_mode_specs(base_spec)
    old_spec = mode_specs["magpie_parity"]
    enhanced_spec = mode_specs["magpie_x_enhanced"]
    engineered_df, improved_spec, engineered_groups = build_improved_low_circularity_spec(df, old_spec)

    for column in improved_spec.feature_columns:
        if matches_explicit_forbidden_predictor(column) or is_prior_predictor_column(column):
            raise ValueError(f"Forbidden prior predictor column present in improved low-circularity spec: {column}")

    classifications = classify_columns(
        df=df,
        base_spec=base_spec,
        old_spec=old_spec,
        improved_spec=improved_spec,
        removed_columns=removed_columns,
        engineered_groups=engineered_groups,
    )
    write_feature_audit(
        output_dir,
        input_path=args.input,
        classifications=classifications,
        old_feature_count=len(old_spec.feature_columns),
        improved_feature_count=len(improved_spec.feature_columns),
        enhanced_feature_count=len(enhanced_spec.feature_columns),
        engineered_groups=engineered_groups,
    )

    metric_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    selected_params_by_sample: dict[int, dict[str, object]] = {}

    for sample_size in args.sample_sizes:
        sample_idx_df = balanced_sample(df.reset_index(names="_source_index"), sample_size=sample_size, seed=args.seed + sample_size)
        source_indices = sample_idx_df["_source_index"].to_numpy(dtype=int)
        y = df.iloc[source_indices]["label"].to_numpy(dtype=int)
        splits = make_splits(y, seed=args.seed, test_size=args.test_size, cv_folds=1)
        fold, train_idx, test_idx = splits[0]

        old_x = df.iloc[source_indices][old_spec.feature_columns].reset_index(drop=True)
        enhanced_x = df.iloc[source_indices][enhanced_spec.feature_columns].reset_index(drop=True)
        improved_x = engineered_df.iloc[source_indices][improved_spec.feature_columns].reset_index(drop=True)

        for mode, x, spec, model, params in [
            ("old_low_circularity", old_x, old_spec, build_model("lightgbm", args.seed + fold), None),
            ("enhanced_dbnsfp_lightgbm", enhanced_x, enhanced_spec, build_model("lightgbm", args.seed + fold), None),
        ]:
            metrics, _ = evaluate_pipeline(
                x=x,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                spec=spec,
                model=model,
                ece_bins=args.ece_bins,
            )
            row = {
                "mode": mode,
                "sample_size": int(sample_size),
                "model": "lightgbm",
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_features": int(len(spec.feature_columns)),
                "selected_params_json": json.dumps(params or {}, sort_keys=True),
                **metrics,
            }
            row["result_tuple"] = result_tuple(row)
            metric_rows.append(row)
            print(f"sample_size={sample_size} mode={mode} accuracy={metrics['accuracy']:.4f} auroc={metrics['auroc']:.4f}")

        selected_params, sample_tuning_rows = tune_lightgbm(
            x=improved_x,
            y=y,
            train_idx=train_idx,
            spec=improved_spec,
            seed=args.seed + sample_size,
            ece_bins=args.ece_bins,
        )
        selected_params_by_sample[int(sample_size)] = selected_params
        best_params_json = json.dumps(selected_params, sort_keys=True)
        for tuning_row in sample_tuning_rows:
            tuning_rows.append(
                {
                    "sample_size": int(sample_size),
                    "candidate_id": tuning_row["candidate_id"],
                    "validation_auprc": tuning_row["validation_auprc"],
                    "validation_auroc": tuning_row["validation_auroc"],
                    "validation_accuracy": tuning_row["validation_accuracy"],
                    "selected": "yes" if json.dumps(tuning_row["params"], sort_keys=True) == best_params_json else "no",
                    "params_json": json.dumps(tuning_row["params"], sort_keys=True),
                }
            )

        metrics, _ = evaluate_pipeline(
            x=improved_x,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            spec=improved_spec,
            model=make_lightgbm_model(selected_params),
            ece_bins=args.ece_bins,
        )
        row = {
            "mode": "improved_low_circularity_lightgbm",
            "sample_size": int(sample_size),
            "model": "lightgbm",
            "fold": int(fold),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(len(improved_spec.feature_columns)),
            "selected_params_json": best_params_json,
            **metrics,
        }
        row["result_tuple"] = result_tuple(row)
        metric_rows.append(row)
        print(
            f"sample_size={sample_size} mode=improved_low_circularity_lightgbm "
            f"accuracy={metrics['accuracy']:.4f} auroc={metrics['auroc']:.4f}"
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["sample_size", "mode"]).reset_index(drop=True)
    tuning_df = pd.DataFrame(tuning_rows).sort_values(["sample_size", "candidate_id"]).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "improved_low_circularity_metrics.csv", index=False)
    tuning_df.to_csv(output_dir / "improved_low_circularity_tuning.csv", index=False)

    write_report(
        output_dir,
        input_path=args.input,
        metrics_df=metrics_df,
        tuning_df=tuning_df,
        sample_sizes=args.sample_sizes,
        seed=args.seed,
        test_size=args.test_size,
        old_feature_count=len(old_spec.feature_columns),
        improved_feature_count=len(improved_spec.feature_columns),
        enhanced_feature_count=len(enhanced_spec.feature_columns),
    )
    metadata = {
        "input": str(args.input),
        "output_dir": str(output_dir),
        "sample_sizes": args.sample_sizes,
        "seed": args.seed,
        "test_size": args.test_size,
        "ece_bins": args.ece_bins,
        "metric_tuple_definition": METRIC_TUPLE_DEFINITION,
        "old_low_circularity_feature_count": len(old_spec.feature_columns),
        "improved_low_circularity_feature_count": len(improved_spec.feature_columns),
        "enhanced_dbnsfp_feature_count": len(enhanced_spec.feature_columns),
        "removed_prior_predictor_columns": removed_columns,
        "engineered_feature_groups": engineered_groups,
        "selected_params_by_sample_size": selected_params_by_sample,
        "explicit_forbidden_predictor_patterns": EXPLICIT_FORBIDDEN_PREDICTORS,
        "broader_removed_predictor_patterns": PRIOR_PREDICTOR_PATTERNS,
        "synthetic_or_toy_data_used": False,
        "optional_databases_downloaded": False,
    }
    (output_dir / "improved_low_circularity_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improve and tune a low-circularity MAGPIE-style LightGBM baseline without prior predictor scores."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-categorical-cardinality", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
