from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "real" / "processed" / "clinvar_exonic_nonsynonymous_balanced.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "real" / "baseline_clinvar_only"

NUMERIC_FEATURES = [
    "pos",
    "ref_length",
    "alt_length",
    "variant_length_change",
    "is_snv",
    "is_insertion",
    "is_deletion",
]

BASE_CATEGORICAL_FEATURES = ["chrom", "ref", "alt", "consequence"]
GENE_FEATURE = "gene"

FORBIDDEN_SYNTHETIC_FILES = [
    ROOT / "outputs" / "real" / "processed" / "annovar.csv",
    ROOT / "outputs" / "real" / "processed" / "dbnsfp.csv",
]


@dataclass(frozen=True)
class SplitData:
    split_name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    heldout_genes: set[str] | None = None


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_one_hot_encoder(*, sparse: bool) -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)


def load_clinvar_only_table(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, nrows=max_rows, dtype={"chrom": str}, low_memory=False)
    required = {"variant_id", "chrom", "pos", "ref", "alt", "consequence", "label", "gene"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    return df


def add_clinvar_only_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["chrom", "ref", "alt", "consequence", "gene"]:
        out[col] = out[col].fillna("UNKNOWN").astype(str)

    out["pos"] = pd.to_numeric(out["pos"], errors="coerce")
    out["pos"] = out["pos"].fillna(out["pos"].median() if out["pos"].notna().any() else 0.0)

    out["ref_length"] = out["ref"].str.len().astype(int)
    out["alt_length"] = out["alt"].str.len().astype(int)
    out["variant_length_change"] = out["alt_length"] - out["ref_length"]
    out["is_snv"] = ((out["ref_length"] == 1) & (out["alt_length"] == 1)).astype(int)
    out["is_insertion"] = (out["variant_length_change"] > 0).astype(int)
    out["is_deletion"] = (out["variant_length_change"] < 0).astype(int)
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out[out["label"].isin([0, 1])].copy()
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)


def random_stratified_split(y: np.ndarray, seed: int) -> SplitData:
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )
    return SplitData("random_split", np.sort(train_idx), np.sort(test_idx))


def gene_heldout_split(df: pd.DataFrame, seed: int) -> SplitData:
    rng = np.random.default_rng(seed)
    genes = np.array(sorted(df[GENE_FEATURE].fillna("UNKNOWN").astype(str).unique()))
    rng.shuffle(genes)
    n_holdout = max(1, int(round(0.2 * len(genes))))
    heldout = set(genes[:n_holdout])
    gene_series = df[GENE_FEATURE].fillna("UNKNOWN").astype(str)
    test_idx = df.index[gene_series.isin(heldout)].to_numpy()
    train_idx = df.index[~gene_series.isin(heldout)].to_numpy()
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Gene-held-out split produced an empty train or test set.")
    overlap = set(df.loc[train_idx, GENE_FEATURE].astype(str)).intersection(set(df.loc[test_idx, GENE_FEATURE].astype(str)))
    if overlap:
        raise ValueError(f"Gene-held-out split has gene overlap: {sorted(list(overlap))[:5]}")
    return SplitData("gene_heldout", np.sort(train_idx), np.sort(test_idx), heldout_genes=heldout)


def sparse_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(sparse=True), categorical_features),
            ("numeric", StandardScaler(with_mean=False), NUMERIC_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def hgb_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    ordinal_features = [c for c in categorical_features if c != "consequence"]
    return ColumnTransformer(
        transformers=[
            (
                "categorical_ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ordinal_features,
            ),
            ("consequence_onehot", make_one_hot_encoder(sparse=False), ["consequence"]),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_models(categorical_features: list[str], seed: int) -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocess", sparse_preprocessor(categorical_features)),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        max_iter=1000,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            steps=[
                ("preprocess", sparse_preprocessor(categorical_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=seed,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
        "HistGradientBoostingClassifier": Pipeline(
            steps=[
                ("preprocess", hgb_preprocessor(categorical_features)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.06,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def safe_score(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if metric_name == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric_name == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        if metric_name == "precision":
            return float(precision_score(y_true, y_pred, zero_division=0))
        if metric_name == "recall":
            return float(recall_score(y_true, y_pred, zero_division=0))
        if metric_name == "f1":
            return float(f1_score(y_true, y_pred, zero_division=0))
        if metric_name == "auroc":
            return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
        if metric_name == "auprc":
            return float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
        if metric_name == "mcc":
            return float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        return float("nan")
    raise ValueError(f"Unknown metric: {metric_name}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc", "auprc", "mcc"]
    metrics = {name: safe_score(name, y_true, y_pred, y_prob) for name in metric_names}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def predict_probability(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    decision = model.decision_function(x_test)
    return 1.0 / (1.0 + np.exp(-decision))


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=160)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["benign", "pathogenic"])
    ax.set_yticks([0, 1], labels=["benign", "pathogenic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def feature_modes() -> dict[str, list[str]]:
    return {
        "without_gene_features": BASE_CATEGORICAL_FEATURES.copy(),
        "with_gene_features": [*BASE_CATEGORICAL_FEATURES, GENE_FEATURE],
    }


def append_predictions(
    rows: list[pd.DataFrame],
    source_df: pd.DataFrame,
    test_idx: np.ndarray,
    split_name: str,
    feature_mode: str,
    model_name: str,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    keep_cols = [c for c in ["variant_id", "gene", "chrom", "pos", "ref", "alt", "consequence", "label"] if c in source_df.columns]
    pred = source_df.loc[test_idx, keep_cols].copy()
    pred["split_type"] = split_name
    pred["feature_mode"] = feature_mode
    pred["model"] = model_name
    pred["probability_pathogenic"] = y_prob
    pred["predicted_label"] = y_pred
    rows.append(pred)


def write_report(
    output_dir: Path,
    input_path: Path,
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    split_summaries: dict[str, dict[str, object]],
    max_rows: int | None,
) -> None:
    report = output_dir / "BASELINE_REPORT.md"
    best_rows = metrics.sort_values(["split_type", "feature_mode", "model"]).copy()
    metric_preview = best_rows[
        ["split_type", "feature_mode", "model", "balanced_accuracy", "f1", "auroc", "auprc", "mcc"]
    ]

    lines: list[str] = []
    lines.append("# ClinVar-Only Baseline Report")
    lines.append("")
    lines.append("This is a ClinVar-only baseline, not the final MAGPIE-style model.")
    lines.append("")
    lines.append("The script used only real ClinVar-derived columns from:")
    lines.append("")
    lines.append(f"```text\n{input_path}\n```")
    lines.append("")
    lines.append("It did not read or use `outputs/real/processed/annovar.csv` or `outputs/real/processed/dbnsfp.csv`; those files are synthetic fallback files in the current repository state.")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Rows used: {len(df):,}")
    lines.append(f"- Benign / likely benign: {(df['label'] == 0).sum():,}")
    lines.append(f"- Pathogenic / likely pathogenic: {(df['label'] == 1).sum():,}")
    if max_rows is not None:
        lines.append(f"- Debug row limit was active: {max_rows:,}")
    lines.append("")
    lines.append("## Feature Modes")
    lines.append("")
    lines.append("- `without_gene_features`: excludes the `gene` column.")
    lines.append("- `with_gene_features`: includes `gene` as a categorical feature. This is a possible gene-identity leakage experiment and should not be interpreted as the cleanest estimate of generalization.")
    lines.append("")
    lines.append("## Splits")
    lines.append("")
    lines.append("- `random_split`: stratified 80/20 row-level split. This may overestimate performance because nearby variants and variants from the same genes can appear in both train and test.")
    lines.append("- `gene_heldout`: holds out 20% of genes completely for testing. This is more realistic because it tests unseen genes and enforces no gene overlap between train and test.")
    lines.append("")
    for split_name, summary in split_summaries.items():
        lines.append(f"### {split_name}")
        lines.append("")
        lines.append(f"- Train rows: {summary['train_rows']:,}")
        lines.append(f"- Test rows: {summary['test_rows']:,}")
        lines.append(f"- Train genes: {summary['train_genes']:,}")
        lines.append(f"- Test genes: {summary['test_genes']:,}")
        lines.append(f"- Gene overlap: {summary['gene_overlap']:,}")
        lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.extend(markdown_table(metric_preview))
    lines.append("")
    lines.append("## Missing MAGPIE Features")
    lines.append("")
    lines.append("Real ANNOVAR and dbNSFP features are still missing. A true MAGPIE-style experiment still needs real annotation tables for consequence-normalized exonic context, allele frequencies, conservation, gene intolerance, SpliceAI, chromatin/epigenetic context where appropriate, and precomputed missense expert scores such as AlphaMissense, ESM1b, REVEL, VARITY, and MutPred.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `metrics.csv`")
    lines.append("- `predictions_random_split.csv`")
    lines.append("- `predictions_gene_heldout.csv`")
    lines.append("- `confusion_matrices/`")
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> list[str]:
    columns = df.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def run(input_path: Path, output_dir: Path, seed: int, max_rows: int | None = None) -> None:
    output_dir = ensure_dir(output_dir)
    cm_dir = ensure_dir(output_dir / "confusion_matrices")
    df = add_clinvar_only_features(load_clinvar_only_table(input_path, max_rows=max_rows))
    if len(df) == 0:
        raise ValueError("No valid rows with binary labels were found.")
    y = df["label"].to_numpy(dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Both benign and pathogenic labels are required.")

    splits = [
        random_stratified_split(y, seed),
        gene_heldout_split(df, seed),
    ]

    metrics_rows: list[dict[str, object]] = []
    predictions_by_split: dict[str, list[pd.DataFrame]] = {"random_split": [], "gene_heldout": []}
    split_summaries: dict[str, dict[str, object]] = {}

    for split in splits:
        train_genes = set(df.loc[split.train_idx, GENE_FEATURE].astype(str))
        test_genes = set(df.loc[split.test_idx, GENE_FEATURE].astype(str))
        split_summaries[split.split_name] = {
            "train_rows": int(len(split.train_idx)),
            "test_rows": int(len(split.test_idx)),
            "train_genes": int(len(train_genes)),
            "test_genes": int(len(test_genes)),
            "gene_overlap": int(len(train_genes.intersection(test_genes))),
        }

        for feature_mode, categorical_features in feature_modes().items():
            x_train = df.iloc[split.train_idx]
            x_test = df.iloc[split.test_idx]
            y_train = y[split.train_idx]
            y_test = y[split.test_idx]
            for model_name, model in build_models(categorical_features, seed).items():
                print(f"Training {model_name} | {feature_mode} | {split.split_name}")
                model.fit(x_train, y_train)
                y_prob = predict_probability(model, x_test)
                y_pred = (y_prob >= 0.5).astype(int)
                row = {
                    "split_type": split.split_name,
                    "feature_mode": feature_mode,
                    "model": model_name,
                    "n_train": int(len(split.train_idx)),
                    "n_test": int(len(split.test_idx)),
                    "n_train_genes": split_summaries[split.split_name]["train_genes"],
                    "n_test_genes": split_summaries[split.split_name]["test_genes"],
                    "gene_overlap": split_summaries[split.split_name]["gene_overlap"],
                    **compute_metrics(y_test, y_pred, y_prob),
                }
                metrics_rows.append(row)
                append_predictions(
                    predictions_by_split[split.split_name],
                    df,
                    split.test_idx,
                    split.split_name,
                    feature_mode,
                    model_name,
                    y_prob,
                    y_pred,
                )
                safe_name = f"{split.split_name}__{feature_mode}__{model_name}.png"
                save_confusion_matrix_plot(
                    y_test,
                    y_pred,
                    cm_dir / safe_name,
                    title=f"{model_name}\n{feature_mode} | {split.split_name}",
                )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    for split_name, rows in predictions_by_split.items():
        predictions = pd.concat(rows, ignore_index=True)
        predictions.to_csv(output_dir / f"predictions_{split_name}.csv", index=False)

    write_json(
        {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "seed": seed,
            "max_rows": max_rows,
            "forbidden_synthetic_files_not_used": [str(p) for p in FORBIDDEN_SYNTHETIC_FILES],
            "split_summaries": split_summaries,
        },
        output_dir / "run_metadata.json",
    )
    write_report(output_dir, input_path, df, metrics, split_summaries, max_rows)
    print(f"Baseline complete. Outputs written to {output_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ClinVar-only baseline models without synthetic ANNOVAR/dbNSFP fallback features.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="ClinVar-only filtered balanced CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory for baseline artifacts.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional debug limit; omit for the full dataset.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args.input, args.output_dir, args.seed, max_rows=args.max_rows)


if __name__ == "__main__":
    main(sys.argv[1:])
