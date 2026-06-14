from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "magpie_x_residual"
DEFAULT_OOF = DEFAULT_OUTPUT_DIR / "oof_tabular_predictions.csv"
DEFAULT_ALPHA = DEFAULT_OUTPUT_DIR / "alphamissense_features.csv"
DEFAULT_FEATURE_TABLE = ROOT / "data" / "interim" / "features" / "basic_feature_table.csv"

FORBIDDEN_FALLBACK_FILES = {
    (ROOT / "outputs" / "real" / "processed" / "annovar.csv").resolve(),
    (ROOT / "outputs" / "real" / "processed" / "dbnsfp.csv").resolve(),
}


def check_not_forbidden(path: Path) -> None:
    if path.resolve() in FORBIDDEN_FALLBACK_FILES:
        raise ValueError(f"Refusing to use known fallback file: {path}")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def safe_logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def is_missense_value(value: object) -> bool:
    if pd.isna(value):
        return False
    return "missense" in str(value).lower()


def load_feature_metadata(feature_table: Path) -> pd.DataFrame:
    check_not_forbidden(feature_table)
    if not feature_table.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table}")
    wanted = {"variant_id", "AF", "consequence", "gene"}
    metadata = pd.read_csv(feature_table, usecols=lambda c: c in wanted, low_memory=False)
    if "variant_id" not in metadata.columns:
        raise ValueError("Feature table must contain variant_id.")
    return metadata.drop_duplicates(subset=["variant_id"])


def load_inputs(oof_path: Path, alphamissense_path: Path, feature_table: Path) -> pd.DataFrame:
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF tabular predictions not found: {oof_path}")
    if not alphamissense_path.exists():
        raise FileNotFoundError(
            f"AlphaMissense feature table not found: {alphamissense_path}. "
            "Run scripts/prepare_alphamissense_features.py first."
        )

    oof = pd.read_csv(oof_path, low_memory=False)
    alpha = pd.read_csv(alphamissense_path, low_memory=False)
    metadata = load_feature_metadata(feature_table)

    required_oof = {"variant_id", "label", "fold", "p_tabular", "logit_tabular", "model_name"}
    missing_oof = sorted(required_oof - set(oof.columns))
    if missing_oof:
        raise ValueError(f"OOF predictions are missing required columns: {missing_oof}")
    if "variant_id" not in alpha.columns:
        raise ValueError("AlphaMissense feature table must contain variant_id.")

    merged = oof.merge(metadata, on="variant_id", how="left", suffixes=("", "_feature"))
    merged = merged.merge(alpha, on="variant_id", how="left", suffixes=("", "_alpha"))

    if "consequence" not in merged.columns and "consequence_feature" in merged.columns:
        merged["consequence"] = merged["consequence_feature"]
    if "gene" not in merged.columns and "gene_feature" in merged.columns:
        merged["gene"] = merged["gene_feature"]
    if "consequence" not in merged.columns:
        merged["consequence"] = "unknown"

    merged["alphamissense_score"] = pd.to_numeric(merged.get("alphamissense_score"), errors="coerce")
    merged["alphamissense_available"] = merged.get("alphamissense_available", False)
    merged["alphamissense_available"] = merged["alphamissense_available"].fillna(False).astype(bool)

    missense_mask = merged["consequence"].map(is_missense_value)
    merged.loc[~missense_mask, "alphamissense_score"] = np.nan
    merged.loc[~missense_mask, "alphamissense_available"] = False
    return add_handcrafted_features(merged)


def af_bin(value: object) -> str:
    af = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(af):
        return "missing"
    if af == 0:
        return "zero"
    if af < 1e-5:
        return "ultra_rare_lt_1e-5"
    if af < 1e-3:
        return "rare_lt_1e-3"
    if af < 1e-2:
        return "low_frequency_lt_1e-2"
    return "common_ge_1e-2"


def add_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_tabular"] = pd.to_numeric(out["p_tabular"], errors="coerce")
    out["logit_tabular"] = pd.to_numeric(out["logit_tabular"], errors="coerce")
    missing_logit = out["logit_tabular"].isna() & out["p_tabular"].notna()
    out.loc[missing_logit, "logit_tabular"] = safe_logit(out.loc[missing_logit, "p_tabular"].to_numpy())
    out["uncertainty_tabular"] = pd.to_numeric(
        out.get("uncertainty_tabular", (out["p_tabular"] - 0.5).abs()),
        errors="coerce",
    )
    out["abs_logit_tabular"] = out["logit_tabular"].abs()
    out["AF_numeric"] = pd.to_numeric(out.get("AF"), errors="coerce")
    out["af_missing"] = out["AF_numeric"].isna()
    out["af_bin"] = out["AF_numeric"].map(af_bin)
    out["alphamissense_missing"] = ~out["alphamissense_available"]
    out["variant_type"] = out["consequence"].fillna("unknown").astype(str)
    out["is_missense"] = out["variant_type"].map(is_missense_value)
    return out


def build_logistic_correction(seed: int) -> Pipeline:
    numeric_columns = [
        "logit_tabular",
        "uncertainty_tabular",
        "abs_logit_tabular",
        "alphamissense_score",
        "alphamissense_available",
        "alphamissense_missing",
        "AF_numeric",
        "af_missing",
        "is_missense",
    ]
    categorical_columns = ["variant_type", "af_bin"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def build_xgboost_correction(seed: int) -> Pipeline:
    try:
        import xgboost
    except ImportError as exc:
        raise ImportError("xgboost is required for --models xgboost. Install it with: pip install xgboost") from exc

    pipeline = build_logistic_correction(seed)
    pipeline.steps[-1] = (
        "model",
        xgboost.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        ),
    )
    return pipeline


def build_correction_model(model_name: str, seed: int) -> Pipeline:
    if model_name == "logistic":
        return build_logistic_correction(seed)
    if model_name == "xgboost":
        return build_xgboost_correction(seed)
    raise ValueError(f"Unknown correction model: {model_name}")


def predict_probability(pipeline: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x_test)[:, 1]
    decision = pipeline.decision_function(x_test)
    return sigmoid(decision)


def feature_columns() -> list[str]:
    return [
        "logit_tabular",
        "uncertainty_tabular",
        "abs_logit_tabular",
        "variant_type",
        "alphamissense_score",
        "alphamissense_available",
        "alphamissense_missing",
        "AF_numeric",
        "af_missing",
        "af_bin",
        "is_missense",
    ]


def run_correction_for_base_model(df: pd.DataFrame, correction_model: str, seed: int) -> pd.DataFrame:
    y = df["label"].astype(int).to_numpy()
    rows: list[pd.DataFrame] = []
    columns = feature_columns()

    for fold in sorted(df["fold"].dropna().unique()):
        test_mask = df["fold"] == fold
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        model = build_correction_model(correction_model, seed + int(fold))
        model.fit(df.loc[train_mask, columns], y[train_mask.to_numpy()])

        p_final = predict_probability(model, df.loc[test_mask, columns])
        z_final = safe_logit(p_final)
        logit_tabular = pd.to_numeric(df.loc[test_mask, "logit_tabular"], errors="coerce").to_numpy(dtype=float)
        delta = z_final - logit_tabular

        keep = [
            column
            for column in [
                "variant_id",
                "label",
                "fold",
                "model_name",
                "consequence",
                "gene",
                "p_tabular",
                "logit_tabular",
                "uncertainty_tabular",
                "variant_type",
                "AF_numeric",
                "af_bin",
                "af_missing",
                "alphamissense_score",
                "alphamissense_available",
                "alphamissense_missing",
            ]
            if column in df.columns
        ]
        out = df.loc[test_mask, keep].copy()
        out = out.rename(columns={"model_name": "base_model_name"})
        out["correction_model"] = correction_model
        out["residual"] = out["label"].astype(float) - out["p_tabular"].astype(float)
        out["delta"] = delta
        out["z_final"] = z_final
        out["p_final"] = p_final
        out["expert_feature_state"] = np.where(
            out["alphamissense_available"],
            "alphamissense_available",
            "no_alphamissense_for_variant",
        )
        rows.append(out)
        print(f"{out['base_model_name'].iloc[0]} | {correction_model} | residual fold {fold} complete")

    return pd.concat(rows, ignore_index=True)


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = load_inputs(args.oof_predictions, args.alphamissense_features, args.feature_table)
    all_predictions: list[pd.DataFrame] = []

    for _, base_df in merged.groupby("model_name", sort=True):
        for correction_model in args.models:
            predictions = run_correction_for_base_model(base_df.reset_index(drop=True), correction_model, args.seed)
            all_predictions.append(predictions)

    if not all_predictions:
        raise ValueError("No residual predictions were generated.")
    output = pd.concat(all_predictions, ignore_index=True)
    output_path = args.output_dir / "residual_predictions.csv"
    output.to_csv(output_path, index=False)

    metadata = {
        "oof_predictions": str(args.oof_predictions),
        "alphamissense_features": str(args.alphamissense_features),
        "feature_table": str(args.feature_table),
        "output": str(output_path),
        "correction_models": args.models,
        "seed": args.seed,
        "feature_columns": feature_columns(),
        "alphamissense_available_rows": int(output["alphamissense_available"].sum()),
        "note": "z_final = z_tabular + delta, where delta is learned from AlphaMissense score, variant type, uncertainty, AF-bin features, and missingness flags. This stage does not train a GNN.",
    }
    write_metadata(args.output_dir / "residual_training_metadata.json", metadata)
    print(f"Saved residual correction predictions to {output_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the immediate MAGPIE-X logit-level residual correction model on the reference 5k benchmark."
    )
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--alphamissense-features", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", choices=["logistic", "xgboost"], default=["logistic"])
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
