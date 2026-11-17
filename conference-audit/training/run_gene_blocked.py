"""Leakage-controlled MAGPIE-X evaluation on the audited 20k sample.

This portable entry point reads an explicitly supplied research-data root and
writes to an explicitly supplied output directory. It reconstructs GENEINFO from
the exact source VCF, joins multi-gene records through connected components,
and therefore prevents every reported gene symbol from crossing an outer
gene-blocked fold.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings("ignore", message="X does not have valid feature names")


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPTS = SOURCE_ROOT / "scripts"
SOURCE_SRC = SOURCE_ROOT / "src"
sys.path.insert(0, str(SOURCE_SCRIPTS))
sys.path.insert(0, str(SOURCE_SRC))

from run_aa_enhanced_low_circularity_comparison import (  # noqa: E402
    AA_FEATURE_COLUMNS,
    build_aa_spec,
    load_aa_columns,
)
from run_circularity_ablation import build_mode_specs  # noqa: E402
from run_magpie_low_circularity_baseline import (  # noqa: E402
    build_improved_low_circularity_spec,
    evaluate_pipeline,
    lightgbm_candidates,
    make_lightgbm_model,
)
from run_sample_size_benchmark import (  # noqa: E402
    FeatureSpec,
    balanced_sample,
    compute_metrics,
    infer_features,
    load_feature_table,
)


BASE_INPUT = AA_INPUT = SOURCE_VCF = ALPHAMISSENSE_20K = None


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0
            return value
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def parse_geneinfo(info: str) -> tuple[str, ...]:
    match = re.search(r"(?:^|;)GENEINFO=([^;]+)", info)
    if not match:
        return ()
    symbols: list[str] = []
    for item in re.split(r"[|,;]+", match.group(1)):
        symbol = item.split(":", 1)[0].strip()
        if symbol and symbol != "." and symbol not in symbols:
            symbols.append(symbol)
    return tuple(symbols)


def reconstruct_gene_components(
    base_df: pd.DataFrame,
    vcf_path: Path,
) -> tuple[pd.Series, dict[str, object]]:
    keys = set(base_df["variant_id"].astype(str))
    genes_by_variant: dict[str, tuple[str, ...]] = {}
    dsu = DisjointSet()
    with gzip.open(vcf_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, ref, alt_field, info = fields[0], fields[1], fields[3], fields[4], fields[7]
            genes = parse_geneinfo(info)
            for alt in alt_field.split(","):
                key = f"{chrom}:{pos}:{ref}>{alt}"
                if key not in keys:
                    continue
                genes_by_variant[key] = genes
                if genes:
                    dsu.find(genes[0])
                    for gene in genes[1:]:
                        dsu.union(genes[0], gene)

    missing = sorted(keys - set(genes_by_variant))
    empty = [key for key, genes in genes_by_variant.items() if not genes]
    if missing or empty:
        raise RuntimeError(
            f"Gene reconstruction is incomplete: unmatched={len(missing)}, empty_gene={len(empty)}"
        )

    component_genes: dict[str, set[str]] = defaultdict(set)
    for genes in genes_by_variant.values():
        for gene in genes:
            component_genes[dsu.find(gene)].add(gene)

    component_ids: dict[str, str] = {}
    for root, genes in component_genes.items():
        signature = "|".join(sorted(genes))
        component_ids[root] = "GC_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]

    variant_components = {
        key: component_ids[dsu.find(genes[0])]
        for key, genes in genes_by_variant.items()
    }
    components = base_df["variant_id"].astype(str).map(variant_components)
    if components.isna().any():
        raise RuntimeError("At least one prepared variant lacks a reconstructed gene component.")

    primary_matches = sum(
        str(primary) == genes_by_variant[str(key)][0]
        for key, primary in zip(base_df["variant_id"], base_df["gene"], strict=True)
    )
    component_variant_counts = Counter(variant_components.values())
    component_gene_counts = [len(genes) for genes in component_genes.values()]
    gene_count_distribution = Counter(len(genes) for genes in genes_by_variant.values())
    audit: dict[str, object] = {
        "prepared_variants": int(len(base_df)),
        "matched_to_exact_source_vcf": int(len(genes_by_variant)),
        "source_vcf": str(vcf_path),
        "source_vcf_release": "2026-05-03",
        "assembly": "GRCh38",
        "primary_gene_matches_parser": int(primary_matches),
        "unique_reported_gene_symbols": int(len(dsu.parent)),
        "multi_gene_variants": int(sum(len(genes) > 1 for genes in genes_by_variant.values())),
        "gene_count_per_variant": {str(k): int(v) for k, v in sorted(gene_count_distribution.items())},
        "connected_gene_components": int(len(component_genes)),
        "largest_component_gene_count": int(max(component_gene_counts)),
        "largest_component_variant_count": int(max(component_variant_counts.values())),
        "component_rule": (
            "All GENEINFO symbols on a variant are unioned; the full connected component is the "
            "StratifiedGroupKFold group. This is stricter than using the parser's first gene."
        ),
    }
    return components.astype(str), audit


def prepare_clean_features() -> tuple[pd.DataFrame, pd.DataFrame, FeatureSpec, dict[str, list[str]]]:
    base_df = load_feature_table(BASE_INPUT)
    aa_columns = load_aa_columns(AA_INPUT, base_df)
    base_spec = infer_features(base_df, max_categorical_cardinality=50)
    mode_specs, _ = build_mode_specs(base_spec)
    combined, improved_spec, engineered_groups = build_improved_low_circularity_spec(
        base_df, mode_specs["magpie_parity"]
    )
    combined = pd.concat([combined, aa_columns], axis=1)
    clean_spec = build_aa_spec(improved_spec)
    if len(clean_spec.feature_columns) != 114:
        raise RuntimeError(f"Expected 114 clean features, found {len(clean_spec.feature_columns)}")
    return base_df, combined, clean_spec, engineered_groups


def candidate_score(
    x: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    spec: FeatureSpec,
    groups: np.ndarray | None,
    grouped: bool,
    seed: int,
    ece_bins: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx])
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(train_idx)), y[train_idx])
    # Match the established protocol's one 80/20 validation split, but keep the
    # validation group-disjoint for the gene-blocked evaluation.
    first_train, first_val = next(iterator)
    inner_splits = [(train_idx[first_train], train_idx[first_val])]

    rows: list[dict[str, object]] = []
    best_params: dict[str, object] | None = None
    best_score = -math.inf
    for candidate_id, params in enumerate(lightgbm_candidates(seed), start=1):
        fold_scores: list[float] = []
        for inner_train, inner_val in inner_splits:
            metrics, _ = evaluate_pipeline(
                x=x,
                y=y,
                train_idx=np.sort(inner_train),
                test_idx=np.sort(inner_val),
                spec=spec,
                model=make_lightgbm_model(params),
                ece_bins=ece_bins,
            )
            fold_scores.append(float(metrics["auprc"]))
        mean_score = float(np.mean(fold_scores))
        row = {
            "candidate_id": candidate_id,
            "mean_inner_auprc": mean_score,
            # A single pre-specified 80/20 validation split mirrors the
            # established pipeline, so a within-candidate SD is undefined.
            "sd_inner_auprc": np.nan,
            "fold_auprc_json": json.dumps(fold_scores),
            "params_json": json.dumps(params, sort_keys=True),
        }
        rows.append(row)
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)
    assert best_params is not None
    return best_params, rows


def make_outer_splits(
    y: np.ndarray,
    groups: np.ndarray,
    grouped: bool,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        return [(a, b) for a, b in splitter.split(np.zeros(len(y)), y, groups)]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(a, b) for a, b in splitter.split(np.zeros(len(y)), y)]


def run_nested_oof(
    *,
    name: str,
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    spec: FeatureSpec,
    grouped: bool,
    seed: int,
    ece_bins: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[tuple[np.ndarray, np.ndarray]]]:
    oof = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    aggregate_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    outer_splits = make_outer_splits(y, groups, grouped, seed)
    for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        if grouped:
            overlap = set(groups[train_idx]) & set(groups[test_idx])
            if overlap:
                raise RuntimeError(f"Outer fold {fold} has {len(overlap)} component overlaps.")
        params, candidate_rows = candidate_score(
            x=x,
            y=y,
            train_idx=train_idx,
            spec=spec,
            groups=groups,
            grouped=grouped,
            seed=seed + fold,
            ece_bins=ece_bins,
        )
        selected_json = json.dumps(params, sort_keys=True)
        for row in candidate_rows:
            tuning_rows.append(
                {
                    "evaluation": name,
                    "outer_fold": fold,
                    **row,
                    "selected": row["params_json"] == selected_json,
                }
            )
        metrics, probabilities = evaluate_pipeline(
            x=x,
            y=y,
            train_idx=np.sort(train_idx),
            test_idx=np.sort(test_idx),
            spec=spec,
            model=make_lightgbm_model(params),
            ece_bins=ece_bins,
        )
        oof[test_idx] = probabilities
        fold_ids[test_idx] = fold
        aggregate_rows.append(
            {
                "evaluation": name,
                "scope": "fold",
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_prevalence": float(np.mean(y[test_idx])),
                "component_overlap": 0 if grouped else int(len(set(groups[train_idx]) & set(groups[test_idx]))),
                **metrics,
            }
        )
        selected.append(params)
        print(
            f"{name} fold={fold} acc={metrics['accuracy']:.4f} "
            f"mcc={metrics['mcc']:.4f} auroc={metrics['auroc']:.4f}",
            flush=True,
        )
    if np.isnan(oof).any() or (fold_ids < 1).any():
        raise RuntimeError(f"Incomplete OOF predictions for {name}.")
    overall = compute_metrics(y, oof, ece_bins)
    crossing_components = int(
        sum(len(np.unique(fold_ids[groups == group])) > 1 for group in np.unique(groups))
    )
    aggregate_rows.append(
        {
            "evaluation": name,
            "scope": "aggregate_oof",
            "fold": 0,
            "n_train": np.nan,
            "n_test": int(len(y)),
            "test_prevalence": float(np.mean(y)),
            "component_overlap": crossing_components,
            **overall,
        }
    )
    return oof, fold_ids, aggregate_rows, tuning_rows, selected, outer_splits


def metric_subset(y: np.ndarray, p: np.ndarray, ece_bins: int) -> dict[str, float]:
    raw = compute_metrics(y, p, ece_bins)
    return {
        key: float(raw[key])
        for key in ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "mcc", "auroc", "auprc", "brier", "ece"]
    }


def paired_component_bootstrap(
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    ece_bins: int,
) -> pd.DataFrame:
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    metric_names = ["accuracy", "mcc", "auroc", "auprc", "brier", "ece"]
    values = {name: [] for name in metric_names}
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_idx = np.concatenate([indices[group] for group in sampled_groups])
        if np.unique(y[sampled_idx]).size < 2:
            continue
        a = metric_subset(y[sampled_idx], left[sampled_idx], ece_bins)
        b = metric_subset(y[sampled_idx], right[sampled_idx], ece_bins)
        for name in metric_names:
            values[name].append(a[name] - b[name])
    rows = []
    for name in metric_names:
        arr = np.asarray(values[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "left_minus_right": metric_subset(y, left, ece_bins)[name]
                - metric_subset(y, right, ece_bins)[name],
                "bootstrap_replicates": int(len(arr)),
                "ci95_low": float(np.quantile(arr, 0.025)),
                "ci95_high": float(np.quantile(arr, 0.975)),
                "two_sided_p": float(
                    min(1.0, 2.0 * min((np.sum(arr <= 0) + 1) / (len(arr) + 1), (np.sum(arr >= 0) + 1) / (len(arr) + 1)))
                ),
            }
        )
    return pd.DataFrame(rows)


def feature_families(feature_columns: list[str]) -> dict[str, list[str]]:
    aa = set(AA_FEATURE_COLUMNS)
    families: dict[str, list[str]] = {
        "population_frequency": [],
        "conservation": [],
        "functional_consequence": [],
        "variant_geometry": [],
        "amino_acid_blosum": [],
    }
    for column in feature_columns:
        low = column.lower()
        if column in aa:
            families["amino_acid_blosum"].append(column)
        elif low.startswith("af") or "allele_frequency" in low or low.startswith(("max_af", "min_af", "mean_af", "median_af", "nonmissing_af", "all_af", "rare_af")):
            families["population_frequency"].append(column)
        elif any(token in low for token in ["gerp", "phylop", "phastcons", "siphy", "fitcons", "conservation", "integrated_confidence"]):
            families["conservation"].append(column)
        elif low.startswith("consequence_") or low in {
            "func_refgene",
            "exonic_func_refgene",
            "is_splice_consequence",
            "is_stop_consequence",
            "is_missense_consequence",
        } or "func_exonic" in low or "transcript_count" in low or "multiple_transcripts" in low:
            families["functional_consequence"].append(column)
        elif any(token in low for token in ["ref_length", "alt_length", "length_change", "length_ratio", "is_snv", "is_insertion", "is_deletion", "is_indel", "length_preserving"]):
            families["variant_geometry"].append(column)
    if any(not members for members in families.values()):
        raise RuntimeError(f"Empty feature family detected: {families}")
    return families


def ablated_spec(spec: FeatureSpec, removed: set[str]) -> FeatureSpec:
    return FeatureSpec(
        feature_columns=[c for c in spec.feature_columns if c not in removed],
        numeric_columns=[c for c in spec.numeric_columns if c not in removed],
        categorical_columns=[c for c in spec.categorical_columns if c not in removed],
        dropped_columns={**spec.dropped_columns, **{c: "feature_family_ablation" for c in removed}},
    )


def run_ablations(
    *,
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    spec: FeatureSpec,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    selected_params: list[dict[str, object]],
    full_metrics: dict[str, float],
    ece_bins: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    families = feature_families(spec.feature_columns)
    rows: list[dict[str, object]] = []
    for family, members in families.items():
        family_spec = ablated_spec(spec, set(members))
        oof = np.full(len(y), np.nan)
        for fold, ((train_idx, test_idx), params) in enumerate(zip(outer_splits, selected_params, strict=True), start=1):
            if set(groups[train_idx]) & set(groups[test_idx]):
                raise RuntimeError(f"Ablation {family} fold {fold} violates component separation.")
            _, probabilities = evaluate_pipeline(
                x=x,
                y=y,
                train_idx=np.sort(train_idx),
                test_idx=np.sort(test_idx),
                spec=family_spec,
                model=make_lightgbm_model(params),
                ece_bins=ece_bins,
            )
            oof[test_idx] = probabilities
        metrics = metric_subset(y, oof, ece_bins)
        rows.append(
            {
                "mode": f"drop_{family}",
                "removed_features": len(members),
                "remaining_features": len(family_spec.feature_columns),
                **metrics,
                "delta_accuracy_vs_full": metrics["accuracy"] - full_metrics["accuracy"],
                "delta_mcc_vs_full": metrics["mcc"] - full_metrics["mcc"],
                "delta_auroc_vs_full": metrics["auroc"] - full_metrics["auroc"],
                "delta_auprc_vs_full": metrics["auprc"] - full_metrics["auprc"],
            }
        )
        print(f"ablation={family} auroc={metrics['auroc']:.4f}", flush=True)
    return pd.DataFrame(rows), families


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    rows = []
    for b in range(n_bins):
        idx = bins == b
        rows.append(
            {
                "bin": b + 1,
                "lower": edges[b],
                "upper": edges[b + 1],
                "n": int(idx.sum()),
                "mean_probability": float(np.mean(p[idx])) if idx.any() else np.nan,
                "observed_fraction": float(np.mean(y[idx])) if idx.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_natural_prevalence(
    *,
    base_df: pd.DataFrame,
    combined: pd.DataFrame,
    components: pd.Series,
    spec: FeatureSpec,
    output_dir: Path,
    ece_bins: int,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled = base_df.reset_index(names="_source_index").sample(n=20_000, replace=False, random_state=31_013)
    source_indices = sampled["_source_index"].to_numpy(dtype=int)
    y = base_df.iloc[source_indices]["label"].to_numpy(dtype=int)
    groups = components.iloc[source_indices].to_numpy(dtype=str)
    x = combined.iloc[source_indices][spec.feature_columns].reset_index(drop=True)
    meta = base_df.iloc[source_indices][["variant_id"]].reset_index(drop=True)
    splits = make_outer_splits(y, groups, grouped=True, seed=31_013)
    oof = np.full(len(y), np.nan)
    folds = np.full(len(y), -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        params = lightgbm_candidates(31_013 + fold)[4]
        _, probabilities = evaluate_pipeline(
            x=x,
            y=y,
            train_idx=np.sort(train_idx),
            test_idx=np.sort(test_idx),
            spec=spec,
            model=make_lightgbm_model(params),
            ece_bins=ece_bins,
        )
        oof[test_idx] = probabilities
        folds[test_idx] = fold
    metrics = metric_subset(y, oof, ece_bins)
    pred = meta.assign(label=y, probability=oof, predicted=(oof >= 0.5).astype(int), fold=folds, component_id=groups)
    pred.to_csv(output_dir / "natural_prevalence_gene_blocked_predictions.csv", index=False)
    reliability_table(y, oof, ece_bins).to_csv(output_dir / "natural_prevalence_reliability.csv", index=False)
    payload = {
        **metrics,
        "n": int(len(y)),
        "pathogenic_fraction": float(np.mean(y)),
        "sampling": "unstratified 20,000-row sample from the 490,403-row prepared eligible table",
        "split": "five-fold connected-gene-component StratifiedGroupKFold",
        "model_selection": "pre-specified LightGBM candidate 5; no tuning on this secondary analysis",
        "interpretation": "prepared-table prevalence sensitivity; not a clinical prevalence estimate",
    }
    (output_dir / "natural_prevalence_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    global BASE_INPUT, AA_INPUT, SOURCE_VCF, ALPHAMISSENSE_20K
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-only", action="store_true", help="Rebuild components and refit both nested OOF arms; skip ablations/comparators/secondary analyses.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1_000)
    args = parser.parse_args()
    BASE_INPUT = args.data_root / "data/interim/features/basic_feature_table.csv"
    AA_INPUT = args.data_root / "data/interim/features/basic_feature_table_with_clean_aa_features.csv"
    SOURCE_VCF = args.data_root / "data/raw/clinvar.vcf.gz"
    ALPHAMISSENSE_20K = args.data_root / "outputs/real/magpie_x_enhanced/alphamissense_20k_features.csv"
    for source in (BASE_INPUT, AA_INPUT, SOURCE_VCF):
        if not source.is_file():
            parser.error(f"Missing required input: {source.name}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    base_df, combined, clean_spec, engineered_groups = prepare_clean_features()
    components, gene_audit = reconstruct_gene_components(base_df, SOURCE_VCF)
    gene_audit["elapsed_seconds_to_reconstruct"] = round(time.time() - started, 3)
    (args.output_dir / "gene_component_audit.json").write_text(
        json.dumps(gene_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame({"variant_id": base_df["variant_id"], "component_id": components}).to_csv(
        args.output_dir / "gene_component_map.csv.gz", index=False, compression="gzip"
    )

    sampled = balanced_sample(
        base_df.reset_index(names="_source_index"),
        sample_size=args.sample_size,
        seed=args.seed + args.sample_size,
    )
    source_indices = sampled["_source_index"].to_numpy(dtype=int)
    y = base_df.iloc[source_indices]["label"].to_numpy(dtype=int)
    groups = components.iloc[source_indices].to_numpy(dtype=str)
    x = combined.iloc[source_indices][clean_spec.feature_columns].reset_index(drop=True)
    meta = base_df.iloc[source_indices][["variant_id", "gene"]].reset_index(drop=True)

    random_prob, random_fold, random_rows, random_tuning, _, _ = run_nested_oof(
        name="random_stratified_5fold",
        x=x,
        y=y,
        groups=groups,
        spec=clean_spec,
        grouped=False,
        seed=args.seed,
        ece_bins=args.ece_bins,
    )
    gene_prob, gene_fold, gene_rows, gene_tuning, gene_params, gene_splits = run_nested_oof(
        name="gene_component_blocked_5fold",
        x=x,
        y=y,
        groups=groups,
        spec=clean_spec,
        grouped=True,
        seed=args.seed,
        ece_bins=args.ece_bins,
    )

    metrics_df = pd.DataFrame([*random_rows, *gene_rows])
    metrics_df.to_csv(args.output_dir / "gene_blocked_metrics.csv", index=False)
    pd.DataFrame([*random_tuning, *gene_tuning]).to_csv(args.output_dir / "nested_tuning.csv", index=False)

    predictions = meta.assign(
        label=y,
        component_id=groups,
        random_fold=random_fold,
        random_probability=random_prob,
        random_prediction=(random_prob >= 0.5).astype(int),
        gene_blocked_fold=gene_fold,
        gene_blocked_probability=gene_prob,
        gene_blocked_prediction=(gene_prob >= 0.5).astype(int),
    )
    predictions.to_csv(args.output_dir / "gene_blocked_predictions.csv", index=False)

    if args.primary_only:
        print("Primary nested OOF retraining complete; individual predictions are private outputs.", flush=True)
        return

    comparison = paired_component_bootstrap(
        y,
        random_prob,
        gene_prob,
        groups,
        n_bootstrap=args.bootstrap,
        seed=91_013,
        ece_bins=args.ece_bins,
    )
    random_correct = (random_prob >= 0.5) == y
    gene_correct = (gene_prob >= 0.5) == y
    random_only = int(np.sum(random_correct & ~gene_correct))
    gene_only = int(np.sum(~random_correct & gene_correct))
    comparison["mcnemar_random_only_correct"] = random_only
    comparison["mcnemar_gene_only_correct"] = gene_only
    comparison["mcnemar_exact_p"] = float(binomtest(min(random_only, gene_only), random_only + gene_only, 0.5).pvalue)
    comparison.to_csv(args.output_dir / "random_vs_gene_paired_statistics.csv", index=False)

    full_gene_metrics = metric_subset(y, gene_prob, args.ece_bins)
    ablations, families = run_ablations(
        x=x,
        y=y,
        groups=groups,
        spec=clean_spec,
        outer_splits=gene_splits,
        selected_params=gene_params,
        full_metrics=full_gene_metrics,
        ece_bins=args.ece_bins,
    )
    full_row = {"mode": "full_114", "removed_features": 0, "remaining_features": 114, **full_gene_metrics}
    ablations = pd.concat([pd.DataFrame([full_row]), ablations], ignore_index=True)
    ablations.to_csv(args.output_dir / "feature_family_ablation_metrics.csv", index=False)
    (args.output_dir / "feature_family_membership.json").write_text(
        json.dumps(families, indent=2, sort_keys=True), encoding="utf-8"
    )

    alpha = pd.read_csv(ALPHAMISSENSE_20K)
    if not alpha["variant_id"].astype(str).equals(meta["variant_id"].astype(str)):
        raise RuntimeError("AlphaMissense 20k sidecar is not aligned to the audited sample.")
    available = alpha["alphamissense_available"].eq(1).to_numpy() & alpha["alphamissense_score"].notna().to_numpy()
    alpha_score = alpha.loc[available, "alphamissense_score"].to_numpy(dtype=float)
    comparator_stats = paired_component_bootstrap(
        y[available],
        gene_prob[available],
        alpha_score,
        groups[available],
        n_bootstrap=args.bootstrap,
        seed=92_013,
        ece_bins=args.ece_bins,
    )
    comparator_stats.to_csv(args.output_dir / "alphamissense_paired_statistics.csv", index=False)
    comparator = {
        "n_exact_shared_variants": int(available.sum()),
        "n_components": int(np.unique(groups[available]).size),
        "clean_gene_blocked": metric_subset(y[available], gene_prob[available], args.ece_bins),
        "alphamissense": metric_subset(y[available], alpha_score, args.ece_bins),
        "primary_statistic": "paired connected-gene-component bootstrap; clean minus AlphaMissense",
        "bootstrap_replicates": args.bootstrap,
        "score_source": str(ALPHAMISSENSE_20K),
    }
    (args.output_dir / "alphamissense_comparator.json").write_text(
        json.dumps(comparator, indent=2, sort_keys=True), encoding="utf-8"
    )
    predictions.loc[available, ["variant_id", "label", "component_id", "gene_blocked_probability"]].assign(
        alphamissense_score=alpha_score
    ).to_csv(args.output_dir / "alphamissense_paired_predictions.csv", index=False)

    run_natural_prevalence(
        base_df=base_df,
        combined=combined,
        components=components,
        spec=clean_spec,
        output_dir=args.output_dir.parent / "prevalence",
        ece_bins=args.ece_bins,
    )

    provenance = {
        "source_repository_commit": "aadbfe11f45524f9a95431c54abedf9cdb83e630",
        "base_input": str(BASE_INPUT),
        "aa_input": str(AA_INPUT),
        "source_vcf": str(SOURCE_VCF),
        "sample_size": args.sample_size,
        "sampling_seed": args.seed + args.sample_size,
        "outer_seed": args.seed,
        "outer_folds": 5,
        "inner_folds": 1,
        "selection_metric": "AUPRC on one nested 80/20 validation split",
        "feature_count": len(clean_spec.feature_columns),
        "feature_columns": clean_spec.feature_columns,
        "engineered_feature_groups": engineered_groups,
        "elapsed_seconds_total": round(time.time() - started, 3),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
