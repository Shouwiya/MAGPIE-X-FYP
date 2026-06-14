"""Feature assembly, missing-value handling, standardization, and manifesting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from magpie_x.data.phenotype_mapper import add_phenotype_groups
from magpie_x.utils.io import ensure_dir, read_table, write_dataframe, write_json
from magpie_x.utils.schema import FeatureSpec, MISSENSE_EXPERT_COLUMNS

AA_PROPERTIES = {
    "A": {"hydropathy": 1.8, "volume": 88.6, "charge": 0},
    "C": {"hydropathy": 2.5, "volume": 108.5, "charge": 0},
    "D": {"hydropathy": -3.5, "volume": 111.1, "charge": -1},
    "E": {"hydropathy": -3.5, "volume": 138.4, "charge": -1},
    "F": {"hydropathy": 2.8, "volume": 189.9, "charge": 0},
    "G": {"hydropathy": -0.4, "volume": 60.1, "charge": 0},
    "H": {"hydropathy": -3.2, "volume": 153.2, "charge": 1},
    "I": {"hydropathy": 4.5, "volume": 166.7, "charge": 0},
    "K": {"hydropathy": -3.9, "volume": 168.6, "charge": 1},
    "L": {"hydropathy": 3.8, "volume": 166.7, "charge": 0},
    "M": {"hydropathy": 1.9, "volume": 162.9, "charge": 0},
    "N": {"hydropathy": -3.5, "volume": 114.1, "charge": 0},
    "P": {"hydropathy": -1.6, "volume": 112.7, "charge": 0},
    "Q": {"hydropathy": -3.5, "volume": 143.8, "charge": 0},
    "R": {"hydropathy": -4.5, "volume": 173.4, "charge": 1},
    "S": {"hydropathy": -0.8, "volume": 89.0, "charge": 0},
    "T": {"hydropathy": -0.7, "volume": 116.1, "charge": 0},
    "V": {"hydropathy": 4.2, "volume": 140.0, "charge": 0},
    "W": {"hydropathy": -0.9, "volume": 227.8, "charge": 0},
    "Y": {"hydropathy": -1.3, "volume": 193.6, "charge": 0},
}

CONSEQUENCE_ORDER = ["missense", "stopgain", "stoploss", "startloss", "frameshift", "nonframeshift"]


@dataclass
class FeatureBundle:
    features: pd.DataFrame
    labels: pd.Series
    variants: pd.DataFrame
    manifest: pd.DataFrame
    feature_names: list[str]
    graph_feature_names: list[str]


def _merge_optional(base: pd.DataFrame, other: pd.DataFrame | None, suffix: str) -> pd.DataFrame:
    if other is None or other.empty:
        return base
    join_cols = ["variant_id"]
    keep = [c for c in other.columns if c not in base.columns or c in join_cols]
    merged = base.merge(other[keep], on=join_cols, how="left", suffixes=("", suffix))
    return merged


def _safe_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _add_biochemical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    specs: list[FeatureSpec] = []
    out = df.copy()
    for prop in ("hydropathy", "volume", "charge"):
        ref_values = out.get("aa_ref", pd.Series("", index=out.index)).map(lambda aa: AA_PROPERTIES.get(str(aa), {}).get(prop, np.nan))
        alt_values = out.get("aa_alt", pd.Series("", index=out.index)).map(lambda aa: AA_PROPERTIES.get(str(aa), {}).get(prop, np.nan))
        name = f"aa_delta_{prop}"
        out[name] = alt_values - ref_values
        specs.append(
            FeatureSpec(
                name=name,
                source="biochemical",
                dtype="numeric",
                group="biochemical",
                missing_strategy="median",
                used_for_graph=True,
                description=f"Alternate minus reference amino acid {prop}.",
            )
        )
    out["aa_grantham_proxy"] = (
        out["aa_delta_hydropathy"].abs() * 20
        + out["aa_delta_volume"].abs() * 0.5
        + out["aa_delta_charge"].abs() * 50
    )
    specs.append(
        FeatureSpec(
            name="aa_grantham_proxy",
            source="biochemical",
            dtype="numeric",
            group="biochemical",
            missing_strategy="median",
            used_for_graph=True,
            description="Lightweight Grantham-like distance proxy for toy and fallback runs.",
        )
    )
    return out, specs


def _feature_specs_for_existing(df: pd.DataFrame, graph_groups: set[str]) -> list[FeatureSpec]:
    catalog: dict[str, tuple[str, str, str, str]] = {
        "gnomad_af": ("gnomAD", "numeric", "frequency", "Population allele frequency."),
        "spliceai_delta_score": ("SpliceAI", "numeric", "splicing", "Maximum SpliceAI delta score."),
        "phylop100": ("dbNSFP", "numeric", "conservation", "phyloP 100-way conservation score."),
        "phastcons100": ("dbNSFP", "numeric", "conservation", "phastCons 100-way conservation probability."),
        "cadd_phred": ("dbNSFP", "numeric", "deleteriousness", "CADD PHRED-like score."),
        "sift_score": ("dbNSFP", "numeric", "missense_expert", "SIFT score; lower is more deleterious."),
        "polyphen2_hdiv_score": ("dbNSFP", "numeric", "missense_expert", "PolyPhen2 HDIV score."),
        "loeuf": ("gnomAD", "numeric", "gene_intolerance", "Gene LOEUF intolerance score."),
        "pli": ("gnomAD", "numeric", "gene_intolerance", "Gene pLI intolerance score."),
        "recombination_rate": ("genomic_context", "numeric", "genomic_context", "Local recombination rate."),
        "chromatin_accessibility": ("epigenetic", "numeric", "epigenetic", "Chromatin accessibility summary signal."),
        "h3k27ac_signal": ("epigenetic", "numeric", "epigenetic", "H3K27ac enhancer-associated signal."),
        "exon_number": ("ANNOVAR", "numeric", "annovar", "Annotated exon number."),
        "cds_position": ("ANNOVAR", "numeric", "annovar", "Coding sequence coordinate."),
    }
    for col in MISSENSE_EXPERT_COLUMNS:
        catalog[col] = ("precomputed_missense_expert", "numeric", "missense_expert", f"Precomputed {col} score.")
    specs = [
        FeatureSpec(
            name=name,
            source=source,
            dtype=dtype,
            group=group,
            missing_strategy="median",
            used_for_graph=group in graph_groups,
            description=description,
        )
        for name, (source, dtype, group, description) in catalog.items()
        if name in df.columns
    ]
    return specs


def _add_consequence_onehot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    out = df.copy()
    specs: list[FeatureSpec] = []
    cons = out["consequence"].fillna("unknown").astype(str).str.lower()
    for name in CONSEQUENCE_ORDER:
        col = f"consequence_{name}"
        out[col] = (cons == name).astype(float)
        specs.append(
            FeatureSpec(
                name=col,
                source="ClinVar/ANNOVAR",
                dtype="binary",
                group="consequence",
                missing_strategy="constant_0",
                used_for_graph=False,
                description=f"One-hot indicator for {name} exonic consequence.",
            )
        )
    return out, specs


def _impute_and_standardize(
    df: pd.DataFrame,
    specs: list[FeatureSpec],
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    features = pd.DataFrame(index=df.index)
    stats: dict[str, dict[str, float]] = {}
    manifest_rows: list[dict[str, object]] = []
    for spec in specs:
        values = pd.to_numeric(df[spec.name], errors="coerce") if spec.name in df.columns else pd.Series(np.nan, index=df.index)
        missingness = float(values.isna().mean())
        if spec.missing_strategy == "constant_0":
            imputed = values.fillna(0.0)
            fill_value = 0.0
        else:
            fill_value = float(values.median()) if values.notna().any() else 0.0
            imputed = values.fillna(fill_value)
        mean = float(imputed.mean())
        std = float(imputed.std(ddof=0))
        if standardize and spec.dtype != "binary":
            denom = std if std > 1e-8 else 1.0
            final = (imputed - mean) / denom
        else:
            denom = 1.0
            final = imputed
        features[spec.name] = final.astype(float)
        stats[spec.name] = {"fill_value": fill_value, "mean": mean, "std": denom}
        manifest_rows.append(
            {
                "feature_name": spec.name,
                "source": spec.source,
                "type": spec.dtype,
                "group": spec.group,
                "missing_strategy": spec.missing_strategy,
                "missingness": missingness,
                "used_for_graph": spec.used_for_graph,
                "used_for_classifier": spec.used_for_classifier,
                "description": spec.description,
            }
        )
    return features, pd.DataFrame(manifest_rows), stats


def build_feature_bundle(
    clinvar: pd.DataFrame,
    annovar: pd.DataFrame | None = None,
    dbnsfp: pd.DataFrame | None = None,
    config: dict | None = None,
) -> FeatureBundle:
    """Build model-ready features and the publication feature manifest."""

    config = config or {}
    graph_groups = set(config.get("graph_feature_groups", ["conservation", "frequency", "biochemical", "gene_intolerance"]))
    df = clinvar.copy()
    df = _merge_optional(df, annovar, "_annovar")
    df = _merge_optional(df, dbnsfp, "_dbnsfp")
    df = add_phenotype_groups(df)

    numeric_candidates = [
        "protein_pos",
        "exon_number",
        "cds_position",
        "gnomad_af",
        "spliceai_delta_score",
        "phylop100",
        "phastcons100",
        "cadd_phred",
        "sift_score",
        "polyphen2_hdiv_score",
        "loeuf",
        "pli",
        "recombination_rate",
        "chromatin_accessibility",
        "h3k27ac_signal",
        *MISSENSE_EXPERT_COLUMNS,
    ]
    df = _safe_numeric(df, numeric_candidates)

    specs = _feature_specs_for_existing(df, graph_groups)
    if config.get("include_biochemical", True):
        df, bio_specs = _add_biochemical_features(df)
        specs.extend(bio_specs)
    if config.get("include_consequence_onehot", True):
        df, consequence_specs = _add_consequence_onehot(df)
        specs.extend(consequence_specs)
    if "protein_pos" in df.columns:
        specs.append(
            FeatureSpec(
                name="protein_pos",
                source="ClinVar/ANNOVAR",
                dtype="numeric",
                group="protein_context",
                missing_strategy="median",
                used_for_graph=False,
                description="Amino-acid coordinate in the canonical or provided transcript.",
            )
        )

    # Preserve order while removing duplicate feature specs.
    deduped: dict[str, FeatureSpec] = {}
    for spec in specs:
        deduped.setdefault(spec.name, spec)
    specs = list(deduped.values())
    features, manifest, stats = _impute_and_standardize(df, specs, standardize=bool(config.get("standardize", True)))
    graph_features = manifest.loc[manifest["used_for_graph"], "feature_name"].tolist()

    variant_cols = [
        c
        for c in [
            "variant_id",
            "chrom",
            "pos",
            "ref",
            "alt",
            "gene",
            "consequence",
            "clinical_significance",
            "submission_date",
            "condition_name",
            "phenotype_group",
            "phenotype_group_id",
            "gnomad_af",
        ]
        if c in df.columns
    ]
    variants = df[variant_cols].copy()
    return FeatureBundle(
        features=features,
        labels=df["label"].astype(int),
        variants=variants,
        manifest=manifest,
        feature_names=features.columns.tolist(),
        graph_feature_names=graph_features,
    )


def save_feature_bundle(bundle: FeatureBundle, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    write_dataframe(bundle.features, output_dir / "features.csv")
    write_dataframe(pd.DataFrame({"label": bundle.labels}), output_dir / "labels.csv")
    variants = bundle.variants.copy()
    if "submission_date" in variants.columns:
        variants["submission_date"] = pd.to_datetime(variants["submission_date"]).dt.strftime("%Y-%m-%d")
    write_dataframe(variants, output_dir / "variants.csv")
    write_dataframe(bundle.manifest, output_dir / "feature_manifest.csv")
    write_json(
        {
            "n_variants": int(len(bundle.labels)),
            "n_features": int(len(bundle.feature_names)),
            "feature_names": bundle.feature_names,
            "graph_feature_names": bundle.graph_feature_names,
        },
        output_dir / "feature_bundle.json",
    )


def load_feature_bundle(output_dir: str | Path) -> FeatureBundle:
    output_dir = Path(output_dir)
    features = read_table(output_dir / "features.csv")
    labels = read_table(output_dir / "labels.csv")["label"].astype(int)
    variants = read_table(output_dir / "variants.csv")
    manifest = read_table(output_dir / "feature_manifest.csv")
    graph_names = manifest.loc[manifest["used_for_graph"].astype(str).str.lower().isin(["true", "1"]), "feature_name"].tolist()
    return FeatureBundle(
        features=features,
        labels=labels,
        variants=variants,
        manifest=manifest,
        feature_names=features.columns.tolist(),
        graph_feature_names=graph_names,
    )

