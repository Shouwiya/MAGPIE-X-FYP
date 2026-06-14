"""Shared schema constants for variant tables and features."""

from __future__ import annotations

from dataclasses import dataclass


CANONICAL_VARIANT_COLUMNS = [
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "gene",
    "consequence",
    "clinical_significance",
    "label",
    "submission_date",
    "condition_name",
]

NONSYNONYMOUS_CONSEQUENCES = {
    "missense",
    "stopgain",
    "stoploss",
    "startloss",
    "frameshift",
    "nonframeshift",
}

MISSENSE_EXPERT_COLUMNS = [
    "alphamissense_score",
    "esm1b_score",
    "revel_score",
    "varity_score",
    "mutpred_score",
]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    dtype: str
    group: str
    missing_strategy: str
    used_for_graph: bool
    used_for_classifier: bool = True
    description: str = ""


def canonical_variant_id(chrom: str | int, pos: str | int, ref: str, alt: str) -> str:
    return f"{chrom}:{pos}:{ref}>{alt}"

