"""Optional mapping from ClinVar condition names to coarse HPO-like groups."""

from __future__ import annotations

import re

import pandas as pd

PHENOTYPE_GROUPS = [
    "neurologic",
    "cardiovascular",
    "metabolic",
    "cancer",
    "immune",
    "skeletal",
    "renal",
    "sensory",
    "other",
]

KEYWORDS = {
    "neurologic": ["seizure", "epilep", "developmental", "neuro", "ataxia", "intellectual"],
    "cardiovascular": ["cardio", "heart", "aortic", "arrhythm", "myopathy", "hypertrophic"],
    "metabolic": ["metabolic", "enzyme", "mitochond", "organic acid", "storage", "inborn"],
    "cancer": ["cancer", "tumor", "tumour", "carcinoma", "neoplasm", "oncology"],
    "immune": ["immune", "immunodeficiency", "autoimmune", "infection"],
    "skeletal": ["skeletal", "bone", "dysplasia", "cranio", "short stature"],
    "renal": ["renal", "kidney", "nephro", "tubular"],
    "sensory": ["retinal", "vision", "hearing", "deaf", "ocular", "dystrophy"],
}


def map_condition_to_group(condition_name: object) -> str:
    text = re.sub(r"[_/;|]+", " ", str(condition_name or "").lower())
    for group, keywords in KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return group
    return "other"


def add_phenotype_groups(df: pd.DataFrame, condition_col: str = "condition_name") -> pd.DataFrame:
    out = df.copy()
    out["phenotype_group"] = out[condition_col].map(map_condition_to_group)
    out["phenotype_group_id"] = out["phenotype_group"].map({g: i for i, g in enumerate(PHENOTYPE_GROUPS)}).astype(int)
    return out

