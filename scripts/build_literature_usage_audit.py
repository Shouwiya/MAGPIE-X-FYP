#!/usr/bin/env python3
"""Build auditable literature and SwissProt-use classifications for the workbook."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = ROOT / "scripts" / "build_literature_results_table.py"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real" / "literature"

CLINVAR_SUPPLEMENT = {1, 6, 7, 8, 9, 11, 14, 16, 18, 19}
CLINVAR_FIGURE = {4, 5, 10, 13, 17, 24}
CLINVAR_RANKING_ONLY = {2, 23}
SWISS_PAPERS = {1, 2, 6, 11, 17, 18}
PROTEIN_ASSAY_PAPERS = {5, 9, 14, 18, 23}
ACMG_PAPERS = {1}
PREPRINT_PAPERS = {3, 21, 22, 25}
MISSENSE_ONLY_PAPERS = {
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    17,
    18,
    20,
    21,
    22,
    23,
    24,
    25,
}


def load_rows() -> tuple[list[str], list[list[object]]]:
    spec = importlib.util.spec_from_file_location("literature_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.HEADERS), list(module.ROWS)


def uses_clinvar(dataset: str, external: str) -> bool:
    return "clinvar" in f"{dataset} {external}".lower()


def clinvar_reason(paper_no: int, uses: bool, tuple_text: str) -> str:
    if not uses:
        return "Dataset not used by paper"
    if paper_no in CLINVAR_RANKING_ONLY:
        return "Reported only as AUROC/AUPRC"
    if paper_no in CLINVAR_SUPPLEMENT:
        return "Supplementary table extraction required"
    if paper_no in CLINVAR_FIGURE:
        return "Figure-only value; needs manual extraction"
    if paper_no == 12:
        return "Dataset not used by paper"
    if paper_no == 15:
        return "Not reported by paper"
    if paper_no == 20:
        return "Dataset not used by paper"
    if paper_no in {21, 25}:
        return "Preprint/unverified"
    if tuple_text != "(-, -, -, -, -, -, -, -, -, -)":
        return "Not reported by paper"
    return "Not reported by paper"


def swiss_reason(paper_no: int, uses: bool, tuple_text: str) -> str:
    if not uses:
        return "Dataset not used by paper"
    if paper_no in {1, 2}:
        return "Reported only as AUROC/AUPRC"
    if paper_no in {6, 11}:
        return "Supplementary table extraction required"
    if paper_no in {17, 18}:
        return "Dataset not applicable to model scope"
    if tuple_text != "(-, -, -, -, -, -, -, -, -, -)":
        return "Not reported by paper"
    return "Not reported by paper"


def other_reason(paper_no: int, external: str) -> str:
    if paper_no in ACMG_PAPERS:
        return "Supplementary table extraction required"
    if paper_no in PROTEIN_ASSAY_PAPERS:
        return "Figure-only value; needs manual extraction"
    if paper_no in SWISS_PAPERS:
        return "Dataset not used by paper"
    if paper_no in MISSENSE_ONLY_PAPERS:
        return "Variant type mismatch"
    if external and external != "-":
        return "External validation exists but not SwissProt"
    return "Dataset not used by paper"


def extraction_status(
    paper_no: int,
    clin_tuple: str,
    rare_tuple: str,
    swiss_tuple: str,
) -> str:
    tuples = (clin_tuple, rare_tuple, swiss_tuple)
    any_verified = any(value != "(-, -, -, -, -, -, -, -, -, -)" for value in tuples)
    if paper_no in PREPRINT_PAPERS:
        if any_verified:
            return "Partial metrics extracted; preprint/unverified"
        return "No verified tuple; preprint/manual PDF check required"
    if any_verified:
        return "Partial verified metrics extracted; full tuple not reported"
    return "No verified full tuple; manual paper/supplement extraction required"


def usefulness(paper_no: int, clinvar: bool, swiss: bool) -> str:
    if paper_no in {1, 2}:
        return "High: direct multi-type ClinVar and SwissProt-style comparator"
    if clinvar and swiss:
        return "High: ClinVar plus SwissProt-style resource comparator"
    if clinvar:
        if paper_no in MISSENSE_ONLY_PAPERS:
            return "Medium: useful ClinVar comparator with missense-only scope"
        return "High: useful ClinVar pathogenicity comparator"
    if paper_no in {3, 20, 22}:
        return "Medium: method/background comparator; no matching verified dataset tuple"
    return "Medium: useful methodological context; benchmark scope differs"


def swiss_metadata(paper_no: int) -> tuple[str, str, str]:
    mapping = {
        1: (
            "SwissProt orthogonal set",
            "1,308 pathogenic + 12,075 benign",
            "Multi-type exonic variants",
        ),
        2: ("SwissProt-Test", "Not extracted", "Multi-type exonic variants"),
        6: ("SwissVar", "Not extracted", "Rare missense variants"),
        11: (
            "SwissVar independent accumulation test",
            "Not extracted",
            "Missense variants",
        ),
        17: (
            "UniProt pathogenic variants used in model development",
            "12,875 UniProt positives reported in training inputs",
            "Missense variants",
        ),
        18: (
            "UniProt variants used in model development",
            "Not separately extracted",
            "Missense variants",
        ),
    }
    return mapping.get(
        paper_no,
        ("Not used", "Not applicable", "Not applicable"),
    )


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    headers, rows = load_rows()
    source = pd.DataFrame(rows, columns=headers)
    audit_rows: list[dict[str, object]] = []
    swiss_rows: list[dict[str, object]] = []

    for _, row in source.iterrows():
        paper_no = int(row["Paper No."])
        clinvar = uses_clinvar(row["Dataset Used"], row["External Validation"])
        swiss = paper_no in SWISS_PAPERS
        clin_reason = clinvar_reason(
            paper_no, clinvar, row["ClinVar Result Tuple"]
        )
        swiss_missing_reason = swiss_reason(
            paper_no, swiss, row["SwissProt/HUMSAVAR Result Tuple"]
        )
        other_missing_reason = other_reason(paper_no, row["External Validation"])
        status = extraction_status(
            paper_no,
            row["ClinVar Result Tuple"],
            row["ClinVarRare Result Tuple"],
            row["SwissProt/HUMSAVAR Result Tuple"],
        )
        final_usefulness = usefulness(paper_no, clinvar, swiss)

        audit_rows.append(
            {
                "Paper No.": paper_no,
                "Uses ClinVar?": "Yes" if clinvar else "No",
                "Uses SwissProt/HUMSAVAR/SwissVar/UniProt?": (
                    "Yes" if swiss else "No"
                ),
                "Reason ClinVar metric missing": clin_reason,
                "Reason SwissProt metric missing": swiss_missing_reason,
                "Reason other benchmark metric missing": other_missing_reason,
                "Metric extraction status": status,
                "Final comparison usefulness": final_usefulness,
                "Uses ProteinGym/MAVE/DMS?": (
                    "Yes" if paper_no in PROTEIN_ASSAY_PAPERS else "No"
                ),
                "Uses ACMG/guideline validation?": (
                    "Yes" if paper_no in ACMG_PAPERS else "No"
                ),
                "Missense-only scope?": (
                    "Yes" if paper_no in MISSENSE_ONLY_PAPERS else "No"
                ),
            }
        )

        swiss_name, swiss_size, swiss_type = swiss_metadata(paper_no)
        if swiss:
            if paper_no in {17, 18}:
                not_used_reason = (
                    "UniProt contributes model-development variants, but no "
                    "separate verified SwissProt-style external test tuple was extracted."
                )
            elif row["SwissProt/HUMSAVAR Result Tuple"] == (
                "(-, -, -, -, -, -, -, -, -, -)"
            ):
                not_used_reason = (
                    "SwissProt-style resource is used, but the full requested "
                    "metric tuple requires supplementary/manual extraction."
                )
            else:
                not_used_reason = (
                    "SwissProt-style evaluation is present; only verified "
                    "reported metrics are populated."
                )
        else:
            not_used_reason = (
                "Not a SwissProt/HUMSAVAR evaluation paper; included only for "
                "ClinVar/model comparison."
            )

        swiss_rows.append(
            {
                "Paper No.": paper_no,
                "Title": row["Title"],
                "Year": int(row["Year"]),
                "Proposed Approach / Model": row["Proposed Approach / Model"],
                "Description": row["Description"],
                "Main dataset used by paper": row["Dataset Used"],
                "Uses ClinVar?": "Yes" if clinvar else "No",
                "Uses SwissProt / HUMSAVAR / SwissVar / UniProt?": (
                    "Yes" if swiss else "No"
                ),
                "SwissProt dataset name used": swiss_name,
                "SwissProt sample size": swiss_size,
                "SwissProt variant type": swiss_type,
                "SwissProt metric tuple": row[
                    "SwissProt/HUMSAVAR Result Tuple"
                ],
                "If SwissProt not used, reason": not_used_reason,
                "Is this paper still useful for our project?": final_usefulness,
                "Notes / Limitations": row["Notes / Limitations"],
                "Source Link": row["Source Link"],
            }
        )

    audit = pd.DataFrame(audit_rows)
    swiss_table = pd.DataFrame(swiss_rows)
    threshold_present = source[
        source.apply(
            lambda row: any(
                value != "-"
                for value in row["ClinVar Result Tuple"]
                .strip("()")
                .replace(" ", "")
                .split(",")[:6]
            )
            or any(
                value != "-"
                for value in row["ClinVarRare Result Tuple"]
                .strip("()")
                .replace(" ", "")
                .split(",")[:6]
            ),
            axis=1,
        )
    ]
    ranking_only = source[
        source["Paper No."].isin(sorted(CLINVAR_RANKING_ONLY))
    ]
    summary = {
        "total_papers": int(len(source)),
        "papers_using_clinvar": int((audit["Uses ClinVar?"] == "Yes").sum()),
        "papers_using_swissprot_style": int(
            (
                audit["Uses SwissProt/HUMSAVAR/SwissVar/UniProt?"] == "Yes"
            ).sum()
        ),
        "papers_using_both_clinvar_and_swissprot_style": int(
            (
                (audit["Uses ClinVar?"] == "Yes")
                & (
                    audit["Uses SwissProt/HUMSAVAR/SwissVar/UniProt?"]
                    == "Yes"
                )
            ).sum()
        ),
        "papers_using_proteingym_mave_dms": int(
            (audit["Uses ProteinGym/MAVE/DMS?"] == "Yes").sum()
        ),
        "papers_using_acmg_guideline_validation": int(
            (audit["Uses ACMG/guideline validation?"] == "Yes").sum()
        ),
        "papers_with_full_metric_tuple": 0,
        "papers_with_auroc_auprc_only": int(len(ranking_only)),
        "papers_with_no_verified_threshold_metrics": int(
            len(source) - len(threshold_present)
        ),
        "reason_counts": {
            "Dataset not used by paper": int(
                (
                    audit[
                        [
                            "Reason ClinVar metric missing",
                            "Reason SwissProt metric missing",
                            "Reason other benchmark metric missing",
                        ]
                    ]
                    == "Dataset not used by paper"
                )
                .sum()
                .sum()
            ),
            "Metric not reported": int(
                (
                    audit[
                        [
                            "Reason ClinVar metric missing",
                            "Reason SwissProt metric missing",
                        ]
                    ]
                    == "Not reported by paper"
                )
                .sum()
                .sum()
            ),
            "Threshold not provided": int(
                (
                    audit[
                        [
                            "Reason ClinVar metric missing",
                            "Reason SwissProt metric missing",
                        ]
                    ]
                    == "Reported only as AUROC/AUPRC"
                )
                .sum()
                .sum()
            ),
            "Supplementary extraction required": int(
                (
                    audit[
                        [
                            "Reason ClinVar metric missing",
                            "Reason SwissProt metric missing",
                            "Reason other benchmark metric missing",
                        ]
                    ]
                    == "Supplementary table extraction required"
                )
                .sum()
                .sum()
            ),
            "Not applicable due to variant type": int(
                (
                    audit["Reason other benchmark metric missing"]
                    == "Variant type mismatch"
                ).sum()
            ),
            "External validation uses a different benchmark": int(
                (
                    audit["Reason other benchmark metric missing"]
                    == "External validation exists but not SwissProt"
                ).sum()
            ),
            "Paper is preprint/unverified": int(len(PREPRINT_PAPERS)),
        },
    }
    return audit, swiss_table, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build literature dataset-use audit sidecars."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit, swiss_table, summary = build_tables()
    audit.to_csv(args.output_dir / "literature_dataset_use_audit.csv", index=False)
    swiss_table.to_csv(
        args.output_dir / "swissprot_comparison_rows.csv", index=False
    )
    (args.output_dir / "dataset_use_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
