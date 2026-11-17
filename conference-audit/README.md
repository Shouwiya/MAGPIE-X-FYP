# Companion: What Pooled AUROC Hides

Companion for **What Pooled AUROC Hides: A Consequence-Stratified Pathogenicity Audit** (ICIC 2026, paper 53). This directory is distinct from the older FYP experiments in the repository root.

## What is reproducible here?

This provides **aggregate verification** plus optional **prepared-input primary retraining**. It is not complete raw-data-to-paper replication. The aggregate commands do not retrain models or recreate bootstrap samples. See [training/README.md](training/README.md) for the separate training command, which needs the full repository checkout and separately obtained research inputs.

Version `conference-audit-v1.1` adds a portable primary refit. A fresh-environment author-side run on 17 November 2026 rebuilt gene components and refitted both nested OOF arms: all ten selected configurations and fold assignments matched; all metrics agreed within 1e-10; the maximum probability difference over 20,000 aligned rows was 0.0. This is not independent external validation or raw-annotation replication. The v1 tag preserves the aggregate-only release.

The manuscript sources are an analysis snapshot, not an approved camera-ready submission. No IEEE PDF eXpress approval is implied. The default build is anonymous; authorship metadata must be resolved separately for submission.

## Quick start (tested with Python 3.12 on Windows)

From this directory, create a fresh environment and install the pinned dependencies:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS alternative: source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_evidence.py
python generate_assets.py
python verify_evidence.py
```

Expected verification output begins `PASS: counts, decomposition, seeds, configurations...`. On assertion failure, do not edit the evidence to force a pass: investigate the inconsistency. Asset generation intentionally overwrites derived files in `figures/`, `tables/`, and the Wilson-interval evidence outputs; run it in a disposable checkout when comparing with the retained snapshot. No network or original research folders are needed after dependency installation.

`requirements-tested-windows-py312.txt` records all resolved versions in the tested environment, including transitive dependencies. The four-package requirements file pins the direct dependencies. Other operating systems have not been tested by this release.

For the manuscript, use an IEEEtran-capable LaTeX installation: `latexmk -pdf main.tex`. Alternatively upload sources, sections, tables, figures and references to Overleaf and select `main.tex` (pdfLaTeX). Python is not required to compile the retained figures. The retained research report and current conference requirements are separate checks.

## Contents and checks

- `evidence/`: 37 aggregate CSV/JSON records, not individual variants or patient records.
- `verify_evidence.py`: stratum totals/confusion counts; AUROC pair decomposition; seed means and sample SDs; feature-family coverage; selected configurations; paired metric differences; Wilson intervals; exploratory seed sign tests; and numeric manuscript anchors.
- `generate_assets.py`: recreates manuscript tables and vector figures from aggregates. Plotting a saved confidence interval does not reproduce its bootstrap calculation.
- `threshold_uncertainty.py`: recomputes nominal Wilson intervals from aggregate counts; these intervals do not account for clustering.
- `evidence/tuning_manifest.json`: all six candidate configurations and outer-fold selections.
- `evidence/recovered_model_provenance.json`: recovered constructor settings, package versions, field schema and recorded author-side reproduction check.
- `evidence/control_protocol.json`: seeds and control specifications.
- `evidence/feature_manifest.json`, `feature_family_membership.json`: retained fields and feature families.
- `DATA_AND_RETRAINING.md`: exact boundary of the release and remaining dependencies.
- `PAPER_AUDIT.md`: reviewer mapping and submission issues still requiring attention.

## Important interpretation limits

AP is non-interpolated average precision; historical `auprc` fields denote AP. Confidence intervals from saved predictions are conditional on fitted predictions, not retraining uncertainty. Seed sign tests are exploratory and unadjusted for multiple metrics. The clinical/source evidence can share information with learned annotations; excluding explicit predictor scores does not establish absence of leakage.

No independent or temporal validation of this conference audit is supplied. The root FYP README's SwissProt experiment is a different experiment and must not be presented as validation of this exact conference protocol without a separate overlap/protocol audit.

## Privacy, licensing, citation

Excluded: row-level variants, clinical labels, predictions, feature vectors, fitted models, proprietary annotation databases, credentials, personal contact/address records, and private checkpoints. Obtain upstream data under its own terms. This package grants no rights to redistribute upstream resources. No new repository-wide software license is asserted by this release.

When citing, identify the paper title, repository URL, **commit hash**, and this directory. No DOI or archival deposit is claimed. Use a fixed commit rather than relying on a moving branch.
