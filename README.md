# MAGPIE-X

## Accepted conference paper companion

For **What Pooled AUROC Hides: A Consequence-Stratified Pathogenicity Audit**
(ICIC 2026, paper 53), start with [conference-audit/README.md](conference-audit/README.md).
It contains aggregate evidence, recorded model configurations, verification code,
figure/table regeneration instructions, and a prepared-input primary retraining
entry point. It is not complete raw-data-to-paper replication. The FYP experiments below, including SwissProt evaluation,
are separate from that conference audit and must not be conflated with it.

MAGPIE-X is a final-year research project for pathogenicity prediction of human
protein-altering variants. The repository separates two experimental tracks:

- **MAGPIE-parity low-circularity** uses ANNOVAR/refGene consequence features,
  gnomAD allele frequencies, conservation, variant type/length, biochemical
  amino-acid deltas, BLOSUM62, and optionally exact-matched SpliceAI features.
- **MAGPIE-X circularity-risk** intentionally adds prior pathogenicity
  predictors from dbNSFP, AlphaMissense where matched, and controlled graph
  residual fusion. These results are reported separately because the external
  predictors can share evidence with clinical labels.

The repository also contains a single-machine simulation of horizontal
federated learning. It is an experimental comparison, not a deployed privacy
or security system.

## Repository Contents

```text
scripts/        Data preparation, matching, modeling, evaluation, and workbook tools
src/magpie_x/   Reusable parsing, feature, graph, model, and evaluation modules
configs/        Small legacy/default configuration files used by CLI utilities
docs/           Feature audits, experiment plans, and methodological notes
paper/          LaTeX paper source and bibliography
outputs/real/   Selected compact final reports and metric tables only
```

The final clarified comparison workbook is:

`FYP_Literature_Results_Table_Omar_FINAL_CLARIFIED.xlsx`

## Data Policy

Raw and derived genomic resources are deliberately excluded from GitHub,
including ClinVar, SpliceAI, gnomAD, dbNSFP, AlphaMissense, ANNOVAR databases,
SwissProt/HUMSAVAR source files, feature tables, prediction sidecars, and model
binaries. They must be obtained from their official sources under the relevant
licenses and placed locally according to the paths accepted by each script.

The committed reports contain aggregate metrics only. The older SwissProt/HUMSAVAR
experiment excluded exact test keys from its training sample, but its available
features were reused by exact ClinVar variant join. Its saved no-exact-overlap
evaluation had zero usable rows. It is an external-label/reference-model check,
not fully independent validation of this conference audit. Fresh annotation of
non-overlapping variants is still required.

## Installation

Python 3.11 was used for the final experiments.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

`pysam` is used for indexed VCF matching. The SpliceAI matching work was run in
WSL Ubuntu with `bcftools`, `tabix`, `bgzip`, and `samtools` available.

## Main Experiment Scripts

- `scripts/run_magpie_low_circularity_baseline.py`
- `scripts/run_aa_enhanced_low_circularity_comparison.py`
- `scripts/run_spliceai_5k_low_circularity_comparison.py`
- `scripts/run_spliceai_10k_20k_low_circularity_comparison.py`
- `scripts/run_magpie_x_5k_full_fusion.py`
- `scripts/run_magpie_x_10k_20k_controlled.py`
- `scripts/federated/run_federated_sample_size_benchmark.py`
- `scripts/evaluate_on_swissprot.py`

Use each script's `--help` output for its exact inputs. The scripts do not
download protected or large annotation databases automatically.

## Reported Metrics

The standard result tuple is:

`(Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, MCC, AUROC, AUPRC, Brier, ECE)`

Compact final reports are retained under:

- `outputs/real/final_comparison/`
- `outputs/real/magpie_parity/`
- `outputs/real/magpie_x_enhanced/`
- `outputs/real/federated/`

## Reproducibility Notes

- Sample-size comparisons use deterministic balanced ClinVar-derived samples
  and paired train/test splits where stated in the reports.
- SpliceAI and AlphaMissense matching uses genomic allele keys and does not use
  clinical labels.
- MAGPIE-parity excludes prior predictor-score fields.
- MAGPIE-X and FL-B/FL-C include prior predictor fields and therefore retain
  high circularity risk.
- Federated experiments simulate five clients on one machine. LightGBM client
  probabilities are soft-voted; only neural models use parameter averaging.
- The full 490k feature table and all raw resources are intentionally absent
  from this public snapshot.

## Research Use

This project is for research and academic evaluation only. It is not clinical
decision support and must not be used to diagnose patients or determine
medical treatment.
