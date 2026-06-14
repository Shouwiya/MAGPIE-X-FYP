# SwissProt/HUMSAVAR Integration Plan

Generated: 2026-06-04

Purpose: add SwissProt/HUMSAVAR as an external validation path for MAGPIE-X. This is separate from random ClinVar split benchmarking and is intended to test generalization.

## Ground Rules

- Use real HUMSAVAR only: `data/external/humsavar/humsavar.txt`.
- Do not fabricate genomic coordinates.
- Do not fill missing mappings manually.
- Do not use synthetic or toy data for reported results.
- Do not train full models automatically.
- Keep SwissProt/HUMSAVAR results separate from random ClinVar split results.

## MAGPIE Reference Logic

MAGPIE used SwissProt as an orthogonal validation dataset. The relevant design idea is:

- remove uncertain-significance variants;
- remove synonymous SNVs;
- retain exonic/protein-altering variants;
- remove variants located in genes seen in ClinVarTrain to reduce leakage.

For this repository, the practical implementation is:

- HUMSAVAR `US` rows are excluded during parsing;
- HUMSAVAR is already a single amino acid variant file, so parsed supervised rows are missense/protein-altering candidates;
- rsID mapping is attempted only against real local ClinVar `variant_summary.txt.gz`;
- rows that cannot be mapped to one unique GRCh38 `chrom,pos,ref,alt` tuple are kept as unmapped;
- gene overlap with the current ClinVar-derived feature table is recorded as an audit field before any held-out filtering is applied.

## Step A: Parse HUMSAVAR

Script:

```powershell
python scripts/parse_humsavar.py --input data/external/humsavar/humsavar.txt --output-dir outputs/real/swissprot
```

Outputs:

- `outputs/real/swissprot/humsavar_parsed.csv`
- `outputs/real/swissprot/humsavar_excluded_uncertain.csv`
- `outputs/real/swissprot/humsavar_parse_manifest.json`

Label rule:

- `LP/P` -> `1`
- `LB/B` -> `0`
- `US` -> excluded

## Step B: Map To Genomic Variants

Script:

```powershell
python scripts/map_humsavar_to_variants.py --input outputs/real/swissprot/humsavar_parsed.csv --variant-summary data/raw/clinvar/latest/variant_summary.txt.gz --output-dir outputs/real/swissprot --clinvar-feature-table data/interim/features/basic_feature_table.csv
```

Mapping rule:

- use dbSNP rsID;
- require GRCh38 rows in ClinVar `variant_summary.txt.gz`;
- require exactly one unique `chrom,pos,ref,alt` tuple per rsID;
- otherwise write the HUMSAVAR row to the unmapped file.

Outputs:

- `outputs/real/swissprot/humsavar_mapped_variants.csv`
- `outputs/real/swissprot/humsavar_unmapped_variants.csv`
- `outputs/real/swissprot/SWISSPROT_MAPPING_REPORT.md`
- `outputs/real/swissprot/humsavar_mapping_manifest.json`

Current audit result:

- supervised HUMSAVAR rows: 73,056;
- mapped rows: 30,122;
- unmapped rows: 42,934;
- mapped rows after excluding genes seen in the current ClinVar-derived feature table: 370.

The last number is an approximation for leakage review, not a final benchmark cohort.

## Step C: Build SwissProt Feature Table

Script:

```powershell
python scripts/build_swissprot_feature_table.py --mapped outputs/real/swissprot/humsavar_mapped_variants.csv --output-dir outputs/real/swissprot
```

This writes:

- `outputs/real/swissprot/swissprot_variant_table.csv`
- `outputs/real/swissprot/swissprot_humsavar.avinput`
- `outputs/real/swissprot/SWISSPROT_ANNOVAR_COMMANDS.txt`

It does not run ANNOVAR automatically. After real ANNOVAR/dbNSFP annotation is available, pass the real multianno file:

```powershell
python scripts/build_swissprot_feature_table.py --mapped outputs/real/swissprot/humsavar_mapped_variants.csv --output-dir outputs/real/swissprot --annovar-multianno outputs/real/swissprot/swissprot_humsavar.hg38_multianno.txt
```

Expected full feature output after annotation:

- `outputs/real/swissprot/swissprot_feature_table.csv`

## Step D: Evaluate On SwissProt/HUMSAVAR

Script:

```powershell
python scripts/evaluate_on_swissprot.py --swissprot-features outputs/real/swissprot/swissprot_feature_table.csv --model-path <compatible_model.pkl>
```

If no compatible trained model exists, reference training must be explicitly approved before using:

```powershell
python scripts/evaluate_on_swissprot.py --swissprot-features outputs/real/swissprot/swissprot_feature_table.csv --train-reference --clinvar-train-size 5000 --model lightgbm --seed 13
```

Outputs:

- `outputs/real/swissprot/swissprot_metrics.csv`
- `outputs/real/swissprot/swissprot_predictions.csv`
- `outputs/real/swissprot/SWISSPROT_EVALUATION_REPORT.md`

Metric tuple:

`(Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, MCC, AUROC, AUPRC, Brier, ECE)`

## Reporting Language

Use this wording in the paper:

SwissProt/HUMSAVAR external validation is separate from random ClinVar split and is used to test generalization.

Do not report SwissProt/HUMSAVAR as final until mapping, annotation, gene-overlap filtering, and external-evaluation metrics are all verified.
