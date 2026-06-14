# Dataset Status And Experiment Modes

## Already Downloaded Real Data

- Latest ClinVar GRCh38: `data/raw/clinvar/latest/clinvar_20260517.vcf.gz`
- ClinVar variant summary: `data/raw/clinvar/latest/variant_summary.txt.gz`
- Historic ClinVar GRCh38: `data/raw/clinvar/historic/clinvar_20201219.vcf.gz`
- Current feature table: `data/interim/features/basic_feature_table.csv`
- ANNOVAR/refGene: `third_party/annovar/humandb/hg38_refGene*`
- gnomAD30 genome via ANNOVAR: `third_party/annovar/humandb/hg38_gnomad30_genome*`
- dbNSFP42a via ANNOVAR: `third_party/annovar/humandb/hg38_dbnsfp42a*`
- AlphaMissense hg38: `data/external/alphamissense/AlphaMissense_hg38.tsv.gz`
- AlphaMissense gene file: `data/external/alphamissense/AlphaMissense_gene_hg38.tsv.gz`
- HUMSAVAR: `data/external/humsavar/humsavar.txt`

The current `basic_feature_table.csv` exists and can be used for the reference 5k benchmark, but it was built before the newest ClinVar `20260517` download. A later rebuild should use `data/raw/clinvar/latest/clinvar_20260517.vcf.gz`.

## Forbidden Research Inputs

Do not use:

- `outputs/real/processed/annovar.csv`
- `outputs/real/processed/dbnsfp.csv`

Toy or synthetic data may remain in the repository for unit tests and smoke tests, but it must never be used for research results.

## Future/Later Datasets

- SpliceAI
- ESM1b
- ChromHMM
- OMIM
- STRING
- ACMG benchmark data
- ProteinGym/MAVE

Do not download these during the current phase.

## Experiment Modes

### 1. `magpie_parity_mode`

Purpose: create a cleaner MAGPIE-style baseline and avoid overclaiming independence from prior predictor scores.

This mode should emphasize broad annotation groups:

- Consequence type.
- Allele frequency and AF bins.
- Conservation summaries.
- Basic gene/transcript annotation.
- Carefully selected non-leaky annotations.

Reports using this mode should describe it as a MAGPIE-style parity or sanity-check mode.

### 2. `magpie_x_enhanced_mode`

Purpose: evaluate the full MAGPIE-X residual fusion idea.

This mode may use:

- dbNSFP predictor scores.
- AlphaMissense missense-only scores.
- Variant graph/GNN context.
- Residual fusion on top of tabular OOF predictions.

dbNSFP predictor scores are useful, but they can make the model more like a meta-predictor because dbNSFP contains outputs from many existing variant-effect tools. Reports must state this honestly and avoid implying that all evidence sources are fully independent.

## Current Safe Run Level

Allowed now:

- Compile checks.
- `--help` checks.
- Reference 5k graph-build smoke tests.
- 100 to 200 node graph visualization smoke tests.

Not allowed unless explicitly approved:

- Full 490k graph construction or GNN training.
- Long model training.
- New large dataset downloads.
