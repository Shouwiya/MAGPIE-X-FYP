# MAGPIE-X Main Contribution

**Title:** MAGPIE-X: Residual Fusion of MAGPIE-Style Annotation, AlphaMissense, and Graph Neural Network Evidence for Human Exonic Variant Pathogenicity Prediction

MAGPIE-X is an evidence-fusion pipeline for exonic variant pathogenicity prediction. Its contribution is not that MAGPIE-style annotation, AlphaMissense, or GNNs are new by themselves. The contribution is to combine them in a staged residual framework and test whether independent evidence can correct remaining errors of a strong tabular baseline.

## Evidence Sources

1. **MAGPIE-style annotation evidence**

   The tabular model uses real annotation features from ClinVar, ANNOVAR/refGene, gnomAD allele frequency, and dbNSFP-derived predictors/conservation scores. XGBoost or LightGBM provides `p_tabular`.

2. **AlphaMissense protein evidence**

   AlphaMissense provides missense-specific protein-level evidence. It is used only as an auxiliary numeric score for missense variants. Its class or score is not treated as a ground-truth label.

3. **GNN relational/context evidence**

   The immediate GNN graph uses variants as nodes. Edges connect variants by same gene, same consequence type, and kNN biological annotation similarity. The GNN learns whether a variant's local context supports increasing or decreasing the tabular prediction.

## Residual Fusion

MAGPIE-X uses logit-level residual correction:

```text
p_tabular = prediction from XGBoost/LightGBM
z_tabular = log(p_tabular / (1 - p_tabular))
delta = correction learned from AlphaMissense + GNN context + uncertainty + variant type
z_final = z_tabular + delta
p_final = sigmoid(z_final)
```

This design avoids naive voting. A baseline that is already confident should not be shifted unless the additional evidence gives a reason. A baseline that is uncertain can be adjusted more strongly if AlphaMissense or graph context supports a correction.

## What Is Immediate Versus Later

Immediate:

- Reference 5k tabular OOF baseline.
- AlphaMissense missense-only feature matching.
- Variant similarity graph construction on reference 5k.
- Lightweight GNN scripts for explicit-approved runs only.
- Residual comparison reports.

Later:

- Latest ClinVar rebuild using `clinvar_20260517.vcf.gz`.
- Gene-held-out and temporal validation.
- SpliceAI, ESM1b, ChromHMM, OMIM, STRING, ACMG, and ProteinGym/MAVE.
- Larger graph training after the reference 5k workflow is validated.

## Honest Claims

MAGPIE-X should be described as an investigation of residual evidence fusion, not as a replacement for established tools. Reports must state which feature mode was used and whether dbNSFP predictor scores were included.
