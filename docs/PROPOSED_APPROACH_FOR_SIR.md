# Proposed Approach For Sir

**Project title:** MAGPIE-X: Residual Fusion of MAGPIE-Style Annotation, AlphaMissense, and Graph Neural Network Evidence for Human Exonic Variant Pathogenicity Prediction

MAGPIE-X combines three complementary evidence sources:

- MAGPIE-style tabular annotation evidence from ClinVar, ANNOVAR, gnomAD, and dbNSFP-derived features.
- AlphaMissense protein-level evidence for missense variants only.
- GNN relational/context evidence between variants, with optional gene nodes later.

The main idea is not simple voting. The tabular model first predicts pathogenicity, then residual fusion learns how much to adjust that prediction using AlphaMissense, graph context, uncertainty, and variant type.

```text
p_tabular = prediction from XGBoost/LightGBM
z_tabular = log(p_tabular / (1 - p_tabular))
delta = correction learned from AlphaMissense + GNN context + uncertainty + variant type
z_final = z_tabular + delta
p_final = sigmoid(z_final)
```

## A. Fusion Of MAGPIE And Deep Architecture

**Proposed method:** MAGPIE-style features plus GNN graph residual correction.

In the immediate graph direction, variants are graph nodes. Edges connect variants using biologically meaningful relations:

- Same-gene edges: variants annotated to the same gene.
- Same-consequence edges: variants with the same functional consequence type.
- kNN similarity edges: variants close in biological annotation space, using features such as AF, CADD, REVEL, SIFT, PolyPhen, GERP, phyloP, and phastCons.

The GNN learns relational/context evidence. In residual mode, the GNN does not replace the MAGPIE-style model; it learns `delta_gnn`, a correction to the tabular logit. Optional later graph extensions can add gene nodes and STRING gene-gene edges, but the immediate implementation already includes a variant graph and lightweight GNN on the reference 5k benchmark.

## B. Fusion Of MAGPIE Features And Handcrafted/Expert Features

**Proposed method:** MAGPIE-style features plus AlphaMissense score, AF bins, missingness flags, uncertainty features, and variant-type indicators.

The immediate handcrafted/expert fusion features are:

- AlphaMissense score for missense variants only.
- AlphaMissense availability flag.
- AF bins, such as missing, zero, ultra-rare, rare, low-frequency, and common.
- Missingness flags for AF and AlphaMissense.
- Tabular uncertainty, such as `|p_tabular - 0.5|` and `|logit(p_tabular)|`.
- Variant-type indicators from consequence annotations.

AlphaMissense is auxiliary evidence, not a ground-truth label. It must not be applied to non-missense variants, and missing AlphaMissense values must remain missing.

## C. Applying An Approach From Another Research Problem

**Proposed method:** logit-level residual correction inspired by boosting and residual learning.

Boosting and residual learning both focus on modeling what the previous stage did not explain. MAGPIE-X applies the same idea at the probability-logit level:

```text
z_tabular = logit(p_tabular)
z_final = z_tabular + delta
p_final = sigmoid(z_final)
```

The correction model learns remaining errors of the tabular baseline using AlphaMissense, graph context, uncertainty, and variant type. This makes the contribution an evidence-fusion and error-correction design, rather than a claim that any one feature source is sufficient by itself.

## Immediate Implementation Order

1. Generate out-of-fold tabular predictions from `data/interim/features/basic_feature_table.csv` on the reference 5k benchmark.
2. Match `data/external/alphamissense/AlphaMissense_hg38.tsv.gz` to missense variants by chromosome, position, reference allele, and alternate allele.
3. Build a capped variant similarity graph on the reference 5k benchmark.
4. Train only lightweight GNN experiments after explicit approval.
5. Compare tabular baseline, tabular plus AlphaMissense residual, GNN-only, tabular plus GNN residual, and tabular plus AlphaMissense plus GNN residual when all inputs exist.

## Guardrails

- Do not use `outputs/real/processed/annovar.csv`.
- Do not use `outputs/real/processed/dbnsfp.csv`.
- Do not use toy or synthetic data for research results.
- Do not download more datasets in this phase.
- Do not run full 490k graph training.
- Do not claim AlphaMissense labels are ground truth.
- Do not fabricate AlphaMissense, ESM1b, or other missing expert scores.
- Keep SpliceAI, ESM1b, ChromHMM, OMIM, STRING, ACMG, and ProteinGym/MAVE as future/later extensions.
