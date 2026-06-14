# MAGPIE-X Residual Fusion Plan

## Research Question

MAGPIE-X will test:

> Can independent protein and graph evidence correct the remaining errors of a strong MAGPIE-style tabular model?

This is a hypothesis, not a pre-proven method.

## Stage 1: Strong MAGPIE-Style Tabular Baseline

Use the current real feature table as the base:

```text
data/interim/features/basic_feature_table.csv
```

This table is intended to represent real ANNOVAR/gnomAD/dbNSFP annotation evidence. The baseline should remain the anchor model because MAGPIE already shows that multimodal annotations and gradient-boosted tabular learning are a strong foundation for exonic pathogenicity prediction.

Reference:

- MAGPIE. https://link.springer.com/article/10.1186/s13073-023-01274-4

Outputs:

```text
p_tabular
z_tabular = logit(p_tabular)
fold assignments
out-of-fold predictions
error table: y_true, p_tabular, z_tabular, residual/error category
```

## Stage 2: Error Analysis Before Fusion

Before adding new evidence, identify where the tabular model fails:

- false positives
- false negatives
- low-confidence correct calls
- high-confidence wrong calls
- missense-only errors
- non-missense errors
- rare-variant errors
- gene-held-out errors

This decides whether residual correction is worth building. If the tabular model errors are mostly from leakage or label noise, graph/protein fusion may not help.

## Stage 3: Missense-Only Protein Expert Evidence

Use AlphaMissense and ESM/ESM1b only for missense variants.

Allowed:

- precomputed AlphaMissense scores
- precomputed ESM1b or ESM variant-effect scores
- missingness indicators for variants where the expert score is unavailable

Not allowed in the first residual-fusion implementation:

- large ESM inference
- AlphaMissense retraining
- applying missense scores to frameshift, stopgain, splice donor/acceptor, or in-frame indel variants as if they were defined

References:

- AlphaMissense. https://pubmed.ncbi.nlm.nih.gov/37733863/
- ESM1b disease variant effects. https://www.nature.com/articles/s41588-023-01465-0

## Stage 4: Graph Context Evidence

Use graph modeling as context evidence, inspired by VEGN.

Possible graph nodes:

- variant nodes
- gene nodes
- optional transcript nodes later

Possible edges:

- variant to gene
- variant to nearest variants by annotation similarity
- optional gene-gene interaction edges if a real, versioned source is added

The graph component should output a correction signal or embedding, not replace the tabular baseline in the first implementation.

Reference:

- VEGN. https://arxiv.org/abs/2106.13642

Leakage rule:

- For gene-held-out testing, held-out gene labels and held-out gene variant nodes must not leak into training supervision.
- Any graph result must be reported separately for random and gene-held-out splits.

## Stage 5: Residual Fusion Model

Let the tabular model produce:

```text
p_tabular
z_tabular = logit(p_tabular)
```

Let external evidence produce:

```text
x_protein = AlphaMissense/ESM features for missense variants
x_graph = graph embedding or graph score
```

Train a residual correction model:

```text
delta = f(x_protein, x_graph, masks, simple metadata)
```

Final score:

```text
z_final = z_tabular + delta
p_final = sigmoid(z_final)
```

Model choices for the first residual correction:

- logistic regression on `z_tabular`, protein scores, graph score, and masks
- or a shallow gradient-boosted model predicting the residual/correction

Keep the first residual model small. The goal is to test whether independent evidence helps, not to hide the result inside an overfit stack.

## Stage 6: Evaluation

Run the same evidence stack through increasingly difficult settings:

| Evaluation | Purpose |
| --- | --- |
| `balanced_5k_cv` | Verify feature pipeline and reference-level behavior |
| random stratified split | Basic internal comparability |
| gene-held-out split | Test unseen-gene generalization |
| rare-variant evaluation | Test low-AF clinical difficulty |
| imbalanced clinical-style evaluation | Test prevalence-sensitive behavior |
| external SwissProt/ACMG-style validation | Test independence from ClinVar-like training labels |
| calibration study | Test whether probabilities are trustworthy |

Primary metrics:

- AUROC
- AUPRC
- balanced accuracy
- F1
- MCC
- Brier score
- ECE

## Stage 7: Disease-Specific Future Work

ProPath and DYNA support disease-specific fine-tuning as a future branch, but not the immediate general MAGPIE-X pipeline.

Only implement disease-specific fine-tuning after selecting a domain and curating disease-specific labels.

References:

- ProPath. https://arxiv.org/abs/2311.03429
- DYNA. https://arxiv.org/abs/2406.00164

## Overclaiming Guardrail

- We do not claim to invent GNNs.
- We do not claim to invent AlphaMissense or ESM.
- We do not claim this exact fusion is already proven.
- We do not claim protein missense scores solve non-missense pathogenicity.
- Our contribution is applying and evaluating residual evidence fusion over a MAGPIE-style pathogenicity baseline.

## Minimum Implementation Order

1. Freeze tabular baseline folds and save out-of-fold `p_tabular`.
2. Build an error table.
3. Add precomputed missense expert features only where valid.
4. Build a leakage-aware graph context score or embedding.
5. Train a small residual correction model.
6. Evaluate random, gene-held-out, rare, imbalanced, external, and calibrated performance.
7. Report ablations: tabular only, tabular + protein, tabular + graph, tabular + protein + graph.

