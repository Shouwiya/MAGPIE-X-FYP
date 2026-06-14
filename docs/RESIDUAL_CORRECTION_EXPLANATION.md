# Residual Correction Explanation

## Core Hypothesis

MAGPIE-X should test the following hypothesis:

> Can independent protein and graph evidence correct the remaining errors of a strong MAGPIE-style tabular model?

This is deliberately modest. We are not claiming that a residual-fusion pathogenicity model is already established. We are proposing a measurable experiment.

## Why Start With A Tabular Model?

MAGPIE-style tabular models already use the evidence most clinicians and variant scientists expect:

- population frequency
- conservation
- consequence type
- gene-level constraint
- splicing annotations
- biochemical amino-acid change features
- meta-predictor and dbNSFP-style scores

MAGPIE reports strong performance using this family of features and performs harder validation such as gene-based splitting, rare variant testing, and orthogonal SwissProt/ACMG-style validation. That makes the tabular model the correct base learner.

Citation:

- MAGPIE: accurate pathogenic prediction for multiple variant types using machine learning approach. https://link.springer.com/article/10.1186/s13073-023-01274-4

## Why Residual Correction?

If the tabular model is already strong, replacing it outright is risky. A more controlled strategy is to ask what the tabular model still gets wrong.

Residual correction treats the tabular model as the anchor:

```text
tabular annotations -> tabular model -> z_tabular
extra evidence      -> correction model -> delta
z_final = z_tabular + delta
p_final = sigmoid(z_final)
```

This is inspired by:

- gradient boosting, where later learners improve earlier predictions by fitting residual or pseudo-residual signal
- residual learning, where a model learns an additive correction instead of relearning the full mapping

Citations:

- Greedy Function Approximation: A Gradient Boosting Machine. https://doi.org/10.1214/aos/1013203451
- Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

## What Evidence Can Correct?

### Missense Protein Evidence

AlphaMissense and ESM1b should be treated as missense-only experts.

They may help where the tabular model has weak protein-level evidence, for example:

- substitutions in conserved domains
- substitutions where generic dbNSFP scores disagree
- variants in proteins where sequence/structure context matters

They should not be used for non-missense variants as if they were universal pathogenicity scores.

Citations:

- AlphaMissense. https://pubmed.ncbi.nlm.nih.gov/37733863/
- ESM1b variant effect prediction. https://www.nature.com/articles/s41588-023-01465-0

### Graph Context Evidence

Graph evidence may help when individual variant rows are sparse. VEGN motivates a heterogeneous graph where variants and genes exchange information through gene membership and gene-gene context.

MAGPIE-X can test graph evidence in a safer residual form:

```text
variant/gene/context graph -> graph encoder -> delta_graph
```

The critical risk is graph leakage. For gene-held-out tests, graph construction and message passing must not allow held-out gene labels to influence training predictions.

Citation:

- VEGN: Variant Effect Prediction with Graph Neural Networks. https://arxiv.org/abs/2106.13642

## Proposed Residual Fusion Formula

Let:

```text
p_tabular = tabular probability
z_tabular = logit(p_tabular)
x_missense = AlphaMissense/ESM evidence for missense variants
x_graph = graph/context embedding or score
```

Learn:

```text
delta = f(x_missense, x_graph, metadata)
```

Then:

```text
z_final = z_tabular + delta
p_final = sigmoid(z_final)
```

For non-missense variants:

```text
x_missense = missing / masked
delta_missense_component = 0 unless explicitly learned from missingness-safe indicators
```

This keeps missense-specific experts from contaminating variant types they do not cover.

## Current No-GNN Implementation Stage

The first implementation deliberately excludes graph neural networks. The goal is to prove that the residual-correction plumbing works before adding VEGN-style graph context.

Implemented stage scripts:

| Script | Purpose |
| --- | --- |
| `scripts/train_oof_tabular_predictions.py` | Train LightGBM and/or XGBoost with 5-fold CV and save tabular out-of-fold probabilities |
| `scripts/prepare_external_expert_features.py` | Attach optional precomputed AlphaMissense and ESM/ESM1b scores without fabricating missing data |
| `scripts/train_residual_correction_model.py` | Train fold-wise residual correction models using tabular logits, uncertainty, consequence type, and optional expert scores |
| `scripts/compare_baseline_vs_residual.py` | Compare tabular-only and residual-corrected predictions using discrimination and calibration metrics |

If no AlphaMissense or ESM files are provided, the correction model is labeled as `no external expert features available`. That run is useful as a framework smoke test, but it should not be presented as protein-expert fusion.

## What Would Count As Success?

Residual correction is successful only if it improves held-out performance without increasing leakage or destroying calibration.

Minimum evidence:

- better or equal AUROC/AUPRC on `balanced_5k_cv`
- improved balanced accuracy or MCC on gene-held-out testing
- improved rare-variant AUPRC/MCC
- no worsening, or a post-calibration improvement, in Brier score and ECE
- clear ablation showing which evidence source helps

## What Would Count As Failure?

The hypothesis should be rejected or revised if:

- residual fusion only improves random splits
- graph evidence collapses under gene-held-out testing
- missense-only experts improve missense variants but degrade mixed exonic evaluation
- calibration worsens substantially
- gains disappear after removing gene or allele-frequency leakage

## Overclaiming Guardrail

- We do not claim to invent residual learning.
- We do not claim to invent GNNs.
- We do not claim to invent AlphaMissense or ESM.
- We do not claim this exact MAGPIE-X fusion is already proven.
- The contribution is evaluating whether independent protein and graph evidence can correct residual errors of a strong MAGPIE-style tabular baseline.
