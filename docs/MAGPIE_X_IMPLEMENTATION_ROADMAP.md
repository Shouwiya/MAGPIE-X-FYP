# MAGPIE-X Implementation Roadmap

## Current Position

MAGPIE-X now has a real MAGPIE-style feature foundation:

```text
data/interim/features/basic_feature_table.csv
```

This table was built from real ClinVar-derived variants and real ANNOVAR annotation resources, including `refGene`, `gnomad30_genome`, and `dbnsfp42a`. The known synthetic fallback files under `outputs/real/processed/annovar.csv` and `outputs/real/processed/dbnsfp.csv` must not be used for MAGPIE-X experiments.

The current reference 5k benchmark shows that the real tabular feature pipeline is functioning at a strong baseline level:

| Model | Accuracy | AUROC |
| --- | ---: | ---: |
| LightGBM | 0.9580 | 0.9845 |
| XGBoost | 0.9572 | 0.9884 |
| RandomForest | 0.9556 | 0.9887 |

These results should be treated as a starting point, not the final MAGPIE-X contribution.

## Research Hypothesis

MAGPIE-X should test a specific residual-evidence question:

> Can independent protein and graph evidence correct the remaining errors of a strong MAGPIE-style tabular model?

This does not claim that our exact fusion method already exists. It is a testable engineering and evaluation hypothesis.

## Phase 1: No-GNN Residual Framework

Phase 1 builds the residual correction machinery before adding graph neural networks.

### Step 1: Save Tabular OOF Predictions

Script:

```text
scripts/train_oof_tabular_predictions.py
```

Purpose:

- train LightGBM and/or XGBoost with 5-fold CV
- save out-of-fold `p_tabular`
- compute `logit_tabular`
- compute `uncertainty_tabular = abs(p_tabular - 0.5)`
- preserve `variant_id`, `label`, `fold`, `model_name`, `consequence`, and `gene` where available

Output:

```text
outputs/real/magpie_x_residual/oof_tabular_predictions.csv
```

### Step 2: Attach Optional External Expert Features

Script:

```text
scripts/prepare_external_expert_features.py
```

Purpose:

- accept optional user-provided AlphaMissense score files
- accept optional user-provided ESM/ESM1b score files
- never fabricate missing expert scores
- apply expert scores only to missense variants
- write availability flags even when score files are absent

Output:

```text
outputs/real/magpie_x_residual/external_expert_features.csv
outputs/real/magpie_x_residual/missing_expert_features_report.md
```

### Step 3: Train Residual Correction Models

Script:

```text
scripts/train_residual_correction_model.py
```

Purpose:

- merge tabular OOF predictions with optional expert features
- compute `residual = label - p_tabular`
- train a small correction model fold-wise
- prefer logit-space correction:

```text
z_final = logit_tabular + delta
p_final = sigmoid(z_final)
```

Supported first-stage correction models:

- LogisticRegression
- XGBoost, if installed

If AlphaMissense/ESM files are absent, this still runs as an uncertainty-plus-variant-type correction baseline and is clearly labeled as `no external expert features available`.

### Step 4: Compare Baseline And Residual Scores

Script:

```text
scripts/compare_baseline_vs_residual.py
```

Purpose:

- compare tabular baseline probabilities against residual-corrected probabilities
- report accuracy, balanced accuracy, F1, AUROC, AUPRC, MCC, Brier score, and ECE

Outputs:

```text
outputs/real/magpie_x_residual/residual_predictions.csv
outputs/real/magpie_x_residual/residual_metrics.csv
outputs/real/magpie_x_residual/RESIDUAL_CORRECTION_REPORT.md
```

## Phase 2: Graph Context After Residual Framework Works

VEGN motivates graph-context modeling with variants and genes in a heterogeneous graph. MAGPIE-X should not train a GNN until Phase 1 can produce stable OOF predictions, expert-feature masks, residual predictions, and comparison reports.

Phase 2 should add graph evidence as another correction signal:

```text
delta = f(logit_tabular, uncertainty, missense_expert_features, graph_context)
```

The graph must be built with leakage controls, especially for gene-held-out evaluation.

## Phase 3: Robustness Evaluation

The main research contribution should come from hard evaluations:

- gene-held-out testing
- rare-variant evaluation
- imbalanced clinical-style testing
- calibration
- external SwissProt/ACMG-style evaluation where available
- later graph/context augmentation

## Literature Roles

| Literature | Role |
| --- | --- |
| MAGPIE | Strong tabular multimodal annotation baseline with gene-aware and external validation precedent |
| AlphaMissense | Missense-only protein/structure/evolution expert, used only through precomputed scores |
| ESM/ESM1b | Missense-only protein language model expert, used only through precomputed scores for now |
| VEGN | Graph-context inspiration for later variant-gene graph modeling |
| ProPath/DYNA | Future disease-specific extension after choosing a disease domain |
| Gradient boosting and residual learning | Conceptual basis for learning corrections after a strong base model |

## Overclaiming Guardrail

- We do not claim to invent MAGPIE.
- We do not claim to invent AlphaMissense or ESM.
- We do not claim to invent GNNs.
- We do not claim this exact residual fusion is already proven.
- We do not claim missense-only protein experts apply to all exonic variant types.
- Our contribution is applying and evaluating residual evidence fusion over a MAGPIE-style pathogenicity baseline.

## First Commands After Approval

Do not start full training until explicitly approved. When approved, Phase 1 can be run as:

```bash
python scripts/train_oof_tabular_predictions.py --benchmark balanced_5k_cv --models lightgbm xgboost
python scripts/prepare_external_expert_features.py
python scripts/train_residual_correction_model.py --models logistic xgboost
python scripts/compare_baseline_vs_residual.py
```

