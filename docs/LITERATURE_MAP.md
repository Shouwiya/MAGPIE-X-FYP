# MAGPIE-X Literature Map

## Framing

MAGPIE-X should be framed as a testable residual-evidence hypothesis, not as a claim that the exact method already exists:

> Can independent protein and graph evidence correct the remaining errors of a strong MAGPIE-style tabular model?

The literature supports each component separately. The research question is whether combining them as residual evidence over a MAGPIE-style baseline improves robustness on hard evaluations.

## Method Roles

| Method | Role in MAGPIE-X | What It Supports | What It Does Not Prove |
| --- | --- | --- | --- |
| MAGPIE | Strong tabular annotation baseline | Multimodal exonic annotations, LightGBM-style tabular modeling, gene-aware evaluation, rare/external validation | Does not prove our residual fusion layer or graph/protein correction improves errors |
| VEGN | Graph-context inspiration | Variant-gene heterogeneous graph reasoning where variants exchange information through genes | Does not prove our exact MAGPIE-X graph construction or residual fusion |
| AlphaMissense | Missense-only protein/structure/evolution expert | Precomputed missense pathogenicity evidence from a protein-structure/evolution model | Should not be applied to frameshift, stopgain, splice, or in-frame indel variants as if it covered them |
| ESM / ESM1b | Missense-only protein language model expert | Precomputed missense effect scores from protein sequence language modeling | Does not justify running large ESM inference or treating PLM scores as universal for all variant types |
| ProPath / DYNA | Future disease-specific extension | Disease-specific fine-tuning can be useful when a disease domain is chosen | Not part of the current general MAGPIE-X baseline |
| Residual correction | Fusion design hypothesis | Inspired by boosting and residual learning: model remaining error after a strong baseline | Not already proven for this exact pathogenicity pipeline |

## A. MAGPIE: Tabular Baseline

MAGPIE is the closest baseline family for MAGPIE-X because it predicts pathogenicity across multiple exonic variant types using multimodal annotations and a gradient-boosted tabular model. The MAGPIE paper reports very high internal performance, uses gene-based splitting to reduce leakage from gene-level features, evaluates low-frequency/rare variants, and includes external/orthogonal validation using SwissProt-like and ACMG-guided sets.

MAGPIE therefore defines the tabular baseline standard: ANNOVAR-style annotation, population frequency, conservation, splicing, gene-level, biochemical, and epigenomic feature families should be treated as the foundation before claiming novelty.

Primary citation:

- MAGPIE: accurate pathogenic prediction for multiple variant types using machine learning approach. Genome Medicine. https://link.springer.com/article/10.1186/s13073-023-01274-4

## B. VEGN: Graph-Context Inspiration

VEGN motivates the idea that variants should not always be treated as independent rows. It models variant effect prediction with a graph neural network over a heterogeneous graph containing variants and genes. Variants connect to genes, genes connect through interaction structure, and information can propagate through the graph.

For MAGPIE-X, VEGN supports graph-context exploration. It does not mean our exact graph design already exists. Our plan should be described as applying graph-context evidence to MAGPIE-style pathogenicity prediction and testing whether it corrects residual errors.

Primary citation:

- VEGN: Variant Effect Prediction with Graph Neural Networks. arXiv. https://arxiv.org/abs/2106.13642

## C. AlphaMissense: Missense-Only Expert

AlphaMissense is a missense variant pathogenicity predictor derived from AlphaFold-related protein structural and evolutionary modeling. It is relevant only where the variant is missense and can be mapped to an amino-acid substitution.

MAGPIE-X should consume AlphaMissense as a precomputed expert score for missense variants. It should not apply AlphaMissense to frameshift, stopgain, stoploss, splice donor/acceptor, or in-frame indel variants as if the score were defined for them.

Primary citation:

- Accurate proteome-wide missense variant effect prediction with AlphaMissense. Science / PubMed. https://pubmed.ncbi.nlm.nih.gov/37733863/

## D. ESM / ESM1b: Missense-Only Protein Language Expert

ESM1b-based work supports genome-wide missense variant effect prediction from protein language models. The cited Nature Genetics work uses ESM1b to score possible missense variants and benchmark them against clinical and functional datasets.

For MAGPIE-X, ESM/ESM1b evidence should be introduced as precomputed missense scores. We should not run large-scale ESM inference yet, and we should not treat ESM scores as available for non-missense variant types.

Primary citation:

- Genome-wide prediction of disease variant effects with a deep protein language model. Nature Genetics. https://www.nature.com/articles/s41588-023-01465-0

## E. ProPath / DYNA: Future Disease-Specific Extension

ProPath and DYNA motivate a later disease-specific branch. Their shared idea is that generic variant effect predictors may be less appropriate when disease mechanism matters, and disease-specific fine-tuning can improve pathogenicity prediction in chosen clinical domains.

For the current project, ProPath/DYNA should remain future work. We should not implement disease-specific fine-tuning unless we intentionally choose a disease area such as cardiomyopathy, arrhythmia, retinal disease, or cancer predisposition and build a curated disease-domain dataset.

Primary citations:

- ProPath: Disease-Specific Protein Language Model for Variant Pathogenicity. arXiv. https://arxiv.org/abs/2311.03429
- DYNA: Disease-Specific Language Model for Variant Pathogenicity. arXiv. https://arxiv.org/abs/2406.00164

## F. Residual Correction: Fusion Hypothesis

The residual-correction idea is inspired by two broad machine learning traditions:

- Gradient boosting, where sequential learners fit residual error or pseudo-residual signal after previous learners.
- Residual learning, where models learn an additive correction rather than replacing the original signal.

MAGPIE-X can use this as a disciplined fusion hypothesis:

1. Train a strong MAGPIE-style tabular model.
2. Compute `p_tabular` and `logit_tabular`.
3. Learn a correction `delta` using independent evidence from AlphaMissense, ESM, and graph context.
4. Combine as `z_final = z_tabular + delta`.
5. Convert to probability with `p_final = sigmoid(z_final)`.

This is plausible, but not proven. The research result must come from evaluation.

General method citations:

- Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics. https://doi.org/10.1214/aos/1013203451
- Deep Residual Learning for Image Recognition. CVPR. https://arxiv.org/abs/1512.03385

## Overclaiming Guardrail

- We do not claim to invent GNNs.
- We do not claim to invent AlphaMissense or ESM.
- We do not claim this exact fusion is already proven.
- We do not claim that protein missense experts apply to every exonic variant type.
- Our contribution is applying and evaluating residual evidence fusion over a MAGPIE-style pathogenicity baseline.

## Evidence-to-Experiment Mapping

| Evidence Source | Variant Scope | How To Use | Evaluation Needed |
| --- | --- | --- | --- |
| MAGPIE-style tabular features | Mixed exonic protein-altering variants | Primary baseline and reference score | Random, gene-held-out, rare, imbalanced, external |
| AlphaMissense | Missense only | Residual feature or expert correction | Missense-only ablation and full mixed-type ablation with missingness handled |
| ESM1b | Missense only | Residual feature or expert correction | Same as AlphaMissense; use precomputed scores |
| Graph context | Mixed variants if graph built safely | Residual correction from neighborhood/gene context | Must avoid split leakage, especially gene-held-out |
| ProPath/DYNA-style disease specificity | Disease-specific subsets only | Future disease-domain model | Only after selecting a disease domain |

## Current Implementation Mapping

The current implementation is Phase 1 of the residual plan: no GNN training yet.

| Component | Implemented In | Status |
| --- | --- | --- |
| MAGPIE-style tabular OOF score | `scripts/train_oof_tabular_predictions.py` | Ready for approved training runs |
| AlphaMissense optional score ingestion | `scripts/prepare_external_expert_features.py` | Wrapper ready; requires user-provided score file |
| ESM/ESM1b optional score ingestion | `scripts/prepare_external_expert_features.py` | Wrapper ready; requires user-provided score file |
| Logit-space residual correction | `scripts/train_residual_correction_model.py` | Ready for approved training runs after OOF predictions exist |
| Baseline versus residual comparison | `scripts/compare_baseline_vs_residual.py` | Ready after residual predictions exist |
| VEGN-style graph context | Not implemented in Phase 1 | Phase 2 only |
| ProPath/DYNA disease-specific tuning | Not implemented | Future work only |
