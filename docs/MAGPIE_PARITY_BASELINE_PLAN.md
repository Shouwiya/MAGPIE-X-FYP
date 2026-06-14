# MAGPIE-Parity Baseline Plan

Generated: 2026-06-05

Status: plan only. No long training was run for this planning step.

## Objective

Refocus the FYP around a MAGPIE-style baseline first, then add MAGPIE-X enhancements. The near-term goal is the closest possible MAGPIE-parity baseline using currently available data, while clearly stating that exact MAGPIE reproduction is not claimed until feature parity and validation parity are complete.

## Reference Protocol To Approximate

- Main model: LightGBM.
- Training labels: benign/likely benign vs pathogenic/likely pathogenic; remove conflicting/unknown.
- Variant types: nonsynonymous SNV/missense, stop-gain, start-loss, frameshift, nonframeshift, stop-loss.
- MAGPIE added 11,998 gnomAD ultra-rare benign variants with AF between 1e-5 and 1e-3.
- MAGPIE used a gene-based split to reduce gene-level leakage.
- MAGPIE excluded prior pathogenicity predictor scores to minimize prior bias.

## Current Data We Can Use

- ClinVar-derived feature table: `data/interim/features/basic_feature_table.csv`.
- ANNOVAR/refGene functional annotations.
- gnomAD30-style population AF columns.
- dbNSFP42a raw annotations, with prior predictor scores removed for parity mode.
- HUMSAVAR/SwissProt mapping audit and feature-reuse evaluation.
- AlphaMissense exists as a project resource, but it is not part of MAGPIE-parity baseline.

## Baseline Naming

Use these names consistently:

- `MAGPIE-parity-current`: closest current low-circularity baseline using available clean features.
- `MAGPIE-X enhanced`: later mode that can include dbNSFP predictor scores, AlphaMissense, ESM, GNN residuals, and fusion.
- `Validation stage`: gene-held-out, temporal, rare-variant, SwissProt no-overlap, calibration, and external/federated checks.

Do not call the current baseline an exact MAGPIE reproduction.

## Clean Feature Engineering Plan

Use only existing real data. Do not download optional databases for this phase.

| Feature group | Planned current implementation | Circularity note |
| --- | --- | --- |
| AF bins and flags | `af_max`, `af_min`, `af_mean`, `af_range`, rare/common bins, missingness indicators | Clean population features; not label-derived. |
| Consequence/type one-hot | Existing `consequence_*`, `func_refgene`, `exonic_func_refgene`, splice/stop/indel/missense summary flags | Clean functional effects. |
| Conservation | Raw GERP, phyloP, phastCons, SiPhy, fitCons summaries and missingness | Use cautiously; exclude learned pathogenicity predictors. |
| Variant length/type | `ref_length`, `alt_length`, `variant_length_change`, SNV/indel flags, length ratios | Clean variant structure features. |
| AA property deltas | Parse unambiguous `aa_change_refgene` missense substitutions; derive charge, polarity, hydrophobicity, volume deltas | Clean biochemical features if derived from amino-acid identities only. |
| BLOSUM-like feature | Derive BLOSUM62 score for unambiguous substitutions; missing otherwise | Clean substitution similarity feature if not trained on ClinVar labels. |
| Transcript/consequence counts | Count transcript consequences from AAChange text when available | Avoid transcript IDs as high-cardinality identifiers. |
| Missingness indicators | Add indicators for AF, conservation, AA-derived, and consequence fields | Clean as long as missingness is annotation availability, not label source. |

## Explicitly Excluded From MAGPIE-Parity Baseline

Do not include SIFT, PolyPhen, REVEL, CADD, MutationTaster, MetaSVM, MetaLR, FATHMM, PROVEAN, M-CAP, AlphaMissense, ESM, ClinPred, BayesDel, DANN, Eigen, GenoCanyon, LINSIGHT, MetaRNN, MutPred, MVP, MPC, PrimateAI, or similar prior pathogenicity predictor outputs in the parity baseline.

Do not add gene identity as a direct random-split feature because same-gene leakage is a known risk.

## Evaluation Plan

First approved training pass:

- Use LightGBM.
- Run only 5k, 10k, and 20k balanced samples.
- Compare old low-circularity, improved MAGPIE-parity-current, and enhanced dbNSFP context.
- Report the tuple: `(Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, MCC, AUROC, AUPRC, Brier, ECE)`.
- Keep enhanced dbNSFP mode labelled as circularity-risk context, not parity.

Next validation pass after approval:

- Gene-held-out split.
- Rare-variant subset with AF < 0.01.
- SwissProt/HUMSAVAR no-exact-overlap evaluation after real annotation for currently unjoined mapped variants.
- Temporal validation if ClinVar dates are available.
- Calibration audit with saved predictions.

## Current Gap-Closing Priority

1. Add AA property delta and BLOSUM-like features from `aa_change_refgene`.
2. Add stricter variant-type filtering to match MAGPIE variant classes.
3. Add gene-held-out split before any final claim.
4. Annotate the 8,312 mapped HUMSAVAR variants that did not join to the current feature table.
5. Only after the parity baseline is stable, move to MAGPIE-X enhancement.

## Claim Boundary

Current claim allowed:

> We are building a closest-possible MAGPIE-style baseline with current data and low-circularity features.

Current claim not allowed:

> We reproduced MAGPIE exactly or proved final improvement over MAGPIE.
