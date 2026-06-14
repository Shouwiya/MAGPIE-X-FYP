# SpliceAI 10k/20k Low-Circularity Report

Generated: 2026-06-07T11:45:10.078464+00:00

Metric tuple: `(Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, MCC, AUROC, AUPRC, Brier, ECE)`

## Configuration

- Exact approved 10k and 20k balanced samples from the AA/BLOSUM benchmark.
- Identical sample order, split seed, and test rows within each paired comparison.
- LightGBM only, using the established six-candidate validation-AUPRC tuning grid.
- AA/BLOSUM low-circularity baseline compared with the same baseline plus aggregated SpliceAI features.
- Unavailable SpliceAI values remain missing with an explicit availability flag.

## Metrics

| sample_size | mode | n_features | spliceai_available | result_tuple |
| --- | --- | --- | --- | --- |
| 10000 | aa_blosum_low_circularity | 114 | 9161 | (0.9350, 0.9350, 0.9350, 0.9350, 0.9350, 0.8700, 0.9824, 0.9816, 0.0493, 0.0267) |
| 10000 | aa_blosum_spliceai_low_circularity | 126 | 9161 | (0.9350, 0.9353, 0.9307, 0.9400, 0.9300, 0.8700, 0.9820, 0.9792, 0.0495, 0.0260) |
| 20000 | aa_blosum_low_circularity | 114 | 18349 | (0.9467, 0.9468, 0.9456, 0.9480, 0.9455, 0.8935, 0.9875, 0.9865, 0.0401, 0.0097) |
| 20000 | aa_blosum_spliceai_low_circularity | 126 | 18349 | (0.9450, 0.9449, 0.9472, 0.9425, 0.9475, 0.8900, 0.9859, 0.9846, 0.0420, 0.0140) |

## Paired Changes

- 10k: accuracy 0.9350 -> 0.9350 (+0.0000); MCC 0.8700 -> 0.8700; AUROC 0.9824 -> 0.9820; AUPRC 0.9816 -> 0.9792; Brier 0.0493 -> 0.0495; ECE 0.0267 -> 0.0260.
- 20k: accuracy 0.9467 -> 0.9450 (-0.0018); MCC 0.8935 -> 0.8900; AUROC 0.9875 -> 0.9859; AUPRC 0.9865 -> 0.9846; Brier 0.0401 -> 0.0420; ECE 0.0097 -> 0.0140.

## Guardrails

Neither mode uses SIFT, PolyPhen, REVEL, CADD, MutationTaster, MetaSVM, MetaLR, FATHMM, PROVEAN, M-CAP, AlphaMissense, ESM, GNN outputs, or similar prior pathogenicity predictors. The protected full AA feature table was read only.

SpliceAI improved the 5k pilot accuracy, but final benefit is evaluated at 10k/20k and later under gene-held-out/external validation. The completed paired checks were accuracy-neutral at 10k and slightly worse at 20k, so they do not confirm a consistent benefit. These random-split results remain preliminary and are not a full MAGPIE reproduction.
