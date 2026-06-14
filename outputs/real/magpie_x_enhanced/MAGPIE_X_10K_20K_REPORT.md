# MAGPIE-X Controlled 10k/20k Report

Generated: 2026-06-08T02:17:05.596730+00:00

Metric tuple: `(Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, MCC, AUROC, AUPRC, Brier, ECE)`

Rows C-F use each sample size's exact approved A/B variant order and stratified seed-13 split. SwissProt/HUMSAVAR is not used in training. All modes are reported as high circularity risk.

## Verified Metrics

| sample_size | experiment_id | model | n_features | number_risky_prior_predictor_fields | spliceai_available_rows | alphamissense_matched_rows | result_tuple |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10000 | C | lightgbm | 200 | 86 | 0 | 0 | (0.9670, 0.9669, 0.9708, 0.9630, 0.9710, 0.9340, 0.9923, 0.9924, 0.0267, 0.0172) |
| 10000 | D | lightgbm | 212 | 86 | 9161 | 0 | (0.9660, 0.9659, 0.9688, 0.9630, 0.9690, 0.9320, 0.9931, 0.9923, 0.0283, 0.0215) |
| 10000 | E | lightgbm | 202 | 86 | 0 | 5501 | (0.9660, 0.9659, 0.9688, 0.9630, 0.9690, 0.9320, 0.9932, 0.9932, 0.0260, 0.0164) |
| 10000 | F | lightgbm_graphsage_residual | 214 | 86 | 9161 | 5501 | (0.9685, 0.9685, 0.9699, 0.9670, 0.9700, 0.9370, 0.9934, 0.9930, 0.0259, 0.0102) |
| 20000 | C | lightgbm | 200 | 86 | 0 | 0 | (0.9680, 0.9681, 0.9657, 0.9705, 0.9655, 0.9360, 0.9930, 0.9915, 0.0247, 0.0080) |
| 20000 | D | lightgbm | 212 | 86 | 18349 | 0 | (0.9690, 0.9691, 0.9648, 0.9735, 0.9645, 0.9380, 0.9925, 0.9904, 0.0256, 0.0180) |
| 20000 | E | lightgbm | 202 | 86 | 0 | 10864 | (0.9700, 0.9700, 0.9686, 0.9715, 0.9685, 0.9400, 0.9934, 0.9919, 0.0239, 0.0073) |
| 20000 | F | lightgbm_graphsage_residual | 214 | 86 | 18349 | 10864 | (0.9688, 0.9688, 0.9676, 0.9700, 0.9675, 0.9375, 0.9942, 0.9934, 0.0227, 0.0058) |

## Exact Definitions

### C. magpie_x_dbnsfp

- Feature sources: ANNOVAR/refGene functional consequence; gnomAD allele frequency and derived rarity/missingness summaries; non-predictor conservation; variant type/length; amino-acid physicochemical deltas; BLOSUM62; 86 dbNSFP prior pathogenicity/damagingness score, rank, and prediction fields.
- Total tabular features: 200.
- Risky prior-predictor fields: 86.
- What `+` means: + dbNSFP means 86 published prior-predictor fields are concatenated to the same clean variant rows; no new labels or dataset are added.
- Circularity risk: **High**.
- Limitation: High circularity risk from predictor reuse and possible source-training overlap; random balanced split is not gene-held-out or external validation.

### D. magpie_x_dbnsfp_spliceai

- Feature sources: Mode C feature sources; official masked SpliceAI v1.3 hg38 exact-match max DS_AG/AL/DG/DL, corresponding DP_AG/AL/DG/DL, max DS, availability, record count, and source type.
- Total tabular features: 212.
- Risky prior-predictor fields: 86.
- What `+` means: + SpliceAI means 12 exact-matched mechanistic splicing columns are added to mode C on the same variant IDs.
- Circularity risk: **High**.
- Limitation: Mode C circularity risk remains; SpliceAI coverage is incomplete and transcript/masking/normalization limitations apply.

### E. magpie_x_dbnsfp_alphamissense

- Feature sources: Mode C feature sources; AlphaMissense hg38 exact-match pathogenicity score and availability flag; AlphaMissense class is excluded.
- Total tabular features: 202.
- Risky prior-predictor fields: 86.
- What `+` means: + AlphaMissense means an exact-matched expert missense score and availability flag are added to mode C; no SpliceAI or clinical labels are added.
- Circularity risk: **High**.
- Limitation: Mode C circularity risk remains; AlphaMissense is missense-only and may overlap external expert-model training evidence.

### F. magpie_x_graphsage_residual

- Feature sources: Mode C feature sources; SpliceAI DS/DP sidecar; AlphaMissense score and availability; tabular p/logit; transductive variant graph with same-gene, same-consequence, and kNN feature-similarity edges.
- Total tabular features: 214.
- Risky prior-predictor fields: 86.
- What `+` means: + GraphSAGE means a graph-derived delta corrects the logit of the 214-feature mode C + SpliceAI + AlphaMissense LightGBM; it is residual correction, not stacking.
- Circularity risk: **High**.
- Limitation: High circularity plus transductive graph/same-gene context risk; test features participate in message passing although test labels remain hidden.

## Scale Comparison

- 10k best verified row: F with accuracy 0.9685, AUROC 0.9934, and AUPRC 0.9930.
- 20k best verified row: E with accuracy 0.9700, AUROC 0.9934, and AUPRC 0.9919.

## Graph Runtime Gate

- 10k GraphSAGE completed: yes.
- 20k GraphSAGE completed: yes.

No full-490k or federated training was run. These are random balanced ClinVar-derived comparisons and remain preliminary until gene-held-out, temporal, and external validation.
