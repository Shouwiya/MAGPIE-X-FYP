# Controlled MAGPIE-Parity versus MAGPIE-X 5k Comparison

Generated: 2026-06-07T15:04:40.174182+00:00

Metric tuple: `(Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, MCC, AUROC, AUPRC, Brier, ECE)`

## Controlled Dataset And Split

- Training/evaluation dataset: Controlled ClinVar-derived balanced 5k sample; 2,500 benign/likely benign and 2,500 pathogenic/likely pathogenic variants.
- Split: Exact shared stratified random 80/20 split: 4,000 train and 1,000 test (sample/split seed 13; identical variant IDs in modes A-G).
- SwissProt/HUMSAVAR used in this training: **No**. SwissProt/HUMSAVAR is external validation and will later test generalization of the ClinVar-trained model.
- Labels used to match SpliceAI or AlphaMissense: **No**.
- No mode adds new clinical labels or a new training dataset.

## Results

| mode_id | long_descriptive_mode_name | ml_model | number_total_tabular_features | number_clean_biological_features | number_risky_prior_predictor_fields | fusion_type | circularity_risk_level | metric_tuple |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | MAGPIE-parity low-circularity LightGBM using ANNOVAR/refGene + gnomAD AF + conservation + variant type/length + AA/BLOSUM62 | LightGBM binary classifier | 114 | 114 | 0 | No fusion; one clean tabular feature matrix | Low-circularity | (0.9260, 0.9267, 0.9176, 0.9360, 0.9160, 0.8522, 0.9788, 0.9738, 0.0537, 0.0324) |
| B | MAGPIE-parity low-circularity LightGBM with SpliceAI DS/DP feature fusion on the same ClinVar variants | LightGBM binary classifier | 126 | 126 | 0 | Tabular feature fusion: 12 SpliceAI columns concatenated to the same 5k variant rows | Low-circularity | (0.9340, 0.9343, 0.9306, 0.9380, 0.9300, 0.8680, 0.9796, 0.9718, 0.0507, 0.0383) |
| C | MAGPIE-X circularity-risk LightGBM using clean features + 86 dbNSFP prior predictor fields | LightGBM binary classifier | 200 | 114 | 86 | Tabular feature fusion: 86 prior predictor fields concatenated to the same clean variant rows | High circularity risk | (0.9590, 0.9590, 0.9599, 0.9580, 0.9600, 0.9180, 0.9865, 0.9797, 0.0337, 0.0206) |
| D | MAGPIE-X circularity-risk LightGBM using clean features + 86 dbNSFP prior predictor fields + SpliceAI DS/DP feature fusion | LightGBM binary classifier | 212 | 126 | 86 | Tabular feature fusion: dbNSFP predictor and SpliceAI columns concatenated to the same 5k rows | High circularity risk | (0.9620, 0.9622, 0.9583, 0.9660, 0.9580, 0.9240, 0.9856, 0.9772, 0.0324, 0.0266) |
| E | MAGPIE-X circularity-risk LightGBM using clean features + 86 dbNSFP prior predictor fields + AlphaMissense feature fusion | LightGBM binary classifier | 202 | 114 | 86 | Tabular feature fusion: AlphaMissense score/availability concatenated to the same 5k rows | High circularity risk | (0.9560, 0.9558, 0.9597, 0.9520, 0.9600, 0.9120, 0.9864, 0.9804, 0.0354, 0.0247) |
| F | MAGPIE-X circularity-risk LightGBM using clean features + 86 dbNSFP prior predictor fields + SpliceAI + AlphaMissense feature fusion | LightGBM binary classifier | 214 | 126 | 86 | Tabular feature fusion: dbNSFP, SpliceAI, and AlphaMissense columns concatenated to the same 5k rows | High circularity risk | (0.9610, 0.9611, 0.9583, 0.9640, 0.9580, 0.9220, 0.9853, 0.9756, 0.0332, 0.0255) |
| G | MAGPIE-X same-split two-layer GraphSAGE residual logit correction over the mode-F LightGBM prediction | LightGBM tabular model + two-layer GraphSAGE residual correction (64 hidden units, dropout 0.25, 50 fixed epochs) | 214 | 126 | 86 | Residual logit correction: z_final = logit(p_tabular) + delta_GraphSAGE; p_final = sigmoid(z_final) | High circularity and transductive graph risk | (0.9590, 0.9592, 0.9545, 0.9640, 0.9540, 0.9180, 0.9861, 0.9760, 0.0328, 0.0151) |

## What The Plus Sign Means

- `+ SpliceAI`: the 12 DS/DP, maximum-score, availability, record-count, and source-type columns are exact-matched to the same ClinVar variant rows. This is tabular feature fusion, not a new dataset or label source.
- `+ AlphaMissense`: `alphamissense_score` and `alphamissense_available` are exact-matched to the same hg38 variant rows. AlphaMissense is an external expert prediction, not a clinical label.
- `+ dbNSFP prior predictors`: 86 published predictor score/rank/prediction fields are added to the same rows. This is intentionally high-circularity.
- `+ GraphSAGE residual correction`: a graph model predicts `delta_gnn` for the same variant IDs and applies `z_final = logit(p_tabular) + delta_gnn`; this is residual logit correction, not feature stacking.
- None of these additions mixes SwissProt/HUMSAVAR into the controlled 5k training sample.

## Feature Group Definitions

### Clean MAGPIE-Parity Features

The 114-feature clean matrix contains ANNOVAR/refGene functional consequence, gnomAD allele-frequency and rarity/missingness summaries, non-predictor conservation, variant type/length, amino-acid physicochemical deltas, and BLOSUM62. These are treated as low-circularity biological annotations.

### SpliceAI Feature Fusion

`Official masked SpliceAI v1.3 hg38 exact-match sidecar: max_DS_AG, max_DS_AL, max_DS_DG, max_DS_DL, DP_AG, DP_AL, DP_DG, DP_DL, max_spliceai_DS, spliceai_available, spliceai_record_count, spliceai_source_types`. Real scores are available for 4,582 of 5,000 variants; unmatched variants retain explicit unavailability rather than fabricated scores.

### AlphaMissense Feature Fusion

`AlphaMissense hg38 exact-match sidecar: alphamissense_score and alphamissense_available; AlphaMissense class was not used`. Exact hg38 matching provides scores for 2,713 variants among 2,893 missense variants. Matching uses normalized chromosome, position, REF, and ALT only.

### dbNSFP Prior-Predictor Fields

The following 86 fields are useful for performance but carry circularity risk because they are already pathogenicity or damagingness predictors:

- **SIFT / SIFT4G:** `SIFT_score`, `SIFT_converted_rankscore`, `SIFT4G_score`, `SIFT4G_converted_rankscore`, `SIFT_pred`, `SIFT4G_pred`
- **PolyPhen-2:** `Polyphen2_HDIV_score`, `Polyphen2_HDIV_rankscore`, `Polyphen2_HVAR_score`, `Polyphen2_HVAR_rankscore`, `Polyphen2_HDIV_pred`, `Polyphen2_HVAR_pred`
- **LRT:** `LRT_score`, `LRT_converted_rankscore`, `LRT_pred`
- **MutationTaster:** `MutationTaster_score`, `MutationTaster_converted_rankscore`, `MutationTaster_pred`
- **MutationAssessor:** `MutationAssessor_score`, `MutationAssessor_rankscore`, `MutationAssessor_pred`
- **FATHMM:** `FATHMM_score`, `FATHMM_converted_rankscore`, `fathmm-MKL_coding_score`, `fathmm-MKL_coding_rankscore`, `fathmm-XF_coding_score`, `fathmm-XF_coding_rankscore`, `FATHMM_pred`, `fathmm-MKL_coding_pred`, `fathmm-XF_coding_pred`
- **PROVEAN:** `PROVEAN_score`, `PROVEAN_converted_rankscore`, `PROVEAN_pred`
- **VEST4:** `VEST4_score`, `VEST4_rankscore`
- **MetaSVM / MetaLR / MetaRNN:** `MetaSVM_score`, `MetaSVM_rankscore`, `MetaLR_score`, `MetaLR_rankscore`, `MetaRNN_score`, `MetaRNN_rankscore`, `MetaSVM_pred`, `MetaLR_pred`, `MetaRNN_pred`
- **M-CAP:** `M-CAP_score`, `M-CAP_rankscore`, `M-CAP_pred`
- **REVEL:** `REVEL_score`, `REVEL_rankscore`
- **MutPred:** `MutPred_score`, `MutPred_rankscore`
- **MVP / MPC:** `MVP_score`, `MVP_rankscore`, `MPC_score`, `MPC_rankscore`
- **PrimateAI:** `PrimateAI_score`, `PrimateAI_rankscore`, `PrimateAI_pred`
- **DEOGEN2:** `DEOGEN2_score`, `DEOGEN2_rankscore`, `DEOGEN2_pred`
- **BayesDel:** `BayesDel_addAF_score`, `BayesDel_addAF_rankscore`, `BayesDel_noAF_score`, `BayesDel_noAF_rankscore`, `BayesDel_addAF_pred`, `BayesDel_noAF_pred`
- **ClinPred:** `ClinPred_score`, `ClinPred_rankscore`, `ClinPred_pred`
- **LIST-S2:** `LIST-S2_score`, `LIST-S2_rankscore`, `LIST-S2_pred`
- **CADD:** `CADD_raw`, `CADD_raw_rankscore`, `CADD_phred`
- **DANN:** `DANN_score`, `DANN_rankscore`
- **Eigen / Eigen-PC:** `Eigen-raw_coding`, `Eigen-raw_coding_rankscore`, `Eigen-PC-raw_coding`, `Eigen-PC-raw_coding_rankscore`
- **GenoCanyon:** `GenoCanyon_score`, `GenoCanyon_rankscore`
- **LINSIGHT:** `LINSIGHT`, `LINSIGHT_rankscore`

### Graph Module

- Architecture: two-layer GraphSAGE, 64 hidden units, dropout 0.25, 50 fixed epochs.
- Nodes: 5,000 variant nodes only. No separate gene nodes are present.
- Directed edges: 100,310 total: 14,442 same-gene capped edges, 29,994 same-consequence capped edges, and 55,874 kNN feature-similarity edges.
- No gene-gene interaction edge is used because no real interaction resource is part of this experiment.
- Node inputs: selected clean and predictor-rich tabular values, SpliceAI, AlphaMissense, consequence one-hot indicators, `p_tabular`, and `logit_tabular`.
- Training labels are visible only for the 4,000 training nodes. No validation labels are used, and test labels are hidden until final metrics.
- The graph is transductive: test-node features and edges are visible, but test labels are not.

## Scientific Interpretation

- **Mode A:** Provides the defensible low-circularity reference; it does not reproduce every MAGPIE feature group or gene-held-out split. Limitation: Random balanced ClinVar split; partial MAGPIE parity; no gene-held-out, temporal, rare-variant, or external validation.
- **Mode B:** Improves 5k accuracy and MCC over mode A, but lowers AUPRC and worsens ECE; larger 10k/20k checks were not consistently positive. Limitation: 418 variants lacked real SpliceAI records; random balanced split and transcript/masking/normalization limitations remain.
- **Mode C:** Substantially outperforms mode A on this random split, but the gain cannot be interpreted as low-circularity evidence. Limitation: Predictor reuse and training-data overlap may inflate performance; separate gene-held-out/external validation is required.
- **Mode D:** Achieves the highest controlled 5k accuracy (0.9620), while AUROC/AUPRC and calibration do not uniformly improve over mode C. Limitation: Best accuracy is random-split, predictor-rich evidence, not a low-circularity or external-generalization claim.
- **Mode E:** Does not improve accuracy over mode C, although AUPRC is slightly higher; the expert feature is not universally beneficial. Limitation: AlphaMissense is missense-only and available for 2,713/5,000 rows; it is an expert predictor with circularity/overlap risk.
- **Mode F:** Accuracy is below mode D, so adding AlphaMissense to the SpliceAI model does not improve the primary controlled result. Limitation: Combines multiple expert predictors with overlapping biological scope; random-split performance may overstate generalization.
- **Mode G:** Does not improve accuracy over mode F or the best mode D; it improves ECE versus mode F, indicating a calibration effect. Limitation: Transductive random-split graph permits test features and same-gene neighborhoods in message passing; external and gene-held-out validation are still required.

The best controlled 5k accuracy is mode D at 0.9620. That mode is not low-circularity: it includes 86 dbNSFP prior predictor fields. Mode G is a valid same-split residual correction experiment, but it does not improve accuracy over mode F or mode D; its clearest change is lower ECE relative to mode F.

No 10k/20k fusion, full-490k training, federated learning, or new SwissProt/HUMSAVAR training was performed.

`FYP_Literature_Results_Table_Omar.xlsx` was not modified because it is a literature-results workbook, whereas modes A--G are results from this project's controlled experiment. Mixing them would blur source provenance; the project results are documented in the controlled CSV, mode explanation table, consolidated matrix, and paper.
