# Federated Learning Report

Generated: 2026-06-14T11:08:07.055826+00:00

Metric tuple: `(Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, MCC, AUROC, AUPRC, Brier, ECE)`

## Experimental Setup

- Real balanced ClinVar-derived samples: 5k, 10k, 15k, 20k, and 25k.
- One stratified 80/20 global train/test split per sample size.
- Five simulated clients partition only the training portion.
- IID clients preserve approximately equal class balance.
- Non-IID clients use reproducible Dirichlet label skew with alpha=0.3.
- A shared preprocessing transform is fit only on the global training partition to keep one feature space across clients; this is a simulation convenience, not a deployment privacy claim.
- SwissProt/HUMSAVAR is not used in training and remains external validation.
- This is single-machine simulated horizontal federated learning, not a privacy or security deployment.

## Feature Modes

- FL-A: MAGPIE-parity low-circularity: ANNOVAR/refGene + gnomAD AF + conservation + variant type/length + AA/BLOSUM62.
- FL-B: MAGPIE-X circularity-risk: FL-A + 86 dbNSFP prior-predictor fields.
- FL-C: MAGPIE-X circularity-risk: FL-B + 12 exact-matched SpliceAI DS/DP fields; available only at 5k, 10k, and 20k because those exact sidecars already exist.

## Best Federated Result Per Sample Size

| sample_size | feature_mode | client_split_type | algorithm | circularity_risk | result_tuple |
| --- | --- | --- | --- | --- | --- |
| 5000 | FL-C | iid | Federated ensemble / soft-voting probability aggregation | high circularity risk | (0.9560, 0.9563, 0.9488, 0.9640, 0.9480, 0.9121, 0.9845, 0.9784, 0.0354, 0.0249) |
| 10000 | FL-B | iid | Federated ensemble / soft-voting probability aggregation | high circularity risk | (0.9680, 0.9679, 0.9699, 0.9660, 0.9700, 0.9360, 0.9928, 0.9926, 0.0253, 0.0142) |
| 15000 | FL-B | iid | Federated ensemble / soft-voting probability aggregation | high circularity risk | (0.9637, 0.9637, 0.9640, 0.9633, 0.9640, 0.9273, 0.9905, 0.9883, 0.0298, 0.0142) |
| 20000 | FL-C | non_iid | FedAvg MLP weighted model averaging | high circularity risk | (0.9677, 0.9677, 0.9680, 0.9675, 0.9680, 0.9355, 0.9935, 0.9928, 0.0259, 0.0087) |
| 25000 | FL-B | iid | Federated ensemble / soft-voting probability aggregation | high circularity risk | (0.9644, 0.9645, 0.9625, 0.9664, 0.9624, 0.9288, 0.9914, 0.9895, 0.0287, 0.0109) |

## Guardrails

- FedAvg and FedProx average neural-network parameters by client sample count.
- The LightGBM method averages predicted probabilities and is not called FedAvg.
- No GraphSAGE federated experiment is claimed.
- No SwissProt/HUMSAVAR labels or variants are used for training.
- Predictor-rich FL-B/FL-C results retain high circularity risk.
