# GNN + AlphaMissense Residual Fusion Plan

This plan makes both AlphaMissense and GNN evidence immediate MAGPIE-X components.

## Stage 1: Tabular Baseline

Train XGBoost or LightGBM on real MAGPIE-style annotation features and generate out-of-fold predictions on the reference 5k benchmark:

```text
p_tabular
z_tabular = logit(p_tabular)
uncertainty = |p_tabular - 0.5|
```

These OOF predictions are required for residual training because they prevent the correction model from seeing overfit in-sample tabular scores.

## Stage 2: AlphaMissense Features

Match `AlphaMissense_hg38.tsv.gz` to variants by:

```text
chrom, pos, ref, alt
```

Rules:

- Use AlphaMissense only for missense variants.
- Do not use the AlphaMissense class as a ground-truth label.
- Keep missing scores missing.
- Add `alphamissense_available` as a missingness/support flag.

## Stage 3: Variant Graph

Build a variant graph on the reference 5k benchmark.

Nodes:

- One node per variant.

Edges:

- Same-gene edges, capped per node.
- Same-consequence edges, capped per node.
- kNN similarity edges from biological annotation features:
  `AF`, `CADD_phred`, `REVEL_score`, `SIFT_score`, `Polyphen2_HDIV_score`, `Polyphen2_HVAR_score`, `GERP++_RS`, `phyloP100way_vertebrate`, `phastCons100way_vertebrate`.

Optional later:

- Gene nodes.
- STRING gene-gene edges.
- Gene-level intolerance or disease-gene metadata.

## Stage 4: GNN Residual Model

The GNN can be evaluated in two modes:

- `gnn-only`: the GNN directly predicts pathogenicity from graph context and node features.
- `residual`: the GNN predicts `delta_gnn`, which adjusts the tabular logit.

Residual mode:

```text
z_final = z_tabular + delta_gnn
p_final = sigmoid(z_final)
```

Input node features can include selected numeric annotation features, one-hot consequence, AlphaMissense score/availability, and OOF tabular logits.

## Stage 5: Comparison

Compare:

- Tabular baseline.
- Tabular plus AlphaMissense residual.
- GNN only.
- Tabular plus GNN residual.
- Tabular plus AlphaMissense plus GNN residual, if all inputs exist.

Metrics:

- Accuracy
- Balanced accuracy
- F1
- AUROC
- AUPRC
- MCC
- Brier score
- ECE

## Leakage Controls

- Labels are used only for supervised training and evaluation splits.
- Labels are not used as graph features.
- OOF tabular predictions are used for residual training.
- Future gene-held-out mode must remove test-gene supervision from training.
- Same-gene and same-consequence edges are capped per node to avoid graph explosion.
