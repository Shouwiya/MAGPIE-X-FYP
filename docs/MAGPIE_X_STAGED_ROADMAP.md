# MAGPIE-X Staged Roadmap

Generated: 2026-06-05

Status: planning document only. No training was run for this roadmap.

## Stage 1: MAGPIE-Parity-Current Baseline

Goal: build the strongest low-circularity LightGBM baseline possible with the current feature table.

Inputs:

- ClinVar-derived real feature table.
- ANNOVAR/refGene functional annotations.
- gnomAD30-style allele-frequency columns.
- Non-predictor conservation and variant-structure features.
- Engineered AF, consequence, length, missingness, AA-property, and BLOSUM-like features where derivable.

Excluded:

- Prior pathogenicity predictor scores.
- AlphaMissense.
- ESM.
- GNN residual/fusion.
- Direct gene identity in random split.

Deliverables:

- Feature gap analysis.
- MAGPIE-parity baseline plan.
- LightGBM 5k/10k/20k low-circularity results.
- Gene-held-out validation plan.

Claim boundary:

- Closest possible MAGPIE-style baseline with current data.
- Not exact MAGPIE reproduction.

## Stage 2: MAGPIE-X Enhanced Residual Fusion

Goal: test whether additional evidence improves on the Stage 1 baseline.

Potential enhancements:

- dbNSFP prior predictor scores as explicitly labelled enhanced/circularity-risk context.
- AlphaMissense as an enhancement feature, not parity.
- ESM/protein language model features as enhancement features, not parity.
- GNN residual correction over the tabular logit.
- Residual fusion that reports whether the graph/protein branch fixes errors not already solved by the tabular model.

Required reporting:

- Always separate Stage 1 parity results from Stage 2 enhanced results.
- Do not present predictor-score gains as independent biological learning.
- Use ablation tables to show what each enhancement adds.

## Stage 3: Robustness Validation

Goal: decide whether any MAGPIE-X improvement is credible.

Required checks:

- Gene-held-out split to reduce same-gene leakage.
- Temporal ClinVar validation if dates are available.
- Rare-variant validation with AF < 0.01.
- SwissProt/HUMSAVAR no-exact-overlap external validation after real annotation.
- Calibration audit with saved predictions and verified ECE.
- Federated or held-site evaluation later if available and approved.

## Supervisor-Facing Narrative

The FYP should be framed as:

1. Reconstruct a low-circularity MAGPIE-style baseline as closely as current data allows.
2. Quantify what is missing from exact MAGPIE parity.
3. Add MAGPIE-X enhancements only after the baseline is clear.
4. Validate with stricter splits before making final improvement claims.

This framing avoids overclaiming and directly addresses why random-split accuracy can be high.
