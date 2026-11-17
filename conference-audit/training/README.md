# Prepared-input primary retraining

This entry point removes the old dependency on a sibling conference-paper-v2 folder. It imports the model/preprocessing helpers already present under this repository's `scripts/` directory. Use the **whole repository checkout**, not this folder alone.

It reconstructs gene components from the source ClinVar VCF, samples the balanced 20k set, and refits both five-fold nested OOF arms, including all six inner candidates per fold. It does not load the historical fitted models or prediction checkpoints to train. This is not a raw-annotation reconstruction or independent-cohort validation.

## Inputs (not redistributed)

Under an explicitly supplied data root:

- `data/interim/features/basic_feature_table.csv`
- `data/interim/features/basic_feature_table_with_clean_aa_features.csv`
- `data/raw/clinvar.vcf.gz`

Use the recorded releases/schema/hashes in the companion. The script checks the 114-field count but this alone does not guarantee matching inputs. Legally obtain and prepare these inputs as described in `../DATA_AND_RETRAINING.md`.

From `conference-audit/training`, in a fresh Python 3.12 environment:

```sh
python -m pip install -r requirements.txt
python -B run_gene_blocked.py --data-root YOUR_DATA_ROOT --output-dir YOUR_PRIVATE_OUTPUT_DIR --primary-only
python -B verify_primary.py --run-dir YOUR_PRIVATE_OUTPUT_DIR
```

Replace the two uppercase placeholders with local paths. Never point the output directory at a public release folder. Outputs include individual variants, labels, component assignments and predictions: keep them private. Source inputs are read-only. The run uses all CPU cores and loads the prepared tables into RAM; runtime depends on hardware.

The historical extended path (omitting `--primary-only`) additionally requires an AlphaMissense sidecar and runs ablations/secondary analyses. It is retained for inspection but is **not the validated entry point for this release**. The ten repeated-seed and newer consequence/control experiments are not all reproduced by the primary command. Do not describe it as complete paper replication.
