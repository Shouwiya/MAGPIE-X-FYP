# Data and full-retraining boundary

## Recorded inputs

The retained lineage identifies ClinVar GRCh38 release **2026-05-03**, with 490,403 eligible unique variants. The audit uses a balanced 20,000-variant sample (sample seed 20013); original outer split seed 13; repeated paired seeds 13-22. See `source_lineage.json`, `control_protocol.json`, and `tuning_manifest.json` under `evidence/` for recorded counts and settings.

Annotation records identify ANNOVAR refGene, gnomAD30_genome and dbnsfp42a on hg38. `annotation_snapshot.json` records hashes of the currently recovered installation, not contemporaneous proof of unchanged historical binaries. The dbNSFP catalogue date is 2021-07-10. The exact AlphaMissense release is unverified. Do not substitute a newer release and call the result an exact replication.

Upstream acquisition starts with the official ClinVar archive (https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/), ANNOVAR documentation (https://annovar.openbioinformatics.org/), gnomAD (https://gnomad.broadinstitute.org/downloads), dbNSFP (https://www.dbnsfp.org/), and AlphaMissense (https://github.com/google-deepmind/alphamissense). These are acquisition entry points, not a verified automated download recipe. Check availability and applicable terms yourself. This release does not redistribute their databases.

## What an exact full rerun still requires

1. Recover and legally obtain the recorded source snapshots and the original preprocessing/annotation/transcript policy.
2. Rebuild the 114-field table. `feature_manifest.json` specifies field names, not a complete raw-annotation construction recipe. `recovered_model_provenance.json` splits numeric and categorical fields and records input hashes.
3. Reconstruct the gene-component map by unioning all co-reported raw ClinVar gene symbols before sampling, plus the eligible-row selection, sampling and fold construction. Individual assignments are not distributed.
4. Use `training/run_gene_blocked.py --primary-only` for the portable primary refit. It imports helpers from the full repository and reconstructs components from the VCF without historical prediction checkpoints for fitting. The newer consequence/control and ten paired-seed experiments still need additional porting; the root FYP entry points alone do not recreate the full audit.
5. Run preprocessing within training folds, the specified nested selection, paired splits/ablations, and conditional component bootstrap; compare exported aggregates with the retained evidence using appropriate tolerances.

The recorded training environment includes LightGBM 4.6.0, scikit-learn 1.7.0, NumPy 2.3.0 and pandas 2.3.0. The aggregate-only requirements intentionally omit LightGBM because no model is trained here. Recorded settings include balanced class weights and `subsample_freq=0`; row-subsampling fractions therefore do not activate row bagging.

The delta in `recovered_model_provenance.json` is historical, not a rerun by `verify_evidence.py`. A separate primary refit completed on 17 November 2026 (see `training/VALIDATION.md`). Hashes cannot reconstruct absent inputs. Complete raw-annotation-to-all-experiments replication remains unresolved.

## External/temporal validation

No new external validation is claimed. An appropriate follow-up would freeze preprocessing/model choices, audit variant and gene overlap with an independent cohort, then report class counts and stratum-specific performance/uncertainty. A temporal analysis needs verified first-availability dates and an explicit annotation-time policy; last-evaluated dates or file timestamps are not a substitute. This is a proposed protocol, not an experimental result.
