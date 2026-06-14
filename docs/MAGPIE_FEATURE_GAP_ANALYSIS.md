# MAGPIE Feature Gap Analysis

Generated: 2026-06-07

Status: updated after completed SpliceAI 5k, 10k, and 20k matching and paired low-circularity evaluation.

Current clean feature table: `data/interim/features/basic_feature_table_with_clean_aa_features.csv`

## MAGPIE Reference Facts

The MAGPIE paper setup to approximate is:

- ClinVar accessed: 2022-06-24.
- Initial filtered ClinVar variants: 78,089.
- Variant types: nonsynonymous SNV/missense, stop-gain, start-loss, frameshift, nonframeshift, stop-loss.
- Labels: benign/likely benign vs pathogenic/likely pathogenic; conflicting/unknown labels removed.
- Added 11,998 gnomAD ultra-rare benign variants with AF between 1e-5 and 1e-3.
- Used a gene-based split to reduce gene-level leakage.
- ClinVarTest: 4,356 pathogenic + 4,310 benign variants.
- ClinVarTest result: AUC 0.995, AUPRC 0.995.
- ClinVarRare: AF < 0.01, 4,881 variants, AUC 0.992, accuracy 0.95.
- SwissProt: 80,840 labelled variants, 71,400 explicitly pathogenic/benign.
- SwissProt filtering: unique rsID mapping, exonic variants, uncertain labels removed, synonymous SNVs removed, genes seen in ClinVarTrain removed.
- Main model: LightGBM.
- Feature categories: epigenomics, functional effects, splicing effects, population features, biochemical properties, conservation.
- Prior pathogenicity predictor scores were excluded to minimize prior bias.

## Current Project State

The project currently has ANNOVAR/refGene, gnomAD30, dbNSFP42a, ClinVar, HUMSAVAR/SwissProt, and AlphaMissense resources available in the broader workspace. The current modeling feature table contains ANNOVAR/refGene-style annotations, population allele frequencies, dbNSFP predictor outputs, conservation fields, consequence flags, and variant length/type features.

The current low-circularity mode is still partial MAGPIE parity because it removes prior predictor-score columns but does not yet reconstruct every MAGPIE feature group. Clean amino-acid and BLOSUM62 features have now been built from `aa_change_refgene` for 219,823 of 490,403 rows with unambiguous canonical substitutions. The enhanced dbNSFP mode uses prior pathogenicity predictor scores and therefore has circularity risk.

The AA/BLOSUM-enhanced LightGBM result is scale-dependent. Accuracy changed from 0.9280 to 0.9210 at 5k, from 0.9300 to 0.9350 at 10k, and from 0.9420 to 0.9467 at 20k. This supports retaining the clean feature group for larger approved checks, but it is not universal improvement evidence.

Official masked hg38 SpliceAI v1.3 files have now been matched to the separate label-free 5k pilot VCF. All 5,000 query IDs are represented in the long-form output, but only 4,582 have at least one real SpliceAI score record; 418 are unmatched placeholders. Per-query aggregation preserves the long-form file and uses score-wise maxima with paired DP positions. On the shared 5k split, AA/BLOSUM plus SpliceAI changed LightGBM accuracy from 0.9260 to 0.9340 and AUROC from 0.9788 to 0.9796, while AUPRC changed from 0.9738 to 0.9718 and ECE from 0.0324 to 0.0383. This mixed pilot result is preliminary.

The exact approved AA/BLOSUM 10k and 20k samples were then matched and evaluated. Real SpliceAI scores were available for 9,161/10,000 and 18,349/20,000 variants. At 10k, accuracy was unchanged at 0.9350, while AUROC changed from 0.9824 to 0.9820 and AUPRC from 0.9816 to 0.9792. At 20k, accuracy changed from 0.9467 to 0.9450, AUROC from 0.9875 to 0.9859, and AUPRC from 0.9865 to 0.9846. The larger paired checks therefore do not confirm a consistent random-split benefit from adding SpliceAI.

## Feature Gap Table

| MAGPIE feature group | Current status | Evidence in current table | Action for closest current parity |
| --- | --- | --- | --- |
| Functional/consequence effects | Present | `func_refgene`, `exonic_func_refgene`, `consequence_*`, `is_snv`, `is_insertion`, `is_deletion` | Keep and expand with consequence/type one-hot and interactions that do not use labels. |
| Population features | Present | `AF`, `AF_raw`, sex-specific and ancestry-specific AF columns from gnomAD-style annotations | Keep; engineer AF bins, rare/common flags, missingness indicators, and AF summary statistics. |
| Conservation | Partially present | `GERP++`, `phyloP`, `phastCons`, `SiPhy`, `integrated_fitCons` fields | Keep raw/non-predictor conservation and summary/missingness features. Treat broad learned pathogenicity-style conservation scores cautiously. |
| Splicing effects | Present in audited 5k/10k/20k sidecars | Official masked hg38 SpliceAI v1.3 exact matching covered 4,582/5,000, 9,161/10,000, and 18,349/20,000 query variants with real scores. Aggregated `DS_*`, paired `DP_*`, availability, source, and record-count fields remain outside the full feature table. | Retain as an audited clean feature group, but do not add it to the default baseline or full 490k table because 10k/20k did not show a consistent benefit. Revisit under gene-held-out/external validation. |
| Epigenomics | Missing | No `ChromHMM`, enhancer, promoter, histone-mark, or tissue regulatory state columns found | Do not fabricate. Add ChromHMM/epigenomic annotations only in a future approved annotation pass. |
| Biochemical amino-acid properties | Present for parseable substitutions | `aa_ref`, `aa_alt`, `aa_substitution_available`, molecular-weight, hydrophobicity, polarity, charge, aromatic, and acid/base change fields are present in the clean AA feature table. | Keep as clean Stage 1 features; preserve missing values for noncanonical, ambiguous, stop, and indel protein changes. |
| BLOSUM-like substitution score | Present for parseable substitutions | `BLOSUM62_score` is present for the same 219,823 unambiguous canonical substitutions. | Keep in Stage 1; do not impute scores for unavailable substitutions. |
| OMIM/inheritance/gene-disease features | Missing | No OMIM or inheritance columns found | Do not download now. List as future parity gap. |
| Protein domain/context | Partially present but risky/sparse | `Interpro_domain` exists but is high-cardinality text and not currently used directly | Use only engineered counts or coarse domain-presence flags if feasible; avoid memorizing high-cardinality strings. |
| Gene identity | Risky/circular for random split | `gene`, `gene_annovar`, `GTEx_V8_gene` exist | Do not add gene identity to random-split parity baseline because same-gene leakage is a known risk. |
| Tissue/expression context | Partially present | `GTEx_V8_tissue` exists | May keep if cardinality is controlled and not label-derived; verify leakage and missingness. |
| Prior pathogenicity predictor scores | Present but circular/risky | SIFT, PolyPhen, REVEL, CADD, MutationTaster, MetaSVM, MetaLR, FATHMM, PROVEAN, M-CAP, AlphaMissense-related if present, ESM if present, and many dbNSFP predictors | Exclude from MAGPIE-parity baseline. Use only in explicitly labelled MAGPIE-X enhanced/circularity-risk mode. |

## Specific Required Checks

| Feature requested | Status | Current evidence |
| --- | --- | --- |
| SpliceAI features | Present for 5k/10k/20k evaluations | Audit, long-form matches, aggregate sidecars, temporary tables, metrics, and reports are in `outputs/real/magpie_parity/`. True score availability is 4,582/5,000, 9,161/10,000, and 18,349/20,000. |
| ChromHMM features | Missing | No column name contains `ChromHMM` or epigenomic state. |
| OMIM/inheritance features | Missing | No OMIM/inheritance columns present. |
| Biochemical amino-acid property deltas | Present/partial coverage | Clean deltas exist for 219,823 rows with unambiguous canonical substitutions. |
| BLOSUM features | Present/partial coverage | Built-in BLOSUM62 scores exist for the same parseable substitutions. |
| Population AF features | Present | gnomAD-style AF columns are present. |
| Conservation features | Present/partial | GERP, phyloP, phastCons, SiPhy, fitCons fields are present. |
| Functional/consequence features | Present | ANNOVAR/refGene and consequence one-hot columns are present. |
| Prior predictor-score features | Present but circular/risky | Many dbNSFP predictor scores and predictions are present and must be excluded from parity. |

## Current Parity Conclusion

The closest current MAGPIE-style baseline should be called `MAGPIE-parity-current` or `partial MAGPIE-parity`, not an exact MAGPIE reproduction. It can use LightGBM with clean biological features and the completed AA/BLOSUM feature group. SpliceAI is available in audited 5k/10k/20k sidecars and has not been merged into the main table. Exact MAGPIE parity remains blocked by full-table SpliceAI coverage, ChromHMM/epigenomic, OMIM/inheritance, the exact 2022-06-24 ClinVar split, and MAGPIE gene-held-out partitions.

SpliceAI improved the 5k pilot accuracy, but final benefit is evaluated at 10k/20k and later under gene-held-out/external validation. The completed 10k comparison was accuracy-neutral and the 20k comparison was slightly worse on accuracy, AUROC, AUPRC, Brier, and ECE. SpliceAI remains a clean MAGPIE-parity splicing feature group, separate from prior pathogenicity predictor scores, but it is not promoted into the default baseline on current evidence.

The enhanced MAGPIE-X mode should be staged after this baseline and must be reported separately because dbNSFP predictor-score columns can inflate random-split performance.
