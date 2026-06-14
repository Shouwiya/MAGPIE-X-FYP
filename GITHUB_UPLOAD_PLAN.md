# GitHub Upload Plan

Generated: 2026-06-14

Status: clean snapshot prepared locally. Nothing is to be pushed without
explicit approval.

## Safety Decision

Do not push `master`.

The existing `master` history contains committed virtual-environment and raw
ClinVar blobs, including files above GitHub's 100 MB limit. `.gitignore` cannot
remove blobs already present in history. The public snapshot must therefore be
an unrelated, parentless branch named `github-clean`.

## Files Included

The clean snapshot includes only:

- `.gitignore`
- `README.md`
- `GITHUB_UPLOAD_PLAN.md`
- `requirements.txt`
- `pyproject.toml`
- `environment.yml`
- `configs/`
- `scripts/`, excluding caches and bytecode
- `src/magpie_x/`, excluding caches and bytecode
- `docs/`
- `paper/main.tex`
- `paper/references.bib`
- `FYP_Literature_Results_Table_Omar_FINAL_CLARIFIED.xlsx`
- the seven requested compact final Markdown reports
- the four requested compact metric/audit CSV tables

The selected snapshot is approximately 1.4 MB and contains about 125 files.
The exact count and byte size are verified from the staged clean branch before
commit.

## Selected Output Artifacts

```text
outputs/real/final_comparison/MAGPIE_PARITY_AND_X_RESULTS_MATRIX.md
outputs/real/federated/FEDERATED_LEARNING_REPORT.md
outputs/real/federated/federated_algorithm_comparison.md
outputs/real/federated/federated_limitations.md
outputs/real/federated/federated_5k_10k_15k_20k_25k_metrics.csv
outputs/real/federated/federated_client_distribution_audit.csv
outputs/real/magpie_x_enhanced/MAGPIE_X_10K_20K_REPORT.md
outputs/real/magpie_x_enhanced/MAGPIE_X_5K_FULL_FUSION_REPORT.md
outputs/real/magpie_x_enhanced/magpie_x_10k_20k_metrics.csv
outputs/real/magpie_x_enhanced/magpie_x_5k_full_fusion_metrics.csv
outputs/real/magpie_parity/SPLICEAI_10K_20K_LOW_CIRCULARITY_REPORT.md
```

## Excluded Content

- `data/raw/`, `data/external/`, and `data/interim/`
- ClinVar, SpliceAI, gnomAD, dbNSFP, AlphaMissense, SwissProt/HUMSAVAR, and
  ANNOVAR raw or derived resources
- `third_party/`
- `.venv/`, `venv/`, environment folders, caches, and bytecode
- VCF, BCF, FASTA, BAM/CRAM, indexes, compressed TSV resources, and avinput
- model binaries and serialized objects: PKL, PT, PTH, JOBLIB, ONNX, CKPT, NPZ
- large feature, prediction, match, and intermediate CSV/TSV files
- screenshots, temporary smoke outputs, logs, and generated previews
- duplicate and older Excel workbooks
- secrets, credentials, API tokens, and machine-local configuration

## Size Gates

- No selected file may exceed 25 MB.
- The total commit should remain under 20 MB.
- All files above 10 MB must be printed before commit.
- Staging must be rejected if a forbidden path or extension appears.

## Local Clean-Branch Workflow

The safe implementation uses a separate orphan worktree so the dirty `master`
worktree remains untouched:

```powershell
git worktree add --orphan -b github-clean D:\FYP_MAGPIE_X_github_clean
```

Only the approved files are copied into that worktree, audited, and committed:

```powershell
Set-Location D:\FYP_MAGPIE_X_github_clean
git add --all
git status --short
git ls-files
git commit -m "Initial clean FYP project snapshot"
```

The resulting commit must have zero parents:

```powershell
git rev-list --parents -n 1 github-clean
```

## Push Gate

No remote is currently configured. After the GitHub repository URL is
provided, configure it from either worktree:

```powershell
git remote add origin https://github.com/<USER>/<REPOSITORY>.git
```

Review with a dry run:

```powershell
git push --dry-run origin github-clean:main
```

Only after explicit approval:

```powershell
git push -u origin github-clean:main
```

Never push `master`, and do not change the GitHub default branch without
explicit approval.
