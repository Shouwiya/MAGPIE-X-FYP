# Primary refit validation - 17 November 2026

Run in the fresh Windows/Python 3.12 environment with scikit-learn 1.7.0 and LightGBM 4.6.0. Used the public repository helpers, not imports from the historical experiment directory. Prepared inputs were read-only; new outputs remain in a private scratch directory.

Scope: --primary-only, 20,000 rows, sample seed 20013, outer seed 13, all six candidates for each outer fold of both arms. Gene components were rebuilt from ClinVar VCF. No saved model or prediction file was used for fitting.

- Base-table SHA256 matches: bc81bec115c9367eea3b02236926d9215cb5a8187dff0b7a49c47266b8c36cc0.
- AA-table SHA256 matches: 4afa75a5e0271088f2612cb43cf0b7588c44631b6f9a3cce4532b19d36416506.
- Reconstructed 14,514 full-source components, 490,403 matched eligible variants, and 54,602 multi-gene records, matching retained counts.
- verify_primary.py PASS: fold/pooled metrics agree within 1e-10 and all ten selected configurations match.
- Additional private comparison: 20,000 rows align, both fold assignments are identical, maximum absolute probability difference across both arms is 0.0.
- pip check: no broken requirements. Additional resolved dependencies: joblib 1.6.0, threadpoolctl 3.8.0, cloudpickle 3.1.2.

This is author-side prepared-input reproducibility, not independent raw-data reconstruction or external-cohort validation. It does not reproduce all secondary experiments or establish absence of leakage.
