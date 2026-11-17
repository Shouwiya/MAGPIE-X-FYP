# Release validation (17 November 2026)

Performed in a new Windows Python 3.12 virtual environment, separate from the original research environment:

- Installed `requirements.txt`; `pip check` reports no broken requirements.
- `python -B verify_evidence.py`: PASS before asset regeneration.
- `python -B generate_assets.py`: PASS; regenerates four vector figures, three tables, and nominal Wilson intervals.
- `python -B verify_evidence.py`: PASS after regeneration.
- `python -B check_release.py`: PASS for common-secret/local-path heuristics, forbidden artifacts and row-level CSV columns. This is not a comprehensive security certification.
- Exported the staged files to a ZIP, extracted them into a separate directory, and reran verification and asset generation successfully without the original research folders.
- Tectonic 0.17.0 compiled the anonymous analysis snapshot: six pages, 20 bibliography entries, no unresolved references or overfull boxes. All six pages were rendered and visually inspected. This snapshot remains US Letter, not the final A4 camera-ready version; minor underfull-box warnings remain.

Two portability defects were repaired: SciPy is explicitly pinned, and Matplotlib uses the non-interactive Agg backend rather than requiring Tk.

No model was retrained by the aggregate workflow described above. A subsequent primary refit is documented separately in `training/VALIDATION.md` for v1.1. No new external/temporal cohort was evaluated. Historical records under `evidence/` remain historical records.
