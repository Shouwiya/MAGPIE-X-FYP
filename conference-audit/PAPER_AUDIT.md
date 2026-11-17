# Reviewer response and remaining paper checks

## Reproducibility

Addressed in part: release aggregate evidence, exact recorded configurations/field manifests, verification and figure-generation scripts, pinned dependencies and explicit execution instructions. SciPy is now an explicit dependency because the verifier imports `scipy.stats.binomtest`.

Not addressed in full: independent end-to-end model retraining. This package must not be described as fully reproducing raw annotation, folds, model fitting, or bootstrap intervals from individual predictions. See DATA_AND_RETRAINING.md.

## External validation

Not experimentally addressed by this release. The manuscript correctly states that no independent, ancestry-stratified or prospective validation was performed. Do not repurpose the older FYP SwissProt result as evidence for this audit without checking model, input, overlap and protocol compatibility.

## Repository link

The earlier 20-reference manuscript had no GitHub URL and said that no public companion was deposited. The companion manuscript now links to the conference-audit-v1 version tag and explicitly limits the reproducibility claim. No DOI is claimed.

## Camera-ready items (not completed by this code release)

- Resolve final authorship metadata with all authors and organizers before submission; do not silently add/remove/reorder authors.
- Prepare A4 IEEE camera-ready formatting and required copyright notice; the retained anonymous manuscript source is not that finalized version.
- Disclose actual AI assistance accurately and obtain the authors' final review and approval.
- Check final six-page/20-reference layout, citations, similarity requirements and reference quality. Aggregate number-anchor tests are not a complete scientific or bibliographic peer review.
- Run the approved final PDF through IEEE PDF eXpress. This release is not a PDF eXpress check or a conference submission.
- Complete copyright and registration through the authorized author. No such action is performed here.
