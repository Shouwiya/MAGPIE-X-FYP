"""Compare a fresh primary refit with published aggregates; never export individual rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    actual = pd.read_csv(args.run_dir / 'gene_blocked_metrics.csv')
    expected = pd.read_csv(ROOT / 'evidence/primary_metrics.csv')
    keys = ['evaluation', 'scope', 'fold']
    expected = expected.set_index(keys).loc[actual.set_index(keys).index]
    actual = actual.set_index(keys)
    metrics = ['accuracy', 'f1', 'precision', 'recall_sensitivity', 'specificity',
               'mcc', 'auroc', 'auprc', 'brier', 'ece']
    np.testing.assert_allclose(actual[metrics], expected[metrics], rtol=0, atol=1e-10)
    tuning = pd.read_csv(args.run_dir / 'nested_tuning.csv')
    chosen = tuning[tuning.selected.astype(str).str.lower().eq('true')]
    manifest = json.loads((ROOT / 'evidence/tuning_manifest.json').read_text())
    for selection in manifest['fold_selections']:
        row = chosen[(chosen.evaluation == selection['evaluation']) &
                     (chosen.outer_fold == selection['outer_fold'])]
        assert len(row) == 1
        observed = json.loads(row.iloc[0].params_json)
        candidate = manifest['candidate_configurations'][str(selection['candidate_id'])]
        assert all(observed[k] == v for k, v in candidate.items())
    print('PASS: all primary fold/pooled metrics agree within 1e-10 and all ten selected configurations match.')
    print('Prepared-input author-side retraining only; not independent data or raw annotation validation.')


if __name__ == '__main__':
    main()
