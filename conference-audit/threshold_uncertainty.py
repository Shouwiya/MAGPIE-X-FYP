"""Count-based, nominal Wilson intervals; NOT score-level AUROC uncertainty.

Only public-safe aggregate counts are read or exported. Independent-row
intervals omit gene dependence, CV training dependence and retraining variance.
"""
from pathlib import Path
import hashlib
import json
from math import sqrt
from statistics import NormalDist
import pandas as pd

ROOT = Path(__file__).resolve().parent


def wilson(success, n):
    assert 0 <= success <= n and n > 0
    z = NormalDist().inv_cdf(.975)
    p = success / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0., center-half), min(1., center+half)


def main():
    source = ROOT/'evidence/consequence_metrics.csv'
    rows = pd.read_csv(source)
    for i,r in rows.iterrows():
        assert r.n == r.n_positive+r.n_negative
        assert r.n_positive == r.tp+r.fn and r.n_negative == r.tn+r.fp
        for prefix,success,n in [('sens',r.tp,r.n_positive),('spec',r.tn,r.n_negative)]:
            lo,hi = wilson(success,n)
            rows.loc[i,prefix+'_low'] = lo
            rows.loc[i,prefix+'_high'] = hi
    rows.to_csv(ROOT/'evidence/consequence_threshold_intervals.csv',index=False)
    result = {'method':'two-sided Wilson score, nominal 95%, independent Bernoulli rows',
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'strata':len(rows),'new_training':False,'auroc_intervals_computed':False,
              'limitations':['No gene-component dependence adjustment','No refitting uncertainty',
                            'Not a replacement for stratum AUROC/bootstrap intervals'],
              'eligible_pathogenic_fraction':331524/490403,
              'unstratified_sample_pathogenic_fraction':13543/20000}
    (ROOT/'evidence/threshold_interval_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: nominal threshold intervals for ten strata; no AUROC intervals claimed.')


if __name__ == '__main__':
    main()
