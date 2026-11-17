"""Verify calculations using only aggregate evidence included in the ZIP."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
E = ROOT / "evidence"


def main():
    strata = pd.read_csv(E / "consequence_metrics.csv")
    assert int(strata.n.sum()) == 20000
    assert int(strata.n_positive.sum()) == int(strata.n_negative.sum()) == 10000
    np.testing.assert_array_equal(strata.n, strata.n_positive + strata.n_negative)
    primary = pd.read_csv(E / "primary_metrics.csv")
    grouped = primary[(primary.scope == "aggregate_oof") &
                      (primary.evaluation == "gene_component_blocked_5fold")].iloc[0]
    for k in ["tp", "tn", "fp", "fn"]:
        assert int(strata[k].sum()) == int(grouped[k])
    weights = strata.n_positive * strata.n_negative
    w = weights.sum() / 100000000
    within = np.dot(weights, strata.auroc) / weights.sum()
    across = (grouped.auroc - w * within) / (1 - w)
    saved = json.loads((E / "consequence_pair_decomposition.json").read_text())
    for key, value in {"within_pair_fraction": w,
                       "within_consequence_pair_weighted_auroc": within,
                       "across_consequence_pair_auroc": across,
                       "pooled_auroc": grouped.auroc}.items():
        np.testing.assert_allclose(value, saved[key], rtol=0, atol=1e-12)
    seed = pd.read_csv(E / "seed_metrics.csv")
    summary = pd.read_csv(E / "seed_summary.csv").set_index("metric")
    for metric, row in summary.iterrows():
        np.testing.assert_allclose(seed[metric].mean(), row["mean"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(seed[metric].std(ddof=1), row["sample_sd"], rtol=0, atol=1e-12)
    selections = json.loads((E / "tuning_manifest.json").read_text())
    assert len(selections["candidate_configurations"]) == 6
    random = [r["candidate_id"] for r in selections["fold_selections"] if r["evaluation"].startswith("random")]
    gene = [r["candidate_id"] for r in selections["fold_selections"] if r["evaluation"].startswith("gene")]
    assert random == [5, 5, 5, 5, 5] and gene == [5, 6, 1, 6, 5]
    families = json.loads((E / "feature_family_membership.json").read_text())
    assert {k: len(v) for k, v in families.items()} == {"population_frequency": 41,
        "conservation": 30, "functional_consequence": 19, "variant_geometry": 14, "amino_acid_blosum": 10}
    fields = [f for family in families.values() for f in family]
    manifest = json.loads((E / "feature_manifest.json").read_text())
    assert len(fields) == len(set(fields)) == 114 and set(fields) == set(manifest["features"])
    random_oof = primary[(primary.scope == "aggregate_oof") &
                        (primary.evaluation == "random_stratified_5fold")].iloc[0]
    intervals = pd.read_csv(E / "split_intervals.csv").set_index("metric")
    for metric, row in intervals.iterrows():
        np.testing.assert_allclose(random_oof[metric]-grouped[metric], row.left_minus_right, atol=1e-12, rtol=0)
    abl = pd.read_csv(E / "ablation_metrics.csv").set_index("mode")
    for mode, row in abl.drop(index="full_114").iterrows():
        assert row.removed_features + row.remaining_features == 114
        for metric in ["accuracy", "mcc", "auroc", "auprc"]:
            if "delta_"+metric+"_vs_full" in row:
                np.testing.assert_allclose(row[metric]-abl.loc["full_114", metric], row["delta_"+metric+"_vs_full"], atol=1e-12, rtol=0)
    comp = pd.read_csv(E / "comparator_summary.csv").set_index("comparator")
    comp_int = pd.read_csv(E / "comparator_intervals.csv")
    for _, row in comp_int.iterrows():
        c = comp.loc[row.comparator]
        np.testing.assert_allclose(c["model_"+row.metric]-c["comparator_"+row.metric], row.model_minus_comparator, atol=1e-12, rtol=0)
    alpha = json.loads((E / "alphamissense_summary.json").read_text())
    for _, row in pd.read_csv(E / "alphamissense_intervals.csv").iterrows():
        np.testing.assert_allclose(alpha["model"][row.metric]-alpha["score"][row.metric], row.left_minus_right, atol=1e-12, rtol=0)
    frameshift = strata.set_index("consequence").loc["frameshift_variant"]
    assert frameshift.n_negative == frameshift.fp == 68 and frameshift.tn == frameshift.specificity == 0
    assert len(seed) == 10 and seed.loc[seed.accuracy.idxmax(), "seed"] == 13
    np.testing.assert_allclose(seed.accuracy.mean(), .940975, atol=1e-12, rtol=0)
    text = (ROOT / "main.tex").read_text()+''.join(p.read_text() for p in (ROOT/'sections').glob('*.tex'))
    for number in ["490,403", "114", "86", "20,000", "10,000", "0.984", "0.879", "0.084", "0.027", "0.141", "0.94097", "0.00441", "0.94675", "80.11", "0.952", "68", "0.67602", "0.67715"]:
        assert number in text, f"Missing manuscript anchor: {number}"
    from threshold_uncertainty import wilson
    ci = pd.read_csv(E/'consequence_threshold_intervals.csv')
    for _,r in ci.iterrows():
        for prefix,k,n in [('sens',r.tp,r.n_positive),('spec',r.tn,r.n_negative)]:
            np.testing.assert_allclose(wilson(k,n), [r[prefix+'_low'],r[prefix+'_high']],atol=1e-12,rtol=0)
    assert abs(wilson(0,68)[1]-.0534713365200705)<1e-12
    assert 'fixed-prediction intervals' in text
    assert f'{331524/490403:.5f}' in text
    for required in ['0.330 pp','0.441 pp','strictly increasing recalibration','no external funding','no conflict of interest']:
        assert required in text,required
    # New source-aligned consequence analysis and retained constructor evidence.
    new=pd.read_csv(E/'balanced_stratum_intervals.csv')
    full=new[new.score.eq('gene_blocked_probability')].set_index('consequence')
    assert len(full)==10 and full.n.sum()==20000
    old=strata.set_index('consequence').loc[full.index]
    dropped=new[new.score.eq('drop_consequence_probability')].set_index('consequence').loc[full.index]
    assert len(dropped)==10 and dropped.n.sum()==20000
    for key in ['n','n_positive','n_negative']:
        np.testing.assert_array_equal(full[key],dropped[key])
    for key in ['auroc','auprc','sensitivity','specificity']:
        np.testing.assert_allclose(dropped[key]-full[key],dropped['delta_'+key],atol=1e-12,rtol=0)
    assert dropped.loc['frameshift_variant','specificity']==0
    for oldkey,newkey in [('auroc','auroc'),('auprc','auprc'),('recall_sensitivity','sensitivity'),('specificity','specificity')]:
        np.testing.assert_allclose(old[oldkey],full[newkey],atol=1e-12,rtol=0)
        assert (full[newkey+'_low']<=full[newkey+'_high']).all()
    secondary=pd.read_csv(E/'unstratified_stratum_intervals.csv')
    assert secondary.n.sum()==20000 and secondary.n_positive.sum()==13543
    assert (new.valid_bootstrap_replicates==2000).all()
    assert json.loads((E/'stratum_interval_audit.json').read_text())['paired_ablation']
    provenance=json.loads((E/'recovered_model_provenance.json').read_text())
    assert provenance['subsample_freq']==0 and provenance['class_weight']=='balanced'
    assert provenance['full_oof_max_probability_delta']<1.2e-16
    assert len(provenance['numeric_fields'])+len(provenance['categorical_fields'])==114
    paired=pd.read_csv(E/'paired_seed_metrics.csv')
    assert paired.seed.tolist()==list(range(13,23))
    ps=pd.read_csv(E/'paired_seed_summary.csv').set_index('metric')
    from scipy.stats import binomtest
    for metric in ['accuracy','auroc','auprc','mcc']:
        v=paired['delta_'+metric]
        np.testing.assert_allclose(v,paired['random_'+metric]-paired['blocked_'+metric],atol=1e-12,rtol=0)
        np.testing.assert_allclose([v.mean(),v.std(ddof=1)],[ps.loc[metric,'mean_delta'],ps.loc[metric,'sample_sd']],atol=1e-12,rtol=0)
        assert ps.loc[metric,'positive']==(v>0).sum() and ps.loc[metric,'negative']==(v<0).sum()
        expected=binomtest(int((v>0).sum()),int((v!=0).sum()),.5).pvalue
        assert abs(expected-ps.loc[metric,'two_sided_sign_p'])<1e-12
    control=pd.read_csv(E/'control_metrics.csv').set_index('mode')
    assert abs(control.loc['fixed_B_random','auroc']-random_oof.auroc)<1e-12
    assert '0.892' in text and '0.557' in text
    snapshots=json.loads((E/'annotation_snapshot.json').read_text())
    assert len(snapshots['files'])==6 and all(len(f['sha256'])==64 for f in snapshots['files'])
    coverage=pd.read_csv(E/'allele_type_coverage.csv').set_index('allele_type')
    assert coverage.n.sum()==20000 and coverage.loc['length_changing','n']==4719
    assert (coverage.loc['length_changing',['CADD_available','REVEL_available','AlphaMissense_available']]==0).all()
    af=json.loads((E/'coverage_audit.json').read_text())
    assert af['AF_missing']==11838 and abs(af['AF_missing_fraction']-.5919)<1e-12
    for required in ['0.186','0.102','0.066','0.049','0.058','0.090','59.19','4,719']:
        assert required in text,required
    config = (ROOT / "tables/configurations.tex").read_text()
    for line in ["500 & 700 & 1,000", ".05 & .04 & .025", "31 / 20 & 31 / 30 & 63 / 40", "not set & 6 & 8", ".90 / .90 & .95 / .95 & .90 / .80", ".00 / .00 & .00 / 1.00 & .05 / 1.50"]:
        assert line in config
    print("PASS: counts, decomposition, seeds, configurations, family coverage, paired/comparator/ablation differences, and manuscript anchors.")
    print("This checks aggregate arithmetic, not raw-data provenance or model retraining.")


if __name__ == "__main__":
    main()
