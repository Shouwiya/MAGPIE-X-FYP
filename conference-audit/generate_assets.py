"""Generate manuscript tables and vector figures from aggregate evidence."""
from pathlib import Path
import json
import matplotlib as mpl
mpl.use("Agg")  # Non-interactive backend: works on fresh installs and CI without Tk.
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
E = ROOT / "evidence"
FIG = ROOT / "figures"
TABLE = ROOT / "tables"
TEAL, BLUE, RUST, INK, MUTED = "#007F7A", "#32698B", "#B34F34", "#172B3A", "#677984"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.2, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#96A3AB", "axes.linewidth": .65,
    "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight", pad_inches=.035,
                metadata={"Title": name.replace('_', ' '), "Author": "",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)


def design():
    fig, ax = plt.subplots(figsize=(7.15, 1.88))
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    ax.set(xlim=(0, 7.15), ylim=(.78, 2.65)); ax.axis("off")

    def box(x, y, w, h, title, body, color, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.045",
                                   edgecolor=color, facecolor=fill, lw=.9))
        ax.text(x+.09, y+h-.10, title, fontsize=8, weight="bold", color=color, va="top")
        ax.text(x+.09, y+h-.31, body, fontsize=7.5, color=INK, va="top", linespacing=1.25)
    def arrow(a,b,color=MUTED):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=.9, color=color,
                    shrinkA=2, shrinkB=2))

    box(.08, 1.08, 1.28, 1.06, "SOURCE", "ClinVar GRCh38\n2026-05-03\n490,403 unique\neligible variants", INK, "#F5F7F8")
    box(1.65, 1.80, 2.16, .68, "FEATURE INPUTS", "114 retained annotation fields\n86 prior-predictor fields excluded", TEAL, "#F1F8F6")
    box(1.65, .86, 2.16, .75, "FULL-SOURCE COMPONENTS", "Transitive unions of reported genes\nLinks through unsampled records\nPartition labels, never model inputs", INK, "#F5F7F8")
    arrow((1.36,1.88),(1.65,2.10)); arrow((1.36,1.35),(1.65,1.23))
    ax.text(5.60,2.56,"SAME 20,000 ROWS: 10,000 PER CLASS",ha="center",va="center",fontsize=7.5,weight="bold",color=INK)
    box(4.22,1.80,2.75,.60,"RANDOM FIVE-FOLD OOF","Variants stratified by class",BLUE,"#F3F7FA")
    box(4.22,.86,2.75,.75,"GENE-COMPONENT-BLOCKED OOF","One outer test fold per component\nComponent-disjoint inner tuning",RUST,"#FCF4F0")
    arrow((3.81,2.10),(4.22,2.10),TEAL)
    # Input feature branch to the grouped model; a separate arrow carries groups.
    ax.plot([4.00,4.00],[2.10,1.46],color=TEAL,lw=.9)
    ax.plot([3.81,4.00],[2.10,2.10],color=TEAL,lw=.9)
    arrow((4.00,1.46),(4.22,1.46),TEAL)
    arrow((3.81,1.13),(4.22,1.13),INK)
    save(fig,"experimental_design")


def uncertainty():
    stats=pd.read_csv(E/"split_intervals.csv").set_index("metric")
    seeds=pd.read_csv(E/"paired_seed_metrics.csv")
    assert len(seeds)==10, 'Complete paired-seed experiment before generating final Figure 2'
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.15,2.35),gridspec_kw={"width_ratios":[.94,1.06]})
    fig.subplots_adjust(left=.11,right=.99,bottom=.22,top=.86,wspace=.50)
    for y,(metric,label) in zip([2,1,0],[("accuracy","Accuracy"),("auroc","AUROC"),("auprc","AP")]):
        r=stats.loc[metric]; value=r.left_minus_right*100
        ax.errorbar(value,y,xerr=[[value-r.ci95_low*100],[r.ci95_high*100-value]],fmt='o',color=BLUE,
                    lw=1.3,capsize=3,ms=4)
        ax.text(.59,y,f"{value:.3f}",va="center",ha="left",fontsize=8,color=INK)
    ax.axvline(0,color=MUTED,lw=.8,ls="--")
    ax.set(yticks=[2,1,0],yticklabels=["Accuracy","AUROC","AP"],xlim=(-.06,.69),ylim=(-.6,2.6),
           xlabel="Random minus blocked (percentage points)")
    ax.set_title("(a) Paired partition effect",loc="left",weight="bold",pad=10)
    ax.set_xticks([0,.2,.4,.6]); ax.grid(axis='x',color="#DDE4E8",lw=.5); ax.set_axisbelow(True)
    values=seeds.delta_accuracy*100
    mean,sd=values.mean(),values.std(ddof=1)
    bx.axhspan(mean-sd,mean+sd,color=BLUE,alpha=.10,lw=0)
    bx.axhline(mean,color=BLUE,lw=1.2)
    bx.axhline(0,color=MUTED,lw=.8,ls='--')
    bx.scatter(seeds.seed,values,color=INK,s=20,zorder=3)
    bx.scatter([13],[values[seeds.seed.eq(13)].iloc[0]],facecolors='white',edgecolors=RUST,
               linewidths=1.5,s=46,zorder=4)
    bx.set(xlim=(12.5,22.5),ylim=(min(-.04,values.min()-.05,mean-sd-.04),max(values.max()+.07,mean+sd+.04)),
           xticks=list(range(13,23)),xlabel='Paired OOF protocol seed',ylabel='Accuracy difference (pp)')
    bx.set_title('(b) Paired accuracy across seeds',loc='left',weight='bold',pad=10)
    bx.grid(axis='y',color="#DDE4E8",lw=.5); bx.set_axisbelow(True)
    # Caption defines the mean, SD shading and outlined seed; no legend over data.
    bx.text(22.4,mean+.008,'Mean',ha='right',va='bottom',fontsize=7,color=BLUE)
    save(fig,"split_and_seed_uncertainty")


def evidence():
    abl=pd.read_csv(E/"ablation_metrics.csv").set_index("mode")
    strata=pd.read_csv(E/"balanced_stratum_intervals.csv")
    without=strata[strata.score.eq('drop_consequence_probability')].set_index('consequence')
    strata=strata[strata.score.eq('gene_blocked_probability')]
    labels=[("drop_conservation","Conservation / functional (30)"),
            ("drop_population_frequency","Population frequency (41)"),
            ("drop_amino_acid_blosum","Amino acid / BLOSUM62 (10)"),
            ("drop_functional_consequence","Consequence (19)"),
            ("drop_variant_geometry","Geometry / type (14)")]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.15,2.95),gridspec_kw={"width_ratios":[1,1]})
    fig.subplots_adjust(left=.22,right=.99,bottom=.18,top=.85,wspace=1.08)
    for y,(key,label) in zip(range(4,-1,-1),labels):
        value=-abl.loc[key,"delta_auroc_vs_full"]*100
        ax.plot([0,value],[y,y],color=TEAL,lw=2)
        ax.plot(value,y,'o',color=TEAL,ms=4)
        ax.text(value+.045,y,f"{value:.2f}",va="center",fontsize=7.7,color=INK)
    ax.set(yticks=list(range(4,-1,-1)),yticklabels=[x[1] for x in labels],xlim=(0,1.88),ylim=(-.5,4.6),
           xlabel="AUROC loss (percentage points)")
    ax.tick_params(axis='y',labelsize=7.3,length=0)
    ax.set_title("(a) Remove one feature family",loc="left",weight="bold",pad=10)
    ax.grid(axis='x',color="#DDE4E8",lw=.5); ax.set_axisbelow(True)
    names={"missense_variant":"Missense","frameshift_variant":"Frameshift","nonsense":"Nonsense",
           "splice_donor_variant":"Splice donor","splice_acceptor_variant":"Splice acceptor",
           "inframe_deletion":"In-frame deletion","inframe_insertion":"In-frame insertion",
           "initiator_codon_variant":"Initiator codon"}
    s=strata[strata.n>=80].copy()
    ys=np.arange(len(s))[::-1]
    s=s.sort_values('n',ascending=False)
    bx.axvline(.5,color=MUTED,lw=.8,ls='--')
    for i,(y,(_,r)) in enumerate(zip(ys,s.iterrows())):
        d=without.loc[r.consequence]
        assert r.n==d.n and r.n_positive==d.n_positive and r.n_negative==d.n_negative
        for row,offset,marker,color,label in [(r,.15,'o',BLUE,'Full (114)'),
                (d,-.15,'s',RUST,'No consequence (95)')]:
            bx.errorbar(row.auroc,y+offset,
                xerr=[[max(0,row.auroc-row.auroc_low)],[max(0,row.auroc_high-row.auroc)]],
                fmt=marker,color=color,ms=3,capsize=1.8,lw=.9,label=label if i==0 else None)
    bx.set(yticks=ys,yticklabels=s.consequence.map(names),xlim=(.3,1.015),ylim=(-.65,8.65),xlabel="Within-consequence AUROC")
    bx.legend(loc='upper left',frameon=False,fontsize=6.6,handlelength=1.1,
              borderpad=0,labelspacing=.25,borderaxespad=.15)
    bx.tick_params(axis='y',length=0,labelsize=7.3)
    bx.set_xticks(np.arange(.3,1.01,.1)); bx.grid(axis='x',color="#DDE4E8",lw=.5); bx.set_axisbelow(True)
    bx.set_title("(b) Consequence ablation (95% CI)",loc="left",weight="bold",pad=10)
    save(fig,"evidence_and_consequences")


def reliability():
    a=pd.read_csv(E/"balanced_reliability.csv")
    b=pd.read_csv(E/"prepared_table_reliability.csv").dropna()
    fig,ax=plt.subplots(figsize=(3.5,2.9),layout="constrained")
    ax.plot([0,1],[0,1],ls="--",color=MUTED,lw=.8,label="Ideal")
    ax.plot(a.mean_probability,a.observed_fraction,'o-',color=TEAL,ms=3,label="Balanced sample")
    ax.plot(b.mean_probability,b.observed_fraction,'s-',color=RUST,ms=3,label="Prepared-table sample")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted probability",ylabel="Observed pathogenic fraction")
    ax.set_title("Sampling-procedure sensitivity",loc="left",weight="bold")
    ax.legend(loc="upper left",frameon=False)
    save(fig,"reliability_supplement")


def tables():
    primary=pd.read_csv(E/"primary_metrics.csv")
    seed=pd.read_csv(E/"seed_summary.csv").set_index("metric")
    seedruns=pd.read_csv(E/"seed_metrics.csv")
    # Reproduction reference is taken directly from the retained seed-13 run.
    seed_col="seed" if "seed" in seedruns else "split_seed"
    base=seedruns[seedruns[seed_col]==13].iloc[0]
    metrics=["accuracy","mcc","auroc","auprc","brier","ece"]
    rows=[]
    for label,arm in [("Random stratified, five-fold OOF","random_stratified_5fold"),
                      ("Gene-component-blocked, five-fold OOF","gene_component_blocked_5fold")]:
        r=primary[(primary.evaluation==arm)&(primary.scope=="aggregate_oof")].iloc[0]
        rows.append(label+" & "+" & ".join(f"{r[m]:.3f}" for m in metrics)+r" \\")
    controls=pd.read_csv(E/'control_metrics.csv').set_index('mode')
    for label,key in [('First-gene blocked, nested tuning','first_gene_nested'),
                      ('Component blocked, fixed B','fixed_B_component'),
                      ('Consequence-only logistic, blocked','consequence_only_logit'),
                      ('AF-only ranking, blocked','AF_only_ranking')]:
        r=controls.loc[key]
        rows.append(label+' & '+' & '.join('--' if pd.isna(r[m]) else f'{r[m]:.3f}' for m in metrics)+r' \\')
    rows.append(r"\midrule")
    # Match the established 5-decimal display of the halfway accuracy mean.
    vals=[f"${seed.loc[m,'mean']:.3f}\\pm{seed.loc[m,'sample_sd']:.3f}$" for m in metrics]
    rows.append("Random 80/20, ten-seed mean $\\pm$ SD & "+" & ".join(vals)+r" \\")
    out=r"""\begin{table*}[t]
\caption{OOF evaluations on the same 20,000 rows and separate holdout sensitivity. Original OOF arms use nested tuning; fixed-B random equals the original random arm. AF ranking supplies no threshold/probability metrics. Each holdout tests 4,000 variants. Entries are rounded independently; differences use unrounded values. Full precision is in the companion.}
\label{tab:primary}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Evaluation & Accuracy & MCC & AUROC & AP & Brier & ECE \\
\midrule
"""+"\n".join(rows)+r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (TABLE/"primary_results.tex").write_text(out)
    c=pd.read_csv(E/"comparator_summary.csv")
    a=json.loads((E/"alphamissense_summary.json").read_text())
    c=pd.concat([c,pd.DataFrame([{"comparator":"AlphaMissense","n_exact_shared_variants":a['n'],
          "n_components":a['n_components'],"pathogenic_fraction":a['pathogenic_fraction'],
          "model_auroc":a['model']['auroc'],"comparator_auroc":a['score']['auroc'],
          "model_auprc":a['model']['auprc'],"comparator_auprc":a['score']['auprc']}])],ignore_index=True)
    rows=[]
    for _,r in c.iterrows():
        rows.append(f"{r.comparator} & {int(r.n_exact_shared_variants):,} & {100*r.n_exact_shared_variants/20000:.1f} & {r.pathogenic_fraction:.3f} & {r.model_auroc:.3f} & {r.comparator_auroc:.3f} & {r.model_auprc:.3f} & {r.comparator_auprc:.3f}"+r" \\")
    (TABLE/"comparators.tex").write_text(r"""\begin{table*}[t]
\caption{Exact-row comparisons, not a common leaderboard. Each score and the blocked local model (MX) use identical available rows. Coverage is relative to the 20,000-variant sample; $\pi$ is the subset's pathogenic fraction. Subsets differ between comparators, but not between models within a row.}
\label{tab:comparators}
\centering\footnotesize
\setlength{\tabcolsep}{6.2pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Score & Shared $n$ & Coverage (\%) & $\pi$ & MX AUROC & Score AUROC & MX AP & Score AP \\
\midrule
"""+"\n".join(rows)+r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    c=pd.read_csv(E/"balanced_stratum_intervals.csv")
    c=c[c.score.eq('gene_blocked_probability')].sort_values('n',ascending=False)
    names={"missense_variant":"Missense","frameshift_variant":"Frameshift","nonsense":"Nonsense",
           "splice_donor_variant":"Splice donor","splice_acceptor_variant":"Splice acceptor",
           "inframe_deletion":"In-frame deletion","inframe_insertion":"In-frame insertion",
           "initiator_codon_variant":"Initiator codon"}
    rows=[]
    for _,r in c[c.n>=80].iterrows():
        rows.append(f"{names[r.consequence]} & {int(r.n):,} & {int(r.n_positive):,}/{int(r.n_negative):,} & {r.pathogenic_fraction:.3f} & {r.auprc:.3f} & {r.auroc:.3f} [{r.auroc_low:.3f}, {r.auroc_high:.3f}] & {r.sensitivity:.3f} & {r.specificity:.3f}"+r" \\")
    (TABLE/"consequence_audit.tex").write_text(r"""\begin{table*}[t]
\caption{Consequence-specific OOF results ($n\geq80$). P/B: pathogenic/benign including likely labels; $\pi=P/n$. Brackets: marginal 95\% component-bootstrap AUROC intervals (2,000 replicates within each stratum; fixed predictions, no refitting). Sensitivity/specificity use threshold 0.5. AP must be interpreted alongside $\pi$, particularly for frameshift and nonsense.}
\label{tab:consequence}
\centering\footnotesize
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Consequence & $n$ & P/B & $\pi$ & AP & AUROC [95\% CI] & Sens. & Spec. \\
\midrule
"""+"\n".join(rows)+r"""
\bottomrule
\end{tabular}
\end{table*}
""")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True); TABLE.mkdir(exist_ok=True)
    from threshold_uncertainty import main as threshold_intervals
    threshold_intervals()
    design(); uncertainty(); evidence(); reliability(); tables()
    print("Generated four vector figures (three in manuscript) and three data-driven tables.")
