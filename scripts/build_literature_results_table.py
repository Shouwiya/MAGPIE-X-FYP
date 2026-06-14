from __future__ import annotations

import html
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "FYP_Literature_Results_Table_Omar.xlsx"

METRIC_TUPLE = "(Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, MCC, AUROC, AUPRC, Brier, ECE)"
MISSING_TUPLE = "(-, -, -, -, -, -, -, -, -, -)"

HEADERS = [
    "Paper No.",
    "Title",
    "Year",
    "Proposed Approach / Model",
    "Description",
    "Dataset Used",
    "Sample Size",
    "Variant Type",
    "ClinVar Result Tuple",
    "ClinVarRare Result Tuple",
    "SwissProt/HUMSAVAR Result Tuple",
    "ACMG Result Tuple",
    "ProteinGym/MAVE Result Tuple",
    "External Validation",
    "Source Link",
    "Notes / Limitations",
]


ROWS = [
    [
        1,
        "MAGPIE: accurate pathogenic prediction for multiple variant types using machine learning approach",
        2024,
        "LightGBM with multimodal annotations",
        "Multi-type exonic pathogenicity predictor using population, functional, conservation, biochemical, splicing, and epigenetic annotations.",
        "ClinVarTrain/ClinVarTest; ClinVarRare; SwissProt orthogonal set; ACMG-guided set",
        "ClinVarTest: 4,356 pathogenic + 4,310 benign; ClinVarRare: 4,881; SwissProt: 1,308 pathogenic + 12,075 benign",
        "Missense, stop-gain, start-loss, frameshift, nonframeshift, stop-loss",
        MISSING_TUPLE,
        "(0.950, -, -, -, -, -, 0.992, -, -, -)",
        "(-, -, -, -, -, -, 0.970, 0.880, -, -)",
        MISSING_TUPLE,
        MISSING_TUPLE,
        "SwissProt and ACMG-guided orthogonal datasets",
        "https://link.springer.com/article/10.1186/s13073-023-01274-4",
        "ClinVarTest exact tuple requires supplementary-table extraction; rare and orthogonal AUROC/AUPRC values are from article text.",
    ],
    [
        2,
        "A graph neural network approach for accurate prediction of pathogenicity in multi-type variants",
        2025,
        "GNN-MAP graph attention network",
        "Constructs variant similarity graphs from multimodal annotations and predicts pathogenicity with GNN layers.",
        "ClinVar CVTesting; VarRareTest; SwissProt-Test",
        "CVTesting: 5,312 pathogenic + 4,023 benign; VarRareTest: 1,960 pathogenic + 2,870 benign",
        "Multi-type exonic variants",
        "(-, -, -, -, -, -, 0.992, 0.993, -, -)",
        "(-, -, -, -, -, -, 0.988, 0.978, -, -)",
        "(-, -, -, -, -, -, 0.993, 0.961, -, -)",
        MISSING_TUPLE,
        MISSING_TUPLE,
        "SwissProt-Test",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC12008122/",
        "Useful graph comparator for MAGPIE-X; still needs careful leakage review because graph construction can connect related variants.",
    ],
    [
        3,
        "VEGN: Variant Effect Prediction with Graph Neural Networks",
        2021,
        "Variant-gene heterogeneous GNN",
        "Connects variants to genes and propagates information through given or learned gene-gene edges.",
        "Variant-gene graph benchmarks described in paper",
        "Not extracted",
        "Variant effect prediction",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Graph-based comparison against existing state-of-the-art models",
        "https://arxiv.org/abs/2106.13642",
        "Graph inspiration; exact metric tuple should be extracted from the PDF before formal comparison.",
    ],
    [
        4,
        "Accurate proteome-wide missense variant effect prediction with AlphaMissense",
        2023,
        "AlphaMissense",
        "DeepMind missense predictor combining protein structure and evolutionary information at proteome scale.",
        "ClinVar and functional assay benchmarks; proteome-wide missense catalog",
        "All possible human missense variants; exact benchmark counts require paper extraction",
        "Missense SNVs only",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Functional assay and clinical benchmark evaluation",
        "https://pubmed.ncbi.nlm.nih.gov/37733863/",
        "Use only where amino-acid substitutions are defined; do not apply to frameshift, nonsense, splice, or indel variants as a universal score.",
    ],
    [
        5,
        "Genome-wide prediction of disease variant effects with a deep protein language model",
        2023,
        "ESM1b protein language model scoring",
        "Scores all possible missense variants using log-likelihood ratio from a protein language model.",
        "ClinVar; HGMD/gnomAD; 28 deep mutational scan datasets",
        "ClinVar benchmark: 36,537 variants in 2,765 genes; HGMD/gnomAD: 30,497 variants in 1,991 genes",
        "Missense variants",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Deep mutational scan datasets",
        "https://www.nature.com/articles/s41588-023-01465-0",
        "Article reports ROC-AUC and balanced PRC-AUC figures; exact values need figure/table extraction.",
    ],
    [
        6,
        "REVEL: an ensemble method for predicting the pathogenicity of rare missense variants",
        2016,
        "REVEL ensemble learner",
        "Combines multiple missense prediction tools and conservation features for rare missense pathogenicity.",
        "HGMD; ESP; SwissVar; ClinVar",
        "ClinVar test set: 1,953 pathogenic + 2,406 benign",
        "Rare missense variants",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "SwissVar disease variants and neutral exome variants",
        "https://www.sciencedirect.com/science/article/pii/S0002929716303706",
        "Important baseline and also a prior predictor column; include in enhanced mode but remove for parity/circularity ablation.",
    ],
    [
        7,
        "A general framework for estimating the relative pathogenicity of human genetic variants",
        2014,
        "CADD",
        "Contrasts fixed or nearly fixed human variants with simulated variants to score deleteriousness genome-wide.",
        "Simulated variants; human-derived alleles; ClinVar/ExAC comparisons",
        "Not extracted",
        "Coding and noncoding SNVs/indels",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "ClinVar pathogenic vs frequent ExAC-style variants in later benchmark descriptions",
        "https://www.nature.com/articles/ng.2892",
        "CADD is a powerful prior score; high performance with CADD-like fields can raise circularity concerns.",
    ],
    [
        8,
        "ClinPred: prediction tool to identify disease-relevant nonsynonymous single-nucleotide variants",
        2018,
        "ClinPred gradient boosting predictor",
        "Supervised nsSNV pathogenicity predictor trained from ClinVar-derived labels and evaluated on clinical exomes.",
        "ClinVar January 2016; rare disease exomes",
        "Clinical exome validation included 31 exome cases and 25 known disease variants",
        "Nonsynonymous SNVs",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "FORGE Canada and Care4Rare Canada exomes",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC6174354/",
        "ClinVar-derived model; remove ClinPred score columns in parity/circularity ablation.",
    ],
    [
        9,
        "Improved pathogenicity prediction for rare human missense variants",
        2021,
        "VARITY",
        "Missense predictor designed for rare variants, with balanced precision-recall style evaluation.",
        "ClinVar rare variants; functional assay comparisons",
        "Not extracted",
        "Rare missense variants",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Deep mutational scanning and rare ClinVar comparisons",
        "https://www.sciencedirect.com/science/article/pii/S0002929721003207",
        "Paper reports AUBPRC and AUROC comparisons; exact tuples need supplementary-table extraction.",
    ],
    [
        10,
        "Disease variant prediction with deep generative models of evolutionary data",
        2021,
        "EVE",
        "Unsupervised/evolutionary generative model for protein variant effects using multiple sequence alignments.",
        "ClinVar annotations; population frequencies; EVE portal data",
        "Not extracted",
        "Missense variants in disease-related genes",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Functional and clinical validation through EVE resources",
        "https://www.nature.com/articles/s41586-021-04043-8",
        "Coverage depends on MSA availability; useful as missense-only protein/evolution comparator.",
    ],
    [
        11,
        "Inferring the molecular and phenotypic impact of amino acid variants with MutPred2",
        2020,
        "MutPred2",
        "Predicts pathogenicity and molecular mechanism hypotheses for amino-acid substitutions.",
        "HGMD; SwissVar; dbSNP; ClinVar/SwissVar independent test",
        "Training: 53,180 pathogenic + 206,946 unlabeled variants",
        "Missense variants",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Independent ClinVar and SwissVar accumulation test",
        "https://www.nature.com/articles/s41467-020-19669-x",
        "Prior predictor score; exact performance tuple should be extracted before final table submission.",
    ],
    [
        12,
        "M-CAP eliminates a majority of variants of uncertain significance in clinical exomes at high sensitivity",
        2016,
        "M-CAP",
        "Clinical missense pathogenicity classifier optimized for high sensitivity and VUS reduction.",
        "Clinical exomes and rare missense benchmark data",
        "Not extracted",
        "Rare missense variants",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Reported 95% sensitivity while dismissing 60% of rare missense VUS",
        "https://www.nature.com/articles/ng.3703",
        "High-sensitivity operating point is not a complete metric tuple; keep as external note unless confusion-matrix metrics are extracted.",
    ],
    [
        13,
        "Predicting the clinical impact of human mutation with deep neural networks",
        2018,
        "PrimateAI",
        "Deep residual neural network using primate and vertebrate sequence context to infer missense pathogenicity.",
        "ClinVar; de novo mutation and disease-gene benchmarks",
        "Not extracted",
        "Missense variants",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Developmental disorder and ClinVar-related benchmarks",
        "https://pubmed.ncbi.nlm.nih.gov/30038395/",
        "Deep learning baseline; prior score columns should be removed in parity mode.",
    ],
    [
        14,
        "The landscape of tolerated genetic variation in humans and primates",
        2023,
        "PrimateAI-3D",
        "3D convolutional model trained with common primate variation and protein structures to classify protein-altering variants.",
        "Primate variation catalog; ClinVar; DMS; patient cohorts",
        "Sequenced 703 individuals from 231 primate species; aggregated 809 individuals from 233 species",
        "Protein-altering variants, especially missense",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "ClinVar, DMS, ASD/NDD/CHD cohort benchmarks",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC10187174/",
        "Important newer protein-structure/evolution comparator; exact tuple needs benchmark table extraction.",
    ],
    [
        15,
        "FATHMM-XF: accurate prediction of pathogenic point mutations via extended features",
        2018,
        "FATHMM-XF",
        "Genome-wide pathogenic point mutation predictor using extended feature sets.",
        "ClinVar unseen coding and noncoding examples",
        "Not extracted",
        "Coding and noncoding point mutations",
        "(0.890, -, -, -, -, -, 0.970, -, -, -)",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "ClinVar unseen examples",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC5860356/",
        "The 0.89 accuracy and 0.97 AUC are reported for unseen ClinVar examples in source text/snippets; verify against Table 1 before final thesis submission.",
    ],
    [
        16,
        "MetaRNN: differentiating rare pathogenic and rare benign missense SNVs and InDels using deep learning",
        2022,
        "MetaRNN and MetaRNN-indel",
        "Deep learning model for rare nsSNVs and nonframeshift indels using independent test sets from ClinVar/HGMD and other resources.",
        "ClinVar; HGMD; TP53 functional mutation dataset; cancer driver variants",
        "RNTS rare test set exact count not extracted",
        "Rare missense SNVs and nonframeshift indels",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "TP53 functional and cancer driver variant sets",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC9548151/",
        "Useful because it covers rare missense SNVs and nonframeshift indels; exact tuple needs paper extraction.",
    ],
    [
        17,
        "MVP predicts the pathogenicity of missense variants by deep learning",
        2021,
        "MVP deep residual network",
        "Deep learning missense pathogenicity predictor using many correlated annotations and separate models by gene constraint context.",
        "HGMD; UniProt; ClinVar; cancer hotspots; de novo developmental disorder variants",
        "Training positives included 22,390 HGMD + 12,875 UniProt + 4,424 ClinVar variants",
        "Missense variants",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Cancer hotspot and de novo variant benchmarks",
        "https://www.nature.com/articles/s41467-020-20847-0",
        "MVP/gMVP scores are prior predictors; handle carefully in circularity ablation.",
    ],
    [
        18,
        "Predicting functional effect of missense variants using graph attention neural networks",
        2022,
        "gMVP graph attention neural network",
        "Represents protein context as a graph with amino-acid node features and co-evolution edge features.",
        "ClinVar; HGMD; UniProt; DMS; de novo variant benchmarks",
        "Training after filtering: 59,701 positives + 59,701 negatives",
        "Missense variants",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "DMS, autism/NDD/CHD de novo benchmarks",
        "https://www.nature.com/articles/s42256-022-00561-w",
        "Graph/protein-context comparator; exact clinical metric tuple needs paper table extraction.",
    ],
    [
        19,
        "CAPICE: a computational method for Consequence-Agnostic Pathogenicity Interpretation of Clinical Exome variations",
        2020,
        "CAPICE XGBoost",
        "Consequence-agnostic pathogenicity predictor for clinical exome variants, including rare and ultra-rare variants.",
        "ClinVar high-confidence labels; clinical exome variation",
        "Not extracted",
        "SNVs and indels across consequence types",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Rare and ultra-rare variant benchmarks",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7446154/",
        "Good comparator for consequence-agnostic modeling; exact tuple needs supplementary extraction.",
    ],
    [
        20,
        "LIST-S2: taxonomy-aware deleteriousness prediction for amino acid mutations",
        2020,
        "LIST-S2",
        "Species-specific conservation method for deleteriousness prediction of protein amino-acid mutations.",
        "Protein sequence conservation resources; benchmarked in MAGPIE/GNN-MAP comparisons",
        "Not extracted",
        "Missense/amino-acid substitutions",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Precomputed proteome-level scores",
        "https://precomputed.list-s2.msl.ubc.ca/about",
        "Usually a prior predictor feature rather than an independent validation dataset.",
    ],
    [
        21,
        "DYNA: Disease-Specific Language Model for Variant Pathogenicity",
        2024,
        "Disease-specific PLM fine-tuning",
        "Fine-tunes genomic/protein foundation model evidence for disease-specific variant pathogenicity.",
        "Held-out rare variant testing; ClinVar replication",
        "Not extracted",
        "Disease-specific missense/pathogenicity context",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "ClinVar replication described by authors",
        "https://arxiv.org/abs/2406.00164",
        "Future-work comparator unless the FYP selects a disease-specific evaluation domain.",
    ],
    [
        22,
        "ProPath: Disease-Specific Protein Language Model for Variant Pathogenicity",
        2023,
        "Disease-specific protein language model",
        "Uses disease-specific fine-tuning to improve pathogenicity prediction relative to generic protein language model scores.",
        "Disease-specific benchmark datasets",
        "Not extracted",
        "Disease-specific missense variants",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Disease-specific held-out benchmarks",
        "https://arxiv.org/abs/2311.03429",
        "Relevant for future extension; not part of current general MAGPIE-X baseline.",
    ],
    [
        23,
        "Predicting the pathogenicity of missense variants using features derived from AlphaFold2",
        2023,
        "AlphScore / AlphaFold2 structural features",
        "Uses AlphaFold2-derived structural features and combines them with established prediction scores.",
        "ClinVar_test; DMS datasets",
        "ClinVar_test: 9,224 benign + 11,844 pathogenic missense variants",
        "Missense variants",
        "(-, -, -, -, -, -, 0.799, -, -, -)",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "(-, -, -, -, -, -, -, -, -, -)",
        "Deep mutational scan datasets",
        "https://academic.oup.com/bioinformatics/article/39/5/btad280/7135835",
        "AlphScore alone AUROC was 0.799 on the cited ClinVar holdout; same source reports CADD 0.885 and REVEL 0.929 on that holdout.",
    ],
    [
        24,
        "Disease-specific variant pathogenicity prediction using multimodal biomedical language models",
        2025,
        "DIVA multimodal biomedical language model",
        "Predicts disease type and deleteriousness for missense variants using multimodal biomedical language-model evidence.",
        "AlphaMissense cancer hotspot benchmark; ClinVar-curated test variants",
        "Not extracted",
        "Missense variants",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Disease-specific prediction benchmark",
        "https://pubmed.ncbi.nlm.nih.gov/41000707/",
        "Newer disease-specific comparator; exact metrics require full paper table extraction.",
    ],
    [
        25,
        "AnnotateMissense: a genome-wide annotation and benchmarking framework for missense pathogenicity prediction",
        2026,
        "AnnotateMissense XGBoost/deep benchmark framework",
        "Integrates dbNSFP, ANNOVAR, AlphaMissense, ESM-derived features, conservation, population frequency, prior predictors, and engineered missense features.",
        "ClinVar-labelled missense variants; temporal ClinVar validation",
        "132,714 ClinVar-labelled missense variants",
        "Missense variants",
        "(0.8798, 0.8750, -, -, -, 0.7613, -, -, -, -)",
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        MISSING_TUPLE,
        "Temporal ClinVar validation; preprint",
        "https://arxiv.org/abs/2605.24520",
        "Preprint as of 2026; useful recent benchmark but must be labeled as not peer reviewed.",
    ],
]

PROJECT_ROWS = [
    [
        "Tabular baseline",
        "reference_5k split only",
        5000,
        "(0.9707, 0.9709, -, -, -, 0.9415, 0.9946, 0.9938, 0.0210, 0.0088)",
        "Preliminary reference result; not a final claim.",
    ],
    [
        "GNN only",
        "reference_5k split only",
        5000,
        "(0.9453, 0.9460, -, -, -, 0.8909, 0.9847, 0.9785, 0.0403, 0.0090)",
        "GNN-only is below tabular baseline on this split.",
    ],
    [
        "Tabular + GNN residual",
        "reference_5k split only",
        5000,
        "(0.9773, 0.9776, -, -, -, 0.9549, 0.9958, 0.9958, 0.0215, 0.0163)",
        "Preliminary residual result; final improvement cannot be claimed yet.",
    ],
]


def col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_xml(row_idx: int, col_idx: int, value, style: int | None = None) -> str:
    ref = f"{col_name(col_idx)}{row_idx}"
    style_attr = f' s="{style}"' if style is not None else ""
    if value is None:
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    text = html.escape(str(value), quote=False)
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{text}</t></is></c>'


def worksheet_xml(rows: list[list], widths: list[int], freeze: bool = True, autofilter: bool = True) -> str:
    max_row = len(rows)
    max_col = max(len(row) for row in rows)
    dimension = f"A1:{col_name(max_col)}{max_row}"
    cols = "".join(
        f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>'
        for idx, width in enumerate(widths, start=1)
    )
    sheet_view = '<sheetViews><sheetView workbookViewId="0">'
    if freeze:
        sheet_view += '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
    sheet_view += '</sheetView></sheetViews>'
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        row_style = 1 if row_idx == 1 else 2
        cells = "".join(cell_xml(row_idx, col_idx, value, row_style) for col_idx, value in enumerate(row, start=1))
        height = ' ht="32" customHeight="1"' if row_idx == 1 else ""
        sheet_rows.append(f'<row r="{row_idx}"{height}>{cells}</row>')
    filter_xml = f'<autoFilter ref="{dimension}"/>' if autofilter else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>'
        f"{sheet_view}"
        f"<cols>{cols}</cols>"
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        f"{filter_xml}"
        "</worksheet>"
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{html.escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets>"
        "</workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(rels)}"
        "</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    sheet_overrides = "".join(
        '<Override PartName="/xl/worksheets/sheet{idx}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'.format(idx=idx)
        for idx in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f"{sheet_overrides}"
        "</Types>"
    )


def package_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="10"/><name val="Calibri"/></font>'
        '<font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Calibri"/></font>'
        '</fonts>'
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="2">'
        '<border><left/><right/><top/><bottom/><diagonal/></border>'
        '<border><left style="thin"/><right style="thin"/><top style="thin"/><bottom style="thin"/><diagonal/></border>'
        '</borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="3">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )


def core_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:title>FYP Literature Results Table Omar</dc:title>'
        '<dc:creator>MAGPIE-X</dc:creator>'
        '<cp:lastModifiedBy>MAGPIE-X</cp:lastModifiedBy>'
        '<dcterms:created xsi:type="dcterms:W3CDTF">2026-06-04T00:00:00Z</dcterms:created>'
        '<dcterms:modified xsi:type="dcterms:W3CDTF">2026-06-04T00:00:00Z</dcterms:modified>'
        '</cp:coreProperties>'
    )


def app_xml(sheet_names: list[str]) -> str:
    titles = "".join(f"<vt:lpstr>{html.escape(name)}</vt:lpstr>" for name in sheet_names)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>Microsoft Excel</Application>'
        '<DocSecurity>0</DocSecurity>'
        '<ScaleCrop>false</ScaleCrop>'
        '<HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>3</vt:i4></vt:variant></vt:vector></HeadingPairs>'
        f'<TitlesOfParts><vt:vector size="{len(sheet_names)}" baseType="lpstr">{titles}</vt:vector></TitlesOfParts>'
        '</Properties>'
    )


def build_workbook() -> None:
    literature_rows = [HEADERS, *ROWS]
    definition_rows = [
        ["Metric Tuple", METRIC_TUPLE],
        ["Rule", "Use '-' for missing metrics. Do not fabricate results."],
        ["Accuracy", "Fraction of correct benign/pathogenic predictions."],
        ["F1-score", "Harmonic mean of precision and recall."],
        ["Precision", "Positive predictive value for pathogenic label."],
        ["Recall/Sensitivity", "Pathogenic variants correctly recovered."],
        ["Specificity", "Benign variants correctly rejected."],
        ["MCC", "Matthews correlation coefficient."],
        ["AUROC", "Area under ROC curve."],
        ["AUPRC", "Area under precision-recall curve."],
        ["Brier", "Mean squared probability calibration error; lower is better."],
        ["ECE", "Expected calibration error; lower is better."],
    ]
    project_rows = [
        ["Model", "Scope", "Sample Size", "Metric Tuple", "Notes"],
        *PROJECT_ROWS,
    ]
    sheets = [
        ("Literature Results", literature_rows, [10, 45, 10, 28, 48, 42, 36, 28, 30, 30, 34, 30, 30, 35, 55, 52]),
        ("Metric Tuple Definition", definition_rows, [24, 110]),
        ("Project Reference 5k", project_rows, [28, 28, 14, 80, 52]),
    ]

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        zf.writestr("_rels/.rels", package_rels_xml())
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("docProps/app.xml", app_xml([sheet[0] for sheet in sheets]))
        zf.writestr("xl/workbook.xml", workbook_xml([sheet[0] for sheet in sheets]))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheets)))
        zf.writestr("xl/styles.xml", styles_xml())
        for idx, (_, rows, widths) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml(rows, widths))


def validate_workbook() -> None:
    required = {
        "[Content_Types].xml",
        "_rels/.rels",
        "xl/workbook.xml",
        "xl/_rels/workbook.xml.rels",
        "xl/styles.xml",
        "xl/worksheets/sheet1.xml",
        "xl/worksheets/sheet2.xml",
        "xl/worksheets/sheet3.xml",
    }
    with zipfile.ZipFile(OUTPUT, "r") as zf:
        names = set(zf.namelist())
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"Workbook package is missing required parts: {missing}")
        for name in required:
            if name.endswith(".xml"):
                ET.fromstring(zf.read(name))


def main() -> None:
    build_workbook()
    validate_workbook()
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

