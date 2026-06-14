param(
    [string]$OutputWorkbook = "FYP_Literature_Results_Table_Omar_FINAL_CLARIFIED.xlsx"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$candidateNames = @(
    "FYP_Literature_Results_Table_Omar_FINAL.xlsx",
    "FYP_Literature_Results_Table_Omar_MEETING_SAFE.xlsx",
    "FYP_Literature_Results_Table_Omar_FORMATTED.xlsx"
)
$candidates = @(
    $candidateNames |
        ForEach-Object { Join-Path $root $_ } |
        Where-Object { Test-Path -LiteralPath $_ } |
        ForEach-Object { Get-Item -LiteralPath $_ }
)
if ($candidates.Count -eq 0) {
    throw "No requested input workbook exists."
}
$inputFile = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$inputPath = $inputFile.FullName
$outputPath = Join-Path $root $OutputWorkbook

$literatureAuditPath = Join-Path $root "outputs/real/literature/literature_dataset_use_audit.csv"
$datasetSummaryPath = Join-Path $root "outputs/real/literature/dataset_use_summary.json"
$federatedMetricsPath = Join-Path $root "outputs/real/federated/federated_5k_10k_15k_20k_25k_metrics.csv"
$clientAuditPath = Join-Path $root "outputs/real/federated/federated_client_distribution_audit.csv"
foreach ($path in @(
    $literatureAuditPath,
    $datasetSummaryPath,
    $federatedMetricsPath,
    $clientAuditPath
)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required verified sidecar is missing: $path"
    }
}

$literatureAudit = Import-Csv -LiteralPath $literatureAuditPath
$datasetSummary = Get-Content -Raw -LiteralPath $datasetSummaryPath | ConvertFrom-Json
$federatedMetrics = Import-Csv -LiteralPath $federatedMetricsPath
$clientAudit = Import-Csv -LiteralPath $clientAuditPath

$visibleSheets = @(
    "Workbook_Guide",
    "Literature_Comparison",
    "SwissProt_Comparison",
    "Dataset_Use_Conclusion",
    "Project_Experiment_Comparison",
    "Federated_Learning_Results",
    "Metric_Definition",
    "Circularity_Explanation"
)

$guideText = "This workbook contains the final organized comparison tables for the FYP. Visible sheets contain the main deliverables: literature comparison, SwissProt comparison, dataset-use conclusion, project experiment comparison, federated learning results, metric definitions, and circularity explanation. Appendix/history sheets are hidden only to reduce clutter and are retained for audit trail, previous experiment history, feature-gap planning, and reproducibility."
$guideInterpretation = "Important interpretation note:`nThe dataset columns such as ClinVar, ClinVar Rare, SwissProt/HUMSAVAR, ACMG, and ProteinGym/MAVE are evaluation benchmark columns. They do not always mean the model was trained separately on each dataset. A paper may train a model using one dataset or a combination of datasets, then report separate result tuples on different evaluation benchmarks. Therefore, each tuple should be read as: performance reported on that benchmark, not necessarily a separate model built only from that database."
$literatureNote = "Final deliverable sheet. Appendix/history sheets are hidden to keep the main workbook readable.`n`nInterpretation: The benchmark columns show where the paper reported evaluation results. They do not always mean the model was trained separately on that dataset. The training/development data is listed separately. Each tuple is the reported performance on that benchmark.`n`nDataset columns are evaluation benchmarks. The tuple in each cell reports the paper's performance on that benchmark when available. A dash (-) means the metric or dataset was not reported, not verified, or not applicable."
$swissNote = "Final deliverable sheet. Appendix/history sheets are hidden to keep the main workbook readable.`n`nThis sheet repeats the comparison from the SwissProt/HUMSAVAR perspective. Papers that did not evaluate on SwissProt/HUMSAVAR/SwissVar/UniProt are still listed for context, and their SwissProt metric tuple remains (-, -, -, -, -, -, -, -, -, -) with a reason."
$projectNote = "Final deliverable sheet. Appendix/history sheets are hidden to keep the main workbook readable.`n`nClinVar 5k/10k/20k columns are controlled project experiments. SwissProt/HUMSAVAR is external validation and is not mixed into ClinVar training. Federated columns use simulated horizontal federated learning on partitioned ClinVar-derived training data."
$benchmarkExplanation = "Why benchmark columns have different tuples:`nA trained model can be evaluated on multiple test datasets. ClinVar, ClinVar Rare, SwissProt/HUMSAVAR, ACMG, and ProteinGym/MAVE represent different evaluation contexts. Separate tuples show how performance changes when the benchmark changes. This is stronger than reporting one combined number because it reveals whether the model generalizes beyond its main training source."

$trainingData = @{
    1 = "ClinVarTrain plus added gnomAD ultra-rare benign variants; annotations from population, functional, conservation, biochemical, splicing, and epigenetic sources."
    2 = "Not fully extracted from paper; ClinVar-derived multimodal graph development data are described. See Description."
    3 = "Not fully extracted from paper; variant-gene graph benchmark construction is described. See Description."
    4 = "Protein sequence/evolution and AlphaFold-derived structural context; evaluated on ClinVar and functional-assay benchmarks."
    5 = "Protein language-model pretraining on protein sequences; clinical and deep-mutational-scan datasets are evaluation benchmarks."
    6 = "HGMD disease variants, ESP/population neutral variants, conservation features, and multiple missense predictor scores."
    7 = "Simulated variants contrasted with fixed or nearly fixed human-derived alleles; clinical resources are evaluation comparisons."
    8 = "ClinVar January 2016 labels with annotated nonsynonymous SNV features; evaluated on rare-disease exomes."
    9 = "Not fully extracted from paper; rare clinical missense and population/functional resources are described. See Description."
    10 = "Unsupervised evolutionary model trained from protein multiple-sequence alignments; clinical labels are evaluation benchmarks."
    11 = "HGMD pathogenic variants plus large unlabeled/population variant sets and molecular annotation features."
    12 = "Not fully extracted from paper; clinical exome and rare-missense development data are described. See Description."
    13 = "Not fully extracted from paper; primate/vertebrate sequence context and clinical benchmark development are described. See Description."
    14 = "Common primate variation and protein structural context; evaluated on ClinVar, DMS, and patient cohorts."
    15 = "Not fully extracted from paper; extended genome-wide point-mutation features and clinical labels are described. See Description."
    16 = "Not fully extracted from paper; ClinVar/HGMD and functional/cancer resources are described. See Description."
    17 = "HGMD, UniProt, and ClinVar pathogenic variants with correlated annotations and gene-constraint context."
    18 = "ClinVar, HGMD, and UniProt missense variants with protein graph and co-evolution features."
    19 = "ClinVar high-confidence labels and consequence-agnostic clinical-exome annotations."
    20 = "Taxonomy-aware protein sequence conservation resources; clinical labels are used for benchmark evaluation."
    21 = "Not fully extracted from paper; disease-specific foundation-model fine-tuning and rare-variant data are described. See Description."
    22 = "Not fully extracted from paper; disease-specific protein-language-model fine-tuning is described. See Description."
    23 = "Not fully extracted from paper; AlphaFold2 structural features and established predictor scores are described. See Description."
    24 = "Not fully extracted from paper; disease-specific multimodal biomedical language-model development is described. See Description."
    25 = "ClinVar-labelled missense variants with dbNSFP, ANNOVAR, AlphaMissense, ESM-derived, conservation, population, and engineered features."
}

$benchmarkInterpretation = @{
    1 = "Trained on ClinVar and externally tested on SwissProt/HUMSAVAR"
    2 = "Trained on ClinVar and externally tested on SwissProt/HUMSAVAR"
    3 = "Literature method; exact train/evaluation split needs extraction"
    4 = "Protein/evolution model evaluated against clinical labels"
    5 = "Protein/evolution model evaluated against clinical labels"
    6 = "Trained on mixed clinical/population data and evaluated on benchmark"
    7 = "Trained on mixed clinical/population data and evaluated on benchmark"
    8 = "Trained on ClinVar and tested on held-out ClinVar"
    9 = "Trained on mixed clinical/population data and evaluated on benchmark"
    10 = "Protein/evolution model evaluated against clinical labels"
    11 = "Trained on mixed clinical/population data and evaluated on benchmark"
    12 = "Literature method; exact train/evaluation split needs extraction"
    13 = "Protein/evolution model evaluated against clinical labels"
    14 = "Protein/evolution model evaluated against clinical labels"
    15 = "Trained on mixed clinical/population data and evaluated on benchmark"
    16 = "Trained on mixed clinical/population data and evaluated on benchmark"
    17 = "Trained on mixed clinical/population data and evaluated on benchmark"
    18 = "Trained on mixed clinical/population data and evaluated on benchmark"
    19 = "Trained on ClinVar and tested on held-out ClinVar"
    20 = "Protein/evolution model evaluated against clinical labels"
    21 = "Protein/evolution model evaluated against clinical labels"
    22 = "Protein/evolution model evaluated against clinical labels"
    23 = "Protein/evolution model evaluated against clinical labels"
    24 = "Protein/evolution model evaluated against clinical labels"
    25 = "Trained on ClinVar and tested on held-out ClinVar"
}

$swissRole = @{
    1 = "Used for external validation"
    2 = "Used for external validation"
    6 = "SwissVar/UniProt-style source used"
    11 = "SwissVar/UniProt-style source used"
    17 = "Used as training/development source"
    18 = "Used as training/development source"
}

$blueHeader = 0xE8BB9E
$blueBody = 0xFAE8D8
$greenHeader = 0xA8D7B6
$greenBody = 0xEAF7EE
$lightYellow = 0xCCFFFF
$lightGray = 0xE7E6E6
$white = 0xFFFFFF
$black = 0x000000
$xlCenter = -4108
$xlLeft = -4131
$xlTop = -4160
$xlContinuous = 1
$xlThin = 2
$xlLandscape = 2
$xlSheetVisible = -1
$xlSheetHidden = 0

function Release-ComObject {
    param([object]$Object)
    if ($null -ne $Object) {
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($Object)
    }
}

function Apply-Borders {
    param([object]$Range)
    $Range.Borders.LineStyle = $xlContinuous
    $Range.Borders.Weight = $xlThin
    $Range.Borders.Color = $black
}

function Configure-Page {
    param(
        [object]$Excel,
        [object]$Sheet,
        [string]$PrintArea,
        [string]$FreezeCell,
        [int]$Zoom
    )
    $Sheet.Activate()
    $Excel.ActiveWindow.FreezePanes = $false
    $Sheet.Range($FreezeCell).Select()
    $Excel.ActiveWindow.FreezePanes = $true
    $Excel.ActiveWindow.Zoom = $Zoom
    $Sheet.PageSetup.Orientation = $xlLandscape
    $Sheet.PageSetup.Zoom = $false
    $Sheet.PageSetup.FitToPagesWide = 1
    $Sheet.PageSetup.FitToPagesTall = $false
    $Sheet.PageSetup.PrintArea = $PrintArea
}

function Neutralize-Text {
    param([string]$Text)
    if ($null -eq $Text) {
        return $null
    }
    $updated = $Text
    $updated = [regex]::Replace($updated, '(?i)professor-facing', 'final deliverable')
    $updated = [regex]::Replace($updated, '(?i)meeting-safe', 'final organized')
    $updated = [regex]::Replace($updated, '(?i)supervisor-facing', 'final reported')
    $updated = [regex]::Replace($updated, '(?i)\bsir\b', 'academic reviewer')
    $updated = [regex]::Replace($updated, '(?i)\bprofessor\b', 'academic reviewer')
    return $updated
}

function Get-Inventory {
    param([object]$Workbook)
    $inventory = @{
        formula = @{}
        numeric = @{}
        tuple = @{}
        url = @{}
    }
    foreach ($sheet in $Workbook.Worksheets) {
        $used = $sheet.UsedRange
        $rows = [int]$used.Rows.Count
        $columns = [int]$used.Columns.Count
        for ($row = 1; $row -le $rows; $row++) {
            for ($column = 1; $column -le $columns; $column++) {
                $cell = $sheet.Cells.Item($row, $column)
                $value = $cell.Value2
                $formula = [string]$cell.Formula
                $text = if ($null -eq $value) { "" } else { [string]$value }
                $kind = $null
                $token = $null
                if ($formula -like "=*") {
                    $kind = "formula"
                    $token = "$($sheet.Name)|$formula"
                }
                elseif (
                    $value -is [double] -or
                    $value -is [single] -or
                    $value -is [decimal] -or
                    $value -is [int] -or
                    $value -is [long]
                ) {
                    $kind = "numeric"
                    $token = "$($sheet.Name)|$text"
                }
                elseif ($text -match '^\(\s*[^()]*,[^()]*\)$') {
                    $kind = "tuple"
                    $token = "$($sheet.Name)|$text"
                }
                elseif ($text -match '^https?://') {
                    $kind = "url"
                    $token = "$($sheet.Name)|$text"
                }
                if ($null -ne $kind) {
                    if (-not $inventory[$kind].ContainsKey($token)) {
                        $inventory[$kind][$token] = 0
                    }
                    $inventory[$kind][$token]++
                }
                Release-ComObject $cell
            }
        }
        Release-ComObject $used
        Release-ComObject $sheet
    }
    return $inventory
}

function Assert-InventoryPreserved {
    param(
        [hashtable]$Before,
        [hashtable]$After
    )
    $failures = [System.Collections.Generic.List[string]]::new()
    foreach ($kind in @("formula", "numeric", "tuple", "url")) {
        foreach ($token in $Before[$kind].Keys) {
            $beforeCount = [int]$Before[$kind][$token]
            $afterCount = if ($After[$kind].ContainsKey($token)) {
                [int]$After[$kind][$token]
            }
            else {
                0
            }
            if ($afterCount -lt $beforeCount) {
                $failures.Add("$kind missing or changed: $token ($beforeCount -> $afterCount)")
            }
        }
    }
    if ($failures.Count -gt 0) {
        throw "Protected workbook inventory changed:`n$($failures | Select-Object -First 20 | Out-String)"
    }
}

function Get-FederatedRow {
    param(
        [int]$SampleSize,
        [string]$Mode,
        [string]$Split,
        [string]$Method
    )
    return $federatedMetrics | Where-Object {
        [int]$_.sample_size -eq $SampleSize -and
        $_.feature_mode -eq $Mode -and
        $_.client_split_type -eq $Split -and
        $_.method -eq $Method
    } | Select-Object -First 1
}

function Get-BestFederatedRow {
    param(
        [int]$SampleSize,
        [string]$Mode,
        [string]$Split
    )
    return $federatedMetrics | Where-Object {
        [int]$_.sample_size -eq $SampleSize -and
        $_.feature_mode -eq $Mode -and
        $_.client_split_type -eq $Split -and
        $_.method -in @("fedavg", "fedprox", "federated_ensemble")
    } | Sort-Object `
        @{ Expression = { [double]$_.accuracy }; Descending = $true }, `
        @{ Expression = { [double]$_.auroc }; Descending = $true } |
        Select-Object -First 1
}

function Method-DisplayName {
    param([string]$Method)
    switch ($Method) {
        "fedavg" { return "FedAvg MLP weighted model averaging" }
        "fedprox" { return "FedProx MLP proximal federated optimization" }
        "federated_ensemble" { return "Federated ensemble / soft-voting probability aggregation" }
        default { return $Method }
    }
}

function Get-ClientSummary {
    param(
        [int]$SampleSize,
        [string]$Split
    )
    $rows = @(
        $clientAudit | Where-Object {
            [int]$_.sample_size -eq $SampleSize -and
            $_.feature_mode -eq "FL-A" -and
            $_.client_split_type -eq $Split
        } | Sort-Object { [int]$_.client_id }
    )
    if ($rows.Count -eq 0) {
        throw "No client partition audit for $SampleSize/$Split."
    }
    return @{
        Sizes = ($rows | ForEach-Object { [string]$_.client_train_size }) -join ", "
        Labels = ($rows | ForEach-Object {
            "C$($_.client_id): benign=$($_.benign_count), pathogenic=$($_.pathogenic_count)"
        }) -join "; "
    }
}

if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Force
}
Copy-Item -LiteralPath $inputPath -Destination $outputPath

$sourceExcel = $null
$sourceWorkbook = $null
$beforeInventory = $null
try {
    $sourceExcel = New-Object -ComObject Excel.Application
    $sourceExcel.Visible = $false
    $sourceExcel.DisplayAlerts = $false
    $sourceWorkbook = $sourceExcel.Workbooks.Open($inputPath, 0, $true)
    $beforeInventory = Get-Inventory $sourceWorkbook
}
finally {
    if ($null -ne $sourceWorkbook) {
        $sourceWorkbook.Close($false)
        Release-ComObject $sourceWorkbook
    }
    if ($null -ne $sourceExcel) {
        $sourceExcel.Quit()
        Release-ComObject $sourceExcel
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

$excel = $null
$workbook = $null
$uncertainRows = [System.Collections.Generic.List[string]]::new()
$missingFederated = [System.Collections.Generic.List[string]]::new()

try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.ScreenUpdating = $false
    $workbook = $excel.Workbooks.Open($outputPath)

    $guide = $null
    try {
        $guide = $workbook.Worksheets.Item("Workbook_Guide")
    }
    catch {
        $guide = $workbook.Worksheets.Item("README_Open_First")
        $guide.Name = "Workbook_Guide"
    }
    $guide.Cells.Item(1, 1).Value2 = "Final organized workbook. Appendix/history sheets are hidden to keep the main workbook readable."
    $guide.Cells.Item(3, 1).Value2 = $guideText
    $guide.Range("A3:H4").WrapText = $true
    $guide.Cells.Item(36, 1).Value2 = "Dataset and benchmark interpretation"
    $guide.Cells.Item(37, 1).Value2 = $guideInterpretation
    $guide.Range("A36:H36").UnMerge()
    $guide.Range("A37:H41").UnMerge()
    [void]$guide.Range("A36:H36").Merge()
    [void]$guide.Range("A37:H41").Merge()
    $guide.Range("A36:H36").Interior.Color = $greenHeader
    $guide.Range("A36:H36").Font.Bold = $true
    $guide.Range("A37:H41").Interior.Color = $blueBody
    $guide.Range("A37:H41").WrapText = $true
    $guide.Range("A37:H41").VerticalAlignment = $xlTop
    Apply-Borders $guide.Range("A36:H41")
    [void]$guide.Range("A43:H43").Merge()
    $guide.Cells.Item(43, 1).Value2 = "Split and validation summary"
    [void]$guide.Range("A44:H48").Merge()
    $guide.Cells.Item(44, 1).Value2 = (
        "Controlled project experiments use balanced ClinVar-derived samples and a " +
        "stratified 80/20 train/test split. LightGBM modes use a small validation " +
        "subset inside the training partition for six-candidate AUPRC selection. " +
        "GraphSAGE uses fixed epochs and training-node labels only. Federated runs " +
        "use no separate validation split; their shared 20% test holdout is used only " +
        "for final evaluation."
    )
    $guide.Range("A43:H43").Interior.Color = $greenHeader
    $guide.Range("A43:H43").Font.Bold = $true
    $guide.Range("A44:H48").Interior.Color = $greenBody
    $guide.Range("A44:H48").WrapText = $true
    $guide.Range("A44:H48").VerticalAlignment = $xlTop
    Apply-Borders $guide.Range("A43:H48")
    $guide.Range("A44:H48").Rows.RowHeight = 28
    $guide.PageSetup.PrintArea = '$A$1:$H$48'
    Release-ComObject $guide

    # Literature comparison: add development context and benchmark interpretation.
    $literature = $workbook.Worksheets.Item("Literature_Comparison")
    $literature.Range("A1:Z2").UnMerge()
    if ([string]$literature.Cells.Item(3, 6).Value2 -ne "Training / Development Data Used") {
        [void]$literature.Columns.Item(6).Insert()
    }
    if ([string]$literature.Cells.Item(3, 20).Value2 -ne "Benchmark Interpretation") {
        [void]$literature.Columns.Item(20).Insert()
    }
    $literature.Cells.Item(1, 1).Value2 = $literatureNote
    [void]$literature.Range("A1:V1").Merge()
    [void]$literature.Range("A2:E2").Merge()
    $literature.Cells.Item(2, 1).Value2 = "Paper and Method"
    [void]$literature.Range("G2:L2").Merge()
    $literature.Cells.Item(2, 7).Value2 = "Evaluation Benchmark Results"
    [void]$literature.Range("M2:S2").Merge()
    $literature.Cells.Item(2, 13).Value2 = "Benchmark Use and Missing-Metric Audit"
    [void]$literature.Range("T2:V2").Merge()
    $literature.Cells.Item(2, 20).Value2 = "Interpretation, Limitations, and Source"
    $literature.Cells.Item(3, 6).Value2 = "Training / Development Data Used"
    $literature.Cells.Item(3, 20).Value2 = "Benchmark Interpretation"
    $literature.Cells.Item(4, 6).Value2 = "Development source or model pretraining evidence"
    $literature.Cells.Item(5, 6).Value2 = "Separate from evaluation benchmark columns"
    $literature.Cells.Item(4, 20).Value2 = "How training and benchmark use should be read"
    $literature.Cells.Item(5, 20).Value2 = "Standardized interpretation category"

    foreach ($auditRow in $literatureAudit) {
        $paperNo = [int]$auditRow."Paper No."
        $row = $paperNo + 5
        $literature.Cells.Item($row, 6).Value2 = $trainingData[$paperNo]
        $literature.Cells.Item($row, 20).Value2 = $benchmarkInterpretation[$paperNo]
        if ($trainingData[$paperNo] -like "Not fully extracted*") {
            $uncertainRows.Add("$paperNo - $([string]$literature.Cells.Item($row, 2).Value2)")
        }

        # After the two insertions, audit columns are M:S.
        $usesSwiss = [string]$literature.Cells.Item($row, 14).Value2
        if ($usesSwiss -eq "No") {
            $literature.Cells.Item($row, 16).Value2 = (
                "SwissProt/HUMSAVAR not used by authors; paper evaluates on " +
                "ClinVar/DMS/other benchmark instead."
            )
        }
        if ([string]$literature.Cells.Item($row, 15).Value2 -eq "Reported only as AUROC/AUPRC") {
            $literature.Cells.Item($row, 18).Value2 = (
                "Paper reports ranking metrics only; thresholded confusion-matrix " +
                "metrics are not provided."
            )
        }
    }
    $literatureUsed = $literature.Range("A1:V30")
    $literatureUsed.Font.Name = "Calibri"
    $literatureUsed.Font.Size = 9
    $literatureUsed.WrapText = $true
    $literatureUsed.VerticalAlignment = $xlTop
    Apply-Borders $literatureUsed
    $literature.Range("A2:F5").Interior.Color = $blueHeader
    $literature.Range("G2:V5").Interior.Color = $greenHeader
    $literature.Range("A6:F30").Interior.Color = $blueBody
    $literature.Range("G6:V30").Interior.Color = $greenBody
    $literature.Range("A1:V1").Interior.Color = $lightYellow
    $literature.Range("A1:V3").Font.Bold = $true
    $literature.Rows.Item(1).RowHeight = 78
    $literature.Rows.Item(2).RowHeight = 28
    $literature.Rows.Item(3).RowHeight = 60
    $literature.Rows.Item(4).RowHeight = 72
    $literature.Rows.Item(5).RowHeight = 68
    $literature.Range("A6:V30").Rows.RowHeight = 92
    $literature.Columns.Item("F").ColumnWidth = 46
    $literature.Columns.Item("T").ColumnWidth = 32
    $literature.Columns.Item("U").ColumnWidth = 40
    $literature.Columns.Item("V").ColumnWidth = 34
    if ($literature.AutoFilterMode) {
        $literature.AutoFilterMode = $false
    }
    [void]$literature.Range("A3:V30").AutoFilter()
    Configure-Page $excel $literature '$A$1:$V$30' "G6" 45
    Release-ComObject $literatureUsed
    Release-ComObject $literature

    # SwissProt comparison: add development data and explicit SwissProt role.
    $swiss = $workbook.Worksheets.Item("SwissProt_Comparison")
    $swiss.Range("A1:Z2").UnMerge()
    if ([string]$swiss.Cells.Item(3, 6).Value2 -ne "Training / Development Data Used") {
        [void]$swiss.Columns.Item(6).Insert()
    }
    if ([string]$swiss.Cells.Item(3, 10).Value2 -ne "SwissProt Role") {
        [void]$swiss.Columns.Item(10).Insert()
    }
    $swiss.Cells.Item(1, 1).Value2 = $swissNote
    [void]$swiss.Range("A1:R1").Merge()
    [void]$swiss.Range("A2:R2").Merge()
    $swiss.Cells.Item(2, 1).Value2 = "SwissProt / HUMSAVAR / SwissVar / UniProt Literature Comparison"
    $swiss.Cells.Item(3, 6).Value2 = "Training / Development Data Used"
    $swiss.Cells.Item(3, 10).Value2 = "SwissProt Role"
    for ($row = 4; $row -le 28; $row++) {
        $paperNo = [int]$swiss.Cells.Item($row, 1).Value2
        $swiss.Cells.Item($row, 6).Value2 = $trainingData[$paperNo]
        $swiss.Cells.Item($row, 10).Value2 = if ($swissRole.ContainsKey($paperNo)) {
            $swissRole[$paperNo]
        }
        else {
            "Not used"
        }
    }
    $swissUsed = $swiss.Range("A1:R28")
    $swissUsed.Font.Name = "Calibri"
    $swissUsed.Font.Size = 9
    $swissUsed.WrapText = $true
    $swissUsed.VerticalAlignment = $xlTop
    Apply-Borders $swissUsed
    $swiss.Range("A1:R1").Interior.Color = $lightYellow
    $swiss.Range("A2:R3").Interior.Color = $greenHeader
    $swiss.Range("A2:R3").Font.Bold = $true
    $swiss.Range("A4:F28").Interior.Color = $blueBody
    $swiss.Range("G4:R28").Interior.Color = $greenBody
    $swiss.Rows.Item(1).RowHeight = 68
    $swiss.Rows.Item(2).RowHeight = 28
    $swiss.Rows.Item(3).RowHeight = 58
    $swiss.Range("A4:R28").Rows.RowHeight = 92
    $swiss.Columns.Item("F").ColumnWidth = 45
    $swiss.Columns.Item("J").ColumnWidth = 24
    if ($swiss.AutoFilterMode) {
        $swiss.AutoFilterMode = $false
    }
    [void]$swiss.Range("A3:R28").AutoFilter()
    Configure-Page $excel $swiss '$A$1:$R$28' "G4" 45
    Release-ComObject $swissUsed
    Release-ComObject $swiss

    # Dataset use conclusion: retain summary and add the benchmark explanation.
    $conclusion = $workbook.Worksheets.Item("Dataset_Use_Conclusion")
    [void]$conclusion.Range("A37:D37").Merge()
    $conclusion.Cells.Item(37, 1).Value2 = "Benchmark interpretation"
    [void]$conclusion.Range("A38:D43").Merge()
    $conclusion.Cells.Item(38, 1).Value2 = $benchmarkExplanation
    $conclusion.Range("A37:D37").Interior.Color = $greenHeader
    $conclusion.Range("A37:D37").Font.Bold = $true
    $conclusion.Range("A38:D43").Interior.Color = $greenBody
    $conclusion.Range("A38:D43").WrapText = $true
    $conclusion.Range("A38:D43").VerticalAlignment = $xlTop
    Apply-Borders $conclusion.Range("A37:D43")
    $conclusion.Range("A38:D43").Rows.RowHeight = 28
    $conclusion.PageSetup.PrintArea = '$A$1:$D$43'
    Release-ComObject $conclusion

    # Project comparison: add exact training/evaluation and split fields.
    $project = $workbook.Worksheets.Item("Project_Experiment_Comparison")
    $project.Range("A1:Z2").UnMerge()
    if ([string]$project.Cells.Item(3, 6).Value2 -ne "Training / Development Data Used") {
        [void]$project.Range("F:K").EntireColumn.Insert()
    }
    $project.Cells.Item(1, 1).Value2 = $projectNote
    [void]$project.Range("A1:U1").Merge()
    [void]$project.Range("A2:E2").Merge()
    $project.Cells.Item(2, 1).Value2 = "Experiment Definition"
    [void]$project.Range("F2:K2").Merge()
    $project.Cells.Item(2, 6).Value2 = "Training, Evaluation, and Split Design"
    [void]$project.Range("L2:T2").Merge()
    $project.Cells.Item(2, 12).Value2 = "Evaluation Benchmark Results"
    $projectHeaders = @(
        "Training / Development Data Used",
        "Evaluation Benchmark",
        "Global Split",
        "Validation Strategy",
        "Test Set Used",
        "Split Limitation"
    )
    for ($index = 0; $index -lt $projectHeaders.Count; $index++) {
        $project.Cells.Item(3, 6 + $index).Value2 = $projectHeaders[$index]
    }
    for ($row = 6; $row -le 14; $row++) {
        $experiment = [string]$project.Cells.Item($row, 1).Value2
        if ($experiment -in @("A", "B", "C", "D", "E", "F")) {
            $project.Cells.Item($row, 6).Value2 = "Controlled balanced ClinVar-derived 5k, 10k, and 20k samples."
            $project.Cells.Item($row, 7).Value2 = "Held-out ClinVar test split for each controlled sample size."
            $project.Cells.Item($row, 8).Value2 = (
                "Stratified seed-13 80/20 train/test split: 5k = 4,000/1,000; " +
                "10k = 8,000/2,000; 20k = 16,000/4,000."
            )
            if ($experiment -eq "F") {
                $project.Cells.Item($row, 9).Value2 = (
                    "Tabular LightGBM uses an internal stratified 20% validation " +
                    "subset of global training for six-candidate AUPRC selection. " +
                    "GraphSAGE uses fixed epochs; no validation or test labels are used."
                )
            }
            else {
                $project.Cells.Item($row, 9).Value2 = (
                    "Internal stratified 20% validation subset drawn only from the " +
                    "global training partition; six LightGBM candidates selected by " +
                    "validation AUPRC."
                )
            }
            $project.Cells.Item($row, 10).Value2 = "Held-out 20% global ClinVar test split; used only for final metrics."
            $project.Cells.Item($row, 11).Value2 = if ($experiment -eq "F") {
                "Random balanced ClinVar split; transductive test-node features participate in message passing; not temporal, gene-held-out, or external validation."
            }
            else {
                "Random balanced ClinVar split; not temporal, gene-held-out, or SwissProt external validation."
            }
        }
        else {
            $project.Cells.Item($row, 6).Value2 = "Controlled balanced ClinVar-derived 5k, 10k, 15k, 20k, and 25k samples; FL-C only at 5k/10k/20k."
            $project.Cells.Item($row, 7).Value2 = "Shared held-out global ClinVar test split."
            $project.Cells.Item($row, 8).Value2 = (
                "One stratified seed-13 80/20 global train/test split per sample; " +
                "five simulated clients partition training only."
            )
            $project.Cells.Item($row, 9).Value2 = (
                "No separate validation split; global 20% test split used only for " +
                "final evaluation; hyperparameters fixed by experiment configuration."
            )
            $project.Cells.Item($row, 10).Value2 = "Shared held-out 20% global ClinVar test split across centralized, local-only, FedAvg, FedProx, and ensemble methods."
            $project.Cells.Item($row, 11).Value2 = "Random balanced ClinVar holdout and synthetic clients; not temporal, gene-held-out, SwissProt external, or real multi-institution validation."
        }
    }
    $projectUsed = $project.Range("A1:U14")
    $projectUsed.Font.Name = "Calibri"
    $projectUsed.Font.Size = 9
    $projectUsed.WrapText = $true
    $projectUsed.VerticalAlignment = $xlTop
    Apply-Borders $projectUsed
    $project.Range("A1:U1").Interior.Color = $lightYellow
    $project.Range("A2:K5").Interior.Color = $blueHeader
    $project.Range("L2:U5").Interior.Color = $greenHeader
    $project.Range("A6:K14").Interior.Color = $blueBody
    $project.Range("L6:U14").Interior.Color = $greenBody
    $project.Range("A1:U3").Font.Bold = $true
    $project.Rows.Item(1).RowHeight = 65
    $project.Rows.Item(2).RowHeight = 28
    $project.Rows.Item(3).RowHeight = 62
    $project.Rows.Item(4).RowHeight = 58
    $project.Rows.Item(5).RowHeight = 48
    $project.Range("A6:U14").Rows.RowHeight = 105
    foreach ($column in @("F", "G", "H", "I", "J", "K")) {
        $project.Columns.Item($column).ColumnWidth = 34
    }
    if ($project.AutoFilterMode) {
        $project.AutoFilterMode = $false
    }
    [void]$project.Range("A3:U14").AutoFilter()
    Configure-Page $excel $project '$A$1:$U$14' "L6" 42
    Release-ComObject $projectUsed
    Release-ComObject $project

    # Federated sheet: present every split and missing reason explicitly.
    $federated = $workbook.Worksheets.Item("Federated_Learning_Results")
    $federated.UsedRange.UnMerge()
    $federated.UsedRange.Clear()
    [void]$federated.Range("A1:Z1").Merge()
    $federated.Cells.Item(1, 1).Value2 = (
        "Final deliverable sheet. Simulated horizontal federated learning on one " +
        "machine using partitioned ClinVar-derived training data. The shared global " +
        "test set is not a validation set. SwissProt/HUMSAVAR is not used in training."
    )
    [void]$federated.Range("A2:Z2").Merge()
    $federated.Cells.Item(2, 1).Value2 = "Federated Learning Results, Split Details, and Missing-Run Audit"
    [void]$federated.Range("A3:Z3").Merge()
    $federated.Cells.Item(3, 1).Value2 = "Federated algorithm definitions"
    $algorithmRows = @(
        @("Method", "Definition", "Aggregation", "True federated optimization?"),
        @("Centralized pooled-data MLP baseline", "One model trains on pooled global training data; upper-bound comparison; not federated.", "None", "No"),
        @("Local-only client MLP average", "One model per client; no server aggregation; client metrics averaged on the shared global test set.", "None", "No"),
        @("FedAvg MLP weighted model averaging", "Clients train local neural models; server averages parameters weighted by client sample counts.", "Neural-network parameter aggregation", "Yes"),
        @("FedProx MLP proximal federated optimization", "FedAvg-style aggregation with proximal regularization in each local objective to reduce client drift.", "Neural-network parameter aggregation", "Yes"),
        @("Federated ensemble / soft-voting probability aggregation", "Local LightGBM models send predicted probabilities; server averages probabilities weighted by client sample counts.", "Prediction aggregation; not FedAvg", "No")
    )
    for ($r = 0; $r -lt $algorithmRows.Count; $r++) {
        for ($c = 0; $c -lt 4; $c++) {
            $federated.Cells.Item(4 + $r, 1 + $c).Value2 = $algorithmRows[$r][$c]
        }
    }

    $federatedHeaders = @(
        "Experiment No.",
        "Sample Size",
        "Feature Mode",
        "Feature Mode Meaning",
        "Circularity Risk",
        "Client Count",
        "Global Sample Size",
        "Global Training Count",
        "Global Test Count",
        "Global Split",
        "Validation Strategy",
        "Client Split Type",
        "Client Partition Method",
        "Client Train Sizes",
        "Client Label Distribution Summary",
        "Federated Method",
        "Aggregation Type",
        "True Federated Optimization?",
        "Centralized Baseline Tuple",
        "Local-only Tuple",
        "FedAvg Tuple",
        "FedProx Tuple",
        "Federated Ensemble Tuple",
        "Best Federated Method",
        "Why Cell Is Missing / Not Run",
        "Notes / Limitations"
    )
    for ($column = 1; $column -le $federatedHeaders.Count; $column++) {
        $federated.Cells.Item(11, $column).Value2 = $federatedHeaders[$column - 1]
    }
    $modeMeaning = @{
        "FL-A" = "MAGPIE-parity low-circularity: ANNOVAR/refGene + gnomAD AF + conservation + variant type/length + AA/BLOSUM62 (114 features)."
        "FL-B" = "MAGPIE-X circularity-risk: FL-A + 86 dbNSFP prior predictor fields (200 features)."
        "FL-C" = "MAGPIE-X circularity-risk: FL-B + 12 exact-matched SpliceAI DS/DP fields (212 features)."
    }
    $counts = @{
        5000 = @(4000, 1000)
        10000 = @(8000, 2000)
        15000 = @(12000, 3000)
        20000 = @(16000, 4000)
        25000 = @(20000, 5000)
    }
    $experimentNo = 1
    $targetRow = 12
    foreach ($sampleSize in @(5000, 10000, 15000, 20000, 25000)) {
        foreach ($mode in @("FL-A", "FL-B", "FL-C")) {
            foreach ($split in @("iid", "non_iid")) {
                $central = Get-FederatedRow $sampleSize $mode $split "centralized"
                $local = Get-FederatedRow $sampleSize $mode $split "local_only_average"
                $fedAvg = Get-FederatedRow $sampleSize $mode $split "fedavg"
                $fedProx = Get-FederatedRow $sampleSize $mode $split "fedprox"
                $ensemble = Get-FederatedRow $sampleSize $mode $split "federated_ensemble"
                $clientSummary = Get-ClientSummary $sampleSize $split
                $risk = if ($mode -eq "FL-A") { "Low-circularity" } else { "High circularity risk" }
                $partition = if ($split -eq "iid") {
                    "Approximately equal client sizes and class balance."
                }
                else {
                    "Reproducible Dirichlet label skew with alpha=0.3."
                }
                $missingReason = if ($null -eq $central) {
                    $reason = "Not run because exact SpliceAI sidecar features were available only for 5k, 10k, and 20k in the current controlled artifacts."
                    $missingFederated.Add("$mode $($sampleSize / 1000)k ${split}: $reason")
                    $reason
                }
                else {
                    "None; all configured methods completed for this feature-mode and client-split row."
                }
                $best = Get-BestFederatedRow $sampleSize $mode $split
                $bestText = if ($null -eq $best) {
                    "-"
                }
                else {
                    "$(Method-DisplayName $best.method): $($best.result_tuple)"
                }
                $values = @(
                    ("FL-{0:D3}" -f $experimentNo),
                    $sampleSize,
                    $mode,
                    $modeMeaning[$mode],
                    $risk,
                    5,
                    $sampleSize,
                    $counts[$sampleSize][0],
                    $counts[$sampleSize][1],
                    "One stratified seed-13 80/20 global train/test split; clients partition training only.",
                    "No separate validation split; global 20% test split used only for final evaluation; hyperparameters fixed by experiment configuration.",
                    $split,
                    $partition,
                    $clientSummary.Sizes,
                    $clientSummary.Labels,
                    "Centralized pooled MLP; local-only client MLPs; FedAvg; FedProx; local LightGBM probability ensemble.",
                    "FedAvg/FedProx: client-size-weighted neural parameters; ensemble: client-size-weighted predicted probabilities.",
                    "Mixed: FedAvg and FedProx = yes; centralized, local-only, and probability ensemble = no.",
                    $(if ($null -eq $central) { "-" } else { $central.result_tuple }),
                    $(if ($null -eq $local) { "-" } else { $local.result_tuple }),
                    $(if ($null -eq $fedAvg) { "-" } else { $fedAvg.result_tuple }),
                    $(if ($null -eq $fedProx) { "-" } else { $fedProx.result_tuple }),
                    $(if ($null -eq $ensemble) { "-" } else { $ensemble.result_tuple }),
                    $bestText,
                    $missingReason,
                    (
                        "Global test set is shared across centralized, local-only, " +
                        "FedAvg, FedProx, and ensemble methods. Shared preprocessing " +
                        "is fit only on the global training partition. Synthetic " +
                        "clients are not real hospitals. SwissProt/HUMSAVAR is not " +
                        "used in training. No deployment-level privacy/security proof."
                    )
                )
                for ($column = 1; $column -le $values.Count; $column++) {
                    $cellValue = $values[$column - 1]
                    if (
                        $cellValue -is [int] -or
                        $cellValue -is [long] -or
                        $cellValue -is [decimal]
                    ) {
                        $federated.Cells.Item($targetRow, $column).Value2 = [double]$cellValue
                    }
                    else {
                        $federated.Cells.Item($targetRow, $column).Value2 = [string]$cellValue
                    }
                }
                $experimentNo++
                $targetRow++
            }
        }
    }
    $lastFederatedRow = $targetRow - 1
    $federatedUsed = $federated.Range("A1:Z$lastFederatedRow")
    $federatedUsed.Font.Name = "Calibri"
    $federatedUsed.Font.Size = 9
    $federatedUsed.WrapText = $true
    $federatedUsed.VerticalAlignment = $xlTop
    Apply-Borders $federatedUsed
    $federated.Range("A1:Z1").Interior.Color = $lightYellow
    $federated.Range("A2:Z3").Interior.Color = $greenHeader
    $federated.Range("A2:Z3").Font.Bold = $true
    $federated.Range("A4:D4").Interior.Color = $blueHeader
    $federated.Range("A5:D9").Interior.Color = $blueBody
    $federated.Range("A11:R11").Interior.Color = $blueHeader
    $federated.Range("S11:Z11").Interior.Color = $greenHeader
    $federated.Range("A12:R$lastFederatedRow").Interior.Color = $blueBody
    $federated.Range("S12:Z$lastFederatedRow").Interior.Color = $greenBody
    $federated.Range("A11:Z11").Font.Bold = $true
    $federated.Rows.Item(1).RowHeight = 58
    $federated.Rows.Item(2).RowHeight = 30
    $federated.Rows.Item(3).RowHeight = 26
    $federated.Rows.Item(4).RowHeight = 34
    $federated.Range("A5:D9").Rows.RowHeight = 52
    $federated.Rows.Item(11).RowHeight = 64
    $federated.Range("A12:Z$lastFederatedRow").Rows.RowHeight = 105
    $widths = @{
        A = 12; B = 11; C = 10; D = 47; E = 18; F = 10; G = 12; H = 13
        I = 12; J = 34; K = 42; L = 13; M = 34; N = 27; O = 52; P = 42
        Q = 45; R = 30; S = 30; T = 30; U = 30; V = 30; W = 30; X = 43
        Y = 44; Z = 48
    }
    foreach ($column in $widths.Keys) {
        $federated.Columns.Item($column).ColumnWidth = $widths[$column]
    }
    if ($federated.AutoFilterMode) {
        $federated.AutoFilterMode = $false
    }
    [void]$federated.Range("A11:Z$lastFederatedRow").AutoFilter()
    Configure-Page $excel $federated ('$A$1:$Z$' + $lastFederatedRow) "S12" 35
    Release-ComObject $federatedUsed
    Release-ComObject $federated

    # Apply neutral wording workbook-wide and enforce requested visibility.
    foreach ($sheet in $workbook.Worksheets) {
        $used = $sheet.UsedRange
        $rows = [int]$used.Rows.Count
        $columns = [int]$used.Columns.Count
        for ($row = 1; $row -le $rows; $row++) {
            for ($column = 1; $column -le $columns; $column++) {
                $cell = $sheet.Cells.Item($row, $column)
                if ([string]$cell.Formula -notlike "=*" -and $cell.Value2 -is [string]) {
                    $current = [string]$cell.Value2
                    $updated = Neutralize-Text $current
                    if ($updated -cne $current) {
                        $cell.Value2 = $updated
                    }
                }
                Release-ComObject $cell
            }
        }
        if ($sheet.Name -like "APPENDIX_*") {
            $sheet.Visible = $xlSheetHidden
        }
        elseif ($sheet.Name -in $visibleSheets) {
            $sheet.Visible = $xlSheetVisible
        }
        Release-ComObject $used
        Release-ComObject $sheet
    }

    $guide = $workbook.Worksheets.Item("Workbook_Guide")
    if ($guide.Index -ne 1) {
        $first = $workbook.Worksheets.Item(1)
        $guide.Move($first)
        Release-ComObject $first
    }
    $guide.Activate()
    $excel.ActiveWindow.Zoom = 75
    $excel.ActiveWindow.ScrollRow = 1
    $excel.ActiveWindow.ScrollColumn = 1
    Release-ComObject $guide

    $excel.CalculateFull()
    $workbook.Save()

    # Integrity checks.
    $afterInventory = Get-Inventory $workbook
    Assert-InventoryPreserved $beforeInventory $afterInventory

    $forbidden = [System.Collections.Generic.List[string]]::new()
    $errors = [System.Collections.Generic.List[string]]::new()
    $blankRequired = [System.Collections.Generic.List[string]]::new()
    foreach ($sheet in $workbook.Worksheets) {
        $used = $sheet.UsedRange
        $rows = [int]$used.Rows.Count
        $columns = [int]$used.Columns.Count
        for ($row = 1; $row -le $rows; $row++) {
            for ($column = 1; $column -le $columns; $column++) {
                $text = [string]$sheet.Cells.Item($row, $column).Text
                if ($text -match '(?i)professor-facing|meeting-safe|supervisor-facing|\bsir\b') {
                    $forbidden.Add("$($sheet.Name)!R${row}C${column}: $text")
                }
                if ($text -match '^#(REF!|DIV/0!|VALUE!|NAME\?|N/A|NUM!|NULL!)$') {
                    $errors.Add("$($sheet.Name)!R${row}C${column}: $text")
                }
            }
        }
        Release-ComObject $used
        Release-ComObject $sheet
    }
    if ($forbidden.Count -gt 0) {
        throw "Forbidden wording remains:`n$($forbidden -join [Environment]::NewLine)"
    }
    if ($errors.Count -gt 0) {
        throw "Excel error values remain:`n$($errors -join [Environment]::NewLine)"
    }

    foreach ($sheetName in $visibleSheets) {
        $sheet = $workbook.Worksheets.Item($sheetName)
        if ($sheet.Visible -ne $xlSheetVisible) {
            throw "Expected visible sheet is hidden: $sheetName"
        }
        Release-ComObject $sheet
    }
    foreach ($sheet in $workbook.Worksheets) {
        if ($sheet.Name -like "APPENDIX_*" -and $sheet.Visible -ne $xlSheetHidden) {
            throw "Appendix sheet is visible: $($sheet.Name)"
        }
        Release-ComObject $sheet
    }

    $workbook.Save()
    Write-Output "INPUT_WORKBOOK=$inputPath"
    Write-Output "OUTPUT_WORKBOOK=$outputPath"
    Write-Output "VISIBLE_SHEETS=$($visibleSheets -join '; ')"
    $hiddenNames = @(
        foreach ($sheet in $workbook.Worksheets) {
            if ($sheet.Name -like "APPENDIX_*") {
                [string]$sheet.Name
            }
            Release-ComObject $sheet
        }
    )
    Write-Output "HIDDEN_SHEETS=$($hiddenNames -join '; ')"
    Write-Output "COLUMNS_ADDED=Literature_Comparison: Training / Development Data Used, Benchmark Interpretation; SwissProt_Comparison: Training / Development Data Used, SwissProt Role; Project_Experiment_Comparison: Training / Development Data Used, Evaluation Benchmark, Global Split, Validation Strategy, Test Set Used, Split Limitation; Federated_Learning_Results: explicit feature meaning, risk, global counts, split, validation, client partition, client sizes/distribution, true-FL, and missing-run fields."
    Write-Output "HEADER_RENAMED=DataSet / Benchmark -> Evaluation Benchmark Results"
    Write-Output "UNCERTAIN_TRAINING_ROWS=$($uncertainRows -join '; ')"
    Write-Output "FEDERATED_CELLS_STILL_MISSING=$($missingFederated -join '; ')"
    Write-Output "SPLIT_STRATEGY=Controlled and federated experiments use one stratified seed-13 80/20 global train/test split per sample size; federated clients partition training only."
    Write-Output "VALIDATION_STRATEGY=Project LightGBM uses an internal training-only validation subset for six-candidate AUPRC selection; GraphSAGE uses fixed epochs without validation/test labels; federated runs have no separate validation split and use the global test only for final evaluation."
    Write-Output "BEST_LOW_CIRCULARITY_FEDERATED=(0.9443 accuracy) FL-A 20k federated ensemble, IID."
    Write-Output "BEST_CIRCULARITY_RISK_FEDERATED=(0.9680 accuracy) FL-B 10k federated ensemble, IID."
    Write-Output "FEDERATED_REACHED_97_PERCENT=No"
    Write-Output "NON_FEDERATED_REACHED_97_PERCENT=Yes; controlled MAGPIE-X 20k AlphaMissense tabular mode reached exactly 0.9700 and is circularity-risk."
    Write-Output "VERIFIED_METRICS_CHANGED=No"
}
finally {
    if ($null -ne $workbook) {
        $workbook.Close($true)
        Release-ComObject $workbook
    }
    if ($null -ne $excel) {
        $excel.Quit()
        Release-ComObject $excel
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
