param(
    [string]$WorkbookPath = "FYP_Literature_Results_Table_Omar_FORMATTED.xlsx",
    [string]$LiteratureAuditCsv = "outputs/real/literature/literature_dataset_use_audit.csv",
    [string]$SwissProtCsv = "outputs/real/literature/swissprot_comparison_rows.csv",
    [string]$DatasetSummaryJson = "outputs/real/literature/dataset_use_summary.json",
    [string]$FederatedMetricsCsv = "outputs/real/federated/federated_5k_10k_15k_20k_25k_metrics.csv"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workbookFullPath = (Resolve-Path (Join-Path $root $WorkbookPath)).Path
$auditFullPath = (Resolve-Path (Join-Path $root $LiteratureAuditCsv)).Path
$swissFullPath = (Resolve-Path (Join-Path $root $SwissProtCsv)).Path
$summaryFullPath = (Resolve-Path (Join-Path $root $DatasetSummaryJson)).Path
$federatedFullPath = (Resolve-Path (Join-Path $root $FederatedMetricsCsv)).Path
$backupDirectory = Join-Path $root "outputs/real/workbook_backups"
$backupPath = Join-Path $backupDirectory "FYP_Literature_Results_Table_Omar_FORMATTED.pre_literature_federated_update.xlsx"

$blueHeader = 0xE8BB9E
$blueBody = 0xFAE8D8
$greenHeader = 0xA8D7B6
$greenBody = 0xEAF7EE
$lightYellow = 0xCCFFFF
$lightRed = 0xDDD9F2
$lightGray = 0xE7E6E6
$white = 0xFFFFFF
$black = 0x000000

$xlCenter = -4108
$xlLeft = -4131
$xlTop = -4160
$xlContinuous = 1
$xlThin = 2
$xlLandscape = 2
$xlCalculationAutomatic = -4105

function Release-ComObject {
    param([object]$Object)
    if ($null -ne $Object) {
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($Object)
    }
}

function Set-CellValue {
    param(
        [object]$Sheet,
        [int]$Row,
        [int]$Column,
        [object]$Value
    )
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        $Sheet.Cells.Item($Row, $Column).Value2 = "-"
    }
    else {
        $Sheet.Cells.Item($Row, $Column).Value2 = [string]$Value
    }
}

function Apply-Borders {
    param([object]$Range)
    $Range.Borders.LineStyle = $xlContinuous
    $Range.Borders.Weight = $xlThin
    $Range.Borders.Color = $black
}

function Delete-SheetIfPresent {
    param(
        [object]$Workbook,
        [string]$Name
    )
    $sheet = $null
    try {
        $sheet = $Workbook.Worksheets.Item($Name)
    }
    catch {
        $sheet = $null
    }
    if ($null -ne $sheet) {
        $sheet.Delete()
        Release-ComObject $sheet
    }
}

function Add-Hyperlink {
    param(
        [object]$Sheet,
        [int]$Row,
        [int]$Column,
        [string]$Url
    )
    if ($Url -match "^https?://") {
        $cell = $Sheet.Cells.Item($Row, $Column)
        [void]$Sheet.Hyperlinks.Add(
            $cell,
            $Url,
            [Type]::Missing,
            [Type]::Missing,
            $Url
        )
        $cell.Font.Color = 0xFF0000
        $cell.Font.Underline = $true
        Release-ComObject $cell
    }
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

function Method-DisplayName {
    param([string]$Method)
    switch ($Method) {
        "centralized" { return "Centralized pooled-data MLP baseline" }
        "local_only_average" { return "Local-only client average" }
        "fedavg" { return "FedAvg MLP weighted model averaging" }
        "fedprox" { return "FedProx MLP proximal federated optimization" }
        "federated_ensemble" { return "Federated ensemble / soft-voting probability aggregation" }
        default { return $Method }
    }
}

function Get-FederatedRow {
    param(
        [array]$Metrics,
        [int]$SampleSize,
        [string]$FeatureMode,
        [string]$SplitType,
        [string]$Method
    )
    return $Metrics | Where-Object {
        [int]$_.sample_size -eq $SampleSize -and
        $_.feature_mode -eq $FeatureMode -and
        $_.client_split_type -eq $SplitType -and
        $_.method -eq $Method
    } | Select-Object -First 1
}

function Get-BestFederatedRow {
    param(
        [array]$Metrics,
        [int]$SampleSize,
        [string]$FeatureMode,
        [string]$SplitType = ""
    )
    $rows = $Metrics | Where-Object {
        [int]$_.sample_size -eq $SampleSize -and
        $_.feature_mode -eq $FeatureMode -and
        $_.method -in @("fedavg", "fedprox", "federated_ensemble") -and
        ([string]::IsNullOrEmpty($SplitType) -or $_.client_split_type -eq $SplitType)
    }
    return $rows | Sort-Object `
        @{ Expression = { [double]$_.accuracy }; Descending = $true }, `
        @{ Expression = { [double]$_.auroc }; Descending = $true } |
        Select-Object -First 1
}

$literatureAudit = Import-Csv -LiteralPath $auditFullPath
$swissRows = Import-Csv -LiteralPath $swissFullPath
$datasetSummary = Get-Content -LiteralPath $summaryFullPath -Raw | ConvertFrom-Json
$federatedMetrics = Import-Csv -LiteralPath $federatedFullPath

New-Item -ItemType Directory -Path $backupDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $backupPath)) {
    Copy-Item -LiteralPath $workbookFullPath -Destination $backupPath
}

$excel = $null
$workbook = $null

try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.ScreenUpdating = $false
    $workbook = $excel.Workbooks.Open($workbookFullPath)
    $excel.Calculation = $xlCalculationAutomatic

    # Literature_Comparison: insert the seven requested audit columns before Notes.
    $literature = $workbook.Worksheets.Item("Literature_Comparison")
    if ([string]$literature.Cells.Item(3, 12).Value2 -ne "Uses ClinVar?") {
        $literature.Range("L:R").EntireColumn.Insert()
    }
    $literature.Range("A1:T1").UnMerge()
    [void]$literature.Range("A1:T1").Merge()
    $literature.Cells.Item(1, 1).Value2 = (
        "A dash (-) preserves unreported, unverified, or non-applicable metrics; " +
        "columns N-P explain why, without fabricating values."
    )
    $literature.Range("F2:T2").UnMerge()
    [void]$literature.Range("F2:T2").Merge()
    $literature.Cells.Item(2, 6).Value2 = "DataSet / Benchmark and Missing-Metric Audit"

    $newHeaders = @(
        "Uses ClinVar?",
        "Uses SwissProt/HUMSAVAR/SwissVar/UniProt?",
        "Reason ClinVar metric missing",
        "Reason SwissProt metric missing",
        "Reason other benchmark metric missing",
        "Metric extraction status",
        "Final comparison usefulness"
    )
    for ($i = 0; $i -lt $newHeaders.Count; $i++) {
        Set-CellValue $literature 3 (12 + $i) $newHeaders[$i]
    }
    $metadataRow4 = @(
        "Paper-level use flag",
        "Paper-level use flag",
        "Standardized reason category",
        "Standardized reason category",
        "Standardized reason category",
        "Extraction audit",
        "Project relevance"
    )
    $metadataRow5 = @(
        "Yes / No",
        "Yes / No",
        "Explains ClinVar and ClinVarRare tuple gaps",
        "Explains SwissProt-style tuple gaps",
        "Explains ACMG, ProteinGym/MAVE, or other gaps",
        "Verified partial / manual extraction / preprint",
        "High / Medium with scope note"
    )
    for ($i = 0; $i -lt $metadataRow4.Count; $i++) {
        Set-CellValue $literature 4 (12 + $i) $metadataRow4[$i]
        Set-CellValue $literature 5 (12 + $i) $metadataRow5[$i]
    }
    Set-CellValue $literature 4 11 "Paper-specific external benchmark below"
    Set-CellValue $literature 4 19 "Paper-specific limitation below"
    Set-CellValue $literature 4 20 "Paper-specific source link below"
    Set-CellValue $literature 5 11 "Varies by paper"
    Set-CellValue $literature 5 19 "See each paper row"
    Set-CellValue $literature 5 20 "See each paper row"

    foreach ($auditRow in $literatureAudit) {
        $paperNumber = [int]$auditRow."Paper No."
        $targetRow = $paperNumber + 5
        Set-CellValue $literature $targetRow 12 $auditRow."Uses ClinVar?"
        Set-CellValue $literature $targetRow 13 $auditRow."Uses SwissProt/HUMSAVAR/SwissVar/UniProt?"
        Set-CellValue $literature $targetRow 14 $auditRow."Reason ClinVar metric missing"
        Set-CellValue $literature $targetRow 15 $auditRow."Reason SwissProt metric missing"
        Set-CellValue $literature $targetRow 16 $auditRow."Reason other benchmark metric missing"
        Set-CellValue $literature $targetRow 17 $auditRow."Metric extraction status"
        Set-CellValue $literature $targetRow 18 $auditRow."Final comparison usefulness"
    }

    $literatureUsed = $literature.Range("A1:T30")
    $literatureUsed.Font.Name = "Calibri"
    $literatureUsed.Font.Size = 9
    $literatureUsed.WrapText = $true
    $literatureUsed.VerticalAlignment = $xlTop
    Apply-Borders $literatureUsed
    $literature.Range("A3:E5").Interior.Color = $blueHeader
    $literature.Range("F3:T5").Interior.Color = $greenHeader
    $literature.Range("A6:E30").Interior.Color = $blueBody
    $literature.Range("F6:T30").Interior.Color = $greenBody
    $literature.Range("A1:T1").Interior.Color = $lightYellow
    $literature.Range("A1:T3").Font.Bold = $true
    $literature.Range("A1:T3").VerticalAlignment = $xlCenter
    $literature.Columns.Item("A").ColumnWidth = 8
    $literature.Columns.Item("B").ColumnWidth = 34
    $literature.Columns.Item("C").ColumnWidth = 8
    $literature.Columns.Item("D").ColumnWidth = 28
    $literature.Columns.Item("E").ColumnWidth = 48
    foreach ($column in @("F", "G", "H", "I", "J")) {
        $literature.Columns.Item($column).ColumnWidth = 22
    }
    $literature.Columns.Item("K").ColumnWidth = 28
    $literature.Columns.Item("L").ColumnWidth = 12
    $literature.Columns.Item("M").ColumnWidth = 15
    foreach ($column in @("N", "O", "P")) {
        $literature.Columns.Item($column).ColumnWidth = 25
    }
    $literature.Columns.Item("Q").ColumnWidth = 30
    $literature.Columns.Item("R").ColumnWidth = 34
    $literature.Columns.Item("S").ColumnWidth = 38
    $literature.Columns.Item("T").ColumnWidth = 34
    $literature.Rows.Item(1).RowHeight = 30
    $literature.Rows.Item(2).RowHeight = 26
    $literature.Rows.Item(3).RowHeight = 55
    $literature.Rows.Item(4).RowHeight = 72
    $literature.Rows.Item(5).RowHeight = 64
    $literature.Range("A6:T30").Rows.RowHeight = 82
    if ($literature.AutoFilterMode) {
        $literature.AutoFilterMode = $false
    }
    [void]$literature.Range("A3:T30").AutoFilter()
    Configure-Page $excel $literature '$A$1:$T$30' "F6" 55
    Release-ComObject $literatureUsed

    # SwissProt-focused comparison sheet.
    Delete-SheetIfPresent $workbook "SwissProt_Comparison"
    $swissSheet = $workbook.Worksheets.Add(
        [Type]::Missing,
        $literature
    )
    $swissSheet.Name = "SwissProt_Comparison"
    $swissHeaders = @(
        "Paper No.",
        "Title",
        "Year",
        "Proposed Approach / Model",
        "Description",
        "Main dataset used by paper",
        "Uses ClinVar?",
        "Uses SwissProt / HUMSAVAR / SwissVar / UniProt?",
        "SwissProt dataset name used",
        "SwissProt sample size",
        "SwissProt variant type",
        "SwissProt metric tuple",
        "If SwissProt not used, reason",
        "Is this paper still useful for our project?",
        "Notes / Limitations",
        "Source Link"
    )
    [void]$swissSheet.Range("A1:P1").Merge()
    $swissSheet.Cells.Item(1, 1).Value2 = (
        "A dash (-) means the requested metric was not reported, not verified, " +
        "or not applicable; column M gives the paper-specific explanation."
    )
    [void]$swissSheet.Range("A2:P2").Merge()
    $swissSheet.Cells.Item(2, 1).Value2 = "SwissProt / HUMSAVAR / SwissVar / UniProt Literature Comparison"
    for ($column = 1; $column -le $swissHeaders.Count; $column++) {
        Set-CellValue $swissSheet 3 $column $swissHeaders[$column - 1]
    }
    $rowNumber = 4
    foreach ($row in $swissRows) {
        for ($column = 1; $column -le $swissHeaders.Count; $column++) {
            $header = $swissHeaders[$column - 1]
            Set-CellValue $swissSheet $rowNumber $column $row.$header
        }
        Add-Hyperlink $swissSheet $rowNumber 16 ([string]$row."Source Link")
        $rowNumber++
    }
    $swissLastRow = $rowNumber - 1
    $swissUsed = $swissSheet.Range("A1:P$swissLastRow")
    $swissUsed.Font.Name = "Calibri"
    $swissUsed.Font.Size = 9
    $swissUsed.WrapText = $true
    $swissUsed.VerticalAlignment = $xlTop
    Apply-Borders $swissUsed
    $swissSheet.Range("A1:P1").Interior.Color = $lightYellow
    $swissSheet.Range("A2:P3").Interior.Color = $greenHeader
    $swissSheet.Range("A2:P3").Font.Bold = $true
    $swissSheet.Range("A2:P3").VerticalAlignment = $xlCenter
    $swissSheet.Range("A4:E$swissLastRow").Interior.Color = $blueBody
    $swissSheet.Range("F4:P$swissLastRow").Interior.Color = $greenBody
    $swissSheet.Columns.Item("A").ColumnWidth = 8
    $swissSheet.Columns.Item("B").ColumnWidth = 34
    $swissSheet.Columns.Item("C").ColumnWidth = 8
    $swissSheet.Columns.Item("D").ColumnWidth = 28
    $swissSheet.Columns.Item("E").ColumnWidth = 42
    $swissSheet.Columns.Item("F").ColumnWidth = 36
    $swissSheet.Columns.Item("G").ColumnWidth = 12
    $swissSheet.Columns.Item("H").ColumnWidth = 17
    $swissSheet.Columns.Item("I").ColumnWidth = 30
    $swissSheet.Columns.Item("J").ColumnWidth = 28
    $swissSheet.Columns.Item("K").ColumnWidth = 22
    $swissSheet.Columns.Item("L").ColumnWidth = 27
    $swissSheet.Columns.Item("M").ColumnWidth = 40
    $swissSheet.Columns.Item("N").ColumnWidth = 34
    $swissSheet.Columns.Item("O").ColumnWidth = 38
    $swissSheet.Columns.Item("P").ColumnWidth = 34
    $swissSheet.Rows.Item(1).RowHeight = 30
    $swissSheet.Rows.Item(2).RowHeight = 28
    $swissSheet.Rows.Item(3).RowHeight = 55
    $swissSheet.Range("A4:P$swissLastRow").Rows.RowHeight = 86
    [void]$swissSheet.Range("A3:P$swissLastRow").AutoFilter()
    $swissPrintArea = '$A$1:$P$' + $swissLastRow
    Configure-Page $excel $swissSheet $swissPrintArea "F4" 55
    Release-ComObject $swissUsed

    # Dataset-use conclusion and empty-cell explanation.
    Delete-SheetIfPresent $workbook "Dataset_Use_Conclusion"
    $conclusionSheet = $workbook.Worksheets.Add(
        [Type]::Missing,
        $swissSheet
    )
    $conclusionSheet.Name = "Dataset_Use_Conclusion"
    [void]$conclusionSheet.Range("A1:D1").Merge()
    $conclusionSheet.Cells.Item(1, 1).Value2 = "Dataset Use and Missing-Metric Conclusion"
    [void]$conclusionSheet.Range("A2:D2").Merge()
    $conclusionSheet.Cells.Item(2, 1).Value2 = (
        "Counts are derived from the 25-paper literature audit. " +
        "No unreported metric is inferred or fabricated."
    )
    $conclusionSheet.Cells.Item(4, 1).Value2 = "Table 1: Literature coverage"
    $conclusionSheet.Cells.Item(5, 1).Value2 = "Coverage item"
    $conclusionSheet.Cells.Item(5, 2).Value2 = "Paper count"
    $coverageLabels = @(
        "Total papers",
        "Papers using ClinVar",
        "Papers using SwissProt/HUMSAVAR/SwissVar/UniProt",
        "Papers using both ClinVar and SwissProt-style validation",
        "Papers using ProteinGym/MAVE/DMS",
        "Papers using ACMG/guideline validation",
        "Papers with full metric tuple",
        "Papers with AUROC/AUPRC only",
        "Papers with no verified threshold metrics"
    )
    $coverageValues = @(
        [int]$datasetSummary.total_papers,
        [int]$datasetSummary.papers_using_clinvar,
        [int]$datasetSummary.papers_using_swissprot_style,
        [int]$datasetSummary.papers_using_both_clinvar_and_swissprot_style,
        [int]$datasetSummary.papers_using_proteingym_mave_dms,
        [int]$datasetSummary.papers_using_acmg_guideline_validation,
        [int]$datasetSummary.papers_with_full_metric_tuple,
        [int]$datasetSummary.papers_with_auroc_auprc_only,
        [int]$datasetSummary.papers_with_no_verified_threshold_metrics
    )
    for ($i = 0; $i -lt $coverageLabels.Count; $i++) {
        Set-CellValue $conclusionSheet (6 + $i) 1 $coverageLabels[$i]
        $conclusionSheet.Cells.Item(6 + $i, 2).Value2 = $coverageValues[$i]
    }
    $conclusionSheet.Cells.Item(16, 1).Value2 = "Table 2: Why cells are empty"
    $conclusionSheet.Cells.Item(17, 1).Value2 = "Reason category"
    $conclusionSheet.Cells.Item(17, 2).Value2 = "Occurrences in audit fields"
    $reasonLabels = @(
        "Dataset not used by paper",
        "Metric not reported",
        "Threshold not provided",
        "Supplementary extraction required",
        "Not applicable due to variant type",
        "External validation uses a different benchmark",
        "Paper is preprint/unverified"
    )
    $reasonValues = @(
        [int]$datasetSummary.reason_counts."Dataset not used by paper",
        [int]$datasetSummary.reason_counts."Metric not reported",
        [int]$datasetSummary.reason_counts."Threshold not provided",
        [int]$datasetSummary.reason_counts."Supplementary extraction required",
        [int]$datasetSummary.reason_counts."Not applicable due to variant type",
        [int]$datasetSummary.reason_counts."External validation uses a different benchmark",
        [int]$datasetSummary.reason_counts."Paper is preprint/unverified"
    )
    for ($i = 0; $i -lt $reasonLabels.Count; $i++) {
        Set-CellValue $conclusionSheet (18 + $i) 1 $reasonLabels[$i]
        $conclusionSheet.Cells.Item(18 + $i, 2).Value2 = $reasonValues[$i]
    }
    $conclusionSheet.Cells.Item(26, 1).Value2 = "Table 3: Final conclusion"
    [void]$conclusionSheet.Range("A27:D31").Merge()
    $conclusionSheet.Cells.Item(27, 1).Value2 = (
        "The table contains many '-' entries because variant pathogenicity papers " +
        "do not standardize their reported metrics or datasets. Most papers report " +
        "AUROC/AUPRC, while the requested tuple also requires threshold-dependent " +
        "metrics such as accuracy, F1, precision, recall, specificity, MCC, " +
        "calibration metrics, Brier, and ECE. Therefore, '-' indicates " +
        "unreported/unverified/not applicable values, not missing work."
    )
    [void]$conclusionSheet.Range("A33:D35").Merge()
    $conclusionSheet.Cells.Item(33, 1).Value2 = (
        "Metric tuple order: (Accuracy, F1-score, Precision, Recall/Sensitivity, " +
        "Specificity, MCC, AUROC, AUPRC, Brier, ECE)."
    )
    $conclusionUsed = $conclusionSheet.Range("A1:D35")
    $conclusionUsed.Font.Name = "Calibri"
    $conclusionUsed.Font.Size = 11
    $conclusionUsed.WrapText = $true
    $conclusionUsed.VerticalAlignment = $xlTop
    Apply-Borders $conclusionSheet.Range("A5:B14")
    Apply-Borders $conclusionSheet.Range("A17:B24")
    Apply-Borders $conclusionSheet.Range("A26:D31")
    $conclusionSheet.Range("A1:D1").Interior.Color = $blueHeader
    $conclusionSheet.Range("A2:D2").Interior.Color = $lightYellow
    foreach ($rangeAddress in @("A4:B5", "A16:B17", "A26:D26")) {
        $conclusionSheet.Range($rangeAddress).Interior.Color = $greenHeader
        $conclusionSheet.Range($rangeAddress).Font.Bold = $true
    }
    $conclusionSheet.Range("A6:B14").Interior.Color = $greenBody
    $conclusionSheet.Range("A18:B24").Interior.Color = $greenBody
    $conclusionSheet.Range("A27:D31").Interior.Color = $blueBody
    $conclusionSheet.Range("A33:D35").Interior.Color = $lightGray
    $conclusionSheet.Range("A1:D1").Font.Bold = $true
    $conclusionSheet.Range("A1:D1").Font.Size = 16
    $conclusionSheet.Columns.Item("A").ColumnWidth = 54
    $conclusionSheet.Columns.Item("B").ColumnWidth = 22
    $conclusionSheet.Columns.Item("C").ColumnWidth = 28
    $conclusionSheet.Columns.Item("D").ColumnWidth = 28
    $conclusionSheet.Rows.Item(1).RowHeight = 30
    $conclusionSheet.Rows.Item(2).RowHeight = 35
    $conclusionSheet.Range("A27:D31").Rows.RowHeight = 34
    Configure-Page $excel $conclusionSheet '$A$1:$D$35' "A5" 80
    Release-ComObject $conclusionUsed

    # Federated_Learning_Results: one comparison row per size/mode/split.
    Delete-SheetIfPresent $workbook "Federated_Learning_Results"
    $federatedSheet = $workbook.Worksheets.Add(
        [Type]::Missing,
        $conclusionSheet
    )
    $federatedSheet.Name = "Federated_Learning_Results"
    $federatedHeaders = @(
        "Experiment No.",
        "Sample Size",
        "Feature Mode",
        "Client Count",
        "Client Split Type",
        "Federated Method",
        "Aggregation Type",
        "Description",
        "Centralized Baseline Tuple",
        "Local-only Tuple",
        "FedAvg Tuple",
        "FedProx Tuple",
        "Federated Ensemble Tuple",
        "Best Federated Method",
        "Notes / Limitations"
    )
    [void]$federatedSheet.Range("A1:O1").Merge()
    $federatedSheet.Cells.Item(1, 1).Value2 = (
        "Simulated horizontal federated learning on partitioned ClinVar-derived " +
        "training data; SwissProt/HUMSAVAR is not used in training."
    )
    [void]$federatedSheet.Range("A2:O2").Merge()
    $federatedSheet.Cells.Item(2, 1).Value2 = "Federated Learning Results by Sample Size, Feature Mode, and Client Split"
    for ($column = 1; $column -le $federatedHeaders.Count; $column++) {
        Set-CellValue $federatedSheet 3 $column $federatedHeaders[$column - 1]
    }
    $modeDescriptions = @{
        "FL-A" = "MAGPIE-parity low-circularity: ANNOVAR/refGene + gnomAD AF + conservation + variant type/length + AA/BLOSUM62 (114 features)."
        "FL-B" = "MAGPIE-X circularity-risk: FL-A + 86 dbNSFP prior predictor fields (200 features)."
        "FL-C" = "MAGPIE-X circularity-risk: FL-B + 12 exact-matched SpliceAI DS/DP aggregate fields (212 features)."
    }
    $experimentNumber = 1
    $federatedRow = 4
    foreach ($sampleSize in @(5000, 10000, 15000, 20000, 25000)) {
        foreach ($mode in @("FL-A", "FL-B", "FL-C")) {
            foreach ($splitType in @("iid", "non_iid")) {
                $central = Get-FederatedRow $federatedMetrics $sampleSize $mode $splitType "centralized"
                $local = Get-FederatedRow $federatedMetrics $sampleSize $mode $splitType "local_only_average"
                $fedAvg = Get-FederatedRow $federatedMetrics $sampleSize $mode $splitType "fedavg"
                $fedProx = Get-FederatedRow $federatedMetrics $sampleSize $mode $splitType "fedprox"
                $ensemble = Get-FederatedRow $federatedMetrics $sampleSize $mode $splitType "federated_ensemble"

                Set-CellValue $federatedSheet $federatedRow 1 ("FL-{0:D3}" -f $experimentNumber)
                Set-CellValue $federatedSheet $federatedRow 2 $sampleSize
                Set-CellValue $federatedSheet $federatedRow 3 $mode
                Set-CellValue $federatedSheet $federatedRow 4 5
                Set-CellValue $federatedSheet $federatedRow 5 $splitType
                Set-CellValue $federatedSheet $federatedRow 6 (
                    "Centralized control; local-only; FedAvg MLP; FedProx MLP; " +
                    "tree probability ensemble"
                )
                Set-CellValue $federatedSheet $federatedRow 7 (
                    "FedAvg/FedProx: client-sample-weighted neural parameters; " +
                    "ensemble: weighted predicted probabilities"
                )
                Set-CellValue $federatedSheet $federatedRow 8 $modeDescriptions[$mode]

                if ($null -eq $central) {
                    foreach ($column in 9..13) {
                        Set-CellValue $federatedSheet $federatedRow $column "-"
                    }
                    Set-CellValue $federatedSheet $federatedRow 14 "-"
                    Set-CellValue $federatedSheet $federatedRow 15 (
                        "Not run: FL-C requires exact SpliceAI sidecars, which are " +
                        "available only for the approved 5k, 10k, and 20k samples."
                    )
                }
                else {
                    Set-CellValue $federatedSheet $federatedRow 9 $central.result_tuple
                    Set-CellValue $federatedSheet $federatedRow 10 $local.result_tuple
                    Set-CellValue $federatedSheet $federatedRow 11 $fedAvg.result_tuple
                    Set-CellValue $federatedSheet $federatedRow 12 $fedProx.result_tuple
                    Set-CellValue $federatedSheet $federatedRow 13 $ensemble.result_tuple
                    $best = Get-BestFederatedRow $federatedMetrics $sampleSize $mode $splitType
                    Set-CellValue $federatedSheet $federatedRow 14 (
                        "$(Method-DisplayName $best.method): $($best.result_tuple)"
                    )
                    $riskNote = if ($mode -eq "FL-A") {
                        "Low-circularity feature mode."
                    }
                    else {
                        "High circularity risk because prior predictor fields are included."
                    }
                    Set-CellValue $federatedSheet $federatedRow 15 (
                        "$riskNote Simulated on one machine; no deployment-level " +
                        "privacy, secure aggregation, or real multi-hospital claim. " +
                        "Tree aggregation is soft-voting, not FedAvg. Shared " +
                        "preprocessing is fit on the global training partition."
                    )
                }
                $experimentNumber++
                $federatedRow++
            }
        }
    }
    $federatedLastRow = $federatedRow - 1
    $federatedUsed = $federatedSheet.Range("A1:O$federatedLastRow")
    $federatedUsed.Font.Name = "Calibri"
    $federatedUsed.Font.Size = 9
    $federatedUsed.WrapText = $true
    $federatedUsed.VerticalAlignment = $xlTop
    Apply-Borders $federatedUsed
    $federatedSheet.Range("A1:O1").Interior.Color = $lightYellow
    $federatedSheet.Range("A2:O3").Interior.Color = $greenHeader
    $federatedSheet.Range("A2:O3").Font.Bold = $true
    $federatedSheet.Range("A4:H$federatedLastRow").Interior.Color = $blueBody
    $federatedSheet.Range("I4:O$federatedLastRow").Interior.Color = $greenBody
    $federatedSheet.Columns.Item("A").ColumnWidth = 12
    $federatedSheet.Columns.Item("B").ColumnWidth = 12
    $federatedSheet.Columns.Item("C").ColumnWidth = 10
    $federatedSheet.Columns.Item("D").ColumnWidth = 11
    $federatedSheet.Columns.Item("E").ColumnWidth = 13
    $federatedSheet.Columns.Item("F").ColumnWidth = 35
    $federatedSheet.Columns.Item("G").ColumnWidth = 42
    $federatedSheet.Columns.Item("H").ColumnWidth = 52
    foreach ($column in @("I", "J", "K", "L", "M")) {
        $federatedSheet.Columns.Item($column).ColumnWidth = 31
    }
    $federatedSheet.Columns.Item("N").ColumnWidth = 43
    $federatedSheet.Columns.Item("O").ColumnWidth = 46
    $federatedSheet.Rows.Item(1).RowHeight = 30
    $federatedSheet.Rows.Item(2).RowHeight = 28
    $federatedSheet.Rows.Item(3).RowHeight = 55
    $federatedSheet.Range("A4:O$federatedLastRow").Rows.RowHeight = 92
    [void]$federatedSheet.Range("A3:O$federatedLastRow").AutoFilter()
    $federatedPrintArea = '$A$1:$O$' + $federatedLastRow
    Configure-Page $excel $federatedSheet $federatedPrintArea "I4" 50
    Release-ComObject $federatedUsed

    # Project_Experiment_Comparison: add explicit federated rows G-I.
    $project = $workbook.Worksheets.Item("Project_Experiment_Comparison")
    foreach ($row in 6..11) {
        $existingNote = [string]$project.Cells.Item($row, 15).Value2
        if ($existingNote -notmatch "Federated columns are '-'") {
            Set-CellValue $project $row 15 (
                "$existingNote Federated columns are '-' because rows A-F are " +
                "centralized or graph comparisons; see rows G-I and the " +
                "Federated_Learning_Results sheet."
            )
        }
    }
    $project.Range("A12:O14").Clear()
    $projectModes = @(
        @{
            Id = "G"
            Mode = "FL-A"
            Name = "MAGPIE-parity low-circularity federated comparison"
            Model = "FedAvg/FedProx MLP and federated LightGBM probability ensemble"
            Description = (
                "Balanced ClinVar-derived samples; five simulated clients; IID and " +
                "non-IID training partitions; shared 20% global test set. Features: " +
                "ANNOVAR/refGene + gnomAD AF + conservation + variant type/length + " +
                "AA/BLOSUM62. Low-circularity."
            )
        },
        @{
            Id = "H"
            Mode = "FL-B"
            Name = "MAGPIE-X circularity-risk federated comparison with dbNSFP predictors"
            Model = "FedAvg/FedProx MLP and federated LightGBM probability ensemble"
            Description = (
                "Same federated protocol as FL-A. Features: 114 clean features + " +
                "86 dbNSFP prior predictor fields. High circularity risk; reported " +
                "separately from MAGPIE-parity low-circularity evaluation."
            )
        },
        @{
            Id = "I"
            Mode = "FL-C"
            Name = "MAGPIE-X circularity-risk federated comparison with dbNSFP + SpliceAI"
            Model = "FedAvg/FedProx MLP and federated LightGBM probability ensemble"
            Description = (
                "Same federated protocol as FL-B plus 12 exact-matched SpliceAI " +
                "aggregate fields. Available only for approved 5k, 10k, and 20k " +
                "samples with existing sidecars."
            )
        }
    )
    $projectRow = 12
    foreach ($modeInfo in $projectModes) {
        Set-CellValue $project $projectRow 1 $modeInfo.Id
        Set-CellValue $project $projectRow 2 $modeInfo.Name
        Set-CellValue $project $projectRow 3 2026
        Set-CellValue $project $projectRow 4 $modeInfo.Model
        Set-CellValue $project $projectRow 5 $modeInfo.Description
        foreach ($column in 6..9) {
            Set-CellValue $project $projectRow $column "-"
        }
        $sampleToColumn = @{
            5000 = 10
            10000 = 11
            15000 = 12
            20000 = 13
            25000 = 14
        }
        $missingSizes = [System.Collections.Generic.List[string]]::new()
        foreach ($sampleSize in @(5000, 10000, 15000, 20000, 25000)) {
            $best = Get-BestFederatedRow $federatedMetrics $sampleSize $modeInfo.Mode
            if ($null -eq $best) {
                Set-CellValue $project $projectRow $sampleToColumn[$sampleSize] "-"
                $missingSizes.Add(("{0}k" -f ($sampleSize / 1000)))
            }
            else {
                Set-CellValue $project $projectRow $sampleToColumn[$sampleSize] $best.result_tuple
            }
        }
        $note = (
            "Columns F-I are '-' because this row summarizes federated ClinVar " +
            "experiments rather than the centralized 5k/10k/20k or SwissProt columns. " +
            "Best federated tuple selects FedAvg, FedProx, or the explicitly named " +
            "probability ensemble across IID/non-IID. SwissProt/HUMSAVAR is not used in training. " +
            "Shared preprocessing is fit on the global training partition."
        )
        if ($missingSizes.Count -gt 0) {
            $note += (
                " FL-C is '-' at " + ($missingSizes -join " and ") +
                " because exact SpliceAI sidecars were not available and scores were not fabricated."
            )
        }
        Set-CellValue $project $projectRow 15 $note
        $projectRow++
    }
    $projectUsed = $project.Range("A1:O14")
    $projectUsed.WrapText = $true
    $projectUsed.VerticalAlignment = $xlTop
    Apply-Borders $projectUsed
    $project.Range("A12:E14").Interior.Color = $blueBody
    $project.Range("F12:O14").Interior.Color = $greenBody
    $project.Range("A12:O14").Rows.RowHeight = 96
    if ($project.AutoFilterMode) {
        $project.AutoFilterMode = $false
    }
    [void]$project.Range("A3:O14").AutoFilter()
    Configure-Page $excel $project '$A$1:$O$14' "F6" 55
    Release-ComObject $projectUsed

    $workbook.Save()
    $excel.CalculateFull()
    $workbook.Save()

    Write-Output "UPDATED_WORKBOOK=$workbookFullPath"
    Write-Output "BACKUP_WORKBOOK=$backupPath"
    Write-Output "LITERATURE_PAPERS=$($literatureAudit.Count)"
    Write-Output "FEDERATED_RESULT_ROWS=$($federatedMetrics.Count)"
    Write-Output "FEDERATED_SHEET_ROWS=$($federatedLastRow - 3)"
}
finally {
    foreach ($object in @(
        $project,
        $federatedSheet,
        $conclusionSheet,
        $swissSheet,
        $literature
    )) {
        Release-ComObject $object
    }
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
