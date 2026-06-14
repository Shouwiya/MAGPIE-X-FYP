param(
    [string]$InputWorkbook = "FYP_Literature_Results_Table_Omar.xlsx",
    [string]$OutputWorkbook = "FYP_Literature_Results_Table_Omar_FORMATTED.xlsx",
    [string]$PreviewDirectory = "outputs/real/workbook_previews"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$inputPath = (Resolve-Path (Join-Path $root $InputWorkbook)).Path
$outputPath = Join-Path $root $OutputWorkbook
$previewPath = Join-Path $root $PreviewDirectory

$blueHeader = 0xE8BB9E
$blueBody = 0xFAE8D8
$greenHeader = 0xA8D7B6
$greenBody = 0xEAF7EE
$darkBlue = 0x804000
$darkGreen = 0x335522
$white = 0xFFFFFF
$lightYellow = 0xCCFFFF
$lightRed = 0xDDD9F2

$xlCenter = -4108
$xlLeft = -4131
$xlTop = -4160
$xlContinuous = 1
$xlThin = 2
$xlLandscape = 2
$xlTypePDF = 0
$xlScreen = 1
$xlPicture = -4147

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
    $Range.Borders.Color = 0x000000
}

function Add-HyperlinkIfValid {
    param(
        [object]$Sheet,
        [int]$Row,
        [int]$Column,
        [string]$Text
    )
    if ($Text -match "^https?://") {
        $cell = $Sheet.Cells.Item($Row, $Column)
        [void]$Sheet.Hyperlinks.Add($cell, $Text, [Type]::Missing, [Type]::Missing, $Text)
        $cell.Font.Color = 0xFF0000
        $cell.Font.Underline = $true
        Release-ComObject $cell
    }
}

function Configure-ComparisonSheet {
    param(
        [object]$Excel,
        [object]$Sheet,
        [int]$IdentityColumns,
        [int]$LastColumn,
        [int]$LastRow,
        [int]$DataStartRow
    )
    $used = $Sheet.Range($Sheet.Cells.Item(1, 1), $Sheet.Cells.Item($LastRow, $LastColumn))
    $used.Font.Name = "Calibri"
    $used.Font.Size = 10
    $used.WrapText = $true
    $used.VerticalAlignment = $xlTop
    Apply-Borders $used

    $identityHeader = $Sheet.Range($Sheet.Cells.Item(2, 1), $Sheet.Cells.Item(5, $IdentityColumns))
    $identityHeader.Interior.Color = $blueHeader
    $identityHeader.Font.Bold = $true
    $identityHeader.VerticalAlignment = $xlCenter

    $datasetHeader = $Sheet.Range($Sheet.Cells.Item(2, $IdentityColumns + 1), $Sheet.Cells.Item(5, $LastColumn))
    $datasetHeader.Interior.Color = $greenHeader
    $datasetHeader.Font.Bold = $true
    $datasetHeader.VerticalAlignment = $xlCenter

    if ($LastRow -ge $DataStartRow) {
        $identityBody = $Sheet.Range($Sheet.Cells.Item($DataStartRow, 1), $Sheet.Cells.Item($LastRow, $IdentityColumns))
        $identityBody.Interior.Color = $blueBody
        $datasetBody = $Sheet.Range($Sheet.Cells.Item($DataStartRow, $IdentityColumns + 1), $Sheet.Cells.Item($LastRow, $LastColumn))
        $datasetBody.Interior.Color = $greenBody
    }

    $note = $Sheet.Range($Sheet.Cells.Item(1, 1), $Sheet.Cells.Item(1, $LastColumn))
    $note.Merge()
    $note.Value2 = "Use '-' for unreported or unverified metrics."
    $note.Interior.Color = $lightYellow
    $note.Font.Bold = $true
    $note.HorizontalAlignment = $xlLeft
    $note.VerticalAlignment = $xlCenter
    $Sheet.Rows.Item(1).RowHeight = 24

    $group = $Sheet.Range(
        $Sheet.Cells.Item(2, $IdentityColumns + 1),
        $Sheet.Cells.Item(2, $LastColumn)
    )
    $group.Merge()
    $group.Value2 = "DataSet / Benchmark"
    $group.HorizontalAlignment = $xlCenter
    $group.VerticalAlignment = $xlCenter
    $group.Font.Bold = $true
    $group.Font.Size = 12

    $Sheet.Rows.Item(2).RowHeight = 25
    $Sheet.Rows.Item(3).RowHeight = 46
    $Sheet.Rows.Item(4).RowHeight = 82
    $Sheet.Rows.Item(5).RowHeight = 34

    $headerRow = $Sheet.Range($Sheet.Cells.Item(3, 1), $Sheet.Cells.Item($LastRow, $LastColumn))
    [void]$headerRow.AutoFilter()

    $Sheet.Activate()
    $Excel.ActiveWindow.FreezePanes = $false
    $Sheet.Cells.Item(6, $IdentityColumns + 1).Select()
    $Excel.ActiveWindow.FreezePanes = $true
    $Excel.ActiveWindow.Zoom = 70

    $Sheet.PageSetup.Orientation = $xlLandscape
    $Sheet.PageSetup.Zoom = $false
    $Sheet.PageSetup.FitToPagesWide = 1
    $Sheet.PageSetup.FitToPagesTall = $false
    $Sheet.PageSetup.CenterHorizontally = $true
    $Sheet.PageSetup.PrintTitleRows = '$1:$5'
    $Sheet.PageSetup.PrintArea = $used.Address()

    Release-ComObject $identityHeader
    Release-ComObject $datasetHeader
    Release-ComObject $used
}

function Export-RangePreview {
    param(
        [object]$Workbook,
        [object]$Sheet,
        [string]$OutputFile
    )
    $used = $Sheet.UsedRange
    $width = [Math]::Min([Math]::Max($used.Width, 800), 3200)
    $height = [Math]::Min([Math]::Max($used.Height, 500), 3600)
    $used.CopyPicture($xlScreen, $xlPicture)
    $chartObject = $Sheet.ChartObjects().Add(0, 0, $width, $height)
    $chart = $chartObject.Chart
    $chart.Paste()
    [void]$chart.Export($OutputFile, "PNG")
    $chartObject.Delete()
    Release-ComObject $chart
    Release-ComObject $chartObject
    Release-ComObject $used
}

if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Force
}
Copy-Item -LiteralPath $inputPath -Destination $outputPath
New-Item -ItemType Directory -Path $previewPath -Force | Out-Null

$excel = $null
$sourceWorkbook = $null
$workbook = $null
$createdSheets = @()
$manualVerification = [System.Collections.Generic.List[string]]::new()

try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.ScreenUpdating = $false

    $sourceWorkbook = $excel.Workbooks.Open($inputPath, 0, $true)
    $sourceSheet = $sourceWorkbook.Worksheets.Item("Literature Results")
    $sourceRange = $sourceSheet.UsedRange
    $sourceValues = $sourceRange.Value2
    $literatureRows = $sourceRange.Rows.Count - 1

    $workbook = $excel.Workbooks.Open($outputPath)

    foreach ($sheetName in @(
        "Literature_Comparison",
        "Project_Experiment_Comparison",
        "Metric_Definition",
        "Circularity_Explanation"
    )) {
        $existing = $null
        try {
            $existing = $workbook.Worksheets.Item($sheetName)
        }
        catch {
            $existing = $null
        }
        if ($null -ne $existing) {
            $existing.Delete()
            Release-ComObject $existing
        }
    }

    foreach ($sheetName in @(
        "Circularity_Explanation",
        "Metric_Definition",
        "Project_Experiment_Comparison",
        "Literature_Comparison"
    )) {
        $newSheet = $workbook.Worksheets.Add($workbook.Worksheets.Item(1))
        $newSheet.Name = $sheetName
        $createdSheets += $sheetName
        Release-ComObject $newSheet
    }
    [array]::Reverse($createdSheets)

    # Literature comparison
    $literature = $workbook.Worksheets.Item("Literature_Comparison")
    $literatureHeaders = @(
        "Paper No.",
        "Title",
        "Year",
        "Proposed Approach / Model",
        "Description",
        "ClinVar",
        "ClinVar Rare",
        "SwissProt / HUMSAVAR",
        "ACMG",
        "ProteinGym / MAVE",
        "External Validation",
        "Notes / Limitations",
        "Source Link"
    )
    for ($column = 1; $column -le $literatureHeaders.Count; $column++) {
        Set-CellValue $literature 3 $column $literatureHeaders[$column - 1]
    }
    Set-CellValue $literature 4 1 "Dataset / source links"
    Set-CellValue $literature 4 2 "-"
    Set-CellValue $literature 4 3 "-"
    Set-CellValue $literature 4 4 "-"
    Set-CellValue $literature 4 5 "Paper-specific dataset details are retained in each Description cell."
    $literatureLinks = @(
        "https://www.ncbi.nlm.nih.gov/clinvar/",
        "https://www.ncbi.nlm.nih.gov/clinvar/",
        "https://www.uniprot.org/help/humsavar",
        "https://pubmed.ncbi.nlm.nih.gov/25741868/",
        "https://proteingym.org/",
        "-",
        "-",
        "Paper-specific source links below"
    )
    for ($index = 0; $index -lt $literatureLinks.Count; $index++) {
        $column = 6 + $index
        Set-CellValue $literature 4 $column $literatureLinks[$index]
        Add-HyperlinkIfValid $literature 4 $column $literatureLinks[$index]
    }
    Set-CellValue $literature 5 1 "Sample size"
    for ($column = 2; $column -le 5; $column++) {
        Set-CellValue $literature 5 $column "-"
    }
    foreach ($column in 6..10) {
        Set-CellValue $literature 5 $column "Varies by paper; see Description"
    }
    foreach ($column in 11..13) {
        Set-CellValue $literature 5 $column "-"
    }

    for ($sourceRow = 2; $sourceRow -le $sourceRange.Rows.Count; $sourceRow++) {
        $targetRow = $sourceRow + 4
        $datasetDetails = [string]$sourceValues[$sourceRow, 6]
        $sampleDetails = [string]$sourceValues[$sourceRow, 7]
        $variantType = [string]$sourceValues[$sourceRow, 8]
        $validation = [string]$sourceValues[$sourceRow, 17]
        $risk = [string]$sourceValues[$sourceRow, 18]
        $description = (
            "{0} Dataset: {1}. Sample size: {2}. Variant type: {3}. " +
            "Validation: {4}. Circularity/leakage assessment: {5}."
        ) -f (
            [string]$sourceValues[$sourceRow, 5],
            $datasetDetails,
            $sampleDetails,
            $variantType,
            $validation,
            $risk
        )

        $values = @(
            $sourceValues[$sourceRow, 1],
            $sourceValues[$sourceRow, 2],
            $sourceValues[$sourceRow, 3],
            $sourceValues[$sourceRow, 4],
            $description,
            $sourceValues[$sourceRow, 9],
            $sourceValues[$sourceRow, 10],
            $sourceValues[$sourceRow, 11],
            $sourceValues[$sourceRow, 12],
            $sourceValues[$sourceRow, 13],
            $sourceValues[$sourceRow, 14],
            $sourceValues[$sourceRow, 16],
            $sourceValues[$sourceRow, 15]
        )
        for ($column = 1; $column -le $values.Count; $column++) {
            Set-CellValue $literature $targetRow $column $values[$column - 1]
        }
        Add-HyperlinkIfValid $literature $targetRow 13 ([string]$sourceValues[$sourceRow, 15])

        $tuples = @(
            [string]$sourceValues[$sourceRow, 9],
            [string]$sourceValues[$sourceRow, 10],
            [string]$sourceValues[$sourceRow, 11],
            [string]$sourceValues[$sourceRow, 12],
            [string]$sourceValues[$sourceRow, 13]
        )
        $notes = [string]$sourceValues[$sourceRow, 16]
        $allUnavailable = ($tuples | Where-Object { $_ -notmatch "^\(-,\s*-,\s*-,\s*-,\s*-,\s*-,\s*-,\s*-,\s*-,\s*-\)$" }).Count -eq 0
        $needsExtraction = $notes -match "(?i)requires|need(s)? .*extract|not extracted|exact .*extract|pending"
        if ($allUnavailable -or $needsExtraction) {
            $manualVerification.Add(
                ("Paper {0}: {1}" -f $sourceValues[$sourceRow, 1], $sourceValues[$sourceRow, 2])
            )
            $literature.Cells.Item($targetRow, 12).Interior.Color = $lightRed
        }
        $literature.Rows.Item($targetRow).RowHeight = 92
    }
    $literature.Columns.ColumnWidth = 18
    $literature.Columns.Item(1).ColumnWidth = 9
    $literature.Columns.Item(2).ColumnWidth = 42
    $literature.Columns.Item(3).ColumnWidth = 9
    $literature.Columns.Item(4).ColumnWidth = 28
    $literature.Columns.Item(5).ColumnWidth = 52
    foreach ($column in 6..10) {
        $literature.Columns.Item($column).ColumnWidth = 29
    }
    $literature.Columns.Item(11).ColumnWidth = 31
    $literature.Columns.Item(12).ColumnWidth = 43
    $literature.Columns.Item(13).ColumnWidth = 34
    Configure-ComparisonSheet $excel $literature 5 13 ($literatureRows + 5) 6

    # Project experiment comparison
    $project = $workbook.Worksheets.Item("Project_Experiment_Comparison")
    $projectHeaders = @(
        "Experiment No.",
        "Mode Name",
        "Year",
        "Proposed Approach / Model",
        "Description",
        "ClinVar 5k",
        "ClinVar 10k",
        "ClinVar 20k",
        "SwissProt / HUMSAVAR",
        "Federated 5k",
        "Federated 10k",
        "Federated 15k",
        "Federated 20k",
        "Federated 25k",
        "Notes / Limitations"
    )
    for ($column = 1; $column -le $projectHeaders.Count; $column++) {
        Set-CellValue $project 3 $column $projectHeaders[$column - 1]
    }
    Set-CellValue $project 4 1 "Dataset / source links"
    foreach ($column in 2..5) {
        Set-CellValue $project 4 $column "-"
    }
    foreach ($column in 6..8) {
        Set-CellValue $project 4 $column "https://www.ncbi.nlm.nih.gov/clinvar/"
        Add-HyperlinkIfValid $project 4 $column "https://www.ncbi.nlm.nih.gov/clinvar/"
    }
    Set-CellValue $project 4 9 "https://www.uniprot.org/help/humsavar"
    Add-HyperlinkIfValid $project 4 9 "https://www.uniprot.org/help/humsavar"
    foreach ($column in 10..15) {
        Set-CellValue $project 4 $column "-"
    }
    Set-CellValue $project 5 1 "Sample size"
    foreach ($column in 2..5) {
        Set-CellValue $project 5 $column "-"
    }
    $projectSampleSizes = @("5,000", "10,000", "20,000", "External validation", "5,000", "10,000", "15,000", "20,000", "25,000", "-")
    for ($index = 0; $index -lt $projectSampleSizes.Count; $index++) {
        Set-CellValue $project 5 (6 + $index) $projectSampleSizes[$index]
    }

    $fullMetrics = Import-Csv (Join-Path $root "outputs/real/magpie_x_enhanced/magpie_x_5k_full_fusion_metrics.csv")
    $largerMetrics = Import-Csv (Join-Path $root "outputs/real/magpie_parity/spliceai_10k_20k_low_circularity_metrics.csv")
    $largerEnhancedMetrics = Import-Csv (Join-Path $root "outputs/real/magpie_x_enhanced/magpie_x_10k_20k_metrics.csv")
    $metricById = @{}
    foreach ($row in $fullMetrics) {
        $metricById[$row.mode_id] = $row.metric_tuple
    }
    $largerByKey = @{}
    foreach ($row in $largerMetrics) {
        $largerByKey["$($row.mode)_$($row.sample_size)"] = $row.result_tuple
    }
    $largerEnhancedByKey = @{}
    foreach ($row in $largerEnhancedMetrics) {
        $largerEnhancedByKey["$($row.experiment_id)_$($row.sample_size)"] = $row.result_tuple
    }

    $projectRows = @(
        @{
            Id = "A"
            Name = "MAGPIE-parity low-circularity LightGBM using ANNOVAR/refGene + gnomAD AF + conservation + AA/BLOSUM"
            Model = "LightGBM binary classifier"
            Description = "Dataset: controlled balanced ClinVar-derived 5k, 10k, and 20k samples; paired 80/20 split within each sample. Features: ANNOVAR/refGene consequence, gnomAD allele frequency, conservation, variant type/length, amino-acid physicochemical deltas, and BLOSUM62. '+' is not applicable. Risk: low-circularity; 86 dbNSFP prior predictors, AlphaMissense, SpliceAI, and graph outputs are excluded. Limitation: random split and partial MAGPIE parity; gene-held-out and external validation remain pending."
            Five = $metricById["A"]
            Ten = $largerByKey["aa_blosum_low_circularity_10000"]
            Twenty = $largerByKey["aa_blosum_low_circularity_20000"]
            Notes = "Low-circularity reference. SwissProt/HUMSAVAR is external validation and is not mixed into this training."
        },
        @{
            Id = "B"
            Name = "MAGPIE-parity low-circularity LightGBM with SpliceAI DS/DP feature fusion"
            Model = "LightGBM binary classifier"
            Description = "Dataset: the same controlled ClinVar-derived samples and paired splits as experiment A. Features: all clean A features plus 12 exact-matched SpliceAI DS/DP, maximum-score, availability, record-count, and source-type columns. '+ SpliceAI' means columns are added to the same variant rows; no new clinical labels or dataset are introduced. Risk: low-circularity. Limitation: SpliceAI coverage is incomplete and its 5k improvement was not consistently confirmed at 10k/20k."
            Five = $metricById["B"]
            Ten = $largerByKey["aa_blosum_spliceai_low_circularity_10000"]
            Twenty = $largerByKey["aa_blosum_spliceai_low_circularity_20000"]
            Notes = "Real SpliceAI scores: 4,582/5,000, 9,161/10,000, and 18,349/20,000. Matching used no labels."
        },
        @{
            Id = "C"
            Name = "MAGPIE-X circularity-risk LightGBM using clean features + dbNSFP prior predictor fields"
            Model = "LightGBM binary classifier"
            Description = "Dataset: controlled balanced ClinVar-derived 5k, 10k, and 20k samples; exact paired seed-13 80/20 split within each size. Features: 114 clean biological features plus 86 dbNSFP prior pathogenicity/damagingness score, rank, and prediction fields (200 total). '+ dbNSFP' means expert predictor outputs are concatenated to the same rows. Risk: high circularity risk. Limitation: predictor reuse and random/same-gene overlap may inflate performance; this is not a low-circularity result."
            Five = $metricById["C"]
            Ten = $largerEnhancedByKey["C_10000"]
            Twenty = $largerEnhancedByKey["C_20000"]
            Notes = "Verified at 5k/10k/20k on paired random splits. Do not describe this row as MAGPIE-parity or low-circularity."
        },
        @{
            Id = "D"
            Name = "MAGPIE-X circularity-risk LightGBM using clean features + dbNSFP prior predictor fields + SpliceAI DS/DP"
            Model = "LightGBM binary classifier"
            Description = "Dataset: the same controlled ClinVar-derived 5k, 10k, and 20k samples and paired splits as experiments A-C. Features: 126 clean-plus-SpliceAI features and the same 86 dbNSFP prior predictor fields (212 total). '+ SpliceAI' means matched DS/DP columns are added to the same rows; no labels are added. Risk: high circularity risk because dbNSFP prior predictors remain. Limitation: SpliceAI coverage is incomplete and results remain random-split evidence."
            Five = $metricById["D"]
            Ten = $largerEnhancedByKey["D_10000"]
            Twenty = $largerEnhancedByKey["D_20000"]
            Notes = "SpliceAI available: 4,582/5,000, 9,161/10,000, and 18,349/20,000. Matching used no labels."
        },
        @{
            Id = "E"
            Name = "MAGPIE-X circularity-risk LightGBM using clean features + dbNSFP prior predictor fields + AlphaMissense"
            Model = "LightGBM binary classifier"
            Description = "Dataset: the same controlled ClinVar-derived 5k, 10k, and 20k samples and paired splits. Features: 114 clean biological features, 86 dbNSFP prior predictor fields, alphamissense_score, and alphamissense_available (202 total); SpliceAI is excluded. '+ AlphaMissense' means an hg38 expert missense score is exact-matched by chromosome, position, REF, and ALT; it is not a clinical label. Risk: high circularity risk. Limitation: AlphaMissense is missense-only."
            Five = $metricById["E"]
            Ten = $largerEnhancedByKey["E_10000"]
            Twenty = $largerEnhancedByKey["E_20000"]
            Notes = "AlphaMissense matched 2,713/5,000, 5,501/10,000, and 10,864/20,000 variants. Matching used no labels or AlphaMissense class."
        },
        @{
            Id = "F"
            Name = "MAGPIE-X same-split two-layer GraphSAGE residual logit correction"
            Model = "LightGBM + two-layer GraphSAGE residual correction"
            Description = "Dataset: exact controlled ClinVar-derived 5k, 10k, and 20k samples and paired seed-13 splits. Tabular input: 114 clean + 86 dbNSFP prior predictors + 12 SpliceAI + 2 AlphaMissense fields (214 total). Graph: variant nodes with directed same-gene, same-consequence, and kNN feature-similarity edges. GraphSAGE predicts delta_gnn and applies z_final = logit(p_tabular) + delta_gnn; this is residual correction, not stacking. Training labels are visible only for training nodes; validation/test labels are hidden. Risk: high circularity and transductive graph risk."
            Five = $metricById["G"]
            Ten = $largerEnhancedByKey["F_10000"]
            Twenty = $largerEnhancedByKey["F_20000"]
            Notes = "Graphs: 5k 5,000 nodes/100,310 edges; 10k 10,000/213,762; 20k 20,000/454,644. No validation/test labels were used for graph supervision."
        }
    )

    $targetRow = 6
    foreach ($experiment in $projectRows) {
        $values = @(
            $experiment.Id,
            $experiment.Name,
            "2026",
            $experiment.Model,
            $experiment.Description,
            $experiment.Five,
            $experiment.Ten,
            $experiment.Twenty,
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            $experiment.Notes
        )
        for ($column = 1; $column -le $values.Count; $column++) {
            Set-CellValue $project $targetRow $column $values[$column - 1]
        }
        $project.Rows.Item($targetRow).RowHeight = 118
        $targetRow++
    }
    $project.Columns.ColumnWidth = 18
    $project.Columns.Item(1).ColumnWidth = 11
    $project.Columns.Item(2).ColumnWidth = 45
    $project.Columns.Item(3).ColumnWidth = 9
    $project.Columns.Item(4).ColumnWidth = 32
    $project.Columns.Item(5).ColumnWidth = 62
    foreach ($column in 6..14) {
        $project.Columns.Item($column).ColumnWidth = 29
    }
    $project.Columns.Item(15).ColumnWidth = 48
    Configure-ComparisonSheet $excel $project 5 15 11 6

    # Metric definition
    $metric = $workbook.Worksheets.Item("Metric_Definition")
    $metric.Range("A1:D1").Merge()
    $metric.Range("A1").Value2 = "Use '-' for unreported or unverified metrics."
    $metric.Range("A1:D1").Interior.Color = $lightYellow
    $metric.Range("A1:D1").Font.Bold = $true
    $metric.Range("A2:D2").Merge()
    $metric.Range("A2").Value2 = "Metric tuple = (Accuracy, F1-score, Precision, Recall/Sensitivity, Specificity, MCC, AUROC, AUPRC, Brier, ECE)"
    $metric.Range("A2:D2").Interior.Color = $greenHeader
    $metric.Range("A2:D2").Font.Bold = $true
    $metric.Range("A3:D3").Value2 = @("Metric", "Definition", "Direction", "Tuple position")
    $metric.Range("A3:A13").Interior.Color = $blueHeader
    $metric.Range("B3:D13").Interior.Color = $greenBody
    $metric.Range("A3:D3").Font.Bold = $true
    $metricRows = @(
        @("Accuracy", "Fraction of all benign/pathogenic predictions that are correct.", "Higher is better", "1"),
        @("F1-score", "Harmonic mean of precision and recall for the pathogenic class.", "Higher is better", "2"),
        @("Precision", "Fraction of predicted pathogenic variants that are truly pathogenic.", "Higher is better", "3"),
        @("Recall/Sensitivity", "Fraction of truly pathogenic variants correctly identified.", "Higher is better", "4"),
        @("Specificity", "Fraction of truly benign variants correctly rejected.", "Higher is better", "5"),
        @("MCC", "Balanced correlation between predictions and labels across all confusion-matrix cells.", "Higher is better", "6"),
        @("AUROC", "Area under the receiver operating characteristic curve across thresholds.", "Higher is better", "7"),
        @("AUPRC", "Area under the precision-recall curve across thresholds.", "Higher is better", "8"),
        @("Brier", "Mean squared error between predicted probabilities and binary outcomes.", "Lower is better", "9"),
        @("ECE", "Expected calibration error between confidence and observed correctness.", "Lower is better", "10")
    )
    for ($index = 0; $index -lt $metricRows.Count; $index++) {
        $row = 4 + $index
        for ($column = 1; $column -le 4; $column++) {
            Set-CellValue $metric $row $column $metricRows[$index][$column - 1]
        }
        $metric.Rows.Item($row).RowHeight = 34
    }
    $metric.Columns.Item(1).ColumnWidth = 24
    $metric.Columns.Item(2).ColumnWidth = 68
    $metric.Columns.Item(3).ColumnWidth = 18
    $metric.Columns.Item(4).ColumnWidth = 15
    $metric.Range("A1:D13").WrapText = $true
    $metric.Range("A1:D13").VerticalAlignment = $xlTop
    Apply-Borders $metric.Range("A1:D13")
    [void]$metric.Range("A3:D13").AutoFilter()
    $metric.Activate()
    $excel.ActiveWindow.FreezePanes = $false
    $metric.Range("B4").Select()
    $excel.ActiveWindow.FreezePanes = $true
    $excel.ActiveWindow.Zoom = 90

    # Circularity explanation
    $circularity = $workbook.Worksheets.Item("Circularity_Explanation")
    $circularity.Range("A1:D1").Merge()
    $circularity.Range("A1").Value2 = "Use '-' for unreported or unverified metrics."
    $circularity.Range("A1:D1").Interior.Color = $lightYellow
    $circularity.Range("A1:D1").Font.Bold = $true
    $circularity.Range("A2:D2").Merge()
    $circularity.Range("A2").Value2 = "Circularity and evidence-source explanation"
    $circularity.Range("A2:D2").Interior.Color = $greenHeader
    $circularity.Range("A2:D2").Font.Bold = $true
    $circularity.Range("A3:D3").Value2 = @("Evidence / mode", "Definition", "Risk level", "Reporting rule")
    $circularity.Range("A3:A9").Interior.Color = $blueHeader
    $circularity.Range("B3:D9").Interior.Color = $greenBody
    $circularity.Range("A3:D3").Font.Bold = $true
    $circularityRows = @(
        @("Low-circularity mode", "Excludes prior pathogenicity predictors and uses clean biological annotations.", "Lower", "Report as MAGPIE-parity-current, not exact MAGPIE reproduction."),
        @("Circularity-risk MAGPIE-X mode", "Includes dbNSFP predictor fields such as SIFT, PolyPhen, REVEL, CADD, MutationTaster, MetaSVM, MetaLR, FATHMM, PROVEAN, and M-CAP.", "High", "Report separately from low-circularity modes."),
        @("SpliceAI", "Biological splicing-effect feature source matched to the same variants; it is not a clinical label.", "Mechanistic feature", "State DS/DP feature fusion and matching coverage."),
        @("AlphaMissense", "External expert missense predictor feature matched by hg38 chrom, pos, REF, and ALT; it is not a clinical label.", "Expert-predictor risk", "State missense-only coverage and do not call it a label."),
        @("GraphSAGE", "Graph-derived same-split residual logit correction using training-node labels only.", "Transductive graph risk", "Use only when sample/split alignment and hidden test labels are verified."),
        @("SwissProt / HUMSAVAR", "External validation resource used to test generalization of ClinVar-trained models.", "External evidence", "Do not mix it into the controlled ClinVar 5k training sample.")
    )
    for ($index = 0; $index -lt $circularityRows.Count; $index++) {
        $row = 4 + $index
        for ($column = 1; $column -le 4; $column++) {
            Set-CellValue $circularity $row $column $circularityRows[$index][$column - 1]
        }
        $circularity.Rows.Item($row).RowHeight = 60
    }
    $circularity.Columns.Item(1).ColumnWidth = 31
    $circularity.Columns.Item(2).ColumnWidth = 75
    $circularity.Columns.Item(3).ColumnWidth = 24
    $circularity.Columns.Item(4).ColumnWidth = 52
    $circularity.Range("A1:D9").WrapText = $true
    $circularity.Range("A1:D9").VerticalAlignment = $xlTop
    Apply-Borders $circularity.Range("A1:D9")
    [void]$circularity.Range("A3:D9").AutoFilter()
    $circularity.Activate()
    $excel.ActiveWindow.FreezePanes = $false
    $circularity.Range("B4").Select()
    $excel.ActiveWindow.FreezePanes = $true
    $excel.ActiveWindow.Zoom = 85

    foreach ($sheetName in @(
        "Literature_Comparison",
        "Project_Experiment_Comparison",
        "Metric_Definition",
        "Circularity_Explanation"
    )) {
        $sheet = $workbook.Worksheets.Item($sheetName)
        $sheet.Cells.HorizontalAlignment = $xlLeft
        $sheet.Cells.VerticalAlignment = $xlTop
        $sheet.Rows.Item(3).VerticalAlignment = $xlCenter
        $sheet.Rows.Item(3).Font.Bold = $true
        Release-ComObject $sheet
    }

    $workbook.Worksheets.Item("Literature_Comparison").Activate()
    $workbook.Save()

    foreach ($sheetName in @(
        "Literature_Comparison",
        "Project_Experiment_Comparison",
        "Metric_Definition",
        "Circularity_Explanation"
    )) {
        $sheet = $workbook.Worksheets.Item($sheetName)
        $safeName = $sheetName.ToLowerInvariant()
        Export-RangePreview $workbook $sheet (Join-Path $previewPath "$safeName.png")
        Release-ComObject $sheet
    }

    Write-Output "OUTPUT_FILE=$outputPath"
    Write-Output ("SHEETS_CREATED=" + (@(
        "Literature_Comparison",
        "Project_Experiment_Comparison",
        "Metric_Definition",
        "Circularity_Explanation"
    ) -join ", "))
    Write-Output "LITERATURE_ROWS=$literatureRows"
    Write-Output "PROJECT_EXPERIMENT_ROWS=$($projectRows.Count)"
    if ($manualVerification.Count -eq 0) {
        Write-Output "MANUAL_METRIC_VERIFICATION=None"
    }
    else {
        Write-Output "MANUAL_METRIC_VERIFICATION_COUNT=$($manualVerification.Count)"
        foreach ($item in $manualVerification) {
            Write-Output "MANUAL_METRIC_VERIFICATION=$item"
        }
    }
}
finally {
    if ($null -ne $workbook) {
        $workbook.Close($true)
    }
    if ($null -ne $sourceWorkbook) {
        $sourceWorkbook.Close($false)
    }
    if ($null -ne $excel) {
        $excel.Quit()
    }
    Release-ComObject $sourceRange
    Release-ComObject $sourceSheet
    Release-ComObject $workbook
    Release-ComObject $sourceWorkbook
    Release-ComObject $excel
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
