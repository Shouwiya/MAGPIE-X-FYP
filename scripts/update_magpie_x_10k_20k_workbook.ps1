param(
    [string]$WorkbookPath = "FYP_Literature_Results_Table_Omar_FORMATTED.xlsx",
    [string]$MetricsPath = "outputs/real/magpie_x_enhanced/magpie_x_10k_20k_metrics.csv"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workbookFullPath = (Resolve-Path (Join-Path $root $WorkbookPath)).Path
$metricsFullPath = (Resolve-Path (Join-Path $root $MetricsPath)).Path
$metrics = Import-Csv $metricsFullPath

$byKey = @{}
foreach ($metric in $metrics) {
    $key = "$($metric.experiment_id)_$($metric.sample_size)"
    if ($byKey.ContainsKey($key)) {
        throw "Duplicate metric row: $key"
    }
    $byKey[$key] = $metric
}
foreach ($id in @("C", "D", "E", "F")) {
    foreach ($sampleSize in @(10000, 20000)) {
        $key = "${id}_${sampleSize}"
        if (-not $byKey.ContainsKey($key)) {
            throw "Missing verified metric row: $key"
        }
    }
}

$descriptions = @{
    C = "Dataset: controlled balanced ClinVar-derived 5k, 10k, and 20k samples; exact paired seed-13 80/20 split within each size. Features: 114 clean biological features plus 86 dbNSFP prior pathogenicity/damagingness score, rank, and prediction fields (200 total). '+ dbNSFP' means expert predictor outputs are concatenated to the same rows. Risk: high circularity risk. Limitation: predictor reuse and random/same-gene overlap may inflate performance; this is not a low-circularity result."
    D = "Dataset: the same controlled ClinVar-derived 5k, 10k, and 20k samples and paired splits as experiments A-C. Features: 126 clean-plus-SpliceAI features and the same 86 dbNSFP prior predictor fields (212 total). '+ SpliceAI' means matched DS/DP columns are added to the same rows; no labels are added. Risk: high circularity risk because dbNSFP prior predictors remain. Limitation: SpliceAI coverage is incomplete and results remain random-split evidence."
    E = "Dataset: the same controlled ClinVar-derived 5k, 10k, and 20k samples and paired splits. Features: 114 clean biological features, 86 dbNSFP prior predictor fields, alphamissense_score, and alphamissense_available (202 total); SpliceAI is excluded. '+ AlphaMissense' means an hg38 expert missense score is exact-matched by chromosome, position, REF, and ALT; it is not a clinical label. Risk: high circularity risk. Limitation: AlphaMissense is missense-only."
    F = "Dataset: exact controlled ClinVar-derived 5k, 10k, and 20k samples and paired seed-13 splits. Tabular input: 114 clean + 86 dbNSFP prior predictors + 12 SpliceAI + 2 AlphaMissense fields (214 total). Graph: variant nodes with directed same-gene, same-consequence, and kNN feature-similarity edges. GraphSAGE predicts delta_gnn and applies z_final = logit(p_tabular) + delta_gnn; this is residual correction, not stacking. Training labels are visible only for training nodes; validation/test labels are hidden. Risk: high circularity and transductive graph risk."
}
$notes = @{
    C = "Verified at 5k/10k/20k on paired random splits. Do not describe this row as MAGPIE-parity or low-circularity."
    D = "SpliceAI available: 4,582/5,000, 9,161/10,000, and 18,349/20,000. Matching used no labels."
    E = "AlphaMissense matched 2,713/5,000, 5,501/10,000, and 10,864/20,000 variants. Matching used no labels or AlphaMissense class."
    F = "Graphs: 5k 5,000 nodes/100,310 edges; 10k 10,000/213,762; 20k 20,000/454,644. No validation/test labels were used for graph supervision."
}

$excel = $null
$workbook = $null
$sheet = $null
$changes = New-Object System.Collections.Generic.List[string]
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $workbook = $excel.Workbooks.Open($workbookFullPath, 0, $false)
    $sheet = $workbook.Worksheets.Item("Project_Experiment_Comparison")

    $rows = @{ C = 8; D = 9; E = 10; F = 11 }
    foreach ($id in @("C", "D", "E", "F")) {
        $row = $rows[$id]
        $storedId = [string]$sheet.Cells.Item($row, 1).Value2
        if ($storedId -ne $id) {
            throw "Expected experiment $id in A$row, found '$storedId'."
        }
        $updates = @{
            "E$row" = $descriptions[$id]
            "G$row" = $byKey["${id}_10000"].result_tuple
            "H$row" = $byKey["${id}_20000"].result_tuple
            "O$row" = $notes[$id]
        }
        foreach ($address in $updates.Keys) {
            $oldValue = [string]$sheet.Range($address).Value2
            $newValue = [string]$updates[$address]
            $sheet.Range($address).Value2 = $newValue
            $changes.Add("$address`t$oldValue`t$newValue")
        }
    }

    $workbook.Save()
    Write-Output "UPDATED_WORKBOOK=$workbookFullPath"
    foreach ($change in $changes) {
        Write-Output "CELL_CHANGE=$change"
    }
}
finally {
    if ($null -ne $sheet) {
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($sheet)
    }
    if ($null -ne $workbook) {
        $workbook.Close($false)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($workbook)
    }
    if ($null -ne $excel) {
        $excel.Quit()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
