param(
    [string]$InputWorkbook = "FYP_Literature_Results_Table_Omar_FORMATTED.xlsx",
    [string]$OutputWorkbook = "FYP_Literature_Results_Table_Omar_MEETING_SAFE.xlsx"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$inputPath = (Resolve-Path (Join-Path $root $InputWorkbook)).Path
$outputPath = Join-Path $root $OutputWorkbook

$finalSheets = @(
    "Literature_Comparison",
    "SwissProt_Comparison",
    "Dataset_Use_Conclusion",
    "Project_Experiment_Comparison",
    "Federated_Learning_Results",
    "Metric_Definition",
    "Circularity_Explanation"
)

$appendixRenameMap = [ordered]@{
    "Literature Results" = "APPENDIX_Literature_Results_Old"
    "Metric Tuple Definition" = "APPENDIX_Metric_Tuple_Def_Old"
    "Project Reference 5k" = "APPENDIX_Project_Reference_5k"
    "MAGPIE-X Sample Size Results" = "APPENDIX_MAGPIE_X_Size_History"
    "SwissProt HUMSAVAR Audit" = "APPENDIX_SwissProt_HUM_Audit"
    "Circularity Ablation" = "APPENDIX_Circularity_Ablation"
    "MAGPIE Feature Gap" = "APPENDIX_MAGPIE_Feature_Gap"
    "MAGPIE Parity Plan" = "APPENDIX_MAGPIE_Parity_Plan"
    "MAGPIE-X Staged Roadmap" = "APPENDIX_MAGPIE_X_Roadmap"
}

$professorNote = "Professor-facing final sheet. Appendix/history sheets are hidden and retained only for audit."
$readmeExplanation = "This workbook contains the final professor-facing comparison tables for the FYP. Visible sheets are final deliverables. Hidden appendix/archive sheets are retained for audit trail, older experiment history, feature-gap planning, and reproducibility. The main results should be read from Literature_Comparison, SwissProt_Comparison, Project_Experiment_Comparison, and Federated_Learning_Results."

$blueHeader = 0xE8BB9E
$blueBody = 0xFAE8D8
$greenHeader = 0xA8D7B6
$greenBody = 0xEAF7EE
$lightYellow = 0xCCFFFF
$lightRed = 0xDDD9F2
$lightGray = 0xE7E6E6
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

function Get-SheetSnapshot {
    param([object]$Sheet)
    $used = $Sheet.UsedRange
    $rows = [int]$used.Rows.Count
    $columns = [int]$used.Columns.Count
    $values = $used.Value2
    $formulas = $used.Formula
    $cells = [System.Collections.Generic.List[object]]::new()
    for ($row = 1; $row -le $rows; $row++) {
        for ($column = 1; $column -le $columns; $column++) {
            $value = if ($rows -eq 1 -and $columns -eq 1) {
                $values
            }
            else {
                $values[$row, $column]
            }
            $formula = if ($rows -eq 1 -and $columns -eq 1) {
                $formulas
            }
            else {
                $formulas[$row, $column]
            }
            $cells.Add([pscustomobject]@{
                Row = $row
                Column = $column
                Value = if ($null -eq $value) { $null } else { [string]$value }
                Formula = if ($null -eq $formula) { $null } else { [string]$formula }
            })
        }
    }
    Release-ComObject $used
    return [pscustomobject]@{
        Rows = $rows
        Columns = $columns
        Cells = $cells
    }
}

function Find-ExperimentRow {
    param(
        [object]$Sheet,
        [string]$ExperimentId
    )
    $used = $Sheet.UsedRange
    $rowCount = [int]$used.Rows.Count
    for ($row = 1; $row -le $rowCount; $row++) {
        if ([string]$Sheet.Cells.Item($row, 1).Value2 -eq $ExperimentId) {
            Release-ComObject $used
            return $row
        }
    }
    Release-ComObject $used
    throw "Experiment row '$ExperimentId' was not found."
}

function Set-FreezePane {
    param(
        [object]$Excel,
        [object]$Sheet,
        [string]$Cell
    )
    $Sheet.Visible = $xlSheetVisible
    $Sheet.Activate()
    $Excel.ActiveWindow.FreezePanes = $false
    $Sheet.Range($Cell).Select()
    $Excel.ActiveWindow.FreezePanes = $true
}

if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Force
}
Copy-Item -LiteralPath $inputPath -Destination $outputPath

$sourceExcel = $null
$sourceWorkbook = $null
$sourceSnapshots = @{}
$projectFacts = @{}

try {
    $sourceExcel = New-Object -ComObject Excel.Application
    $sourceExcel.Visible = $false
    $sourceExcel.DisplayAlerts = $false
    $sourceWorkbook = $sourceExcel.Workbooks.Open($inputPath, 0, $true)

    foreach ($sheet in $sourceWorkbook.Worksheets) {
        $sourceSnapshots[$sheet.Name] = Get-SheetSnapshot $sheet
        Release-ComObject $sheet
    }

    $projectSheet = $sourceWorkbook.Worksheets.Item("Project_Experiment_Comparison")
    $rowA = Find-ExperimentRow $projectSheet "A"
    $rowE = Find-ExperimentRow $projectSheet "E"
    $rowH = Find-ExperimentRow $projectSheet "H"
    $projectFacts = @{
        LowCircularityName = [string]$projectSheet.Cells.Item($rowA, 2).Value2
        LowCircularityTuple = [string]$projectSheet.Cells.Item($rowA, 8).Value2
        CircularityRiskName = [string]$projectSheet.Cells.Item($rowE, 2).Value2
        CircularityRiskTuple = [string]$projectSheet.Cells.Item($rowE, 8).Value2
        FederatedName = [string]$projectSheet.Cells.Item($rowH, 2).Value2
        FederatedTuple = [string]$projectSheet.Cells.Item($rowH, 11).Value2
    }
    Release-ComObject $projectSheet
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

try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.ScreenUpdating = $false
    $workbook = $excel.Workbooks.Open($outputPath)

    # Rename the retained history/audit sheets before changing visibility.
    foreach ($oldName in $appendixRenameMap.Keys) {
        $sheet = $workbook.Worksheets.Item($oldName)
        $sheet.Name = $appendixRenameMap[$oldName]
        Release-ComObject $sheet
    }

    # Add a professor-facing README as the first sheet.
    $readme = $workbook.Worksheets.Add($workbook.Worksheets.Item(1))
    $readme.Name = "README_Open_First"
    [void]$readme.Range("A1:H1").Merge()
    $readme.Cells.Item(1, 1).Value2 = $professorNote
    [void]$readme.Range("A3:H4").Merge()
    $readme.Cells.Item(3, 1).Value2 = $readmeExplanation

    $readme.Cells.Item(6, 1).Value2 = "Main task checklist"
    $readme.Cells.Item(7, 1).Value2 = "Task"
    $readme.Cells.Item(7, 2).Value2 = "Status"
    $readme.Cells.Item(7, 3).Value2 = "Read this sheet"
    $checklist = @(
        @("Literature comparison with descriptions and metric tuple", "Complete", "Literature_Comparison"),
        @("SwissProt-focused literature comparison", "Complete", "SwissProt_Comparison"),
        @("Explanation for unreported / non-applicable cells", "Complete", "Dataset_Use_Conclusion"),
        @("Project modes and sample-size comparison", "Complete", "Project_Experiment_Comparison"),
        @("Federated 5k/10k/15k/20k/25k comparison", "Complete", "Federated_Learning_Results"),
        @("Metric definitions and tuple order", "Complete", "Metric_Definition"),
        @("Circularity-risk interpretation", "Complete", "Circularity_Explanation")
    )
    for ($index = 0; $index -lt $checklist.Count; $index++) {
        $targetRow = 8 + $index
        for ($column = 1; $column -le 3; $column++) {
            $readme.Cells.Item($targetRow, $column).Value2 = $checklist[$index][$column - 1]
        }
    }

    $readme.Cells.Item(17, 1).Value2 = "Key verified results"
    $readme.Cells.Item(18, 1).Value2 = "Result category"
    $readme.Cells.Item(18, 2).Value2 = "Precisely defined mode"
    $readme.Cells.Item(18, 3).Value2 = "Metric tuple"
    $readme.Cells.Item(19, 1).Value2 = "Best low-circularity result"
    $readme.Cells.Item(19, 2).Value2 = $projectFacts.LowCircularityName + " (20k controlled ClinVar-derived split; accuracy 0.9467)"
    $readme.Cells.Item(19, 3).Value2 = $projectFacts.LowCircularityTuple
    $readme.Cells.Item(20, 1).Value2 = "Best controlled circularity-risk result"
    $readme.Cells.Item(20, 2).Value2 = $projectFacts.CircularityRiskName + " (20k; accuracy 0.9700)"
    $readme.Cells.Item(20, 3).Value2 = $projectFacts.CircularityRiskTuple
    $readme.Cells.Item(21, 1).Value2 = "Best federated result"
    $readme.Cells.Item(21, 2).Value2 = $projectFacts.FederatedName + " (10k IID probability ensemble; accuracy 0.9680)"
    $readme.Cells.Item(21, 3).Value2 = $projectFacts.FederatedTuple

    $readme.Cells.Item(23, 1).Value2 = "Important interpretation warnings"
    [void]$readme.Range("A24:H25").Merge()
    $readme.Cells.Item(24, 1).Value2 = "Warning: SwissProt/HUMSAVAR is external validation and is not mixed into ClinVar training."
    [void]$readme.Range("A27:H28").Merge()
    $readme.Cells.Item(27, 1).Value2 = "Warning: FL-B/FL-C and MAGPIE-X enhanced modes use dbNSFP prior predictor fields and are circularity-risk."
    [void]$readme.Range("A30:H31").Merge()
    $readme.Cells.Item(30, 1).Value2 = "A dash (-) means a result was not reported, not verified, not applicable, or not run with an approved matching sidecar. The relevant sheet explains the reason; no value is fabricated."
    [void]$readme.Range("A33:H34").Merge()
    $readme.Cells.Item(33, 1).Value2 = "Hidden APPENDIX_ sheets are intentionally retained as the audit trail. Use Excel's Unhide command only when older experiment history, feature planning, or reproducibility evidence is needed."

    $readmeUsed = $readme.Range("A1:H34")
    $readmeUsed.Font.Name = "Calibri"
    $readmeUsed.Font.Size = 11
    $readmeUsed.WrapText = $true
    $readmeUsed.VerticalAlignment = $xlTop
    $readme.Range("A1:H1").Interior.Color = $lightYellow
    $readme.Range("A1:H1").Font.Bold = $true
    $readme.Range("A3:H4").Interior.Color = $blueBody
    $readme.Range("A3:H4").Font.Size = 13
    foreach ($address in @("A6:C7", "A17:C18", "A23:H23")) {
        $readme.Range($address).Interior.Color = $greenHeader
        $readme.Range($address).Font.Bold = $true
    }
    $readme.Range("A8:C14").Interior.Color = $greenBody
    $readme.Range("A19:C21").Interior.Color = $blueBody
    $readme.Range("A24:H25").Interior.Color = $lightYellow
    $readme.Range("A27:H28").Interior.Color = $lightRed
    $readme.Range("A30:H31").Interior.Color = $lightGray
    $readme.Range("A33:H34").Interior.Color = $lightGray
    Apply-Borders $readme.Range("A7:C14")
    Apply-Borders $readme.Range("A18:C21")
    Apply-Borders $readme.Range("A24:H25")
    Apply-Borders $readme.Range("A27:H28")
    Apply-Borders $readme.Range("A30:H31")
    Apply-Borders $readme.Range("A33:H34")
    $readme.Columns.Item("A").ColumnWidth = 36
    $readme.Columns.Item("B").ColumnWidth = 72
    $readme.Columns.Item("C").ColumnWidth = 50
    foreach ($column in @("D", "E", "F", "G", "H")) {
        $readme.Columns.Item($column).ColumnWidth = 14
    }
    $readme.Rows.Item(1).RowHeight = 30
    $readme.Rows.Item(3).RowHeight = 36
    $readme.Rows.Item(4).RowHeight = 36
    $readme.Range("A8:C14").Rows.RowHeight = 30
    $readme.Range("A19:C21").Rows.RowHeight = 50
    $readme.Range("A24:H25").Rows.RowHeight = 28
    $readme.Range("A27:H28").Rows.RowHeight = 28
    $readme.Range("A30:H31").Rows.RowHeight = 30
    $readme.Range("A33:H34").Rows.RowHeight = 30
    $readme.PageSetup.Orientation = $xlLandscape
    $readme.PageSetup.Zoom = $false
    $readme.PageSetup.FitToPagesWide = 1
    $readme.PageSetup.FitToPagesTall = 1
    $readme.PageSetup.PrintArea = '$A$1:$H$34'
    Release-ComObject $readmeUsed

    # Add the professor-facing note without moving or replacing any data rows.
    foreach ($sheetName in $finalSheets) {
        $sheet = $workbook.Worksheets.Item($sheetName)
        $existingNote = [string]$sheet.Cells.Item(1, 1).Value2
        if ($existingNote -notlike "$professorNote*") {
            $sheet.Cells.Item(1, 1).Value2 = $professorNote + [Environment]::NewLine + $existingNote
        }
        $sheet.Cells.Item(1, 1).WrapText = $true
        $sheet.Rows.Item(1).RowHeight = [Math]::Max(40, [double]$sheet.Rows.Item(1).RowHeight)
        Release-ComObject $sheet
    }

    # Put the final professor-facing sheets first in the exact requested order.
    $orderedVisibleSheets = @("README_Open_First") + $finalSheets
    $previous = $workbook.Worksheets.Item("README_Open_First")
    for ($index = 1; $index -lt $orderedVisibleSheets.Count; $index++) {
        $sheet = $workbook.Worksheets.Item($orderedVisibleSheets[$index])
        $sheet.Move([Type]::Missing, $previous)
        Release-ComObject $previous
        $previous = $sheet
    }
    Release-ComObject $previous

    # Hide retained appendix/history sheets.
    foreach ($newName in $appendixRenameMap.Values) {
        $sheet = $workbook.Worksheets.Item($newName)
        $sheet.Visible = $xlSheetHidden
        Release-ComObject $sheet
    }

    # Ensure the visible final sheets retain useful frozen headers.
    $freezeMap = [ordered]@{
        "README_Open_First" = "A8"
        "Literature_Comparison" = "F6"
        "SwissProt_Comparison" = "F4"
        "Dataset_Use_Conclusion" = "A5"
        "Project_Experiment_Comparison" = "F6"
        "Federated_Learning_Results" = "I4"
        "Metric_Definition" = "A4"
        "Circularity_Explanation" = "A4"
    }
    foreach ($sheetName in $freezeMap.Keys) {
        $sheet = $workbook.Worksheets.Item($sheetName)
        Set-FreezePane $excel $sheet $freezeMap[$sheetName]
        Release-ComObject $sheet
    }

    $readme.Activate()
    $excel.ActiveWindow.Zoom = 75
    $excel.ActiveWindow.ScrollRow = 1
    $excel.ActiveWindow.ScrollColumn = 1
    $workbook.Save()

    # Integrity check against the source snapshot.
    $allowedTopNoteChanges = [System.Collections.Generic.HashSet[string]]::new(
        [string[]]$finalSheets
    )
    $comparisonFailures = [System.Collections.Generic.List[string]]::new()

    foreach ($sourceName in $sourceSnapshots.Keys) {
        $targetName = if ($appendixRenameMap.Contains($sourceName)) {
            $appendixRenameMap[$sourceName]
        }
        else {
            $sourceName
        }
        $sheet = $workbook.Worksheets.Item($targetName)
        $targetSnapshot = Get-SheetSnapshot $sheet
        $sourceSnapshot = $sourceSnapshots[$sourceName]

        if ($targetSnapshot.Rows -ne $sourceSnapshot.Rows -or
            $targetSnapshot.Columns -ne $sourceSnapshot.Columns) {
            $comparisonFailures.Add(
                "$sourceName dimensions changed from " +
                "$($sourceSnapshot.Rows)x$($sourceSnapshot.Columns) to " +
                "$($targetSnapshot.Rows)x$($targetSnapshot.Columns)."
            )
        }

        $cellCount = [Math]::Min(
            $sourceSnapshot.Cells.Count,
            $targetSnapshot.Cells.Count
        )
        for ($index = 0; $index -lt $cellCount; $index++) {
            $sourceCell = $sourceSnapshot.Cells[$index]
            $targetCell = $targetSnapshot.Cells[$index]
            $isAllowedNote = (
                $allowedTopNoteChanges.Contains($sourceName) -and
                $sourceCell.Row -eq 1 -and
                $sourceCell.Column -eq 1
            )
            if (-not $isAllowedNote -and (
                $sourceCell.Value -cne $targetCell.Value -or
                $sourceCell.Formula -cne $targetCell.Formula
            )) {
                $comparisonFailures.Add(
                    "$sourceName R$($sourceCell.Row)C$($sourceCell.Column) changed."
                )
                if ($comparisonFailures.Count -ge 20) {
                    break
                }
            }
        }
        Release-ComObject $sheet
        if ($comparisonFailures.Count -ge 20) {
            break
        }
    }

    if ($comparisonFailures.Count -gt 0) {
        throw (
            "Cell-value preservation check failed:`n" +
            ($comparisonFailures -join [Environment]::NewLine)
        )
    }

    # Scan all used cells for Excel error values.
    $excelErrors = [System.Collections.Generic.List[string]]::new()
    foreach ($sheet in $workbook.Worksheets) {
        $used = $sheet.UsedRange
        $rows = [int]$used.Rows.Count
        $columns = [int]$used.Columns.Count
        $values = $used.Value2
        for ($row = 1; $row -le $rows; $row++) {
            for ($column = 1; $column -le $columns; $column++) {
                $value = if ($rows -eq 1 -and $columns -eq 1) {
                    $values
                }
                else {
                    $values[$row, $column]
                }
                if ([string]$value -match '^#(REF!|DIV/0!|VALUE!|NAME\?|N/A|NUM!|NULL!)$') {
                    $excelErrors.Add("$($sheet.Name)!R${row}C${column}=$value")
                }
            }
        }
        Release-ComObject $used
        Release-ComObject $sheet
    }
    if ($excelErrors.Count -gt 0) {
        throw "Excel error values found:`n$($excelErrors -join [Environment]::NewLine)"
    }

    # Verify exact sheet order and visibility.
    for ($index = 0; $index -lt $orderedVisibleSheets.Count; $index++) {
        $sheet = $workbook.Worksheets.Item($index + 1)
        if ($sheet.Name -ne $orderedVisibleSheets[$index]) {
            throw "Unexpected sheet order at position $($index + 1): $($sheet.Name)"
        }
        if ($sheet.Visible -ne $xlSheetVisible) {
            throw "Expected visible sheet is hidden: $($sheet.Name)"
        }
        Release-ComObject $sheet
    }
    foreach ($appendixName in $appendixRenameMap.Values) {
        $sheet = $workbook.Worksheets.Item($appendixName)
        if ($sheet.Visible -ne $xlSheetHidden) {
            throw "Expected appendix sheet is not hidden: $appendixName"
        }
        Release-ComObject $sheet
    }

    $workbook.Save()

    Write-Output "OUTPUT_WORKBOOK=$outputPath"
    Write-Output "VISIBLE_SHEETS=$($orderedVisibleSheets -join '; ')"
    Write-Output "HIDDEN_APPENDIX_SHEETS=$($appendixRenameMap.Values -join '; ')"
    Write-Output "RENAMED_SHEETS=$(
        ($appendixRenameMap.GetEnumerator() | ForEach-Object {
            "$($_.Key) -> $($_.Value)"
        }) -join '; '
    )"
    Write-Output "METRIC_VALUES_CHANGED=No"
    Write-Output "EXCEL_ERROR_VALUES=0"
}
finally {
    Release-ComObject $readme
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
