param(
    [string]$InputWorkbook = "FYP_Literature_Results_Table_Omar_MEETING_SAFE.xlsx",
    [string]$OutputWorkbook = "FYP_Literature_Results_Table_Omar_FINAL.xlsx"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$inputPath = (Resolve-Path (Join-Path $root $InputWorkbook)).Path
$outputPath = Join-Path $root $OutputWorkbook

$oldGuideName = "README_Open_First"
$newGuideName = "Workbook_Guide"
$visibleSheetNames = @(
    $newGuideName,
    "Literature_Comparison",
    "SwissProt_Comparison",
    "Dataset_Use_Conclusion",
    "Project_Experiment_Comparison",
    "Federated_Learning_Results",
    "Metric_Definition",
    "Circularity_Explanation"
)
$appendixSheetNames = @(
    "APPENDIX_Literature_Results_Old",
    "APPENDIX_Metric_Tuple_Def_Old",
    "APPENDIX_Project_Reference_5k",
    "APPENDIX_MAGPIE_X_Size_History",
    "APPENDIX_SwissProt_HUM_Audit",
    "APPENDIX_Circularity_Ablation",
    "APPENDIX_MAGPIE_Feature_Gap",
    "APPENDIX_MAGPIE_Parity_Plan",
    "APPENDIX_MAGPIE_X_Roadmap"
)

$finalSheetNote = "Final deliverable sheet. Appendix/history sheets are hidden to keep the main workbook readable."
$guideText = "This workbook contains the final organized comparison tables for the FYP. The visible sheets contain the main deliverables: literature comparison, SwissProt comparison, dataset-use conclusion, project experiment comparison, federated learning results, metric definitions, and circularity explanation. Appendix sheets are hidden only to reduce clutter; they are retained for audit trail, previous experiment history, feature-gap planning, and reproducibility."
$interpretationNote = "The dataset columns such as ClinVar, ClinVar Rare, SwissProt/HUMSAVAR, ACMG, and ProteinGym/MAVE are evaluation benchmark columns. They do not always mean the model was trained separately on each dataset. A paper may train a model using one dataset or a combination of datasets, then report separate result tuples on different evaluation benchmarks. Therefore, each tuple should be read as: performance reported on that benchmark, not necessarily a separate model built only from that database."
$literatureClarification = "Dataset columns are evaluation benchmarks. The tuple in each cell reports the paper's performance on that benchmark when available. A dash (-) means the metric or dataset was not reported, not verified, or not applicable."
$swissClarification = "This sheet repeats the comparison from the SwissProt/HUMSAVAR perspective. Papers that did not evaluate on SwissProt/HUMSAVAR/SwissVar/UniProt are still listed only for context, and their SwissProt metric tuple remains (-, -, -, -, -, -, -, -, -, -) with a reason."
$projectClarification = "ClinVar 5k/10k/20k columns are controlled project experiments. SwissProt/HUMSAVAR is external validation and is not mixed into ClinVar training. Federated columns use simulated horizontal federated learning on partitioned ClinVar-derived training data."

$greenHeader = 0xA8D7B6
$blueBody = 0xFAE8D8
$black = 0x000000
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

function Get-CellValue {
    param(
        [object]$Values,
        [int]$Rows,
        [int]$Columns,
        [int]$Row,
        [int]$Column
    )
    if ($Rows -eq 1 -and $Columns -eq 1) {
        return $Values
    }
    return $Values[$Row, $Column]
}

function Get-ProtectedInventory {
    param([object]$Workbook)
    $inventory = [System.Collections.Generic.List[object]]::new()
    foreach ($sheet in $Workbook.Worksheets) {
        $used = $sheet.UsedRange
        $rows = [int]$used.Rows.Count
        $columns = [int]$used.Columns.Count
        $values = $used.Value2
        $formulas = $used.Formula
        for ($row = 1; $row -le $rows; $row++) {
            for ($column = 1; $column -le $columns; $column++) {
                $value = Get-CellValue $values $rows $columns $row $column
                $formula = Get-CellValue $formulas $rows $columns $row $column
                $valueText = if ($null -eq $value) { $null } else { [string]$value }
                $formulaText = if ($null -eq $formula) { $null } else { [string]$formula }
                $isFormula = $formulaText -like "=*"
                $isNumeric = (
                    $value -is [double] -or
                    $value -is [single] -or
                    $value -is [decimal] -or
                    $value -is [int] -or
                    $value -is [long]
                )
                $isTuple = (
                    $null -ne $valueText -and
                    $valueText -match '^\(\s*[^()]*,[^()]*\)$'
                )
                if ($isFormula -or $isNumeric -or $isTuple) {
                    $inventory.Add([pscustomobject]@{
                        Sheet = [string]$sheet.Name
                        Row = $row
                        Column = $column
                        Value = $valueText
                        Formula = $formulaText
                        Kind = if ($isFormula) {
                            "formula"
                        }
                        elseif ($isTuple) {
                            "tuple"
                        }
                        else {
                            "numeric"
                        }
                    })
                }
            }
        }
        Release-ComObject $used
        Release-ComObject $sheet
    }
    return $inventory
}

function Replace-NeutralTerms {
    param([string]$Text)
    if ($null -eq $Text) {
        return $null
    }
    $updated = $Text
    $updated = [regex]::Replace(
        $updated,
        '(?i)professor-facing comparison tables',
        'final FYP comparison tables'
    )
    $updated = [regex]::Replace(
        $updated,
        '(?i)professor-facing final sheet',
        'Final deliverable sheet'
    )
    $updated = [regex]::Replace(
        $updated,
        '(?i)professor-facing',
        'final'
    )
    $updated = [regex]::Replace(
        $updated,
        '(?i)meeting-safe',
        'final organized version'
    )
    $updated = [regex]::Replace(
        $updated,
        '(?i)supervisor-facing',
        'final reported'
    )
    $updated = [regex]::Replace(
        $updated,
        '(?i)\bprofessor\b',
        'academic reviewer'
    )
    $updated = [regex]::Replace(
        $updated,
        '(?i)\bsir\b',
        'academic reviewer'
    )
    return $updated
}

function Set-TopNote {
    param(
        [object]$Sheet,
        [string]$Clarification = ""
    )
    $current = [string]$Sheet.Cells.Item(1, 1).Value2
    $lines = $current -split "(`r`n|`n|`r)"
    $remaining = @(
        $lines | Where-Object {
            -not [string]::IsNullOrWhiteSpace($_) -and
            $_ -notmatch '(?i)^Professor-facing final sheet\.' -and
            $_ -notmatch '(?i)^Final deliverable sheet\.'
        }
    )
    $parts = [System.Collections.Generic.List[string]]::new()
    $parts.Add($finalSheetNote)
    if (-not [string]::IsNullOrWhiteSpace($Clarification)) {
        $parts.Add($Clarification)
    }
    foreach ($line in $remaining) {
        $neutral = Replace-NeutralTerms $line
        if (-not [string]::IsNullOrWhiteSpace($neutral)) {
            $parts.Add($neutral)
        }
    }
    $Sheet.Cells.Item(1, 1).Value2 = $parts -join [Environment]::NewLine
    $Sheet.Cells.Item(1, 1).WrapText = $true
    $Sheet.Rows.Item(1).RowHeight = 58
}

if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Force
}
Copy-Item -LiteralPath $inputPath -Destination $outputPath

$sourceExcel = $null
$sourceWorkbook = $null
$protectedInventory = $null
try {
    $sourceExcel = New-Object -ComObject Excel.Application
    $sourceExcel.Visible = $false
    $sourceExcel.DisplayAlerts = $false
    $sourceWorkbook = $sourceExcel.Workbooks.Open($inputPath, 0, $true)
    $protectedInventory = Get-ProtectedInventory $sourceWorkbook
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
$guide = $null

try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.ScreenUpdating = $false
    $workbook = $excel.Workbooks.Open($outputPath)

    $guide = $workbook.Worksheets.Item($oldGuideName)
    $guide.Name = $newGuideName
    $guide.Cells.Item(1, 1).Value2 = $finalSheetNote
    $guide.Cells.Item(1, 1).WrapText = $true
    $guide.Rows.Item(1).RowHeight = 30
    $guide.Cells.Item(3, 1).Value2 = $guideText
    $guide.Range("A3:H4").WrapText = $true

    $guide.Cells.Item(33, 1).Value2 = "Appendix/history sheets are hidden to keep the main workbook readable. They remain available through Excel's Unhide command for audit trail, previous experiment history, feature-gap planning, and reproducibility."
    $guide.Range("A33:H34").WrapText = $true

    [void]$guide.Range("A36:H36").Merge()
    $guide.Cells.Item(36, 1).Value2 = "Important interpretation note"
    [void]$guide.Range("A37:H41").Merge()
    $guide.Cells.Item(37, 1).Value2 = $interpretationNote
    $guide.Range("A36:H36").Interior.Color = $greenHeader
    $guide.Range("A36:H36").Font.Bold = $true
    $guide.Range("A37:H41").Interior.Color = $blueBody
    $guide.Range("A37:H41").WrapText = $true
    $guide.Range("A37:H41").VerticalAlignment = -4160
    Apply-Borders $guide.Range("A36:H41")
    $guide.Rows.Item(36).RowHeight = 26
    $guide.Range("A37:H41").Rows.RowHeight = 30
    $guide.PageSetup.PrintArea = '$A$1:$H$41'
    $guide.PageSetup.Orientation = $xlLandscape
    $guide.PageSetup.Zoom = $false
    $guide.PageSetup.FitToPagesWide = 1
    $guide.PageSetup.FitToPagesTall = 1

    $sheet = $workbook.Worksheets.Item("Literature_Comparison")
    Set-TopNote $sheet $literatureClarification
    Release-ComObject $sheet

    $sheet = $workbook.Worksheets.Item("SwissProt_Comparison")
    Set-TopNote $sheet $swissClarification
    Release-ComObject $sheet

    $sheet = $workbook.Worksheets.Item("Project_Experiment_Comparison")
    Set-TopNote $sheet $projectClarification
    Release-ComObject $sheet

    foreach ($sheetName in @(
        "Dataset_Use_Conclusion",
        "Federated_Learning_Results",
        "Metric_Definition",
        "Circularity_Explanation"
    )) {
        $sheet = $workbook.Worksheets.Item($sheetName)
        Set-TopNote $sheet
        Release-ComObject $sheet
    }

    # Apply required neutral replacements everywhere, including hidden appendices.
    foreach ($sheet in $workbook.Worksheets) {
        $used = $sheet.UsedRange
        $rows = [int]$used.Rows.Count
        $columns = [int]$used.Columns.Count
        $values = $used.Value2
        $formulas = $used.Formula
        for ($row = 1; $row -le $rows; $row++) {
            for ($column = 1; $column -le $columns; $column++) {
                $value = Get-CellValue $values $rows $columns $row $column
                $formula = Get-CellValue $formulas $rows $columns $row $column
                if ($null -eq $value -or [string]$formula -like "=*") {
                    continue
                }
                if ($value -isnot [string]) {
                    continue
                }
                $updated = Replace-NeutralTerms ([string]$value)
                if ($updated -cne [string]$value) {
                    $sheet.Cells.Item($row, $column).Value2 = $updated
                }
            }
        }
        Release-ComObject $used
        Release-ComObject $sheet
    }

    # Confirm the final guide stays first and sheet visibility is unchanged.
    if ($guide.Index -ne 1) {
        $firstSheet = $workbook.Worksheets.Item(1)
        $guide.Move($firstSheet)
        Release-ComObject $firstSheet
    }
    foreach ($sheetName in $visibleSheetNames) {
        $sheet = $workbook.Worksheets.Item($sheetName)
        $sheet.Visible = $xlSheetVisible
        Release-ComObject $sheet
    }
    foreach ($sheetName in $appendixSheetNames) {
        $sheet = $workbook.Worksheets.Item($sheetName)
        $sheet.Visible = $xlSheetHidden
        Release-ComObject $sheet
    }

    $guide.Activate()
    $excel.ActiveWindow.Zoom = 75
    $excel.ActiveWindow.ScrollRow = 1
    $excel.ActiveWindow.ScrollColumn = 1
    $workbook.Save()

    # Verify formulas, numeric values, and exact tuple strings.
    $protectedFailures = [System.Collections.Generic.List[string]]::new()
    foreach ($item in $protectedInventory) {
        $targetSheetName = if ($item.Sheet -eq $oldGuideName) {
            $newGuideName
        }
        else {
            $item.Sheet
        }
        $sheet = $workbook.Worksheets.Item($targetSheetName)
        $cell = $sheet.Cells.Item($item.Row, $item.Column)
        $actualValue = if ($null -eq $cell.Value2) {
            $null
        }
        else {
            [string]$cell.Value2
        }
        $actualFormula = if ($null -eq $cell.Formula) {
            $null
        }
        else {
            [string]$cell.Formula
        }
        if ($actualValue -cne $item.Value -or $actualFormula -cne $item.Formula) {
            $protectedFailures.Add(
                "$($item.Kind) changed at $targetSheetName!" +
                "R$($item.Row)C$($item.Column)"
            )
        }
        Release-ComObject $cell
        Release-ComObject $sheet
        if ($protectedFailures.Count -ge 20) {
            break
        }
    }
    if ($protectedFailures.Count -gt 0) {
        throw "Protected metric/formula check failed:`n$($protectedFailures -join [Environment]::NewLine)"
    }

    # Confirm the requested forbidden terms are absent and Excel errors are absent.
    $forbiddenMatches = [System.Collections.Generic.List[string]]::new()
    $excelErrors = [System.Collections.Generic.List[string]]::new()
    foreach ($sheet in $workbook.Worksheets) {
        $used = $sheet.UsedRange
        $rows = [int]$used.Rows.Count
        $columns = [int]$used.Columns.Count
        $values = $used.Value2
        for ($row = 1; $row -le $rows; $row++) {
            for ($column = 1; $column -le $columns; $column++) {
                $value = Get-CellValue $values $rows $columns $row $column
                $text = if ($null -eq $value) { "" } else { [string]$value }
                if ($text -match '(?i)professor-facing|meeting-safe|supervisor-facing|\bsir\b|\bprofessor\b') {
                    $forbiddenMatches.Add("$($sheet.Name)!R${row}C${column}: $text")
                }
                if ($text -match '^#(REF!|DIV/0!|VALUE!|NAME\?|N/A|NUM!|NULL!)$') {
                    $excelErrors.Add("$($sheet.Name)!R${row}C${column}: $text")
                }
            }
        }
        Release-ComObject $used
        Release-ComObject $sheet
    }
    if ($forbiddenMatches.Count -gt 0) {
        throw "Forbidden wording remains:`n$($forbiddenMatches -join [Environment]::NewLine)"
    }
    if ($excelErrors.Count -gt 0) {
        throw "Excel error values remain:`n$($excelErrors -join [Environment]::NewLine)"
    }

    foreach ($sheetName in $visibleSheetNames) {
        $sheet = $workbook.Worksheets.Item($sheetName)
        if ($sheet.Visible -ne $xlSheetVisible) {
            throw "Expected visible sheet is hidden: $sheetName"
        }
        Release-ComObject $sheet
    }
    foreach ($sheetName in $appendixSheetNames) {
        $sheet = $workbook.Worksheets.Item($sheetName)
        if ($sheet.Visible -ne $xlSheetHidden) {
            throw "Expected appendix sheet is visible: $sheetName"
        }
        Release-ComObject $sheet
    }

    $workbook.Save()

    Write-Output "OUTPUT_WORKBOOK=$outputPath"
    Write-Output "TERMS_REPLACED=professor-facing -> final deliverable/final wording; professor-facing comparison tables -> final FYP comparison tables; meeting-safe -> final organized version; supervisor-facing -> final reported; sir/professor -> neutral academic wording"
    Write-Output "VISIBLE_SHEETS=$($visibleSheetNames -join '; ')"
    Write-Output "HIDDEN_APPENDIX_SHEETS=$($appendixSheetNames -join '; ')"
    Write-Output "METRICS_CHANGED=No"
    Write-Output "FORMULAS_CHANGED=No"
    Write-Output "FORBIDDEN_TERMS_REMAINING=0"
    Write-Output "EXCEL_ERROR_VALUES=0"
}
finally {
    Release-ComObject $guide
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
