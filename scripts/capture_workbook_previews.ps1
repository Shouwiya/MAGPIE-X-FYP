param(
    [string]$WorkbookPath = "FYP_Literature_Results_Table_Omar_FORMATTED.xlsx",
    [string]$OutputDirectory = "outputs/real/workbook_previews"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workbookFullPath = (Resolve-Path (Join-Path $root $WorkbookPath)).Path
$outputFullPath = Join-Path $root $OutputDirectory
New-Item -ItemType Directory -Path $outputFullPath -Force | Out-Null
$outputFullPath = (Resolve-Path $outputFullPath).Path

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32Capture {
  [StructLayout(LayoutKind.Sequential)]
  public struct RECT {
    public int Left;
    public int Top;
    public int Right;
    public int Bottom;
  }
  [DllImport("user32.dll")]
  public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
  [DllImport("user32.dll")]
  public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")]
  public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, uint nFlags);
}
"@
Add-Type -AssemblyName System.Drawing

$excel = $null
$workbook = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $true
    $excel.DisplayAlerts = $false
    $excel.WindowState = -4137
    $workbook = $excel.Workbooks.Open($workbookFullPath, 0, $true)

    $captures = @(
        @{ Sheet = "Workbook_Guide"; Zoom = 75; File = "workbook_guide_screen.png" },
        @{ Sheet = "README_Open_First"; Zoom = 75; File = "readme_open_first_screen.png" },
        @{ Sheet = "Project_Experiment_Comparison"; Zoom = 42; File = "project_screen.png" },
        @{ Sheet = "Literature_Comparison"; Zoom = 38; File = "literature_screen.png" },
        @{ Sheet = "SwissProt_Comparison"; Zoom = 40; File = "swissprot_comparison_screen.png" },
        @{ Sheet = "Dataset_Use_Conclusion"; Zoom = 75; File = "dataset_use_conclusion_screen.png" },
        @{ Sheet = "Federated_Learning_Results"; Zoom = 38; File = "federated_learning_screen.png" },
        @{ Sheet = "Metric_Definition"; Zoom = 80; File = "metric_screen.png" },
        @{ Sheet = "Circularity_Explanation"; Zoom = 65; File = "circularity_screen.png" }
    )

    foreach ($capture in $captures) {
        $sheet = $null
        try {
            $sheet = $workbook.Worksheets.Item($capture.Sheet)
        }
        catch {
            Write-Output "SKIPPED_MISSING_SHEET=$($capture.Sheet)"
            continue
        }
        $sheet.Activate()
        [void][Win32Capture]::SetForegroundWindow([IntPtr]$excel.Hwnd)
        $excel.ActiveWindow.Zoom = $capture.Zoom
        $excel.ActiveWindow.ScrollRow = 1
        $excel.ActiveWindow.ScrollColumn = 1
        Start-Sleep -Milliseconds 1000

        $rect = New-Object Win32Capture+RECT
        [void][Win32Capture]::GetWindowRect([IntPtr]$excel.Hwnd, [ref]$rect)
        $width = $rect.Right - $rect.Left
        $height = $rect.Bottom - $rect.Top
        if ($width -le 0 -or $height -le 0) {
            throw "Excel window capture returned invalid dimensions."
        }

        $bitmap = New-Object System.Drawing.Bitmap $width, $height
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $hdc = $graphics.GetHdc()
        $printed = [Win32Capture]::PrintWindow([IntPtr]$excel.Hwnd, $hdc, 2)
        $graphics.ReleaseHdc($hdc)
        if (-not $printed) {
            throw "Excel PrintWindow capture failed for $($capture.Sheet)."
        }
        $target = Join-Path $outputFullPath $capture.File
        $bitmap.Save($target, [System.Drawing.Imaging.ImageFormat]::Png)
        $graphics.Dispose()
        $bitmap.Dispose()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($sheet)
        Write-Output "CAPTURED=$target"
    }
}
finally {
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
