<#
.SYNOPSIS
    Proves the packaged OriginStack.exe actually works, without needing a
    human to visually inspect the tkinter window (unreliable on headless CI
    runners anyway, and unnecessary locally).

.DESCRIPTION
    The desktop app is a native tkinter window (no local HTTP server as of
    2026-08 -- it used to be a pywebview-wrapped dashboard). This script
    proves the packaged exe works two ways:

    1. A normal launch: polls for a window with the right title to appear
       (proves every bundled import -- astropy/scipy/astro_native/rawpy/
       tifffile/tkinter -- actually resolved at runtime and the app didn't
       crash on startup), then reads the startup log
       (`desktop_app.py::_log_startup_status`) to confirm astro_native
       loaded rather than silently falling back to numpy -- shipping that
       fallback's perf profile by accident defeats the point of bundling
       astro_native at all, so this fails hard rather than warning.

    2. A `--verify-headless` launch: runs a real stack with multiple
       parallel workers through the exact same frozen entry point, watching
       for extra GUI windows -- regression guard for a real bug that
       shipped: without `multiprocessing.freeze_support()` as the first
       statement in desktop_app.py's `__main__` guard, Phase 1's
       ProcessPoolExecutor 'spawn' workers each re-executed the frozen exe
       from scratch, opening one full GUI window per CPU core (observed in
       the wild: a 12-core machine opened 12 windows). Worker *processes*
       are expected and fine -- the check is for extra visible *windows*,
       i.e. MainWindowTitle. `--verify-headless` itself opens no window (it
       skips the GUI/mainloop entirely), so any window appearing during this
       phase is the regression.
#>
param(
    [string]$ExePath = "$PSScriptRoot\dist\OriginStack\OriginStack.exe",
    [string]$PythonExe = "$PSScriptRoot\.build_venv\Scripts\python.exe"
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ExePath)) { throw "Build did not produce $ExePath" }
$exeName = [System.IO.Path]::GetFileNameWithoutExtension($ExePath)
$logPath = "$env:LOCALAPPDATA\OriginStack\logs\desktop_app.log"
if (Test-Path $logPath) { Remove-Item $logPath -Force }

# ── 1. Normal launch: window appears, native kernel loaded ────────────────
$proc = Start-Process -FilePath $ExePath -PassThru
try {
    $ok = $false
    for ($i = 0; $i -lt 30; $i++) {
        $w = Get-Process -Name $exeName -ErrorAction SilentlyContinue |
            Where-Object { $_.MainWindowTitle -ne '' }
        if ($w) { $ok = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ok) {
        throw "OriginStack window never appeared -- check $env:LOCALAPPDATA\OriginStack\logs\desktop_app_crash.log"
    }
    Write-Host "Window appeared"

    $statusOk = $false
    for ($i = 0; $i -lt 10; $i++) {
        if (Test-Path $logPath) {
            $line = Get-Content $logPath -Tail 1
            if ($line -match 'ACTIVE') { $statusOk = $true; break }
            if ($line -match 'numpy fallback') {
                throw "astro_native did not load in the packaged build -- shipping numpy fallback silently"
            }
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $statusOk) { throw "Startup log never confirmed astro_native active -- check $logPath" }
    Write-Host "astro_native active ($(Get-Content $logPath -Tail 1))"
} finally {
    Get-Process -Name $exeName -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1

# ── 2. --verify-headless: real Phase 1 multiprocessing check ──────────────
if (-not (Test-Path $PythonExe)) { $PythonExe = "python" }
$synthDir = "$PSScriptRoot\..\synthetic_data"
if (-not (Test-Path $synthDir)) {
    Push-Location "$PSScriptRoot\.."
    & $PythonExe tools\create_synthetic.py
    Pop-Location
}
if (-not (Test-Path $synthDir)) { throw "synthetic_data was not created -- cannot run the Phase 1 check" }

$outPath = "$env:TEMP\originstack_verify_out.fits"
$headlessArgs = @('--verify-headless', '-d', (Resolve-Path $synthDir).Path, '-o', $outPath,
                  '--parallel', '4', '--debayer-method', 'malvar',
                  '--white-balance', 'grayworld', '--stack-method', 'median')
$headlessProc = Start-Process -FilePath $ExePath -ArgumentList $headlessArgs -PassThru

$maxWindows = 0  # --verify-headless opens no window of its own
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if ($headlessProc.HasExited) { break }
    $windowed = Get-Process -Name $exeName -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -ne '' }
    $wCount = ($windowed | Measure-Object).Count
    if ($wCount -gt $maxWindows) { $maxWindows = $wCount }
}
if (-not $headlessProc.HasExited) {
    $headlessProc.WaitForExit(30000) | Out-Null
}
if ($maxWindows -gt 0) {
    throw "--verify-headless opened $maxWindows GUI window(s) (expected 0) -- " +
          "multiprocessing.freeze_support() regression in desktop_app.py"
}
if (-not (Test-Path $outPath)) {
    throw "--verify-headless did not produce $outPath -- run failed (check console/log output above)"
}
Write-Host "Phase 1 multiprocessing check passed (no extra GUI windows)"

# ── 3. Graceful shutdown: confirm nothing is left running ─────────────────
Get-Process -Name $exeName -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
$stillUp = Get-Process -Name $exeName -ErrorAction SilentlyContinue
if ($stillUp) {
    throw "OriginStack process(es) still running after cleanup"
}

Write-Host "Verification passed: $ExePath"
