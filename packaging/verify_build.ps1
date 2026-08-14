<#
.SYNOPSIS
    Proves the packaged OriginStack.exe actually works, without needing a
    human to visually inspect the pywebview window (unreliable on headless
    CI runners anyway, and unnecessary locally).

.DESCRIPTION
    src/webview.py's ThreadingHTTPServer starts before webview.create_window(),
    so launching the real .exe and polling its own HTTP port proves every
    bundled import (astropy/scipy/astro_native/rawpy/tifffile/pywebview)
    actually resolved at runtime. /api/health additionally proves
    astro_native loaded rather than silently falling back to numpy --
    shipping that fallback's perf profile by accident defeats the point of
    bundling astro_native at all, so this fails hard rather than warning.

    Also runs a real stack through the dashboard's own POST /api/start with
    multiple parallel workers, watching for extra GUI windows -- regression
    guard for a real bug that shipped: without multiprocessing.freeze_support()
    as the first statement in desktop_app.py's __main__ guard, Phase 1's
    ProcessPoolExecutor 'spawn' workers each re-executed the frozen exe from
    scratch, opening one full GUI window per CPU core (observed in the wild:
    a 12-core machine opened 12 windows). Worker *processes* are expected and
    fine -- the check is for extra visible *windows*, i.e. MainWindowTitle.
#>
param(
    [string]$ExePath = "$PSScriptRoot\dist\OriginStack\OriginStack.exe",
    [string]$PythonExe = "$PSScriptRoot\.build_venv\Scripts\python.exe"
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ExePath)) { throw "Build did not produce $ExePath" }

# 1. Launch the packaged exe as a real child process -- not `python -m ...`,
#    must exercise the actual frozen bootloader + bundled interpreter.
$proc = Start-Process -FilePath $ExePath -PassThru

try {
    # 2. Poll the dashboard's own port until it responds.
    $ok = $false
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri 'http://127.0.0.1:8765/' -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -eq 200) { $ok = $true; break }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $ok) {
        $log = "$env:LOCALAPPDATA\OriginStack\logs\desktop_app_crash.log"
        throw "Dashboard server never came up on :8765 -- check $log"
    }
    Write-Host "Dashboard server responded on :8765"

    # 3. Confirm astro_native (not the numpy fallback) actually loaded.
    $health = Invoke-WebRequest -Uri 'http://127.0.0.1:8765/api/health' -UseBasicParsing
    $healthObj = $health.Content | ConvertFrom-Json
    if (-not $healthObj.native) {
        throw "astro_native did not load in the packaged build -- shipping numpy fallback silently"
    }
    Write-Host "astro_native active (version $($healthObj.version))"

    # 4. Real Phase 1 multiprocessing check -- start an actual stack with
    #    multiple workers and watch for extra GUI windows the whole time it
    #    runs. Total process count rising is expected and correct (that's
    #    ProcessPoolExecutor doing its job); a second *windowed* process is
    #    the bug.
    if (-not (Test-Path $PythonExe)) { $PythonExe = "python" }
    $synthDir = "$PSScriptRoot\..\synthetic_data"
    if (-not (Test-Path $synthDir)) {
        Push-Location "$PSScriptRoot\.."
        & $PythonExe tools\create_synthetic.py
        Pop-Location
    }
    if (-not (Test-Path $synthDir)) { throw "synthetic_data was not created -- cannot run the Phase 1 check" }

    $body = @{
        directory = (Resolve-Path $synthDir).Path
        output = "$env:TEMP\originstack_verify_out.fits"
        parallel = 4
        background_extraction = $false
        denoise = $false
        star_reduce = $false
        local_contrast = $false
    } | ConvertTo-Json
    $startResp = Invoke-WebRequest -Uri 'http://127.0.0.1:8765/api/start' -Method POST `
        -Body $body -ContentType 'application/json' -UseBasicParsing
    if ($startResp.StatusCode -ne 202) { throw "POST /api/start returned $($startResp.StatusCode)" }

    # Deliberately NOT polling /events here: it's a Server-Sent-Events stream
    # that never closes on its own, and Invoke-WebRequest does not reliably
    # honor -TimeoutSec against an open streaming connection on Windows
    # PowerShell 5.1 -- it can block for minutes instead of the requested 1s,
    # hanging this whole script. A fixed wall-clock window is simpler and
    # sufficient: the synthetic 6-frame run finishes in ~8s, so 30s of
    # 1-second window-count samples comfortably covers the whole run without
    # needing a "done" signal at all.
    $maxWindows = 1  # the process we already launched
    $exeName = [System.IO.Path]::GetFileNameWithoutExtension($ExePath)
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        $windowed = Get-Process -Name $exeName -ErrorAction SilentlyContinue |
            Where-Object { $_.MainWindowTitle -ne '' }
        $wCount = ($windowed | Measure-Object).Count
        if ($wCount -gt $maxWindows) { $maxWindows = $wCount }
    }
    if ($maxWindows -gt 1) {
        throw "Phase 1 opened $maxWindows GUI windows (expected 1) -- " +
              "multiprocessing.freeze_support() regression in desktop_app.py"
    }
    Write-Host "Phase 1 multiprocessing check passed (no extra GUI windows)"
} finally {
    # 5. Graceful shutdown: kill every OriginStack process (main + any
    #    workers still winding down) and confirm the port frees up.
    Get-Process -Name ([System.IO.Path]::GetFileNameWithoutExtension($ExePath)) -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 1
$stillUp = $true
try {
    Invoke-WebRequest -Uri 'http://127.0.0.1:8765/' -UseBasicParsing -TimeoutSec 2 | Out-Null
} catch {
    $stillUp = $false
}
if ($stillUp) {
    throw "Port 8765 still responding after process kill -- server thread didn't die"
}

Write-Host "Verification passed: $ExePath"
