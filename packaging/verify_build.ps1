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
#>
param(
    [string]$ExePath = "$PSScriptRoot\dist\OriginStack\OriginStack.exe"
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
} finally {
    # 4. Graceful shutdown: kill the process and confirm the port frees up.
    if (-not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
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
