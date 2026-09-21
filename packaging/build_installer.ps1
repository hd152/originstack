<#
.SYNOPSIS
    Builds the Windows installer (Setup.exe) from an existing PyInstaller build.

.DESCRIPTION
    Run build_windows.ps1 first: it produces packaging\dist\OriginStack\. This wraps that folder
    with Inno Setup (packaging\originstack.iss) into
    packaging\dist\OriginStack-<VERSION>-setup.exe, versioned from the repo's VERSION file.

    Inno Setup is a build-time tool, not a runtime dependency. Install it with
    `winget install JRSoftware.InnoSetup` (or `choco install innosetup`); this script looks for
    ISCC.exe on PATH and in the usual install folders.
#>
param(
    [string]$IsccPath = ''
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent

if (-not (Test-Path "$PSScriptRoot\dist\OriginStack\OriginStack.exe")) {
    throw "packaging\dist\OriginStack\OriginStack.exe not found -- run packaging\build_windows.ps1 first"
}

if (-not $IsccPath) {
    $candidates = @(
        (Get-Command iscc -ErrorAction SilentlyContinue | ForEach-Object { $_.Source }),
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }
    if (-not $candidates) {
        throw "Inno Setup (ISCC.exe) not found. Install it: winget install JRSoftware.InnoSetup"
    }
    $IsccPath = @($candidates)[0]
}

$version = (Get-Content "$Root\VERSION" -Raw).Trim()
Write-Host "Building OriginStack $version installer with $IsccPath"
& $IsccPath "/DAppVersion=$version" "$PSScriptRoot\originstack.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed (exit $LASTEXITCODE)" }

$setup = "$PSScriptRoot\dist\OriginStack-$version-setup.exe"
if (-not (Test-Path $setup)) { throw "Installer was not produced at $setup" }
Write-Host ("Installer: {0} ({1:N0} MB)" -f $setup, ((Get-Item $setup).Length / 1MB))
