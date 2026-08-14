<#
.SYNOPSIS
    End-to-end build of the OriginStack desktop app into a standalone
    Windows onedir bundle via PyInstaller.

.DESCRIPTION
    Creates a fresh packaging-only venv (never reuses a dev venv -- must not
    carry over `maturin develop`'s editable/.pth astro_native install, which
    is invisible to PyInstaller's static analysis), installs everything the
    packaged app needs (including the desktop-app extras requirements.txt
    only documents as optional), builds ext/astro_native as a real wheel,
    then invokes PyInstaller.

.PARAMETER PythonVersion
    Python version for the `py` launcher, e.g. '3.12'. Defaults to whatever
    `py` resolves to with no version flag (this machine currently only has
    3.14 installed). PyInstaller/pyo3 tooling tends to lag a few months
    behind the newest CPython release -- if the build fails against the
    default, install Python 3.12 (`py install 3.12`, or python.org) and
    rerun with `-PythonVersion 3.12`.

.PARAMETER Clean
    Delete and recreate the packaging venv instead of reusing it.
#>
param(
    [string]$PythonVersion = '',
    [switch]$Clean
)
$ErrorActionPreference = 'Stop'
$Root = Resolve-Path "$PSScriptRoot\.."
Set-Location $Root

# 1. Fresh venv.
$venvPath = "$Root\packaging\.build_venv"
if ($Clean -and (Test-Path $venvPath)) { Remove-Item -Recurse -Force $venvPath }
if (-not (Test-Path $venvPath)) {
    if ($PythonVersion) {
        & py "-$PythonVersion" -m venv $venvPath
    } else {
        & py -m venv $venvPath
    }
}
$py = "$venvPath\Scripts\python.exe"
& $py -m pip install --upgrade pip

# 2. Core + desktop-app-only deps that requirements.txt intentionally
#    documents but doesn't install (rawpy/tifffile/pywebview are commented-
#    out "optional" per requirements.txt -- the packaged exe needs all
#    three bundled for a real astrophotography workflow).
& $py -m pip install -r "$Root\requirements.txt"
& $py -m pip install "rawpy>=0.19" "tifffile>=2023.1" "pywebview>=5.0" `
                      "pyinstaller>=6.0" "pyinstaller-hooks-contrib" "maturin>=1.7,<2.0"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# 3. Build astro_native as a REAL wheel (not `maturin develop`'s editable
#    install, which PyInstaller's analysis cannot see into) and install it.
#    Deliberately do NOT set RUSTFLAGS=target-cpu=native here -- CLAUDE.md's
#    own native-build docs warn that flag produces a wheel that "may use
#    instructions older CPUs lack" and must not be redistributed.
Push-Location "$Root\ext\astro_native"
try {
    & cargo --version
    if ($LASTEXITCODE -ne 0) { throw "Rust toolchain not found (cargo --version failed)" }
    & $py -m maturin build --release
    if ($LASTEXITCODE -ne 0) { throw "maturin build failed" }
    $wheel = Get-ChildItem "target\wheels\astro_native-*.whl" | Select-Object -First 1
    if (-not $wheel) { throw "maturin build produced no wheel in target\wheels\" }
    & $py -m pip install --force-reinstall $wheel.FullName
} finally {
    Pop-Location
}

# 4. Sanity: fail loudly now if astro_native silently fell back to numpy,
#    rather than shipping the numpy-fallback perf profile by accident.
& $py -c "import astro_native; print('astro_native OK:', astro_native.__version__)"
if ($LASTEXITCODE -ne 0) { throw "astro_native failed to import after packaging install" }

# 5. Invoke PyInstaller from the packaging venv, so its own hook-discovery
#    entry-point scan sees the exact same site-packages the app will ship.
& $py -m PyInstaller "$Root\packaging\originstack.spec" `
    --distpath "$Root\packaging\dist" `
    --workpath "$Root\packaging\build" `
    --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

# 6. Zip dist/OriginStack/ (onedir) for distribution. PyInstaller's own
#    subprocess (or an AV real-time scan of the just-written .pyd/.zip
#    payload) can still hold a handle on a just-closed file for a moment --
#    retry rather than fail the whole build over a transient lock.
$version = (Get-Content "$Root\VERSION" -Raw).Trim()
$zipName = "OriginStack-$version-windows-x64.zip"
$zipPath = "$Root\packaging\dist\$zipName"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
$zipped = $false
for ($attempt = 1; $attempt -le 5; $attempt++) {
    try {
        Compress-Archive -Path "$Root\packaging\dist\OriginStack\*" -DestinationPath $zipPath -ErrorAction Stop
        $zipped = $true
        break
    } catch {
        Write-Host "Compress-Archive attempt $attempt failed ($($_.Exception.Message)) -- retrying..."
        if (Test-Path $zipPath) { Remove-Item $zipPath -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 3
    }
}
if (-not $zipped) { throw "Compress-Archive failed after 5 attempts -- a file in dist\OriginStack\ stayed locked" }

# Sanity: a onedir build this size zipping to a few hundred bytes means the
# archive is silently missing its payload (the exact failure mode a
# non-terminating Compress-Archive error inside the loop above can produce)
# -- fail loudly instead of shipping a broken zip.
$zipSizeMb = (Get-Item $zipPath).Length / 1MB
if ($zipSizeMb -lt 50) { throw "Zip suspiciously small (${zipSizeMb}MB) -- likely missing files" }
Write-Host "Built: packaging\dist\$zipName ($([math]::Round($zipSizeMb, 1)) MB)"
