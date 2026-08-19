# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for OriginStack's desktop app.

Build via packaging/build_windows.ps1 (which sets up the packaging venv,
builds ext/astro_native as a real wheel, and invokes PyInstaller against this
spec) rather than running `pyinstaller` directly against this file.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None
ROOT = Path(SPECPATH).parent  # packaging/ -> repo root

# Defense-in-depth collect_all() for packages whose own PyInstaller hooks
# already handle most of this. rawpy is the one *verified* real risk: it
# ships sibling DLLs (raw_r.dll, vcomp140.dll) next to its .pyd that
# PyInstaller's binary walker may or may not follow depending on version.
datas, binaries, hiddenimports = [], [], []
for pkg in ('rawpy',):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

datas += [(str(ROOT / 'VERSION'), '.')]

a = Analysis(
    [str(ROOT / 'desktop_app.py')],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    # Deliberately NOT listing numpy/astropy/scipy/tqdm/PIL/psutil/rawpy/
    # tifffile/astro_native here -- every import in src/ (including all 10
    # try/except-wrapped `import astro_native` sites) is a plain static
    # `import x` statement, which PyInstaller's AST-based modulegraph finds
    # on its own. numpy self-registers its own hook; scipy/astropy don't,
    # which is what `pyinstaller-hooks-contrib` (installed by
    # build_windows.ps1) covers -- their curated hooks handle astropy's
    # IERS/leap-second/units data files and scipy's non-statically-visible
    # compiled submodules better than a blind collect_all would.
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / 'packaging' / 'runtime_hook_stdout.py')],
    # tkinter is the desktop app's UI toolkit as of 2026-08 (replaced a
    # pywebview-wrapped local HTTP dashboard) -- no longer excluded. Only
    # its own test suite is dead weight.
    excludes=['cupy', 'tkinter.test', 'matplotlib.tests'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# astropy_iers_data (Earth-rotation/leap-second tables, ~8.5MB): nothing in
# src/ touches astropy.time.Time/EarthLocation/AltAz or any other frame that
# needs Earth-orientation parameters -- confirmed by grepping every astropy
# import in src/ (only astropy.io.fits, astropy.stats, astropy.wcs.WCS,
# astropy.coordinates.SkyCoord(frame='icrs'), astropy.units, astropy.table --
# none of which are IERS-dependent) and empirically, by running the exact
# WCS.wcs_pix2world + SkyCoord ICRS separation calls annotation.py/
# color_calibrate.py make with warnings promoted to errors and network
# access disabled: zero IERS access. pyinstaller-hooks-contrib's astropy
# hook bundles it unconditionally just because `astropy` is imported at all.
a.datas = [d for d in a.datas if 'astropy_iers_data' not in d[0]]

# TRIED TWICE AND REVERTED: numpy.libs/ and scipy.libs/ both contain a
# byte-identical libscipy_openblas64_-*.dll (numpy 2.x vendors the same
# scipy-openblas64 wheel scipy does), ~19.5MB -- looked like a free dedup.
#
# Attempt 1: filter a.binaries to drop numpy.libs' copy. Broke numpy outright
# ("DLL load failed while importing _multiarray_umath") -- numpy's own
# loader only searches its own numpy.libs/, not scipy.libs/, just because
# the bytes happen to match.
#
# Attempt 2: added a runtime hook (os.add_dll_directory + SetDllDirectoryW +
# PATH, all three) registering scipy.libs/ as a search directory before
# numpy imports. Still broke, and a debug-logged run of the actual packaged
# exe revealed why the first theory was wrong: scipy.libs/ itself was left
# holding only the *non*-64 openblas variant once a.binaries was filtered at
# all -- even though the filter only ever matched 'numpy.libs' destination
# paths. So this isn't a DLL-search-path problem: something in PyInstaller's
# own binary-collection merge silently drops the file from scipy.libs/ the
# moment a.binaries is edited post-Analysis, in a way not understood from
# here without much deeper PyInstaller-internals digging than a ~4-5MB
# compressed-zip win justifies. Left unfiltered. Do not re-attempt this
# without first understanding *why* editing a.binaries perturbs scipy.libs/
# -- both prior attempts were caught by packaging/verify_build.ps1 (a real
# packaged-exe run) before shipping, not by pytest, which runs against the
# dev venv's separate numpy/scipy install and can't see this class of bug.

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OriginStack',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,     # UPX + numpy/scipy .pyd files is a known source of both
                    # AV false positives and occasional load failures.
    console=False,  # windowed -- this is why runtime_hook_stdout.py and
                     # src/desktop_app.py's _fatal() dialog exist: there is
                     # no console for a bare print()/traceback to reach.
    icon=str(ROOT / 'packaging' / 'icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='OriginStack',   # -> dist/OriginStack/ (onedir -- see build_windows.ps1
                            # for why: onefile re-extracts the whole numpy/
                            # scipy/astropy payload to a temp dir on every
                            # launch, multi-second delay each time).
)
