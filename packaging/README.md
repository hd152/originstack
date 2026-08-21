# Packaging the desktop app (Windows)

Produces a standalone, double-click-runnable `OriginStack.exe` -- no Python
or pip install required by the end user.

## Prerequisites (build machine only)

- A Rust toolchain (`cargo --version` must work) -- needed to build
  `ext/astro_native` as a real wheel.
- Python for the packaging venv. `build_windows.ps1` defaults to whatever
  `py` resolves to with no version flag. PyInstaller and pyo3's tooling tend
  to lag a few months behind the newest CPython release; if the default
  build fails, install Python 3.12 (`py install 3.12`, or the python.org
  installer) and rerun with `-PythonVersion 3.12`.

## Building

```powershell
.\packaging\build_windows.ps1
# or, if the default Python is too new for the current PyInstaller/pyo3:
.\packaging\build_windows.ps1 -PythonVersion 3.12
```

Output: `packaging\dist\OriginStack\` (onedir -- starts instantly, unlike
onefile which re-extracts the whole numpy/scipy/astropy payload to a temp
dir on every launch) and a zipped `packaging\dist\OriginStack-<version>-
windows-x64.zip` for distribution.

Verify the build actually works (launches the real exe, confirms the window
appears, confirms `astro_native` loaded rather than silently falling back to
numpy, runs a real stack with multiple parallel workers and confirms no
extra GUI windows open -- regression guard for a real bug that shipped once:
a frozen ProcessPoolExecutor worker without `multiprocessing.freeze_support()`
re-launches the whole app instead of running as a worker -- confirms clean
shutdown):

```powershell
.\packaging\verify_build.ps1
```

## What's bundled vs. not

Bundled: numpy, astropy, scipy, tqdm, Pillow, psutil, rawpy (camera RAW),
tifffile (TIFF I/O), tkinter (the desktop app's UI toolkit, stdlib), and
`astro_native` (built from a real `maturin build --release` wheel, not the
dev-mode `maturin develop` editable install -- see `originstack.spec`'s
comments for why that distinction matters for PyInstaller).

Not bundled: `cupy`/GPU acceleration. Not viable in a generic packaged exe
(requires the end user's own CUDA install); the app already degrades to CPU
gracefully (`src/gpu_context.py`), so `--use-gpu` simply isn't available in
the packaged build.

## Known limitations

- **Unsigned exe.** No code-signing certificate is used for this build, so
  Windows SmartScreen and some antivirus engines will flag it on first run.
  There's no code fix for this without a paid cert; the available mitigation
  is submitting the built `OriginStack.exe` to Microsoft's file-submission
  portal (https://www.microsoft.com/en-us/wdsi/filesubmission) after each
  release, which reduces false-positive flagging over time.
- **Windows only.** No macOS/Linux packaging in this pass.
- **No installer wizard.** Ships as a zipped folder, not an Inno
  Setup/MSI installer. A reasonable follow-up if a smoother install
  experience (Start Menu shortcut, uninstaller entry) is wanted later.

## Release automation

`.github/workflows/release.yml` builds and attaches the Windows zip to an
already-existing GitHub Release on any `v*` tag push -- it does **not**
create the release itself (matching this repo's established manual
`gh release create` workflow). Create the release first, then push the tag
(or push the tag after creating the release, either order works as long as
the release exists by the time the workflow's upload step runs).
