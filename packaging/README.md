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
numpy, runs a real stack with multiple parallel workers -- with
`--originvision` on, so a self-disable warning fails the check if the native
scorer or `originvision.onnx` didn't make it into the bundle -- and
confirms no extra GUI windows open -- regression guard for a real bug that
shipped once: a frozen ProcessPoolExecutor worker without
`multiprocessing.freeze_support()` re-launches the whole app instead of
running as a worker -- then confirms clean shutdown):

```powershell
.\packaging\verify_build.ps1
```

## What's bundled vs. not

Bundled: numpy, astropy, scipy, tqdm, Pillow, psutil, rawpy (camera RAW),
tifffile (TIFF I/O), tkinter (the desktop app's UI toolkit, stdlib),
`astro_native` (built from a real `maturin build --release` wheel, not the
dev-mode `maturin develop` editable install -- see `originstack.spec`'s
comments for why that distinction matters for PyInstaller), and the
`src/data/originvision.onnx` model (~11 MB) for `--originvision`.

Not bundled:
- `cupy`/GPU acceleration -- not viable in a generic packaged exe (requires
  the end user's own CUDA install); the app degrades to CPU gracefully
  (`src/gpu_context.py`), so `--use-gpu` isn't available in the packaged
  build.
- `onnxruntime` -- `--originvision` inference is the native
  `astro_native.originvision_score` kernel (pure-Rust `tract`) in the
  packaged app; the Python `onnxruntime` path is only a source-checkout
  fallback for when `ext/astro_native/` isn't built.

## Known limitations

- **Unsigned exe.** No code-signing certificate is used for this build, so
  Windows SmartScreen and some antivirus engines will flag it on first run.
  There's no code fix for this without a paid cert; the available mitigation
  is submitting the built `OriginStack.exe` to Microsoft's file-submission
  portal (https://www.microsoft.com/en-us/wdsi/filesubmission) after each
  release, which reduces false-positive flagging over time.
- **Windows only.** No macOS/Linux packaging in this pass.
- **The installer is unsigned too**, so SmartScreen treats `OriginStack-<version>-setup.exe` like the exe.
- **Run it extracted or installed, never from inside the zip.** Double-clicking `OriginStack.exe` in
  Explorer's zip view extracts only the exe (into a `...zip.aca` temp folder), not its `_internal\`
  folder, and fails with "Failed to load Python DLL". Use the installer, or **Extract All** first.

## Installer

`packaging\build_installer.ps1` wraps `dist\OriginStack\` in a normal `OriginStack-<VERSION>-setup.exe`
with Inno Setup (`packaging\originstack.iss`; a build-time tool only -- `winget install
JRSoftware.InnoSetup`). It installs per user into `%LOCALAPPDATA%\Programs\OriginStack` with no admin
prompt (the wizard's usual "for all users" choice is offered), adds a Start Menu entry (and an optional
desktop shortcut) and an uninstaller. The release workflow builds it after the PyInstaller step, installs
it silently on the runner, runs `verify_build.ps1` against the *installed* exe, and only then attaches it
to the release next to the zip.

## Release automation

`.github/workflows/release.yml` builds and attaches the Windows zip to an
already-existing GitHub Release on any `v*` tag push -- it does **not**
create the release itself (matching this repo's established manual
`gh release create` workflow). Create the release first, then push the tag
(or push the tag after creating the release, either order works as long as
the release exists by the time the workflow's upload step runs).
