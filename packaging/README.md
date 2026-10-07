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
numpy, runs a real stack with multiple parallel workers, and
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
comments for why that distinction matters for PyInstaller), the
`src/data/transient_triage.onnx` model (~100 KB) for `--transient-triage`, and
the example images on the desktop app's target cards (`src/data/examples/`).

Not bundled:
- `cupy`/GPU acceleration -- not viable in a generic packaged exe (requires
  the end user's own CUDA install); the app degrades to CPU gracefully
  (`src/gpu_context.py`), so `--use-gpu` isn't available in the packaged
  build.
- `onnxruntime` -- `--transient-triage` inference is the native
  `astro_native.transient_triage_score` kernel (pure-Rust `tract`).

## Known limitations

- **Unsigned until SignPath is set up.** Without the SignPath variables below, releases are unsigned and
  Windows SmartScreen and some antivirus engines flag them on first run. See "Code signing".
- **Platforms.** Windows (zip + installer), Linux (tarball) and macOS (unsigned `.app` zip) -- see the
  sections below.
- **Run it extracted or installed, never from inside the zip.** Double-clicking `OriginStack.exe` in
  Explorer's zip view extracts only the exe (into a `...zip.aca` temp folder), not its `_internal\`
  folder, and fails with "Failed to load Python DLL". Use the installer, or **Extract All** first.

## Linux

`packaging/build_linux.sh` builds the same PyInstaller bundle on Linux and packs it as
`OriginStack-<VERSION>-linux-x64.tar.gz` (+ `.sha256`) containing the bundle, `install.sh` (per-user install
into `~/.local/share`, a menu entry, `--uninstall`) and a short README. `packaging/verify_build.sh` is the
counterpart of `verify_build.ps1`: it launches the bundle under a display (or `xvfb-run`), requires the startup
log to say the native kernels loaded, then runs `--verify-headless` with four workers on synthetic data.

Needs a Python **with tkinter** (`apt install python3-tk`, or a distribution of Python that bundles Tk), a Rust
toolchain, and `libtk8.6`/`libtcl8.6` present at build time so PyInstaller can collect them. The bundle needs a glibc at
least as new as the build machine's, so the release workflow builds on `ubuntu-22.04` (glibc 2.35) rather than
the newest image. To try it without a release, run the "Release" workflow by hand from the Actions tab: the Linux
job runs and keeps the tarball as a workflow artifact.

Local test from Windows: WSL2 works (WSLg provides the display). `sudo` is not needed if you install a Python with
Tk via `uv python install 3.12` and Rust via `rustup`; a Python whose Tcl/Tk lives outside the default library path (uv's) also
needs `LD_LIBRARY_PATH` pointing at its `lib/` while building, or the bundle misses `libtcl9tk9.0.so`.
`verify_build.sh` exists because both of the problems found this way -- that missing library and a Phase 1 deadlock
from `fork` -- only showed up by running the built bundle.

## macOS

`packaging/build_macos.sh` builds the same PyInstaller spec on macOS; on `darwin` the spec also wraps the
onedir in `OriginStack.app` (`BUNDLE`), and the script packs it with `ditto` (which keeps the bundle's
symlinks; `zip -r` does not) as `OriginStack-<VERSION>-macos-<arch>.zip` (+ `.sha256`), `<arch>` being the
build machine's `arm64` or `x86_64` -- there is no universal2 build. The `.icns` is generated from
`assets/icon.png` with `sips` + `iconutil`; if that fails the app gets the generic icon. astro_native is built
as a real wheel for the host architecture, without `target-cpu=native` (the script refuses it).
`packaging/verify_build_macos.sh` is the counterpart of `verify_build.ps1`: it launches the app and checks it
stays up with no crash log and that the startup log says the native kernels are active (`SKIP_GUI_CHECK=1`
skips this part), then runs `--verify-headless` with four workers on `tools/create_synthetic.py` data and
requires a FITS, exit code 0, and exactly one startup-log line (a second one means a spawned worker re-ran
the app: the `freeze_support()` regression). Logs go to `~/Library/Logs/OriginStack/`.

Needs a Python **with tkinter** (python.org and `actions/setup-python` builds include Tk; Homebrew's needs
`brew install python-tk`), a Rust toolchain and the Xcode command-line tools. The release workflow builds on
`macos-14` (arm64) and `macos-15-intel` (x86_64; GitHub retired `macos-13`). The minimum macOS the app runs on
is set by the wheels pip picks on the build machine (scipy's arm64 Accelerate wheels need macOS 14), not only by
`MACOSX_DEPLOYMENT_TARGET` (11.0, which covers the Rust wheel). To try it without a release, run the "Release"
workflow by hand from the Actions tab; both macOS jobs keep their zip as a workflow artifact.

**Unsigned and not notarized.** PyInstaller ad-hoc signs the binaries (Apple silicon will not run unsigned
code at all), but there is no Developer ID signature, so Gatekeeper blocks the first launch of a downloaded
copy ("cannot be opened because the developer cannot be verified" / "is damaged"). Either right-click (or
Control-click) `OriginStack.app` > **Open** > **Open** once, or clear the quarantine flag:

```bash
xattr -dr com.apple.quarantine /path/to/OriginStack.app
```

Adding signing later needs an Apple Developer Program membership ($99/year): import a "Developer ID
Application" certificate into a temporary keychain on the runner (from repo secrets), pass
`codesign_identity='Developer ID Application: ...'` and an entitlements file to `EXE`/`BUNDLE` in
`originstack.spec` (hardened runtime needs at least `com.apple.security.cs.allow-unsigned-executable-memory`
and `com.apple.security.cs.disable-library-validation` for a PyInstaller app), then notarize the zip with
`xcrun notarytool submit --wait` and `xcrun stapler staple OriginStack.app` before re-zipping.

## Installer

`packaging\build_installer.ps1` wraps `dist\OriginStack\` in a normal `OriginStack-<VERSION>-setup.exe`
with Inno Setup (`packaging\originstack.iss`; a build-time tool only -- `winget install
JRSoftware.InnoSetup`). It installs per user into `%LOCALAPPDATA%\Programs\OriginStack` with no admin
prompt (the wizard's usual "for all users" choice is offered), adds a Start Menu entry (and an optional
desktop shortcut) and an uninstaller. The release workflow builds it after the PyInstaller step, installs
it silently on the runner, runs `verify_build.ps1` against the *installed* exe, and only then attaches it
to the release next to the zip.

## Code signing (SignPath Foundation)

`release.yml` signs `OriginStack.exe` (then re-zips) and the installer through
[SignPath](https://signpath.org/) -- free for open-source projects. Signing is **skipped** unless the repo
variable `SIGNPATH_ORGANIZATION_ID` is set, so releases keep working before approval.

One-time setup (maintainer):
1. Apply at https://signpath.org/apply (public repo, OSI licence, a code-signing policy page linked from
   the README are required -- see [`CODE_SIGNING_POLICY.md`](../CODE_SIGNING_POLICY.md), linked from the
   README's top link bar).
2. In SignPath create project `originstack` with signing policy `release-signing`, and two artifact
   configurations, both for a GitHub-uploaded zip: `exe` (`<pe-file path="OriginStack.exe">` sign) and
   `installer` (`<pe-file path="OriginStack-*-setup.exe">` sign). Trust the GitHub.com connector and
   restrict the policy to release tags.
3. In GitHub: repo variable `SIGNPATH_ORGANIZATION_ID`, repo secret `SIGNPATH_API_TOKEN`.

Each release needs **two manual approvals** in SignPath (one for the exe, then one for the installer built from it), because the Foundation requires an approver for every signing request. The workflow waits up to two hours for each, so approve them promptly after tagging or the build fails.

Only the top-level exe and installer are signed; the bundled `.pyd`/`.dll` files stay unsigned. The
workflow fails the release if a signed file's Authenticode status is not `Valid`.

## Release automation

`.github/workflows/release.yml` builds and attaches the Windows zip and installer, the Linux
tarball and the two macOS zips to an already-existing GitHub Release on any `v*` tag push -- it does **not**
create the release itself (matching this repo's established manual
`gh release create` workflow). Create the release first, then push the tag
(or push the tag after creating the release, either order works as long as
the release exists by the time the workflow's upload step runs).
