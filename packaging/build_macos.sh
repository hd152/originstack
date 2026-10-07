#!/usr/bin/env bash
# Builds the OriginStack desktop app into OriginStack.app with PyInstaller and packs it as
# OriginStack-<VERSION>-macos-<arch>.zip (+ .sha256), arch = arm64 or x86_64 (the build machine's).
#
# The macOS counterpart of build_windows.ps1 / build_linux.sh: a fresh packaging-only venv (never
# a dev venv, whose `maturin develop` editable install PyInstaller cannot see into), the desktop
# extras requirements.txt only documents, astro_native built as a real wheel, then PyInstaller
# against packaging/originstack.spec (which adds the .app BUNDLE on macOS).
#
#   PYTHON=python3.12 ./packaging/build_macos.sh        # which interpreter (default python3)
#   CLEAN=1 ./packaging/build_macos.sh                  # recreate the venv
#
# Needs: a Python with tkinter (python.org / actions/setup-python builds bundle Tk; Homebrew's
# python needs `brew install python-tk`), a Rust toolchain (cargo), and Xcode command-line tools
# (sips, iconutil, ditto, codesign ship with macOS itself). Not signed or notarized -- see
# packaging/README.md ("macOS").
#
# The result runs on the macOS it was built on and later, at best: the Rust wheel targets
# MACOSX_DEPLOYMENT_TARGET (default 11.0), but numpy/scipy wheels pip picks for the build machine
# may need a newer macOS (e.g. scipy's Accelerate-based arm64 wheels need 14.0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
VENV="$ROOT/packaging/.build_venv_macos"
WORK="$ROOT/packaging/build_macos"
ARCH="$(uname -m)"                       # arm64 | x86_64
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"

[ "$(uname -s)" = "Darwin" ] || { echo "ERROR: build_macos.sh must run on macOS." >&2; exit 1; }
"$PY" -c "import tkinter; tkinter.Tcl()" 2>/dev/null || {
    echo "ERROR: $PY has no working tkinter. Use a python.org/setup-python build, or 'brew install python-tk'." >&2
    exit 1
}
command -v cargo >/dev/null || { echo "ERROR: Rust toolchain not found (cargo)." >&2; exit 1; }
# Never ship a host-tuned build (see CLAUDE.md, "Run-time AVX2 dispatch"): the wheel/app would
# crash with an illegal instruction on older CPUs.
case "${RUSTFLAGS:-}" in *target-cpu=native*)
    echo "ERROR: RUSTFLAGS contains target-cpu=native -- not allowed for a redistributable build." >&2; exit 1;;
esac

[ -n "${CLEAN:-}" ] && rm -rf "$VENV"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
VPY="$VENV/bin/python"
"$VPY" -m pip install --quiet --upgrade pip

# Core + the extras the packaged app bundles (see build_windows.ps1 for why).
"$VPY" -m pip install --quiet -r "$ROOT/requirements.txt"
"$VPY" -m pip install --quiet "psutil>=5.9" "rawpy>=0.19" "tifffile>=2023.1" \
    "pyinstaller>=6.0" "pyinstaller-hooks-contrib" "maturin>=1.7,<2.0"

# astro_native as a REAL wheel, for this machine's architecture. Clear old wheels first --
# target/wheels/ accumulates one per version and a stale one would install silently.
( cd "$ROOT/ext/astro_native"
  rm -f target/wheels/astro_native-*.whl
  "$VPY" -m maturin build --release
  wheels=(target/wheels/astro_native-*.whl)
  [ "${#wheels[@]}" -eq 1 ] || { echo "ERROR: expected exactly 1 wheel, found ${#wheels[@]}" >&2; exit 1; }
  "$VPY" -m pip install --quiet --force-reinstall "${wheels[0]}" )

# Fail now if the native module silently fell back to numpy or is a stale version.
cargo_version="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$ROOT/ext/astro_native/Cargo.toml" | head -1)"
installed_version="$("$VPY" -c 'import astro_native; print(astro_native.__version__)')"
echo "astro_native OK: $installed_version ($ARCH)"
[ "$installed_version" = "$cargo_version" ] || {
    echo "ERROR: astro_native version mismatch: Cargo.toml $cargo_version, installed $installed_version" >&2; exit 1; }

# .icns from assets/icon.png (512x512). Optional: the spec falls back to the generic icon.
mkdir -p "$WORK"
ICNS="$WORK/OriginStack.icns"
rm -f "$ICNS"
if command -v sips >/dev/null && command -v iconutil >/dev/null; then
    SET="$WORK/OriginStack.iconset"
    rm -rf "$SET"; mkdir -p "$SET"
    for s in 16 32 128 256 512; do
        sips -z "$s" "$s" "$ROOT/assets/icon.png" --out "$SET/icon_${s}x${s}.png" >/dev/null
        d=$((s * 2))
        if [ "$d" -le 512 ]; then
            sips -z "$d" "$d" "$ROOT/assets/icon.png" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
        fi
    done
    iconutil -c icns "$SET" -o "$ICNS" || { echo "WARNING: iconutil failed; building without an icon"; rm -f "$ICNS"; }
else
    echo "WARNING: sips/iconutil not found; building without an app icon"
fi
export ORIGINSTACK_ICNS="$ICNS"

rm -rf "$ROOT/packaging/dist/OriginStack" "$ROOT/packaging/dist/OriginStack.app"
"$VPY" -m PyInstaller "$ROOT/packaging/originstack.spec" \
    --distpath "$ROOT/packaging/dist" --workpath "$WORK" --noconfirm

APP="$ROOT/packaging/dist/OriginStack.app"
[ -x "$APP/Contents/MacOS/OriginStack" ] || { echo "ERROR: build did not produce $APP" >&2; exit 1; }

# Pack with ditto, not zip: it keeps the bundle's symlinks (PyInstaller 6 links
# Contents/Frameworks <-> Contents/Resources) and extended attributes, which `zip -r` breaks.
VERSION="$(tr -d '[:space:]' < "$ROOT/VERSION")"
NAME="OriginStack-$VERSION-macos-$ARCH"
ZIP="$ROOT/packaging/dist/$NAME.zip"
rm -f "$ZIP" "$ZIP.sha256"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
( cd "$ROOT/packaging/dist" && shasum -a 256 "$NAME.zip" > "$NAME.zip.sha256" )

size_mb=$(( $(wc -c < "$ZIP") / 1048576 ))
[ "$size_mb" -ge 50 ] || { echo "ERROR: archive suspiciously small (${size_mb} MB)" >&2; exit 1; }
echo "Built: packaging/dist/$NAME.zip (${size_mb} MB)"
