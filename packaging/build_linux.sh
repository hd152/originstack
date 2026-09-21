#!/usr/bin/env bash
# Builds the OriginStack desktop app into a standalone Linux onedir bundle with PyInstaller
# and packs it as OriginStack-<VERSION>-linux-x64.tar.gz (+ .sha256).
#
# The Linux counterpart of build_windows.ps1: a fresh packaging-only venv (never a dev venv,
# whose `maturin develop` editable install PyInstaller cannot see into), the desktop extras
# requirements.txt only documents, astro_native built as a real wheel, then PyInstaller
# against packaging/originstack.spec.
#
#   PYTHON=python3.12 ./packaging/build_linux.sh        # which interpreter (default python3)
#   CLEAN=1 ./packaging/build_linux.sh                  # recreate the venv
#
# Needs: Python with tkinter (the desktop app is a Tk window), a Rust toolchain (cargo), and
# for portability of the result, an old-enough glibc: the bundle needs at least the glibc of the
# machine that built it. The release workflow builds on ubuntu-22.04 (glibc 2.35).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
VENV="$ROOT/packaging/.build_venv_linux"

"$PY" -c "import tkinter" 2>/dev/null || {
    echo "ERROR: $PY has no tkinter. Install it (Debian/Ubuntu: apt install python3-tk) or use a Python that bundles it (e.g. uv's)." >&2
    exit 1
}
command -v cargo >/dev/null || { echo "ERROR: Rust toolchain not found (cargo)." >&2; exit 1; }

[ -n "${CLEAN:-}" ] && rm -rf "$VENV"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
VPY="$VENV/bin/python"
"$VPY" -m pip install --quiet --upgrade pip

# Core + the extras the packaged app bundles (see build_windows.ps1 for why).
"$VPY" -m pip install --quiet -r "$ROOT/requirements.txt"
"$VPY" -m pip install --quiet "psutil>=5.9" "rawpy>=0.19" "tifffile>=2023.1" \
    "pyinstaller>=6.0" "pyinstaller-hooks-contrib" "maturin>=1.7,<2.0"

# astro_native as a REAL wheel. No RUSTFLAGS=target-cpu=native: that wheel may use instructions
# older CPUs lack and must never be redistributed. Clear old wheels first -- target/wheels/
# accumulates one per version and picking "the" wheel from a stale directory silently installs an
# old build.
( cd "$ROOT/ext/astro_native"
  rm -f target/wheels/astro_native-*.whl
  "$VPY" -m maturin build --release
  wheels=(target/wheels/astro_native-*.whl)
  [ "${#wheels[@]}" -eq 1 ] || { echo "ERROR: expected exactly 1 wheel, found ${#wheels[@]}" >&2; exit 1; }
  "$VPY" -m pip install --quiet --force-reinstall "${wheels[0]}" )

# Fail now if the native module silently fell back to numpy or is a stale version.
cargo_version="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$ROOT/ext/astro_native/Cargo.toml" | head -1 | tr -d '')"
installed_version="$("$VPY" -c 'import astro_native; print(astro_native.__version__)')"
echo "astro_native OK: $installed_version"
[ "$installed_version" = "$cargo_version" ] || {
    echo "ERROR: astro_native version mismatch: Cargo.toml $cargo_version, installed $installed_version" >&2; exit 1; }

"$VPY" -m PyInstaller "$ROOT/packaging/originstack.spec" \
    --distpath "$ROOT/packaging/dist" --workpath "$ROOT/packaging/build_linux" --noconfirm

# Pack: OriginStack-<ver>-linux-x64/{OriginStack/ (the bundle), install.sh, icon.png, README.txt}
VERSION="$(tr -d '[:space:]' < "$ROOT/VERSION")"
NAME="OriginStack-$VERSION-linux-x64"
STAGE="$ROOT/packaging/dist/$NAME"
rm -rf "$STAGE" "$ROOT/packaging/dist/$NAME.tar.gz"
mkdir -p "$STAGE"
cp -a "$ROOT/packaging/dist/OriginStack" "$STAGE/OriginStack"
cp "$ROOT/packaging/linux/install.sh" "$STAGE/install.sh"
cp "$ROOT/packaging/linux/README.txt" "$STAGE/README.txt"
cp "$ROOT/assets/icon.png" "$STAGE/icon.png"
chmod +x "$STAGE/install.sh" "$STAGE/OriginStack/OriginStack"
tar -C "$ROOT/packaging/dist" -czf "$ROOT/packaging/dist/$NAME.tar.gz" "$NAME"
( cd "$ROOT/packaging/dist" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )

size_mb=$(( $(stat -c %s "$ROOT/packaging/dist/$NAME.tar.gz") / 1048576 ))
[ "$size_mb" -ge 50 ] || { echo "ERROR: archive suspiciously small (${size_mb} MB)" >&2; exit 1; }
echo "Built: packaging/dist/$NAME.tar.gz (${size_mb} MB)"
