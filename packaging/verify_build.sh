#!/usr/bin/env bash
# Proves the packaged Linux bundle actually works, without a person looking at the window.
# The Linux counterpart of verify_build.ps1:
#
#   1. Normal launch: the app must still be running after a few seconds (a broken bundled
#      import or a Tk failure exits at once through desktop_app.py's _fatal), and its startup
#      log must say astro_native loaded -- not the numpy fallback, which would silently ship a
#      much slower build.
#   2. --verify-headless: a real multi-worker stack through the same frozen entry point, so a
#      missing multiprocessing.freeze_support() (every pool worker re-running the whole app)
#      or a missing native/bundled file fails here rather than on a user's machine.
#
#   ./packaging/verify_build.sh [path/to/OriginStack] [python-with-numpy-astropy]
#
# Needs a display: uses $DISPLAY when set, else re-runs itself under xvfb-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="${1:-$ROOT/packaging/dist/OriginStack/OriginStack}"
PY="${2:-$ROOT/packaging/.build_venv_linux/bin/python}"
[ -x "$EXE" ] || { echo "ERROR: build did not produce $EXE" >&2; exit 1; }

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    command -v xvfb-run >/dev/null || { echo "ERROR: no display and no xvfb-run" >&2; exit 1; }
    exec xvfb-run -a "$0" "$@"
fi

STATE="$(mktemp -d)"
export XDG_STATE_HOME="$STATE"
LOG="$STATE/OriginStack/logs/desktop_app.log"
trap '[ -n "${APP_PID:-}" ] && kill "$APP_PID" 2>/dev/null || true; rm -rf "$STATE"' EXIT

# ── 1. normal launch ─────────────────────────────────────────────────────────
"$EXE" >"$STATE/app.out" 2>&1 &
APP_PID=$!
ok=""
for _ in $(seq 1 30); do
    kill -0 "$APP_PID" 2>/dev/null || { echo "ERROR: OriginStack exited during startup"; cat "$STATE/app.out"; \
        cat "$STATE/OriginStack/logs/desktop_app_crash.log" 2>/dev/null || true; exit 1; }
    if [ -f "$LOG" ] && grep -q "ACTIVE" "$LOG"; then ok=1; fi
    if [ -f "$LOG" ] && grep -q "numpy fallback" "$LOG"; then
        echo "ERROR: astro_native did not load in the packaged build -- shipping numpy fallback silently"; exit 1
    fi
    sleep 0.5
done
[ -n "$ok" ] || { echo "ERROR: startup log never confirmed astro_native active ($LOG)"; exit 1; }
kill -0 "$APP_PID" 2>/dev/null || { echo "ERROR: OriginStack did not stay running"; exit 1; }
echo "App running; $(tail -1 "$LOG" | sed 's/^\[[^]]*\] //')"
kill "$APP_PID" 2>/dev/null || true
wait "$APP_PID" 2>/dev/null || true
APP_PID=

# ── 2. --verify-headless: real Phase 1 multiprocessing ──────────────────────
SYNTH="$ROOT/synthetic_data"
[ -d "$SYNTH" ] || ( cd "$ROOT" && "$PY" tools/create_synthetic.py )
[ -d "$SYNTH" ] || { echo "ERROR: synthetic_data was not created" >&2; exit 1; }
OUT="$STATE/verify_out.fits"
# --originvision exercises the bundled native scorer and model file.
"$EXE" --verify-headless -d "$SYNTH" -o "$OUT" --parallel 4 --debayer-method malvar \
    --white-balance grayworld --stack-method median --originvision >"$STATE/headless.out" 2>&1 || {
    echo "ERROR: --verify-headless failed"; tail -40 "$STATE/headless.out"; exit 1; }
[ -s "$OUT" ] || { echo "ERROR: --verify-headless did not produce $OUT"; tail -40 "$STATE/headless.out"; exit 1; }
if grep -qi "originvision.*\(disabled\|unavailable\|not found\)" "$STATE/headless.out"; then
    echo "ERROR: --originvision self-disabled in the frozen build (native scorer or model missing)"
    grep -i originvision "$STATE/headless.out" | head; exit 1
fi
echo "Headless stack passed ($(stat -c %s "$OUT") bytes)"
echo "Linux build verified."
