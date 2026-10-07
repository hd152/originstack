#!/usr/bin/env bash
# Proves the packaged OriginStack.app actually works, without a person looking at the window.
# The macOS counterpart of verify_build.ps1 / verify_build.sh:
#
#   1. Normal launch (skip with SKIP_GUI_CHECK=1): the app must still be running after a few
#      seconds with no crash log (a broken bundled import or a Tk failure goes through
#      desktop_app.py's _fatal, which writes desktop_app_crash.log), and its startup log must say
#      astro_native loaded -- not the numpy fallback, which would silently ship a much slower build.
#   2. --verify-headless: a real multi-worker stack of tools/create_synthetic.py data through the
#      same frozen entry point. Must exit 0, write the FITS, and log exactly ONE startup line: every
#      process that runs desktop_app.main() logs one, so a second line means a spawned Phase 1
#      worker re-ran the app (missing multiprocessing.freeze_support() -- on Windows that opened
#      one GUI window per core). Also re-checks astro_native from that run's startup line.
#
#   ./packaging/verify_build_macos.sh [path/to/OriginStack.app] [python-with-numpy-astropy]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${1:-$ROOT/packaging/dist/OriginStack.app}"
PY="${2:-$ROOT/packaging/.build_venv_macos/bin/python}"
EXE="$APP/Contents/MacOS/OriginStack"
[ -x "$EXE" ] || { echo "ERROR: build did not produce $EXE" >&2; exit 1; }
[ -x "$PY" ] || PY=python3

# Isolated HOME: desktop_app._log_dir() is ~/Library/Logs/OriginStack on macOS, so this both
# finds the logs and keeps a previous run's lines out of the count below.
STATE="$(mktemp -d)"
export HOME="$STATE"
LOGDIR="$STATE/Library/Logs/OriginStack"
LOG="$LOGDIR/desktop_app.log"
CRASH="$LOGDIR/desktop_app_crash.log"
APP_PID=""
HL_PID=""
cleanup() {
    [ -n "$APP_PID" ] && kill "$APP_PID" 2>/dev/null || true
    [ -n "$HL_PID" ] && kill "$HL_PID" 2>/dev/null || true
    pkill -f "$EXE" 2>/dev/null || true
    rm -rf "$STATE"
}
trap cleanup EXIT

check_native() {   # $1 = log line
    case "$1" in
        *"numpy fallback"*) echo "ERROR: astro_native did not load in the packaged build -- shipping numpy fallback silently"; exit 1;;
        *ACTIVE*) ;;
        *) echo "ERROR: startup log line does not confirm astro_native: $1"; exit 1;;
    esac
}

# ── 1. normal launch ─────────────────────────────────────────────────────────
if [ -z "${SKIP_GUI_CHECK:-}" ]; then
    "$EXE" >"$STATE/app.out" 2>&1 &
    APP_PID=$!
    for _ in $(seq 1 16); do          # ~8 s: past startup, Tk window creation and App()
        if ! kill -0 "$APP_PID" 2>/dev/null; then
            echo "ERROR: OriginStack exited during startup"; cat "$STATE/app.out"
            cat "$CRASH" 2>/dev/null || true; exit 1
        fi
        sleep 0.5
    done
    if [ -s "$CRASH" ]; then
        echo "ERROR: OriginStack hit a fatal error on startup:"; cat "$CRASH"; exit 1
    fi
    [ -f "$LOG" ] || { echo "ERROR: no startup log at $LOG"; cat "$STATE/app.out"; exit 1; }
    check_native "$(tail -1 "$LOG")"
    echo "App running; $(tail -1 "$LOG" | sed 's/^\[[^]]*\] //')"
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
    APP_PID=""
    rm -f "$LOG"
fi

# ── 2. --verify-headless: real Phase 1 multiprocessing ──────────────────────
SYNTH="$ROOT/synthetic_data"
[ -d "$SYNTH" ] || ( cd "$ROOT" && "$PY" tools/create_synthetic.py )
[ -d "$SYNTH" ] || { echo "ERROR: synthetic_data was not created" >&2; exit 1; }
OUT="$STATE/verify_out.fits"
"$EXE" --verify-headless -d "$SYNTH" -o "$OUT" --parallel 4 --debayer-method malvar \
    --white-balance grayworld --stack-method median >"$STATE/headless.out" 2>&1 &
HL_PID=$!
# No `timeout` on stock macOS: poll. A freeze_support regression would leave workers sitting in
# a Tk mainloop, i.e. hang -- the timeout turns that into a failure.
for _ in $(seq 1 600); do             # 10 min
    kill -0 "$HL_PID" 2>/dev/null || break
    sleep 1
done
if kill -0 "$HL_PID" 2>/dev/null; then
    echo "ERROR: --verify-headless did not finish in 10 minutes"; tail -40 "$STATE/headless.out"
    [ -f "$LOG" ] && cat "$LOG"; exit 1
fi
rc=0; wait "$HL_PID" || rc=$?
HL_PID=""
[ "$rc" -eq 0 ] || { echo "ERROR: --verify-headless exited $rc"; tail -40 "$STATE/headless.out"; exit 1; }
[ -s "$OUT" ] || { echo "ERROR: --verify-headless did not produce $OUT"; tail -40 "$STATE/headless.out"; exit 1; }

[ -f "$LOG" ] || { echo "ERROR: --verify-headless wrote no startup log at $LOG"; exit 1; }
starts="$(grep -c " starting -- " "$LOG" || true)"
if [ "$starts" -ne 1 ]; then
    echo "ERROR: $starts processes ran desktop_app.main() during --verify-headless (expected 1) --"
    echo "       multiprocessing.freeze_support() regression in desktop_app.py"; cat "$LOG"; exit 1
fi
check_native "$(tail -1 "$LOG")"
echo "Headless stack passed ($(wc -c < "$OUT" | tr -d ' ') bytes, 1 app process, astro_native active)"

pkill -f "$EXE" 2>/dev/null || true
sleep 1
if pgrep -f "$EXE" >/dev/null; then
    echo "ERROR: OriginStack process(es) still running after cleanup"; exit 1
fi
echo "macOS build verified: $APP"
