# Changelog

All notable user-facing changes to OriginStack are documented here. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/); versions
match the `VERSION` file and `v*` git tags.

## [1.0.0] - Unreleased

### Changed

- **Desktop app rewritten as a native window.** `python desktop_app.py` (and
  the packaged `OriginStack.exe`) is now a genuine `tkinter` window (stdlib)
  instead of a `pywebview`-wrapped local HTTP dashboard. No more Microsoft
  Edge WebView2 Runtime dependency — one fewer thing that can be missing on
  an end user's machine. The Setup form, live progress/log, and preview
  panel (zoom/pan, live re-stretch, before/after wipe-slider compare,
  per-frame thumbnail ring) all carry over with the same functionality.
- The standalone `--web-view` CLI flag (browser-tab dashboard) is removed;
  the desktop app is now the only GUI surface. The CLI itself is unchanged.
- **`--astrollm` no longer scores every accepted light frame by default.**
  It now samples 3 frames (fast, ~8s each) to feed the target-classification
  prior and a defect-nudge, same as before but bounded in time regardless of
  session size. Full per-frame scoring is still available, opt-in, via the
  new `--astrollm-score-all` flag. `--astrollm-score-all` alone (without
  `--astrollm`) is a no-op and now warns at startup.

### Internal

- `src/webview.py`'s HTTP/SSE transport is gone. Its state/data logic (log
  buffer, phase/progress, named preview slots, per-frame thumbnail ring,
  on-demand re-stretch) moved to `src/ui_events.py`, polled in-process by
  the tkinter window instead of pushed over a socket.
- `src/webview_control.py` renamed to `src/desktop_control.py` (schema
  introspection + `RunManager` are unchanged; only the name, since
  "webview" no longer describes anything in this codebase).
- `packaging/originstack.spec` bundles `tkinter` instead of excluding it;
  drops the `pywebview`/`pythonnet`/`clr_loader` PyInstaller hooks.
  `packaging/verify_build.ps1` proves the packaged build works via a window
  appearing + a startup log line (native-kernel check) and a new
  `--verify-headless` desktop-app flag (multiprocessing regression check),
  replacing the old HTTP-endpoint polling.

## Earlier development

Versions prior to 1.0.0 (`v0.1.0` – `v0.9.0`) predate this changelog; see
the [GitHub Releases](https://github.com/hd152/originstack/releases) page
and `git log` for that history.
