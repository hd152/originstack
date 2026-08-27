# Changelog

All notable user-facing changes to OriginStack are documented here. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/); versions
match the `VERSION` file and `v*` git tags.

## [Unreleased]

### Added

- **`--photometry`: absolute aperture photometry on the linear stack.**
  Detects stars, cone-searches Gaia DR3 around the field (via the header
  WCS — a Celestron Origin `info.json` session solve or `--plate-solve`),
  cross-matches, and does per-channel circular-aperture photometry
  (aperture + sigma-clipped sky annulus). Fits a robust per-channel
  photometric zero point `m_cal = m_inst - k·X + ZP` against Gaia G/BP/RP
  (RP→R, G→G, BP→B — a coarse OSC mapping, not a filter-matched
  transform), with an airmass term `X` derived from the site GPS +
  observation time in `info.json` when present (otherwise extinction is
  folded into the zero point). Writes `<output>_photometry.csv` (per-star
  RA/Dec, flux/mag/mag-err/SNR per channel, saturation flag) and
  `MAGZP_R/G/B` + `MAGZPE_R/G/B` header keywords. `--photometry-extinction-k`
  overrides the nominal per-band extinction coefficients (R=0.09, G=0.15,
  B=0.23 mag/airmass). Needs a WCS; no per-sub light curves in this first
  cut (the streaming stacker doesn't retain frames), and the Poisson error
  term is only included when the header carries a real `GAIN`/`EGAIN`.

## [1.0.0] - 2026-08-21

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
- **Six new native (Rust) kernels** (`ext/astro_native` 0.17.0 → 0.18.0),
  closing out the remaining Python-loop hot paths identified by a full
  profiling pass: `fit_moffat_native` (saturated-star repair's Moffat wing
  fit — removes scipy `curve_fit`'s per-iteration Python callback, 39x on a
  single fit), `mesh_median_grid` (background-extraction mesh median, on
  the default pipeline, 34x), `local_normalize_grid`
  (`--local-normalize`'s per-frame coarse background grid, 8x),
  `stamp_star_disks` (star-removal mask, on by default, 50x on a
  4000-star field), `bresenham_line_native` (`--trail-reject`'s line
  rasterization, 163x), and `radial_bin_median` (`--comet-mode`'s radial
  profile, 39x). Every kernel keeps its original numpy/scipy path as an
  automatic fallback when the native extension isn't built; see
  `CLAUDE.md`'s "Native (Rust) acceleration" section for kernel-by-kernel
  detail and `tests/test_native.py` for parity tests.

## Earlier development

Versions prior to 1.0.0 (`v0.1.0` – `v0.9.0`) predate this changelog; see
the [GitHub Releases](https://github.com/hd152/originstack/releases) page
and `git log` for that history.
