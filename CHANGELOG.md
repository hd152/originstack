# Changelog

All notable user-facing changes to OriginStack are documented here. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/); versions
match the `VERSION` file and `v*` git tags.

## [Unreleased]

## [2.0.0] - 2026-09-07

### Breaking

- **`--astrollm*` → `--originvision*`.** The upstream model project renamed
  itself (`astrollm` → `originvision`), and OriginStack follows: every
  `--astrollm`, `--astrollm-score-all`, `--astrollm-model` etc. is now
  `--originvision*`, the `ASTROLLM_DIR` env var is `ORIGINVISION_DIR`, and
  `FrameInfo.metrics['astrollm']` is `['originvision']`. **No aliases** —
  a command line still passing `--astrollm` will error. `--astrollm-python`
  / `--astrollm-script` / `--astrollm-timeout` were already inert after the
  scorer moved in-process and are removed entirely.
- **Star removal is opt-in.** The `<output>_starless.fits` sidecar
  surprised users who never asked for it. It is **off by default** now;
  `--remove-stars` enables it. `--no-remove-stars` is kept as a hidden
  no-op so existing command lines don't error. `--auto` no longer turns it
  on for any target type. The main output (with stars) is unaffected
  either way.

### Added

- **`--originvision` inference runs fully in-process in Rust.** The new
  `astro_native.originvision_score` kernel does the whole path —
  preprocessing (percentile stretch, resize/centre-crop, comet shape-gate)
  plus the ONNX forward pass, via the pure-Rust `tract` runtime — with **no
  Python ONNX dependency and no extra DLL**. A source checkout without
  `ext/astro_native/` built falls back to a Python `onnxruntime` path;
  `onnxruntime` is no longer bundled in the packaged app. The exported
  model ships inside the package at `src/data/originvision.onnx`.
- **`--originvision-model PATH`** overrides the bundled model (e.g. to test
  a newer checkpoint). The bundled model is now "v4" — a from-scratch,
  no-SSL 7-task run (best-by-category checkpoint, epoch 15). vs the
  previous model it matches or beats on category, reject-recall and
  quality; it is ~0.07 lower on exposure-class accuracy.
- **`--photometry`: absolute aperture photometry on the linear stack.**
  Detects stars, cone-searches Gaia DR3 around the field (via the header
  WCS — a Celestron Origin `info.json` session solve or `--plate-solve`),
  cross-matches, and does per-channel **partial-pixel** circular-aperture
  photometry (supersampled aperture edge + robust sky annulus). Fits a
  robust per-channel photometric zero point
  `m_cal = m_inst - k·X + ZP + CT·(BP-RP - ref)` against Gaia G/BP/RP
  (RP→R, G→G, BP→B — a coarse OSC mapping, not a filter-matched
  transform), with an airmass term `X` from the site GPS + observation
  time in `info.json` when present (otherwise extinction folds into the
  zero point). Writes `<output>_photometry.csv` (per-star RA/Dec,
  flux/mag/mag-err/SNR per channel, saturation flag) and
  `MAGZP_R/G/B` / `MAGZPE_R/G/B` / `MAGCT_R/G/B` / `PHOTGAIN` header
  keywords. Flags: `--photometry-color-terms` (fit the per-channel colour
  term instead of a plain median), `--photometry-extinction-k` (override
  the nominal `k` = R 0.09 / G 0.15 / B 0.23 mag/airmass),
  `--photometry-gain` (electrons/ADU for the Poisson error term).
- **`--photometry-timeseries`: per-frame differential light curves.**
  Runs a separate pass over the registered subs (right after Phase 3),
  aperture-photometering a fixed Gaia star list on every frame and
  ensemble-differential-calibrating a per-frame zero point (removes
  transparency / airmass drift). Writes `<output>_lightcurves.csv` (one
  row per frame × star — MJD, airmass, per-channel mag/mag-err) and
  `<output>_lightcurve_stats.csv` (per star — mean, rms, MAD, reduced χ²
  vs a constant, and a `variable` flag). `--photometry-target "RA,DEC"` or
  `"px:X,Y"` marks and reports one star. Needs a session `info.json` WCS
  (`--plate-solve` runs too late for per-frame work); differential only,
  no absolute zero point.
- **Photon-transfer gain / read-noise from raw calibration frames.** When
  `--photometry`/`--photometry-timeseries` is set and no `--photometry-gain`
  is given, a Janesick two-frame difference over ≥2 bias + ≥2 flat frames
  estimates the sensor gain (e-/ADU) and read noise (e-), feeding a real
  Poisson term into the per-star photometric errors.
- **Quality-sweep score cache.** `--quality-sweep` writes a versioned
  `.sweepcache.json` at the sweep root, keyed by each frame's mtime+size,
  so re-sweeps of a mostly-unchanged collection only re-score new or
  changed frames. `--sweep-no-cache` ignores and does not write it.

### Changed

- **`--auto` now skips the sky-residual correction passes for emission and
  reflection nebulae too, not just galaxies.** `remove_sky_residual`'s
  mesh background fit reads a frame-filling nebula as elevated background
  and subtracts through it (confirmed on a real Lagoon Nebula session —
  the passes removed over half the nebulosity). DBE's own protected pass
  already does the main background-flattening job for these targets. Small
  nebulae on empty sky (low emission/reflection blend weight) keep the
  step. Override either way with an explicit `--skip-step`.
- **`--fix-atmospheric-dispersion` now auto-derives `--plate-scale` and
  `--zenith-angle`** (from the header WCS, and from the `info.json` GPS +
  observation time) when they are not given. `--parallactic-angle` stays
  required — a wrong value shifts colour channels the wrong way and
  mapping it onto the detector needs the image north angle.
- **Desktop app layout.** The GHS-slider **STRETCH** panel is removed
  (`ui_events.restretch` / the preview's `replace_pixels` stay as
  API). RECENT FRAMES moves next to the preview on the right, so the whole
  left column below the pipeline bar is the log. Form field labels are now
  human ("Light frames", "Stack method", …) instead of the raw `--flag`
  text, with the flag name in the hover tooltip. Previews and the
  thumbnail ring clear at the start of each new run.
- **`psutil` is an optional dependency now, not a core one.** Every use is
  already `try/except`-guarded with a fixed fallback; installing it lets
  the pipeline size workers and memmaps to real free RAM.

### Internal

- **`astro_native` 0.18.0 → 0.21.0.** New kernels: `originvision_score`
  (full `--originvision` inference via `tract`), `fit_psf_moffat2d_native`
  / `fit_psf_gauss2d_native` (2D star-profile PSF fits — removes scipy
  `curve_fit`'s per-iteration Python callback, ~4× on a 30-star PSF
  estimate), `aperture_photometry_batch` (partial-pixel aperture
  photometry for N centres at once, ~150× at time-series scale). Release
  profile `panic = "abort"` → `"unwind"` so a malformed
  `--originvision-model` raises a Python exception instead of aborting the
  process; `lto` `true` → `"thin"` (tract's ~120-crate dep tree made fat
  LTO builds punishing).
- **`vendor/astrollm/` → `vendor/originvision/`** — an upstream provenance
  snapshot only, not on the runtime path.
- **Refactor: shared photometry primitives.** New `src/photometry_core.py`
  (aperture-photometry kernel + dispatcher, WCS→pixel projection, field
  centre/radius) and `src/observing_geometry.py` (alt/az, airmass, zenith
  & parallactic angle from lat/long + RA/Dec + UTC). `color_calibrate` and
  `photometric_calibration`'s per-star aperture loops are now thin
  wrappers over the shared native kernel.
- **`postprocess_stack` hardening.** A failure in the star-removal /
  starless-sidecar block can no longer abort the output stage and leave an
  orphan `_starless.fits` with no main output — it is now a warning.
- **Dependency trim.** `onnxruntime` is no longer installed by
  `build_windows.ps1` or collected into the packaged app.

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
- **`--originvision` no longer scores every accepted light frame by default.**
  It now samples 3 frames (fast, ~8s each) to feed the target-classification
  prior and a defect-nudge, same as before but bounded in time regardless of
  session size. Full per-frame scoring is still available, opt-in, via the
  new `--originvision-score-all` flag. `--originvision-score-all` alone (without
  `--originvision`) is a no-op and now warns at startup.

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

