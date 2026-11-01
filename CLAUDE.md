# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
pip install pytest
# Optional: GPU support (NVIDIA + CUDA): the CuPy wheel for your CUDA version
pip install cupy-cuda12x   # see requirements-gpu.txt
# Optional: native (Rust) acceleration — see "Native (Rust) build" below
pip install maturin && (cd ext/astro_native && maturin develop --release)
```

### Run tests
```bash
pytest -q
# Full suite, parallel (pytest-xdist): ~4.5 min on the 8-core laptop. Tests never reach the
# network (tests/conftest.py _no_network)
pytest -q -n auto
# Run a single test
pytest tests/test_core.py::test_calculate_shift_recovery -v
```
`-n auto` is deliberately not the default: xdist workers break `--pdb`/`breakpoint()` and
interleave `-s` output. CI uses it on both jobs.

### Lint
```bash
pip install -r requirements-dev.txt          # ruff + test tooling
python -m ruff check .                        # style: pyflakes/pycodestyle/imports/logging
python -m ruff check --fix .                  # apply the safe autofixes
python tools/lint_conventions.py              # project conventions (see below)
python tools/lint_conventions.py --git origin/main   # + diff-scoped rules (print noise, Cargo bump)
```
Both are gated in CI (the `lint` job) and re-checked by `tests/test_lint_conventions.py`.
Ruff config lives in [pyproject.toml](pyproject.toml) — kept deliberately narrow (high-signal
rules only). The convention linter ([tools/lint_conventions.py](tools/lint_conventions.py))
enforces repo rules ruff can't express: **OS001** `logging.getLogger` must pass `"originstack"`,
not `__name__`/root; **OS002** module-level optional-dependency imports in `src/` must be
`try/except`-guarded; **OS003** (warn) no bare `print()` in library code — use `safe_print`;
**OS004** (warn) every native `#[pyfunction]` needs a test reference; **OS006** (warn) no broad `except Exception: pass` (log at debug instead); **OS005** (warn, `--git`)
bump `ext/astro_native/Cargo.toml` when the crate source changes.

### Run the stacker
```bash
python originstack.py -d lights/ -o stacked.fits
python originstack.py -d lights/ -o stacked.fits -v
python originstack.py -d session/ -o combined.fits --debug intermediates -v
# Phase 4 only, on an earlier run's linear output (uses its _config.toml)
python originstack.py --from-stack stacked.fits -o tweak.fits --stretch arcsinh
```

### CI smoke test (generates synthetic data then stacks it)
```bash
python tools/create_synthetic.py
python originstack.py -d synthetic_data -o ci_synthetic_stack.fits --debayer-method malvar --white-balance grayworld --stack-method median
```

### Debug registration issues
```bash
python originstack.py -d lights/ -o stacked.fits --debug registration
# Output diagnostics go to _registration_debug/
```

### Working in parallel sessions
Several agent sessions share this machine, and a branch switch in the shared
`C:\source\originstack` checkout moves every session's working tree (commits and uncommitted
files from different sessions have landed in the wrong PR before). So:
- Do multi-step work in your own worktree from the start, and remove it when done:
  ```bash
  git fetch origin && git worktree add -b <branch> ../originstack-<topic> origin/main
  git worktree remove ../originstack-<topic>        # after the PR
  ```
  (the worktree uses the same `.venv` Python; `python originstack.py` runs that worktree's code)
- Never leave uncommitted changes in the shared checkout; run `git status -sb` before any commit,
  reset or branch switch, and stage files by name (not `commit -a`).
- Benchmarks and training write tens of GB (`bench_vs_siril.py` work dirs, 4.9 GB banks under
  `C:\nn_work`): keep the logs, delete the work dirs and banks once the results are recorded.
  A nearly full temp disk also changes results (Phase 1 compression switches on).

## Architecture

[originstack.py](originstack.py) is a thin entry-point shim re-exporting the public symbols of `src/` (tests import from `originstack`); [desktop_app.py](desktop_app.py) is the equally thin shim for the desktop app. History, measurements and investigation notes that used to live here are in [dev-notes/](dev-notes/):
[module-notes.md](dev-notes/module-notes.md) (full per-module history), [pipeline-notes.md](dev-notes/pipeline-notes.md) (phases, checkpoints, merge, memory, hierarchical mode, solving, photometry, debayering), [native-kernels.md](dev-notes/native-kernels.md) (per-kernel design and timings), [siril-comparison.md](dev-notes/siril-comparison.md), [phase4-quality.md](dev-notes/phase4-quality.md). Read the relevant note before re-investigating a module's behaviour or performance.

### Modules (`src/`)

| Module | Contents |
|--------|----------|
| `gpu_context.py` | `GpuContext` (`xp` = numpy or cupy), `get_gpu()` singleton; `--use-gpu` |
| `models.py` | `Config` (all magic-number constants), `FrameInfo`, `ProcessingStats`, `RunCancelled` |
| `utils.py` | `safe_print`/print helpers, `header_get_first`, `parse_timestamp` (the one home for Origin's `-0700` offset), `disable_astropy_network`, `embed_to_shape`, `native_status`, `read_version`, `should_check_for_update`, `tqdm` (the one guarded import; plain iterator without it) |
| `io_fits.py` | FITS load/save, `load_frame` (the single format dispatcher), `make_master`, `populate_fits_header`, preview rendering (`render_preview_float`, white-point floor; stretches linear/arcsinh/ghs/masked/multiscale/seti/noisefloor/oklab/clahe), `preview_stretch_kw(args)` (the one place preview settings are read from `args`) |
| `io_raw.py` / `io_tiff.py` / `io_xisf.py` / `io_ser.py` | Camera RAW (rawpy), TIFF (tifffile), XISF 1.0 (no dep), SER video (no dep; one file -> virtual frames `path::index`) |
| `xisf_writer.py` | Minimal XISF 1.0 writer |
| `frame_discovery.py` | `discover_frames`, `classify_frame`, `select_matching_darks` |
| `robust_pca.py` | Robust-PCA masters (`--master-method robust_pca`), `--flat-from-lights` (downsampled per CFA plane) |
| `dark_temp_model.py` | Per-pixel dark-vs-temperature polynomial (`--dark-temp-model`) |
| `vignette_calib.py` | Per-instrument vignetting map, apply side (`--vignette-map`; built by `tools/build_vignette_map.py`) |
| `debayer.py` | Calibration, hot pixels (Bayer statistical fix with star-support rule), RCD (default) / Malvar / Menon debayer, white balance, session-constant CFA equalisation (`--no-session-cfa-eq`), `--spike-reject`, `luminance()` (0.299/0.587/0.114, native for f32; use it rather than writing the expression out) |
| `banding.py` | Row/column banding removal (`--banding-removal`) |
| `trail_reject.py` | Satellite/aircraft trail rejection (`--trail-reject`), Phase 1 |
| `local_normalize.py` | Per-frame additive background matching before rejection (`--local-normalize`). Still exists; only the old Phase 4 local-normalisation *step* was removed |
| `quality.py` | `compute_quality_metrics`, FWHM, `estimate_bortle` (heuristic) |
| `star_detect.py` / `matched_filter.py` | `detect_stars_matched_filter` (the detector; native + numpy); point-source matched filter (`--matched-filter`) |
| `quality_sweep.py` | `--quality-sweep`: score every light in a folder tree |
| `frame_processor.py` | Phase 1 workers, `execute_frame_processing`, `quality_gate`, `phase1_uses_gpu` (`--gpu-phase1`), `phase1_layout`, pre-gradient removal |
| `frame_store.py` | Per-session frame arrays in RAM (shared memory) or temp memmap (`--frame-store auto|ram|disk`) |
| `registration.py` | `calculate_shift`, pyramid + FFT shifts, affine/RANSAC, `registration_stars`, residual gate, `run_registration_phase`, `fit_displacement_field` (`--elastic-registration`) |
| `affine_fit.py` / `phase_correlate.py` / `blind_match.py` | Rigid RANSAC; subpixel phase correlation; blind unknown-rotation star match |
| `transparency.py` | Per-frame transparency (`--transparency-min`); `to_aligned_yx` is the inverse of `apply_transform` |
| `distortion.py` | Radial distortion model (`--distortion-model`); null result on Origin data |
| `stacking.py` | Combines (sigma-clip, percentile, ESD, linear fit, IVW, wavelet, median), drizzle (resample / `--drizzle-method splat`, kernels lanczos/psf/magic, `--super-res-iters`), lacosmic, online sigma-clip, `run_stacking_phase`; the numpy combines share one tile driver (`_tile_grid`, `_combine_tiles`) |
| `proper_coadd.py` | Zackay-Ofek proper coaddition, on by default (`--no-proper-coadd`) |
| `cfa_drizzle.py` | Bayer drizzle (`--cfa-drizzle`, opt-in, not recommended for Origin data) |
| `full_field.py` | Extend the stack beyond the common crop (`--full-field`, on by default at FRAC 0.5; `--no-full-field` / `--full-field 0` off; skipped silently with drizzle, comet mode, elastic registration, uncertainty map) |
| `merge.py` | `--merge` and the hierarchical combine (`merge_previous_stacks`) |
| `mosaic.py` | Multi-panel stitching (`--mosaic`) |
| `stream_stack.py` / `live_stack.py` | Two-pass O(1)-memory stack (`--stream`); real-time stacking (`--live`) |
| `noise_validation.py` / `noise_model.py` | Odd/even noise maps (`--noise-validate`); measured gain + correlated-noise model for photometry |
| `moving_objects.py` | Mover detection / tracked stack (`--moving-objects[-stack]`) |
| `session_report.py` / `session_info.py` | Tracking/trend report (`--session-report`); parse capture-app `info.json`, stack WCS |
| `dither_report.py` / `aberration.py` | Dither coverage (`--dither-report`); field aberration/tilt (`--aberration-report`) |
| `postprocess.py` | Phase 4 chain `postprocess_stack`, `_postprocess_early` (cacheable early steps); `StepSnapshots` + `PHASE4_TOGGLES` (desktop app: switch steps off/on after a run, resuming at the first changed step) |
| `background.py` | Mesh/DBE background (DBE and `--bg-method wavelet` share `_patch_background_extraction`; only the surface fit differs), sky residual, edge bands, `gaussian_filter_ds`, `gaussian_blur_spatial` |
| `denoising.py` | Directional wavelet (`--denoiser wavelet`, `--wavelet-protect`; `curvelet` alias), ACDNR, bilateral, aniso, `shrink_stars` (the star-reduction step), multiscale local contrast, chroma NR |
| `wavelet.py` | Native-backed `wavedec2`/`waverec2` |
| `nn_denoise.py` | `--denoiser nn`: Noise2Noise U-Net on native `nn_conv3x3` (weights `src/data/nn_denoise.npz`), input in noise units, low-frequency restore (`nn_restore_sigma` 4 px), sets `args._preview_noise_scale`. `--deconvolve nn`: same network, `src/data/nn_deconv.npz` (Pan-STARRS truth), restore `nn_deconv_restore_sigma`, then mixed with the nn denoiser (`nn_deconv_blend`: 0.75, `--auto` galaxies 0.35); turns the other denoisers off and marks them explicit. Stars outside the trained width (`TRAIN_FWHM_RANGE`) are resampled first (`measure_star_fwhm`, `nn_scale_normalise`); narrow stars get a star-aware restore (`nn_deconv_star_restore_frac`, faded out 3.6-4.0 px). `--auto` picks it (`auto_settings._apply_nn_deconvolution`); off Origin it is tentative and `postprocess._apply_auto_nn_guard` reverts it when `nn_stack_guard` trips (no stars / <30 usable / FWHM outside 1.1-14 px / ellipticity >0.35). Score star colour on the linear output (`bench_phase4` `*_lin`). Detail, tools and results: [module-notes](dev-notes/module-notes.md#detail-moved-out-of-claudemd), [nn-deconv-runbook](dev-notes/nn-deconv-runbook.md) |
| `psf_deconvolution.py` | PSF estimation; RL / RL-SV / TV / sparse FISTA (`--deconvolve`) |
| `sky_model.py` | EXPERIMENTAL physical sky model (`--bg-method physical`); `remove_physical_sky_with_reason` declines on narrow fields -> DBE |
| `star_removal.py` / `star_repair.py` | `--remove-stars` (opt-in), `--starless-process`, `--layered-stretch`; saturated-core repair (`--repair-stars`) |
| `star_n2n.py` | `--remove-stars-method nn` (EXPERIMENTAL): Noisier2Noise star removal on the linear stack (weights `src/data/nn_starless.npz`); post steps `constrain_star_layer`, `guard_holes`, `remove_residual_cores`, `fill_saturated`. Train/score tools and results: [star-n2n runbook](dev-notes/star-n2n-runbook.md) |
| `source_separation.py` / `exposure_fusion.py` | NMF star/nebula split (`--nmf-separate`); Mertens fusion (`--hdr-blend-mode fusion`) |
| `atmospheric_dispersion.py` | EXPERIMENTAL dispersion correction (`--fix-atmospheric-dispersion`) |
| `uncertainty.py` | Monte Carlo Phase 4 uncertainty (`--uncertainty-propagate`), confidence map, `--error-aware-stretch` |
| `color_calibrate.py` / `photometric_calibration.py` | Gaia colour calibration on the linear stack, on by default (`--no-color-calibrate`, methods solar/colorindex/spcc); gray-locus white balance |
| `channel_combine.py` | `combine` subcommand: LRGB/SHO/HOO, SCNR, `--continuum` subtraction |
| `difference_imaging.py` / `transient_triage.py` | ZOGY subtraction + detection (`--transient-detect REF.fits`); CNN triage (`--transient-triage`; numpy forward pass reading the bundled ONNX, native tract kernel in a `triage`-feature build) |
| `photometry_core.py` | `aperture_photometry_batch`, WCS helpers (`_celestial_wcs`, `_pixel_coords`, `_field_centre_and_radius`) shared by photometry, colour calibration, annotation, mosaic |
| `camera_profile.py` | Camera from the FITS `CAMERA` keyword; per-model constants in `src/data/camera_profiles/<model>.json` (raw gain per ISO, colour slopes), gain checked on two of the session's lights before photometry (`resolve_camera` -> `args._camera`, `verify_gain`). Details: [camera-profile](dev-notes/camera-profile.md) |
| `photometry.py` / `photometry_timeseries.py` / `lightcurve_analysis.py` / `gain_ptc.py` | `--photometry`; `--photometry-timeseries`; `--lightcurve-analysis`; PTC gain from bias/flat pairs |
| `observing_geometry.py` | Alt/az, airmass, parallactic angle (closed-form fallbacks, no IERS needed) |
| `plate_solve.py` / `local_solve.py` | `--plate-solve` backend dispatch; built-in Gaia-tile solver |
| `annotation.py` | `--annotate` (SIMBAD) |
| `net_query.py` | All HTTP (Gaia/VizieR/SIMBAD/astrometry.net/Horizons/update check); `--offline` guard |
| `gaia_cache.py` | Local cache of brightest-first Gaia cone queries (`star_index_dir()/gaia_cones`); answers only when the cached rows are the server's own answer (cone contained, entry untruncated or still >= n rows, float32 G for the `min_mag` cut); misses fetch a 5%-wider cone with 25% more rows so the next night hits |
| `target_inference.py` / `auto_settings.py` | Target from headers/folder/SIMBAD; `--auto` advisor (blended presets) |
| `checkpoint.py` / `cleanup.py` | Checkpoint/resume + `stack_fingerprint`; temp-file registry |
| `pipeline.py` | `stack_target` (wires the phases), `postprocess_from_stack` (`--from-stack`), `save_final_preview` (writes a run's preview JPEG and keeps its source for the desktop re-stretch) |
| `cli.py` | `process_directory`, `parse_args`, `main`, `_build_masters`; removed flags are hidden `_RemovedFlag` actions |
| `health_check.py` | `run_health_check` |
| `ui_events.py` / `desktop_control.py` / `desktop_app.py` / `native_dialog.py` / `notify.py` | Desktop app: event sink polled by tkinter (`remember_final_preview`/`restretch`: pick another stretch for the finished run's JPEG), form-from-argparse + `RunManager` (cooperative cancel), the tkinter window, error dialogs, balloon notifications |
| `i18n.py` | Desktop-app translations: `_()`/`N_()`, catalogs in `src/locales/<code>.json` keyed by the English text, language from `$ORIGINSTACK_LANG` / the saved menu choice / the OS |

### Four-phase pipeline
1. **Phase 1 — Process & quality** (`frame_processor.py`): load, calibrate, hot pixels, debayer, white balance, quality metrics, patch scores, in parallel; hard limits, outlier and percentile gates.
2. **Phase 2 — Registration** (`registration.py`): pyramid seed + FFT subpixel shift, optional affine via star matching + seeded RANSAC (validated: shift < 10% of frame, rotation < 5 deg), shift-outlier rescue by blind match, residual gate (relative fallback so it can't reject a whole session), optional elastic field.
3. **Phase 3 — Stacking** (`stacking.py`): warp into the common crop and combine; proper coadd by default.
4. **Phase 4 — Post-processing** (`postprocess.py`), in order: (1) per-channel hot pixels; (2) star detection (one mask reused); (3) DBE/mesh background; (4) chroma NR; (5) sky floor; (7) sky residual; (8) sky pedestal; (9-11b) denoiser — bilateral / ACDNR / aniso (kappa = `aniso_kappa_sigma` x sky sigma) / directional wavelet (default); (12) `--deconvolve`; (13) star reduction (`shrink_stars`, `--no-star-reduce`); (14) multiscale local contrast (`--no-local-contrast`); (15a) edge bands; (15) final sky flatten + neutralise. Steps are skippable via `--skip-step NAME`. The preview stretch is per-channel by default (`--stretch-color channel`); `--stretch-color preserve` applies the curve to luminance and keeps RGB ratios.

### Rules that must hold

**Settings, checkpoints, `--from-stack`**
- `stack_fingerprint` hashes lights + Phase 1 (`p1`) and Phase 2-3 (`p23`) settings, taken before `--auto` mutates `args`. **A new Phase 1-3 attribute not in those CLI groups must be added to `_P1_EXTRA`/`_P23_EXTRA`** (Phase-4-only members of those groups go in `_P23_IGNORE`), or changing it silently reuses a stale checkpoint.
- `--from-stack` runs Phase 4 on a previous run's linear FITS with its `<stem>_config.toml`; never writes the input. The early-step cache key is collected from `_postprocess_early`'s own source (`_early_args_read`); a new local crossing the `_postprocess_early` boundary must join its return value. Names read only for side-file paths go in `_EARLY_KEY_IGNORE` (`output` was keyed: every new `-o` missed); a step that writes a side file there must make `_early_cache_usable` false. `save_effective_config` writes lists too (`skip_step`; dropping them re-ran skipped steps on `--from-stack`).
- Saved configs bake in `--auto`'s derived choices (e.g. `galaxy_mode`) as if user-set; re-running from a config is not the same as re-running `--auto`. `--auto` runs after Phase 1, so its presets must not set a Phase 1 attribute (`pre_gradient_removal` did: it never ran under `--auto`, only from the saved config; `TestNoPhase1Settings`).
- `--auto` mutates the one shared `args`; `cli._snapshot_args`/`_restore_args` reset it per target, copy mutable containers (`_copy_arg_value`) and delete attributes a target created. Accumulating settings (e.g. `skip_step`) leak otherwise. Tested by `tests/test_auto_state_isolation.py` against the real helpers.
- `_want_combine_sessions` honours a `combine_sessions=True` *value* (saved configs), not only argv.

**Linear data**
- The main output FITS is the linear pre-Phase-4 stack (`RAWSTACK=True`, `NFRAMES`/`INTGTIME`/`TOTEXP`). `--merge`, `--transient-detect` and `--from-stack` refuse inputs without `RAWSTACK`. Photometry, colour calibration and difference imaging run on the linear stack, never the post-processed one.
- `--merge`: flux-match previous stacks to the current one (gain from pixels bright in both; sky offset as a robust quadratic surface fitted to block medians of `ref - gain*img`, `_sky_offset_surface` -- a constant offset left seams and a red rim on a five-session Lagoon combine; clipped cores neutralised after the mean, `_clipped_weight(grow_px=_CLIP_GROW_PX)`), inverse-noise-variance weighted mean per footprint, `embed_to_shape` before registration, headers summed; no `--drizzle-scale > 1`. Hierarchical combine uses the same `merge_previous_stacks` (rotation-aware), reference = most `INTGTIME`; Phase 4 runs once on the combined stack (`args._defer_phase4` for the per-session runs).
- Multi-session pooling vs split is decided from predicted avoidable field rotation (`_predict_rotation_spread`, 3 deg); lights come from `discover_frames`, not a glob.

**Uncertainty / Phase 4**
- Any Phase 4 step that writes a file or calls the network belongs in `uncertainty._QUIET_OFF`, or each Monte Carlo realization writes it silently.
- A new resumable Phase 4 step after the early block needs `_at('name', enabled)` as the first test of its `if` in `postprocess_stack`, an entry in `SNAP_STEPS` (run order) and `PHASE4_TOGGLES`, and a label in `desktop_app._PP_STEP_LABELS` (+ catalog entries). `tests/test_phase4_snapshots.py` checks a resumed run is bit-identical to a full run; state a step leaves in a local other than `stacked`/`_dn_stars`/`args._preview_noise_scale` is not in the snapshot.
- `reduce_chroma_noise` (and anything right after DBE) must not clip negatives: the sky is centred on zero there.
- DBE samples each channel against its own sky (`background._channel_sky`), not the luminance sky: on a colour-calibrated stack G/B sat far above it and fell back to the mesh.
- Stretch one image; never recombine separately stretched layers.
- Never rebuild RGB as `img * new_lum / lum` on sky-centred data: lum ~ 0 with channels of opposite sign gave 1e18 in `multiscale_local_contrast` (Orion); fade to an additive change near zero.
- The preview white point (p99.5) clips star cores by design; an *extended* region above it (largest connected region >= `Config.PREVIEW_ROLLOFF_MIN_AREA`) is rolled off instead (`io_fits._preserving_preview`). Gate on connected area, not blurred area fraction (a dense star field passes the latter); 0.02% of the frame (M57 0.037%, star cores <= 0.012%). It must reach every `--stretch-color` path: wired to `preserve` only, it silently stopped when the default went back to `channel`. The faint-wing lift that comes with it is for nebulae only (`black_sigma <= 0`).
- Preview JPEGs take their settings from `io_fits.preview_stretch_kw(args)` (the run's main preview via `pipeline.save_final_preview`); hand-built kwargs drifted: the live/stream/mosaic/comet/annotated previews left out `color` and rendered `preserve` after the default became `channel`.

**Streaming memory**
- Never hold all frames in memory unless `frame_store` decided they fit (available RAM minus 20% of total minus worker reserve). Output must be bit-identical across RAM/disk placement (`tests/test_frame_store.py`). Shared-memory arrays are opened in workers with `track=False`. A combine over a disk-backed aligned stack goes through `stacking._combine_in_bands` (per-pixel kernel, so bit-identical): one native call over a 60 GB memmap ran silent for minutes and paged the machine. Long per-frame loops report through `ui_events.progress` and check `args._cancel_event`; an `except Exception` fallback around them must re-raise `RunCancelled`. `frame_store` refuses an array the temp disk cannot hold (`FrameStoreSpaceError`) -- the sparse file would otherwise fail only when written.

**Native (Rust) kernels** ([ext/astro_native/](ext/astro_native/), details in [dev-notes/native-kernels.md](dev-notes/native-kernels.md))
- Layout: one module per domain under `ext/astro_native/src/` (`combine`, `warp`, `filters`, `hotpix`, `rcd`, ...; the list is in the `lib.rs` doc comment). Items are `pub(crate)` and re-exported from `lib.rs`, each module starts with `use crate::*;`, and a new `#[pyfunction]` is registered in the `#[pymodule]` at the bottom of `lib.rs`. Shared helpers: `stats` (`median_inplace`, `select_middle`, `median_f64`, `median_f32_as_f64`, `sigma_clipped_median_f32`: add order statistics there, not as a local copy), `parallel::par_rows` (one task per output row), `avx2_twin!` (run-time AVX2 dispatcher over a portable body; hand-written intrinsic twins keep multiply and add separate). Every `unsafe` carries a `// SAFETY:` comment, every `unsafe fn` a `# Safety` doc.
- Every kernel needs a numpy mirror (fallback) and a parity test in `tests/test_native.py` (auto-skips without the module); OS004 lints for test references. Transient triage's native kernel sits behind the Cargo feature `triage` (off by default; `maturin build --release --features triage`); its fallback is `transient_triage._score_numpy`, which reads the ONNX weights itself and supports only the `TriageNet` architecture.
- Default is bit-identity with the numpy path: same f32 ops in the same order, no FMA; numpy 2 does `f32_array * python_float` in f32, so round scalars to f32 first; `np.median` of an even count is `(a+b)/2` in f32. Where identity is impossible (parallel reductions, histogram edges), say so and test with tolerance.
- **Measure numpy's real baseline before porting.** Ports win when numpy/scipy does something wasteful (Python loops, full-frame work for a sparse result, needless copies), not by default.
- Releases build for baseline x86-64; never ship `-C target-cpu=native`. Use run-time `is_x86_feature_detected!` twins (no FMA) for SIMD.
- `panic = "unwind"` in the release profile is load-bearing when built with `--features triage` (tract parser panics are caught and become `RuntimeError`).
- Bump `ext/astro_native/Cargo.toml`'s version when the crate source changes (OS005). After swapping a wheel, **confirm `astro_native.__version__`**. In Git Bash, `$TEMP` is a backslash path and `pip install $TEMP/...*.whl` does not expand — use a POSIX path.
- When shimming `_native` in a test, capture the real module first: a wrapper delegating to itself recursed, fell back to scipy silently, and compared against the wrong resampler.

**Experiments and profiling**
- Spawned pool workers don't inherit monkeypatches; an experiment hook must load in every process (`sitecustomize.py` on `PYTHONPATH`).
- `--from-stack` (and `tools/bench_phase4.py`) reuse `<stem>_phase4cache.pkl` while the *settings* match: an experiment that patches a `Config` constant of an early step (DBE, sky) gets the cached result. Delete the cache between such runs.
- Profile with nothing else running; this machine's timings move up to ~40% between identical runs.
- Compare stacks against Siril with shared-star widths on the same stars, per channel, at matched resolution (`tools/bench_vs_siril.py`, `--os-args=...` needs the `=`); never compare single-frame FWHM with stack FWHM.

**Packaged app / network**
- The frozen exe has no IERS tables (`packaging/originstack.spec` strips `astropy_iers_data`): anything on the default path must not need `Time.sidereal_time("apparent")` or `AltAz` (use `sky_model`/`observing_geometry` closed forms). A fail-soft `except` there hides the divergence.
- All HTTP goes through `net_query`'s `_http_*` helpers (the `--offline` guard). The default run does query SIMBAD; keep the README "Network use" table accurate.
- Every desktop-app failure path goes through `_fatal()` (windowed exe has no console). Logging uses `logging.getLogger("originstack")` (OS001); optional imports in `src/` are `try/except`-guarded (OS002); library code uses `safe_print` (OS003).
- `save_effective_config` writes strings through `_toml_str` (Windows paths).

**Translations (desktop app and website)**
- Every user-visible string in `desktop_app.py` goes through `_()` (or `N_()` in a module-level table, translated where shown). Keys stay English: never translate a value that is submitted, compared or used as a dict key (group titles, dests, choices). The log, progress text and the CLI stay English on purpose.
- The Setup form translates at display time: group titles, field labels, summaries and full help are looked up by their English text, so changing an option's help in `cli.py` makes its translations stale. `python tools/i18n_strings.py` shows coverage per language, `--missing <code>` lists what needs translating; `python tools/check_translations.py` must pass (placeholders and HTML tags kept). Missing text falls back to English.
- The English pages in `docs/` are the website source. After editing `index.html`, `stack-celestron-origin.html` or `privacy.html`, run `python tools/build_site_i18n.py` to regenerate `docs/<lang>/`, the switcher, `hreflang` alternates and the sitemap (a test fails when they are stale). New or changed paragraphs need `i18n/site/<code>.json` entries (`--missing <code>`), or they stay English on the translated pages. Never edit `docs/<lang>/` by hand.
- Translations were machine-drafted (2026-10) and not reviewed by native speakers.

**Domain conventions worth knowing**
- Origin FITS have RA increasing with +x (not textbook east-left); `info.json`'s solve describes the session's *first* sub. `pipeline._settle_stack_wcs` maps it through that sub's transform + crop, then refines with `local_solve` (`--no-wcs-refine`); the result is `args._stack_wcs`. A WCS on the `(3, H, W)` cube must be built with `naxis=2`.
- Catalogue TAP queries must be ordered (`ORDER BY phot_g_mean_mag`); `TOP n` alone returns an arbitrary subset per call.
- Origin `EGAIN` is ~5x too high; photometry uses the measured gain from `noise_model` / PTC.
- Registration RANSAC is seeded (`_RANSAC_SEED`) so two identical runs are bit-identical. Session reports use the frame *centre's* displacement, not the transform's corner translation.
- A flat's `DATE-OBS` is `'0-00-00T00:00:00'`; take light headers from `discover_frames`. A FITS `TIMEZONE` is validated as `+/-HHMM` before use.
- Mono SER/XISF are replicated to 3 channels at load; a mono TIFF is ambiguous and treated as Bayer unless a `.json` sidecar gives `bayerPattern`.
- `--use-gpu` stays opt-in: no gain measured on a 4 GB card; the native Lanczos-3 warp beats the GPU warp whenever it can run, and Phase 1 uses the CPU pool unless VRAM allows `os.cpu_count()` GPU workers (`args._phase1_gpu`). Any array from the GPU path must go through `to_host` before numpy-only calls (a bare `except` hid this once). History: [module-notes](dev-notes/module-notes.md#detail-moved-out-of-claudemd).
- Comet mode (`registration.run_comet_registration_phase`) finds the comet by its motion against the stars and keeps each frame's star rotation; `_comet_blended` subtracts the comet's modelled smear from the star stack and adds the comet once (`stacking.blend_comet_star_stacks`). Detail: [module-notes](dev-notes/module-notes.md#detail-moved-out-of-claudemd).
- Cancel is cooperative (`args._cancel_event`, checked between Phase 1 frames and between targets); pools are shut down with `cancel_futures=True`.

### Tried and reverted — don't retry blind
Details in the linked notes.
- Fused native `compute_quality_metrics`/`validate_image_data` stats via one sort: slower than numpy's partitioning ([native-kernels](dev-notes/native-kernels.md)).
- Sort-based MAD sigma-clip / sort-once u32 keys for patch-weighted combine: 13-45% slower than quickselect.
- Windowed selection predicted from the neighbour pixel; radix-histogram median: no faster.
- Native rustfft `rfft2_batch` for proper coadd; Moffat closed-form spectrum; batched/AVX2 coadd accumulate; `FILE_ATTRIBUTE_TEMPORARY`; Phase 1 arrays in RAM; file writes instead of memmap slot assignment.
- Phase 1 write-behind (a writer thread per worker) and reused per-worker output slots: no gain, Phase 1 is CPU/memory-throughput bound on 8 cores; more workers than physical cores is slower ([native-kernels](dev-notes/native-kernels.md)).
- Pre-touching frame-store pages from a parent thread: a worker still faults every page in its own view (60-99 vs 68-72 ms per 75 MB slot; floor 10 ms) ([pipeline-notes](dev-notes/pipeline-notes.md)).
- Measuring proper-coadd PSFs inside the alignment loop (GIL contention).
- Subsampled sigma-clipped medians for CFA equalisation (period-4 structure biases them).
- RCD per-Bayer-site stages (stops vectorising); RCD strips without a buffer pool.
- `--cfa-drizzle` as a default; kernel-regression (Handheld Multi-Frame Super-Resolution) merge of the Bayer samples: stars 8-45% wider and 1.3-2.9x noisier than RCD + proper coadd at matched width; demosaic-free luminance ([siril-comparison](dev-notes/siril-comparison.md)); ubercal flat from stars, single- and multi-night -- unconstrained on the 13 local sessions (<= 0.4 deg rotation per session, <= 0.5 deg between nights); untested on strongly rotating data such as the Lagoon set ([native-kernels](dev-notes/native-kernels.md)).
- Per-frame lacosmic as a noise reducer (it was smoothing; fixed noise model now) ([siril-comparison](dev-notes/siril-comparison.md)).
- Proper coadd outside the core in `--full-field` (normalised convolution, per-tile, MTF filter, Wiener) ([module-notes](dev-notes/module-notes.md)).
- Widening `--use-gpu` coverage (fewer VRAM-bound workers); GPU stays opt-in.
- Removed denoisers NLM, BM3D, MMT, non-adaptive wavelet + Noise2Self calibration ([module-notes](dev-notes/module-notes.md)).
- Physical sky model on ~1 deg fields (declines by design).
- Starless-layer deconvolution and separate-layer stretch.
- AVX2 lanes for the Lanczos-3 warp row (one `[f64; 4]` lane per channel under `avx2_twin!`): bit-identical, no faster (probably the fixed-order dependent add chains, not lane width; interleaving two output pixels is untried) ([native-kernels](dev-notes/native-kernels.md)).
- Camera-wide CFA equalisation prior: 2x2 green offsets vary 5-15 ADU between sessions, 0.4 ADU within one ([camera-profile](dev-notes/camera-profile.md)).
- Per-unit bad-pixel / vignetting library: the session dark already holds the recurring bad pixels; a leave-one-out vignetting map gains 0-5% after a quadratic ([camera-profile](dev-notes/camera-profile.md)).
- Sharper learned deconvolution from more of the same data: a stricter seeing cut (top 10% is 3% narrower than top 30%), cross-night pairs (x1) and a 2-4x wider U-Net all land on v3's width-noise curve when compared at equal noise (`tools/eval_nn_matched_noise.py`); run-to-run spread ~1.5% ([nn-deconv-runbook](dev-notes/nn-deconv-runbook.md)).
- `--flat-from-lights` under `--auto` for sessions without flats: the target leaks into the low-rank flat unless it moves by more than its own size; removed ~95% of M31's disk on an S50 Pro ([module-notes](dev-notes/module-notes.md)).
- Variance-vs-signal slope for raw Origin gain: the stepped raw values make it quantisation-dominated; use the two-point ratio (`camera_profile.raw_pair_gain`).

### Native (Rust) build
```bash
# into a virtualenv:
cd ext/astro_native && maturin develop --release
# system Python (no venv): build a wheel and install it
cd ext/astro_native && python -m maturin build --release
pip install --force-reinstall target/wheels/astro_native-*.whl
# optional transient-triage model support (tract ONNX runtime; off by default)
python -m maturin build --release --features triage
```
PyPI wheels (Windows x64, Linux x86_64 manylinux, macOS arm64 and x86_64, plus an sdist) come from `.github/workflows/native-wheels.yml`: it builds and smoke-tests on every PR that touches `ext/astro_native/`, and publishes only on a `native-v<Cargo version>` tag (or a manual run with `publish`), through the `pypi` environment with trusted publishing. The crate version lives in `Cargo.toml` only (`pyproject.toml` reads it).
The startup banner reports native status (`native_status()`); accelerated steps log `[rust] ...`. Benchmark kernel changes with [tools/bench_native.py](tools/bench_native.py).

### Optional dependencies
`tqdm`, `Pillow`, `psutil`, `cupy`, `rawpy`, `tifffile`, `astro_native` — each guarded, features degrade gracefully.

### Parallelism
`-j N` / `--parallel N`: `0` auto (single-threaded processes on 3/4 of the physical cores: Phase 1 is memory-bound, 6 x 1 beat 8 x 2 by 7-10% on an 8-core laptop; `frame_processor.phase1_layout`), `1` sequential, `N` = N single-threaded processes. `ProcessPoolExecutor` for CPU, `ThreadPoolExecutor` for GPU/thread paths.

### Packaging the desktop app (Windows)
```powershell
.\packaging\build_windows.ps1
.\packaging\verify_build.ps1   # launches the exe, checks the window, astro_native loaded, clean shutdown
```
PyInstaller onedir build in `packaging\dist\OriginStack\` plus `OriginStack-<VERSION>-windows-x64.zip`. `VERSION` (repo root) is read by `read_version()`; bump it with each release tag. `ext/astro_native` must be installed as a real wheel (not `maturin develop`) for PyInstaller to see it; `build_windows.ps1` does this. Release builds do not include the `triage` feature. `.github/workflows/release.yml` attaches the zip to an existing `v*` release. See [packaging/README.md](packaging/README.md).
