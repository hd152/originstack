# OriginStack — Architecture & Feature Reference

## Project Overview

OriginStack is a professional-grade Python pipeline for stacking and post-processing astronomical FITS images. It was designed for the Celestron Origin smart telescope and runs efficiently on ordinary laptop hardware using a streaming memory architecture: frames are loaded, processed, and freed one at a time, keeping peak memory constant regardless of total frame count.

For installation, quick start, and common recipes, see [README.md](README.md).

---

## Module Inventory

| Module | Lines | Contents |
|--------|-------|----------|
| `src/gpu_context.py` | 270 | `GpuContext`, CUDA stream contexts, `get_gpu()` singleton |
| `src/models.py` | 215 | `Config`, `FrameInfo`, `ProcessingStats` |
| `src/utils.py` | 214 | Print helpers, `format_time`, `get_memory_usage_mb`, `native_status()` |
| `src/io_fits.py` | 575 | FITS load/save, `load_frame` (format dispatcher), `make_master`, `populate_fits_header` |
| `src/robust_pca.py` | 224 | Robust PCA (Principal Component Pursuit) master calibration frames (`--master-method robust_pca`); same decomposition backs `--flat-from-lights` — see feature 35 |
| `src/dark_temp_model.py` | 124 | Temperature-interpolated dark current model (`--dark-temp-model`) — see feature 34 |
| `src/io_raw.py` | 204 | Camera RAW load (rawpy) — CR2/CR3/NEF/ARW/DNG/ORF/RW2/RAF/PEF/3FR/MRW/X3F/IIQ |
| `src/io_tiff.py` | 88 | TIFF load (tifffile) |
| `src/io_xisf.py` | 192 | XISF 1.0 load (hand-rolled, no dependency) |
| `src/io_ser.py` | 200 | SER (planetary video) load — one file expands to many virtual frames |
| `src/frame_discovery.py` | 299 | `discover_frames`, `classify_frame`, `select_matching_darks` |
| `src/debayer.py` | 928 | Debayering (Malvar/Menon2007), hot pixels, white balance, CA correction, `autodetect_bayer_orientation` |
| `src/matched_filter.py` | 55 | Point-source matched filter (`--matched-filter`) — post-stack SNR-optimal detection map, see feature 28 |
| `src/atmospheric_dispersion.py` | 117 | Software atmospheric dispersion correction (`--fix-atmospheric-dispersion`, experimental, not wired into `--auto`) |
| `src/vignette_calib.py` | 77 | Per-instrument vignetting/background calibration map load + apply (`--vignette-map`; built offline by `tools/build_vignette_map.py`) |
| `src/quality.py` | 902 | `compute_quality_metrics`, star detection, FWHM, `estimate_bortle` |
| `src/star_detect.py` | 270 | `detect_stars_matched_filter` — default star detector, native Rust + numpy mirror |
| `src/affine_fit.py` | 202 | `fit_rigid_ransac` — 2D rigid-transform RANSAC, native + numpy |
| `src/phase_correlate.py` | 103 | `phase_cross_correlation` — subpixel FFT registration |
| `src/blind_match.py` | 230 | `match_rigid_unknown_rotation` — blind (unknown-rotation) star match, used by `--merge` |
| `src/psf_deconvolution.py` | 719 | PSF estimation (Moffat/Gaussian), Richardson-Lucy (global + spatially-variant), TV, `sparse_wavelet_deconvolve` |
| `src/background.py` | 1506 | DBE (robust local regression, Rust-accelerated), mesh/wavelet sky extraction, residual removal, exclusion-mask support (`--galaxy-mode`) |
| `src/denoising.py` | 1789 | Wavelet (BayesShrink), MMT, ACDNR, bilateral, NLM, aniso, curvelet-inspired directional wavelet, `--variance-stabilize`, star reduction, local contrast |
| `src/self_supervised_calibration.py` | 103 | Noise2Self-style denoiser parameter calibration (`--denoise-strength-calibrate`) — see feature 36 |
| `src/wavelet.py` | 258 | `wavedec2`/`waverec2` — native 2D wavelet transform (bior1.3 + db4) |
| `src/registration.py` | 2421 | `calculate_shift`, affine/RANSAC, `calc_common_crop`, `run_registration_phase`, `fit_displacement_field` (`--elastic-registration`) |
| `src/stacking.py` | 2607 | Sigma-clip, percentile, ESD, linear-fit, IVW, wavelet-subband, drizzle (Lanczos-3/PSF-matched/Magic Kernel), IBP super-res, `run_stacking_phase` |
| `src/frame_processor.py` | 1494 | Parallel workers, `execute_frame_processing`, `quality_gate` |
| `src/postprocess.py` | 877 | `postprocess_stack` — up to 20-step post-processing chain |
| `src/aberration.py` | 313 | Field aberration/tilt inspector (`--aberration-report`) — see feature 37 |
| `src/dither_report.py` | 112 | Dither-coverage uniformity diagnostic (`--dither-report`) — see feature 38 |
| `src/star_repair.py` | 163 | Saturated star core repair (`--repair-stars`) — see feature 39 |
| `src/star_removal.py` | 128 | Star removal, on by default (`--no-remove-stars`) — see feature 40 |
| `src/trail_reject.py` | 243 | Satellite/aircraft trail rejection (`--trail-reject`) — see feature 41 |
| `src/local_normalize.py` | 118 | Per-frame Local Normalization (`--local-normalize`), pre-combine — see feature 42 |
| `src/live_stack.py` | 339 | Real-time stacking (`--live`) — see feature 44 |
| `src/stream_stack.py` | 498 | Two-pass streaming stack of an already-complete directory (`--stream`) — see feature 43 |
| `src/ui_events.py` | 375 | In-process UI event/state sink for the desktop app — log/phase/progress state, named milestone previews with on-demand re-stretch, per-frame thumbnail ring |
| `src/desktop_control.py` | 220 | Desktop-app control layer — form schema introspection, form-to-argv translation, `RunManager` |
| `src/desktop_app.py` | 1031 | Native tkinter desktop app (`python desktop_app.py`) — Setup form, live progress/log, interactive preview (zoom/pan/re-stretch/wipe compare/thumbnail ring) |
| `src/native_dialog.py` | 48 | Native Windows message boxes via ctypes — fatal-error / close-confirm dialogs for the packaged desktop app |
| `src/notify.py` | 70 | Best-effort native Windows balloon-tip notification (no-op elsewhere) |
| `src/channel_combine.py` | 625 | `combine` subcommand — LRGB/narrowband palettes, SCNR, continuum subtraction |
| `src/exposure_fusion.py` | 153 | Mertens multiresolution exposure fusion (`--hdr-blend-mode fusion`) — see feature 22 |
| `src/source_separation.py` | 148 | NMF star/nebula source separation (`--nmf-separate`) — see feature 45 |
| `src/auto_settings.py` | 943 | Heuristic target classifier, `apply_auto_settings` (`--auto`) |
| `src/target_inference.py` | 670 | Target type inference — heuristic galaxy/nebula/starfield classification |
| `src/health_check.py` | 191 | Frame consistency and calibration quality analysis |
| `src/session_info.py` | 208 | Session metadata reader (Celestron Origin `info.json` — GPS, filter, ISO, orientation) |
| `src/plate_solve.py` | 343 | Astrometry.net + ASTAP + SIMBAD identification |
| `src/net_query.py` | 398 | Direct-HTTP replacements for astroquery services (Gaia, VizieR, SIMBAD, astrometry.net, JPL Horizons) — stdlib urllib, no dependency |
| `src/color_calibrate.py` | 572 | Photometric colour calibration using plate-solved star colours (`--color-calibrate`, `--color-calibrate-method {colorindex,spcc}`) |
| `src/photometric_calibration.py` | 181 | Gray-locus photometric colour calibration (`--photometric-calibration`) |
| `src/annotation.py` | 224 | Object annotation (`--annotate`) — see feature 20 |
| `src/astrollm.py` | 319 | astrollm integration (`--astrollm` / `--astrollm-score-all`) — see feature 46 |
| `src/mosaic.py` | 320 | WCS-based mosaic stitching (`--mosaic`) |
| `src/checkpoint.py` | 327 | Checkpoint save/load for pre-post-processing stack (`--keep-checkpoint`) |
| `src/merge.py` | 279 | Incremental stacking — register + weighted-merge previous linear stacks (`--merge`) |
| `src/quality_sweep.py` | 270 | Collection quality sweep — recursive scoring + reversible flagging (`--quality-sweep`) |
| `src/xisf_writer.py` | 107 | XISF 1.0 format writer (`--export xisf`) |
| `src/cleanup.py` | 47 | Global temp-file registry — auto-remove all registered paths on exit or interrupt |
| `src/pipeline.py` | 1197 | Four-phase orchestrator, `stack_target` |
| `src/cli.py` | 1911 | `process_directory`, `parse_args`, `main` |
| `originstack.py` | 179 | Backward-compatibility re-export shim |
| `desktop_app.py` (root) | 24 | Thin root shim — `from src.desktop_app import main` |
| `ext/astro_native/` (Rust) | 4915 | Optional PyO3/maturin crate (30+ kernels): stacking combines (incl. Linear Fit Clipping, inverse-variance-weighted, online streaming sigma-clip), fused patch-weighted combine, Lanczos-3 warp (alignment + drizzle) and PSF-matched-kernel warp, anisotropic diffusion, L.A.Cosmic, median filter, DBE surface fit + patch sampler, Malvar + Menon2007 debayer, bilateral filter, matched-filter star detection, rigid-transform RANSAC, blind (unknown-rotation) star-pattern match, 2D wavelet transform, BM3D block-matching fallback, hot-pixel fix/replace, robust-PCA Gram-matrix SVD kernels, continuum-subtraction moments (numpy fallback when absent) |

**Total: ~30,000 lines** (Python, `src/` alone; excludes the Rust crate). Tests in `tests/test_core.py` import symbols directly from `originstack`; `tests/test_native.py` covers the Rust kernels (auto-skips if unbuilt).

---

## Core Architecture

### Streaming Memory Model

Frames are never all loaded at once. The processing loop is:

```
load → calibrate → debayer → quality-analyse → shift-calculate → accumulate → free → next frame
```

Peak memory is bounded by 1–2 frames, not the total frame count. This is the fundamental design constraint — never accumulate all frames in memory simultaneously.

**Memory comparison (50 × 4096×4096 float32 RGB):**
- Traditional approach: 10–20 GB
- OriginStack: 0.4–1.2 GB (~13× reduction)
- Tested up to 500 frames with constant memory usage

### Four-Phase Pipeline

```
Phase 1  ──  Frame Processing & Quality Analysis
             Load → calibrate → debayer → quality metrics → reject bad frames
             Parallelised via ProcessPoolExecutor / ThreadPoolExecutor

Phase 2  ──  Registration
             Coarse-to-fine pyramid seed (half-res) + downsampled FFT residual
             correlation with parabolic sub-pixel refinement; optional
             star-match affine/RANSAC for field rotation
             Reference frame selected by quality + pyramid consensus

Phase 3  ──  Stacking
             Align frames → crop to valid overlap region → accumulate → combine
             9 methods: mean, median, sigma_clip, winsorized, percentile, esd,
             linear_fit, ivw, wavelet, plus drizzle super-resolution (any
             rejection method) and streaming/live variants (--stream, --live)

Phase 4  ──  Post-Processing
             up to 20-step chain: background → denoising → deconvolution → contrast
```

---

## Feature Reference

### 1. Frame Discovery & Classification

- Automatically identifies frame types from filenames and FITS headers
- Filename patterns: `light_*.fit`, `dark_*.fit`, `flat_*.fit`, `bias_*.fit`
- Header keywords: `IMAGETYP`, `FRAME`
- Heuristic: zero-exposure frames classified as bias
- Default: unidentified files treated as lights
- **Input formats**: FITS (`.fit`/`.fits`, always), camera RAW (`.cr2`/`.cr3`/`.nef`/`.arw`/
  `.dng`/`.orf`/`.rw2`/`.raf`/`.pef`/`.3fr`/`.mrw`/`.x3f`/`.iiq`, needs `rawpy`), TIFF
  (`.tif`/`.tiff`, needs `tifffile`), XISF (`.xisf`, no dependency), SER (`.ser`, no
  dependency) — freely mixable within one directory. All formats route through
  `src/io_fits.py::load_frame`, the single load dispatcher `make_master` and every other
  loader use. RAW/TIFF/XISF-without-header-info fall back to filename-substring
  classification (`IMAGETYP` has no equivalent in those formats); a `.json` sidecar
  (`_merge_json_sidecar`) can backfill `EXPTIME`/`ISOSPEED`/`GAIN`/`CCD-TEMP`/`BAYERPAT` for
  any format. A `.ser` file's frames are expanded into one virtual `FrameInfo` per frame
  (`path::index`) since a single SER file can hold thousands of frames — see feature 33 below

### 2. Calibration

- **Master frame creation**: `--master-method {median,mean,robust_pca}` combine bias/darks/flats per subfolder (default: median). `robust_pca` (opt-in, needs >= `Config.ROBUST_PCA_MIN_FRAMES` frames) splits the stack into a low-rank component (the true shared pattern — flat vignetting, fixed dark current) and a sparse component (dust motes that shifted between sessions, transient hot pixels) via Principal Component Pursuit — see feature 35. `--auto` narrowly upgrades median→robust_pca per calibration type when its frame count falls in a sweet-spot range
- **Synthetic flats** (`--flat-from-lights`): when no dedicated flats exist, reuses the same robust-PCA decomposition on a capped sample of *light* frames — see feature 35
- **Dark selection**: matches lights by ISO, exposure time, and sensor dimensions; `--dark-temp-model` interpolates dark current across sensor temperature instead of nearest-match selection — see feature 34
- **Calibration order**: bias subtraction → dark subtraction (exposure-scaled) → flat division → optional `--vignette-map` subtraction (per-instrument map built offline from many past sessions; applied per-frame right after debayer, in native sensor pixel space)
- **Smoothing**: masters are Gaussian-smoothed (per-Bayer-channel for flats) to avoid adding correlated noise
- **Hot pixel map**: built from dark frame before smoothing; applied per Bayer sub-channel
- **Flat normalisation**: per Bayer channel; division-by-zero protected

### 3. Bayer Pattern Debayering

- **Auto-detection**: from FITS headers (`BAYERPAT`, `COLORTYP`); `--no-bayer-autodetect` disables a secondary row-orientation check (some capture software writes a `BAYERPAT` that doesn't match the actual row orientation — checked once per session against the reference frame's G1/G2 balance and corrected if it exceeds the sensor-noise range)
- **Supported patterns**: RGGB, BGGR, GRBG, GBRG (default: RGGB)
- **Algorithm**: `--debayer-method {malvar,menon2007}` (default: `malvar`), both native Rust kernels with a numpy fallback, no external dependency. `malvar` — Malvar-He-Cutler, a single fused per-pixel gather. `menon2007` — Menon (2007) DDFAPD directional filtering, higher fidelity on fine periodic detail but ~4-5x slower even natively; a synthetic-astro-frame benchmark shows only a modest gain on typical smooth-sky-plus-point-source astro data, so `malvar` stays the default. Both validated bit-exact against the `colour-demosaicing` reference package. (`bilinear` — pure NumPy — still exists internally as the base function but isn't CLI-reachable; `vng`, formerly OpenCV's Variable Number of Gradients, was removed once it became a pure alias for `malvar`.)
- **Green equalization**: corrects G1/G2 channel mismatch in CMOS sensors
- **Output**: RGB images in (H, W, 3) float32 format

### 4. Hot Pixel Removal

- Per-Bayer-channel MAD-based detection and bilinear interpolation
- Optional pre-built map from dark frame (more accurate for known-bad pixels)
- Applied in raw (Bayer) space before debayering

### 5. White Balance

- `grayworld` (default) — scale each channel to equalise mean
- `whitepatch` — scale channels to 99.5th-percentile white point
- `none` — skip white balance entirely

### 6. Chromatic Aberration Correction

- Realigns red and blue channels to the green channel (sub-pixel phase correlation on a 2x-downsampled estimate)
- **Session-constant**: CA is a property of the optics, so it is measured once on three sample frames and the median shifts applied to every frame; shifts below 0.25 px skip the warp entirely
- Enabled by default; disable with `--no-ca-correction`
- **Software atmospheric dispersion correction** (`--fix-atmospheric-dispersion`, experimental): a separate, unrelated effect — shifts R/B channels toward green's position to correct chromatic atmospheric refraction (worse at low altitude), derived from first principles (Filippenko 1982's refraction formula) since no established software reference implementation exists to port. Off by default, not wired into `--auto`. Needs `--observer-site`, `--parallactic-angle`/`--zenith-angle` or plate-solve-derived equivalents; see `--help` for the full flag set (`src/atmospheric_dispersion.py`)

### 7. Cosmic Ray Rejection (L.A.Cosmic)

- Per-frame rejection using a Laplacian edge-detection noise model (Rust-accelerated)
- Applied in Phase 1 before stacking accumulation
- **Automatic gating**: skipped when >=20 frames are stacked with a rejection method — per-pixel stack rejection removes cosmic rays statistically better than per-frame detection. Force with `--cosmic-ray-rejection`, disable with `--no-cosmic-ray-rejection`; drizzle (no per-pixel rejection) keeps it on

### 8. Quality Analysis & Frame Rejection

Per-frame metrics computed in Phase 1:

| Metric | Description |
|--------|-------------|
| Brightness | Median pixel value |
| Contrast | Standard deviation |
| Star count | matched-filter (native Rust/numpy, src/star_detect.py) |
| FWHM | Median Full Width at Half Maximum of detected stars |
| SNR | Signal-to-noise estimate |
| Quality score | Composite weighted score |

**Rejection behaviour:**
- Percentile-based: default rejects the lowest 50th percentile (`--quality-threshold 50`)
- Hard rejection: brightness < 10 (blank/corrupt), contrast < 1 (flat/underexposed)
- Rejection reason logged per frame when using `-v`

**Quality-weighted stacking** (config keys `weight_snr`, `weight_fwhm`, `weight_stars`):  
Frames with better quality contribute more to the final stack.

### 9. Image Registration

**Translation (default):**
1. Coarse-to-fine pyramid seed — integer-precision, computed at half resolution (the full-res level is redundant with step 2) and shared with reference selection
2. FFT residual cross-correlation on a 2x-downsampled grid (scipy pocketfft, next-fast-length padding) with parabolic sub-pixel refinement — ~0.05 px final accuracy
3. Centroid of thresholded bright regions (fallback)

The post-registration residual check verifies alignment on the riskiest ~20% of frames (largest shifts + a deterministic sample) and escalates to all frames only if any sampled frame fails.

**Affine correction (default on, `--no-affine` to disable):**
- Star catalogue extraction with DAOStarFinder
- Nearest-neighbour star matching
- RANSAC-based affine transform estimation (rotation + scale + translation)
- Handles field rotation from polar alignment error and differential refraction

**Validation and dither detection:**
- Shifts > 10% of image dimension are rejected as unrealistic (translation fallback path)
- The affine fit is independently sanity-checked (shift > 10% of frame size, or rotation > `Config.AFFINE_MAX_ROTATION_DEG` = 5deg) before being accepted; a bad RANSAC star match can converge on a wildly wrong but internally-consistent transform, so this can't be caught by the translation-only guard. Falls back to translation-only registration on failure.
- Post-registration residual check re-detects stars in each aligned frame and measures RMS centroid error (sampled ~20% + riskiest-by-shift frames, escalating to all frames if any sample fails). Frames exceeding `Config.REG_RESIDUAL_MAX_PX` (1.5px) are dropped from the stack by default -- `--no-reg-residual-reject` to keep them (still annotated via `reg_residual_px` in metrics); `--no-reg-residual-check` skips the check entirely.
- Optional elastic (non-rigid) local correction extends this further — see feature 32 below.
- Dither pattern detection → auto-selects sigma-clip stacking method

### 10. Anti-Black-Border Cropping

- Calculates the valid overlap region present in all aligned frames
- Applied after registration, before stacking accumulation
- 2-pixel safety margin
- Crop amount reported in verbose output

### 11. Stacking Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `auto` | Selects percentile (<8 frames) or sigma_clip (≥8 frames) | General use |
| `sigma_clip` | MAD-based iterative rejection, configurable σ and iterations | Most sessions |
| `winsorized` | Like sigma_clip but clips to boundary instead of rejecting | Smooth backgrounds |
| `percentile` | Rejects outside [low%, high%] percentile range | <8 frames |
| `esd` | Grubbs/ESD statistical test | <15 frames |
| `median` | Robust, uses memory-mapped temp files for large datasets | Low frame count |
| `mean` | No rejection, fastest | Pre-calibrated data |
| `linear_fit` | PixInsight-style Linear Fit Clipping — fits a line to each pixel's sorted per-frame values, rejects outliers from the residual, iterates; more robust to non-Gaussian tails than sigma-clip's mean/std test | Non-Gaussian noise, mixed-quality sessions |
| `ivw` | Inverse-variance-weighted combine — Gauss-Markov-optimal linear combiner, weights each frame by `1/noise²` from its own measured background sigma; optional Poisson shot-noise term via `--config ivw_gain`. Does not reject outliers (pair with `--cosmic-ray-rejection`/`--trail-reject`); `--uncertainty-map` additionally writes a `<output>_sigma.fits` per-pixel standard-error sidecar | Frames with very different noise levels |
| `wavelet` | Wavelet-subband stacking — combines each subband separately before reconstructing | Structure-preserving combine |

**Drizzle super-resolution** (`--drizzle-scale 2.0`):
- `--drizzle-kernel {lanczos3,psf,magic}` (default `lanczos3`): Lanczos-3 sub-pixel accumulation (native Rust warp, ~26x vs the scipy path; all channels in one pass); `psf` resamples with the session's own estimated PSF as a Wiener-regularized matched filter (mild built-in sharpening, tune via `--config drizzle_psf_wiener_k`, falls back to lanczos3 if PSF estimation fails); `magic` uses Costella's base Magic Kernel (quadratic B-spline, provably non-negative/ringing-free but softer than Lanczos-3)
- Requires dithered frames to produce true resolution gain
- `--drizzle-pixfrac` controls the tent-kernel pixel fraction (< 1.0 = sharper, noisier)
- `--super-res-iters N`: Iterative Back-Projection (Irani & Peleg 1991) refines the drizzle output for N iterations afterward — forward-simulates what each original frame should look like given the current estimate, back-projects the residual correction. Not yet compatible with `--elastic-registration`

**Pre-combine normalization** (`--local-normalize`): additively matches every frame's background to the per-frame median before the rejection combine — removes per-frame gradients from moonlight/light-pollution drift/thin cloud and sharpens sigma-clip. Applies to rejection stack methods, not plain mean or drizzle. Off by default.

### 12. Background Extraction

| Method | Description |
|--------|-------------|
| `dbe` (default) | Dynamic Background Extraction — Gaussian-weighted robust local regression (Tukey IRLS) over sampled patches; bounded by construction (no runaway extrapolation near bright stars). Patch sampling and surface fit are Rust-accelerated |
| `mesh` | Legacy polynomial grid (faster, less accurate) |
| `wavelet` | Starlet (à trous) multiscale background estimate — `--bg-wavelet-scales N` (default 6) sets the dyadic scale count; structure smaller than roughly `2**N` cells is treated as sky and removed |

**DBE details:**
- Star mask applied to sampling patches to avoid fitting to sources
- **Extended-source (galaxy) exclusion masking** (`--galaxy-mode`): excludes a generous ellipse around the detected galaxy/extended source from background sampling entirely, since a smooth-surface model can't tell a tapering galaxy halo apart from real gradient at its edge. The ellipse is fit to the object's own second-moment shape (largest-area connected blob, not simply the brightest pixel, with edge/corner-artifact rejection) so it tracks real elongation. `--galaxy-mask-radius PX` overrides the fitted semi-major axis; `--galaxy-center X,Y` skips detection entirely and centers the exclusion on a given pixel coordinate (the reliable option on fields where auto-detection picks the wrong object). Auto-enabled by `--auto` for galaxy-leaning targets; the same exclusion mask is honoured by every `remove_sky_residual` pass, not just the first DBE call, including DBE's dense-field fallback
- Sky floor normalisation removes constant per-channel pedestal after background subtraction

### 13. Denoising

The primary luma denoiser is selected with a single flag —
`--denoiser {auto,wavelet,mmt,bm3d,acdnr,nlm,bilateral,aniso,curvelet,none}`
(default `auto`, which resolves to `curvelet` unless a preset/`--auto`
selects otherwise) — and the auto-advisor enforces **one primary**
(precedence BM3D > MMT > wavelet > ACDNR): layering several full-frame
smoothers compounds smoothing without adding selectivity. Chroma noise
reduction is separate and always available. Per-denoiser tuning lives in
the config-file tier (see `--config`).

| Denoiser | Notes | Config keys |
|----------|-------|-------------|
| `curvelet` (`auto` default) | Adaptive BayesShrink DWT (like `wavelet`) but the per-subband threshold is locally reduced wherever a structure-tensor coherence map detects elongated structure (filaments, galaxy arms), protecting it more than an isotropic threshold would. Curvelet/shearlet-*inspired*, not an actual ridgelet/shearlet transform | `directional_protect_strength` (default 0.6; 0 = identical to `wavelet`) |
| `wavelet` | Plain adaptive BayesShrink per subband, luma/chroma split, star-protected, strength auto-tuned from SNR | `denoise_strength`, `denoise_adaptive`, `auto_denoise_strength`, `denoise_chroma_boost` |
| `mmt` | Multiscale Median Transform — robust to Poisson+read noise, best edge preservation (Rust-accelerated median cascade, ~10x) | `denoise_mmt_levels`, `denoise_mmt_strength` |
| `bm3d` | Collaborative filtering, near-optimal, slower (auto-enabled by the advisor when SNR/frame count justify it) | `bm3d_sigma`, `bm3d_stride`, `bm3d_search_window`, `bm3d_group_size` |
| `acdnr` | Contrast-gated sky smoothing — flat sky smoothed, structure preserved | `denoise_acdnr_sigma`, `denoise_acdnr_k` |
| `nlm` | Non-local means (native/numpy fast NL-means, box-filter accelerated) | `denoise_nlm_strength`, `denoise_nlm_blend` |
| `bilateral` | Edge-preserving bilateral filter, joint colour-space weighting (Rust-accelerated) | `denoise_bilateral_sigma_color`, `denoise_bilateral_sigma_space` |
| `aniso` | Perona-Malik anisotropic diffusion (Rust-accelerated, ~37x) | `aniso_iterations`, `aniso_kappa`, `aniso_gamma`, `aniso_option` |
| `none` | Disable luma denoising | — |

**Chroma noise reduction** (default on, `--no-chroma-nr`): Gaussian smoothing of
the chroma channels; fine pass (`chroma_nr_sigma`) plus an optional object-masked
coarse pass for medium-scale colour blotches (`chroma_nr_large_sigma`, auto-set
for galaxy targets).

**Variance stabilisation** (`--variance-stabilize`): applies a generalized
Anscombe transform to the luma plane before wavelet thresholding (both
`wavelet` and `curvelet`), inverting it after — makes BayesShrink's single
per-subband noise estimate valid across the whole brightness range instead
of just near sky level, since shot noise on bright pixels is Poisson, not
Gaussian. Gain/read-noise are self-estimated from the image's own local
mean-variance relationship.

**Self-supervised strength calibration** (`--denoise-strength-calibrate`,
`--denoiser wavelet` only): a Noise2Self-style calibrator masks a small
random pixel subset (replaced with a 4-neighbour average), denoises, scores
the prediction at the masked positions against their true values, sweeps
`--denoise-strength` and picks the minimum — an alternative to the default
SNR-heuristic that needs no ground truth. Verified to select near the true
MSE-minimising parameter on synthetic ground truth before being trusted for
real use.

### 14. PSF Deconvolution (`--deconvolve {off,rl,rl-sv,tv,sparse}`)

- `rl` — Richardson-Lucy iterative deconvolution (GPU-accelerated via cupy FFT under `--use-gpu`)
- `rl-sv` — spatially-variant Richardson-Lucy (`richardson_lucy_svpsf`): the PSF is allowed to vary across the field instead of one global kernel
- `tv` — Total-Variation regularised variant, sharper edges at the cost of speed
- `sparse` — FISTA (Beck & Teboulle 2009), L1-regularised in this project's own wavelet basis (`src/wavelet.py`) rather than TV's spatial-gradient basis; same forward/adjoint PSF convolution and positivity-pedestal pattern as `rl`/`tv`
- PSF model: Moffat (default) or Gaussian; FWHM estimated from detected stars
- Config keys: `deconvolve_iterations`, `deconvolve_fwhm`, `deconvolve_psf_model`, `deconvolve_blind_psf` (empirical PSF from a star median stack), `tv_lambda`, `tv_iterations`, `deconvolve_sparse_lambda`
- Off by default; the advisor gates it on Strehl/dispersion measurements and disables it where RL ringing would hurt (undersampled wide-field stars)

### 15. Star Reduction (default on)

- Reduces star halo prominence for galaxy and nebula targets
- Blurs a copy of the image and blends back over stars identified by the star mask
- `--star-reduce-factor` (blend strength), `--star-reduce-sigma` (blur radius)
- Disable with `--no-star-reduce`

### 16. Local Contrast Enhancement (default on)

- Multiscale local contrast enhancement (MLCE)
- Enhances fine detail without affecting global dynamic range
- `--local-contrast-strength`
- Disable with `--no-local-contrast`

### 17. SCNR — Subtractive Chromatic Noise Reduction

- Suppresses green (or cyan) colour cast in OSC/DSLR images
- `--scnr-amount`, `--scnr-target {green,cyan}`

### 18. Photometric Colour Calibration

- **Gray-locus method** (`--photometric-calibration`): calibrates colours by fitting to a neutral gray locus
- **Astrometry-based** (`--color-calibrate`, needs `--plate-solve`): full photometric calibration using field stars via aperture photometry on Gaia/2MASS catalogues, fits per-channel scale factors. `--color-calibrate-method {colorindex,spcc}`: `colorindex` (default) uses a fixed Gaia BP-RP→B-V formula; `spcc` integrates a blackbody spectrum at each star's Gaia `teff_gspphot` against generic per-channel Gaussian response curves (an honest generic default, not a claim of matching a specific camera's real QE/filter curves), falling back to `colorindex` per-star when no Teff estimate is available

### 19. Plate Solving

- **nova.astrometry.net** (default): cloud-based, requires `ASTROMETRY_API_KEY` (direct HTTP, `src/net_query.py`, no extra package)
- **ASTAP** (`--plate-solver astap`): local solver, no API key required
- On success: writes WCS (CRVAL, CRPIX, CD matrix) to FITS header
- Object identification via SIMBAD database
- Unlocks `--color-calibrate`

### 20. Object Annotation (`--annotate`)

- Circles and labels bright stars and named deep-sky objects (galaxies, nebulae, clusters) on a copy of the preview (`<output>_annotated.jpg`); the main FITS/TIFF/JPG output is untouched
- Needs a WCS solution (`--plate-solve`, or a session `info.json` that already provided one) — skipped with a message otherwise
- Two separate live SIMBAD cone-search queries: stars via a magnitude-limited join (`--annotate-mag-limit`), DSOs via a curated object-type set restricted to Messier/NGC/IC name prefixes (avoids flooding a rich field with thousands of obscure catalog entries)

### 21. Comet Mode (`--comet-mode`)

- Runs two registration passes: one tracking stars, one tracking the comet nucleus
- Saves `<output>_comet.fits` (comet-registered stack) alongside the star-registered stack
- `--comet-xy`/`--comet-search-radius` seed and bound nucleus tracking; `--comet-affine` allows an affine (not just translation) comet-frame fit
- `--comet-blend-sigma` controls blend transition width; `--coma-mask-radius` excludes the coma from star-registration background sampling
- `--comet-radial-renorm` flattens the coma's radial brightness falloff; `--comet-larson-sekanina` applies a Larson-Sekanina rotational-gradient filter to reveal jets/structure (`--comet-ls-rotation` sets the rotation angle)
- `--comet-designation` + `--observer-site` enable JPL Horizons ephemeris lookup for automatic nucleus position seeding

### 22. HDR Combining (`--hdr-combine SHORT_STACK.fits`)

- Blends a separate short-exposure stack into the highlight regions of the main stack
- Recovers detail in saturated bright areas (star cores, galaxy nuclei)
- `--hdr-short-exptime`, `--hdr-long-exptime` specify the exposure times for proper scaling
- `--hdr-blend-mode {threshold,fusion}` (default `threshold`): `threshold` is the original sigmoid-threshold spatial blend; `fusion` uses Mertens multiresolution exposure fusion (Laplacian pyramid weighted by local contrast/saturation/well-exposedness) instead, avoiding the seam a hard/sigmoid threshold can leave at the transition band

### 23. Mosaic Stitching (`--mosaic`)

- Stitches per-subfolder stacks into a single wide-field mosaic
- WCS-based reprojection via the `reproject` package
- Requires plate solving to succeed for each panel

### 24. Incremental Stacking (`--merge`)

- The main output FITS is the **linear pre-post-processing stack** (`RAWSTACK=True`) carrying `NFRAMES`/`INTGTIME`/`TOTEXP` headers
- `--merge PREV.fits [...]` processes only the new session through Phases 1-3, registers each previous stack onto the new grid (blind rigid star-pattern match first, `src/blind_match.py` — nights differ by arbitrary field rotation on alt-az mounts, and this makes no assumption about the angle — translation-seeded star-catalog RANSAC and translation-only fallbacks), and combines as a per-pixel `NFRAMES`-weighted mean inside each warped footprint
- Phase 4 runs once on the merged result; header aggregates are summed, so the output chains into future merges
- Guards: hard error on registration failure, <25% footprint overlap, or <0.15 aligned-luminance correlation (wrong-target protection); refuses non-linear inputs and `--drizzle-scale > 1`
- No cross-session outlier rejection (each session already rejected internally)
- Coalesces with `--keep-checkpoint`: the checkpoint stores the session-only stack, so a resumed run re-applies the merge idempotently (~45 s post-processing iteration on a merged stack)

### 25. Desktop App (`python desktop_app.py`)

- Native tkinter window (stdlib, no external UI toolkit or runtime dependency) — replaced a `pywebview`-wrapped local HTTP dashboard (2026-08)
- Setup form auto-built from the CLI's own argument parser (one tab per argparse group), so it can't drift out of sync with the flags it exposes
- Live phase stepper with timings, active-loop progress bar, log stream, per-frame quality ticker, and an interactive preview (zoom/pan, live re-stretch from a retained linear source, before/after wipe-slider compare, per-frame thumbnail ring), and a completion summary card
- Zero overhead on a plain CLI run: the underlying event-sink singleton's publish methods are no-ops until the desktop app attaches it; preview JPEG encoding is throttled
- Closing the window mid-run asks for confirmation; a native OS notification fires on completion

### 26. Collection Quality Sweep (`--quality-sweep`)

- Recursively walks the tree under `-d`, scores every light frame (uncalibrated debayered luminance through `compute_quality_metrics`), and applies the pipeline's own `quality_gate` per folder — hard rejects, statistical outliers, and the folder-relative score threshold (`--quality-threshold`)
- Dry-run report by default; `--apply` renames flagged files to `*.fits.rejected` (invisible to frame discovery, which matches only `.fit`/`.fits`); `--sweep-undo` restores them
- `--quality-report PATH` writes the per-frame CSV; darks/flats/bias and pipeline outputs are excluded by the standard classifier

### 27. Hierarchical Processing

- **Auto-detected**: FITS in the root directory → single-folder mode; subdirectories with FITS → hierarchical mode
- Each subfolder processed independently with its own calibration frames
- Final combination: registered and mean-combined with crop to common valid dimensions
- `--debug intermediates`: saves per-subfolder stacks
- `--combine-sessions`: pools all lights from all subfolders into one unified stack

### 28. Preview & Output Formats

| Format | Flag | Description |
|--------|------|-------------|
| FITS float32 | (always) | Primary output; (3, H, W) layout for maximum compatibility |
| Preview JPEG | (always) | GHS, arcsinh, or linear stretch; 95% quality |
| TIFF 32-bit | `--export tiff` | 32-bit float RGB for external tools |
| XISF 1.0 | `--export xisf` | PixInsight native format |
| Per-frame JPEGs | `--export-frames-dir` | Stretched preview for every accepted frame |
| Star mask FITS | `--debug masks` | Star detection mask as `<output>_star_mask.fits` |
| Matched-filter SNR map | `--matched-filter` | `<output>_matched_filter.fits` — stacked image correlated with its own estimated PSF, the SNR-optimal linear detector for a known-shape point source; complementary to `--stack-method ivw`, not a replacement |
| Uncertainty map | `--uncertainty-map` | `<output>_sigma.fits` — per-pixel standard error, `--stack-method ivw` only |
| Starless sidecar | (automatic unless `--no-remove-stars`) | `<output>_starless.fits` — the fully post-processed image with stars inpainted out; the main output keeps stars |
| Source-separation components | `--nmf-separate` | `<output>_star_component.fits` / `_nebula_component.fits` |

**FITS Header metadata populated:**
- Stacking: frame count, rejection count, method, reference frame
- Calibration: which bias/dark/flat masters were applied
- Registration: mean/std shift, maximum shift magnitude
- Quality: mean brightness, contrast, quality scores
- Timing: Phase 1/2/3 durations
- Source metadata: telescope, instrument, observer, exposure time (from input headers)

### 29. Stretch Methods (Preview JPEG)

| Method | Flag | Description |
|--------|------|-------------|
| GHS *(default)* | `--stretch ghs` | Generalised Hyperbolic Stretch; tunable via config keys `ghs_b`, `ghs_sp`, `ghs_hp` (auto-set per target) |
| Arcsinh | `--stretch arcsinh` | Arcsinh stretch; good for galaxies |
| Linear | `--stretch linear` | No stretch; useful for already-processed data |

### 30. GPU Acceleration

- CuPy backend replaces NumPy/SciPy throughout the pipeline
- `GpuContext` provides a uniform interface: `xp` (array ops), `xndimage`, `xsignal`
- Auto-limits parallel workers based on available VRAM
- Enable with `--use-gpu`; requires `cupy-cuda*` installed (see `requirements-gpu.txt`)

### 31. Auto Target Detection (`--auto`, on by default; `--no-auto` to disable)

Heuristic target classifier — analyses frame metrics (star count, brightness distribution, contrast) and automatically applies optimised parameter sets for the detected target type. No external dependencies or API keys required. Any explicit CLI flag you pass overrides the value the advisor would have picked.

### 32. Elastic (Non-Rigid) Local Registration (`--elastic-registration`)

Corrects spatially-varying distortion a single global affine per frame can't fix — differential atmospheric refraction, field rotation across the frame, tube flexure. Replaces an earlier `--optical-flow` feature that was architecturally broken (dense Farneback optical flow on empty sky, a second separate `cv2.remap` pass that compounded blur instead of composing with the affine warp, and silently dropped under `--drizzle-scale > 1`/`--stack-method mean`) and was never wired up as a real CLI flag, documented, or tested.

- Reuses star catalogues already computed in Phase 1 and the post-registration residual check's star-matching to get per-frame matched (reference, aligned) star correspondences.
- Fits a smooth per-frame local displacement field (dy, dx) from those sparse correspondences via DBE's Gaussian-weighted local-linear regression + Tukey-biweight IRLS kernel (`src/background.py`, already Rust-accelerated and parity-tested), repurposed for a 2-channel displacement instead of a scalar brightness surface. Stored as a small coarse grid, sampled on demand — never materialised at full resolution.
- Composed into the *same* single resampling pass as the affine warp (`apply_transform`'s `local_field` parameter: source coordinate = affine source coordinate minus the rotated sampled displacement, one `map_coordinates` call) rather than a second warp — this is the fix for the old feature's double-warp blur.
- Works under `--drizzle-scale > 1` too (`_drizzle_one` subtracts the sampled field from its per-output-pixel coordinate mapping) — unlike the old feature, which silently dropped correction there.
- Falls back to the frame's existing unmodified affine/translation warp when fewer than `Config.LOCAL_WARP_MIN_STARS` (12) stars matched. Fitted displacement clamped to `Config.LOCAL_WARP_MAX_DISPLACEMENT_PX` (8px), which also expands `calc_common_crop`'s safety margin by the actual observed max fitted magnitude.
- Off by default (opt-in, higher-risk correction; not auto-enabled by `--auto`). Forces the residual-check star-match pass to run even under `--no-reg-residual-check` (needs the correspondences; the RMS-reject gate itself stays off).

### 33. Multi-Format Input (RAW, TIFF, XISF, SER)

Extends the FITS-only loader to camera RAW, TIFF, XISF, and SER — all dispatched through a single `src/io_fits.py::load_frame`, including `make_master`, so calibration masters can be built from any mix of formats.

- **RAW** (`src/io_raw.py`, rawpy) — reads the undemosaiced sensor mosaic (`raw.raw_image_visible`, not rawpy's own demosaic), per-channel black-level subtraction, white-level normalisation to [0,1]. Bayer pattern derived from `raw.color_desc`/`raw.raw_pattern`. EXIF (`EXPTIME`/`ISO`/`INSTRUME`/`TELESCOP`) via Pillow, best-effort.
- **TIFF** (`src/io_tiff.py`, tifffile) — integer sample values are *not* rescaled (cast to float32 as-is, same convention as `load_fits`), so a TIFF light stays on the same ADU-count scale as sibling FITS/RAW calibration frames. A 3-channel TIFF is treated as already-debayered RGB and passed through untouched; a 2-D (mono) TIFF is genuinely ambiguous (real mono capture vs. undemosaiced Bayer export — no TIFF tag distinguishes them) and falls through to the existing session-Bayer/RGGB default, same as a headerless FITS Bayer frame. A `.json` sidecar with `bayerPattern` overrides this.
- **XISF** (`src/io_xisf.py`, no dependency) — hand-rolled reader, the read-side counterpart to `xisf_writer.py`. Covers uncompressed, attached (not embedded/inline) pixel data, Float32/UInt16/UInt32/UInt8 sample format, mono or RGB, planar or normal pixel storage — sufficient to round-trip this codebase's own writer output and straightforward external exports. Compressed or embedded-block XISF raises a clear, specific error rather than silently misreading. `FITSKeywords` the writer embedded are recovered back into the header dict. Mono XISF is replicated to 3 channels at load time (XISF is a processed/calibrated format, never a raw Bayer mosaic, so it must never be routed through the debayer step).
- **SER** (`src/io_ser.py`, no dependency) — FireCapture/SharpCap planetary/lucky-imaging video. A single `.ser` file holds many frames, breaking the "one file = one frame" assumption everywhere else — `discover_frames` runs a pre-pass (`expand_ser_files`) that parses the 178-byte header once per file (`lru_cache`d) and synthesises one virtual `FrameInfo.path` per frame (`"<real_path>::<index>"`). `::` isn't a filesystem separator, so filename-substring classification, dict/set keys, logging, and checkpoint JSON serialisation all work unmodified on virtual paths. Per-frame pixel reads use `np.memmap` for O(1) offset seeking, always copied out of the memmap before returning (mutated in place downstream — a bare memmap view would corrupt the source file). `ColorID` maps to `BAYERPAT` for the 4 standard Bayer patterns; `MONO` is replicated to 3 channels (same reasoning as XISF); `RGB`/`BGR` are 3-channel-per-frame with a channel-order fix for BGR; the rare CMY-filter Bayer variants (`ColorID` 16-19) aren't supported by this pipeline's 4-pattern debayer and raise a clear error instead of silently mismapping. `--quality-sweep --apply`'s reject-rename step skips SER virtual paths (not real, renameable filesystem entries) with a warning rather than erroring.
- No CLI flags — format is detected purely by extension (or the `::` marker for SER), matching how FITS/RAW have always worked.

### 34. Temperature-Interpolated Dark Model (`--dark-temp-model`)

- Fits a per-pixel low-order polynomial of dark signal vs. sensor temperature across the *whole* dark library in one vectorised `np.linalg.lstsq` call, then evaluates it at each light frame's own temperature
- An alternative to `select_matching_darks`'s nearest-temperature selection, letting a smaller library cover a wider temperature range via interpolation
- Assumes the library is already homogeneous in ISO/gain/exposure time; needs >= 3 distinct temperatures in the library, falls back to nearest-match selection otherwise

### 35. Robust-PCA Master Calibration & Synthetic Flats (`--master-method robust_pca`, `--flat-from-lights`)

- Principal Component Pursuit: splits a bias/dark/flat stack into a low-rank component (the true shared pattern — flat-field vignetting, fixed dark current) and a sparse component (dust motes that shifted between sessions, transient hot pixels), instead of the default per-pixel median treating every outlier independently
- Opt-in only: needs >= `Config.ROBUST_PCA_MIN_FRAMES` frames, loads the full stack into memory, slower than median/mean. `--auto` narrowly upgrades median→robust_pca per calibration type when its frame count falls in `[ROBUST_PCA_MIN_FRAMES, ROBUST_PCA_AUTO_MAX_FRAMES]`
- `--flat-from-lights`: reuses the same decomposition on a capped sample of *light* frames when no dedicated flats exist — the low-rank component approximates vignetting/dust, stars and nebula structure fall into the sparse component since dithering shifts them frame-to-frame while vignetting stays sensor-locked. Approximate; opt-in
- Solver internals are Rust-accelerated (Gram-matrix thin-SVD trick, ~9x over direct `np.linalg.svd` at realistic shapes)

### 36. Self-Supervised Denoiser Parameter Calibration (`--denoise-strength-calibrate`)

See feature 13 (Denoising) for details — Noise2Self-style parameter selection for `--denoiser wavelet`, no ground truth needed.

### 37. Field Aberration / Tilt Report (`--aberration-report`)

- Per-cell FWHM/elongation map across the field, diagnosing sensor tilt or field curvature
- Writes an annotated PNG; diagnostic only, no effect on the stack

### 38. Dither Coverage Report (`--dither-report`)

- Bins each frame's sub-pixel registration-shift phase into a grid histogram over one output-pixel cell and reports how uniformly it's sampled
- Drizzle output quality is bounded by dither uniformity and nothing else in the pipeline measures it directly
- Writes `<stem>_dither.png`; diagnostic only, no effect on the stack

### 39. Saturated Star Core Repair (`--repair-stars`)

- Per-channel Moffat wing fit refills clipped (saturated) star cores
- Runs before star reduction/removal in the post-processing chain

### 40. Star Removal (on by default; `--no-remove-stars` to disable)

- Inpaints each detected star with local background (normalised-convolution fill), on by default
- Runs last, on the fully post-processed image; writes a `<output>_starless.fits` sidecar — the main output keeps stars
- `--auto` turns it off for star-dominant target types (`globular_cluster`, `star_field`, `wide_field`) where the stars are the target

### 41. Satellite/Aircraft Trail Rejection (`--trail-reject`)

- Per-frame Hough line detection + local-background inpaint, applied in Phase 1 before stacking
- Off by default; `--auto` can enable it defensively (e.g. when astrollm flags a defective frame — see feature 46)

### 42. Per-Frame Local Normalization (`--local-normalize`)

See feature 11 (Stacking Methods) for details — additive per-frame background match before the rejection combine, distinct from the now-removed post-processing local-variance-equalisation step.

### 43. Streaming Two-Pass Stack (`--stream`)

- Two-pass streaming stack of an *already-complete* directory: O(1) full-resolution memory via an online (single-pass) sigma-clip Welford accumulator, instead of materializing the whole `(N,H,W,C)` aligned stack the default `--stack-method` needs
- `--stream-burnin N` (default 10): frames MAD-rejected as a batch to seed the running sigma-clip state before streaming begins
- `--stream-sigma SIGMA` (default: `--rejection-sigma`)
- v1 limitations vs. the default pipeline: reference frame picked by quality score alone, hard-limit quality gating only, no `--elastic-registration`/drizzle/patch-weighted combine/`--merge`. Mutually exclusive with `--live`

### 44. Real-Time (Live) Stacking (`--live`)

- Watches the capture directory and folds each new sub into a running weighted-mean stack as it lands, pushing the growing result and a running SNR to whatever UI is attached (console-only on a plain CLI run; the desktop app shows it live)
- `--live-interval SEC` (default 4): directory poll interval
- `--live-duration MIN` (default: none, runs until Ctrl-C): optional time limit
- Mutually exclusive with `--stream`

### 45. NMF Source Separation (`--nmf-separate`)

- Non-negative matrix factorization factors the stacked image's per-pixel channel vector into 2 non-negative sources (spectral basis + spatial activation map each), labelling the higher peak-to-mean-activation component as "stellar"
- Offered alongside, not replacing, star removal (feature 40) — separates signal into components rather than discarding a masked region, suiting a downstream continuum-subtraction-style use better
- Non-convex, no global-optimum guarantee; writes `<stem>_star_component.fits`/`_nebula_component.fits`

### 46. astrollm Integration (`--astrollm`, `--astrollm-score-all`)

- Optional integration with astrollm, a separately-trained image classifier (external repo, invoked as a per-image subprocess — no network call)
- `--astrollm` (needs `--astrollm-dir`, or the individual `--astrollm-python`/`-script`/`-checkpoint`/`-timeout`/`-workers` overrides): when `--auto` is also active (the default), samples 3 light frames spread through the session (fast, ~8s each). The sampled category feeds the same target-classification prior SIMBAD/header metadata uses; a defect flag nudges settings defensively (enables `--trail-reject`, boosts chroma denoising strength). Never auto-rejects a frame — advisory only, this model is still finishing its first training run
- `--astrollm-score-all`: also scores every accepted light frame with astrollm (much slower — minutes, not seconds, on a large session). Has no effect without `--astrollm` also set (warns at startup if passed alone)
- Scores are stored in `FrameInfo.metrics['astrollm']` and logged, but never set `accepted` or feed `metrics['score']`

---

## Complete CLI Reference

The CLI exposes the decisions a user actually makes; fine-grained tuning
parameters live in the **config-file tier** — set them in a TOML file loaded
with `--config` (keys listed per feature above and in `parse_args`
`set_defaults`). `--help` renders these groups with full descriptions.

### Core

| Flag | Description |
|------|-------------|
| `-d, --directory PATH` | Input directory (required) |
| `-o, --output PATH` | Output FITS path (not required with `--health-check` / `--dry-run`) |
| `--cal-dir PATH` | External calibration library; best-matching frames auto-selected |
| `--config PATH` | Load parameters from TOML file; CLI args override |
| `--preset NAME` | quick, quality, galaxy, nebula, narrowband, starfield, planetary, lunar |
| `--no-auto` | Disable the heuristic target classifier + parameter advisor (on by default) |
| `--dry-run` | Show resolved parameters and resource estimates, no processing |
| `--health-check` | Analyse frames + calibration without stacking |
| `--quality-sweep [--apply]` | Recursively flag poor lights across a collection (dry-run default) |
| `--sweep-undo` | Restore files flagged by a previous sweep |
| `-v, --verbose` | Per-frame output |
| `-j, --parallel N` | Worker count (0 = auto, 1 = sequential) |
| `--use-gpu` | CuPy GPU acceleration (experimental) |
| `--stream` | Two-pass streaming stack of a complete directory (feature 43); mutually exclusive with `--live` |
| `--live` | Real-time stacking of a directory as frames land (feature 44); mutually exclusive with `--stream` |
| `--vignette-map PATH` | Per-instrument vignetting/background calibration map, subtracted per-frame after debayer |

### Frames & calibration (Phase 1)

| Flag | Default | Description |
|------|---------|-------------|
| `--debayer-method {malvar,menon2007}` | malvar | Both native Rust kernels; menon2007 is higher fidelity, ~4-5x slower |
| `--no-bayer-autodetect` | on | Disable the Bayer row-orientation autodetection check |
| `--white-balance` | grayworld | none, grayworld, whitepatch |
| `--quality-threshold N` | 50 | Reject frames scoring below N% of the session reference |
| `--no-quality-filter` | — | Keep every frame |
| `--no-ca-correction` | — | Disable chromatic aberration correction |
| `--fix-atmospheric-dispersion` | off | Experimental software ADC correction (feature 6); not wired into `--auto` |
| `--cosmic-ray-rejection` / `--no-cosmic-ray-rejection` | auto | Force / disable L.A.Cosmic (auto: skipped on >=20-frame rejection stacks) |
| `--master-method {median,mean,robust_pca}` | median | Master bias/dark/flat combine method (feature 35) |
| `--flat-from-lights` | off | Derive a synthetic flat from the light frames via robust-PCA (feature 35) |
| `--dark-temp-model` | off | Temperature-interpolated dark current model (feature 34) |
| `--trail-reject` | off | Satellite/aircraft trail rejection (feature 41) |

### Registration & stacking (Phases 2-3)

| Flag | Default | Description |
|------|---------|-------------|
| `--stack-method` | auto | mean, median, sigma_clip, winsorized, percentile, esd, linear_fit, ivw, wavelet, auto |
| `--rejection-sigma N` | 3.0 | Sigma threshold for sigma_clip/winsorized |
| `--rejection-iters N` | 3 | Clipping iterations |
| `--local-normalize` | off | Additive per-frame background match before the rejection combine (feature 42) |
| `--uncertainty-map` | off | With `--stack-method ivw`, write a `<output>_sigma.fits` per-pixel standard-error map |
| `--drizzle-scale N` | 1.0 | Super-resolution scale (2.0 = 2x; needs dithered frames) |
| `--drizzle-pixfrac N` | 1.0 | Drizzle tent-kernel pixel fraction |
| `--drizzle-kernel {lanczos3,psf,magic}` | lanczos3 | Drizzle resample kernel (feature 11) |
| `--super-res-iters N` | 0 | Iterative Back-Projection super-res refinement passes after drizzle (feature 11) |
| `--no-registration` | — | Disable alignment (pre-aligned frames) |
| `--no-affine` | — | Translation-only registration |
| `--no-reg-residual-reject` | — | Keep frames failing the post-registration residual check (dropped by default) |
| `--no-reg-residual-check` | — | Skip the post-registration residual check entirely |
| `--elastic-registration` | off | Fit + apply a per-frame local (non-rigid) displacement field on top of the global affine |

### Post-processing (Phase 4)

| Flag | Default | Description |
|------|---------|-------------|
| `--denoiser NAME` | auto (curvelet) | Primary luma denoiser: wavelet, mmt, bm3d, acdnr, nlm, bilateral, aniso, curvelet, none |
| `--denoise-strength N` | 3.0 | Luma denoise threshold factor |
| `--denoise-strength-calibrate` | off | Self-supervised strength selection, `--denoiser wavelet` only (feature 36) |
| `--variance-stabilize` | off | Generalized Anscombe transform before wavelet/curvelet thresholding |
| `--deconvolve {off,rl,rl-sv,tv,sparse}` | off | Richardson-Lucy (global or spatially-variant), TV, or sparse-wavelet (FISTA) deconvolution |
| `--repair-stars` | off | Saturated star core repair via Moffat wing fit (feature 39) |
| `--bg-method` | dbe | dbe, mesh, wavelet |
| `--bg-wavelet-scales N` | 6 | `--bg-method wavelet` starlet scale count |
| `--no-background-extraction` | — | Disable background extraction |
| `--galaxy-mode` | off | Exclude an extended-source ellipse from background sampling (feature 12); auto-enabled for galaxy targets by `--auto` |
| `--galaxy-center X,Y` | — | Skip auto-detection, center the `--galaxy-mode` exclusion here |
| `--galaxy-mask-radius PX` | auto | Override the fitted exclusion ellipse's semi-major axis |
| `--no-chroma-nr` | — | Disable chroma noise reduction |
| `--no-star-reduce` | — | Disable star halo softening |
| `--no-remove-stars` | on by default | Disable star removal / `<output>_starless.fits` sidecar (feature 40) |
| `--no-local-contrast` | — | Disable multiscale local contrast |
| `--scnr` | off | Subtractive green-cast removal |
| `--photometric-calibration` | off | Gray-locus colour calibration |
| `--halo-removal` | off | Fit + subtract bright-star halos |
| `--hdr-combine PATH` | — | Blend a short-exposure stack into saturated regions |
| `--hdr-blend-mode {threshold,fusion}` | threshold | Blend method for `--hdr-combine` |
| `--nmf-separate` | off | NMF star/nebula source separation (feature 45) |
| `--skip-step STEP` | — | Skip a named post-processing step (repeatable) |

### Output, preview & plate solving

| Flag | Default | Description |
|------|---------|-------------|
| `--stretch` | ghs | Preview stretch: linear, arcsinh, ghs |
| `--preview-black-sigma N` | auto | Preview black point in sky-sigma (auto-set per target and stack depth) |
| `--export FMT[,FMT]` | — | Extra formats: tiff, xisf |
| `--matched-filter` | off | Write a per-stack matched-filter SNR map (feature 28) |
| `--plate-solve` | off | Solve WCS (`--plate-solver astap|astrometry`, `--astap-path`) |
| `--color-calibrate` | off | Photometric calibration from plate-solved stars (`--color-calibrate-method {colorindex,spcc}`) |
| `--annotate` | off | Label stars + named DSOs on a copy of the preview, needs a WCS (feature 20) |
| `--annotate-mag-limit N` | — | Faintest star magnitude to annotate |
| `--aberration-report` | off | Field aberration/tilt diagnostic PNG (feature 37) |
| `--dither-report` | off | Dither-coverage uniformity diagnostic PNG (feature 38) |

### astrollm (feature 46)

`--astrollm` (needs `--astrollm-dir` or the individual overrides) plus:
`--astrollm-score-all`, `--astrollm-dir`, `--astrollm-python`,
`--astrollm-script`, `--astrollm-checkpoint`, `--astrollm-timeout`,
`--astrollm-workers`. See `--help` for details.

### Multi-session, merge & checkpoint

| Flag | Description |
|------|-------------|
| `--merge STACK.fits [...]` | Incremental stacking: fold previous linear stacks into this run |
| `--combine-sessions` | Pool all subfolder lights into one unified stack |
| `--mosaic` | Stitch per-subfolder stacks via WCS reprojection |
| `--keep-checkpoint` | Persist the raw stack; re-runs skip Phases 1-3 |
| `--no-resume` | Ignore an existing checkpoint |

### Comet mode

`--comet-mode` plus: `--comet-xy`, `--comet-search-radius`, `--comet-affine`,
`--comet-blend-sigma`, `--coma-mask-radius`, `--comet-radial-renorm`,
`--comet-larson-sekanina`, `--comet-ls-rotation`, `--comet-designation`,
`--observer-site`, `--comet-detrail`. See `--help` for details.

### Diagnostics & debugging

| Flag | Description |
|------|-------------|
| `--debug KIND[,KIND]` | registration, diagnostic (pre-step FITS snapshots), intermediates, masks |
| `--quality-report PATH` | Per-frame quality metrics CSV |
| `--export-frames-dir PATH` | Stretched JPEG per accepted frame |
| `--log-level` / `--log-file` | Logging control |

Valid step names for `--skip-step`: `hot_pixel`, `background`, `chroma_nr`, `sky_floor`, `wavelet`, `sky_residual`, `sky_pedestal`, `nlm`, `bilateral`, `mmt`, `acdnr`, `bm3d`, `aniso`, `scnr`, `photo_cal`, `deconvolve`, `star_reduce`, `local_contrast`, `sky_neutralize`, `star_remove`

---

## Example Commands

```bash
# Single folder, all defaults
python originstack.py -d lights/ -o stacked.fits

# Verbose with auto target detection (recommended)
python originstack.py -d lights/ -o stacked.fits -v --auto

# External calibration library
python originstack.py --cal-dir calibration/ -d lights/ -o stacked.fits --auto -v

# Incremental stacking: fold last week's stack into tonight's run
python originstack.py -d tonight/ -o m51_v2.fits --auto --merge m51_v1.fits

# Galaxy preset with RL deconvolution
python originstack.py -d lights/ -o galaxy.fits --preset galaxy --deconvolve rl -v

# Explicit denoiser choice
python originstack.py -d lights/ -o out.fits --auto --denoiser mmt -v

# Super-resolution drizzle
python originstack.py -d lights/ -o drizzled.fits --drizzle-scale 2.0 -v

# Plate solve + colour calibration
python originstack.py -d lights/ -o stacked.fits --plate-solve --color-calibrate -v

# Debug registration problems
python originstack.py -d lights/ -o stacked.fits --debug registration

# Iterate on post-processing settings (re-runs skip Phases 1-3)
python originstack.py -d lights/ -o stacked.fits --auto --keep-checkpoint

# Minimal run (turn off default post-processing)
python originstack.py -d lights/ -o stacked.fits   --no-star-reduce --no-local-contrast --no-background-extraction --denoiser none

# Galaxy target with a manual exclusion-mask center (bypasses auto-detection)
python originstack.py -d lights/ -o m51.fits --auto --galaxy-mode --galaxy-center 1420,930 -v

# astrollm-assisted classification (fast, 3-frame sample) plus a full-session scan
python originstack.py -d lights/ -o stacked.fits --auto --astrollm --astrollm-dir C:/source/astrollm --astrollm-score-all -v
```

---

## Dependencies

### Required (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| `numpy >= 1.20` | Array operations |
| `astropy >= 5.0` | FITS I/O |
| `scipy >= 1.7` | Shifting, interpolation, Gaussian filters |
| `tqdm >= 4.65` | Progress bars (falls back to plain iterator) |
| `Pillow >= 9.0` | Preview JPEG generation |
| `psutil >= 5.9` | Memory usage reporting |

### Optional (install separately)

| Package | Feature Unlocked |
|---------|-----------------|
| `cupy-cuda*` | GPU acceleration (`--use-gpu`; see `requirements-gpu.txt`) |
| `reproject` | Mosaic WCS reprojection (`--mosaic`) |
| `tifffile` | TIFF input and 32-bit TIFF output (`--export tiff`) |
| `rawpy` | Camera RAW input (CR2/CR3/NEF/ARW/DNG/…); RAW files are silently excluded from discovery when absent |
| `astro_native` (Rust) | 30+ native kernels; numpy fallback when absent (see below) |

`opencv-python`, `astroalign`, `scikit-image`, `PyWavelets`, and `astroquery` are not used anywhere in this codebase — Malvar/Menon2007 debayer and the bilateral filter are native Rust kernels (numpy fallback if `astro_native` isn't built); `--merge`'s cross-night registration is `src/blind_match.py`, also native; NLM denoising and Richardson-Lucy's CPU fallback are native/numpy now; the wavelet denoiser and multiscale-entropy seeing metric's transform are native (`src/wavelet.py`); every network catalogue lookup (astrometry.net, Gaia, VizieR, SIMBAD, JPL Horizons) is direct HTTP via `src/net_query.py` (stdlib urllib) — none of them need an external dependency.

---

## Known Limitations

**Registration:**
- Very sparse star fields or pure extended nebulae (no stars) may fall back to centroid matching
- Use `--no-registration` for pre-aligned frames

**Debayering:**
- Both `--debayer-method` choices (`malvar` default, `menon2007`) are native Rust kernels with a numpy fallback; the internal `bilinear` base function isn't CLI-reachable and exists only as a fallback path, so this is a non-issue in practice

**Hierarchical mode:**
- Targets with very different crop amounts produce shape mismatches — handled by resizing to minimum common dimensions (minor quality trade-off)
- Consider processing targets with very different field-of-views separately

**Drizzle:**
- Requires dithered frames for a true resolution benefit; without dithering it simply upsamples

**GPU:**
- CuPy is experimental; not all code paths are GPU-accelerated; best used with `--parallel 1`

---

## Design Patterns

1. **Streaming Architecture** — one frame at a time; free immediately after accumulation
2. **Dataclasses** — type-safe config (`Config`) and per-frame metadata (`FrameInfo`)
3. **Optional dependency pattern** — every optional import wrapped in `try/except` with `HAS_X` flag
4. **Singleton** — `get_gpu()` returns the module-level `GpuContext` instance
5. **Strategy Pattern** — debayer method, stacking method, stretch method all selected by name string
6. **Separation of Concerns** — each `src/` module owns exactly one domain; `pipeline.py` wires them
