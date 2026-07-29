# OriginStack — Architecture & Feature Reference

## Project Overview

OriginStack is a professional-grade Python pipeline for stacking and post-processing astronomical FITS images. It was designed for the Celestron Origin smart telescope and runs efficiently on ordinary laptop hardware using a streaming memory architecture: frames are loaded, processed, and freed one at a time, keeping peak memory constant regardless of total frame count.

For installation, quick start, and common recipes, see [README.md](README.md).

---

## Module Inventory

| Module | Lines | Contents |
|--------|-------|----------|
| `src/gpu_context.py` | ~160 | `GpuContext`, CUDA stream contexts, `get_gpu()` singleton |
| `src/models.py` | ~90 | `Config`, `FrameInfo`, `ProcessingStats` |
| `src/utils.py` | ~160 | Print helpers, `format_time`, `get_memory_usage_mb` |
| `src/io_fits.py` | ~315 | FITS load/save, `load_frame` (format dispatcher), `make_master`, `populate_fits_header` |
| `src/io_raw.py` | ~170 | Camera RAW load (rawpy) — CR2/CR3/NEF/ARW/DNG/ORF/RW2/RAF/PEF/3FR/MRW/X3F/IIQ |
| `src/io_tiff.py` | ~90 | TIFF load (tifffile) |
| `src/io_xisf.py` | ~200 | XISF 1.0 load (hand-rolled, no dependency) |
| `src/io_ser.py` | ~190 | SER (planetary video) load — one file expands to many virtual frames |
| `src/frame_discovery.py` | ~140 | `discover_frames`, `classify_frame`, `select_matching_darks` |
| `src/debayer.py` | ~305 | Debayering, hot pixels, white balance, CA correction |
| `src/quality.py` | ~375 | `compute_quality_metrics`, star detection, FWHM |
| `src/psf_deconvolution.py` | ~215 | PSF estimation (Moffat/Gaussian), Richardson-Lucy |
| `src/background.py` | ~1100 | DBE (robust local regression, Rust-accelerated), mesh-based sky extraction, floor normalisation |
| `src/denoising.py` | ~895 | Wavelet (BayesShrink), bilateral, NLM, MMT, ACDNR, GHS/arcsinh stretch |
| `src/registration.py` | ~685 | `calculate_shift`, affine/RANSAC, `calc_common_crop`, `run_registration_phase`, `fit_displacement_field` (`--elastic-registration`) |
| `src/stacking.py` | ~770 | Sigma-clip, percentile, ESD, drizzle, Lanczos, `run_stacking_phase` |
| `src/plate_solve.py` | ~160 | Astrometry.net + ASTAP + SIMBAD identification |
| `src/pipeline.py` | ~265 | Four-phase orchestrator, `stack_target` |
| `src/frame_processor.py` | ~430 | `_process_single_frame`, `execute_frame_processing`, `quality_gate` |
| `src/postprocess.py` | ~315 | `postprocess_stack` — up to 20-step post-processing chain |
| `src/auto_settings.py` | ~360 | Heuristic target classifier, `apply_auto_settings` (`--auto`) |
| `src/health_check.py` | ~190 | Frame consistency and calibration quality analysis |
| `src/session_info.py` | ~200 | Session metadata reader (Celestron Origin `info.json` — GPS, filter, ISO, orientation) |
| `src/target_inference.py` | ~670 | Target type inference — heuristic galaxy/nebula/starfield classification |
| `src/color_calibrate.py` | ~395 | Photometric colour calibration using plate-solved star colours |
| `src/photometric_calibration.py` | ~300 | Gray-locus photometric colour calibration (`--photometric-calibration`) |
| `src/mosaic.py` | ~315 | WCS-based mosaic stitching (`--mosaic`) |
| `src/checkpoint.py` | ~225 | Checkpoint save/load for pre-post-processing stack (`--keep-checkpoint`) |
| `src/merge.py` | ~230 | Incremental stacking — register + weighted-merge previous linear stacks (`--merge`) |
| `src/quality_sweep.py` | ~220 | Collection quality sweep — recursive scoring + reversible flagging (`--quality-sweep`) |
| `src/webview.py` | ~380 | Live stacking dashboard — stdlib HTTP + SSE (`--web-view`) |
| `src/xisf_writer.py` | ~105 | XISF 1.0 format writer (`--export xisf`) |
| `src/channel_combine.py` | ~310 | Multi-channel combination (L-RGB, OSC + narrowband workflows) |
| `src/features.py` | ~90 | Low-level feature extraction helpers |
| `src/cli.py` | ~685 | `process_directory`, `parse_args`, `main` |
| `astro_stack.py` | ~170 | Backward-compatibility re-export shim |
| `ext/astro_native/` (Rust) | — | Optional PyO3/maturin crate (13 kernels): stacking combines, fused patch-weighted combine, Lanczos-3 warp (alignment + drizzle), anisotropic diffusion, L.A.Cosmic, median filter, DBE surface fit + patch sampler (numpy fallback when absent) |

**Total: ~10,800 lines** (Python). Tests in `tests/test_core.py` import symbols directly from `astro_stack`; `tests/test_native.py` covers the Rust kernels (auto-skips if unbuilt).

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
             7 methods: mean, median, sigma_clip, winsorized, percentile, esd, drizzle

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

- **Master frame creation**: median-combine bias/darks/flats per subfolder
- **Dark selection**: matches lights by ISO, exposure time, and sensor dimensions
- **Calibration order**: bias subtraction → dark subtraction (exposure-scaled) → flat division
- **Smoothing**: masters are Gaussian-smoothed (per-Bayer-channel for flats) to avoid adding correlated noise
- **Hot pixel map**: built from dark frame before smoothing; applied per Bayer sub-channel
- **Flat normalisation**: per Bayer channel; division-by-zero protected

### 3. Bayer Pattern Debayering

- **Auto-detection**: from FITS headers (`BAYERPAT`, `COLORTYP`)
- **Supported patterns**: RGGB, BGGR, GRBG, GBRG (default: RGGB)
- **Algorithms**:
  - `bilinear` — pure NumPy, fast, default
  - `malvar` — Malvar-He-Cutler algorithm, higher quality; native Rust kernel with a numpy fallback, no external dependency
  - `vng` — alias for `malvar` (was OpenCV's Variable Number of Gradients; OpenCV is no longer a dependency of this codebase)
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
| Star count | DAOStarFinder (photutils) or SEP C-backend |
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

**Drizzle super-resolution** (`--drizzle-scale 2.0`):
- Lanczos-3 sub-pixel accumulation onto the upsampled grid (native Rust warp, ~26x vs the scipy path; all channels in one pass)
- Requires dithered frames to produce true resolution gain
- `--drizzle-pixfrac` controls the tent-kernel pixel fraction (< 1.0 = sharper, noisier)

### 12. Background Extraction

| Method | Description |
|--------|-------------|
| `dbe` (default) | Dynamic Background Extraction — Gaussian-weighted robust local regression (Tukey IRLS) over sampled patches; bounded by construction (no runaway extrapolation near bright stars). Patch sampling and surface fit are Rust-accelerated |
| `mesh` | Legacy polynomial grid (faster, less accurate) |
| `graxpert` | AI-powered gradient removal via GraXpert subprocess |

**DBE details:**
- Star mask applied to sampling patches to avoid fitting to sources
- Extended galaxy/nebula masking preserves target structure
- Sky floor normalisation removes constant per-channel pedestal after background subtraction

### 13. Denoising

The primary luma denoiser is selected with a single flag —
`--denoiser {auto,wavelet,mmt,bm3d,acdnr,nlm,bilateral,aniso,none}` — and the
auto-advisor enforces **one primary** (precedence BM3D > MMT > wavelet > ACDNR):
layering several full-frame smoothers compounds smoothing without adding
selectivity. Chroma noise reduction is separate and always available.
Per-denoiser tuning lives in the config-file tier (see `--config`).

| Denoiser | Notes | Config keys |
|----------|-------|-------------|
| `wavelet` (default) | BayesShrink adaptive per subband, luma/chroma split, star-protected, strength auto-tuned from SNR | `denoise_strength`, `denoise_adaptive`, `auto_denoise_strength`, `denoise_chroma_boost` |
| `mmt` | Multiscale Median Transform — robust to Poisson+read noise, best edge preservation (Rust-accelerated median cascade, ~10x) | `denoise_mmt_levels`, `denoise_mmt_strength` |
| `bm3d` | Collaborative filtering, near-optimal, slower (auto-enabled by the advisor when SNR/frame count justify it) | `bm3d_sigma`, `bm3d_stride`, `bm3d_search_window`, `bm3d_group_size` |
| `acdnr` | Contrast-gated sky smoothing — flat sky smoothed, structure preserved | `denoise_acdnr_sigma`, `denoise_acdnr_k` |
| `nlm` | Non-local means (skimage.restoration) | `denoise_nlm_strength`, `denoise_nlm_blend` |
| `bilateral` | Edge-preserving bilateral filter, joint colour-space weighting (Rust-accelerated) | `denoise_bilateral_sigma_color`, `denoise_bilateral_sigma_space` |
| `aniso` | Perona-Malik anisotropic diffusion (Rust-accelerated, ~37x) | `aniso_iterations`, `aniso_kappa`, `aniso_gamma`, `aniso_option` |
| `none` | Disable luma denoising | — |

**Chroma noise reduction** (default on, `--no-chroma-nr`): Gaussian smoothing of
the chroma channels; fine pass (`chroma_nr_sigma`) plus an optional object-masked
coarse pass for medium-scale colour blotches (`chroma_nr_large_sigma`, auto-set
for galaxy targets).

### 14. PSF Deconvolution (`--deconvolve {off,rl,tv}`)

- `rl` — Richardson-Lucy iterative deconvolution (GPU-accelerated via cupy FFT under `--use-gpu`)
- `tv` — Total-Variation regularised variant, sharper edges at the cost of speed
- PSF model: Moffat (default) or Gaussian; FWHM estimated from detected stars
- Config keys: `deconvolve_iterations`, `deconvolve_fwhm`, `deconvolve_psf_model`, `deconvolve_blind_psf` (empirical PSF from a star median stack), `tv_lambda`, `tv_iterations`
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
- **Gaia DR3 extension** (`--gaia-calibration`): extends calibration using Gaia stellar colours (requires `--plate-solve`)
- **Astrometry-based** (`--color-calibrate`): full photometric calibration using field stars

### 19. Plate Solving

- **nova.astrometry.net** (default): cloud-based, requires `astroquery` and `ASTROMETRY_API_KEY`
- **ASTAP** (`--plate-solver astap`): local solver, no API key required
- On success: writes WCS (CRVAL, CRPIX, CD matrix) to FITS header
- Object identification via SIMBAD database
- Unlocks `--color-calibrate` and `--gaia-calibration`

### 20. Star Removal (`--star-remove`)

- Invokes Starnet++ binary (auto-detected on PATH or via `--starnet-path`)
- Saves `<output>_starless.fits` and `<output>_stars.fits`
- Useful for separate nebula/star processing workflows

### 21. Comet Mode (`--comet-mode`)

- Runs two registration passes: one tracking stars, one tracking the comet nucleus
- Saves `<output>_comet.fits` (comet-registered stack) alongside the star-registered stack
- `--comet-blend-sigma` controls blend transition width

### 22. HDR Combining (`--hdr-combine SHORT_STACK.fits`)

- Blends a separate short-exposure stack into the highlight regions of the main stack
- Recovers detail in saturated bright areas (star cores, galaxy nuclei)
- `--hdr-short-exptime`, `--hdr-long-exptime` specify the exposure times for proper scaling

### 23. Mosaic Stitching (`--mosaic`)

- Stitches per-subfolder stacks into a single wide-field mosaic
- WCS-based reprojection via the `reproject` package
- Requires plate solving to succeed for each panel

### 24. Incremental Stacking (`--merge`)

- The main output FITS is the **linear pre-post-processing stack** (`RAWSTACK=True`) carrying `NFRAMES`/`INTGTIME`/`TOTEXP` headers
- `--merge PREV.fits [...]` processes only the new session through Phases 1-3, registers each previous stack onto the new grid (astroalign triangle matching first — nights differ by arbitrary field rotation on alt-az mounts — star-catalog RANSAC and translation fallbacks), and combines as a per-pixel `NFRAMES`-weighted mean inside each warped footprint
- Phase 4 runs once on the merged result; header aggregates are summed, so the output chains into future merges
- Guards: hard error on registration failure, <25% footprint overlap, or <0.15 aligned-luminance correlation (wrong-target protection); refuses non-linear inputs and `--drizzle-scale > 1`
- No cross-session outlier rejection (each session already rejected internally)
- Coalesces with `--keep-checkpoint`: the checkpoint stores the session-only stack, so a resumed run re-applies the merge idempotently (~45 s post-processing iteration on a merged stack)

### 25. Live Web View (`--web-view`)

- Pure-stdlib local dashboard (`http.server` + Server-Sent Events, no dependencies, localhost only, default port 8765 via `--web-view-port`)
- Live phase stepper with timings, active-loop progress bar, log stream, per-frame quality ticker, preview images at milestones (post-stack linear preview, each post-processing step, final), and a completion summary card
- Zero overhead when the flag is absent: the singleton's publish methods are no-ops until started; preview JPEG encoding is throttled
- Server keeps serving the final state after the run completes (Ctrl+C to exit)

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
| `--web-view` | Live dashboard while stacking (`--web-view-port`, default 8765) |
| `-v, --verbose` | Per-frame output |
| `-j, --parallel N` | Worker count (0 = auto, 1 = sequential) |
| `--use-gpu` | CuPy GPU acceleration (experimental) |

### Frames & calibration (Phase 1)

| Flag | Default | Description |
|------|---------|-------------|
| `--debayer-method` | bilinear | bilinear, malvar (native Rust), vng (alias for malvar) |
| `--white-balance` | grayworld | none, grayworld, whitepatch |
| `--quality-threshold N` | 50 | Reject frames scoring below N% of the session reference |
| `--no-quality-filter` | — | Keep every frame |
| `--no-ca-correction` | — | Disable chromatic aberration correction |
| `--cosmic-ray-rejection` / `--no-cosmic-ray-rejection` | auto | Force / disable L.A.Cosmic (auto: skipped on >=20-frame rejection stacks) |

### Registration & stacking (Phases 2-3)

| Flag | Default | Description |
|------|---------|-------------|
| `--stack-method` | auto | mean, median, sigma_clip, winsorized, percentile, esd, trimmed_mean, auto |
| `--rejection-sigma N` | 3.0 | Sigma threshold for sigma_clip/winsorized |
| `--rejection-iters N` | 3 | Clipping iterations |
| `--drizzle-scale N` | 1.0 | Super-resolution scale (2.0 = 2x; needs dithered frames) |
| `--drizzle-pixfrac N` | 1.0 | Drizzle tent-kernel pixel fraction |
| `--no-registration` | — | Disable alignment (pre-aligned frames) |
| `--no-affine` | — | Translation-only registration |
| `--no-reg-residual-reject` | — | Keep frames failing the post-registration residual check (dropped by default) |
| `--no-reg-residual-check` | — | Skip the post-registration residual check entirely |
| `--elastic-registration` | off | Fit + apply a per-frame local (non-rigid) displacement field on top of the global affine |

### Post-processing (Phase 4)

| Flag | Default | Description |
|------|---------|-------------|
| `--denoiser NAME` | auto | Primary luma denoiser: wavelet, mmt, bm3d, acdnr, nlm, bilateral, aniso, none |
| `--denoise-strength N` | 3.0 | Luma denoise threshold factor |
| `--deconvolve {off,rl,tv}` | off | Richardson-Lucy or TV-regularised deconvolution |
| `--bg-method` | dbe | dbe, mesh, graxpert (`--graxpert-path`) |
| `--no-background-extraction` | — | Disable background extraction |
| `--no-chroma-nr` | — | Disable chroma noise reduction |
| `--no-star-reduce` | — | Disable star halo softening |
| `--no-local-contrast` | — | Disable multiscale local contrast |
| `--scnr` | off | Subtractive green-cast removal |
| `--photometric-calibration` | off | Gray-locus colour calibration |
| `--gaia-calibration` | off | Gaia DR3 extension (needs `--plate-solve`) |
| `--halo-removal` | off | Fit + subtract bright-star halos |
| `--hdr-combine PATH` | — | Blend a short-exposure stack into saturated regions |
| `--skip-step STEP` | — | Skip a named post-processing step (repeatable) |

### Output, preview & plate solving

| Flag | Default | Description |
|------|---------|-------------|
| `--stretch` | ghs | Preview stretch: linear, arcsinh, ghs |
| `--preview-black-sigma N` | auto | Preview black point in sky-sigma (auto-set per target and stack depth) |
| `--export FMT[,FMT]` | — | Extra formats: tiff, xisf |
| `--plate-solve` | off | Solve WCS (`--plate-solver astap|astrometry`, `--astap-path`) |
| `--color-calibrate` | off | Photometric calibration from plate-solved stars |
| `--star-remove` | off | Starnet++ star removal (`--starnet-path`) |

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
python astro_stack.py -d lights/ -o stacked.fits

# Verbose with auto target detection (recommended)
python astro_stack.py -d lights/ -o stacked.fits -v --auto

# External calibration library
python astro_stack.py --cal-dir calibration/ -d lights/ -o stacked.fits --auto -v

# Incremental stacking: fold last week's stack into tonight's run
python astro_stack.py -d tonight/ -o m51_v2.fits --auto --merge m51_v1.fits

# Galaxy preset with RL deconvolution
python astro_stack.py -d lights/ -o galaxy.fits --preset galaxy --deconvolve rl -v

# Explicit denoiser choice
python astro_stack.py -d lights/ -o out.fits --auto --denoiser mmt -v

# Super-resolution drizzle
python astro_stack.py -d lights/ -o drizzled.fits --drizzle-scale 2.0 -v

# Plate solve + colour calibration
python astro_stack.py -d lights/ -o stacked.fits --plate-solve --color-calibrate -v

# Debug registration problems
python astro_stack.py -d lights/ -o stacked.fits --debug registration

# Iterate on post-processing settings (re-runs skip Phases 1-3)
python astro_stack.py -d lights/ -o stacked.fits --auto --keep-checkpoint

# Minimal run (turn off default post-processing)
python astro_stack.py -d lights/ -o stacked.fits   --no-star-reduce --no-local-contrast --no-background-extraction --denoiser none
```

---

## Dependencies

### Required (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| `numpy >= 1.20` | Array operations |
| `astropy >= 5.0` | FITS I/O |
| `scipy >= 1.7` | Shifting, interpolation, Gaussian filters |
| `scikit-image >= 0.21` | Phase cross-correlation, morphology |
| `photutils >= 1.5` | Star detection (degrades gracefully without it) |
| `tqdm >= 4.65` | Progress bars (falls back to plain iterator) |
| `Pillow >= 9.0` | Preview JPEG generation |
| `psutil >= 5.9` | Memory usage reporting |

### Optional (install separately)

| Package | Feature Unlocked |
|---------|-----------------|
| `PyWavelets` | Wavelet denoising (`--denoise`) |
| `sep` | Alternate star-detection backend (default `matched-filter` needs no extra package and is faster on real data) |
| `astroquery >= 0.4.6` | Plate solving via nova.astrometry.net |
| `cupy-cuda*` | GPU acceleration (`--use-gpu`; see `requirements-gpu.txt`) |
| `reproject` | Mosaic WCS reprojection (`--mosaic`) |
| `astroalign` | Triangle-pattern registration for `--merge` (cross-night field rotation) |
| `tifffile` | 16-bit TIFF output |

`opencv-python` is not used anywhere in this codebase — Malvar/VNG debayer and the bilateral filter are native Rust kernels (numpy fallback if `astro_native` isn't built).

---

## Known Limitations

**Registration:**
- Very sparse star fields or pure extended nebulae (no stars) may fall back to centroid matching
- Use `--no-registration` for pre-aligned frames

**Debayering:**
- Bilinear (default) can produce colour fringing; use `--debayer-method malvar` for better quality (native Rust kernel, no extra dependency)

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
