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
| `src/io_fits.py` | ~315 | FITS load/save, `make_master`, `populate_fits_header` |
| `src/frame_discovery.py` | ~140 | `discover_frames`, `classify_frame`, `select_matching_darks` |
| `src/debayer.py` | ~305 | Debayering, hot pixels, white balance, CA correction |
| `src/quality.py` | ~375 | `compute_quality_metrics`, star detection, FWHM |
| `src/psf_deconvolution.py` | ~215 | PSF estimation (Moffat/Gaussian), Richardson-Lucy |
| `src/background.py` | ~1000 | DBE (RBF thin-plate spline), mesh-based sky extraction, floor normalisation |
| `src/denoising.py` | ~895 | Wavelet (BayesShrink), bilateral, NLM, MMT, ACDNR, GHS/arcsinh stretch |
| `src/registration.py` | ~685 | `calculate_shift`, affine/RANSAC, `calc_common_crop`, `run_registration_phase` |
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
| `src/xisf_writer.py` | ~105 | XISF 1.0 format writer (`--output-xisf`) |
| `src/channel_combine.py` | ~310 | Multi-channel combination (L-RGB, OSC + narrowband workflows) |
| `src/features.py` | ~90 | Low-level feature extraction helpers |
| `src/cli.py` | ~685 | `process_directory`, `parse_args`, `main` |
| `astro_stack.py` | ~170 | Backward-compatibility re-export shim |

**Total: ~10,800 lines.** Tests in `tests/test_core.py` import symbols directly from `astro_stack`.

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
             Calculate per-frame shifts (phase correlation + affine/RANSAC)
             Select reference frame (highest quality score)

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
  - `malvar` — Malvar-He-Cutler algorithm, higher quality, requires OpenCV
  - `vng` — Variable Number of Gradients, requires OpenCV
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

- Realigns red and blue channels to the green channel
- Uses phase cross-correlation for sub-pixel channel shift estimation
- Enabled by default; disable with `--no-ca-correction`

### 7. Cosmic Ray Rejection (L.A.Cosmic)

- Per-frame rejection using a Laplacian edge-detection noise model
- Applied in Phase 1 before stacking accumulation
- Tunable via `--cr-sigclip` (detection sigma) and `--cr-objlim` (object rejection ratio)
- Enabled by default; disable with `--no-cosmic-ray-rejection`

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

**Quality-weighted stacking** (`--weight-snr`, `--weight-fwhm`, `--weight-stars`):  
Frames with better quality contribute more to the final stack.

### 9. Image Registration

**Translation (default):**
1. Phase cross-correlation (scikit-image) — sub-pixel accurate
2. FFT cross-correlation on downscaled image (fallback)
3. Centroid of thresholded bright regions (second fallback)

**Affine correction (default on, `--no-affine` to disable):**
- Star catalogue extraction with DAOStarFinder
- Nearest-neighbour star matching
- RANSAC-based affine transform estimation (rotation + scale + translation)
- Handles field rotation from polar alignment error and differential refraction

**Validation and dither detection:**
- Shifts > 10% of image dimension are rejected as unrealistic
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
- Lanczos-interpolated sub-pixel accumulation onto upsampled grid
- Requires dithered frames to produce true resolution gain
- Drop size controls the `pixfrac` parameter (`--drizzle-drop-size`)

### 12. Background Extraction

| Method | Description |
|--------|-------------|
| `dbe` (default) | Dynamic Background Extraction — RBF thin-plate spline fit to sampled patches |
| `mesh` | Legacy polynomial grid (faster, less accurate) |
| `graxpert` | AI-powered gradient removal via GraXpert subprocess |

**DBE details:**
- Star mask applied to sampling patches to avoid fitting to sources
- Extended galaxy/nebula masking preserves target structure
- Sky floor normalisation removes constant per-channel pedestal after background subtraction

### 13. Denoising

#### Wavelet (default on)
- Separates image into luma and chroma components
- BayesShrink adaptive thresholding per subband (or fixed threshold)
- Auto-tunes strength from per-frame SNR estimate
- Star mask applied to protect stellar morphology
- `--denoise-strength`, `--denoise-adaptive`, `--auto-denoise-strength`

#### Chroma Noise Reduction (default on)
- Gaussian blur applied to chroma channels (separate from wavelet)
- `--chroma-nr-sigma` controls blur radius

#### Non-Local Means (`--denoise-nlm`)
- Patch-based denoising using skimage or OpenCV backend
- `--denoise-nlm-strength`, `--denoise-nlm-blend`

#### Bilateral Filter (`--denoise-bilateral`, requires OpenCV)
- Edge-preserving smoothing
- `--denoise-bilateral-sigma-color`, `--denoise-bilateral-sigma-space`

#### Multiscale Median Transform (`--denoise-mmt`)
- Decomposes image into median-residual layers, thresholds each
- `--denoise-mmt-levels`, `--denoise-mmt-strength`

#### ACDNR (`--denoise-acdnr`)
- Adaptive contrast-based denoising using local contrast maps
- `--denoise-acdnr-sigma`, `--denoise-acdnr-k`

#### BM3D (`--denoise-bm3d`)
- Block-matching and 3D collaborative filtering (luminance only)

#### Anisotropic Diffusion (`--denoise-aniso`)
- Perona-Malik diffusion (edge-preserving smoothing)

### 14. PSF Deconvolution (`--deconvolve`)

- Richardson-Lucy iterative deconvolution
- PSF model: Moffat (default) or Gaussian
- PSF FWHM estimated automatically from detected stars or specified manually
- TV-regularised variant (`--deconvolve-tv`) for sharper edges at the cost of speed
- Blind PSF estimation from star median stack (`--deconvolve-blind-psf`)
- `--deconvolve-iterations`, `--deconvolve-fwhm`, `--deconvolve-psf-model`

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

### 24. Hierarchical Processing

- **Auto-detected**: FITS in the root directory → single-folder mode; subdirectories with FITS → hierarchical mode
- Each subfolder processed independently with its own calibration frames
- Final combination: registered and mean-combined with crop to common valid dimensions
- `--keep-intermediates`: saves per-subfolder stacks
- `--combine-sessions`: pools all lights from all subfolders into one unified stack

### 25. Preview & Output Formats

| Format | Flag | Description |
|--------|------|-------------|
| FITS float32 | (always) | Primary output; (3, H, W) layout for maximum compatibility |
| Preview JPEG | (always) | GHS, arcsinh, or linear stretch; 95% quality |
| TIFF 32-bit | `--output-tiff` | 32-bit float RGB for external tools |
| XISF 1.0 | `--output-xisf` | PixInsight native format |
| Per-frame JPEGs | `--export-frames-dir` | Stretched preview for every accepted frame |
| Star mask FITS | `--export-masks` | Star detection mask as `<output>_star_mask.fits` |

**FITS Header metadata populated:**
- Stacking: frame count, rejection count, method, reference frame
- Calibration: which bias/dark/flat masters were applied
- Registration: mean/std shift, maximum shift magnitude
- Quality: mean brightness, contrast, quality scores
- Timing: Phase 1/2/3 durations
- Source metadata: telescope, instrument, observer, exposure time (from input headers)

### 26. Stretch Methods (Preview JPEG)

| Method | Flag | Description |
|--------|------|-------------|
| GHS *(default)* | `--stretch ghs` | Generalised Hyperbolic Stretch; tunable via `--ghs-b`, `--ghs-sp`, `--ghs-hp` |
| Arcsinh | `--stretch arcsinh` | Arcsinh stretch; good for galaxies |
| Linear | `--stretch linear` | No stretch; useful for already-processed data |

### 27. GPU Acceleration

- CuPy backend replaces NumPy/SciPy throughout the pipeline
- `GpuContext` provides a uniform interface: `xp` (array ops), `xndimage`, `xsignal`
- Auto-limits parallel workers based on available VRAM
- Enable with `--use-gpu`; requires `cupy-cuda*` installed (see `requirements-gpu.txt`)

### 28. Auto Target Detection (`--auto`)

Heuristic target classifier — analyses frame metrics (star count, brightness distribution, contrast) and automatically applies optimised parameter sets for the detected target type. No external dependencies or API keys required.

---

## Complete CLI Reference

### Required Arguments

| Flag | Description |
|------|-------------|
| `-d, --directory PATH` | Input directory |
| `-o, --output PATH` | Output FITS path (not required with `--health-check` or `--dry-run`) |

### Configuration & Presets

| Flag | Description |
|------|-------------|
| `--config PATH` | Load parameters from TOML file; CLI args override |
| `--preset NAME` | Apply named preset (quick, quality, galaxy, nebula, narrowband, starfield, planetary, lunar) |
| `--dry-run` | Discover frames, show resolved parameters, estimate resources — no processing |
| `--auto` | Heuristic target classifier |

### Registration

| Flag | Description |
|------|-------------|
| `--no-registration` | Disable all alignment |
| `--no-affine` | Translation-only (disable rotation+scale) |
| `--skip-phase-correlation` | Use fallback methods only |
| `--debug-registration` | Write diagnostics to `_registration_debug/` |

### Quality Filtering

| Flag | Default | Description |
|------|---------|-------------|
| `--quality-filter / --no-quality-filter` | on | Enable/disable frame rejection |
| `--quality-threshold N` | 50 | Reject frames below Nth percentile |

### Stacking

| Flag | Default | Description |
|------|---------|-------------|
| `--stack-method METHOD` | auto | mean, median, sigma_clip, winsorized, percentile, esd, auto |
| `--rejection-sigma N` | 3.0 | Sigma threshold for sigma_clip/winsorized |
| `--rejection-iters N` | 3 | Clipping iterations |
| `--rejection-estimator {mad,std}` | mad | Spread estimator |
| `--percentile-low N` | 20 | Lower rejection percentile |
| `--percentile-high N` | 80 | Upper rejection percentile |
| `--esd-max-outliers N` | 0 (= N//4) | Max outliers per pixel |
| `--esd-significance N` | 0.05 | ESD significance level |
| `--weight-snr N` | 1.0 | SNR weight exponent (0 = disable) |
| `--weight-fwhm N` | 1.0 | FWHM weight exponent (0 = disable) |
| `--weight-stars N` | 1.0 | Star-count weight exponent (0 = disable) |
| `--weight-noise` | off | Add 1/noise² weighting |

### Debayering & Colour

| Flag | Default | Description |
|------|---------|-------------|
| `--debayer-method METHOD` | bilinear | bilinear, malvar (requires cv2), vng (requires cv2) |
| `--white-balance METHOD` | grayworld | none, grayworld, whitepatch |

### Background Extraction

| Flag | Default | Description |
|------|---------|-------------|
| `--background-extraction / --no-background-extraction` | on | Enable/disable |
| `--bg-method {mesh,dbe,graxpert}` | dbe | Extraction method |
| `--dbe-patch-size N` | 64 | DBE sampling patch size (px) |
| `--bg-mesh-size N` | 64 | Legacy mesh cell size (px) |
| `--bg-filter-size N` | 3 | Mesh smoothing filter size (must be odd) |
| `--bg-clip-sigma N` | 3.0 | Star rejection sigma |
| `--graxpert-path PATH` | (auto) | Explicit path to GraXpert binary |

### Denoising

| Flag | Default | Description |
|------|---------|-------------|
| `--denoise / --no-denoise` | on | Wavelet denoising (requires pywt) |
| `--denoise-strength N` | 3.0 | Luma threshold factor |
| `--denoise-adaptive / --no-denoise-adaptive` | on | BayesShrink per-subband |
| `--auto-denoise-strength / --no-auto-denoise-strength` | on | Auto-tune from SNR |
| `--denoise-chroma-boost N` | 2.0 | Chroma threshold multiplier |
| `--chroma-nr / --no-chroma-nr` | on | Chroma Gaussian smoothing |
| `--chroma-nr-sigma N` | 2.0 | Chroma blur radius (px) |
| `--denoise-nlm` | off | Non-local means |
| `--denoise-nlm-strength N` | 1.0 | NLM strength multiplier |
| `--denoise-nlm-blend N` | 0.5 | NLM blend fraction 0–1 |
| `--denoise-bilateral` | off | Bilateral filter (requires cv2) |
| `--denoise-bilateral-sigma-color N` | (auto) | Value-similarity scale in ADU |
| `--denoise-bilateral-sigma-space N` | 3.0 | Spatial smoothing radius (px) |
| `--denoise-mmt` | off | Multiscale Median Transform |
| `--denoise-mmt-levels N` | 4 | MMT decomposition depth |
| `--denoise-mmt-strength N` | 3.0 | MMT soft-threshold multiplier |
| `--denoise-acdnr` | off | ACDNR adaptive contrast denoising |
| `--denoise-acdnr-sigma N` | 1.5 | ACDNR Gaussian radius (px) |
| `--denoise-acdnr-k N` | 3.0 | ACDNR contrast threshold multiplier |
| `--denoise-bm3d` | off | BM3D collaborative filter (luma only) |
| `--denoise-aniso` | off | Perona-Malik anisotropic diffusion |

### Deconvolution

| Flag | Default | Description |
|------|---------|-------------|
| `--deconvolve / --no-deconvolve` | off | Richardson-Lucy deconvolution |
| `--deconvolve-iterations N` | 15 | RL iteration count |
| `--deconvolve-fwhm N` | (auto) | Override PSF FWHM (px) |
| `--deconvolve-psf-model {moffat,gaussian}` | moffat | PSF model |
| `--deconvolve-tv` | off | Total Variation regularisation |
| `--deconvolve-blind-psf` | off | Empirical PSF from star median |

### Other Post-Processing

| Flag | Default | Description |
|------|---------|-------------|
| `--local-normalize` | off | Vignette residual removal |
| `--local-normalize-sigma N` | 50 | Gaussian sigma (px) |
| `--star-reduce / --no-star-reduce` | on | Soften star halos |
| `--star-reduce-factor N` | 0.4 | Blend fraction 0–1 |
| `--star-reduce-sigma N` | 1.5 | Gaussian blur radius (px) |
| `--local-contrast / --no-local-contrast` | on | Multiscale local contrast |
| `--local-contrast-strength N` | 0.7 | Strength 0–1 |
| `--ca-correction / --no-ca-correction` | on | Chromatic aberration correction |
| `--cosmic-ray-rejection / --no-cosmic-ray-rejection` | on | L.A.Cosmic per-frame |
| `--cr-sigclip N` | 4.5 | L.A.Cosmic detection sigma |
| `--cr-objlim N` | 5.0 | L.A.Cosmic object rejection ratio |
| `--scnr` | off | Subtractive Chromatic Noise Reduction |
| `--scnr-amount N` | 1.0 | SCNR amount |
| `--scnr-target {green,cyan}` | green | SCNR target colour |

### Stretch & Output

| Flag | Default | Description |
|------|---------|-------------|
| `--stretch {linear,arcsinh,ghs}` | ghs | Preview JPEG stretch |
| `--ghs-b N` | 8.0 | GHS stretch factor |
| `--ghs-sp N` | 0.15 | GHS symmetry point 0–1 |
| `--ghs-hp N` | 0.95 | GHS highlights protection 0–1 |
| `--output-tiff` | off | Write 32-bit TIFF alongside FITS |
| `--output-xisf` | off | Write XISF 1.0 format |
| `--export-frames-dir PATH` | off | Write per-frame JPEG previews |
| `--export-masks` | off | Save star mask as `_star_mask.fits` |

### Super-Resolution

| Flag | Default | Description |
|------|---------|-------------|
| `--drizzle-scale N` | 1.0 | Output scale (1.0 = off, 2.0 = 2×) |
| `--drizzle-drop-size N` | 0.7 | Pixfrac 0.5–1.0 |

### Advanced & External Tools

| Flag | Description |
|------|-------------|
| `--plate-solve` | Plate solve via astrometry.net/ASTAP |
| `--plate-solver {astap,astrometry}` | Solver backend (default: astrometry) |
| `--astap-path PATH` | Explicit ASTAP binary path |
| `--color-calibrate` | Photometric colour calibration (requires `--plate-solve`) |
| `--photometric-calibration` | Gray-locus colour calibration (no plate-solve dependency) |
| `--gaia-calibration` | Gaia DR3 colour calibration (requires `--plate-solve`) |
| `--star-remove` | Remove stars via Starnet++ |
| `--starnet-path PATH` | Explicit Starnet++ binary path |
| `--comet-mode` | Dual-register for comet tracking |
| `--comet-blend-sigma N` | Blend transition sigma (px) |
| `--hdr-combine PATH` | Blend short-exposure stack for HDR |
| `--hdr-short-exptime N` | Short exposure time (s) |
| `--hdr-long-exptime N` | Long exposure time (s) |
| `--mosaic` | Stitch per-subfolder stacks into mosaic |

### Checkpointing & Diagnostics

| Flag | Description |
|------|-------------|
| `--keep-checkpoint` | Save raw pre-post-processing stack |
| `--no-resume` | Ignore existing checkpoint |
| `--skip-step STEP` | Skip a named post-processing step (repeatable) |
| `--diagnostic` | Save FITS snapshot before each post-processing step |
| `--diagnostic-dir PATH` | Directory for diagnostic snapshots |
| `--quality-report PATH` | Write per-frame metrics to CSV |

Valid step names for `--skip-step`: `hot_pixel`, `background`, `chroma_nr`, `sky_floor`, `local_normalize`, `wavelet`, `sky_residual`, `nlm`, `bilateral`, `mmt`, `acdnr`, `bm3d`, `aniso`, `scnr`, `photo_cal`, `deconvolve`, `star_reduce`, `local_contrast`, `star_remove`

### Infrastructure

| Flag | Default | Description |
|------|---------|-------------|
| `-j, --parallel N` | 0 (auto) | Worker count (1 = sequential) |
| `--use-gpu` | off | CuPy GPU acceleration |
| `-v, --verbose` | off | Detailed per-frame output |
| `--log-level LEVEL` | WARNING | Stderr log level (DEBUG, INFO, WARNING, ERROR) |
| `--log-file PATH` | off | Write full DEBUG log to file |
| `--health-check` | — | Analyse frames + calibration without stacking |
| `--keep-intermediates` | off | Save per-subfolder stacks (hierarchical mode) |
| `--combine-sessions` | off | Pool all lights from subfolders into one stack |

---

## Example Commands

```bash
# Single folder, all defaults
python astro_stack.py -d lights/ -o stacked.fits

# Verbose with auto target detection
python astro_stack.py -d lights/ -o stacked.fits -v --auto

# Hierarchical session with intermediates
python astro_stack.py -d session/ -o combined.fits --keep-intermediates -v

# Galaxy preset with deconvolution
python astro_stack.py -d lights/ -o galaxy.fits --preset galaxy --deconvolve -v

# Maximum quality
python astro_stack.py -d lights/ -o best.fits --preset quality \
  --denoise-nlm --denoise-bilateral --denoise-mmt --denoise-acdnr \
  --deconvolve --local-normalize -v

# Super-resolution drizzle
python astro_stack.py -d lights/ -o drizzled.fits --drizzle-scale 2.0 -v

# Plate solve + colour calibration
python astro_stack.py -d lights/ -o stacked.fits --plate-solve --color-calibrate -v

# Debug registration problems
python astro_stack.py -d lights/ -o stacked.fits --debug-registration

# Health check only
python astro_stack.py -d lights/ --health-check


# Minimal run (turn off new-default post-processing)
python astro_stack.py -d lights/ -o stacked.fits \
  --no-star-reduce --no-local-contrast --no-background-extraction
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
| `opencv-python` | Malvar/VNG debayer, bilateral filter, some NLM paths |
| `PyWavelets` | Wavelet denoising (`--denoise`) |
| `sep` | 5–10× faster star detection than DAOStarFinder |
| `astroquery >= 0.4.6` | Plate solving via nova.astrometry.net |
| `cupy-cuda*` | GPU acceleration (`--use-gpu`; see `requirements-gpu.txt`) |
| `reproject` | Mosaic WCS reprojection (`--mosaic`) |
| `tifffile` | 16-bit TIFF output |

---

## Known Limitations

**Registration:**
- Very sparse star fields or pure extended nebulae (no stars) may fall back to centroid matching
- Use `--no-registration` for pre-aligned frames

**Debayering:**
- Bilinear (default) can produce colour fringing; use `--debayer-method malvar` for better quality (requires OpenCV)

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
