# Astrophotography FITS Stacker - Complete Project Specification

## Project Overview
A professional-grade Python tool for stacking astronomical FITS images from telescopes (particularly Celestron Origin). Processes unlimited images on limited hardware using streaming architecture.

## Code Structure

The codebase is split into focused modules under `src/`. `astro_stack.py` (root) is a thin backward-compatibility shim.

| Module | Lines | Contents |
|--------|-------|----------|
| `src/gpu_context.py` | ~125 | `GpuContext`, CUDA stream contexts, `get_gpu()` |
| `src/models.py` | ~80 | `Config`, `FrameInfo`, `ProcessingStats` |
| `src/utils.py` | ~115 | Print helpers, `format_time`, `get_memory_usage_mb` |
| `src/io_fits.py` | ~310 | FITS load/save, `make_master`, `populate_fits_header` |
| `src/frame_discovery.py` | ~125 | `discover_frames`, `classify_frame`, `select_matching_darks` |
| `src/debayer.py` | ~250 | Debayering, hot pixels, white balance |
| `src/quality.py` | ~300 | `compute_quality_metrics`, star detection, FWHM |
| `src/psf_deconvolution.py` | ~195 | PSF estimation (Moffat/Gaussian), Richardson-Lucy |
| `src/background.py` | ~700 | Mesh-based sky extraction, residual removal, floor normalisation |
| `src/denoising.py` | ~450 | Wavelet, bilateral, NLM, chroma NR, arcsinh stretch |
| `src/registration.py` | ~480 | `calculate_shift`, affine/RANSAC, `calc_common_crop` |
| `src/stacking.py` | ~370 | Sigma-clip combine, drizzle, Lanczos resampling |
| `src/plate_solve.py` | ~150 | Astrometry.net + SIMBAD identification |
| `src/pipeline.py` | ~730 | Three-phase pipeline, parallel workers, `stack_target` |
| `src/health_check.py` | ~180 | Frame consistency and calibration analysis |
| `src/cli.py` | ~480 | `process_directory`, `parse_args`, `main` |
| `astro_stack.py` | ~165 | Backward-compat re-export shim |

**Total: ~5,200 lines. Tests: `tests/test_core.py` imports symbols directly from `astro_stack`.**

## Core Architecture

### Memory Management
- **Streaming architecture**: Never loads all images at once
- **Per-image processing**: Load → Process → Stack → Free → Next
- **Memory usage**: Constant ~1-2 images worth regardless of total count
- **Traditional approach**: 10-20GB for 50 images
- **This approach**: 0.4-1.2GB for 50 images (~13x reduction)

### Three-Phase Processing
1. **Phase 1: Validation & Quality Analysis** — Load, calibrate, debayer, analyse quality, reject bad frames
2. **Phase 2: Registration** — Calculate alignment shifts, select reference frame
3. **Phase 3: Stacking** — Align, crop valid region, accumulate, combine

## Key Features

### 1. Automatic Frame Detection & Classification
- Automatically identifies file types from filenames and FITS headers
- Patterns: `light_*.fit`, `dark_*.fit`, `flat_*.fit`, `bias_*.fit`
- Header keywords: `IMAGETYP`, `FRAME`
- Heuristics: Zero exposure = bias
- Default: Unidentified files treated as lights (safe)

### 2. Automatic Calibration
- **Per-target calibration**: Each subfolder uses its own calibration frames
- **Master frame creation**: Median combines darks/flats/bias
- **Dark matching**: Selects darks matching light frame ISO, exposure time, and dimensions
- **Calibration smoothing**: Masters are Gaussian-smoothed (per Bayer channel for flats) to avoid adding correlated noise to lights
- **Hot pixel map**: Built from dark frame before smoothing; applied per Bayer sub-channel
- **Calibration order**: Bias subtraction → Dark subtraction (exposure-scaled) → Flat division
- **Flat normalisation**: Per Bayer channel, avoids division by zero

### 3. Bayer Pattern Debayering
- **Auto-detection**: From FITS headers (`BAYERPAT`, `COLORTYP`)
- **Supported patterns**: RGGB, BGGR, GRBG, GBRG; default RGGB
- **Algorithms**:
  - `bilinear` (default) — pure numpy, fast
  - `malvar` — Malvar-He-Cutler, higher quality, requires cv2
  - `vng` — Variable Number of Gradients, requires cv2
- **Output**: RGB images in (H, W, 3) format

### 4. White Balance
- **Gray-world** (default): scale each channel to equalise mean
- **White-patch**: scale to 99.5th-percentile white point
- **None**: skip white balance

### 5. Hot Pixel Removal
- Per-Bayer-channel MAD-based detection and interpolation
- Optional pre-built map from dark frame (more accurate)
- Applied in raw space before debayering

### 6. Quality Analysis & Filtering
- **Metrics calculated per frame**:
  - Brightness (median pixel value)
  - Contrast (standard deviation)
  - Star count (via photutils DAOStarFinder, optional)
  - FWHM (median Full Width at Half Maximum of detected stars)
  - SNR estimate
  - Composite quality score
- **Filtering**: Percentile-based rejection (default: reject lowest 25%, i.e. keep best 75%)
- Quality filter is **on by default** (`--no-quality-filter` to disable)
- Hard rejection: blank/corrupt frames (brightness < 10), flat/underexposed (contrast < 1)

### 7. Image Registration (Alignment)
- **Primary**: Phase cross-correlation (skimage) — sub-pixel accurate
- **Fallback**: FFT cross-correlation on downscaled image
- **Second fallback**: Centroid of thresholded bright regions
- **Affine registration** (default): Star matching + RANSAC for rotation and scale correction; disable with `--no-affine`
- **Reference frame**: Highest-quality accepted frame
- **Validation**: Rejects unrealistic shifts (> 10% of image size)
- **Dither detection**: Automatically selects sigma-clip stacking when dithering is detected

### 8. Automatic Cropping (Anti-Black-Border)
- Calculates valid overlap region present in all aligned frames
- Reports crop amount (rows and columns removed)
- 2-pixel safety margin

### 9. Background Extraction
- **Enabled by default** (`--no-background-extraction` to disable)
- Mesh-based sky estimation with configurable cell size (`--bg-mesh-size`, default 64px)
- Sigma-clipped statistics per cell to reject stars and extended objects
- Spline interpolation across the mesh produces a smooth background model
- Galaxy/extended-object masking to preserve nebulosity
- Sky floor normalisation: subtracts constant sky pedestal per channel

### 10. Stacking Methods
- `mean` — fastest, no rejection
- `median` — robust, uses temporary memmaps for large datasets
- `sigma_clip` (default when dithering detected) — MAD-based tiled sigma-clipping with configurable sigma and iterations
- **Winsorize option**: clip outliers to boundary value instead of rejecting
- **Drizzle**: Lanczos-interpolated sub-pixel accumulation for super-resolution (`--drizzle-scale`)

### 11. Post-Stack Processing Chain
Applied in order after stacking:
1. Background extraction (if enabled)
2. Wavelet denoising (luma + chroma, star-protected) — `--denoise`
3. Non-local means denoising — `--denoise-nlm`
4. Bilateral filter denoising — `--denoise-bilateral`
5. Chroma noise reduction — `--chroma-nr` (on by default)
6. Local normalisation (vignette residual removal) — `--local-normalize`
7. Richardson-Lucy deconvolution (PSF sharpening) — `--deconvolve`

### 12. Plate Solving
- Via nova.astrometry.net (requires `astroquery` and `ASTROMETRY_API_KEY`)
- Writes WCS solution to output FITS header
- Identifies deep-sky objects in the field via SIMBAD
- Enable with `--plate-solve`

### 13. Hierarchical Processing
- **Auto-detection**: FITS in root → single mode; subfolders with FITS → hierarchical mode
- Each subfolder stacked independently with its own calibration frames
- Final combination: registered and averaged with common-dimension crop
- `--keep-intermediates` saves individual subfolder stacks

### 14. GPU Acceleration
- CuPy backend, opt-in via `--use-gpu`
- `GpuContext` provides uniform `xp`/`xndimage`/`xsignal` interface (numpy or cupy)
- Auto-limits parallel workers based on available VRAM

### 15. Output Generation
- **FITS format**: (3, H, W) float32 for maximum compatibility
- **Metadata**: frame count, rejection count, target name, processing parameters
- **Preview JPEG**: arcsinh or linear stretch (default arcsinh), 95% quality
- **Plate-solve WCS**: written to FITS header when solved

### 16. Parallel Processing
- `-j N` workers (0 = auto-detect CPU count, 1 = sequential)
- `ProcessPoolExecutor` for CPU, auto-limited for GPU

### 17. Health Check
- `--health-check`: analyse calibration quality and frame consistency without stacking
- Reports bias noise, dark thermal current rate, flat vignetting, ISO mismatches

## Directory Structure Support

### Single Folder
```
lights/
├── dark_001.fit (optional)
├── flat_001.fit (optional)
├── bias_001.fit (optional)
├── light_001.fit
├── light_002.fit
└── ...
```

### Hierarchical
```
session/
├── M31/
│   ├── dark_001.fit
│   ├── flat_001.fit
│   ├── bias_001.fit
│   └── light_001.fit (many)
├── M42/
│   ├── dark_001.fit
│   └── light_001.fit (many)
└── NGC7000/
    └── light_001.fit (many, no calibration OK)
```

## Command Line Interface

### Required Arguments
- `-d, --directory` — Input directory (single or with subfolders)
- `-o, --output` — Output FITS file path (optional with `--health-check`)

### Registration
- `--no-registration` — Disable all alignment
- `--no-affine` — Use translation-only registration (no rotation/scale)
- `--skip-phase-correlation` — Skip phase correlation, use fallback methods only (debug)
- `--debug-registration` — Write detailed diagnostics to `_registration_debug/` (implies `-v`)

### Quality Filtering
- `--no-quality-filter` — Keep all frames regardless of quality
- `--quality-threshold N` — Reject frames below Nth percentile (default: 25 = keep best 75%)

### Stacking
- `--stack-method {mean,median,sigma_clip}` — Override auto-selected method
- `--rejection-sigma N` — Sigma threshold for sigma_clip (default: 3.0)
- `--rejection-iters N` — Clipping iterations for sigma_clip (default: 3)
- `--winsorize` — Clip outliers to boundary instead of rejecting

### Debayering & Colour
- `--debayer-method {bilinear,malvar,vng}` — Demosaicing algorithm (default: bilinear)
- `--white-balance {none,grayworld,whitepatch}` — White balance method (default: grayworld)

### Background Extraction
- `--no-background-extraction` — Disable mesh-based background removal
- `--bg-mesh-size N` — Grid cell size in pixels (default: 64)
- `--bg-filter-size N` — Median filter size for grid smoothing (default: 3, must be odd)
- `--bg-clip-sigma N` — Sigma for star rejection (default: 3.0)

### Denoising
- `--denoise` — Enable wavelet denoising
- `--denoise-strength N` — Wavelet luma threshold factor (default: 3.0)
- `--denoise-chroma-boost N` — Chroma threshold multiplier (default: 2.0)
- `--denoise-nlm` — Enable non-local means after wavelet
- `--denoise-nlm-strength N` — NLM strength multiplier (default: 1.0)
- `--denoise-nlm-blend N` — Blend fraction 0–1 (default: 0.5)
- `--denoise-bilateral` — Enable bilateral filter
- `--denoise-bilateral-sigma-color N` — Value-similarity scale in ADU (default: auto)
- `--denoise-bilateral-sigma-space N` — Spatial smoothing radius in pixels (default: 3.0)
- `--no-chroma-nr` — Disable chroma noise reduction (on by default)
- `--chroma-nr-sigma N` — Gaussian sigma for chroma smoothing (default: 2.0)

### Deconvolution
- `--deconvolve` — Enable Richardson-Lucy sharpening
- `--deconvolve-iterations N` — RL iterations (default: 15)
- `--deconvolve-fwhm N` — Override auto-estimated PSF FWHM in pixels
- `--deconvolve-psf-model {moffat,gaussian}` — PSF model (default: moffat)

### Other Post-Processing
- `--local-normalize` — Remove vignetting residuals via local normalisation
- `--local-normalize-sigma N` — Gaussian sigma (default: 50)
- `--stretch {linear,arcsinh}` — Preview stretch method (default: arcsinh)

### Super-Resolution
- `--drizzle-scale N` — Drizzle output scale factor (1.0 = disabled, 2.0 = 2× super-res)
- `--drizzle-drop-size N` — Pixfrac/drop size 0.5–1.0 (default: 0.7)

### Infrastructure
- `--use-gpu` — CuPy GPU acceleration (experimental)
- `--plate-solve` — Plate solve via astrometry.net
- `-j N, --parallel N` — Worker count (0 = auto, 1 = sequential, default: 1)
- `--keep-intermediates` — Save per-subfolder stacks in hierarchical mode
- `-v, --verbose` — Detailed per-frame output
- `--health-check` — Analyse frames and calibration without stacking

### Example Commands
```bash
# Single folder, basic
python astro_stack.py -d lights/ -o stacked.fits

# With verbose output
python astro_stack.py -d lights/ -o stacked.fits -v

# Hierarchical with quality filtering and intermediates
python astro_stack.py -d session/ -o combined.fits --keep-intermediates -v

# Full post-processing pipeline
python astro_stack.py -d lights/ -o stacked.fits --denoise --deconvolve --stretch arcsinh

# Check calibration quality without stacking
python astro_stack.py -d lights/ --health-check

# Debug registration problems
python astro_stack.py -d lights/ -o stacked.fits --debug-registration

# GPU + drizzle super-resolution
python astro_stack.py -d lights/ -o stacked.fits --use-gpu --drizzle-scale 2.0
```

## Dependencies

### Required
- `numpy >= 1.20` — Array operations
- `astropy >= 5.0` — FITS file I/O
- `scipy >= 1.7` — Image shifting, interpolation, Gaussian filters
- `scikit-image >= 0.21` — Phase cross-correlation, morphology, restoration

### Included in `requirements.txt` (strongly recommended)
- `photutils >= 1.5` — Star detection (quality analysis; degrades without it)
- `tqdm >= 4.65` — Progress bars (falls back to plain iterator)
- `Pillow >= 9.0` — Preview JPEG generation
- `psutil >= 5.9` — Memory usage reporting

### Optional (install separately as needed)
- `opencv-python` (`cv2`) — Malvar/VNG debayer, bilateral denoising
- `pywt` (`PyWavelets`) — Wavelet denoising (`--denoise`)
- `cupy-cudaXXx` — GPU acceleration (see `requirements-gpu.txt`)
- `astroquery >= 0.4.6` — Plate solving (see `requirements-astrometry.txt`)

## Performance Characteristics

### Memory
- **50 images, 4096×4096**: 0.4-1.2 GB (vs 10-20 GB traditional)
- **Scalability**: Tested up to 500 images, constant memory
- **Peak usage**: 1-2 images worth at any time

### Speed (CPU, 8 threads)
- **Mean stacking**: 30-60s for 50 images
- **Sigma-clip stacking**: 60-120s for 50 images
- **Quality filtering**: saves time by rejecting bad frames early

### Output Quality
- **Alignment accuracy**: Sub-pixel (phase correlation)
- **Dynamic range**: Full float32 precision
- **Crop amount**: Typically 50-100 pixels per edge for well-tracked sessions

## Known Limitations

### Registration
- Very sparse star fields or pure extended nebulae without stars may cause fallback to centroid method
- Use `--no-registration` for pre-aligned frames or problematic targets

### Debayering
- Bilinear (default) is fast but can produce colour fringing; use `malvar` for better quality (requires cv2)

### Hierarchical Mode
- Targets with very different crop amounts produce shape mismatches; handled by resizing to minimum common dimensions (slight quality loss)
- Better to process targets with very different tracking separately

### Drizzle
- Requires dithered frames to produce sharper output; with no dither it just upsamples

## Design Patterns Used

1. **Streaming Architecture** — Process one frame at a time, free immediately
2. **Dataclasses** — Type-safe configuration (`Config`) and per-frame info (`FrameInfo`)
3. **Optional dependency pattern** — Every optional import wrapped in `try/except` with `HAS_X` flag
4. **Singleton** — `get_gpu()` returns the module-level `GpuContext` instance
5. **Strategy Pattern** — Debayer method, stacking method, stretch method all dispatched by name
6. **Separation of Concerns** — Each `src/` module owns one domain; `pipeline.py` wires them together
