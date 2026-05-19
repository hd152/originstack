# OriginStack

**Streaming FITS stacker for astrophotography — runs on ordinary hardware, scales to any frame count.**

OriginStack is a full-featured Python pipeline for stacking and processing astronomical FITS images. It was designed for the Celestron Origin smart telescope but works with FITS files from any OSC/DSLR/mirrorless camera. The core design principle is a **streaming architecture**: frames are loaded, processed, and freed one at a time, so memory usage stays constant regardless of how many frames you have.

---

## Sample Result

![Whirlpool Galaxy (M51)](sample/whirlz1.jpg)

*Whirlpool Galaxy (M51) — stacked and processed entirely with OriginStack from raw Celestron Origin FITS frames.*

---

## Sample Output

<details>
<summary>Click to expand — verbose run on 300 Whirlpool Galaxy frames</summary>

```
======================================================================
Astrophotography FITS Stacker
======================================================================
Input:  lights/Whirlpool_Galaxy
Output: whirlpool.fits
  Compute: CPU

Discovering frames...
  Mode: Single folder
  Found 303 FITS files: 300 lights, 1 darks, 1 flats, 1 bias

Creating master calibration frames...
  [OK] Master bias:  1 frames -> 2048x3056
  [OK] Master dark:  1 frames -> 2048x3056
  [OK] Master flat:  1 frames -> 2048x3056
  [OK] Hot pixel map: 214576 pixels from dark frame
    Bias:  pedestal=4990.7 ADU  noise=0.8 ADU  -> Good (low read noise)
    Dark:  median=4917.1 ADU  temp=31.3°C  exp=30.0s  ISO=200  -> OK
    Flat:  R=0.765/G1=1.114/G2=1.122/B=0.922  vignetting=3.4%  -> Good

======================================================================
PHASE 1: PROCESSING & QUALITY ANALYSIS
======================================================================
  Processing 300 frames in parallel (8 workers)...

  Frame Quality Details:
  ---------------------------------------------------------------------------------------------------------------
  Frame                            Bright       Bg   Noise   SNR  Stars   FWHM    Sharp      Score  St
  ---------------------------------------------------------------------------------------------------------------
  Light0001.fits                   4558.1   4558.1  101.99   1.7     78    7.0   236280       56.4   [OK]
  Light0002.fits                   4531.2   4531.2  101.13   1.7     77    7.0   224233       56.4   [OK]
  Light0003.fits                   4499.2   4499.2  104.04   1.6     73    7.1   248395       53.5   [OK]
  Light0004.fits                   4491.5   4491.5  100.00   1.7     72    6.8   217981       58.1   [OK]
  Light0005.fits                   4458.1   4458.1  104.04   1.6     71    6.6   244558       57.8   [OK]
  ...
  ---------------------------------------------------------------------------------------------------------------
  [OK] Accepted: 287/300 (95.7%)
  [X] Rejected:  13 (quality threshold)

  Target: Whirlpool Galaxy [Galaxy]  conf=90%  source=header

======================================================================
PHASE 2: REGISTRATION
======================================================================
  Reference frame: Light0009.fits (score=60.9)
  Calculating shifts for 287 frames...
    Light0001.fits: affine shift=(-6.0, +5.1) px, rotation=+0.000 deg
    Light0002.fits: affine shift=(-2.6, +1.4) px, rotation=+0.000 deg
    Light0003.fits: affine shift=(-1.8, +1.1) px, rotation=-0.003 deg
    Light0004.fits: affine shift=(-0.4, +0.4) px, rotation=-0.003 deg
    Light0005.fits: affine shift=(+0.0, +0.0) px, rotation=+0.000 deg  [reference]
    ...
  Shift statistics:
    X: mean=+13.7px, std=16.1px, range=[-6.0, +33.9]
    Y: mean=-11.7px, std=13.8px, range=[-30.4, +5.1]
    Magnitude: mean=20.8px, max=43.3px
  Dither pattern detected — sigma_clip stacking recommended
  Tip: dithered data detected — add --drizzle-scale 2.0 for super-resolution

======================================================================
PHASE 3: STACKING
======================================================================
  Method: sigma_clip (sigma=3.0, iters=3, estimator=MAD)
  Quality weights: min=0.937, max=1.000, mean=0.972
  Tiled sigma-clip: 96 tiles of 256x256

======================================================================
PHASE 4: POST-PROCESSING
======================================================================
  Removing residual hot pixels (per-channel)...
  [OK] Per-channel hot pixel removal: 144 pixels fixed (5.9s)
    Post-processing star mask: 115 stars

  Applying Dynamic Background Extraction (patch=64px, RBF thin-plate-spline)...
  [OK] Dynamic Background Extraction (24.5s)

  Applying chroma noise reduction (sigma=2.0)...
  [OK] Chroma noise reduction (1.8s)

  Applying adaptive wavelet denoising (BayesShrink, chroma_factor=2.0)...
  [OK] Wavelet denoise (2.1s)

  Correcting sky residuals...
  [OK] Sky residual correction (35.1s)

  Applying star reduction (factor=0.40, blur_sigma=1.5)...
  [OK] Star reduction (0.9s)

  Applying multiscale local contrast enhancement (strength=0.70)...
  [OK] Local contrast enhancement (3.2s)

  Output size: 3036x2030 (cropped 20x18 pixels)

======================================================================
SUMMARY
======================================================================
  Frames analyzed:  300
  Frames stacked:   287 (95.7%)
  Integration time: 2h 23m
  Output:           whirlpool.fits (3036x2030x3)
  Preview:          whirlpool.jpg (ghs stretch)
  Avg FWHM:         6.73 px (best: 6.38)
  Avg SNR:          1.7  (best: 1.7)
  Processing time:  18m 42s
    Quality+Load:   4m 21s
    Registration:   3m 15s
    Stacking:       1m 44s
    Post-process:   9m 22s
  Peak memory:      1477.4 MB
======================================================================
```

</details>

---

## Features at a Glance

### Core Pipeline
- **Four-phase pipeline**: quality analysis → registration → stacking → post-processing
- **Streaming memory model**: ~0.4–1.2 GB for 50 × 4K frames (vs 10–20 GB with traditional approaches)
- **Automatic calibration**: bias, dark, and flat master frames built and applied automatically
- **Parallel processing**: multi-core via `ProcessPoolExecutor`; GPU acceleration via CuPy (`--use-gpu`)

### Registration & Alignment
- Sub-pixel phase cross-correlation (scikit-image) with FFT and centroid fallbacks
- **Affine registration** via star matching + RANSAC — corrects rotation, scale, and translation
- Automatic dither detection → selects sigma-clip stacking automatically
- Crops to the valid common region across all frames (no black borders)

### Stacking Methods (7)
| Method | Best For |
|--------|----------|
| `auto` *(default)* | Selects automatically based on frame count |
| `sigma_clip` | Most sessions (MAD-based iterative rejection) |
| `percentile` | Fewer than 8 frames |
| `esd` | Fewer than 15 frames (Grubbs/ESD statistical test) |
| `winsorized` | Like sigma_clip but clips to boundary |
| `median` | Robust, no tuning required |
| `mean` | Fastest, no rejection |

Drizzle super-resolution (`--drizzle-scale 2.0`) uses Lanczos-interpolated sub-pixel accumulation.

### Quality Filtering
- Per-frame metrics: brightness, contrast, star count, FWHM, SNR, composite score
- Percentile-based rejection (default: keep best 75%, `--quality-threshold`)
- Hard rejection: blank, corrupt, or severely underexposed frames
- Quality-weighted stacking (SNR, FWHM, star count weighting)

### Post-Processing Chain (up to 20 steps)
Applied in order after stacking. Steps marked ✅ are on by default; ❌ must be explicitly enabled:

1. ✅ Hot pixel removal on stacked image
2. ✅ Star mask generation (protects structure in subsequent steps)
3. ✅ Background extraction (DBE via RBF thin-plate spline, or legacy mesh, or GraXpert AI)
4. ✅ Chroma noise reduction
5. ✅ Sky floor normalisation (per-channel pedestal removal)
6. ❌ Local normalisation — `--local-normalize`
7. ✅ Wavelet denoising — BayesShrink adaptive, auto-tuned from SNR
8. ✅ Sky residual correction (second pass after denoising)
9. ❌ Non-local means denoising — `--denoise-nlm`
10. ❌ Bilateral filter — `--denoise-bilateral`
11. ❌ Multiscale Median Transform (MMT) — `--denoise-mmt`
12. ❌ ACDNR adaptive contrast denoising — `--denoise-acdnr`
13. ❌ BM3D collaborative filter — `--denoise-bm3d`
14. ❌ Perona-Malik anisotropic diffusion — `--denoise-aniso`
15. ❌ Subtractive Chromatic Noise Reduction — `--scnr`
16. ❌ Photometric colour calibration — `--photometric-calibration`
17. ❌ Richardson-Lucy deconvolution — `--deconvolve`
18. ✅ Star reduction (softens star cores) — `--no-star-reduce` to disable
19. ✅ Multiscale local contrast enhancement (MLCE) — `--no-local-contrast` to disable
20. ❌ Star removal via Starnet++ — `--star-remove`

### Presets
Eight built-in target presets tune all parameters at once:

```bash
--preset galaxy       # GHS stretch, star reduction, bilateral filter
--preset nebula       # GHS stretch, MMT + ACDNR denoising
--preset narrowband   # Tuned for Ha/OIII/SII narrow-band data
--preset starfield    # No star reduction, minimal processing
--preset planetary    # No background extraction, deconvolution enabled
--preset lunar        # Linear stretch, no star reduction
--preset quick        # Mean stack, minimal post-processing (fastest)
--preset quality      # All denoisers, sigma-clip, deconvolution (best output)
```

### Advanced Features
- **Plate solving** via ASTAP or nova.astrometry.net — writes WCS to FITS header, identifies objects via SIMBAD
- **Photometric colour calibration** — gray-locus method, optional Gaia DR3 extension
- **Star removal** via Starnet++ — saves `_starless.fits` and `_stars.fits`
- **Comet nucleus tracking** — dual-registered stacks (`_comet.fits`)
- **HDR combining** — blends short/long exposure stacks for high-dynamic-range targets
- **Mosaic stitching** — WCS-based reprojection via `reproject` (`--mosaic`)
- **Checkpointing** — save raw pre-post stack for iterative post-processing (`--keep-checkpoint`)
- **Diagnostic snapshots** — FITS snapshots before each post-processing step (`--diagnostic`)
- **Quality CSV** — per-frame metrics exported for external analysis (`--quality-report`)

---

## Installation

Requires Python 3.9+.

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/originstack.git
cd originstack

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Optional: GPU support (requires CUDA toolkit)
pip install -r requirements-gpu.txt

# 5. Optional: plate solving
pip install -r requirements-astrometry.txt

# 6. Optional: higher-quality debayer and bilateral denoising
pip install opencv-python

# 7. Optional: wavelet denoising
pip install PyWavelets
```

**Optional dependencies** — all gracefully degraded when absent:

| Package | Feature |
|---------|---------|
| `opencv-python` | Malvar/VNG debayer, bilateral filter |
| `PyWavelets` | Wavelet denoising |
| `sep` | 5–10× faster star detection |
| `astroquery` | Plate solving via astrometry.net |
| `cupy-cuda*` | GPU acceleration |
| `reproject` | Mosaic stitching |

---

## Quick Start

### Single folder of lights

```bash
python astro_stack.py -d lights/ -o stacked.fits
```

OriginStack will automatically detect any calibration frames (`dark_*.fit`, `flat_*.fit`, `bias_*.fit`) in the same directory, build master frames, stack the lights, and write a FITS file and a preview JPEG.

### Verbose output with quality metrics

```bash
python astro_stack.py -d lights/ -o stacked.fits -v
```

Shows per-frame brightness, contrast, star count, SNR, and shift magnitude as each frame is processed.

### Auto target detection

```bash
python astro_stack.py -d lights/ -o stacked.fits --auto
```

Analyses your frames and applies optimised settings for the detected target type (galaxy, nebula, star field, etc.) — no manual tuning required.

### Hierarchical session (multiple targets in one night)

```bash
python astro_stack.py -d session/ -o combined.fits --keep-intermediates -v
```

Where `session/` contains one subfolder per target. Each subfolder is stacked independently with its own calibration frames, then combined into a single output.

---

## Usage Examples

### Galaxy (e.g., M51, M81)

```bash
python astro_stack.py -d lights/ -o galaxy.fits \
  --preset galaxy \
  --debayer-method malvar \
  --stack-method sigma_clip \
  --rejection-sigma 2.8 \
  --deconvolve \
  -v
```

The `galaxy` preset applies GHS stretching and star reduction. Adding `--deconvolve` sharpens fine detail in spiral arms.

### Emission nebula (e.g., Orion, Rosette)

```bash
python astro_stack.py -d lights/ -o nebula.fits \
  --preset nebula \
  --denoise-mmt \
  --denoise-acdnr \
  --stretch ghs --ghs-b 8.0 \
  -v
```

### Narrow-band (Ha/OIII/SII)

```bash
python astro_stack.py -d ha_lights/ -o ha_stack.fits \
  --preset narrowband \
  --no-white-balance \
  --scnr \
  --stack-method sigma_clip --rejection-sigma 3.0 \
  -v
```

### Planetary / lunar

```bash
python astro_stack.py -d frames/ -o jupiter.fits \
  --preset planetary \
  --deconvolve --deconvolve-iterations 20 \
  --no-background-extraction \
  --no-star-reduce \
  -v
```

### Maximum quality — everything on

```bash
python astro_stack.py -d lights/ -o best.fits \
  --preset quality \
  --debayer-method malvar \
  --denoise-nlm --denoise-bilateral --denoise-mmt --denoise-acdnr \
  --deconvolve \
  --local-normalize \
  --stack-method sigma_clip --rejection-sigma 2.5 \
  --weight-snr 2.0 --weight-fwhm 2.0 \
  -v
```

### Super-resolution drizzle (requires dithered frames)

```bash
python astro_stack.py -d lights/ -o drizzled.fits \
  --drizzle-scale 2.0 \
  --drizzle-drop-size 0.7 \
  -v
```

### Plate solving + colour calibration

```bash
# Set your API key first
export ASTROMETRY_API_KEY=your_key_here   # Linux/macOS
set ASTROMETRY_API_KEY=your_key_here      # Windows

python astro_stack.py -d lights/ -o stacked.fits \
  --plate-solve \
  --color-calibrate \
  -v
```

### Debug registration problems

```bash
python astro_stack.py -d lights/ -o stacked.fits --debug-registration
```

Writes PNG overlay images and shift statistics to `_registration_debug/`. Use this when frames aren't aligning correctly.

### Health check without stacking

```bash
python astro_stack.py -d lights/ --health-check
```

Analyses calibration quality (bias noise, dark thermal current, flat vignetting) and reports any ISO or dimension mismatches — without actually stacking anything.

### Save a config file for reuse

```bash
# First run with --dry-run to see resolved parameters
python astro_stack.py -d lights/ -o stacked.fits --preset galaxy --deconvolve --dry-run

# Then use --config to reapply the same settings
python astro_stack.py -d lights/ -o stacked.fits --config my_settings.toml
```

---

## Directory Structure

### Single folder mode

```
lights/
├── bias_001.fit          (optional)
├── dark_001.fit          (optional)
├── flat_001.fit          (optional)
├── light_001.fit
├── light_002.fit
└── ...
```

### Hierarchical mode (multi-target session)

```
session/
├── M31/
│   ├── dark_001.fit
│   ├── flat_001.fit
│   └── light_001.fit ... light_NNN.fit
├── M42/
│   ├── dark_001.fit
│   └── light_001.fit ... light_NNN.fit
└── NGC7000/
    └── light_001.fit ... light_NNN.fit   (no calibration = OK)
```

OriginStack auto-detects which mode to use: FITS files in the root → single-folder mode; subdirectories containing FITS → hierarchical mode.

---

## Post-Processing Default Flags

Most post-processing is **on by default**. Here are the disable flags:

| Feature | Default | Disable with |
|---------|---------|-------------|
| Background extraction (DBE) | ✅ on | `--no-background-extraction` |
| Wavelet denoising | ✅ on | `--no-denoise` |
| Chroma noise reduction | ✅ on | `--no-chroma-nr` |
| Star reduction | ✅ on | `--no-star-reduce` |
| Local contrast enhancement | ✅ on | `--no-local-contrast` |
| Chromatic aberration correction | ✅ on | `--no-ca-correction` |
| Cosmic ray rejection | ✅ on | `--no-cosmic-ray-rejection` |
| Quality filtering | ✅ on | `--no-quality-filter` |
| Affine registration | ✅ on | `--no-affine` |
| NLM denoising | ❌ off | `--denoise-nlm` |
| Bilateral denoising | ❌ off | `--denoise-bilateral` |
| MMT denoising | ❌ off | `--denoise-mmt` |
| ACDNR denoising | ❌ off | `--denoise-acdnr` |
| Richardson-Lucy deconvolution | ❌ off | `--deconvolve` |
| Local normalisation | ❌ off | `--local-normalize` |

---

## CLI Reference (Abridged)

```
python astro_stack.py -d <dir> -o <output.fits> [options]
```

| Flag | Description |
|------|-------------|
| `-d, --directory` | Input directory (required) |
| `-o, --output` | Output FITS path (required unless `--health-check` or `--dry-run`) |
| `--preset NAME` | Apply named preset (quick, quality, galaxy, nebula, narrowband, starfield, planetary, lunar) |
| `--config PATH` | Load parameters from TOML file |
| `--auto` | Heuristic target classifier — detect and optimise automatically |
| `--stack-method METHOD` | Stacking algorithm (auto, mean, median, sigma_clip, percentile, esd, winsorized) |
| `--debayer-method METHOD` | Debayer algorithm (bilinear, malvar, vng) |
| `--white-balance METHOD` | White balance (grayworld, whitepatch, none) |
| `--bg-method METHOD` | Background extraction (dbe, mesh, graxpert) |
| `--drizzle-scale N` | Super-resolution scale (1.0 = off, 2.0 = 2×) |
| `--deconvolve` | Enable Richardson-Lucy PSF deconvolution |
| `--plate-solve` | Plate solve via astrometry.net (requires astroquery + API key) |
| `--star-remove` | Remove stars via Starnet++ (saves _starless.fits + _stars.fits) |
| `--comet-mode` | Dual-register for comet nucleus tracking |
| `--hdr-combine PATH` | Blend short-exposure stack for HDR |
| `--mosaic` | Stitch per-subfolder stacks via WCS reprojection |
| `--keep-checkpoint` | Save raw pre-post-processing stack for re-processing |
| `--diagnostic` | Save FITS snapshot before each post-processing step |
| `--quality-report PATH` | Write per-frame quality metrics to CSV |
| `--dry-run` | Discover frames, show parameters, estimate resources — no processing |
| `--health-check` | Analyse calibration and frames without stacking |
| `--debug-registration` | Write registration diagnostics to `_registration_debug/` |
| `--use-gpu` | Enable CuPy GPU acceleration |
| `-j N, --parallel N` | Worker count (0 = auto-detect) |
| `-v, --verbose` | Detailed per-frame output |

For the full CLI reference with all flags and defaults, see [PROJECT_SPEC.md](PROJECT_SPEC.md).

---

## Performance

| Scenario | Memory | Time (8-core CPU) |
|----------|--------|-------------------|
| 50 × 4096×4096 frames | 0.4–1.2 GB | 30–120 s |
| Traditional stacker | 10–20 GB | varies |
| 500 frames (tested) | same 0.4–1.2 GB | ~10× longer |

Memory usage is bounded by the streaming architecture — frames are loaded one at a time and freed immediately after accumulation.

---

## Architecture Overview

```
astro_stack.py                  ← thin backward-compatibility entry point
└── src/
    ├── cli.py                  ← argument parsing, process_directory(), main()
    ├── pipeline.py             ← four-phase orchestrator (stack_target)
    ├── frame_processor.py      ← Phase 1: parallel per-frame load/calibrate/quality
    ├── registration.py         ← Phase 2: shift calculation, affine/RANSAC
    ├── stacking.py             ← Phase 3: alignment, cropping, combine
    ├── postprocess.py          ← Phase 4: up to 20-step post-processing chain
    ├── debayer.py              ← Bayer demosaicing, hot pixels, white balance
    ├── quality.py              ← star detection, FWHM, quality metrics
    ├── background.py           ← DBE, mesh sky extraction, floor normalisation
    ├── denoising.py            ← wavelet, NLM, bilateral, MMT, ACDNR, stretch
    ├── psf_deconvolution.py    ← PSF estimation, Richardson-Lucy
    ├── io_fits.py              ← FITS load/save, master frame creation
    ├── frame_discovery.py      ← automatic frame classification
    ├── auto_settings.py        ← heuristic target classifier (--auto)
    ├── plate_solve.py          ← astrometry.net + SIMBAD
    ├── gpu_context.py          ← CPU/GPU abstraction (numpy ↔ cupy)
    ├── models.py               ← Config, FrameInfo, ProcessingStats
    ├── health_check.py         ← calibration analysis
    └── utils.py                ← print helpers, formatting
```

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for a detailed architecture and feature reference.

---

## Testing

```bash
pip install pytest
pytest -q

# Run a specific test
pytest tests/test_core.py::test_calculate_shift_recovery -v

# CI smoke test (generates synthetic data, then stacks it)
python tools/create_synthetic.py
python astro_stack.py -d synthetic_data -o ci_synthetic_stack.fits \
  --debayer-method malvar --white-balance grayworld --stack-method median
```

---

## Diagnostics & Troubleshooting

**Frames not aligning?** Use `--debug-registration`:
```bash
python astro_stack.py -d lights/ -o out.fits --debug-registration
# Diagnostics written to _registration_debug/
```

**Want shift and quality data per frame?** Run with `-v`:
```bash
python astro_stack.py -d lights/ -o out.fits -v 2>&1 | tee run.log
```

**Not sure what's happening?** Run with `--dry-run` first:
```bash
python astro_stack.py -d lights/ -o out.fits --dry-run
```

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for guidance on interpreting shift patterns and quality metrics.

---

## Plate Solving

Requires `astroquery` and a free API key from [nova.astrometry.net](https://nova.astrometry.net/api_help).

```bash
pip install astroquery
export ASTROMETRY_API_KEY=your_key_here

python astro_stack.py -d lights/ -o stacked.fits --plate-solve --color-calibrate
```

When plate solving succeeds, WCS keywords (CRVAL, CRPIX, CD matrix) are written to the FITS header and the field's primary object is identified via the SIMBAD database. The output FITS will then display coordinate grids in DS9, AstroImageJ, PixInsight, and similar tools.

Alternatively, use the ASTAP solver:
```bash
python astro_stack.py -d lights/ -o stacked.fits --plate-solve --plate-solver astap
```
