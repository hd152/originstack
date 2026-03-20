# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
pip install pytest
# Optional: GPU support (requires CUDA)
pip install -r requirements-gpu.txt
```

### Run tests
```bash
pytest -q
# Run a single test
pytest tests/test_core.py::test_calculate_shift_recovery -v
```

### Run the stacker
```bash
python astro_stack.py -d lights/ -o stacked.fits
python astro_stack.py -d lights/ -o stacked.fits -v
python astro_stack.py -d session/ -o combined.fits --keep-intermediates -v
```

### CI smoke test (generates synthetic data then stacks it)
```bash
python tools/create_synthetic.py
python astro_stack.py -d synthetic_data -o ci_synthetic_stack.fits --debayer-method malvar --white-balance grayworld --stack-method median
```

### Debug registration issues
```bash
python astro_stack.py -d lights/ -o stacked.fits --debug-registration
# Output diagnostics go to _registration_debug/
```

## Architecture

The pipeline is split across `src/` modules. [astro_stack.py](astro_stack.py) is a thin backward-compatibility shim that re-exports all public symbols. Tests import directly from `astro_stack`. The source modules are:

| Module | Contents |
|--------|----------|
| [src/gpu_context.py](src/gpu_context.py) | `GpuContext`, `get_gpu()` singleton |
| [src/models.py](src/models.py) | `Config`, `FrameInfo`, `ProcessingStats` |
| [src/utils.py](src/utils.py) | Print helpers, `format_time`, `get_memory_usage_mb` |
| [src/io_fits.py](src/io_fits.py) | FITS load/save, `make_master`, `populate_fits_header` |
| [src/frame_discovery.py](src/frame_discovery.py) | `discover_frames`, `classify_frame`, `select_matching_darks` |
| [src/debayer.py](src/debayer.py) | Debayering, hot pixels, white balance |
| [src/quality.py](src/quality.py) | `compute_quality_metrics`, star detection, FWHM |
| [src/psf_deconvolution.py](src/psf_deconvolution.py) | PSF estimation, Richardson-Lucy deconvolution |
| [src/background.py](src/background.py) | Mesh-based sky extraction, residual removal |
| [src/denoising.py](src/denoising.py) | Wavelet, bilateral, NLM, arcsinh stretch |
| [src/registration.py](src/registration.py) | `calculate_shift`, affine transform, `calc_common_crop` |
| [src/stacking.py](src/stacking.py) | Sigma-clip combine, drizzle, Lanczos |
| [src/plate_solve.py](src/plate_solve.py) | Astrometry.net plate solving |
| [src/pipeline.py](src/pipeline.py) | Three-phase pipeline, parallel workers, `stack_target` |
| [src/health_check.py](src/health_check.py) | `run_health_check` |
| [src/cli.py](src/cli.py) | `process_directory`, `parse_args`, `main` |

### Three-phase processing pipeline
1. **Validation & Quality Analysis** — Load each frame, compute brightness/contrast/star-count/FWHM/SNR, reject below-threshold frames. Quality filter is on by default; controlled by `--quality-threshold` (default: 25th percentile, i.e. keep best 75%).
2. **Registration** — Calculate per-frame alignment shifts using phase cross-correlation (skimage) with FFT cross-correlation fallback, then optional affine (rotation+scale) via star matching + RANSAC. Reference frame is the highest-quality accepted frame.
3. **Stacking** — Shift-align each frame, crop to the valid region common to all frames (eliminates black borders), accumulate into running mean/median/sigma-clip combine.

### Streaming memory model
Frames are processed one at a time: load → process → accumulate → free. Memory usage stays at ~1-2 frames regardless of total frame count. This is the core design constraint — never accumulate all frames in memory.

### Key classes and singletons
- **`GpuContext`** — CPU/GPU abstraction; exposes `xp` (numpy or cupy), `xndimage`, `xsignal`. Accessed via module-level `get_gpu()` in `src/gpu_context.py`. GPU path is opt-in via `--use-gpu`.
- **`Config`** — Central class for magic number constants (thresholds, defaults) in `src/models.py`.
- **`FrameInfo`** — Dataclass holding path, type, header, quality metrics, and computed shift for each frame.
- Master calibration frames (bias/dark/flat) are plain numpy arrays stored in a `masters` dict passed through the pipeline.

### Hierarchical vs. single-folder mode
Auto-detected at runtime:
- FITS files in root → single-folder mode (one output FITS).
- Subfolders containing FITS → hierarchical mode (each subfolder stacked independently, then combined; shape mismatches handled by resizing to minimum common dimensions).

### Optional dependencies (graceful degradation)
Each `src/` module wraps its optional imports in `try/except`. Features degrade gracefully when packages are absent:
- `photutils` — star detection in quality analysis
- `tqdm` — progress bars (falls back to plain iterator)
- `Pillow` — preview JPEG generation
- `psutil` — memory usage reporting
- `cupy` — GPU acceleration (`--use-gpu`)
- `cv2` (OpenCV) — advanced debayer methods (Malvar, VNG), bilateral filter denoising
- `pywt` — wavelet denoising (`--denoise`)
- `astroquery` — plate solving via nova.astrometry.net (`--plate-solve`)

### Plate solving
Requires `astroquery` installed and `ASTROMETRY_API_KEY` environment variable set. Enable with `--plate-solve`.

### Debayering options
- `bilinear` (default, pure numpy)
- `malvar` (higher quality, requires cv2)
- `vng` (requires cv2)

### Parallelism
- `-j N` controls worker count for frame processing via `ProcessPoolExecutor`/`ThreadPoolExecutor`.
- GPU mode auto-limits workers based on available VRAM.
