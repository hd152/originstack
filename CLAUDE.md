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
| [src/background.py](src/background.py) | Mesh-based sky extraction, DBE, residual removal |
| [src/denoising.py](src/denoising.py) | Wavelet, MMT, ACDNR, bilateral, NLM, star reduction, local contrast |
| [src/registration.py](src/registration.py) | `calculate_shift`, affine/RANSAC, `calc_common_crop`, `run_registration_phase` |
| [src/stacking.py](src/stacking.py) | Sigma-clip, percentile, ESD, drizzle, Lanczos, `run_stacking_phase` |
| [src/frame_processor.py](src/frame_processor.py) | Parallel workers, `execute_frame_processing`, `quality_gate` |
| [src/postprocess.py](src/postprocess.py) | Full post-processing chain: `postprocess_stack` |
| [src/auto_settings.py](src/auto_settings.py) | Heuristic target classifier and parameter advisor (`--auto`) |
| [src/ai_advisor.py](src/ai_advisor.py) | Claude API parameter advisor and session report (`--ai-advisor`) |
| [src/plate_solve.py](src/plate_solve.py) | Astrometry.net plate solving |
| [src/pipeline.py](src/pipeline.py) | Thin orchestrator: `stack_target` wires all four phases |
| [src/health_check.py](src/health_check.py) | `run_health_check` |
| [src/cli.py](src/cli.py) | `process_directory`, `parse_args`, `main` |

### Four-phase processing pipeline
1. **Phase 1 — Process & Quality Analysis** — Load, calibrate, debayer, and quality-analyse each frame in parallel (ProcessPool, ThreadPool, or sequential). Hard-limit rejection, statistical outlier detection, and percentile quality threshold filter out bad frames. Handled by `frame_processor.py`.
2. **Phase 2 — Registration** — Calculate per-frame alignment shifts via phase cross-correlation (skimage) with FFT fallback, then optional affine (rotation+scale) via star matching + RANSAC. Reference frame is the highest-quality accepted frame. Handled by `run_registration_phase` in `registration.py`.
3. **Phase 3 — Stacking** — Align, crop to the valid common region (eliminates black borders), and combine via the selected method (mean, median, sigma-clip, percentile, ESD, or drizzle). Handled by `run_stacking_phase` in `stacking.py`.
4. **Phase 4 — Post-processing** — Full chain applied to the stacked image (see below). Handled by `postprocess_stack` in `postprocess.py`.

### Post-processing chain (Phase 4)
Applied in order; most steps are on by default:
1. Per-channel hot pixel removal on stacked image
2. Star detection (single pass; mask reused by all steps below)
3. Dynamic Background Extraction (DBE) or legacy mesh extraction
4. Chroma noise reduction
5. Sky floor correction (per-channel pedestal removal)
6. Local normalisation (vignette residual removal) — `--local-normalize`
7. Wavelet denoising (luma/chroma split, star-protected, adaptive BayesShrink by default)
8. Sky residual correction (broad + fine passes after background extraction)
9. NLM denoising — `--denoise-nlm`
10. Bilateral filter denoising — `--denoise-bilateral`
11. MMT denoising (multiscale median transform) — `--denoise-mmt`
12. ACDNR denoising (adaptive contrast-based) — `--denoise-acdnr`
13. Richardson-Lucy deconvolution — `--deconvolve`
14. Star reduction (halo softening) — on by default, `--no-star-reduce`
15. Multiscale local contrast enhancement — on by default, `--no-local-contrast`

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
- `pywt` — wavelet denoising
- `astroquery` — plate solving via nova.astrometry.net (`--plate-solve`)
- `anthropic` — AI parameter advisor (`--ai-advisor`, `--ai-report`)

### Plate solving
Requires `astroquery` installed and `ASTROMETRY_API_KEY` environment variable set. Enable with `--plate-solve`.

### Debayering options
- `bilinear` (default, pure numpy)
- `malvar` (higher quality, requires cv2)
- `vng` (requires cv2)

### Parallelism
- `-j N` (or `--parallel N`) controls worker count. `0` = auto (default), `1` = sequential, `N` = N processes.
- Uses `ProcessPoolExecutor` for CPU multi-process, `ThreadPoolExecutor` for GPU or thread-parallel paths.
- GPU mode auto-limits workers based on available VRAM.
