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
python astro_stack.py -d lights/ -o stacked.fits -v --quality-filter
python astro_stack.py -d session/ -o combined.fits --quality-filter --keep-intermediates -v
```

### CI smoke test (generates synthetic data then stacks it)
```bash
python tools/create_synthetic.py
python astro_stack.py -d synthetic_data -o ci_synthetic_stack.fits --quality-filter --debayer-method malvar --white-balance grayworld --stack-method median
```

### Debug registration issues
```bash
python astro_stack.py -d lights/ -o stacked.fits --debug-registration
# Output diagnostics go to _registration_debug/
```

## Architecture

The entire stacking pipeline lives in a single file: [astro_stack.py](astro_stack.py) (~3,900 lines). Tests are in [tests/test_core.py](tests/test_core.py) which imports functions directly from `astro_stack`.

### Three-phase processing pipeline
1. **Validation & Quality Analysis** — Load each frame, compute brightness/contrast/star-count/SNR, reject below-threshold frames. Controlled by `--quality-filter` and `--quality-threshold`.
2. **Registration** — Calculate per-frame alignment shifts using phase cross-correlation (skimage) with FFT cross-correlation fallback. Reference frame is the highest-quality accepted frame.
3. **Stacking** — Shift-align each frame, crop to the valid region common to all frames (eliminates black borders), accumulate into running mean/median/sigma-clip combine.

### Streaming memory model
Frames are processed one at a time: load → process → accumulate → free. Memory usage stays at ~1-2 frames regardless of total frame count. This is the core design constraint — never accumulate all frames in memory.

### Key classes and singletons
- **`GpuContext`** — CPU/GPU abstraction; exposes `xp` (numpy or cupy), `xndimage`, `xsignal`. Accessed via module-level `get_gpu()`. GPU path is opt-in via `--use-gpu`.
- **`Config`** — Central class for magic number constants (thresholds, defaults).
- **`CalibrationFrames`** — Holds master bias/dark/flat arrays built by median-combining calibration files before the main loop.
- **`QualityMetrics`** — Dataclass returned by `compute_quality_metrics()`.

### Hierarchical vs. single-folder mode
Auto-detected at runtime:
- FITS files in root → single-folder mode (one output FITS).
- Subfolders containing FITS → hierarchical mode (each subfolder stacked independently, then combined; shape mismatches handled by resizing to minimum common dimensions).

### Optional dependencies (graceful degradation)
All optional imports are wrapped in `try/except` at the top of `astro_stack.py`. Features degrade gracefully when packages are absent:
- `photutils` — star detection in quality analysis
- `tqdm` — progress bars (falls back to plain iterator)
- `Pillow` — preview JPEG generation
- `cupy` — GPU acceleration (`--use-gpu`)
- `cv2` — advanced debayer methods (Malvar, VNG)
- `pywt` — wavelet denoising
- `astroquery` — plate solving via nova.astrometry.net

### Plate solving
Requires `astroquery` installed and `ASTROMETRY_API_KEY` environment variable set. Attempted by default when the library is present; skipped with `--skip-plate-solve`.

### Debayering options
- `bilinear` (default, pure numpy)
- `malvar` (higher quality, requires cv2)
- `vng` (requires cv2)

### Parallelism
- `-j N` controls worker count for frame processing via `ProcessPoolExecutor`/`ThreadPoolExecutor`.
- GPU mode auto-limits workers based on available VRAM.
