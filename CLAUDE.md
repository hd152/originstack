# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
pip install pytest
# Optional: GPU support (requires CUDA)
pip install -r requirements-gpu.txt
# Optional: native (Rust) acceleration — see "Native (Rust) acceleration" below
pip install maturin && (cd ext/astro_native && maturin develop --release)
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
4. Chroma noise reduction — fine pass; optional coarse pass for medium-scale colour blotches (`--chroma-nr-large-sigma`, object-masked)
5. Sky floor correction (per-channel pedestal removal)
6. Wavelet denoising (luma/chroma split, star-protected, adaptive BayesShrink by default)
7. Sky residual correction (broad + fine passes after background extraction)
8. Sky pedestal — scalar lift off zero before the non-negativity clips (prevents black-hole clipping); skippable via `--skip-step sky_pedestal`
9. NLM denoising — `--denoise-nlm`
10. Bilateral filter denoising — `--denoise-bilateral`
11. MMT denoising (multiscale median transform) — `--denoise-mmt`
12. ACDNR denoising (adaptive contrast-based) — `--denoise-acdnr`
13. Anisotropic diffusion — `--denoise-aniso` (native/Rust accelerated)
14. Richardson-Lucy deconvolution — `--deconvolve` (GPU-accelerated via cupy FFT when `--use-gpu`)
15. Star reduction (halo softening) — on by default, `--no-star-reduce`
16. Multiscale local contrast enhancement — on by default, `--no-local-contrast`
17. Final sky flattening + neutralisation — masked large-scale per-channel background removal → neutral grey; skippable via `--skip-step sky_neutralize`

The old local-normalisation step (`--local-normalize`) was **removed**: it did local variance equalisation (÷ local σ), which amplifies background noise; gradient/vignette residual is handled by `--pre-gradient-removal` + DBE + sky-floor + sky-residual.

The preview JPEG black point is set per target by the auto-advisor (`preview_black_sigma`, overridable with `--preview-black-sigma`); higher values (2–3) clip the sky-noise tail to black for a small target on empty sky.

### Incremental stacking (`--merge`)
The main output FITS is the linear pre-post-processing stack (`RAWSTACK=True`) with `NFRAMES`/`INTGTIME`/`TOTEXP` headers. `--merge PREV.fits [...]` processes only the new session through Phases 1-3, registers each previous stack onto the new grid (star-match affine in [src/merge.py](src/merge.py) — nights differ by arbitrary field rotation on alt-az mounts — translation fallback, hard error on failure or <25% overlap), and combines as a per-pixel `NFRAMES`-weighted mean inside each warped footprint before Phase 4 runs once on the result. Header aggregates are summed, so the output chains into future merges. There is no cross-session outlier rejection (each session already rejected internally); not supported with `--drizzle-scale > 1`.

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
- `astro_native` — optional Rust hot-path kernels (see below); numpy fallback when absent

### Native (Rust) acceleration
[ext/astro_native/](ext/astro_native/) is a PyO3/maturin crate of hot-path kernels, all with a numpy fallback. Coverage:
- **Stacking combines** (`src/stacking.py`): `sigma_clip_combine` (~37×), `esd_combine` (~24×), `percentile_clip_combine` (~13×), `median_combine` (~6×), `trimmed_mean_combine` (~4×). ESD's Student-t critical-value table is precomputed in Python (`_esd_lambda_table`) and passed to Rust — exact parity, no stats crate. Native path is taken when a rejection mask is not requested and the input is a C-contiguous float32 `(N,H,W,C)` array; the aligned stack memmap qualifies, so Rust views it zero-copy and the streaming memmap model is preserved.
- **Drizzle resample** (`src/stacking.py`): both drizzle paths route through `warp_affine_lanczos3` (~26×, all channels in one pass; true Lanczos-3 vs the scipy order-5 quintic approximation it replaced). The kernel's separable fast path covers any diagonal matrix — drizzle's no-rotation `diag(1/scale)`+offset mapping as well as pure translations.
- **Fused patch-weighted combine** (`src/stacking.py` `run_stacking_phase`): `patch_weighted_sigma_combine` (~100×) — the consensus/patch-weighted path did two numpy passes (sigma-clip `return_mask=True` + `patch_weighted_mean_combine`); the Rust kernel fuses sigma-clip rejection + quality weighting in one pass with no `(N,H,W,C)` mask array. Used for the sigma_clip/winsorized patch methods.
- **Per-frame warp** (`src/registration.py` `apply_transform`): `warp_affine_lanczos3`, a 2D Lanczos-3 affine/shift resample (~5×). CPU path only (GPU unchanged). This is a quality-validated change, not numeric parity: FWHM and flux match scipy order-3, and the mild Lanczos ringing is incoherent across dithered frames (averages to zero in the stack).
- **Anisotropic diffusion** (`src/denoising.py` `anisotropic_diffusion`): native Jacobi iteration with periodic boundary (~37×), exact parity.
- **DBE surface fit** (`src/background.py` `_fit_background_surface`): `dbe_fit_surface`, Gaussian-weighted local-linear regression with Tukey-biweight IRLS (~2.4× over the scipy RBF it replaced). Not just a speedup — the RBF (thin-plate spline + hard outlier-rejection loop) was unbounded and could extrapolate wildly into rejected-sample gaps near bright stars; the local fit is bounded near the sample values by construction. Numpy mirror in `_dbe_fit_surface_numpy`, parity-tested.
- **Cosmic-ray rejection** (`src/stacking.py` `lacosmic_reject`): `lacosmic_reject_native`, L.A.Cosmic-style Laplacian spike detection + 5×5 median replacement, f32 internally (~2×+ vs numpy; the memory-traffic halving matters more than raw compute under 16-way ProcessPool contention).
- **Median filter** (`median_filter_native`): windowed median (reflect boundary) used by lacosmic and the hot-pixel detectors. Interior fast path with contiguous reads; 3×3 uses Paeth's 19-op branchless median network (~13×), 5×5 quickselect (~1.6×).

The startup banner reports native status (`native_status()` in `src/utils.py`); each accelerated step logs a `[rust] …` line. Separately, Richardson-Lucy deconvolution runs on the GPU (cupy FFT) when `--use-gpu` is active (`_rl_deconvolve_xp`), validated against skimage on the numpy backend.

Kernel internals (why they're fast): a blocked **gather-transpose** turns the per-pixel N-sample gather (N huge-stride read streams that defeat the prefetcher/TLB) into sequential per-frame row-segment copies through an L2-resident pixel-major tile; medians use **quickselect** (`select_nth_unstable`, O(n)) instead of a full sort; the warp has a **separable fast path** for pure translation (per-image wx table + per-row wy — zero `sin()` calls per pixel) plus an interior path with no per-tap bounds checks. Measured on top of the first-generation kernels: sigma-clip ~2.6×, winsorized ~3.0×, median ~4.0×, ESD ~1.7×, fused patch combine ~2.8×, shift warp ~6.5×, affine warp ~2.8×, aniso ~4.0× ([tools/bench_native.py](tools/bench_native.py) reproduces the numbers).

Build (needs a Rust toolchain + `pip install maturin`):
```bash
# into a virtualenv:
cd ext/astro_native && maturin develop --release
# system Python (no venv): build a wheel and install it
cd ext/astro_native && python -m maturin build --release
pip install --force-reinstall target/wheels/astro_native-*.whl
```
Optionally set `RUSTFLAGS="-C target-cpu=native"` before building for extra auto-vectorisation — safe when the wheel stays on the machine that built it (do not redistribute such a wheel; it may use instructions older CPUs lack).

Parity vs the numpy reference is covered by [tests/test_native.py](tests/test_native.py) (auto-skips if the module is absent). Benchmark with [tools/bench_native.py](tools/bench_native.py) before/after kernel changes.

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
