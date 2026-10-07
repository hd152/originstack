"""OriginStack -- Astro FITS Stream Stacker

Features:
- Streaming processing (constant memory)
- Calibration (bias/dark/flat)
- Debayering (bilinear, Malvar -- native Rust, no OpenCV dependency)
- Quality analysis (brightness, contrast, star count, FWHM)
- Registration (sub-pixel phase correlation, FFT cross-correlation, affine/star-matching)
- Automatic cropping, hierarchical processing, preview generation
- Intelligent background extraction (mesh-based sigma-clipped sky removal with star masking)
- GPU acceleration via CuPy (--use-gpu) with automatic CPU fallback
- Parallel frame processing via multiprocessing (-j)
- Quality-weighted stacking, MAD-based sigma clipping, winsorized combine
- Wavelet denoising, local normalization, arcsinh preview stretch
- Richardson-Lucy deconvolution with automatic PSF estimation (Moffat/Gaussian)
- Lanczos-interpolated drizzle for sub-pixel super-resolution
- White balance, hot pixel removal, gradient removal

Usage: python originstack.py -d INPUT_DIR -o OUTPUT.fits [options]

NOTE: This file is a thin entry-point shim; all implementation lives
      under src/.
"""
from __future__ import annotations

# Re-exports used by tests (``from originstack import X`` / ``originstack.X``)
# and the entry point. Implementation lives under src/.
from src.background import (
    sky_floor_normalize,
)
from src.cli import (
    main,
)
from src.debayer import (
    apply_hot_pixel_map_bayer,
    build_hot_pixel_map,
    correct_chromatic_aberration,
    debayer,
    remove_hot_pixels,
    remove_hot_pixels_bayer,
    white_balance_grayworld,
    white_balance_whitepatch,
)
from src.denoising import (
    arcsinh_stretch,
    reduce_chroma_noise,
)
from src.frame_discovery import (
    classify_frame,
    discover_frames,
    select_matching_darks,
)
from src.gpu_context import (
    GpuContext,
)
from src.models import (
    Config,
    FrameInfo,
    ProcessingStats,
)
from src.psf_deconvolution import (
    estimate_psf,
    make_synthetic_psf,
    richardson_lucy_deconvolve,
)
from src.quality import (
    compute_quality_metrics,
    generate_star_mask,
    validate_image_data,
)
from src.registration import (
    apply_transform,
    calc_common_crop,
    calculate_shift,
    detect_dither,
)
from src.stacking import (
    _esd_lambda_table,
    _sigma_clip_tile,
    esd_combine,
    ivw_combine,
    lacosmic_reject,
    linear_fit_clip_combine,
    median_combine,
    online_sigma_clip_fold_frame,
    online_sigma_clip_seed_burnin,
    patch_weighted_mean_combine,
    percentile_clip_combine,
    sigma_clip_combine,
)
from src.utils import (
    format_time,
    read_version,
    safe_print,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
