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

# background
from src.background import (
    apply_background_extraction,
    extract_background,
    remove_sky_residual,
    sky_floor_normalize,
)

# cli
from src.cli import (
    main,
    parse_args,
    process_directory,
)

# debayer
from src.debayer import (
    apply_hot_pixel_map_bayer,
    build_hot_pixel_map,
    correct_chromatic_aberration,
    debayer,
    debayer_bilinear,
    debayer_malvar,
    remove_hot_pixels,
    remove_hot_pixels_bayer,
    remove_hot_pixels_rgb,
    white_balance_grayworld,
    white_balance_whitepatch,
)

# denoising
from src.denoising import (
    adaptive_wavelet_denoise,
    arcsinh_stretch,
    bilateral_denoise,
    nlm_denoise,
    reduce_chroma_noise,
    wavelet_denoise,
)

# frame_discovery
from src.frame_discovery import (
    classify_frame,
    discover_frames,
    select_matching_darks,
)
from src.frame_processor import (
    _init_worker_shm,
    _parallel_frame_worker,
    _process_single_frame,
    _worker_masters,
)

# ---------------------------------------------------------------------------
# Re-export everything from the src sub-modules so that:
#   from originstack import X
# works without reaching into src/ directly.
# ---------------------------------------------------------------------------
# gpu_context
from src.gpu_context import (
    GpuContext,
    _CudaStreamContext,
    _NullContext,
    get_gpu,
)

# health_check
from src.health_check import run_health_check

# io_fits
from src.io_fits import (
    _read_fits_header,
    load_fits,
    make_master,
    populate_fits_header,
    save_preview_rgb,
)

# models
from src.models import (
    Config,
    FrameInfo,
    ProcessingStats,
)

# pipeline
from src.pipeline import stack_target

# plate_solve
from src.plate_solve import solve_plate

# psf_deconvolution
from src.psf_deconvolution import (
    estimate_psf,
    make_synthetic_psf,
    richardson_lucy_deconvolve,
)

# quality
from src.quality import (
    compute_quality_metrics,
    generate_star_mask,
    measure_fwhm,
    validate_image_data,
)

# registration
from src.registration import (
    apply_shift,
    apply_transform,
    calc_common_crop,
    calculate_shift,
    calculate_shift_pyramid,
    detect_dither,
    match_stars_affine,
)

# stacking
from src.stacking import (
    _esd_lambda_table,
    _lanczos_resample_frame,
    _sigma_clip_tile,
    drizzle_combine,
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

# utils
from src.utils import (
    format_time,
    get_logger,
    get_memory_usage_mb,
    print_header,
    print_phase,
    print_quality_table,
    read_version,
    safe_print,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
