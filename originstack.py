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

# models
from src.models import (
    Config,
    FrameInfo,
    ProcessingStats,
)

# utils
from src.utils import (
    safe_print,
    print_header,
    print_quality_table,
    print_phase,
    format_time,
    get_memory_usage_mb,
    setup_logging,
    get_logger,
    read_version,
)

# io_fits
from src.io_fits import (
    _read_fits_header,
    load_fits,
    make_master,
    save_preview_rgb,
    populate_fits_header,
)

# frame_discovery
from src.frame_discovery import (
    discover_frames,
    classify_frame,
    select_matching_darks,
)

# debayer
from src.debayer import (
    debayer_bilinear,
    debayer_malvar,
    debayer,
    white_balance_grayworld,
    white_balance_whitepatch,
    remove_hot_pixels,
    remove_hot_pixels_bayer,
    build_hot_pixel_map,
    apply_hot_pixel_map_bayer,
    remove_hot_pixels_rgb,
    correct_chromatic_aberration,
)

# quality
from src.quality import (
    generate_star_mask,
    measure_fwhm,
    validate_image_data,
    compute_quality_metrics,
)

# psf_deconvolution
from src.psf_deconvolution import (
    estimate_psf,
    make_synthetic_psf,
    richardson_lucy_deconvolve,
)

# background
from src.background import (
    extract_background,
    apply_background_extraction,
    remove_sky_residual,
    sky_floor_normalize,
)

# denoising
from src.denoising import (
    wavelet_denoise,
    adaptive_wavelet_denoise,
    bilateral_denoise,
    nlm_denoise,
    reduce_chroma_noise,
    arcsinh_stretch,
)

# registration
from src.registration import (
    match_stars_affine,
    apply_transform,
    calculate_shift,
    calculate_shift_pyramid,
    apply_shift,
    calc_common_crop,
    detect_dither,
)

# stacking
from src.stacking import (
    drizzle_combine,
    _lanczos_resample_frame,
    _sigma_clip_tile,
    sigma_clip_combine,
    online_sigma_clip_seed_burnin,
    online_sigma_clip_fold_frame,
    median_combine,
    percentile_clip_combine,
    trimmed_mean_combine,
    esd_combine,
    _esd_lambda_table,
    linear_fit_clip_combine,
    ivw_combine,
    patch_weighted_mean_combine,
    lacosmic_reject,
)

# plate_solve
from src.plate_solve import solve_plate

# pipeline
from src.pipeline import stack_target
from src.frame_processor import (
    _process_single_frame,
    _init_worker_shm,
    _parallel_frame_worker,
    _worker_masters,
)

# health_check
from src.health_check import run_health_check

# cli
from src.cli import (
    process_directory,
    parse_args,
    main,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
