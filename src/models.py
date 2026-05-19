"""Data models and configuration constants."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


class Config:
    """Central configuration for magic numbers and thresholds."""
    HOT_PIXEL_THRESHOLD = 12.0
    HOT_PIXEL_BAYER_THRESHOLD = 5.0  # Lower for Bayer detection (MAD-based, robust)
    WHITE_PATCH_PERCENTILE = 99.5
    MAX_SHIFT_FRACTION = 0.1
    STAR_DETECTION_SIGMA = 5.0
    CROP_MARGIN = 2
    XCORR_DOWNSCALE_TARGET = 256  # Target size for cross-correlation
    CENTROID_PERCENTILES = [95, 90, 85, 80]
    QUALITY_LOW_BRIGHTNESS = 10
    QUALITY_LOW_CONTRAST = 1
    LARGE_SHIFT_WARNING_PX = 20
    MIN_RECOMMENDED_FRAMES = 10
    PREVIEW_JPEG_QUALITY = 95
    PREVIEW_STRETCH_PERCENTILES = (1, 99)
    TILE_SIZE = 256  # Tile size for tiled sigma-clip (pixels)
    FWHM_CUTOUT_RADIUS = 10  # Cutout radius for FWHM measurement
    FWHM_MAX_STARS = 50  # Max stars to measure for FWHM
    ARCSINH_STRETCH_FACTOR = 5.0  # Default arcsinh stretch factor
    STAR_MASK_MAX_STARS = 500  # Max stars for mask generation
    AFFINE_MAX_STARS = 80  # Max stars for affine matching
    AFFINE_MATCH_RADIUS = 10.0  # Max pixel distance for star matching
    GPU_PHASE1_WORKER_MB = 250.0   # VRAM per thread for debayer+hotpix+wb
    GPU_FFT_WORKER_MB = 800.0      # VRAM per thread for padded complex128 FFT
    GPU_ALIGN_WORKER_MB = 250.0    # VRAM per thread for ndimage.shift on 3-ch image
    GPU_VRAM_RESERVE_MB = 512.0    # Reserved for CuPy overhead / driver
    RL_PSF_CUTOUT_RADIUS = 15      # Radius for star cutouts used in PSF estimation
    RL_PSF_MAX_STARS = 30          # Max stars to sample for PSF building
    RL_PSF_MIN_STARS = 5           # Min successful fits for reliable PSF
    RL_PSF_SIZE = 31               # Output PSF kernel size (odd)
    RL_DEFAULT_ITERATIONS = 15     # Default Richardson-Lucy iterations
    BORDER_FRAC = 0.12             # Fraction of image border used for sky reference

    # Dynamic Background Extraction (DBE)
    DBE_PATCH_SIZE = 64            # Candidate background patch size in pixels
    DBE_MASKED_FRAC_THRESH = 0.30 # Max allowed emission-masked fraction per patch
    DBE_OUTLIER_SIGMA = 2.5       # Sigma for iterative RBF residual rejection
    DBE_OUTLIER_ITERS = 3         # Max outlier rejection passes
    DBE_MIN_SAMPLES = 20          # Min accepted patches before falling back to mesh
    DBE_MAX_SAMPLES = 2000        # Cap to keep RBF O(N²) tractable on large images
    DBE_DENSE_FIELD_THRESH = 0.70 # Emission-mask coverage above which dense-field fallback is used
    DBE_RBF_KERNEL = 'thin_plate_spline'
    DBE_RBF_SMOOTHING = 0.0

    # BM3D denoising
    BM3D_SIGMA_PSD = 0.0            # 0 = auto-estimate from sky noise
    BM3D_BLOCK_SIZE = 8
    BM3D_GROUP_SIZE = 8
    BM3D_SEARCH_WINDOW = 16

    # Anisotropic diffusion (Perona-Malik)
    ANISO_ITERATIONS = 20
    ANISO_KAPPA = 30.0              # Gradient threshold (ADU)
    ANISO_GAMMA = 0.1               # Time step (0 < γ ≤ 0.25 for stability)

    # SCNR
    SCNR_AMOUNT = 1.0               # 1.0 = full neutralisation

    # Adaptive MTF
    MTF_TARGET_BG = 0.15            # Target output level for sky background

    # Blind PSF estimation
    BLIND_PSF_ITERATIONS = 8        # RL iterations for PSF update

    # Total Variation deconvolution
    TV_LAMBDA = 0.02                # TV regularisation weight
    TV_ITERATIONS = 50              # Gradient descent steps

    # Polynomial distortion correction
    POLY_DISTORTION_DEGREE = 2      # Polynomial degree (2=quadratic, 3=cubic)
    POLY_MIN_STARS = 12             # Min matched pairs for reliable poly fit

    # HDR stack blending
    HDR_TRANSITION_WIDTH = 0.1      # Fractional signal range for sigmoid blend

    # Strehl / atmospheric dispersion
    STREHL_CUTOUT_RADIUS = 20       # Cutout half-size for Strehl measurement
    DISP_CUTOUT_RADIUS = 10         # Cutout half-size for dispersion centroid

    # Brenner / wavelet entropy quality metrics
    WAVELET_ENTROPY_LEVELS = 4      # Wavelet decomposition levels for entropy ratio

    # Zernike PSF decomposition
    ZERNIKE_CUTOUT_RADIUS = 15      # Half-size for Zernike PSF cutout extraction
    ZERNIKE_MAX_ORDER = 4           # Max radial order (covers 15 modes: piston through spherical)
    ZERNIKE_MAX_STARS = 15          # Stars to sample for Zernike decomposition

    # Registration enhancements
    SHIFT_OUTLIER_SIGMA = 3.5       # MAD-sigma threshold for pre-registration outlier rejection
    REG_RESIDUAL_MAX_PX = 1.5       # Max post-registration centroid RMS (px) before rejection
    ALIGNMENT_CENTRALITY_WEIGHT = 0.3   # Blend weight: 0=pure quality score, 1=pure centrality
    REF_PYRAMID_WORKERS = 4         # Max parallel workers for cheap reference-selection pass

    # Patch-based local registration (lucky imaging mode)
    PATCH_GRID_SIZE = 8             # NxN grid for patch quality map (8x8 = 64 patches)
    PATCH_MIN_SIZE = 64             # Minimum patch dimension in pixels


@dataclass
class FrameInfo:
    path: str
    type: str  # 'light','dark','flat','bias'
    header: dict
    accepted: bool = True
    metrics: Optional[Dict] = None
    shift: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ProcessingStats:
    """Track timing and statistics during processing."""
    start_time: float = field(default_factory=time.time)
    discovery_time: float = 0.0
    calibration_time: float = 0.0
    quality_time: float = 0.0
    registration_time: float = 0.0
    stacking_time: float = 0.0
    post_processing_time: float = 0.0
    total_frames: int = 0
    accepted_frames: int = 0
    rejected_frames: int = 0
    errors: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_shape: Optional[Tuple[int, int]] = None
    cropped_pixels: Optional[Tuple[int, int]] = None
    peak_memory_mb: float = 0.0

    def total_time(self) -> float:
        return time.time() - self.start_time

    def add_error(self, path: str, error: str):
        self.errors.append((path, error))

    def add_warning(self, warning: str):
        self.warnings.append(warning)
