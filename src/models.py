"""Data models and configuration constants."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


class Config:
    """Central configuration for magic numbers and thresholds."""
    HOT_PIXEL_THRESHOLD = 12.0
    HOT_PIXEL_BAYER_THRESHOLD = 5.0  # Lower for Bayer detection (MAD-based, robust)
    CA_MIN_SHIFT_PX = 0.25           # Session CA below this: skip the correction warp entirely
    MERGE_MIN_OVERLAP = 0.25         # --merge: min warped-footprint overlap before refusing
    MERGE_MIN_CORRELATION = 0.15     # --merge: min aligned-vs-new luminance correlation
    WHITE_PATCH_PERCENTILE = 99.5
    CROP_MARGIN = 2
    XCORR_DOWNSCALE_TARGET = 256  # Target size for cross-correlation
    CENTROID_PERCENTILES = [95, 90, 85, 80]
    QUALITY_LOW_BRIGHTNESS = 10
    QUALITY_LOW_CONTRAST = 1
    LARGE_SHIFT_WARNING_PX = 20
    MIN_RECOMMENDED_FRAMES = 10
    PREVIEW_JPEG_QUALITY = 95
    PREVIEW_STRETCH_PERCENTILES = (1, 99)
    PREVIEW_MAX_DIMENSION = 8192
    TILE_SIZE = 256  # Tile size for tiled sigma-clip (pixels)
    FWHM_CUTOUT_RADIUS = 10  # Cutout radius for FWHM measurement
    FWHM_MAX_STARS = 50  # Max stars to measure for FWHM
    ARCSINH_STRETCH_FACTOR = 5.0  # Default arcsinh stretch factor
    STAR_MASK_MAX_STARS = 500  # Max stars for mask generation
    AFFINE_MAX_STARS = 80  # Max stars for affine matching
    AFFINE_MATCH_RADIUS = 10.0  # Max pixel distance for star matching
    AFFINE_MAX_ROTATION_DEG = 20.0  # Reject an affine fit rotating more than this (bad RANSAC match)
    MAX_REALISTIC_SHIFT_FRAC = 0.3  # Reject a shift/affine translation exceeding this fraction of frame W/H (bad match, not real drift)
    GPU_PHASE1_WORKER_MB = 450.0   # VRAM per thread: raw+cal+green_eq+debayer+hotpix+wb peak
    GPU_FFT_WORKER_MB = 800.0      # VRAM per thread for padded complex128 FFT
    GPU_ALIGN_WORKER_MB = 250.0    # VRAM per thread for ndimage.shift on 3-ch image
    GPU_VRAM_RESERVE_MB = 768.0    # Reserved for CuPy kernel cache / driver / masters
    GPU_POOL_FREE_INTERVAL = 32    # Free CuPy memory pool every N completed GPU frames
    RL_PSF_CUTOUT_RADIUS = 15      # Radius for star cutouts used in PSF estimation
    RL_PSF_MAX_STARS = 30          # Max stars to sample for PSF building
    RL_PSF_MIN_STARS = 5           # Min successful fits for reliable PSF
    RL_PSF_SIZE = 31               # Output PSF kernel size (odd)
    RL_DEFAULT_ITERATIONS = 15     # Default Richardson-Lucy iterations
    BORDER_FRAC = 0.12             # Fraction of image border used for sky reference

    # Dynamic Background Extraction (DBE)
    DBE_PATCH_SIZE = 64            # Candidate background patch size in pixels
    DBE_MASKED_FRAC_THRESH = 0.30 # Max allowed emission-masked fraction per patch
    DBE_OUTLIER_SIGMA = 2.5       # Robust-fit outlier tolerance (scales the Tukey biweight cutoff)
    DBE_OUTLIER_ITERS = 3         # IRLS reweighting passes in the surface fit
    DBE_MIN_SAMPLES = 20          # Min accepted patches before falling back to mesh
    DBE_MAX_SAMPLES = 4000        # Sample cap (local regression is O(N) per eval point)
    DBE_DENSE_FIELD_THRESH = 0.70 # Emission-mask coverage above which dense-field fallback is used
    DBE_FIT_SIGMA_PATCHES = 1.25  # Local-regression Gaussian bandwidth, in units of patch_size
                                  # (swept 1.0-2.0 on real data: 1.0-1.25 matches the old RBF's
                                  # large-scale flatness; larger trades flatness for smoothness)

    # BM3D denoising
    BM3D_SIGMA_PSD = 0.0            # 0 = auto-estimate from sky noise
    BM3D_SEARCH_WINDOW = 16

    # Blind PSF estimation
    BLIND_PSF_ITERATIONS = 8        # RL iterations for PSF update

    # Total Variation deconvolution
    TV_LAMBDA = 0.02                # TV regularisation weight
    TV_ITERATIONS = 50              # Gradient descent steps

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
    REG_RESIDUAL_MAX_PX = 1.5       # Floor for the post-registration centroid RMS reject threshold (px)
    REG_RESIDUAL_SIGMA_MULT = 3.0   # Threshold = max(floor, this * expected centroid noise sigma)
    REG_RESIDUAL_MAX_PX_CAP = 6.0   # Ceiling: never waive the check past this, even on very noisy subs
    ALIGNMENT_CENTRALITY_WEIGHT = 0.3   # Blend weight: 0=pure quality score, 1=pure centrality

    # Patch-based local registration (lucky imaging mode)
    PATCH_GRID_SIZE = 8             # NxN grid for patch quality map (8x8 = 64 patches)
    PATCH_MIN_SIZE = 64             # Minimum patch dimension in pixels

    # Elastic (non-rigid, per-patch) local registration -- corrects spatially-varying
    # distortion (differential atmospheric refraction, field rotation, tube flexure) a
    # single global affine per frame can't fix. Provisional/unvalidated against real
    # data (unlike DBE_FIT_SIGMA_PATCHES's real-data sweep) -- revisit after testing.
    LOCAL_WARP_MIN_STARS = 12           # Min matched stars to attempt a per-frame fit;
                                         # below this, fall back to affine-only unchanged
    LOCAL_WARP_MAX_DISPLACEMENT_PX = 8.0  # Clamp on fitted displacement magnitude (px);
                                           # also drives calc_common_crop's safety margin
    LOCAL_WARP_GRID_SIZE = 24           # Coarse (Gc,Gc) field grid, sampled on demand by
                                         # consumers -- never materialised at full res
    LOCAL_WARP_BANDWIDTH_FRAC = 0.35    # Gaussian local-regression bandwidth as a fraction
                                         # of min(H,W) -- displacement varies on far larger
                                         # spatial scales than DBE's background patches,
                                         # from far sparser samples (dozens of stars vs
                                         # thousands of patches)
    LOCAL_WARP_OUTLIER_SIGMA = 2.5      # Tukey-biweight IRLS outlier tolerance
    LOCAL_WARP_OUTLIER_ITERS = 3        # IRLS reweighting passes

    # Comet tracking / nucleus detection
    COMET_DOG_SIGMA_SMALL = 2.0     # DoG small sigma (suppresses point sources)
    COMET_DOG_SIGMA_LARGE = 10.0    # DoG large sigma (enhances diffuse coma core)

    # Robust-PCA (Principal Component Pursuit) master calibration frames
    ROBUST_PCA_MIN_FRAMES = 5       # Below this, the low-rank/sparse split is
                                     # underdetermined -- make_master falls back to median
    ROBUST_PCA_AUTO_MAX_FRAMES = 10 # --auto only auto-upgrades median->robust_pca for a
                                     # calibration type at or below this frame count.
                                     # Cost scales O(N^2 x pixels); measured 1264s/~21min
                                     # at N=20 -- too slow for a silent --auto default at
                                     # that scale (see _build_masters's comment). At
                                     # N<=10 that scales to ~(10/20)^2 * 1264s ~= 316s
                                     # (~5.3min), judged tolerable for --auto; still
                                     # opt-in via --master-method robust_pca above this
    ROBUST_PCA_MAX_ITERS = 50       # IALM iterations (each is one economy SVD of an
                                     # (N, H*W*C) matrix, via src.robust_pca's
                                     # Gram-matrix-trick + native gram_matrix_wide/
                                     # small_times_wide kernels -- ~9x over a direct
                                     # np.linalg.svd call on a realistic problem shape
                                     # (N=20, P=18M), ~2.3x end-to-end after the
                                     # non-SVD per-iteration ops that don't benefit;
                                     # measured 1264s/~21min full run, down from
                                     # 2910s/~48.5min pre-optimization. Bounded, not
                                     # adaptive-early-exit beyond the tolerance below)
    ROBUST_PCA_TOL = 1e-7           # Relative Frobenius-norm residual convergence tolerance

    # PSF-kernel drizzle resampling
    DRIZZLE_PSF_KERNEL_SIZE = 9     # Tap radius for the drizzle resample kernel when
                                     # --drizzle-kernel psf is set -- deliberately much
                                     # smaller than RL_PSF_SIZE (31): this is a per-pixel
                                     # resample tap count, not a deconvolution kernel
    DRIZZLE_PSF_PHASES = 16         # Subpixel phase quantization per axis for the
                                     # precomputed tap-weight table

    # Iterative back-projection (IBP, Irani & Peleg 1991) super-resolution refinement
    IBP_RELAX = 0.15                # Step-size / damping factor on the per-iteration
                                     # back-projected update. Swept on synthetic
                                     # known-ground-truth super-res data (see
                                     # tests/test_ibp_super_res.py): 0.5 (the original
                                     # guess) reliably makes RMSE *worse*, not better --
                                     # the direction of the update is correct (verified:
                                     # RMSE drops monotonically with relax at a single
                                     # iteration) but 0.5 overshoots. 0.15 consistently
                                     # improved RMSE across 5 synthetic seeds; combined
                                     # with ~5 iterations (RMSE bottoms out there, then
                                     # rises again -- classic IBP noise amplification).
    DRIZZLE_PSF_WIENER_K = 0.02     # Wiener regularization constant for the PSF inverse
                                     # filter the tap table is built from (see
                                     # build_drizzle_psf_table). Using the raw PSF shape
                                     # as the resample kernel measurably BROADENS stars
                                     # (convolving a sigma~2px star with a sigma~2px
                                     # kernel gives sigma*sqrt(2) -- verified empirically,
                                     # not just theory). This value was swept on a
                                     # synthetic Gaussian-PSF star: ~0.03 is close to
                                     # neutral (no broadening), ~0.01 gives ~9% FWHM
                                     # sharpening; noise response stays well-behaved
                                     # (suppressed, not amplified) across that whole
                                     # range, so 0.02 is a conservative middle default,
                                     # not a hard optimum

    # astrollm (external defect/quality/category classifier, --astrollm)
    ASTROLLM_TIMEOUT_S = 60.0       # Per-frame subprocess timeout for infer.py
    ASTROLLM_OUTLIER_SIGMA = 2.0    # Session-relative quality_score flag threshold
                                     # (advisory logging only -- see src/astrollm.py)


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
