"""Data models and configuration constants."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class RunCancelled(Exception):
    """Raised at a cooperative checkpoint (``args._cancel_event`` set) to
    unwind a run cleanly -- not a failure. Checked in the Phase 1 per-frame
    loops (``frame_processor.execute_frame_processing``, the highest-value
    spot: usually the longest-running phase) and between targets in
    ``cli.process_directory`` (multi-session/hierarchical runs). Phases 2-4
    of a single target aren't interruptible yet -- once one starts it runs
    to completion, same as before this existed."""


class Config:
    """Central configuration for magic numbers and thresholds."""
    HOT_PIXEL_THRESHOLD = 12.0
    HOT_PIXEL_BAYER_THRESHOLD = 5.0  # Lower for Bayer detection (MAD-based, robust)
    HOT_PIXEL_STAR_SUPPORT = 3.0     # Bayer hot-pixel test: keep a flagged pixel whose neighbours are > this many sigma high (it is a star)
    SPIKE_REJECT_SIGMA = 5.0         # --spike-reject: excess over the 8 same-plane neighbours' median, in plane sigma
    SPIKE_REJECT_CONTRAST = 5.0      # --spike-reject: excess must also be this many times the neighbours' own lift (sharpness)
    SPIKE_REJECT_SUPPORT_FRAC = 0.15 # --spike-reject: a 1-px mosaic neighbour above this fraction of the peak 'supports' it; 3+ of 4 = star
    OVEREXPOSED_FRACTION = 0.10      # Reject a sub with more raw pixels than this at full scale (measured: every normal session on the dev machine <= 2.1%, NGC 2244; overexposed SWAN subs 52% and 99.8%)
    CA_MIN_SHIFT_PX = 0.25           # Session CA below this: skip the correction warp entirely
    SESSION_CFA_MIN_FRAMES = 12      # Fewer lights than this: per-frame CFA equalisation, not one session estimate
    SESSION_CFA_PROBE_FRAMES = 8     # Frames spread through the session that the estimate is taken from
    MERGE_MIN_OVERLAP = 0.25        # --merge: min warped-footprint overlap before refusing
    MERGE_MIN_CORRELATION = 0.15     # --merge: min aligned-vs-new luminance correlation
    WHITE_PATCH_PERCENTILE = 99.5
    CROP_MARGIN = 2
    CENTROID_PERCENTILES = [95, 90, 85, 80]
    QUALITY_LOW_BRIGHTNESS = 10
    QUALITY_LOW_CONTRAST = 1
    LARGE_SHIFT_WARNING_PX = 20
    MIN_RECOMMENDED_FRAMES = 10
    PREVIEW_JPEG_QUALITY = 95
    PREVIEW_STRETCH_PERCENTILES = (1, 99)
    # Lowest preview white point, in sky sigma above the sky median. The white
    # point is the 99.5th luminance percentile, which on a small target in an
    # empty field (a galaxy covering ~0.5% of the frame) falls on barely-above-sky
    # pixels: measured 5.3 sigma (Black Eye) and 6.8 sigma (Sunflower), so sky
    # noise spanned the whole display range (a field of white speckles) and the
    # galaxy core clipped flat. 50 sigma shows their rings/arms on a clean sky;
    # frames where the percentile is already higher (Crab +70 sigma) are
    # unchanged, Hercules (+37) nearly so. 100 sigma started to dim the galaxies.
    PREVIEW_WHITE_MIN_SIGMA = 50.0
    # Highlight roll-off above the preview white point, for an *extended* bright
    # region (a nebula core) rather than star cores: on a real Orion stack the
    # 99.5th-percentile white point sat at 68k ADU against a core reaching 166k,
    # so the whole Trapezium region (25.6k connected pixels, 0.44% of the frame)
    # clipped flat white while the FITS showed its structure. Applied when the
    # largest connected region above white covers at least
    # PREVIEW_ROLLOFF_MIN_AREA of the frame (Orion 0.44%; star cores only on
    # Black Eye / Whirlpool / a dense Flaming Star field: 0.003-0.012%). Below
    # white the usual curve is scaled into [0, KNEE]; above it a log curve of
    # STRENGTH maps white..p99.99 into [KNEE, 1].
    PREVIEW_ROLLOFF_MIN_AREA = 0.001
    PREVIEW_ROLLOFF_KNEE = 0.85
    PREVIEW_ROLLOFF_STRENGTH = 8.0
    # Colour calibration of a stack with clipped star cores: a channel has a
    # clipped plateau when at least this many separate regions (stars) sit within
    # 2% of its maximum. Saturated cores stack to nearly one level (six on a real
    # Sunflower stack within 1.7%); a smooth unclipped galaxy core near the
    # maximum is one region, and unclipped star peaks rarely agree to 2%.
    CLIP_PLATEAU_MIN_STARS = 3
    PREVIEW_MAX_DIMENSION = 8192
    TILE_SIZE = 256  # Tile size for tiled sigma-clip (pixels)
    FWHM_CUTOUT_RADIUS = 10  # Cutout radius for FWHM measurement
    FWHM_MAX_STARS = 50  # Max stars to measure for FWHM
    ARCSINH_STRETCH_FACTOR = 5.0  # Default arcsinh stretch factor
    STAR_MASK_MAX_STARS = 500  # Max stars for mask generation
    AFFINE_MAX_STARS = 80  # Max stars for affine matching
    REG_MIN_STARS = 12  # Registration catalog thinner than this is re-detected at lower thresholds (noisy/hazy subs)
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
    # Anisotropic diffusion edge threshold, in units of the image's sky sigma
    # (postprocess._aniso_kappa). It was a fixed 30 ADU, which against stack
    # sky sigmas of ~35-100+ ADU meant a different filter on every session.
    # 1.0 trades noise against detail. tools/bench_denoise_quality.py's
    # curvelet+aniso chain (option 2, 15 iterations, six synthetic scenes):
    # fine structure kept 0.95-1.0 at 0.5, 0.80-0.93 at 1.0 (0.80 on the noisy
    # nebula), 0.62-0.84 at 2.0. tools/bench_phase4.py on real nebula stacks:
    # display sky noise roughly halves per doubling (Veil 6.1 / 3.0 / 1.5,
    # Crab 2.7 / 1.4 / 0.6 at 0.5 / 1 / 2). The old fixed 30 ADU was 3 sigma on
    # Veil, 0.65 on Crab and 0.3 on Trifid -- three different filters.
    ANISO_KAPPA_SIGMA = 1.0
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
    ROBUST_PCA_AUTO_MAX_FRAMES = 25 # --auto only auto-upgrades median->robust_pca for a
                                     # calibration type at or below this frame count.
                                     # History: originally set to 10 after an N=20 real-
                                     # shape (2048x3056 mono) measurement of 1264s/~21min
                                     # was judged too slow for a silent --auto default, and
                                     # N=15's ~717s/~12min (extrapolated then, from a
                                     # measured N^1.42 exponent) was judged not worth
                                     # widening for. Both those costs were dominated by
                                     # robust_pca_decompose's per-iteration plain-numpy
                                     # elementwise arithmetic -- once that got fused into
                                     # native kernels (robust_pca_pre_svd_input/_iterate,
                                     # the L-update matmul routed through small_times_wide;
                                     # see CHANGELOG "Several real perf fixes"), the SVD
                                     # step dominates more and the real N-exponent measured
                                     # steeper (~N^1.91, tools/bench_robust_pca_scale.py) --
                                     # but the *absolute* cost dropped enough that it no
                                     # longer matters: direct real-shape measurements
                                     # (same 2048x3056 mono frames, tested up to N=30 for
                                     # a fit, not just extrapolated) gave N=10 45s, N=15
                                     # 89s, N=20 164s, N=25 258s, N=30 377s/~6.3min --
                                     # i.e. today's N=25 costs less than half of what N=10
                                     # cost when this threshold was first judged tolerable.
                                     # Widened to 25 on that basis; N=30's ~6.3min sits
                                     # right at the old N=10 tolerance bar, so left as the
                                     # next candidate rather than taken now. Opt-in via
                                     # --master-method robust_pca above this frame count
                                     # regardless
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
    FLAT_FROM_LIGHTS_DOWNSAMPLE = 4 # --flat-from-lights only (never real bias/dark/flat
                                     # robust_pca): block-averages each of the 4 Bayer
                                     # sub-planes by this factor before decomposition,
                                     # cutting P (and the O(N^2 x P) cost) by ~16x -- the
                                     # low-rank content this path actually wants (flat-field
                                     # vignetting, dust motes) is smooth well above a 4-pixel
                                     # scale, unlike a real dark/bias master's per-pixel hot
                                     # pixels, which is why this knob doesn't exist for those.
                                     # Not yet measured end-to-end against a real vignetted
                                     # session (only against a flat synthetic frame with no
                                     # vignetting to recover) -- see robust_pca_master's
                                     # bayer_block_downsample/_upsample for the mechanism.


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
