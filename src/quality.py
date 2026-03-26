"""Frame quality analysis: star detection, FWHM measurement, quality metrics."""
from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional

import numpy as np

from src.models import Config

try:
    from photutils.detection import DAOStarFinder
except Exception:
    DAOStarFinder = None

try:
    from astropy.stats import sigma_clipped_stats
except Exception:
    sigma_clipped_stats = None

# Module-level imports for scipy — avoids repeated sys.modules lookups and
# attribute resolution on every call to compute_quality_metrics.
try:
    from scipy.ndimage import laplace, maximum_filter, gaussian_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    laplace = maximum_filter = gaussian_filter = None
    _SCIPY_AVAILABLE = False


def generate_star_mask(shape: Tuple[int, int], star_positions, fwhm: float = 3.0) -> np.ndarray:
    """Generate a float mask with Gaussian PSFs at detected star positions.

    Fast path: centroid coordinates are extracted with a fully vectorised
    operation, bounds-clipped, and scattered onto a point image in one
    indexing call.  scipy.ndimage.gaussian_filter then blurs the whole
    array in a single pass — replacing N per-star np.exp evaluations.
    """
    mask = np.zeros(shape, dtype=np.float32)
    if star_positions is None or len(star_positions) == 0:
        return mask

    sigma = fwhm / 2.355
    H, W = shape
    n_stars = min(len(star_positions), Config.STAR_MASK_MAX_STARS)

    if gaussian_filter is not None:
        # Vectorised centroid extraction and scatter — no Python loop.
        ys = np.round(np.array([float(star_positions[i]['ycentroid'])
                                 for i in range(n_stars)])).astype(int)
        xs = np.round(np.array([float(star_positions[i]['xcentroid'])
                                 for i in range(n_stars)])).astype(int)
        in_bounds = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        point_mask = np.zeros(shape, dtype=np.float32)
        # np.maximum.at handles multiple stars landing on the same pixel safely.
        np.maximum.at(point_mask, (ys[in_bounds], xs[in_bounds]), 1.0)
        blurred = gaussian_filter(point_mask, sigma=sigma)
        max_val = blurred.max()
        if max_val > 0:
            mask = blurred / max_val  # normalise to [0, 1]
        return mask

    # Slow fallback (scipy unavailable): original per-star Gaussian loop.
    radius = int(3 * sigma) + 1
    for i in range(n_stars):
        star = star_positions[i]
        y = int(round(float(star['ycentroid'])))
        x = int(round(float(star['xcentroid'])))
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        gaussian = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], gaussian)
    return mask


def measure_fwhm(img: np.ndarray, star_positions, cutout_radius: int = None) -> float:
    """Measure median FWHM from star cutouts using half-max area method.

    Performance improvements vs original:
    - Border stars are pre-filtered vectorised before the loop.
    - Background estimated with np.partition (O(n)) not np.percentile (O(n log n)).
    - Low-contrast check reuses already-computed peak/bg range instead of
      calling np.std(cutout) — a full extra traversal per star.
    - Breaks early once 20 reliable FWHM samples are collected.
    """
    if cutout_radius is None:
        cutout_radius = Config.FWHM_CUTOUT_RADIUS
    if star_positions is None or len(star_positions) == 0:
        return 0.0

    H, W = img.shape
    fwhms = []
    n_stars = min(len(star_positions), Config.FWHM_MAX_STARS)

    # Sort by flux (brightest first) for more reliable measurements.
    try:
        sorted_idx = np.argsort(star_positions['flux'])[::-1]
    except (KeyError, TypeError):
        sorted_idx = range(n_stars)

    # Vectorised border pre-filter — compute validity mask before the loop.
    try:
        all_y = np.array([float(star_positions[i]['ycentroid']) for i in sorted_idx[:n_stars]])
        all_x = np.array([float(star_positions[i]['xcentroid']) for i in sorted_idx[:n_stars]])
    except (TypeError, IndexError):
        all_y = np.array([float(star_positions[idx]['ycentroid']) for idx in sorted_idx[:n_stars]])
        all_x = np.array([float(star_positions[idx]['xcentroid']) for idx in sorted_idx[:n_stars]])

    iy = np.round(all_y).astype(int)
    ix = np.round(all_x).astype(int)
    border_valid = (
        (iy >= cutout_radius) & (iy < H - cutout_radius) &
        (ix >= cutout_radius) & (ix < W - cutout_radius)
    )

    for orig_idx, y, x, is_valid in zip(sorted_idx[:n_stars], iy, ix, border_valid):
        if not is_valid:
            continue

        cutout = img[y - cutout_radius:y + cutout_radius + 1,
                     x - cutout_radius:x + cutout_radius + 1].astype(np.float64)

        peak = float(np.max(cutout))

        # O(n) partial sort for background estimate instead of O(n log n) percentile.
        flat = cutout.ravel()
        k = flat.size // 4
        bg = float(np.partition(flat, k)[k])

        # Low-contrast check: reuse peak-bg signal instead of calling np.std —
        # avoids a full extra traversal per star cutout.
        signal = peak - bg
        if signal < bg * 0.5 or signal < 10.0:
            continue

        half_max = (peak + bg) / 2.0
        above_half = int(np.sum(cutout > half_max))
        fwhm_est = 2.0 * np.sqrt(above_half / np.pi)
        if 0.5 <= fwhm_est < cutout_radius * 2.5:
            fwhms.append(fwhm_est)

        # Median stabilises after ~20 good samples — stop early.
        if len(fwhms) >= 20:
            break

    return float(np.median(fwhms)) if fwhms else 0.0


def validate_image_data(img: np.ndarray, name: str = "") -> Tuple[bool, Optional[str]]:
    """Validate image data for common issues. Returns (is_valid, error_message).

    Performance improvements vs original:
    - Single np.isfinite pass; NaN/Inf counts only computed on the error path.
    - np.std computed once and reused.
    - max_val derived from percentile batch call, eliminating a standalone
      np.max traversal.
    - Zero boolean array computed once and reused for the fraction check.
    """
    # One isfinite pass — NaN/Inf counts only paid for on the failure path.
    if not np.isfinite(img).all():
        nan_count = int(np.isnan(img).sum())
        inf_count = int(np.isinf(img).sum())
        return False, f"contains {nan_count} NaN and {inf_count} Inf values"

    # Single std computation, reused in the error message.
    img_std = float(np.std(img))
    if img_std < 0.1:
        return False, f"flat image (std={img_std:.3f})"

    # Batch percentile call — p100 == max for finite arrays, avoids a
    # separate np.max traversal.
    p01, p99, p_max = np.percentile(img, [1, 99, 100])
    max_val = float(p_max)

    if max_val > 0:
        saturated_fraction = float(np.sum(img >= max_val * 0.999)) / img.size
        if saturated_fraction > 0.95:
            return False, f"saturated ({saturated_fraction*100:.1f}% at max)"

    # Compute the boolean zero mask once; .sum() reuses it without a second
    # comparison pass.
    zero_fraction = float((img == 0).sum()) / img.size
    if zero_fraction > 0.5:
        return False, f"mostly zeros ({zero_fraction*100:.1f}%)"

    if p99 - p01 < 10:
        return False, f"insufficient dynamic range ({p99-p01:.1f})"

    return True, None


def _detect_stars_multi_fwhm(bg_sub: np.ndarray, threshold: float):
    """Run DAOStarFinder at FWHM 2, 3, 5, 8 and return the best quality-filtered table.

    Short-circuits as soon as a trial yields >=20 quality stars.  If the strict
    roundness/sharpness filter rejects everything, retries with relaxed thresholds.
    Returns None when DAOStarFinder is unavailable or no sources are found at all.

    Performance improvements vs original:
    - Roundness filter fused into a single np.maximum(|r1|, |r2|) pass instead
      of two separate np.abs calls with independent boolean arrays.
    - Relaxed fallback recomputes the same fused roundness on all_raw_sources
      once, not twice as in the original duplicated mask logic.
    """
    if DAOStarFinder is None:
        return None

    best_sources = None
    best_quality_count = 0
    all_raw_sources = None

    for trial_fwhm in (2.0, 3.0, 5.0, 8.0):
        daof = DAOStarFinder(fwhm=trial_fwhm, threshold=threshold)
        trial_sources = daof(bg_sub)
        if trial_sources is None or len(trial_sources) == 0:
            continue
        if all_raw_sources is None or len(trial_sources) > len(all_raw_sources):
            all_raw_sources = trial_sources

        # Fused roundness: one np.maximum call instead of two np.abs + two comparisons.
        max_roundness = np.maximum(
            np.abs(trial_sources['roundness1']),
            np.abs(trial_sources['roundness2'])
        )
        sharpness = trial_sources['sharpness']
        quality_mask = (max_roundness < 0.5) & (sharpness > 0.3) & (sharpness < 0.9)
        quality_count = int(np.sum(quality_mask))
        if quality_count > best_quality_count:
            best_quality_count = quality_count
            best_sources = trial_sources[quality_mask]
        if best_quality_count >= 20:
            break

    # Strict filter found nothing — retry with relaxed thresholds.
    if best_sources is None and all_raw_sources is not None and len(all_raw_sources) > 0:
        max_roundness_relaxed = np.maximum(
            np.abs(all_raw_sources['roundness1']),
            np.abs(all_raw_sources['roundness2'])
        )
        sharpness_relaxed = all_raw_sources['sharpness']
        relaxed_mask = (
            (max_roundness_relaxed < 0.7) &
            (sharpness_relaxed > 0.1) &
            (sharpness_relaxed < 1.0)
        )
        if np.sum(relaxed_mask) > 0:
            best_sources = all_raw_sources[relaxed_mask]
            logging.debug(
                f"DAOStarFinder: strict filter found 0 stars, "
                f"relaxed filter found {len(best_sources)}"
            )

    return best_sources


def compute_quality_metrics(img: np.ndarray) -> Dict:
    """Comprehensive quality analysis with multiple metrics.

    Performance improvements vs original:
    - img_s cast to float32 once upfront — all downstream operations share
      the dtype without repeated implicit upcasting.
    - brightness derived from the already-computed p50 percentile — no
      separate np.median traversal.
    - Laplacian skips the per-call astype(float32) since img_s is already float32.
    - Dead elif/re-import branch for unavailable scipy removed — if the
      module-level import failed a per-call import will fail identically.
    - Return dict values explicitly cast to float to avoid returning numpy
      scalars, which can cause downstream serialisation surprises.
    """
    _min_dim = min(img.shape)
    _ds = 4 if _min_dim >= 2048 else (2 if _min_dim >= 1024 else 1)
    # Cast to float32 once — shared by stats, percentile, and laplace below.
    img_s = (img[::_ds, ::_ds] if _ds > 1 else img).astype(np.float32)
    img_s_stars = img  # star detection always at full resolution

    # All percentiles in one pass — p50 replaces a separate np.median call.
    p01, p05, p25, p50, p75, p95, p99 = np.percentile(img_s, [1, 5, 25, 50, 75, 95, 99])
    brightness = float(p50)  # median == p50, no extra traversal

    mean = float(np.mean(img_s))
    contrast = float(np.std(img_s))

    # Signal-to-noise estimation
    snr = 0.0
    background = mean
    noise = contrast

    _scs_bg_mean = _scs_bg_median = _scs_bg_std = None
    if sigma_clipped_stats is not None:
        try:
            _scs_bg_mean, _scs_bg_median, _scs_bg_std = sigma_clipped_stats(
                img_s, sigma=3.0, maxiters=3
            )
            background = float(_scs_bg_median)
            noise = float(_scs_bg_std)
            snr = (p95 - background) / (noise + 1e-12) if noise > 0 else 0.0
        except Exception:
            snr = (p95 - mean) / (contrast + 1e-12)
    else:
        snr = (p95 - mean) / (contrast + 1e-12)

    # Star detection
    star_count = 0
    star_snr = 0.0
    sources_s = None

    if DAOStarFinder is not None and _scs_bg_std is not None:
        try:
            threshold = 5.0 * float(_scs_bg_std)
            bg_sub = img_s_stars - float(_scs_bg_median)
            sources_s = _detect_stars_multi_fwhm(bg_sub, threshold)
            if sources_s is not None and len(sources_s) > 0:
                star_count = len(sources_s)
                star_snr = float(np.median(sources_s['peak'])) / (noise + 1e-12)
        except Exception as e:
            logging.debug(f"DAOStarFinder failed: {type(e).__name__}: {e}")
            sources_s = None

    # Fallback: local-maxima detection.  The dead elif/re-import branch from
    # the previous version is removed — if the module-level scipy import
    # failed, a per-call import will fail identically and is not worth keeping.
    if star_count == 0 and maximum_filter is not None:
        try:
            threshold = background + 5.0 * noise
            local_max = maximum_filter(img_s_stars, size=11)
            detected_peaks = (img_s_stars == local_max) & (img_s_stars > threshold)
            star_count = min(int(np.sum(detected_peaks)), 500)
            if star_count > 0:
                star_snr = float(np.median(img_s_stars[detected_peaks])) / (noise + 1e-12)
        except Exception:
            star_count = 0

    # Laplacian sharpness — img_s is already float32, no cast needed.
    sharpness = 0.0
    if laplace is not None:
        try:
            sharpness = float(np.var(laplace(img_s)))
        except Exception:
            sharpness = 0.0

    # FWHM at full resolution
    fwhm = 0.0
    if star_count > 0 and sources_s is not None:
        fwhm = measure_fwhm(img_s_stars, sources_s)

    # Composite quality score
    star_factor = min(star_count / 50.0, 1.0) if star_count > 0 else 0.01
    snr_factor = min(snr / 10.0, 1.0) if snr > 0 else 0.01
    fwhm_factor = (
        max(0.1, 1.0 / (1.0 + max(0.0, fwhm - 2.0) ** 2 * 0.1))
        if fwhm > 0 else 1.0
    )
    score = brightness * contrast * star_factor * snr_factor * fwhm_factor * 100.0

    return {
        'brightness': brightness,
        'mean': mean,
        'contrast': contrast,
        'snr': float(snr),
        'star_count': star_count,
        'star_snr': float(star_snr),
        'sharpness': sharpness,
        'fwhm': fwhm,
        'background': float(background),
        'noise': float(noise),
        'score': float(score),
        'p01': float(p01),
        'p50': float(p50),
        'p75': float(p75),
        'p95': float(p95),
        'p99': float(p99),
        'dynamic_range': float(p99 - p01),
        '_star_sources': sources_s,
    }