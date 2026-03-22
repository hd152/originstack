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


def generate_star_mask(shape: Tuple[int, int], star_positions, fwhm: float = 3.0) -> np.ndarray:
    """Generate a float mask with Gaussian PSFs at detected star positions."""
    mask = np.zeros(shape, dtype=np.float32)
    if star_positions is None or len(star_positions) == 0:
        return mask
    sigma = fwhm / 2.355
    radius = int(3 * sigma) + 1
    H, W = shape
    n_stars = min(len(star_positions), Config.STAR_MASK_MAX_STARS)
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
    """Measure median FWHM from star cutouts using half-max area method."""
    if cutout_radius is None:
        cutout_radius = Config.FWHM_CUTOUT_RADIUS
    if star_positions is None or len(star_positions) == 0:
        return 0.0
    H, W = img.shape
    fwhms = []
    n_stars = min(len(star_positions), Config.FWHM_MAX_STARS)
    # Sort by flux (brightest first) for more reliable measurements
    try:
        sorted_idx = np.argsort(star_positions['flux'])[::-1]
    except (KeyError, TypeError):
        sorted_idx = range(n_stars)
    for idx in sorted_idx[:n_stars]:
        star = star_positions[idx]
        y = int(round(float(star['ycentroid'])))
        x = int(round(float(star['xcentroid'])))
        if (y < cutout_radius or y >= H - cutout_radius or
                x < cutout_radius or x >= W - cutout_radius):
            continue
        cutout = img[y - cutout_radius:y + cutout_radius + 1,
                     x - cutout_radius:x + cutout_radius + 1].astype(np.float64)
        peak = np.max(cutout)
        bg = np.percentile(cutout, 25)
        # Skip if star has very low contrast relative to noise
        if (peak - bg) < np.std(cutout) * 2.0:
            continue
        half_max = (peak + bg) / 2.0
        above_half = np.sum(cutout > half_max)
        fwhm_est = 2.0 * np.sqrt(above_half / np.pi)
        if 0.5 <= fwhm_est < cutout_radius * 2.5:
            fwhms.append(fwhm_est)
    return float(np.median(fwhms)) if fwhms else 0.0


def validate_image_data(img: np.ndarray, name: str = "") -> Tuple[bool, Optional[str]]:
    """Validate image data for common issues. Returns (is_valid, error_message)."""

    # Check for NaN or Inf
    if not np.isfinite(img).all():
        nan_count = np.isnan(img).sum()
        inf_count = np.isinf(img).sum()
        return False, f"contains {nan_count} NaN and {inf_count} Inf values"

    # Check for completely flat/dead image
    if np.std(img) < 0.1:
        return False, f"flat image (std={np.std(img):.3f})"

    # Check for saturated image (>95% of pixels at max value)
    max_val = np.max(img)
    if max_val > 0:
        saturated_fraction = np.sum(img >= max_val * 0.999) / img.size
        if saturated_fraction > 0.95:
            return False, f"saturated ({saturated_fraction*100:.1f}% at max)"

    # Check for mostly zeros (dead/corrupt)
    zero_fraction = np.sum(img == 0) / img.size
    if zero_fraction > 0.5:
        return False, f"mostly zeros ({zero_fraction*100:.1f}%)"

    # Check dynamic range
    p01, p99 = np.percentile(img, [1, 99])
    if p99 - p01 < 10:
        return False, f"insufficient dynamic range ({p99-p01:.1f})"

    return True, None


def _detect_stars_multi_fwhm(bg_sub: np.ndarray, threshold: float):
    """Run DAOStarFinder at FWHM 2, 3, 5, 8 and return the best quality-filtered table.

    Short-circuits as soon as a trial yields >=20 quality stars.  If the strict
    roundness/sharpness filter rejects everything, retries with relaxed thresholds
    so that slightly overexposed or soft stars still register.  Returns None when
    DAOStarFinder is unavailable or no sources are found at all.
    """
    if DAOStarFinder is None:
        return None
    best_sources = None
    best_quality_count = 0
    all_raw_sources = None  # best raw result across trials, kept for relaxed fallback

    for trial_fwhm in (2.0, 3.0, 5.0, 8.0):
        daof = DAOStarFinder(fwhm=trial_fwhm, threshold=threshold)
        trial_sources = daof(bg_sub)
        if trial_sources is None or len(trial_sources) == 0:
            continue
        if all_raw_sources is None or len(trial_sources) > len(all_raw_sources):
            all_raw_sources = trial_sources
        round_ok = (np.abs(trial_sources['roundness1']) < 0.5) & \
                   (np.abs(trial_sources['roundness2']) < 0.5)
        sharp_ok = (trial_sources['sharpness'] > 0.3) & \
                   (trial_sources['sharpness'] < 0.9)
        quality_mask = round_ok & sharp_ok
        quality_count = int(np.sum(quality_mask))
        if quality_count > best_quality_count:
            best_quality_count = quality_count
            best_sources = trial_sources[quality_mask]
        if best_quality_count >= 20:
            break

    # Strict filter found nothing — retry with relaxed thresholds so slightly
    # overexposed, soft, or elongated stars still contribute to registration.
    if best_sources is None and all_raw_sources is not None and len(all_raw_sources) > 0:
        round_ok = (np.abs(all_raw_sources['roundness1']) < 0.7) & \
                   (np.abs(all_raw_sources['roundness2']) < 0.7)
        sharp_ok = (all_raw_sources['sharpness'] > 0.1) & \
                   (all_raw_sources['sharpness'] < 1.0)
        relaxed_mask = round_ok & sharp_ok
        if np.sum(relaxed_mask) > 0:
            best_sources = all_raw_sources[relaxed_mask]
            logging.debug(f"DAOStarFinder: strict filter found 0 stars, "
                          f"relaxed filter found {len(best_sources)}")

    return best_sources


def compute_quality_metrics(img: np.ndarray) -> Dict:
    """Comprehensive quality analysis with multiple metrics."""

    # Downsample large images for cheap scalar stats only.
    # Star detection and FWHM run at full resolution — DAOStarFinder's
    # hardcoded FWHM trials (3, 5, 8 px) assume full-res star profiles;
    # downsampling compresses stars below the detection threshold.
    _min_dim = min(img.shape)
    _ds = 4 if _min_dim >= 2048 else (2 if _min_dim >= 1024 else 1)
    img_s = img[::_ds, ::_ds] if _ds > 1 else img
    img_s_stars = img  # always full resolution

    # Basic statistics
    brightness = float(np.median(img_s))
    mean = float(np.mean(img_s))
    contrast = float(np.std(img_s))

    # Percentiles for outlier detection
    p01, p05, p25, p50, p75, p95, p99 = np.percentile(img_s, [1, 5, 25, 50, 75, 95, 99])

    # Signal-to-noise estimation
    # Use sigma-clipped statistics if available
    snr = 0.0
    background = mean
    noise = contrast

    _scs_bg_mean = _scs_bg_median = _scs_bg_std = None
    if sigma_clipped_stats is not None:
        try:
            _scs_bg_mean, _scs_bg_median, _scs_bg_std = sigma_clipped_stats(img_s, sigma=3.0, maxiters=3)
            background = float(_scs_bg_median)
            noise = float(_scs_bg_std)
            snr = (p95 - background) / (noise + 1e-12) if noise > 0 else 0.0
        except:
            snr = (p95 - mean) / (contrast + 1e-12)
    else:
        snr = (p95 - mean) / (contrast + 1e-12)

    # Star detection
    star_count = 0
    star_snr = 0.0

    sources_s = None  # star positions in img_s_stars coordinates
    if DAOStarFinder is not None and _scs_bg_std is not None:
        try:
            bg_mean, bg_median, bg_std = _scs_bg_mean, _scs_bg_median, _scs_bg_std
            # Threshold for background-subtracted image: N * sigma above zero
            threshold = 5.0 * float(bg_std)
            bg_sub = img_s_stars - float(bg_median)

            # Try multiple FWHM values to handle varying seeing/debayer methods.
            # DAOStarFinder is sensitive to the fwhm parameter — a mismatch
            # causes it to reject real stars via its sharpness/roundness criteria.
            sources_s = _detect_stars_multi_fwhm(bg_sub, threshold)

            if sources_s is not None and len(sources_s) > 0:
                star_count = len(sources_s)
                # Calculate median star SNR
                star_peaks = sources_s['peak']
                star_snr = float(np.median(star_peaks)) / (noise + 1e-12)
        except Exception as e:
            logging.debug(f"DAOStarFinder failed: {type(e).__name__}: {e}")
            sources_s = None

    # Fallback star detection using local maxima
    if star_count == 0:
        try:
            # Find bright local maxima — use a larger neighbourhood and stricter
            # threshold to avoid counting noise peaks as stars
            threshold = background + 5.0 * noise
            from scipy.ndimage import maximum_filter
            local_max = maximum_filter(img_s_stars, size=11)
            detected_peaks = (img_s_stars == local_max) & (img_s_stars > threshold)
            star_count = min(int(np.sum(detected_peaks)), 500)

            if star_count > 0:
                peak_values = img_s_stars[detected_peaks]
                star_snr = float(np.median(peak_values)) / (noise + 1e-12)
        except:
            star_count = 0

    # Focus/sharpness metric using Laplacian variance (cheap; use heavily downsampled img_s)
    try:
        from scipy.ndimage import laplace
        laplacian = laplace(img_s.astype(np.float32))
        sharpness = float(np.var(laplacian))
    except:
        sharpness = 0.0

    # FWHM measured at full resolution — no scaling needed
    fwhm = 0.0
    if star_count > 0 and sources_s is not None:
        fwhm = measure_fwhm(img_s_stars, sources_s)

    # Star positions are already full-resolution coordinates
    sources = sources_s

    # Composite quality score
    star_factor = min(star_count / 50.0, 1.0) if star_count > 0 else 0.01
    snr_factor = min(snr / 10.0, 1.0) if snr > 0 else 0.01
    # Penalize poor focus (high FWHM) — prefer tighter stars
    fwhm_factor = 1.0
    if fwhm > 0:
        fwhm_factor = max(0.1, 1.0 / (1.0 + max(0, fwhm - 2.0) ** 2 * 0.1))

    score = brightness * contrast * star_factor * snr_factor * fwhm_factor * 100.0

    return {
        'brightness': brightness,
        'mean': mean,
        'contrast': contrast,
        'snr': snr,
        'star_count': star_count,
        'star_snr': star_snr,
        'sharpness': sharpness,
        'fwhm': fwhm,
        'background': background,
        'noise': noise,
        'score': score,
        'p01': p01,
        'p50': p50,
        'p75': p75,
        'p95': p95,
        'p99': p99,
        'dynamic_range': p99 - p01,
        '_star_sources': sources,
    }
