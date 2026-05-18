"""Frame quality analysis: star detection, FWHM measurement, quality metrics."""
from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional

import numpy as np

from src.models import Config

DAOStarFinder = None

try:
    import sep as _sep_module
    _SEP_AVAILABLE = True
except ImportError:
    _sep_module = None
    _SEP_AVAILABLE = False


def _ensure_photutils():
    global DAOStarFinder
    if DAOStarFinder is None:
        try:
            from photutils.detection import DAOStarFinder as _dao
            if callable(_dao):
                DAOStarFinder = _dao
        except Exception:
            pass
    return DAOStarFinder


def _sep_detect_stars(img_2d: np.ndarray, noise: float) -> Optional[object]:
    """SEP (SourceExtractor) star detection — ~5-10x faster than DAOStarFinder.

    Returns a photutils-compatible structured array or None on failure/unavailable.
    """
    if not _SEP_AVAILABLE:
        return None
    try:
        data = np.ascontiguousarray(img_2d, dtype=np.float64)
        bkg = _sep_module.Background(data, bw=64, bh=64, fw=3, fh=3)
        bkg_sub = np.ascontiguousarray(data - bkg, dtype=np.float64)
        thresh = max(3.0 * float(bkg.globalrms), 5.0 * noise, 1e-6)
        raw = _sep_module.extract(bkg_sub, thresh, minarea=9)
    except Exception:
        return None
    if raw is None or len(raw) == 0:
        return None

    dt = np.dtype([
        ('xcentroid', np.float64), ('ycentroid', np.float64),
        ('flux', np.float64), ('peak', np.float64),
        ('roundness1', np.float64), ('roundness2', np.float64),
        ('sharpness', np.float64),
        ('a', np.float64), ('b', np.float64), ('theta', np.float64),
    ])
    out = np.zeros(len(raw), dtype=dt)
    out['xcentroid'] = raw['x']
    out['ycentroid'] = raw['y']
    out['flux'] = raw['flux']
    out['peak'] = raw['peak']
    a = np.maximum(raw['a'], 1e-6)
    b = np.maximum(raw['b'], 1e-6)
    roundness = 1.0 - np.minimum(b, a) / np.maximum(b, a)  # 0=circular, 1=linear
    out['roundness1'] = roundness
    out['roundness2'] = roundness
    out['sharpness'] = 0.5  # neutral; passes the (0.3, 0.9) quality filter
    out['a'] = a
    out['b'] = b
    out['theta'] = raw['theta']  # position angle in radians (SEP convention)

    quality_mask = roundness < 0.5
    if np.sum(quality_mask) == 0:
        quality_mask = roundness < 0.7
    filtered = out[quality_mask]
    return filtered if len(filtered) > 0 else None

# Module-level imports for scipy — avoids repeated sys.modules lookups and
# attribute resolution on every call to compute_quality_metrics.
try:
    from scipy.ndimage import laplace, maximum_filter, gaussian_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    laplace = maximum_filter = gaussian_filter = None
    _SCIPY_AVAILABLE = False


def generate_star_mask(shape: Tuple[int, int], star_positions: Optional[object], fwhm: float = 3.0) -> np.ndarray:
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
        ys = np.round(np.asarray(star_positions['ycentroid'][:n_stars], dtype=np.float64)).astype(int)
        xs = np.round(np.asarray(star_positions['xcentroid'][:n_stars], dtype=np.float64)).astype(int)
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


def measure_fwhm(img: np.ndarray, star_positions: Optional[object], cutout_radius: Optional[int] = None) -> float:
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
    except (KeyError, TypeError, ValueError):
        sorted_idx = range(n_stars)

    # Vectorised border pre-filter — column slicing + fancy indexing in C.
    indices = sorted_idx[:n_stars]
    all_y = np.asarray(star_positions['ycentroid'][indices], dtype=np.float64)
    all_x = np.asarray(star_positions['xcentroid'][indices], dtype=np.float64)

    iy = np.round(all_y).astype(int)
    ix = np.round(all_x).astype(int)
    border_valid = (
        (iy >= cutout_radius) & (iy < H - cutout_radius) &
        (ix >= cutout_radius) & (ix < W - cutout_radius)
    )

    valid_mask = border_valid.nonzero()[0]
    for idx in valid_mask:
        y, x = iy[idx], ix[idx]
        cutout = img[y - cutout_radius:y + cutout_radius + 1,
                     x - cutout_radius:x + cutout_radius + 1].astype(np.float32)

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

    # count_nonzero avoids allocating a full boolean array.
    zero_fraction = (img.size - np.count_nonzero(img)) / img.size
    if zero_fraction > 0.5:
        return False, f"mostly zeros ({zero_fraction*100:.1f}%)"

    if p99 - p01 < 10:
        return False, f"insufficient dynamic range ({p99-p01:.1f})"

    return True, None


def _detect_stars_multi_fwhm(bg_sub: np.ndarray, threshold: float) -> Optional[object]:
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
    _ensure_photutils()
    if DAOStarFinder is None:
        return None

    best_sources = None
    best_quality_count = 0
    all_raw_sources = None

    for trial_fwhm in (3.0, 5.0, 2.0, 8.0):  # 3px most common seeing; short-circuits early
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


def estimate_strehl_ratio(img: np.ndarray, star_positions,
                           fwhm: float = 0.0) -> float:
    """Estimate a Strehl-proxy metric from bright star PSF profiles.

    True Strehl ratio requires knowledge of the telescope aperture and
    wavelength.  This function computes a normalised proxy:

        S_proxy = peak_norm / peak_gaussian

    where ``peak_norm`` is the measured peak of the background-subtracted,
    unit-integral star profile, and ``peak_gaussian`` is the expected peak
    of a Gaussian PSF with the same measured FWHM:

        peak_gaussian = 4 ln(2) / (π · FWHM²)

    For a perfect Gaussian PSF the ratio is exactly 1.0.  Aberrated or
    atmospherically blurred PSFs have S_proxy < 1.0; over-sharp / spiky
    PSFs (e.g. from lucky imaging) may exceed 1.0.

    Args:
        img:            Float32 stacked image (H, W, 3).
        star_positions: Source table (DAOStarFinder output).
        fwhm:           Pre-measured median FWHM in pixels; if 0 it is
                        re-estimated from cutouts.

    Returns:
        Strehl proxy (float).  0.0 if estimation fails.
    """
    if star_positions is None or len(star_positions) == 0:
        return 0.0

    H, W = img.shape[:2]
    lum = (img if img.ndim == 2
           else 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])

    cutout_r = Config.STREHL_CUTOUT_RADIUS
    try:
        sorted_idx = np.argsort(star_positions['flux'])[::-1]
    except (KeyError, TypeError):
        sorted_idx = range(len(star_positions))

    measured_peaks = []
    measured_fwhms = []

    for idx in list(sorted_idx)[:20]:
        star = star_positions[idx]
        yc = int(round(float(star['ycentroid'])))
        xc = int(round(float(star['xcentroid'])))
        if (yc < cutout_r or yc >= H - cutout_r or
                xc < cutout_r or xc >= W - cutout_r):
            continue

        cutout = lum[yc - cutout_r:yc + cutout_r + 1,
                     xc - cutout_r:xc + cutout_r + 1].astype(np.float64)
        peak = float(cutout.max())
        bg = float(np.percentile(cutout, 25))
        if (peak - bg) < 10.0:
            continue
        # Skip saturated stars
        if int(np.sum(cutout >= peak - 1e-6 * max(peak, 1.0))) > 4:
            continue

        cutout_sub = np.maximum(cutout - bg, 0.0)
        total = cutout_sub.sum()
        if total < 1e-12:
            continue
        norm_peak = float(cutout_sub.max()) / total
        measured_peaks.append(norm_peak)

        # Per-star FWHM estimate
        half_max_val = cutout_sub.max() * 0.5
        above = int(np.sum(cutout_sub > half_max_val))
        if above > 0:
            measured_fwhms.append(2.0 * np.sqrt(above / np.pi))

        if len(measured_peaks) >= 10:
            break

    if not measured_peaks:
        return 0.0

    if fwhm <= 0.0:
        fwhm = float(np.median(measured_fwhms)) if measured_fwhms else 4.0
    if fwhm < 0.5:
        fwhm = 0.5

    # Gaussian peak for unit-integral normalised PSF
    gaussian_peak = 4.0 * np.log(2.0) / (np.pi * fwhm ** 2)

    mean_peak = float(np.median(measured_peaks))
    strehl_proxy = mean_peak / (gaussian_peak + 1e-12)
    return float(np.clip(strehl_proxy, 0.0, 5.0))


def measure_atmospheric_dispersion(img: np.ndarray, star_positions,
                                    cutout_radius: Optional[int] = None) -> float:
    """Measure RGB centroid offsets per star as an atmospheric dispersion proxy.

    Atmospheric dispersion shifts shorter wavelengths (blue) more than longer
    ones (red) along the altitude axis, stretching each star into a tiny
    spectrum.  This function measures the RMS separation between R, G, and B
    channel centroids for a set of bright stars; the result is a proxy for
    how much dispersion is degrading the data.

    A value < 0.3 px is negligible.  Values > 1 px indicate significant
    dispersion that warrants Atmospheric Dispersion Corrector use.

    Args:
        img:            Float32 stacked image (H, W, 3).
        star_positions: Source table (DAOStarFinder).
        cutout_radius:  Half-size of extraction window (default
                        Config.DISP_CUTOUT_RADIUS = 10 px).

    Returns:
        Median RGB centroid separation in pixels.  0.0 on failure.
    """
    if star_positions is None or len(star_positions) == 0:
        return 0.0
    if img.ndim != 3 or img.shape[2] < 3:
        return 0.0

    if cutout_radius is None:
        cutout_radius = Config.DISP_CUTOUT_RADIUS

    H, W = img.shape[:2]
    try:
        sorted_idx = np.argsort(star_positions['flux'])[::-1]
    except (KeyError, TypeError):
        sorted_idx = range(len(star_positions))

    dispersions = []
    yy, xx = np.mgrid[:2 * cutout_radius + 1, :2 * cutout_radius + 1]

    for idx in list(sorted_idx)[:30]:
        star = star_positions[idx]
        y0 = int(round(float(star['ycentroid'])))
        x0 = int(round(float(star['xcentroid'])))
        if (y0 < cutout_radius or y0 >= H - cutout_radius or
                x0 < cutout_radius or x0 >= W - cutout_radius):
            continue

        centroids = []
        for c in range(3):
            cut = img[y0 - cutout_radius:y0 + cutout_radius + 1,
                      x0 - cutout_radius:x0 + cutout_radius + 1, c].astype(np.float64)
            bg = float(np.percentile(cut, 25))
            cut = np.maximum(cut - bg, 0.0)
            total = cut.sum()
            if total < 1e-9:
                break
            cy = float(np.sum(yy * cut)) / total
            cx = float(np.sum(xx * cut)) / total
            centroids.append((cy, cx))

        if len(centroids) != 3:
            continue

        # RMS distance between all channel-pair centroids
        dists = []
        for i in range(3):
            for j in range(i + 1, 3):
                dy = centroids[i][0] - centroids[j][0]
                dx = centroids[i][1] - centroids[j][1]
                dists.append(np.sqrt(dy ** 2 + dx ** 2))
        dispersions.append(float(np.mean(dists)))

        if len(dispersions) >= 15:
            break

    return float(np.median(dispersions)) if dispersions else 0.0


def compute_brenner_sharpness(img: np.ndarray) -> float:
    """Brenner gradient function: mean of squared 2-pixel horizontal differences.

    More noise-robust than the Laplacian for star fields because the step size
    of two pixels suppresses high-frequency shot noise while still capturing the
    steep edges of star PSF cores.  A sharp frame scores higher than a blurred
    or trailed one of equal SNR.
    """
    lum = (img if img.ndim == 2
           else 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])
    lum = lum.astype(np.float64)
    diff = lum[:, 2:] - lum[:, :-2]
    return float(np.mean(diff * diff))


def measure_psf_anisotropy(sources) -> Tuple[float, float, str]:
    """Compute PSF ellipticity and position-angle scatter from SEP source catalog.

    Uses the semi-major/minor axes (a, b) and orientation angle (theta) stored
    in the extended SEP source array returned by _sep_detect_stars.

    Returns:
        median_ellipticity: median (a²-b²)/(a²+b²) across sources; 0=circular.
        pa_scatter_deg:     circular standard deviation of position angles (°).
        interpretation:     'isotropic' | 'tracking_drift' | 'field_curvature'

    Interpretation heuristics:
        - Low ellipticity → round stars, no issue.
        - High ellipticity + low PA scatter → uniform elongation direction
          (tracking drift or atmospheric dispersion along one axis).
        - High ellipticity + high PA scatter → direction varies spatially
          (field curvature, coma, or optical distortion across the FOV).
    """
    if sources is None or len(sources) == 0:
        return 0.0, 0.0, 'isotropic'

    if 'a' not in sources.dtype.names or 'b' not in sources.dtype.names:
        return 0.0, 0.0, 'isotropic'

    a = np.asarray(sources['a'], dtype=np.float64)
    b = np.asarray(sources['b'], dtype=np.float64)
    a2, b2 = a ** 2, b ** 2
    ellipticity = (a2 - b2) / np.maximum(a2 + b2, 1e-12)
    med_e = float(np.median(ellipticity))

    pa_scatter = 0.0
    if 'theta' in sources.dtype.names and len(sources) >= 3:
        angles = np.asarray(sources['theta'], dtype=np.float64)
        sin_mean = float(np.mean(np.sin(2.0 * angles)))
        cos_mean = float(np.mean(np.cos(2.0 * angles)))
        R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
        # Circular SD in degrees (factor of 2 because PA is π-periodic)
        pa_scatter = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(R, 1e-12)), 0.0))) * 0.5)

    if med_e < 0.15:
        interp = 'isotropic'
    elif pa_scatter < 20.0:
        interp = 'tracking_drift'
    else:
        interp = 'field_curvature'

    return float(med_e), float(pa_scatter), interp


def compute_multiscale_entropy(img: np.ndarray, levels: int = None) -> float:
    """Wavelet entropy ratio: fine-scale / coarse-scale detail energy.

    Decomposes luminance into ``levels`` wavelet scales using a db4 filter.
    A sharp, well-focused frame concentrates energy in fine (high-frequency)
    scales; a blurry or turbulence-smeared frame redistributes energy to
    coarser scales.  The ratio of finest-scale entropy to coarsest-scale
    entropy is therefore a seeing-quality indicator independent of SNR.

    Requires pywt.  Returns 0.0 when pywt is unavailable or the image is too
    small for the requested decomposition depth.
    """
    try:
        import pywt as _pywt
    except ImportError:
        return 0.0

    if levels is None:
        from src.models import Config
        levels = Config.WAVELET_ENTROPY_LEVELS

    lum = (img if img.ndim == 2
           else 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])
    lum = lum.astype(np.float32)

    try:
        coeffs = _pywt.wavedec2(lum, 'db4', level=levels)
    except Exception:
        return 0.0

    def _band_entropy(band: np.ndarray) -> float:
        flat = band.ravel().astype(np.float64)
        norm = np.linalg.norm(flat)
        if norm < 1e-12:
            return 0.0
        p = (flat / norm) ** 2
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log2(p)))

    # coeffs[0] = approximation; coeffs[1..N] = detail tuples per level
    # Level 1 = finest detail, level N = coarsest detail
    entropies_per_level = []
    for level_detail in coeffs[1:]:
        level_ent = sum(_band_entropy(b) for b in level_detail)
        entropies_per_level.append(level_ent)

    if len(entropies_per_level) < 2:
        return 0.0

    coarse = entropies_per_level[-1]
    fine = entropies_per_level[0]
    return float(fine / (coarse + 1e-12))


def _zernike_radial(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
    """Radial polynomial R_n^m_abs(rho) for Zernike basis."""
    from math import factorial
    result = np.zeros(len(rho), dtype=np.float64)
    half = (n - m_abs) // 2
    for s in range(half + 1):
        num = factorial(n - s)
        den = factorial(s) * factorial((n + m_abs) // 2 - s) * factorial(half - s)
        result += ((-1) ** s) * (num / den) * rho ** (n - 2 * s)
    return result


def _build_zernike_matrix(rho: np.ndarray, theta: np.ndarray, max_order: int) -> np.ndarray:
    """Build (N_pixels, N_modes) Zernike basis matrix up to radial order max_order.

    Uses OSA/ANSI normalisation so that each mode has unit RMS on the unit disk.
    Modes are ordered as: (0,0), (1,-1), (1,1), (2,-2), (2,0), (2,2), ...
    """
    cols = []
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2 if n > 0 else 1):
            m_abs = abs(m)
            R = _zernike_radial(n, m_abs, rho)
            if m == 0:
                norm = np.sqrt(n + 1)
                cols.append(norm * R)
            elif m > 0:
                norm = np.sqrt(2.0 * (n + 1))
                cols.append(norm * R * np.cos(m_abs * theta))
            else:
                norm = np.sqrt(2.0 * (n + 1))
                cols.append(norm * R * np.sin(m_abs * theta))
    return np.column_stack(cols) if len(cols) > 1 else np.array(cols).T


def decompose_psf_zernike(img: np.ndarray, star_positions,
                           cutout_radius: int = None,
                           max_order: int = None) -> Dict:
    """Decompose star PSFs into Zernike modes; return per-mode coefficients and RMS.

    Extracts a cutout centred on each bright star, fits the background-subtracted
    PSF to an orthonormal Zernike basis (up to radial order ``max_order``), and
    returns the median coefficients across stars.

    Optical aberration RMS = sqrt(sum of squared coefficients for modes Z4+),
    i.e. ignoring piston (Z1), tip (Z2), and tilt (Z3).

    Returns a dict with keys:
        'zernike_coeffs'  : list of median coefficients (one per mode)
        'zernike_rms'     : optical aberration RMS (modes 4+ only)
        'zernike_defocus' : Z(2,0) defocus coefficient
        'zernike_astig'   : sqrt(Z(2,-2)²+Z(2,2)²) astigmatism magnitude
        'zernike_coma'    : sqrt(Z(3,-1)²+Z(3,1)²) coma magnitude
        'zernike_spherical': |Z(4,0)| spherical aberration
    """
    if cutout_radius is None:
        from src.models import Config
        cutout_radius = Config.ZERNIKE_CUTOUT_RADIUS
    if max_order is None:
        from src.models import Config
        max_order = Config.ZERNIKE_MAX_ORDER
    max_stars = Config.ZERNIKE_MAX_STARS if 'Config' in dir() else 15

    empty = {
        'zernike_coeffs': [], 'zernike_rms': 0.0,
        'zernike_defocus': 0.0, 'zernike_astig': 0.0,
        'zernike_coma': 0.0, 'zernike_spherical': 0.0,
    }

    if star_positions is None or len(star_positions) == 0:
        return empty

    H, W = img.shape[:2]
    lum = (img if img.ndim == 2
           else 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])

    try:
        sorted_idx = list(np.argsort(star_positions['flux'])[::-1])
    except (KeyError, TypeError):
        sorted_idx = list(range(len(star_positions)))

    # Build pixel grid for the cutout once (reused for every star)
    d = cutout_radius
    size = 2 * d + 1
    yy, xx = np.mgrid[-d:d + 1, -d:d + 1].astype(np.float64)
    r = np.sqrt(xx ** 2 + yy ** 2)
    unit_mask = (r <= d)
    rho_flat = (r[unit_mask] / d).ravel()         # normalised radius in [0,1]
    theta_flat = np.arctan2(yy[unit_mask], xx[unit_mask]).ravel()
    Z = _build_zernike_matrix(rho_flat, theta_flat, max_order)

    all_coeffs = []
    for idx in sorted_idx[:max_stars]:
        star = star_positions[idx]
        yc = int(round(float(star['ycentroid'])))
        xc = int(round(float(star['xcentroid'])))
        if yc < d or yc >= H - d or xc < d or xc >= W - d:
            continue

        cut = lum[yc - d:yc + d + 1, xc - d:xc + d + 1].astype(np.float64)
        bg = float(np.percentile(cut, 25))
        cut_sub = np.maximum(cut - bg, 0.0)
        peak = cut_sub.max()
        if peak < 10.0:
            continue
        # Skip saturated stars (flat top)
        if int(np.sum(cut_sub >= peak * 0.99)) > 6:
            continue

        psf_flat = cut_sub[unit_mask].ravel()
        total = psf_flat.sum()
        if total < 1e-9:
            continue
        psf_norm = psf_flat / total

        try:
            coeffs, _, _, _ = np.linalg.lstsq(Z, psf_norm, rcond=None)
            all_coeffs.append(coeffs)
        except np.linalg.LinAlgError:
            continue

    if not all_coeffs:
        return empty

    med_coeffs = np.median(np.array(all_coeffs), axis=0)

    # Mode index map: (0,0)=0, (1,-1)=1, (1,1)=2, (2,-2)=3, (2,0)=4, (2,2)=5,
    # (3,-3)=6, (3,-1)=7, (3,1)=8, (3,3)=9, (4,-4)=10, (4,-2)=11, (4,0)=12, ...
    # Aberration RMS from mode index 3 onward (skip piston + tip + tilt at 0,1,2)
    aberr_rms = float(np.sqrt(np.sum(med_coeffs[3:] ** 2))) if len(med_coeffs) > 3 else 0.0

    # Extract named aberrations by mode index
    defocus = float(med_coeffs[4]) if len(med_coeffs) > 4 else 0.0
    astig = (float(np.sqrt(med_coeffs[3] ** 2 + med_coeffs[5] ** 2))
             if len(med_coeffs) > 5 else 0.0)
    coma = (float(np.sqrt(med_coeffs[7] ** 2 + med_coeffs[8] ** 2))
            if len(med_coeffs) > 8 else 0.0)
    spherical = float(abs(med_coeffs[12])) if len(med_coeffs) > 12 else 0.0

    return {
        'zernike_coeffs': med_coeffs.tolist(),
        'zernike_rms': aberr_rms,
        'zernike_defocus': defocus,
        'zernike_astig': astig,
        'zernike_coma': coma,
        'zernike_spherical': spherical,
    }


def compute_quality_metrics(img: np.ndarray, quick: bool = False,
                            advanced_metrics: bool = True) -> Dict:
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

    # Star detection on a 2x-downsampled image — DAOStarFinder runs an FFT
    # matched filter over the whole frame; half-res cuts that cost by 4x.
    # Coordinates are scaled back to full-res after detection.
    _star_ds = 2 if _min_dim >= 1500 else 1
    img_s_stars = img[::_star_ds, ::_star_ds] if _star_ds > 1 else img

    # All percentiles in one pass — p50 replaces a separate np.median call.
    p01, p50, p75, p95, p99 = np.percentile(img_s, [1, 50, 75, 95, 99])
    brightness = float(p50)  # median == p50, no extra traversal

    mean = float(np.mean(img_s))
    contrast = float(np.std(img_s))

    # MAD-based background/noise — single pass, no iterative sigma clipping.
    # 1.4826 * MAD is an unbiased estimator of Gaussian sigma.
    _bg_median = float(np.median(img_s))
    _bg_mad = float(np.median(np.abs(img_s - _bg_median)))
    background = _bg_median
    noise = max(1.4826 * _bg_mad, 1e-6)
    snr = (p95 - background) / noise

    # Star detection
    star_count = 0
    star_snr = 0.0
    sources_s = None
    fwhm = 0.0

    if not quick:
        # Try SEP first (C backend, ~5-10x faster than DAOStarFinder).
        sources_s = _sep_detect_stars(img_s_stars, noise)
        if sources_s is not None and len(sources_s) > 0:
            star_count = len(sources_s)
            star_snr = float(np.median(sources_s['peak'])) / (noise + 1e-12)
        else:
            # DAOStarFinder fallback.
            _ensure_photutils()
            if DAOStarFinder is not None:
                try:
                    threshold = 5.0 * noise
                    bg_sub = img_s_stars - background
                    sources_s = _detect_stars_multi_fwhm(bg_sub, threshold)
                    if sources_s is not None and len(sources_s) > 0:
                        star_count = len(sources_s)
                        star_snr = float(np.median(sources_s['peak'])) / (noise + 1e-12)
                except Exception as e:
                    logging.debug(f"DAOStarFinder failed: {type(e).__name__}: {e}")
                    sources_s = None

        # Fallback: local-maxima detection when photutils/SEP are unavailable.
        if star_count == 0 and maximum_filter is not None:
            try:
                threshold = background + 5.0 * noise
                local_max = maximum_filter(img_s_stars, size=11)
                detected_peaks = (img_s_stars == local_max) & (img_s_stars > threshold)
                peak_ys, peak_xs = np.nonzero(detected_peaks)
                peak_vals = img_s_stars[peak_ys, peak_xs]
                order = np.argsort(peak_vals)[::-1]
                peak_ys = peak_ys[order]
                peak_xs = peak_xs[order]
                star_count = min(len(peak_ys), 500)
                if star_count > 0:
                    star_snr = float(np.median(peak_vals)) / (noise + 1e-12)
                    sources_s = np.zeros(star_count, dtype=[
                        ('ycentroid', np.float32),
                        ('xcentroid', np.float32),
                        ('peak', np.float32),
                    ])
                    sources_s['ycentroid'] = peak_ys[:star_count].astype(np.float32)
                    sources_s['xcentroid'] = peak_xs[:star_count].astype(np.float32)
                    sources_s['peak'] = peak_vals[order[:star_count]].astype(np.float32)
            except Exception:
                star_count = 0

        # FWHM — measured in downsampled space, scaled back to full-res pixels.
        if star_count > 0 and sources_s is not None:
            fwhm = measure_fwhm(img_s_stars, sources_s) * _star_ds

    # Strehl proxy and atmospheric dispersion — expensive per-star cutout work;
    # only run when the caller opts in via advanced_metrics.
    strehl = 0.0
    dispersion_px = 0.0
    if advanced_metrics and not quick and star_count > 0 and sources_s is not None:
        fwhm_detect = fwhm / _star_ds if _star_ds > 1 else fwhm  # downsampled-space FWHM
        try:
            strehl = estimate_strehl_ratio(img_s_stars, sources_s, fwhm=fwhm_detect)
        except Exception:
            strehl = 0.0
        if img_s_stars.ndim == 3:
            try:
                dispersion_px = measure_atmospheric_dispersion(img_s_stars, sources_s)
            except Exception:
                dispersion_px = 0.0

    # Laplacian sharpness — img_s is already float32, no cast needed.
    sharpness = 0.0
    if laplace is not None:
        try:
            sharpness = float(np.var(laplace(img_s)))
        except Exception:
            sharpness = 0.0

    # Brenner gradient sharpness — noise-robust complement to Laplacian.
    brenner = 0.0
    try:
        brenner = compute_brenner_sharpness(img_s)
    except Exception:
        brenner = 0.0

    # Wavelet multi-scale entropy ratio — captures seeing quality independently of SNR.
    wavelet_entropy_ratio = 0.0
    if not quick:
        try:
            wavelet_entropy_ratio = compute_multiscale_entropy(img_s)
        except Exception:
            wavelet_entropy_ratio = 0.0

    # PSF anisotropy — ellipticity and PA scatter from SEP semi-axes.
    psf_ellipticity = 0.0
    psf_pa_scatter = 0.0
    psf_anisotropy_type = 'isotropic'
    if not quick and sources_s is not None and len(sources_s) > 0:
        try:
            psf_ellipticity, psf_pa_scatter, psf_anisotropy_type = measure_psf_anisotropy(sources_s)
        except Exception:
            pass

    # Zernike PSF decomposition — optical aberration fingerprint.
    zernike_result: Dict = {}
    if advanced_metrics and not quick and star_count > 0 and sources_s is not None:
        try:
            zernike_result = decompose_psf_zernike(img_s_stars, sources_s)
        except Exception:
            zernike_result = {}

    # Composite quality score (0–100 range).
    # Normalisation targets realistic single-frame values:
    #   SNR ~2 = good sky-limited frame; FWHM ~4px = good seeing (gentle penalty above that).
    if quick:
        # SNR-only score for the initial quality gate (no star detection overhead)
        score = min(max(snr / 2.0, 0.01), 1.0) * 100.0
    else:
        star_factor = min(star_count / 50.0, 1.0) if star_count > 0 else 0.01
        snr_factor = min(snr / 2.0, 1.0) if snr > 0 else 0.01
        fwhm_factor = (
            max(0.1, 1.0 / (1.0 + max(0.0, fwhm - 2.0) ** 2 * 0.02))
            if fwhm > 0 else 1.0
        )
        score = snr_factor * star_factor * fwhm_factor * 100.0

    return {
        'brightness': brightness,
        'mean': mean,
        'contrast': contrast,
        'snr': float(snr),
        'star_count': star_count,
        'star_snr': float(star_snr),
        'sharpness': sharpness,
        'brenner': float(brenner),
        'wavelet_entropy_ratio': float(wavelet_entropy_ratio),
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
        'strehl': float(strehl),
        'dispersion_px': float(dispersion_px),
        'psf_ellipticity': float(psf_ellipticity),
        'psf_pa_scatter': float(psf_pa_scatter),
        'psf_anisotropy_type': psf_anisotropy_type,
        'zernike_rms': float(zernike_result.get('zernike_rms', 0.0)),
        'zernike_defocus': float(zernike_result.get('zernike_defocus', 0.0)),
        'zernike_astig': float(zernike_result.get('zernike_astig', 0.0)),
        'zernike_coma': float(zernike_result.get('zernike_coma', 0.0)),
        'zernike_spherical': float(zernike_result.get('zernike_spherical', 0.0)),
        '_star_sources': sources_s,
    }