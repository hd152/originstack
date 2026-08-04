"""PSF estimation and Richardson-Lucy deconvolution."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import signal as _scipy_signal

from src.models import Config

try:
    from scipy.optimize import curve_fit
    HAS_CURVE_FIT = True
except Exception:
    HAS_CURVE_FIT = False

def estimate_psf(img: np.ndarray, star_positions,
                 cutout_radius: int = None, psf_size: int = None,
                 model: str = 'moffat') -> Tuple[Optional[np.ndarray], float]:
    """Estimate the point spread function from star cutouts via profile fitting.

    Fits a 2D Moffat (default) or Gaussian model to bright, unsaturated stars
    and builds a normalized PSF kernel from the median fit parameters.

    Returns (psf_kernel, fwhm_pixels).  Returns (None, 0.0) if too few stars
    are successfully fit.
    """
    if star_positions is None or len(star_positions) < Config.RL_PSF_MIN_STARS:
        logging.warning("Too few stars for PSF estimation "
                        f"({0 if star_positions is None else len(star_positions)} < {Config.RL_PSF_MIN_STARS})")
        return None, 0.0

    if psf_size is None:
        psf_size = Config.RL_PSF_SIZE

    if not HAS_CURVE_FIT:
        logging.warning("scipy.optimize.curve_fit unavailable; cannot estimate PSF")
        return None, 0.0

    if cutout_radius is None:
        cutout_radius = Config.RL_PSF_CUTOUT_RADIUS
    if psf_size is None:
        psf_size = Config.RL_PSF_SIZE

    H, W = img.shape[:2]
    lum = img if img.ndim == 2 else (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])
    img_max = float(lum.max())

    # Sort by flux (brightest first), skip saturated
    try:
        sorted_idx = np.argsort(star_positions['flux'])[::-1]
    except (KeyError, TypeError):
        sorted_idx = range(len(star_positions))

    # 2D Moffat: I(x,y) = A * (1 + ((x-x0)^2 + (y-y0)^2) / alpha^2)^(-beta) + bg
    def moffat_2d(coords, amplitude, x0, y0, alpha, beta, background):
        y, x = coords
        r2 = (x - x0) ** 2 + (y - y0) ** 2
        return (amplitude * (1.0 + r2 / (alpha ** 2)) ** (-beta) + background).ravel()

    # 2D Gaussian: used as fallback when model='gaussian'
    def gaussian_2d(coords, amplitude, x0, y0, sigma, background):
        y, x = coords
        r2 = (x - x0) ** 2 + (y - y0) ** 2
        return (amplitude * np.exp(-r2 / (2.0 * sigma ** 2)) + background).ravel()

    alphas = []
    betas = []
    sigmas = []
    n_tried = 0

    for idx in sorted_idx:
        if n_tried >= Config.RL_PSF_MAX_STARS:
            break
        star = star_positions[idx]
        yc = int(round(float(star['ycentroid'])))
        xc = int(round(float(star['xcentroid'])))
        if (yc < cutout_radius or yc >= H - cutout_radius or
                xc < cutout_radius or xc >= W - cutout_radius):
            continue

        cutout = lum[yc - cutout_radius:yc + cutout_radius + 1,
                     xc - cutout_radius:xc + cutout_radius + 1].astype(np.float64)
        peak = float(np.max(cutout))
        bg = float(np.percentile(cutout, 25))

        # Skip saturated stars (flat-topped peak = clipped at ADC max) or low-contrast
        # A star is "saturated" if multiple pixels share the exact peak value (flat top)
        n_at_peak = np.sum(cutout >= peak - 1e-6 * max(peak, 1.0))
        if n_at_peak > 4 or (peak - bg) < np.std(cutout) * 2.0:
            continue
        n_tried += 1

        sz = cutout.shape[0]
        yg, xg = np.mgrid[0:sz, 0:sz]
        center = sz / 2.0

        try:
            if model == 'moffat':
                p0 = [peak - bg, center, center, 2.0, 3.0, bg]
                bounds = ([0, center - 3, center - 3, 0.5, 1.0, 0],
                          [peak * 2, center + 3, center + 3, 20.0, 10.0, peak])
                popt, _ = curve_fit(moffat_2d, (yg, xg), cutout.ravel(),
                                    p0=p0, bounds=bounds, maxfev=2000)
                alphas.append(popt[3])
                betas.append(popt[4])
            else:
                p0 = [peak - bg, center, center, 2.0, bg]
                bounds = ([0, center - 3, center - 3, 0.3, 0],
                          [peak * 2, center + 3, center + 3, 20.0, peak])
                popt, _ = curve_fit(gaussian_2d, (yg, xg), cutout.ravel(),
                                    p0=p0, bounds=bounds, maxfev=2000)
                sigmas.append(popt[3])
        except (RuntimeError, ValueError):
            continue

    # Require minimum successful fits
    if model == 'moffat':
        if len(alphas) < Config.RL_PSF_MIN_STARS:
            logging.warning(f"PSF estimation: only {len(alphas)} Moffat fits succeeded "
                            f"(need {Config.RL_PSF_MIN_STARS})")
            return None, 0.0
        med_alpha = float(np.median(alphas))
        med_beta = float(np.median(betas))
        fwhm = 2.0 * med_alpha * np.sqrt(2.0 ** (1.0 / med_beta) - 1.0)
        # Generate Moffat PSF kernel
        half = psf_size // 2
        yg, xg = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
        r2 = xg ** 2 + yg ** 2
        psf = (1.0 + r2 / (med_alpha ** 2)) ** (-med_beta)
    else:
        if len(sigmas) < Config.RL_PSF_MIN_STARS:
            logging.warning(f"PSF estimation: only {len(sigmas)} Gaussian fits succeeded "
                            f"(need {Config.RL_PSF_MIN_STARS})")
            return None, 0.0
        med_sigma = float(np.median(sigmas))
        fwhm = 2.355 * med_sigma
        half = psf_size // 2
        yg, xg = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
        r2 = xg ** 2 + yg ** 2
        psf = np.exp(-r2 / (2.0 * med_sigma ** 2))

    psf /= psf.sum()
    logging.info(f"PSF estimated: model={model}, FWHM={fwhm:.2f}px, "
                 f"from {len(alphas) if model == 'moffat' else len(sigmas)} stars")
    return psf.astype(np.float64), float(fwhm)


def make_synthetic_psf(fwhm: float, psf_size: int = None, model: str = 'gaussian') -> np.ndarray:
    """Generate a synthetic PSF kernel from a given FWHM.

    Used as fallback when star-based PSF estimation is unavailable (too few
    stars, or scipy.optimize.curve_fit absent). Returns a normalized 2D kernel.
    """
    if psf_size is None:
        psf_size = Config.RL_PSF_SIZE
    half = psf_size // 2
    yg, xg = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    r2 = xg ** 2 + yg ** 2
    if model == 'moffat':
        alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / 3.0) - 1.0))  # assume beta=3
        psf = (1.0 + r2 / (alpha ** 2)) ** (-3.0)
    else:
        sigma = fwhm / 2.355
        psf = np.exp(-r2 / (2.0 * sigma ** 2))
    psf /= psf.sum()
    return psf


def estimate_psf_blind(img: np.ndarray, star_positions,
                        psf_size: int = None,
                        iterations: int = None) -> Tuple[Optional[np.ndarray], float]:
    """Empirical (model-free) PSF estimation by stacking bright star cutouts.

    Instead of fitting a parametric model, this function extracts cutouts
    around bright unsaturated stars, background-subtracts each, normalises
    them to unit integral, and returns the median stack.  The result is the
    *actual* on-sky PSF shape, including any asymmetry, coma, or tracking
    errors that a Gaussian/Moffat model cannot capture.

    The ``iterations`` parameter optionally runs Richardson-Lucy blind
    deconvolution update steps on top of the median-stack PSF to sharpen
    the estimate.  Each step refines the PSF by computing:

        psf_new = psf * correlate(img / (psf * img), img_reversed)

    Args:
        img:            Float32 stacked image (H, W, 3).
        star_positions: Source table from detect_stars_auto.
        psf_size:       Output kernel side length (default Config.RL_PSF_SIZE).
        iterations:     RL blind update iterations (0 = median stack only).

    Returns:
        (psf_kernel, fwhm_pixels) — (None, 0.0) if estimation fails.
    """
    if star_positions is None or len(star_positions) < Config.RL_PSF_MIN_STARS:
        logging.warning("Blind PSF: too few stars (%d)", 0 if star_positions is None else len(star_positions))
        return None, 0.0

    if psf_size is None:
        psf_size = Config.RL_PSF_SIZE
    if iterations is None:
        iterations = Config.BLIND_PSF_ITERATIONS

    H, W = img.shape[:2]
    lum = (img if img.ndim == 2
           else 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])

    half = psf_size // 2
    cutout_r = half + 4   # slightly larger than psf_size for clean border

    try:
        sorted_idx = np.argsort(star_positions['flux'])[::-1]
    except (KeyError, TypeError):
        sorted_idx = range(len(star_positions))

    stacked_cutouts = []
    for idx in list(sorted_idx)[:Config.RL_PSF_MAX_STARS]:
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
        signal = peak - bg
        if signal < bg * 0.5 or signal < 10.0:
            continue

        # Skip saturated (flat-topped) stars
        if int(np.sum(cutout >= peak - 1e-6 * max(peak, 1.0))) > 4:
            continue

        cutout = cutout - bg
        cutout = np.maximum(cutout, 0.0)
        total = cutout.sum()
        if total < 1e-12:
            continue
        cutout /= total

        # Centre-crop to psf_size
        c = cutout_r
        psf_cutout = cutout[c - half:c + half + 1, c - half:c + half + 1]
        if psf_cutout.shape != (psf_size, psf_size):
            continue
        stacked_cutouts.append(psf_cutout)

    if len(stacked_cutouts) < Config.RL_PSF_MIN_STARS:
        logging.warning("Blind PSF: only %d usable stars (need %d)",
                        len(stacked_cutouts), Config.RL_PSF_MIN_STARS)
        return None, 0.0

    psf = np.median(np.array(stacked_cutouts), axis=0)
    psf = np.maximum(psf, 0.0)
    psf /= psf.sum()

    # Optional blind RL refinement (pure scipy.signal, no skimage involved)
    if iterations > 0:
        from scipy.signal import fftconvolve
        lum_pos = lum - lum.min() + 1e-6
        psf_est = psf.copy()
        for _ in range(iterations):
            conv = fftconvolve(lum_pos, psf_est, mode='same')
            ratio = lum_pos / (conv + 1e-12)
            update = fftconvolve(ratio, psf_est[::-1, ::-1], mode='same')
            # PSF update: cross-correlate ratio with lum
            psf_update = fftconvolve(ratio[::-1, ::-1], lum_pos, mode='same')
            # Trim to psf_size
            cy, cx = np.unravel_index(np.argmax(psf_update), psf_update.shape)
            y0 = max(0, cy - half)
            x0 = max(0, cx - half)
            patch = psf_update[y0:y0 + psf_size, x0:x0 + psf_size]
            if patch.shape == (psf_size, psf_size):
                patch = np.maximum(patch, 0.0)
                s = patch.sum()
                if s > 1e-12:
                    psf_est = 0.7 * psf_est + 0.3 * patch / s

        psf = np.maximum(psf_est, 0.0)
        psf /= psf.sum()

    # Radial apodization — force the kernel to zero at its edge.
    # An empirical PSF from stacked star cutouts retains non-zero energy in the
    # square corners (star wings, residual noise), and blind RL refinement can
    # inject blocky off-centre structure. FFT deconvolution with such a
    # hard-edged square kernel produces square ringing ("boxes") around every
    # point source. A Tukey (flat-core, cosine-taper) radial window keeps the
    # PSF core/wings intact while tapering the outer edge smoothly to zero,
    # yielding circular support and eliminating the box artefacts.
    yy, xx = np.mgrid[0:psf_size, 0:psf_size]
    r = np.sqrt((yy - half) ** 2 + (xx - half) ** 2) / float(max(half, 1))
    taper_start = 0.6                       # inner 60% radius: unwindowed
    w = np.ones_like(r)
    edge = r >= 1.0
    taper = (r >= taper_start) & (~edge)
    w[taper] = 0.5 * (1.0 + np.cos(np.pi * (r[taper] - taper_start)
                                   / (1.0 - taper_start)))
    w[edge] = 0.0
    psf = psf * w
    _s = psf.sum()
    if _s > 1e-12:
        psf /= _s

    # Estimate FWHM from the median PSF
    half_max = psf.max() * 0.5
    above = int(np.sum(psf > half_max))
    fwhm = 2.0 * np.sqrt(above / np.pi) if above > 0 else float(psf_size // 4)

    logging.info("Blind PSF: stacked %d stars, FWHM≈%.2f px", len(stacked_cutouts), fwhm)
    return psf.astype(np.float64), fwhm


def tv_regularized_deconvolve(img: np.ndarray, psf: np.ndarray,
                               iterations: int = None,
                               lambda_tv: float = None,
                               star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Total Variation regularized deconvolution via gradient descent.

    Minimises the Tikhonov-TV functional:

        E(x) = ½ ‖H x − y‖² + λ ‖∇x‖_TV

    where H is convolution with the PSF, y is the observed image, and the
    TV norm ‖∇x‖_TV = Σ √(|∂x/∂r|² + |∂x/∂c|² + ε) promotes piecewise-
    smooth solutions while allowing sharp edges (galaxy arms, nebula filaments)
    — unlike Tikhonov-L2 regularisation which blurs them.

    Each gradient descent step:
      1. Data gradient:  ∇E_data = H^T (H x − y)   (correlation with PSF)
      2. TV gradient:    ∇E_tv   = −div( ∇x / |∇x|_ε )
      3. Update:         x ← x − step · (∇E_data + λ ∇E_tv)

    Operates on luminance only (YCbCr split) to avoid colour artefacts.

    Args:
        img:        Float32 stacked image (H, W, 3).
        psf:        Normalised 2-D PSF kernel.
        iterations: Gradient descent steps (default Config.TV_ITERATIONS = 50).
        lambda_tv:  TV regularisation strength (default Config.TV_LAMBDA = 0.02).
                    Larger → smoother; smaller → sharper but noisier.
        star_mask:  Optional float mask (1 = star core) blended from original
                    to avoid ringing artefacts on bright star cores.

    Returns:
        Deconvolved float32 image (H, W, 3).
    """
    from scipy.signal import fftconvolve

    if iterations is None:
        iterations = Config.TV_ITERATIONS
    if lambda_tv is None:
        lambda_tv = Config.TV_LAMBDA

    src = img.astype(np.float64)
    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    psf_d = psf.astype(np.float64)
    psf_d /= psf_d.sum()
    psf_flip = psf_d[::-1, ::-1]

    # Pedestal (RL requires strictly positive input)
    y_min = float(Y.min())
    pedestal = max(-y_min + 1e-6, 1e-6)
    y = Y + pedestal

    # Step size: spectral norm of H^T H ≈ max(|FFT(psf)|²) = 1 (normalised PSF)
    step = 0.9 / (1.0 + lambda_tv * 8.0)

    x = y.copy()
    eps_tv = 1e-6   # smoothing constant for TV gradient

    for _ in range(iterations):
        # Data gradient
        Hx = fftconvolve(x, psf_d, mode='same')
        data_grad = fftconvolve(Hx - y, psf_flip, mode='same')

        # TV gradient (anisotropic: discrete divergence of normalised gradient)
        dx = np.roll(x, -1, axis=1) - x
        dy = np.roll(x, -1, axis=0) - x
        norm_grad = np.sqrt(dx ** 2 + dy ** 2 + eps_tv)
        Px = dx / norm_grad
        Py = dy / norm_grad
        tv_grad = (Px - np.roll(Px, 1, axis=1)
                   + Py - np.roll(Py, 1, axis=0))

        x = x - step * (data_grad + lambda_tv * tv_grad)
        np.clip(x, 1e-12, None, out=x)

    Y_d = x - pedestal

    if star_mask is not None:
        Y_d = Y_d * (1.0 - star_mask) + Y * star_mask

    R = Y_d + 1.40200 * Cr
    G = Y_d - 0.34414 * Cb - 0.71414 * Cr
    B = Y_d + 1.77200 * Cb
    result = np.stack([R, G, B], axis=2)
    np.clip(result, 0.0, None, out=result)
    return result.astype(np.float32)


def _rl_deconvolve_xp(image, psf, iterations, xp, xsignal):
    """Richardson-Lucy iteration on an arbitrary array backend (numpy or cupy).

    Mirrors skimage.restoration.richardson_lucy (clip=False): start from a flat
    0.5 estimate and iterate im *= conv(image / conv(im, psf), psf_mirror) using
    FFT convolutions. With xp=cupy/xsignal=cupyx this runs on the GPU.
    """
    image = xp.asarray(image, dtype=xp.float32)
    psf = xp.asarray(psf, dtype=xp.float32)
    im_deconv = xp.full(image.shape, 0.5, dtype=xp.float32)
    psf_mirror = psf[::-1, ::-1]
    for _ in range(int(iterations)):
        conv = xsignal.fftconvolve(im_deconv, psf, mode='same')
        # FFT convolution of nonnegative inputs can still ring to ~0 (or
        # slightly negative) at float32 precision; an unguarded division
        # here blows one such pixel up to Inf/NaN, which fftconvolve's
        # global support then spreads to the entire frame on the very next
        # iteration. Floor the denominator instead of dividing raw.
        conv = xp.clip(conv, 1e-6, None)
        relative_blur = image / conv
        im_deconv = im_deconv * xsignal.fftconvolve(relative_blur, psf_mirror, mode='same')
        im_deconv = xp.clip(im_deconv, 0.0, None)
    return im_deconv


def richardson_lucy_deconvolve(img: np.ndarray, psf: np.ndarray,
                                iterations: int = None,
                                star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Richardson-Lucy deconvolution for sharpening the stacked image.

    Operates on luminance only (YCbCr decomposition) to avoid colour
    artifacts, then recombines with the original chrominance channels.

    star_mask (float [0,1], 1=star core): if provided, the deconvolved
    result is blended back with the original at star positions to prevent
    ringing artifacts on bright star cores.

    Runs on the GPU (cupy FFT convolution) when --use-gpu is active; otherwise
    runs the same FFT-based iteration on the numpy/scipy CPU backend.
    """
    from src.gpu_context import get_gpu
    _gpu = get_gpu()
    _use_gpu = _gpu.active and _gpu.xsignal is not None and hasattr(_gpu.xsignal, 'fftconvolve')

    if iterations is None:
        iterations = Config.RL_DEFAULT_ITERATIONS

    src = img.astype(np.float64)

    # Work in YCbCr — deconvolve luminance only
    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    # RL requires strictly positive input; add a small pedestal
    y_min = float(Y.min())
    pedestal = max(-y_min + 1e-6, 1e-6)
    Y_pos = Y + pedestal

    if _use_gpu:
        try:
            Y_deconv = _gpu.to_host(
                _rl_deconvolve_xp(Y_pos, psf, iterations, _gpu.xp, _gpu.xsignal))
            logging.info("Richardson-Lucy deconvolution ran on GPU")
        except Exception as exc:
            if _gpu.is_oom(exc):
                # Permanent fallback (matches debayer.py's GPU call sites),
                # not just free_pool(): a capacity OOM won't resolve itself
                # by freeing the pool once, and richardson_lucy_svpsf calls
                # this once per tile, so leaving the GPU "active" here would
                # just repeat the same OOM on every remaining tile.
                _gpu.disable()
            logging.debug("GPU RL failed (%s); falling back to CPU", exc)
            Y_deconv = _rl_deconvolve_xp(Y_pos, psf, iterations, np, _scipy_signal)
    else:
        Y_deconv = _rl_deconvolve_xp(Y_pos, psf, iterations, np, _scipy_signal)
    Y_deconv = np.asarray(Y_deconv, dtype=np.float64) - pedestal

    # Star protection: blend original at star cores
    if star_mask is not None:
        Y_deconv = Y_deconv * (1.0 - star_mask) + Y * star_mask

    # YCbCr -> RGB
    R = Y_deconv + 1.40200 * Cr
    G = Y_deconv - 0.34414 * Cb - 0.71414 * Cr
    B = Y_deconv + 1.77200 * Cb
    result = np.stack([R, G, B], axis=2)
    np.clip(result, 0.0, None, out=result)

    return result.astype(np.float32)


def _shift_sources(sources, dx: float, dy: float):
    """Return a copy of a star catalogue with centroids shifted by (-dx, -dy)
    (into a sub-image whose top-left is at (dx, dy) in the full frame)."""
    out = np.array(sources).copy()
    out['xcentroid'] = np.asarray(sources['xcentroid'], dtype=np.float64) - dx
    out['ycentroid'] = np.asarray(sources['ycentroid'], dtype=np.float64) - dy
    return out


def _feather_window(h: int, w: int, my: int, mx: int) -> np.ndarray:
    """Raised-cosine blend weight: 1.0 across the core, ramping to ~0 over the
    ``my``/``mx``-pixel margins so overlapping tiles sum seamlessly."""
    def ramp(n: int, m: int) -> np.ndarray:
        v = np.ones(n, dtype=np.float64)
        if m > 0:
            t = np.linspace(0.0, np.pi, 2 * m)
            edge = 0.5 * (1.0 - np.cos(t[:m]))          # 0 -> 1 over m px
            v[:m] = edge
            v[-m:] = edge[::-1]
        return v
    wy = ramp(h, min(my, h // 2))
    wx = ramp(w, min(mx, w // 2))
    return np.outer(wy, wx)


def richardson_lucy_svpsf(img: np.ndarray, sources, iterations: int = 15,
                          n_tiles: int = 3, model: str = 'moffat',
                          overlap: float = 0.35, star_mask: Optional[np.ndarray] = None,
                          verbose: bool = False) -> np.ndarray:
    """Spatially-variant Richardson-Lucy deconvolution.

    A single global PSF over-sharpens the frame centre and rings the corners
    when the optics' PSF varies across the field (off-axis aberration, tilt,
    field curvature). This fits a *separate* PSF from the local stars of each
    tile in an ``n_tiles x n_tiles`` grid, deconvolves each tile's luminance
    with its own PSF, and blends the overlapping tiles with a raised-cosine
    window so there is no seam. Chrominance is preserved (luminance-only
    deconvolution, YCbCr recombination), matching ``richardson_lucy_deconvolve``.

    Tiles with too few stars fall back to the global PSF, so the result is never
    worse than the single-PSF path. Returns a new (H, W, 3) float32 image.
    """
    from src.gpu_context import get_gpu
    _gpu = get_gpu()
    _use_gpu = _gpu.active and _gpu.xsignal is not None and hasattr(_gpu.xsignal, 'fftconvolve')
    if sources is None or len(sources) == 0:
        return img

    src = img.astype(np.float64)
    Y = 0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr = 0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]
    H, W = Y.shape

    # Global PSF as the per-tile fallback; abort if even that fails.
    global_psf, _gf = estimate_psf(img, sources, model=model)
    if global_psf is None:
        logging.info("SV-PSF: global PSF estimation failed — no deconvolution")
        return img

    from scipy import signal as _sig
    xs_all = np.asarray(sources['xcentroid'], dtype=np.float64)
    ys_all = np.asarray(sources['ycentroid'], dtype=np.float64)

    th = int(np.ceil(H / n_tiles))
    tw = int(np.ceil(W / n_tiles))
    my = int(overlap * th)
    mx = int(overlap * tw)

    Y_acc = np.zeros((H, W), dtype=np.float64)
    W_acc = np.zeros((H, W), dtype=np.float64)
    n_local = 0
    for ty in range(n_tiles):
        for tx in range(n_tiles):
            cy0, cy1 = ty * th, min((ty + 1) * th, H)
            cx0, cx1 = tx * tw, min((tx + 1) * tw, W)
            # Expanded (with-margin) region for PSF context + feather.
            ey0, ey1 = max(0, cy0 - my), min(H, cy1 + my)
            ex0, ex1 = max(0, cx0 - mx), min(W, cx1 + mx)

            in_tile = ((xs_all >= ex0) & (xs_all < ex1)
                       & (ys_all >= ey0) & (ys_all < ey1))
            psf = global_psf
            if int(in_tile.sum()) >= Config.RL_PSF_MIN_STARS:
                sub_src = _shift_sources(sources[in_tile], ex0, ey0)
                sub_img = Y[ey0:ey1, ex0:ex1]
                local_psf, _lf = estimate_psf(sub_img, sub_src, model=model)
                if local_psf is not None:
                    psf = local_psf
                    n_local += 1

            tileY = Y[ey0:ey1, ex0:ex1]
            pedestal = max(-float(tileY.min()) + 1e-6, 1e-6)
            tileY_pos = tileY + pedestal
            if _use_gpu:
                try:
                    dec = _gpu.to_host(_rl_deconvolve_xp(tileY_pos, psf, iterations,
                                                         _gpu.xp, _gpu.xsignal))
                except Exception:
                    dec = _rl_deconvolve_xp(tileY_pos, psf, iterations, np, _sig)
            else:
                dec = _rl_deconvolve_xp(tileY_pos, psf, iterations, np, _sig)
            dec = np.asarray(dec, dtype=np.float64) - pedestal

            wwin = _feather_window(ey1 - ey0, ex1 - ex0, my, mx)
            Y_acc[ey0:ey1, ex0:ex1] += dec * wwin
            W_acc[ey0:ey1, ex0:ex1] += wwin

    Y_deconv = Y_acc / np.maximum(W_acc, 1e-9)
    if star_mask is not None:
        Y_deconv = Y_deconv * (1.0 - star_mask) + Y * star_mask
    if verbose:
        logging.info(f"SV-PSF: {n_local}/{n_tiles * n_tiles} tiles used a local PSF")

    R = Y_deconv + 1.40200 * Cr
    G = Y_deconv - 0.34414 * Cb - 0.71414 * Cr
    B = Y_deconv + 1.77200 * Cb
    result = np.stack([R, G, B], axis=2)
    np.clip(result, 0.0, None, out=result)
    return result.astype(np.float32)
