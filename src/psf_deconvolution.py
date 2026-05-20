"""PSF estimation and Richardson-Lucy deconvolution."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from src.models import Config

try:
    from scipy.optimize import curve_fit
    HAS_CURVE_FIT = True
except Exception:
    HAS_CURVE_FIT = False

try:
    from skimage.restoration import denoise_nl_means, estimate_sigma, richardson_lucy
    HAS_SKIMAGE_RESTORATION = True
except Exception:
    HAS_SKIMAGE_RESTORATION = False

try:
    from photutils.psf import EPSFBuilder
    from photutils.psf import extract_stars as _phot_extract_stars
    from astropy.nddata import NDData as _NDData
    from astropy.table import Table as _AstroTable
    HAS_PHOTUTILS_PSF = True
except Exception:
    HAS_PHOTUTILS_PSF = False


def _estimate_psf_epsf(img: np.ndarray, star_positions,
                       psf_size: int) -> Tuple[Optional[np.ndarray], float]:
    """Empirical PSF via photutils EPSFBuilder.

    Stacks oversampled star cutouts without assuming any parametric model,
    producing a more accurate kernel than the Moffat/Gaussian fitting approach
    when stars are well-sampled (typical for stacked deep-sky images).
    Returns (psf_kernel, fwhm_px) or (None, 0.0) on failure.
    """
    if not HAS_PHOTUTILS_PSF:
        return None, 0.0
    if star_positions is None or len(star_positions) < Config.RL_PSF_MIN_STARS:
        return None, 0.0

    lum = (img if img.ndim == 2
           else 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])

    try:
        # Sort by flux, take brightest unsaturated stars
        try:
            order = np.argsort(star_positions['flux'])[::-1][:Config.RL_PSF_MAX_STARS]
        except (KeyError, TypeError):
            order = range(min(Config.RL_PSF_MAX_STARS, len(star_positions)))

        xs = [float(star_positions[i]['xcentroid']) for i in order]
        ys = [float(star_positions[i]['ycentroid']) for i in order]

        tbl = _AstroTable()
        tbl['x'] = xs
        tbl['y'] = ys

        stars = _phot_extract_stars(
            _NDData(data=lum.astype(np.float64)), tbl, size=psf_size
        )
        if len(stars) < Config.RL_PSF_MIN_STARS:
            return None, 0.0

        epsf, _ = EPSFBuilder(oversampling=2, maxiters=10,
                               progress_bar=False)(stars)

        kernel = np.array(epsf.data, dtype=np.float64)
        kernel = np.maximum(kernel, 0.0)
        total = kernel.sum()
        if total <= 0.0:
            return None, 0.0
        kernel /= total

        # Estimate FWHM from the fraction of kernel above half-maximum
        half_max = kernel.max() / 2.0
        fwhm = float(np.sqrt(np.sum(kernel >= half_max) / np.pi) * 2.0)

        logging.info("PSF estimated via EPSFBuilder: FWHM=%.2f px, "
                     "from %d stars", fwhm, len(stars))
        return kernel, fwhm
    except Exception as exc:
        logging.debug("EPSFBuilder failed: %s", exc)
        return None, 0.0


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

    # Prefer photutils EPSFBuilder (empirical, model-free) when available
    epsf_kernel, epsf_fwhm = _estimate_psf_epsf(img, star_positions, psf_size)
    if epsf_kernel is not None:
        return epsf_kernel, epsf_fwhm

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

    Used as fallback when star-based PSF estimation is unavailable (e.g. no
    photutils).  Returns a normalized 2D kernel.
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
        star_positions: Source table from DAOStarFinder / detect_stars.
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

    # Optional blind RL refinement
    if iterations > 0 and HAS_SKIMAGE_RESTORATION:
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


def richardson_lucy_deconvolve(img: np.ndarray, psf: np.ndarray,
                                iterations: int = None,
                                star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Richardson-Lucy deconvolution for sharpening the stacked image.

    Operates on luminance only (YCbCr decomposition) to avoid colour
    artifacts, then recombines with the original chrominance channels.

    star_mask (float [0,1], 1=star core): if provided, the deconvolved
    result is blended back with the original at star positions to prevent
    ringing artifacts on bright star cores.
    """
    if not HAS_SKIMAGE_RESTORATION:
        logging.warning("skimage.restoration not available; skipping Richardson-Lucy deconvolution")
        return img
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

    Y_deconv = richardson_lucy(Y_pos, psf, num_iter=iterations, clip=False)
    Y_deconv = Y_deconv - pedestal

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
