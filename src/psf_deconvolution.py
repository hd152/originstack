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


def estimate_psf(img: np.ndarray, star_positions,
                 cutout_radius: int = None, psf_size: int = None,
                 model: str = 'moffat') -> Tuple[Optional[np.ndarray], float]:
    """Estimate the point spread function from star cutouts via profile fitting.

    Fits a 2D Moffat (default) or Gaussian model to bright, unsaturated stars
    and builds a normalized PSF kernel from the median fit parameters.

    Returns (psf_kernel, fwhm_pixels).  Returns (None, 0.0) if too few stars
    are successfully fit.
    """
    if not HAS_CURVE_FIT:
        logging.warning("scipy.optimize.curve_fit unavailable; cannot estimate PSF")
        return None, 0.0
    if star_positions is None or len(star_positions) < Config.RL_PSF_MIN_STARS:
        logging.warning("Too few stars for PSF estimation "
                        f"({0 if star_positions is None else len(star_positions)} < {Config.RL_PSF_MIN_STARS})")
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
