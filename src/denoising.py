"""Denoising and image processing: wavelet, bilateral, NLM, local normalize, arcsinh."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from scipy import ndimage

from src.models import Config
from src.utils import safe_print, get_logger
from src.background import _estimate_sky_sigma

_log = get_logger()

try:
    import pywt
    HAS_PYWT = True
except Exception:
    HAS_PYWT = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

try:
    from skimage.restoration import denoise_nl_means, estimate_sigma, richardson_lucy
    HAS_SKIMAGE_RESTORATION = True
except Exception:
    HAS_SKIMAGE_RESTORATION = False

try:
    from astropy.stats import sigma_clipped_stats
except Exception:
    sigma_clipped_stats = None


def _bayesshrink_threshold(coeffs: np.ndarray, sigma_noise: float) -> float:
    """BayesShrink adaptive threshold for one wavelet subband.

    Estimates the signal standard deviation from the observed subband variance
    minus the noise variance and computes T = sigma_noise² / sigma_signal.
    This per-subband threshold adapts naturally: high-noise subbands (e.g.
    finest detail levels of a faint stack) receive a larger threshold and are
    smoothed more aggressively, while signal-rich subbands (coarser scales
    with nebula structure) receive a smaller threshold that preserves detail.

    Returns ``inf`` for subbands that appear to be pure noise (signal variance
    <= 0), causing all coefficients to be zeroed via soft thresholding.
    """
    sigma_sq_y = float(np.mean(coeffs ** 2))
    sigma_sq_s = max(sigma_sq_y - sigma_noise ** 2, 0.0)
    if sigma_sq_s < 1e-30:
        return float('inf')
    return sigma_noise ** 2 / np.sqrt(sigma_sq_s)


def adaptive_wavelet_denoise(img: np.ndarray, wavelet: str = 'bior1.3',
                              levels: int = 4,
                              chroma_factor: float = 2.0,
                              star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Adaptive multi-scale wavelet denoising using BayesShrink thresholds.

    Unlike ``wavelet_denoise`` which applies a single global
    ``threshold_factor × sigma`` to every subband, this function computes a
    separate BayesShrink threshold for **each** detail subband and channel.
    Subbands dominated by noise receive a large threshold (heavy smoothing);
    subbands with genuine signal receive a small threshold (gentle smoothing).

    This produces better-preserved fine nebula filaments and star halos compared
    to the global-threshold approach, while suppressing sky background noise at
    least as effectively.

    Args:
        img: Float32 stacked image (H, W, 3).
        wavelet: Wavelet family (default ``'bior1.3'``).
        levels: Maximum decomposition depth (default 4).
        chroma_factor: Multiplier applied to the noise estimate for the Cb/Cr
                       chroma channels (default 2.0).  Higher values remove more
                       colour speckle at the cost of slight chroma blurring.
        star_mask: Optional float mask (0–1, 1 = star core).  Star pixels are
                   blended back from the original to avoid core softening.

    Returns:
        Denoised float32 image (H, W, 3).
    """
    if not HAS_PYWT:
        _log.warning("pywt not installed, skipping adaptive wavelet denoise")
        return img

    h, w = img.shape[0], img.shape[1]
    src = img.astype(np.float64)

    # RGB -> YCbCr (ITU-R BT.601)
    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    def _adaptive_denoise_plane(plane: np.ndarray, chroma_mult: float) -> np.ndarray:
        max_level = pywt.dwt_max_level(min(plane.shape), pywt.Wavelet(wavelet).dec_len)
        use_levels = min(levels, max_level)
        if use_levels < 1:
            return plane

        coeffs = pywt.wavedec2(plane, wavelet, level=use_levels)

        # Global noise estimate from finest-level HH subband (standard MAD estimator)
        sigma_noise = np.median(np.abs(coeffs[-1][-1])) / 0.6745
        sigma_noise = max(sigma_noise * chroma_mult, 1e-12)

        new_coeffs = [coeffs[0]]  # keep approximation coefficients unchanged
        for detail_level in coeffs[1:]:
            new_detail = []
            for d in detail_level:
                threshold = _bayesshrink_threshold(d, sigma_noise)
                new_detail.append(pywt.threshold(d, threshold, mode='soft'))
            new_coeffs.append(tuple(new_detail))

        return pywt.waverec2(new_coeffs, wavelet)[:h, :w]

    with ThreadPoolExecutor(max_workers=3) as executor:
        f_Y  = executor.submit(_adaptive_denoise_plane, Y,  1.0)
        f_Cb = executor.submit(_adaptive_denoise_plane, Cb, chroma_factor)
        f_Cr = executor.submit(_adaptive_denoise_plane, Cr, chroma_factor)
        Y_d, Cb_d, Cr_d = f_Y.result(), f_Cb.result(), f_Cr.result()

    # YCbCr -> RGB
    R = Y_d + 1.40200 * Cr_d
    G = Y_d - 0.34414 * Cb_d - 0.71414 * Cr_d
    B = Y_d + 1.77200 * Cb_d
    result = np.stack([R, G, B], axis=2)

    if star_mask is not None:
        mask3 = star_mask[:, :, np.newaxis]
        result = result * (1.0 - mask3) + src * mask3

    return result.astype(np.float32)


def wavelet_denoise(img: np.ndarray, wavelet: str = 'bior1.3',
                    levels: int = 4, threshold_factor: float = 3.0,
                    chroma_factor: float = 2.0,
                    star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Multi-scale wavelet denoising with luma/chroma split and star protection.

    Operates in YCbCr colour space so that chroma channels (Cb, Cr) can receive
    a stronger threshold (chroma_factor x threshold_factor) while luminance is
    handled conservatively.  This removes colour speckle in sky background more
    aggressively without softening fine luminance structure in nebulae.

    If star_mask is provided (float [0,1], 1=star core), the denoised result is
    blended back with the original at star positions so that star cores are not
    softened and their colours are preserved.
    """
    if not HAS_PYWT:
        _log.warning("pywt not installed, skipping wavelet denoise")
        return img

    h, w = img.shape[0], img.shape[1]
    src = img.astype(np.float64)

    # RGB -> YCbCr (ITU-R BT.601 coefficients)
    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    def _denoise_plane(plane, factor):
        max_level = pywt.dwt_max_level(min(plane.shape), pywt.Wavelet(wavelet).dec_len)
        use_levels = min(levels, max_level)
        if use_levels < 1:
            return plane
        coeffs = pywt.wavedec2(plane, wavelet, level=use_levels)
        detail_hh = coeffs[-1][-1]
        sigma_noise = np.median(np.abs(detail_hh)) / 0.6745
        threshold = factor * sigma_noise
        new_coeffs = [coeffs[0]]
        for detail_level in coeffs[1:]:
            new_coeffs.append(tuple(
                pywt.threshold(d, threshold, mode='soft') for d in detail_level
            ))
        return pywt.waverec2(new_coeffs, wavelet)[:h, :w]

    chroma_thresh = threshold_factor * chroma_factor
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_Y  = executor.submit(_denoise_plane, Y,  threshold_factor)
        f_Cb = executor.submit(_denoise_plane, Cb, chroma_thresh)
        f_Cr = executor.submit(_denoise_plane, Cr, chroma_thresh)
        Y_d, Cb_d, Cr_d = f_Y.result(), f_Cb.result(), f_Cr.result()

    # YCbCr -> RGB
    R = Y_d + 1.40200 * Cr_d
    G = Y_d - 0.34414 * Cb_d - 0.71414 * Cr_d
    B = Y_d + 1.77200 * Cb_d
    result = np.stack([R, G, B], axis=2)

    # Star protection: blend back original at star core positions
    if star_mask is not None:
        mask3 = star_mask[:, :, np.newaxis]
        result = result * (1.0 - mask3) + src * mask3

    return result.astype(np.float32)


def bilateral_denoise(img: np.ndarray, sigma_color: float = None,
                      sigma_space: float = 3.0) -> np.ndarray:
    """Edge-preserving bilateral filter denoising (second-pass after wavelet).

    Each output pixel is a Gaussian-weighted average of neighbours that are
    close in *both* space (sigma_space pixels) and value (sigma_color ADU).
    Unlike NLM, the weight of each neighbour is determined independently, so
    there is no "patch pool" whose size varies across the image.  The result
    is spatially uniform: sky noise is reduced by the same factor everywhere
    regardless of whether the pixel sits in open sky or in a gap between
    nebula structures.

    sigma_color:  Value similarity scale in ADU.  Pixels differing by more
                  than ~2xsigma_color are not mixed.  If None (default) it
                  is auto-estimated from the sky noise via adjacent-pixel diffs.
                  A good manual range is 1-5x the stack sky noise.
    sigma_space:  Spatial smoothing radius in pixels (default 3.0).  Larger
                  values smooth over bigger areas but are slower.
    """
    if not HAS_CV2:
        _log.warning("Bilateral denoising requires cv2; skipping")
        return img

    img_max = float(img.max())
    if img_max < 1e-12:
        return img

    if sigma_color is None:
        sigma_color = _estimate_sky_sigma(img)
    sigma_color = float(sigma_color)

    # cv2.bilateralFilter accepts float32 and operates in ADU space directly.
    # d=-1 tells cv2 to derive the pixel neighbourhood diameter from sigma_space.
    # We clamp d to avoid extreme runtimes on large sigma_space values.
    d = min(int(round(6.0 * sigma_space + 1)) | 1, 21)  # odd, max 21 px

    img_f32 = img.astype(np.float32)
    result = cv2.bilateralFilter(img_f32, d=d,
                                 sigmaColor=sigma_color,
                                 sigmaSpace=float(sigma_space))
    return result.astype(np.float32)


def nlm_denoise(img: np.ndarray, h: float = 1.0,
                patch_size: int = 5, patch_distance: int = 7,
                blend: float = 0.5) -> np.ndarray:
    """Non-local means denoising for faint extended nebulosity.

    Searches for similar patches across the image and averages them, which
    smooths large featureless sky and faint nebula regions while preserving
    sharp edges like galaxy arms and star-forming filaments.

    Uses skimage.restoration.denoise_nl_means (operates on float, no quantisation
    loss) when available, with cv2.fastNlMeansDenoisingColored as fallback.

    h:               Filter strength multiplier relative to auto-estimated noise
                     sigma.  1.0 is conservative; 2-3 for heavy sky noise.
    patch_size:      Patch half-size in pixels for similarity comparison (default 5).
    patch_distance:  Search half-window in pixels for candidate patches (default 7).
    blend:           Fraction of NLM result to mix with the original (0-1).
                     blend=1.0 is pure NLM; blend=0.5 (default) mixes equally.
                     Lower values prevent the NLM non-uniformity artifact: NLM
                     over-smooths featureless sky (many matching patches) while
                     under-smoothing sky islands near nebula structures (fewer
                     patches).  With blend=alpha, output variance approx alpha^2*sigma^2/N + (1-alpha)^2*sigma^2,
                     so the (1-alpha)^2*sigma^2 term dominates and the spatial variation in N
                     becomes invisible.  blend=0.5 reduces noise by ~30% while
                     keeping uniformity within ~3%.
    """
    img_max = float(img.max())
    if img_max < 1e-12:
        return img

    sky_sigma = _estimate_sky_sigma(img)
    _log.debug("NLM: sky_sigma=%.4f, img_max=%.1f", sky_sigma, img_max)

    # Pedestal trick: add 3*sigma before NLM and remove it after.
    # Without this, NLM patches that straddle exact-zero sky pixels anchor their
    # weighted average toward zero, pulling positive-noise pixels down to 0 and
    # creating the "leopard print" pattern (large black splotches).  Adding a
    # pedestal converts the half-rectified distribution into a proper Gaussian so
    # NLM can smooth it symmetrically.
    pedestal = 3.0 * sky_sigma
    img_ped = img.astype(np.float64) + pedestal
    ped_max = float(img_ped.max())

    blend = float(np.clip(blend, 0.0, 1.0))
    img_f64 = img.astype(np.float64)

    if HAS_SKIMAGE_RESTORATION:
        img_norm = img_ped.astype(np.float32) / ped_max
        h_norm = h * sky_sigma / ped_max
        denoised = denoise_nl_means(
            img_norm, h=h_norm, fast_mode=True,
            patch_size=patch_size, patch_distance=patch_distance,
            channel_axis=-1)
        nlm_result = denoised.astype(np.float64) * ped_max - pedestal
        result = blend * nlm_result + (1.0 - blend) * img_f64
        return result.astype(np.float32)

    if HAS_CV2:
        img8 = np.clip(img_ped / ped_max * 255, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
        h_cv = max(1, int(round(h * sky_sigma / ped_max * 255)))
        tw = patch_size * 2 + 1
        sw = patch_distance * 2 + 1
        denoised_bgr = cv2.fastNlMeansDenoisingColored(bgr, None, h_cv, h_cv, tw, sw)
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        nlm_result = denoised_rgb.astype(np.float64) / 255.0 * ped_max - pedestal
        result = blend * nlm_result + (1.0 - blend) * img_f64
        return result.astype(np.float32)

    _log.warning("NLM denoising requires skimage.restoration or cv2; skipping")
    return img


def local_normalize(img: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """Local normalization to remove flat-field residuals and vignetting."""
    result = np.empty_like(img)

    def _normalize_channel(c):
        channel = img[:, :, c].astype(np.float64)
        local_mean = ndimage.gaussian_filter(channel, sigma=sigma)
        local_sq_mean = ndimage.gaussian_filter(channel ** 2, sigma=sigma)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
        return c, (channel - local_mean) / (local_std + 1e-12)

    with ThreadPoolExecutor(max_workers=img.shape[2]) as executor:
        for c, ch_result in executor.map(_normalize_channel, range(img.shape[2])):
            result[:, :, c] = ch_result
    # Re-scale to original data range (positive values)
    result = result - result.min()
    orig_max = np.max(img)
    if result.max() > 0 and orig_max > 0:
        result = result / result.max() * orig_max
    return result.astype(np.float32)


def reduce_chroma_noise(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Remove chroma (color) noise from sky background using luminance-protected smoothing.

    Stars and bright objects are masked out before the blur so their chroma
    never bleeds into surrounding pixels (which caused the halos/streaks in the
    naive approach).  Only dark background pixels contribute to, and receive,
    the smoothed chroma.  Stars/objects get their original chroma back exactly.

    Algorithm:
      1. Compute luminance and sigma-clipped sky statistics.
      2. Build a soft sky-mask (1 = background, 0 = star/bright object).
      3. For each channel: blur (chroma * sky_mask) and normalise by
         blurred(sky_mask) - this is a masked/weighted Gaussian that cannot
         receive contamination from bright pixels.
      4. Reconstruct: sky pixels use smooth chroma, bright pixels use original.
    """
    lum = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1]
           + 0.114 * img[:, :, 2]).astype(np.float64)

    # Sky statistics: sigma-clipped to exclude stars, so protect ramp is
    # correctly calibrated even when background extraction has clipped the
    # sky to >=0 (which makes lum[lum <= median] a list of exact zeros ->
    # std=0 -> protect_range~epsilon -> every non-zero pixel treated as a star ->
    # chroma NR silently disabled for all sky pixels).
    if sigma_clipped_stats is not None:
        try:
            _, sky_med, sky_std = sigma_clipped_stats(lum.ravel(), sigma=3.0, maxiters=5)
            sky_med = float(sky_med)
            sky_std = float(sky_std)
        except Exception:
            sky_med = float(np.median(lum))
            sky_std = 0.0
    else:
        sky_med = float(np.median(lum))
        sky_std = 0.0
    # If std is still near zero (e.g., all sky is exactly 0), estimate from
    # the non-zero pixels which represent the positive half of the noise dist.
    # Their std ~= 0.603*sigma_sky, so scale up to recover the true noise level.
    if sky_std < 0.5:
        pos = lum[lum > 0]
        if pos.size > 100:
            try:
                if sigma_clipped_stats is not None:
                    _, _, sky_std = sigma_clipped_stats(pos, sigma=3.0, maxiters=3)
                else:
                    sky_std = float(np.std(pos))
                sky_std = float(sky_std) / 0.603  # half-normal correction
            except Exception:
                pass

    # protect = 0 -> sky (smooth), protect = 1 -> star (leave alone)
    # Ramp from sky_med to sky_med + 3*sky_std
    protect_range = max(3.0 * sky_std, np.finfo(np.float64).eps)
    protect = np.clip((lum - sky_med) / protect_range, 0.0, 1.0)
    sky_mask = 1.0 - protect  # float [0,1]

    result = np.empty_like(img, dtype=np.float64)
    blurred_weight = ndimage.gaussian_filter(sky_mask, sigma=sigma)
    safe_weight = np.maximum(blurred_weight, 1e-9)

    for c in range(img.shape[2]):
        chroma = img[:, :, c].astype(np.float64) - lum
        # Weighted blur: star pixels contribute 0, background contributes 1
        smooth_chroma = ndimage.gaussian_filter(chroma * sky_mask, sigma=sigma) / safe_weight
        # Stars keep original chroma; background gets smoothed chroma
        out_chroma = chroma * protect + smooth_chroma * sky_mask
        result[:, :, c] = lum + out_chroma

    return np.clip(result, 0, None).astype(np.float32)


def arcsinh_stretch(img: np.ndarray, factor: float = None) -> np.ndarray:
    """Non-linear arcsinh stretch with sigma-clipped sky background estimation.

    Estimates the true sky background via iterative sigma-clipping, sets it as
    the black point, then auto-tunes the arcsinh factor so the sky maps to a
    target display level (~15 %).  This preserves faint nebulosity and avoids
    the flat, grey-sky look produced by simple percentile clipping.
    """
    flat = img.ravel().astype(np.float64)
    # Sigma-clipped sky estimate (3 iterations, 2.5-sigma)
    med = np.median(flat)
    for _ in range(3):
        mad = np.median(np.abs(flat - med))
        sig = 1.4826 * mad
        flat = flat[np.abs(flat - med) < 2.5 * sig]
        if len(flat) < 100:
            break
        med = np.median(flat)
    bg = float(med)
    bg_sigma = float(np.std(flat)) if len(flat) > 1 else 1.0

    # Black point: just below the sky floor
    black = max(bg - 1.0 * bg_sigma, 0.0)
    # White point: bright stars / bright nebula cap
    white = np.percentile(img, 99.8)
    span = white - black
    if span < 1e-12:
        return np.zeros_like(img)

    norm = np.clip((img - black) / span, 0.0, 1.0)

    # Auto-tune arcsinh factor so sky maps to ~15 % of output range
    if factor is None:
        target_bg = 0.15
        bg_norm = float(np.clip((bg - black) / span, 1e-6, 1.0))
        factor = Config.ARCSINH_STRETCH_FACTOR
        for f in (3.0, 5.0, 10.0, 20.0, 50.0, 100.0):
            if np.arcsinh(bg_norm * f) / np.arcsinh(f) >= target_bg:
                factor = f
                break

    stretched = np.arcsinh(norm * factor) / np.arcsinh(factor)
    return np.clip(stretched, 0.0, 1.0)
