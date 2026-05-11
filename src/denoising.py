"""Denoising and image processing: wavelet, bilateral, NLM, local normalize, arcsinh."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from src.models import Config
from src.utils import safe_print, get_logger
from src.background import _estimate_sky_sigma

_log = get_logger()

pywt = None
HAS_PYWT = False
cv2 = None
HAS_CV2 = False
denoise_nl_means = estimate_sigma = richardson_lucy = None
HAS_SKIMAGE_RESTORATION = False


def _ensure_pywt():
    global pywt, HAS_PYWT
    if not HAS_PYWT:
        try:
            import pywt as _pywt
            if hasattr(_pywt, 'dwt_max_level'):
                pywt = _pywt
                HAS_PYWT = True
        except Exception:
            pass
    return pywt


def _ensure_cv2():
    global cv2, HAS_CV2
    if not HAS_CV2:
        try:
            import cv2 as _cv2
            if hasattr(_cv2, 'bilateralFilter'):
                cv2 = _cv2
                HAS_CV2 = True
        except Exception:
            pass
    return cv2


def _ensure_skimage_restoration():
    global denoise_nl_means, estimate_sigma, richardson_lucy, HAS_SKIMAGE_RESTORATION
    if not HAS_SKIMAGE_RESTORATION:
        try:
            from skimage.restoration import (
                denoise_nl_means as _dnlm,
                estimate_sigma as _es,
                richardson_lucy as _rl)
            if callable(_dnlm):
                denoise_nl_means = _dnlm
                estimate_sigma = _es
                richardson_lucy = _rl
                HAS_SKIMAGE_RESTORATION = True
        except Exception:
            pass
    return HAS_SKIMAGE_RESTORATION

try:
    from astropy.stats import sigma_clipped_stats
except Exception:
    
    sigma_clipped_stats = None


def estimate_denoise_strength(stacked: np.ndarray, fwhm_mean: float = 0.0) -> float:
    """Auto-tune wavelet denoise threshold_factor from stacked image noise level.

    Measures sky background noise and maps signal-to-noise ratio to a denoise
    strength.  Noisy stacks (low SNR) receive a higher threshold to suppress
    sky graininess; clean high-SNR stacks receive a lower threshold to avoid
    over-smoothing faint nebula detail.

    Args:
        stacked: Float32 stacked RGB image (H, W, 3).
        fwhm_mean: Mean star FWHM in pixels from quality analysis (0 = unknown).

    Returns:
        Recommended threshold_factor for ``wavelet_denoise`` (1.0–5.5).
    """
    sky_sigma = _estimate_sky_sigma(stacked)
    if sky_sigma < 1e-10:
        return 3.0

    green = stacked[:, :, 1].astype(np.float64)
    if sigma_clipped_stats is not None:
        try:
            _, bg_median, _ = sigma_clipped_stats(green, sigma=3.0, maxiters=5)
            bg_median = float(bg_median)
        except Exception:
            bg_median = float(np.median(green))
    else:
        bg_median = float(np.median(green))

    p95 = float(np.percentile(green, 95))
    signal = max(p95 - bg_median, 0.0)
    stack_snr = signal / sky_sigma

    # Map SNR → strength:  SNR=1→4.5,  SNR=10→3.0,  SNR=100→1.5
    strength = 4.5 - 1.5 * np.log10(max(stack_snr, 1.0))

    # FWHM modulation: large PSF spreads noise to coarser scales, allowing a
    # slightly more aggressive threshold; reference is 4 px (typical).
    if fwhm_mean > 0.0:
        strength *= float(np.clip(fwhm_mean / 4.0, 0.8, 1.3))

    return float(np.clip(strength, 1.0, 5.5))


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
    _ensure_pywt()
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

    Y_d  = _adaptive_denoise_plane(Y,  1.0)
    Cb_d = _adaptive_denoise_plane(Cb, chroma_factor)
    Cr_d = _adaptive_denoise_plane(Cr, chroma_factor)

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
    _ensure_pywt()
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
    Y_d  = _denoise_plane(Y,  threshold_factor)
    Cb_d = _denoise_plane(Cb, chroma_thresh)
    Cr_d = _denoise_plane(Cr, chroma_thresh)

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


def bilateral_denoise(img: np.ndarray, sigma_color: Optional[float] = None,
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
    _ensure_cv2()
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

    _ensure_skimage_restoration()
    _ensure_cv2()
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


def _median_filter_fast(plane: np.ndarray, ksize: int) -> np.ndarray:
    """Median filter with cv2 float32 fast-path and scipy fallback.

    cv2.medianBlur supports float32 input for ksize 3 and 5 (SIMD-accelerated).
    For ksize > 5, cv2 only accepts uint8, which is insufficient precision for
    astrophoto data; scipy.ndimage.median_filter (C-based histogram algorithm)
    is used instead and handles arbitrary odd kernel sizes on float64 directly.

    Args:
        plane: 2-D float64 array.
        ksize: Odd kernel side length (3, 5, 9, 17, …).

    Returns:
        Median-filtered float64 array of the same shape.
    """
    _ensure_cv2()
    if HAS_CV2 and ksize in (3, 5):
        return cv2.medianBlur(plane.astype(np.float32), ksize).astype(np.float64)
    return ndimage.median_filter(plane, size=ksize)


def mmt_denoise(img: np.ndarray, levels: int = 4, threshold_factor: float = 3.0,
                chroma_factor: float = 2.0,
                star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Multiscale Median Transform (MMT) denoising.

    Decomposes the image into detail layers using successive median filters
    (kernel sizes 3, 5, 9, 17 px for 4 levels).  Each layer captures structure
    at one spatial scale; noise is estimated per-layer via the MAD estimator
    and removed via soft thresholding.  The image is reconstructed from the
    thresholded layers plus the coarsest background residual.

    Advantages over DWT (wavelet_denoise):
    - Median filters are robust to non-Gaussian noise (Poisson + read noise).
    - Better edge preservation in thin filaments and star halos.
    - More effective against residual hot pixels that survive calibration.

    Operates in YCbCr space; chroma channels receive a larger effective
    threshold (chroma_factor × luma threshold) so colour speckle is removed
    more aggressively than luminance structure.

    Args:
        img:              Float32 stacked image (H, W, 3).
        levels:           Number of decomposition scales (default 4 → kernel
                          sizes 3, 5, 9, 17 px).
        threshold_factor: Noise-sigma multiplier for soft thresholding
                          (default 3.0).  Larger → more aggressive noise removal.
        chroma_factor:    Multiplier on the noise estimate for Cb/Cr channels
                          (default 2.0).  Higher removes more colour speckle at
                          the cost of slight chroma blurring.
        star_mask:        Optional float mask (0–1, 1 = star core).  Star pixels
                          are blended back from the original to protect cores.

    Returns:
        Denoised float32 image (H, W, 3).
    """
    h, w = img.shape[:2]
    src = img.astype(np.float64)

    # RGB → YCbCr (ITU-R BT.601)
    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    def _mmt_plane(plane: np.ndarray, chroma_mult: float) -> np.ndarray:
        prev = plane.copy()
        detail_layers = []
        for k in range(levels):
            ksize = 2 ** (k + 1) + 1  # 3, 5, 9, 17 for k = 0..3
            blurred = _median_filter_fast(prev, ksize)
            detail_layers.append(prev - blurred)
            prev = blurred  # carry residual to next finer scale

        # Noise estimate from the finest detail layer (MAD, consistent with
        # the estimator used in adaptive_wavelet_denoise)
        sigma_noise = np.median(np.abs(detail_layers[0])) / 0.6745
        sigma_noise = max(sigma_noise * chroma_mult, 1e-12)

        # Soft-threshold each detail layer.  Noise amplitude decays across
        # scales roughly as 1/sqrt(k+1) for the median cascade (empirically
        # matches PixInsight MMT behaviour); coarser scales therefore receive
        # a smaller threshold so genuine large-scale structure is preserved.
        result = prev.copy()  # start from coarsest background residual
        for k, layer in enumerate(detail_layers):
            scale_sigma = sigma_noise / np.sqrt(float(k + 1))
            threshold = threshold_factor * scale_sigma
            thresholded = np.sign(layer) * np.maximum(np.abs(layer) - threshold, 0.0)
            result = result + thresholded

        return result

    Y_d  = _mmt_plane(Y,  1.0)
    Cb_d = _mmt_plane(Cb, chroma_factor)
    Cr_d = _mmt_plane(Cr, chroma_factor)

    # YCbCr → RGB
    R = Y_d + 1.40200 * Cr_d
    G = Y_d - 0.34414 * Cb_d - 0.71414 * Cr_d
    B = Y_d + 1.77200 * Cb_d
    result = np.stack([R, G, B], axis=2)

    if star_mask is not None:
        mask3 = star_mask[:, :, np.newaxis]
        result = result * (1.0 - mask3) + src * mask3

    return result.astype(np.float32)


def acdnr_denoise(img: np.ndarray, smoothing_sigma: float = 1.5,
                  contrast_k: float = 3.0, chroma_factor: float = 2.0,
                  star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Adaptive Contrast-based Denoising with Noise Reduction (ACDNR-style).

    Computes a per-pixel adaptive weight from local luminance contrast relative
    to the sky-noise level::

        w(x,y) = exp(-0.5 * (contrast(x,y) / (k * sigma_noise))^2)

    where ``contrast(x,y) = |luma(x,y) - gaussian_smooth(luma, sigma)|``.

    Pixels in flat sky regions (contrast << k·σ) receive w ≈ 1 and are
    fully smoothed.  Pixels near nebula filaments or galaxy edges (contrast >>
    k·σ) receive w ≈ 0 and are left unchanged.  The transition is controlled
    by ``contrast_k``: smaller values smooth more structure aggressively;
    larger values restrict smoothing to featureless sky only.

    Chroma channels (Cb/Cr) use the same luma-derived contrast mask but with
    a higher effective threshold (k * chroma_factor), so colour speckle in the
    sky background is always removed more aggressively than luma detail.

    Args:
        img:             Float32 stacked image (H, W, 3).
        smoothing_sigma: Gaussian σ for contrast detection and smoothing
                         (default 1.5 px).  Larger values remove coarser
                         spatial noise but blur fine structure.
        contrast_k:      Noise-sigma multiplier for the contrast threshold
                         (default 3.0).  Lower → more aggressive; higher →
                         sky-only smoothing.
        chroma_factor:   Chroma channels use k * chroma_factor as threshold
                         (default 2.0), making them 2× more aggressively
                         denoised than luma.
        star_mask:       Optional float mask (0–1, 1 = star core).  Star
                         pixels are blended back from the original.

    Returns:
        Denoised float32 image (H, W, 3).
    """
    src = img.astype(np.float64)

    luma = (0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1]
            + 0.11400 * src[:, :, 2])

    sigma_noise = float(_estimate_sky_sigma(img))
    if sigma_noise < 1e-10:
        return img.copy()

    # Local contrast map at the smoothing scale
    smooth_luma = ndimage.gaussian_filter(luma, sigma=smoothing_sigma)
    contrast = np.abs(luma - smooth_luma)

    # YCbCr split (same BT.601 coefficients as the other denoisers)
    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    # Adaptive weights: Gaussian decay around the noise threshold
    luma_thr   = max(contrast_k * sigma_noise, 1e-12)
    chroma_thr = max(contrast_k * sigma_noise * max(chroma_factor, 1.0), 1e-12)
    luma_w   = np.exp(-0.5 * (contrast / luma_thr)   ** 2)
    chroma_w = np.exp(-0.5 * (contrast / chroma_thr) ** 2)

    # Smooth each YCbCr plane, then adaptively blend
    Y_smooth  = ndimage.gaussian_filter(Y,  sigma=smoothing_sigma)
    Cb_smooth = ndimage.gaussian_filter(Cb, sigma=smoothing_sigma)
    Cr_smooth = ndimage.gaussian_filter(Cr, sigma=smoothing_sigma)

    Y_d  = luma_w   * Y_smooth  + (1.0 - luma_w)   * Y
    Cb_d = chroma_w * Cb_smooth + (1.0 - chroma_w) * Cb
    Cr_d = chroma_w * Cr_smooth + (1.0 - chroma_w) * Cr

    # YCbCr → RGB
    R = Y_d + 1.40200 * Cr_d
    G = Y_d - 0.34414 * Cb_d - 0.71414 * Cr_d
    B = Y_d + 1.77200 * Cb_d
    result = np.stack([R, G, B], axis=2)

    if star_mask is not None:
        mask3 = star_mask[:, :, np.newaxis]
        result = result * (1.0 - mask3) + src * mask3

    return result.astype(np.float32)


def local_normalize(img: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """Local normalization to remove flat-field residuals and vignetting."""
    result = np.empty_like(img)

    for c in range(img.shape[2]):
        channel = img[:, :, c].astype(np.float64)
        local_mean = ndimage.gaussian_filter(channel, sigma=sigma)
        local_sq_mean = ndimage.gaussian_filter(channel ** 2, sigma=sigma)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
        result[:, :, c] = (channel - local_mean) / (local_std + 1e-12)
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


def generalized_hyperbolic_stretch(
        img: np.ndarray,
        b: float = 8.0,
        SP: float = 0.15,
        LP: float = 0.0,
        HP: float = 0.95,
        black_point: Optional[float] = None,
        white_point: Optional[float] = None) -> np.ndarray:
    """Generalized Hyperbolic Stretch (GHS) for galaxy/nebula imaging.

    The state-of-the-art stretch algorithm for deep-sky display.  Unlike the
    classic arcsinh stretch (which applies a fixed symmetric curve), GHS gives
    independent control over four parameters that together handle the extreme
    dynamic range in galaxy images:

    b  — Stretch factor.  0 = linear; 5 = moderate; 8–12 = galaxy-optimised.
         Higher values push faint outer spiral arms and dust lanes into the
         displayable range while compressing the bright nucleus.
    SP — Symmetry Point [0–1 normalised].  The pivot of the stretch: the curve
         applies equal emphasis to data above and below SP.  Setting SP well
         below the galaxy core (0.10–0.20) lifts faint outer structure
         disproportionately relative to the bright inner regions — exactly what
         is needed for objects like M64 where the outer arms are orders of
         magnitude fainter than the nucleus.
    LP — Linear Point [0–1].  Black-point cut-in: values ≤ LP map to 0.
         All normalised sky noise below LP is clipped to black.  Typical: 0–0.05.
    HP — Highlights Protection [0–1].  Values ≥ HP map to 1.  Protects the
         bright nucleus and star cores from blowing out to pure white while the
         faint outer arms are being stretched into visibility.  Typical: 0.85–0.98.

    The image is normalised via the same sigma-clipped sky estimation used by
    ``arcsinh_stretch``, so sky → ~0 and bright stars → ~1 before the GHS
    transform is applied, ensuring the parameters are object-independent.

    Reference: Cranfield & Symons (2021), https://ghsastro.co.uk/
    """
    # --- Normalise to [0, 1] using sigma-clipped sky statistics ---
    if black_point is None or white_point is None:
        flat = img.ravel().astype(np.float64)
        med = float(np.median(flat))
        for _ in range(3):
            mad = np.median(np.abs(flat - med))
            sig = 1.4826 * mad
            flat = flat[np.abs(flat - med) < 2.5 * sig]
            if len(flat) < 100:
                break
            med = float(np.median(flat))
        bg = med
        bg_sigma = float(np.std(flat)) if len(flat) > 1 else 1.0
        black_point = max(bg - 1.0 * bg_sigma, 0.0)
        white_point = float(np.percentile(img, 99.9))
    span = white_point - black_point
    if span < 1e-12:
        return np.zeros_like(img, dtype=np.float32)

    norm = np.clip((img.astype(np.float64) - black_point) / span, 0.0, 1.0)

    # --- Apply GHS piecewise transform ---
    if abs(b) < 1e-6:
        # b ≈ 0: degenerate case — linear transform over [LP, HP]
        if HP > LP:
            out = np.clip((norm - LP) / (HP - LP), 0.0, 1.0)
        else:
            out = norm.copy()
        return out.astype(np.float32)

    # arcsinh evaluated at LP and HP establishes the normalisation range
    ghs_lp = float(np.arcsinh(b * (LP - SP)))
    ghs_hp = float(np.arcsinh(b * (HP - SP)))
    denom = ghs_hp - ghs_lp
    if abs(denom) < 1e-12:
        return np.zeros_like(img, dtype=np.float32)

    # Core GHS transform: arcsinh-based, centred on SP
    core = (np.arcsinh(b * (norm - SP)) - ghs_lp) / denom

    # Piecewise: below LP → black, above HP → white, middle → GHS curve
    out = np.where(norm <= LP, 0.0, np.where(norm >= HP, 1.0, core))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def multiscale_local_contrast(
        img: np.ndarray,
        strength: float = 0.7,
        scales: Tuple[int, ...] = (2, 12, 40),
        scale_weights: Tuple[float, ...] = (0.3, 0.6, 0.1),
        star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Multiscale local contrast enhancement (MLCE) for galaxy structure.

    Applies luminance-domain unsharp masking simultaneously at fine, medium,
    and coarse spatial scales with a mid-tone protection mask that:

      • Suppresses enhancement in the sky background (avoids amplifying noise).
      • Protects bright nuclei and star cores from ringing or blowout.
      • Focuses full enhancement on the mid-tone range where galaxy spiral arms,
        dust lanes, and star-forming regions live.

    For the Black Eye Galaxy (M64) the medium scale (σ ≈ 12 px) is the most
    valuable: it precisely targets the width of the characteristic dark dust
    band, creating the stark contrast that makes this galaxy recognisable.

    Args:
        img:           Float32 stacked RGB image (H, W, 3), linear scale.
        strength:      Overall enhancement multiplier (0 = off, 1 = full).
                       Typical range 0.4–0.9 for galaxy imaging.
        scales:        Gaussian σ values (px) for each detail layer.
                       Default (2, 12, 40) = fine / medium / coarse.
        scale_weights: Relative weight of each scale (sum need not equal 1).
                       Default gives primary weight to the medium-scale layer.
        star_mask:     Float mask (1 = star core).  Star pixels receive no
                       contrast enhancement — their halos must not grow.

    Returns:
        Enhanced float32 image (H, W, 3), non-negative.
    """
    lum = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1]
           + 0.114 * img[:, :, 2]).astype(np.float64)

    # Mid-tone protection mask —————————————————————————————————————————————
    # Ramp from 0 at the sky floor to 1 over 2×sky_sigma, then ramp back
    # to 0 as we approach the top 3% (bright nucleus / saturated stars).
    sky_sigma = float(_estimate_sky_sigma(img))
    sky_floor = float(np.median(lum)) + 1.5 * sky_sigma
    highlight_cap = float(np.percentile(lum, 97))

    # Low ramp: 0 at sky_floor → 1 at sky_floor + 2*sky_sigma
    low_ramp_range = max(2.0 * sky_sigma, 1e-6)
    mask = np.clip((lum - sky_floor) / low_ramp_range, 0.0, 1.0)

    # High ramp: 1 below highlight_cap → 0 at highlight_cap + 0.2*(cap-floor)
    hi_transition = max((highlight_cap - sky_floor) * 0.2, 1.0)
    mask *= np.clip(1.0 - (lum - highlight_cap) / hi_transition, 0.0, 1.0)

    # Star protection: no enhancement at star core positions
    if star_mask is not None:
        mask *= (1.0 - star_mask.astype(np.float64))

    # Multiscale detail injection ——————————————————————————————————————————
    enhanced_lum = lum.copy()
    for sigma, w in zip(scales, scale_weights):
        if w <= 0 or strength <= 0:
            continue
        blurred = ndimage.gaussian_filter(lum, sigma=float(sigma))
        detail = lum - blurred          # high-frequency detail at this scale
        enhanced_lum += strength * w * detail * mask

    # Reconstruct RGB by the luminance ratio (hue/saturation preserved)
    safe_lum = np.where(lum > 1e-10, lum, 1e-10)
    ratio = enhanced_lum / safe_lum
    result = img.astype(np.float64) * ratio[:, :, np.newaxis]
    return np.clip(result, 0.0, None).astype(np.float32)


def reduce_stars(
        img: np.ndarray,
        star_mask: np.ndarray,
        reduction_factor: float = 0.4,
        blur_sigma: float = 1.5) -> np.ndarray:
    """Reduce star prominence to improve galaxy-to-star visual balance.

    Stars in galaxy images compete visually with the delicate structure of
    spiral arms and dust lanes.  This function softens star cores by blending
    the original image with a Gaussian-blurred version at star positions,
    making stars appear slightly smaller and less dominant without erasing them.

    The effect is purposefully subtle: star colours and relative brightnesses
    are preserved; only their apparent angular size is reduced.  This is the
    same technique used in post-processing tools like StarXTerminator (when
    operated in 'reduce' mode rather than 'remove' mode).

    Args:
        img:              Float32 stacked RGB image (H, W, 3).
        star_mask:        Float mask (1 = star core, 0 = background).
                          If None, returns the image unchanged.
        reduction_factor: Blend fraction toward the blurred image at star
                          positions (0 = no change, 1 = full blur).
                          Typical: 0.3–0.6 for subtle to moderate reduction.
        blur_sigma:       Gaussian blur radius for the replacement (px).
                          Larger values give softer but dimmer star cores.

    Returns:
        Float32 image (H, W, 3) with reduced star sizes, non-negative.
    """
    if star_mask is None:
        return img

    reduction_factor = float(np.clip(reduction_factor, 0.0, 1.0))
    if reduction_factor <= 0.0:
        return img

    # Blur per spatial dimension only (channel axis excluded)
    blurred = ndimage.gaussian_filter(
        img.astype(np.float64),
        sigma=(blur_sigma, blur_sigma, 0))

    blend = (star_mask * reduction_factor).astype(np.float64)
    mask3 = blend[:, :, np.newaxis]
    result = img.astype(np.float64) * (1.0 - mask3) + blurred * mask3
    return np.clip(result, 0.0, None).astype(np.float32)


def bm3d_denoise(img: np.ndarray, sigma_psd: float = 0.0,
                  block_size: int = 8, stride: Optional[int] = None,
                  search_window: int = 16, group_size: int = 8,
                  star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """BM3D collaborative-filter denoising on the luminance channel.

    Implements the core BM3D pipeline (Dabov et al. 2007):
      Step 1 — hard-thresholding in joint 3-D DCT domain.
      Step 2 — Wiener filter using the Step-1 estimate as pilot.

    Similar-looking 8×8 patches are grouped into a 3-D stack, a 3-D DCT is
    applied (2-D spatial + 1-D across the group axis), coefficients are
    thresholded or Wiener-filtered, then the result is inverse-transformed and
    aggregated back into the image using weighted overlap-add.

    Only luminance is processed; chroma channels receive mild Gaussian
    smoothing proportional to the estimated noise level to keep colour noise
    suppressed without blurring colour gradients.

    Args:
        img:          Float32 stacked image (H, W, 3).
        sigma_psd:    Noise standard deviation in image units.  0 = auto from
                      sky-background estimate.
        block_size:   Patch side length in pixels (default 8).
        stride:       Step between reference block centres (default: auto,
                      8 px for images > 1500 px, 4 px otherwise).
        search_window: Half-size of the local block-matching window (pixels).
        group_size:   Maximum number of similar blocks per group (default 8).
        star_mask:    Optional float mask (1 = star core) blended back to
                      preserve star colours and prevent core softening.

    Returns:
        Denoised float32 image (H, W, 3).

    Notes:
        Runtime scales with image area ÷ stride².  Expect 15–60 s on a
        2 K image at stride=8.  Use ``--bm3d-stride 16`` for large images.
    """
    try:
        from scipy.fft import dctn, idctn
    except ImportError:
        from scipy.fftpack import dctn, idctn

    src = img.astype(np.float64)
    H, W = src.shape[:2]

    if stride is None:
        stride = 8 if max(H, W) > 1500 else 4

    if sigma_psd <= 0.0:
        sigma_psd = float(_estimate_sky_sigma(img))
    if sigma_psd < 1e-9:
        return img.copy()

    # YCbCr decomposition
    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    bs = block_size
    sw = search_window

    # Reflective padding avoids border artefacts
    pad = sw + bs
    Yp = np.pad(Y, pad, mode='reflect')

    # Reference block grid on the original (unpadded) image
    ref_ys = np.arange(0, H - bs + 1, stride)
    ref_xs = np.arange(0, W - bs + 1, stride)
    ny, nx = len(ref_ys), len(ref_xs)

    # --- Precompute 2-D DCTs of all reference blocks in one batched call ---
    all_ref = np.array([[Y[yr:yr + bs, xr:xr + bs]
                         for xr in ref_xs] for yr in ref_ys])  # (ny, nx, bs, bs)
    all_dcts = dctn(all_ref.reshape(ny * nx, bs, bs),
                    axes=(1, 2), norm='ortho')            # (N, bs, bs)
    all_dcts_flat = all_dcts.reshape(ny * nx, bs * bs)   # (N, bs²)

    # Hard-threshold value (global, Step 1)
    ht_threshold = sigma_psd * np.sqrt(2.0 * np.log(float(bs * bs)))
    # Distance threshold: blocks whose mean squared pixel diff < this are "similar"
    dist_threshold = (ht_threshold * 0.5) ** 2 * bs * bs

    max_dy = max(1, sw // stride)
    max_dx = max(1, sw // stride)

    # Accumulation arrays (Step 1 and Step 2)
    acc1 = np.zeros((H, W), dtype=np.float64)
    wgt1 = np.zeros((H, W), dtype=np.float64)
    acc2 = np.zeros((H, W), dtype=np.float64)
    wgt2 = np.zeros((H, W), dtype=np.float64)

    for iy in range(ny):
        yr = ref_ys[iy]
        iy_lo = max(0, iy - max_dy)
        iy_hi = min(ny - 1, iy + max_dy)

        for ix in range(nx):
            xr = ref_xs[ix]
            ix_lo = max(0, ix - max_dx)
            ix_hi = min(nx - 1, ix + max_dx)

            i_ref = iy * nx + ix
            ref_dct_flat = all_dcts_flat[i_ref]       # (bs²,)

            # Candidate block indices in the search window
            cand_iy, cand_ix = np.mgrid[iy_lo:iy_hi + 1, ix_lo:ix_hi + 1]
            cand_idx = (cand_iy * nx + cand_ix).ravel()   # (N_cand,)

            # Vectorised L2 distance in DCT domain (proportional to pixel-domain L2)
            diffs = all_dcts_flat[cand_idx] - ref_dct_flat   # (N_cand, bs²)
            dists = np.einsum('ij,ij->i', diffs, diffs)       # (N_cand,)

            # Keep similar blocks, capped at group_size
            similar_mask = dists < dist_threshold
            similar_idx = cand_idx[similar_mask]
            if len(similar_idx) == 0:
                similar_idx = cand_idx[:1]   # always include self
            if len(similar_idx) > group_size:
                order = np.argsort(dists[similar_mask])[:group_size]
                similar_idx = similar_idx[order]

            # Pixel-domain group blocks
            sim_iy = similar_idx // nx
            sim_ix = similar_idx % nx
            group_blks = all_ref[sim_iy, sim_ix, :, :]   # (N_g, bs, bs)
            N_g = len(similar_idx)

            # --- Step 1: Hard thresholding in full 3-D DCT domain ---
            spec3 = dctn(group_blks, axes=(0, 1, 2), norm='ortho')
            ht = np.where(np.abs(spec3) >= ht_threshold, spec3, 0.0)
            n_nz = max(1, int(np.count_nonzero(ht)))
            w1 = 1.0 / n_nz
            denoised1 = idctn(ht, axes=(0, 1, 2), norm='ortho')

            for k, (gi, gj) in enumerate(zip(sim_iy, sim_ix)):
                yr2, xr2 = ref_ys[gi], ref_xs[gj]
                acc1[yr2:yr2 + bs, xr2:xr2 + bs] += w1 * denoised1[k]
                wgt1[yr2:yr2 + bs, xr2:xr2 + bs] += w1

    pilot = np.where(wgt1 > 0, acc1 / wgt1, Y)

    # --- Step 2: Wiener filter using pilot estimate ---
    pilot_dcts = dctn(np.array([[pilot[yr:yr + bs, xr:xr + bs]
                                  for xr in ref_xs] for yr in ref_ys]).reshape(ny * nx, bs, bs),
                      axes=(1, 2), norm='ortho')       # (N, bs, bs)

    for iy in range(ny):
        yr = ref_ys[iy]
        iy_lo = max(0, iy - max_dy)
        iy_hi = min(ny - 1, iy + max_dy)

        for ix in range(nx):
            xr = ref_xs[ix]
            ix_lo = max(0, ix - max_dx)
            ix_hi = min(nx - 1, ix + max_dx)

            i_ref = iy * nx + ix
            ref_dct_flat = all_dcts_flat[i_ref]

            cand_iy, cand_ix = np.mgrid[iy_lo:iy_hi + 1, ix_lo:ix_hi + 1]
            cand_idx = (cand_iy * nx + cand_ix).ravel()

            diffs = all_dcts_flat[cand_idx] - ref_dct_flat
            dists = np.einsum('ij,ij->i', diffs, diffs)
            similar_mask = dists < dist_threshold
            similar_idx = cand_idx[similar_mask]
            if len(similar_idx) == 0:
                similar_idx = cand_idx[:1]
            if len(similar_idx) > group_size:
                order = np.argsort(dists[similar_mask])[:group_size]
                similar_idx = similar_idx[order]

            sim_iy = similar_idx // nx
            sim_ix = similar_idx % nx
            noisy_group = all_ref[sim_iy, sim_ix, :, :]     # (N_g, bs, bs)
            pilot_group = np.array([pilot[ref_ys[gi]:ref_ys[gi] + bs,
                                         ref_xs[gj]:ref_xs[gj] + bs]
                                     for gi, gj in zip(sim_iy, sim_ix)])

            spec_noisy = dctn(noisy_group, axes=(0, 1, 2), norm='ortho')
            spec_pilot = dctn(pilot_group, axes=(0, 1, 2), norm='ortho')

            pilot_sq = spec_pilot ** 2
            wiener = pilot_sq / (pilot_sq + sigma_psd ** 2 + 1e-30)
            spec_filt = wiener * spec_noisy

            wiener_w = float(np.sum(wiener ** 2)) / max(len(similar_idx), 1)
            w2 = 1.0 / max(wiener_w, 1e-12)
            denoised2 = idctn(spec_filt, axes=(0, 1, 2), norm='ortho')

            for k, (gi, gj) in enumerate(zip(sim_iy, sim_ix)):
                yr2, xr2 = ref_ys[gi], ref_xs[gj]
                acc2[yr2:yr2 + bs, xr2:xr2 + bs] += w2 * denoised2[k]
                wgt2[yr2:yr2 + bs, xr2:xr2 + bs] += w2

    Y_d = np.where(wgt2 > 0, acc2 / wgt2, pilot)

    # Chroma: Gaussian proportional to noise level (fast, avoids colour noise)
    img_scale = float(np.percentile(img, 95)) + 1e-9
    chroma_sigma_px = float(np.clip(sigma_psd / img_scale * 3.0, 0.5, 4.0))
    Cb_d = ndimage.gaussian_filter(Cb, sigma=chroma_sigma_px)
    Cr_d = ndimage.gaussian_filter(Cr, sigma=chroma_sigma_px)

    R = Y_d + 1.40200 * Cr_d
    G = Y_d - 0.34414 * Cb_d - 0.71414 * Cr_d
    B = Y_d + 1.77200 * Cb_d
    result = np.stack([R, G, B], axis=2)

    if star_mask is not None:
        mask3 = star_mask[:, :, np.newaxis]
        result = result * (1.0 - mask3) + src * mask3

    return result.astype(np.float32)


def anisotropic_diffusion(img: np.ndarray, iterations: int = 20,
                           kappa: float = 30.0, gamma: float = 0.1,
                           option: int = 1,
                           star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Perona-Malik anisotropic diffusion for edge-preserving noise reduction.

    Iterates the PDE:  ∂I/∂t = div( c(|∇I|) · ∇I )
    where the conduction coefficient c(·) inhibits diffusion across edges.

    Two conduction functions are available:
      option=1: c(d) = exp(-(d/κ)²)          — favours high-contrast edges
      option=2: c(d) = 1 / (1 + (d/κ)²)     — favours wide regions

    Unlike Gaussian smoothing, fine nebula filaments and galaxy arms whose
    gradient magnitude exceeds κ are preserved while flat-sky regions (gradients
    ≪ κ) are smoothed heavily.

    Args:
        img:        Float32 stacked image (H, W, 3).
        iterations: Number of time steps (default 20; more = smoother).
        kappa:      Gradient edge threshold in ADU (default 30).  Set to ~3×
                    sky noise for conservative structure preservation.
        gamma:      Time step; must satisfy 0 < γ ≤ 0.25 for numerical
                    stability (default 0.1).
        option:     Conduction function choice (1 or 2).
        star_mask:  Optional float mask (1 = star core) blended back.

    Returns:
        Denoised float32 image (H, W, 3).
    """
    src = img.astype(np.float64)
    gamma = float(np.clip(gamma, 1e-6, 0.25))
    result = src.copy()

    for _ in range(iterations):
        for c in range(3):
            ch = result[:, :, c]

            dN = np.roll(ch, -1, axis=0) - ch
            dS = np.roll(ch,  1, axis=0) - ch
            dE = np.roll(ch, -1, axis=1) - ch
            dW = np.roll(ch,  1, axis=1) - ch

            if option == 1:
                cN = np.exp(-(dN / kappa) ** 2)
                cS = np.exp(-(dS / kappa) ** 2)
                cE = np.exp(-(dE / kappa) ** 2)
                cW = np.exp(-(dW / kappa) ** 2)
            else:
                cN = 1.0 / (1.0 + (dN / kappa) ** 2)
                cS = 1.0 / (1.0 + (dS / kappa) ** 2)
                cE = 1.0 / (1.0 + (dE / kappa) ** 2)
                cW = 1.0 / (1.0 + (dW / kappa) ** 2)

            result[:, :, c] = ch + gamma * (cN * dN + cS * dS + cE * dE + cW * dW)

    if star_mask is not None:
        mask3 = star_mask[:, :, np.newaxis]
        result = result * (1.0 - mask3) + src * mask3

    return np.clip(result, 0.0, None).astype(np.float32)


def scnr(img: np.ndarray, amount: float = 1.0,
         target: str = 'green') -> np.ndarray:
    """Subtractive Chromatic Noise Reduction (SCNR).

    Neutralises an unwanted colour cast (most often a green bias in OSC/DSLR
    images caused by the 2:1 green-pixel Bayer pattern) by replacing each
    target-channel pixel with the smaller of its value and the per-pixel
    average of the two other channels.

    The ``amount`` parameter controls the blend between the corrected and
    original value (1.0 = full correction, 0.0 = no change):

        out = lerp(original, min(original, average_mask), amount)

    Args:
        img:    Float32 stacked image (H, W, 3).
        amount: Correction strength [0, 1] (default 1.0 = full).
        target: Which channel to neutralise: 'green' (default), 'red', or
                'blue'.

    Returns:
        Colour-corrected float32 image (H, W, 3), same dynamic range.
    """
    channel_map = {'red': 0, 'green': 1, 'blue': 2}
    tc = channel_map.get(target, 1)
    others = [i for i in range(3) if i != tc]

    src = img.astype(np.float64)
    result = src.copy()

    avg_mask = (src[:, :, others[0]] + src[:, :, others[1]]) * 0.5
    corrected = np.minimum(src[:, :, tc], avg_mask)
    result[:, :, tc] = src[:, :, tc] * (1.0 - amount) + corrected * amount

    return np.clip(result, 0.0, None).astype(np.float32)


def adaptive_mtf(img: np.ndarray,
                 target_bg: float = 0.15,
                 shadows: float = 0.0,
                 highlights: float = 1.0) -> np.ndarray:
    """Adaptive Midtone Transfer Function (MTF) auto-stretch.

    Derives the sky-background level via iterative sigma-clipping, then
    computes the midtone parameter *m* such that the background maps to
    ``target_bg`` in the output (default 15 %, matching PixInsight's
    AutoSTF convention).

    The MTF curve is:
        f(x) = (m - 1) · x / ((2m - 1) · x - m)

    which is a rational function passing through (0, 0), (m, 0.5), (1, 1).
    It amplifies faint nebulosity near the sky floor while compressing
    bright stars and galaxy cores.

    Args:
        img:       Float32 stacked image (H, W, 3), linear scale.
        target_bg: Target output level for sky background (default 0.15).
        shadows:   Black-point clip (values ≤ shadows → 0).
        highlights: White-point clip (values ≥ highlights → 1).

    Returns:
        Stretched float32 image in [0, 1] (display-ready).

    Notes:
        This is a display stretch, not a denoising step — it should be
        applied after all linear-space processing is complete.  Applying
        denoising to the output of adaptive_mtf will not work correctly.
    """
    src = img.astype(np.float64)

    # Sigma-clipped sky estimate (identical to arcsinh_stretch logic)
    flat = src.ravel()
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

    black = max(bg - bg_sigma, 0.0)
    white = float(np.percentile(img, 99.9))
    span = white - black
    if span < 1e-12:
        return np.zeros_like(img, dtype=np.float32)

    # Normalise to [0, 1]
    norm = np.clip((src - black) / span, 0.0, 1.0)

    # Apply shadow / highlight clipping
    if highlights > shadows:
        norm = np.clip((norm - shadows) / (highlights - shadows), 0.0, 1.0)

    # Compute MTF midtone parameter m such that f(bg_norm) = target_bg
    bg_norm = float(np.clip((bg - black) / span, 1e-6, 0.9999))
    bg_norm = float(np.clip((bg_norm - shadows) / max(highlights - shadows, 1e-9), 1e-6, 0.9999))
    # Solve: target = (m-1)*bg_n / ((2m-1)*bg_n - m)
    # → m * (target*(2*bg_n - 1) - bg_n + 1) = target * bg_n
    # → m = target * bg_n / (target*(2*bg_n - 1) - bg_n + 1)  [when denominator ≠ 0]
    tb = float(np.clip(target_bg, 0.01, 0.49))
    denom = tb * (2.0 * bg_norm - 1.0) - bg_norm + 1.0
    if abs(denom) < 1e-9 or bg_norm < 1e-6:
        m = 0.5
    else:
        m = float(np.clip(tb * bg_norm / denom, 0.001, 0.999))

    # Apply MTF: f(x) = (m-1)*x / ((2m-1)*x - m)
    denom_map = (2.0 * m - 1.0) * norm - m
    # Avoid division by zero (occurs at x=m/(2m-1))
    safe_denom = np.where(np.abs(denom_map) < 1e-12, 1e-12, denom_map)
    stretched = (m - 1.0) * norm / safe_denom

    # Boundary enforcement: x=0 → 0, x=1 → 1
    stretched = np.where(norm <= 0.0, 0.0, np.where(norm >= 1.0, 1.0, stretched))

    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def arcsinh_stretch(img: np.ndarray, factor: Optional[float] = None,
                    black_point: Optional[float] = None,
                    white_point: Optional[float] = None) -> np.ndarray:
    """Non-linear arcsinh stretch with sigma-clipped sky background estimation.

    Estimates the true sky background via iterative sigma-clipping, sets it as
    the black point, then auto-tunes the arcsinh factor so the sky maps to a
    target display level (~15 %).  This preserves faint nebulosity and avoids
    the flat, grey-sky look produced by simple percentile clipping.

    When black_point and white_point are provided (e.g. pre-computed from
    luminance), the per-channel stats step is skipped so all channels share
    the same normalization range, preserving cross-channel color ratios.
    """
    if black_point is None or white_point is None:
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
        black_point = max(bg - 1.0 * bg_sigma, 0.0)
        white_point = float(np.percentile(img, 99.8))
    else:
        bg = black_point  # used below for factor auto-tuning
        bg_sigma = 0.0
    span = white_point - black_point
    if span < 1e-12:
        return np.zeros_like(img)

    norm = np.clip((img - black_point) / span, 0.0, 1.0)

    # Auto-tune arcsinh factor so sky maps to ~15 % of output range
    if factor is None:
        target_bg = 0.15
        bg_norm = float(np.clip((bg - black_point) / span, 1e-6, 1.0))
        factor = getattr(Config, 'ARCSINH_STRETCH_FACTOR', 10.0)
        for f in (3.0, 5.0, 10.0, 20.0, 50.0, 100.0):
            if np.arcsinh(bg_norm * f) / np.arcsinh(f) >= target_bg:
                factor = f
                break

    stretched = np.arcsinh(norm * factor) / np.arcsinh(factor)
    return np.clip(stretched, 0.0, 1.0)
