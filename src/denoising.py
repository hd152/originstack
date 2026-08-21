"""Denoising and image processing: wavelet, bilateral, NLM, local normalize, arcsinh."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from src.models import Config
from src.utils import safe_print, get_logger
from src.background import _estimate_sky_sigma, gaussian_filter_ds
from src import wavelet
_log = get_logger()

# Optional native (Rust) kernels — graceful degradation to numpy if absent.
try:
    import astro_native as _native
    _HAS_NATIVE = True
except Exception:
    _native = None
    _HAS_NATIVE = False

try:
    import bm3d as _bm3d_pkg
    HAS_BM3D_PKG = True
except Exception:
    _bm3d_pkg = None  # type: ignore[assignment]
    HAS_BM3D_PKG = False

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


def adaptive_wavelet_denoise(img: np.ndarray, levels: int = 4,
                              chroma_factor: float = 2.0,
                              star_mask: Optional[np.ndarray] = None,
                              variance_stabilize: bool = False) -> np.ndarray:
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
        levels: Maximum decomposition depth (default 4).
        chroma_factor: Multiplier applied to the noise estimate for the Cb/Cr
                       chroma channels (default 2.0).  Higher values remove more
                       colour speckle at the cost of slight chroma blurring.
        star_mask: Optional float mask (0–1, 1 = star core).  Star pixels are
                   blended back from the original to avoid core softening.

    Returns:
        Denoised float32 image (H, W, 3).
    """
    h, w = img.shape[0], img.shape[1]
    src = img.astype(np.float64)

    # RGB -> YCbCr (ITU-R BT.601)
    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    def _adaptive_denoise_plane(plane: np.ndarray, chroma_mult: float) -> np.ndarray:
        max_level = wavelet.dwt_max_level(min(plane.shape))
        use_levels = min(levels, max_level)
        if use_levels < 1:
            return plane

        coeffs = wavelet.wavedec2(plane, use_levels)

        # Global noise estimate from finest-level HH subband (standard MAD estimator)
        sigma_noise = np.median(np.abs(coeffs[-1][-1])) / 0.6745
        sigma_noise = max(sigma_noise * chroma_mult, 1e-12)

        new_coeffs = [coeffs[0]]  # keep approximation coefficients unchanged
        for detail_level in coeffs[1:]:
            new_detail = []
            for d in detail_level:
                threshold = _bayesshrink_threshold(d, sigma_noise)
                new_detail.append(wavelet.soft_threshold(d, threshold))
            new_coeffs.append(tuple(new_detail))

        return wavelet.waverec2(new_coeffs)[:h, :w]

    if variance_stabilize:
        # See wavelet_denoise's identical guard for why luma only: it's the
        # plane whose noise is genuinely photon-limited, which is what the
        # generalized Anscombe transform's Poisson+Gaussian model assumes.
        gain, sigma = _estimate_noise_level_function(Y)
        Y_stab = _generalized_anscombe(np.maximum(Y, 0.0), gain, sigma)
        Y_d = _inverse_generalized_anscombe(
            _adaptive_denoise_plane(Y_stab, 1.0), gain, sigma)
    else:
        Y_d = _adaptive_denoise_plane(Y, 1.0)
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


def _structure_tensor_coherence(plane: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Local structure-tensor coherence, ``(lambda1-lambda2)/(lambda1+lambda2)``
    of the Gaussian-windowed gradient outer-product tensor. Near 1 on a
    straight edge/filament (one dominant local gradient direction), near 0
    on isotropic structure (noise, point-like blobs, flat sky) -- the
    standard anisotropy measure ``anisotropic_diffusion`` doesn't compute
    explicitly (it diffuses by a conductance function of gradient
    *magnitude* alone, not orientation coherence).
    """
    gy, gx = np.gradient(plane.astype(np.float64))
    jxx = ndimage.gaussian_filter(gx * gx, sigma)
    jyy = ndimage.gaussian_filter(gy * gy, sigma)
    jxy = ndimage.gaussian_filter(gx * gy, sigma)
    trace = jxx + jyy
    disc = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy ** 2, 0.0))
    lam1 = 0.5 * (trace + disc)
    lam2 = 0.5 * (trace - disc)
    denom = lam1 + lam2
    coherence = np.where(denom > 1e-12, (lam1 - lam2) / np.maximum(denom, 1e-12), 0.0)
    return np.clip(coherence, 0.0, 1.0)


def _resize_to(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    zy, zx = shape[0] / arr.shape[0], shape[1] / arr.shape[1]
    return ndimage.zoom(arr, (zy, zx), order=1)


def directional_wavelet_denoise(img: np.ndarray, levels: int = 4,
                                chroma_factor: float = 2.0,
                                star_mask: Optional[np.ndarray] = None,
                                protect_strength: float = 0.6,
                                coherence_sigma: float = 1.5) -> np.ndarray:
    """Directional (curvelet/shearlet-*inspired*) adaptive wavelet
    denoising -- ``--denoiser curvelet``.

    Plain BayesShrink (``adaptive_wavelet_denoise``) applies one threshold
    per wavelet subband uniformly across the whole plane: isotropic in
    space. Curvelets/shearlets instead use genuinely directional basis
    functions, so elongated structure (nebula filaments, galaxy arms)
    survives thresholding better than an isotropic wavelet basis naturally
    allows. This function approximates that practical benefit WITHOUT
    implementing a full ridgelet/Radon-based transform or a perfect-
    reconstruction directional filter bank: it computes a per-pixel
    structure-tensor coherence map (``_structure_tensor_coherence`` -- near
    1 on a straight edge/filament, near 0 on isotropic noise or point-like
    blobs), resizes it to match each decomposition level's detail-subband
    resolution, and locally *reduces* the BayesShrink threshold wherever
    coherence is high -- still this project's own validated wavelet
    transform and per-subband noise estimate (``_bayesshrink_threshold``),
    just made spatially adaptive instead of one scalar per subband.

    Deliberately NOT named a claim of implementing curvelets/shearlets
    themselves -- named for the practical goal it approximates. Applied to
    luma only (chroma channels get the same uniform BayesShrink
    ``adaptive_wavelet_denoise`` already uses -- chroma structure isn't
    what this is meant to protect).

    ``protect_strength``: 0 = falls back to plain uniform BayesShrink
    (protection off); clamped to at most 0.95 so even maximally-coherent
    pixels retain a little thresholding (a thin linear artifact -- a
    satellite trail sliver, a hot column -- is also "coherent" by this
    measure, and shouldn't pass through completely untouched).
    """
    h, w = img.shape[0], img.shape[1]
    src = img.astype(np.float64)
    protect_strength = float(np.clip(protect_strength, 0.0, 0.95))

    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    coherence = _structure_tensor_coherence(Y, sigma=coherence_sigma)

    def _denoise_plane(plane, chroma_mult, use_coherence):
        max_level = wavelet.dwt_max_level(min(plane.shape))
        use_levels = min(levels, max_level)
        if use_levels < 1:
            return plane
        coeffs = wavelet.wavedec2(plane, use_levels)
        sigma_noise = np.median(np.abs(coeffs[-1][-1])) / 0.6745
        sigma_noise = max(sigma_noise * chroma_mult, 1e-12)

        new_coeffs = [coeffs[0]]
        for detail_level in coeffs[1:]:
            new_detail = []
            for d in detail_level:
                base_threshold = _bayesshrink_threshold(d, sigma_noise)
                if use_coherence and np.isfinite(base_threshold):
                    coh = _resize_to(coherence, d.shape)
                    local_threshold = np.maximum(
                        base_threshold * (1.0 - protect_strength * coh), 0.0)
                else:
                    local_threshold = base_threshold
                new_detail.append(wavelet.soft_threshold(d, local_threshold))
            new_coeffs.append(tuple(new_detail))
        return wavelet.waverec2(new_coeffs)[:h, :w]

    Y_d  = _denoise_plane(Y,  1.0, True)
    Cb_d = _denoise_plane(Cb, chroma_factor, False)
    Cr_d = _denoise_plane(Cr, chroma_factor, False)

    R = Y_d + 1.40200 * Cr_d
    G = Y_d - 0.34414 * Cb_d - 0.71414 * Cr_d
    B = Y_d + 1.77200 * Cb_d
    result = np.stack([R, G, B], axis=2)

    if star_mask is not None:
        mask3 = star_mask[:, :, np.newaxis]
        result = result * (1.0 - mask3) + src * mask3

    return result.astype(np.float32)


def _estimate_noise_level_function(plane: np.ndarray, tile: int = 16) -> Tuple[float, float]:
    """Estimate an approximate (gain, read_noise_sigma) pair from the
    plane's own local mean-variance relationship (a lightweight photon
    transfer curve fit), so the generalized Anscombe transform below
    doesn't need the caller to supply exact sensor calibration data.

    Splits the plane into tiles, takes each tile's (mean, variance) as one
    sample, and fits ``variance ~= (1/gain) * mean + read_noise_sigma^2`` by
    least squares restricted to the lower half of tiles by mean brightness
    -- background-dominated tiles follow the shot-noise relationship;
    star/nebula-structure tiles have inflated variance from real signal,
    not noise, and would bias the fit if included.
    """
    h, w = plane.shape
    ny, nx = h // tile, w // tile
    if ny * nx < 10:
        return 1.0, 0.0  # too few tiles to fit -- identity-ish transform

    # Vectorized block-reduce: crop to the largest exact-multiple-of-tile
    # region (a trailing partial-tile strip, if any, is dropped -- same
    # effect as the plain Python double loop this replaced, which skipped
    # any undersized trailing patch via a size check), then reshape to
    # (ny, tile, nx, tile) and reduce over the two tile axes in one call --
    # no per-tile Python loop over what can be thousands of tiles on a
    # full-resolution frame.
    cropped = plane[:ny * tile, :nx * tile]
    blocks = cropped.reshape(ny, tile, nx, tile)
    means_arr = blocks.mean(axis=(1, 3)).ravel()
    varis_arr = blocks.var(axis=(1, 3)).ravel()
    order = np.argsort(means_arr)
    means_arr, varis_arr = means_arr[order], varis_arr[order]
    cut = max(10, len(means_arr) // 2)
    m_bg, v_bg = means_arr[:cut], varis_arr[:cut]
    if np.ptp(m_bg) < 1e-6:
        return 1.0, float(max(np.median(v_bg), 0.0))

    slope, intercept = np.polyfit(m_bg, v_bg, 1)
    gain = 1.0 / max(slope, 1e-6)
    read_var = max(intercept, 0.0)
    return gain, float(np.sqrt(read_var))


def _generalized_anscombe(x: np.ndarray, gain: float, sigma: float) -> np.ndarray:
    """Forward generalized Anscombe transform: maps a Poisson(shot noise,
    scaled by ``gain``) + Gaussian(``sigma``) signal to one with
    approximately unit variance everywhere, regardless of brightness --
    the assumption BayesShrink's single per-subband threshold estimate
    (from the finest detail subband's MAD) actually needs to hold.
    """
    return (2.0 / gain) * np.sqrt(np.maximum(gain * x + 0.375 * gain ** 2 + sigma ** 2, 0.0))


def _inverse_generalized_anscombe(z: np.ndarray, gain: float, sigma: float) -> np.ndarray:
    """Algebraic (exact-inverse-of-the-forward-map) inverse of
    ``_generalized_anscombe``. Not the "optimal unbiased inverse" (Makitalo
    & Foi 2011), which needs a precomputed correction table -- the plain
    algebraic inverse is a standard, simpler approximation, adequate at
    the moderate-to-high SNR this pipeline's stacked images sit at, with a
    small known bias only at very low counts (per the same literature).
    """
    return ((z * gain / 2.0) ** 2 - 0.375 * gain ** 2 - sigma ** 2) / gain


def wavelet_denoise(img: np.ndarray, levels: int = 4, threshold_factor: float = 3.0,
                    chroma_factor: float = 2.0,
                    star_mask: Optional[np.ndarray] = None,
                    variance_stabilize: bool = False) -> np.ndarray:
    """Multi-scale wavelet denoising with luma/chroma split and star protection.

    Operates in YCbCr colour space so that chroma channels (Cb, Cr) can receive
    a stronger threshold (chroma_factor x threshold_factor) while luminance is
    handled conservatively.  This removes colour speckle in sky background more
    aggressively without softening fine luminance structure in nebulae.

    If star_mask is provided (float [0,1], 1=star core), the denoised result is
    blended back with the original at star positions so that star cores are not
    softened and their colours are preserved.
    """
    h, w = img.shape[0], img.shape[1]
    src = img.astype(np.float64)

    # RGB -> YCbCr (ITU-R BT.601 coefficients)
    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    def _denoise_plane(plane, factor):
        max_level = wavelet.dwt_max_level(min(plane.shape))
        use_levels = min(levels, max_level)
        if use_levels < 1:
            return plane
        coeffs = wavelet.wavedec2(plane, use_levels)
        detail_hh = coeffs[-1][-1]
        sigma_noise = np.median(np.abs(detail_hh)) / 0.6745
        threshold = factor * sigma_noise
        new_coeffs = [coeffs[0]]
        for detail_level in coeffs[1:]:
            new_coeffs.append(tuple(
                wavelet.soft_threshold(d, threshold) for d in detail_level
            ))
        return wavelet.waverec2(new_coeffs)[:h, :w]

    chroma_thresh = threshold_factor * chroma_factor
    if variance_stabilize:
        # Only luma: it's the plane whose noise is genuinely photon-limited
        # (shot noise from the actual signal), which is what the generalized
        # Anscombe transform's Poisson+Gaussian model assumes. Cb/Cr are
        # differences of positive quantities, not counts, so GAT doesn't
        # have the same physical grounding there -- left as-is.
        gain, sigma = _estimate_noise_level_function(Y)
        Y_stab = _generalized_anscombe(np.maximum(Y, 0.0), gain, sigma)
        Y_d = _inverse_generalized_anscombe(
            _denoise_plane(Y_stab, threshold_factor), gain, sigma)
    else:
        Y_d = _denoise_plane(Y, threshold_factor)
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
    img_max = float(img.max())
    if img_max < 1e-12:
        return img

    if sigma_color is None:
        sigma_color = _estimate_sky_sigma(img)
    sigma_color = float(sigma_color)

    # Neighbourhood radius derived from sigma_space (was cv2's d=-1 auto rule);
    # clamped to avoid extreme runtimes on large sigma_space values.
    radius = min(int(round(3.0 * sigma_space)), 10)  # max 21x21 window

    img_f32 = img.astype(np.float32)
    if _HAS_NATIVE and hasattr(_native, 'bilateral_filter'):
        try:
            return np.asarray(_native.bilateral_filter(
                np.ascontiguousarray(img_f32), sigma_color, float(sigma_space), radius))
        except Exception:
            pass
    return _bilateral_filter_numpy(img_f32, sigma_color, float(sigma_space), radius)


def _bilateral_filter_numpy(img: np.ndarray, sigma_color: float, sigma_space: float,
                            radius: int) -> np.ndarray:
    """Joint (colour-space) bilateral filter, pure numpy -- native Rust dispatch
    happens one level up in ``bilateral_denoise``. Vectorised over the whole
    image per kernel tap ((2*radius+1)^2 iterations, each an O(H*W*C) numpy op)
    rather than a per-pixel python loop -- the same tap-loop pattern as the
    Malvar debayer's numpy fallback, just with a runtime-sized window instead
    of a fixed 5x5. The colour-similarity weight uses the joint Euclidean
    distance across all 3 channels per neighbour (like cv2.bilateralFilter's
    multi-channel mode), not independent per-channel weights, so it doesn't
    introduce colour fringing at edges.
    """
    H, W, C = img.shape
    img64 = img.astype(np.float64)
    padded = np.pad(img64, ((radius, radius), (radius, radius), (0, 0)), mode='reflect')

    acc = np.zeros((H, W, C), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)
    inv_2s2 = 1.0 / (2.0 * sigma_space * sigma_space)
    inv_2c2 = 1.0 / (2.0 * sigma_color * sigma_color)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[radius + dy:radius + dy + H, radius + dx:radius + dx + W, :]
            spatial_w = np.exp(-(dy * dy + dx * dx) * inv_2s2)
            color_dist2 = np.sum((neighbor - img64) ** 2, axis=-1)
            w = spatial_w * np.exp(-color_dist2 * inv_2c2)
            acc += neighbor * w[:, :, np.newaxis]
            wsum += w

    return (acc / np.maximum(wsum[:, :, np.newaxis], 1e-12)).astype(np.float32)


def _nlm_patch_distance(diff2_outer: np.ndarray, patch_size: int, H: int, W: int,
                        m: int) -> np.ndarray:
    """Sum of a per-pixel squared-difference map over a ``patch_size x
    patch_size`` window, for every pixel of the central (H, W) region.

    ``diff2_outer`` must already be the (H+2m, W+2m) map (m = patch_size//2)
    so the box filter never needs its own boundary handling: any of
    scipy's filter modes agree exactly once cropped back to the central
    (H, W) region, since the crop removes precisely the filter_radius-wide
    border where boundary mode would otherwise matter."""
    boxed = ndimage.uniform_filter(diff2_outer, size=patch_size, mode='nearest')
    return boxed[m:m + H, m:m + W] * (patch_size * patch_size)


def _nlm_denoise_numpy(img: np.ndarray, h: float, patch_size: int,
                       patch_distance: int) -> np.ndarray:
    """Fast non-local means, jointly across channels -- Darbon et al.'s
    box-filter acceleration of the Buades-Coll-Morel NL-means algorithm
    (the same algorithm family skimage.restoration.denoise_nl_means's own
    docstring cites for its fast_mode). This is a faithful reimplementation
    of the *published* algorithm, not a bit-exact port of skimage's fast
    path: that path is a compiled Cython kernel with no .pyx source shipped
    in the installed package (only a compiled .pyd), so there was nothing
    to port literally, unlike phase_correlate.py/wavelet.py where the
    reference math is fully specified. Validated for equivalent denoising
    *behaviour* against real skimage instead (comparable noise reduction,
    edge preservation, and monotonic response to h) -- see
    tests/test_denoising.py.

    Patch distance is the mean squared per-pixel-per-channel difference
    over a (patch_size x patch_size) window (via a box filter, computed
    once per candidate shift rather than per pixel -- the source of the
    "fast" complexity class: image.size * patch_distance**2). The exact
    self-match (zero shift) is excluded from the main weighted sum and
    re-added with weight = the max weight of every other candidate for
    that pixel, the standard NL-means fix (Buades, Coll & Morel, IPOL
    2011) for the degenerate case where the trivial self-distance-0 match
    would otherwise always win outright (weight=1, the maximum possible)
    and suppress denoising.
    """
    H, W, C = img.shape
    m = patch_size // 2
    d = int(patch_distance)
    pad = d + m
    img64 = img.astype(np.float64)
    padded = np.pad(img64, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    base = pad - m
    a_outer = padded[base:base + H + 2 * m, base:base + W + 2 * m, :]

    patch_area = patch_size * patch_size
    h2 = max(h, 1e-12) ** 2
    norm = C * h2

    acc = np.zeros((H, W, C), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)
    wmax = np.zeros((H, W), dtype=np.float64)

    for dy in range(-d, d + 1):
        for dx in range(-d, d + 1):
            if dy == 0 and dx == 0:
                continue
            b_outer = padded[base + dy:base + dy + H + 2 * m,
                             base + dx:base + dx + W + 2 * m, :]
            diff2_outer = np.sum((a_outer - b_outer) ** 2, axis=-1)
            dist = _nlm_patch_distance(diff2_outer, patch_size, H, W, m) / patch_area
            w = np.exp(-dist / norm)
            neighbor = padded[pad + dy:pad + dy + H, pad + dx:pad + dx + W, :]
            acc += neighbor * w[:, :, np.newaxis]
            wsum += w
            np.maximum(wmax, w, out=wmax)

    # Re-add the excluded self-match with the standard max-weight fix.
    center = padded[pad:pad + H, pad:pad + W, :]
    acc += center * wmax[:, :, np.newaxis]
    wsum += wmax

    return (acc / np.maximum(wsum[:, :, np.newaxis], 1e-12)).astype(np.float32)


def nlm_denoise(img: np.ndarray, h: float = 1.0,
                patch_size: int = 5, patch_distance: int = 7,
                blend: float = 0.5) -> np.ndarray:
    """Non-local means denoising for faint extended nebulosity.

    Searches for similar patches across the image and averages them, which
    smooths large featureless sky and faint nebula regions while preserving
    sharp edges like galaxy arms and star-forming filaments.

    Native/numpy fast NL-means (_nlm_denoise_numpy, no external dependency --
    see its docstring for the algorithm and validation notes).

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

    img_norm = img_ped.astype(np.float32) / ped_max
    h_norm = h * sky_sigma / ped_max
    denoised = _nlm_denoise_numpy(
        img_norm, h=h_norm, patch_size=patch_size, patch_distance=patch_distance)
    nlm_result = denoised.astype(np.float64) * ped_max - pedestal
    result = blend * nlm_result + (1.0 - blend) * img_f64
    return result.astype(np.float32)


def _median_filter_fast(plane: np.ndarray, ksize: int) -> np.ndarray:
    """Median filter: native Rust fast path (any odd size), scipy fallback.

    Args:
        plane: 2-D float64 array.
        ksize: Odd kernel side length (3, 5, 9, 17, …).

    Returns:
        Median-filtered float64 array of the same shape.
    """
    # Native path first: rayon-parallel, interior fast path, any odd size.
    # Median is an order statistic, so filtering the f32-cast plane selects
    # the same sample values the f64 filter would (input images are f32;
    # only the YCbCr mixing introduces sub-f32 bits, ~1e-7 relative).
    if _HAS_NATIVE:
        try:
            return _native.median_filter_native(
                np.ascontiguousarray(plane, dtype=np.float32),
                int(ksize)).astype(np.float64)
        except Exception:
            pass
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

    if _HAS_NATIVE:
        safe_print(f"    [rust] MMT median cascade ({levels} levels x 3 planes)")

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


def reduce_chroma_noise(img: np.ndarray, sigma: float = 2.0,
                        sigma_large: float = 0.0,
                        large_strength: float = 0.7,
                        star_mask: Optional[np.ndarray] = None) -> np.ndarray:
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

    ``star_mask`` (optional): a real per-pixel star-PSF mask (e.g. from
    ``generate_star_mask``, 1 = star core, smooth falloff). When given, this
    replaces the luminance-threshold heuristic for what counts as "protected"
    -- the heuristic flags *any* pixel brighter than sky+3sigma, which for a
    galaxy or bright nebula target means the entire object is treated as one
    giant protected star and never receives chroma smoothing at all (visible
    as an under-denoised speckled/mottled texture across the whole target,
    while the surrounding sky is clean). A real star mask only protects
    actual point sources, so extended structure still gets smoothed.
    """
    lum = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1]
           + 0.114 * img[:, :, 2]).astype(np.float64)

    if star_mask is not None:
        protect = np.clip(np.asarray(star_mask, dtype=np.float64), 0.0, 1.0)
    else:
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

    # Optional coarse pass — smooths medium-scale colour blotches (walking /
    # chroma-noise mottle, tens of px) that the fine pass leaves untouched.
    # Same object masking, so star/galaxy colour is preserved; only sky chroma
    # is flattened, blended in by large_strength.
    do_large = sigma_large > 0.0
    if do_large:
        weight_large = np.maximum(
            gaussian_filter_ds(sky_mask, sigma=sigma_large), 1e-9)
        blend_large = np.clip(sky_mask * float(large_strength), 0.0, 1.0)

    for c in range(img.shape[2]):
        chroma = img[:, :, c].astype(np.float64) - lum
        # Weighted blur: star pixels contribute 0, background contributes 1
        smooth_chroma = ndimage.gaussian_filter(chroma * sky_mask, sigma=sigma) / safe_weight
        # Stars keep original chroma; background gets smoothed chroma
        out_chroma = chroma * protect + smooth_chroma * sky_mask
        if do_large:
            coarse = (gaussian_filter_ds(out_chroma * sky_mask,
                                         sigma=sigma_large) / weight_large)
            out_chroma = out_chroma * (1.0 - blend_large) + coarse * blend_large
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
        black_point = bg - 1.0 * bg_sigma
        white_point = float(np.percentile(img, 99.9))
    span = white_point - black_point
    if span < 1e-12 or white_point <= 0.0:
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
    if sigma_psd <= 0.0:
        sigma_psd = float(_estimate_sky_sigma(img))
    if sigma_psd < 1e-9:
        return img.copy()

    # Fast path: use the bm3d package when installed (significantly faster than
    # the pure-scipy DCT fallback below and handles all stages in one call).
    if HAS_BM3D_PKG:
        try:
            src_f = img.astype(np.float64)
            img_max = float(src_f.max()) or 1.0
            Y_raw = (0.29900 * src_f[:, :, 0] + 0.58700 * src_f[:, :, 1]
                     + 0.11400 * src_f[:, :, 2])
            Y_norm = Y_raw / img_max
            sigma_norm = sigma_psd / img_max
            Y_denoised = _bm3d_pkg.bm3d(
                Y_norm,
                sigma_psd=sigma_norm,
                stage_arg=_bm3d_pkg.BM3DStages.ALL_STAGES,
            )
            Y_d = Y_denoised * img_max
            # Chroma: mild Gaussian smoothing proportional to noise level
            Cb = -0.16875 * src_f[:, :, 0] - 0.33126 * src_f[:, :, 1] + 0.50000 * src_f[:, :, 2]
            Cr =  0.50000 * src_f[:, :, 0] - 0.41869 * src_f[:, :, 1] - 0.08131 * src_f[:, :, 2]
            chroma_sigma = max(1.0, sigma_psd / img_max * 3.0)
            Cb_d = ndimage.gaussian_filter(Cb, sigma=chroma_sigma)
            Cr_d = ndimage.gaussian_filter(Cr, sigma=chroma_sigma)
            R = Y_d + 1.40200 * Cr_d
            G = Y_d - 0.34414 * Cb_d - 0.71414 * Cr_d
            B = Y_d + 1.77200 * Cb_d
            result = np.clip(np.stack([R, G, B], axis=2), 0.0, None).astype(np.float32)
            if star_mask is not None:
                mask3 = star_mask[:, :, np.newaxis]
                result = (result * (1.0 - mask3) + img * mask3).astype(np.float32)
            return result
        except Exception:
            pass  # fall through to pure-scipy DCT implementation

    try:
        from scipy.fft import dctn, idctn
    except ImportError:
        from scipy.fftpack import dctn, idctn

    src = img.astype(np.float64)
    H, W = src.shape[:2]

    if stride is None:
        stride = 8 if max(H, W) > 1500 else 4

    # YCbCr decomposition
    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    bs = block_size
    sw = search_window

    Y_d = None
    if _HAS_NATIVE and hasattr(_native, 'bm3d_denoise_native'):
        try:
            Y_d = np.asarray(_native.bm3d_denoise_native(
                np.ascontiguousarray(Y, dtype=np.float64),
                int(bs), int(stride), int(sw), int(group_size), float(sigma_psd)))
        except Exception:
            Y_d = None

    if Y_d is None:
        Y_d = _bm3d_step12_numpy(Y, bs, stride, sw, group_size, sigma_psd, dctn, idctn)

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


def _bm3d_step12_numpy(Y: np.ndarray, bs: int, stride: int, sw: int, group_size: int,
                       sigma_psd: float, dctn, idctn) -> np.ndarray:
    """Pure-scipy-DCT Step1 (hard-threshold) + Step2 (Wiener) BM3D core --
    the numpy fallback for bm3d_denoise when the native kernel is
    unavailable. See bm3d_denoise_native in ext/astro_native/src/lib.rs for
    the ported version (same algorithm, no behaviour change)."""
    H, W = Y.shape

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

    return np.where(wgt2 > 0, acc2 / wgt2, pilot)


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

    # Native fast path (Rust): identical Jacobi iteration with periodic boundary.
    if _HAS_NATIVE and img.ndim == 3 and img.shape[2] == 3:
        try:
            result = _native.anisotropic_diffusion(
                np.ascontiguousarray(img, dtype=np.float32),
                int(iterations), float(kappa), float(gamma), int(option))
            safe_print(f"    [rust] anisotropic diffusion ({iterations} iters)")
            if star_mask is not None:
                mask3 = star_mask[:, :, np.newaxis]
                result = result * (1.0 - mask3) + src * mask3
            return np.clip(result, 0.0, None).astype(np.float32)
        except Exception as _exc:
            _log.debug("native anisotropic_diffusion failed (%s); using numpy", _exc)

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
        black_point = bg - 1.0 * bg_sigma
        white_point = float(np.percentile(img, 99.8))
    else:
        bg = black_point  # used below for factor auto-tuning
        bg_sigma = 0.0
    span = white_point - black_point
    if span < 1e-12 or white_point <= 0.0:
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


def remove_star_halos(img: np.ndarray, star_sources, fwhm: float,
                      protection_radius: float = 2.0) -> np.ndarray:
    """Fit and subtract Gaussian PSF halos from bright stars.

    For each bright star (above 95th percentile of flux), fits a scaled Gaussian
    and subtracts the predicted halo beyond protection_radius * fwhm from center.
    """
    if star_sources is None or len(star_sources) == 0 or fwhm <= 0:
        return img

    H, W = img.shape[:2]
    result = img.copy()

    try:
        fluxes = np.asarray(star_sources['flux'], dtype=np.float64)
    except (KeyError, TypeError):
        return img

    flux_thresh = float(np.percentile(fluxes, 95))
    bright_mask = fluxes >= flux_thresh
    bright_stars = star_sources[bright_mask]

    if len(bright_stars) == 0:
        return img

    sigma = fwhm / 2.355
    protect_radius_px = protection_radius * fwhm

    lum = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])

    for star in bright_stars:
        try:
            yc = int(round(float(star['ycentroid'])))
            xc = int(round(float(star['xcentroid'])))
        except (KeyError, TypeError):
            continue

        if yc < 0 or yc >= H or xc < 0 or xc >= W:
            continue

        r_cut = int(min(protect_radius_px * 3, 50))
        y0, y1 = max(0, yc - r_cut), min(H, yc + r_cut + 1)
        x0, x1 = max(0, xc - r_cut), min(W, xc + r_cut + 1)
        if y1 <= y0 or x1 <= x0:
            continue

        cut_lum = lum[y0:y1, x0:x1]
        peak = float(cut_lum.max())
        bg = float(np.percentile(cut_lum, 25))
        if peak - bg < 10.0:
            continue

        r_work = int(5 * sigma) + r_cut
        wy0, wy1 = max(0, yc - r_work), min(H, yc + r_work + 1)
        wx0, wx1 = max(0, xc - r_work), min(W, xc + r_work + 1)

        yy, xx = np.mgrid[wy0:wy1, wx0:wx1]
        dist2 = (yy - yc) ** 2 + (xx - xc) ** 2
        gaussian = (peak - bg) * np.exp(-dist2 / (2 * sigma ** 2))

        protect_mask = dist2 < protect_radius_px ** 2
        halo_subtract = np.where(protect_mask, 0.0, gaussian)

        for c in range(img.shape[2] if img.ndim == 3 else 1):
            if img.ndim == 3:
                result[wy0:wy1, wx0:wx1, c] = np.clip(
                    result[wy0:wy1, wx0:wx1, c] - halo_subtract, 0.0, None)
            else:
                result[wy0:wy1, wx0:wx1] = np.clip(
                    result[wy0:wy1, wx0:wx1] - halo_subtract, 0.0, None)

    return result.astype(img.dtype)


# ---------------------------------------------------------------------------
# Comet-specific filters
# ---------------------------------------------------------------------------

def radial_renormalize(img: np.ndarray, nucleus_y: float, nucleus_x: float,
                       smooth_sigma: float = 20.0, n_bins: int = 200) -> np.ndarray:
    """Radial renormalization filter for comet coma structure enhancement.

    Divides the image by a radially-smoothed profile centred on the nucleus,
    flattening the steep coma gradient to reveal jets and fine structure.

    Args:
        img:          Float32 (H, W, 3) or (H, W) stacked image.
        nucleus_y:    Row coordinate of the comet nucleus.
        nucleus_x:    Column coordinate of the comet nucleus.
        smooth_sigma: Gaussian sigma for smoothing the radial profile (degrees
                      of the radial bin profile, not pixels).
        n_bins:       Number of radial bins for the profile estimate.

    Returns:
        Float32 image of the same shape with the coma gradient flattened.
    """
    ndim_orig = img.ndim
    if ndim_orig == 2:
        img = img[:, :, np.newaxis]

    H, W, C = img.shape
    img_f = img.astype(np.float64)

    # Build radial distance map
    yy, xx = np.mgrid[:H, :W]
    radii = np.sqrt((yy - nucleus_y) ** 2 + (xx - nucleus_x) ** 2).astype(np.float64)
    max_radius = float(radii.max())
    if max_radius < 1.0:
        out = img_f.astype(np.float32)
        if ndim_orig == 2:
            out = out[:, :, 0]
        return out

    # Process each channel independently
    result = np.zeros_like(img_f)
    for c in range(C):
        channel = img_f[:, :, c]
        # Build radial profile: median per bin. The numpy reference rebuilds
        # a boolean mask over the whole image once per bin (n_bins full-image
        # passes) just to select each bin's pixels; the native kernel buckets
        # every pixel by radial bin in one O(H*W) pass instead.
        if _HAS_NATIVE and hasattr(_native, 'radial_bin_median'):
            profile = np.asarray(_native.radial_bin_median(
                np.ascontiguousarray(radii, dtype=np.float64),
                np.ascontiguousarray(channel, dtype=np.float64),
                max_radius, n_bins))
        else:
            bin_edges = np.linspace(0.0, max_radius + 1.0, n_bins + 1)
            profile = np.zeros(n_bins, dtype=np.float64)
            for b in range(n_bins):
                in_bin = (radii >= bin_edges[b]) & (radii < bin_edges[b + 1])
                if in_bin.any():
                    profile[b] = float(np.median(channel[in_bin]))

        # Smooth the profile
        from scipy.ndimage import gaussian_filter1d
        profile_smooth = gaussian_filter1d(profile, sigma=smooth_sigma)

        # Interpolate profile to full image
        bin_indices = np.clip(
            ((radii / max_radius) * (n_bins - 1)).astype(int), 0, n_bins - 1
        )
        model = profile_smooth[bin_indices]

        # Scale: protect against near-zero model values
        scale = np.where(model > 1e-12, model, 1e-12)
        # Preserve overall brightness: multiply by mean model
        mean_model = float(np.mean(profile_smooth[profile_smooth > 1e-12])) if np.any(profile_smooth > 1e-12) else 1.0
        renormed = (channel / scale) * mean_model
        result[:, :, c] = renormed

    out = result.astype(np.float32)
    if ndim_orig == 2:
        out = out[:, :, 0]
    return out


def larson_sekanina(img: np.ndarray, nucleus_y: float, nucleus_x: float,
                    rotation_deg: float = 15.0, dr: float = 0.0) -> np.ndarray:
    """Larson-Sekanina rotational difference filter for comet jet detection.

    Subtracts a rotationally-shifted copy of the image from the original,
    revealing asymmetric jet structure in the coma.

    Args:
        img:          Float32 (H, W, 3) or (H, W) image.
        nucleus_y:    Row coordinate of the comet nucleus (rotation centre).
        nucleus_x:    Column coordinate of the comet nucleus (rotation centre).
        rotation_deg: Rotation angle in degrees for the difference.
        dr:           Optional radial shift of the rotated copy in pixels
                      (positive = away from nucleus).

    Returns:
        Float32 image of the same shape with jets enhanced.
    """
    try:
        from scipy.ndimage import rotate as _rotate, shift as _shift
    except ImportError:
        safe_print("  WARNING: larson_sekanina requires scipy — skipping")
        return img.astype(np.float32)

    ndim_orig = img.ndim
    if ndim_orig == 2:
        img = img[:, :, np.newaxis]

    H, W, C = img.shape
    img_f = img.astype(np.float64)
    original_max = float(img_f.max()) or 1.0

    # Rotate around nucleus: scipy.ndimage.rotate rotates around image centre.
    # We compensate by shifting the image so nucleus is at centre, rotating, then shifting back.
    centre_y, centre_x = H / 2.0 - 0.5, W / 2.0 - 0.5
    shift_to_centre = (centre_y - nucleus_y, centre_x - nucleus_x)
    shift_back = (nucleus_y - centre_y, nucleus_x - centre_x)

    result = np.zeros_like(img_f)
    for c in range(C):
        ch = img_f[:, :, c]
        # Shift nucleus to image centre
        shifted = _shift(ch, shift=shift_to_centre, order=3, mode='constant', cval=0.0)
        # Rotate
        rotated = _rotate(shifted, angle=rotation_deg, reshape=False,
                          order=3, mode='constant', cval=0.0)
        # Optional radial shift: shift away from image centre (now = nucleus)
        if abs(dr) > 0.1:
            # Direction along the average gradient (use identity for simplicity: shift along Y)
            rotated = _shift(rotated, shift=(dr, 0.0), order=1, mode='constant', cval=0.0)
        # Shift nucleus back to original position
        rotated = _shift(rotated, shift=shift_back, order=3, mode='constant', cval=0.0)
        # Larson-Sekanina: original minus rotated-shifted copy
        diff = ch - rotated
        result[:, :, c] = diff

    # Clip to [0, original_max] and re-normalise
    result = np.clip(result, 0.0, original_max)
    if result.max() > 1e-12:
        result = result / result.max() * original_max

    out = result.astype(np.float32)
    if ndim_orig == 2:
        out = out[:, :, 0]
    return out
