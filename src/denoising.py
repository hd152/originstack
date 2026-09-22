"""Denoising and image processing: curvelet/wavelet, ACDNR, bilateral, anisotropic, arcsinh."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from src import wavelet
from src.background import (
    _estimate_sky_sigma,
    _gaussian_blur,
    gaussian_blur_spatial,
    gaussian_filter_ds,
)
from src.models import Config
from src.utils import get_logger, safe_print

_log = get_logger()

# Optional native (Rust) kernels — graceful degradation to numpy if absent.
try:
    import astro_native as _native
    _HAS_NATIVE = True
except Exception:
    _native = None
    _HAS_NATIVE = False

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
    jxx = _gaussian_blur(gx * gx, sigma)
    jyy = _gaussian_blur(gy * gy, sigma)
    jxy = _gaussian_blur(gx * gy, sigma)
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
                                coherence_sigma: float = 1.5,
                                variance_stabilize: bool = False) -> np.ndarray:
    """Directional (curvelet/shearlet-*inspired*) adaptive wavelet
    denoising -- ``--denoiser curvelet``.

    Plain BayesShrink (the plain BayesShrink wavelet denoiser (this function at ``protect_strength=0``)) applies one threshold
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
    the plain BayesShrink wavelet denoiser (this function at ``protect_strength=0``) already uses -- chroma structure isn't
    what this is meant to protect).

    ``protect_strength``: 0 = falls back to plain uniform BayesShrink
    (protection off); clamped to at most 0.95 so even maximally-coherent
    pixels retain a little thresholding (a thin linear artifact -- a
    satellite trail sliver, a hot column -- is also "coherent" by this
    measure, and shouldn't pass through completely untouched).

    ``variance_stabilize`` applies a generalized Anscombe transform to the
    luma plane before thresholding and inverts it after (luma only: it's the
    plane whose noise is genuinely photon-limited, which is what the
    Poisson+Gaussian model behind the transform assumes).
    """
    h, w = img.shape[0], img.shape[1]
    src = img.astype(np.float64)
    protect_strength = float(np.clip(protect_strength, 0.0, 0.95))

    Y  =  0.29900 * src[:, :, 0] + 0.58700 * src[:, :, 1] + 0.11400 * src[:, :, 2]
    Cb = -0.16875 * src[:, :, 0] - 0.33126 * src[:, :, 1] + 0.50000 * src[:, :, 2]
    Cr =  0.50000 * src[:, :, 0] - 0.41869 * src[:, :, 1] - 0.08131 * src[:, :, 2]

    coherence = _structure_tensor_coherence(Y, sigma=coherence_sigma)
    if variance_stabilize:
        # Stabilise first: the coherence map is computed on the raw luma so
        # the structure test isn't distorted by the transform's compression.
        gain, sigma_ro = _estimate_noise_level_function(Y)

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

    if variance_stabilize:
        Y_d = _inverse_generalized_anscombe(
            _denoise_plane(_generalized_anscombe(np.maximum(Y, 0.0), gain, sigma_ro),
                           1.0, True), gain, sigma_ro)
    else:
        Y_d = _denoise_plane(Y, 1.0, True)
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
    smooth_luma = _gaussian_blur(luma, smoothing_sigma)
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
    Y_smooth  = _gaussian_blur(Y,  smoothing_sigma)
    Cb_smooth = _gaussian_blur(Cb, smoothing_sigma)
    Cr_smooth = _gaussian_blur(Cr, smoothing_sigma)

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
    blurred_weight = _gaussian_blur(sky_mask, sigma)
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
        smooth_chroma = _gaussian_blur(chroma * sky_mask, sigma) / safe_weight
        # Stars keep original chroma; background gets smoothed chroma
        out_chroma = chroma * protect + smooth_chroma * sky_mask
        if do_large:
            coarse = (gaussian_filter_ds(out_chroma * sky_mask,
                                         sigma=sigma_large) / weight_large)
            out_chroma = out_chroma * (1.0 - blend_large) + coarse * blend_large
        result[:, :, c] = lum + out_chroma

    # No non-negativity clip: this runs right after background extraction, which
    # centres the sky on zero, and the sky pedestal that keeps noise above zero is
    # applied later. Clipping here half-wave-rectifies the sky noise (50% exact
    # zeros + positive spikes) -- the stretch then renders the spikes as white dots.
    return result.astype(np.float32)


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


# Percentile at which the local-contrast detail source is clipped, so a bright
# star's flux cannot smear outward through the blur and carve a dark ring in
# the nebulosity around it. Chosen by measurement on a real stack: p99 cuts the
# ring from -6.1% to -0.5% while keeping ~81% of the contrast gain; p97/p95
# remove the ring entirely but progressively undo the enhancement itself.
_DETAIL_CLIP_PERCENTILE = 99.0


def multiscale_local_contrast(
        img: np.ndarray,
        strength: float = 0.7,
        scales: Tuple[int, ...] = (2, 12, 40),
        scale_weights: Tuple[float, ...] = (0.3, 0.6, 0.1),
        star_mask: Optional[np.ndarray] = None,
        detail_clip_percentile: Optional[float] = _DETAIL_CLIP_PERCENTILE
        ) -> np.ndarray:
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

        detail_clip_percentile: Percentile at which the detail source is
                       clipped (the bright-star collar guard, see below).
                       ``None`` disables it -- correct for an image that has
                       no stars in it (the ``--starless-process`` layer).

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

    # Detail is measured against a PEAK-CLIPPED luminance ————————————————————
    # Otherwise a bright star's flux leaks into its own background estimate:
    # blurring at these scales smears the core outward, so in the annulus just
    # beyond the protected core `blurred` far exceeds `lum`, `detail` goes
    # strongly negative, and the enhancement *subtracts* real nebulosity --
    # a black collar around every bright star, starting exactly where core
    # protection stops.
    #
    # Measured on a real Lagoon stack at the strength --auto selects for an
    # emission nebula (0.749): the background around bright stars darkened by
    # 6.1% on average and 14.2% at worst. Clipping the detail source at the
    # 99th percentile caps how much stellar flux can smear outward and brings
    # that to 0.5% / 1.8%, while retaining ~81% of the contrast gain away from
    # stars. Clipping harder (p97, p95) removes the ring completely but walks
    # the enhancement back toward doing nothing at all.
    #
    # Note this deliberately does NOT key off `star_mask`: that mask is a
    # narrow fwhm-3 core covering ~0.4% of pixels, far smaller than the wings
    # that actually pollute a sigma-12 blur. An earlier attempt to fix this by
    # infilling masked pixels measured beautifully on a synthetic scene with a
    # hard disk mask and changed real output by 0.01%.
    detail_src = lum if detail_clip_percentile is None else np.minimum(
        lum, float(np.percentile(lum, detail_clip_percentile)))

    # Multiscale detail injection ——————————————————————————————————————————
    enhanced_lum = lum.copy()
    for sigma, w in zip(scales, scale_weights):
        if w <= 0 or strength <= 0:
            continue
        blurred = _gaussian_blur(detail_src, float(sigma))
        detail = detail_src - blurred   # high-frequency detail at this scale
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
    blurred = gaussian_blur_spatial(img.astype(np.float64), blur_sigma)

    blend = (star_mask * reduction_factor).astype(np.float64)
    mask3 = blend[:, :, np.newaxis]
    result = img.astype(np.float64) * (1.0 - mask3) + blurred * mask3
    return np.clip(result, 0.0, None).astype(np.float32)


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
        from scipy.ndimage import rotate as _rotate
        from scipy.ndimage import shift as _shift
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
