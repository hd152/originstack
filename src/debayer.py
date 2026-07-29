"""Debayering, white balance, and hot pixel removal."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy import ndimage

from src.gpu_context import get_gpu
from src.models import Config
from src.utils import get_logger

_log = get_logger()

# Optional native (Rust) kernels — graceful degradation to numpy if absent.
try:
    import astro_native as _native
    _HAS_NATIVE = True
except Exception:
    _native = None
    _HAS_NATIVE = False

cv2 = None
HAS_CV2 = False


def _ensure_cv2():
    global cv2, HAS_CV2
    if not HAS_CV2:
        try:
            import cv2 as _cv2
            if hasattr(_cv2, 'cvtColor'):
                cv2 = _cv2
                HAS_CV2 = True
        except Exception:
            pass
    return cv2

try:
    from skimage.registration import phase_cross_correlation as _pcc
    _HAS_PCC = True
except Exception:
    _pcc = None
    _HAS_PCC = False

# FIX #10: Module-level constant — avoids reallocating the kernel on every
# upsample() call inside debayer_bilinear().
_BILINEAR_KERNEL = np.array(
    [[0.25, 0.5, 0.25],
     [0.5,  1.0, 0.5],
     [0.25, 0.5, 0.25]],
    dtype=np.float32,
)
_BILINEAR_KERNEL /= _BILINEAR_KERNEL.sum()
_BILINEAR_KERNEL_GPU: dict = {}   # xp-id → device copy, uploaded once per session

# Bayer pattern → (r_offset, g1_offset, g2_offset, b_offset)
# Each offset is (row, col) into the 2×2 Bayer tile.
_PATTERN_OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    'RGGB': ((0, 0), (0, 1), (1, 0), (1, 1)),  # R G1 G2 B
    'BGGR': ((1, 1), (0, 1), (1, 0), (0, 0)),  # B G1 G2 R  → swap R↔B
    'GRBG': ((0, 1), (0, 0), (1, 1), (1, 0)),  # R G1 G2 B (shifted)
    'GBRG': ((1, 0), (0, 0), (1, 1), (0, 1)),  # R G1 G2 B (shifted)
}


def _sigma_clipped_median(arr, sigma: float = 3.0, iters: int = 3, xp=np) -> float:
    x = arr.ravel()
    for _ in range(iters):
        med = float(xp.median(x))
        std = float(xp.std(x))
        if std < 1e-12:
            break
        x = x[xp.abs(x - med) < sigma * std]
        if len(x) == 0:
            break
    return float(xp.median(x)) if len(x) > 0 else float(xp.median(arr))


def green_equalize(raw, pattern: str = 'RGGB'):
    """Scale the G2 sub-channel to match G1's sigma-clipped median.

    CMOS sensors have two physically distinct green sub-pixels (G1, G2) per
    Bayer tile that often differ by a few percent in sensitivity.  After
    bilinear debayering, the mismatch propagates as a 2-pixel-period
    checkerboard across the green channel (and therefore luminance).  Scaling
    G2 to match G1 before debayering removes the artifact entirely.

    The correction is capped at ±20 % to guard against bad frames where one
    sub-channel is near zero (saturated sky, very low counts, etc.).
    Accepts both numpy and CuPy arrays; output matches the input type.
    """
    offsets = _PATTERN_OFFSETS.get(pattern.upper())
    if offsets is None:
        return raw
    try:
        import cupy as _cp
        xp = _cp.get_array_module(raw)
    except ImportError:
        xp = np
    (_, _), (g1_r, g1_c), (g2_r, g2_c), (_, _) = offsets
    raw_f = xp.array(raw, dtype=xp.float32)
    g1 = raw_f[g1_r::2, g1_c::2].ravel()
    g2 = raw_f[g2_r::2, g2_c::2].ravel()
    # On GPU use a direct median (2 GPU→CPU syncs) instead of sigma-clipped
    # (12+ syncs). For typical astrophotography data the difference is <0.1%.
    if xp is np:
        g1_med = _sigma_clipped_median(g1, xp=xp)
        g2_med = _sigma_clipped_median(g2, xp=xp)
    else:
        g1_med = float(xp.median(g1))
        g2_med = float(xp.median(g2))
    if g2_med > 1e-6 and abs(g1_med / g2_med - 1.0) < 0.2:
        raw_f[g2_r::2, g2_c::2] *= g1_med / g2_med
    return raw_f


def _equalize_bayer_grid(rgb: np.ndarray) -> np.ndarray:
    """Remove 2×2 position-dependent green bias introduced by edge-aware CFA interpolation.

    Malvar/VNG debayering produces systematically different green values at
    interpolated positions (R and B cells) vs source positions (G1 and G2 cells).
    After stacking many frames, this coherent per-pixel offset survives noise
    averaging and becomes a visible checkerboard in the sky background.

    Uses sigma-clipped medians to estimate the sky-level offset for each of the
    four Bayer-position sub-channels, then subtracts the deviation from the
    per-image mean.  Guards skip corrections outside the plausible 0.01–100 ADU
    range to avoid modifying high-SNR targets or corrupted frames.
    """
    G = rgb[:, :, 1]
    ee = _sigma_clipped_median(G[::2,  ::2])   # R positions (even row, even col)
    eo = _sigma_clipped_median(G[::2,  1::2])  # G1 positions (even row, odd col)
    oe = _sigma_clipped_median(G[1::2, ::2])   # G2 positions (odd row, even col)
    oo = _sigma_clipped_median(G[1::2, 1::2])  # B positions (odd row, odd col)
    overall = (ee + eo + oe + oo) / 4.0
    spread = max(abs(ee - overall), abs(eo - overall),
                 abs(oe - overall), abs(oo - overall))
    if spread < 0.01 or spread > 100.0:
        return rgb
    result = rgb.copy()
    G_out = result[:, :, 1]
    G_out[::2,  ::2]  = np.clip(G[::2,  ::2]  - float(ee - overall), 0, None)
    G_out[::2,  1::2] = np.clip(G[::2,  1::2] - float(eo - overall), 0, None)
    G_out[1::2, ::2]  = np.clip(G[1::2, ::2]  - float(oe - overall), 0, None)
    G_out[1::2, 1::2] = np.clip(G[1::2, 1::2] - float(oo - overall), 0, None)
    return result


def debayer_bilinear(raw: np.ndarray, pattern: str = 'RGGB', method: str = 'bilinear') -> np.ndarray:
    # FIX #1: honour the `pattern` argument — previously hardcoded to RGGB.
    # FIX #2: `method` param is accepted for API compatibility but not used
    #          (bilinear is the only variant here); document this explicitly.
    gpu = get_gpu()
    xp = gpu.xp
    raw = gpu.to_device(raw)
    H, W = raw.shape

    offsets = _PATTERN_OFFSETS.get(pattern.upper())
    if offsets is None:
        raise ValueError(f"Unknown Bayer pattern '{pattern}'. "
                         f"Expected one of {list(_PATTERN_OFFSETS)}")

    (r_r, r_c), (g1_r, g1_c), (g2_r, g2_c), (b_r, b_c) = offsets

    r  = raw[r_r::2,  r_c::2]
    g1 = raw[g1_r::2, g1_c::2]
    g2 = raw[g2_r::2, g2_c::2]
    b  = raw[b_r::2,  b_c::2]

    # FIX #6 & #10: Use kron for channel expansion instead of a sparse
    # zero-filled array; reuse the pre-allocated module-level kernel.
    _xp_id = id(xp)
    if _xp_id not in _BILINEAR_KERNEL_GPU:
        _BILINEAR_KERNEL_GPU[_xp_id] = xp.array(_BILINEAR_KERNEL)
    kernel = _BILINEAR_KERNEL_GPU[_xp_id]
    # Pre-allocate once; each upsample() call zeros and refills it.
    # convolve() returns a new array so the previous result is safe.
    expanded = xp.zeros((H, W), dtype=xp.float32)

    def upsample(ch, r_offset, c_offset):
        expanded[:] = 0
        expanded[r_offset::2, c_offset::2] = ch
        return gpu.xndimage.convolve(expanded, kernel, mode='mirror')

    # FIX #7: Average the two green upsamples in one expression — same cost,
    # but expressed clearly; a future GPU kernel could fuse these.
    out = xp.zeros((H, W, 3), dtype=xp.float32)
    out[:, :, 0] = upsample(r,  r_r,  r_c)
    out[:, :, 1] = 0.5 * (upsample(g1, g1_r, g1_c) + upsample(g2, g2_r, g2_c))
    out[:, :, 2] = upsample(b,  b_r,  b_c)
    return out


# Malvar-He-Cutler (2004) kernels -- "High-Quality Linear Interpolation for
# Demosaicing of Bayer-Patterned Color Images". Coefficients as published
# (Table 1), normalised by 8; verified against the reference implementation
# in the `colour-demosaicing` package (bit-exact on interior pixels, see
# tests/test_debayer_malvar.py). The old cv2-EA path required 16-bit
# requantization (real precision loss) and only ran when cv2 was installed;
# this operates directly on the native float32 data and has no dependency.
_MALVAR_G_AT_RB = np.array([
    [0.0, 0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0,  2.0, 0.0, 0.0],
    [-1.0, 2.0, 4.0, 2.0, -1.0],
    [0.0, 0.0,  2.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0],
], dtype=np.float64) / 8.0

# R at green in an R row / B column (and B at green in a B row / R column).
_MALVAR_RG_RB_BG_BR = np.array([
    [0.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, -1.0, 0.0, -1.0, 0.0],
    [-1.0, 4.0, 5.0, 4.0, -1.0],
    [0.0, -1.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.0],
], dtype=np.float64) / 8.0

# R at green in a B row / R column (and B at green in an R row / B column) --
# the transpose of the kernel above.
_MALVAR_RG_BR_BG_RB = _MALVAR_RG_RB_BG_BR.T

# R at B (and B at R).
_MALVAR_R_AT_B = np.array([
    [0.0, 0.0, -1.5, 0.0, 0.0],
    [0.0, 2.0, 0.0, 2.0, 0.0],
    [-1.5, 0.0, 6.0, 0.0, -1.5],
    [0.0, 2.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, -1.5, 0.0, 0.0],
], dtype=np.float64) / 8.0


def _debayer_malvar_numpy(raw: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """Malvar-He-Cutler demosaicing, pure numpy (native Rust dispatch happens
    one level up in ``debayer_malvar``)."""
    offsets = _PATTERN_OFFSETS.get(pattern.upper())
    if offsets is None:
        raise ValueError(f"Unknown Bayer pattern '{pattern}'. "
                         f"Expected one of {list(_PATTERN_OFFSETS)}")
    (r_r, r_c), _, _, (b_r, b_c) = offsets
    raw64 = np.asarray(raw, dtype=np.float64)
    H, W = raw64.shape

    row = np.arange(H) % 2
    col = np.arange(W) % 2
    r_row = (row == r_r)[:, None]
    r_col = (col == r_c)[None, :]
    b_row = (row == b_r)[:, None]
    b_col = (col == b_c)[None, :]

    g_at_rb = ndimage.convolve(raw64, _MALVAR_G_AT_RB, mode='mirror')
    rg_rb_bg_br = ndimage.convolve(raw64, _MALVAR_RG_RB_BG_BR, mode='mirror')
    rg_br_bg_rb = ndimage.convolve(raw64, _MALVAR_RG_BR_BG_RB, mode='mirror')
    r_at_b = ndimage.convolve(raw64, _MALVAR_R_AT_B, mode='mirror')

    is_r = r_row & r_col
    is_b = b_row & b_col

    R = np.where(is_r, raw64, 0.0)
    R = np.where(r_row & b_col, rg_rb_bg_br, R)
    R = np.where(b_row & r_col, rg_br_bg_rb, R)
    R = np.where(is_b, r_at_b, R)

    B = np.where(is_b, raw64, 0.0)
    B = np.where(b_row & r_col, rg_rb_bg_br, B)
    B = np.where(r_row & b_col, rg_br_bg_rb, B)
    B = np.where(is_r, r_at_b, B)

    G = np.where(is_r | is_b, g_at_rb, raw64)

    return np.stack([R, G, B], axis=-1).astype(np.float32)


def debayer_malvar(raw: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """Malvar-He-Cutler demosaicing (native Rust kernel with a numpy
    fallback -- see ``_debayer_malvar_numpy`` for the algorithm/validation
    notes). No longer depends on cv2."""
    if _HAS_NATIVE and hasattr(_native, 'debayer_malvar'):
        raw_np = np.ascontiguousarray(raw, dtype=np.float32)
        try:
            out = _native.debayer_malvar(raw_np, pattern.upper())
        except Exception:
            out = None
        if out is not None:
            return _equalize_bayer_grid(out)
    return _equalize_bayer_grid(_debayer_malvar_numpy(raw, pattern))


def debayer_vng(raw: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """VNG (Variable Number of Gradients) debayering via OpenCV."""
    _ensure_cv2()
    if not HAS_CV2:
        return debayer_malvar(raw, pattern)
    pat_map = {
        'RGGB': getattr(cv2, 'COLOR_BAYER_RG2BGR_VNG', None),
        'BGGR': getattr(cv2, 'COLOR_BAYER_BG2BGR_VNG', None),
        'GRBG': getattr(cv2, 'COLOR_BAYER_GR2BGR_VNG', None),
        'GBRG': getattr(cv2, 'COLOR_BAYER_GB2BGR_VNG', None),
    }
    code = pat_map.get(pattern.upper())
    if code is None:
        return debayer_malvar(raw, pattern)
    raw_np = np.asarray(raw, dtype=np.float32)
    max_val = raw_np.max()
    if max_val <= 0:
        return np.zeros((*raw_np.shape, 3), dtype=np.float32)
    # VNG requires uint8 input in OpenCV >= 4.x
    raw_u8 = np.clip(raw_np / max_val * 255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(raw_u8, code)
    result = bgr[:, :, ::-1].astype(np.float32) / 255.0 * max_val
    return _equalize_bayer_grid(result)


def debayer(raw: np.ndarray, pattern: str = 'RGGB', method: str = 'bilinear') -> np.ndarray:
    """Dispatch to the appropriate debayering method."""
    if method == 'vng':
        return debayer_vng(raw, pattern)
    elif method == 'malvar':
        return debayer_malvar(raw, pattern)
    else:
        return debayer_bilinear(raw, pattern, method)


def white_balance_grayworld(rgb: np.ndarray) -> np.ndarray:
    gpu = get_gpu()
    xp = gpu.xp
    img = xp.asarray(rgb, dtype=xp.float32)
    mean = img.mean(axis=(0, 1))
    scale = mean.mean() / (mean + 1e-12)
    return xp.clip(img * scale, 0, None)


def white_balance_whitepatch(rgb: np.ndarray, pct: Optional[float] = None) -> np.ndarray:
    gpu = get_gpu()
    xp = gpu.xp
    if pct is None:
        pct = Config.WHITE_PATCH_PERCENTILE
    img = xp.asarray(rgb, dtype=xp.float32)
    # Compute all per-channel stats on GPU, avoiding per-channel CPU syncs.
    # Guard against saturated channels: fall back to channel mean when the
    # pct-th percentile hits or exceeds the image max (fully clipped channel).
    scales  = xp.stack([xp.percentile(img[:, :, c], pct) for c in range(3)])
    img_max = img.max()
    means   = img.mean(axis=(0, 1))
    bad     = (scales >= img_max * 0.999) | (scales < 1e-12)
    scales  = xp.where(bad, means + 1e-12, scales)
    scales  = scales / (scales.mean() + 1e-12)
    return xp.clip(img / scales[None, None, :], 0, None)


def build_hot_pixel_map(dark: np.ndarray, sigma_threshold: float = 5.0) -> np.ndarray:
    """Build a boolean hot pixel map from an unsmoothed dark frame.

    Detects pixels with dark current significantly above the background.
    Must be called BEFORE Gaussian smoothing of the master dark.
    """
    dark_f = dark.astype(np.float32)
    med = np.median(dark_f)
    # Use MAD-based sigma for robustness against the hot pixels themselves
    mad = np.median(np.abs(dark_f - med))
    sigma = mad * 1.4826  # MAD to Gaussian sigma conversion
    if sigma < 1e-6:
        sigma = float(np.std(dark_f))
    return dark_f > (med + sigma_threshold * sigma)


_DETECT = object()  # sentinel: use default threshold for statistical detection


def fix_hot_pixels(data: np.ndarray, mode: str = 'auto',
                   threshold: Optional[float] = _DETECT,
                   hot_map: Optional[np.ndarray] = None) -> np.ndarray:
    """Unified hot pixel detection and replacement.

    Modes:
        'bayer'  — Process each 2x2 Bayer sub-channel independently.
                   If *hot_map* is provided, those pixels are replaced.
                   Statistical detection also runs unless *threshold=None*.
        'rgb'    — Detect on luminance, replace all 3 channels.
        'mono'   — Single-channel statistical detection (GPU-accelerated).
        'auto'   — Infer from data shape: 2D → bayer, 3D → rgb.

    Args:
        threshold: Sigma multiplier for statistical detection. Pass *None*
                   to skip detection (useful when only applying a hot_map).
                   Defaults to the per-mode Config value.

    All modes use MAD-based sigma for robust noise estimation.
    """
    if mode == 'auto':
        mode = 'bayer' if data.ndim == 2 else 'rgb'

    if mode == 'bayer':
        return _fix_hot_bayer(data, threshold, hot_map)
    elif mode == 'rgb':
        rgb_fixed, _lum = _fix_hot_rgb(data, threshold)
        return rgb_fixed
    elif mode == 'mono':
        return _fix_hot_mono(data, threshold)
    else:
        raise ValueError(f"Unknown hot pixel mode: {mode!r}")


def _fix_hot_bayer(data: np.ndarray, threshold: Optional[float] = _DETECT,
                   hot_map: Optional[np.ndarray] = None) -> np.ndarray:
    """Bayer-aware hot pixel fix: apply pre-built map and/or statistical detection.

    Merged implementation: median_filter is computed once per sub-channel and
    reused for both hot-map replacement and statistical detection, halving the
    number of median filter calls when both are active.
    """
    if data.ndim != 2:
        return data
    result = data.astype(np.float32, copy=True)

    has_map = hot_map is not None and hot_map.shape == data.shape and np.any(hot_map)
    do_stat = threshold is not None
    if do_stat and threshold is _DETECT:
        threshold = Config.HOT_PIXEL_BAYER_THRESHOLD

    if not has_map and not do_stat:
        return result

    for dy in range(2):
        for dx in range(2):
            sub = result[dy::2, dx::2]
            # Compute median once; used for both map application and statistics.
            med = ndimage.median_filter(sub, size=3)

            if has_map:
                map_mask = hot_map[dy::2, dx::2]
                if np.any(map_mask):
                    sub[map_mask] = med[map_mask]
                    # sub is a view of result — no need to write back.

            if do_stat:
                diff = sub - med
                mad = np.median(np.abs(diff))
                sigma = mad * 1.4826
                if sigma < 1e-6:
                    continue
                stat_mask = diff > threshold * sigma
                if np.any(stat_mask):
                    sub[stat_mask] = med[stat_mask]

    return result


def _median_filter3(arr, xp, _nd):
    """3x3 median filter, native (Rust) on the CPU/numpy path when available —
    ~10x faster than scipy's generic rank filter for this small, fixed
    footprint. GPU (cupy) path is untouched; falls back to scipy on any error.
    """
    if (xp is np and _HAS_NATIVE and arr.dtype == np.float32
            and arr.ndim == 2 and arr.flags['C_CONTIGUOUS']):
        try:
            return _native.median_filter_native(arr, 3)
        except Exception:
            pass
    return _nd.median_filter(arr, size=3)


def _fix_hot_rgb_impl(rgb, threshold, xp, _nd):
    """Pure implementation of RGB hot-pixel correction; works with numpy or CuPy.

    Detection uses a single luma median filter (1 pass instead of 3).
    Replacement uses uniform_filter (box mean) which is ~5x faster than
    median on GPU and gives equivalent quality for isolated hot pixels.
    """
    lum     = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    med_lum = _median_filter3(lum, xp, _nd)
    diff    = lum - med_lum
    mad     = float(xp.median(xp.abs(diff)))
    sigma   = mad * 1.4826
    if sigma < 1e-6:
        sigma = float(xp.std(diff))
    mask = diff > threshold * sigma
    if not bool(xp.any(mask)):
        return rgb, lum
    result = xp.empty_like(rgb)
    for c in range(rgb.shape[2]):
        ch = rgb[:, :, c]
        result[:, :, c] = xp.where(mask, _nd.uniform_filter(ch, size=3), ch)
    lum_fixed = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
    return result, lum_fixed


def _fix_hot_rgb(rgb: np.ndarray, threshold: Optional[float] = _DETECT):
    """Detect hot pixels on luminance, fix all 3 channels.

    Computes per-channel medians once (3 passes), reconstructs median
    luminance from them, and reuses medians for replacement.

    Returns (rgb_fixed, lum) so the caller can reuse the luminance array
    instead of recomputing it.  Falls back to CPU scipy on GPU OOM.
    """
    gpu = get_gpu()
    if threshold is _DETECT:
        threshold = Config.HOT_PIXEL_THRESHOLD
    if gpu.active:
        try:
            return _fix_hot_rgb_impl(gpu.xp.asarray(rgb), threshold, gpu.xp, gpu.xndimage)
        except Exception as exc:
            if gpu.is_oom(exc):
                gpu.disable()
            else:
                raise
    return _fix_hot_rgb_impl(np.asarray(rgb), threshold, np, ndimage)


def _fix_hot_mono_impl(img, threshold, xp, _nd):
    """Pure implementation of mono hot-pixel correction; works with numpy or CuPy."""
    med   = _median_filter3(img, xp, _nd)
    diff  = img - med
    mad   = float(xp.median(xp.abs(diff)))
    sigma = mad * 1.4826
    if sigma < 1e-6:
        sigma = float(xp.std(diff))
    mask = diff > threshold * sigma
    if not bool(xp.any(mask)):
        return img
    return xp.where(mask, _nd.uniform_filter(img, size=3), img)


def _fix_hot_mono(img: np.ndarray, threshold: Optional[float] = _DETECT) -> np.ndarray:
    """Single-channel hot pixel detection and replacement.  Falls back to CPU on GPU OOM."""
    gpu = get_gpu()
    if threshold is _DETECT:
        threshold = Config.HOT_PIXEL_THRESHOLD
    if gpu.active:
        try:
            return _fix_hot_mono_impl(gpu.xp.asarray(img), threshold, gpu.xp, gpu.xndimage)
        except Exception as exc:
            if gpu.is_oom(exc):
                gpu.disable()
            else:
                raise
    return _fix_hot_mono_impl(np.asarray(img), threshold, np, ndimage)


# Legacy aliases for backwards compatibility
remove_hot_pixels = _fix_hot_mono
remove_hot_pixels_bayer = lambda data, threshold=_DETECT: fix_hot_pixels(data, mode='bayer', threshold=threshold)
apply_hot_pixel_map_bayer = lambda data, hot_map: fix_hot_pixels(data, mode='bayer', hot_map=hot_map, threshold=None)


def _block_avg_2x(a: np.ndarray) -> np.ndarray:
    """2x2 block-average downsample (even-cropped). Preserves input dtype —
    call on float32 and cast to float64 afterward (on the now-small array) to
    avoid a wasted full-resolution float64 copy."""
    h2 = (a.shape[0] // 2) * 2
    w2 = (a.shape[1] // 2) * 2
    c = a[:h2, :w2]
    return (c[::2, ::2] + c[1::2, ::2] + c[::2, 1::2] + c[1::2, 1::2]) * 0.25


def correct_chromatic_aberration(rgb: np.ndarray, max_shift_px: float = 5.0,
                                  upsample: int = 10,
                                  downsample: int = 2) -> np.ndarray:
    """Correct lateral chromatic aberration by sub-pixel per-channel registration.

    Registers the red and blue channels against the green channel using phase
    cross-correlation and applies a sub-pixel shift to bring them into alignment.
    Correction is intentionally limited to ``max_shift_px`` pixels to avoid
    over-correcting in frames where phase correlation fails.

    Requires skimage (``phase_cross_correlation``).  Returns the original image
    unchanged if the dependency is missing or registration fails.

    The shift *estimate* runs on a ``downsample``x block-averaged copy of each
    channel — CA is a smooth, near-constant sub-pixel offset across the frame
    (lens dispersion), not fine per-pixel detail, so it survives a 2x
    downsample essentially exactly while cutting the dominant FFT cost by
    ~4x (N log N). Measured on real 233-frame runs this step was 3-4x slower
    under full parallel load than in isolation — consistent with the full-res
    FFTs being memory-bandwidth bound, which downsampling directly reduces.
    The recovered shift is scaled back up and applied to the FULL-resolution
    channel (accuracy of the *applied* correction is unaffected; only the
    *estimation* resolution changes) via the native Lanczos-3 warp when
    available, else scipy's cubic-spline shift.

    Args:
        rgb: Float32 image (H, W, 3), R/G/B order.
        max_shift_px: Maximum plausible CA shift in pixels (default 5).
                      Corrections larger than this are silently suppressed.
        upsample: Sub-pixel upsample factor for phase correlation (default 10),
                  applied at the downsampled scale.
        downsample: Block-average factor for the correlation estimate (default
                    2). Set to 1 to correlate at full resolution (old behaviour).
    """
    shifts = measure_chromatic_aberration(rgb, max_shift_px=max_shift_px,
                                          upsample=upsample,
                                          downsample=downsample)
    return apply_chromatic_aberration(rgb, shifts)


def measure_chromatic_aberration(rgb: np.ndarray, max_shift_px: float = 5.0,
                                 upsample: int = 10,
                                 downsample: int = 2) -> dict:
    """Measure the R/B channel offsets against G (see
    ``correct_chromatic_aberration``). Returns ``{0: (sy, sx) | None,
    2: (sy, sx) | None}`` keyed by channel index; None where measurement
    failed or exceeded ``max_shift_px``. Measurement and application are
    split so the session-constant CA (lens dispersion is fixed in the sensor
    frame for a whole session) can be measured once on a few sample frames
    and applied to every frame."""
    shifts: dict = {0: None, 2: None}
    if not _HAS_PCC or rgb.ndim != 3 or rgb.shape[2] != 3:
        return shifts

    # Downsample the cheap float32 data FIRST, cast to float64 only the small
    # result. Casting the full-res channel to float64 before downsampling (the
    # original order) wastes a 50MB copy per channel that gets thrown away
    # immediately — exactly the memory traffic this function is trying to cut.
    g_small = (_block_avg_2x(rgb[:, :, 1]) if downsample >= 2
              else rgb[:, :, 1]).astype(np.float64)
    g_std = g_small.std()
    if g_std < 1e-12:
        return shifts
    g_norm = (g_small - g_small.mean()) / g_std

    for c_idx in (0, 2):  # Red, Blue
        ch_small = (_block_avg_2x(rgb[:, :, c_idx]) if downsample >= 2
                   else rgb[:, :, c_idx]).astype(np.float64)
        ch_std = ch_small.std()
        if ch_std < 1e-12:
            continue
        ch_norm = (ch_small - ch_small.mean()) / ch_std
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                shift, error, _ = _pcc(g_norm, ch_norm, upsample_factor=upsample)
            scale = 2.0 if downsample >= 2 else 1.0
            shift = shift * scale
            if (np.isfinite(shift).all()
                    and np.abs(shift[0]) <= max_shift_px
                    and np.abs(shift[1]) <= max_shift_px):
                shifts[c_idx] = (float(shift[0]), float(shift[1]))
                _log.debug("CA measure ch%d: shift=(%.3f, %.3f) err=%.4f",
                           c_idx, float(shift[0]), float(shift[1]), float(error))
        except Exception as exc:
            _log.debug("CA measure ch%d failed: %s", c_idx, exc)

    return shifts


def apply_chromatic_aberration(rgb: np.ndarray, shifts: dict) -> np.ndarray:
    """Apply pre-measured CA channel shifts (see
    ``measure_chromatic_aberration``). Returns the input unchanged when no
    channel has a valid shift."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not shifts:
        return rgb
    if not any(shifts.get(c) is not None for c in (0, 2)):
        return rgb
    result = rgb.copy()
    H, W = rgb.shape[:2]
    for c_idx in (0, 2):
        shift = shifts.get(c_idx)
        if shift is None:
            continue
        if _HAS_NATIVE:
            try:
                off = [-float(shift[0]), -float(shift[1])]
                result[:, :, c_idx] = _native.warp_affine_lanczos3(
                    np.ascontiguousarray(rgb[:, :, c_idx:c_idx + 1]),
                    [1.0, 0.0, 0.0, 1.0], off, H, W, 0.0)[:, :, 0]
                continue
            except Exception:
                pass
        result[:, :, c_idx] = ndimage.shift(
            rgb[:, :, c_idx], shift=shift,
            order=3, mode='reflect').astype(np.float32)
    return result


def remove_hot_pixels_rgb(rgb: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """Detect hot pixels on luminance, fix all 3 channels. Returns corrected RGB."""
    if threshold is None:
        threshold = Config.HOT_PIXEL_THRESHOLD
    rgb_fixed, _lum = _fix_hot_rgb(rgb, threshold=threshold)
    return rgb_fixed


def remove_hot_pixels_rgb_with_lum(rgb: np.ndarray, threshold: Optional[float] = None):
    """Like remove_hot_pixels_rgb but also returns the luminance as (rgb_fixed, lum).

    Use this in performance-critical paths to avoid recomputing luminance after
    hot pixel removal.
    """
    if threshold is None:
        threshold = Config.HOT_PIXEL_THRESHOLD
    return _fix_hot_rgb(rgb, threshold=threshold)