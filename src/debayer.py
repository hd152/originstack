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

# Bayer pattern → (r_offset, g1_offset, g2_offset, b_offset)
# Each offset is (row, col) into the 2×2 Bayer tile.
_PATTERN_OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    'RGGB': ((0, 0), (0, 1), (1, 0), (1, 1)),  # R G1 G2 B
    'BGGR': ((1, 1), (0, 1), (1, 0), (0, 0)),  # B G1 G2 R  → swap R↔B
    'GRBG': ((0, 1), (0, 0), (1, 1), (1, 0)),  # R G1 G2 B (shifted)
    'GBRG': ((1, 0), (0, 0), (1, 1), (0, 1)),  # R G1 G2 B (shifted)
}


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
    kernel = xp.array(_BILINEAR_KERNEL)

    def upsample(ch, r_offset, c_offset):
        # kron doubles the grid without an explicit sparse intermediate.
        # We still need to place the channel at the correct sub-pixel offset,
        # so we pad, kron, then slice back to (H, W).
        expanded = xp.zeros((H, W), dtype=xp.float32)
        expanded[r_offset::2, c_offset::2] = ch
        return gpu.xndimage.convolve(expanded, kernel, mode='mirror')

    # FIX #7: Average the two green upsamples in one expression — same cost,
    # but expressed clearly; a future GPU kernel could fuse these.
    out = xp.zeros((H, W, 3), dtype=xp.float32)
    out[:, :, 0] = upsample(r,  r_r,  r_c)
    out[:, :, 1] = 0.5 * (upsample(g1, g1_r, g1_c) + upsample(g2, g2_r, g2_c))
    out[:, :, 2] = upsample(b,  b_r,  b_c)
    return out


def debayer_malvar(raw: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """Malvar-He-Cutler demosaicing.

    Uses OpenCV's edge-aware (EA) implementation when cv2 is available —
    this is the correct Malvar-He-Cutler algorithm.  Falls back to bilinear
    when cv2 is absent; the old simplified difference-kernel approach is NOT
    used because its kR/kB kernels sum to zero, which destroys R and B signal
    whenever the per-channel sky background levels differ (e.g. after flat
    calibration normalises a colour-imbalanced sensor like the IMX178).
    """
    _ensure_cv2()
    if HAS_CV2:
        pat_map = {
            'RGGB': getattr(cv2, 'COLOR_BAYER_RG2BGR_EA', None),
            'BGGR': getattr(cv2, 'COLOR_BAYER_BG2BGR_EA', None),
            'GRBG': getattr(cv2, 'COLOR_BAYER_GR2BGR_EA', None),
            'GBRG': getattr(cv2, 'COLOR_BAYER_GB2BGR_EA', None),
        }
        code = pat_map.get(pattern.upper())
        if code is not None:
            raw_np = np.asarray(raw, dtype=np.float32)
            max_val = raw_np.max()
            if max_val <= 0:
                return np.zeros((*raw_np.shape, 3), dtype=np.float32)
            raw_u16 = np.clip(raw_np / max_val * 65535, 0, 65535).astype(np.uint16)
            rgb = cv2.cvtColor(raw_u16, code)
            return rgb.astype(np.float32) / 65535.0 * max_val

    # Fallback: bilinear (correct, though lower quality than Malvar)
    return debayer_bilinear(raw, pattern)


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
    rgb = cv2.cvtColor(raw_u8, code)
    return rgb.astype(np.float32) / 255.0 * max_val


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
    img = xp.array(rgb, dtype=xp.float32, copy=True)
    mean = img.mean(axis=(0, 1))
    scale = mean.mean() / (mean + 1e-12)
    return xp.clip(img * scale, 0, None)


def white_balance_whitepatch(rgb: np.ndarray, pct: Optional[float] = None) -> np.ndarray:
    gpu = get_gpu()
    xp = gpu.xp
    if pct is None:
        pct = Config.WHITE_PATCH_PERCENTILE
    img = xp.array(rgb, dtype=xp.float32, copy=True)
    scales = xp.array([float(xp.percentile(img[:, :, c], pct)) for c in range(3)])

    # FIX #4: Guard against saturated / clipped channels.  A percentile at or
    # above the image maximum means that channel is fully saturated; dividing
    # by it would collapse the whole channel to near-zero.  Fall back to the
    # channel mean in that case so we at least do something sensible.
    img_max = float(img.max())
    for c in range(3):
        if float(scales[c]) >= img_max * 0.999 or float(scales[c]) < 1e-12:
            scales[c] = float(img[:, :, c].mean()) + 1e-12

    scales = scales / (scales.mean() + 1e-12)
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
        return _fix_hot_rgb(data, threshold)
    elif mode == 'mono':
        return _fix_hot_mono(data, threshold)
    else:
        raise ValueError(f"Unknown hot pixel mode: {mode!r}")


def _fix_hot_bayer(data: np.ndarray, threshold: Optional[float] = _DETECT,
                   hot_map: Optional[np.ndarray] = None) -> np.ndarray:
    """Bayer-aware hot pixel fix: apply pre-built map and/or statistical detection."""
    if data.ndim != 2:
        return data
    result = data.astype(np.float32, copy=True)

    # Apply pre-built dark-frame map if provided
    if hot_map is not None and hot_map.shape == data.shape and np.any(hot_map):
        for dy in range(2):
            for dx in range(2):
                sub = result[dy::2, dx::2]
                mask = hot_map[dy::2, dx::2]
                if np.any(mask):
                    med = ndimage.median_filter(sub, size=3)
                    sub[mask] = med[mask]
                    result[dy::2, dx::2] = sub

    # Statistical detection per sub-channel (skip if threshold is None)
    if threshold is None:
        return result
    if threshold is _DETECT:
        threshold = Config.HOT_PIXEL_BAYER_THRESHOLD
    for dy in range(2):
        for dx in range(2):
            sub = result[dy::2, dx::2]
            med = ndimage.median_filter(sub, size=3)
            diff = sub - med
            mad = np.median(np.abs(diff))
            sigma = mad * 1.4826
            if sigma < 1e-6:
                continue
            mask = diff > threshold * sigma
            if np.any(mask):
                sub[mask] = med[mask]
                result[dy::2, dx::2] = sub
    return result


def _fix_hot_rgb(rgb: np.ndarray, threshold: Optional[float] = _DETECT) -> np.ndarray:
    """Detect hot pixels on luminance, fix all 3 channels.

    Computes per-channel medians once (3 passes), reconstructs median
    luminance from them, and reuses medians for replacement.
    """
    gpu = get_gpu()
    xp = gpu.xp
    if threshold is _DETECT:
        threshold = Config.HOT_PIXEL_THRESHOLD

    ch_meds = [gpu.xndimage.median_filter(rgb[:, :, c], size=3) for c in range(rgb.shape[2])]
    lum     = 0.299 * rgb[:, :, 0]   + 0.587 * rgb[:, :, 1]   + 0.114 * rgb[:, :, 2]
    med_lum = 0.299 * ch_meds[0]     + 0.587 * ch_meds[1]     + 0.114 * ch_meds[2]

    diff = lum - med_lum
    mad = float(xp.median(xp.abs(diff)))
    sigma = mad * 1.4826
    if sigma < 1e-6:
        sigma = float(xp.std(diff))

    mask = diff > threshold * sigma
    if not bool(xp.any(mask)):
        return rgb

    result = xp.array(rgb, copy=True)
    for c in range(rgb.shape[2]):
        result[:, :, c][mask] = ch_meds[c][mask]
    return result


def _fix_hot_mono(img: np.ndarray, threshold: Optional[float] = _DETECT) -> np.ndarray:
    """Single-channel hot pixel detection and replacement (GPU-accelerated)."""
    gpu = get_gpu()
    xp = gpu.xp
    if threshold is _DETECT:
        threshold = Config.HOT_PIXEL_THRESHOLD
    med = gpu.xndimage.median_filter(img, size=3)
    diff = img - med

    mad = float(xp.median(xp.abs(diff)))
    sigma = mad * 1.4826
    if sigma < 1e-6:
        sigma = float(xp.std(diff))

    mask = diff > threshold * sigma
    if not bool(xp.any(mask)):
        return img
    img_fixed = xp.array(img, copy=True)
    img_fixed[mask] = med[mask]
    return img_fixed


# Legacy aliases for backwards compatibility
remove_hot_pixels = _fix_hot_mono
remove_hot_pixels_bayer = lambda data, threshold=_DETECT: fix_hot_pixels(data, mode='bayer', threshold=threshold)
remove_hot_pixels_rgb = lambda rgb, threshold=_DETECT: fix_hot_pixels(rgb, mode='rgb', threshold=threshold)
apply_hot_pixel_map_bayer = lambda data, hot_map: fix_hot_pixels(data, mode='bayer', hot_map=hot_map, threshold=None)


def correct_chromatic_aberration(rgb: np.ndarray, max_shift_px: float = 5.0,
                                  upsample: int = 10) -> np.ndarray:
    """Correct lateral chromatic aberration by sub-pixel per-channel registration.

    Registers the red and blue channels against the green channel using phase
    cross-correlation and applies a sub-pixel shift to bring them into alignment.
    Correction is intentionally limited to ``max_shift_px`` pixels to avoid
    over-correcting in frames where phase correlation fails.

    Requires skimage (``phase_cross_correlation``).  Returns the original image
    unchanged if the dependency is missing or registration fails.

    Args:
        rgb: Float32 image (H, W, 3), R/G/B order.
        max_shift_px: Maximum plausible CA shift in pixels (default 5).
                      Corrections larger than this are silently suppressed.
        upsample: Sub-pixel upsample factor for phase correlation (default 10).
    """
    if not _HAS_PCC or rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb

    result = rgb.copy()
    g = rgb[:, :, 1].astype(np.float64)
    g_std = g.std()
    if g_std < 1e-12:
        return rgb
    g_norm = (g - g.mean()) / g_std

    # FIX #5: `import warnings` moved to module level — was inside this loop,
    # causing a redundant (though cached) import on every iteration.
    for c_idx in (0, 2):  # Red, Blue
        ch = rgb[:, :, c_idx].astype(np.float64)
        ch_std = ch.std()
        if ch_std < 1e-12:
            continue
        ch_norm = (ch - ch.mean()) / ch_std
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                shift, error, _ = _pcc(g_norm, ch_norm, upsample_factor=upsample)
            if (np.isfinite(shift).all()
                    and np.abs(shift[0]) <= max_shift_px
                    and np.abs(shift[1]) <= max_shift_px):
                result[:, :, c_idx] = ndimage.shift(
                    rgb[:, :, c_idx], shift=shift,
                    order=3, mode='reflect').astype(np.float32)
                _log.debug("CA correction ch%d: shift=(%.3f, %.3f) err=%.4f",
                           c_idx, float(shift[0]), float(shift[1]), float(error))
        except Exception as exc:
            _log.debug("CA correction ch%d failed: %s", c_idx, exc)

    return result


def background_gradient_subtract(img: np.ndarray) -> np.ndarray:
    gpu = get_gpu()
    # FIX #9: Use img.shape[:2] so min() operates on spatial dims (H, W) only.
    # Previously min(img.shape) could return 3 (the channel count) for RGB
    # input, causing sigma to collapse to max(15, 0) = 15 regardless of image
    # size.
    sigma = max(15, min(img.shape[:2]) // 20)
    blurred = gpu.xndimage.gaussian_filter(img, sigma=sigma)
    return img - blurred


def remove_hot_pixels_rgb(rgb: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """Detect hot pixels on luminance, fix all 3 channels.

    FIX #8: Reduced from 4 median_filter passes (1 luminance + 3 channels)
    to 3 passes by computing per-channel medians up front and reusing them
    both for mask derivation (via luminance) and for pixel replacement.
    """
    gpu = get_gpu()
    xp = gpu.xp
    if threshold is None:
        threshold = Config.HOT_PIXEL_THRESHOLD

    # Compute per-channel medians once — reused for both detection and repair.
    ch_meds = [gpu.xndimage.median_filter(rgb[:, :, c], size=3) for c in range(rgb.shape[2])]

    # Reconstruct median luminance from per-channel medians (no extra filter pass).
    lum      = 0.299 * rgb[:, :, 0]    + 0.587 * rgb[:, :, 1]    + 0.114 * rgb[:, :, 2]
    med_lum  = 0.299 * ch_meds[0]      + 0.587 * ch_meds[1]      + 0.114 * ch_meds[2]

    diff = lum - med_lum

    # FIX #3 (applied here too): MAD-based sigma instead of std.
    mad = float(xp.median(xp.abs(diff)))
    sigma = mad * 1.4826
    if sigma < 1e-6:
        sigma = float(xp.std(diff))

    mask = diff > threshold * sigma
    if not bool(xp.any(mask)):
        return rgb

    result = xp.array(rgb, copy=True)
    for c in range(rgb.shape[2]):
        result[:, :, c][mask] = ch_meds[c][mask]
    return result