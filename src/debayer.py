"""Debayering, white balance, and hot pixel removal."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.gpu_context import get_gpu
from src.models import Config
from src.utils import get_logger

_log = get_logger()

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

try:
    from skimage.registration import phase_cross_correlation as _pcc
    _HAS_PCC = True
except Exception:
    _pcc = None
    _HAS_PCC = False


def debayer_bilinear(raw, pattern: str = 'RGGB', method: str = 'bilinear'):
    gpu = get_gpu()
    xp = gpu.xp
    raw = gpu.to_device(raw)
    H, W = raw.shape
    pat = pattern.upper()
    r = raw[0::2, 0::2]
    g1 = raw[0::2, 1::2]
    g2 = raw[1::2, 0::2]
    b = raw[1::2, 1::2]

    def upsample(ch, r_offset, c_offset):
        outc = xp.zeros_like(raw)
        outc[r_offset::2, c_offset::2] = ch
        kernel = xp.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]],
                          dtype=xp.float32)
        kernel = kernel / kernel.sum()
        return gpu.xndimage.convolve(outc, kernel, mode='mirror')

    out = xp.zeros((H, W, 3), dtype=xp.float32)
    out[:, :, 0] = upsample(r, 0, 0)
    out[:, :, 1] = 0.5 * (upsample(g1, 0, 1) + upsample(g2, 1, 0))
    out[:, :, 2] = upsample(b, 1, 1)
    return out


def debayer_malvar(raw, pattern: str = 'RGGB'):
    """Malvar-He-Cutler demosaicing (simplified kernels)."""
    gpu = get_gpu()
    xp = gpu.xp
    raw = gpu.to_device(raw)
    H, W = raw.shape
    kG = xp.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
                   [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], dtype=xp.float32) / 8.0
    kR = xp.array([[0, 0, 1, 0, 0], [0, -2, 0, -2, 0], [1, 0, 4, 0, 1],
                   [0, -2, 0, -2, 0], [0, 0, 1, 0, 0]], dtype=xp.float32) / 8.0
    kB = kR[::-1, ::-1]
    out = xp.zeros((H, W, 3), dtype=xp.float32)
    out[:, :, 0] = gpu.xndimage.convolve(raw, kR, mode='mirror')
    out[:, :, 1] = gpu.xndimage.convolve(raw, kG, mode='mirror')
    out[:, :, 2] = gpu.xndimage.convolve(raw, kB, mode='mirror')
    # Malvar kernels have negative coefficients that can produce negative values;
    # clip to zero since negative light is physically meaningless
    xp.clip(out, 0, None, out=out)
    return out


def debayer_vng(raw, pattern: str = 'RGGB'):
    """VNG (Variable Number of Gradients) debayering via OpenCV."""
    if not HAS_CV2:
        return debayer_malvar(raw, pattern)
    pat_map = {
        'RGGB': cv2.COLOR_BAYER_RG2RGB_VNG,
        'BGGR': cv2.COLOR_BAYER_BG2RGB_VNG,
        'GRBG': cv2.COLOR_BAYER_GR2RGB_VNG,
        'GBRG': cv2.COLOR_BAYER_GB2RGB_VNG,
    }
    code = pat_map.get(pattern.upper())
    if code is None:
        return debayer_malvar(raw, pattern)
    raw_np = np.asarray(raw, dtype=np.float32)
    max_val = raw_np.max()
    if max_val <= 0:
        return np.zeros((*raw_np.shape, 3), dtype=np.float32)
    raw_u16 = np.clip(raw_np / max_val * 65535, 0, 65535).astype(np.uint16)
    rgb = cv2.cvtColor(raw_u16, code)
    return (rgb.astype(np.float32) / 65535.0 * max_val)


def debayer(raw, pattern: str = 'RGGB', method: str = 'bilinear'):
    """Dispatch to the appropriate debayering method."""
    if method == 'vng':
        return debayer_vng(raw, pattern)
    elif method == 'malvar':
        return debayer_malvar(raw, pattern)
    else:
        return debayer_bilinear(raw, pattern, method)


def white_balance_grayworld(rgb):
    gpu = get_gpu()
    xp = gpu.xp
    img = xp.array(rgb, dtype=xp.float32, copy=True)
    mean = img.mean(axis=(0, 1))
    scale = mean.mean() / (mean + 1e-12)
    return xp.clip(img * scale, 0, None)


def white_balance_whitepatch(rgb, pct: float = None):
    gpu = get_gpu()
    xp = gpu.xp
    if pct is None:
        pct = Config.WHITE_PATCH_PERCENTILE
    img = xp.array(rgb, dtype=xp.float32, copy=True)
    scales = xp.array([float(xp.percentile(img[:, :, c], pct)) for c in range(3)])
    scales = scales / (scales.mean() + 1e-12)
    return xp.clip(img / scales[None, None, :], 0, None)


def remove_hot_pixels(img, threshold: float = None):
    gpu = get_gpu()
    xp = gpu.xp
    if threshold is None:
        threshold = Config.HOT_PIXEL_THRESHOLD
    med = gpu.xndimage.median_filter(img, size=3)
    diff = img - med
    sigma = float(xp.std(diff))
    # threshold is a sigma multiplier (e.g. 12.0 = 12-sigma detection)
    mask = diff > threshold * sigma
    if not bool(xp.any(mask)):
        return img
    img_fixed = xp.array(img, copy=True)
    img_fixed[mask] = med[mask]
    return img_fixed


def remove_hot_pixels_bayer(data: np.ndarray, threshold: float = None) -> np.ndarray:
    """Detect and replace hot pixels on raw Bayer data per sub-channel.

    Each 2x2 Bayer sub-channel (R, G1, G2, B) is processed independently,
    so single-color hot pixels are detected at full strength — unlike
    luminance-based detection which dilutes them by 70-89%.

    Uses MAD-based sigma (robust to outliers) instead of std, preventing
    hot pixels from inflating the noise estimate and hiding themselves.
    """
    if threshold is None:
        threshold = Config.HOT_PIXEL_BAYER_THRESHOLD
    if data.ndim != 2:
        return data
    result = data.astype(np.float32, copy=True)
    for dy in range(2):
        for dx in range(2):
            sub = result[dy::2, dx::2]
            med = ndimage.median_filter(sub, size=3)
            diff = sub - med
            mad = np.median(np.abs(diff))
            sigma = mad * 1.4826  # MAD to Gaussian sigma
            if sigma < 1e-6:
                continue
            mask = diff > threshold * sigma
            if np.any(mask):
                sub[mask] = med[mask]
                result[dy::2, dx::2] = sub
    return result


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


def apply_hot_pixel_map_bayer(data: np.ndarray, hot_map: np.ndarray) -> np.ndarray:
    """Replace hot pixels in Bayer data using same-color median neighbors.

    Processes each Bayer sub-channel (R, G1, G2, B) independently so the
    3x3 median filter only uses same-color pixels (equivalent to 6x6 on
    the full grid).  This preserves color accuracy.
    """
    if hot_map is None or not np.any(hot_map):
        return data
    result = data.astype(np.float32, copy=True)
    for dy in range(2):
        for dx in range(2):
            sub = result[dy::2, dx::2]
            mask = hot_map[dy::2, dx::2]
            if np.any(mask):
                med = ndimage.median_filter(sub, size=3)
                sub[mask] = med[mask]
                result[dy::2, dx::2] = sub
    return result


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

    for c_idx in (0, 2):  # Red, Blue
        ch = rgb[:, :, c_idx].astype(np.float64)
        ch_std = ch.std()
        if ch_std < 1e-12:
            continue
        ch_norm = (ch - ch.mean()) / ch_std
        try:
            import warnings
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


def background_gradient_subtract(img):
    gpu = get_gpu()
    blurred = gpu.xndimage.gaussian_filter(img, sigma=max(15, min(img.shape) // 20))
    return img - blurred


def remove_hot_pixels_rgb(rgb, threshold: float = None):
    """Detect hot pixels on luminance (1 filter pass), fix all 3 channels."""
    gpu = get_gpu()
    xp = gpu.xp
    if threshold is None:
        threshold = Config.HOT_PIXEL_THRESHOLD
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    med_lum = gpu.xndimage.median_filter(lum, size=3)
    diff = lum - med_lum
    sigma = float(xp.std(diff))
    # threshold is a sigma multiplier (e.g. 12.0 = 12-sigma detection)
    mask = diff > threshold * sigma
    if not bool(xp.any(mask)):
        return rgb
    result = xp.array(rgb, copy=True)
    for c in range(rgb.shape[2]):
        ch_med = gpu.xndimage.median_filter(rgb[:, :, c], size=3)
        result[:, :, c][mask] = ch_med[mask]
    return result
