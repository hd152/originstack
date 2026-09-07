"""Pure image preprocessing - percentile stretch and resize/center-crop.

Split out from data/cache.py so the inference path can reuse the exact same
preprocessing without dragging in astropy (FITS) or torch. Only cv2 + numpy.
data/cache.py and infer.py re-import these; infer_onnx.py imports them here
directly.
"""

import cv2
import numpy as np


def stretch_to_uint8(img, low_pct=0.5, high_pct=99.5):
    """Percentile clip + linear stretch, applied per-channel. Raw Bayer
    channels have very different gains (2x as many green photosites, plus
    QE/filter differences) so a single global percentile across all
    channels leaves a strong color cast; stretching each channel against
    its own percentile range is a cheap approximation of white balance."""
    if img.ndim == 2:
        lo, hi = np.percentile(img, [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1.0
        out = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        return (out * 255).astype(np.uint8)

    out = np.empty(img.shape, dtype=np.uint8)
    for c in range(img.shape[2]):
        lo, hi = np.percentile(img[..., c], [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1.0
        chan = np.clip((img[..., c].astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        out[..., c] = (chan * 255).astype(np.uint8)
    return out


def resize_center_crop(img, size):
    """Resize shorter side to `size`, center-crop to size x size square.
    Avoids distorting star shapes, which matters for the quality/defect task."""
    h, w = img.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    return resized[top : top + size, left : left + size]


def apply_background_correction(img_uint8, grid):
    """Flatten background unevenness in a uint8 image using a predicted
    background grid (the models.heads.BackgroundGridHead output - a small
    grid_size x grid_size array in the same [0,1]-ish units as the
    normalized image it was regressed against; see
    data/dataset.py background_grid_target).

    `grid` is upsampled (bicubic) to img_uint8's own resolution, then its
    *deviation from its own mean* is subtracted from every channel equally
    - not the raw background level itself, so overall exposure is
    preserved and only the unevenness (gradient / light-pollution
    asymmetry / vignetting) is removed. Same correction applied to all
    channels: the grid is a single luminance-proxy prediction (see
    background_grid_target's per-channel-averaged target), not a per-
    channel one, so this flattens brightness unevenness, not color cast.

    img_uint8 can be any channel order (BGR or RGB) or grayscale - treated
    identically since the correction is applied per-pixel, uniformly across
    whatever channels are present."""
    h, w = img_uint8.shape[:2]
    grid_np = np.asarray(grid, dtype=np.float32)
    upsampled = cv2.resize(grid_np, (w, h), interpolation=cv2.INTER_CUBIC)
    correction = (upsampled - upsampled.mean()) * 255.0
    if img_uint8.ndim == 3:
        correction = correction[:, :, None]
    corrected = img_uint8.astype(np.float32) - correction
    return np.clip(corrected, 0, 255).astype(np.uint8)
