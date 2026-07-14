"""Satellite / aircraft trail rejection (per frame, pre-stack).

Sigma-clip and friends only reject a trail where enough *other* frames cover
the same pixel cleanly — which fails at low frame counts (a bright Starlink
streak in 1 of 5 subs survives into the stack). This module finds long straight
trails in each frame *before* it enters the stack and erases them by filling the
streak pixels with the local background (normalised-convolution inpaint), so the
result is robust for any stacking method and any frame count.

Detection: bright-pixel threshold → probabilistic Hough transform (skimage) to
find straight segments longer than a fraction of the frame. Compact sources
(stars) never form long collinear runs, so they are left alone. A global
area-fraction guard aborts the fill if detection went haywire (e.g. a diffraction
spike lattice), so a bad detection can never gut the frame.

Degrades gracefully: if scikit-image is unavailable the frame is returned
untouched.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.utils import safe_print

try:
    from skimage.transform import probabilistic_hough_line as _phl
    from skimage.draw import line as _draw_line
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover - optional dependency
    _phl = _draw_line = None
    _HAS_SKIMAGE = False

try:
    from scipy.ndimage import binary_dilation, gaussian_filter
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    binary_dilation = gaussian_filter = None
    _HAS_SCIPY = False


def detect_trail_mask(lum: np.ndarray, min_len_frac: float = 0.18,
                      thresh_sigma: float = 3.0, width: int = 4,
                      max_area_frac: float = 0.08) -> Tuple[Optional[np.ndarray], int]:
    """Return (boolean trail mask, n_segments) for a luminance frame, or
    (None, 0) if no trail was found / detection is unavailable."""
    if not (_HAS_SKIMAGE and _HAS_SCIPY):
        return None, 0
    H, W = lum.shape[:2]
    lum = np.asarray(lum, dtype=np.float32)

    med = float(np.median(lum))
    sig = 1.4826 * float(np.median(np.abs(lum - med)))
    if sig <= 0:
        sig = float(np.std(lum)) or 1.0
    binary = lum > (med + thresh_sigma * sig)
    if binary.sum() < H:  # essentially nothing bright
        return None, 0

    min_len = max(30, int(min_len_frac * min(H, W)))
    line_gap = max(3, int(0.02 * min(H, W)))
    try:
        segments = _phl(binary, threshold=10, line_length=min_len, line_gap=line_gap)
    except Exception:
        return None, 0
    if not segments:
        return None, 0

    mask = np.zeros((H, W), dtype=bool)
    for (x0, y0), (x1, y1) in segments:
        rr, cc = _draw_line(int(y0), int(x0), int(y1), int(x1))
        rr = np.clip(rr, 0, H - 1)
        cc = np.clip(cc, 0, W - 1)
        mask[rr, cc] = True
    if width > 0:
        mask = binary_dilation(mask, iterations=int(width))

    # Guard: if the "trail" covers too much of the frame the detection is
    # almost certainly wrong (dense field, big galaxy edge) — do nothing.
    if mask.mean() > max_area_frac:
        return None, 0
    return mask, len(segments)


def reject_trails(rgb: np.ndarray, lum: Optional[np.ndarray] = None,
                  min_len_frac: float = 0.18, width: int = 4,
                  fill_sigma: float = 12.0,
                  verbose: bool = False) -> Tuple[np.ndarray, int]:
    """Detect straight trails and inpaint them with local background.

    Args:
        rgb: (H, W, 3) float32 frame.
        lum: optional precomputed luminance (else derived).
        fill_sigma: Gaussian bandwidth (px) of the normalised-convolution
            background used to refill trail pixels.

    Returns (cleaned_rgb, n_segments). Unchanged (and n=0) when no trail found.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb, 0
    if lum is None:
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mask, n = detect_trail_mask(lum, min_len_frac=min_len_frac, width=width)
    if mask is None or n == 0:
        return rgb, 0

    out = rgb.astype(np.float32, copy=True)
    keep = (~mask).astype(np.float32)
    # Normalised convolution: smooth background from the non-trail pixels only,
    # so the fill never smears trail flux back into the gap.
    denom = gaussian_filter(keep, sigma=fill_sigma)
    denom = np.maximum(denom, 1e-6)
    for c in range(3):
        ch = out[:, :, c]
        bg = gaussian_filter(ch * keep, sigma=fill_sigma) / denom
        ch[mask] = bg[mask]
        out[:, :, c] = ch
    if verbose:
        safe_print(f"      trail rejection: masked {n} segment(s), "
                   f"{int(mask.sum())} px inpainted")
    return out, n
