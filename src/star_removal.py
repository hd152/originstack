"""Star removal: inpaint detected stars with local background.

Produces a "starless" sidecar image for downstream nebula/background work
(aggressive stretch, external DBE-style extraction, blended star
recombination) without a full ML star-extraction model (StarNet/
StarXTerminator-style -- large embedded network weights, extra dependency,
out of scope for this project's no-heavy-ML policy). Same normalised-
convolution inpaint technique src/trail_reject.py uses for satellite trails:
fill each star's footprint from the surrounding background, weighted only by
non-star pixels so star flux never smears back into its own gap.

Per-star footprint radius scales with the star's measured peak brightness --
saturated/bloomed stars need a much bigger disk than faint point sources, and
a single fixed radius (fine for trail_reject's near-constant trail width)
would under-erase bright stars and over-erase faint ones.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.utils import safe_print

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    gaussian_filter = None
    _HAS_SCIPY = False

# Safety cap, same spirit as trail_reject's max_area_frac guard: a mask this
# large means star detection went haywire (huge galaxy core, comet coma
# misread as a field of giant "stars") -- abort rather than gut the frame.
_MAX_MASK_AREA_FRAC = 0.6
_MAX_STARS = 4000


def build_star_mask(shape: Tuple[int, int], sources, fwhm: float,
                    radius_scale: float = 1.8,
                    max_radius_frac: float = 0.02,
                    ) -> Tuple[Optional[np.ndarray], float]:
    """Boolean mask covering each detected star's footprint.

    Each star's disk radius scales with sqrt(peak / median_peak) off a
    ``radius_scale * fwhm`` base, clamped to [0.8*fwhm, max_radius_frac *
    min(H, W)]. Returns (mask, max_radius_used_px), or (None, 0.0) if no
    stars were usable or the mask blew up past the safety cap.
    """
    if sources is None or len(sources) == 0:
        return None, 0.0
    H, W = shape
    n = min(len(sources), _MAX_STARS)
    peaks = np.asarray(sources['peak'][:n], dtype=np.float64)
    positive = peaks[peaks > 0]
    med_peak = float(np.median(positive)) if positive.size else 1.0
    med_peak = max(med_peak, 1e-6)

    min_r = max(2.0, fwhm * 0.8)
    max_r = max(min_r, max_radius_frac * min(H, W))

    ys_c = np.asarray(sources['ycentroid'][:n], dtype=np.float64)
    xs_c = np.asarray(sources['xcentroid'][:n], dtype=np.float64)

    mask = np.zeros((H, W), dtype=bool)
    max_r_used = 0.0
    for i in range(n):
        r = radius_scale * fwhm * np.sqrt(max(peaks[i], 0.0) / med_peak)
        r = float(np.clip(r, min_r, max_r))
        cy, cx = ys_c[i], xs_c[i]
        y0, y1 = max(0, int(cy - r)), min(H, int(cy + r) + 1)
        x0, x1 = max(0, int(cx - r)), min(W, int(cx + r) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        mask[y0:y1, x0:x1] |= disk
        max_r_used = max(max_r_used, r)

    if not mask.any():
        return None, 0.0
    if mask.mean() > _MAX_MASK_AREA_FRAC:
        return None, 0.0
    return mask, max_r_used


def remove_stars(rgb: np.ndarray, sources, fwhm: float,
                 radius_scale: float = 1.8,
                 verbose: bool = False) -> Tuple[np.ndarray, int]:
    """Inpaint detected stars with normalised-convolution local background.

    Args:
        rgb: (H, W, 3) float32 image.
        sources: structured star-source array (quality.detect_stars_auto's
            output), or None.
        fwhm: representative stellar FWHM (px), used as the base disk size.

    Returns (starless_rgb, n_stars_removed) -- a copy of ``rgb`` unchanged
    (n=0) when scipy is unavailable, no stars were found, or the mask tripped
    the safety cap.
    """
    if not _HAS_SCIPY or rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb.copy(), 0
    mask, max_r = build_star_mask(rgb.shape[:2], sources, fwhm, radius_scale=radius_scale)
    if mask is None:
        return rgb.copy(), 0

    # Normalised convolution: smooth background from the non-star pixels
    # only, so the fill never smears star flux back into the gap. Bandwidth
    # scales with the largest star disk actually used, so a field of mostly
    # small stars gets a tight fill while a few bloated saturated stars still
    # get a wide enough one.
    fill_sigma = max(6.0, 1.5 * max_r)
    out = rgb.astype(np.float32, copy=True)
    keep = (~mask).astype(np.float32)
    denom = gaussian_filter(keep, sigma=fill_sigma)
    denom = np.maximum(denom, 1e-6)
    for c in range(3):
        ch = out[:, :, c]
        bg = gaussian_filter(ch * keep, sigma=fill_sigma) / denom
        ch[mask] = bg[mask]
        out[:, :, c] = ch

    n_stars = min(len(sources), _MAX_STARS)
    if verbose:
        safe_print(f"    Star removal: masked {n_stars} star(s), "
                   f"{int(mask.sum())} px inpainted (max disk radius {max_r:.1f}px)")
    return out, n_stars
