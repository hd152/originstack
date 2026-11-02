"""Star width across the field, and an equalising second pass for a tilted PSF.

Every Origin stack measured (five, across nights and targets) has a clean linear gradient of star
FWHM along one sensor axis: the left edge is 12-55% wider than the right, the plane fit leaving
1-2.5% (``dev-notes/stack-optics.md``). ``--deconvolve nn`` flattens most of it (Omega max/min
1.55 -> 1.33) but not all. ``equalise_tilt`` runs the network a second time per strip along the gradient and blends each strip's
result in just far enough that the strip's star width reaches the sharpest strip's (measured on the
strip's own stars before and after; never beyond it). Strips at the sharp end are left alone.

Five stacks, `dev-notes/stack-optics.md`: star aperture flux conserved to 0.5-2%, 24 px-smoothed
luminance within 1%, sky noise 2-9% lower (the network denoises as well).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

_log = logging.getLogger("originstack")

MIN_STARS_PER_CELL = 6


def star_fwhm(ch: np.ndarray, y: float, x: float, half: int = 10) -> Optional[float]:
    """FWHM (px) of one star from adaptive Gaussian-weighted second moments (Bernstein & Jarvis
    2002 style): the centroid and sigma are iterated with a Gaussian window of the current sigma,
    so no half-max pixel count (``quality.measure_fwhm`` returns values quantised to 1.732 steps)
    and the background cannot drag it wide. Exact for a Gaussian star. None for a stamp that
    leaves the frame or holds no signal."""
    H, W = ch.shape
    iy, ix = int(round(y)), int(round(x))
    if iy - half < 0 or ix - half < 0 or iy + half >= H or ix + half >= W:
        return None
    w = ch[iy - half:iy + half + 1, ix - half:ix + half + 1].astype(np.float64)
    ring = np.concatenate([w[0], w[-1], w[1:-1, 0], w[1:-1, -1]])
    w = w - np.median(ring)
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    cy, cx, sig = 0.0, 0.0, 1.8
    for _ in range(6):
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))
        wg = w * g
        tot = wg.sum()
        if tot <= 0:
            return None
        cy, cx = (wg * yy).sum() / tot, (wg * xx).sum() / tot
        m2 = (wg * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / tot / 2.0
        v = sig ** 2
        denom = v - m2
        if denom <= 1e-6:
            return None
        s2 = m2 * v / denom                 # a star of variance s2 seen through a window of variance v
        if not np.isfinite(s2) or s2 <= 0.2:
            return None
        sig = float(np.sqrt(s2))
    return 2.3548 * sig


def bright_unsaturated(lum: np.ndarray, sources, n: int, sigma: float) -> List[int]:
    """Indices (brightest first, at most ``n``) of stars 25 sigma above the sky and below 70% of the
    frame's 99.99th percentile, away from the border."""
    top = float(np.percentile(lum, 99.99))
    med = float(np.median(lum))
    H, W = lum.shape
    keep: List[int] = []
    for i in np.argsort(-np.asarray(sources['flux'])):
        y, x = int(round(float(sources['ycentroid'][i]))), int(round(float(sources['xcentroid'][i])))
        if 12 < y < H - 12 and 12 < x < W - 12 and sigma * 25 < lum[y, x] - med < 0.7 * top:
            keep.append(int(i))
        if len(keep) >= n:
            break
    return keep


def cell_map(lum: np.ndarray, sources, grid: int, sigma: float) -> np.ndarray:
    """Median adaptive-moment FWHM per cell of a GxG grid (NaN: fewer than 6 good stars)."""
    H, W = lum.shape
    idx = bright_unsaturated(lum, sources, 1500, sigma)
    ys = np.asarray(sources['ycentroid'], float)[idx]
    xs = np.asarray(sources['xcentroid'], float)[idx]
    f = np.array([star_fwhm(lum, y, x) or np.nan for y, x in zip(ys, xs)])
    out = np.full((grid, grid), np.nan)
    for gy in range(grid):
        for gx in range(grid):
            m = ((ys >= gy * H / grid) & (ys < (gy + 1) * H / grid)
                 & (xs >= gx * W / grid) & (xs < (gx + 1) * W / grid) & np.isfinite(f))
            if m.sum() >= MIN_STARS_PER_CELL:
                out[gy, gx] = float(np.median(f[m]))
    return out


def _strip_widths(lum: np.ndarray, pos: np.ndarray, across: np.ndarray, edges: np.ndarray
                  ) -> np.ndarray:
    """Median star FWHM in each strip ``[edges[i], edges[i+1])`` of the coordinate ``pos`` (NaN: fewer
    than 6 measurable stars). ``lum`` is in the same orientation as the star coordinates."""
    out = np.full(len(edges) - 1, np.nan)
    f = np.array([star_fwhm(lum, a, b) or np.nan for a, b in zip(across, pos)])
    for i in range(len(edges) - 1):
        m = (pos >= edges[i]) & (pos < edges[i + 1]) & np.isfinite(f)
        if m.sum() >= MIN_STARS_PER_CELL:
            out[i] = float(np.median(f[m]))
    return out


def equalise_tilt(img: np.ndarray, deconvolve: Callable[[np.ndarray, float], np.ndarray],
                  sigma: float, n_strips: int = 6, min_ratio: float = 1.12, pad: int = 80,
                  feather: int = 40, tolerance: float = 0.04
                  ) -> Tuple[np.ndarray, Optional[Dict[str, object]]]:
    """Second deconvolver pass per strip along the star-width gradient of ``img`` (H, W, 3).

    The strips are cut along whichever sensor axis shows the larger max/min width. The sharpest
    strip is the target. Every strip wider than ``target * (1 + tolerance)`` is run once through
    ``deconvolve(segment, 1.0)``; its widths before (``w0``) and after (``w1``) are measured on the
    strip's own stars and the output is the blend ``segment + s * (full - segment)`` with
    ``s = (w0 - target) / (w0 - w1)`` clipped to [0, 1], so the strip lands on the target width, and
    never beyond it. A strip the pass does not narrow is left as it was.

    Returns ``(image, info)``; ``info`` is None (and ``image`` the input) when the stars are not
    tilted by ``min_ratio`` or too few were measured."""
    from src.debayer import luminance
    from src.quality import detect_stars_auto
    lum = luminance(img).astype(np.float32)
    src = detect_stars_auto(lum, float(sigma), background=float(np.median(lum)))
    if src is None or len(src) < 64:
        return img, None
    idx = bright_unsaturated(lum, src, 1500, float(sigma))
    ys = np.asarray(src['ycentroid'], float)[idx]
    xs = np.asarray(src['xcentroid'], float)[idx]
    H, W = lum.shape
    wx = _strip_widths(lum, xs, ys, np.linspace(0, W, n_strips + 1))
    wy = _strip_widths(np.ascontiguousarray(lum.T), ys, xs, np.linspace(0, H, n_strips + 1))

    def spread(w):
        return float(np.nanmax(w) / np.nanmin(w)) if np.isfinite(w).sum() >= 3 else 0.0
    axis = 1 if spread(wx) >= spread(wy) else 0                    # 1: strips along x
    w0 = wx if axis == 1 else wy
    if spread(w0) < min_ratio:
        return img, None
    work = img if axis == 1 else np.ascontiguousarray(img.transpose(1, 0, 2))
    lum_w = lum if axis == 1 else np.ascontiguousarray(lum.T)
    pos, across = (xs, ys) if axis == 1 else (ys, xs)
    L = work.shape[1]
    edges = np.linspace(0, L, n_strips + 1).astype(int)
    target = float(np.nanmin(w0))
    out = np.zeros_like(work, dtype=np.float32)
    wsum = np.zeros(L)
    blends: List[float] = []
    for i in range(n_strips):
        x0, x1 = int(edges[i]), int(edges[i + 1])
        lo, hi = max(0, x0 - pad), min(L, x1 + pad)
        seg = work[:, lo:hi]
        s = 0.0
        if np.isfinite(w0[i]) and w0[i] > target * (1.0 + tolerance):
            full = deconvolve(np.ascontiguousarray(seg), 1.0) if axis == 1 else np.asarray(
                deconvolve(np.ascontiguousarray(seg.transpose(1, 0, 2)), 1.0)).transpose(1, 0, 2)
            full = np.asarray(full, dtype=np.float32)
            core = (pos >= x0) & (pos < x1)
            lum_full = luminance(full).astype(np.float32)
            f1 = np.array([star_fwhm(lum_full, a, b - lo) or np.nan
                           for a, b in zip(across[core], pos[core])])
            w1 = float(np.nanmedian(f1)) if np.isfinite(f1).sum() >= MIN_STARS_PER_CELL else np.nan
            if np.isfinite(w1) and w0[i] - w1 > 1e-3:
                s = float(np.clip((w0[i] - target) / (w0[i] - w1), 0.0, 1.0))
            if s > 0.0:
                seg = seg + np.float32(s) * (full - seg)
        blends.append(round(s, 2))
        xs_ = np.arange(lo, hi)
        w = np.clip((xs_ - (x0 - feather)) / feather, 0, 1) * np.clip(((x1 + feather) - xs_) / feather, 0, 1)
        out[:, lo:hi] += np.asarray(seg, dtype=np.float32) * w[None, :, None].astype(np.float32)
        wsum[lo:hi] += w
    out /= np.maximum(wsum, 1e-6)[None, :, None].astype(np.float32)
    if axis == 0:
        out = np.ascontiguousarray(out.transpose(1, 0, 2))
    return out, {'axis': 'x' if axis == 1 else 'y',
                 'strip_fwhm': [round(float(v), 2) if np.isfinite(v) else None for v in w0],
                 'target': round(target, 2), 'blends': blends}
