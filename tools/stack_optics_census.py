"""What the optics leave in a finished stack: PSF variation across the field, colour-dependent
widths and lateral chromatic aberration.

``tools/psf_census.py`` looks at single frames; this looks at the stack, which is what the
post-coadd ideas (PSF homogenisation, a radial CA model) would correct. Field rotation during an
alt-az session is small (a few tenths of a degree), so what is fixed to the sensor stays fixed in
the stack. Per stack (``*.fits`` with ``RAWSTACK``):

  * FWHM of the unsaturated stars in each cell of a GxG grid (luminance, adaptive Gaussian-weighted
    moments; ``quality.measure_fwhm`` is quantised to 1.732 steps), the
    max/min over cells, a plane fit (tilt) and the residual after it (curvature, coma);
  * the median FWHM of the same stars in R, G and B (``colour_fwhm_ratio``);
  * lateral chromatic aberration: the centroid of each bright star in R and B against G, fitted
    as ``offset = a + b * (position - centre)``; ``b * half-diagonal`` is the radial shift at the
    corner, the number a radial CA model (Rd = b Ru^3 + c Ru^2 + v Ru, Lensfun) would remove.

Gates from the research report: a position-dependent PSF is worth building when max/min cell FWHM
is at least 1.05 (or ellipticity 0.05); a radial CA model when the corner shift is at least
0.15 px.

Usage:
  python tools/stack_optics_census.py STACK.fits [STACK2.fits ...] [--grid 4] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MIN_STARS_PER_CELL = 6
CENTROID_HALF = 4           # px: centroid window half-size (9x9)


def plane_fit(fwhm_grid: np.ndarray) -> Dict[str, float]:
    """Tilt and residual of a (G, G) cell map (NaN cells ignored): ``fwhm = c + a*x + b*y`` with x, y in
    [-0.5, 0.5] across the field. ``tilt_px`` = peak-to-peak of the plane across the field."""
    g = fwhm_grid.shape[0]
    yy, xx = np.mgrid[0:g, 0:g]
    x = (xx + 0.5) / g - 0.5
    y = (yy + 0.5) / g - 0.5
    ok = np.isfinite(fwhm_grid)
    if ok.sum() < 6:
        return {'tilt_px': float('nan'), 'resid_rel': float('nan')}
    A = np.column_stack([np.ones(ok.sum()), x[ok], y[ok]])
    coef, *_ = np.linalg.lstsq(A, fwhm_grid[ok], rcond=None)
    res = fwhm_grid[ok] - A @ coef
    mean = float(np.mean(fwhm_grid[ok]))
    return {'tilt_px': float(np.hypot(coef[1], coef[2])), 'resid_rel': float(np.std(res) / mean),
            'tilt_direction_deg': float(np.degrees(np.arctan2(coef[2], coef[1])))}


def star_fwhm(ch: np.ndarray, y: float, x: float, half: int = 10) -> Optional[float]:
    """FWHM (px) of one star from adaptive Gaussian-weighted second moments (Bernstein & Jarvis 2002
    style): the centroid and sigma are iterated with a Gaussian window of the current sigma, so the
    estimate does not depend on a half-max pixel count (``quality.measure_fwhm`` returns values
    quantised to 1.732 steps) and the background cannot drag it wide. The window is the true
    sigma-weighted moment; for a Gaussian star it returns the FWHM exactly. None for a stamp that
    leaves the frame or has no signal."""
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
        m2 = (wg * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / tot / 2.0      # per-axis variance of the weighted star
        # a Gaussian star of variance s^2 seen through a Gaussian window of variance v: m2 = s^2 v / (s^2 + v)
        v = sig ** 2
        denom = v - m2
        if denom <= 1e-6:
            return None
        s2 = m2 * v / denom
        if not np.isfinite(s2) or s2 <= 0.2:
            return None
        sig = float(np.sqrt(s2))
    return 2.3548 * sig


def _bright_unsaturated(stack: np.ndarray, sources, n: int, sigma: float) -> List[int]:
    lum = stack.mean(axis=2)
    top = float(np.percentile(lum, 99.99))
    H, W = lum.shape
    keep = []
    for i in np.argsort(-np.asarray(sources['flux'])):
        y, x = int(round(float(sources['ycentroid'][i]))), int(round(float(sources['xcentroid'][i])))
        if 12 < y < H - 12 and 12 < x < W - 12 and sigma * 25 < lum[y, x] - np.median(lum) < 0.7 * top:
            keep.append(int(i))
        if len(keep) >= n:
            break
    return keep


def cell_map(stack: np.ndarray, sources, grid: int, sigma: float, channel: Optional[int] = None) -> np.ndarray:
    """Median adaptive-moment FWHM per cell of a GxG grid, from the unsaturated stars of ``sources``
    (luminance, or one channel); NaN where a cell has fewer than ``MIN_STARS_PER_CELL`` good stars."""
    H, W, _ = stack.shape
    img = stack.mean(axis=2) if channel is None else stack[:, :, channel]
    idx = _bright_unsaturated(stack, sources, 1500, sigma)
    ys = np.asarray(sources['ycentroid'], float)[idx]
    xs = np.asarray(sources['xcentroid'], float)[idx]
    f = np.array([star_fwhm(img, y, x) or np.nan for y, x in zip(ys, xs)])
    out = np.full((grid, grid), np.nan)
    for gy in range(grid):
        for gx in range(grid):
            m = ((ys >= gy * H / grid) & (ys < (gy + 1) * H / grid)
                 & (xs >= gx * W / grid) & (xs < (gx + 1) * W / grid) & np.isfinite(f))
            if m.sum() >= MIN_STARS_PER_CELL:
                out[gy, gx] = float(np.median(f[m]))
    return out


def _centroids(ch: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Background-subtracted centroid of each star in one channel, (N, 2) as (dy, dx) from the
    integer start; NaN when the window leaves the frame."""
    H, W = ch.shape
    h = CENTROID_HALF
    out = np.full((len(ys), 2), np.nan)
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float64)
    for i, (y, x) in enumerate(zip(np.round(ys).astype(int), np.round(xs).astype(int))):
        if y - h < 0 or x - h < 0 or y + h >= H or x + h >= W:
            continue
        w = ch[y - h:y + h + 1, x - h:x + h + 1].astype(np.float64)
        w = w - np.median(np.concatenate([w[0], w[-1], w[:, 0], w[:, -1]]))
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s <= 0:
            continue
        out[i] = ((w * yy).sum() / s, (w * xx).sum() / s)
    return out


def lateral_ca(stack: np.ndarray, sources, n_stars: int = 400) -> Dict[str, float]:
    """Fit centroid(R) - centroid(G) and (B) - (G) against position; return the corner shift in px.

    Uses the brightest unsaturated stars. ``corner_shift_px`` is |b| * half-diagonal for the
    linear (radial scale) term; ``mean_offset_px`` the constant term (a global shift, which the
    existing per-session CA probe already removes)."""
    H, W, _ = stack.shape
    order = np.argsort(-np.asarray(sources['flux']))
    lum = stack.mean(axis=2)
    top = float(np.percentile(lum, 99.99))
    keep = []
    for i in order[:n_stars * 3]:
        y, x = int(round(float(sources['ycentroid'][i]))), int(round(float(sources['xcentroid'][i])))
        if CENTROID_HALF < y < H - CENTROID_HALF and CENTROID_HALF < x < W - CENTROID_HALF \
                and lum[y, x] < 0.8 * top:
            keep.append(i)
        if len(keep) >= n_stars:
            break
    if len(keep) < 40:
        return {'n_stars': len(keep)}
    ys = np.asarray(sources['ycentroid'])[keep]
    xs = np.asarray(sources['xcentroid'])[keep]
    cen = [_centroids(stack[:, :, c], ys, xs) for c in range(3)]
    px = (xs - W / 2) / (W / 2)
    py = (ys - H / 2) / (H / 2)
    half_diag = float(np.hypot(W, H) / 2)
    out: Dict[str, float] = {'n_stars': len(keep)}
    for name, c in (('R', 0), ('B', 2)):
        d = cen[c] - cen[1]                                   # (dy, dx) of c relative to G
        ok = np.isfinite(d).all(axis=1) & (np.abs(d) < 2.0).all(axis=1)
        if ok.sum() < 30:
            continue
        A = np.column_stack([np.ones(ok.sum()), px[ok], py[ok]])
        cx, *_ = np.linalg.lstsq(A, d[ok, 1], rcond=None)
        cy, *_ = np.linalg.lstsq(A, d[ok, 0], rcond=None)
        # radial scale term: the gradient of the dx with x and dy with y; corner shift in px
        scale = 0.5 * (cx[1] / (W / 2) + cy[2] / (H / 2))     # px of shift per px of radius
        out[f'{name}_minus_G_mean_offset_px'] = float(np.hypot(cx[0], cy[0]))
        out[f'{name}_minus_G_corner_shift_px'] = float(scale * half_diag)
        out[f'{name}_minus_G_scatter_px'] = float(1.4826 * np.median(np.abs(
            d[ok] - np.median(d[ok], axis=0))))
    return out


def channel_fwhm(stack: np.ndarray, sources, sigma: float) -> Dict[str, float]:
    idx = _bright_unsaturated(stack, sources, 600, sigma)
    ys = np.asarray(sources['ycentroid'], float)[idx]
    xs = np.asarray(sources['xcentroid'], float)[idx]
    med = []
    for c in range(3):
        v = np.array([star_fwhm(stack[:, :, c], y, x) or np.nan for y, x in zip(ys, xs)])
        med.append(float(np.nanmedian(v)))
    return {'fwhm_R': med[0], 'fwhm_G': med[1], 'fwhm_B': med[2],
            'colour_fwhm_ratio_B_over_G': med[2] / med[1] if med[1] > 0 else float('nan'),
            'colour_fwhm_ratio_R_over_G': med[0] / med[1] if med[1] > 0 else float('nan')}


def census(path: str, grid: int) -> Dict[str, object]:
    from src.background import _pixel_noise_sigma
    from src.debayer import luminance
    from src.merge import load_merge_stack
    from src.quality import detect_stars_auto
    stack, _meta = load_merge_stack(path)
    lum = luminance(stack).astype(np.float32)
    sig = _pixel_noise_sigma(lum)
    src = detect_stars_auto(lum, float(sig), background=float(np.median(lum)))
    res: Dict[str, object] = {'stack': os.path.basename(path), 'stars': 0 if src is None else len(src)}
    if src is None or len(src) < 60:
        return res
    grid_map = cell_map(stack, src, grid, float(sig))
    ok = np.isfinite(grid_map)
    res['cells_measured'] = int(ok.sum())
    if ok.sum() >= 6:
        res['fwhm_median'] = float(np.nanmedian(grid_map))
        res['fwhm_max_over_min'] = float(np.nanmax(grid_map) / np.nanmin(grid_map))
        res.update({f'plane_{k}': v for k, v in plane_fit(grid_map).items()})
        res['fwhm_by_column'] = [round(float(v), 2) for v in np.nanmedian(grid_map, axis=0)]
        res['fwhm_by_row'] = [round(float(v), 2) for v in np.nanmedian(grid_map, axis=1)]
    res.update(channel_fwhm(stack, src, float(sig)))
    res.update(lateral_ca(stack, src))
    return res


def verdict(r: Dict[str, object]) -> List[str]:
    out = []
    ratio = r.get('fwhm_max_over_min')
    if ratio is not None:
        out.append(("cell FWHM max/min %.2f >= 1.05: a position-dependent PSF is worth building"
                    if ratio >= 1.05 else
                    "cell FWHM max/min %.2f < 1.05: one PSF describes the field") % ratio)
    shifts = [abs(float(r[k])) for k in ('R_minus_G_corner_shift_px', 'B_minus_G_corner_shift_px') if k in r]
    if shifts:
        s = max(shifts)
        out.append(("corner CA shift %.2f px >= 0.15: a radial CA model would remove it"
                    if s >= 0.15 else "corner CA shift %.2f px < 0.15: no radial CA model needed") % s)
    return out


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument('stacks', nargs='+')
    ap.add_argument('--grid', type=int, default=4)
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    results = []
    for p in a.stacks:
        r = census(p, a.grid)
        results.append(r)
        print(f"\n{r['stack']}: {r['stars']} stars, {r.get('cells_measured', 0)} cells")
        for k, v in r.items():
            if k in ('stack', 'stars', 'cells_measured'):
                continue
            print(f"  {k:34s} {v:.4f}" if isinstance(v, float) else f"  {k:34s} {v}")
        for line in verdict(r):
            print("  ->", line)
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
