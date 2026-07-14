"""Field aberration / optical-tilt inspector.

Bins detected stars into a grid over the frame and measures, per cell, the
median star FWHM plus the intensity-weighted elongation (magnitude + position
angle). From the spatial pattern it diagnoses the common optical faults an
astrophotographer chases:

  * **Sensor tilt** — a roughly *linear* FWHM gradient across the frame (one
    edge sharp, the opposite edge bloated). Reported with the "downhill"
    (toward-focus) direction.
  * **Field curvature / backfocus (spacing) error** — FWHM grows *radially*
    from the frame centre with stars elongated *tangentially* (curvature) or
    *radially* (coma from wrong corrector spacing).
  * **Astigmatism** — high elongation with no consistent radial/linear pattern.

Everything here is measured directly from the luminance image + a star
centroid catalogue (no dependency on the detector's shape columns), so it works
with either the SEP or DAOStarFinder catalogues the pipeline produces. Output is
a printed summary, a returned dict (for the FITS header / JSON), and — when
Pillow is available — an annotated PNG with one ellipse per populated cell,
sized by FWHM and coloured green→red worst-to-best.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except Exception:  # pragma: no cover - optional dependency
    Image = ImageDraw = None  # type: ignore[assignment]
    _HAS_PIL = False

from src.utils import safe_print


def _star_shape(cutout: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Measure (fwhm, ellipticity, angle_rad) of a single background-subtracted
    star cutout via the half-max area + intensity-weighted second moments.

    ``ellipticity`` = 1 - b/a in [0, 1) (0 = round); ``angle`` is the position
    angle of the major axis measured from +x (columns) toward +y (rows), in
    radians. Returns None for low-contrast / unusable cutouts."""
    c = cutout.astype(np.float64)
    peak = float(c.max())
    # Robust local background: lower-quartile of the cutout.
    flat = c.ravel()
    bg = float(np.partition(flat, flat.size // 4)[flat.size // 4])
    signal = peak - bg
    if signal < 10.0 or signal < bg * 0.3:
        return None

    half = (peak + bg) / 2.0
    mask = c > half
    n_above = int(mask.sum())
    if n_above < 3:
        return None
    fwhm = 2.0 * np.sqrt(n_above / np.pi)

    # Intensity-weighted second moments over the above-half-max footprint.
    ys, xs = np.nonzero(mask)
    wts = (c[ys, xs] - bg)
    wsum = float(wts.sum())
    if wsum <= 0:
        return None
    yc = float((wts * ys).sum() / wsum)
    xc = float((wts * xs).sum() / wsum)
    dy = ys - yc
    dx = xs - xc
    mxx = float((wts * dx * dx).sum() / wsum)
    myy = float((wts * dy * dy).sum() / wsum)
    mxy = float((wts * dx * dy).sum() / wsum)

    # Eigen-decomposition of the 2x2 moment matrix -> major/minor axes.
    tr = mxx + myy
    det = mxx * myy - mxy * mxy
    disc = max(tr * tr / 4.0 - det, 0.0)
    lam1 = tr / 2.0 + np.sqrt(disc)  # major
    lam2 = tr / 2.0 - np.sqrt(disc)  # minor
    if lam1 <= 1e-9:
        return fwhm, 0.0, 0.0
    a = np.sqrt(max(lam1, 0.0))
    b = np.sqrt(max(lam2, 0.0))
    ellip = float(1.0 - b / a) if a > 1e-9 else 0.0
    angle = 0.5 * np.arctan2(2.0 * mxy, mxx - myy)
    return fwhm, ellip, float(angle)


def analyze_field_aberration(
    lum: np.ndarray,
    sources: object,
    grid: int = 5,
    max_stars: int = 4000,
    cutout_radius: int = 9,
    output_png: Optional[str] = None,
    verbose: bool = False,
) -> Optional[Dict]:
    """Grid-binned field aberration analysis.

    Args:
        lum: 2-D luminance image (float).
        sources: star catalogue with ``xcentroid`` / ``ycentroid`` (and
            optionally ``flux``) columns.
        grid: cells per axis (grid x grid map).
        output_png: if given (and Pillow available), write an annotated map.

    Returns a summary dict, or None when too few stars were measurable.
    """
    if sources is None or len(sources) == 0:
        return None
    H, W = lum.shape[:2]
    r = int(cutout_radius)

    try:
        xs_all = np.asarray(sources['xcentroid'], dtype=np.float64)
        ys_all = np.asarray(sources['ycentroid'], dtype=np.float64)
    except Exception:
        return None
    try:
        order = np.argsort(np.asarray(sources['flux'], dtype=np.float64))[::-1]
    except Exception:
        order = np.arange(len(xs_all))
    order = order[:max_stars]

    # Per-star measurements collected into grid cells.
    cell_fwhm: List[List[float]] = [[] for _ in range(grid * grid)]
    cell_ellip: List[List[float]] = [[] for _ in range(grid * grid)]
    cell_vec = np.zeros((grid * grid, 2), dtype=np.float64)  # summed unit elong vectors
    cell_cnt = np.zeros(grid * grid, dtype=np.int64)
    all_fwhm: List[float] = []

    lum_f = lum.astype(np.float32, copy=False)
    for idx in order:
        x = xs_all[idx]
        y = ys_all[idx]
        ix = int(round(x))
        iy = int(round(y))
        if ix < r or ix >= W - r or iy < r or iy >= H - r:
            continue
        cutout = lum_f[iy - r:iy + r + 1, ix - r:ix + r + 1]
        shape = _star_shape(cutout)
        if shape is None:
            continue
        fwhm, ellip, ang = shape
        if not (0.5 <= fwhm < r * 2.5):
            continue
        gx = min(int(x / W * grid), grid - 1)
        gy = min(int(y / H * grid), grid - 1)
        cell = gy * grid + gx
        cell_fwhm[cell].append(fwhm)
        cell_ellip[cell].append(ellip)
        # Elongation direction is orientation (mod pi); double the angle so
        # opposite-pointing axes add coherently, weight by ellipticity.
        cell_vec[cell, 0] += ellip * np.cos(2.0 * ang)
        cell_vec[cell, 1] += ellip * np.sin(2.0 * ang)
        cell_cnt[cell] += 1
        all_fwhm.append(fwhm)

    if len(all_fwhm) < max(8, grid):
        return None

    med_fwhm = np.full(grid * grid, np.nan)
    med_ellip = np.full(grid * grid, np.nan)
    ang_map = np.full(grid * grid, np.nan)
    for c in range(grid * grid):
        if cell_cnt[c] >= 2:
            med_fwhm[c] = float(np.median(cell_fwhm[c]))
            med_ellip[c] = float(np.median(cell_ellip[c]))
            if np.hypot(*cell_vec[c]) > 1e-9:
                ang_map[c] = 0.5 * np.arctan2(cell_vec[c, 1], cell_vec[c, 0])

    fwhm_grid = med_fwhm.reshape(grid, grid)
    valid = np.isfinite(med_fwhm)
    n_valid = int(valid.sum())

    summary: Dict = {
        'grid': grid,
        'n_stars_measured': len(all_fwhm),
        'fwhm_median': float(np.nanmedian(med_fwhm)),
        'fwhm_min': float(np.nanmin(med_fwhm)),
        'fwhm_max': float(np.nanmax(med_fwhm)),
        'ellipticity_median': float(np.nanmedian(med_ellip)),
        'fwhm_grid': fwhm_grid.tolist(),
    }
    # Corner-to-corner spread as a fraction of the best cell — the headline
    # "how uneven is the field" number.
    if summary['fwhm_min'] > 1e-6:
        summary['fwhm_spread_pct'] = 100.0 * (summary['fwhm_max'] - summary['fwhm_min']) / summary['fwhm_min']
    else:
        summary['fwhm_spread_pct'] = 0.0

    # --- Tilt: least-squares plane fit of FWHM over cell-centre coordinates. ---
    gy_i, gx_i = np.mgrid[0:grid, 0:grid]
    cx = (gx_i.ravel() + 0.5) / grid - 0.5   # normalised [-0.5, 0.5]
    cy = (gy_i.ravel() + 0.5) / grid - 0.5
    tilt_dir = None
    tilt_grad = 0.0
    curv_corr = 0.0
    if n_valid >= 4:
        A = np.column_stack([np.ones(n_valid), cx[valid], cy[valid]])
        coef, *_ = np.linalg.lstsq(A, med_fwhm[valid], rcond=None)
        gxc, gyc = float(coef[1]), float(coef[2])
        tilt_grad = float(np.hypot(gxc, gyc))          # FWHM px across full frame
        # Downhill (toward best focus) direction, image convention (y down).
        ang = np.degrees(np.arctan2(-gyc, -gxc)) % 360.0
        tilt_dir = _compass(ang)
        # --- Field curvature: correlation of FWHM with radius from centre. ---
        radius = np.hypot(cx[valid], cy[valid])
        if radius.std() > 1e-9 and med_fwhm[valid].std() > 1e-9:
            curv_corr = float(np.corrcoef(radius, med_fwhm[valid])[0, 1])

    summary['tilt_gradient_px'] = tilt_grad
    summary['tilt_direction'] = tilt_dir
    summary['curvature_corr'] = curv_corr

    # --- Diagnosis heuristics (thresholds tuned for typical DSO frames). ---
    diag: List[str] = []
    spread = summary['fwhm_spread_pct']
    if spread < 12.0:
        diag.append('Field is even — no significant tilt or curvature.')
    else:
        if tilt_grad > 0.6 and (curv_corr < 0.5):
            diag.append(f'Sensor tilt likely: FWHM rises ~{tilt_grad:.1f}px across '
                        f'the frame; adjust tilt toward {tilt_dir} (the soft side).')
        if curv_corr > 0.55:
            diag.append('Field curvature / backfocus (spacing) error: FWHM grows '
                        'radially from centre — check corrector-to-sensor spacing.')
        if summary['ellipticity_median'] > 0.30 and curv_corr <= 0.55 and tilt_grad <= 0.6:
            diag.append('Astigmatism / guiding: high elongation with no clean '
                        'radial or linear pattern.')
        if not diag:
            diag.append(f'Uneven field ({spread:.0f}% FWHM spread) — mild tilt/spacing.')
    summary['diagnosis'] = diag

    if verbose or True:
        safe_print("  Field aberration inspector:")
        safe_print(f"    Stars measured: {len(all_fwhm)}  |  FWHM median "
                   f"{summary['fwhm_median']:.2f}px  (best {summary['fwhm_min']:.2f} / "
                   f"worst {summary['fwhm_max']:.2f}, spread {spread:.0f}%)")
        safe_print(f"    Median ellipticity: {summary['ellipticity_median']:.2f}"
                   + (f"  |  tilt grad {tilt_grad:.2f}px toward {tilt_dir}"
                      if tilt_dir else ""))
        for d in diag:
            safe_print(f"    → {d}")

    if output_png and _HAS_PIL:
        try:
            _render_png(fwhm_grid, med_ellip.reshape(grid, grid),
                        ang_map.reshape(grid, grid), W, H, output_png, summary)
            safe_print(f"    Saved: {os.path.basename(output_png)}")
        except Exception as exc:  # pragma: no cover - rendering is best-effort
            safe_print(f"    WARNING: aberration PNG failed: {exc}")

    return summary


_COMPASS = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']


def _compass(angle_deg: float) -> str:
    """8-point compass label for an image-space angle (0=+x/right, 90=up)."""
    return _COMPASS[int(((angle_deg + 22.5) % 360) / 45.0)]


def _render_png(fwhm_grid: np.ndarray, ellip_grid: np.ndarray,
                ang_grid: np.ndarray, W: int, H: int, path: str,
                summary: Dict) -> None:
    """Annotated aberration map: one ellipse per populated cell, sized by FWHM,
    oriented by elongation, coloured green (best) → red (worst)."""
    grid = fwhm_grid.shape[0]
    scale = 900.0 / max(W, H)
    cw = W * scale / grid
    ch = H * scale / grid
    iw, ih = int(W * scale), int(H * scale)
    img = Image.new('RGB', (iw, ih + 40), (18, 18, 22))
    d = ImageDraw.Draw(img)

    fmin = summary['fwhm_min']
    fmax = max(summary['fwhm_max'], fmin + 1e-6)
    for gy in range(grid):
        for gx in range(grid):
            f = fwhm_grid[gy, gx]
            cx = (gx + 0.5) * cw
            cy = (gy + 0.5) * ch
            # cell border
            d.rectangle([gx * cw, gy * ch, (gx + 1) * cw, (gy + 1) * ch],
                        outline=(60, 60, 70))
            if not np.isfinite(f):
                continue
            t = (f - fmin) / (fmax - fmin)   # 0 best -> 1 worst
            col = (int(40 + 215 * t), int(220 - 180 * t), 60)
            e = ellip_grid[gy, gx] if np.isfinite(ellip_grid[gy, gx]) else 0.0
            ang = ang_grid[gy, gx] if np.isfinite(ang_grid[gy, gx]) else 0.0
            rmaj = 0.42 * min(cw, ch) * (0.4 + 0.6 * t + 0.5 * e)
            rmin = rmaj * (1.0 - min(e, 0.85))
            ca, sa = np.cos(ang), np.sin(ang)
            pts = []
            for k in range(24):
                th = 2 * np.pi * k / 24
                ex = rmaj * np.cos(th)
                ey = rmin * np.sin(th)
                pts.append((cx + ex * ca - ey * sa, cy + ex * sa + ey * ca))
            d.polygon(pts, outline=col)
            d.text((gx * cw + 3, gy * ch + 3), f"{f:.1f}", fill=col)
    d.text((6, ih + 12),
           f"FWHM {summary['fwhm_median']:.2f}px  spread {summary['fwhm_spread_pct']:.0f}%"
           f"   ellip {summary['ellipticity_median']:.2f}"
           + (f"   tilt->{summary['tilt_direction']}" if summary['tilt_direction'] else ""),
           fill=(200, 200, 210))
    img.save(path, format='PNG')
