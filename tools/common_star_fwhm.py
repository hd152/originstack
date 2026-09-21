"""Star sharpness of two stacks, measured on the SAME stars and each stack's own pixel grid.

Comparing "the FWHM of stack A" with "the FWHM of stack B" is unreliable when each is measured
over its own detected star list: which stars are picked (bright saturated ones widen the
result, faint ones narrow it) changes the number by more than the difference being claimed.
This measures one shared set instead:

  1. Detect stars in stack B and keep well-behaved ones: bright enough to fit, not saturated,
     isolated, away from the edge.
  2. Map those positions into stack A's pixel coordinates with a rigid star-pattern match. Only
     coordinates are transformed; neither image is resampled, so neither is blurred by the
     comparison itself.
  3. Fit a circular Gaussian + background to every star in both stacks and report the median
     FWHM and the spread, on the stars both fits accept.

Public entry point: ``common_star_fwhm(cube_a, cube_b)`` -> dict.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

_HALF = 8           # cutout half-width in pixels
_FWHM_PER_SIGMA = 2.3548


def _gauss_fit(img, x, y):
    """(fwhm, amplitude, peak, x0, y0) of the star near (x, y), or None if the fit is not usable."""
    xi, yi = int(round(x)), int(round(y))
    if xi < _HALF + 1 or yi < _HALF + 1 or xi >= img.shape[1] - _HALF - 1 or yi >= img.shape[0] - _HALF - 1:
        return None
    cut = img[yi - _HALF:yi + _HALF + 1, xi - _HALF:xi + _HALF + 1].astype(np.float64)
    yy, xx = np.mgrid[-_HALF:_HALF + 1, -_HALF:_HALF + 1]
    edge = np.concatenate([cut[0], cut[-1], cut[1:-1, 0], cut[1:-1, -1]])
    bg0, amp0 = float(np.median(edge)), float(cut.max() - np.median(edge))
    if amp0 <= 0:
        return None

    def resid(p):
        a, dx, dy, sig, bg = p
        return ((bg + a * np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * sig ** 2))) - cut).ravel()

    try:
        sol = least_squares(resid, [amp0, 0.0, 0.0, 1.8, bg0],
                            bounds=([0, -3, -3, 0.5, -np.inf], [np.inf, 3, 3, 8.0, np.inf]), max_nfev=60)
    except Exception:
        return None
    a, dx, dy, sig, bg = sol.x
    return _FWHM_PER_SIGMA * sig, a, float(cut.max()), xi + dx, yi + dy


def _luma(cube):
    lum = np.asarray(cube, dtype=np.float32).mean(0)
    return lum - np.median(lum)


def common_star_fwhm(cube_a, cube_b, max_stars=400):
    """FWHM of the same stars in two (3, H, W) linear stacks. Returns a dict with the median
    FWHM of each (a, b), their spread (IQR), the star count, and the median per-star ratio a/b."""
    from src.blind_match import match_rigid_unknown_rotation
    from src.star_detect import detect_stars_matched_filter as detect

    la, lb = _luma(cube_a), _luma(cube_b)
    sa, sb = detect(la), detect(lb)
    sa, sb = sa[np.argsort(-sa["flux"])], sb[np.argsort(-sb["flux"])]
    rigid = match_rigid_unknown_rotation(sb, sa, max_stars=60, pixel_tol=2.0)     # b -> a coordinates
    if rigid is None:
        return None
    p = rigid.params

    xs, ys = np.asarray(sb["xcentroid"], float), np.asarray(sb["ycentroid"], float)
    order = np.argsort(-np.asarray(sb["flux"]))
    sat_b = np.percentile(lb, 99.99) * 0.6
    keep = []
    for i in order:
        d2 = (xs - xs[i]) ** 2 + (ys - ys[i]) ** 2
        d2[i] = np.inf
        if d2.min() < 12 ** 2:                       # not isolated
            continue
        keep.append(i)
        if len(keep) >= 6 * max_stars:
            break

    fa, fb = [], []
    for i in keep:
        rb = _gauss_fit(lb, xs[i], ys[i])
        if rb is None or rb[2] >= sat_b or rb[1] < 40 * 1.0:
            continue
        ax = p[0, 0] * xs[i] + p[0, 1] * ys[i] + p[0, 2]
        ay = p[1, 0] * xs[i] + p[1, 1] * ys[i] + p[1, 2]
        ra = _gauss_fit(la, ax, ay)
        if ra is None or ra[2] >= np.percentile(la, 99.99) * 0.6:
            continue
        if not (1.0 < rb[0] < 14 and 1.0 < ra[0] < 14):
            continue
        fa.append(ra[0]); fb.append(rb[0])
        if len(fa) >= max_stars:
            break
    if len(fa) < 20:
        return None
    fa, fb = np.array(fa), np.array(fb)
    q = lambda v: (float(np.percentile(v, 25)), float(np.percentile(v, 75)))       # noqa: E731
    return {"n": len(fa), "a": float(np.median(fa)), "b": float(np.median(fb)),
            "a_iqr": q(fa), "b_iqr": q(fb), "ratio_a_over_b": float(np.median(fa / fb))}
