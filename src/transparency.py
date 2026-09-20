"""Per-frame sky transparency from a fixed ensemble of stars.

Thin cloud, haze and dew dim the stars in a sub without necessarily changing its
FWHM or its measured SNR much (the background rises as the stars fall, and SNR
is a ratio), so the quality gate's metrics can pass a frame that has lost a
third of its light. This measures it directly: after registration, each frame's
stars are mapped into the reference frame's coordinates, matched to the
reference catalogue, and the median flux ratio of the matched (unsaturated,
well-detected) stars is that frame's transparency, normalised to the session
median so 1.0 is a typical frame.

Registration's transforms are the ones ``apply_transform`` uses, so the mapping
here is its inverse: a native pixel ``r`` (row, col) lands at
``R^-1 r + t`` (or ``r + shift`` for a pure translation), with ``R`` and ``t``
read exactly as ``apply_transform`` reads them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    cKDTree = None
    _HAS_SCIPY = False

_LOW_FLUX_PCT = 25.0    # drop the faintest reference stars (noisy fluxes)
_HIGH_FLUX_PCT = 90.0   # and the brightest (saturated / non-linear)


def to_aligned_yx(yx: np.ndarray, transform: Optional[Any],
                  shift: Optional[Tuple[float, float]]) -> np.ndarray:
    """Native (row, col) positions -> reference-aligned (row, col)."""
    yx = np.asarray(yx, dtype=np.float64)
    if transform is not None:
        m = transform.params
        R = m[:2, :2]
        t_rc = np.array([m[1, 2], m[0, 2]])
        return yx @ np.linalg.inv(R).T + t_rc
    if shift is not None:
        return yx + np.asarray(shift, dtype=np.float64)
    return yx.copy()


def _xy(stars) -> np.ndarray:
    return np.column_stack([np.asarray(stars['ycentroid'], dtype=np.float64),
                            np.asarray(stars['xcentroid'], dtype=np.float64)])


def _exptime(frame) -> float:
    try:
        v = float((frame.header or {}).get('EXPTIME', 0) or 0)
    except (TypeError, ValueError):
        v = 0.0
    return v if v > 0 else 1.0


def match_flux_ratio(ref_stars, frame_stars, transform, shift,
                     ref_exptime: float = 1.0, frame_exptime: float = 1.0,
                     tol: float = 4.0, min_matches: int = 8
                     ) -> Tuple[float, int]:
    """(median flux ratio frame/reference, matches used); (nan, n) if too few.

    ``tol`` is in pixels and deliberately generous: the residual gate's floor is
    1.5 px and real sessions sit near 2, so a tight radius would match only the
    best-registered stars."""
    if not _HAS_SCIPY or ref_stars is None or frame_stars is None:
        return float('nan'), 0
    if len(ref_stars) < min_matches or len(frame_stars) < min_matches:
        return float('nan'), 0
    ref_yx = _xy(ref_stars)
    ref_flux = np.asarray(ref_stars['flux'], dtype=np.float64) / ref_exptime
    lo, hi = np.percentile(ref_flux, [_LOW_FLUX_PCT, _HIGH_FLUX_PCT])
    use = (ref_flux >= lo) & (ref_flux <= hi) & (ref_flux > 0)
    if int(use.sum()) < min_matches:
        return float('nan'), int(use.sum())
    frm_yx = to_aligned_yx(_xy(frame_stars), transform, shift)
    frm_flux = np.asarray(frame_stars['flux'], dtype=np.float64) / frame_exptime

    tree_ref = cKDTree(ref_yx)
    tree_frm = cKDTree(frm_yx)
    d, i_ref = tree_ref.query(frm_yx, distance_upper_bound=tol)
    ok = np.isfinite(d)
    # mutual nearest neighbours only: a crowded field would otherwise give two
    # frame stars the same reference partner
    d_back, i_back = tree_frm.query(ref_yx[np.where(ok, i_ref, 0)])
    mutual = ok & (i_back == np.arange(len(frm_yx)))
    sel = np.where(mutual)[0]
    sel = sel[use[i_ref[sel]]]
    if sel.size < min_matches:
        return float('nan'), int(sel.size)
    ratio = frm_flux[sel] / ref_flux[i_ref[sel]]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size < min_matches:
        return float('nan'), int(ratio.size)
    return float(np.median(ratio)), int(ratio.size)


def measure_transparency(final: Sequence[Any], shifts: Sequence[Any],
                         transforms: Sequence[Any], ref_stars,
                         ref_frame: Optional[Any] = None,
                         tol: float = 4.0) -> Dict[str, Any]:
    """Set ``metrics['transparency']`` (relative, session median = 1.0) and
    ``metrics['transparency_n']`` on every frame; return a summary dict.

    Frames with no catalogue or too few matches get NaN and are never gated.
    """
    n = len(final)
    ref_exp = _exptime(ref_frame) if ref_frame is not None else 1.0
    raw = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    for j, f in enumerate(final):
        stars = f.metrics.get('_star_sources') if getattr(f, 'metrics', None) else None
        r, k = match_flux_ratio(ref_stars, stars, transforms[j], shifts[j],
                                ref_exptime=ref_exp, frame_exptime=_exptime(f), tol=tol)
        raw[j], counts[j] = r, k
    good = np.isfinite(raw)
    med = float(np.median(raw[good])) if good.any() else float('nan')
    rel = raw / med if good.any() and med > 0 else raw
    for j, f in enumerate(final):
        if getattr(f, 'metrics', None) is not None:
            f.metrics['transparency'] = float(rel[j]) if np.isfinite(rel[j]) else float('nan')
            f.metrics['transparency_n'] = int(counts[j])
    fin = rel[np.isfinite(rel)]
    return {
        'measured': int(good.sum()), 'frames': n,
        'median_matches': int(np.median(counts[good])) if good.any() else 0,
        'min': float(fin.min()) if fin.size else float('nan'),
        'p10': float(np.percentile(fin, 10)) if fin.size else float('nan'),
        'max': float(fin.max()) if fin.size else float('nan'),
        'values': rel,
    }


def transparency_keep_mask(values: np.ndarray, threshold: float) -> List[bool]:
    """True = keep. Unmeasured (NaN) frames are always kept."""
    if threshold <= 0:
        return [True] * len(values)
    return [bool((not np.isfinite(v)) or v >= threshold) for v in values]
