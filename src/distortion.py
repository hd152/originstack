"""Session-wide radial distortion model (``--distortion-model``).

Every frame of a session passes through the same optics, so a single radial
distortion, ``u = c + (p - c) (1 + a1 r^2 + a2 r^4)`` with ``r`` the distance
from the optical centre in units of half the frame diagonal, relates each
frame's pixel coordinates ``p`` to undistorted sky-plane coordinates ``u``. On
an alt-az mount the frames are additionally rotated and shifted against each
other, and a rigid registration of *distorted* coordinates cannot be exact: how
wrong it is depends on where a star sits and on how far the frame is rotated.
That is a large part of the 1.5-2 px registration residual real sessions carry.

Elastic registration fits an unconstrained displacement field per frame from
that frame's own star matches. This instead fits the two coefficients once for
the whole session, from every frame's matches, by minimising the residual of
per-frame rigid fits in undistorted space -- so a frame with few stars is still
corrected, and the model is one interpretable number pair.

The fitted model is applied through the *same* mechanism elastic registration
uses (a coarse per-frame displacement field composed into the single resample
pass), built analytically from the model instead of fitted from noisy per-frame
star pairs.

All coordinates here are (row, col).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

from src.transparency import to_aligned_yx

_log = logging.getLogger("originstack")

_MIN_PAIRS_PER_FRAME = 12
_MAX_FIT_FRAMES = 30
_MIN_IMPROVEMENT = 0.10      # relative rms reduction needed before the model is applied
_MIN_ABS_GAIN_PX = 0.10


class RadialModel:
    """u = c + (p - c) * (1 + a1 r^2 + a2 r^4),  r = |p - c| / rho."""

    def __init__(self, a1: float, a2: float, center_yx: Tuple[float, float], rho: float):
        self.a1, self.a2 = float(a1), float(a2)
        self.c = np.asarray(center_yx, dtype=np.float64)
        self.rho = float(rho)

    def undistort(self, p: np.ndarray) -> np.ndarray:
        d = np.asarray(p, dtype=np.float64) - self.c
        r2 = (d ** 2).sum(axis=-1, keepdims=True) / self.rho ** 2
        return self.c + d * (1.0 + self.a1 * r2 + self.a2 * r2 ** 2)

    def distort(self, u: np.ndarray, iters: int = 12) -> np.ndarray:
        """Inverse of ``undistort`` by fixed-point iteration."""
        d_u = np.asarray(u, dtype=np.float64) - self.c
        d = d_u.copy()
        for _ in range(iters):
            r2 = (d ** 2).sum(axis=-1, keepdims=True) / self.rho ** 2
            d = d_u / (1.0 + self.a1 * r2 + self.a2 * r2 ** 2)
        return self.c + d


def _rigid(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation ``R`` (2x2, acting on (row, col)) and translation ``t`` minimising
    |dst - (R src + t)|^2 (Umeyama, no scale)."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, md - R @ ms


def collect_pairs(final: Sequence[Any], shifts: Sequence[Any], transforms: Sequence[Any],
                  ref_stars, tol: float = 3.0) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Per frame: (ref_yx, frame_yx) matched stars in NATIVE pixel coordinates.

    Frame stars are mapped into the reference by the frame's registration (with
    the convention verified in ``src.transparency``) only to find their partners
    (mutual nearest neighbours within ``tol``); the pairs returned are the
    untransformed native positions the distortion model is about."""
    if not _HAS_SCIPY or ref_stars is None or len(ref_stars) < _MIN_PAIRS_PER_FRAME:
        return [None] * len(final)
    ref_yx = np.column_stack([np.asarray(ref_stars['ycentroid'], float),
                              np.asarray(ref_stars['xcentroid'], float)])
    tree_ref = cKDTree(ref_yx)
    out: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    for j, f in enumerate(final):
        stars = (getattr(f, 'metrics', None) or {}).get('_star_sources')
        if stars is None or len(stars) < _MIN_PAIRS_PER_FRAME:
            out.append(None)
            continue
        native = np.column_stack([np.asarray(stars['ycentroid'], float),
                                  np.asarray(stars['xcentroid'], float)])
        aligned = to_aligned_yx(native, transforms[j], shifts[j])
        d, i_ref = tree_ref.query(aligned, distance_upper_bound=tol)
        ok = np.isfinite(d)
        if not ok.any():
            out.append(None)
            continue
        back_d, back_i = cKDTree(aligned).query(ref_yx[np.where(ok, i_ref, 0)])
        mutual = ok & (back_i == np.arange(len(aligned)))
        sel = np.where(mutual)[0]
        if len(sel) < _MIN_PAIRS_PER_FRAME:
            out.append(None)
            continue
        out.append((ref_yx[i_ref[sel]], native[sel]))
    return out


def _objective(params: np.ndarray, pairs, center, rho) -> float:
    m = RadialModel(params[0], params[1], center, rho)
    tot, n = 0.0, 0
    for ref, frm in pairs:
        u_ref, u_frm = m.undistort(ref), m.undistort(frm)
        R, t = _rigid(u_frm, u_ref)
        r = u_ref - (u_frm @ R.T + t)
        tot += float((r ** 2).sum())
        n += len(r)
    return float(np.sqrt(tot / max(n, 1)))


def fit_radial_distortion(pairs: Sequence[Optional[Tuple[np.ndarray, np.ndarray]]],
                          shape_hw: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    """Fit (a1, a2) by minimising the rigid-fit residual over the frames' star
    pairs. Returns the model, the rms before/after, and per-frame rigid transforms
    in undistorted space; None when there are too few usable frames."""
    if not _HAS_SCIPY:
        return None
    H, W = shape_hw
    center = ((H - 1) / 2.0, (W - 1) / 2.0)
    rho = 0.5 * float(np.hypot(H, W))
    idx = [j for j, p in enumerate(pairs) if p is not None]
    if len(idx) < 5:
        return None
    # frames spread over the whole session (rotation/shift diversity is what
    # makes the distortion observable), capped for speed
    pick = [idx[k] for k in np.unique(np.linspace(0, len(idx) - 1, min(_MAX_FIT_FRAMES, len(idx))).astype(int))]
    sub = [pairs[j] for j in pick]
    rms0 = _objective(np.zeros(2), sub, center, rho)
    res = minimize(_objective, x0=np.array([0.0, 0.0]), args=(sub, center, rho),
                   method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-4, 'maxiter': 400,
                            'initial_simplex': np.array([[0, 0], [0.01, 0], [0, 0.01]])})
    a1, a2 = (float(np.clip(res.x[0], -0.3, 0.3)), float(np.clip(res.x[1], -0.3, 0.3)))
    model = RadialModel(a1, a2, center, rho)
    rms1 = _objective(np.array([a1, a2]), sub, center, rho)
    # held-out check: frames the fit did not see
    rest = [pairs[j] for j in idx if j not in set(pick)]
    held0 = _objective(np.zeros(2), rest, center, rho) if rest else float('nan')
    held1 = _objective(np.array([a1, a2]), rest, center, rho) if rest else float('nan')
    rigid: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for j in idx:
        ref, frm = pairs[j]
        rigid[j] = _rigid(model.undistort(frm), model.undistort(ref))
    return {'model': model, 'rms_rigid_px': rms0, 'rms_model_px': rms1,
            'heldout_rigid_px': held0, 'heldout_model_px': held1,
            'frames_used': len(pick), 'frames_matched': len(idx),
            'pairs': int(sum(len(p[0]) for p in sub)), 'rigid': rigid}


def is_significant(fit: Dict[str, Any]) -> bool:
    """Apply the model only if it clearly helps, on the held-out frames when there
    are any (otherwise on the fitting set)."""
    r0 = fit['heldout_rigid_px'] if np.isfinite(fit['heldout_rigid_px']) else fit['rms_rigid_px']
    r1 = fit['heldout_model_px'] if np.isfinite(fit['heldout_model_px']) else fit['rms_model_px']
    return bool(r0 - r1 >= _MIN_ABS_GAIN_PX and (r0 - r1) / max(r0, 1e-9) >= _MIN_IMPROVEMENT)


def build_displacement_fields(fit: Dict[str, Any], shifts: Sequence[Any],
                              transforms: Sequence[Any], shape_hw: Tuple[int, int],
                              grid: int = 16) -> List[Optional[np.ndarray]]:
    """Per-frame (grid, grid, 2) displacement fields realising the model, for
    ``apply_transform(local_field=...)``.

    For each frame, take the reference grid nodes ``g``. The sky point seen at
    ``g`` is at undistorted ``u = U(g)``, at ``R_j^-1 (u - t_j)`` in frame j's
    undistorted coordinates and so at ``D(...)`` in its native pixels; running
    that through the registration the warp already applies gives where the
    *rigid* warp puts it. The field at ``g`` is (reference position) - (that
    position) -- the same sign and (row, col) order elastic registration's
    ``fit_displacement_field`` returns and ``apply_transform`` consumes.

    The model gives the displacement analytically at every node, so it is stored
    directly rather than passed through ``fit_displacement_field``: that
    fitter smooths (Gaussian-weighted local regression, built for noisy star
    matches) and shrank this smooth field to 65-75% of its true amplitude in
    testing, leaving a third of the error in place. Bilinear interpolation
    between 16x16 nodes is exact to well under 0.1 px for a distortion this
    smooth."""
    H, W = shape_hw
    model: RadialModel = fit['model']
    gy, gx = np.meshgrid(np.linspace(0, H - 1, grid), np.linspace(0, W - 1, grid), indexing='ij')
    g = np.column_stack([gy.ravel(), gx.ravel()])
    u = model.undistort(g)
    fields: List[Optional[np.ndarray]] = []
    for j in range(len(transforms)):
        if j not in fit['rigid']:
            fields.append(None)
            continue
        R, t = fit['rigid'][j]
        u_frame = (u - t) @ np.linalg.inv(R).T            # inverse of  u_ref = R u_frame + t
        p_native = model.distort(u_frame)
        aligned = to_aligned_yx(p_native, transforms[j], shifts[j] if j < len(shifts) else None)
        fields.append((g - aligned).reshape(grid, grid, 2).astype(np.float32))
    return fields


def format_summary(fit: Dict[str, Any], applied: bool) -> str:
    m: RadialModel = fit['model']
    lines = [f"  Distortion model: a1 = {m.a1:+.5f}, a2 = {m.a2:+.5f} (centre "
             f"({m.c[1]:.0f}, {m.c[0]:.0f}), r in units of half the diagonal)",
             f"    star-match residual after rigid registration {fit['rms_rigid_px']:.2f} px -> "
             f"{fit['rms_model_px']:.2f} px with the model "
             f"({fit['frames_used']} frames, {fit['pairs']} pairs)"]
    if np.isfinite(fit['heldout_rigid_px']):
        lines.append(f"    held-out frames: {fit['heldout_rigid_px']:.2f} px -> "
                     f"{fit['heldout_model_px']:.2f} px")
    lines.append("    applied to every frame's resample" if applied else
                 "    not significant enough to apply (registration left as is)")
    return "\n".join(lines)
