"""Blind (unknown-rotation) rigid star-pattern matching -- replaces astroalign
for the cases where two star catalogs need registering with no prior
knowledge of the rotation between them (cross-night --merge on an alt-az
mount, or the in-session fallback when the near-zero-rotation RANSAC match
in match_stars_affine fails to find enough inliers).

Unlike match_stars_affine (registration.py), which assumes rotation ~ 0
after an initial translation estimate and finds correspondences by simple
nearest-neighbour matching, this has no such assumption -- it has to solve
correspondence and rotation simultaneously.

Algorithm (a from-scratch design, not astroalign's triangle-invariant
geometric hashing, though similar in spirit -- see module docstring in
src/affine_fit.py for the sibling design philosophy: same-instrument
observations mean the scale between the two catalogs is always exactly 1,
which removes a whole degree of freedom astroalign has to handle):

1. Take up to `max_stars` brightest points from each catalog.
2. Compute all pairwise point-to-point distances within each catalog
   (C(n,2) values). Euclidean distance between two points is invariant
   under rotation and translation, so a distance shared between a src pair
   and a dst pair (within tolerance) is a *candidate correspondence* for
   those two points, without needing to know the rotation angle at all.
3. Each matched pair-of-pairs gives a closed-form 2-point rigid transform
   (the vector between the two points fixes the rotation angle; either
   point then fixes the translation) -- two candidates per match, one per
   pairing order (which src point maps to which dst point).
4. Each candidate transform is applied to every src point; the count of
   src points landing within `pixel_tol` of a real dst point (a KDTree
   nearest-neighbour query) is its consensus score. This is exactly a
   RANSAC consensus count, just with an *informed* hypothesis proposal
   (distance-matched pairs) instead of uniform random sampling -- uniform
   random 2-point sampling would need to random-guess the correct pairing
   out of up to n*m candidates with no shortcut, which is why
   match_stars_affine's random-sample RANSAC doesn't work here but does
   work once an initial correspondence guess (near-zero rotation) narrows
   the search.
5. The highest-consensus hypothesis's inlier correspondences are refit with
   src/affine_fit.py's `_umeyama_2d` (the same closed-form least-squares
   solver RANSAC itself uses) for the final transform, rather than trusting
   the minimal 2-point seed.

Dense star fields (a globular cluster core, the validation target for this
module) can have many spuriously-matching pairwise distances -- the reason
this is still robust is that a *wrong* candidate transform only explains
the 2 points that generated it, while the *correct* transform is the only
one with consensus across many independently-matched pairs simultaneously.
Downstream callers (src/merge.py) additionally verify the aligned result
correlates with the reference before accepting it, so a failure mode here
is caught, not silently corrupted.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from src.affine_fit import RigidTransform, _umeyama_2d

_MAX_HYPOTHESES = 20000


def _extract_xy(stars, max_stars: int) -> np.ndarray:
    """Brightest-first (x, y) array from a _SOURCES_DTYPE-compatible catalog."""
    if stars is None or len(stars) == 0:
        return np.empty((0, 2))
    try:
        order = np.argsort(stars['flux'])[::-1]
    except (KeyError, TypeError, ValueError, IndexError):
        order = np.arange(len(stars))
    order = order[:max_stars]
    return np.array([(float(stars[i]['xcentroid']), float(stars[i]['ycentroid']))
                     for i in order])


def _rigid_from_two_points(p_i: np.ndarray, p_j: np.ndarray,
                           q_a: np.ndarray, q_b: np.ndarray) -> Optional[RigidTransform]:
    """Closed-form rigid transform mapping p_i->q_a, p_j->q_b exactly
    (proper rotation only, no reflection -- physically correct for the
    same-camera-across-nights case this module targets)."""
    v_src = p_j - p_i
    len_src = float(np.hypot(v_src[0], v_src[1]))
    if len_src < 1e-6:
        return None
    v_dst = q_b - q_a
    theta = np.arctan2(v_dst[1], v_dst[0]) - np.arctan2(v_src[1], v_src[0])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    t = q_a - R @ p_i
    params = np.eye(3)
    params[:2, :2] = R
    params[:2, 2] = t
    return RigidTransform(params)


def match_rigid_unknown_rotation(src_stars, dst_stars,
                                 max_stars: int = 40,
                                 pixel_tol: float = 3.0,
                                 dist_rel_tol: float = 0.01,
                                 min_inliers: int = 6) -> Optional[RigidTransform]:
    """Find the rigid (rotation+translation) transform mapping src_stars onto
    dst_stars with no assumption about the rotation angle. Returns None if
    fewer than `min_inliers` points achieve consensus.

    src_stars/dst_stars: _SOURCES_DTYPE-compatible structured arrays (or
    anything indexable with 'flux'/'xcentroid'/'ycentroid' fields).
    """
    src = _extract_xy(src_stars, max_stars)
    dst = _extract_xy(dst_stars, max_stars)
    n, m = len(src), len(dst)
    if n < 4 or m < 4:
        return None

    src_i, src_j = np.triu_indices(n, k=1)
    src_d = np.hypot(src[src_i, 0] - src[src_j, 0], src[src_i, 1] - src[src_j, 1])
    dst_i, dst_j = np.triu_indices(m, k=1)
    dst_d = np.hypot(dst[dst_i, 0] - dst[dst_j, 0], dst[dst_i, 1] - dst[dst_j, 1])

    # Drop pairs too short to give a well-conditioned rotation estimate --
    # a tiny separation amplifies centroid noise into a large angle error.
    min_sep = max(pixel_tol * 4.0, 10.0)
    keep = src_d >= min_sep
    src_i, src_j, src_d = src_i[keep], src_j[keep], src_d[keep]
    if len(src_d) == 0 or len(dst_d) == 0:
        return None

    order = np.argsort(dst_d)
    dst_d_sorted = dst_d[order]
    dst_i_sorted = dst_i[order]
    dst_j_sorted = dst_j[order]

    dst_tree = cKDTree(dst)

    best_inliers = 0
    best_transform: Optional[RigidTransform] = None
    tested = 0
    target_inliers = int(np.ceil(0.9 * min(n, m)))

    for k in range(len(src_d)):
        if tested >= _MAX_HYPOTHESES:
            break
        d = src_d[k]
        tol = max(pixel_tol * 2.0, d * dist_rel_tol)
        lo = np.searchsorted(dst_d_sorted, d - tol, side='left')
        hi = np.searchsorted(dst_d_sorted, d + tol, side='right')
        if lo >= hi:
            continue
        i, j = src_i[k], src_j[k]
        p_i, p_j = src[i], src[j]

        for cand in range(lo, hi):
            ci, cj = dst_i_sorted[cand], dst_j_sorted[cand]
            for a, b in ((ci, cj), (cj, ci)):
                tested += 1
                transform = _rigid_from_two_points(p_i, p_j, dst[a], dst[b])
                if transform is None:
                    continue
                R = transform.params[:2, :2]
                t = transform.params[:2, 2]
                pred = src @ R.T + t
                dists, _ = dst_tree.query(pred, k=1)
                n_in = int(np.count_nonzero(dists < pixel_tol))
                if n_in > best_inliers:
                    best_inliers = n_in
                    best_transform = transform
                    if best_inliers >= target_inliers:
                        break
                if tested >= _MAX_HYPOTHESES:
                    break
            if tested >= _MAX_HYPOTHESES or best_inliers >= target_inliers:
                break
        if best_inliers >= target_inliers:
            break

    if best_transform is None or best_inliers < min_inliers:
        return None

    # Final least-squares refit over all inliers of the winning hypothesis,
    # not just its minimal 2-point seed.
    R = best_transform.params[:2, :2]
    t = best_transform.params[:2, 2]
    pred = src @ R.T + t
    dists, idx = dst_tree.query(pred, k=1)
    inlier_mask = dists < pixel_tol
    params = _umeyama_2d(src[inlier_mask], dst[idx[inlier_mask]])
    if params is None:
        return best_transform
    return RigidTransform(params)
