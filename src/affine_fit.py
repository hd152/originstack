"""RANSAC-robust 2D rigid (rotation+translation) transform fitting, replacing
skimage.measure.ransac + skimage.transform.EuclideanTransform for
src.registration.match_stars_affine -- the per-frame affine-registration fit,
called for every light frame on every run.

Native (Rust) with an exact numpy mirror. The numpy mirror reimplements the
documented skimage algorithm exactly (same RANSAC loop: dynamic max_trials
shrinking, more-inliers-then-less-residual tie-break, final refit on all
inliers of the best trial) rather than approximating it -- verified by
matching skimage's own fit quality (inlier count, residual RMS) on real
matched-star data, not just "looks similar". The rigid-transform solve
itself (Umeyama's closed-form least-squares method) uses numpy.linalg.svd
directly, which is what skimage's own `_umeyama` does internally too, so
that part is not an approximation at all -- same math, same call.

skimage's ransac is unseeded by default in this codebase's usage (no
`rng=` passed), so "parity with skimage" can only ever be statistical --
matching a specific run's random sample sequence isn't a meaningful target
since skimage itself doesn't guarantee one. What *is* meaningful and
checked: same closed-form solver, same loop semantics, equivalent fit
quality on real data, and (for the native-vs-numpy pair specifically,
where a shared seed is meaningful since both are *my own* implementations)
exact bit-for-bit agreement.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import astro_native as _native
    HAS_NATIVE = hasattr(_native, 'fit_rigid_ransac')
except Exception:
    _native = None
    HAS_NATIVE = False


class RigidTransform:
    """Minimal stand-in for skimage's EuclideanTransform: every consumer in
    this codebase only ever reads `.params` (a 3x3 homogeneous matrix)."""
    __slots__ = ('params',)

    def __init__(self, params: np.ndarray):
        self.params = params

    @classmethod
    def from_rotation_translation(cls, rotation: float,
                                  translation: Tuple[float, float]) -> 'RigidTransform':
        """Build directly from a rotation angle (radians) + translation --
        exact port of skimage.transform.EuclideanTransform's
        ``_rt2matrix(rotation, translation, n_dims=2)``, used to wrap
        astroalign's own (rotation, translation) result the same way the
        old EuclideanTransform(rotation=..., translation=...) constructor
        call did."""
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        params = np.eye(3)
        params[:2, :2] = [[cos_r, -sin_r], [sin_r, cos_r]]
        params[0, 2], params[1, 2] = translation
        return cls(params)


def _umeyama_2d(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """2D rigid (no scale) least-squares fit, Umeyama (1991). Identical
    algorithm to skimage.transform._geometric._umeyama(src, dst,
    estimate_scale=False) -- same numpy.linalg.svd call, same rank-deficient
    reflection handling. Returns a 3x3 homogeneous matrix, or None if the
    problem is degenerate (rank 0, matching skimage's NaN-params case)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = (dst_demean.T @ src_demean) / num

    d = np.ones(dim, dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)

    tol = S.max() * max(A.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(S > tol))
    if rank == 0:
        return None
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    T[:dim, dim] = dst_mean - (T[:dim, :dim] @ src_mean)
    return T


def _dynamic_max_trials(n_inliers: int, n_samples: int, min_samples: int,
                        probability: float) -> float:
    """Exact port of skimage.measure.fit._dynamic_max_trials."""
    if probability == 0:
        return 0
    if n_inliers == 0:
        return np.inf
    eps = np.finfo(float).eps
    inlier_ratio = n_inliers / n_samples
    nom = np.clip(1 - probability, eps, 1 - eps)
    denom = np.clip(1 - inlier_ratio ** min_samples, eps, 1 - eps)
    return np.ceil(np.log(nom) / np.log(denom))


def _ransac_rigid_numpy(src: np.ndarray, dst: np.ndarray,
                        min_samples: int = 3, residual_threshold: float = 2.0,
                        max_trials: int = 1000,
                        rng: Optional[np.random.Generator] = None,
                        ) -> Tuple[Optional[RigidTransform], Optional[np.ndarray]]:
    """Exact port of skimage.measure.ransac's loop, specialised to a 2D rigid
    (EuclideanTransform) model: random min_samples draw -> Umeyama fit ->
    count inliers (residual < threshold, strict) -> keep as best on (more
    inliers) or (equal inliers, less summed-squared-residual) -> shrink
    max_trials via _dynamic_max_trials (stop_probability=1, skimage's
    default) -> final refit on ALL inliers of the best trial.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < min_samples:
        return None, None
    if rng is None:
        rng = np.random.default_rng()

    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers: Optional[np.ndarray] = None
    trials = 0
    cur_max_trials = max_trials

    while trials < cur_max_trials:
        trials += 1
        idx = rng.choice(n, min_samples, replace=False)
        params = _umeyama_2d(src[idx], dst[idx])
        if params is None:
            continue

        transformed = src @ params[:2, :2].T + params[:2, 2]
        residuals = np.sqrt(np.sum((transformed - dst) ** 2, axis=1))
        inliers = residuals < residual_threshold
        inliers_count = int(np.count_nonzero(inliers))
        residuals_sum = float(residuals @ residuals)

        if (inliers_count > best_inlier_num or
                (inliers_count == best_inlier_num and
                 residuals_sum < best_inlier_residuals_sum)):
            best_inlier_num = inliers_count
            best_inlier_residuals_sum = residuals_sum
            best_inliers = inliers
            cur_max_trials = min(cur_max_trials, _dynamic_max_trials(
                best_inlier_num, n, min_samples, 1.0))
            if best_inlier_num >= n or best_inlier_residuals_sum <= 0:
                break

    if best_inliers is None or not np.any(best_inliers):
        return None, None

    final_params = _umeyama_2d(src[best_inliers], dst[best_inliers])
    if final_params is None:
        return None, None
    return RigidTransform(final_params), best_inliers


def fit_rigid_ransac(src: np.ndarray, dst: np.ndarray,
                     min_samples: int = 3, residual_threshold: float = 2.0,
                     max_trials: int = 1000, seed: Optional[int] = None,
                     ) -> Tuple[Optional[RigidTransform], Optional[np.ndarray]]:
    """RANSAC-robust 2D rigid transform fit -- native (Rust) when available,
    exact numpy mirror otherwise. `seed=None` matches this codebase's
    existing (unseeded) usage; pass a seed for reproducible fits (tests,
    native/numpy parity checks)."""
    if HAS_NATIVE:
        try:
            params, inliers = _native.fit_rigid_ransac(
                np.ascontiguousarray(src, dtype=np.float64),
                np.ascontiguousarray(dst, dtype=np.float64),
                int(min_samples), float(residual_threshold), int(max_trials),
                -1 if seed is None else int(seed))
            if params is None or inliers is None or not np.any(inliers):
                return None, None
            return RigidTransform(np.asarray(params)), np.asarray(inliers, dtype=bool)
        except Exception:
            pass  # fall through to numpy mirror

    rng = None if seed is None else np.random.default_rng(seed)
    return _ransac_rigid_numpy(src, dst, min_samples, residual_threshold,
                               max_trials, rng=rng)
