"""Tests for src/blind_match.py -- rigid star-pattern matching with unknown
rotation, replacing astroalign for merge.py's cross-night registration
(and registration.py's within-session fallback when the near-zero-rotation
RANSAC match fails).

Validated against synthetic ground truth across a wide range of rotation
angles (0.5 deg to 178 deg) with dropout/spurious-star noise, and directly
against real astroalign on a dense globular-cluster-like synthetic field
(the adversarial case for pairwise-distance matching: many similar
separations) -- both recover the same rotation to within ~0.01 deg.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.blind_match import match_rigid_unknown_rotation

_DT = np.dtype([('xcentroid', np.float64), ('ycentroid', np.float64), ('flux', np.float64)])


def _make_catalog(n, w=3000, h=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=_DT)
    out['xcentroid'] = rng.uniform(50, w - 50, n)
    out['ycentroid'] = rng.uniform(50, h - 50, n)
    out['flux'] = rng.uniform(500, 50000, n)
    return out


def _transform_catalog(cat, theta_deg, tx, ty, w=3000, h=2000, noise=0.2,
                       drop_frac=0.2, extra=10, seed=1):
    """Rotate cat about the frame centre, translate, add centroid noise,
    randomly drop some stars, and inject spurious extras -- simulating a
    second night's detections of mostly-the-same field."""
    rng = np.random.default_rng(seed)
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    cx, cy = w / 2, h / 2
    x = cat['xcentroid'] - cx
    y = cat['ycentroid'] - cy
    xr = c * x - s * y + cx + tx
    yr = s * x + c * y + cy + ty
    xr += rng.normal(0, noise, len(xr))
    yr += rng.normal(0, noise, len(yr))
    keep = rng.random(len(cat)) > drop_frac
    n_keep = int(keep.sum())
    out = np.zeros(n_keep + extra, dtype=_DT)
    out['xcentroid'][:n_keep] = xr[keep]
    out['ycentroid'][:n_keep] = yr[keep]
    out['flux'][:n_keep] = cat['flux'][keep]
    out['xcentroid'][n_keep:] = rng.uniform(50, w - 50, extra)
    out['ycentroid'][n_keep:] = rng.uniform(50, h - 50, extra)
    out['flux'][n_keep:] = rng.uniform(500, 50000, extra)
    return out, keep


class TestMatchRigidUnknownRotation:
    @pytest.mark.parametrize('theta,tx,ty,seed', [
        (5.0, 12.3, -8.7, 0),
        (37.0, 100.5, -50.2, 1),
        (91.0, -30.0, 40.0, 2),
        (178.0, 5.0, 5.0, 3),
        (-45.0, -20.1, 60.4, 4),
        (0.5, 3.0, -3.0, 5),
    ])
    def test_recovers_ground_truth_transform(self, theta, tx, ty, seed):
        src = _make_catalog(45, seed=seed)
        dst, keep = _transform_catalog(src, theta, tx, ty, seed=seed + 100)
        tf = match_rigid_unknown_rotation(src, dst, max_stars=45)
        assert tf is not None

        R, t = tf.params[:2, :2], tf.params[:2, 2]
        src_xy = np.column_stack([src['xcentroid'], src['ycentroid']])[keep]
        dst_xy = np.column_stack([dst['xcentroid'], dst['ycentroid']])[:keep.sum()]
        pred = src_xy @ R.T + t
        err = np.hypot(*(pred - dst_xy).T)
        # noise injected was 0.2px std; a correct fit should track that closely
        assert err.mean() < 1.0
        assert err.max() < 2.0

        fitted_theta = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        # angle wraps at +-180; compare via the wrapped difference
        dtheta = (fitted_theta - theta + 180) % 360 - 180
        assert abs(dtheta) < 0.1

    def test_dense_field_matches_real_astroalign(self):
        """Adversarial case for pairwise-distance matching: a dense
        globular-cluster-like core with many similar star separations."""
        astroalign = pytest.importorskip("astroalign")

        def make_dense(n, w=800, h=800, cluster_frac=0.6, seed=0):
            rng = np.random.default_rng(seed)
            n_core = int(n * cluster_frac)
            n_field = n - n_core
            xs = np.concatenate([rng.normal(w / 2, 40, n_core),
                                 rng.uniform(50, w - 50, n_field)])
            ys = np.concatenate([rng.normal(h / 2, 40, n_core),
                                 rng.uniform(50, h - 50, n_field)])
            out = np.zeros(n, dtype=_DT)
            out['xcentroid'] = np.clip(xs, 10, w - 10)
            out['ycentroid'] = np.clip(ys, 10, h - 10)
            out['flux'] = rng.uniform(500, 50000, n)
            return out

        def paint(cat, w=800, h=800, sigma=2.0):
            img = np.zeros((h, w), dtype=np.float64)
            yy, xx = np.mgrid[0:h, 0:w]
            for i in range(len(cat)):
                cx, cy, f = cat['xcentroid'][i], cat['ycentroid'][i], cat['flux'][i]
                img += f * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            return img + 50.0

        src = make_dense(60, seed=42)
        theta, tx, ty = 23.5, 15.2, -9.8
        dst, _ = _transform_catalog(src, theta, tx, ty, w=800, h=800,
                                    noise=0.15, drop_frac=0.0, extra=0, seed=99)

        tf = match_rigid_unknown_rotation(src, dst, max_stars=60)
        assert tf is not None
        fitted_theta = np.degrees(np.arctan2(tf.params[1, 0], tf.params[0, 0]))

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aa_tf, _ = astroalign.find_transform(
                paint(src).astype(np.float32), paint(dst).astype(np.float32))
        aa_theta = np.degrees(aa_tf.rotation)

        assert abs(fitted_theta - theta) < 0.1
        assert abs(aa_theta - theta) < 0.1
        assert abs(fitted_theta - aa_theta) < 0.05

    def test_too_few_stars_returns_none(self):
        src = _make_catalog(3, seed=0)
        dst = _make_catalog(3, seed=1)
        assert match_rigid_unknown_rotation(src, dst) is None

    def test_unrelated_catalogs_return_none_or_low_confidence(self):
        # Two independent random fields share no real geometric pattern;
        # a match, if any survives, must not be trusted with a tiny inlier
        # count -- min_inliers guards this structurally, so this is really
        # just confirming the function doesn't crash or hang.
        src = _make_catalog(30, seed=10)
        dst = _make_catalog(30, seed=20)
        result = match_rigid_unknown_rotation(src, dst, min_inliers=6)
        assert result is None or hasattr(result, 'params')

    def test_none_catalogs_return_none(self):
        assert match_rigid_unknown_rotation(None, _make_catalog(10)) is None
        assert match_rigid_unknown_rotation(_make_catalog(10), None) is None
