"""Tests for src/affine_fit.py -- RANSAC-robust 2D rigid transform fit,
replacing skimage.measure.ransac + skimage.transform.EuclideanTransform.

See the module docstring for why parity with skimage is statistical (its
usage here is unseeded) rather than bit-exact; these tests check
correctness against synthetic ground truth and internal consistency,
which is what's actually verifiable and matters.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.affine_fit import (
    fit_rigid_ransac, _ransac_rigid_numpy, _umeyama_2d, RigidTransform,
)


def _synthetic_correspondences(n_inliers=40, n_outliers=15, theta_deg=3.7,
                               tx=12.3, ty=-7.8, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    theta = np.radians(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src_in = rng.uniform(0, 500, (n_inliers, 2))
    dst_in = (R @ src_in.T).T + np.array([tx, ty]) + rng.normal(0, noise, (n_inliers, 2))
    src_out = rng.uniform(0, 500, (n_outliers, 2))
    dst_out = rng.uniform(0, 500, (n_outliers, 2))
    return np.vstack([src_in, src_out]), np.vstack([dst_in, dst_out]), n_inliers, theta, np.array([tx, ty])


class TestUmeyama2D:
    def test_exact_rigid_transform_recovered(self):
        src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        theta = np.radians(15.0)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        dst = (R @ src.T).T + np.array([5.0, 3.0])
        params = _umeyama_2d(src, dst)
        fitted_theta = np.degrees(np.arctan2(params[1, 0], params[0, 0]))
        assert abs(fitted_theta - 15.0) < 1e-8
        assert np.allclose(params[:2, 2], [5.0, 3.0], atol=1e-8)

    def test_identity_for_identical_points_set(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 1.0]])
        params = _umeyama_2d(pts, pts)
        assert np.allclose(params, np.eye(3), atol=1e-10)

    def test_degenerate_identical_points_returns_none(self):
        deg = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        assert _umeyama_2d(deg, deg) is None

    def test_reflection_is_corrected_to_proper_rotation(self):
        # A point configuration whose naive best-fit A has det<0 must still
        # produce a proper rotation (det=+1), not a mirror flip -- this is
        # exactly what Umeyama's d-vector correction exists for.
        src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        dst = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]])  # mirrored
        params = _umeyama_2d(src, dst)
        det = params[0, 0] * params[1, 1] - params[0, 1] * params[1, 0]
        assert det > 0  # proper rotation, not a reflection


class TestRansacRigidNumpy:
    def test_recovers_synthetic_ground_truth(self):
        src, dst, n_true, theta, t = _synthetic_correspondences()
        model, inliers = _ransac_rigid_numpy(src, dst, min_samples=3,
                                             residual_threshold=2.0, max_trials=1000,
                                             rng=np.random.default_rng(1))
        assert inliers.sum() >= n_true - 2
        fitted_theta = np.degrees(np.arctan2(model.params[1, 0], model.params[0, 0]))
        assert abs(fitted_theta - np.degrees(theta)) < 0.1
        assert np.allclose(model.params[:2, 2], t, atol=0.2)

    def test_too_few_points_returns_none(self):
        model, inliers = _ransac_rigid_numpy(
            np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 0.0], [1.0, 1.0]]),
            min_samples=3)
        assert model is None and inliers is None

    def test_exact_min_samples_fit(self):
        src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        theta = np.radians(15.0)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        dst = (R @ src.T).T + np.array([5.0, 3.0])
        model, inliers = _ransac_rigid_numpy(src, dst, min_samples=3,
                                             residual_threshold=1.0, max_trials=100)
        assert inliers.sum() == 3

    def test_matches_skimage_statistically(self):
        # This is a nice-to-have cross-check against the real skimage, not a
        # correctness gate (that's the synthetic-ground-truth tests above).
        # skip -- don't fail -- if skimage.transform isn't genuinely usable:
        # some other test module (test_main.py, for its own offline-mock
        # isolation) installs a process-global empty stub at
        # sys.modules['skimage.transform'] via setdefault(), which only
        # "wins" when nothing has *really* imported that module first. That
        # used to always be true by the time tests ran (src/registration.py
        # itself imported EuclideanTransform at module load time); now that
        # this codebase's own code no longer needs skimage for this, test
        # order can leave the stub in place instead of the real module.
        try:
            from skimage.transform import EuclideanTransform
            from skimage.measure import ransac as sk_ransac
        except ImportError:
            pytest.skip("skimage.transform unavailable or stubbed by another "
                       "test module's sys.modules mocking in this run")

        src, dst, n_true, theta, t = _synthetic_correspondences()
        sk_counts, my_counts = [], []
        for trial in range(10):
            _, in_sk = sk_ransac((src, dst), EuclideanTransform, min_samples=3,
                                 residual_threshold=2.0, max_trials=1000)
            sk_counts.append(int(in_sk.sum()))
            _, in_my = _ransac_rigid_numpy(src, dst, min_samples=3,
                                           residual_threshold=2.0, max_trials=1000)
            my_counts.append(int(in_my.sum()))
        # both should reliably converge on the true inlier set
        assert min(sk_counts) >= n_true - 2
        assert min(my_counts) >= n_true - 2


class TestFitRigidRansacDispatch:
    def test_returns_rigid_transform_with_params(self):
        src, dst, n_true, theta, t = _synthetic_correspondences()
        model, inliers = fit_rigid_ransac(src, dst, seed=3)
        assert isinstance(model, RigidTransform)
        assert model.params.shape == (3, 3)
        assert inliers.sum() >= n_true - 2

    def test_seeded_calls_are_reproducible(self):
        src, dst, *_ = _synthetic_correspondences()
        m1, in1 = fit_rigid_ransac(src, dst, seed=99)
        m2, in2 = fit_rigid_ransac(src, dst, seed=99)
        assert np.array_equal(in1, in2)
        assert np.allclose(m1.params, m2.params)

    def test_none_on_too_few_points(self):
        model, inliers = fit_rigid_ransac(
            np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert model is None and inliers is None
