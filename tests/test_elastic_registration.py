"""Tests for elastic (non-rigid, per-patch) local registration.

fit_displacement_field reuses DBE's Gaussian-weighted local-linear-regression
+ Tukey-biweight IRLS kernel (repurposed for a 2-channel displacement field
instead of a scalar brightness surface) -- these tests check the fit recovers
a known field from sparse correspondences, degrades safely below the star
floor, and clamps oversized displacement. apply_transform's composed-warp
round trip checks the affine+local-field sign/rotation composition is
correct empirically, not just derived on paper.
"""
from __future__ import annotations

import unittest

import numpy as np

from src.models import Config
from src.registration import (
    apply_transform,
    calc_common_crop,
    fit_displacement_field,
    sample_displacement_field,
)


class TestFitDisplacementField(unittest.TestCase):
    """fit_displacement_field: recovery, star-count floor, clamping."""

    def test_fit_recovers_linear_gradient(self):
        H, W = 512, 512
        rng = np.random.default_rng(0)
        n = 60
        ref_xy = np.column_stack([
            rng.uniform(20, W - 20, n), rng.uniform(20, H - 20, n),
        ])  # (x, y)
        A, B = 3.0, -2.0  # px amplitude of the ground-truth gradient
        gt_dy = A * (ref_xy[:, 0] / W - 0.5)
        gt_dx = B * (ref_xy[:, 1] / H - 0.5)
        frame_xy = np.column_stack([ref_xy[:, 0] - gt_dx, ref_xy[:, 1] - gt_dy])

        field = fit_displacement_field(ref_xy, frame_xy, H, W)
        self.assertIsNotNone(field)
        self.assertEqual(field.shape, (Config.LOCAL_WARP_GRID_SIZE,
                                       Config.LOCAL_WARP_GRID_SIZE, 2))

        dy, dx = sample_displacement_field(field, H, W, ref_xy[:, 1], ref_xy[:, 0])
        np.testing.assert_allclose(dy, gt_dy, atol=0.5)
        np.testing.assert_allclose(dx, gt_dx, atol=0.5)

    def test_returns_none_below_star_floor(self):
        H, W = 256, 256
        n = Config.LOCAL_WARP_MIN_STARS - 1
        rng = np.random.default_rng(2)
        ref_xy = np.column_stack([
            rng.uniform(20, W - 20, n), rng.uniform(20, H - 20, n),
        ])
        frame_xy = ref_xy + 1.0
        field = fit_displacement_field(ref_xy, frame_xy, H, W)
        self.assertIsNone(field)

    def test_clamps_oversized_displacement(self):
        H, W = 256, 256
        n = 30
        rng = np.random.default_rng(1)
        ref_xy = np.column_stack([
            rng.uniform(20, W - 20, n), rng.uniform(20, H - 20, n),
        ])
        frame_xy = ref_xy.copy()
        frame_xy[:, 1] -= 50.0  # huge uniform y-displacement everywhere

        field = fit_displacement_field(ref_xy, frame_xy, H, W)
        self.assertIsNotNone(field)
        mag = np.hypot(field[..., 0], field[..., 1])
        self.assertLessEqual(float(mag.max()),
                             Config.LOCAL_WARP_MAX_DISPLACEMENT_PX + 1e-6)


class TestApplyTransformLocalField(unittest.TestCase):
    """Composed affine+local-field warp: verifies the sign/rotation
    composition (src(o) = R @ (o - D(o)) + t) empirically, not just on paper.
    """

    def test_composed_warp_samples_expected_source_point(self):
        from skimage.transform import EuclideanTransform

        H, W = 128, 128
        rng = np.random.default_rng(3)
        centers = rng.uniform(20, H - 20, size=(6, 2))
        amps = rng.uniform(0.5, 1.0, size=6)

        def f(y, x):
            val = np.zeros_like(np.asarray(y, dtype=np.float64))
            for (cy, cx), a in zip(centers, amps):
                val += a * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 6.0 ** 2))
            return val

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        raw = f(yy, xx).astype(np.float32)[:, :, np.newaxis]

        angle = np.deg2rad(4.0)
        tx, ty = 5.0, -3.0
        transform = EuclideanTransform(rotation=angle, translation=(tx, ty))

        dy0, dx0 = 2.5, -1.5  # constant local field applied everywhere
        field = np.zeros((Config.LOCAL_WARP_GRID_SIZE,
                          Config.LOCAL_WARP_GRID_SIZE, 2), dtype=np.float32)
        field[..., 0] = dy0
        field[..., 1] = dx0

        result = apply_transform(raw, transform=transform, local_field=field)

        R = transform.params[:2, :2]
        t_xy = transform.params[:2, 2]
        t_rowcol = np.array([t_xy[1], t_xy[0]])
        D = np.array([dy0, dx0])

        for oy, ox in [(64, 64), (40, 90), (90, 40)]:
            o = np.array([float(oy), float(ox)])
            # Matches apply_transform's pre-existing affine convention
            # (offset = -R @ t_rowcol, i.e. aligned[o] = raw[R @ (o -
            # t_rowcol)]) with the local field subtracted the same way the
            # translation already is: aligned[o] = raw[R @ (o - t_rowcol - D)].
            expected_src = R @ (o - t_rowcol - D)
            expected_val = float(f(np.array([expected_src[0]]),
                                   np.array([expected_src[1]]))[0])
            got_val = float(result[oy, ox, 0])
            self.assertAlmostEqual(
                got_val, expected_val, delta=0.05,
                msg=f"composed warp sign/rotation mismatch at o=({oy},{ox})")


class TestCalcCommonCropMargin(unittest.TestCase):
    """calc_common_crop's extra_margin_px shrinks the rectangle exactly."""

    def test_extra_margin_shrinks_by_exact_amount(self):
        shifts = [(0.0, 0.0), (2.0, -3.0), (-1.0, 4.0)]
        shape = (200, 300)
        no_margin = calc_common_crop(shifts, shape, extra_margin_px=0.0)
        with_margin = calc_common_crop(shifts, shape, extra_margin_px=5.0)
        top0, bottom0, left0, right0 = no_margin
        top1, bottom1, left1, right1 = with_margin
        self.assertEqual(top1, top0 + 5)
        self.assertEqual(bottom1, bottom0 - 5)
        self.assertEqual(left1, left0 + 5)
        self.assertEqual(right1, right0 - 5)


if __name__ == '__main__':
    unittest.main()
