"""Tests for fit_displacement_field_flow: the dense pyramidal-LK alternative
to fit_displacement_field's sparse matched-star fit (--elastic-registration-method
flow).

Unlike fit_displacement_field (fit from a handful of star correspondences),
this recovers a field directly from two luminance images -- these tests check
it recovers a known smooth synthetic field, returns None on a featureless
(flat) frame, and clamps oversized displacement the same way the star-based
fitter does, since both must satisfy the exact same downstream contract
(sample_displacement_field / apply_transform's local_field composition).
"""
from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage

from src.models import Config
from src.registration import fit_displacement_field_flow, sample_displacement_field


def _synthetic_starfield(H: int, W: int, seed: int = 0, n_stars: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = np.full((H, W), 20.0)
    for _ in range(n_stars):
        cy, cx = rng.uniform(10, H - 10), rng.uniform(10, W - 10)
        amp = rng.uniform(50, 200)
        sigma = rng.uniform(2, 4)
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return img


class TestFitDisplacementFieldFlow(unittest.TestCase):

    def test_recovers_smooth_sinusoidal_field(self):
        H, W = 128, 128
        ref = _synthetic_starfield(H, W, seed=0)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

        dy_true = 2.0 * np.sin(2 * np.pi * xx / W)
        dx_true = 1.5 * np.cos(2 * np.pi * yy / H)
        # frame_warped(o) = ref(o + D_true(o)) satisfies
        # frame_warped(o - D_true(o)) ~= ref(o) for smooth D, matching the
        # backward-warp convention fit_displacement_field_flow solves for.
        frame_warped = ndimage.map_coordinates(ref, [yy + dy_true, xx + dx_true],
                                               order=3, mode='reflect')

        field = fit_displacement_field_flow(ref.astype(np.float32),
                                            frame_warped.astype(np.float32), H, W)
        self.assertIsNotNone(field)
        self.assertEqual(field.shape, (Config.LOCAL_WARP_GRID_SIZE,
                                       Config.LOCAL_WARP_GRID_SIZE, 2))

        dy_rec, dx_rec = sample_displacement_field(field, H, W, yy, xx)
        # Approximate recovery (numpy LK on a coarse grid, not exact) --
        # mean error well within the ground-truth amplitude (+/-2px / +/-1.5px).
        self.assertLess(float(np.abs(dy_rec - dy_true).mean()), 1.0)
        self.assertLess(float(np.abs(dx_rec - dx_true).mean()), 1.0)

    def test_returns_none_on_featureless_frame(self):
        flat = np.full((64, 64), 100.0, dtype=np.float32)
        field = fit_displacement_field_flow(flat, flat, 64, 64)
        self.assertIsNone(field)

    def test_clamps_oversized_displacement(self):
        H, W = 128, 128
        ref = _synthetic_starfield(H, W, seed=1)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        # Huge uniform shift, far beyond LOCAL_WARP_MAX_DISPLACEMENT_PX.
        shifted = ndimage.map_coordinates(ref, [yy + 30.0, xx + 30.0],
                                          order=3, mode='reflect')

        field = fit_displacement_field_flow(ref.astype(np.float32),
                                            shifted.astype(np.float32), H, W)
        self.assertIsNotNone(field)
        mag = np.hypot(field[..., 0], field[..., 1])
        self.assertLessEqual(float(mag.max()),
                             Config.LOCAL_WARP_MAX_DISPLACEMENT_PX + 1e-3)

    def test_identical_frames_give_near_zero_field(self):
        H, W = 96, 96
        ref = _synthetic_starfield(H, W, seed=2)
        field = fit_displacement_field_flow(ref.astype(np.float32), ref.astype(np.float32), H, W)
        self.assertIsNotNone(field)
        mag = np.hypot(field[..., 0], field[..., 1])
        self.assertLess(float(mag.max()), 0.5)


if __name__ == '__main__':
    unittest.main()
