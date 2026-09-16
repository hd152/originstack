"""Tests for iterative back-projection (IBP) super-resolution refinement
(--super-res-iters).

iterative_back_projection refines a drizzle output by forward-simulating
each original frame from the current estimate (inverse of drizzle's own
affine mapping, reused via _drizzle_matrix), comparing to what was actually
observed, and back-projecting the residual. These tests build a synthetic
scenario with a KNOWN high-res ground truth, synthetically resample it into
several low-res frames (via the exact same _drizzle_matrix mapping IBP
itself uses -- so any test failure reflects a real bug in IBP's math, not a
mismatch between the test's own forward model and the implementation's),
drizzle-combine them, then confirm IBP genuinely reduces error against the
known ground truth -- not just "runs without crashing".
"""
from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

from src.stacking import _drizzle_matrix, iterative_back_projection


def _gaussian_psf(size: int = 15, sigma: float = 1.0) -> np.ndarray:
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def _build_synthetic_scenario(seed=0, n_final=8, out_hw=64, raw_hw=32,
                              drizzle_scale=2.0, psf_sigma=1.0, read_noise=0.5):
    """Known high-res truth -> N synthetic low-res raw frames (via the same
    affine mapping IBP itself uses) -> naive initial drizzle estimate."""
    rng = np.random.default_rng(seed)
    out_h = out_w = out_hw
    H = W = raw_hw
    inv_scale = 1.0 / drizzle_scale
    top, left = 0.0, 0.0

    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float64)
    truth = 50.0 + 200.0 * np.exp(-((yy - out_h / 2) ** 2 + (xx - out_w / 2) ** 2)
                                  / (2 * 3.0 ** 2))
    psf = _gaussian_psf(sigma=psf_sigma)

    shifts = []
    transforms = [None] * n_final
    mem_rgb = np.zeros((n_final, H, W, 1), dtype=np.float32)
    final_indices = list(range(n_final))
    weights = np.ones(n_final, dtype=np.float64)

    for j in range(n_final):
        dy, dx = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        shifts.append((dy, dx))
        M, off = _drizzle_matrix(None, (dy, dx), top, left, inv_scale)
        raw = ndimage.affine_transform(truth, M, offset=off, output_shape=(H, W),
                                       order=3, mode='constant', cval=0.0)
        raw = fftconvolve(raw, psf, mode='same')
        raw = raw + rng.normal(0, read_noise, raw.shape)
        mem_rgb[j, :, :, 0] = raw

    acc = np.zeros((out_h, out_w, 1))
    wsum = np.zeros((out_h, out_w))
    for j in range(n_final):
        M, off = _drizzle_matrix(None, shifts[j], top, left, inv_scale)
        warped = ndimage.affine_transform(mem_rgb[j, :, :, 0], M, offset=off,
                                          output_shape=(out_h, out_w), order=3,
                                          mode='constant', cval=0.0)
        cov = ndimage.affine_transform(np.ones((H, W)), M, offset=off,
                                       output_shape=(out_h, out_w), order=1,
                                       mode='constant', cval=0.0)
        acc[:, :, 0] += warped * cov
        wsum += cov
    stacked0 = (acc[:, :, 0] / np.maximum(wsum, 1e-9))[..., np.newaxis].astype(np.float32)

    return dict(truth=truth, psf=psf, mem_rgb=mem_rgb, final_indices=final_indices,
               shifts=shifts, transforms=transforms, weights=weights,
               stacked0=stacked0, top=top, left=left, drizzle_scale=drizzle_scale,
               H=H, W=W)


class TestIterativeBackProjection(unittest.TestCase):

    def test_reduces_error_vs_ground_truth(self):
        sc = _build_synthetic_scenario(seed=0)
        rmse0 = float(np.sqrt(np.mean((sc['stacked0'][:, :, 0] - sc['truth']) ** 2)))

        refined = iterative_back_projection(
            sc['stacked0'], sc['mem_rgb'], sc['final_indices'], sc['shifts'],
            sc['transforms'], sc['weights'], sc['top'], sc['left'], sc['drizzle_scale'],
            sc['psf'], sc['H'], sc['W'], 1, iters=5, relax=0.15)

        rmse1 = float(np.sqrt(np.mean((refined[:, :, 0] - sc['truth']) ** 2)))
        self.assertLess(rmse1, rmse0,
                        f"IBP should reduce RMSE vs ground truth (before={rmse0:.3f}, "
                        f"after={rmse1:.3f})")

    def test_reduces_error_across_multiple_seeds(self):
        # Guard against the previous (broken-sign) implementation's failure
        # mode passing by coincidence on a single seed.
        improved = 0
        for seed in range(4):
            sc = _build_synthetic_scenario(seed=seed)
            rmse0 = float(np.sqrt(np.mean((sc['stacked0'][:, :, 0] - sc['truth']) ** 2)))
            refined = iterative_back_projection(
                sc['stacked0'], sc['mem_rgb'], sc['final_indices'], sc['shifts'],
                sc['transforms'], sc['weights'], sc['top'], sc['left'], sc['drizzle_scale'],
                sc['psf'], sc['H'], sc['W'], 1, iters=5, relax=0.15)
            rmse1 = float(np.sqrt(np.mean((refined[:, :, 0] - sc['truth']) ** 2)))
            if rmse1 < rmse0:
                improved += 1
        self.assertGreaterEqual(improved, 3, "IBP should improve RMSE on most seeds")

    def test_output_shape_dtype_finite(self):
        sc = _build_synthetic_scenario(seed=1)
        refined = iterative_back_projection(
            sc['stacked0'], sc['mem_rgb'], sc['final_indices'], sc['shifts'],
            sc['transforms'], sc['weights'], sc['top'], sc['left'], sc['drizzle_scale'],
            sc['psf'], sc['H'], sc['W'], 1, iters=3, relax=0.15)
        self.assertEqual(refined.shape, sc['stacked0'].shape)
        self.assertEqual(refined.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(refined)))

    def test_zero_iterations_returns_input_unchanged(self):
        sc = _build_synthetic_scenario(seed=2)
        refined = iterative_back_projection(
            sc['stacked0'], sc['mem_rgb'], sc['final_indices'], sc['shifts'],
            sc['transforms'], sc['weights'], sc['top'], sc['left'], sc['drizzle_scale'],
            sc['psf'], sc['H'], sc['W'], 1, iters=0, relax=0.15)
        np.testing.assert_allclose(refined, sc['stacked0'], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
