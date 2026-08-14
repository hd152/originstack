"""Tests for wavelet-domain multi-frame combine (--stack-method wavelet).

wavelet_combine decomposes every aligned frame, sigma-clip-combines each
subband's coefficients across frames (reusing the native sigma_clip_combine
kernel on coefficient arrays instead of pixel arrays), then reconstructs.
These tests check: basic shape/finiteness, that a strong single-frame
outlier (cosmic-ray-like) is still rejected comparably to pixel-domain
sigma-clip, and that combining a clean multi-frame stack recovers the
ground truth to within the expected noise-averaging tolerance.
"""
from __future__ import annotations

import unittest

import numpy as np

from src.stacking import sigma_clip_combine, wavelet_combine


def _synthetic_truth(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    truth = 100.0 + 50.0 * np.exp(-((yy - h // 2) ** 2 + (xx - w // 2) ** 2) / (2 * 8.0 ** 2))
    # Faint smooth filament, low contrast relative to sky.
    truth += 4.0 * np.sin(xx / 3.0) * np.exp(-((yy - h * 0.6) ** 2) / (2 * 10.0 ** 2))
    return truth


class TestWaveletCombineBasics(unittest.TestCase):

    def test_shape_dtype_finite(self):
        rng = np.random.default_rng(0)
        n, h, w = 6, 48, 48
        truth = _synthetic_truth(h, w)
        stack = np.stack([truth + rng.normal(0, 3.0, (h, w)) for _ in range(n)], axis=0)
        mem = stack[..., np.newaxis].astype(np.float32)

        out = wavelet_combine(mem, levels=3, sigma=3.0, max_iters=3)
        self.assertEqual(out.shape, (h, w, 1))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_recovers_ground_truth_within_noise_tolerance(self):
        rng = np.random.default_rng(1)
        n, h, w = 10, 48, 48
        truth = _synthetic_truth(h, w)
        noise_sigma = 3.0
        stack = np.stack([truth + rng.normal(0, noise_sigma, (h, w)) for _ in range(n)], axis=0)
        mem = stack[..., np.newaxis].astype(np.float32)

        out = wavelet_combine(mem, levels=3, sigma=3.0, max_iters=3)
        rmse = float(np.sqrt(np.mean((out[..., 0] - truth) ** 2)))
        # Expected residual noise after averaging N clean frames ~ sigma/sqrt(N);
        # allow generous slack for wavelet-domain reconstruction overhead.
        expected = noise_sigma / np.sqrt(n)
        self.assertLess(rmse, expected * 2.5)


class TestWaveletCombineOutlierRejection(unittest.TestCase):

    def test_rejects_cosmic_ray_like_outlier_comparably_to_sigma_clip(self):
        rng = np.random.default_rng(2)
        n, h, w = 8, 48, 48
        truth = _synthetic_truth(h, w)
        stack = np.stack([truth + rng.normal(0, 3.0, (h, w)) for _ in range(n)], axis=0)
        stack[3, 20, 20] += 500.0  # single-frame spike
        mem = stack[..., np.newaxis].astype(np.float32)

        out_wavelet = wavelet_combine(mem, levels=3, sigma=2.5, max_iters=3)
        out_sigma_clip = sigma_clip_combine(mem, sigma=2.5, max_iters=3)

        true_val = truth[20, 20]
        # Both must reject the +500 spike -- well below a naive-mean blowup
        # (naive mean would land near true_val + 500/8 ~= true_val + 62.5).
        self.assertLess(abs(float(out_wavelet[20, 20, 0]) - true_val), 15.0)
        self.assertLess(abs(float(out_sigma_clip[20, 20, 0]) - true_val), 15.0)

    def test_faint_filament_error_not_worse_than_sigma_clip(self):
        rng = np.random.default_rng(3)
        n, h, w = 8, 48, 48
        truth = _synthetic_truth(h, w)
        stack = np.stack([truth + rng.normal(0, 3.0, (h, w)) for _ in range(n)], axis=0)
        stack[3, 20, 20] += 500.0
        mem = stack[..., np.newaxis].astype(np.float32)

        out_wavelet = wavelet_combine(mem, levels=3, sigma=2.0, max_iters=3)
        out_sigma_clip = sigma_clip_combine(mem, sigma=2.0, max_iters=3)

        mse_wavelet = float(np.mean((out_wavelet[..., 0] - truth) ** 2))
        mse_sigma_clip = float(np.mean((out_sigma_clip[..., 0] - truth) ** 2))
        # Not a strict "always better" claim (see plan discussion) -- just
        # confirm no meaningful regression vs the pixel-domain method.
        self.assertLess(mse_wavelet, mse_sigma_clip * 1.25)


if __name__ == '__main__':
    unittest.main()
