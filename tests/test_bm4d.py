"""Tests for BM4D-style cross-frame collaborative-filter denoising
(--bm4d-prefilter, src/bm4d.py).

Unlike bm3d_denoise's spatial self-similarity search within one image, this
exploits exact cross-frame correspondence already established by alignment
-- these tests check it substantially reduces per-frame noise on a synthetic
multi-frame stack with known ground truth, that it beats naive N-frame
averaging (the point of also doing DCT-domain collaborative filtering, not
just temporal averaging), and edge-case handling (too few frames, zero
noise).
"""
from __future__ import annotations

import unittest

import numpy as np

from src.bm4d import bm4d_denoise_stack, _estimate_stack_sigma


def _synthetic_truth(H: int, W: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    truth = 100.0 + 50.0 * np.exp(-((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (2 * 10.0 ** 2))
    truth += 8.0 * np.sin(xx / 5.0) * np.exp(-((yy - H * 0.73) ** 2) / (2 * 15.0 ** 2))
    return truth


class TestBm4dDenoiseStack(unittest.TestCase):

    def test_shape_dtype_finite(self):
        rng = np.random.default_rng(0)
        n, H, W = 10, 64, 64
        truth = _synthetic_truth(H, W)
        stack = np.stack([truth + rng.normal(0, 5.0, (H, W)) for _ in range(n)],
                         axis=0).astype(np.float32)
        out = bm4d_denoise_stack(stack)
        self.assertEqual(out.shape, stack.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_substantially_reduces_noise(self):
        rng = np.random.default_rng(1)
        n, H, W = 15, 96, 96
        truth = _synthetic_truth(H, W)
        noise_sigma = 6.0
        stack = np.stack([truth + rng.normal(0, noise_sigma, (H, W)) for _ in range(n)],
                         axis=0).astype(np.float32)

        denoised = bm4d_denoise_stack(stack, block_size=8)
        rmse_noisy = float(np.sqrt(np.mean((stack - truth[None]) ** 2)))
        rmse_denoised = float(np.sqrt(np.mean((denoised - truth[None]) ** 2)))
        self.assertLess(rmse_denoised, rmse_noisy * 0.3)

    def test_beats_naive_frame_mean(self):
        """The whole point of DCT-domain collaborative filtering on top of
        cross-frame correspondence: it should do better than just averaging
        the N frames together (which discards per-frame structure)."""
        rng = np.random.default_rng(2)
        n, H, W = 15, 96, 96
        truth = _synthetic_truth(H, W)
        noise_sigma = 6.0
        stack = np.stack([truth + rng.normal(0, noise_sigma, (H, W)) for _ in range(n)],
                         axis=0).astype(np.float32)

        denoised = bm4d_denoise_stack(stack, block_size=8)
        rmse_denoised = float(np.sqrt(np.mean((denoised - truth[None]) ** 2)))

        naive_mean = stack.mean(axis=0)
        rmse_naive = float(np.sqrt(np.mean(
            (np.broadcast_to(naive_mean, stack.shape) - truth[None]) ** 2)))
        self.assertLess(rmse_denoised, rmse_naive)

    def test_too_few_frames_returns_unchanged(self):
        rng = np.random.default_rng(3)
        H, W = 32, 32
        truth = _synthetic_truth(H, W)
        stack = np.stack([truth + rng.normal(0, 5.0, (H, W)) for _ in range(2)],
                         axis=0).astype(np.float32)
        out = bm4d_denoise_stack(stack)
        np.testing.assert_array_equal(out, stack.astype(np.float32))

    def test_zero_noise_returns_unchanged(self):
        H, W = 32, 32
        truth = _synthetic_truth(H, W)
        stack = np.stack([truth] * 5, axis=0).astype(np.float32)
        out = bm4d_denoise_stack(stack, sigma_psd=0.0)
        # Flat/identical frames -> near-zero estimated sigma -> early-return path.
        np.testing.assert_allclose(out, stack, atol=1e-3)

    def test_large_n_all_frames_denoised(self):
        """Every frame must actually be denoised, not just a subsample of
        them -- regression test for a bug caught during development where a
        since-removed group_cap silently left frames beyond the cap
        completely untouched (still full noise, flagged by a stray
        RuntimeWarning from a 0/0 divide on their never-written weight)."""
        rng = np.random.default_rng(4)
        n, H, W = 40, 48, 48
        truth = _synthetic_truth(H, W)
        noise_sigma = 6.0
        stack = np.stack([truth + rng.normal(0, noise_sigma, (H, W)) for _ in range(n)],
                         axis=0).astype(np.float32)
        out = bm4d_denoise_stack(stack, block_size=8)
        self.assertEqual(out.shape, stack.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        # Every single frame's own RMSE must have genuinely improved --
        # not just the stack average (which a partially-denoised stack
        # could still pass).
        rmse_noisy_per_frame = np.sqrt(np.mean((stack - truth[None]) ** 2, axis=(1, 2)))
        rmse_denoised_per_frame = np.sqrt(np.mean((out - truth[None]) ** 2, axis=(1, 2)))
        self.assertTrue(np.all(rmse_denoised_per_frame < rmse_noisy_per_frame * 0.5))


class TestEstimateStackSigma(unittest.TestCase):

    def test_recovers_known_noise_sigma(self):
        rng = np.random.default_rng(5)
        H, W = 128, 128
        truth = np.full((H, W), 500.0)
        true_sigma = 4.0
        stack = np.stack([truth + rng.normal(0, true_sigma, (H, W)) for _ in range(2)], axis=0)
        est = _estimate_stack_sigma(stack)
        self.assertAlmostEqual(est, true_sigma, delta=true_sigma * 0.15)

    def test_zero_for_identical_frames(self):
        H, W = 32, 32
        stack = np.stack([np.full((H, W), 100.0)] * 3, axis=0)
        self.assertLess(_estimate_stack_sigma(stack), 1e-9)

    def test_zero_for_single_frame(self):
        stack = np.zeros((1, 16, 16))
        self.assertEqual(_estimate_stack_sigma(stack), 0.0)


if __name__ == '__main__':
    unittest.main()
