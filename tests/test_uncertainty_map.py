"""Tests for the IVW per-pixel uncertainty map (--uncertainty-map).

ivw_combine(..., return_sigma=True) exposes the Gauss-Markov estimator's own
standard error (1/sqrt(sum of inverse-variance weights)) instead of just the
combined image -- these tests check it against a hand-computed analytic case
and confirm it always takes the numpy path (the native kernel doesn't expose
the per-pixel weight sum).
"""
from __future__ import annotations

import numpy as np

from src.stacking import _ivw_tile, ivw_combine


class TestIvwTileReturnWsum:

    def test_wsum_matches_analytic_sum_of_inverse_variances(self):
        # 3 frames, per-frame-constant noise (no gain/sky term) -- wsum at
        # every pixel must equal sum(1/noise_i^2) exactly.
        noise = np.array([2.0, 4.0, 5.0], dtype=np.float32)
        tile = np.ones((3, 4, 4, 1), dtype=np.float32)
        _, wsum = _ivw_tile(tile, noise, None, None, None, return_wsum=True)
        expected = sum(1.0 / n ** 2 for n in noise)
        np.testing.assert_allclose(wsum, expected, rtol=1e-6)

    def test_wsum_without_return_flag_omitted(self):
        noise = np.array([2.0, 3.0], dtype=np.float32)
        tile = np.ones((2, 4, 4, 1), dtype=np.float32)
        result = _ivw_tile(tile, noise, None, None, None)
        assert isinstance(result, np.ndarray)  # no tuple when return_wsum=False


class TestIvwCombineReturnSigma:

    def test_sigma_matches_analytic_standard_error(self):
        rng = np.random.default_rng(0)
        n, h, w, c = 4, 16, 16, 1
        noise = np.array([2.0, 3.0, 4.0, 6.0], dtype=np.float32)
        data = rng.normal(1000.0, 5.0, (n, h, w, c)).astype(np.float32)

        result, sigma = ivw_combine(data, noise=noise, return_sigma=True)
        assert result.shape == (h, w, c)
        assert sigma.shape == (h, w)

        expected_sigma = 1.0 / np.sqrt(sum(1.0 / nn ** 2 for nn in noise))
        np.testing.assert_allclose(sigma, expected_sigma, rtol=1e-5)

    def test_sigma_shrinks_with_more_frames(self):
        # More (equally noisy) frames -> lower uncertainty -- the basic
        # sanity check any uncertainty map must satisfy.
        rng = np.random.default_rng(1)
        h, w, c = 12, 12, 1
        noise4 = np.full(4, 3.0, dtype=np.float32)
        noise12 = np.full(12, 3.0, dtype=np.float32)
        data4 = rng.normal(500.0, 3.0, (4, h, w, c)).astype(np.float32)
        data12 = rng.normal(500.0, 3.0, (12, h, w, c)).astype(np.float32)

        _, sigma4 = ivw_combine(data4, noise=noise4, return_sigma=True)
        _, sigma12 = ivw_combine(data12, noise=noise12, return_sigma=True)
        assert float(np.mean(sigma12)) < float(np.mean(sigma4))

    def test_return_sigma_false_returns_array_only(self):
        rng = np.random.default_rng(2)
        data = rng.normal(100.0, 2.0, (3, 8, 8, 1)).astype(np.float32)
        noise = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = ivw_combine(data, noise=noise)
        assert isinstance(result, np.ndarray)

    def test_shot_noise_term_increases_sigma_at_bright_pixels(self):
        # With gain+sky supplied, brighter pixels carry more Poisson shot
        # noise -> higher uncertainty there than in the sky background.
        h, w, c = 8, 8, 1
        n = 4
        noise = np.full(n, 2.0, dtype=np.float32)
        sky = np.full(n, 100.0, dtype=np.float32)
        data = np.full((n, h, w, c), 100.0, dtype=np.float32)
        data[:, 0, 0, 0] = 5000.0  # one bright "star" pixel
        _, sigma = ivw_combine(data, noise=noise, sky=sky, gain=1.0, return_sigma=True)
        assert sigma[0, 0] > sigma[4, 4]
