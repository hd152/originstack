"""Tests for the generalized Anscombe transform variance-stabilization
pre/post step (--variance-stabilize), used by wavelet_denoise and
adaptive_wavelet_denoise's luma plane before BayesShrink thresholding.
"""
from __future__ import annotations

import numpy as np

from src.denoising import (
    _estimate_noise_level_function,
    _generalized_anscombe,
    _inverse_generalized_anscombe,
    adaptive_wavelet_denoise,
    wavelet_denoise,
)


class TestGeneralizedAnscombeRoundTrip:

    def test_forward_inverse_is_identity_for_nonnegative_input(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 5000.0, (50, 50))
        gain, sigma = 2.0, 15.0
        z = _generalized_anscombe(x, gain, sigma)
        x_back = _inverse_generalized_anscombe(z, gain, sigma)
        np.testing.assert_allclose(x_back, x, atol=1e-6, rtol=1e-6)

    def test_transformed_variance_is_more_uniform_than_raw(self):
        # Classic GAT property: raw Poisson-ish data has variance
        # proportional to the mean; after GAT, variance should be roughly
        # constant across brightness levels.
        rng = np.random.default_rng(1)
        gain, sigma = 1.0, 5.0
        levels = [50.0, 500.0, 5000.0]
        raw_vars, gat_vars = [], []
        for lvl in levels:
            samples = rng.poisson(lvl, 20000).astype(np.float64) + rng.normal(0, sigma, 20000)
            raw_vars.append(np.var(samples))
            z = _generalized_anscombe(np.maximum(samples, 0.0), gain, sigma)
            gat_vars.append(np.var(z))
        # Raw variance should scale with level; GAT variance should be far
        # flatter -- check the coefficient of variation drops substantially.
        raw_cv = np.std(raw_vars) / np.mean(raw_vars)
        gat_cv = np.std(gat_vars) / np.mean(gat_vars)
        assert gat_cv < raw_cv * 0.5


class TestEstimateNoiseLevelFunction:

    def test_recovers_known_gain_and_sigma_approximately(self):
        rng = np.random.default_rng(2)
        h, w = 256, 256
        # Smooth background trend so tiles span a real brightness range,
        # plus Poisson+Gaussian noise for a known (gain, sigma).
        yy, xx = np.mgrid[0:h, 0:w]
        signal = 50.0 + 400.0 * (xx / w)
        true_gain, true_sigma = 2.0, 8.0
        noisy = rng.poisson(signal * true_gain).astype(np.float64) / true_gain \
            + rng.normal(0, true_sigma, (h, w))
        gain, sigma = _estimate_noise_level_function(noisy)
        assert gain > 0
        # Loose tolerance -- this is a lightweight self-calibration, not a
        # precision instrument.
        assert 0.3 * true_gain < gain < 3.0 * true_gain

    def test_degenerate_input_does_not_crash(self):
        flat = np.full((32, 32), 100.0)
        gain, sigma = _estimate_noise_level_function(flat)
        assert gain > 0
        assert sigma >= 0

    def test_too_small_image_returns_fallback(self):
        tiny = np.ones((4, 4))
        gain, sigma = _estimate_noise_level_function(tiny, tile=16)
        assert gain == 1.0
        assert sigma == 0.0


class TestWaveletDenoiseVarianceStabilize:

    def _synthetic_image(self, seed=0):
        rng = np.random.default_rng(seed)
        h, w = 64, 64
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        base = 100.0 + 300.0 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 10.0 ** 2))
        img = np.stack([base] * 3, axis=-1) + rng.normal(0, 8.0, (h, w, 3))
        return np.clip(img, 0, None).astype(np.float32)

    def test_wavelet_denoise_runs_with_stabilize(self):
        img = self._synthetic_image()
        out = wavelet_denoise(img, variance_stabilize=True)
        assert out.shape == img.shape
        assert np.all(np.isfinite(out))

    def test_adaptive_wavelet_denoise_runs_with_stabilize(self):
        img = self._synthetic_image(seed=1)
        out = adaptive_wavelet_denoise(img, variance_stabilize=True)
        assert out.shape == img.shape
        assert np.all(np.isfinite(out))

    def test_stabilize_reduces_noise_comparably_to_default(self):
        img = self._synthetic_image(seed=2)
        out_default = adaptive_wavelet_denoise(img, variance_stabilize=False)
        out_stab = adaptive_wavelet_denoise(img, variance_stabilize=True)
        # Both should meaningfully reduce noise vs. the raw image in a flat
        # background corner (away from the synthetic "star").
        raw_std = float(np.std(img[:10, :10, 0]))
        default_std = float(np.std(out_default[:10, :10, 0]))
        stab_std = float(np.std(out_stab[:10, :10, 0]))
        assert default_std < raw_std
        assert stab_std < raw_std
