"""Tests for Noise2Self-style self-supervised denoiser parameter
calibration (--denoise-strength-calibrate, src/self_supervised_calibration.py).

The core mechanism (build_masked_image + calibrate_denoiser_param) is
validated against synthetic ground truth with a simple, easy-to-reason-
about toy denoiser (Gaussian blur, one scalar parameter, well-understood
bias-variance tradeoff) *before* trusting it for the real wavelet-strength
integration -- same discipline this session already applied to
optimal_continuum_scale after that implementation's first design turned
out to pick the wrong scale.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from src.self_supervised_calibration import (
    build_masked_image,
    calibrate_denoiser_param,
    calibrate_wavelet_strength,
)


def _synthetic_scene(seed=0, h=96, w=96, noise_sigma=15.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    clean = 100.0 + 80.0 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 10.0 ** 2))
    clean += 20.0 * np.sin(xx / 6.0) * np.exp(-((yy - h * 0.7) ** 2) / (2 * 15.0 ** 2))
    noisy = clean + rng.normal(0, noise_sigma, (h, w))
    return clean, noisy.astype(np.float32)


class TestBuildMaskedImage:

    def test_shapes(self):
        img = np.random.default_rng(0).uniform(0, 100, (32, 32)).astype(np.float32)
        masked, ys, xs, true_vals = build_masked_image(img, mask_frac=0.05, seed=1)
        assert masked.shape == img.shape
        n_expected = int(32 * 32 * 0.05)
        assert len(ys) == n_expected
        assert len(xs) == n_expected
        assert len(true_vals) == n_expected

    def test_masked_positions_differ_from_original(self):
        rng = np.random.default_rng(2)
        img = rng.uniform(0, 100, (40, 40)).astype(np.float32)
        masked, ys, xs, true_vals = build_masked_image(img, mask_frac=0.05, seed=3)
        # At least most masked positions should have actually changed
        # (neighbor-average != original value, for generic random data).
        changed = np.abs(masked[ys, xs] - img[ys, xs]) > 1e-6
        assert changed.mean() > 0.8

    def test_true_vals_match_original_image(self):
        img = np.random.default_rng(4).uniform(0, 100, (32, 32)).astype(np.float32)
        _, ys, xs, true_vals = build_masked_image(img, mask_frac=0.03, seed=5)
        np.testing.assert_allclose(true_vals, img[ys, xs])


class TestCalibrateDenoiserParamAgainstGroundTruth:
    """The critical validation: does the self-supervised loss actually pick
    a parameter close to the one that minimizes TRUE error against known
    ground truth? Toy denoiser: Gaussian blur, sigma is the parameter --
    too small leaves noise, too large blurs away the signal, exactly the
    bias-variance tradeoff any denoiser strength parameter faces.
    """

    def test_selects_near_optimal_gaussian_blur_sigma(self):
        clean, noisy = _synthetic_scene(seed=10, noise_sigma=15.0)
        sigma_grid = np.array([0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0])

        # Ground truth: the sigma that actually minimizes MSE vs the known
        # clean image (only computable here because this is a synthetic
        # test -- a real run never has this).
        true_mse = np.array([
            np.mean((gaussian_filter(noisy, s) - clean) ** 2) for s in sigma_grid
        ])
        true_best = sigma_grid[np.argmin(true_mse)]

        def _blur_fn(x, sigma):
            return gaussian_filter(x, sigma)

        chosen, losses = calibrate_denoiser_param(
            noisy, _blur_fn, sigma_grid, mask_frac=0.05, seed=11)

        assert len(losses) == len(sigma_grid)
        # Within one grid step of the true optimum, not necessarily exact
        # -- self-supervised loss is an unbiased *estimator*, not a perfect
        # oracle, on a single finite noisy realization.
        true_idx = int(np.argmin(true_mse))
        chosen_idx = int(np.argmin(losses))
        assert abs(chosen_idx - true_idx) <= 1, (
            f"chosen sigma={chosen} (idx {chosen_idx}) vs true-optimal "
            f"sigma={true_best} (idx {true_idx})")

    def test_consistent_across_multiple_noise_realizations(self):
        # Guard against a single-seed false positive the same way this
        # session's IBP tests already do.
        hits = 0
        for seed in range(5):
            clean, noisy = _synthetic_scene(seed=seed, noise_sigma=12.0)
            sigma_grid = np.array([0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0])
            true_mse = np.array([
                np.mean((gaussian_filter(noisy, s) - clean) ** 2) for s in sigma_grid
            ])
            true_idx = int(np.argmin(true_mse))

            def _blur_fn(x, sigma):
                return gaussian_filter(x, sigma)

            _, losses = calibrate_denoiser_param(
                noisy, _blur_fn, sigma_grid, mask_frac=0.05, seed=seed + 100)
            chosen_idx = int(np.argmin(losses))
            if abs(chosen_idx - true_idx) <= 1:
                hits += 1
        assert hits >= 4

    def test_extreme_undersmoothing_scores_worse_than_moderate(self):
        clean, noisy = _synthetic_scene(seed=20, noise_sigma=20.0)
        sigma_grid = np.array([0.1, 2.0])

        def _blur_fn(x, sigma):
            return gaussian_filter(x, sigma)

        _, losses = calibrate_denoiser_param(
            noisy, _blur_fn, sigma_grid, mask_frac=0.05, seed=21)
        assert losses[1] < losses[0]  # moderate smoothing beats almost none


class TestCalibrateWaveletStrength:

    def test_runs_and_returns_value_in_grid(self):
        _, noisy = _synthetic_scene(seed=30, noise_sigma=10.0)
        img = np.stack([noisy] * 3, axis=-1).astype(np.float32)
        grid = (1.0, 2.0, 3.0, 4.0, 5.0)
        chosen, losses = calibrate_wavelet_strength(img, param_grid=grid, seed=31)
        assert chosen in grid
        assert len(losses) == len(grid)
        assert np.all(np.isfinite(losses))

    def test_higher_noise_prefers_stronger_or_equal_smoothing(self):
        # Not a strict monotonic guarantee (single noisy realization,
        # discrete grid), but a noisier stack should not be calibrated to
        # noticeably *less* smoothing than a cleaner one on average.
        grid = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        _, low_noise = _synthetic_scene(seed=40, noise_sigma=4.0)
        _, high_noise = _synthetic_scene(seed=40, noise_sigma=25.0)
        img_low = np.stack([low_noise] * 3, axis=-1).astype(np.float32)
        img_high = np.stack([high_noise] * 3, axis=-1).astype(np.float32)
        chosen_low, _ = calibrate_wavelet_strength(img_low, param_grid=grid, seed=41)
        chosen_high, _ = calibrate_wavelet_strength(img_high, param_grid=grid, seed=41)
        assert chosen_high >= chosen_low
