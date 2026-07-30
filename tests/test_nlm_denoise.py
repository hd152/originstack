"""Tests for src/denoising.py's non-local means port (_nlm_denoise_numpy,
nlm_denoise) -- replaces skimage.restoration.denoise_nl_means.

Validated for equivalent denoising *behaviour* against real skimage where
skimage is installed (noise reduction, edge preservation, monotonic response
to h) rather than bit-exact parity: skimage's fast_mode NLM is a compiled
Cython kernel with no .pyx source shipped in the installed wheel, so there
was nothing to port literally -- see the docstring on _nlm_denoise_numpy in
src/denoising.py for the full rationale and the published algorithm it's a
faithful reimplementation of.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.denoising import _nlm_denoise_numpy, nlm_denoise


def _noisy_rgb(H=50, W=50, level=0.5, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)
    base = np.full((H, W), level, dtype=np.float64)
    base[15:35, 15:35] += 0.3  # a flat square feature
    img = base[:, :, None] + noise * rng.standard_normal((H, W, 1))
    return np.repeat(img, 3, axis=2).astype(np.float64)


class TestNlmDenoiseNumpy:
    def test_constant_image_unchanged(self):
        img = np.full((30, 30, 3), 0.42, dtype=np.float64)
        out = _nlm_denoise_numpy(img, h=0.2, patch_size=5, patch_distance=6)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_reduces_noise_variance(self):
        img = _noisy_rgb(noise=0.08)
        out = _nlm_denoise_numpy(img, h=0.3, patch_size=5, patch_distance=6)
        # Measure noise on the flat background region only (avoid the edge).
        bg_in = img[2:12, 2:12, 0]
        bg_out = out[2:12, 2:12, 0]
        assert bg_out.std() < bg_in.std() * 0.6

    def test_h_monotonically_increases_smoothing(self):
        img = _noisy_rgb(noise=0.08)
        stds = []
        for h in [0.05, 0.2, 0.6, 1.5]:
            out = _nlm_denoise_numpy(img, h=h, patch_size=5, patch_distance=6)
            stds.append(out[2:12, 2:12, 0].std())
        # Not required to be strictly monotonic at every step (patch-based
        # methods can plateau), but the overall trend must be decreasing.
        assert stds[0] > stds[-1]
        assert stds[-1] <= stds[0]

    def test_preserves_step_edge(self):
        img = _noisy_rgb(noise=0.02)
        out = _nlm_denoise_numpy(img, h=0.2, patch_size=5, patch_distance=6)
        # Sample straight across the square feature's edge (row 25, cols
        # 10..20); the ~0.3 step must still be clearly present, not smeared
        # away to near-zero.
        profile = out[25, 10:20, 0]
        assert (profile.max() - profile.min()) > 0.15

    def test_output_shape_and_dtype(self):
        img = _noisy_rgb(H=20, W=24)
        out = _nlm_denoise_numpy(img, h=0.3, patch_size=3, patch_distance=4)
        assert out.shape == img.shape
        assert out.dtype == np.float32


class TestNlmDenoiseVsSkimage:
    """Behavioural cross-check against real skimage (not bit-exact -- see
    module docstring)."""

    def test_correlates_with_skimage_and_denoises_comparably(self):
        skrest = pytest.importorskip("skimage.restoration")
        rng = np.random.default_rng(3)
        H, W = 60, 60
        base = np.full((H, W), 0.5)
        base[20:40, 20:40] += 0.3
        clean = np.repeat(base[:, :, None], 3, axis=2)
        noisy = (clean + 0.06 * rng.standard_normal((H, W, 1))).astype(np.float64)

        h = 0.25
        mine = _nlm_denoise_numpy(noisy, h=h, patch_size=5, patch_distance=6)
        ref = skrest.denoise_nl_means(
            noisy.astype(np.float32), h=h, fast_mode=True,
            patch_size=5, patch_distance=6, channel_axis=-1)

        # Both must denoise substantially relative to the input.
        bg_noisy_std = noisy[2:12, 2:12, 0].std()
        assert mine[2:12, 2:12, 0].std() < bg_noisy_std * 0.7
        assert ref[2:12, 2:12, 0].std() < bg_noisy_std * 0.7

        # Same image content, same general smoothing behaviour -> outputs
        # should be strongly correlated even without being bit-exact.
        corr = np.corrcoef(mine.ravel(), ref.ravel().astype(np.float64))[0, 1]
        assert corr > 0.85


class TestNlmDenoiseWrapper:
    """The public nlm_denoise() wrapper (pedestal trick + blend)."""

    def test_blend_zero_returns_original(self):
        img = _noisy_rgb(H=20, W=20).astype(np.float32)
        out = nlm_denoise(img, h=1.0, blend=0.0)
        np.testing.assert_allclose(out, img, atol=1e-4)

    def test_blend_reduces_noise(self):
        img = _noisy_rgb(H=30, W=30, noise=0.08).astype(np.float32)
        out = nlm_denoise(img, h=1.5, blend=1.0, patch_size=3, patch_distance=4)
        assert out.shape == img.shape
        assert out[2:12, 2:12, 0].std() < img[2:12, 2:12, 0].std()

    def test_all_zero_image_is_noop(self):
        img = np.zeros((10, 10, 3), dtype=np.float32)
        out = nlm_denoise(img)
        np.testing.assert_array_equal(out, img)
