"""Tests for src/denoising.py's bilateral filter -- replaces cv2.bilateralFilter,
the last cv2-only feature in this codebase with no fallback (VNG debayer already
aliased to Malvar, NLM already preferred skimage.restoration). This codebase now
has no cv2 dependency at all.

Native/numpy parity is exact (same mirror_idx / np.pad(mode='reflect') boundary
convention on both sides -- see src/denoising.py's _bilateral_filter_numpy and
ext/astro_native/src/lib.rs's bilateral_filter docstrings). Quality properties
(flat-field invariance, edge preservation, noise reduction) are checked against
synthetic ground truth rather than against cv2, since we're replacing it.
"""
from __future__ import annotations

import numpy as np
import pytest

import src.denoising as dn


class TestBilateralFilterProperties:
    def test_flat_field_unchanged(self):
        flat = np.full((30, 30, 3), 500.0, dtype=np.float32)
        out = dn.bilateral_denoise(flat, sigma_color=20.0, sigma_space=3.0)
        np.testing.assert_allclose(out, 500.0, atol=1e-3)

    def test_hard_edge_not_blurred_across(self):
        h, w = 60, 60
        img = np.zeros((h, w, 3), dtype=np.float32)
        img[:, w // 2:] = 1000.0
        out = dn.bilateral_denoise(img, sigma_color=20.0, sigma_space=3.0)
        # far from the edge, values should stay exactly at the two levels
        assert np.allclose(out[:, :w // 2 - 5], 0.0, atol=1.0)
        assert np.allclose(out[:, w // 2 + 5:], 1000.0, atol=1.0)

    def test_reduces_noise_on_flat_region(self):
        rng = np.random.default_rng(1)
        noisy = (500 + rng.normal(0, 30, (100, 120, 3))).astype(np.float32)
        out = dn.bilateral_denoise(noisy, sigma_color=60.0, sigma_space=3.0)
        assert out.std() < noisy.std() * 0.5

    def test_zero_image_short_circuits(self):
        img = np.zeros((10, 10, 3), dtype=np.float32)
        out = dn.bilateral_denoise(img, sigma_color=10.0, sigma_space=3.0)
        np.testing.assert_array_equal(out, img)

    def test_auto_sigma_color_when_none(self):
        rng = np.random.default_rng(2)
        img = (500 + rng.normal(0, 20, (40, 40, 3))).astype(np.float32)
        out = dn.bilateral_denoise(img, sigma_color=None, sigma_space=3.0)
        assert out.shape == img.shape


class TestBilateralFilterNativeNumpyParity:
    def test_native_matches_numpy_exactly(self):
        if not dn._HAS_NATIVE or not hasattr(dn._native, 'bilateral_filter'):
            pytest.skip("astro_native bilateral_filter kernel not built")
        rng = np.random.default_rng(3)
        img = rng.uniform(0, 1000, (40, 50, 3)).astype(np.float32)
        sigma_color, sigma_space, radius = 50.0, 3.0, 9
        native_out = np.asarray(
            dn._native.bilateral_filter(np.ascontiguousarray(img), sigma_color, sigma_space, radius))
        numpy_out = dn._bilateral_filter_numpy(img, sigma_color, sigma_space, radius)
        np.testing.assert_allclose(native_out, numpy_out, atol=1e-4)

    @pytest.mark.parametrize('radius', [1, 3, 9])
    def test_native_matches_numpy_various_radii(self, radius):
        if not dn._HAS_NATIVE or not hasattr(dn._native, 'bilateral_filter'):
            pytest.skip("astro_native bilateral_filter kernel not built")
        rng = np.random.default_rng(4)
        img = rng.uniform(0, 1000, (25, 30, 3)).astype(np.float32)
        native_out = np.asarray(
            dn._native.bilateral_filter(np.ascontiguousarray(img), 40.0, 2.0, radius))
        numpy_out = dn._bilateral_filter_numpy(img, 40.0, 2.0, radius)
        np.testing.assert_allclose(native_out, numpy_out, atol=1e-4)
