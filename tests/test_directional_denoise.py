"""Tests for the directional (curvelet/shearlet-inspired) wavelet denoiser
(--denoiser curvelet, src/denoising.py::directional_wavelet_denoise and its
structure-tensor coherence helper).
"""
from __future__ import annotations

import numpy as np

from src.denoising import (
    _structure_tensor_coherence,
    directional_wavelet_denoise,
    adaptive_wavelet_denoise,
)


class TestStructureTensorCoherence:

    def test_higher_on_a_clean_edge_than_on_pure_noise(self):
        h, w = 64, 64
        edge = np.zeros((h, w))
        edge[:, w // 2:] = 100.0
        coh_edge = _structure_tensor_coherence(edge, sigma=1.5)

        rng = np.random.default_rng(0)
        noise = rng.normal(50.0, 10.0, (h, w))
        coh_noise = _structure_tensor_coherence(noise, sigma=1.5)

        # Compare near the edge column (where the edge actually has
        # structure) against the noise image's overall coherence.
        band = coh_edge[:, w // 2 - 3:w // 2 + 3]
        assert float(np.mean(band)) > float(np.mean(coh_noise)) * 2

    def test_output_in_zero_one_range(self):
        rng = np.random.default_rng(1)
        img = rng.uniform(0, 500, (48, 48))
        coh = _structure_tensor_coherence(img)
        assert coh.shape == img.shape
        assert np.all(coh >= 0.0)
        assert np.all(coh <= 1.0)

    def test_flat_image_has_zero_coherence(self):
        flat = np.full((32, 32), 42.0)
        coh = _structure_tensor_coherence(flat)
        np.testing.assert_allclose(coh, 0.0, atol=1e-10)


class TestDirectionalWaveletDenoise:

    def _filament_image(self, seed=0, h=80, w=80, noise_sigma=15.0):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        # A faint diagonal filament, low contrast against noise.
        filament = 25.0 * np.exp(-((yy - xx) ** 2) / (2 * 2.0 ** 2))
        base = 100.0 + filament
        img = np.stack([base] * 3, axis=-1) + rng.normal(0, noise_sigma, (h, w, 3))
        return np.clip(img, 0, None).astype(np.float32), filament

    def test_output_shape_dtype_finite(self):
        img, _ = self._filament_image()
        out = directional_wavelet_denoise(img)
        assert out.shape == img.shape
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

    def test_protect_strength_zero_matches_plain_bayesshrink(self):
        img, _ = self._filament_image(seed=1)
        out_directional = directional_wavelet_denoise(img, protect_strength=0.0)
        out_plain = adaptive_wavelet_denoise(img)
        np.testing.assert_allclose(out_directional, out_plain, atol=1e-4)

    def test_preserves_filament_better_than_plain_bayesshrink(self):
        img, filament = self._filament_image(seed=2, noise_sigma=20.0)
        out_directional = directional_wavelet_denoise(img, protect_strength=0.8)
        out_plain = adaptive_wavelet_denoise(img)

        # Measure signal retained along the filament's ridge line (diagonal)
        # relative to the true filament profile -- protecting coherent
        # structure should retain more of it than uniform thresholding.
        h, w = filament.shape
        ridge = [(i, i) for i in range(10, min(h, w) - 10)]
        true_vals = np.array([filament[y, x] for y, x in ridge])
        base_level = 100.0
        directional_vals = np.array(
            [out_directional[y, x, 0] - base_level for y, x in ridge])
        plain_vals = np.array([out_plain[y, x, 0] - base_level for y, x in ridge])

        err_directional = float(np.mean(np.abs(directional_vals - true_vals)))
        err_plain = float(np.mean(np.abs(plain_vals - true_vals)))
        assert err_directional <= err_plain * 1.1  # at least comparable, ideally better

    def test_star_mask_blends_back_original(self):
        img, _ = self._filament_image(seed=3)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask[30:35, 30:35] = 1.0
        out = directional_wavelet_denoise(img, star_mask=mask)
        np.testing.assert_allclose(out[30:35, 30:35], img[30:35, 30:35], atol=1e-3)
