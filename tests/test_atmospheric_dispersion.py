"""Tests for software atmospheric dispersion correction
(--fix-atmospheric-dispersion, src/atmospheric_dispersion.py).

Flagged in the project's own research as experimental/first-principles
(no established software reference implementation exists to validate
against) -- these tests check the refractive-index formula against a
well-known independent physical fact (air's refractive index at visible
wavelengths), and check the differential-refraction/pixel-shift logic
behaves in the physically-expected directions, rather than asserting exact
values from a reference implementation that doesn't exist.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.atmospheric_dispersion import (
    _refractive_index_air_minus_one,
    differential_refraction_arcsec,
    correct_atmospheric_dispersion,
)


class TestRefractiveIndex:

    def test_matches_well_known_visible_ballpark(self):
        # Air's refractive index at visible wavelengths is a well-known,
        # independently-verifiable fact: n ~= 1.00027-1.00028 (280ish ppm
        # above 1), regardless of which specific dispersion formula is used.
        n_minus_1 = _refractive_index_air_minus_one(np.array([500.0]))[0]
        assert 2.5e-4 < n_minus_1 < 3.0e-4

    def test_decreases_with_wavelength_in_visible_range(self):
        # Normal dispersion: shorter wavelengths refract more.
        n_blue = _refractive_index_air_minus_one(np.array([450.0]))[0]
        n_red = _refractive_index_air_minus_one(np.array([650.0]))[0]
        assert n_blue > n_red

    def test_positive_for_all_visible_wavelengths(self):
        wl = np.linspace(400.0, 700.0, 20)
        n_minus_1 = _refractive_index_air_minus_one(wl)
        assert np.all(n_minus_1 > 0)


class TestDifferentialRefraction:

    def test_zero_for_equal_wavelengths(self):
        sep = differential_refraction_arcsec(550.0, 550.0, zenith_angle_deg=45.0)
        assert abs(sep) < 1e-9

    def test_zero_at_zenith(self):
        # At zenith angle 0, tan(0)=0 -- no differential refraction
        # regardless of wavelength separation.
        sep = differential_refraction_arcsec(450.0, 650.0, zenith_angle_deg=0.0)
        assert abs(sep) < 1e-9

    def test_increases_with_zenith_angle(self):
        sep_30 = abs(differential_refraction_arcsec(450.0, 650.0, zenith_angle_deg=30.0))
        sep_60 = abs(differential_refraction_arcsec(450.0, 650.0, zenith_angle_deg=60.0))
        assert sep_60 > sep_30

    def test_increases_with_wavelength_separation(self):
        sep_narrow = abs(differential_refraction_arcsec(500.0, 550.0, zenith_angle_deg=50.0))
        sep_wide = abs(differential_refraction_arcsec(400.0, 700.0, zenith_angle_deg=50.0))
        assert sep_wide > sep_narrow

    def test_reasonable_magnitude_at_moderate_altitude(self):
        # Sanity: at 45deg zenith angle (45deg altitude), full visible-band
        # (400-700nm) dispersion should be on the order of ~1 arcsec, not
        # micro-arcsec or degrees -- this is the well-known real-world scale
        # atmospheric dispersion correctors are built to address.
        sep = abs(differential_refraction_arcsec(400.0, 700.0, zenith_angle_deg=45.0))
        assert 0.1 < sep < 10.0


class TestCorrectAtmosphericDispersion:

    def _test_image(self, h=48, w=48):
        rng = np.random.default_rng(0)
        base = rng.uniform(0, 100, (h, w))
        return np.stack([base] * 3, axis=-1).astype(np.float32)

    def test_output_shape_dtype(self):
        img = self._test_image()
        out = correct_atmospheric_dispersion(
            img, plate_scale_arcsec_px=1.0, zenith_angle_deg=45.0,
            parallactic_angle_deg=0.0)
        assert out.shape == img.shape
        assert out.dtype == np.float32

    def test_reference_channel_unchanged(self):
        img = self._test_image()
        out = correct_atmospheric_dispersion(
            img, plate_scale_arcsec_px=1.0, zenith_angle_deg=45.0,
            parallactic_angle_deg=0.0, reference_index=1)
        np.testing.assert_allclose(out[:, :, 1], img[:, :, 1])

    def test_non_reference_channels_shifted_at_nonzero_zenith(self):
        img = self._test_image()
        out = correct_atmospheric_dispersion(
            img, plate_scale_arcsec_px=0.5, zenith_angle_deg=60.0,
            parallactic_angle_deg=0.0, reference_index=1)
        assert not np.allclose(out[:, :, 0], img[:, :, 0])
        assert not np.allclose(out[:, :, 2], img[:, :, 2])

    def test_negligible_shift_at_zenith(self):
        img = self._test_image()
        out = correct_atmospheric_dispersion(
            img, plate_scale_arcsec_px=1.0, zenith_angle_deg=0.0,
            parallactic_angle_deg=0.0, reference_index=1)
        np.testing.assert_allclose(out[:, :, 0], img[:, :, 0], atol=1e-6)
        np.testing.assert_allclose(out[:, :, 2], img[:, :, 2], atol=1e-6)

    def test_rejects_non_rgb_input(self):
        img = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            correct_atmospheric_dispersion(
                img, plate_scale_arcsec_px=1.0, zenith_angle_deg=45.0,
                parallactic_angle_deg=0.0)
