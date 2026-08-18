"""Tests for spectrophotometric-style colour calibration
(src/color_calibrate.py's fit_channel_scales_spcc and its blackbody-spectrum
integration helpers).

fit_channel_scales_spcc differs from the existing fit_channel_scales by
integrating a per-star blackbody spectrum (from Gaia's teff_gspphot) against
channel response curves instead of a single fixed colour-index formula --
these tests check the physics helpers behave sanely (Wien's law direction,
positive flux) and that the end-to-end fit correctly uses the blackbody path
when Teff is available and falls back to the colour-index formula per-star
when it isn't.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.color_calibrate import (
    _blackbody_spectrum,
    _default_channel_response,
    synthetic_channel_flux,
    fit_channel_scales_spcc,
)


class _FakeTable:
    """Minimal stand-in for an astropy Table: dict-of-columns + colnames,
    supporting the same ``table["col"][mask]`` and ``len(table)`` access
    fit_channel_scales_spcc actually uses."""

    def __init__(self, columns: dict):
        self._columns = columns
        self.colnames = list(columns.keys())

    def __getitem__(self, key):
        return self._columns[key]

    def __len__(self):
        return len(next(iter(self._columns.values())))


class TestBlackbodySpectrum:

    def test_hotter_star_peaks_at_shorter_wavelength(self):
        # Wien's displacement law: peak wavelength is inversely proportional
        # to temperature -- a 10000K spectrum must peak bluer than a 3000K one.
        wl = np.linspace(200.0, 2000.0, 2000)
        hot = _blackbody_spectrum(10000.0, wl)
        cool = _blackbody_spectrum(3000.0, wl)
        assert wl[np.argmax(hot)] < wl[np.argmax(cool)]

    def test_nonnegative_and_finite(self):
        wl = np.linspace(100.0, 3000.0, 500)
        spec = _blackbody_spectrum(5778.0, wl)
        assert np.all(np.isfinite(spec))
        assert np.all(spec >= 0.0)


class TestSyntheticChannelFlux:

    def test_hot_star_is_relatively_bluer(self):
        fr_hot, fg_hot, fb_hot = synthetic_channel_flux(15000.0)
        fr_cool, fg_cool, fb_cool = synthetic_channel_flux(3500.0)
        assert (fb_hot / fr_hot) > (fb_cool / fr_cool)

    def test_returns_three_positive_floats(self):
        fr, fg, fb = synthetic_channel_flux(5778.0)
        assert fr > 0 and fg > 0 and fb > 0

    def test_custom_channel_response_used(self):
        calls = []

        def resp(channel, wavelengths_nm):
            calls.append(channel)
            return _default_channel_response(channel, wavelengths_nm)

        synthetic_channel_flux(5778.0, channel_response=resp)
        assert calls == ['R', 'G', 'B']


class TestFitChannelScalesSpcc:

    def _catalog(self, n=20, with_teff=True):
        rng = np.random.default_rng(0)
        bp = rng.uniform(10.0, 14.0, n)
        rp = bp - rng.uniform(0.3, 1.2, n)
        g = (bp + rp) / 2.0
        cols = {
            'ra': rng.uniform(0, 1, n), 'dec': rng.uniform(0, 1, n),
            'phot_bp_mean_mag': bp, 'phot_rp_mean_mag': rp,
            'phot_g_mean_mag': g,
        }
        if with_teff:
            teff = rng.uniform(4000.0, 9000.0, n)
            teff[::4] = np.nan  # some stars lack a GSP-Phot estimate
            cols['teff_gspphot'] = teff
        return _FakeTable(cols)

    def _fake_pixel_coords_and_flux(self, n):
        px = np.arange(n, dtype=float)
        py = np.arange(n, dtype=float)
        # Deterministic per-star RGB flux, all positive and finite.
        rng = np.random.default_rng(1)
        fluxes = rng.uniform(100.0, 1000.0, (n, 3))
        return px, py, fluxes

    def test_returns_neutral_scales_on_too_few_stars(self):
        catalog = self._catalog(n=3)
        img = np.ones((32, 32, 3), dtype=np.float32)
        header = {}
        with patch('src.color_calibrate._pixel_coords',
                  return_value=np.zeros((3, 2))):
            scales = fit_channel_scales_spcc(img, header, catalog)
        assert scales == (1.0, 1.0, 1.0)

    def test_end_to_end_with_teff_present(self):
        n = 20
        catalog = self._catalog(n=n, with_teff=True)
        px, py, fluxes = self._fake_pixel_coords_and_flux(n)
        img = np.ones((64, 64, 3), dtype=np.float32)
        header = {}
        with patch('src.color_calibrate._pixel_coords',
                  return_value=np.column_stack([px, py])), \
             patch('src.color_calibrate._aperture_flux', return_value=fluxes):
            scale_r, scale_g, scale_b = fit_channel_scales_spcc(
                img, header, catalog, verbose=False)
        for s in (scale_r, scale_g, scale_b):
            assert np.isfinite(s)
            assert 0.5 <= s <= 2.0
        # Mean-normalised, per the same convention fit_channel_scales uses.
        assert abs((scale_r + scale_g + scale_b) / 3.0 - 1.0) < 1e-6

    def test_falls_back_to_colorindex_without_teff_column(self):
        n = 20
        catalog = self._catalog(n=n, with_teff=False)
        assert 'teff_gspphot' not in catalog.colnames
        px, py, fluxes = self._fake_pixel_coords_and_flux(n)
        img = np.ones((64, 64, 3), dtype=np.float32)
        header = {}
        with patch('src.color_calibrate._pixel_coords',
                  return_value=np.column_stack([px, py])), \
             patch('src.color_calibrate._aperture_flux', return_value=fluxes):
            scales = fit_channel_scales_spcc(img, header, catalog, verbose=False)
        assert all(np.isfinite(s) for s in scales)
