"""Tests for Gaia DR3 astrometric distortion correction
(--gaia-distortion-correction).

fit_gaia_distortion_field cross-matches Gaia-predicted pixel positions
(from an existing WCS) against detected reference-frame star centroids, then
reuses fit_displacement_field's DBE-style local regression to fit the
residual as a session-wide distortion field. These tests build a synthetic
WCS + a synthetic Gaia catalog + detected positions with a KNOWN injected
distortion, and confirm the fitted field recovers it -- not just "runs
without crashing". The sign convention was verified by hand against
apply_transform's local_field composition before locking in these
expectations (see module history: the first draft of this test had the sign
backwards, not the implementation -- corrected before assuming either was
right).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

from src.gaia_distortion import fit_gaia_distortion_field, GAIA_MIN_MATCHES
from src.registration import sample_displacement_field


def _synthetic_wcs_header(H: int, W: int) -> dict:
    return {
        'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
        'CRVAL1': 180.0, 'CRVAL2': 30.0,
        'CRPIX1': W / 2.0, 'CRPIX2': H / 2.0,
        'CDELT1': -0.0005, 'CDELT2': 0.0005,
        'CUNIT1': 'deg', 'CUNIT2': 'deg',
    }


def _linear_distortion(px: np.ndarray, H: int, W: int) -> np.ndarray:
    """Smooth synthetic distortion: linear gradient in x, quadratic-ish in y."""
    x, y = px[:, 0], px[:, 1]
    dx = 3.0 * (x - W / 2) / (W / 2)
    dy = 2.0 * ((y - H / 2) / (H / 2)) ** 2 * np.sign(y - H / 2 + 1e-9)
    return np.column_stack([dx, dy])


def _build_scenario(H=512, W=512, M=60, seed=0):
    hdr = _synthetic_wcs_header(H, W)
    wcs = WCS(hdr, naxis=2)
    rng = np.random.default_rng(seed)
    true_px = np.column_stack([rng.uniform(20, W - 20, M), rng.uniform(20, H - 20, M)])
    radec = wcs.all_pix2world(true_px, 0)

    dist = _linear_distortion(true_px, H, W)
    detected_px = true_px + dist

    dt = np.dtype([('xcentroid', np.float64), ('ycentroid', np.float64), ('flux', np.float64)])
    ref_stars = np.zeros(M, dtype=dt)
    ref_stars['xcentroid'] = detected_px[:, 0]
    ref_stars['ycentroid'] = detected_px[:, 1]
    ref_stars['flux'] = 1000.0

    fake_table = Table({'ra': radec[:, 0], 'dec': radec[:, 1],
                        'phot_g_mean_mag': np.full(M, 12.0)})
    return hdr, ref_stars, true_px, dist, fake_table, H, W


class TestFitGaiaDistortionField(unittest.TestCase):

    def test_recovers_known_injected_distortion(self):
        hdr, ref_stars, true_px, dist, fake_table, H, W = _build_scenario()
        with patch('src.net_query.gaia_cone_search', return_value=fake_table):
            field = fit_gaia_distortion_field(hdr, ref_stars, H, W)

        self.assertIsNotNone(field)
        dy_rec, dx_rec = sample_displacement_field(field, H, W, true_px[:, 1], true_px[:, 0])
        # Convention: field = predicted - detected = -dist (verified against
        # apply_transform's src(o) = o - D(o) composition -- see module docstring).
        self.assertLess(float(np.mean(np.abs(dx_rec - (-dist[:, 0])))), 0.5)
        self.assertLess(float(np.mean(np.abs(dy_rec - (-dist[:, 1])))), 0.5)

    def test_returns_none_without_wcs(self):
        _, ref_stars, _, _, fake_table, H, W = _build_scenario()
        with patch('src.net_query.gaia_cone_search', return_value=fake_table):
            field = fit_gaia_distortion_field({}, ref_stars, H, W)
        self.assertIsNone(field)

    def test_returns_none_below_min_matches(self):
        hdr, ref_stars, true_px, dist, fake_table, H, W = _build_scenario(M=GAIA_MIN_MATCHES - 1)
        with patch('src.net_query.gaia_cone_search', return_value=fake_table):
            field = fit_gaia_distortion_field(hdr, ref_stars, H, W)
        self.assertIsNone(field)

    def test_returns_none_on_gaia_query_failure(self):
        hdr, ref_stars, _, _, _, H, W = _build_scenario()
        with patch('src.net_query.gaia_cone_search', side_effect=RuntimeError("network down")):
            field = fit_gaia_distortion_field(hdr, ref_stars, H, W)
        self.assertIsNone(field)

    def test_returns_none_on_empty_gaia_table(self):
        hdr, ref_stars, _, _, _, H, W = _build_scenario()
        with patch('src.net_query.gaia_cone_search', return_value=None):
            field = fit_gaia_distortion_field(hdr, ref_stars, H, W)
        self.assertIsNone(field)

    def test_returns_none_with_no_ref_stars(self):
        hdr, _, _, _, fake_table, H, W = _build_scenario()
        with patch('src.net_query.gaia_cone_search', return_value=fake_table):
            field = fit_gaia_distortion_field(hdr, None, H, W)
        self.assertIsNone(field)


if __name__ == '__main__':
    unittest.main()
