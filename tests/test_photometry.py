"""Tests for src/photometry.py -- absolute aperture photometry (--photometry).

Builds a synthetic RGB star field with a known TAN WCS, mocks the Gaia
cone search with a catalogue whose RA/Dec project exactly onto the planted
stars, and checks that run_photometry recovers per-channel magnitudes that
match the catalogue (the zero point + aperture-fraction offset is common to
every star, so the *calibrated* magnitudes are the meaningful check) and
writes a well-formed CSV.
"""
from __future__ import annotations

import csv
import os
from unittest.mock import patch

import numpy as np
import pytest

from astropy.io import fits
from astropy.wcs import WCS

from src.photometry import run_photometry


class _Args:
    photometry_extinction_k = None
    photometry_color_terms = False
    photometry_gain = None
    verbose = False


def _make_wcs_header(H, W, scale_arcsec=2.0):
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = W
    hdr["NAXIS2"] = H
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = W / 2.0
    hdr["CRPIX2"] = H / 2.0
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 0.0
    s = scale_arcsec / 3600.0
    hdr["CD1_1"] = -s
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = s
    return hdr


def _add_gaussian(plane, x, y, total_flux, sigma=1.6):
    H, W = plane.shape
    r = int(np.ceil(4 * sigma))
    x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    g *= total_flux / (2 * np.pi * sigma ** 2)
    plane[y0:y1, x0:x1] += g


class _FakeColumn(np.ndarray):
    pass


class _FakeTable:
    """Minimal astropy-Table stand-in: table['col'] -> ndarray, len()."""

    def __init__(self, cols):
        self._cols = {k: np.asarray(v) for k, v in cols.items()}
        self.colnames = list(cols.keys())

    def __getitem__(self, key):
        return self._cols[key]

    def __len__(self):
        return len(next(iter(self._cols.values())))


@pytest.fixture
def synthetic_field(tmp_path):
    rng = np.random.default_rng(1234)
    H = W = 320
    hdr = _make_wcs_header(H, W)
    wcs = WCS(hdr)

    n_stars = 40
    margin = 24
    xs = rng.uniform(margin, W - margin, n_stars)
    ys = rng.uniform(margin, H - margin, n_stars)
    # Keep every star high-SNR: the photometric-scatter assertion below is a
    # statement about the method, not about shot noise on faint sources.
    g_mag = rng.uniform(10.5, 13.0, n_stars)
    bp_mag = g_mag + rng.uniform(0.1, 0.6, n_stars)
    rp_mag = g_mag - rng.uniform(0.1, 0.6, n_stars)

    ZP_TRUE = 20.0  # instrumental: flux = 10**(-0.4*(mag - ZP_TRUE))
    img = np.full((H, W, 3), 12.0, dtype=np.float64)
    for i in range(n_stars):
        for ci, m in enumerate((rp_mag[i], g_mag[i], bp_mag[i])):
            flux = 10.0 ** (-0.4 * (m - ZP_TRUE))
            _add_gaussian(img[..., ci], xs[i], ys[i], flux)
    img += rng.normal(0.0, 1.0, img.shape)
    img = np.clip(img, 0.0, None)

    ra, dec = wcs.all_pix2world(xs, ys, 0)
    catalog = _FakeTable({
        "source_id": np.arange(1000, 1000 + n_stars),
        "ra": ra, "dec": dec,
        "phot_g_mean_mag": g_mag,
        "phot_bp_mean_mag": bp_mag,
        "phot_rp_mean_mag": rp_mag,
    })

    out_path = str(tmp_path / "stack.fits")
    return img, hdr, catalog, out_path, dict(g=g_mag, bp=bp_mag, rp=rp_mag)


def test_run_photometry_recovers_catalog_magnitudes(synthetic_field):
    img, hdr, catalog, out_path, truth = synthetic_field

    with patch("src.net_query.gaia_cone_search", return_value=catalog):
        summary = run_photometry(img, hdr, _Args(), None, out_path)

    assert summary is not None
    assert summary["n_matched"] >= 20
    # No session GPS/time -> extinction folded into the zero point.
    assert summary["airmass"] is None
    for ch in ("R", "G", "B"):
        assert ch in summary["zeropoints"]
        assert summary["zeropoints"][ch]["zp_err"] < 0.05

    csv_path = summary["csv_path"]
    assert csv_path and os.path.isfile(csv_path)
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == summary["n_matched"]

    # Calibrated magnitudes must track the catalogue (common ZP + aperture
    # fraction cancels in the comparison against the same catalogue).
    dg = [float(r["mag_g"]) - float(r["gaia_g"]) for r in rows if r["mag_g"]]
    dr = [float(r["mag_r"]) - float(r["gaia_rp"]) for r in rows if r["mag_r"]]
    db = [float(r["mag_b"]) - float(r["gaia_bp"]) for r in rows if r["mag_b"]]
    for d in (dg, dr, db):
        d = np.asarray(d)
        assert len(d) >= 15
        # The mean offset is the (channel-independent) aperture fraction;
        # scatter about it is what photometric quality means here.
        assert np.std(d - np.median(d)) < 0.05


def test_color_term_fit_recovers_injected_slope(tmp_path):
    """A colour-dependent instrumental offset is recovered by
    --photometry-color-terms and left as scatter without it."""
    rng = np.random.default_rng(99)
    H = W = 320
    hdr = _make_wcs_header(H, W)
    wcs = WCS(hdr)

    n = 45
    xs = rng.uniform(24, W - 24, n)
    ys = rng.uniform(24, H - 24, n)
    g_mag = rng.uniform(10.5, 12.5, n)
    bp = g_mag + rng.uniform(0.0, 1.4, n)      # wide colour spread
    rp = g_mag - rng.uniform(0.0, 1.4, n)
    bp_rp = bp - rp
    ref = float(np.median(bp_rp))
    ZP_TRUE = 20.0
    CT_TRUE = {0: -0.10, 1: 0.06, 2: 0.14}     # mag per mag BP-RP, per channel

    img = np.full((H, W, 3), 12.0, dtype=np.float64)
    for i in range(n):
        for ci, cat_m in enumerate((rp[i], g_mag[i], bp[i])):
            m_eff = cat_m - ZP_TRUE - CT_TRUE[ci] * (bp_rp[i] - ref)
            flux = 10.0 ** (-0.4 * m_eff)
            _add_gaussian(img[..., ci], xs[i], ys[i], flux)
    img += rng.normal(0.0, 1.0, img.shape)
    img = np.clip(img, 0.0, None)

    ra, dec = wcs.all_pix2world(xs, ys, 0)
    catalog = _FakeTable({
        "source_id": np.arange(1, n + 1), "ra": ra, "dec": dec,
        "phot_g_mean_mag": g_mag, "phot_bp_mean_mag": bp,
        "phot_rp_mean_mag": rp,
    })
    out = str(tmp_path / "s.fits")

    args_ct = _Args()
    args_ct.photometry_color_terms = True
    with patch("src.net_query.gaia_cone_search", return_value=catalog):
        s_ct = run_photometry(img, hdr, args_ct, None, out)
        s_plain = run_photometry(img, hdr, _Args(), None, out)

    assert s_ct is not None and s_plain is not None
    assert s_ct["color_terms_fitted"] and not s_plain["color_terms_fitted"]
    for ch, ci in (("R", 0), ("G", 1), ("B", 2)):
        assert s_ct["zeropoints"][ch]["ct"] == pytest.approx(CT_TRUE[ci], abs=0.02)
        # colour term absorbed -> tighter zero-point residual
        assert s_ct["zeropoints"][ch]["zp_err"] < s_plain["zeropoints"][ch]["zp_err"]


def test_run_photometry_needs_wcs(synthetic_field):
    img, hdr, catalog, out_path, _ = synthetic_field
    bare = fits.Header()
    with patch("src.net_query.gaia_cone_search", return_value=catalog):
        assert run_photometry(img, bare, _Args(), None, out_path) is None


def test_run_photometry_rejects_mono(synthetic_field):
    img, hdr, catalog, out_path, _ = synthetic_field
    assert run_photometry(img[..., 0], hdr, _Args(), None, out_path) is None


class _SI:
    has_gps = True
    latitude = 40.0
    longitude = -105.0
    altitude = 1600.0
    date_time = None


def test_airmass_wiring_over_a_day():
    """_airmass returns a sane value while the target is up, None otherwise."""
    from src.photometry import _airmass

    hits = 0
    for h in range(0, 24, 2):
        hdr = fits.Header()
        hdr["DATE-OBS"] = f"2026-03-15T{h:02d}:00:00"
        X = _airmass(hdr, _SI(), 180.0, 0.0)
        if X is not None:
            hits += 1
            assert X >= 1.0
    assert hits >= 3  # RA 180/Dec 0 is above the horizon much of the day


def test_extinction_term_applied_with_gps(synthetic_field):
    img, hdr, catalog, out_path, _ = synthetic_field
    from src.photometry import _airmass

    good_time = None
    for h in range(0, 24):
        probe = fits.Header()
        probe["DATE-OBS"] = f"2026-03-15T{h:02d}:00:00"
        if _airmass(probe, _SI(), 180.0, 0.0) is not None:
            good_time = probe["DATE-OBS"]
            break
    assert good_time is not None
    hdr["DATE-OBS"] = good_time

    with patch("src.net_query.gaia_cone_search", return_value=catalog):
        summary = run_photometry(img, hdr, _Args(), _SI(), out_path)

    assert summary is not None
    assert summary["airmass"] is not None and summary["airmass"] >= 1.0
    assert summary["extinction_k"]["G"] == pytest.approx(0.15)
