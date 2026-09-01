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

from src.photometry import _fit_zeropoint_colorterm, run_photometry
from tests._photometry_helpers import FakeTable as _FakeTable
from tests._photometry_helpers import add_gaussian as _add_gaussian
from tests._photometry_helpers import make_wcs_header as _make_wcs_header


class _Args:
    photometry_extinction_k = None
    photometry_color_terms = False
    photometry_gain = None
    verbose = False


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


# ---------------------------------------------------------------------------
# Review follow-ups: colour-term fit flag + saturation heuristic
# ---------------------------------------------------------------------------

def test_fit_zeropoint_colorterm_flags_when_slope_not_fitted():
    rng = np.random.default_rng(5)
    color = rng.uniform(0.4, 2.2, 12)
    resid = 19.0 + 0.08 * (color - color.mean()) + rng.normal(0, 0.01, 12)

    zp, zp_err, ct, n, fitted = _fit_zeropoint_colorterm(
        resid, color, float(np.median(color)), fit_ct=True)
    assert fitted is True
    assert ct == pytest.approx(0.08, abs=0.03)

    # Only 6 stars -> not enough to fit a slope; falls back to a median.
    zp, zp_err, ct, n, fitted = _fit_zeropoint_colorterm(
        resid[:6], color[:6], float(np.median(color[:6])), fit_ct=True)
    assert fitted is False
    assert ct == 0.0

    # fit_ct=False is always a median, never "fitted".
    _, _, ct, _, fitted = _fit_zeropoint_colorterm(
        resid, color, float(np.median(color)), fit_ct=False)
    assert fitted is False and ct == 0.0


def test_bright_unsaturated_cluster_not_flagged(tmp_path):
    """A tight cluster of bright (but non-clipping) stars must not be flagged
    saturated when the header carries no SATURATE level."""
    rng = np.random.default_rng(11)
    H = W = 320
    hdr = _make_wcs_header(H, W)
    wcs = WCS(hdr)
    n = 30
    xs = rng.uniform(24, W - 24, n)
    ys = rng.uniform(24, H - 24, n)
    # 8 stars within ~0.15 mag of each other near the top, rest fainter.
    g = np.concatenate([rng.uniform(10.40, 10.55, 8), rng.uniform(12.0, 13.5, n - 8)])
    bp = g + 0.3
    rp = g - 0.3
    img = np.full((H, W, 3), 12.0)
    for i in range(n):
        for ci, m in enumerate((rp[i], g[i], bp[i])):
            _add_gaussian(img[..., ci], xs[i], ys[i], 10.0 ** (-0.4 * (m - 20.0)))
    img += rng.normal(0, 1, img.shape)
    img = np.clip(img, 0, None)
    ra, dec = wcs.all_pix2world(xs, ys, 0)
    catalog = _FakeTable({"source_id": np.arange(n), "ra": ra, "dec": dec,
                          "phot_g_mean_mag": g, "phot_bp_mean_mag": bp,
                          "phot_rp_mean_mag": rp})
    with patch("src.net_query.gaia_cone_search", return_value=catalog):
        s = run_photometry(img, hdr, _Args(), None, str(tmp_path / "s.fits"))
    assert s is not None
    with open(s["csv_path"], newline="") as fh:
        rows = list(csv.DictReader(fh))
    n_sat = sum(int(r["saturated"]) for r in rows)
    # At most the single literal-max-pixel star; the bright cluster is fine.
    assert n_sat <= 1
