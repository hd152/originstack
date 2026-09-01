"""Tests for src/photometry_timeseries.py -- per-frame differential light curves.

Builds a short synthetic run: N registered frames of the same star field
with one injected variable star, a session info.json-style WCS, and a
mocked Gaia catalogue that projects onto the planted stars. Checks that
the variable star is flagged and a constant star is not, and that the
--photometry-target path reports stats.
"""
from __future__ import annotations

import csv
import math
import os
import types
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from src.photometry_timeseries import run_timeseries_photometry
from src.session_info import SessionInfo
from tests._photometry_helpers import FakeTable as _FakeTable


class _Args:
    verbose = False
    photometry_target = None
    photometry_gain = None


def _session_info(H, W, scale_arcsec=2.0):
    si = SessionInfo()
    si.image_width = W
    si.image_height = H
    si.ra_rad = math.radians(180.0)
    si.dec_rad = math.radians(0.0)
    si.fov_x_rad = math.radians(W * scale_arcsec / 3600.0)
    si.fov_y_rad = math.radians(H * scale_arcsec / 3600.0)
    si.orientation_rad = 0.0
    si.latitude = 40.0
    si.longitude = -105.0
    si.altitude = 1600.0
    si.date_time = "2026-03-15T05:00:00"
    si.total_duration_ms = 40 * 60 * 1000.0
    return si



@pytest.fixture
def synthetic_run(tmp_path):
    rng = np.random.default_rng(20)
    H = W = 300
    N = 14
    si = _session_info(H, W)

    from src.session_info import build_wcs_keywords
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = W
    hdr["NAXIS2"] = H
    for k, v in build_wcs_keywords(si).items():
        hdr[k] = v
    wcs = WCS(hdr)

    n_stars = 26
    margin = 30
    xs = rng.uniform(margin, W - margin, n_stars)
    ys = rng.uniform(margin, H - margin, n_stars)
    g_mag = rng.uniform(11.0, 13.5, n_stars)
    bp = g_mag + rng.uniform(0.2, 0.7, n_stars)
    rp = g_mag - rng.uniform(0.2, 0.7, n_stars)

    var_idx = 7                      # this star pulses +/- ~0.25 mag
    const_bright_idx = int(np.argmin(g_mag))
    base_flux = 10.0 ** (-0.4 * (g_mag - 20.0))

    yy, xx = np.mgrid[0:H, 0:W]
    frames = np.empty((N, H, W, 3), np.float32)
    for j in range(N):
        img = np.full((H, W, 3), 20.0, np.float64)
        amp = 1.0 + 0.25 * math.sin(2 * math.pi * j / N)
        for i in range(n_stars):
            f = base_flux[i] * (amp if i == var_idx else 1.0)
            for ci, m_off in enumerate((rp[i] - g_mag[i], 0.0, bp[i] - g_mag[i])):
                fc = f * 10.0 ** (-0.4 * m_off)
                img[..., ci] += fc * np.exp(
                    -(((xx - xs[i]) ** 2 + (yy - ys[i]) ** 2) / (2 * 1.8 ** 2)))
        img += rng.normal(0.0, 1.0, img.shape)
        frames[j] = np.clip(img, 0, None).astype(np.float32)

    ra, dec = wcs.all_pix2world(xs, ys, 0)
    catalog = _FakeTable({
        "source_id": np.arange(5000, 5000 + n_stars),
        "ra": ra, "dec": dec,
        "phot_g_mean_mag": g_mag,
        "phot_bp_mean_mag": bp,
        "phot_rp_mean_mag": rp,
    })

    final = []
    for j in range(N):
        f = types.SimpleNamespace()
        f.path = str(tmp_path / f"sub_{j:03d}.fits")
        t = 5.0 / 24.0 + (40.0 / 60.0 / 24.0) * (j / (N - 1))
        f.header = {"DATE-OBS": (
            "2026-03-15T" + _hms(t))}
        final.append(f)

    out_path = str(tmp_path / "stack.fits")
    stacked = frames.mean(axis=0)
    return dict(frames=frames, final=final, stacked=stacked, si=si,
               catalog=catalog, out_path=out_path, var_id=5000 + var_idx,
               const_id=5000 + const_bright_idx, N=N, H=H, W=W,
               var_xy=(xs[var_idx], ys[var_idx]))


def _hms(day_frac):
    secs = int(round(day_frac * 86400)) % 86400
    return f"{secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}"


def _run(sr, args=None):
    args = args or _Args()
    N = sr["N"]
    with patch("src.net_query.gaia_cone_search", return_value=sr["catalog"]):
        return run_timeseries_photometry(
            sr["final"], list(range(N)), sr["frames"],
            [(0.0, 0.0)] * N, [None] * N, None,
            (0, sr["H"], 0, sr["W"]), sr["stacked"], sr["si"], args,
            sr["out_path"])


def test_flags_variable_and_not_constant(synthetic_run):
    summary = _run(synthetic_run)
    assert summary is not None
    assert summary["n_frames"] == synthetic_run["N"]
    assert os.path.isfile(summary["stats_csv"])
    assert os.path.isfile(summary["lightcurves_csv"])

    with open(summary["stats_csv"], newline="") as fh:
        stats = {int(r["source_id"]): r for r in csv.DictReader(fh)}

    var_row = stats[synthetic_run["var_id"]]
    const_row = stats[synthetic_run["const_id"]]
    assert var_row["variable"] == "1"
    assert float(var_row["rms_g"]) > 0.05
    assert const_row["variable"] == "0"
    assert float(const_row["rms_g"]) < float(var_row["rms_g"])
    assert summary["n_variable_candidates"] >= 1

    # long-format light curve has N x n_stars rows
    with open(summary["lightcurves_csv"], newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == summary["n_frames"] * summary["n_stars"]


def test_target_selection_by_pixel(synthetic_run):
    args = _Args()
    vx, vy = synthetic_run["var_xy"]
    args.photometry_target = f"px:{vx:.1f},{vy:.1f}"
    summary = _run(synthetic_run, args)
    assert summary is not None
    assert summary["target"] is not None
    assert int(summary["target"]["source_id"]) == synthetic_run["var_id"]
    assert summary["target"]["n_points"] >= synthetic_run["N"] - 2


def test_needs_session_wcs(synthetic_run):
    sr = synthetic_run
    bare_si = SessionInfo()          # no WCS fields -> has_wcs is False
    N = sr["N"]
    with patch("src.net_query.gaia_cone_search", return_value=sr["catalog"]):
        out = run_timeseries_photometry(
            sr["final"], list(range(N)), sr["frames"],
            [(0.0, 0.0)] * N, [None] * N, None,
            (0, sr["H"], 0, sr["W"]), sr["stacked"], bare_si, _Args(),
            sr["out_path"])
    assert out is None
