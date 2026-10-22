"""AAVSO Extended Format report (src/aavso_report.py) from a synthetic light-curve CSV."""
import csv
import math

import numpy as np
import pytest

pytest.importorskip("astropy")


def _wcs_header():
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [100.0, 20.0]
    w.wcs.crpix = [501.0, 501.0]
    w.wcs.cdelt = [-0.0004, 0.0004]
    return w.to_header(), w


def _sexa(ra, dec):
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra, dec, unit="deg")
    return (c.ra.to_string(unit="hourangle", sep=":", precision=3),
            c.dec.to_string(sep=":", precision=2, alwayssign=True))


# comparison stars (x, y, V); the target at (500, 500)
_COMPS = [(200.0, 300.0, 11.0), (700.0, 650.0, 12.0), (400.0, 800.0, 12.5), (820.0, 180.0, 13.0)]
_TARGET_XY = (500.0, 500.0)
ZP_TRUE = 21.3        # V - instrumental of this camera


def _write_lc(path, n=12, target_mag=12.4, noise=0.0, seed=0):
    """Light curves as photometry_timeseries writes them: per frame a different
    (Gaia ensemble) offset that the VSP re-zeroing must remove."""
    rng = np.random.default_rng(seed)
    stars = [("T", *_TARGET_XY, target_mag)] + [(f"C{i}", x, y, v) for i, (x, y, v) in enumerate(_COMPS)]
    fields = ["frame", "filename", "mjd", "airmass", "source_id", "x", "y", "gaia_g", "is_target",
              "mag_r", "magerr_r", "mag_g", "magerr_g", "mag_b", "magerr_b", "flag"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for j in range(n):
            frame_off = 0.3 * math.sin(j)               # transparency / Gaia-zp wobble per frame
            for sid, x, y, v in stars:
                m = v - ZP_TRUE + frame_off + noise * rng.standard_normal()
                w.writerow({"frame": j, "filename": f"f{j}", "mjd": 60000.0 + j / 1440.0, "airmass": 1.2,
                            "source_id": sid, "x": x, "y": y, "gaia_g": v, "is_target": int(sid == "T"),
                            "mag_r": "", "magerr_r": "", "mag_g": round(m, 5), "magerr_g": 0.01,
                            "mag_b": "", "magerr_b": "", "flag": ""})


def _vsp(w):
    phot = []
    for i, (x, y, v) in enumerate(_COMPS):
        ra, dec = (float(t) for t in w.all_pix2world(x, y, 0))
        sra, sdec = _sexa(ra, dec)
        phot.append({"auid": f"000-AAA-00{i}", "label": str(int(v * 10)), "ra": sra, "dec": sdec,
                     "bands": [{"band": "V", "mag": v, "error": 0.02}, {"band": "B", "mag": v + 0.6, "error": 0.02}]})
    tra, tdec = (float(t) for t in w.all_pix2world(*_TARGET_XY, 0))
    sra, sdec = _sexa(tra, tdec)
    return {"chartid": "X12345ABC", "star": "ZZ Tst", "auid": "000-ZZZ-999", "ra": sra, "dec": sdec,
            "photometry": phot}


def test_report_recovers_target_on_v_scale(tmp_path):
    from src.aavso_report import build_report, parse_vsp
    hdr, w = _wcs_header()
    lc = str(tmp_path / "x_lightcurves.csv")
    _write_lc(lc)
    s = build_report(lc, hdr, parse_vsp(_vsp(w)), obscode="ABC", exptime_s=30.0)
    assert "error" not in s
    rows = [r.split(",") for r in s["text"].splitlines() if not r.startswith("#")]
    assert len(rows) == 12
    mags = np.array([float(r[2]) for r in rows])
    np.testing.assert_allclose(mags, 12.4, atol=1e-3)          # per-frame offsets removed
    # check star: the comp closest in brightness to the target (V 12.5), reproduced
    assert s["check"]["v"] == 12.5 and abs(s["check"]["measured"] - 12.5) < 1e-3
    r = rows[0]
    assert r[0] == "ZZ Tst" and r[4] == "TG" and r[5] == "NO" and r[6] == "STD"
    assert r[7] == "ENSEMBLE" and r[8] == "na" and r[13] == "X12345ABC"
    # mid-exposure JD: MJD + 2400000.5 + 15 s
    assert float(r[1]) == pytest.approx(60000.0 + 2400000.5 + 15.0 / 86400.0, abs=1e-5)
    head = [h for h in s["text"].splitlines() if h.startswith("#")]
    assert head[0] == "#TYPE=EXTENDED" and "#OBSCODE=ABC" in head and "#DELIM=," in head and "#DATE=JD" in head
    assert all(len(r) == 15 for r in rows)                       # NOTES carries no extra delimiter


def test_binning_and_missing_obscode(tmp_path):
    from src.aavso_report import build_report, parse_vsp
    hdr, w = _wcs_header()
    lc = str(tmp_path / "x_lightcurves.csv")
    _write_lc(lc, n=12, noise=0.02)
    s = build_report(lc, hdr, parse_vsp(_vsp(w)), exptime_s=30.0, bin_minutes=4.0)
    rows = [r.split(",") for r in s["text"].splitlines() if not r.startswith("#")]
    assert len(rows) == 3                                        # 12 one-minute frames in 4-minute bins
    assert all(abs(float(r[2]) - 12.4) < 0.05 for r in rows)
    assert "#OBSCODE=CHANGEME" in s["text"] and s["obscode_missing"]


def test_target_not_measured_is_an_error(tmp_path):
    from src.aavso_report import build_report, parse_vsp
    hdr, w = _wcs_header()
    lc = str(tmp_path / "x_lightcurves.csv")
    _write_lc(lc)
    s = build_report(lc, hdr, parse_vsp(_vsp(w)), target_ra=float(w.all_pix2world(50, 50, 0)[0]),
                     target_dec=float(w.all_pix2world(50, 50, 0)[1]))
    assert "error" in s and "not among the measured stars" in s["error"]


def test_vsp_query_is_offline_guarded(monkeypatch):
    from src import net_query
    net_query.set_offline(True)
    try:
        with pytest.raises(net_query.OfflineError):
            net_query.aavso_vsp_chart(star="SS Cyg")
    finally:
        net_query.set_offline(False)
