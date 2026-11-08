"""Colour-calibration scale clamps are reported with the values the fit wanted."""
import numpy as np

import src.photometric_calibration as pc
from src import color_calibrate as cc


def test_clamp_scales_names_wanted_values():
    clamped, note = pc.clamp_scales((0.33, 1.0, 0.34))
    assert list(clamped) == [0.5, 1.0, 0.5]
    assert "R x0.330" in note and "B x0.340" in note and "G x" not in note
    clamped, note = pc.clamp_scales((0.7, 1.0, 1.9))
    assert note is None and list(clamped) == [0.7, 1.0, 1.9]


def test_gray_locus_reports_clamp(monkeypatch):
    rng = np.random.default_rng(1)
    g = rng.uniform(1e3, 1e4, 40)
    fluxes = np.stack([3.0 * g, g, 2.9 * g], axis=1) * rng.normal(1, 0.01, (40, 1))
    monkeypatch.setattr(pc, '_aperture_photometry', lambda *a, **k: fluxes)
    img = np.ones((8, 8, 3), np.float32)
    info = {}
    _, scales = pc.photometric_color_calibrate(img, object(), info=info)
    assert scales[0] == 0.5 and scales[2] == 0.5
    assert "R x0.3" in info["clamp_note"] and "B x0.3" in info['clamp_note']
    # in range: no note
    monkeypatch.setattr(pc, '_aperture_photometry',
                        lambda *a, **k: np.stack([0.8 * g, g, 0.9 * g], axis=1))
    info = {}
    pc.photometric_color_calibrate(img, object(), info=info)
    assert info['clamp_note'] is None


def test_fit_clamp_is_printed(capsys):
    assert cc._clamp_report(0.33, 1.0, 0.34) == (0.5, 1.0, 0.5)
    out = capsys.readouterr().out
    assert "wanted R x0.330" in out
    assert cc._clamp_report(0.8, 1.0, 0.9) == (0.8, 1.0, 0.9)
    assert capsys.readouterr().out == ""
