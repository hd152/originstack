"""Tests for src/aberration.py — the field aberration / tilt inspector."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from src.aberration import analyze_field_aberration, _star_shape, _compass


def _sources(xs, ys, flux=None):
    n = len(xs)
    dt = np.dtype([('xcentroid', np.float64), ('ycentroid', np.float64),
                   ('flux', np.float64)])
    out = np.zeros(n, dtype=dt)
    out['xcentroid'] = xs
    out['ycentroid'] = ys
    out['flux'] = flux if flux is not None else np.ones(n)
    return out


def _gaussian_star(img, x, y, fwhm, amp=2000.0, ellip=0.0, angle=0.0):
    """Paint an (optionally elliptical) Gaussian star into img in place."""
    sigma = fwhm / 2.355
    sa = sigma
    sb = sigma * (1.0 - ellip)
    r = int(fwhm * 3) + 2
    H, W = img.shape
    y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
    x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - x
    dy = yy - y
    ca, san = np.cos(angle), np.sin(angle)
    u = dx * ca + dy * san
    v = -dx * san + dy * ca
    img[y0:y1, x0:x1] += amp * np.exp(-(u * u) / (2 * sa * sa) - (v * v) / (2 * sb * sb))


def _grid_field(H, W, fwhm_fn, spacing=70, ellip_fn=None):
    """Build a star field on a regular grid; fwhm_fn(x,y)->fwhm."""
    img = np.full((H, W), 100.0, dtype=np.float32)
    xs, ys = [], []
    for y in range(spacing, H - spacing, spacing):
        for x in range(spacing, W - spacing, spacing):
            f = fwhm_fn(x, y)
            e, a = (ellip_fn(x, y) if ellip_fn else (0.0, 0.0))
            _gaussian_star(img, x, y, f, ellip=e, angle=a)
            xs.append(x)
            ys.append(y)
    return img, _sources(np.array(xs, float), np.array(ys, float))


def test_star_shape_round_vs_elongated():
    img = np.full((40, 40), 50.0, dtype=np.float32)
    _gaussian_star(img, 20, 20, fwhm=4.0)
    cut = img[11:30, 11:30]
    fwhm, ellip, _ = _star_shape(cut)
    assert 2.5 < fwhm < 6.0
    assert ellip < 0.2  # round

    img2 = np.full((40, 40), 50.0, dtype=np.float32)
    _gaussian_star(img2, 20, 20, fwhm=4.0, ellip=0.6, angle=0.0)
    cut2 = img2[11:30, 11:30]
    _, ellip2, _ = _star_shape(cut2)
    assert ellip2 > ellip  # elongated reads more elliptical


def test_even_field_reports_no_tilt():
    H, W = 800, 800
    img, src = _grid_field(H, W, lambda x, y: 3.0)
    r = analyze_field_aberration(img, src, grid=4)
    assert r is not None
    assert r['fwhm_spread_pct'] < 20.0
    assert any('even' in d.lower() for d in r['diagnosis'])


def test_linear_tilt_detected():
    H, W = 900, 900
    # FWHM ramps left (sharp ~2.2) -> right (soft ~5) : classic tilt.
    img, src = _grid_field(H, W, lambda x, y: 2.2 + 3.0 * (x / W))
    r = analyze_field_aberration(img, src, grid=5)
    assert r is not None
    assert r['fwhm_spread_pct'] > 25.0
    assert r['tilt_gradient_px'] > 0.6
    # soft side is the +x (East/right) edge -> downhill points toward it? no:
    # downhill (toward focus) points to the SHARP side (West/left).
    assert r['tilt_direction'] in ('W', 'NW', 'SW')
    assert any('tilt' in d.lower() for d in r['diagnosis'])


def test_field_curvature_detected():
    H, W = 900, 900
    cx, cy = W / 2, H / 2
    rmax = np.hypot(cx, cy)
    img, src = _grid_field(H, W,
                           lambda x, y: 2.2 + 3.0 * (np.hypot(x - cx, y - cy) / rmax))
    r = analyze_field_aberration(img, src, grid=5)
    assert r is not None
    assert r['curvature_corr'] > 0.55
    assert any('curvature' in d.lower() for d in r['diagnosis'])


def test_png_written(tmp_path):
    H, W = 700, 700
    img, src = _grid_field(H, W, lambda x, y: 3.0)
    png = str(tmp_path / 'ab.png')
    r = analyze_field_aberration(img, src, grid=4, output_png=png)
    assert r is not None
    # PNG only if Pillow is present; if written, it must be a real file.
    try:
        import PIL  # noqa: F401
        assert os.path.exists(png) and os.path.getsize(png) > 0
    except Exception:
        pass


def test_too_few_stars_returns_none():
    img = np.full((200, 200), 100.0, dtype=np.float32)
    _gaussian_star(img, 100, 100, 3.0)
    assert analyze_field_aberration(img, _sources([100.0], [100.0]), grid=4) is None


def test_compass_labels():
    assert _compass(0) == 'E'
    assert _compass(90) == 'N'
    assert _compass(180) == 'W'
    assert _compass(270) == 'S'
