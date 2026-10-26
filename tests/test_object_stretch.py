"""--stretch object: white point and symmetry point from the target itself."""
from types import SimpleNamespace

import numpy as np
import pytest

from src import io_fits
from src.io_fits import _object_region, object_anchor, render_preview_float

H, W = 600, 800


def _field(rng, blobs=(), sky=100.0, noise=2.0, n_stars=60):
    """RGB field: sky + noise + point stars + elliptical exponential 'galaxies'
    given as (cy, cx, scale_px, peak)."""
    yy, xx = np.mgrid[:H, :W].astype(np.float64)
    lum = np.full((H, W), sky)
    for cy, cx, sc, pk in blobs:
        r = np.hypot((yy - cy) / sc, (xx - cx) / (1.4 * sc))
        lum += pk * np.exp(-r)
    for _ in range(n_stars):
        y, x, f = rng.uniform(5, H - 5), rng.uniform(5, W - 5), rng.uniform(50, 3000)
        lum += f * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.3 ** 2))
    rgb = lum[..., None] * np.array([1.0, 0.95, 0.9]) + rng.normal(0, noise, (H, W, 3))
    return rgb.astype(np.float32)


def test_small_galaxy_is_found_and_owns_the_curve():
    rng = np.random.default_rng(0)
    rgb = _field(rng, blobs=[(300, 400, 18, 60.0)])
    lum = rgb.mean(-1)
    reg = _object_region(lum, sky=100.0 * 0.95)
    assert reg is not None
    core, faint, frac = reg
    assert 0.0005 < frac < 0.3 and core > faint > 95.0
    ghs = render_preview_float(rgb, stretch='ghs', black_sigma=1.0)
    obj = render_preview_float(rgb, stretch='object', black_sigma=1.0)
    # the galaxy's outer disk gets more of the display range than under plain ghs
    ring = (slice(300 - 4, 300 + 5), slice(400 + 60, 400 + 70))
    assert obj[ring].mean() > ghs[ring].mean() + 0.02
    # the empty sky stays dark
    sky = (slice(20, 60), slice(20, 60))
    assert obj[sky].mean() < 0.25


@pytest.mark.parametrize('blobs', [[], [(300, 400, 400, 30.0)]], ids=['empty sky', 'frame-filling'])
def test_no_separate_object_is_exactly_ghs(blobs):
    rng = np.random.default_rng(1)
    rgb = _field(rng, blobs=blobs)
    assert _object_region(rgb.mean(-1), sky=float(np.median(rgb.mean(-1)))) is None
    np.testing.assert_array_equal(render_preview_float(rgb, stretch='object'),
                                  render_preview_float(rgb, stretch='ghs'))


def test_anchor_picks_the_object_the_telescope_pointed_at():
    rng = np.random.default_rng(2)
    rgb = _field(rng, blobs=[(150, 200, 12, 80.0), (420, 620, 12, 20.0)])
    lum = rgb.mean(-1)
    bright = _object_region(lum, sky=95.0, anchor=(150, 200))
    faint = _object_region(lum, sky=95.0, anchor=(420, 620))
    assert bright is not None and faint is not None
    assert bright[0] > faint[0] + 20.0


def test_object_anchor_maps_the_session_pointing_through_the_stack_wcs():
    cards = {'CTYPE1': ('RA---TAN', ''), 'CTYPE2': ('DEC--TAN', ''),
             'CRVAL1': (180.0, ''), 'CRVAL2': (30.0, ''),
             'CRPIX1': (401.0, ''), 'CRPIX2': (301.0, ''),
             'CD1_1': (-4e-4, ''), 'CD1_2': (0.0, ''), 'CD2_1': (0.0, ''), 'CD2_2': (4e-4, '')}
    si = SimpleNamespace(ra_rad=np.radians(180.0), dec_rad=np.radians(30.0 + 50 * 4e-4))
    y, x = object_anchor(SimpleNamespace(_session_info=si, _stack_wcs=cards))
    assert y == pytest.approx(350.0, abs=0.5) and x == pytest.approx(400.0, abs=0.5)
    assert object_anchor(SimpleNamespace()) is None


def test_preview_kw_carries_the_anchor_to_every_preview(monkeypatch):
    monkeypatch.setattr(io_fits, 'object_anchor', lambda args: (12.0, 34.0))
    kw = io_fits.preview_stretch_kw(SimpleNamespace(stretch='object'))
    assert kw['stretch'] == 'object' and kw['anchor'] == (12.0, 34.0)
