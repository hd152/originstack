"""The survey H-alpha reference for DBE (src/halpha_reference.py): HEALPix lookup, the scale
fit and its three ways of declining, and the Phase 4 hook."""
import argparse

import numpy as np
import pytest
from astropy.io import fits

from src import halpha_reference as hr

NSIDE = 64


def test_ang2pix_nest_base_faces():
    """nside 1: the pixel is the base face (Gorski et al. 2005, fig. 4)."""
    theta = np.array([0.2, 0.2, np.pi / 2, np.pi / 2, np.pi - 0.2, np.pi - 0.2])
    phi = np.array([np.pi / 4, 3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4, 7 * np.pi / 4])
    assert hr.ang2pix_nest(1, theta, phi).tolist() == [0, 1, 4, 5, 8, 11]


def test_ang2pix_nest_equal_area_and_nesting():
    """Every pixel is hit about equally often by uniform points, and a pixel's four
    children at twice the resolution are ``4 p .. 4 p + 3`` (the NESTED property)."""
    rng = np.random.default_rng(1)
    z = rng.uniform(-1, 1, 400_000)
    theta, phi = np.arccos(z), rng.uniform(0, 2 * np.pi, z.size)
    p8 = hr.ang2pix_nest(8, theta, phi)
    counts = np.bincount(p8, minlength=12 * 64)
    assert counts.size == 12 * 64 and counts.min() > 0
    assert counts.std() / counts.mean() < 0.08          # Poisson: 1/sqrt(520) = 0.044
    assert np.array_equal(hr.ang2pix_nest(16, theta, phi) // 4, p8)


def test_icrs_to_galactic_matches_astropy():
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.default_rng(2)
    ra, dec = rng.uniform(0, 360, 200), np.degrees(np.arcsin(rng.uniform(-1, 1, 200)))
    lon, lat = hr.icrs_to_galactic(ra, dec)
    g = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs').galactic
    assert np.max(np.abs(lat - g.b.rad)) < 1e-7
    assert np.max(np.abs(np.angle(np.exp(1j * (lon - g.l.rad))))) < 1e-6


def _write_map(path, value_of_lb):
    """A NESTED HEALPix table whose pixels hold ``value_of_lb(l, b)`` (mean over the points
    that fall in each)."""
    rng = np.random.default_rng(3)
    z = rng.uniform(-1, 1, 2_000_000)
    lon = rng.uniform(0, 2 * np.pi, z.size)
    pix = hr.ang2pix_nest(NSIDE, np.arccos(z), lon)
    n = 12 * NSIDE * NSIDE
    s = np.bincount(pix, weights=value_of_lb(lon, np.arcsin(z)), minlength=n)
    c = np.bincount(pix, minlength=n)
    m = (s / np.maximum(c, 1)).astype(np.float32)
    col = fits.Column(name='TEMPERATURE', format='E', array=m)
    hdu = fits.BinTableHDU.from_columns([col])
    hdu.header['ORDERING'] = 'NESTED'
    hdu.header['NSIDE'] = NSIDE
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    return m


def test_load_map_and_lookup(tmp_path, monkeypatch):
    p = str(tmp_path / 'map.fits')
    _write_map(p, lambda lon, lat: 100.0 * np.cos(lat) * (1.5 + np.cos(lon)))
    monkeypatch.setenv('ORIGINSTACK_HALPHA_MAP', p)
    hr._map_cache.clear()
    m = hr.load_map(download=False)
    assert m is not None and m.size == 12 * NSIDE * NSIDE
    ra, dec = np.array([10.0, 83.8, 250.0]), np.array([41.0, -5.4, -30.0])
    lon, lat = hr.icrs_to_galactic(ra, dec)
    want = 100.0 * np.cos(lat) * (1.5 + np.cos(lon))
    assert np.allclose(hr.halpha_at(m, ra, dec), want, atol=4.0)      # one pixel is ~0.9 deg


def test_missing_map_declines_without_network(tmp_path, monkeypatch):
    """No map on disk: the download is attempted through net_query (blocked in tests) and
    the reference is declined, not raised."""
    monkeypatch.setenv('ORIGINSTACK_HALPHA_MAP', str(tmp_path / 'absent.fits'))
    hr._map_cache.clear()
    rgb = np.zeros((512, 640, 3), np.float32)
    model, note = hr.reference_model(rgb, fits.Header())
    assert model is None and 'map' in note


def _scene(seed=0, H=1024, W=1536):
    """(nebula shape in Rayleigh, gradient) on an H x W frame: a blob the quadratic cannot
    follow, on a tilted sky."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    neb = 300.0 * np.exp(-(((xx - 0.3 * W) / 260.0) ** 2 + ((yy - 0.6 * H) / 200.0) ** 2))
    neb += 150.0 * np.exp(-(((xx - 0.75 * W) / 180.0) ** 2 + ((yy - 0.3 * H) / 150.0) ** 2))
    grad = 400.0 + 0.2 * xx - 0.15 * yy + 1e-4 * (xx - W / 2) ** 2
    return neb.astype(np.float32), grad.astype(np.float32)


def test_fit_scales_recovers_the_scale():
    neb, grad = _scene()
    rng = np.random.default_rng(4)
    true = np.array([0.8, 0.35, 0.3], np.float32)
    rgb = neb[:, :, None] * true + grad[:, :, None] + rng.normal(0, 20.0, neb.shape + (3,)).astype(np.float32)
    scales, note = hr.fit_scales(rgb, neb)
    assert scales is not None, note
    assert np.allclose(scales, true, rtol=0.08)


def test_fit_scales_declines():
    neb, grad = _scene()
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 20.0, neb.shape + (3,)).astype(np.float32)
    # no emission in the field
    s, note = hr.fit_scales(grad[:, :, None] + noise, neb * 0.01)
    assert s is None and 'no H-alpha structure' in note
    # the map is itself a smooth gradient over the field: scale not measurable
    s, note = hr.fit_scales(grad[:, :, None] * np.ones(3, np.float32) + noise, grad)
    assert s is None and 'smooth gradient' in note
    # the image does not contain the map's shape
    s, note = hr.fit_scales(grad[:, :, None] + noise, neb)
    assert s is None and 'does not follow' in note


def test_phase4_hook_never_raises():
    from src.postprocess import _halpha_reference_model
    rgb = np.zeros((256, 256, 3), np.float32)
    args = argparse.Namespace(_stack_header=None, _stack_wcs=None)
    assert _halpha_reference_model(rgb, args) is None


def test_setting_is_off_by_default_and_quiet_in_monte_carlo():
    from src import cli
    from src.uncertainty import _QUIET_OFF
    args = cli.parse_args(['-d', 'x', '-o', 'y.fits'])
    assert args.dbe_halpha_reference is False
    assert 'dbe_halpha_reference' in _QUIET_OFF


@pytest.mark.parametrize('shape', [(64, 64, 3)])
def test_small_frame_declines(shape):
    s, note = hr.fit_scales(np.zeros(shape, np.float32), np.zeros(shape[:2], np.float32))
    assert s is None
