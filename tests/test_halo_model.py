"""Gaia-anchored extended-PSF halo subtraction (src/halo_model.py) on a synthetic field.

Stars with a Gaussian core and an r^-2.5 wing, fluxes from a known magnitude relation, the
bright ones clipped; a mocked Gaia cone search; a TAN WCS at 1 arcsec/pixel.
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from src import halo_model, net_query

H = W = 900
SIGMA = 3.0
A = 9.0            # log10 flux at G = 0
KAPPA = 1.5e-2       # wing amplitude per unit flux
CLIP = 2500.0
RA0, DEC0 = 150.0, 20.0


def _wcs_header():
    w = WCS(naxis=2)
    w.wcs.crpix = [W / 2 + 0.5, H / 2 + 0.5]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    hdr = w.to_header()
    hdr['NAXIS'] = 2
    hdr['NAXIS1'], hdr['NAXIS2'] = W, H
    return hdr, w


def _psf(r2, flux):
    core = flux / (2 * np.pi * 1.8 ** 2) * np.exp(-r2 / (2 * 1.8 ** 2))
    wing = flux * KAPPA * (1 + r2 / 16.0) ** -1.25
    return core + wing


@pytest.fixture(scope='module')
def field():
    rng = np.random.default_rng(3)
    hdr, w = _wcs_header()
    yy, xx = np.mgrid[:H, :W].astype(np.float64)
    stars = []
    for gy, gx in [(150, 150), (150, 450), (150, 750), (450, 150), (450, 750), (750, 150),
                   (750, 450), (750, 750)]:
        stars.append((gy, gx, rng.uniform(6.5, 8.5)))
    stars.append((450, 450, 7.0))                       # the star the nebula sits next to
    for _ in range(30):
        stars.append((rng.uniform(30, H - 30), rng.uniform(30, W - 30), rng.uniform(11.8, 14.0)))
    img = np.full((H, W), 500.0)
    truth_wing = np.zeros((H, W))
    for y, x, g in stars:
        f = 10.0 ** (A - 0.4 * g)
        r2 = (yy - y) ** 2 + (xx - x) ** 2
        img += _psf(r2, f)
        truth_wing += f * KAPPA * (1 + r2 / 16.0) ** -1.25
    blob = 40.0 * np.exp(-((yy - 450) ** 2 + (xx - 520) ** 2) / (2 * 25.0 ** 2))   # 13 sigma, 70 px from star
    img += blob
    img = np.minimum(img + rng.normal(0, SIGMA, img.shape), CLIP)
    stack = np.stack([img, img * 0.8, img * 0.6], axis=2).astype(np.float32)
    ra, dec = w.all_pix2world([s[1] for s in stars], [s[0] for s in stars], 0)
    cat = Table({'ra': ra, 'dec': dec, 'phot_g_mean_mag': [s[2] for s in stars],
                 'phot_bp_mean_mag': [s[2] + 0.5 for s in stars],
                 'phot_rp_mean_mag': [s[2] - 0.3 for s in stars]})
    return stack, hdr, cat, truth_wing, blob


def _excess(lum, cy, cx, r):
    yy, xx = np.mgrid[:H, :W]
    d = np.hypot(yy - cy, xx - cx)
    ring = (d >= r) & (d < r + 4)
    sky = (d >= 120) & (d < 140)
    return float(np.median(lum[ring]) - np.median(lum[sky]))


def test_flux_scale_recovers_the_magnitude_relation():
    g = np.linspace(9, 14, 40)
    colour = np.linspace(0.3, 1.6, 40)
    flux = 10.0 ** (7.9 - 0.4 * g + 0.2 * (colour - 0.9))
    a, b, scatter = halo_model.fit_flux_scale(flux, g, colour)
    assert abs(a - (7.9 - 0.2 * 0.9)) < 0.02 and abs(b - 0.2) < 0.02 and scatter < 0.01
    assert halo_model.fit_flux_scale(flux[:4], g[:4], colour[:4]) is None
    pred = halo_model.predicted_flux(a, b, np.array([10.0]), np.array([np.nan]))
    assert pred[0] > 0


def test_wings_are_removed_and_the_core_and_nebula_are_not(field, monkeypatch):
    stack, hdr, cat, wing, blob = field
    monkeypatch.setattr(net_query, 'gaia_cone_search', lambda *a, **k: cat)
    out, info = halo_model.subtract_halos(stack, hdr, fwhm=4.2)
    assert info is not None and 'declined' not in info
    lum0, lum1 = stack[:, :, 1].astype(np.float64), out[:, :, 1].astype(np.float64)
    for cy, cx in [(150, 150), (750, 750), (150, 750)]:
        for r in (30, 50):
            before, after = _excess(lum0, cy, cx, r), _excess(lum1, cy, cx, r)
            assert before > 2 * SIGMA                              # a real halo was there
            assert abs(after) < 0.3 * before                             # and is mostly gone
    # the core inside r_in is untouched
    assert abs(float(out[150, 150, 1]) - float(stack[150, 150, 1])) < 1.0
    # the nebula 70 px from a bright star keeps its flux: what leaves is the star's wing there
    # (~19 ADU), not the 40 ADU blob -- the amplitude comes from the catalogue, not the pixels
    removed = (stack[:, :, 0] - out[:, :, 0]).astype(np.float64)
    box = (slice(440, 460), slice(510, 530))
    # a far wing is under-subtracted (the window edge and the local sky baseline absorb some),
    # never over-subtracted into the nebula
    assert 0.45 * wing[box].mean() < removed[box].mean() < wing[box].mean() + 4.0


def test_no_halo_means_nothing_is_taken(monkeypatch):
    rng = np.random.default_rng(8)
    hdr, w = _wcs_header()
    yy, xx = np.mgrid[:H, :W].astype(np.float64)
    img = np.full((H, W), 500.0)
    stars = [(rng.uniform(40, H - 40), rng.uniform(40, W - 40), rng.uniform(8.0, 13.0)) for _ in range(40)]
    for y, x, g in stars:
        r2 = (yy - y) ** 2 + (xx - x) ** 2
        img += 10.0 ** (A - 0.4 * g) / (2 * np.pi * 1.8 ** 2) * np.exp(-r2 / (2 * 1.8 ** 2))
    img = np.minimum(img + rng.normal(0, SIGMA, img.shape), CLIP)
    stack = np.stack([img] * 3, axis=2).astype(np.float32)
    ra, dec = w.all_pix2world([s[1] for s in stars], [s[0] for s in stars], 0)
    cat = Table({'ra': ra, 'dec': dec, 'phot_g_mean_mag': [s[2] for s in stars],
                 'phot_bp_mean_mag': [s[2] + 0.5 for s in stars],
                 'phot_rp_mean_mag': [s[2] - 0.3 for s in stars]})
    monkeypatch.setattr(net_query, 'gaia_cone_search', lambda *a, **k: cat)
    out, info = halo_model.subtract_halos(stack, hdr, fwhm=4.2)
    diff = (stack - out)[:, :, 0].astype(np.float64)
    assert np.abs(diff).max() < 3 * SIGMA                                # no halo found: no moat dug


def test_declines_without_catalogue_wcs_or_stars(field, monkeypatch):
    stack, hdr, cat, _w, _b = field
    monkeypatch.setattr(net_query, 'gaia_cone_search', lambda *a, **k: None)
    out, info = halo_model.subtract_halos(stack, hdr, fwhm=4.2)
    assert out is stack and info['declined'] == 'no Gaia catalogue'
    out, info = halo_model.subtract_halos(stack, fits.Header(), fwhm=4.2)
    assert out is stack and info['declined'] == 'no WCS'
    monkeypatch.setattr(net_query, 'gaia_cone_search', lambda *a, **k: cat[:5])
    out, info = halo_model.subtract_halos(stack, hdr, fwhm=4.2)
    assert out is stack and 'declined' in info
