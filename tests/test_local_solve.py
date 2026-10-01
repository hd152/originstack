"""Built-in plate solver (src/local_solve.py) against a synthetic Gaia index.

The index is written to a temp directory by hand (no network), a star field is
rendered through a known TAN WCS -- rotated, and in both parities -- and the
solver must recover that WCS from a hint that is off in position and scale.
"""
import math

import numpy as np
import pytest

from src import local_solve as ls


def _write_index(root, ra0, dec0, radius, n_per_deg2=150, seed=0):
    rng = np.random.default_rng(seed)
    for b, r in ls.tiles_for_cone(ra0, dec0, radius):
        ra_lo, ra_hi, dec_lo, dec_hi = ls.tile_bounds(b, r)
        # uniform on the sphere inside the tile
        area = (math.radians(ra_hi - ra_lo)
                * (math.sin(math.radians(dec_hi)) - math.sin(math.radians(dec_lo))))
        n = int(n_per_deg2 * area * (180 / math.pi) ** 2)
        ra = rng.uniform(ra_lo, ra_hi, n)
        dec = np.degrees(np.arcsin(rng.uniform(math.sin(math.radians(dec_lo)),
                                                math.sin(math.radians(dec_hi)), n)))
        mag = rng.uniform(8.0, 15.0, n)
        arr = np.column_stack([ra, dec, mag])
        np.save(ls._tile_path(b, r, str(root)), arr[np.argsort(mag)])


def _render(cat, crval, cd, shape, fwhm=4.0, seed=1):
    """Stars through a TAN WCS (CRPIX = image centre), Gaussian PSF + noise."""
    h, w = shape
    xi, eta = ls.project_tan(cat[:, 0], cat[:, 1], *crval)
    inv = np.linalg.inv(cd)
    pix = (np.column_stack([xi, eta]) @ inv.T) + [(w - 1) / 2.0, (h - 1) / 2.0]
    img = np.zeros(shape, np.float32)
    s = fwhm / 2.3548
    yy, xx = np.mgrid[-8:9, -8:9]
    for (x, y), m in zip(pix, cat[:, 2]):
        if not (8 <= x < w - 9 and 8 <= y < h - 9):
            continue
        xi0, yi0 = int(x), int(y)
        amp = 10 ** (-0.4 * (m - 15.0)) * 400.0
        img[yi0 - 8:yi0 + 9, xi0 - 8:xi0 + 9] += amp * np.exp(
            -((xx - (x - xi0)) ** 2 + (yy - (y - yi0)) ** 2) / (2 * s * s))
    rng = np.random.default_rng(seed)
    return img + 1000.0 + rng.normal(0, 8.0, shape).astype(np.float32)


def _cd(scale_arcsec, rot_deg, parity):
    s = scale_arcsec / 3600.0
    c, n = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    return np.array([[-s * c * parity, s * n], [-s * n * parity, -s * c]])


@pytest.mark.parametrize('rot,parity', [(0.0, 1), (37.0, 1), (-120.0, -1), (7.0, -1)])
def test_recovers_rotated_and_mirrored_wcs(tmp_path, rot, parity):
    ra0, dec0 = 198.95, 42.03
    _write_index(tmp_path, ra0, dec0, 3.0)
    cat, cov = ls.cone_catalog(ra0, dec0, 1.2, fetch=False, root=str(tmp_path))
    assert cov == 1.0
    cd = _cd(1.48, rot, parity)
    img = _render(cat, (ra0, dec0), cd, (900, 1300))
    # hint 0.25 deg off, scale 3% off
    hint = ls.SolveHint(ra0 + 0.25 / math.cos(math.radians(dec0)), dec0 - 0.1,
                        search_radius=0.5, scale=1.48 * 1.03, scale_tol=0.05)
    sol = ls.solve_local(img, hint, fetch=False, root=str(tmp_path))
    assert sol is not None
    assert ls.angular_sep_deg(sol.crval[0], sol.crval[1], ra0, dec0) * 3600 < 1.0
    np.testing.assert_allclose(sol.cd, cd, atol=2e-3 * 1.48 / 3600)
    assert sol.rms_px < 0.5 and sol.n_match >= 12
    assert np.sign(np.linalg.det(sol.cd)) == np.sign(np.linalg.det(cd))


def test_no_match_on_the_wrong_patch_of_sky(tmp_path):
    ra0, dec0 = 120.0, -10.0
    _write_index(tmp_path, ra0, dec0, 6.0, seed=3)
    cat, _ = ls.cone_catalog(ra0, dec0, 1.2, fetch=False, root=str(tmp_path))
    img = _render(cat, (ra0, dec0), _cd(1.5, 10, 1), (800, 1200))
    # a hint 4 deg away with a tight search radius never sees the true field
    hint = ls.SolveHint(ra0 + 4.0, dec0, search_radius=0.3, scale=1.5, scale_tol=0.02)
    assert ls.solve_local(img, hint, fetch=False, root=str(tmp_path)) is None


def test_solve_header_writes_a_consistent_wcs(tmp_path, monkeypatch):
    from astropy.io import fits
    from astropy.wcs import WCS
    ra0, dec0 = 83.63, 22.01
    _write_index(tmp_path, ra0, dec0, 3.0, seed=5)
    monkeypatch.setenv('ORIGINSTACK_STAR_INDEX', str(tmp_path))
    cat, _ = ls.cone_catalog(ra0, dec0, 1.2, fetch=False)
    cd = _cd(1.476, 25.0, 1)
    img = _render(cat, (ra0, dec0), cd, (800, 1200))
    hdr = fits.Header()
    hdr['RA'], hdr['DEC'] = ra0 + 0.2, dec0 - 0.2      # pointing keywords only
    hdr['FOCALLEN'], hdr['XPIXSZ'] = 335.0, 2.4
    hdr['CROTA2'] = 99.0                                 # stale keyword must go
    assert ls.solve_header(img, hdr, fetch=False)
    assert 'CROTA2' not in hdr and hdr['PLTSOLVR'] == 'local'
    w = WCS(hdr, naxis=2)
    ra, dec = w.all_pix2world([[599.5, 399.5]], 0)[0]
    assert ls.angular_sep_deg(ra, dec, ra0, dec0) * 3600 < 1.0


def test_no_hint_no_solve():
    from astropy.io import fits
    assert ls.solve_header(np.zeros((50, 50), np.float32), fits.Header()) is False


@pytest.mark.parametrize('ra,dec,r', [(359.5, 10.0, 2.0), (0.3, -40.0, 3.0),
                                     (10.0, 88.5, 3.0), (200.0, -89.0, 2.0),
                                     (180.0, 0.0, 0.5)])
def test_tiles_for_cone_cover_every_nearby_point(ra, dec, r):
    """Every point within the cone falls in one of the returned tiles (RA wrap, poles)."""
    tiles = set(ls.tiles_for_cone(ra, dec, r))
    rng = np.random.default_rng(0)
    for _ in range(2000):
        rr = r * math.sqrt(rng.uniform())
        a = rng.uniform(0, 2 * math.pi)
        pra, pdec = ls.deproject_tan(math.degrees(math.tan(math.radians(rr))) * math.cos(a),
                                     math.degrees(math.tan(math.radians(rr))) * math.sin(a),
                                     ra, dec)
        b = min(ls._N_BANDS - 1, int((float(pdec) + 90.0) // ls.BAND_DEG))
        n = ls._n_ra(b)
        t = (b, min(n - 1, int((float(pra) % 360.0) // (360.0 / n))))
        assert t in tiles, (pra, pdec, t)


def test_tan_projection_round_trip():
    ra = np.array([10.0, 10.5, 9.2, 359.9])
    dec = np.array([-30.0, -29.1, -31.4, -30.2])
    xi, eta = ls.project_tan(ra, dec, 10.0, -30.0)
    r2, d2 = ls.deproject_tan(xi, eta, 10.0, -30.0)
    np.testing.assert_allclose(ls.angular_sep_deg(r2, d2, ra, dec), 0, atol=1e-9)


@pytest.mark.parametrize('val,hours,expect', [
    ('13 15 49.3', True, 198.95541666), ('-05:23:28', False, -5.39111111),
    ('+42 01 45', False, 42.02916667), (198.9, True, 198.9), ('12h30m00s', True, 187.5),
    ('-00 30 00', False, -0.5), ('junk', False, None)])
def test_sexagesimal(val, hours, expect):
    got = ls._sexagesimal(val, hours)
    if expect is None:
        assert got is None
    else:
        assert got == pytest.approx(expect, abs=1e-6)


def test_fetch_is_skipped_offline(tmp_path, monkeypatch):
    from src import net_query
    calls = []
    monkeypatch.setattr(ls, 'fetch_tile', ls.fetch_tile.__wrapped__)   # the real one
    monkeypatch.setattr(net_query, 'tap_query', lambda *a, **k: calls.append(a))
    net_query.set_offline(True)
    try:
        cat, cov = ls.cone_catalog(10.0, 10.0, 0.5, fetch=True, root=str(tmp_path))
    finally:
        net_query.set_offline(False)
    assert calls == [] and len(cat) == 0 and cov == 0.0
