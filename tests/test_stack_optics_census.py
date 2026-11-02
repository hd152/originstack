"""tools/stack_optics_census.py: width estimator, tilt fit, cell map and lateral CA on synthetic stars."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import stack_optics_census as so  # noqa: E402

H, W = 600, 800


def _gauss_stamp(shape, y, x, fwhm, amp):
    sig = fwhm / 2.3548
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float64)
    return amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sig ** 2))


def test_star_fwhm_recovers_gaussian_widths():
    rng = np.random.default_rng(0)
    for fwhm in (2.8, 3.5, 5.0, 6.5):
        img = 100 + _gauss_stamp((60, 60), 30.3, 29.6, fwhm, 3000) + rng.normal(0, 3, (60, 60))
        est = so.star_fwhm(img, 30.3, 29.6)
        assert est is not None and abs(est - fwhm) / fwhm < 0.05, (fwhm, est)
    assert so.star_fwhm(np.zeros((60, 60)), 30, 30) is None
    assert so.star_fwhm(np.ones((60, 60)), 3, 3) is None                 # stamp leaves the frame


def test_plane_fit_finds_a_pure_tilt():
    g = 4
    xx = np.tile(np.arange(g), (g, 1))
    grid = 3.0 + 0.5 * xx
    out = so.plane_fit(grid)
    assert abs(out['tilt_px'] - 2.0) < 1e-9                              # 0.5 px per cell x 4 cells = 2 px across
    assert out['resid_rel'] < 1e-9
    assert abs(abs(out['tilt_direction_deg']) % 360 - 0) < 1e-6
    assert np.isnan(so.plane_fit(np.full((4, 4), np.nan))['tilt_px'])


def _star_field(fwhm_of_x, n=260, seed=1, ca_corner=0.0):
    rng = np.random.default_rng(seed)
    stack = np.full((H, W, 3), 100.0)
    ys = rng.uniform(25, H - 25, n)
    xs = rng.uniform(25, W - 25, n)
    flux = np.sort(rng.uniform(800, 4000, n))[::-1]
    half_diag = np.hypot(W, H) / 2
    for y, x, a in zip(ys, xs, flux):
        fw = fwhm_of_x(x)
        for c in range(3):
            dy = dx = 0.0
            if c == 0 and ca_corner:                                  # R shifted radially, growing with radius
                k = ca_corner / half_diag
                dy, dx = k * (y - H / 2), k * (x - W / 2)
            cy, cx = y + dy, x + dx
            y0, y1 = max(0, int(cy) - 18), min(H, int(cy) + 19)
            x0, x1 = max(0, int(cx) - 18), min(W, int(cx) + 19)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
            sig = fw / 2.3548
            stack[y0:y1, x0:x1, c] += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))
    stack += rng.normal(0, 2.0, stack.shape)
    sources = np.zeros(n, dtype=[('xcentroid', 'f8'), ('ycentroid', 'f8'), ('flux', 'f8')])
    sources['xcentroid'], sources['ycentroid'], sources['flux'] = xs, ys, flux
    return stack.astype(np.float32), sources


def test_cell_map_follows_a_tilted_psf():
    stack, src = _star_field(lambda x: 3.0 + 2.0 * x / W)             # 3 px on the left, 5 on the right
    grid = so.cell_map(stack, src, 4, 2.0)
    cols = np.nanmedian(grid, axis=0)
    assert np.all(np.diff(cols) > 0.2)
    assert abs(cols[0] - 3.25) < 0.4 and abs(cols[-1] - 4.75) < 0.4
    assert np.nanstd(grid - np.nanmean(grid, axis=0, keepdims=True)) < 0.2      # nothing left once the column trend is out
    pf = so.plane_fit(grid)
    assert pf['resid_rel'] < 0.1 and 1.0 < pf['tilt_px'] < 3.0


def test_lateral_ca_recovers_the_corner_shift():
    stack, src = _star_field(lambda x: 3.5, ca_corner=0.4)
    res = so.lateral_ca(stack, src, n_stars=200)
    assert abs(res['R_minus_G_corner_shift_px'] - 0.4) < 0.12
    assert abs(res['B_minus_G_corner_shift_px']) < 0.1
    assert 'no radial CA' in so.verdict({'R_minus_G_corner_shift_px': 0.05})[0]
    assert 'would remove' in so.verdict({'R_minus_G_corner_shift_px': 0.25})[0]
    assert 'worth building' in so.verdict({'fwhm_max_over_min': 1.3})[0]
