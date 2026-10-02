"""Shot-noise model from same-colour pixel pairs, correlated-noise factor, photometry
Poisson coefficients (src/noise_model.py, src/photometry.py)."""
import types

import numpy as np
import pytest
from scipy import ndimage as ndi

from src import noise_model as nm
from src.debayer import _rcd_raw


def _frames(n=8, gain=0.02, wb=(1.6, 1.0, 1.5), seed=0, shape=(240, 320)):
    """Debayered RGGB frames whose raw samples are Poisson electrons / gain, scaled per
    channel by ``wb`` (as white balance does), with a sky gradient, stars and a dark
    offset (a constant, which the slope must ignore)."""
    rng = np.random.default_rng(seed)
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    chan = np.empty(shape, int)
    chan[0::2, 0::2], chan[0::2, 1::2], chan[1::2, 0::2], chan[1::2, 1::2] = 0, 1, 1, 2
    out = []
    for j in range(n):
        sky_e = 300 + 80 * j + 150 * (xx / W) + 60 * (yy / H)       # electrons, varies within/across frames
        img_e = sky_e.copy()
        for _ in range(25):
            y, x = rng.uniform(10, H - 10), rng.uniform(10, W - 10)
            img_e += 4000 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 4.0)
        e = rng.poisson(img_e).astype(np.float64)
        adu = e / gain + 50.0                                      # + a dark-current offset
        adu *= np.asarray(wb)[chan]
        out.append(_rcd_raw(adu.astype(np.float32), 'RGGB'))
    return out


def test_noise_model_recovers_gain_and_white_balance():
    gain, wb = 0.02, (1.6, 1.0, 1.5)
    frames = _frames(gain=gain, wb=wb)
    m = nm.measure_noise_model(frames, range(len(frames)), 'RGGB')
    assert m is not None
    want = np.asarray(wb) / gain          # var per ADU of signal: k_c = w_c / g
    np.testing.assert_allclose(m['k'], want, rtol=0.06)


def test_noise_model_declines_without_a_pattern():
    frames = _frames(n=3)
    assert nm.measure_noise_model(frames, range(3), '') is None
    assert nm.measure_noise_model(frames, [], 'RGGB') is None


def test_temporal_noise_white_and_correlated():
    rng = np.random.default_rng(1)
    static = rng.normal(0, 30, (512, 512, 3)) + np.linspace(0, 200, 512)[None, :, None]
    a = static + rng.normal(0, 5, static.shape)
    b = static + rng.normal(0, 5, static.shape)
    st, rs = nm.temporal_noise(a, b, blocks=(8, 16))
    np.testing.assert_allclose(st, 5.0, rtol=0.05)             # static structure cancelled
    for r in rs.values():
        np.testing.assert_allclose(r, 1.0, rtol=0.15)          # white noise: no correlation
    smooth = lambda x: np.stack([ndi.gaussian_filter(x[..., c], 1.0) for c in range(3)], -1)
    a2, b2 = smooth(rng.normal(0, 5, static.shape)), smooth(rng.normal(0, 5, static.shape))
    _, rs2 = nm.temporal_noise(a2, b2, blocks=(16,))
    assert (rs2[16] > 5).all()                                 # sigma-1 smoothing: R ~ 4 pi sigma^2 ~ 12
    assert nm.corr_factor_at(rs2, 9.0) is not None and nm.corr_factor_at({}, 9.0) is None


def test_poisson_coefficients_precedence_and_stack_factor():
    from astropy.io import fits

    from src.photometry import poisson_coefficients
    h = fits.Header()
    h['EGAIN'] = 0.5
    h['NFRAMES'] = 10
    a = types.SimpleNamespace(photometry_gain=None)
    c, src = poisson_coefficients(h, a, stacked=False)
    np.testing.assert_allclose(c, 2.0)
    assert src == 'FITS gain'
    c, _ = poisson_coefficients(h, a, stacked=True)            # no recorded factor: 1 / NFRAMES
    np.testing.assert_allclose(c, 0.2)
    a._noise_model_k = [3.0, 2.0, 4.0]
    a._stack_poisson_factor = [0.01, 0.01, 0.02]
    c, src = poisson_coefficients(h, a, stacked=True)
    np.testing.assert_allclose(c, [0.03, 0.02, 0.08])
    assert src.startswith('noise model')
    a.photometry_gain = 4.0                                     # the user's gain wins
    c, src = poisson_coefficients(h, a, stacked=False)
    np.testing.assert_allclose(c, 0.25)
    assert src == '--photometry-gain'
    assert poisson_coefficients(fits.Header(), types.SimpleNamespace(photometry_gain=None))[0] is None


def test_proper_coadd_poisson_factor_equal_frames():
    """Equal frames (same F, same noise): the factor is exactly 1 / N."""
    from src import proper_coadd as pc
    rng = np.random.default_rng(2)
    H, W, N = 320, 400, 12
    yy, xx = np.mgrid[:H, :W].astype(float)
    base = np.full((H, W), 100.0)
    for _ in range(30):
        y, x = rng.uniform(20, H - 20), rng.uniform(20, W - 20)
        base += rng.lognormal(7.5, 0.8) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5 ** 2))
    A = np.stack([base + rng.normal(0, 5, (H, W)) for _ in range(N)]).astype(np.float32)[..., None].repeat(3, -1)
    stats = {}
    out = pc.proper_coadd(np.ascontiguousarray(A), A.mean(0), fwhm=3.5, verbose=False, stats=stats)
    assert out is not None
    np.testing.assert_allclose(stats['poisson_factor'], 1.0 / stats['frames'], rtol=0.15)
