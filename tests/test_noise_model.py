"""Shot-noise model from same-colour pixel pairs, correlated-noise factor, photometry
Poisson coefficients (src/noise_model.py, src/photometry.py)."""
import types

import numpy as np
import pytest
from scipy import ndimage as ndi

from src import noise_model as nm
from src.debayer import _rcd_raw


def _session(tmp_path, n=6, gain=0.02, wb=(1.6, 1.0, 1.5), pedestal=500.0, seed=0,
             shape=(320, 400)):
    """Raw RGGB lights on disk (Poisson electrons / gain + pedestal, under 30%
    vignetting) and the processed frames Phase 1 would make of them: pedestal off,
    flat-fielded, white-balanced, RCD-debayered. The flat makes the sky uniform while
    its noise still grows toward the corners -- the case a variance-vs-signal slope on
    processed frames got wrong."""
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    flat = 1.0 - 0.3 * (((yy - H / 2) / H) ** 2 + ((xx - W / 2) / W) ** 2) * 4
    chan = np.empty(shape, int)
    chan[0::2, 0::2], chan[0::2, 1::2], chan[1::2, 0::2], chan[1::2, 1::2] = 0, 1, 1, 2
    paths, frames = [], []
    for j in range(n):
        e = rng.poisson((400.0 + 30 * j) * flat).astype(np.float64)
        raw = e / gain + pedestal
        p = tmp_path / f"Light{j:04d}.fits"
        fits.PrimaryHDU(raw.astype(np.float32)).writeto(p)
        paths.append(str(p))
        proc = (raw - pedestal) / flat * np.asarray(wb)[chan]
        frames.append(_rcd_raw(proc.astype(np.float32), 'RGGB'))
    # the coefficient varies across a flat-fielded frame (higher in the corners); the
    # model reports its median over the interior it samples (2 x the per-plane border)
    b = 2 * min(100, H // 16, W // 16)
    want = np.array([np.median(wb[c] / flat[b:-b, b:-b]) for c in range(3)]) / gain
    return frames, paths, want


def test_noise_model_from_raw_gain_and_processing_scale(tmp_path):
    frames, paths, want = _session(tmp_path)
    m = nm.measure_noise_model(frames, range(len(frames)), paths, 'RGGB', 0.02, 500.0)
    assert m is not None and m['frames'] >= 3
    np.testing.assert_allclose(m['k'], want, rtol=0.02)


def test_noise_model_declines_without_pattern_or_gain(tmp_path):
    frames, paths, _ = _session(tmp_path, n=2)
    assert nm.measure_noise_model(frames, range(2), paths, '', 0.02, 500.0) is None
    assert nm.measure_noise_model(frames, range(2), paths, 'RGGB', None, 500.0) is None
    assert nm.measure_noise_model(frames, [], [], 'RGGB', 0.02, 500.0) is None
    assert nm.measure_noise_model(frames, range(2), paths[:1], 'RGGB', 0.02, 500.0) is None


def test_raw_gain_source_order(monkeypatch):
    import argparse

    from src import camera_profile as cp
    from src.pipeline import _raw_gain_for_noise_model
    lights = [types.SimpleNamespace(path='a'), types.SimpleNamespace(path='b')]
    masters = {'bias': np.full((64, 64), 4990.0, np.float32)}
    args = argparse.Namespace(_camera={'gain': 0.0165, 'label': 'Origin178', 'model': 'Origin178',
                                       'iso': 200}, _session_bayer='RGGB')
    g, src, ped = _raw_gain_for_noise_model(args, lights, masters)
    assert g == pytest.approx(0.0165) and 'camera profile' in src and ped == pytest.approx(4990.0)
    # no usable profile gain: measured on the session's raw lights
    monkeypatch.setattr(cp, 'measure_raw_gain', lambda *a, **k: np.array([0.031, 0.030, 0.032]))
    args._camera = None
    g, src, ped = _raw_gain_for_noise_model(args, lights, masters)
    assert g == pytest.approx(0.031) and 'measured' in src
    # no bias and no profile: no pedestal, no model
    assert _raw_gain_for_noise_model(args, lights, {'bias': None}) == (None, None, None)


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
