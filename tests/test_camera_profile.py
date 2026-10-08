"""Camera profiles (src/camera_profile.py): identification, the raw-gain estimator,
the per-session check of the shipped gain, and its place in photometry's gain order."""
import argparse

import numpy as np
import pytest

from src import camera_profile as cp
from src.photometry import _read_gain, poisson_coefficients


def _hdr(**kw):
    h = {'CAMERA': 'Origin178-932f562381e97ce70', 'ISOSPEED': 200, 'CREATOR': 'Origin 1.4.6084',
         'NAXIS1': 3056, 'NAXIS2': 2048, 'BAYERPAT': 'RGGB', 'EGAIN': 0.07417014}
    h.update(kw)
    return h


def test_identify_origin_header():
    cam = cp.identify(_hdr())
    assert cam.model == 'Origin178' and cam.unit == '932f562381e97ce70'
    assert cam.iso == 200 and cam.firmware == 'Origin 1.4.6084'


def test_identify_unknown_and_missing():
    assert cp.identify({}) is None
    assert cp.identify(None) is None
    cam = cp.identify({'CAMERA': 'ZWO ASI2600MC'})
    assert cam.unit is None and cp.load_profile(cam.model) is None


def test_shipped_profile_and_geometry_check():
    cam = cp.identify(_hdr())
    prof = cp.profile_for(cam, (2048, 3056))
    assert prof is not None
    assert cp.table_gain(prof, 200) == pytest.approx(0.0165)
    assert cp.table_gain(prof, 500) == pytest.approx(0.0064)
    assert cp.table_gain(prof, 2000) is None          # not measured: no guess
    assert cp.profile_for(cam, (1080, 1920)) is None  # binned/cropped frames: not this mode


def _poisson_pair(gain, sky_e, pedestal=4990.0, step=16.0, floor_sd=0.0, shape=(400, 600), seed=1):
    """Two raw frames: photons ~ Poisson(sky_e * vignetting), ADU = e/gain + pedestal +
    read noise, quantised to ``step`` ADU like the Origin's stepped values."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    vig = 1.0 - 0.3 * (((yy - shape[0] / 2) / shape[0]) ** 2 + ((xx - shape[1] / 2) / shape[1]) ** 2)
    out = []
    for _ in range(2):
        e = rng.poisson(sky_e * vig)
        adu = e / gain + pedestal + rng.normal(0, floor_sd, shape)
        out.append((np.round(adu / step) * step).astype(np.float32))
    return out


def test_raw_pair_gain_recovers_gain_through_quantisation():
    a, b = _poisson_pair(0.0165, sky_e=150.0)    # ~9000 ADU over the pedestal
    g = cp.raw_pair_gain(a, b, 'RGGB', 4990.0)
    assert np.allclose(g, 0.0165, rtol=0.03), g


def test_floor_correction_on_a_dim_sky():
    # read + dark noise in the variance makes a dim sky read low; the floor removes it
    sd = 120.0
    a, b = _poisson_pair(0.0165, sky_e=45.0, floor_sd=sd, seed=3)    # ~2700 ADU of sky
    raw = np.nanmedian(cp.raw_pair_gain(a, b, 'RGGB', 4990.0))
    fixed = np.nanmedian(cp.raw_pair_gain(a, b, 'RGGB', 4990.0, floor_var=[sd * sd] * 3))
    assert raw < 0.0165 * 0.95
    assert fixed == pytest.approx(0.0165, rel=0.04)


def test_verify_gain_flags_a_mismatch(monkeypatch):
    prof = cp.load_profile('Origin178')
    monkeypatch.setattr(cp, 'measure_raw_gain', lambda *a, **k: np.array([0.030, 0.031, 0.030]))
    g_tab, g_meas, ok = cp.verify_gain(prof, 200, ['a', 'b', 'c'], 'RGGB')
    assert g_tab == pytest.approx(0.0165) and g_meas == pytest.approx(0.030) and not ok
    monkeypatch.setattr(cp, 'measure_raw_gain', lambda *a, **k: np.array([0.0163, 0.0166, 0.0164]))
    assert cp.verify_gain(prof, 200, ['a', 'b', 'c'], 'RGGB')[2]


class _F:
    def __init__(self, header, path='x.fits'):
        self.header, self.path = header, path


def test_resolve_camera(monkeypatch):
    args = argparse.Namespace()
    info = cp.resolve_camera([_F(_hdr())], args)
    assert info['gain'] == pytest.approx(0.0165) and args._camera is info
    monkeypatch.setattr(cp, 'verify_gain', lambda *a, **k: (0.0165, 0.031, False))
    info = cp.resolve_camera([_F(_hdr())], args, verify=True)
    assert info['gain'] is None and info['gain_measured'] == pytest.approx(0.031)
    assert cp.resolve_camera([_F({'CAMERA': 'Other-12345678'})], args) is None
    assert args._camera is None


def test_read_gain_order():
    hdr = _hdr()
    args = argparse.Namespace(photometry_gain=None, _ptc_gain_e_per_adu=0.5,
                              _camera={'gain': 0.0165, 'label': 'Origin178', 'iso': 200})
    assert _read_gain(hdr, args) == pytest.approx(0.0165)          # profile before PTC
    args.photometry_gain = 0.02
    assert _read_gain(hdr, args) == pytest.approx(0.02)            # explicit override first
    args.photometry_gain, args._camera = None, {'gain': None}
    assert _read_gain(hdr, args) == pytest.approx(0.5)             # contradicted profile: PTC
    args._ptc_gain_e_per_adu = None
    assert _read_gain(hdr, args) == pytest.approx(0.07417014)      # then the header


def test_poisson_label_names_the_profile():
    args = argparse.Namespace(photometry_gain=None, _noise_model_k=None,
                              _camera={'gain': 0.0165, 'label': 'Origin178 #932f56', 'iso': 200},
                              _stack_poisson_factor=None)
    c, src = poisson_coefficients({'NFRAMES': 1}, args, stacked=False)
    assert np.allclose(c, 1 / 0.0165) and 'camera profile' in src
