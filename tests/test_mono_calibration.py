"""Monochrome lights through master bias/dark/flat calibration (frame_processor, cli._build_masters).

A synthetic mono session with known truth -- vignetting, dark current, a bias pattern and a
0.4% sprinkle of hot pixels -- is calibrated and compared with the noise-free sky. The
frames are calibrated as the single plane they are and replicated after the hot-pixel fix
(masters 2-D, not three identical planes); with the planes carried as (H, W, 3) the hot-pixel
map and the mosaic hot-pixel pass were skipped and the hot pixels stayed (median +338 ADU on a
stronger version of this session).
"""
import argparse
import os

import numpy as np
import pytest
from astropy.io import fits

from src import io_fits
from src.cli import _build_masters
from src.frame_discovery import discover_frames
from src.frame_processor import _process_single_frame
from src.models import ProcessingStats

H, W = 96, 128
EXPTIME = 30.0
SKY = 400.0


def _make_session(folder, bayerpat=None):
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r2 = ((yy - H / 2) / (H / 2)) ** 2 + ((xx - W / 2) / (W / 2)) ** 2
    vign = 1.0 - 0.35 * r2 / 2
    bias = 100.0 + 4.0 * np.sin(xx / 17.0)
    rate = 0.8 + 0.4 * rng.random((H, W))
    hot = rng.random((H, W)) < 0.004
    rate = np.where(hot, 15.0 + 15.0 * rng.random((H, W)), rate)      # 450-900 ADU in 30 s

    def write(name, arr, kind, et):
        hdu = fits.PrimaryHDU(np.clip(np.round(arr), 0, 65535).astype(np.uint16))
        hdu.header['IMAGETYP'] = kind
        hdu.header['EXPTIME'] = et
        hdu.header['FILTER'] = 'L'
        hdu.header['GAIN'] = 120
        hdu.header['CCD-TEMP'] = -10.0
        if bayerpat:
            hdu.header['BAYERPAT'] = bayerpat
        hdu.writeto(os.path.join(folder, name), overwrite=True)

    for i in range(3):
        write(f'Light_{i}.fits', SKY * vign + rate * EXPTIME + bias + rng.normal(0, 3.0, (H, W)), 'LIGHT', EXPTIME)
    for i in range(5):
        write(f'Dark_{i}.fits', rate * EXPTIME + bias + rng.normal(0, 3.0, (H, W)), 'DARK', EXPTIME)
        write(f'Flat_{i}.fits', 20000.0 * vign + bias + rate * 2 + rng.normal(0, 25.0, (H, W)), 'FLAT', 2.0)
        write(f'Bias_{i}.fits', bias + rng.normal(0, 3.0, (H, W)), 'BIAS', 0.0)
    return vign, hot


def _args():
    return argparse.Namespace(master_method='median', auto=False, dark_temp_model=False,
                              flat_from_lights=False, vignette_map=None, photometry=False,
                              photometry_timeseries=False, _explicit_cli_dests=set())


def _process(frames, masters, **kw):
    li = frames['light'][0]
    return _process_single_frame(li.path, li.header, masters, 'rcd', 'none',
                                 quality_level='quick', skip_quality=True, **kw)


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    folder = str(tmp_path_factory.mktemp('mono'))
    vign, hot = _make_session(folder)
    frames = discover_frames(folder)
    masters = _build_masters(frames, ProcessingStats(), _args())
    return frames, masters, vign, hot


def test_masters_are_single_planes(session):
    _, masters, _, hot = session
    for k in ('bias', 'dark', 'flat', 'hot_pixel_map'):
        assert masters[k].shape == (H, W), k
    assert masters['bias'].dtype == masters['dark'].dtype == masters['flat'].dtype == np.float32
    n = int(masters['hot_pixel_map'].sum())
    assert hot.sum() * 0.9 <= n <= hot.sum() * 1.3        # not tripled by replicated planes


def test_calibration_removes_dark_vignetting_and_hot_pixels(session):
    frames, masters, vign, hot = session
    res = _process(frames, masters)
    assert res['error'] is None
    rgb = res['rgb']
    assert rgb.shape == (H, W, 3)
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 2])
    lum = rgb[..., 0]
    inner = np.zeros((H, W), bool)
    inner[4:-4, 4:-4] = True
    sky = inner & ~hot
    # flat-field normalised: the sky is SKY x the flat's median, uniform across the frame
    assert abs(lum[sky].mean() / (SKY * np.median(vign)) - 1) < 0.02
    c = lum[H // 2 - 10:H // 2 + 10, W // 2 - 10:W // 2 + 10].mean()
    k = 12
    corners = np.mean([lum[4:4 + k, 4:4 + k].mean(), lum[4:4 + k, -4 - k:-4].mean(),
                       lum[-4 - k:-4, 4:4 + k].mean(), lum[-4 - k:-4, -4 - k:-4].mean()])
    assert abs(c / corners - 1) < 0.02
    # hot pixels (450-900 ADU of dark current) are gone, not left at hundreds of ADU
    resid = lum[hot & inner] - lum[sky].mean()
    assert abs(np.median(resid)) < 15 and np.percentile(resid, 95) < 40


def test_fast_and_plain_load_paths_agree(session, monkeypatch):
    frames, masters, _, _ = session
    fast = _process(frames, masters)['rgb']
    monkeypatch.setattr(io_fits, 'read_fits_be16', lambda path: None)
    plain = _process(frames, masters)['rgb']
    np.testing.assert_array_equal(fast, plain)


def test_banding_removal_and_spike_reject_run_on_the_plane(session):
    frames, masters, _, _ = session
    res = _process(frames, masters, banding=(1.0, 3.0), spike_reject=True)
    assert res['error'] is None and res['rgb'].shape == (H, W, 3)


def test_cfa_probe_reports_nothing_for_a_mono_frame(session):
    frames, masters, _, _ = session
    li = frames['light'][0]
    res = _process_single_frame(li.path, {}, masters, 'rcd', 'none', cfa_probe=True)
    assert res['error'] is None and res['cfa_stats'] is None


def test_the_same_data_with_a_bayer_header_still_goes_through_the_bayer_path(tmp_path):
    folder = str(tmp_path)
    _make_session(folder, bayerpat='RGGB')
    frames = discover_frames(folder)
    masters = _build_masters(frames, ProcessingStats(), _args())
    assert masters['dark'].shape == (H, W) and masters['hot_pixel_map'].shape == (H, W)
    res = _process(frames, masters)
    assert res['error'] is None and res['rgb'].shape == (H, W, 3)
    assert not np.array_equal(res['rgb'][..., 0], res['rgb'][..., 1])   # debayered: channels differ


def test_flat_master_of_a_mono_session_is_smoothed_as_one_plane(session):
    _, masters, _, _ = session
    flat = masters['flat'] / np.median(masters['flat'])
    # the 2x2 colour-plane blur a mosaic flat gets leaves a comb on a smooth flat: a mono
    # flat's pixel-to-pixel step stays far below the vignetting's own gradient
    assert np.abs(np.diff(flat, axis=1)).mean() < 0.01
