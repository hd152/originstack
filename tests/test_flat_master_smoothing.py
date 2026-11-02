"""Master-flat smoothing: a single pre-averaged flat keeps its real pixel-response pattern.

``_build_masters`` blurs the master flat with a sigma that shrinks with the number of frames
averaged. A lone file (Origin ships one library flat) is a master, not one noisy frame; the
old n == 1 value of 15 px erased the structure the lights show in their sensor-locked pattern.
"""
import argparse
import os

import numpy as np
from astropy.io import fits

from src.cli import _build_masters, _flat_smooth_sigma
from src.frame_discovery import discover_frames
from src.models import Config, ProcessingStats

H, W = 128, 192
PERIOD = 64.0          # px: a pixel-response ripple the old sigma-15 blur removed (~1%)
AMP = 0.02


def _write(folder, name, arr, kind, et):
    hdu = fits.PrimaryHDU(np.clip(np.round(arr), 0, 65535).astype(np.uint16))
    hdu.header['IMAGETYP'] = kind
    hdu.header['EXPTIME'] = et
    hdu.header['FILTER'] = 'L'
    hdu.header['BAYERPAT'] = 'RGGB'
    hdu.header['GAIN'] = 120
    hdu.header['CCD-TEMP'] = -10.0
    hdu.writeto(os.path.join(folder, name), overwrite=True)


def _args():
    return argparse.Namespace(master_method='median', auto=False, dark_temp_model=False,
                              flat_from_lights=False, vignette_map=None, photometry=False,
                              photometry_timeseries=False, _explicit_cli_dests=set())


def _flat_masters(tmp_path, n_flats):
    xx = np.arange(W)[None, :].repeat(H, axis=0)
    ripple = 1.0 + AMP * np.sin(2 * np.pi * xx / PERIOD)
    rng = np.random.default_rng(3)
    folder = str(tmp_path)
    _write(folder, 'Light_0.fits', np.full((H, W), 500.0), 'LIGHT', 10.0)
    for i in range(n_flats):
        _write(folder, f'Flat_{i}.fits', 20000.0 * ripple + rng.normal(0, 20.0, (H, W)),
               'FLAT', 1.0)
    masters = _build_masters(discover_frames(folder), ProcessingStats(), _args())
    return masters['flat'], ripple


def _ripple_gain(flat, ripple):
    """Regression slope of the master flat on the true ripple, interior columns only."""
    f = flat[:, 32:-32] / np.median(flat) - 1.0
    r = ripple[:, 32:-32] - 1.0
    return float((f * r).sum() / (r * r).sum())


def test_sigma_rule():
    assert _flat_smooth_sigma(1) == Config.FLAT_SINGLE_MASTER_SIGMA
    assert _flat_smooth_sigma(0) == Config.FLAT_SINGLE_MASTER_SIGMA
    # n >= 2 keeps the established rule
    assert _flat_smooth_sigma(2) == 15
    assert _flat_smooth_sigma(4) == 7
    assert _flat_smooth_sigma(25) == 3
    assert _flat_smooth_sigma(400) == 1


def test_single_flat_keeps_pixel_response_pattern(tmp_path):
    flat, ripple = _flat_masters(tmp_path, 1)
    # sigma 4 on a 32 px-per-period (Bayer plane) ripple keeps ~70%; sigma 15 kept ~1%
    assert _ripple_gain(flat, ripple) > 0.5


def test_many_flats_still_smoothed_as_before(tmp_path):
    flat, ripple = _flat_masters(tmp_path, 4)       # sigma 7: unchanged by the single-master rule
    assert _ripple_gain(flat, ripple) < 0.5
