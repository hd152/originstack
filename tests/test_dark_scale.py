"""Per-session dark glow scale (src/dark_scale.py): fit, acceptance rules, Phase 1 hook."""
import os
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from src.dark_scale import (
    apply_session_dark_scale,
    decide_dark_scale,
    fit_glow_delta,
    measure_session_dark_scale,
)
from src.frame_processor import _process_single_frame

H, W = 512, 640
BLOCK = 64


def _glow():
    """A corner glow: 0 in the middle, ~150 ADU in one corner (not a quadratic)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return 150.0 * np.exp(-((yy - H) ** 2 + (xx - W) ** 2) / (2 * 140.0 ** 2)).astype(np.float32)


def _sky(rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return 900.0 + 0.05 * xx + 0.03 * yy + 0.00002 * (xx - 300) ** 2


def _session(tmp_path, delta_true, n=10, exp=20.0, dark_exp=20.0, seed=0):
    """Lights whose dark term is s * (1 + delta_true) times the library dark's glow."""
    rng = np.random.default_rng(seed)
    glow = _glow()
    bias = np.full((H, W), 100.0, dtype=np.float32)
    dark = bias + glow
    s = exp / dark_exp
    paths = []
    for i in range(n):
        light = (bias + _sky(rng) + s * (1.0 + delta_true) * glow
                 + rng.normal(0, 4.0, (H, W))).astype(np.float32)
        hdu = fits.PrimaryHDU(light)
        hdu.header['EXPTIME'] = exp
        hdu.header['IMAGETYP'] = 'LIGHT'
        p = os.path.join(str(tmp_path), f'L{i}.fits')
        hdu.writeto(p, overwrite=True)
        paths.append(p)
    lights = [SimpleNamespace(path=p) for p in paths]
    masters = {'bias': bias, 'dark': dark, 'flat': None, 'dark_exptime': dark_exp}
    return lights, masters


def test_fit_recovers_known_glow_multiple():
    rng = np.random.default_rng(1)
    gb = _glow().reshape(8, 64, 10, 64).mean(axis=(1, 3))
    yy, xx = np.mgrid[0:8, 0:10]
    cal = 30.0 + 2.0 * xx + 1.5 * yy + 0.4 * gb + rng.normal(0, 0.5, gb.shape)
    out = fit_glow_delta(cal, gb)
    assert out is not None
    delta, se = out
    assert abs(delta - 0.4) < 0.05 and 0 < se < 0.05


def test_fit_ignores_a_pure_sky_polynomial():
    rng = np.random.default_rng(2)
    gb = _glow().reshape(8, 64, 10, 64).mean(axis=(1, 3))
    yy, xx = np.mgrid[0:8, 0:10]
    cal = 30.0 + 2.0 * xx + 1.5 * yy + 0.3 * xx * yy + rng.normal(0, 0.5, gb.shape)
    delta, se = fit_glow_delta(cal, gb)
    assert abs(delta) < 3 * se + 0.02


def test_cold_session_gets_a_glow_correction(tmp_path):
    lights, masters = _session(tmp_path, delta_true=-0.5)
    res = measure_session_dark_scale(lights, masters)
    assert res is not None
    assert abs(res['delta'] - (-0.5)) < 0.05
    apply_session_dark_scale(masters, lights)
    assert abs(masters['dark_scale_adjust'] - 0.5) < 0.06


def test_exposure_ratio_other_than_one_is_divided_out(tmp_path):
    # lights twice the dark's exposure: the residual glow is s * delta, the fit reports delta
    lights, masters = _session(tmp_path, delta_true=-0.3, exp=40.0, dark_exp=20.0)
    res = measure_session_dark_scale(lights, masters)
    assert res is not None and abs(res['delta'] - (-0.3)) < 0.05


def test_matched_session_is_left_alone(tmp_path):
    lights, masters = _session(tmp_path, delta_true=0.005)
    assert measure_session_dark_scale(lights, masters) is None
    apply_session_dark_scale(masters, lights)
    assert 'dark_scale_adjust' not in masters


def test_opt_out_and_temperature_model_and_short_sessions(tmp_path):
    lights, masters = _session(tmp_path, delta_true=-0.5)
    apply_session_dark_scale(masters, lights, SimpleNamespace(session_dark_scale=False))
    assert 'dark_scale_adjust' not in masters
    masters['dark_from_temp_model'] = True
    apply_session_dark_scale(masters, lights)
    assert 'dark_scale_adjust' not in masters
    assert measure_session_dark_scale(lights[:3], {**masters, 'dark_from_temp_model': False}) is None


def test_implausible_or_noisy_fits_are_refused():
    assert decide_dark_scale([-1.3] * 6, [0.02] * 6) is None           # would add the dark back
    assert decide_dark_scale([0.95] * 6, [0.02] * 6) is None           # more glow than the dark has
    cold = decide_dark_scale([-0.93] * 6, [0.08] * 6)                  # Markarian, 19-21 C
    assert cold is not None and -0.93 < cold['used'] < -0.9
    assert decide_dark_scale([-1.1] * 6, [0.02] * 6)['used'] == -1.0   # clamped: scale 0, not negative
    assert decide_dark_scale([-0.4] * 6, [0.15] * 6) is None           # loosely determined
    assert decide_dark_scale([-0.9, -0.1, -0.5, 0.2, -0.7, 0.1], [0.02] * 6) is None  # probes disagree
    assert decide_dark_scale([-0.4] * 3, [0.02] * 3) is None           # too few probes
    ok = decide_dark_scale([-0.4] * 6, [0.02] * 6)
    assert ok is not None and -0.4 < ok['used'] < -0.39                # barely shrunk


def test_phase1_applies_the_adjust(tmp_path):
    glow = _glow()
    noise = np.random.default_rng(5).normal(0, 3.0, glow.shape)
    light = (200.0 + 0.5 * glow + noise).astype(np.float32)
    hdu = fits.PrimaryHDU(light)
    hdu.header['EXPTIME'] = 20.0
    p = os.path.join(str(tmp_path), 'one.fits')
    hdu.writeto(p, overwrite=True)

    def run(adjust):
        m = {'dark': glow.astype(np.float32), 'dark_exptime': 20.0}
        if adjust is not None:
            m['dark_scale_adjust'] = adjust
        r = _process_single_frame(p, {}, m, 'rcd', 'none', quality_level='quick',
                                  skip_quality=True)
        assert r['error'] is None
        return r['rgb'][..., 0].astype(np.float64)

    plain, adjusted = run(None), run(0.5)
    # the light holds half the library dark's glow: the plain exposure-ratio scale over-subtracts
    # it (200 - 0.5 g); a scale of 0.5 leaves the flat 200 in the glow corner too
    corner = (slice(420, 500), slice(520, 630))
    assert glow[corner].mean() > 60
    assert abs(adjusted[corner].mean() - 200.0) < 1.0
    assert plain[corner].mean() < 200.0 - 20.0
