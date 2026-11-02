"""Effective integration time (src/effective_exposure.py) and its frame-report column."""
from types import SimpleNamespace

import numpy as np

from src.effective_exposure import (
    LOW_T_EFF,
    effective_integration,
    frame_t_eff,
    session_fiducials,
)
from src.frame_report import build_frame_report


def _f(fwhm, bg, tr=1.0, exp=20.0, name='a'):
    return SimpleNamespace(path=name, header={'EXPTIME': exp}, shift=(0.0, 0.0),
                           metrics={'fwhm': fwhm, 'background': bg, 'transparency': tr,
                                    'star_count': 100, 'score': 1.0, 'snr': 5.0,
                                    'ellipticity': 0.1})


def test_t_eff_formula():
    assert frame_t_eff({'fwhm': 4.0, 'background': 100.0, 'transparency': 1.0}, 4.0, 100.0) == 1.0
    # twice the FWHM: a quarter; twice the sky: half; 0.8 transparency: 0.64
    assert abs(frame_t_eff({'fwhm': 8.0, 'background': 100.0}, 4.0, 100.0) - 0.25) < 1e-9
    assert abs(frame_t_eff({'fwhm': 4.0, 'background': 200.0}, 4.0, 100.0) - 0.5) < 1e-9
    assert abs(frame_t_eff({'fwhm': 4.0, 'background': 100.0, 'transparency': 0.8}, 4.0, 100.0) - 0.64) < 1e-9
    assert frame_t_eff({'fwhm': 4.0}, 4.0, 100.0) is None
    assert frame_t_eff({'fwhm': float('nan'), 'background': 1.0}, 4.0, 1.0) is None
    assert abs(frame_t_eff({'fwhm': 4.0, 'background': 100.0, 'transparency': float('nan')},
                           4.0, 100.0) - 1.0) < 1e-9


def test_session_time_in_median_quality_seconds():
    frames = [_f(4.0, 100.0, name=str(i)) for i in range(8)]
    frames += [_f(4.0, 100.0, tr=0.5, name='cloud')]             # a clouded frame counts a quarter
    res = effective_integration(frames)
    assert res['n'] == 9
    assert abs(res['teff_time'] - (8 * 20 + 0.25 * 20)) < 1e-6
    assert res['ratio'] < 1.0
    fid = session_fiducials(frames)
    assert fid['fwhm'] == 4.0 and fid['background'] == 100.0
    assert effective_integration([SimpleNamespace(metrics={}, header={})]) is None


def test_frame_report_column_and_low_value_flag():
    lights = [_f(4.0, 100.0, name=f'ok{i}.fits') for i in range(7)] + [_f(9.0, 100.0, name='soft.fits')]
    rows = {r['filename']: r for r in build_frame_report(lights, {}, lights)}
    assert rows['ok0.fits']['t_eff'] == 1.0 and rows['ok0.fits']['notes'] == ''
    assert rows['soft.fits']['t_eff'] < LOW_T_EFF
    assert 'effective exposure' in rows['soft.fits']['notes']
    assert np.isfinite(rows['soft.fits']['t_eff'])
