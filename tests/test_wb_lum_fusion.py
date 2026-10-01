"""Phase 1 takes the fused white-balance + luminance path whenever nothing after
white balance rewrites the image -- including a session CA measurement that found
nothing to shift -- and the luminance it returns is that of the final RGB frame."""
import numpy as np
import pytest

import src.debayer as D
import src.frame_processor as FP


def _mosaic(seed=0, shape=(96, 128)):
    rng = np.random.default_rng(seed)
    m = rng.normal(1000, 30, shape).astype(np.float32)
    m[0::2, 0::2] *= 0.7                                    # colour cast for gray-world to fix
    m[1::2, 1::2] *= 1.3
    m[40:43, 60:63] += 20000.0                              # a bright star
    return m


def _run(monkeypatch, **kw):
    calls = []
    real = FP.white_balance_grayworld_lum
    monkeypatch.setattr(FP, 'white_balance_grayworld_lum',
                        lambda rgb: calls.append(1) or real(rgb))
    res = FP._process_single_frame('x.fits', {}, {}, 'malvar', 'grayworld', session_bayer='RGGB',
                                   preloaded_data=(_mosaic(), {}), skip_quality=True, **kw)
    assert res['error'] is None
    return res, bool(calls)


@pytest.mark.parametrize('kw,fused', [
    ({}, True),
    ({'ca_correction': True, 'ca_shifts': {0: None, 2: None}}, True),   # measured: nothing to shift
    ({'ca_correction': True, 'ca_shifts': {0: (0.6, -0.4), 2: None}}, False),
])
def test_fused_path_and_its_luminance(monkeypatch, kw, fused):
    res, used = _run(monkeypatch, **kw)
    assert used is fused
    rgb, lum = res['rgb'], res['lum']
    np.testing.assert_array_equal(lum, D.luminance(np.ascontiguousarray(rgb)))


def test_fused_equals_separate_steps(monkeypatch):
    a, used = _run(monkeypatch)
    assert used
    # separate steps: balance only, the luminance recompute then runs on its own
    monkeypatch.setattr(FP, 'white_balance_grayworld_lum',
                        lambda rgb: (D.white_balance_grayworld(rgb, inplace=True), None))
    b = FP._process_single_frame('x.fits', {}, {}, 'malvar', 'grayworld', session_bayer='RGGB',
                                 preloaded_data=(_mosaic(), {}), skip_quality=True)
    np.testing.assert_array_equal(a['rgb'], b['rgb'])
    np.testing.assert_array_equal(a['lum'], b['lum'])
