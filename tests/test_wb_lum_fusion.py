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


_INTO = D._HAS_NATIVE and hasattr(D._native, 'white_balance_grayworld_lum_into')


def _frame(seed, h, w, nan=False, cap=None):
    rng = np.random.default_rng(seed)
    f = (np.array([900.0, 1000.0, 1100.0], np.float32) + rng.normal(0, 25, (h, w, 3))).astype(np.float32)
    f[h // 3:h // 3 + 3, w // 2:w // 2 + 3] += 60000.0                 # a star past the ceiling
    if cap is not None:
        np.minimum(f, cap, out=f)                                     # lots of saturated pixels
    if nan:
        f[1, 2, 1] = np.nan
    return np.ascontiguousarray(f)


def _u(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


@pytest.mark.skipif(not _INTO, reason='astro_native without white_balance_grayworld_lum_into')
@pytest.mark.parametrize('case', ['plain', 'nan', 'saturated', 'inf', 'zeros', 'small', 'odd'])
@pytest.mark.parametrize('step', [1, 3, 4])
def test_lum_into_bit_identical(case, step):
    """white_balance_grayworld_lum_into: rgb and lum == white_balance_grayworld_lum_inplace,
    stats == validate_frame_stats(lum), sample == lum[::step, ::step], bit for bit."""
    h, w = {'small': (20, 30), 'odd': (101, 67)}.get(case, (90, 120))
    f = _frame(3, h, w, nan=case == 'nan', cap=1200.0 if case == 'saturated' else None)
    if case == 'inf':
        f[5, 6, 2] = np.inf
    if case == 'zeros':
        f[:] = 0.0
        f[0, 0] = 5.0
    a = f.copy()
    la = D._native.white_balance_grayworld_lum_inplace(a)
    want = D._native.validate_frame_stats(la)
    b = f.copy()
    lb = np.full((h, w), 7.0, np.float32)
    sm = np.full((-(-h // step), -(-w // step)), 9.0, np.float32)
    got = D._native.white_balance_grayworld_lum_into(b, lb, sm, step)
    np.testing.assert_array_equal(_u(b), _u(a))
    np.testing.assert_array_equal(_u(lb), _u(la))
    np.testing.assert_array_equal(_u(sm), _u(la[::step, ::step]))
    assert got[0] == want[0] and got[2:] == want[2:]
    assert _u(np.float32(got[1])) == _u(np.float32(want[1]))
    # no sample requested: same image, luminance and stats
    c, lc = f.copy(), np.empty((h, w), np.float32)
    assert D._native.white_balance_grayworld_lum_into(c, lc)[2:] == want[2:]
    np.testing.assert_array_equal(_u(lc), _u(la))


@pytest.mark.skipif(not _INTO, reason='astro_native without white_balance_grayworld_lum_into')
@pytest.mark.parametrize('case', ['plain', 'nan', 'saturated', 'flat', 'small'])
def test_validate_with_precomputed_stats_same_verdict(case):
    """validate_image_data with the kernel's stats and sample gives the same verdict and
    message as computing them itself."""
    from src.quality import validate_image_data
    h, w = (20, 30) if case == 'small' else (90, 120)
    f = _frame(4, h, w, nan=case == 'nan', cap=1200.0 if case == 'saturated' else None)
    if case == 'flat':
        f[:] = 1000.0
    a = f.copy()
    la = D.white_balance_grayworld_lum(a)[1]
    b = f.copy()
    out = D.white_balance_grayworld_lum_into(b, np.empty((h, w), np.float32))
    assert out is not None
    lb, stats, sample = out
    assert (sample is None) == (min(h, w) < 64)
    assert validate_image_data(lb, 'f', native_stats=stats, sample=sample) == validate_image_data(la, 'f')


@pytest.mark.skipif(not _INTO, reason='astro_native without white_balance_grayworld_lum_into')
def test_out_lum_slot_written_and_identical(monkeypatch):
    """With out_lum the luminance lands in the slot itself; rgb, lum and metrics are those
    of the path that returns a fresh luminance array."""
    slot = np.full((96, 128), -1.0, np.float32)
    a = FP._process_single_frame('x.fits', {}, {}, 'malvar', 'grayworld', session_bayer='RGGB',
                                 preloaded_data=(_mosaic(), {}), out_lum=slot)
    assert a['error'] is None
    assert a['lum'] is slot
    monkeypatch.setattr(FP, 'white_balance_grayworld_lum_into', lambda rgb, out: None)
    b = FP._process_single_frame('x.fits', {}, {}, 'malvar', 'grayworld', session_bayer='RGGB',
                                 preloaded_data=(_mosaic(), {}))
    np.testing.assert_array_equal(_u(a['rgb']), _u(b['rgb']))
    np.testing.assert_array_equal(_u(slot), _u(b['lum']))
    ma = {k: v for k, v in a['metrics'].items() if k != '_patch_scores'}
    mb = {k: v for k, v in b['metrics'].items() if k != '_patch_scores'}
    assert repr(ma) == repr(mb)
    np.testing.assert_array_equal(a['metrics']['_patch_scores'], b['metrics']['_patch_scores'])
