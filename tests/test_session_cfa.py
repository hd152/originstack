"""Session-constant CFA equalisation (debayer.py: cfa_frame_stats / combine_cfa_stats /
set_session_cfa, frame_processor.py: _measure_session_cfa).

The per-frame path re-measured six sigma-clipped medians on every frame; the session path
measures them once. These tests pin (a) that a session value taken from a frame reproduces
what the per-frame path does to that frame, (b) that nothing changes when no session value
is set, and (c) that a session whose samples disagree is refused rather than averaged.
"""
import numpy as np
import pytest

from src import debayer as D
from src import frame_processor as FP


@pytest.fixture(autouse=True)
def _clean_session_cfa():
    D.set_session_cfa(None)
    yield
    D.set_session_cfa(None)


def _mosaic(seed=0, g2_gain=1.012, shape=(256, 320)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    sky = 800.0 + 0.4 * xx + 0.2 * yy                       # gentle gradient, like a real sky
    m = (sky + rng.normal(0, 40, shape)).astype(np.float32)
    m[1::2, 0::2] *= np.float32(g2_gain)                    # G2 sites of an RGGB mosaic
    return m


def _samples(n=8, **kw):
    return [D.cfa_frame_stats(_mosaic(seed=i, **kw), 'RGGB') for i in range(n)]


def test_frame_stats_recovers_the_g2_gain():
    st = D.cfa_frame_stats(_mosaic(g2_gain=1.02), 'RGGB')
    # green_equalize scales G2 to G1: g1/g2 ~ 1/1.02
    assert st['gain'] == pytest.approx(1 / 1.02, abs=2e-3)
    assert len(st['grid']) == 4 and abs(sum(st['grid'])) < 1e-6


def test_session_value_reproduces_the_per_frame_gain_step():
    """green_equalize with the session gain == green_equalize measuring it itself."""
    m = _mosaic(seed=3)
    st = D.cfa_frame_stats(m, 'RGGB')
    per_frame = D.green_equalize(m.copy(), pattern='RGGB', inplace=True)
    D.set_session_cfa({'pattern': 'RGGB', 'gain': st['gain'], 'grid': tuple(st['grid']),
                       'apply_grid': True, 'n': 1})
    fixed = D.green_equalize(m.copy(), pattern='RGGB', inplace=True)
    np.testing.assert_allclose(fixed, per_frame, rtol=2e-7, atol=1e-3)


def test_no_session_value_leaves_debayer_malvar_unchanged():
    m = _mosaic(seed=4)
    got = D.debayer_malvar(m, 'RGGB')
    want = D._equalize_bayer_grid(D._malvar_raw(m, 'RGGB'), inplace=True)
    np.testing.assert_array_equal(got, want)


def test_pattern_mismatch_falls_back_to_the_per_frame_path():
    m = _mosaic(seed=5)
    baseline = D.debayer_malvar(m, 'RGGB')
    D.set_session_cfa({'pattern': 'BGGR', 'gain': 0.9, 'grid': (9.0, -9.0, 9.0, -9.0),
                       'apply_grid': True, 'n': 8})
    np.testing.assert_array_equal(D.debayer_malvar(m, 'RGGB'), baseline)
    np.testing.assert_array_equal(D.green_equalize(m.copy(), 'RGGB'),
                                  D.green_equalize(m.copy(), 'RGGB'))


def test_fixed_grid_flattens_an_imposed_2x2_pattern():
    rng = np.random.default_rng(1)
    rgb = rng.normal(5000, 60, (200, 240, 3)).astype(np.float32)
    imposed = (3.0, -1.0, -1.0, -1.0)
    for (a, b), off in zip(D._GRID_PARITY, imposed):
        rgb[a::2, b::2, 1] += np.float32(off)
    q = [float(np.median(rgb[a::2, b::2, 1])) for a, b in D._GRID_PARITY]
    cfg = D.combine_cfa_stats([{'gain': 1.0, 'grid': list(np.array(q) - np.mean(q))}] * 5, 'RGGB')
    D._apply_fixed_grid(rgb, cfg)
    after = [float(np.median(rgb[a::2, b::2, 1])) for a, b in D._GRID_PARITY]
    assert max(after) - min(after) < 0.05


def test_fixed_grid_clips_at_zero_like_the_per_frame_path():
    rgb = np.zeros((16, 16, 3), np.float32)
    D._apply_fixed_grid(rgb, {'apply_grid': True, 'grid': (5.0, 5.0, 5.0, 5.0)})
    assert rgb.min() >= 0.0


def test_negligible_grid_is_not_applied():
    cfg = D.combine_cfa_stats([{'gain': 1.0, 'grid': [0.001, -0.001, 0.001, -0.001]}] * 6, 'RGGB')
    assert cfg['apply_grid'] is False
    rgb = np.full((8, 8, 3), 100.0, np.float32)
    D._apply_fixed_grid(rgb, cfg)
    assert np.all(rgb == 100.0)


def test_combine_needs_enough_samples():
    assert D.combine_cfa_stats(_samples(4), 'RGGB') is None
    assert D.combine_cfa_stats([None] * 8, 'RGGB') is None
    cfg = D.combine_cfa_stats(_samples(8), 'RGGB')
    assert cfg is not None and cfg['pattern'] == 'RGGB' and cfg['n'] == 8


def test_combine_refuses_samples_that_disagree():
    ok = {'gain': 1.0, 'grid': [1.0, -1.0, 0.0, 0.0]}
    wild = {'gain': 1.0, 'grid': [20.0, -20.0, 0.0, 0.0]}
    assert D.combine_cfa_stats([ok] * 4 + [wild] * 4, 'RGGB') is None
    drifting = [{'gain': 1.0 + 0.05 * (i % 2), 'grid': [1.0, -1.0, 0.0, 0.0]} for i in range(8)]
    assert D.combine_cfa_stats(drifting, 'RGGB') is None


def test_stats_reject_a_frame_the_per_frame_path_would_also_skip():
    m = _mosaic()
    m[1::2, 0::2] = 0.0                                     # G2 dead -> gain guard
    assert D.cfa_frame_stats(m, 'RGGB') is None


def test_probe_returns_stats_without_debayering():
    m = _mosaic(seed=6)
    res = FP._process_single_frame('x.fits', {}, {}, 'malvar', 'none', session_bayer='RGGB',
                                   preloaded_data=(m.copy(), {}), cfa_probe=True)
    assert res['error'] is None and res['cfa_stats'] is not None and 'rgb' not in res


class _Args:
    session_cfa_eq = True
    debayer_method = 'malvar'
    white_balance = 'none'
    _session_bayer = 'RGGB'


def _frames(n):
    from src.models import FrameInfo
    return [FrameInfo(path=f'f{i}.fits', type='light', header={}) for i in range(n)]


@pytest.fixture
def probe_returns(monkeypatch):
    def install(stats):
        it = iter(stats * 10)
        monkeypatch.setattr(FP, '_process_single_frame',
                            lambda *a, **k: {'error': None, 'cfa_stats': next(it),
                                             'cfa_pattern': 'RGGB'})
    return install


def test_measure_session_cfa_uses_median_of_probed_frames(probe_returns):
    probe_returns(_samples(8))
    cfg = FP._measure_session_cfa(_frames(40), {}, _Args())
    assert cfg is not None and cfg['n'] == 8


def test_measure_session_cfa_skips_short_sessions_and_opt_out(probe_returns):
    probe_returns(_samples(8))
    assert FP._measure_session_cfa(_frames(6), {}, _Args()) is None
    args = _Args()
    args.session_cfa_eq = False
    assert FP._measure_session_cfa(_frames(40), {}, args) is None
    args = _Args()
    args.debayer_method = 'menon2007'
    assert FP._measure_session_cfa(_frames(40), {}, args) is None


def test_prepare_reuses_the_measured_value_for_the_reload_pass(probe_returns):
    probe_returns(_samples(8))
    args = _Args()
    first = FP._prepare_session_cfa(_frames(40), {}, args)
    assert first is not None and D.get_session_cfa() is first
    # reload_accepted_frames must apply the same numbers, not re-measure
    probe_returns([None] * 8)
    assert FP._prepare_session_cfa(_frames(40), {}, args) is first


def test_decorator_clears_the_global_even_when_the_phase_raises():
    @FP._with_session_cfa
    def phase():
        D.set_session_cfa({'pattern': 'RGGB', 'gain': 1.0, 'grid': (0, 0, 0, 0),
                           'apply_grid': False, 'n': 1})
        raise RuntimeError('boom')
    with pytest.raises(RuntimeError):
        phase()
    assert D.get_session_cfa() is None
