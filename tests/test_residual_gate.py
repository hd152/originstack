"""The absolute registration-residual gate must not reject a whole session."""
import numpy as np

from src.models import Config
from src.registration import _relative_residual_gate


def _res(median, n=145, spread=0.3, seed=0):
    return list(np.abs(np.random.default_rng(seed).normal(median, spread, n)))


def test_a_gate_that_fails_nearly_everything_is_replaced_by_the_sessions_own_spread():
    res = _res(2.4)                                      # every frame ~2.4 px
    n_failed = sum(r > 1.5 for r in res)                 # the 1.5 px floor rejects ~all
    assert n_failed > 0.9 * len(res)
    thr, med = _relative_residual_gate(res, 1.5, n_failed)
    assert 2.3 < med < 2.5
    assert thr > 1.5
    passed = sum(r <= thr for r in res)
    assert passed > 0.95 * len(res)                      # the session is kept...
    res[7] = 5.5                                         # ...but a real outlier still fails
    assert res[7] > thr


def test_a_gate_that_only_catches_a_few_frames_is_left_alone():
    res = _res(0.9)
    res[:6] = [4.0] * 6
    n_failed = sum(r > 1.5 for r in res)
    assert _relative_residual_gate(res, 1.5, n_failed) is None


def test_a_session_that_is_really_misregistered_keeps_the_strict_result():
    res = _res(Config.REG_RESIDUAL_MAX_PX_CAP + 4.0)     # median beyond the cap
    n_failed = sum(r > 1.5 for r in res)
    assert _relative_residual_gate(res, 1.5, n_failed) is None


def test_small_sessions_and_missing_measurements_are_left_alone():
    assert _relative_residual_gate(_res(2.4, n=6), 1.5, 6) is None
    res = [float('nan')] * 100 + [2.4] * 20
    assert _relative_residual_gate(res, 1.5, 120) is None     # too few finite values


def test_threshold_never_exceeds_the_cap():
    res = _res(5.8, spread=0.9)
    n_failed = sum(r > 1.5 for r in res)
    out = _relative_residual_gate(res, 1.5, n_failed)
    assert out is None or out[0] <= Config.REG_RESIDUAL_MAX_PX_CAP
