"""The stack guards that decide whether --auto's off-Origin --deconvolve nn runs
(auto_settings.nn_stack_guard / star_shape_stats) and how Phase 4 takes the choice back."""
import numpy as np
import pytest

from src import auto_settings as a
from src import postprocess as pp
from src.cli import parse_args


def _field(fwhm=3.0, n=60, ratio=1.0, shape=(420, 420), seed=1, theta=0.4):
    """Noisy sky with ``n`` Gaussian stars on a geometric brightness ladder (the brightest few are
    over the saturation cut the fit applies, like real stacks): sigma x, y = fwhm/2.3548
    and that x ``ratio`` (ratio < 1 = elongated), rotated by ``theta``."""
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, 2.0, shape).astype(np.float32)
    sx = fwhm / 2.3548
    sy = sx * ratio
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    pts = [(y, x) for y in range(22, shape[0] - 22, 40) for x in range(22, shape[1] - 22, 40)]
    rng.shuffle(pts)
    c, s = np.cos(theta), np.sin(theta)
    for k, (y, x) in enumerate(pts[:n]):
        u = (xx - x) * c + (yy - y) * s
        v = -(xx - x) * s + (yy - y) * c
        img += (150.0 * 1.1 ** k) * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2)).astype(np.float32)
    return np.repeat(img[..., None], 3, axis=2)


def test_shape_stats_recover_width_and_roundness():
    st = a.star_shape_stats(_field(3.0, ratio=1.0))
    assert st['n'] >= a.NN_MIN_STARS + 5 and abs(st['fwhm'] - 3.0) < 0.3 and st['ellipticity'] < 0.1


def test_shape_stats_see_elongation():
    st = a.star_shape_stats(_field(3.0, ratio=0.5))
    assert 0.4 < st['ellipticity'] < 0.6                 # 1 - 0.5


def test_a_clean_field_passes():
    assert a.nn_stack_guard(_field(3.0, n=60)) is None


def test_no_stars_blocks():
    flat = np.random.default_rng(0).normal(100, 2, (300, 300, 3)).astype(np.float32)
    assert 'no stars' in a.nn_stack_guard(flat)


def test_few_stars_blocks():
    reason = a.nn_stack_guard(_field(3.0, n=14))
    assert reason is not None and 'usable stars' in reason


def test_elongated_stars_block():
    reason = a.nn_stack_guard(_field(3.0, ratio=0.45))
    assert reason is not None and 'elliptical' in reason


@pytest.mark.parametrize('fwhm', [0.8, 16.0])
def test_width_outside_the_limits_blocks(monkeypatch, fwhm):
    monkeypatch.setattr(a, 'star_shape_stats', lambda img: {'n': 80, 'fwhm': fwhm, 'ellipticity': 0.1})
    assert 'outside' in a.nn_stack_guard(np.zeros((8, 8, 3), np.float32))


def _auto_args():
    args = parse_args(['-d', 'x', '-o', 'y.fits'])
    args._auto_nn_fallback = {'deconvolve_mode': 'off', 'deconvolve_nn': False, 'denoise_curvelet': True,
                              'nn_deconv_blend': 0.75}
    args.deconvolve_mode, args.deconvolve_nn, args.denoise_curvelet, args.nn_deconv_blend = 'nn', True, False, 0.35
    return args


def test_postprocess_guard_restores_the_settings_when_blocked(monkeypatch):
    monkeypatch.setattr(a, 'nn_stack_guard', lambda img: 'stars are elliptical (median 0.50 > 0.35)')
    args = _auto_args()
    pp._apply_auto_nn_guard(np.zeros((8, 8, 3), np.float32), args)
    assert (args.deconvolve_mode, args.deconvolve_nn, args.denoise_curvelet, args.nn_deconv_blend) == \
        ('off', False, True, 0.75)
    assert args._auto_nn_fallback is None


def test_postprocess_guard_keeps_the_choice_when_clear_and_runs_once(monkeypatch):
    calls = []
    monkeypatch.setattr(a, 'nn_stack_guard', lambda img: calls.append(1))
    args = _auto_args()
    pp._apply_auto_nn_guard(np.zeros((8, 8, 3), np.float32), args)
    pp._apply_auto_nn_guard(np.zeros((8, 8, 3), np.float32), args)
    assert args.deconvolve_nn is True and args.denoise_curvelet is False and calls == [1]


def test_postprocess_guard_is_a_noop_without_a_fallback(monkeypatch):
    monkeypatch.setattr(a, 'nn_stack_guard', lambda img: 'blocked')
    args = parse_args(['-d', 'x', '-o', 'y.fits', '--deconvolve', 'nn'])
    pp._apply_auto_nn_guard(np.zeros((8, 8, 3), np.float32), args)     # a user's --deconvolve nn stays
    assert args.deconvolve_nn is True
