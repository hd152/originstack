"""tools/common_star_fwhm.py: star width on the same stars in two stacks, each on its own grid.

The benchmark once compared each stack's FWHM over its own detected star list and reported an
advantage that came from which stars were picked. These pin the replacement on synthetic fields
whose true widths are known.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import common_star_fwhm as cs  # noqa: E402

K = 2.3548


def _field(sigma, shift=(0.0, 0.0), seed=0, shape=(700, 900), noise=8.0, sat=False):
    """A (3, H, W) cube of the same star grid, Gaussian stars of the given sigma, shifted."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    img = np.full(shape, 500.0)
    stars = np.random.default_rng(1)        # same positions and brightnesses for every call
    for gy in range(40, shape[0] - 40, 60):
        for gx in range(40, shape[1] - 40, 60):
            y = gy + stars.uniform(-10, 10) + shift[0]
            x = gx + stars.uniform(-10, 10) + shift[1]
            amp = stars.uniform(600, 4000)
            img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    img += rng.normal(0, noise, shape)
    return np.stack([img, img, img]).astype(np.float32)


def test_single_star_fit_recovers_the_true_fwhm():
    yy, xx = np.mgrid[:60, :60]
    img = np.random.default_rng(0).normal(0, 5, (60, 60)) + 300 * np.exp(-((xx - 30.3) ** 2 + (yy - 29.6) ** 2) / (2 * 2.0 ** 2))
    fwhm, amp, peak, x0, y0 = cs._gauss_fit(img, 30, 30)
    assert fwhm == pytest.approx(K * 2.0, abs=0.06)
    assert (x0, y0) == pytest.approx((30.3, 29.6), abs=0.1)


def test_reports_the_true_width_ratio_between_two_stacks():
    a = _field(sigma=2.0, seed=2)
    b = _field(sigma=2.6, shift=(4.0, -3.0), seed=3)         # softer and offset: coordinates must be mapped
    r = cs.common_star_fwhm(a, b)
    assert r is not None and r['n'] >= 60
    assert r['a'] == pytest.approx(K * 2.0, rel=0.03)
    assert r['b'] == pytest.approx(K * 2.6, rel=0.03)
    assert r['ratio_a_over_b'] == pytest.approx(2.0 / 2.6, rel=0.03)


def test_identical_stacks_give_a_ratio_of_one():
    a = _field(sigma=2.2, seed=4)
    r = cs.common_star_fwhm(a, a.copy())
    assert r['ratio_a_over_b'] == pytest.approx(1.0, abs=0.01)


def test_ratio_does_not_depend_on_which_stack_has_more_detections():
    """The old benchmark's flaw: extra faint detections in one stack moved its 'FWHM'. Here one
    stack has 10x the noise (so a different star list is detected) at the same true width."""
    clean = _field(sigma=2.3, seed=5, noise=4.0)
    noisy = _field(sigma=2.3, shift=(2.0, 1.0), seed=6, noise=40.0)
    r = cs.common_star_fwhm(clean, noisy)
    assert r is not None
    assert r['ratio_a_over_b'] == pytest.approx(1.0, abs=0.05)


def test_returns_none_when_there_is_nothing_to_compare():
    empty = np.full((3, 300, 300), 500.0, dtype=np.float32)
    assert cs.common_star_fwhm(empty, empty) is None
