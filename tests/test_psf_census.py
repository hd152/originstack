"""tools/psf_census.py: the lucky-patch statistics and verdicts."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import psf_census as pc  # noqa: E402


def _matrix(rng, n=40, m=4, seeing=0.15, static=0.0, lucky=0.0, meas=0.02):
    base = 4.4 * (1 + seeing * rng.standard_normal((n, 1)))
    stat = 1 + static * np.array([0.0, 1.0, 0.5, -0.5])[:m][None, :]
    inter = 1 + lucky * rng.standard_normal((n, m))
    truth = base * stat * inter
    noise_half = meas * 4.4 * rng.standard_normal((n, m)) * np.sqrt(2.0)     # a - b of two half estimates
    obs = truth + 0.5 * meas * 4.4 * rng.standard_normal((n, m))
    return obs, noise_half * 1.0


def test_pure_seeing_and_noise_gives_no_excess():
    rng = np.random.default_rng(0)
    F, nd = _matrix(rng)
    st = pc.census_statistics(F, nd)
    assert st['excess_rel'] < 0.02
    assert st['frame_scatter_rel'] > 0.08                       # the seeing varies a lot...
    assert 'no lucky-patch signal' in pc.verdict(st)[0]         # ...and is removed first


def test_frame_by_cell_variation_is_detected():
    rng = np.random.default_rng(1)
    F, nd = _matrix(rng, lucky=0.06)
    st = pc.census_statistics(F, nd)
    assert 0.04 < st['excess_rel'] < 0.09
    assert 'lucky-patch signal present' in pc.verdict(st)[0]


def test_static_field_shape_is_reported_not_counted_as_excess():
    rng = np.random.default_rng(2)
    F, nd = _matrix(rng, static=0.04)
    st = pc.census_statistics(F, nd)
    assert st['static_max_over_min'] > 1.05 and st['excess_rel'] < 0.02
    assert 'a position-dependent PSF would help' in pc.verdict(st)[1]


def test_gaps_and_short_sessions():
    rng = np.random.default_rng(3)
    F, nd = _matrix(rng)
    F[3, 1] = np.nan
    assert pc.census_statistics(F, nd)['n_frames'] == 39        # the frame with a gap is dropped
    assert np.isnan(pc.census_statistics(F[:4], nd[:4])['excess_rel'])
    assert 'not enough' in pc.verdict({'excess_rel': float('nan')})[0]
