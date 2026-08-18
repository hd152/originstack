"""Tests for the dither-coverage uniformity diagnostic (--dither-report)."""
from __future__ import annotations

import numpy as np

from src.dither_report import compute_dither_coverage


class TestComputeDitherCoverage:

    def test_empty_shifts_returns_zero_frames(self):
        stats = compute_dither_coverage([])
        assert stats['n_frames'] == 0
        assert stats['grid_counts'] is None

    def test_uniform_random_dither_scores_high_uniformity(self):
        rng = np.random.default_rng(0)
        shifts = [(float(dy), float(dx)) for dy, dx in
                 rng.uniform(0.0, 5.0, (500, 2))]
        stats = compute_dither_coverage(shifts, grid=8)
        assert stats['n_frames'] == 500
        assert stats['uniformity'] > 0.5
        assert stats['empty_frac'] < 0.2

    def test_identical_shifts_score_low_uniformity(self):
        # Every frame lands in exactly the same sub-pixel bin -- the worst
        # possible dither coverage.
        shifts = [(1.0, 1.0)] * 50
        stats = compute_dither_coverage(shifts, grid=8)
        assert stats['n_frames'] == 50
        assert stats['uniformity'] < 0.2
        assert stats['empty_frac'] > 0.9  # only 1 of 64 bins ever touched

    def test_drizzle_scale_affects_phase_binning(self):
        # (0.5, 0.5) at scale=1 sits at phase 0.5; at scale=2 it wraps to
        # phase 0.0 -- the binning must actually use the supplied scale.
        shifts = [(0.5, 0.5)] * 20
        stats_scale1 = compute_dither_coverage(shifts, grid=4, scale=1.0)
        stats_scale2 = compute_dither_coverage(shifts, grid=4, scale=2.0)
        peak1 = np.unravel_index(np.argmax(stats_scale1['grid_counts']),
                                 stats_scale1['grid_counts'].shape)
        peak2 = np.unravel_index(np.argmax(stats_scale2['grid_counts']),
                                 stats_scale2['grid_counts'].shape)
        assert peak1 != peak2

    def test_grid_counts_sum_to_frame_count(self):
        rng = np.random.default_rng(1)
        shifts = [(float(dy), float(dx)) for dy, dx in rng.uniform(0, 3, (30, 2))]
        stats = compute_dither_coverage(shifts, grid=8)
        assert int(stats['grid_counts'].sum()) == 30
