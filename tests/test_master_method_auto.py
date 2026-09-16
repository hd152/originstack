"""Tests for --auto's narrow robust_pca master-combine auto-enable
(src/cli.py::_build_masters).

Gated per calibration type (bias/dark/flat counts often differ) on frame
count in [Config.ROBUST_PCA_MIN_FRAMES, Config.ROBUST_PCA_AUTO_MAX_FRAMES] --
below that floor robust_pca is underdetermined anyway (make_master's own
existing fallback), above the ceiling it's too slow to silently add to
--auto's runtime (see _build_masters's comment: benchmarked ~21min at N=20).
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

from src.cli import _build_masters
from src.models import Config


def _frames(n: int):
    return [SimpleNamespace(header={}, path=f'f{i}.fits') for i in range(n)]


def _args(**overrides):
    base = dict(auto=True, master_method='median', _explicit_cli_dests=set())
    base.update(overrides)
    return argparse.Namespace(**base)


class TestRobustPcaAutoMasterMethod:

    def test_auto_upgrades_bias_in_band(self):
        n = Config.ROBUST_PCA_MIN_FRAMES + 1
        assert n <= Config.ROBUST_PCA_AUTO_MAX_FRAMES
        frames = {'light': [], 'dark': [], 'flat': [], 'bias': _frames(n)}
        with patch('src.cli.make_master', return_value=None) as mm:
            _build_masters(frames, args=_args())
        mm.assert_called_once()
        assert mm.call_args.kwargs['method'] == 'robust_pca'

    def test_stays_median_below_min_frames(self):
        n = Config.ROBUST_PCA_MIN_FRAMES - 1
        frames = {'light': [], 'dark': [], 'flat': [], 'bias': _frames(n)}
        with patch('src.cli.make_master', return_value=None) as mm:
            _build_masters(frames, args=_args())
        assert mm.call_args.kwargs['method'] == 'median'

    def test_stays_median_above_auto_ceiling(self):
        n = Config.ROBUST_PCA_AUTO_MAX_FRAMES + 5
        frames = {'light': [], 'dark': [], 'flat': [], 'bias': _frames(n)}
        with patch('src.cli.make_master', return_value=None) as mm:
            _build_masters(frames, args=_args())
        assert mm.call_args.kwargs['method'] == 'median'

    def test_no_auto_stays_median_in_band(self):
        n = Config.ROBUST_PCA_MIN_FRAMES + 1
        frames = {'light': [], 'dark': [], 'flat': [], 'bias': _frames(n)}
        with patch('src.cli.make_master', return_value=None) as mm:
            _build_masters(frames, args=_args(auto=False))
        assert mm.call_args.kwargs['method'] == 'median'

    def test_explicit_master_method_wins(self):
        n = Config.ROBUST_PCA_MIN_FRAMES + 1
        frames = {'light': [], 'dark': [], 'flat': [], 'bias': _frames(n)}
        with patch('src.cli.make_master', return_value=None) as mm:
            _build_masters(frames, args=_args(_explicit_cli_dests={'master_method'}))
        assert mm.call_args.kwargs['method'] == 'median'

    def test_explicit_robust_pca_used_regardless_of_band(self):
        n = Config.ROBUST_PCA_AUTO_MAX_FRAMES + 20  # well above the auto ceiling
        frames = {'light': [], 'dark': [], 'flat': [], 'bias': _frames(n)}
        with patch('src.cli.make_master', return_value=None) as mm:
            _build_masters(frames, args=_args(master_method='robust_pca',
                                              _explicit_cli_dests={'master_method'}))
        assert mm.call_args.kwargs['method'] == 'robust_pca'

    def test_per_type_gating_bias_small_dark_large(self):
        """A small bias count and a large dark count in the same session --
        each type's own count must be judged independently, not one
        globally-picked method applied to both."""
        n_small = Config.ROBUST_PCA_MIN_FRAMES + 1
        n_large = Config.ROBUST_PCA_AUTO_MAX_FRAMES + 20
        frames = {'light': [], 'dark': _frames(n_large), 'flat': [], 'bias': _frames(n_small)}
        with patch('src.cli.make_master', return_value=None) as mm, \
             patch('src.cli.select_matching_darks', side_effect=lambda lights, d: d):
            _build_masters(frames, args=_args())
        calls_by_len = {len(c.args[0]): c.kwargs['method'] for c in mm.call_args_list}
        assert calls_by_len[n_small] == 'robust_pca'
        assert calls_by_len[n_large] == 'median'
