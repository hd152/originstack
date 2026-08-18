"""Tests for --flat-from-lights (src/cli.py::_build_masters).

Derives a synthetic flat/vignetting map from the light frames themselves via
robust_pca_master when no dedicated flat frames were discovered -- reuses the
exact low-rank/sparse decomposition --master-method robust_pca already uses
for real calibration frames, just pointed at a different frame population.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.cli import _build_masters
from src.models import Config


def _frames(n: int):
    return [SimpleNamespace(header={}, path=f'light{i}.fits') for i in range(n)]


def _args(**overrides):
    base = dict(auto=False, master_method='median', flat_from_lights=True,
                _explicit_cli_dests=set())
    base.update(overrides)
    return argparse.Namespace(**base)


class TestFlatFromLights:

    def test_builds_synthetic_flat_when_none_discovered(self):
        n = Config.ROBUST_PCA_MIN_FRAMES + 2
        frames = {'light': _frames(n), 'dark': [], 'flat': [], 'bias': []}
        fake_flat = np.full((8, 8), 500.0, dtype=np.float32)
        with patch('src.cli.make_master', return_value=fake_flat) as mm:
            masters = _build_masters(frames, args=_args())
        mm.assert_called_once()
        assert mm.call_args.kwargs['method'] == 'robust_pca'
        # masters['flat'] is Gaussian-smoothed per Bayer position after this
        # (existing _build_masters behavior) -- same value here since fake_flat
        # is constant, but not the same object.
        np.testing.assert_allclose(masters['flat'], fake_flat)

    def test_skipped_when_real_flats_exist(self):
        n = Config.ROBUST_PCA_MIN_FRAMES + 2
        frames = {'light': _frames(n), 'dark': [], 'flat': _frames(3), 'bias': []}
        with patch('src.cli.make_master', return_value=np.zeros((4, 4), dtype=np.float32)) as mm, \
             patch('src.cli.select_matching_flats', side_effect=lambda lights, f: f):
            _build_masters(frames, args=_args())
        # Only the real-flat call site should have fired, with the real 3 flat frames.
        mm.assert_called_once()
        assert len(mm.call_args.args[0]) == 3

    def test_skipped_when_flag_off(self):
        n = Config.ROBUST_PCA_MIN_FRAMES + 2
        frames = {'light': _frames(n), 'dark': [], 'flat': [], 'bias': []}
        with patch('src.cli.make_master') as mm:
            masters = _build_masters(frames, args=_args(flat_from_lights=False))
        mm.assert_not_called()
        assert masters['flat'] is None

    def test_skipped_below_min_frames(self):
        n = Config.ROBUST_PCA_MIN_FRAMES - 1
        frames = {'light': _frames(n), 'dark': [], 'flat': [], 'bias': []}
        with patch('src.cli.make_master') as mm:
            masters = _build_masters(frames, args=_args())
        mm.assert_not_called()
        assert masters['flat'] is None

    def test_sample_capped_at_auto_max_frames(self):
        n = Config.ROBUST_PCA_AUTO_MAX_FRAMES * 3
        frames = {'light': _frames(n), 'dark': [], 'flat': [], 'bias': []}
        with patch('src.cli.make_master', return_value=np.zeros((4, 4), dtype=np.float32)) as mm:
            _build_masters(frames, args=_args())
        sample = mm.call_args.args[0]
        assert len(sample) <= Config.ROBUST_PCA_AUTO_MAX_FRAMES

    def test_none_result_leaves_flat_none(self):
        n = Config.ROBUST_PCA_MIN_FRAMES + 2
        frames = {'light': _frames(n), 'dark': [], 'flat': [], 'bias': []}
        with patch('src.cli.make_master', return_value=None):
            masters = _build_masters(frames, args=_args())
        assert masters['flat'] is None
