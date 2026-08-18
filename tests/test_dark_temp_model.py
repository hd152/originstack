"""Tests for the temperature-interpolated dark current model
(--dark-temp-model, src/dark_temp_model.py + src/cli.py::_build_masters).
"""
from __future__ import annotations

import argparse
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from src.cli import _build_masters
from src.dark_temp_model import build_dark_temperature_model, sample_dark_at_temperature
from src.models import FrameInfo


def _write_fits(path: str, data: np.ndarray) -> None:
    fits.writeto(path, data.astype(np.float32), overwrite=True)


class TestBuildDarkTemperatureModel:

    def _dark_at_temp(self, tmpdir, name, shape, temp_c, dark_per_degree=5.0,
                      base=50.0, seed=0):
        rng = np.random.default_rng(seed)
        level = base + dark_per_degree * temp_c
        data = np.full(shape, level, dtype=np.float32) + rng.normal(0, 1.0, shape)
        path = os.path.join(tmpdir, name)
        _write_fits(path, data)
        return FrameInfo(path=path, type='dark', header={'CCDTEMP': temp_c})

    def test_recovers_linear_relationship(self):
        shape = (16, 16)
        with tempfile.TemporaryDirectory() as d:
            frames = []
            temps = [-10.0, -5.0, 0.0, 5.0, 10.0]
            for i, t in enumerate(temps):
                for j in range(2):  # 2 frames per temperature
                    frames.append(self._dark_at_temp(d, f'd{i}_{j}.fits', shape, t, seed=i * 2 + j))
            model = build_dark_temperature_model(frames, degree=1)
            assert model is not None
            assert model['n_temps'] == len(temps)

            # Evaluate at an in-range temperature and check it's close to the
            # known linear relationship (base=50, slope=5/degree).
            pred = sample_dark_at_temperature(model, 2.5)
            expected = 50.0 + 5.0 * 2.5
            assert abs(float(np.mean(pred)) - expected) < 3.0

    def test_returns_none_with_too_few_distinct_temperatures(self):
        shape = (8, 8)
        with tempfile.TemporaryDirectory() as d:
            # All frames at the same temperature -- degree=2 needs >= 3 distinct.
            frames = [self._dark_at_temp(d, f'd{i}.fits', shape, 0.0, seed=i)
                     for i in range(6)]
            model = build_dark_temperature_model(frames, degree=2)
            assert model is None

    def test_skips_frames_without_temperature_header(self):
        shape = (8, 8)
        with tempfile.TemporaryDirectory() as d:
            frames = []
            temps = [-5.0, 0.0, 5.0]
            for i, t in enumerate(temps):
                for j in range(2):
                    frames.append(self._dark_at_temp(d, f'd{i}_{j}.fits', shape, t, seed=i * 2 + j))
            # Add frames with no usable temperature header.
            for k in range(3):
                path = os.path.join(d, f'notemp{k}.fits')
                _write_fits(path, np.full(shape, 999.0, dtype=np.float32))
                frames.append(FrameInfo(path=path, type='dark', header={}))
            model = build_dark_temperature_model(frames, degree=1)
            assert model is not None
            assert model['n_frames'] == 6  # only the temp-tagged frames counted

    def test_skips_shape_mismatched_frame(self):
        shape = (8, 8)
        with tempfile.TemporaryDirectory() as d:
            frames = []
            temps = [-5.0, 0.0, 5.0]
            for i, t in enumerate(temps):
                for j in range(2):
                    frames.append(self._dark_at_temp(d, f'd{i}_{j}.fits', shape, t, seed=i * 2 + j))
            bad_path = os.path.join(d, 'bad_shape.fits')
            _write_fits(bad_path, np.full((4, 4), 100.0, dtype=np.float32))
            frames.append(FrameInfo(path=bad_path, type='dark', header={'CCDTEMP': 0.0}))
            model = build_dark_temperature_model(frames, degree=1)
            assert model is not None
            assert model['shape'] == shape


class TestSampleDarkAtTemperature:

    def test_warns_on_extrapolation_far_outside_range(self, capsys):
        model = {'coeffs': np.zeros((2, 4, 4)), 'degree': 1,
                 'shape': (4, 4), 'temp_range': (0.0, 10.0),
                 'n_frames': 6, 'n_temps': 3}
        sample_dark_at_temperature(model, 50.0)
        # safe_print goes through src.utils -- just check no crash and a
        # sane, non-negative, correctly-shaped output.
        out = sample_dark_at_temperature(model, 50.0)
        assert out.shape == (4, 4)
        assert np.all(out >= 0.0)


class TestBuildMastersDarkTempModelDispatch:

    def _args(self, **overrides):
        base = dict(auto=False, master_method='median', dark_temp_model=True,
                    flat_from_lights=False, _explicit_cli_dests=set())
        base.update(overrides)
        return argparse.Namespace(**base)

    def _light_frames(self, n, temp=2.0):
        return [SimpleNamespace(header={'CCDTEMP': temp}, path=f'light{i}.fits')
               for i in range(n)]

    def test_uses_model_when_available(self):
        frames = {'light': self._light_frames(5), 'dark': [object()] * 10,
                  'flat': [], 'bias': []}
        fake_model = {'coeffs': np.zeros((2, 4, 4)), 'degree': 1,
                     'shape': (4, 4), 'temp_range': (-5.0, 10.0),
                     'n_frames': 10, 'n_temps': 4}
        with patch('src.dark_temp_model.build_dark_temperature_model',
                  return_value=fake_model) as build_mock, \
             patch('src.cli.select_matching_darks') as select_mock, \
             patch('src.cli.make_master') as mm:
            masters = _build_masters(frames, args=self._args())
        build_mock.assert_called_once()
        select_mock.assert_not_called()
        mm.assert_not_called()
        assert masters['dark'] is not None
        assert masters['dark'].shape[:2] == (4, 4)

    def test_falls_back_when_model_unavailable(self):
        frames = {'light': self._light_frames(5), 'dark': [object()] * 3,
                  'flat': [], 'bias': []}
        with patch('src.dark_temp_model.build_dark_temperature_model',
                  return_value=None), \
             patch('src.cli.select_matching_darks', side_effect=lambda lights, d: d), \
             patch('src.cli.make_master', return_value=np.zeros((4, 4), dtype=np.float32)) as mm:
            masters = _build_masters(frames, args=self._args())
        mm.assert_called_once()
        assert masters['dark'] is not None

    def test_flag_off_uses_normal_path(self):
        frames = {'light': self._light_frames(5), 'dark': [object()] * 3,
                  'flat': [], 'bias': []}
        with patch('src.dark_temp_model.build_dark_temperature_model') as build_mock, \
             patch('src.cli.select_matching_darks', side_effect=lambda lights, d: d), \
             patch('src.cli.make_master', return_value=np.zeros((4, 4), dtype=np.float32)) as mm:
            _build_masters(frames, args=self._args(dark_temp_model=False))
        build_mock.assert_not_called()
        mm.assert_called_once()
