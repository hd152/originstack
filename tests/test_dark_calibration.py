"""Calibration with an inconsistent dark library.

A real Origin session (Whirlpool, 2026-07-01) came out magenta and mottled because its folder held a
1 s dark whose bias pedestal was already removed (median 16 ADU against a bias of 4,989), 10 s and 15 s
darks, and 20 s lights. ``--auto`` fitted a temperature model to all of them as if they were one
exposure, evaluated it 4-6 C outside the fitted range (-> a master dark of 0 ADU), scaled the "1 s dark"
by 20, and the calibration's ``+ scale * bias`` term added 100,000 ADU to every pixel.
"""
import argparse
import os

import numpy as np
import pytest
from astropy.io import fits

from src.cli import _build_masters, _guard_dark_calibration, _reconcile_dark_pedestal
from src.dark_temp_model import (
    MAX_EXTRAPOLATION_C,
    build_dark_temperature_model,
    model_unusable_reason,
    sample_dark_at_temperature,
    sample_unusable_reason,
)
from src.debayer import calibrate_frame
from src.frame_discovery import select_matching_darks
from src.models import FrameInfo

SHAPE = (16, 16)
BIAS = 4989.0


def _frame(tmp_path, name, level, exptime, temp, iso=None, kind='dark', seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    data = (level + rng.normal(0, noise, SHAPE)).astype(np.float32)
    path = str(tmp_path / name)
    hdr = fits.Header()
    hdr['EXPTIME'] = exptime
    hdr['CCD-TEMP'] = temp
    fits.writeto(path, data, header=hdr, overwrite=True)
    header = {'EXPTIME': exptime, 'CCD-TEMP': temp, 'NAXIS1': SHAPE[1], 'NAXIS2': SHAPE[0]}
    if iso is not None:
        header['ISO'] = iso
    return FrameInfo(path=path, type=kind, header=header)


# ── the dark temperature model uses one exposure, and will not extrapolate far ──


class TestModelUsesOneExposure:
    def _library(self, tmp_path):
        """What the session had: a 1 s dark without pedestal, three 10 s and three 15 s darks."""
        return [
            _frame(tmp_path, 'd1.fits', 16, 1.0, 27.2, seed=1),
            _frame(tmp_path, 'd2.fits', 4962, 10.0, 28.2, seed=2),
            _frame(tmp_path, 'd3.fits', 4917, 10.0, 28.9, seed=3),
            _frame(tmp_path, 'd4.fits', 4890, 10.0, 29.5, seed=4),
            _frame(tmp_path, 'd5.fits', 4957, 15.0, 29.9, seed=5),
            _frame(tmp_path, 'd6.fits', 4905, 15.0, 30.2, seed=6),
            _frame(tmp_path, 'd7.fits', 4902, 15.0, 28.5, seed=7),
            _frame(tmp_path, 'd8.fits', 4930, 15.0, 29.0, seed=8),
        ]

    def test_only_the_exposure_nearest_the_lights_is_fitted(self, tmp_path):
        model = build_dark_temperature_model(self._library(tmp_path), degree=1, light_exposure=20.0)
        assert model is not None
        assert model['exposure'] == pytest.approx(15.0) and model['n_frames'] == 4
        lo, hi = model['median_range']
        assert 4890 < lo and hi < 4960            # the 1 s dark (16 ADU) did not enter

    def test_without_a_light_exposure_the_larger_group_wins(self, tmp_path):
        frames = self._library(tmp_path)
        model = build_dark_temperature_model(frames, degree=1)
        assert model['exposure'] == pytest.approx(15.0)       # 4 frames against 3 and 1

    def test_no_exposure_with_enough_frames_gives_no_model(self, tmp_path):
        frames = [_frame(tmp_path, 'a.fits', 16, 1.0, 20.0), _frame(tmp_path, 'b.fits', 4900, 10.0, 25.0),
                  _frame(tmp_path, 'c.fits', 4900, 15.0, 30.0), _frame(tmp_path, 'd.fits', 4900, 20.0, 35.0)]
        assert build_dark_temperature_model(frames, degree=1) is None

    def test_mixed_exposures_are_named_in_the_log(self, tmp_path, capsys):
        build_dark_temperature_model(self._library(tmp_path), degree=1, light_exposure=20.0)
        out = capsys.readouterr().out
        assert 'different exposures' in out and '15 s' in out and '1 s' in out and '10 s' in out

    def test_darks_without_an_exposure_header_still_form_one_group(self, tmp_path):
        frames = []
        for i, t in enumerate((0.0, 5.0, 10.0, 15.0)):
            f = _frame(tmp_path, f'n{i}.fits', 100 + 5 * t, 1.0, t, seed=i)
            f.header.pop('EXPTIME')
            frames.append(f)
        model = build_dark_temperature_model(frames, degree=1)
        assert model is not None and model['exposure'] is None and model['n_frames'] == 4


class TestModelIsNotUsedFarOutsideItsRange:
    MODEL = {'temp_range': (28.5, 30.2), 'median_range': (4902.0, 4957.0)}

    def test_within_the_margin_is_fine(self):
        assert model_unusable_reason(self.MODEL, 27.0) is None
        assert model_unusable_reason(self.MODEL, 28.5 - MAX_EXTRAPOLATION_C) is None

    def test_the_session_in_question_is_refused(self):
        why = model_unusable_reason(self.MODEL, 24.2)
        assert why and '24.2 C' in why and '28.5-30.2' in why

    def test_a_sample_far_from_the_fitted_levels_is_refused(self):
        zero = np.zeros(SHAPE, np.float32)
        junk = np.full(SHAPE, 18235.0, np.float32)
        fine = np.full(SHAPE, 4930.0, np.float32)
        assert 'comes out at 0 ADU' in sample_unusable_reason(self.MODEL, zero)
        assert '18235' in sample_unusable_reason(self.MODEL, junk)
        assert sample_unusable_reason(self.MODEL, fine) is None
        assert sample_unusable_reason({'temp_range': (0, 1)}, zero) is None      # no levels recorded

    def test_sampling_itself_still_extrapolates_with_a_warning(self, capsys):
        model = {'coeffs': np.zeros((2, 4, 4)), 'degree': 1, 'shape': (4, 4), 'temp_range': (0.0, 10.0)}
        assert sample_dark_at_temperature(model, 50.0).shape == (4, 4)
        assert 'extrapolation' in capsys.readouterr().out


# ── choosing a dark ─────────────────────────────────────────────────────


class TestDarkSelectionDoesNotScaleAShortDarkUp:
    def _lights(self, n=5):
        return [FrameInfo(path=f'l{i}.fits', type='light', header={
            'EXPTIME': 20.0, 'ISO': 500, 'CCD-TEMP': 24.2, 'NAXIS1': 16, 'NAXIS2': 16}) for i in range(n)]

    def _dark(self, name, exp, temp, iso):
        return FrameInfo(path=name, type='dark', header={
            'EXPTIME': exp, 'ISO': iso, 'CCD-TEMP': temp, 'NAXIS1': 16, 'NAXIS2': 16})

    def test_a_close_exposure_beats_a_close_temperature(self):
        darks = [self._dark('1s.fits', 1.0, 27.2, 2000),               # nearest in temperature
                 self._dark('15a.fits', 15.0, 29.9, 100), self._dark('15b.fits', 15.0, 30.2, 200),
                 self._dark('15c.fits', 15.0, 28.5, 200)]
        names = [os.path.basename(f.path) for f in select_matching_darks(self._lights(), darks)]
        assert '1s.fits' not in names and all(n.startswith('15') for n in names)

    def test_scaling_a_dark_up_by_a_small_factor_is_still_fine(self):
        darks = [self._dark('15.fits', 15.0, 24.0, 500), self._dark('30.fits', 30.0, 24.0, 500)]
        names = [os.path.basename(f.path) for f in select_matching_darks(self._lights(), darks)]
        assert names == ['15.fits', '30.fits'] or names == ['15.fits'] or names == ['30.fits']

    def test_a_single_short_dark_is_still_used_when_it_is_all_there_is(self):
        darks = [self._dark('1s.fits', 1.0, 24.0, 500)]
        assert len(select_matching_darks(self._lights(), darks)) == 1


# ── the bias pedestal ───────────────────────────────────────────────────


class TestDarkWithoutPedestal:
    def _masters(self, dark_level):
        return {'bias': np.full(SHAPE, BIAS, np.float32), 'dark': np.full(SHAPE, dark_level, np.float32)}

    def test_a_dark_without_pedestal_gets_the_bias_added(self, capsys):
        m = self._masters(16.0)
        _reconcile_dark_pedestal(m)
        assert float(np.median(m['dark'])) == pytest.approx(BIAS + 16.0)
        assert m['dark_pedestal_added'] and 'no bias pedestal' in capsys.readouterr().out

    def test_a_normal_dark_is_left_alone(self):
        m = self._masters(4939.0)
        before = m['dark'].copy()
        _reconcile_dark_pedestal(m)
        assert np.array_equal(m['dark'], before) and 'dark_pedestal_added' not in m

    def test_a_long_dark_with_lots_of_dark_current_is_left_alone(self):
        m = self._masters(9000.0)
        _reconcile_dark_pedestal(m)
        assert float(np.median(m['dark'])) == 9000.0

    def test_without_a_bias_nothing_is_done(self):
        m = {'bias': None, 'dark': np.full(SHAPE, 16.0, np.float32)}
        _reconcile_dark_pedestal(m)
        assert float(np.median(m['dark'])) == 16.0

    def test_the_calibration_no_longer_adds_a_hundred_thousand_adu(self):
        """The failure itself: 20 s lights, a 1 s dark with no pedestal, scale 20."""
        raw = np.full(SHAPE, BIAS + 5000.0, np.float32)           # pedestal plus 5,000 ADU of sky
        bias = np.full(SHAPE, BIAS, np.float32)
        dark = np.full(SHAPE, 16.0, np.float32)
        broken, _ = calibrate_frame(raw.copy(), bias, dark, 20.0, None)
        assert float(np.median(broken)) > 90_000                  # what the stack saw
        masters = {'bias': bias, 'dark': dark}
        _reconcile_dark_pedestal(masters)
        fixed, _ = calibrate_frame(raw.copy(), bias, masters['dark'], 20.0, None)
        assert float(np.median(fixed)) == pytest.approx(5000.0 - 20.0 * 16.0, abs=1.0)


# ── the safety net ──────────────────────────────────────────────────────


class TestGuard:
    def _lights(self, tmp_path, level=BIAS + 5200.0, exptime=20.0):
        return [_frame(tmp_path, 'light.fits', level, exptime, 24.2, kind='light')]

    def _masters(self, dark_level, dark_exp=15.0):
        return {'bias': np.full(SHAPE, BIAS, np.float32), 'dark': np.full(SHAPE, dark_level, np.float32),
                'dark_exptime': dark_exp}

    def test_a_dark_that_removes_the_frame_is_dropped(self, tmp_path, capsys):
        m = self._masters(18235.0, dark_exp=10.0)                 # the model's 18 k junk, scale 2
        _guard_dark_calibration(m, self._lights(tmp_path))
        assert m['dark'] is None and m['dark_exptime'] is None and m['dark_dropped']
        assert 'Calibrating without a dark' in capsys.readouterr().out

    def test_a_dark_that_adds_a_huge_offset_is_dropped(self, tmp_path):
        m = self._masters(0.0, dark_exp=1.0)                      # median 0 with a bias of 4,989, scale 20
        _guard_dark_calibration(m, self._lights(tmp_path))
        assert m['dark'] is None

    def test_a_normal_dark_is_kept(self, tmp_path, capsys):
        m = self._masters(4939.0, dark_exp=15.0)
        _guard_dark_calibration(m, self._lights(tmp_path))
        assert m['dark'] is not None and 'WARNING' not in capsys.readouterr().out

    def test_a_missing_light_never_stops_the_run(self):
        m = self._masters(18235.0)
        _guard_dark_calibration(m, [FrameInfo(path='does-not-exist.fits', type='light', header={})])
        assert m['dark'] is not None
        _guard_dark_calibration(m, [])

    def test_no_dark_means_nothing_to_guard(self, tmp_path):
        m = {'bias': None, 'dark': None}
        _guard_dark_calibration(m, self._lights(tmp_path))
        assert m['dark'] is None and 'dark_dropped' not in m


# ── end to end through _build_masters ───────────────────────────────────


class TestBuildMastersOnTheMessyLibrary:
    def _frames(self, tmp_path):
        lights = [_frame(tmp_path, f'light{i}.fits', BIAS + 5200, 20.0, 24.2, iso=500, kind='light', seed=i)
                  for i in range(5)]
        biases = [_frame(tmp_path, f'bias{i}.fits', BIAS, 0.001, 28.0, kind='bias', seed=10 + i)
                  for i in range(3)]
        darks = [_frame(tmp_path, 'dark1.fits', 16, 1.0, 27.2, iso=2000, seed=20),
                 _frame(tmp_path, 'dark2.fits', 4962, 10.0, 28.2, iso=100, seed=21),
                 _frame(tmp_path, 'dark3.fits', 4917, 10.0, 28.9, iso=200, seed=22),
                 _frame(tmp_path, 'dark4.fits', 4890, 10.0, 29.5, iso=2000, seed=23),
                 _frame(tmp_path, 'dark5.fits', 4957, 15.0, 29.9, iso=100, seed=24),
                 _frame(tmp_path, 'dark6.fits', 4905, 15.0, 30.2, iso=200, seed=25),
                 _frame(tmp_path, 'dark7.fits', 4902, 15.0, 28.5, iso=200, seed=26)]
        return {'light': lights, 'bias': biases, 'dark': darks, 'flat': []}

    def _args(self, **kw):
        base = dict(auto=True, master_method='median', dark_temp_model=False, flat_from_lights=False,
                    _explicit_cli_dests=set())
        base.update(kw)
        return argparse.Namespace(**base)

    def _calibrated_sky(self, masters, frames):
        raw = np.full(SHAPE, BIAS + 5200.0, np.float32)
        scale = 20.0 / masters['dark_exptime'] if masters.get('dark') is not None else 1.0
        out, _ = calibrate_frame(raw, masters['bias'], masters['dark'], scale, None)
        return float(np.median(out))

    def test_auto_run_no_longer_adds_a_pedestal(self, tmp_path, capsys):
        """--auto enables the temperature model with these darks (>= 3 distinct temperatures)."""
        frames = self._frames(tmp_path)
        masters = _build_masters(frames, args=self._args())
        sky = self._calibrated_sky(masters, frames)
        assert 3000 < sky < 8000, f'calibrated sky {sky}: the real sky is about 5,200 ADU'
        out = capsys.readouterr().out
        assert 'Dark temperature model not used' in out or 'Master dark:' in out

    def test_the_old_picks_are_gone(self, tmp_path):
        frames = self._frames(tmp_path)
        masters = _build_masters(frames, args=self._args(auto=False))
        assert masters['dark_exptime'] == pytest.approx(15.0)       # the 15 s darks, not the 1 s one
        sky = self._calibrated_sky(masters, frames)
        assert 3000 < sky < 8000

    def test_a_library_of_only_the_pedestal_less_dark_is_corrected(self, tmp_path, capsys):
        frames = self._frames(tmp_path)
        frames['dark'] = frames['dark'][:1]
        masters = _build_masters(frames, args=self._args(auto=False))
        sky = self._calibrated_sky(masters, frames)
        assert 3000 < sky < 8000
        assert 'no bias pedestal' in capsys.readouterr().out
