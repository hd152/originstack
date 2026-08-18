"""Tests for --auto wiring of this session's new features:
denoise_curvelet (emission/reflection nebula), variance_stabilize,
drizzle_kernel=magic, hdr_blend_mode=fusion, color_calibrate_method=spcc
(src/auto_settings.py), and flat_from_lights / dark_temp_model auto-trigger
(src/cli.py::_build_masters).
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src import auto_settings as a
from src.cli import _build_masters
from src.models import Config


def _args(**overrides):
    base = dict(
        _explicit_cli_dests=set(), stack_method='auto', deconvolve=True,
        auto_denoise_strength=True, debayer_method='malvar',
        denoise_mmt=False, denoise_acdnr=False, denoise=False,
        denoise_curvelet=False, denoise_bm3d=False, deconvolve_tv=False,
        patch_registration=False, consensus_ref=False, preview_black_sigma=0.0,
        variance_stabilize=False, drizzle_scale=1.0, drizzle_kernel='lanczos3',
        hdr_combine=None, hdr_blend_mode='threshold',
        color_calibrate=False, color_calibrate_method='colorindex',
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestCurveletForFilamentTargets:

    def test_emission_nebula_anchor_prefers_curvelet_over_mmt(self):
        sig = dict(a._TYPE_ANCHORS['emission_nebula'])
        weights = a._blend_weights(sig)
        args = _args()
        a._apply_dynamic_settings(sig, weights, args)
        assert args.denoise_curvelet is True
        assert args.denoise_mmt is False

    def test_reflection_nebula_anchor_prefers_curvelet_over_mmt(self):
        sig = dict(a._TYPE_ANCHORS['reflection_nebula'])
        weights = a._blend_weights(sig)
        args = _args()
        a._apply_dynamic_settings(sig, weights, args)
        assert args.denoise_curvelet is True
        assert args.denoise_mmt is False

    def test_galaxy_anchor_still_prefers_mmt(self):
        # Galaxy wasn't switched -- only emission/reflection nebula were.
        # _apply_dynamic_settings alone can leave denoise_curvelet True too
        # (a tiny nonzero blend weight on emission/reflection nebula is
        # enough, since only those two presets define that attr at all --
        # same pre-existing behavior denoise_mmt itself would have if it
        # weren't also galaxy's own dominant preset). apply_auto_settings's
        # real pipeline always runs _apply_quality_settings' rule 14 right
        # after, which is what actually resolves the conflict -- so exercise
        # both, matching production usage, rather than half the pipeline.
        sig = dict(a._TYPE_ANCHORS['galaxy'])
        weights = a._blend_weights(sig)
        args = _args()
        a._apply_dynamic_settings(sig, weights, args)
        full_sig = {**sig, 'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                   'median_ellipticity': 0}
        a._apply_quality_settings(full_sig, args, weights=weights)
        assert args.denoise_mmt is True
        assert args.denoise_curvelet is False

    def test_explicit_denoise_curvelet_flag_wins(self):
        sig = dict(a._TYPE_ANCHORS['galaxy'])
        weights = a._blend_weights(sig)
        args = _args(denoise_curvelet=True, _explicit_cli_dests={'denoise_curvelet'})
        a._apply_dynamic_settings(sig, weights, args)
        assert args.denoise_curvelet is True  # not overwritten to False


class TestVarianceStabilizeRule:

    def test_enabled_when_wavelet_primary(self):
        args = _args(denoise=True)
        a._apply_quality_settings({'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                                   'median_ellipticity': 0}, args)
        assert args.variance_stabilize is True

    def test_enabled_when_curvelet_primary(self):
        args = _args(denoise_curvelet=True)
        a._apply_quality_settings({'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                                   'median_ellipticity': 0}, args)
        assert args.variance_stabilize is True

    def test_not_forced_when_neither_active(self):
        args = _args(denoise=False, denoise_curvelet=False, denoise_mmt=True)
        a._apply_quality_settings({'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                                   'median_ellipticity': 0}, args)
        assert args.variance_stabilize is False

    def test_explicit_false_respected(self):
        args = _args(denoise=True, variance_stabilize=False,
                     _explicit_cli_dests={'variance_stabilize'})
        a._apply_quality_settings({'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                                   'median_ellipticity': 0}, args)
        assert args.variance_stabilize is False


class TestDrizzleKernelUpgrade:

    def test_upgrades_to_magic_when_drizzling(self):
        args = _args(drizzle_scale=2.0)
        a._apply_quality_settings({'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                                   'median_ellipticity': 0}, args)
        assert args.drizzle_kernel == 'magic'

    def test_no_change_without_drizzling(self):
        args = _args(drizzle_scale=1.0)
        a._apply_quality_settings({'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                                   'median_ellipticity': 0}, args)
        assert args.drizzle_kernel == 'lanczos3'

    def test_explicit_psf_kernel_not_overridden(self):
        args = _args(drizzle_scale=2.0, drizzle_kernel='psf',
                     _explicit_cli_dests={'drizzle_kernel'})
        a._apply_quality_settings({'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                                   'median_ellipticity': 0}, args)
        assert args.drizzle_kernel == 'psf'


class TestHdrBlendModeUpgrade:

    def test_upgrades_to_fusion_when_hdr_combine_set(self):
        args = _args(hdr_combine='short.fits')
        a._apply_quality_settings({'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                                   'median_ellipticity': 0}, args)
        assert args.hdr_blend_mode == 'fusion'

    def test_no_change_without_hdr_combine(self):
        args = _args(hdr_combine=None)
        a._apply_quality_settings({'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                                   'median_ellipticity': 0}, args)
        assert args.hdr_blend_mode == 'threshold'


class TestColorCalibrateMethodUpgrade:

    def test_upgrades_to_spcc_when_color_calibrate_set(self):
        args = _args(color_calibrate=True)
        a._apply_quality_settings({'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                                   'median_ellipticity': 0}, args)
        assert args.color_calibrate_method == 'spcc'

    def test_no_change_without_color_calibrate(self):
        args = _args(color_calibrate=False)
        a._apply_quality_settings({'n_frames': 10, 'snr': 10, 'fwhm': 2.0,
                                   'star_count': 20, 'strehl': 0, 'dispersion': 0,
                                   'median_ellipticity': 0}, args)
        assert args.color_calibrate_method == 'colorindex'


# ---------------------------------------------------------------------------
# _build_masters auto-trigger for flat_from_lights / dark_temp_model
# ---------------------------------------------------------------------------

def _frames(n: int):
    return [SimpleNamespace(header={}, path=f'f{i}.fits') for i in range(n)]


def _master_args(**overrides):
    base = dict(auto=True, master_method='median', flat_from_lights=False,
                dark_temp_model=False, _explicit_cli_dests=set())
    base.update(overrides)
    return argparse.Namespace(**base)


class TestFlatFromLightsAutoTrigger:

    def test_auto_triggers_without_explicit_flag(self):
        n = Config.ROBUST_PCA_MIN_FRAMES + 2
        frames = {'light': _frames(n), 'dark': [], 'flat': [], 'bias': []}
        with patch('src.cli.make_master', return_value=np.zeros((4, 4), dtype=np.float32)) as mm:
            masters = _build_masters(frames, args=_master_args())
        mm.assert_called_once()
        assert mm.call_args.kwargs['method'] == 'robust_pca'
        assert masters['flat'] is not None

    def test_no_auto_no_trigger(self):
        n = Config.ROBUST_PCA_MIN_FRAMES + 2
        frames = {'light': _frames(n), 'dark': [], 'flat': [], 'bias': []}
        with patch('src.cli.make_master') as mm:
            masters = _build_masters(frames, args=_master_args(auto=False))
        mm.assert_not_called()
        assert masters['flat'] is None


class TestDarkTempModelAutoTrigger:

    def _dark_frame(self, temp):
        return SimpleNamespace(header={'CCDTEMP': temp}, path='d.fits')

    def _lights_with_temp(self, n, temp=0.0):
        # build_dark_temperature_model's caller also needs the LIGHT frames'
        # own temperature to evaluate the fitted model at -- _frames()'s
        # empty-header lights are fine for the flat-from-lights tests above
        # but not here.
        return [SimpleNamespace(header={'CCDTEMP': temp}, path=f'l{i}.fits')
               for i in range(n)]

    def test_auto_triggers_with_enough_distinct_temps(self):
        darks = [self._dark_frame(t) for t in [-10.0, -5.0, 0.0, 5.0, -10.0, -5.0]]
        frames = {'light': self._lights_with_temp(5), 'dark': darks, 'flat': [], 'bias': []}
        fake_model = {'coeffs': np.zeros((2, 4, 4)), 'degree': 1,
                     'shape': (4, 4), 'temp_range': (-10.0, 5.0),
                     'n_frames': 6, 'n_temps': 4}
        with patch('src.dark_temp_model.build_dark_temperature_model',
                  return_value=fake_model) as build_mock, \
             patch('src.cli.select_matching_darks') as select_mock:
            masters = _build_masters(frames, args=_master_args())
        build_mock.assert_called_once()
        select_mock.assert_not_called()
        assert masters['dark'] is not None

    def test_auto_does_not_trigger_with_too_few_distinct_temps(self):
        darks = [self._dark_frame(0.0) for _ in range(6)]  # all same temp
        frames = {'light': _frames(5), 'dark': darks, 'flat': [], 'bias': []}
        with patch('src.dark_temp_model.build_dark_temperature_model') as build_mock, \
             patch('src.cli.select_matching_darks', side_effect=lambda lights, d: d), \
             patch('src.cli.make_master', return_value=np.zeros((4, 4), dtype=np.float32)):
            _build_masters(frames, args=_master_args())
        build_mock.assert_not_called()

    def test_explicit_flag_still_works_regardless_of_temp_spread(self):
        darks = [self._dark_frame(0.0) for _ in range(6)]
        frames = {'light': self._lights_with_temp(5), 'dark': darks, 'flat': [], 'bias': []}
        with patch('src.dark_temp_model.build_dark_temperature_model',
                  return_value=None) as build_mock:
            _build_masters(frames, args=_master_args(dark_temp_model=True))
        build_mock.assert_called_once()
