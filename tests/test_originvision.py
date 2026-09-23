"""Tests for the originvision integration (src/originvision.py + src/originvision_infer.py):
in-process ONNX inference wiring, numpy/scipy preprocessing, and the
advisory-only session-relative flagging.

onnxruntime may or may not be installed in the test environment. Tests that
need a real forward pass skip when it (or the bundled model) is absent;
everything else mocks ``run_originvision_infer`` / ``_originvision_model`` so it
never depends on a real model.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import src.originvision as originvision_mod
import src.originvision_infer as infer_mod
from src.originvision import (
    map_originvision_category,
    run_originvision_infer,
    sample_session_priors,
    score_lights_with_originvision,
    score_master_with_originvision,
)

_HAVE_ORT = infer_mod._ort is not None
_HAVE_NATIVE = infer_mod._HAS_NATIVE_OV
_HAVE_BACKEND = infer_mod.scoring_backend_available()   # native OR onnxruntime
_HAVE_MODEL = infer_mod.resolve_model_path(None) is not None
_real_infer = pytest.mark.skipif(
    not (_HAVE_BACKEND and _HAVE_MODEL),
    reason='no originvision backend (native/onnxruntime) or bundled model')
_need_both = pytest.mark.skipif(
    not (_HAVE_ORT and _HAVE_NATIVE and _HAVE_MODEL),
    reason='need native AND onnxruntime for a cross-backend parity check')


# ---------------------------------------------------------------------------
# originvision_infer: preprocessing ports
# ---------------------------------------------------------------------------

class TestPreprocessing:

    def test_stretch_to_uint8_per_channel_full_range(self):
        img = np.zeros((20, 20, 3), np.float32)
        img[..., 0] = np.linspace(100, 200, 20)[:, None]      # R: 100..200
        img[..., 1] = np.linspace(0, 50, 20)[:, None]         # G: 0..50
        img[..., 2] = 7.0                                     # B: flat
        out = infer_mod._stretch_to_uint8(img)
        assert out.dtype == np.uint8
        # each non-flat channel is stretched to (near) the full 0..255 range
        assert out[..., 0].min() < 5 and out[..., 0].max() > 250
        assert out[..., 1].min() < 5 and out[..., 1].max() > 250
        # flat channel: hi<=lo guard -> all zeros, no divide-by-zero
        assert out[..., 2].max() == 0

    def test_resize_center_crop_exact_square(self):
        img = (np.random.default_rng(0).random((300, 500, 3)) * 255).astype(np.uint8)
        for size in (64, 128, 256, 384):
            out = infer_mod._resize_center_crop(img, size)
            assert out.shape == (size, size, 3)
            assert out.dtype == np.uint8

    def test_resize_center_crop_upscale(self):
        img = (np.random.default_rng(1).random((40, 60, 3)) * 255).astype(np.uint8)
        out = infer_mod._resize_center_crop(img, 128)
        assert out.shape == (128, 128, 3)

    def test_downsample_if_large_is_a_no_op_under_the_cap(self):
        img = (np.random.default_rng(2).random((100, 150, 3)) * 255).astype(np.float32)
        out = infer_mod._downsample_if_large(img, max_long_side=200)
        assert out is img  # identity, not just equal -- no copy when already small

    def test_downsample_if_large_preserves_aspect_and_caps_long_side(self):
        img = (np.random.default_rng(3).random((1000, 2000, 3)) * 255).astype(np.float32)
        out = infer_mod._downsample_if_large(img, max_long_side=500)
        assert out.shape[1] == 500  # long side (width here) hits the cap exactly
        assert abs(out.shape[0] / out.shape[1] - img.shape[0] / img.shape[1]) < 0.01
        assert out.dtype == np.float32

    def test_downsample_if_large_never_crops(self):
        """Unlike _resize_center_crop, this must keep the whole frame --
        cropping here, before the percentile stretch even runs, would change
        which pixels the stretch is computed over."""
        img = (np.random.default_rng(4).random((300, 900, 3)) * 255).astype(np.float32)
        out = infer_mod._downsample_if_large(img, max_long_side=300)
        assert out.shape[0] == 100  # short side scaled down, not cropped away
        assert out.shape[1] == 300


class _FakeCatSession:
    """Minimal onnxruntime-style session with only a `category` head, so the
    comet-suppression branch in `score_rgb` can be exercised without a real
    model or onnxruntime installed."""

    def __init__(self, logits):
        self._logits = np.asarray(logits, np.float32)

    def get_modelmeta(self):
        cats = 'galaxy,nebula,star_cluster,comet'

        class _M:
            custom_metadata_map = {'tasks': 'category', 'head_order': 'category',
                                   'categories': cats}
        return _M()

    def run(self, _outs, _feed):
        return [self._logits[np.newaxis]]


class TestCometSuppression:
    """`comet` is a distrusted class: a top `comet` pick is demoted to the
    runner-up (native kernel and onnxruntime fallback alike)."""

    def _score(self, monkeypatch, logits, *, shape_gate=True):
        monkeypatch.setattr(infer_mod, '_HAS_NATIVE_OV', False)
        monkeypatch.setattr(infer_mod, '_ort', object())          # bypass the None guard
        monkeypatch.setattr(infer_mod, '_get_session',
                            lambda mp: _FakeCatSession(logits))
        return infer_mod.score_rgb(np.zeros((32, 32, 3), np.float32),
                                   shape_gate=shape_gate)

    def test_comet_top_is_demoted_to_runner_up(self, monkeypatch):
        r = self._score(monkeypatch, [0.1, 2.0, 0.1, 5.0])   # argmax = comet, 2nd = nebula
        assert r['category'] == 'nebula'
        assert r['category_shape_gated'] is True

    def test_non_comet_top_is_untouched(self, monkeypatch):
        r = self._score(monkeypatch, [5.0, 2.0, 0.1, 0.1])   # argmax = galaxy
        assert r['category'] == 'galaxy'
        assert r['category_shape_gated'] is False

    def test_shape_gate_false_keeps_comet(self, monkeypatch):
        r = self._score(monkeypatch, [0.1, 2.0, 0.1, 5.0], shape_gate=False)
        assert r['category'] == 'comet'


# ---------------------------------------------------------------------------
# originvision_infer: model resolution / availability gate
# ---------------------------------------------------------------------------

class TestModelResolution:

    def test_bundled_model_path_is_under_src_data(self):
        p = infer_mod.bundled_model_path()
        assert p.replace('\\', '/').endswith('src/data/originvision.onnx')

    def test_resolve_prefers_explicit_when_it_exists(self, tmp_path):
        m = tmp_path / 'custom.onnx'
        m.write_bytes(b'not really onnx')
        assert infer_mod.resolve_model_path(str(m)) == str(m)

    def test_resolve_falls_back_to_bundled_for_bad_explicit(self):
        got = infer_mod.resolve_model_path('/no/such/model.onnx')
        assert got == (infer_mod.bundled_model_path() if _HAVE_MODEL else None)

    def test_score_rgb_none_without_any_backend(self, monkeypatch):
        monkeypatch.setattr(infer_mod, '_ort', None)
        monkeypatch.setattr(infer_mod, '_HAS_NATIVE_OV', False)
        assert infer_mod.score_rgb(np.zeros((32, 32, 3), np.float32)) is None


# ---------------------------------------------------------------------------
# originvision_infer: real forward pass (skips without onnxruntime)
# ---------------------------------------------------------------------------

@_real_infer
class TestRealInference:

    def _synth_rgb(self):
        rng = np.random.default_rng(42)
        h, w = 300, 400
        yy, xx = np.mgrid[0:h, 0:w]
        img = 40 + 0.03 * xx + rng.normal(0, 3, (h, w))
        for _ in range(30):
            cy, cx = rng.integers(30, h - 30), rng.integers(30, w - 30)
            img += rng.uniform(60, 200) * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.uniform(1.5, 3) ** 2))
        img = np.clip(img, 0, 255)
        return np.stack([img, img * 0.9, img * 0.8], -1).astype(np.float32)

    def test_score_rgb_returns_expected_keys(self):
        r = infer_mod.score_rgb(self._synth_rgb())
        assert r is not None
        for k in ('checkpoint_epoch', 'tasks', 'defect_probability', 'is_defective',
                  'quality_score', 'category', 'category_confidence', 'top_categories',
                  'sky_brightness', 'stray_light_gradient', 'stray_light_flag'):
            assert k in r, k
        assert r['category'] in ('galaxy', 'nebula', 'star_cluster', 'comet')
        assert 0.0 <= r['category_confidence'] <= 1.0

    def test_fast_preprocess_defaults_off_and_matches_explicit_false(self):
        """The dormant fast_preprocess opt-in must never change behaviour
        unless a caller explicitly asks for it -- default-arg omission and
        an explicit False must be identical."""
        rgb = self._synth_rgb()
        r_default = infer_mod.score_rgb(rgb)
        r_explicit_false = infer_mod.score_rgb(rgb, fast_preprocess=False)
        assert r_default == r_explicit_false

    def test_fast_preprocess_true_actually_changes_the_input(self):
        """Opting in must take a measurably different (cheaper) path -- a
        small synthetic frame is already near/under the downsample cap, so
        scale up first to guarantee the cap actually bites."""
        big = np.tile(self._synth_rgb(), (3, 3, 1))  # 900x1200, well over 256*4
        r_full = infer_mod.score_rgb(big)
        r_fast = infer_mod.score_rgb(big, fast_preprocess=True)
        assert r_full is not None and r_fast is not None
        assert r_full['quality_score'] != r_fast['quality_score']

    def test_untrained_and_unused_heads_not_surfaced(self):
        """The bundled v4 graph emits 8 outputs incl. `trailing` (untrained,
        excluded from `tasks`) and `background_grid` (trained but unused).
        Neither may leak into the result -- every head is gated on `tasks`."""
        r = infer_mod.score_rgb(self._synth_rgb())
        assert r is not None
        assert 'trailing' not in r['tasks']
        for k in ('trailing_score', 'trailing_flag', 'background_grid',
                  'background_extracted_to'):
            assert k not in r, k

    def test_score_path_png_roundtrip(self, tmp_path):
        pytest.importorskip('PIL')
        from PIL import Image
        p = tmp_path / 'synth.png'
        Image.fromarray(self._synth_rgb().astype(np.uint8), 'RGB').save(p)
        r = infer_mod.score_path(str(p))
        assert r is not None and r['image'] == str(p)
        assert r['category'] in ('galaxy', 'nebula', 'star_cluster', 'comet')

    def test_score_path_bayer_fits(self, tmp_path):
        fits = pytest.importorskip('astropy.io.fits')
        if not hasattr(fits.PrimaryHDU, 'writeto'):
            pytest.skip('astropy.io.fits stubbed by another test module')
        rng = np.random.default_rng(0)
        mosaic = (rng.random((240, 320)) * 3000 + 200).astype(np.uint16)
        hdu = fits.PrimaryHDU(data=mosaic)
        hdu.header['BAYERPAT'] = 'RGGB'
        src = str(tmp_path / 'Light0001.fits')
        hdu.writeto(src, overwrite=True)
        r = infer_mod.score_path(src)
        assert r is not None and r['category'] in (
            'galaxy', 'nebula', 'star_cluster', 'comet')

    @pytest.mark.skipif(not _HAVE_ORT, reason='onnxruntime fallback path only')
    def test_onnxruntime_fallback_session_cache(self, monkeypatch):
        # The native backend caches its session in Rust; this exercises the
        # onnxruntime fallback's _SESSION_CACHE. Isolated dict so it can't
        # race other tests under pytest -n.
        monkeypatch.setattr(infer_mod, '_HAS_NATIVE_OV', False)
        fresh: dict = {}
        monkeypatch.setattr(infer_mod, '_SESSION_CACHE', fresh)
        mp = infer_mod.resolve_model_path(None)
        infer_mod.score_rgb(self._synth_rgb())
        infer_mod.score_rgb(self._synth_rgb())
        assert list(fresh) == [mp]

    @_need_both
    def test_native_matches_onnxruntime(self):
        """Native tract path vs the onnxruntime fallback on the same frame:
        identical category / flags, scalars within tolerance (preprocessing
        differs slightly -- Rust bilinear vs scipy zoom)."""
        rgb = self._synth_rgb()
        native = infer_mod.score_rgb(rgb)
        # force the fallback
        import unittest.mock as _m
        with _m.patch.object(infer_mod, '_HAS_NATIVE_OV', False):
            ort = infer_mod.score_rgb(rgb)
        assert native is not None and ort is not None
        assert native['category'] == ort['category']
        assert native['is_defective'] == ort['is_defective']
        assert native['stray_light_flag'] == ort['stray_light_flag']
        assert native.get('predicted_exposure_s') == ort.get('predicted_exposure_s')
        for k in ('sky_brightness', 'stray_light_gradient', 'defect_probability'):
            assert abs(native[k] - ort[k]) < max(1.0, abs(ort[k]) * 0.05), (k, native[k], ort[k])
        assert abs(native['category_confidence'] - ort['category_confidence']) < 0.03


# ---------------------------------------------------------------------------
# src/originvision.py: run_originvision_infer wrapper
# ---------------------------------------------------------------------------

class TestRunOriginvisionInfer:

    def test_delegates_to_score_path_with_model(self):
        with mock.patch.object(originvision_mod.originvision_infer, 'score_path',
                               return_value={'category': 'galaxy'}) as m:
            out = run_originvision_infer('master.tiff', 'model.onnx')
        assert out == {'category': 'galaxy'}
        m.assert_called_once_with('master.tiff', model_path='model.onnx')

    def test_failure_returns_none(self):
        with mock.patch.object(originvision_mod.originvision_infer, 'score_path',
                               return_value=None):
            assert run_originvision_infer('x.fits') is None


# ---------------------------------------------------------------------------
# src/originvision.py: advisory scoring (mock run_originvision_infer + _originvision_model)
# ---------------------------------------------------------------------------

def _frame(path, accepted=True):
    return SimpleNamespace(path=path, accepted=accepted, metrics={'score': 50.0})


def _args(**overrides):
    base = dict(originvision=True, originvision_score_all=True, originvision_model=None,
                originvision_workers=2)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _model_resolves():
    """Every advisory-path test assumes a usable model; the two 'missing
    model' tests re-patch it to None inside their own ``with`` block."""
    with mock.patch.object(originvision_mod, '_originvision_model', return_value='model.onnx'):
        yield


class TestScoreLightsWithOriginvision:

    def test_disabled_is_noop(self):
        lights = [_frame('a.fits')]
        with mock.patch.object(originvision_mod, 'run_originvision_infer') as m:
            score_lights_with_originvision(lights, _args(originvision=False))
        m.assert_not_called()
        assert 'originvision' not in lights[0].metrics

    def test_missing_model_is_noop(self):
        lights = [_frame('a.fits')]
        with mock.patch.object(originvision_mod, '_originvision_model', return_value=None), \
             mock.patch.object(originvision_mod, 'run_originvision_infer') as m:
            score_lights_with_originvision(lights, _args())
        m.assert_not_called()
        assert 'originvision' not in lights[0].metrics

    def test_originvision_on_but_score_all_off_is_noop(self):
        lights = [_frame('a.fits')]
        with mock.patch.object(originvision_mod, 'run_originvision_infer') as m:
            score_lights_with_originvision(lights, _args(originvision_score_all=False))
        m.assert_not_called()
        assert 'originvision' not in lights[0].metrics

    def test_stores_result_without_touching_accepted_or_score(self):
        lights = [_frame('a.fits'), _frame('b.fits')]
        results = {
            'a.fits': {'quality_score': 400.0, 'is_defective': False, 'stray_light_flag': False},
            'b.fits': {'quality_score': 410.0, 'is_defective': True, 'stray_light_flag': False},
        }
        with mock.patch.object(originvision_mod, 'run_originvision_infer',
                               side_effect=lambda path, *a, **k: results[path]):
            score_lights_with_originvision(lights, _args())
        for f in lights:
            assert f.metrics['originvision'] == results[f.path]
            assert f.accepted is True
            assert f.metrics['score'] == 50.0

    def test_failed_frame_scored_none_does_not_crash(self):
        lights = [_frame('a.fits'), _frame('b.fits')]
        def _side_effect(path, *a, **k):
            return None if path == 'a.fits' else {'quality_score': 100.0}
        with mock.patch.object(originvision_mod, 'run_originvision_infer', side_effect=_side_effect):
            score_lights_with_originvision(lights, _args())
        assert lights[0].metrics['originvision'] is None
        assert lights[1].metrics['originvision'] == {'quality_score': 100.0}

    def test_below_session_average_frame_flagged_in_output(self, capsys):
        lights = [_frame(f'good{i}.fits') for i in range(9)]
        lights.append(_frame('bad.fits'))
        def _side_effect(path, *a, **k):
            if path == 'bad.fits':
                return {'quality_score': 1.0}
            return {'quality_score': 500.0}
        with mock.patch.object(originvision_mod, 'run_originvision_infer', side_effect=_side_effect):
            score_lights_with_originvision(lights, _args())
        out = capsys.readouterr().out
        assert 'below-session-average' in out
        assert 'bad.fits' in out

    def test_rejected_frames_skipped(self):
        lights = [_frame('a.fits', accepted=False)]
        with mock.patch.object(originvision_mod, 'run_originvision_infer') as m:
            score_lights_with_originvision(lights, _args())
        m.assert_not_called()


class TestScoreMasterWithOriginvision:

    def test_disabled_is_noop(self):
        with mock.patch.object(originvision_mod, 'run_originvision_infer') as m:
            score_master_with_originvision('stack.fits', _args(originvision=False), 'galaxy')
        m.assert_not_called()

    def test_category_mismatch_warns(self, caplog):
        result = {'category': 'nebula', 'category_confidence': 0.7,
                 'sky_brightness': 80.0, 'stray_light_gradient': 5.0}
        with mock.patch.object(originvision_mod, 'run_originvision_infer', return_value=result):
            with caplog.at_level('WARNING', logger='originstack'):
                score_master_with_originvision('stack.fits', _args(), 'galaxy')
        assert any('does not match' in r.getMessage() for r in caplog.records)

    def test_category_match_does_not_warn(self, caplog):
        result = {'category': 'galaxy', 'category_confidence': 0.9,
                 'sky_brightness': 80.0, 'stray_light_gradient': 5.0}
        with mock.patch.object(originvision_mod, 'run_originvision_infer', return_value=result):
            with caplog.at_level('WARNING', logger='originstack'):
                score_master_with_originvision('stack.fits', _args(), 'galaxy')
        assert not any('does not match' in r.getMessage() for r in caplog.records)

    def test_coarse_category_vs_fine_inferred_type_does_not_warn(self, caplog):
        result = {'category': 'nebula', 'category_confidence': 0.95,
                 'sky_brightness': 40.0, 'stray_light_gradient': 20.0}
        with mock.patch.object(originvision_mod, 'run_originvision_infer', return_value=result):
            with caplog.at_level('WARNING', logger='originstack'):
                score_master_with_originvision('stack.tiff', _args(), 'emission_nebula')
        assert not any('does not match' in r.getMessage() for r in caplog.records)

    def test_predicted_exposure_printed_when_present(self, capsys):
        result = {'category': 'galaxy', 'category_confidence': 0.9,
                 'sky_brightness': 40.0, 'stray_light_gradient': 5.0,
                 'predicted_exposure_s': 30.0}
        with mock.patch.object(originvision_mod, 'run_originvision_infer', return_value=result):
            score_master_with_originvision('stack.tiff', _args(), 'galaxy')
        assert 'predicted_exposure=30s' in capsys.readouterr().out

    def test_failed_score_logged_not_raised(self, capsys):
        with mock.patch.object(originvision_mod, 'run_originvision_infer', return_value=None):
            score_master_with_originvision('stack.fits', _args(), 'galaxy')
        assert 'failed' in capsys.readouterr().out


class TestMapOriginvisionCategory:

    def test_galaxy_maps_unambiguously(self):
        assert map_originvision_category('galaxy') == 'galaxy'

    def test_star_cluster_maps_to_globular_cluster(self):
        assert map_originvision_category('star_cluster') == 'globular_cluster'

    def test_ambiguous_nebula_returns_none(self):
        assert map_originvision_category('nebula') is None

    def test_unrelated_categories_return_none(self):
        for c in ('comet', 'planet', 'star', 'other'):
            assert map_originvision_category(c) is None

    def test_none_input_returns_none(self):
        assert map_originvision_category(None) is None

    def test_case_insensitive(self):
        assert map_originvision_category('GALAXY') == 'galaxy'


class TestSampleSessionPriors:

    def test_disabled_is_noop(self):
        lights = [_frame(f'{i}.fits') for i in range(10)]
        with mock.patch.object(originvision_mod, 'run_originvision_infer') as m:
            result = sample_session_priors(lights, _args(originvision=False))
        assert result is None
        m.assert_not_called()

    def test_missing_model_is_noop(self):
        lights = [_frame(f'{i}.fits') for i in range(10)]
        with mock.patch.object(originvision_mod, '_originvision_model', return_value=None), \
             mock.patch.object(originvision_mod, 'run_originvision_infer') as m:
            result = sample_session_priors(lights, _args())
        assert result is None
        m.assert_not_called()

    def test_no_accepted_frames_is_noop(self):
        lights = [_frame('a.fits', accepted=False)]
        result = sample_session_priors(lights, _args())
        assert result is None

    def test_samples_a_few_frames_not_all(self):
        lights = [_frame(f'{i}.fits') for i in range(120)]
        payload = {'category': 'galaxy', 'category_confidence': 0.9,
                  'is_defective': False, 'stray_light_flag': False,
                  'defect_probability': 0.1}
        with mock.patch.object(originvision_mod, 'run_originvision_infer', return_value=payload) as m:
            result = sample_session_priors(lights, _args())
        assert result is not None
        assert m.call_count <= 3

    def test_majority_category_wins(self):
        lights = [_frame(f'{i}.fits') for i in range(12)]
        payloads = [
            {'category': 'galaxy', 'category_confidence': 0.9, 'defect_probability': 0.0},
            {'category': 'galaxy', 'category_confidence': 0.8, 'defect_probability': 0.0},
            {'category': 'nebula', 'category_confidence': 0.99, 'defect_probability': 0.0},
        ]
        with mock.patch.object(originvision_mod, 'run_originvision_infer', side_effect=payloads):
            result = sample_session_priors(lights, _args())
        assert result['category'] == 'galaxy'

    def test_defect_flagged_on_any_sample(self):
        lights = [_frame(f'{i}.fits') for i in range(12)]
        payloads = [
            {'category': 'galaxy', 'category_confidence': 0.9, 'is_defective': False,
             'defect_probability': 0.1},
            {'category': 'galaxy', 'category_confidence': 0.9, 'is_defective': True,
             'defect_probability': 0.9},
            {'category': 'galaxy', 'category_confidence': 0.9, 'is_defective': False,
             'defect_probability': 0.1},
        ]
        with mock.patch.object(originvision_mod, 'run_originvision_infer', side_effect=payloads):
            result = sample_session_priors(lights, _args())
        assert result['defect_flagged'] is True

    def test_no_defect_when_all_clean(self):
        lights = [_frame(f'{i}.fits') for i in range(12)]
        payload = {'category': 'galaxy', 'category_confidence': 0.9,
                  'is_defective': False, 'stray_light_flag': False,
                  'defect_probability': 0.05}
        with mock.patch.object(originvision_mod, 'run_originvision_infer', return_value=payload):
            result = sample_session_priors(lights, _args())
        assert result['defect_flagged'] is False

    def test_all_samples_failing_returns_none(self):
        lights = [_frame(f'{i}.fits') for i in range(12)]
        with mock.patch.object(originvision_mod, 'run_originvision_infer', return_value=None):
            result = sample_session_priors(lights, _args())
        assert result is None


# ---------------------------------------------------------------------------
# src/cli.py: --originvision resolution (bundled model, onnxruntime gate)
# ---------------------------------------------------------------------------

class TestCliOriginvisionResolution:

    def _parse(self, tmp_path, *extra):
        from src import cli
        return cli.parse_args(['-d', str(tmp_path), '-o', str(tmp_path / 'o.fits'),
                               '--originvision', *extra])

    @_real_infer
    def test_originvision_stays_enabled_with_bundled_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ORIGINVISION_DIR', raising=False)
        args = self._parse(tmp_path)
        assert args.originvision is True
        assert args.originvision_model is None  # bundled default, resolved at call time

    def test_disabled_without_any_backend(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv('ORIGINVISION_DIR', raising=False)
        monkeypatch.setattr(infer_mod, 'scoring_backend_available', lambda: False)
        args = self._parse(tmp_path)
        assert args.originvision is False
        assert 'no inference backend' in capsys.readouterr().out

    def test_explicit_model_override_is_kept(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ORIGINVISION_DIR', raising=False)
        m = tmp_path / 'v4.onnx'
        m.write_bytes(b'x')
        monkeypatch.setattr(infer_mod, 'scoring_backend_available', lambda: True)
        args = self._parse(tmp_path, '--originvision-model', str(m))
        assert args.originvision is True
        assert args.originvision_model == str(m)

    def test_legacy_dir_flag_still_resolves_a_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ORIGINVISION_DIR', raising=False)
        ck = tmp_path / 'checkpoints'
        ck.mkdir()
        (ck / 'model.onnx').write_bytes(b'x')
        monkeypatch.setattr(infer_mod, 'scoring_backend_available', lambda: True)
        args = self._parse(tmp_path, '--originvision-dir', str(tmp_path))
        assert args.originvision is True
        assert args.originvision_model == str(ck / 'model.onnx')

    def test_legacy_checkpoint_flag_still_resolves_a_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ORIGINVISION_DIR', raising=False)
        m = tmp_path / 'old.onnx'
        m.write_bytes(b'x')
        monkeypatch.setattr(infer_mod, 'scoring_backend_available', lambda: True)
        args = self._parse(tmp_path, '--originvision-checkpoint', str(m))
        assert args.originvision is True
        assert args.originvision_model == str(m)

    def test_bad_explicit_model_disables_not_silent_bundled_fallback(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv('ORIGINVISION_DIR', raising=False)
        monkeypatch.setattr(infer_mod, 'scoring_backend_available', lambda: True)
        bad = str(tmp_path / 'nope.onnx')
        args = self._parse(tmp_path, '--originvision-model', bad)
        assert args.originvision is False
        assert bad in capsys.readouterr().out

    def test_dropped_subprocess_flags_now_rejected(self, tmp_path, monkeypatch):
        # --originvision-timeout / -python / -script were inert no-ops for the
        # removed subprocess scorer; they are gone now (not silently swallowed).
        monkeypatch.delenv('ORIGINVISION_DIR', raising=False)
        for flag, val in (('--originvision-timeout', '120'),
                          ('--originvision-python', 'py.exe'),
                          ('--originvision-script', 's.py')):
            with pytest.raises(SystemExit):
                self._parse(tmp_path, flag, val)


if __name__ == '__main__':
    import unittest
    unittest.main()
