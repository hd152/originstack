"""Tests for the astrollm integration (src/astrollm.py + src/astrollm_infer.py):
in-process ONNX inference wiring, numpy/scipy preprocessing, and the
advisory-only session-relative flagging.

onnxruntime may or may not be installed in the test environment. Tests that
need a real forward pass skip when it (or the bundled model) is absent;
everything else mocks ``run_astrollm_infer`` / ``_astrollm_model`` so it
never depends on a real model.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import src.astrollm as astrollm_mod
import src.astrollm_infer as infer_mod
from src.astrollm import (
    map_astrollm_category,
    run_astrollm_infer,
    sample_session_priors,
    score_lights_with_astrollm,
    score_master_with_astrollm,
)

_HAVE_ORT = infer_mod.onnxruntime_available()
_HAVE_MODEL = infer_mod.resolve_model_path(None) is not None
_real_infer = pytest.mark.skipif(
    not (_HAVE_ORT and _HAVE_MODEL),
    reason='onnxruntime or bundled model not available')


# ---------------------------------------------------------------------------
# astrollm_infer: preprocessing ports
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

    def test_blob_shape_features_centered_single_blob(self):
        g = np.zeros((200, 200), np.uint8)
        g[95:105, 95:105] = 255                    # one blob, dead centre
        n, dist = infer_mod._blob_shape_features(g)
        assert n == 1
        assert dist < 0.05                          # ~centre

    def test_blob_shape_features_offcentre_and_count(self):
        g = np.zeros((200, 200), np.uint8)
        g[5:15, 5:15] = 255                         # corner blob (largest)
        for cx in range(20, 180, 15):              # a row of smaller blobs
            g[100:104, cx:cx + 4] = 255
        n, dist = infer_mod._blob_shape_features(g)
        assert n >= 5
        assert dist > 0.5                           # largest blob is a corner

    def test_blob_shape_features_blank_returns_not_centered(self):
        n, dist = infer_mod._blob_shape_features(np.zeros((50, 50), np.uint8))
        assert n == 0 and dist == 1.0

    def test_gate_comet_only_touches_comet_top(self):
        cats = ['galaxy', 'nebula', 'star_cluster', 'comet']
        probs = np.array([0.7, 0.1, 0.1, 0.1])     # top = galaxy
        gray = np.zeros((100, 100), np.uint8)
        assert infer_mod._gate_comet_prediction(probs, gray, cats) == 0

    def test_gate_comet_demotes_when_shape_disagrees(self):
        cats = ['galaxy', 'nebula', 'star_cluster', 'comet']
        probs = np.array([0.05, 0.25, 0.1, 0.6])   # top = comet
        gray = np.zeros((200, 200), np.uint8)
        gray[5:15, 5:15] = 255                      # bright blob in a corner
        picked = infer_mod._gate_comet_prediction(probs, gray, cats)
        assert picked == 1                          # -> 2nd best (nebula)


# ---------------------------------------------------------------------------
# astrollm_infer: model resolution / availability gate
# ---------------------------------------------------------------------------

class TestModelResolution:

    def test_bundled_model_path_is_under_src_data(self):
        p = infer_mod.bundled_model_path()
        assert p.replace('\\', '/').endswith('src/data/astrollm.onnx')

    def test_resolve_prefers_explicit_when_it_exists(self, tmp_path):
        m = tmp_path / 'custom.onnx'
        m.write_bytes(b'not really onnx')
        assert infer_mod.resolve_model_path(str(m)) == str(m)

    def test_resolve_falls_back_to_bundled_for_bad_explicit(self):
        got = infer_mod.resolve_model_path('/no/such/model.onnx')
        assert got == (infer_mod.bundled_model_path() if _HAVE_MODEL else None)

    def test_score_rgb_none_without_onnxruntime(self, monkeypatch):
        monkeypatch.setattr(infer_mod, '_ort', None)
        assert infer_mod.score_rgb(np.zeros((32, 32, 3), np.float32)) is None


# ---------------------------------------------------------------------------
# astrollm_infer: real forward pass (skips without onnxruntime)
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

    def test_session_cache_reuses_one_inferencesession(self, monkeypatch):
        # Isolated cache dict so this can't race other tests under pytest -n.
        fresh: dict = {}
        monkeypatch.setattr(infer_mod, '_SESSION_CACHE', fresh)
        mp = infer_mod.resolve_model_path(None)
        infer_mod.score_rgb(self._synth_rgb())
        infer_mod.score_rgb(self._synth_rgb())
        assert list(fresh) == [mp]


# ---------------------------------------------------------------------------
# src/astrollm.py: run_astrollm_infer wrapper
# ---------------------------------------------------------------------------

class TestRunAstrollmInfer:

    def test_delegates_to_score_path_with_model(self):
        with mock.patch.object(astrollm_mod.astrollm_infer, 'score_path',
                               return_value={'category': 'galaxy'}) as m:
            out = run_astrollm_infer('master.tiff', 'model.onnx')
        assert out == {'category': 'galaxy'}
        m.assert_called_once_with('master.tiff', model_path='model.onnx')

    def test_failure_returns_none(self):
        with mock.patch.object(astrollm_mod.astrollm_infer, 'score_path',
                               return_value=None):
            assert run_astrollm_infer('x.fits') is None


# ---------------------------------------------------------------------------
# src/astrollm.py: advisory scoring (mock run_astrollm_infer + _astrollm_model)
# ---------------------------------------------------------------------------

def _frame(path, accepted=True):
    return SimpleNamespace(path=path, accepted=accepted, metrics={'score': 50.0})


def _args(**overrides):
    base = dict(astrollm=True, astrollm_score_all=True, astrollm_model=None,
                astrollm_workers=2)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _model_resolves():
    """Every advisory-path test assumes a usable model; the two 'missing
    model' tests re-patch it to None inside their own ``with`` block."""
    with mock.patch.object(astrollm_mod, '_astrollm_model', return_value='model.onnx'):
        yield


class TestScoreLightsWithAstrollm:

    def test_disabled_is_noop(self):
        lights = [_frame('a.fits')]
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer') as m:
            score_lights_with_astrollm(lights, _args(astrollm=False))
        m.assert_not_called()
        assert 'astrollm' not in lights[0].metrics

    def test_missing_model_is_noop(self):
        lights = [_frame('a.fits')]
        with mock.patch.object(astrollm_mod, '_astrollm_model', return_value=None), \
             mock.patch.object(astrollm_mod, 'run_astrollm_infer') as m:
            score_lights_with_astrollm(lights, _args())
        m.assert_not_called()
        assert 'astrollm' not in lights[0].metrics

    def test_astrollm_on_but_score_all_off_is_noop(self):
        lights = [_frame('a.fits')]
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer') as m:
            score_lights_with_astrollm(lights, _args(astrollm_score_all=False))
        m.assert_not_called()
        assert 'astrollm' not in lights[0].metrics

    def test_stores_result_without_touching_accepted_or_score(self):
        lights = [_frame('a.fits'), _frame('b.fits')]
        results = {
            'a.fits': {'quality_score': 400.0, 'is_defective': False, 'stray_light_flag': False},
            'b.fits': {'quality_score': 410.0, 'is_defective': True, 'stray_light_flag': False},
        }
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer',
                               side_effect=lambda path, *a, **k: results[path]):
            score_lights_with_astrollm(lights, _args())
        for f in lights:
            assert f.metrics['astrollm'] == results[f.path]
            assert f.accepted is True
            assert f.metrics['score'] == 50.0

    def test_failed_frame_scored_none_does_not_crash(self):
        lights = [_frame('a.fits'), _frame('b.fits')]
        def _side_effect(path, *a, **k):
            return None if path == 'a.fits' else {'quality_score': 100.0}
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', side_effect=_side_effect):
            score_lights_with_astrollm(lights, _args())
        assert lights[0].metrics['astrollm'] is None
        assert lights[1].metrics['astrollm'] == {'quality_score': 100.0}

    def test_below_session_average_frame_flagged_in_output(self, capsys):
        lights = [_frame(f'good{i}.fits') for i in range(9)]
        lights.append(_frame('bad.fits'))
        def _side_effect(path, *a, **k):
            if path == 'bad.fits':
                return {'quality_score': 1.0}
            return {'quality_score': 500.0}
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', side_effect=_side_effect):
            score_lights_with_astrollm(lights, _args())
        out = capsys.readouterr().out
        assert 'below-session-average' in out
        assert 'bad.fits' in out

    def test_rejected_frames_skipped(self):
        lights = [_frame('a.fits', accepted=False)]
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer') as m:
            score_lights_with_astrollm(lights, _args())
        m.assert_not_called()


class TestScoreMasterWithAstrollm:

    def test_disabled_is_noop(self):
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer') as m:
            score_master_with_astrollm('stack.fits', _args(astrollm=False), 'galaxy')
        m.assert_not_called()

    def test_category_mismatch_warns(self, caplog):
        result = {'category': 'nebula', 'category_confidence': 0.7,
                 'sky_brightness': 80.0, 'stray_light_gradient': 5.0}
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', return_value=result):
            with caplog.at_level('WARNING', logger='originstack'):
                score_master_with_astrollm('stack.fits', _args(), 'galaxy')
        assert any('does not match' in r.getMessage() for r in caplog.records)

    def test_category_match_does_not_warn(self, caplog):
        result = {'category': 'galaxy', 'category_confidence': 0.9,
                 'sky_brightness': 80.0, 'stray_light_gradient': 5.0}
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', return_value=result):
            with caplog.at_level('WARNING', logger='originstack'):
                score_master_with_astrollm('stack.fits', _args(), 'galaxy')
        assert not any('does not match' in r.getMessage() for r in caplog.records)

    def test_coarse_category_vs_fine_inferred_type_does_not_warn(self, caplog):
        result = {'category': 'nebula', 'category_confidence': 0.95,
                 'sky_brightness': 40.0, 'stray_light_gradient': 20.0}
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', return_value=result):
            with caplog.at_level('WARNING', logger='originstack'):
                score_master_with_astrollm('stack.tiff', _args(), 'emission_nebula')
        assert not any('does not match' in r.getMessage() for r in caplog.records)

    def test_predicted_exposure_printed_when_present(self, capsys):
        result = {'category': 'galaxy', 'category_confidence': 0.9,
                 'sky_brightness': 40.0, 'stray_light_gradient': 5.0,
                 'predicted_exposure_s': 30.0}
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', return_value=result):
            score_master_with_astrollm('stack.tiff', _args(), 'galaxy')
        assert 'predicted_exposure=30s' in capsys.readouterr().out

    def test_failed_score_logged_not_raised(self, capsys):
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', return_value=None):
            score_master_with_astrollm('stack.fits', _args(), 'galaxy')
        assert 'failed' in capsys.readouterr().out


class TestMapAstrollmCategory:

    def test_galaxy_maps_unambiguously(self):
        assert map_astrollm_category('galaxy') == 'galaxy'

    def test_star_cluster_maps_to_globular_cluster(self):
        assert map_astrollm_category('star_cluster') == 'globular_cluster'

    def test_ambiguous_nebula_returns_none(self):
        assert map_astrollm_category('nebula') is None

    def test_unrelated_categories_return_none(self):
        for c in ('comet', 'planet', 'star', 'other'):
            assert map_astrollm_category(c) is None

    def test_none_input_returns_none(self):
        assert map_astrollm_category(None) is None

    def test_case_insensitive(self):
        assert map_astrollm_category('GALAXY') == 'galaxy'


class TestSampleSessionPriors:

    def test_disabled_is_noop(self):
        lights = [_frame(f'{i}.fits') for i in range(10)]
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer') as m:
            result = sample_session_priors(lights, _args(astrollm=False))
        assert result is None
        m.assert_not_called()

    def test_missing_model_is_noop(self):
        lights = [_frame(f'{i}.fits') for i in range(10)]
        with mock.patch.object(astrollm_mod, '_astrollm_model', return_value=None), \
             mock.patch.object(astrollm_mod, 'run_astrollm_infer') as m:
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
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', return_value=payload) as m:
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
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', side_effect=payloads):
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
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', side_effect=payloads):
            result = sample_session_priors(lights, _args())
        assert result['defect_flagged'] is True

    def test_no_defect_when_all_clean(self):
        lights = [_frame(f'{i}.fits') for i in range(12)]
        payload = {'category': 'galaxy', 'category_confidence': 0.9,
                  'is_defective': False, 'stray_light_flag': False,
                  'defect_probability': 0.05}
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', return_value=payload):
            result = sample_session_priors(lights, _args())
        assert result['defect_flagged'] is False

    def test_all_samples_failing_returns_none(self):
        lights = [_frame(f'{i}.fits') for i in range(12)]
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', return_value=None):
            result = sample_session_priors(lights, _args())
        assert result is None


# ---------------------------------------------------------------------------
# src/cli.py: --astrollm resolution (bundled model, onnxruntime gate)
# ---------------------------------------------------------------------------

class TestCliAstrollmResolution:

    def _parse(self, tmp_path, *extra):
        from src import cli
        return cli.parse_args(['-d', str(tmp_path), '-o', str(tmp_path / 'o.fits'),
                               '--astrollm', *extra])

    @_real_infer
    def test_astrollm_stays_enabled_with_bundled_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ASTROLLM_DIR', raising=False)
        args = self._parse(tmp_path)
        assert args.astrollm is True
        assert args.astrollm_model is None  # bundled default, resolved at call time

    def test_disabled_without_onnxruntime(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv('ASTROLLM_DIR', raising=False)
        monkeypatch.setattr(infer_mod, '_ort', None)
        args = self._parse(tmp_path)
        assert args.astrollm is False
        assert 'onnxruntime' in capsys.readouterr().out

    def test_explicit_model_override_is_kept(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ASTROLLM_DIR', raising=False)
        m = tmp_path / 'v4.onnx'
        m.write_bytes(b'x')
        monkeypatch.setattr(infer_mod, 'onnxruntime_available', lambda: True)
        args = self._parse(tmp_path, '--astrollm-model', str(m))
        assert args.astrollm is True
        assert args.astrollm_model == str(m)

    def test_legacy_dir_flag_still_resolves_a_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ASTROLLM_DIR', raising=False)
        ck = tmp_path / 'checkpoints'
        ck.mkdir()
        (ck / 'model.onnx').write_bytes(b'x')
        monkeypatch.setattr(infer_mod, 'onnxruntime_available', lambda: True)
        args = self._parse(tmp_path, '--astrollm-dir', str(tmp_path))
        assert args.astrollm is True
        assert args.astrollm_model == str(ck / 'model.onnx')

    def test_legacy_checkpoint_flag_still_resolves_a_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ASTROLLM_DIR', raising=False)
        m = tmp_path / 'old.onnx'
        m.write_bytes(b'x')
        monkeypatch.setattr(infer_mod, 'onnxruntime_available', lambda: True)
        args = self._parse(tmp_path, '--astrollm-checkpoint', str(m))
        assert args.astrollm is True
        assert args.astrollm_model == str(m)

    def test_bad_explicit_model_disables_not_silent_bundled_fallback(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv('ASTROLLM_DIR', raising=False)
        monkeypatch.setattr(infer_mod, 'onnxruntime_available', lambda: True)
        bad = str(tmp_path / 'nope.onnx')
        args = self._parse(tmp_path, '--astrollm-model', bad)
        assert args.astrollm is False
        assert bad in capsys.readouterr().out

    def test_removed_flags_are_inert_not_errors(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ASTROLLM_DIR', raising=False)
        monkeypatch.setattr(infer_mod, 'onnxruntime_available', lambda: True)
        monkeypatch.setattr(infer_mod, 'resolve_model_path', lambda p: p or 'bundled')
        # old command lines that still pass these must not hard-error
        args = self._parse(tmp_path, '--astrollm-timeout', '120',
                           '--astrollm-python', 'py.exe', '--astrollm-script', 's.py')
        assert args.astrollm is True


if __name__ == '__main__':
    import unittest
    unittest.main()
