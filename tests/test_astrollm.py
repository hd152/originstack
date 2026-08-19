"""Tests for astrollm integration (src/astrollm.py): subprocess wiring,
JSON parsing/error handling, and advisory-only session-relative flagging.

astrollm is not installed in the test environment -- every test mocks
subprocess.run (or run_astrollm_infer directly) so nothing here depends on a
real checkpoint/venv.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from types import SimpleNamespace
from unittest import mock

import src.astrollm as astrollm_mod
from src.astrollm import (run_astrollm_infer, score_lights_with_astrollm,
                          score_master_with_astrollm)


def _completed(returncode=0, stdout='', stderr=''):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


class TestRunAstrollmInfer:

    def test_valid_json_parsed(self):
        payload = {'image': 'a.fits', 'is_defective': False, 'quality_score': 400.0,
                  'category': 'galaxy', 'category_confidence': 0.8}
        with mock.patch('subprocess.run', return_value=_completed(stdout=json.dumps(payload) + '\n')):
            result = run_astrollm_infer('a.fits', 'py.exe', 'infer.py', 'model.pt')
        assert result == payload

    def test_nonzero_exit_returns_none(self):
        with mock.patch('subprocess.run', return_value=_completed(returncode=1, stderr='boom')):
            result = run_astrollm_infer('a.fits', 'py.exe', 'infer.py', 'model.pt')
        assert result is None

    def test_timeout_returns_none(self):
        with mock.patch('subprocess.run', side_effect=subprocess.TimeoutExpired(cmd=[], timeout=1)):
            result = run_astrollm_infer('a.fits', 'py.exe', 'infer.py', 'model.pt')
        assert result is None

    def test_garbage_stdout_returns_none(self):
        with mock.patch('subprocess.run', return_value=_completed(stdout='not json')):
            result = run_astrollm_infer('a.fits', 'py.exe', 'infer.py', 'model.pt')
        assert result is None

    def test_missing_binary_returns_none(self):
        with mock.patch('subprocess.run', side_effect=FileNotFoundError('no such file')):
            result = run_astrollm_infer('a.fits', 'py.exe', 'infer.py', 'model.pt')
        assert result is None

    def test_takes_last_stdout_line(self):
        payload = {'quality_score': 1.0}
        stdout = 'some warning line\n' + json.dumps(payload) + '\n'
        with mock.patch('subprocess.run', return_value=_completed(stdout=stdout)):
            result = run_astrollm_infer('a.fits', 'py.exe', 'infer.py', 'model.pt')
        assert result == payload


def _frame(path, accepted=True):
    return SimpleNamespace(path=path, accepted=accepted, metrics={'score': 50.0})


def _args(**overrides):
    base = dict(astrollm=True, astrollm_python='py.exe', astrollm_script='infer.py',
               astrollm_checkpoint='model.pt', astrollm_workers=2, astrollm_timeout=60.0)
    base.update(overrides)
    return argparse.Namespace(**base)


class TestScoreLightsWithAstrollm:

    def test_disabled_is_noop(self):
        lights = [_frame('a.fits')]
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer') as m:
            score_lights_with_astrollm(lights, _args(astrollm=False))
        m.assert_not_called()
        assert 'astrollm' not in lights[0].metrics

    def test_missing_config_is_noop(self):
        lights = [_frame('a.fits')]
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer') as m:
            score_lights_with_astrollm(lights, _args(astrollm_python=None))
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
        assert any('mismatch' in r.message or 'does not match' in r.message
                  for r in caplog.records) or any(
                  'does not match' in r.getMessage() for r in caplog.records)

    def test_category_match_does_not_warn(self, caplog):
        result = {'category': 'galaxy', 'category_confidence': 0.9,
                 'sky_brightness': 80.0, 'stray_light_gradient': 5.0}
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', return_value=result):
            with caplog.at_level('WARNING', logger='originstack'):
                score_master_with_astrollm('stack.fits', _args(), 'galaxy')
        assert not any('does not match' in r.getMessage() for r in caplog.records)

    def test_coarse_category_vs_fine_inferred_type_does_not_warn(self, caplog):
        """astrollm's 'nebula' bucket vs originstack's finer 'emission_nebula'
        (or 'globular_cluster' vs its own 'star_cluster') is a correct call,
        not a mismatch -- word-overlap matching must not flag it."""
        result = {'category': 'nebula', 'category_confidence': 0.95,
                 'sky_brightness': 40.0, 'stray_light_gradient': 20.0}
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', return_value=result):
            with caplog.at_level('WARNING', logger='originstack'):
                score_master_with_astrollm('stack.tiff', _args(), 'emission_nebula')
        assert not any('does not match' in r.getMessage() for r in caplog.records)

    def test_failed_score_logged_not_raised(self, capsys):
        with mock.patch.object(astrollm_mod, 'run_astrollm_infer', return_value=None):
            score_master_with_astrollm('stack.fits', _args(), 'galaxy')
        assert 'failed' in capsys.readouterr().out


if __name__ == '__main__':
    import unittest
    unittest.main()
