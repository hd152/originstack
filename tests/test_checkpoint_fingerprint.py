"""Checkpoints are only reused for the settings and files they were made from.

A checkpoint used to be matched on the set of light-frame paths alone, so
re-running with a different --stack-method (or re-captured files under the same
names) silently reused the old stack.
"""
import os
import time

import pytest

from src import checkpoint as ck
from src.cli import parse_args
from src.models import FrameInfo


@pytest.fixture
def lights(tmp_path):
    out = []
    for i in range(3):
        p = tmp_path / f'Light_{i:03d}.fits'
        p.write_bytes(b'x' * (100 + i))
        out.append(FrameInfo(path=str(p), type='light', header={}))
    return out


def _save(out, lights, args, phase):
    fp = ck.stack_fingerprint(args, lights)
    ck.save_checkpoint(out, phase=phase, lights=lights, final=lights, fingerprint=fp)
    if phase >= 3:
        import numpy as np
        ck.save_raw_stack(out, np.zeros((4, 4, 3), np.float32))


def _resume(out, lights, argv):
    return ck.can_resume(out, lights, ck.stack_fingerprint(parse_args(argv), lights))


BASE = ['-d', 'x']


def test_same_settings_resume_at_the_saved_phase(tmp_path, lights):
    out = str(tmp_path / 'o.fits')
    _save(out, lights, parse_args(BASE), 3)
    ok, phase, _ = _resume(out, lights, BASE)
    assert ok and phase == 3


def test_phase4_only_change_still_resumes_after_phase3(tmp_path, lights):
    out = str(tmp_path / 'o.fits')
    _save(out, lights, parse_args(BASE), 3)
    ok, phase, _ = _resume(out, lights, BASE + ['--denoiser', 'bilateral', '--stretch', 'arcsinh'])
    assert ok and phase == 3


def test_stacking_change_reuses_phase1_only(tmp_path, lights, capsys):
    out = str(tmp_path / 'o.fits')
    _save(out, lights, parse_args(BASE), 3)
    ok, phase, _ = _resume(out, lights, BASE + ['--stack-method', 'median'])
    assert ok and phase == 1
    assert 'stack_method' in capsys.readouterr().out


def test_phase1_change_starts_fresh(tmp_path, lights, capsys):
    out = str(tmp_path / 'o.fits')
    _save(out, lights, parse_args(BASE), 2)
    ok, phase, _ = _resume(out, lights, BASE + ['--debayer-method', 'menon2007'])
    assert not ok and phase == 0
    assert 'debayer_method' in capsys.readouterr().out


def test_edited_light_file_starts_fresh(tmp_path, lights):
    out = str(tmp_path / 'o.fits')
    _save(out, lights, parse_args(BASE), 3)
    with open(lights[1].path, 'ab') as fh:
        fh.write(b'more')
    ok, _, _ = _resume(out, lights, BASE)
    assert not ok


def test_verified_checkpoint_does_not_expire(tmp_path, lights, monkeypatch):
    out = str(tmp_path / 'o.fits')
    _save(out, lights, parse_args(BASE), 3)
    real = time.time
    monkeypatch.setattr(ck.time, 'time', lambda: real() + 10 * 24 * 3600)
    ok, phase, _ = _resume(out, lights, BASE)
    assert ok and phase == 3


def test_legacy_checkpoint_without_fingerprint_behaves_as_before(tmp_path, lights):
    out = str(tmp_path / 'o.fits')
    ck.save_checkpoint(out, phase=2, lights=lights, final=lights)
    ok, phase, _ = _resume(out, lights, BASE + ['--stack-method', 'median'])
    assert ok and phase == 2


def test_every_phase_group_dest_is_fingerprinted():
    fp = ck.stack_fingerprint(parse_args(BASE), [])
    for d in ('stack_method', 'rejection_sigma', 'drizzle_scale', 'elastic_registration'):
        assert d in fp['p23']
    for d in ('debayer_method', 'white_balance', 'spike_reject', 'cosmic_ray_rejection',
              'pre_gradient_removal', 'cal_dir'):
        assert d in fp['p1']
    for d in ('stretch', 'denoiser', 'error_aware_stretch'):
        assert d not in fp['p1'] and d not in fp['p23']


def test_fingerprint_is_json_serialisable(lights):
    import json
    json.dumps(ck.stack_fingerprint(parse_args(BASE), lights))
    assert os.path.exists(lights[0].path)
