"""--target-type: the user tells --auto what was imaged (the desktop app's
target cards set it)."""
from unittest.mock import patch

import pytest

from src import checkpoint as ck
from src import pipeline
from src.cli import parse_args
from src.models import FrameInfo


def test_flag_parses_and_rejects_unknown_types():
    assert parse_args(['-d', 'x', '--target-type', 'galaxy']).target_type == 'galaxy'
    assert parse_args(['-d', 'x']).target_type is None
    with pytest.raises(SystemExit):
        parse_args(['-d', 'x', '--target-type', 'unknown'])


def test_warns_without_auto(capsys):
    parse_args(['-d', 'x', '--no-auto', '--target-type', 'galaxy'])
    assert '--target-type has no effect with --no-auto' in capsys.readouterr().out


def _advise(args, inferred):
    seen = {}

    def fake_advisor(final, a, prior_type=None, prior_confidence=0.0):
        seen.update(prior_type=prior_type, prior_confidence=prior_confidence)

    with patch('src.target_inference.infer_target_from_metadata', return_value=inferred), \
            patch.object(pipeline, '_run_auto_advisor', fake_advisor):
        pipeline._infer_target_and_advise([], args, 'x', use_simbad=False)
    return seen


def test_target_type_overrides_the_inferred_type_at_full_confidence():
    args = parse_args(['-d', 'x', '--target-type', 'emission_nebula'])
    seen = _advise(args, ('M51', 'galaxy', 1.0, 'session'))
    assert seen == {'prior_type': 'emission_nebula', 'prior_confidence': 1.0}
    # The name still comes from the metadata; the type and source are the user's.
    assert (args._inferred_target, args._inferred_type, args._inferred_source) == \
        ('M51', 'emission_nebula', 'user')


def test_without_target_type_the_inferred_type_is_used():
    args = parse_args(['-d', 'x'])
    seen = _advise(args, ('M51', 'galaxy', 0.9, 'header'))
    assert seen == {'prior_type': 'galaxy', 'prior_confidence': 0.9}


def test_changing_target_type_restacks_from_phase1(tmp_path):
    # --auto derives stacking settings from the type, so a phase-3 stack made
    # for one type must not be reused for another.
    p = tmp_path / 'Light_000.fits'
    p.write_bytes(b'x' * 100)
    lights = [FrameInfo(path=str(p), type='light', header={})]
    out = str(tmp_path / 'o.fits')
    fp = ck.stack_fingerprint(parse_args(['-d', 'x']), lights)
    ck.save_checkpoint(out, phase=3, lights=lights, final=lights, fingerprint=fp)
    import numpy as np
    ck.save_raw_stack(out, np.zeros((4, 4, 3), np.float32))
    ok, phase, _ = ck.can_resume(
        out, lights, ck.stack_fingerprint(parse_args(['-d', 'x', '--target-type', 'galaxy']),
                                          lights))
    assert ok and phase == 1
