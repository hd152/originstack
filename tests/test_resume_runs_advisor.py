"""Every way into Phases 2-4 must run target inference and the --auto advisor.
Only the phase-3 resume and the fresh path did: a run resumed from a phase-1
or phase-2 checkpoint registered, stacked and post-processed with bare CLI
defaults (no galaxy mode, SCNR or deconvolution presets)."""
import json
import os
import tempfile
from unittest import mock

import pytest

from src import pipeline
from src.checkpoint import _checkpoint_dir, _ckpt_json_path
from src.models import FrameInfo, ProcessingStats
from tests.test_e2e import _create_synthetic_dataset, _make_minimal_args


def _lights(paths):
    return [FrameInfo(path=p, type='light',
                      header={'BAYERPAT': 'RGGB', 'EXPTIME': 120.0,
                              'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
            for p in paths['light']]


@pytest.mark.parametrize('phase', [1, 2, 3])
def test_resume_from_any_phase_runs_the_advisor(phase):
    with tempfile.TemporaryDirectory() as tmp:
        paths = _create_synthetic_dataset(tmp)
        out = os.path.join(tmp, 'stacked.fits')
        masters = {'dark': None, 'flat': None, 'bias': None, 'dark_exptime': None}
        # First run leaves a phase-3 checkpoint behind.
        first = _make_minimal_args(no_resume=False, keep_checkpoint=True)
        assert pipeline.stack_target(_lights(paths), out, first, masters, ProcessingStats())
        ckpt = _ckpt_json_path(_checkpoint_dir(out))
        with open(ckpt) as fh:
            state = json.load(fh)
        state['phase'] = phase
        with open(ckpt, 'w') as fh:
            json.dump(state, fh)

        calls = []
        again = _make_minimal_args(no_resume=False, keep_checkpoint=True, auto=True)
        with mock.patch.object(pipeline, '_run_auto_advisor',
                               side_effect=lambda final, args, **kw: calls.append(len(final))):
            assert pipeline.stack_target(_lights(paths), out, again, masters,
                                         ProcessingStats())
        assert calls == [len(paths['light'])]
        assert hasattr(again, '_inferred_type')
