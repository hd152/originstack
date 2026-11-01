"""A master whose shape differs from the lights is reported once and disabled."""
import numpy as np

from src import cli
from src.models import FrameInfo


def _light(h, w):
    fi = FrameInfo.__new__(FrameInfo)
    fi.header = {'NAXIS1': w, 'NAXIS2': h}
    return fi


def test_mismatched_masters_warn_once_per_kind(monkeypatch):
    msgs = []
    monkeypatch.setattr(cli, 'safe_print', lambda m="", *a, **k: msgs.append(str(m)))
    frames = {'light': [_light(8, 10), _light(8, 10)]}
    masters = {'dark': np.zeros((6, 6), np.float32),
               'flat': np.ones((8, 10), np.float32),
               'bias': np.zeros((4, 4), np.float32)}
    dropped = cli._drop_mismatched_masters(masters, frames)
    assert dropped == ['bias', 'dark']
    assert masters['dark'] is None and masters['bias'] is None
    assert masters['flat'] is not None
    assert len(msgs) == 2
    assert any('dark master ignored' in m and '6×6' in m and '10×8' in m for m in msgs)
    assert any('bias master ignored' in m for m in msgs)


def test_matching_or_unknown_shape_is_silent(monkeypatch):
    msgs = []
    monkeypatch.setattr(cli, 'safe_print', lambda m="", *a, **k: msgs.append(str(m)))
    masters = {'dark': np.zeros((8, 10), np.float32)}
    assert cli._drop_mismatched_masters(masters, {'light': [_light(8, 10)]}) == []
    assert cli._drop_mismatched_masters(masters, {'light': []}) == []
    nohdr = FrameInfo.__new__(FrameInfo)
    nohdr.header = {}
    assert cli._drop_mismatched_masters(masters, {'light': [nohdr]}) == []
    assert masters['dark'] is not None and not msgs
