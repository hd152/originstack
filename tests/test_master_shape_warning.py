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


def _hl(h, w, **extra):
    fi = FrameInfo.__new__(FrameInfo)
    fi.header = {'NAXIS1': w, 'NAXIS2': h, **extra}
    return fi


def _run(master, light, cal_hdr_frame=None):
    msgs = []
    import unittest.mock as m
    with m.patch.object(cli, 'safe_print', lambda s="", *a, **k: msgs.append(str(s))):
        masters = {'dark': master}
        frames = {'light': [light]}
        if cal_hdr_frame is not None:
            frames['dark'] = [cal_hdr_frame]
        dropped = cli._drop_mismatched_masters(masters, frames)
    return masters['dark'], dropped, msgs


def test_larger_master_cropped_with_header_offset():
    big = np.arange(12 * 16, dtype=np.float32).reshape(12, 16)
    out, dropped, msgs = _run(big, _hl(8, 10, XORGSUBF=4, YORGSUBF=2))
    assert dropped == [] and out.shape == (8, 10)
    assert np.array_equal(out, big[2:10, 4:14])
    assert len(msgs) == 1 and 'cropped' in msgs[0]


def test_larger_master_cropped_centred_when_small_and_symmetric():
    big = np.arange(12 * 18, dtype=np.float32).reshape(12, 18)
    out, dropped, _ = _run(big, _hl(8, 10))
    assert out.shape == (8, 10) and np.array_equal(out, big[2:10, 4:14])


def test_odd_or_asymmetric_crop_dropped():
    big = np.zeros((11, 16), np.float32)          # odd height difference
    out, dropped, msgs = _run(big, _hl(8, 10))
    assert out is None and dropped == ['dark'] and 'not small and symmetric' in msgs[0]
    big = np.zeros((14, 16), np.float32)          # centred offset (3, 3) is odd
    out, dropped, msgs = _run(big, _hl(8, 10))
    assert out is None and 'odd' in msgs[0]


def test_binning_multiple_dropped_with_reason():
    out, dropped, msgs = _run(np.zeros((16, 20), np.float32), _hl(8, 10))
    assert out is None and dropped == ['dark'] and 'binning mismatch' in msgs[0]


def test_binning_header_mismatch_dropped():
    big = np.zeros((12, 16), np.float32)
    cal = _hl(12, 16, XBINNING=2, YBINNING=2)
    out, dropped, msgs = _run(big, _hl(8, 10, XBINNING=1, YBINNING=1), cal)
    assert out is None and 'binning differs' in msgs[0]


def test_mixed_settings_split_into_groups():
    from src.frame_discovery import differing_settings, group_lights_by_filter
    a = [_hl(8, 10, XBINNING=1, YBINNING=1, GAIN=100) for _ in range(2)]
    b = [_hl(8, 10, XBINNING=2, YBINNING=2, GAIN=100)]
    assert differing_settings(a) == []
    assert set(group_lights_by_filter(a)) == {'unfiltered'}   # homogeneous: unchanged tag
    g = group_lights_by_filter(a + b)
    assert set(g) == {'unfiltered_bin1x1', 'unfiltered_bin2x2'}
    assert differing_settings(a + b) == ['binning']
    c = [_hl(8, 10, GAIN=100, FILTER='Clear'), _hl(8, 10, GAIN=200, FILTER='Clear')]
    assert set(group_lights_by_filter(c)) == {'clear_g100', 'clear_g200'}
    # a setting missing from some lights is not guessed to differ
    d = [_hl(8, 10, GAIN=100), _hl(8, 10)]
    assert differing_settings(d) == []
