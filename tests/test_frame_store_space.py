"""frame_store refuses up front when the temp disk cannot hold an array, instead
of failing when the sparse file's pages are written deep into a run."""
import collections

import pytest

import src.frame_store as fs


def test_refuses_with_a_clear_message_when_temp_disk_is_too_small(monkeypatch):
    Usage = collections.namedtuple('Usage', 'total used free')
    monkeypatch.setattr('shutil.disk_usage', lambda p: Usage(100e9, 98e9, 2e9))
    store = fs.FrameStore('disk')
    with pytest.raises(fs.FrameStoreSpaceError, match=r'stack aligned.*GB needed.*GB free'):
        store.create('stack_aligned_', 'float32', (100, 1000, 1000, 3))   # 1.2 GB > 2 - 1 GB
    store.cleanup()


def test_small_arrays_still_go_to_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(fs.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(fs.tempfile, 'tempdir', str(tmp_path))
    store = fs.FrameStore('disk')
    arr, spec = store.create('small_', 'float32', (2, 8, 8, 3))
    assert spec and arr.shape == (2, 8, 8, 3)
    store.cleanup()
