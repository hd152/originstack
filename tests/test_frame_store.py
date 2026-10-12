"""--frame-store: where the per-session frame arrays live (src/frame_store.py)."""
import importlib.util
import multiprocessing as mp
import os
import sys
import tempfile

import numpy as np
import pytest

from src import frame_store as fs


@pytest.fixture
def mem(monkeypatch):
    """Pretend memory (MB): available, total."""
    state = {'avail': 40_000.0, 'total': 64_000.0}
    monkeypatch.setattr(fs, '_available_mb', lambda: (state['avail'], state['total']))
    return state


@pytest.fixture
def disk(monkeypatch):
    """Pretend temp-disk free space (bytes)."""
    import shutil
    from collections import namedtuple
    du = namedtuple('du', 'total used free')
    state = {'free': 500e9, 'total': 1000e9}
    monkeypatch.setattr(shutil, 'disk_usage',
                        lambda p: du(state['total'], state['total'] - state['free'], state['free']))
    return state


def _placed(store, prefix):
    return store.placement[prefix].split()[0]


def test_available_memory_falls_back_to_windows_api_without_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, 'psutil', None)
    monkeypatch.setattr(fs.sys, 'platform', 'win32')
    monkeypatch.setattr(fs, '_windows_memory_mb', lambda: (32_000.0, 64_000.0))
    assert fs._available_mb() == (32_000.0, 64_000.0)


def test_auto_aligned_prefers_ram_when_it_fits(mem, disk):
    s = fs.FrameStore('auto')
    a, spec = s.create('aligned_', 'float32', (4, 100, 100, 3))
    assert isinstance(a, fs._RamArray) and spec == '' and _placed(s, 'aligned_') == 'RAM'
    a.flush()                                 # memmap API is accepted
    s.cleanup()


def test_auto_respects_the_memory_reserve(mem, disk):
    mem['avail'] = 13_000.0                   # 20% of 64 GB reserve = 12.8 GB
    s = fs.FrameStore('auto')
    a, spec = s.create('aligned_', 'float32', (1, 1000, 1000, 1), reserve_mb=1_000.0)
    assert _placed(s, 'aligned_') == 'disk' and os.path.exists(spec)
    s.cleanup()
    assert not os.path.exists(spec)


def test_phase1_compression_is_selected_when_arrays_exceed_safe_ram(mem):
    # Safe budget: 40 GB available - 12.8 GB reserve - 20 GB for workers.
    assert fs.FrameStore.phase1_compression_needed(7_199_000_000, 20_000.0) is False
    assert fs.FrameStore.phase1_compression_needed(7_201_000_000, 20_000.0) is True


def test_phase1_compression_auto_is_undecided_without_memory_telemetry(monkeypatch):
    monkeypatch.setattr(fs, '_available_mb', lambda: (None, None))
    assert fs.FrameStore.phase1_compression_needed(1, 0.0) is None


def test_auto_phase1_arrays_stay_on_disk_unless_the_disk_is_short(mem, disk):
    s = fs.FrameStore('auto')
    a, spec = s.create('rgb_', 'float32', (2, 64, 64, 3), shared=True, prefer='disk')
    assert _placed(s, 'rgb_') == 'disk'
    disk['free'] = 50e9                       # below 10% of a 1 TB disk
    b, spec2 = s.create('lum_', 'float32', (2, 64, 64), shared=True, prefer='disk')
    assert _placed(s, 'lum_') == 'RAM' and spec2.startswith(fs.SHM_PREFIX)
    del a, b
    s.cleanup()


def test_forced_modes(mem, disk):
    s = fs.FrameStore('ram')
    mem['avail'] = 0.0
    a, _ = s.create('x_', 'float32', (10,))
    assert _placed(s, 'x_') == 'RAM'
    s2 = fs.FrameStore('disk')
    b, path = s2.create('y_', 'float32', (10,))
    assert _placed(s2, 'y_') == 'disk' and os.path.exists(path)
    del a, b
    s.cleanup()
    s2.cleanup()


def _child_write(spec, shape, value):
    arr = fs.open_frame_array(spec, 'float32', shape)
    arr[1] = value


def test_compressed_frames_round_trip_and_support_channel_updates(mem, disk):
    pytest.importorskip('astro_native')
    store = fs.FrameStore('disk')
    shape = (3, 8, 7, 3)
    arr, spec = store.create('stack_rgb_', 'float32', shape, compressed=True)
    source = np.random.default_rng(42).normal(size=shape).astype(np.float32)
    for i in range(shape[0]):
        arr[i] = source[i]
    np.testing.assert_array_equal(np.asarray(arr), source)

    arr[1, :, :, 2] = 0.0
    expected = source.copy()
    expected[1, :, :, 2] = 0.0
    np.testing.assert_array_equal(arr[1], expected[1])
    np.testing.assert_array_equal(arr[1:3], expected[1:3])
    assert spec.startswith(fs.COMPRESSED_PREFIX)
    assert store.placement['stack_rgb_'].startswith('compressed')
    paths = list(store._files)
    del arr
    store.cleanup()
    assert all(not os.path.exists(path) for path in paths)


def test_compressed_frame_store_is_writable_from_another_process(mem, disk):
    pytest.importorskip('astro_native')
    store = fs.FrameStore('disk')
    shape = (3, 8, 8)
    arr, spec = store.create('stack_lum_', 'float32', shape, compressed=True)
    p = mp.get_context('spawn').Process(target=_child_write, args=(spec, shape, 7.5))
    p.start()
    p.join(60)
    assert p.exitcode == 0
    np.testing.assert_array_equal(arr[1], np.full(shape[1:], 7.5, dtype=np.float32))
    with pytest.raises(FileNotFoundError):
        _ = arr[0]
    del arr
    store.cleanup()


def test_shared_ram_array_is_written_by_another_process(mem, disk):
    s = fs.FrameStore('ram')
    shape = (3, 8, 8)
    arr, spec = s.create('shm_', 'float32', shape, shared=True)
    arr[:] = 0
    p = mp.get_context('spawn').Process(target=_child_write, args=(spec, shape, 7.5))
    p.start()
    p.join(60)
    assert p.exitcode == 0
    assert np.all(arr[1] == 7.5) and np.all(arr[0] == 0)
    del arr
    s.cleanup()


def test_disk_spec_opens_as_memmap(mem, disk):
    s = fs.FrameStore('disk')
    arr, path = s.create('f_', 'float32', (2, 4))
    arr[:] = 3.0
    arr.flush()
    view = fs.open_frame_array(path, 'float32', (2, 4))
    assert np.all(view == 3.0)
    del view, arr
    s.cleanup()
    assert path.startswith(tempfile.gettempdir()) and not os.path.exists(path)


def test_pipeline_results_do_not_depend_on_placement(tmp_path):
    """The whole stack through stack_target, process-pool Phase 1 included."""
    from astropy.io import fits

    from src.models import FrameInfo, ProcessingStats
    from src.pipeline import stack_target
    from tests.test_e2e import _create_synthetic_dataset, _make_minimal_args

    paths = _create_synthetic_dataset(str(tmp_path), n_lights=6)
    lights = [FrameInfo(path=p, type='light',
                        header={'BAYERPAT': 'RGGB', 'EXPTIME': 120.0,
                                'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
              for p in paths['light']]
    out = {}
    modes = ['disk', 'ram']
    if importlib.util.find_spec('astro_native') is not None:
        modes.append('compressed')
    for mode in modes:
        o = str(tmp_path / f'stack_{mode}.fits')
        args = _make_minimal_args(parallel=2, frame_store=mode if mode != 'compressed' else 'disk',
                                  no_resume=True,
                                  stack_method='sigma_clip')
        args.compress_phase1 = mode == 'compressed'
        stack_target([FrameInfo(path=f.path, type='light', header=dict(f.header))
                      for f in lights], o, args, {}, ProcessingStats())
        out[mode] = np.array(fits.getdata(o, memmap=False))
    np.testing.assert_array_equal(out['disk'], out['ram'])
    if 'compressed' in out:
        np.testing.assert_array_equal(out['disk'], out['compressed'])
