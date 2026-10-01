"""--frame-store: where the per-session frame arrays live (src/frame_store.py)."""
import multiprocessing as mp
import os
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


def test_auto_aligned_prefers_ram_when_it_fits(mem, disk):
    s = fs.FrameStore('auto')
    a, spec = s.create('aligned_', 'float32', (4, 100, 100, 3))
    assert isinstance(a, fs._RamArray) and spec == '' and _placed(s, 'aligned_') == 'RAM'
    a.flush()                                 # memmap API is accepted
    s.cleanup()


def test_auto_respects_the_memory_reserve(mem, disk):
    mem['avail'] = 20_000.0                   # 30% of 64 GB reserve = 19.2 GB
    s = fs.FrameStore('auto')
    a, spec = s.create('aligned_', 'float32', (1, 1000, 1000, 1), reserve_mb=1_000.0)
    assert _placed(s, 'aligned_') == 'disk' and os.path.exists(spec)
    s.cleanup()
    assert not os.path.exists(spec)


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
    for m in ('disk', 'ram'):
        o = str(tmp_path / f'stack_{m}.fits')
        args = _make_minimal_args(parallel=2, frame_store=m, no_resume=True,
                                  stack_method='sigma_clip')
        stack_target([FrameInfo(path=f.path, type='light', header=dict(f.header))
                      for f in lights], o, args, {}, ProcessingStats())
        out[m] = np.array(fits.getdata(o, memmap=False))
    np.testing.assert_array_equal(out['disk'], out['ram'])
