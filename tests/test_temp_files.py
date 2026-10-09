"""Temporary files: --temp-dir, cleanup at the end of every run, and removal of
files a crashed run left behind (an 850-light session filled the system drive and
the reporter found the temp files were never deleted)."""
import os
import tempfile
import time

import src.cleanup as cleanup
import src.frame_store as fs
from src.cli import apply_temp_dir


def test_sweep_removes_only_old_originstack_files(tmp_path):
    old = time.time() - 24 * 3600
    keep, gone = [], []
    for name in ('stack_rgb_abc.dat', 'stack_aligned_x.dat', 'M42_stack.fits', 'M42_stack.jpg',
                 'M42_stack_config.toml'):
        p = tmp_path / name
        p.write_bytes(b'x' * 10)
        os.utime(p, (old, old))
        gone.append(p)
    fresh = tmp_path / 'stack_rgb_new.dat'
    fresh.write_bytes(b'x')                             # a run in progress
    other = tmp_path / 'holiday.jpg'
    other.write_bytes(b'x')
    os.utime(other, (old, old))                         # not ours
    keep += [fresh, other]
    n, freed = cleanup.sweep_orphans([str(tmp_path)])
    assert n == len(gone) and freed == 10 * len(gone)
    assert not any(p.exists() for p in gone) and all(p.exists() for p in keep)


def test_cleanup_now_closes_and_deletes_an_open_frame_store(tmp_path, monkeypatch):
    monkeypatch.setattr(fs.tempfile, 'tempdir', str(tmp_path))
    store = fs.FrameStore('disk')
    arr, path = store.create('stack_aligned_', 'float32', (4, 16, 16, 3))
    arr[:] = 1.0                                       # mapped and written, as mid-run
    assert os.path.exists(path)
    cleanup.cleanup_now()                              # the run failed: nobody called cleanup()
    assert not os.path.exists(path)


def test_temp_dir_applies_and_resets(tmp_path):
    system = tempfile.gettempdir()
    try:
        got = apply_temp_dir(str(tmp_path / 'big'))
        assert got == str(tmp_path / 'big') and os.path.isdir(got)
        assert tempfile.gettempdir() == got and os.environ.get('TEMP') == got
        store = fs.FrameStore('disk')
        _, path = store.create('stack_rgb_', 'float32', (2, 4, 4, 3))
        assert os.path.dirname(path) == got
        store.cleanup()
    finally:
        apply_temp_dir(None)
    assert tempfile.gettempdir() == system


def test_run_manager_records_output_and_cleans_up_after_a_failed_run(tmp_path, monkeypatch):
    from src import cli
    from src import desktop_control as dc
    leftover = tmp_path / 'leftover.dat'

    def boom(directory, output, args):
        leftover.write_bytes(b'x')
        cleanup.register(str(leftover))
        raise RuntimeError('disk full')
    monkeypatch.setattr(cli, 'process_directory', boom)
    rm = dc.RunManager()
    (tmp_path / 'lights').mkdir()
    rm._run(['-d', str(tmp_path / 'lights'), '-o', str(tmp_path / 'out.fits'),
             '--no-color-calibrate', '--offline'])
    assert rm.status == 'error'
    assert not leftover.exists()
    assert rm.last_output == os.path.abspath(str(tmp_path / 'out.fits'))
