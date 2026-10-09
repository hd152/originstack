"""Temporary files: --temp-dir, cleanup at the end of every run, and removal of
files a crashed run left behind (an 850-light session filled the system drive and
the reporter found the temp files were never deleted)."""
import os
import tempfile
import time

import numpy as np
import pytest

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
    n, freed = cleanup.sweep_orphans([str(tmp_path)], min_age_hours=6)
    assert n == len(gone) and freed == 10 * len(gone)
    assert not any(p.exists() for p in gone) and all(p.exists() for p in keep)


@pytest.mark.skipif(os.name != 'nt', reason='Windows refuses to delete an open file')
def test_windows_sweep_takes_any_age_but_never_an_open_file(tmp_path):
    closed = tmp_path / 'stack_rgb_done.dat'
    closed.write_bytes(b'x')
    busy = tmp_path / 'stack_aligned_busy.dat'
    with open(busy, 'wb') as fh:          # another instance still running
        fh.write(b'x')
        fresh_stack = tmp_path / 'M42_stack.fits'   # closed, waiting for the combine
        fresh_stack.write_bytes(b'x')
        n, _ = cleanup.sweep_orphans([str(tmp_path)])
        assert n == 1 and not closed.exists() and busy.exists() and fresh_stack.exists()


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


def test_failed_combine_really_keeps_the_per_session_stacks(tmp_path, monkeypatch):
    """The log says the per-session stacks are kept for a manual --merge; the
    end-of-run cleanup must not then delete them."""
    import argparse
    from unittest import mock

    from astropy.io import fits

    import src.cli as cli
    monkeypatch.setattr(cli.tempfile, 'tempdir', str(tmp_path / 'tmp'))
    os.makedirs(tmp_path / 'tmp')
    root = tmp_path / 'target'
    for i in range(2):
        (root / f'session{i}').mkdir(parents=True)

    def fake_stack_target(frames, output, args, masters, stats):
        hdu = fits.PrimaryHDU(np.zeros((3, 8, 8), np.float32))
        hdu.header['RAWSTACK'] = True
        hdu.header['NFRAMES'] = 5
        hdu.header['INTGTIME'] = 50.0
        hdu.writeto(output)
        return output

    light = mock.Mock(header={'NAXIS1': 8, 'NAXIS2': 8})
    args = argparse.Namespace(skip_step=[], hierarchical=True, mosaic=False,
                              combine_sessions=False, dry_run=False, health_check=False,
                              preset=None, verbose=False, stack_method='auto',
                              _explicit_cli_dests=set())
    with mock.patch.object(cli, 'discover_frames', return_value={
                'light': [light] * 5, 'dark': [], 'flat': [], 'bias': []}), \
         mock.patch.object(cli, '_load_calibration_dir', return_value={
                'dark': [], 'flat': [], 'bias': []}), \
         mock.patch.object(cli, 'group_lights_by_filter', side_effect=lambda lights: {'L': lights}), \
         mock.patch.object(cli, '_build_masters', return_value={}), \
         mock.patch.object(cli, 'stack_target', side_effect=fake_stack_target), \
         mock.patch('src.merge.merge_previous_stacks', side_effect=ValueError('no match')):
        with pytest.raises(ValueError):
            cli.process_directory(str(root), str(tmp_path / 'out.fits'), args)
    cleanup.cleanup_now()
    left = sorted(p.name for p in (tmp_path / 'tmp').glob('*_stack.fits'))
    assert left == ['session0_stack.fits', 'session1_stack.fits']
