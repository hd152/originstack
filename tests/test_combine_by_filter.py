"""A hierarchical (one stack per session) combine never averages stacks shot through
different filters: a real Eagle Nebula folder with two Clear and three Nebula-filter
sessions came out with every session's footprint visible and the nebula gone."""
import argparse
import os
from unittest import mock

import numpy as np
import pytest
from astropy.io import fits

import src.cli as cli


@pytest.fixture(autouse=True)
def _private_temp(tmp_path, monkeypatch):
    # per-session stacks go to tempfile.gettempdir() as <session>_stack.fits: in the
    # shared %TEMP% two xdist workers wrote session0_stack.fits at once and one failed
    d = tmp_path / 'tmp'
    d.mkdir()
    monkeypatch.setattr(cli.tempfile, 'tempdir', str(d))


def _stack(path, filt):
    hdu = fits.PrimaryHDU(np.zeros((3, 8, 8), np.float32))
    if filt is not None:
        hdu.header['FILTER'] = filt
    hdu.writeto(path)
    return str(path)


def test_group_stacks_by_filter(tmp_path):
    a = _stack(tmp_path / 'a.fits', 'Clear')
    b = _stack(tmp_path / 'b.fits', 'Nebula')
    c = _stack(tmp_path / 'c.fits', 'clear ')
    d = _stack(tmp_path / 'd.fits', None)
    assert cli._group_stacks_by_filter([a, b, c, d]) == {
        'clear': [a, c], 'nebula': [b], 'unfiltered': [d]}


def _run(tmp_path, filters):
    """process_directory over len(filters) session folders whose stacks carry those
    FILTER values; returns the (paths, output) of every _combine_stacks call."""
    root = tmp_path / 'target'
    root.mkdir()
    for i in range(len(filters)):
        (root / f'session{i}').mkdir()
    made = iter(filters)

    def fake_stack_target(frames, output, args, masters, stats):
        return _stack(output, next(made))

    calls = []
    light = mock.Mock(header={'NAXIS1': 8, 'NAXIS2': 8})
    args = argparse.Namespace(
        skip_step=[], hierarchical=True, mosaic=False, combine_sessions=False,
        dry_run=False, health_check=False, preset=None, verbose=False,
        stack_method='auto', _explicit_cli_dests=set())
    with mock.patch.object(cli, 'discover_frames', return_value={
                'light': [light] * 5, 'dark': [], 'flat': [], 'bias': []}), \
         mock.patch.object(cli, '_load_calibration_dir', return_value={'dark': [], 'flat': [], 'bias': []}), \
         mock.patch.object(cli, 'group_lights_by_filter', side_effect=lambda lights: {'x': lights}), \
         mock.patch.object(cli, '_build_masters', return_value={}), \
         mock.patch.object(cli, 'stack_target', side_effect=fake_stack_target), \
         mock.patch.object(cli, '_combine_stacks',
                           side_effect=lambda paths, out, a, eff: calls.append((len(paths), out))):
        cli.process_directory(str(root), str(tmp_path / 'out.fits'), args)
    return [(n, os.path.basename(out)) for n, out in calls]


def test_mixed_filter_sessions_are_combined_per_filter(tmp_path):
    calls = _run(tmp_path, ['Clear', 'Clear', 'Nebula', 'Nebula', 'Nebula'])
    assert sorted(calls) == [(2, 'out_clear.fits'), (3, 'out_nebula.fits')]


def test_a_single_filter_still_makes_one_combined_stack(tmp_path):
    assert _run(tmp_path, ['Clear', 'Clear', 'Clear']) == [(3, 'out.fits')]


def test_no_filter_headers_combine_as_before(tmp_path):
    assert _run(tmp_path, [None, None]) == [(2, 'out.fits')]
