"""process_directory's per-target loop must hand each target its own
directory. The calibration-analysis block used to assign the master dark to
``d`` -- the loop variable holding the target directory -- so every session
with darks got ``args._input_directory`` = the dark array, and pipeline.py
silently fell back to the parent folder for info.json and target inference."""
import argparse
import os
import tempfile
from unittest import mock

import numpy as np

from src import cli


def test_each_target_gets_its_own_directory_when_darks_exist():
    seen = []

    def fake_stack_target(frames, outp, args, masters, stats):
        seen.append(args._input_directory)
        return None

    light = mock.Mock(header={'NAXIS1': 8, 'NAXIS2': 8})
    dark = mock.Mock(header={'NAXIS1': 8, 'NAXIS2': 8, 'EXPTIME': 10.0})
    with tempfile.TemporaryDirectory() as root:
        dirs = [os.path.join(root, f"session{i}") for i in range(2)]
        for p in dirs:
            os.mkdir(p)
        args = argparse.Namespace(
            skip_step=[], hierarchical=True, mosaic=False, combine_sessions=False,
            dry_run=False, health_check=False, preset=None, verbose=False,
            stack_method='auto', _explicit_cli_dests=set())
        with mock.patch.object(cli, 'discover_frames', return_value={
                    'light': [light], 'dark': [dark], 'flat': [], 'bias': []}), \
             mock.patch.object(cli, '_load_calibration_dir', return_value={
                    'dark': [], 'flat': [], 'bias': []}), \
             mock.patch.object(cli, 'group_lights_by_filter',
                               side_effect=lambda lights: {'L': lights}), \
             mock.patch.object(cli, '_build_masters', side_effect=lambda f, s, a: {
                    'dark': np.full((8, 8), 50.0, np.float32), 'dark_exptime': 10.0}), \
             mock.patch.object(cli, 'stack_target', side_effect=fake_stack_target):
            cli.process_directory(root, os.path.join(root, 'out.fits'), args)

    assert seen == sorted(dirs)
