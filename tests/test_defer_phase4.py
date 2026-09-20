"""Hierarchical multi-session runs post-process once, on the combined stack.

Each session's stack is only an input to the combine (which reads the linear FITS),
so running Phase 4 on it is wasted -- but a single session, a filter-split group and
the combined stack itself must still get it.
"""
import argparse
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from astropy.io import fits

import src.cli as cli
from tests.test_e2e import TestE2EDrizzlePixfrac, _create_synthetic_dataset


class TestCliSetsDeferFlag(unittest.TestCase):
    def _flags(self, n_subdirs, fits_in_root=False):
        seen = []

        def fake_stack_target(frames, output, args, masters, stats):
            seen.append(getattr(args, '_defer_phase4', None))
            return None

        fake_light = mock.Mock(header={'NAXIS1': 8, 'NAXIS2': 8})
        with tempfile.TemporaryDirectory() as root:
            for i in range(n_subdirs):
                os.mkdir(os.path.join(root, f"session{i}"))
            if fits_in_root:
                open(os.path.join(root, 'a.fits'), 'w').close()
            args = argparse.Namespace(
                skip_step=[], hierarchical=True, mosaic=False, combine_sessions=False,
                dry_run=False, health_check=False, preset=None, verbose=False,
                stack_method='auto', _explicit_cli_dests=set())
            with mock.patch.object(cli, 'discover_frames', return_value={
                        'light': [fake_light] * 5, 'dark': [], 'flat': [], 'bias': []}), \
                 mock.patch.object(cli, '_load_calibration_dir', return_value={
                        'dark': [], 'flat': [], 'bias': []}), \
                 mock.patch.object(cli, 'group_lights_by_filter', side_effect=lambda l: {'L': l}), \
                 mock.patch.object(cli, '_build_masters', return_value={}), \
                 mock.patch.object(cli, 'stack_target', side_effect=fake_stack_target):
                cli.process_directory(root, os.path.join(root, 'out.fits'), args)
            self.assertFalse(getattr(args, '_defer_phase4', False),
                             "the flag must not outlive the stack_target call")
        return seen

    def test_several_sessions_defer_phase4(self):
        self.assertEqual(self._flags(3), [True, True, True])

    def test_one_session_keeps_phase4(self):
        self.assertEqual(self._flags(1), [False])

    def test_single_folder_keeps_phase4(self):
        self.assertEqual(self._flags(0, fits_in_root=True), [False])


class TestStackTargetHonoursDeferFlag(TestE2EDrizzlePixfrac):
    def _count_phase4(self, **overrides):
        import src.pipeline as pl
        calls = []
        real = pl.postprocess_stack

        def counting(*a, **k):
            calls.append(1)
            return real(*a, **k)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, n_lights=4)
            with mock.patch.object(pl, 'postprocess_stack', counting):
                data = self._run_drizzle(tmpdir, paths, 1.0, **overrides)
        return len(calls), data

    def test_deferred_stack_skips_phase4_and_keeps_the_linear_fits(self):
        n_normal, normal = self._count_phase4()
        n_deferred, deferred = self._count_phase4(_defer_phase4=True)
        self.assertEqual(n_normal, 1)
        self.assertEqual(n_deferred, 0)
        # the FITS a combine reads is the linear stack either way
        np.testing.assert_array_equal(normal, deferred)
        self.assertTrue(np.isfinite(deferred).all())


if __name__ == '__main__':
    unittest.main()
