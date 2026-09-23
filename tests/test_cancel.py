"""Tests for cooperative run cancellation (the desktop app's Cancel button):
``RunCancelled`` (src/models.py), ``frame_processor._check_cancel``,
``cli.process_directory``'s per-target guard, and ``RunManager``'s
cancel()/is_cancelling() plumbing (src/desktop_control.py).

Cancellation is cooperative -- a ``threading.Event`` checked at a handful of
checkpoints, not a thread kill -- so these tests exercise the checkpoints
directly rather than timing a real multi-minute stacking run.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from src.models import RunCancelled


class TestCheckCancel(unittest.TestCase):
    def test_raises_when_event_set(self):
        from src.frame_processor import _check_cancel
        ns = argparse.Namespace(_cancel_event=threading.Event())
        ns._cancel_event.set()
        with self.assertRaises(RunCancelled):
            _check_cancel(ns)

    def test_no_op_when_event_unset(self):
        from src.frame_processor import _check_cancel
        ns = argparse.Namespace(_cancel_event=threading.Event())
        _check_cancel(ns)  # must not raise

    def test_no_op_without_a_cancel_event_at_all(self):
        """A plain CLI run never sets args._cancel_event -- must be a no-op,
        not an AttributeError."""
        from src.frame_processor import _check_cancel
        _check_cancel(argparse.Namespace())


class TestExecuteFrameProcessingStopsEarly(unittest.TestCase):
    def test_sequential_path_raises_before_processing_any_frame(self):
        """A pre-cancelled event must stop the sequential dispatch path
        (n < 4, no process pool) before it touches the first frame --
        cheap and deterministic, unlike timing a real multi-frame run."""
        from src.frame_processor import execute_frame_processing
        from src.models import FrameInfo, ProcessingStats

        lights = [FrameInfo(path='does-not-exist.fits', type='light', header={})]
        args = argparse.Namespace(
            parallel=1, verbose=False, debayer_method='malvar', white_balance='grayworld',
            ca_correction=False, cosmic_ray_rejection=False, advanced_metrics=True,
            pre_gradient_removal=False, trail_reject=False,
            _cancel_event=threading.Event())
        args._cancel_event.set()

        with patch('src.frame_processor._process_single_frame') as mock_proc:
            with self.assertRaises(RunCancelled):
                execute_frame_processing(
                    lights, {}, args,
                    mem_rgb=None, mem_lum=None, mm_rgb_path='', mm_lum_path='',
                    cached_lums=[None], rgb_shape=(1, 4, 4, 3), lum_shape=(1, 4, 4),
                    rejected_reasons={}, stats=ProcessingStats())
            mock_proc.assert_not_called()


class TestProcessDirectoryPerTargetGuard(unittest.TestCase):
    def test_raises_before_the_first_target_when_precancelled(self):
        from src.cli import process_directory

        with tempfile.TemporaryDirectory() as tmp:
            for name in ('session_a', 'session_b'):
                d = os.path.join(tmp, name)
                os.makedirs(d)
                open(os.path.join(d, 'light_000.fits'), 'w').close()

            args = argparse.Namespace(hierarchical=True, mosaic=False,
                                      preset=None, _cancel_event=threading.Event())
            args._cancel_event.set()

            with patch('src.cli._want_combine_sessions', return_value=False), \
                 patch('src.cli.discover_frames') as mock_discover:
                with self.assertRaises(RunCancelled):
                    process_directory(tmp, os.path.join(tmp, 'out.fits'), args)
                mock_discover.assert_not_called()


class TestRunManagerCancel(unittest.TestCase):
    def test_cancel_is_a_noop_while_idle(self):
        from src.desktop_control import RunManager
        rm = RunManager()
        rm.cancel()
        self.assertFalse(rm._cancel_event.is_set())

    def test_cancel_sets_the_event_while_running(self):
        from src.desktop_control import RunManager
        rm = RunManager()
        rm.status = 'running'
        rm.cancel()
        self.assertTrue(rm._cancel_event.is_set())

    def test_is_cancelling_reflects_both_status_and_event(self):
        from src.desktop_control import RunManager
        rm = RunManager()
        self.assertFalse(rm.is_cancelling())
        rm.status = 'running'
        self.assertFalse(rm.is_cancelling())
        rm._cancel_event.set()
        self.assertTrue(rm.is_cancelling())
        rm.status = 'ok'
        self.assertFalse(rm.is_cancelling(), "a finished run is not 'cancelling'")

    def test_start_gives_each_run_a_fresh_event(self):
        """A cancel() from a previous run must not leak into the next one."""
        from src.desktop_control import RunManager
        rm = RunManager()
        rm._cancel_event.set()
        with patch('src.cli.process_directory'):
            rm.start({'directory': 'foo', 'output': 'bar.fits'})
            rm.thread.join(timeout=5)
        self.assertEqual(rm.status, 'ok')

    def test_start_threads_its_cancel_event_onto_args(self):
        """frame_processor._check_cancel / cli.process_directory's guard
        read args._cancel_event -- RunManager._run must set it to the
        SAME Event object cancel() sets."""
        from src.desktop_control import RunManager
        rm = RunManager()
        captured = {}

        def _capture(directory, output, args):
            captured['ev'] = args._cancel_event

        with patch('src.cli.process_directory', side_effect=_capture):
            rm.start({'directory': 'foo', 'output': 'bar.fits'})
            rm.thread.join(timeout=5)
        self.assertIs(captured['ev'], rm._cancel_event)

    def test_pipeline_raising_run_cancelled_sets_status_cancelled_not_error(self):
        from src.desktop_control import RunManager
        rm = RunManager()
        with patch('src.cli.process_directory', side_effect=RunCancelled('stop')):
            result = rm.start({'directory': 'foo', 'output': 'bar.fits'})
            self.assertTrue(result['ok'])
            rm.thread.join(timeout=5)
        self.assertEqual(rm.status, 'cancelled')

    def test_run_finished_receives_cancelled_status(self):
        from src.desktop_control import RunManager
        rm = RunManager()
        with patch('src.cli.process_directory', side_effect=RunCancelled('stop')), \
             patch('src.ui_events.get_ui_events') as mock_get_wv:
            mock_wv = mock_get_wv.return_value
            rm.start({'directory': 'foo', 'output': 'bar.fits'})
            rm.thread.join(timeout=5)
        mock_wv.run_finished.assert_called_once_with('cancelled', None)


if __name__ == '__main__':
    unittest.main()
