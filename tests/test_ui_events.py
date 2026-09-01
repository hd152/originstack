"""Tests for src/ui_events.py -- the desktop app's in-process state sink
(replaces the old HTTP/SSE ``WebView``; these tests port ``test_webview.py``'s
state-logic coverage, minus everything that was testing the deleted HTTP
transport)."""
import unittest

import numpy as np

from src.io_fits import preview_jpeg_bytes, render_preview_uint8
from src.ui_events import UIEvents


def _synth_rgb(H=128, W=160):
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(1000, 60, (H, W, 3)), 0, None)
    return img.astype(np.float32)


class TestInactiveNoOp(unittest.TestCase):
    def test_publishes_are_noops_when_inactive(self):
        ui = UIEvents()
        ui.log("hello")
        ui.phase(1, "x")
        ui.progress("p", 1, 10)
        ui.frame_metrics("f.fits", {'score': 1})
        ui.preview(_synth_rgb(), "cap")
        ui.summary(a=1)
        ui.run_started()
        ui.run_finished('error', 'boom')
        self.assertEqual(ui._state['log'], [])
        self.assertEqual(ui._state['phase'], 0)
        self.assertIsNone(ui._preview_bytes)
        self.assertEqual(ui._state['run_status'], 'idle')
        self.assertIsNone(ui._state['run_error'])


class TestRunState(unittest.TestCase):
    """UIEvents.run_started()/run_finished(): the desktop app's run_status
    signal, separate from summary()/done which fire once per stack_target
    call (multiple times in one hierarchical/mosaic run)."""

    def setUp(self):
        self.ui = UIEvents()
        self.ui.active = True

    def test_run_started_sets_running_and_clears_prior_state(self):
        self.ui.log("stale log line")
        self.ui.frame_metrics("f.fits", {'score': 1})
        self.ui.summary(output='old.fits')
        self.ui.phase(2, "Registration")
        self.ui.progress("Registering", 3, 10)
        before_version = self.ui._version

        self.ui.run_started()

        self.assertEqual(self.ui._state['run_status'], 'running')
        self.assertIsNone(self.ui._state['run_error'])
        self.assertEqual(self.ui._state['log'], [])
        self.assertEqual(self.ui._state['frames'], [])
        self.assertIsNone(self.ui._state['summary'])
        self.assertFalse(self.ui._state['done'])
        self.assertEqual(self.ui._state['phase'], 0)
        self.assertEqual(self.ui._state['phases'], {})
        self.assertEqual(self.ui._state['progress'], {'label': '', 'done': 0, 'total': 0})
        self.assertGreater(self.ui._version, before_version)

    def test_run_finished_ok(self):
        self.ui.run_started()
        self.ui.run_finished('ok')
        self.assertEqual(self.ui._state['run_status'], 'ok')
        self.assertIsNone(self.ui._state['run_error'])

    def test_run_finished_error_carries_message(self):
        self.ui.run_started()
        self.ui.run_finished('error', 'directory not found')
        self.assertEqual(self.ui._state['run_status'], 'error')
        self.assertEqual(self.ui._state['run_error'], 'directory not found')

    def test_summary_done_unaffected_by_run_state(self):
        """summary()/done still mean 'a target just finished' -- run_started
        resets them (new run), but run_finished must not touch them, since a
        hierarchical run's last stack_target() call already set them."""
        self.ui.run_started()
        self.ui.summary(output='final.fits')
        self.ui.run_finished('ok')
        self.assertTrue(self.ui._state['done'])
        self.assertEqual(self.ui._state['summary'], {'output': 'final.fits'})


class TestPreviewBytes(unittest.TestCase):
    def test_jpeg_roundtrip(self):
        import io

        from PIL import Image
        data = preview_jpeg_bytes(_synth_rgb(), stretch='ghs')
        self.assertIsNotNone(data)
        im = Image.open(io.BytesIO(data))
        self.assertEqual(im.size, (160, 128))

    def test_render_matches_file_path(self):
        """render_preview_uint8 is the same array save_preview_rgb encodes."""
        import io
        import os
        import tempfile

        from PIL import Image
        rgb = _synth_rgb()
        out = render_preview_uint8(rgb, stretch='ghs')
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, rgb.shape)
        from src.io_fits import save_preview_rgb
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'x.jpg')
            save_preview_rgb(rgb, p, stretch='ghs')
            with Image.open(p) as im:
                self.assertEqual(im.size, (160, 128))


class TestNamedSlotsAndSnapshot(unittest.TestCase):
    """A published preview registers a named slot fetchable by slug and
    listed in ``snapshot()`` for the compare dropdown -- the in-process
    equivalent of the old ``/named.jpg``/SSE-snapshot coverage."""

    def setUp(self):
        self.ui = UIEvents()
        self.ui.active = True

    def test_named_slot_and_snapshot(self):
        self.ui.preview(_synth_rgb(), "Final (post-processed)", slot='final',
                        min_interval=0.0)
        snap = self.ui.snapshot()
        slugs = [n['slug'] for n in snap['named']]
        self.assertIn('final', slugs)
        self.assertTrue(snap['named'][0]['src'])  # linear source retained
        self.assertEqual(snap['latest_slug'], 'final')
        data = self.ui.named_jpeg('final')
        self.assertEqual(data[:2], b'\xff\xd8')  # JPEG magic

    def test_preview_endpoint_equivalent(self):
        self.assertIsNone(self.ui.preview_jpeg())
        self.ui.preview(_synth_rgb(), "test cap", min_interval=0.0)
        data = self.ui.preview_jpeg()
        self.assertGreater(len(data), 1000)
        self.assertEqual(data[:2], b'\xff\xd8')

    def test_frame_thumbnail(self):
        self.ui.frame_preview("light_001.fits", _synth_rgb())
        snap = self.ui.snapshot()
        self.assertEqual(len(snap['frames_img']), 1)
        fid = snap['frames_img'][0]['id']
        data = self.ui.frame_jpeg(fid)
        self.assertEqual(data[:2], b'\xff\xd8')

    def test_restretch_from_source(self):
        """Re-stretch re-encodes the retained linear source with new params."""
        self.ui.preview(_synth_rgb(), "Final", slot='final', min_interval=0.0)
        data = self.ui.restretch('final', {'stretch': 'ghs', 'b': 3, 'sp': 0.2,
                                           'hp': 0.9, 'black': 1.0})
        self.assertEqual(data[:2], b'\xff\xd8')
        self.assertIsNone(self.ui.restretch('nope', {}))


class TestFrameThumbRing(unittest.TestCase):
    def test_ring_is_bounded(self):
        from src.ui_events import _MAX_FRAME_THUMBS
        ui = UIEvents()
        ui.active = True
        for i in range(_MAX_FRAME_THUMBS + 5):
            ui.frame_preview("f%d" % i, _synth_rgb())
        self.assertEqual(len(ui._frame_thumbs), _MAX_FRAME_THUMBS)


if __name__ == '__main__':
    unittest.main()
