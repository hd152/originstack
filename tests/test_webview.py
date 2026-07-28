"""Tests for the live stacking web view (--web-view)."""
import json
import unittest
import urllib.request

import numpy as np

from src.webview import WebView
from src.io_fits import preview_jpeg_bytes, render_preview_uint8


def _synth_rgb(H=128, W=160):
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(1000, 60, (H, W, 3)), 0, None)
    return img.astype(np.float32)


class TestInactiveNoOp(unittest.TestCase):
    def test_publishes_are_noops_when_inactive(self):
        wv = WebView()
        wv.log("hello")
        wv.phase(1, "x")
        wv.progress("p", 1, 10)
        wv.frame_metrics("f.fits", {'score': 1})
        wv.preview(_synth_rgb(), "cap")
        wv.summary(a=1)
        self.assertEqual(wv._state['log'], [])
        self.assertEqual(wv._state['phase'], 0)
        self.assertIsNone(wv._preview_bytes)


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
        import io, os, tempfile
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


class TestServer(unittest.TestCase):
    def setUp(self):
        self.wv = WebView()
        self.url = self.wv.start(port=0)  # ephemeral port
        self.assertIsNotNone(self.url)

    def tearDown(self):
        self.wv.stop()

    def _get(self, path, timeout=5):
        return urllib.request.urlopen(self.url.rstrip('/') + path,
                                      timeout=timeout)

    def test_page_served(self):
        body = self._get('/').read().decode('utf-8')
        self.assertIn('OriginStack', body)
        self.assertIn('EventSource', body)

    def test_events_reflect_state(self):
        self.wv.phase(2, "Registration")
        self.wv.progress("Registering frames", 3, 10)
        self.wv.log("hello from test")
        resp = self._get('/events')
        line = resp.readline().decode('utf-8')
        while not line.startswith('data:'):
            line = resp.readline().decode('utf-8')
        snap = json.loads(line[5:].strip())
        self.assertEqual(snap['phase'], 2)
        self.assertEqual(snap['progress']['done'], 3)
        self.assertIn('hello from test', snap['log'])
        resp.close()

    def test_preview_endpoint(self):
        r404 = None
        try:
            self._get('/preview.jpg')
        except urllib.error.HTTPError as e:
            r404 = e.code
        self.assertEqual(r404, 404)
        self.wv.preview(_synth_rgb(), "test cap", min_interval=0.0)
        data = self._get('/preview.jpg').read()
        self.assertGreater(len(data), 1000)
        self.assertEqual(data[:2], b'\xff\xd8')  # JPEG magic

    def test_named_slot_and_snapshot(self):
        """A published preview registers a named slot fetchable by slug and
        listed in the SSE snapshot for the compare dropdown."""
        self.wv.preview(_synth_rgb(), "Final (post-processed)", slot='final',
                        min_interval=0.0)
        # snapshot lists the slot
        resp = self._get('/events')
        line = resp.readline().decode('utf-8')
        while not line.startswith('data:'):
            line = resp.readline().decode('utf-8')
        snap = json.loads(line[5:].strip())
        resp.close()
        slugs = [n['slug'] for n in snap['named']]
        self.assertIn('final', slugs)
        self.assertTrue(snap['named'][0]['src'])  # linear source retained
        self.assertEqual(snap['latest_slug'], 'final')
        # slot jpeg is fetchable
        data = self._get('/named.jpg?slug=final').read()
        self.assertEqual(data[:2], b'\xff\xd8')

    def test_frame_thumbnail_endpoint(self):
        self.wv.frame_preview("light_001.fits", _synth_rgb())
        resp = self._get('/events')
        line = resp.readline().decode('utf-8')
        while not line.startswith('data:'):
            line = resp.readline().decode('utf-8')
        snap = json.loads(line[5:].strip())
        resp.close()
        self.assertEqual(len(snap['frames_img']), 1)
        fid = snap['frames_img'][0]['id']
        data = self._get('/frame.jpg?id=%d' % fid).read()
        self.assertEqual(data[:2], b'\xff\xd8')

    def test_restretch_from_source(self):
        """Re-stretch re-encodes the retained linear source with new params."""
        self.wv.preview(_synth_rgb(), "Final", slot='final', min_interval=0.0)
        data = self._get('/restretch?slug=final&stretch=ghs&b=3&sp=0.2'
                         '&hp=0.9&black=1.0').read()
        self.assertEqual(data[:2], b'\xff\xd8')
        # unknown slot -> 404
        r404 = None
        try:
            self._get('/restretch?slug=nope')
        except urllib.error.HTTPError as e:
            r404 = e.code
        self.assertEqual(r404, 404)

    def test_page_has_interactive_controls(self):
        body = self._get('/').read().decode('utf-8')
        for marker in ('viewport', 'applyStretch', 'cmpBtn', 'restretch',
                       'strip'):
            self.assertIn(marker, body)


class TestFrameThumbRing(unittest.TestCase):
    def test_ring_is_bounded(self):
        from src.webview import _MAX_FRAME_THUMBS
        wv = WebView()
        wv.active = True  # publish without a server
        for i in range(_MAX_FRAME_THUMBS + 5):
            wv.frame_preview("f%d" % i, _synth_rgb())
        self.assertEqual(len(wv._frame_thumbs), _MAX_FRAME_THUMBS)


if __name__ == '__main__':
    unittest.main()
