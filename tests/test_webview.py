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
        wv.run_started()
        wv.run_finished('error', 'boom')
        self.assertEqual(wv._state['log'], [])
        self.assertEqual(wv._state['phase'], 0)
        self.assertIsNone(wv._preview_bytes)
        self.assertEqual(wv._state['run_status'], 'idle')
        self.assertIsNone(wv._state['run_error'])


class TestRunState(unittest.TestCase):
    """WebView.run_started()/run_finished(): the desktop app's run_status
    signal, separate from summary()/done which fire once per stack_target
    call (multiple times in one hierarchical/mosaic run)."""

    def setUp(self):
        self.wv = WebView()
        self.wv.active = True

    def test_run_started_sets_running_and_clears_prior_state(self):
        self.wv.log("stale log line")
        self.wv.frame_metrics("f.fits", {'score': 1})
        self.wv.summary(output='old.fits')
        self.wv.phase(2, "Registration")
        self.wv.progress("Registering", 3, 10)
        before_version = self.wv._version

        self.wv.run_started()

        self.assertEqual(self.wv._state['run_status'], 'running')
        self.assertIsNone(self.wv._state['run_error'])
        self.assertEqual(self.wv._state['log'], [])
        self.assertEqual(self.wv._state['frames'], [])
        self.assertIsNone(self.wv._state['summary'])
        self.assertFalse(self.wv._state['done'])
        self.assertEqual(self.wv._state['phase'], 0)
        self.assertEqual(self.wv._state['phases'], {})
        self.assertEqual(self.wv._state['progress'], {'label': '', 'done': 0, 'total': 0})
        self.assertGreater(self.wv._version, before_version)

    def test_run_finished_ok(self):
        self.wv.run_started()
        self.wv.run_finished('ok')
        self.assertEqual(self.wv._state['run_status'], 'ok')
        self.assertIsNone(self.wv._state['run_error'])

    def test_run_finished_error_carries_message(self):
        self.wv.run_started()
        self.wv.run_finished('error', 'directory not found')
        self.assertEqual(self.wv._state['run_status'], 'error')
        self.assertEqual(self.wv._state['run_error'], 'directory not found')

    def test_summary_done_unaffected_by_run_state(self):
        """summary()/done still mean 'a target just finished' -- run_started
        resets them (new run), but run_finished must not touch them, since a
        hierarchical run's last stack_target() call already set them."""
        self.wv.run_started()
        self.wv.summary(output='final.fits')
        self.wv.run_finished('ok')
        self.assertTrue(self.wv._state['done'])
        self.assertEqual(self.wv._state['summary'], {'output': 'final.fits'})


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

    def test_api_schema_served(self):
        body = json.loads(self._get('/api/schema').read())
        self.assertIn('Core', body)
        dests = {f['dest'] for fields in body.values() for f in fields}
        self.assertIn('directory', dests)
        self.assertIn('stack_method', dests)

    def test_api_health_served(self):
        body = json.loads(self._get('/api/health').read())
        self.assertIn('native', body)
        self.assertIn('version', body)
        self.assertIsInstance(body['native'], bool)

    def test_api_frame_count_missing_dir_param(self):
        body = json.loads(self._get('/api/frame_count').read())
        self.assertFalse(body['ok'])

    def test_api_frame_count_nonexistent_dir(self):
        body = json.loads(self._get('/api/frame_count?dir=' + 'no_such_dir_xyz').read())
        self.assertFalse(body['ok'])

    def test_api_frame_count_real_directory(self):
        """--web-view's 'how many lights are actually in this folder' check
        (user-reported desktop-app gap): counts must match discover_frames()
        exactly, since that's what Phase 1 will actually see."""
        import os
        import tempfile
        from astropy.io import fits
        with tempfile.TemporaryDirectory() as td:
            for i, kind in enumerate(['light', 'light', 'light', 'dark', 'flat']):
                hdu = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
                hdu.writeto(os.path.join(td, f'{kind}_{i:03d}.fit'), overwrite=True)
            import urllib.parse
            body = json.loads(self._get(
                '/api/frame_count?dir=' + urllib.parse.quote(td)).read())
        self.assertTrue(body['ok'])
        self.assertEqual(body['counts']['light'], 3)
        self.assertEqual(body['counts']['dark'], 1)
        self.assertEqual(body['counts']['flat'], 1)
        self.assertEqual(body['counts']['bias'], 0)

    def test_api_frame_count_hierarchical_directory(self):
        """A folder with no FITS directly in it but subfolders each holding
        a session (the --combine-sessions / hierarchical-mode layout) must
        report the pooled totals and a per-session breakdown, not a false
        'no frames found' -- the exact desktop-app gap reported by a user
        selecting this kind of folder."""
        import os
        import tempfile
        import urllib.parse
        from astropy.io import fits
        with tempfile.TemporaryDirectory() as td:
            for session, lights in [('night1', 3), ('night2', 2)]:
                sdir = os.path.join(td, session)
                os.makedirs(sdir)
                for i in range(lights):
                    hdu = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
                    hdu.writeto(os.path.join(sdir, f'light_{i:03d}.fit'), overwrite=True)
            body = json.loads(self._get(
                '/api/frame_count?dir=' + urllib.parse.quote(td)).read())
        self.assertTrue(body['ok'])
        self.assertEqual(body['counts']['light'], 5)
        self.assertIn('sessions', body)
        names = sorted(s['name'] for s in body['sessions'])
        self.assertEqual(names, ['night1', 'night2'])

    def test_api_frame_count_empty_subfolders_reports_no_frames(self):
        """Subfolders that exist but hold no FITS/RAW/etc must still report
        a plain 'no frames' result, not a hierarchical session list."""
        import os
        import tempfile
        import urllib.parse
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, 'empty_subdir'))
            body = json.loads(self._get(
                '/api/frame_count?dir=' + urllib.parse.quote(td)).read())
        self.assertTrue(body['ok'])
        self.assertEqual(sum(body['counts'].values()), 0)
        self.assertEqual(body.get('sessions'), [])

    def _post(self, path, body, headers=None, timeout=5):
        hdrs = {'Content-Type': 'application/json'}
        hdrs.update(headers or {})
        req = urllib.request.Request(
            self.url.rstrip('/') + path,
            data=json.dumps(body).encode('utf-8'),
            headers=hdrs, method='POST')
        return urllib.request.urlopen(req, timeout=timeout)

    def test_api_start_happy_path(self):
        """POST /api/start accepts a valid form and returns 202 without
        actually running the pipeline (process_directory patched out) --
        this exercises do_POST's full dispatch to RunManager.start()."""
        from unittest.mock import patch
        import src.webview_control as wc
        wc._run_manager = wc.RunManager()  # isolate from other tests
        try:
            with patch('src.cli.process_directory') as mock_pd:
                resp = self._post('/api/start',
                                  {'directory': 'nonexistent_dir_for_test',
                                   'output': 'out.fits'})
                self.assertEqual(resp.status, 202)
                result = json.loads(resp.read())
                self.assertTrue(result['ok'])
                self.assertIsNotNone(wc._run_manager.thread)
                wc._run_manager.thread.join(timeout=5)
                mock_pd.assert_called_once()
                self.assertEqual(mock_pd.call_args[0][0], 'nonexistent_dir_for_test')
        finally:
            wc._run_manager = wc.RunManager()  # restore a clean singleton

    def test_api_start_rejects_non_json_content_type(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post('/api/start', {'directory': 'x'},
                       headers={'Content-Type': 'text/plain'})
        self.assertEqual(ctx.exception.code, 415)

    def test_api_start_rejects_mismatched_origin(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post('/api/start', {'directory': 'x'},
                       headers={'Origin': 'http://evil.example'})
        self.assertEqual(ctx.exception.code, 403)

    def test_api_start_rejects_malformed_body(self):
        req = urllib.request.Request(
            self.url.rstrip('/') + '/api/start', data=b'{not json',
            headers={'Content-Type': 'application/json'}, method='POST')
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 400)

    def test_api_start_rejects_oversized_body(self):
        from src.webview import _MAX_POST_BODY
        req = urllib.request.Request(
            self.url.rstrip('/') + '/api/start',
            data=b'{"directory": "' + b'x' * _MAX_POST_BODY + b'"}',
            headers={'Content-Type': 'application/json'}, method='POST')
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 413)

    def test_api_start_concurrent_returns_409(self):
        import src.webview_control as wc
        rm = wc.RunManager()
        rm.status = 'running'  # simulate an in-progress run
        wc._run_manager = rm
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self._post('/api/start', {'directory': 'x'})
            self.assertEqual(ctx.exception.code, 409)
        finally:
            wc._run_manager = wc.RunManager()


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
