"""Live stacking web view (``--web-view``).

A pure-stdlib local dashboard: a ``ThreadingHTTPServer`` daemon thread in the
main process serves one self-contained HTML page with live phase progress,
the log stream, a per-frame quality ticker, and preview images published at
processing milestones. Server-Sent Events push state snapshots; images are
fetched by version.

The viewer is interactive: zoom/pan the preview, re-stretch it live from the
retained float source, compare any two milestone previews with a wipe slider,
and page through per-frame thumbnails published during Phase 1.

Inactive by default: every publish method is a no-op until ``start()`` is
called, so the pipeline instrumentation costs nothing on normal runs.
Mirrors the ``get_gpu()`` module-level singleton pattern.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

_MAX_LOG_LINES = 500
_MAX_FRAME_ROWS = 60
_MAX_FRAME_THUMBS = 24     # per-frame preview ring
_MAX_NAMED = 16            # retained milestone previews (with float source)
_SRC_MAX_DIM = 1000        # downsized float source kept for re-stretch
_MAX_POST_BODY = 64 * 1024  # cap on POST /api/start body size (bytes)


def _slugify(text: str) -> str:
    s = re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')
    return s or 'preview'


def _downsize_f16(rgb, max_dim: int):
    """Decimate an HWC float image to <= max_dim and store as float16 to bound
    the memory kept for on-demand re-stretch. Aliasing is acceptable for a
    preview source."""
    import numpy as np
    a = np.asarray(rgb)
    if a.ndim == 2:
        a = a[:, :, None]
    h, w = a.shape[:2]
    step = max(1, int(max(h, w) // max_dim))
    if step > 1:
        a = a[::step, ::step]
    return a.astype(np.float16)


class WebView:
    def __init__(self) -> None:
        self.active = False
        self._lock = threading.Lock()
        self._server = None
        self._version = 0          # bumped on every state change (SSE trigger)
        # latest preview (back-compat single slot)
        self._preview_version = 0
        self._preview_bytes: Optional[bytes] = None
        self._preview_caption = ""
        self._last_preview_time = 0.0
        # named milestone previews: slug -> {caption, ver, jpeg, src(f16), kw}
        self._named: "OrderedDict[str, dict]" = OrderedDict()
        self._named_version = 0
        self._latest_slug = ""
        # per-frame thumbnails (ring)
        self._frame_thumbs: "OrderedDict[int, dict]" = OrderedDict()
        self._frame_thumb_seq = 0
        self._frame_thumbs_version = 0
        self._state: Dict[str, Any] = {
            'run': {},                 # target/output/frame counts
            'phase': 0,
            'phase_title': '',
            'phases': {},              # {n: {'title', 'started', 'elapsed'}}
            'progress': {'label': '', 'done': 0, 'total': 0},
            'log': [],
            'frames': [],              # recent per-frame metric rows
            'summary': None,
            'done': False,
            'run_status': 'idle',      # 'idle' | 'running' | 'ok' | 'error'
            'run_error': None,
        }

    # ── publish API (no-ops while inactive) ──────────────────────────────

    def _bump(self) -> None:
        self._version += 1

    def log(self, text: str) -> None:
        if not self.active:
            return
        with self._lock:
            log = self._state['log']
            for line in str(text).split('\n'):
                if line.strip():
                    log.append(line)
            if len(log) > _MAX_LOG_LINES:
                del log[:len(log) - _MAX_LOG_LINES]
            self._bump()

    def set_run_info(self, **info: Any) -> None:
        if not self.active:
            return
        with self._lock:
            self._state['run'].update(info)
            self._bump()

    def phase(self, num: int, title: str) -> None:
        if not self.active:
            return
        now = time.time()
        with self._lock:
            prev = self._state['phase']
            if prev and prev in self._state['phases']:
                p = self._state['phases'][prev]
                p['elapsed'] = now - p['started']
            self._state['phase'] = num
            self._state['phase_title'] = title
            self._state['phases'][num] = {'title': title, 'started': now,
                                          'elapsed': None}
            self._state['progress'] = {'label': '', 'done': 0, 'total': 0}
            self._bump()

    def progress(self, label: str, done: int, total: int) -> None:
        if not self.active:
            return
        with self._lock:
            self._state['progress'] = {'label': label, 'done': int(done),
                                       'total': int(total)}
            self._bump()

    def frame_metrics(self, name: str, metrics: Optional[dict],
                      accepted: bool = True) -> None:
        if not self.active:
            return
        m = metrics or {}
        row = {'name': name,
               'score': round(float(m.get('score', 0.0)), 1),
               'snr': round(float(m.get('snr', 0.0)), 2),
               'stars': int(m.get('star_count', 0)),
               'fwhm': round(float(m.get('fwhm', 0.0)), 2),
               'ok': bool(accepted)}
        with self._lock:
            rows = self._state['frames']
            rows.append(row)
            if len(rows) > _MAX_FRAME_ROWS:
                del rows[:len(rows) - _MAX_FRAME_ROWS]
            self._bump()

    @staticmethod
    def _stretch_kw(args) -> dict:
        """Extract stretch parameters from an args namespace (or defaults)."""
        if args is None:
            return dict(stretch='ghs', ghs_b=8.0, ghs_sp=0.15, ghs_hp=0.95,
                        black_sigma=0.0)
        return dict(
            stretch=getattr(args, 'stretch', 'ghs'),
            ghs_b=float(getattr(args, 'ghs_b', 8.0)),
            ghs_sp=float(getattr(args, 'ghs_sp', 0.15)),
            ghs_hp=float(getattr(args, 'ghs_hp', 0.95)),
            black_sigma=float(getattr(args, 'preview_black_sigma', 0.0) or 0.0))

    def preview(self, rgb, caption: str, args=None, slot: Optional[str] = None,
                min_interval: float = 2.0) -> None:
        """Publish a stretched preview of an HWC float32 image. Encoding is
        throttled so frequent milestones don't spend time on JPEG encodes.
        The float source is retained (downsized) so the viewer can re-stretch
        it on demand, and the preview is registered as a named slot so it can
        be picked for before/after compare."""
        if not self.active:
            return
        now = time.time()
        if now - self._last_preview_time < min_interval:
            return
        kw = self._stretch_kw(args)
        try:
            from src.io_fits import preview_jpeg_bytes
            data = preview_jpeg_bytes(rgb, max_dim=1024, **kw)
        except Exception:
            return
        if not data:
            return
        try:
            src = _downsize_f16(rgb, _SRC_MAX_DIM)
        except Exception:
            src = None
        slug = _slugify(slot or caption)
        with self._lock:
            self._preview_bytes = data
            self._preview_caption = caption
            self._preview_version += 1
            self._last_preview_time = now
            # register / update the named slot
            self._named_version += 1
            self._named[slug] = {'caption': caption, 'ver': self._named_version,
                                 'jpeg': data, 'src': src, 'kw': kw}
            self._named.move_to_end(slug)
            self._latest_slug = slug
            # evict oldest sources beyond the cap (keep 'final' + newest)
            while len(self._named) > _MAX_NAMED:
                for k in list(self._named.keys()):
                    if k != 'final' and k != self._latest_slug:
                        del self._named[k]
                        break
                else:
                    break
            self._bump()

    def frame_preview(self, name: str, rgb, args=None,
                      max_dim: int = 512) -> None:
        """Publish a small thumbnail of a single processed light (Phase 1).
        Stored in a bounded ring; the viewer can page through them."""
        if not self.active:
            return
        kw = self._stretch_kw(args)
        try:
            from src.io_fits import preview_jpeg_bytes
            data = preview_jpeg_bytes(rgb, max_dim=max_dim, **kw)
        except Exception:
            return
        if not data:
            return
        with self._lock:
            self._frame_thumb_seq += 1
            fid = self._frame_thumb_seq
            self._frame_thumbs[fid] = {'id': fid, 'name': name, 'jpeg': data}
            while len(self._frame_thumbs) > _MAX_FRAME_THUMBS:
                self._frame_thumbs.popitem(last=False)
            self._frame_thumbs_version += 1
            self._bump()

    def summary(self, **fields: Any) -> None:
        if not self.active:
            return
        with self._lock:
            self._state['summary'] = fields
            self._state['done'] = True
            self._bump()

    def run_started(self) -> None:
        """Mark the start of a GUI-triggered pipeline run and clear state left
        over from a previous run. ``summary``/``done`` track per-*target*
        completion (set once per ``stack_target`` call, so they can fire
        several times in one hierarchical/mosaic run) -- ``run_status`` tracks
        the whole ``process_directory()`` call instead, which is what a
        long-lived desktop app session actually needs to know."""
        if not self.active:
            return
        with self._lock:
            self._state['log'] = []
            self._state['frames'] = []
            self._state['summary'] = None
            self._state['done'] = False
            self._state['phase'] = 0
            self._state['phase_title'] = ''
            self._state['phases'] = {}
            self._state['progress'] = {'label': '', 'done': 0, 'total': 0}
            self._state['run_status'] = 'running'
            self._state['run_error'] = None
            self._bump()

    def run_finished(self, status: str, error: Optional[str] = None) -> None:
        if not self.active:
            return
        with self._lock:
            self._state['run_status'] = status
            self._state['run_error'] = error
            self._bump()

    # ── re-stretch (on-demand, from retained float source) ────────────────

    def restretch(self, slug: str, params: dict) -> Optional[bytes]:
        with self._lock:
            slot = self._named.get(slug)
            src = slot['src'] if slot else None
        if src is None:
            return None
        try:
            import numpy as np
            from src.io_fits import preview_jpeg_bytes
            f = np.asarray(src, dtype=np.float32)
            return preview_jpeg_bytes(
                f, max_dim=_SRC_MAX_DIM,
                stretch=params.get('stretch', 'ghs'),
                ghs_b=float(params.get('b', 8.0)),
                ghs_sp=float(params.get('sp', 0.15)),
                ghs_hp=float(params.get('hp', 0.95)),
                black_sigma=float(params.get('black', 0.0)))
        except Exception:
            return None

    # ── snapshot for SSE ──────────────────────────────────────────────────

    def _snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = json.loads(json.dumps(self._state))
            snap['version'] = self._version
            snap['preview_version'] = self._preview_version
            snap['preview_caption'] = self._preview_caption
            snap['latest_slug'] = self._latest_slug
            snap['named_version'] = self._named_version
            snap['named'] = [{'slug': k, 'caption': v['caption'],
                              'ver': v['ver'], 'src': v['src'] is not None}
                             for k, v in self._named.items()]
            snap['frames_img_version'] = self._frame_thumbs_version
            snap['frames_img'] = [{'id': v['id'], 'name': v['name']}
                                  for v in self._frame_thumbs.values()]
            return snap

    def _preview(self):
        with self._lock:
            return self._preview_bytes

    def _named_jpeg(self, slug: str):
        with self._lock:
            slot = self._named.get(slug)
            return slot['jpeg'] if slot else None

    def _frame_jpeg(self, fid: int):
        with self._lock:
            slot = self._frame_thumbs.get(fid)
            return slot['jpeg'] if slot else None

    # ── server ────────────────────────────────────────────────────────────

    def start(self, port: int = 8765) -> Optional[str]:
        """Start the dashboard server; returns the URL (None on failure)."""
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import parse_qs, urlparse

        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence per-request stderr noise
                pass

            def _send(self, code, ctype, body, extra=None):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                q = parse_qs(parsed.query)
                if path == '/':
                    self._send(200, 'text/html; charset=utf-8',
                               _PAGE.encode('utf-8'))
                elif path == '/preview.jpg':
                    data = view._preview()
                    if data is None:
                        self._send(404, 'text/plain', b'no preview yet')
                    else:
                        self._send(200, 'image/jpeg', data)
                elif path == '/named.jpg':
                    data = view._named_jpeg((q.get('slug') or [''])[0])
                    if data is None:
                        self._send(404, 'text/plain', b'no such preview')
                    else:
                        self._send(200, 'image/jpeg', data)
                elif path == '/frame.jpg':
                    try:
                        fid = int((q.get('id') or ['0'])[0])
                    except ValueError:
                        fid = 0
                    data = view._frame_jpeg(fid)
                    if data is None:
                        self._send(404, 'text/plain', b'no such frame')
                    else:
                        self._send(200, 'image/jpeg', data)
                elif path == '/restretch':
                    def _f(key, dflt):
                        try:
                            return float((q.get(key) or [dflt])[0])
                        except (TypeError, ValueError):
                            return float(dflt)
                    params = {'stretch': (q.get('stretch') or ['ghs'])[0],
                              'b': _f('b', 8.0), 'sp': _f('sp', 0.15),
                              'hp': _f('hp', 0.95), 'black': _f('black', 0.0)}
                    data = view.restretch((q.get('slug') or [''])[0], params)
                    if data is None:
                        self._send(404, 'text/plain', b'no source for slot')
                    else:
                        self._send(200, 'image/jpeg', data)
                elif path == '/api/schema':
                    from src.webview_control import get_form_schema
                    try:
                        body = json.dumps(get_form_schema()).encode('utf-8')
                        self._send(200, 'application/json', body)
                    except Exception as e:
                        self._send(500, 'application/json',
                                   json.dumps({'error': str(e)}).encode('utf-8'))
                elif path == '/events':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    last = -1
                    try:
                        while True:
                            if view._version != last:
                                last = view._version
                                payload = json.dumps(view._snapshot())
                                self.wfile.write(
                                    f"data: {payload}\n\n".encode('utf-8'))
                                self.wfile.flush()
                            time.sleep(0.25)
                    except (ConnectionAbortedError, ConnectionResetError,
                            BrokenPipeError, OSError):
                        return
                else:
                    self._send(404, 'text/plain', b'not found')

            def do_POST(self):
                if self.path != '/api/start':
                    self._send(404, 'text/plain', b'not found')
                    return
                # CSRF hardening: a browser can only send a strict
                # application/json body via fetch(), which is a non-simple
                # request -- it triggers a CORS preflight, and since this
                # server never answers OPTIONS with CORS headers, the browser
                # refuses to send the real request cross-origin. A classic
                # <form> POST (the no-preflight CSRF vector) can only set
                # Content-Type to x-www-form-urlencoded/multipart/text-plain,
                # never application/json, so it's rejected here regardless.
                ctype = (self.headers.get('Content-Type') or '').split(';')[0].strip().lower()
                if ctype != 'application/json':
                    self._send(415, 'application/json',
                               json.dumps({'ok': False,
                                          'error': 'Content-Type must be application/json'})
                               .encode('utf-8'))
                    return
                # Defense in depth: if a browser sent an Origin header, it
                # must match the Host this server is actually bound to.
                origin = self.headers.get('Origin')
                if origin:
                    from urllib.parse import urlparse as _urlparse
                    if _urlparse(origin).netloc != (self.headers.get('Host') or ''):
                        self._send(403, 'application/json',
                                   json.dumps({'ok': False, 'error': 'origin not allowed'})
                                   .encode('utf-8'))
                        return
                try:
                    length = int(self.headers.get('Content-Length') or 0)
                    if length > _MAX_POST_BODY:
                        self._send(413, 'application/json',
                                   json.dumps({'ok': False, 'error': 'request body too large'})
                                   .encode('utf-8'))
                        return
                    raw = self.rfile.read(length) if length else b'{}'
                    form = json.loads(raw.decode('utf-8')) if raw else {}
                except Exception as e:
                    self._send(400, 'application/json',
                               json.dumps({'ok': False, 'error': f'bad request: {e}'})
                               .encode('utf-8'))
                    return
                from src.webview_control import get_run_manager
                result = get_run_manager().start(form)
                code = 202 if result.get('ok') else 409
                self._send(code, 'application/json', json.dumps(result).encode('utf-8'))

        try:
            self._server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        except OSError as exc:
            from src.utils import safe_print
            safe_print(f"  WARNING: web view failed to bind port {port}: {exc}")
            return None
        self.active = True
        t = threading.Thread(target=self._server.serve_forever,
                             name='webview-http', daemon=True)
        t.start()
        actual_port = self._server.server_address[1]
        return f"http://127.0.0.1:{actual_port}/"

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
        self.active = False


_webview: Optional[WebView] = None


def get_webview() -> WebView:
    global _webview
    if _webview is None:
        _webview = WebView()
    return _webview


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OriginStack — live</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0d1117; color: #d5dae2;
         font: 14px/1.45 system-ui, "Segoe UI", sans-serif; }
  header { padding: 14px 22px; background: #161b22;
           border-bottom: 1px solid #2a313c; display: flex;
           align-items: baseline; gap: 14px; }
  header h1 { font-size: 17px; margin: 0; color: #e8edf4; }
  header .run { color: #8b949e; font-size: 13px; }
  main { display: grid; grid-template-columns: minmax(360px, 1fr) minmax(460px, 1.3fr);
         gap: 16px; padding: 16px 22px; max-width: 1600px; }
  section { background: #161b22; border: 1px solid #2a313c;
            border-radius: 8px; padding: 14px 16px; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
       color: #8b949e; margin: 0 0 10px; }
  .phases { display: flex; gap: 8px; }
  .ph { flex: 1; padding: 8px 10px; border-radius: 6px; background: #0d1117;
        border: 1px solid #2a313c; font-size: 12px; color: #8b949e; }
  .ph.active { border-color: #58a6ff; color: #e8edf4; }
  .ph.done { border-color: #3fb950; color: #b9c4d0; }
  .ph .t { display: block; font-size: 11px; margin-top: 2px; color: #6e7883; }
  .bar { height: 10px; background: #0d1117; border-radius: 5px;
         overflow: hidden; margin-top: 10px; border: 1px solid #2a313c; }
  .bar div { height: 100%; background: linear-gradient(90deg, #1f6feb, #58a6ff);
             width: 0%; transition: width .3s; }
  .plabel { margin-top: 6px; font-size: 12px; color: #8b949e; }
  #log { height: 260px; overflow-y: auto; background: #0d1117;
         border-radius: 6px; padding: 10px 12px; font: 12px/1.5 Consolas,
         monospace; white-space: pre-wrap; color: #9aa4b2; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: right; padding: 3px 8px; border-bottom: 1px solid #21262d; }
  th:first-child, td:first-child { text-align: left; }
  td.bad { color: #f85149; }
  /* viewer */
  .vtools { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
            margin-bottom: 10px; }
  .vtools select, .vtools button { background: #0d1117; color: #d5dae2;
      border: 1px solid #2a313c; border-radius: 6px; padding: 5px 9px;
      font-size: 12px; cursor: pointer; }
  .vtools button:hover, .vtools select:hover { border-color: #58a6ff; }
  .vtools button.on { border-color: #58a6ff; color: #e8edf4; }
  .vtools .spacer { flex: 1; }
  .vtools .z { color: #6e7883; font-size: 12px; min-width: 46px;
               text-align: right; }
  #viewport { position: relative; width: 100%; height: 460px; overflow: hidden;
              background: #000; border-radius: 6px; border: 1px solid #2a313c;
              cursor: grab; touch-action: none; }
  #viewport.drag { cursor: grabbing; }
  #viewport img { position: absolute; top: 0; left: 0; transform-origin: 0 0;
                  image-rendering: auto; user-select: none;
                  -webkit-user-drag: none; max-width: none; }
  #cmpImg { display: none; }
  #wipe { position: absolute; top: 0; bottom: 0; width: 2px; background: #58a6ff;
          display: none; cursor: ew-resize; z-index: 5; }
  #wipe::after { content: '\\21d4'; position: absolute; top: 50%; left: 50%;
      transform: translate(-50%,-50%); background: #58a6ff; color: #0d1117;
      border-radius: 50%; width: 22px; height: 22px; line-height: 22px;
      text-align: center; font-size: 12px; }
  .cap { margin-top: 6px; font-size: 12px; color: #8b949e; }
  /* stretch panel */
  .sgrid { display: grid; grid-template-columns: auto 1fr auto; gap: 6px 10px;
           align-items: center; font-size: 12px; }
  .sgrid label { color: #8b949e; }
  .sgrid input[type=range] { width: 100%; }
  .sgrid .val { color: #d5dae2; min-width: 42px; text-align: right;
                font-variant-numeric: tabular-nums; }
  .srow { display: flex; gap: 8px; margin-top: 10px; }
  .srow button { flex: 1; background: #1f6feb; color: #fff; border: 0;
      border-radius: 6px; padding: 7px; font-size: 12px; cursor: pointer; }
  .srow button.ghost { background: #0d1117; color: #8b949e;
      border: 1px solid #2a313c; }
  .snote { color: #6e7883; font-size: 11px; margin-top: 8px; }
  /* setup panel: quick bools + advanced accordion */
  .quick-bool { display: flex; align-items: center; gap: 6px; font-size: 12px;
                color: #d5dae2; }
  .adv-group-h { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
                 color: #6e7883; margin: 10px 0 4px; }
  .adv-row { display: grid; grid-template-columns: 180px 1fr auto; gap: 8px;
             align-items: center; margin-bottom: 6px; font-size: 12px; }
  .adv-label { color: #8b949e; }
  .adv-input { background: #0d1117; color: #d5dae2; border: 1px solid #2a313c;
               border-radius: 4px; padding: 3px 6px; width: 100%; }
  .adv-browse { background: #0d1117; color: #d5dae2; border: 1px solid #2a313c;
                border-radius: 4px; padding: 3px 8px; font-size: 11px; cursor: pointer; }
  .adv-browse:hover { border-color: #58a6ff; }
  .adv-summary { cursor: pointer; color: #8b949e; font-size: 12px; }
  /* frame strip */
  #strip { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px; }
  #strip figure { margin: 0; flex: 0 0 auto; width: 96px; cursor: pointer;
                  text-align: center; }
  #strip img { width: 96px; height: 72px; object-fit: cover; border-radius: 4px;
               border: 1px solid #2a313c; background: #000; }
  #strip figure:hover img { border-color: #58a6ff; }
  #strip figcaption { font-size: 10px; color: #6e7883; margin-top: 3px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #summary { display: none; border-color: #3fb950; }
  #summary dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 16px;
                margin: 0; font-size: 13px; }
  #summary dt { color: #8b949e; } #summary dd { margin: 0; color: #e8edf4; }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header><h1>OriginStack</h1><span class="run" id="runinfo">connecting…</span></header>
<main>
  <div>
    <section id="setupSection">
      <h2>Setup</h2>
      <div class="sgrid" id="quickFields">
        <label>Directory</label>
        <input type="text" id="f_directory" placeholder="C:\\path\\to\\lights">
        <button id="browseDir" style="display:none">Browse…</button>
        <label>Output</label>
        <input type="text" id="f_output" placeholder="(default: &lt;directory&gt;_stacked.fits)">
        <button id="browseOut" style="display:none">Browse…</button>
        <label>Preset</label>
        <select id="f_preset"><option value="">(none)</option></select>
        <span></span>
        <label>Stack method</label>
        <select id="f_stack_method"></select>
        <span></span>
        <label>Denoiser</label>
        <select id="f_denoiser"></select>
        <span></span>
        <label>Deconvolution</label>
        <select id="f_deconvolve_mode"></select>
        <span></span>
        <label>Drizzle scale</label>
        <select id="f_drizzle_scale">
          <option value="1.0">Off</option>
          <option value="1.5">1.5x</option>
          <option value="2.0">2x</option>
          <option value="3.0">3x</option>
        </select>
        <span></span>
      </div>
      <div class="srow" style="flex-wrap:wrap;margin-top:12px" id="quickBools"></div>
      <details style="margin-top:12px" id="advancedDetails">
        <summary class="adv-summary">Advanced (everything else)</summary>
        <div id="advancedFields" style="margin-top:10px"></div>
      </details>
      <div class="srow" style="margin-top:14px">
        <button id="startBtn">Start</button>
      </div>
      <div class="snote" id="startStatus">Idle.</div>
    </section>
    <section style="margin-top:16px">
      <h2>Pipeline</h2>
      <div class="phases" id="phases">
        <div class="ph" data-n="1">1 · Quality<span class="t"></span></div>
        <div class="ph" data-n="2">2 · Registration<span class="t"></span></div>
        <div class="ph" data-n="3">3 · Stacking<span class="t"></span></div>
        <div class="ph" data-n="4">4 · Post-process<span class="t"></span></div>
      </div>
      <div class="bar"><div id="barfill"></div></div>
      <div class="plabel" id="plabel">&nbsp;</div>
    </section>
    <section style="margin-top:16px">
      <h2>Recent frames</h2>
      <table><thead><tr><th>Frame</th><th>Score</th><th>SNR</th><th>Stars</th>
        <th>FWHM</th></tr></thead><tbody id="frames"></tbody></table>
    </section>
    <section style="margin-top:16px">
      <h2>Stretch</h2>
      <div class="sgrid">
        <label>Black σ</label><input type="range" id="s_black" min="-1" max="4"
          step="0.05" value="0"><span class="val" id="v_black">0.00</span>
        <label>GHS b</label><input type="range" id="s_b" min="0" max="20"
          step="0.1" value="8"><span class="val" id="v_b">8.0</span>
        <label>GHS sp</label><input type="range" id="s_sp" min="0" max="1"
          step="0.005" value="0.15"><span class="val" id="v_sp">0.15</span>
        <label>GHS hp</label><input type="range" id="s_hp" min="0" max="1"
          step="0.005" value="0.95"><span class="val" id="v_hp">0.95</span>
      </div>
      <div class="srow">
        <button id="applyStretch">Apply to view</button>
        <button id="resetStretch" class="ghost">Reset</button>
      </div>
      <div class="snote" id="snote">Adjusts the currently viewed milestone
        preview, re-stretched from its retained linear source.</div>
    </section>
    <section style="margin-top:16px">
      <h2>Log</h2>
      <div id="log"></div>
    </section>
  </div>
  <div>
    <section>
      <h2>Preview</h2>
      <div class="vtools">
        <select id="viewSel" title="Which image to show"></select>
        <button id="cmpBtn" title="Compare two milestones">Compare</button>
        <select id="cmpSel" style="display:none" title="Compare against"></select>
        <div class="spacer"></div>
        <button id="fitBtn">Fit</button>
        <button id="oneBtn">1:1</button>
        <span class="z" id="zlabel">100%</span>
      </div>
      <div id="viewport">
        <img id="mainImg" alt="waiting for first preview…">
        <img id="cmpImg" alt="">
        <div id="wipe"></div>
      </div>
      <div class="cap" id="previewCap">Waiting for the first stack…</div>
    </section>
    <section style="margin-top:16px">
      <h2>Frames</h2>
      <div id="strip"></div>
    </section>
    <section style="margin-top:16px" id="summary">
      <h2>Complete</h2>
      <dl id="summarydl"></dl>
    </section>
  </div>
</main>
<script>
"use strict";
// ── viewer transform state ───────────────────────────────────────────────
const vp = document.getElementById('viewport');
const mainImg = document.getElementById('mainImg');
const cmpImg = document.getElementById('cmpImg');
const wipe = document.getElementById('wipe');
let scale = 1, tx = 0, ty = 0, natW = 0, natH = 0, fitScale = 1;
let follow = true;          // main view follows the latest live preview
let curSlug = '';           // slug currently shown in main view
let manualStretch = false;  // a re-stretch overrides live updates
let cmpOn = false, wipeFrac = 0.5;

function applyTransform() {
  const t = `translate(${tx}px, ${ty}px) scale(${scale})`;
  mainImg.style.transform = t;
  cmpImg.style.transform = t;
  document.getElementById('zlabel').textContent =
    Math.round(scale / fitScale * 100) + '%';
  updateWipe();
}
function computeFit() {
  if (!natW || !natH) return;
  fitScale = Math.min(vp.clientWidth / natW, vp.clientHeight / natH);
  scale = fitScale;
  tx = (vp.clientWidth - natW * scale) / 2;
  ty = (vp.clientHeight - natH * scale) / 2;
  applyTransform();
}
function setOneToOne() {
  const cx = vp.clientWidth / 2, cy = vp.clientHeight / 2;
  const ix = (cx - tx) / scale, iy = (cy - ty) / scale;
  scale = 1;
  tx = cx - ix * scale; ty = cy - iy * scale;
  applyTransform();
}
mainImg.onload = () => {
  const firstLoad = (natW === 0);
  natW = mainImg.naturalWidth; natH = mainImg.naturalHeight;
  mainImg.style.width = natW + 'px'; mainImg.style.height = natH + 'px';
  cmpImg.style.width = natW + 'px'; cmpImg.style.height = natH + 'px';
  if (firstLoad || scale === 0) computeFit();
  else applyTransform();
};
// wheel zoom toward cursor
vp.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = vp.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const ix = (mx - tx) / scale, iy = (my - ty) / scale;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  scale = Math.max(fitScale * 0.5, Math.min(scale * factor, 40));
  tx = mx - ix * scale; ty = my - iy * scale;
  applyTransform();
}, { passive: false });
// drag to pan (or move wipe when grabbing near it)
let dragging = false, wipeDrag = false, lx = 0, ly = 0;
vp.addEventListener('pointerdown', (e) => {
  vp.setPointerCapture(e.pointerId);
  if (cmpOn) {
    const r = vp.getBoundingClientRect();
    if (Math.abs((e.clientX - r.left) - wipeFrac * vp.clientWidth) < 14) {
      wipeDrag = true; return;
    }
  }
  dragging = true; vp.classList.add('drag'); lx = e.clientX; ly = e.clientY;
});
vp.addEventListener('pointermove', (e) => {
  if (wipeDrag) {
    const r = vp.getBoundingClientRect();
    wipeFrac = Math.max(0, Math.min(1, (e.clientX - r.left) / vp.clientWidth));
    updateWipe(); return;
  }
  if (!dragging) return;
  tx += e.clientX - lx; ty += e.clientY - ly; lx = e.clientX; ly = e.clientY;
  applyTransform();
});
function endDrag() { dragging = false; wipeDrag = false; vp.classList.remove('drag'); }
vp.addEventListener('pointerup', endDrag);
vp.addEventListener('pointercancel', endDrag);
function updateWipe() {
  if (!cmpOn) { wipe.style.display = 'none'; cmpImg.style.display = 'none'; return; }
  const px = wipeFrac * vp.clientWidth;
  wipe.style.display = 'block'; wipe.style.left = px + 'px';
  cmpImg.style.display = 'block';
  // cmpImg (B) shows on the right of the wipe; mainImg (A) fully drawn beneath
  cmpImg.style.clipPath = `inset(0 0 0 ${px}px)`;
}
document.getElementById('fitBtn').onclick = computeFit;
document.getElementById('oneBtn').onclick = setOneToOne;
window.addEventListener('resize', () => { if (natW) applyTransform(); });

// ── source selection ─────────────────────────────────────────────────────
const viewSel = document.getElementById('viewSel');
const cmpSel = document.getElementById('cmpSel');
let namedVer = -1, framesVer = -1;
let lastPreviewVer = -1, latestSlug = '';

function loadMain(url, slug) {
  curSlug = slug || '';
  const keep = (natW !== 0);
  if (!keep) scale = 0;   // force fit on first image
  mainImg.src = url;
}
function loadCmp(slug) {
  cmpImg.src = '/named.jpg?slug=' + encodeURIComponent(slug);
}
viewSel.onchange = () => {
  manualStretch = false;
  const v = viewSel.value;
  if (v === '__live__') { follow = true; if (latestSlug) loadMain('/preview.jpg?v=' + lastPreviewVer, latestSlug); }
  else if (v.startsWith('f:')) { follow = false; loadMain('/frame.jpg?id=' + v.slice(2), ''); }
  else { follow = false; loadMain('/named.jpg?slug=' + encodeURIComponent(v), v); }
  refreshStretchNote();
};
document.getElementById('cmpBtn').onclick = () => {
  cmpOn = !cmpOn;
  document.getElementById('cmpBtn').classList.toggle('on', cmpOn);
  cmpSel.style.display = cmpOn ? '' : 'none';
  if (cmpOn && cmpSel.value) loadCmp(cmpSel.value);
  updateWipe();
};
cmpSel.onchange = () => { if (cmpOn) loadCmp(cmpSel.value); };

function rebuildViewOptions(named, frames) {
  const cur = viewSel.value || '__live__';
  let html = '<option value="__live__">Live (latest)</option>';
  named.forEach(n => { html += `<option value="${n.slug}">${esc(n.caption)}</option>`; });
  frames.slice().reverse().forEach(f => {
    html += `<option value="f:${f.id}">frame · ${esc(f.name)}</option>`; });
  viewSel.innerHTML = html;
  viewSel.value = [...viewSel.options].some(o => o.value === cur) ? cur : '__live__';
  // compare dropdown: named slots only
  const curB = cmpSel.value;
  cmpSel.innerHTML = named.map(n =>
    `<option value="${n.slug}">${esc(n.caption)}</option>`).join('');
  if ([...cmpSel.options].some(o => o.value === curB)) cmpSel.value = curB;
  else if (named.length) cmpSel.value = named[0].slug;
}

// ── stretch controls ─────────────────────────────────────────────────────
const sl = { black: g('s_black'), b: g('s_b'), sp: g('s_sp'), hp: g('s_hp') };
function g(id){ return document.getElementById(id); }
function fmt(id, x, d){ g(id).textContent = (+x).toFixed(d); }
function syncLabels() {
  fmt('v_black', sl.black.value, 2); fmt('v_b', sl.b.value, 1);
  fmt('v_sp', sl.sp.value, 3); fmt('v_hp', sl.hp.value, 3);
}
Object.values(sl).forEach(s => s.addEventListener('input', syncLabels));
syncLabels();
function stretchSlug() {
  if (curSlug) return curSlug;
  if (follow && latestSlug) return latestSlug;
  return '';
}
function refreshStretchNote() {
  const slug = stretchSlug();
  const ok = slug && namedHas(slug);
  g('applyStretch').disabled = !ok;
  g('snote').textContent = ok
    ? 'Re-stretches “' + slug + '” from its retained linear source.'
    : 'Select a milestone preview to re-stretch (frame thumbnails have no linear source).';
}
let curNamed = [];
function namedHas(slug){ return curNamed.some(n => n.slug === slug && n.src); }
g('applyStretch').onclick = () => {
  const slug = stretchSlug();
  if (!slug || !namedHas(slug)) return;
  manualStretch = true; follow = false;
  const u = `/restretch?slug=${encodeURIComponent(slug)}&stretch=ghs`
    + `&black=${sl.black.value}&b=${sl.b.value}&sp=${sl.sp.value}&hp=${sl.hp.value}`
    + `&_=${Date.now()}`;
  loadMain(u, slug);
};
g('resetStretch').onclick = () => {
  sl.black.value = 0; sl.b.value = 8; sl.sp.value = 0.15; sl.hp.value = 0.95;
  syncLabels();
  manualStretch = false; follow = true; viewSel.value = '__live__';
  if (latestSlug) loadMain('/preview.jpg?v=' + lastPreviewVer, latestSlug);
};

// ── frame strip ──────────────────────────────────────────────────────────
function rebuildStrip(frames) {
  const el = document.getElementById('strip');
  el.innerHTML = frames.slice().reverse().map(f =>
    `<figure data-id="${f.id}"><img src="/frame.jpg?id=${f.id}" loading="lazy">`
    + `<figcaption>${esc(f.name)}</figcaption></figure>`).join('');
  el.querySelectorAll('figure').forEach(fig => fig.onclick = () => {
    follow = false; manualStretch = false;
    viewSel.value = 'f:' + fig.dataset.id;
    loadMain('/frame.jpg?id=' + fig.dataset.id, '');
    refreshStretchNote();
  });
}

// ── SSE ──────────────────────────────────────────────────────────────────
function esc(s){ return String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
const es = new EventSource('/events');
es.onmessage = (ev) => {
  const s = JSON.parse(ev.data);
  const r = s.run || {};
  document.getElementById('runinfo').textContent =
    (r.target ? r.target + ' — ' : '') + (r.output || '') +
    (r.n_frames ? ' · ' + r.n_frames + ' frames' : '');
  document.querySelectorAll('.ph').forEach(el => {
    const n = +el.dataset.n, info = (s.phases || {})[n];
    el.classList.toggle('active', s.phase === n && !s.done);
    el.classList.toggle('done', !!(info && info.elapsed != null) ||
                        (s.done && !!info));
    el.querySelector('.t').textContent = info ?
      (info.elapsed != null ? info.elapsed.toFixed(0) + 's' :
       ((Date.now()/1000 - info.started).toFixed(0) + 's…')) : '';
  });
  const p = s.progress || {};
  const pct = p.total ? (100 * p.done / p.total) : 0;
  document.getElementById('barfill').style.width = pct + '%';
  document.getElementById('plabel').textContent = p.total ?
    `${p.label}: ${p.done} / ${p.total}` : (s.phase_title || '\\u00a0');
  const tb = document.getElementById('frames');
  tb.innerHTML = (s.frames || []).slice(-12).reverse().map(f =>
    `<tr><td${f.ok ? '' : ' class="bad"'}>${esc(f.name)}</td><td>${f.score}</td>` +
    `<td>${f.snr}</td><td>${f.stars}</td><td>${f.fwhm}</td></tr>`).join('');
  const log = document.getElementById('log');
  const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
  log.textContent = (s.log || []).join('\\n');
  if (stick) log.scrollTop = log.scrollHeight;

  latestSlug = s.latest_slug || latestSlug;
  lastPreviewVer = s.preview_version;
  curNamed = s.named || [];
  // rebuild selectors / strip only when their versions change
  if (s.named_version !== namedVer || s.frames_img_version !== framesVer) {
    namedVer = s.named_version; framesVer = s.frames_img_version;
    rebuildViewOptions(s.named || [], s.frames_img || []);
    rebuildStrip(s.frames_img || []);
    refreshStretchNote();
  }
  // live-follow the latest preview unless the user took control
  if (follow && !manualStretch && s.preview_version > 0
      && mainImg.dataset.pv !== String(s.preview_version)) {
    mainImg.dataset.pv = String(s.preview_version);
    loadMain('/preview.jpg?v=' + s.preview_version, latestSlug);
    document.getElementById('previewCap').textContent = s.preview_caption || '';
  }
  if (s.done && s.summary) {
    const el = document.getElementById('summary');
    el.style.display = 'block';
    document.getElementById('summarydl').innerHTML =
      Object.entries(s.summary).map(([k, v]) =>
        `<dt>${esc(k.replace(/_/g, ' '))}</dt><dd>${esc(v)}</dd>`).join('');
  }
  updateRunStatus(s.run_status || 'idle', s.run_error || null);
};

// ── Setup panel: form schema, submission, run status ─────────────────────
const QUICK_SELECTS = ['preset', 'stack_method', 'denoiser', 'deconvolve_mode'];
const QUICK_BOOLS = [
  ['auto', 'Auto advisor'],
  ['trail_reject', 'Trail reject'],
  ['local_normalize', 'Local normalize'],
  ['repair_stars', 'Repair stars'],
  ['elastic_registration', 'Elastic registration'],
  ['use_gpu', 'Use GPU'],
  ['no_resume', 'Start fresh (ignore checkpoint)'],
  ['verbose', 'Verbose'],
];
const QUICK_DESTS = new Set([
  'directory', 'output', 'drizzle_scale',
  ...QUICK_SELECTS, ...QUICK_BOOLS.map(b => b[0]),
]);
const touched = new Set();
let schemaByDest = {};
let lastRunStatus = 'idle';

function markTouched(dest) { touched.add(dest); }

function fieldValue(dest, kind, el) {
  if (kind === 'bool_true' || kind === 'bool_false') return el.checked;
  if (kind === 'number') return el.value === '' ? null : Number(el.value);
  return el.value;
}

function populateOptions(selectEl, choices, dflt) {
  for (const c of (choices || [])) {
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    if (c === dflt) opt.selected = true;
    selectEl.appendChild(opt);
  }
}

function buildQuickBools() {
  const wrap = document.getElementById('quickBools');
  wrap.innerHTML = '';
  for (const [dest, label] of QUICK_BOOLS) {
    const f = schemaByDest[dest];
    const id = 'f_' + dest;
    const lbl = document.createElement('label');
    lbl.className = 'quick-bool';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.id = id;
    cb.checked = !!(f && f.default);
    cb.onchange = () => markTouched(dest);
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(label));
    wrap.appendChild(lbl);
  }
}

function populateSelect(dest) {
  const f = schemaByDest[dest];
  const el = document.getElementById('f_' + dest);
  if (!f || !el) return;
  populateOptions(el, f.choices, f.default);
  el.onchange = () => markTouched(dest);
}

const advancedBrowseBtns = [];  // {btn, el, widget, dest} -- revealed by enablePywebviewPickers()

function renderAdvancedField(f) {
  const row = document.createElement('div');
  row.className = 'adv-row';
  const lbl = document.createElement('label');
  lbl.className = 'adv-label';
  lbl.textContent = f.flag;
  lbl.title = f.help || '';
  row.appendChild(lbl);
  let el;
  if (f.kind === 'bool_true' || f.kind === 'bool_false') {
    el = document.createElement('input');
    el.type = 'checkbox'; el.checked = !!f.default;
  } else if (f.kind === 'select') {
    el = document.createElement('select');
    populateOptions(el, f.choices, f.default);
  } else {
    el = document.createElement('input');
    el.type = (f.kind === 'number') ? 'number' : 'text';
    if (f.kind === 'number' && f.default != null) el.placeholder = String(f.default);
    if (f.kind === 'text' && f.default != null && f.default !== '') el.placeholder = String(f.default);
    el.className = 'adv-input';
  }
  el.title = f.help || '';
  el.oninput = el.onchange = () => markTouched(f.dest);
  el._kind = f.kind; el._dest = f.dest;
  row.appendChild(el);
  if (f.widget) {
    const btn = document.createElement('button');
    btn.type = 'button'; btn.textContent = 'Browse…'; btn.className = 'adv-browse';
    btn.style.display = 'none';
    row.appendChild(btn);
    advancedBrowseBtns.push({ btn, el, widget: f.widget, dest: f.dest });
  }
  return { row, el };
}

const advancedEls = {};

function buildAdvanced(schema) {
  const wrap = document.getElementById('advancedFields');
  wrap.innerHTML = '';
  for (const [group, fields] of Object.entries(schema)) {
    const shown = fields.filter(f => !QUICK_DESTS.has(f.dest));
    if (!shown.length) continue;
    const h = document.createElement('div');
    h.className = 'adv-group-h';
    h.textContent = group;
    wrap.appendChild(h);
    for (const f of shown) {
      const { row, el } = renderAdvancedField(f);
      advancedEls[f.dest] = { el, kind: f.kind };
      wrap.appendChild(row);
    }
  }
}

async function loadSchema() {
  const resp = await fetch('/api/schema');
  const schema = await resp.json();
  schemaByDest = {};
  for (const fields of Object.values(schema)) {
    for (const f of fields) schemaByDest[f.dest] = f;
  }
  for (const dest of QUICK_SELECTS) populateSelect(dest);
  buildQuickBools();
  buildAdvanced(schema);
  enablePywebviewPickers();
}

function collectForm() {
  const form = {};
  const dir = document.getElementById('f_directory').value.trim();
  if (dir) form.directory = dir;
  const out = document.getElementById('f_output').value.trim();
  if (out) form.output = out;
  for (const dest of QUICK_SELECTS) {
    if (touched.has(dest)) form[dest] = document.getElementById('f_' + dest).value;
  }
  if (touched.has('drizzle_scale')) {
    form.drizzle_scale = Number(document.getElementById('f_drizzle_scale').value);
  }
  for (const [dest] of QUICK_BOOLS) {
    if (touched.has(dest)) form[dest] = document.getElementById('f_' + dest).checked;
  }
  for (const [dest, { el, kind }] of Object.entries(advancedEls)) {
    if (touched.has(dest)) form[dest] = fieldValue(dest, kind, el);
  }
  return form;
}

function updateRunStatus(status, error) {
  lastRunStatus = status;
  const btn = document.getElementById('startBtn');
  const note = document.getElementById('startStatus');
  btn.disabled = (status === 'running');
  if (status === 'running') note.textContent = 'Running…';
  else if (status === 'ok') note.textContent = 'Complete.';
  else if (status === 'error') note.textContent = 'Error: ' + (error || 'unknown error');
  else note.textContent = 'Idle.';
}

document.getElementById('startBtn').onclick = async () => {
  const form = collectForm();
  if (!form.directory) {
    document.getElementById('startStatus').textContent = 'Directory is required.';
    return;
  }
  document.getElementById('startBtn').disabled = true;
  document.getElementById('startStatus').textContent = 'Starting…';
  try {
    const resp = await fetch('/api/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form),
    });
    const result = await resp.json();
    if (!result.ok) {
      document.getElementById('startStatus').textContent = 'Error: ' + result.error;
      document.getElementById('startBtn').disabled = false;
    }
  } catch (e) {
    document.getElementById('startStatus').textContent = 'Error: ' + e;
    document.getElementById('startBtn').disabled = false;
  }
};

function enablePywebviewPickers() {
  if (!window.pywebview) return;
  document.getElementById('browseDir').style.display = '';
  document.getElementById('browseOut').style.display = '';
  document.getElementById('browseDir').onclick = async () => {
    const p = await window.pywebview.api.browse_directory();
    if (p) { document.getElementById('f_directory').value = p; markTouched('directory'); }
  };
  document.getElementById('browseOut').onclick = async () => {
    const p = await window.pywebview.api.browse_output_path();
    if (p) { document.getElementById('f_output').value = p; markTouched('output'); }
  };
  // Advanced-panel path fields (populated by buildAdvanced() -- may run
  // before or after pywebview becomes ready, so this is called again at
  // the end of loadSchema() to cover either ordering).
  for (const { btn, el, widget, dest } of advancedBrowseBtns) {
    btn.style.display = '';
    btn.onclick = async () => {
      const p = await window.pywebview.api.browse_path(widget);
      if (p) { el.value = p; markTouched(dest); }
    };
  }
}
// window.pywebview is injected asynchronously -- it may not exist yet at
// script-load time even inside the native window. pywebview fires
// 'pywebviewready' once the API bridge is actually attached; also check
// immediately in case it's already there (e.g. on a fast reload).
window.addEventListener('pywebviewready', enablePywebviewPickers);
enablePywebviewPickers();

loadSchema();
</script>
</body>
</html>
"""
