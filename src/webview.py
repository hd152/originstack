"""Live stacking web view (``--web-view``).

A pure-stdlib local dashboard: a ``ThreadingHTTPServer`` daemon thread in the
main process serves one self-contained HTML page with live phase progress,
the log stream, a per-frame quality ticker, and preview images published at
processing milestones. Server-Sent Events push state snapshots; the preview
image is fetched by version.

Inactive by default: every publish method is a no-op until ``start()`` is
called, so the pipeline instrumentation costs nothing on normal runs.
Mirrors the ``get_gpu()`` module-level singleton pattern.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

_MAX_LOG_LINES = 500
_MAX_FRAME_ROWS = 60


class WebView:
    def __init__(self) -> None:
        self.active = False
        self._lock = threading.Lock()
        self._server = None
        self._version = 0          # bumped on every state change (SSE trigger)
        self._preview_version = 0
        self._preview_bytes: Optional[bytes] = None
        self._preview_caption = ""
        self._last_preview_time = 0.0
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

    def preview(self, rgb, caption: str, args=None,
                min_interval: float = 2.0) -> None:
        """Publish a stretched preview of an HWC float32 image. Encoding is
        throttled so frequent milestones don't spend time on JPEG encodes."""
        if not self.active:
            return
        now = time.time()
        if now - self._last_preview_time < min_interval:
            return
        try:
            from src.io_fits import preview_jpeg_bytes
            kw = {}
            if args is not None:
                kw = dict(stretch=getattr(args, 'stretch', 'ghs'),
                          ghs_b=float(getattr(args, 'ghs_b', 8.0)),
                          ghs_sp=float(getattr(args, 'ghs_sp', 0.15)),
                          ghs_hp=float(getattr(args, 'ghs_hp', 0.95)),
                          black_sigma=float(getattr(args, 'preview_black_sigma',
                                                    0.0) or 0.0))
            data = preview_jpeg_bytes(rgb, max_dim=1024, **kw)
        except Exception:
            return
        if not data:
            return
        with self._lock:
            self._preview_bytes = data
            self._preview_caption = caption
            self._preview_version += 1
            self._last_preview_time = now
            self._bump()

    def summary(self, **fields: Any) -> None:
        if not self.active:
            return
        with self._lock:
            self._state['summary'] = fields
            self._state['done'] = True
            self._bump()

    # ── snapshot for SSE ──────────────────────────────────────────────────

    def _snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = json.loads(json.dumps(self._state))
            snap['version'] = self._version
            snap['preview_version'] = self._preview_version
            snap['preview_caption'] = self._preview_caption
            return snap

    def _preview(self):
        with self._lock:
            return self._preview_bytes

    # ── server ────────────────────────────────────────────────────────────

    def start(self, port: int = 8765) -> Optional[str]:
        """Start the dashboard server; returns the URL (None on failure)."""
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
                path = self.path.split('?')[0]
                if path == '/':
                    self._send(200, 'text/html; charset=utf-8',
                               _PAGE.encode('utf-8'))
                elif path == '/preview.jpg':
                    data = view._preview()
                    if data is None:
                        self._send(404, 'text/plain', b'no preview yet')
                    else:
                        self._send(200, 'image/jpeg', data)
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
  main { display: grid; grid-template-columns: minmax(380px, 1fr) minmax(420px, 1.2fr);
         gap: 16px; padding: 16px 22px; max-width: 1500px; }
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
  #log { height: 300px; overflow-y: auto; background: #0d1117;
         border-radius: 6px; padding: 10px 12px; font: 12px/1.5 Consolas,
         monospace; white-space: pre-wrap; color: #9aa4b2; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: right; padding: 3px 8px; border-bottom: 1px solid #21262d; }
  th:first-child, td:first-child { text-align: left; }
  td.bad { color: #f85149; }
  #previewImg { width: 100%; border-radius: 6px; background: #000;
                min-height: 200px; }
  .cap { margin-top: 6px; font-size: 12px; color: #8b949e; }
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
    <section>
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
      <h2>Log</h2>
      <div id="log"></div>
    </section>
  </div>
  <div>
    <section>
      <h2>Preview</h2>
      <img id="previewImg" alt="waiting for first preview…">
      <div class="cap" id="previewCap">Waiting for the first stack…</div>
    </section>
    <section style="margin-top:16px" id="summary">
      <h2>Complete</h2>
      <dl id="summarydl"></dl>
    </section>
  </div>
</main>
<script>
let pv = -1;
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
    `<tr><td${f.ok ? '' : ' class="bad"'}>${f.name}</td><td>${f.score}</td>` +
    `<td>${f.snr}</td><td>${f.stars}</td><td>${f.fwhm}</td></tr>`).join('');
  const log = document.getElementById('log');
  const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
  log.textContent = (s.log || []).join('\\n');
  if (stick) log.scrollTop = log.scrollHeight;
  if (s.preview_version !== pv && s.preview_version > 0) {
    pv = s.preview_version;
    document.getElementById('previewImg').src = '/preview.jpg?v=' + pv;
    document.getElementById('previewCap').textContent = s.preview_caption || '';
  }
  if (s.done && s.summary) {
    const el = document.getElementById('summary');
    el.style.display = 'block';
    document.getElementById('summarydl').innerHTML =
      Object.entries(s.summary).map(([k, v]) =>
        `<dt>${k.replace(/_/g, ' ')}</dt><dd>${v}</dd>`).join('');
  }
};
</script>
</body>
</html>
"""
