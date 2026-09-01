"""In-process UI event/state sink for the desktop app.

Replaces ``src/webview.py``'s HTTP/SSE dashboard: the pipeline runs on a
background thread (``src/desktop_control.py``'s ``RunManager``) while
tkinter's mainloop owns the main thread, so state can't be pushed into
widgets directly from pipeline code (tkinter widgets may only be touched from
the main thread). Instead this class holds the same state ``WebView`` used to
serve over SSE, and the GUI polls ``version`` on a ``root.after()`` timer and
re-reads whatever changed -- the same poll-a-version-counter pattern the old
SSE endpoint used internally, just without the HTTP hop.

Inactive by default: every publish method is a no-op until ``attach()`` is
called, so the pipeline instrumentation costs nothing on a plain CLI run.
Mirrors the ``get_gpu()`` module-level singleton pattern.
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

_MAX_LOG_LINES = 500
_MAX_FRAME_ROWS = 60
_MAX_FRAME_THUMBS = 24     # per-frame preview ring
_MAX_NAMED = 16            # retained milestone previews (with float source)
_SRC_MAX_DIM = 1000        # downsized float source kept for re-stretch


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


class UIEvents:
    def __init__(self) -> None:
        self.active = False
        self._lock = threading.Lock()
        self._version = 0          # bumped on every state change (GUI polls this)
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
        # Static for the process lifetime -- computed once rather than on
        # every snapshot() call, unlike the other snap fields above which
        # genuinely change per-run.
        from src.utils import read_version
        self._app_version = read_version()

    # ── attach/detach (replaces WebView's HTTP server lifecycle) ──────────

    def attach(self) -> None:
        """Activate publishing -- called once by the desktop app before a
        run starts. A plain CLI run never calls this, so ``active`` stays
        False and every publish method below stays a true no-op."""
        self.active = True

    def detach(self) -> None:
        self.active = False

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
        Stored in a bounded ring; the viewer can page through them.

        A single unstacked sub's per-pixel colour noise (no rejection-combine
        averaging yet) can blow up into a misleading solid green/blue blob
        once downsized to ring-thumbnail size -- partially desaturated here
        (this path only) so it reads as noisy-but-legible star field instead.
        """
        if not self.active:
            return
        kw = self._stretch_kw(args)
        try:
            from src.io_fits import preview_jpeg_bytes
            data = preview_jpeg_bytes(rgb, max_dim=max_dim, desaturate=0.6, **kw)
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
        self._notify(status, error)

    def _notify(self, status: str, error: Optional[str]) -> None:
        """Best-effort native OS notification -- cosmetic, so a failure here
        must never affect run state. Useful because a long stacking run is
        exactly the kind of thing a user tabs away from."""
        try:
            from src.notify import notify_windows
            if status == 'ok':
                msg = 'Stacking run complete.'
            elif status == 'error':
                msg = f'Run failed: {error or "unknown error"}'
            else:
                return
            notify_windows('OriginStack', msg)
        except Exception:
            pass

    # ── re-stretch (on-demand, from retained float source) ────────────────

    def restretch(self, slug: str, params: dict) -> Optional[bytes]:
        """Re-render a retained milestone at new stretch params. Returns JPEG
        bytes (same as ``preview()``'s stored format) -- the GUI layer
        decodes them into a display image; kept as bytes here rather than a
        PIL Image so this method's tested behavior is unchanged from
        ``WebView.restretch()``."""
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

    # ── snapshot for the GUI's poll loop ───────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """A plain-dict copy of everything the GUI needs to redraw itself --
        the in-process equivalent of what ``WebView``'s SSE endpoint used to
        JSON-serialize per client. No serialization needed here; the caller
        (tkinter, same process) can use the nested dicts/lists directly."""
        import copy
        with self._lock:
            snap = copy.deepcopy(self._state)
            snap['version'] = self._version
            snap['app_version'] = self._app_version
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

    def preview_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._preview_bytes

    def named_jpeg(self, slug: str) -> Optional[bytes]:
        with self._lock:
            slot = self._named.get(slug)
            return slot['jpeg'] if slot else None

    def frame_jpeg(self, fid: int) -> Optional[bytes]:
        with self._lock:
            slot = self._frame_thumbs.get(fid)
            return slot['jpeg'] if slot else None


_ui_events: Optional[UIEvents] = None


def get_ui_events() -> UIEvents:
    global _ui_events
    if _ui_events is None:
        _ui_events = UIEvents()
    return _ui_events
