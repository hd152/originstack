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

import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional

_MAX_LOG_LINES = 500
_MAX_FRAME_ROWS = 60
_MAX_FRAME_THUMBS = 32     # per-frame preview ring (the table lists the last 20 frames)
_MAX_NAMED = 16            # retained milestone previews


def _decimate(rgb, max_dim: int):
    """Average ``rgb`` (HWC float) down by an integer factor to at most ``max_dim`` px.

    Stretching a full 2048x3056 sub to make a 512 px thumbnail took ~2.5 s, nearly all of it
    in the stretch's statistics over pixels the thumbnail then throws away; averaging first
    leaves ~0.2 s, so every processed frame can have a thumbnail rather than every fifth.
    """
    import numpy as np
    a = np.asarray(rgb)
    h, w = a.shape[:2]
    f = -(-max(h, w) // max_dim)
    if f <= 1 or a.ndim != 3:
        return a
    hb, wb = h // f, w // f
    if hb < 1 or wb < 1:
        return a
    return a[:hb * f, :wb * f].reshape(hb, f, wb, f, a.shape[2]).mean(axis=(1, 3), dtype=np.float32)


def _slugify(text: str) -> str:
    s = re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')
    return s or 'preview'


class UIEvents:
    def __init__(self) -> None:
        self.active = False
        self._lock = threading.Lock()
        self._version = 0          # bumped on every state change (GUI polls this)
        # latest preview (back-compat single slot)
        self._preview_version = 0
        self._preview_caption = ""
        self._last_preview_time = 0.0
        # named milestone previews: slug -> {caption, ver, jpeg}
        self._named: "OrderedDict[str, dict]" = OrderedDict()
        self._named_version = 0
        self._latest_slug = ""
        # per-frame thumbnails (ring)
        self._frame_thumbs: "OrderedDict[int, dict]" = OrderedDict()
        self._frame_thumb_seq = 0
        self._frame_thumbs_version = 0
        # The image the run's preview JPEG was stretched from, kept so the
        # user can pick another stretch once the run is done (restretch()).
        self._final_src: Optional[dict] = None
        # What switching Phase 4 steps off and on afterwards needs (remember_postprocess).
        self._pp_session: Optional[dict] = None
        self._state: Dict[str, Any] = {
            'run': {},                 # target/output/frame counts
            'phase': 0,
            'phase_title': '',
            'phases': {},              # {n: {'title', 'started', 'elapsed'}}
            'progress': {'label': '', 'done': 0, 'total': 0,
                         'started': None, 'eta': None},
            'log': [],
            'frames': [],              # recent per-frame metric rows
            'frame_report': [],        # --frame-report: kept/rejected verdict per light
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
            self._state['progress'] = {'label': '', 'done': 0, 'total': 0,
                                       'started': now, 'eta': None}
            self._bump()

    def progress(self, label: str, done: int, total: int) -> None:
        if not self.active:
            return
        with self._lock:
            done = int(done)
            total = int(total)
            previous = self._state['progress']
            started = previous.get('started') or time.time()
            elapsed = max(0.0, time.time() - started)
            eta = None
            if done > 0 and total > done:
                eta = elapsed * (total - done) / done
            self._state['progress'] = {'label': label, 'done': done,
                                       'total': total, 'started': started,
                                       'eta': eta}
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

    def frame_report(self, rows: list) -> None:
        """Publish the per-frame kept/rejected table (``src/frame_report.py``)."""
        if not self.active:
            return
        with self._lock:
            self._state['frame_report'] = [dict(r) for r in rows]
            self._bump()

    @staticmethod
    def _stretch_kw(args) -> dict:
        """Extract stretch parameters from an args namespace (or defaults)."""
        from src.io_fits import preview_stretch_kw
        return preview_stretch_kw(args)

    def _set_named_slot(self, slug: str, caption: str, data: bytes) -> None:
        """Register / update a named viewer slot (caller holds the lock)."""
        self._named_version += 1
        self._named[slug] = {'caption': caption, 'ver': self._named_version,
                             'jpeg': data}
        self._named.move_to_end(slug)
        self._latest_slug = slug
        # evict oldest slots beyond the cap (keep 'final' + newest)
        while len(self._named) > _MAX_NAMED:
            for k in list(self._named.keys()):
                if k != 'final' and k != self._latest_slug:
                    del self._named[k]
                    break
            else:
                break
        self._bump()

    def preview(self, rgb, caption: str, args=None, slot: Optional[str] = None,
                min_interval: float = 2.0) -> None:
        """Publish a stretched preview of an HWC float32 image. Encoding is
        throttled so frequent milestones don't spend time on JPEG encodes.
        The preview is registered as a named slot so it can be picked for
        before/after compare."""
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
        slug = _slugify(slot or caption)
        with self._lock:
            self._preview_caption = caption
            self._preview_version += 1
            self._last_preview_time = now
            self._set_named_slot(slug, caption, data)

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
            data = preview_jpeg_bytes(_decimate(rgb, max_dim), max_dim=max_dim,
                                      desaturate=0.6, **kw)
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

    def remember_final_preview(self, rgb, path: str, args=None, starless=None) -> None:
        """Keep the image ``path`` (the run's preview JPEG) was just stretched
        from, with the settings used, so ``restretch`` can redo it with another
        stretch. Holds a full-resolution float image until the next run starts,
        so it is only kept while the desktop app is attached. The last call
        wins: in a multi-target run that is the last JPEG written."""
        if not self.active:
            return
        kw = self._stretch_kw(args)
        from src.io_fits import preview_exports
        with self._lock:
            self._final_src = {'rgb': rgb, 'starless': starless,
                               'path': os.path.abspath(path), 'kw': kw,
                               'exports': preview_exports(args) if args is not None else ()}
            self._bump()

    def remember_postprocess(self, linear, args, final, stats, snaps) -> None:
        """Keep a finished Phase 4 -- the linear stack it started from, its settings and the
        per-step snapshots -- so ``rerun_postprocess`` can redo it with steps switched on or
        off. Holds the linear stack plus about one image per enabled step until the next
        run starts, so only while the desktop app is attached. The last call wins."""
        if not self.active:
            return
        with self._lock:
            self._pp_session = {'linear': linear, 'args': args, 'final': final,
                                'stats': stats, 'snaps': snaps}
            self._bump()

    def postprocess_steps(self) -> Optional[list]:
        """The Phase 4 steps of the remembered run as {id, label, on}, or None."""
        from src.postprocess import phase4_toggles
        with self._lock:
            sess = self._pp_session
        return None if sess is None else phase4_toggles(sess['args'])

    def rerun_postprocess(self, off: Iterable[str]) -> Optional[bytes]:
        """Redo Phase 4 with exactly the toggleable steps in ``off`` skipped, resuming at the
        earliest step that differs from the last run (``postprocess.StepSnapshots``), then
        re-render the final preview with the stretch it has now (as ``restretch``). Returns
        the viewer JPEG, or None when nothing changed. Slow (seconds to a minute): call it
        off the GUI thread. ``RuntimeError`` when there is no run to redo."""
        from src.postprocess import PHASE4_TOGGLES, phase4_resume_step, postprocess_stack
        with self._lock:
            sess, fs = self._pp_session, self._final_src
        if sess is None or fs is None:
            raise RuntimeError('no finished run to post-process again')
        args, snaps = sess['args'], sess['snaps']
        ids = {i for i, _, _, _ in PHASE4_TOGGLES}
        old = set(getattr(args, 'skip_step', None) or [])
        new = (old - ids) | (set(off) & ids)
        changed = (old ^ new) & ids
        if not changed:
            return None
        resume = phase4_resume_step(changed)
        snaps.resume = None if resume == 'early' else resume
        args.skip_step = sorted(new)
        args._pp_snaps = snaps
        try:
            result = postprocess_stack(sess['linear'].copy(), args, sess['final'], sess['stats'])
        except BaseException:
            # the snapshots no longer match args.skip_step: restore it and start over
            args.skip_step = sorted(old)
            snaps.reset()
            raise
        finally:
            snaps.resume = None
            del args._pp_snaps
        starless = None
        if getattr(args, 'layered_stretch', False) and getattr(args, 'stretch', 'ghs') == 'ghs':
            from src.star_removal import starless_from_image
            starless = starless_from_image(result)
        with self._lock:
            if self._final_src is not fs:
                raise RuntimeError('a new run started')
            kw = self._stretch_kw(args)
            kw['stretch'], kw['color'] = fs['kw']['stretch'], fs['kw']['color']
            fs.update(rgb=result, starless=starless, kw=kw)
            fs.pop('starless_img', None)
        return self.restretch(kw['stretch'], kw['color'])

    def starless_view(self, on: bool) -> bytes:
        """Show the final image without its stars (``on``) or with them again, in the
        viewer's 'final' slot. Stars come out with the remover ``--remove-stars-method
        inpaint`` uses, on the finished image, so it matches the starless sidecar of an
        inpaint run; the nn method works on the linear stack and cannot be applied here.
        Switching on also writes ``<preview stem>_starless.jpg`` (and the run's PNG exports)
        next to the preview; the stars-in files are left alone. The first switch after a
        run or a re-run detects and removes the stars, so call it off the GUI thread.
        Returns the viewer-sized JPEG; ``RuntimeError`` when there is no finished image or
        no usable star detections."""
        from src.io_fits import (
            preview_jpeg_from_uint8,
            render_preview_rgb_float,
            to_uint8,
            write_final_images,
        )
        with self._lock:
            fs = self._final_src
            if fs is None:
                raise RuntimeError('no finished preview to remove stars from')
            cached = fs.get('starless_img')
            rgb, kw = fs['rgb'], dict(fs['kw'])
        if on:
            starless = cached
            if starless is None:
                from src.star_removal import starless_from_image
                starless = starless_from_image(rgb)
                if starless is None:
                    raise RuntimeError('no usable star detections to remove')
            src = starless
        else:
            src = rgb
        img = render_preview_rgb_float(src, starless=None if on else fs['starless'], **kw)
        if img is None:
            raise RuntimeError(f"the {kw['stretch']} stretch is unavailable")
        if on:
            stem = os.path.splitext(fs['path'])[0] + '_starless.jpg'
            write_final_images(img, stem, fs.get('exports', ()))
        data = preview_jpeg_from_uint8(to_uint8(img))
        with self._lock:
            if self._final_src is fs and fs['rgb'] is rgb:    # not replaced by a new run / re-run
                if on:
                    fs['starless_img'] = starless
                caption = 'Final (starless)' if on else 'Final (post-processed)'
                self._set_named_slot('final', caption, data)
        return data

    def restretch(self, stretch: str, color: str) -> bytes:
        """Re-render the remembered final preview with ``stretch``/``color``,
        overwrite its JPEG (and its ``--export png``/``png16`` copies) and update the
        'final' viewer slot. Returns the
        viewer-sized JPEG bytes. Runs the full-resolution stretch, so call it
        off the GUI thread. Raises ``RuntimeError`` when there is nothing to
        re-stretch or the stretch backend is unavailable; ``OSError`` when the
        file cannot be written."""
        from src.io_fits import preview_jpeg_from_uint8, render_preview_rgb_float, to_uint8, write_final_images
        with self._lock:
            src = self._final_src
        if src is None:
            raise RuntimeError('no finished preview to re-stretch')
        kw = dict(src['kw'], stretch=stretch, color=color)
        img = render_preview_rgb_float(src['rgb'], starless=src['starless'], **kw)
        if img is None:
            raise RuntimeError(f'the {stretch} stretch is unavailable')
        write_final_images(img, src['path'], src.get('exports', ()))
        data = preview_jpeg_from_uint8(to_uint8(img))
        with self._lock:
            # A new run may have started while this rendered: its state wins.
            if self._final_src is src:
                src['kw'] = kw
                caption = self._named.get('final', {}).get('caption', 'Final (post-processed)')
                self._set_named_slot('final', caption, data)
        return data

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
            self._state['frame_report'] = []
            self._state['summary'] = None
            self._state['done'] = False
            self._state['phase'] = 0
            self._state['phase_title'] = ''
            self._state['phases'] = {}
            self._state['progress'] = {'label': '', 'done': 0, 'total': 0}
            self._state['run_status'] = 'running'
            self._state['run_error'] = None
            # Drop every preview from the previous run: the single back-compat
            # slot, the retained milestone slots and the per-frame thumbnail
            # ring. Version counters are
            # bumped, not reset, so the GUI's poll loop sees the change and
            # redraws an empty viewer instead of showing last run's images.
            self._preview_caption = ''
            self._preview_version += 1
            self._last_preview_time = 0.0
            self._named.clear()
            self._latest_slug = ''
            self._named_version += 1
            self._frame_thumbs.clear()
            self._frame_thumb_seq = 0
            self._frame_thumbs_version += 1
            self._final_src = None
            self._pp_session = None
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
            snap['named'] = [{'slug': k, 'caption': v['caption'], 'ver': v['ver']}
                             for k, v in self._named.items()]
            snap['frames_img_version'] = self._frame_thumbs_version
            snap['frames_img'] = [{'id': v['id'], 'name': v['name']}
                                  for v in self._frame_thumbs.values()]
            snap['postprocess'] = self._pp_session is not None
            fs = self._final_src
            snap['restretch'] = (None if fs is None else
                                 {'stretch': fs['kw']['stretch'], 'color': fs['kw']['color'],
                                  'path': fs['path']})
            return snap

    def named_jpeg(self, slug: str) -> Optional[bytes]:
        with self._lock:
            slot = self._named.get(slug)
            return slot['jpeg'] if slot else None

    def frame_jpeg(self, fid: int) -> Optional[bytes]:
        with self._lock:
            slot = self._frame_thumbs.get(fid)
            return slot['jpeg'] if slot else None

    def frame_jpeg_for(self, name: str) -> Optional[bytes]:
        """The newest kept thumbnail of the frame called ``name`` (its row in the frames
        table), or None when none was published or it has left the ring."""
        with self._lock:
            for slot in reversed(self._frame_thumbs.values()):
                if slot['name'] == name:
                    return slot['jpeg']
        return None


_ui_events: Optional[UIEvents] = None


def get_ui_events() -> UIEvents:
    global _ui_events
    if _ui_events is None:
        _ui_events = UIEvents()
    return _ui_events
