"""Real-time (live) stacking.

Watches a capture directory and folds each new light frame into a running
stack as it lands — the classic "live stacking" workflow (SharpCap / ASIStudio)
that lets you watch signal build up at the telescope instead of stacking after
the session. Each new sub is calibrated, debayered, registered to the running
reference frame, and accumulated into a per-pixel weighted-mean stack; the
growing result and a running SNR / integration-time readout are pushed to the
``--web-view`` dashboard after every frame.

Design (keeps the streaming, low-memory model):
  * The first accepted frame is the registration reference and seeds the
    accumulators. Every later frame is aligned to it (translation via the
    pyramid + phase-correlation cascade, seeded from the previous shift).
  * Two full-frame float64 buffers only: ``acc`` (Σ weight·pixel) and ``wsum``
    (Σ weight, per pixel, honouring each frame's valid coverage after the
    shift). The live stack is ``acc / wsum`` — O(1) memory in the frame count.
  * Files still being written are skipped until their mtime is a poll-interval
    old, so a half-flushed sub is never read.

Runs until Ctrl-C (or an optional ``--live-duration``); on exit the running
linear stack is written to the output path (plus a preview JPEG), ready to feed
a normal post-processing run.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from src.utils import safe_print, format_time

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _ndi = None
    _HAS_SCIPY = False


def _stack_snr(lum: np.ndarray) -> float:
    """SNR of the running stack: (mean object signal - sky) / sky-noise.

    The object set is the pixels a stable 5-sigma above the sky, so as more
    frames average the sky noise down (signal fixed) the number rises — the
    live "signal building up" indicator."""
    med = float(np.median(lum))
    sig = 1.4826 * float(np.median(np.abs(lum - med)))
    if sig <= 1e-9:
        sig = float(np.std(lum)) or 1.0
    obj = lum > med + 5.0 * sig
    if obj.sum() >= 5:
        signal = float(lum[obj].mean()) - med
    else:
        signal = float(np.percentile(lum, 99.9)) - med
    return max(signal / sig, 0.0)


class LiveStacker:
    """Incremental per-pixel weighted-mean stacker driven by new files."""

    def __init__(self, args, masters: Dict, webview=None):
        self.args = args
        self.masters = masters
        self.wv = webview
        self.ref_lum: Optional[np.ndarray] = None
        self.acc: Optional[np.ndarray] = None     # (H, W, C) float64 Σ w·px
        self.wsum: Optional[np.ndarray] = None    # (H, W) float64 Σ w
        self.n = 0
        self.n_rejected = 0
        self.seed_shift: Tuple[float, float] = (0.0, 0.0)
        self.total_exposure = 0.0
        self.snr_history = []  # list of (n_frames, snr)
        self._sb = getattr(args, '_session_bayer', None)

    # ── per-frame ──────────────────────────────────────────────────────────

    def add_frame(self, path: str, header: Optional[dict] = None) -> bool:
        """Process one light frame and fold it into the running stack.
        Returns True if the frame was accepted."""
        from src.frame_processor import _process_single_frame
        from src.registration import calculate_shift, apply_transform

        try:
            res = _process_single_frame(
                path, header or {}, self.masters,
                self.args.debayer_method, self.args.white_balance,
                ca_correction=getattr(self.args, 'ca_correction', False),
                cosmic_ray_rejection=False,   # per-pixel rejection needs the stack
                advanced_metrics=False,
                session_bayer=self._sb,
                pre_gradient_removal=getattr(self.args, 'pre_gradient_removal', False),
                trail_reject=getattr(self.args, 'trail_reject', False))
        except Exception as exc:
            safe_print(f"  LIVE skip {os.path.basename(path)}: {exc}")
            self.n_rejected += 1
            return False
        if res.get('error'):
            safe_print(f"  LIVE skip {os.path.basename(path)}: {res['error']}")
            self.n_rejected += 1
            return False

        rgb = np.asarray(res['rgb'], dtype=np.float32)
        lum = np.asarray(res['lum'], dtype=np.float32)
        metrics = res.get('metrics') or {}

        # Minimal quality gate: only drop essentially-blank frames. Registration
        # runs on any structure (stars OR nebula via phase-correlation), and a
        # genuinely misregistered frame is caught by the shift sanity check
        # below — so we don't hard-require detected stars here.
        if metrics.get('snr', 1.0) < 0.3:
            safe_print(f"  LIVE reject {os.path.basename(path)}: low quality "
                       f"(stars={metrics.get('star_count', 0)}, "
                       f"snr={metrics.get('snr', 0):.2f})")
            self.n_rejected += 1
            if self.wv is not None:
                self.wv.frame_metrics(os.path.basename(path), metrics, accepted=False)
            return False

        weight = max(float(metrics.get('score', 1.0)), 0.01)

        if self.ref_lum is None:
            # First frame becomes the reference and seeds the accumulators.
            self.ref_lum = lum
            self.acc = rgb.astype(np.float64) * weight
            self.wsum = np.full(lum.shape, weight, dtype=np.float64)
        else:
            if lum.shape != self.ref_lum.shape:
                safe_print(f"  LIVE skip {os.path.basename(path)}: shape "
                           f"{lum.shape} != reference {self.ref_lum.shape}")
                self.n_rejected += 1
                return False
            try:
                sy, sx = calculate_shift(self.ref_lum, lum, seed_shift=self.seed_shift)
            except Exception:
                sy, sx = 0.0, 0.0
            H, W = self.ref_lum.shape
            if not (np.isfinite(sy) and np.isfinite(sx)) or abs(sy) > 0.3 * H or abs(sx) > 0.3 * W:
                safe_print(f"  LIVE reject {os.path.basename(path)}: bad shift "
                           f"({sx:.1f}, {sy:.1f})")
                self.n_rejected += 1
                if self.wv is not None:
                    self.wv.frame_metrics(os.path.basename(path), metrics, accepted=False)
                return False
            self.seed_shift = (sy, sx)
            aligned = apply_transform(rgb, shift=(sy, sx))
            # Coverage: pixels the shift filled from outside the frame are 0.
            cov = _ndi.shift(np.ones((H, W), dtype=np.float32), shift=(sy, sx),
                             order=0, mode='constant', cval=0.0) > 0.5
            w_pix = cov.astype(np.float64) * weight
            self.acc += aligned.astype(np.float64) * w_pix[:, :, None]
            self.wsum += w_pix

        self.n += 1
        try:
            self.total_exposure += float((header or {}).get('EXPTIME', 0) or 0)
        except (TypeError, ValueError):
            pass

        self._publish(os.path.basename(path), metrics)
        return True

    def current_stack(self) -> Optional[np.ndarray]:
        if self.acc is None:
            return None
        safe = np.maximum(self.wsum, 1e-9)
        return (self.acc / safe[:, :, None]).astype(np.float32)

    def _publish(self, name: str, metrics: dict) -> None:
        stack = self.current_stack()
        if stack is None:
            return
        lum = 0.299 * stack[:, :, 0] + 0.587 * stack[:, :, 1] + 0.114 * stack[:, :, 2]
        snr = _stack_snr(lum)
        self.snr_history.append((self.n, snr))
        exp_str = (f", {format_time(self.total_exposure)} integration"
                   if self.total_exposure > 0 else "")
        safe_print(f"  LIVE +{name}: {self.n} stacked, stack SNR {snr:.1f}{exp_str}")
        if self.wv is not None:
            self.wv.set_run_info(target='Live stacking',
                                 frames_stacked=self.n,
                                 rejected=self.n_rejected,
                                 stack_snr=round(snr, 1),
                                 integration=format_time(self.total_exposure)
                                 if self.total_exposure > 0 else f"{self.n} subs")
            self.wv.frame_metrics(name, metrics, accepted=True)
            self.wv.progress('Live stacking', self.n, self.n)
            self.wv.preview(stack, f"{self.n} frames — SNR {snr:.1f}",
                            args=self.args, slot='live', min_interval=1.0)

    # ── watch loop ─────────────────────────────────────────────────────────

    def _discover_new_lights(self, directory: str, seen: set,
                             min_age: float) -> list:
        """Return sorted new light-frame paths whose files are done writing."""
        from src.frame_discovery import discover_frames
        try:
            frames = discover_frames(directory)
        except Exception:
            return []
        out = []
        now = time.time()
        for fi in frames.get('light', []):
            p = fi.path
            if p in seen:
                continue
            # SER virtual paths (real::idx) and real files: age-gate the real file.
            real = p.split('::', 1)[0]
            try:
                if now - os.path.getmtime(real) < min_age:
                    continue  # still being written
            except OSError:
                continue
            out.append((p, fi.header))
        out.sort(key=lambda t: t[0])
        return out

    def run(self, directory: str, interval: float = 4.0,
            duration: Optional[float] = None) -> None:
        """Poll *directory* every *interval* seconds, stacking new frames until
        Ctrl-C or *duration* seconds elapse."""
        safe_print(f"\n  Live stacking: watching {directory} "
                   f"(poll {interval:.0f}s, Ctrl-C to stop)")
        seen: set = set()
        start = time.time()
        try:
            while True:
                new = self._discover_new_lights(directory, seen, min_age=interval)
                for path, header in new:
                    seen.add(path)
                    self.add_frame(path, header)
                if duration is not None and (time.time() - start) >= duration:
                    safe_print(f"  Live stacking: reached {format_time(duration)} limit")
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            safe_print("\n  Live stacking stopped (Ctrl-C)")

    # ── finish ─────────────────────────────────────────────────────────────

    def save(self, output_path: str) -> None:
        stack = self.current_stack()
        if stack is None or self.n == 0:
            safe_print("  Live stacking: no frames stacked — nothing to save")
            return
        from astropy.io import fits
        data_out = np.transpose(stack, (2, 0, 1)).astype(np.float32)
        hdu = fits.PrimaryHDU(data=data_out)
        hdu.header['CREATOR'] = 'originstack.py live'
        hdu.header['COMBINED'] = (True, 'Live-stacked image')
        hdu.header['RAWSTACK'] = (True, 'Linear pre-post-processing live stack')
        hdu.header['NFRAMES'] = (self.n, 'Frames in live stack')
        if self.total_exposure > 0:
            hdu.header['TOTEXP'] = (round(self.total_exposure, 1), 'Total exposure (s)')
        hdu.writeto(output_path, overwrite=True)
        safe_print(f"  Live stack saved: {output_path} ({self.n} frames)")
        try:
            from src.io_fits import save_preview_rgb
            prev = os.path.splitext(output_path)[0] + '.jpg'
            save_preview_rgb(stack, prev, stretch=getattr(self.args, 'stretch', 'ghs'),
                             ghs_b=float(getattr(self.args, 'ghs_b', 8.0)),
                             ghs_sp=float(getattr(self.args, 'ghs_sp', 0.15)),
                             ghs_hp=float(getattr(self.args, 'ghs_hp', 0.95)),
                             black_sigma=float(getattr(self.args, 'preview_black_sigma', 0.0) or 0.0))
            safe_print(f"  Preview: {prev}")
        except Exception:
            pass


def run_live_stack(args) -> int:
    """Entry point for ``--live``: build masters, then watch + stack."""
    if not _HAS_SCIPY:
        safe_print("  ERROR: live stacking requires scipy")
        return 1
    directory = args.directory
    if not os.path.isdir(directory):
        safe_print(f"  ERROR: not a directory: {directory}")
        return 1

    # Build calibration masters from whatever darks/flats/bias exist now.
    from src.frame_discovery import discover_frames
    from src.cli import _build_masters, _load_calibration_dir
    frames = discover_frames(directory)
    extra = _load_calibration_dir(args)
    for k in ('dark', 'flat', 'bias'):
        frames[k] = (frames.get(k) or []) + extra.get(k, [])
    masters = _build_masters(frames, args=args)
    from src.frame_processor import _build_flat_norm
    _build_flat_norm(masters, frames.get('light', []))

    # Session Bayer pattern from the first light header (headerless -> default).
    lights = frames.get('light', [])
    if lights and getattr(args, '_session_bayer', None) is None:
        args._session_bayer = lights[0].header.get('BAYERPAT') or lights[0].header.get('COLORTYP')

    wv = None
    if getattr(args, 'web_view', False) or getattr(args, 'live_web', True):
        from src.webview import get_webview
        wv = get_webview()
        if not wv.active:
            url = wv.start(port=getattr(args, 'web_view_port', 8765))
            if url:
                safe_print(f"  Web view: {url}")

    stacker = LiveStacker(args, masters, webview=wv)
    stacker.run(directory,
                interval=float(getattr(args, 'live_interval', 4.0)),
                duration=(float(args.live_duration) * 60.0
                          if getattr(args, 'live_duration', None) else None))
    stacker.save(args.output)
    if wv is not None:
        wv.summary(frames_stacked=stacker.n, rejected=stacker.n_rejected,
                   output=os.path.basename(args.output))
    return 0
