"""Moving-object (asteroid) detection and tracked stacking (``--moving-objects``).

``--transient-detect`` compares two epochs and comet mode tracks one known
object; nothing linked sources that *move during a session*. On an aligned stack
the stars stand still, so a moving object is what each frame has that the stack
(which rejected it) does not:

  1. ``residual_j = frame_j - stack``  -- stars cancel, the mover remains;
  2. a matched-filter-style smooth (gaussian at the PSF width) and a robust noise
     estimate give per-frame candidates above ``threshold`` sigma, away from stars
     (seeing changes leave residuals on every bright star);
  3. candidates from all frames are linked into straight tracks by voting in
     velocity space: for every trial velocity the detections are back-projected to
     a common epoch, and a cell that collects detections from many *different
     frames* is a track. Chance alignments are negligible: a 4 px cell in a
     megapixel frame collects ~0.005 random detections, so six aligned ones is a
     detection;
  4. each track is refined by a least-squares line fit, reported, and (optionally)
     stacked along its own motion so the object comes up out of the noise.

A source that moves less than ~2 FWHM over the whole session cannot be told from
a residual or a hot pixel and is not reported; satellite and aircraft trails are
streaks, not points, and are handled by ``--trail-reject``.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import ndimage

    from src.background import _gaussian_blur
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    ndimage = None
    _gaussian_blur = None
    _HAS_SCIPY = False

_log = logging.getLogger("originstack")

_LUM = (0.299, 0.587, 0.114)


def _lum(frame: np.ndarray) -> np.ndarray:
    f = np.asarray(frame, dtype=np.float32)
    if f.ndim == 2:
        return f
    return _LUM[0] * f[:, :, 0] + _LUM[1] * f[:, :, 1] + _LUM[2] * f[:, :, 2]


def star_mask(ref_lum: np.ndarray, sources, fwhm: float) -> np.ndarray:
    """Boolean mask of where a moving-object candidate is not credible.

    Around every star of the reference, out to a radius that grows with its
    brightness (a bright star's seeing residual reaches further)."""
    H, W = ref_lum.shape
    mask = np.zeros((H, W), dtype=bool)
    if sources is None or len(sources) == 0:
        return mask
    flux = np.asarray(sources['flux'], dtype=np.float64)
    ref = max(float(np.median(flux[flux > 0])) if (flux > 0).any() else 1.0, 1e-9)
    y = np.asarray(sources['ycentroid'], dtype=np.float64)
    x = np.asarray(sources['xcentroid'], dtype=np.float64)
    rad = fwhm * (1.5 + 1.2 * np.clip(np.log10(np.maximum(flux, ref) / ref), 0.0, 3.0))
    for yy, xx, r in zip(y, x, rad):
        r = float(r)
        y0, y1 = int(max(yy - r, 0)), int(min(yy + r + 1, H))
        x0, x1 = int(max(xx - r, 0)), int(min(xx + r + 1, W))
        if y1 <= y0 or x1 <= x0:
            continue
        gy, gx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (gy - yy) ** 2 + (gx - xx) ** 2 <= r * r
    return mask


def detect_in_residual(residual: np.ndarray, fwhm: float, threshold: float,
                       mask: Optional[np.ndarray] = None, max_detections: int = 150
                       ) -> np.ndarray:
    """Candidates in one residual image: (n, 3) array of (x, y, snr).

    The residual is smoothed at the PSF width and divided by its own robust noise
    (MAD), so the threshold is in sigma of the smoothed image."""
    sm = _gaussian_blur(residual.astype(np.float32), max(fwhm / 2.355, 0.8))
    med = float(np.median(sm))
    sig = 1.4826 * float(np.median(np.abs(sm - med)))
    if not np.isfinite(sig) or sig <= 0:
        return np.zeros((0, 3))
    snr = (sm - med) / sig
    peaks = (snr == ndimage.maximum_filter(snr, size=max(int(fwhm * 1.5) | 1, 3))) & (snr > threshold)
    if mask is not None:
        peaks &= ~mask
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return np.zeros((0, 3))
    vals = snr[ys, xs]
    order = np.argsort(vals)[::-1][:max_detections]
    return np.column_stack([xs[order], ys[order], vals[order]]).astype(np.float64)


def collect_detections(frames: np.ndarray, times_min: np.ndarray, ref_lum: np.ndarray,
                       fwhm: float, threshold: float, mask: Optional[np.ndarray]
                       ) -> np.ndarray:
    """All candidates over all frames: (M, 5) of (frame, t_min, x, y, snr)."""
    rows: List[np.ndarray] = []
    for j in range(frames.shape[0]):
        d = _lum(frames[j]) - ref_lum
        det = detect_in_residual(d, fwhm, threshold, mask)
        if len(det):
            rows.append(np.column_stack([np.full(len(det), j), np.full(len(det), times_min[j]),
                                         det]))
    return np.vstack(rows) if rows else np.zeros((0, 5))


def _vote(det: np.ndarray, t_mid: float, vgrid_x: np.ndarray, vgrid_y: np.ndarray,
          shape: Tuple[int, int], cell: float, valid: np.ndarray
          ) -> Tuple[int, float, float, float, float]:
    """Best (count, vx, vy, x0, y0): the velocity whose back-projected detections
    pile up in one cell. ``valid`` is a (nvy, nvx) mask of velocities to skip."""
    H, W = shape
    ncx, ncy = int(W / cell) + 2, int(H / cell) + 2
    tt = det[:, 1] - t_mid
    frame = det[:, 0].astype(np.int64)
    best = (0, 0.0, 0.0, 0.0, 0.0)
    for iy, vy in enumerate(vgrid_y):
        for ix, vx in enumerate(vgrid_x):
            if not valid[iy, ix]:
                continue
            x0 = det[:, 2] - vx * tt
            y0 = det[:, 3] - vy * tt
            cx = np.floor(x0 / cell).astype(np.int64)
            cy = np.floor(y0 / cell).astype(np.int64)
            ok = (cx >= 0) & (cx < ncx) & (cy >= 0) & (cy < ncy)
            if ok.sum() < best[0]:
                continue
            key = cy[ok] * ncx + cx[ok]
            cnt = np.bincount(key)
            k = int(np.argmax(cnt))
            if cnt[k] <= best[0]:
                continue
            # distinct frames, not just points: one noisy frame may add several
            members = ok.copy()
            members[ok] = key == k
            n_frames = len(np.unique(frame[members]))
            if n_frames > best[0]:
                best = (n_frames, float(vx), float(vy),
                        float(np.median(x0[members])), float(np.median(y0[members])))
    return best


def _fit_track(det: np.ndarray, t_mid: float, vx: float, vy: float, x0: float, y0: float,
               tol: float) -> Optional[Dict[str, Any]]:
    """Members within ``tol`` px of the voted line, refit by least squares."""
    tt = det[:, 1] - t_mid
    for _ in range(3):
        dx = det[:, 2] - (x0 + vx * tt)
        dy = det[:, 3] - (y0 + vy * tt)
        member = np.hypot(dx, dy) <= tol
        if member.sum() < 4:
            return None
        A = np.column_stack([np.ones(member.sum()), tt[member]])
        cx, *_ = np.linalg.lstsq(A, det[member, 2], rcond=None)
        cy, *_ = np.linalg.lstsq(A, det[member, 3], rcond=None)
        x0, vx = float(cx[0]), float(cx[1])
        y0, vy = float(cy[0]), float(cy[1])
    dx = det[:, 2] - (x0 + vx * tt)
    dy = det[:, 3] - (y0 + vy * tt)
    member = np.hypot(dx, dy) <= tol
    rms = float(np.sqrt(np.mean(dx[member] ** 2 + dy[member] ** 2)))
    frames = np.unique(det[member, 0])
    return {'vx': vx, 'vy': vy, 'x0': x0, 'y0': y0, 't_mid': float(t_mid),
            'members': member, 'n_frames': int(len(frames)), 'rms_px': rms,
            'mean_snr': float(np.mean(det[member, 4])),
            'speed_px_per_min': float(np.hypot(vx, vy)),
            'direction_deg': float(np.degrees(np.arctan2(vy, vx)))}


def find_tracks(det: np.ndarray, shape: Tuple[int, int], span_min: float, fwhm: float,
                n_frames_total: int, max_speed: Optional[float] = None, max_tracks: int = 10,
                min_frames: Optional[int] = None) -> List[Dict[str, Any]]:
    """Link detections into straight constant-velocity tracks (see module doc)."""
    if len(det) < 4 or span_min <= 0:
        return []
    t_mid = float(np.median(det[:, 1]))
    cell = max(4.0, 1.2 * fwhm)
    v_min = 2.0 * fwhm / span_min                    # slower than this = not distinguishable
    # Default ceiling 3 px/min (main-belt asteroids move a fraction of that at
    # typical plate scales); raise it with ``max_speed`` for fast movers. Never
    # beyond half a frame over the whole session.
    v_max = max_speed if max_speed else 3.0
    v_max = min(v_max, min(shape) / (2.0 * span_min))
    v_max = max(v_max, 2.0 * v_min)
    dv = max(0.5 * cell / (span_min / 2.0), v_min / 6.0)
    grid = np.arange(-v_max, v_max + dv / 2, dv)
    vx_g = grid
    vy_g = grid
    valid = (vx_g[None, :] ** 2 + vy_g[:, None] ** 2) >= v_min ** 2
    need = min_frames or max(6, int(0.25 * n_frames_total))
    pool = det.copy()
    tracks: List[Dict[str, Any]] = []
    for _ in range(max_tracks):
        if len(pool) < need:
            break
        n, vx, vy, x0, y0 = _vote(pool, t_mid, vx_g, vy_g, shape, cell, valid)
        if n < need:
            break
        trk = _fit_track(pool, t_mid, vx, vy, x0, y0, tol=max(2.0, 0.6 * fwhm))
        if trk is None or trk['n_frames'] < need or trk['speed_px_per_min'] < v_min:
            # this peak was not a real line: drop its detections and go on
            break
        trk['detections'] = pool[trk['members']].copy()
        tracks.append(trk)
        pool = pool[~trk['members']]
    return tracks


def tracked_stack(frames: np.ndarray, times_min: np.ndarray, track: Dict[str, Any],
                  half: int = 40) -> Optional[np.ndarray]:
    """Median of each frame's window centred on the moving object.

    A ``2*half+1`` square is cut around the object's predicted position in every
    frame (bilinear sub-pixel shift), so the object is fixed and the stars smear
    into trails that a median rejects. Returns (2h+1, 2h+1, C) or None."""
    H, W = frames.shape[1:3]
    wins = []
    for j in range(frames.shape[0]):
        tt = times_min[j] - track['t_mid']
        cx = track['x0'] + track['vx'] * tt
        cy = track['y0'] + track['vy'] * tt
        if not (half + 2 <= cx < W - half - 2 and half + 2 <= cy < H - half - 2):
            continue
        ix, iy = int(round(cx)), int(round(cy))
        sub = np.asarray(frames[j, iy - half - 2:iy + half + 3, ix - half - 2:ix + half + 3],
                         dtype=np.float32)
        fy, fx = cy - iy, cx - ix
        shifted = ndimage.shift(sub, (-fy, -fx, 0), order=1, mode='nearest')
        wins.append(shifted[2:-2, 2:-2])
    if len(wins) < 3:
        return None
    return np.median(np.stack(wins), axis=0).astype(np.float32)


def find_moving_objects(frames: np.ndarray, times_min: np.ndarray, reference: np.ndarray,
                        fwhm: float = 4.0, threshold: float = 5.0, sources=None,
                        max_speed: Optional[float] = None) -> Dict[str, Any]:
    """Detect and link moving objects in an (N, H, W, C) aligned stack.

    ``reference`` is the combined (rejection) stack, (H, W, C). Returns
    {'tracks': [...], 'detections': (M, 5), 'n_frames': N, 'span_min': ...}."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required")
    ref_lum = _lum(reference)
    if sources is None:
        try:
            from src.quality import detect_stars_auto
            med = float(np.median(ref_lum))
            sd = 1.4826 * float(np.median(np.abs(ref_lum - med)))
            sources = detect_stars_auto(ref_lum, max(sd, 1e-3), background=med)
        except Exception:
            sources = None
    mask = star_mask(ref_lum, sources, fwhm)
    times_min = np.asarray(times_min, dtype=float)
    det = collect_detections(frames, times_min, ref_lum, fwhm, threshold, mask)
    span = float(times_min.max() - times_min.min())
    tracks = find_tracks(det, ref_lum.shape, span, fwhm, frames.shape[0], max_speed=max_speed)
    return {'tracks': tracks, 'detections': det, 'n_frames': int(frames.shape[0]),
            'span_min': span, 'masked_fraction': float(mask.mean())}


_TRACK_FIELDS = ['track', 'n_frames', 'x_mid', 'y_mid', 'vx_px_per_min', 'vy_px_per_min',
                 'speed_px_per_min', 'direction_deg', 'rms_px', 'mean_snr']


def write_tracks_csv(result: Dict[str, Any], path: str) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(_TRACK_FIELDS)
        for k, t in enumerate(result['tracks'], 1):
            w.writerow([k, t['n_frames'], round(t['x0'], 2), round(t['y0'], 2),
                        round(t['vx'], 4), round(t['vy'], 4), round(t['speed_px_per_min'], 4),
                        round(t['direction_deg'], 1), round(t['rms_px'], 3),
                        round(t['mean_snr'], 2)])


def format_summary(result: Dict[str, Any]) -> str:
    tr = result['tracks']
    head = (f"  Moving objects: {len(result['detections'])} candidate detections in "
            f"{result['n_frames']} frames over {result['span_min']:.0f} min -> "
            f"{len(tr)} linked track(s)")
    lines = [head]
    for k, t in enumerate(tr, 1):
        lines.append(f"    #{k}: {t['n_frames']} frames, {t['speed_px_per_min']:.3f} px/min "
                     f"toward {t['direction_deg']:+.0f} deg, mean SNR {t['mean_snr']:.1f}, "
                     f"line rms {t['rms_px']:.2f} px, near ({t['x0']:.0f}, {t['y0']:.0f}) at mid-session")
    return "\n".join(lines)


def write_outputs(result: Dict[str, Any], frames: np.ndarray, times_min: np.ndarray,
                  base: str, stack_tracks: bool = False) -> List[str]:
    """CSV of tracks (+ optional per-track tracked-stack FITS). Returns paths."""
    paths: List[str] = []
    if not result['tracks']:
        return paths
    csv_path = base + '_moving_objects.csv'
    write_tracks_csv(result, csv_path)
    paths.append(csv_path)
    if stack_tracks:
        from astropy.io import fits
        for k, t in enumerate(result['tracks'][:3], 1):
            img = tracked_stack(frames, times_min, t)
            if img is None:
                continue
            hdr = fits.Header()
            hdr['COMMENT'] = 'median stack along a moving object\'s track (object fixed at centre)'
            hdr['MOVSPEED'] = (t['speed_px_per_min'], 'px/min')
            hdr['MOVDIR'] = (t['direction_deg'], 'deg')
            hdr['NFRAMES'] = t['n_frames']
            p = f"{base}_moving_{k}.fits"
            fits.PrimaryHDU(np.transpose(img, (2, 0, 1)), hdr).writeto(p, overwrite=True)
            paths.append(p)
    return paths


def frame_times_min(final: Sequence[Any]) -> np.ndarray:
    """Minutes since the first frame from DATE-OBS (frame index if unparseable)."""
    from src.utils import parse_timestamp
    t = []
    for j, f in enumerate(final):
        v = (getattr(f, 'header', None) or {}).get('DATE-OBS')
        dt = None
        if v:
            try:
                dt = parse_timestamp(str(v))
            except Exception:
                dt = None
        t.append(dt.timestamp() / 60.0 if dt is not None else float(j))
    t = np.asarray(t, dtype=float)
    return t - t.min() if len(t) else t


def output_base(output_path: Optional[str]) -> Optional[str]:
    return os.path.splitext(output_path)[0] if output_path else None
