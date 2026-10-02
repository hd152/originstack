"""Per-channel shot-noise model measured from the lights themselves.

The debayered frames Phase 1 leaves in ``mem_rgb`` keep every pixel's own-channel
raw sample (RCD and Malvar write it back unchanged), so same-colour pixels two
apart in a row of the mosaic are two samples of nearly the same sky: their
difference is noise, with stars, nebulae and gradients cancelled to first order
(the median-based spread ignores the few pairs that straddle a star). In each
frame the pairs are binned by signal level; the variance-per-pixel against signal
is a straight line whose **slope** k_c is the Poisson conversion of that channel in
the image's own units (variance in ADU^2 per ADU of signal = 1 / effective gain),
white balance and flat normalisation included. The slope, not var/signal, so the
constant terms -- read noise, the shot noise of the subtracted dark current --
drop out.

Why it exists: the Celestron Origin's FITS ``EGAIN`` is 4.6-5x too high against
this measurement (three sessions, two ISO settings; frame-to-frame scatter 0.4%),
and the stack's header has no gain at all, so photometry either left the Poisson
term out or used one ~2x too small in sigma. It is also, in effect, a
photon-transfer curve taken from the lights, with no calibration frames.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

_log = logging.getLogger("originstack")

_CH = {'R': 0, 'G': 1, 'B': 2}
_BIN_PCTS = (5, 15, 25, 35, 45, 60)      # signal bins: the sky end, where structure is rare


def _channel_sites(pattern: str, c: int):
    """(dy, dx) offsets of channel ``c`` in the 2x2 Bayer cell."""
    pat = pattern.strip().upper()
    return [divmod(k, 2) for k, ch in enumerate(pat) if _CH.get(ch) == c]


def pair_samples(frame: np.ndarray, pattern: str, c: int):
    """(difference, mean) of horizontally adjacent same-colour samples (2 px apart) of
    channel ``c`` in a debayered frame that keeps its raw samples."""
    ds, ss = [], []
    for dy, dx in _channel_sites(pattern, c):
        plane = np.asarray(frame[dy::2, dx::2, c], np.float64)
        a, b = plane[:, :-1], plane[:, 1:]
        ds.append((a - b).ravel())
        ss.append((0.5 * (a + b)).ravel())
    return np.concatenate(ds), np.concatenate(ss)


def _robust_pair_var(d: np.ndarray) -> float:
    """Per-pixel variance from pair differences (MAD-based; pairs on a star are outliers)."""
    return float((1.4826 * np.median(np.abs(d - np.median(d)))) ** 2 / 2.0)


def frame_bins(frame: np.ndarray, pattern: str, border: int = 8):
    """Per channel: [(median signal, pair variance), ...] over the sky-end signal bins."""
    f = np.asarray(frame)[border:-border, border:-border]
    out = []
    for c in range(3):
        d, s = pair_samples(f, pattern, c)
        ok = np.isfinite(d) & np.isfinite(s)
        d, s = d[ok], s[ok]
        if s.size < 5000:
            out.append([])
            continue
        q = np.percentile(s, _BIN_PCTS)
        rows = []
        for lo, hi in zip(q[:-1], q[1:]):
            sel = (s >= lo) & (s < hi)
            if sel.sum() >= 1000:
                rows.append((float(np.median(s[sel])), _robust_pair_var(d[sel])))
        out.append(rows)
    return out


def _theil_sen(x: np.ndarray, y: np.ndarray):
    """Median of pairwise slopes and the matching intercept (outlier-robust line)."""
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-9 * max(1.0, float(np.abs(x).max()))
    if ok.sum() < 3:
        return float('nan'), float('nan')
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return slope, float(np.median(y - slope * x))


def measure_noise_model(mem_rgb, frame_indices: Sequence[int], pattern: str,
                        max_frames: int = 16) -> Optional[dict]:
    """Per-channel Poisson slope ``k`` (ADU^2 per ADU) and intercept from a spread of
    frames. Returns None when the pattern is unknown or too little data fits."""
    if not pattern or len(pattern.strip()) != 4:
        return None
    idx = list(frame_indices)
    if not idx:
        return None
    pick = idx[::max(1, len(idx) // max_frames)][:max_frames]
    pts = [[], [], []]
    per_frame = [[], [], []]
    for i in pick:
        fr = mem_rgb[i]
        if np.asarray(fr).ndim != 3 or np.asarray(fr).shape[2] != 3:
            return None
        for c, rows in enumerate(frame_bins(fr, pattern)):
            pts[c].extend(rows)
            if len(rows) >= 3:
                xs, ys = np.array(rows).T
                per_frame[c].append(_theil_sen(xs, ys)[0])
    k, a, spread = [], [], []
    for c in range(3):
        if len(pts[c]) < 6:
            return None
        xs, ys = np.array(pts[c]).T
        slope, icept = _theil_sen(xs, ys)
        if not (np.isfinite(slope) and slope > 0):
            return None
        k.append(slope)
        a.append(icept)
        pf = np.array([v for v in per_frame[c] if np.isfinite(v)])
        spread.append(float(np.std(pf) / np.mean(pf)) if pf.size >= 3 else float('nan'))
    return {'k': np.array(k), 'intercept': np.array(a), 'frames': len(pick),
            'frame_spread': np.array(spread)}


def temporal_noise(a: np.ndarray, b: np.ndarray, blocks=(8, 12, 16, 24)):
    """From two aligned frames ``a``, ``b`` (H, W, 3): the per-channel per-pixel temporal
    noise of one frame, and {block: R} with R = var(block sum) / (block^2 var(pixel)) of
    their difference -- the correlated-noise factor an aperture of that area needs.
    Debayering and the registration warp correlate neighbouring pixels, so a sum over
    an aperture is noisier than area x per-pixel variance (R ~3 in G, ~5 in R/B on real
    Origin data). Differencing two frames cancels static structure (faint sources,
    background ripples), which a single image's block sums would mistake for noise;
    MAD-based throughout, so stars that changed between the frames do not count. A block
    whose R is not finite in every channel is left out."""
    def sd(x):
        x = x[np.isfinite(x)]
        return 1.4826 * float(np.median(np.abs(x - np.median(x)))) if x.size > 100 else float("nan")
    st = np.full(3, np.nan)
    rs = {}
    d_all = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    for c in range(3):
        d = d_all[..., c] - np.median(d_all[::4, ::4, c])
        st[c] = sd(d[::2, ::2]) / np.sqrt(2.0)
    for blk in blocks:
        h, w = (d_all.shape[0] // blk) * blk, (d_all.shape[1] // blk) * blk
        if h < 4 * blk or w < 4 * blk:
            continue
        r = np.full(3, np.nan)
        for c in range(3):
            d = d_all[:h, :w, c] - np.median(d_all[::4, ::4, c])
            sp = st[c] * np.sqrt(2.0)
            sb = sd(d.reshape(h // blk, blk, w // blk, blk).sum((1, 3)))
            if sp > 0 and np.isfinite(sb):
                r[c] = sb ** 2 / (blk * blk * sp ** 2)
        if np.isfinite(r).all():
            rs[blk] = r
    return st, rs


def corr_factor_at(rs: dict, ap_radius: float):
    """R for an aperture of radius ``ap_radius`` (block of equal area), interpolated in log
    block size over the measured blocks; None when nothing was measured."""
    if not rs:
        return None
    b = np.sqrt(np.pi) * float(ap_radius)
    ks = sorted(rs)
    if len(ks) == 1:
        return np.asarray(rs[ks[0]], float)
    x = np.log(np.array(ks, float))
    t = np.clip(np.log(max(b, 1.0)), x[0], x[-1])
    return np.array([np.interp(t, x, [rs[k][c] for k in ks]) for c in range(3)])


def format_noise_model(nm: dict) -> str:
    k = nm['k']
    return ("  Noise model (same-colour pixel pairs, {n} frames): effective gain "
            "R {r:.4f}  G {g:.4f}  B {b:.4f} e-/ADU of this image").format(
                n=nm['frames'], r=1 / k[0], g=1 / k[1], b=1 / k[2])
