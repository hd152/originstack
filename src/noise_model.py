"""Photometry's noise model: the Poisson coefficient of the processed frames and the
correlated-noise factor of aperture sums.

**Shot noise.** A source of ``f`` ADU in the processed image has shot-noise variance
``k_c * f``, with ``k_c = s_c / g``: ``g`` the sensor's gain in e- per *raw* ADU and
``s_c`` how far Phase 1 scaled channel ``c`` (flat, white balance, CFA equalisation).
``g`` comes from the camera profile (src/camera_profile.py, checked on this session)
or from the two-point photon-transfer estimate on this session's raw lights with the
master bias as pedestal; ``s_c`` is the median ratio of a processed own-channel
sample to the same raw pixel above the pedestal (RCD and Malvar keep each pixel's
own-channel sample). The Origin's FITS ``EGAIN`` is ~4.5x too high, so the header
is no substitute.

This used to be the *slope* of same-colour pixel-pair variance against signal on the
processed frames. That fails on flat-fielded data: after the flat a uniform sky has
one signal level but more noise in the vignetted corners, so within a frame the
variance no longer follows the signal. Against raw gain x processing scale it read
0.40-1.33x on four real sessions and came out *negative* on two (Crab, M101)
(dev-notes/camera-profile.md).
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


def processing_scale(processed: np.ndarray, raw: np.ndarray, pattern: str, pedestal: float,
                     min_signal: float = 500.0, border: int = 100) -> Optional[np.ndarray]:
    """Per channel: median of processed own-channel sample / (raw - pedestal), over
    pixels at least ``min_signal`` above the pedestal. None when the shapes differ or a
    channel has too few such pixels."""
    processed = np.asarray(processed)
    raw = np.asarray(raw)
    if processed.ndim != 3 or raw.ndim != 2 or processed.shape[:2] != raw.shape:
        return None
    border = min(border, raw.shape[0] // 16, raw.shape[1] // 16)    # per Bayer plane
    out = np.full(3, np.nan)
    for c in range(3):
        rs = []
        for dy, dx in _channel_sites(pattern, c):
            a = raw[dy::2, dx::2][border:-border or None:3, border:-border or None:3].astype(np.float64) - pedestal
            b = processed[dy::2, dx::2, c][border:-border or None:3, border:-border or None:3].astype(np.float64)
            ok = np.isfinite(a) & np.isfinite(b) & (a > min_signal)
            if ok.sum() >= 1000:
                rs.append(b[ok] / a[ok])
        if rs:
            out[c] = float(np.median(np.concatenate(rs)))
    return out if np.all(np.isfinite(out) & (out > 0)) else None


def measure_noise_model(mem_rgb, frame_indices: Sequence[int], raw_paths: Sequence[str],
                        pattern: str, raw_gain: float, pedestal: float,
                        max_frames: int = 6) -> Optional[dict]:
    """Per-channel Poisson coefficient ``k`` (ADU^2 per ADU of processed signal) from a
    raw gain (e- per raw ADU) and the processing scale measured on up to ``max_frames``
    frames spread through the session. ``raw_paths[j]`` is the raw light behind
    ``mem_rgb[frame_indices[j]]``. None when the pattern, the gain or the scale is
    unusable."""
    from src.io_fits import load_frame
    if not pattern or len(pattern.strip()) != 4 or not raw_gain or raw_gain <= 0:
        return None
    idx = list(frame_indices)
    if not idx or len(raw_paths) != len(idx):
        return None
    step = max(1, len(idx) // max_frames)
    scales = []
    for j in range(0, len(idx), step)[:max_frames]:
        try:
            raw = load_frame(raw_paths[j])[0]
        except Exception:
            continue
        sc = processing_scale(mem_rgb[idx[j]], raw, pattern, pedestal)
        if sc is not None:
            scales.append(sc)
    if not scales:
        return None
    scale = np.median(np.array(scales), axis=0)
    return {'k': scale / float(raw_gain), 'scale': scale, 'gain': float(raw_gain),
            'frames': len(scales)}


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


def format_noise_model(nm: dict, source: str = "") -> str:
    k = nm['k']
    return ("  Noise model ({src}raw gain {g:.4f} e-/ADU, processing scale from {n} frames): "
            "effective gain R {r:.4f}  G {gg:.4f}  B {b:.4f} e-/ADU of this image").format(
                src=f"{source}, " if source else "", g=nm['gain'], n=nm['frames'],
                r=1 / k[0], gg=1 / k[1], b=1 / k[2])
