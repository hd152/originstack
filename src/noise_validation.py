"""Split-half noise measurement and structure confirmation (``--noise-validate``).

Stack the odd-numbered and even-numbered frames separately. The two halves see
the same sky and independent noise, so

  * their difference contains **only noise**: ``var(A-B) = sigma1^2 (1/nA + 1/nB)``
    while the full stack's noise is ``sigma1^2 / N``, which gives the stack's
    *measured* per-pixel noise (per channel, and locally) with no model of the
    sensor and no assumption that the noise is white -- it captures whatever
    resampling, debayering and rejection did to it; and
  * their *agreement* says which structure is real: where a smoothed patch of A
    and the same patch of B are correlated, the signal is repeatable; where they
    are not, it is noise. That local correlation is a confidence map for faint
    structure -- the honest check on anything a denoiser or a contrast step
    brings out of a low-SNR stack.

The halves are plain means (no rejection or weights): the purpose is measuring
the noise floor and repeatability, not producing a science stack.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage

    from src.background import _gaussian_blur
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    ndimage = None
    _gaussian_blur = None
    _HAS_SCIPY = False


def half_means(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """(mean of even-indexed frames, mean of odd-indexed frames, nA, nB) from an
    (N, H, W, C) array or memmap, reading one frame at a time."""
    n = frames.shape[0]
    a = np.zeros(frames.shape[1:], dtype=np.float64)
    b = np.zeros(frames.shape[1:], dtype=np.float64)
    na = nb = 0
    for j in range(n):
        fr = np.asarray(frames[j], dtype=np.float64)
        if j % 2 == 0:
            a += fr
            na += 1
        else:
            b += fr
            nb += 1
    if na == 0 or nb == 0:
        raise ValueError("need at least two frames")
    return (a / na).astype(np.float32), (b / nb).astype(np.float32), na, nb


def _robust_sigma_blocks(d: np.ndarray, block: int) -> np.ndarray:
    """MAD-based sigma of ``d`` in ``block`` x ``block`` cells, bilinearly
    upsampled to d's shape (edges kept). Robust so stars' residual structure and
    a few bad frames do not inflate it."""
    H, W = d.shape
    gy, gx = max(H // block, 1), max(W // block, 1)
    sig = np.empty((gy, gx), dtype=np.float64)
    ys = np.linspace(0, H, gy + 1).astype(int)
    xs = np.linspace(0, W, gx + 1).astype(int)
    for i in range(gy):
        for j in range(gx):
            cell = d[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            sig[i, j] = 1.4826 * float(np.median(np.abs(cell - np.median(cell))))
    if gy == 1 and gx == 1:
        return np.full(d.shape, sig[0, 0], dtype=np.float32)
    zy, zx = H / gy, W / gx
    return ndimage.zoom(sig, (zy, zx), order=1, mode='nearest')[:H, :W].astype(np.float32)


def noise_map(a: np.ndarray, b: np.ndarray, na: int, nb: int, block: int = 64
              ) -> np.ndarray:
    """Per-pixel noise sigma of the FULL (N = nA + nB frame) mean stack, (H, W, C),
    from the difference of the halves."""
    n = na + nb
    scale = 1.0 / np.sqrt(n * (1.0 / na + 1.0 / nb))          # sigma_full = sigma_D * scale
    out = np.empty(a.shape, dtype=np.float32)
    for c in range(a.shape[2]):
        out[:, :, c] = _robust_sigma_blocks((a[:, :, c] - b[:, :, c]).astype(np.float64),
                                            block) * scale
    return out


def consistency_map(a: np.ndarray, b: np.ndarray, smooth: float = 2.0, window: int = 17
                    ) -> np.ndarray:
    """Local Pearson correlation between the halves' luminance, (H, W) in [0, 1].

    Both are smoothed (``smooth`` px) to trade resolution for signal-to-noise,
    then correlated over a ``window`` box after removing each half's local mean.
    Negative correlation carries no information and is clipped to 0."""
    def lum(x):
        return 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]

    la = _gaussian_blur(lum(a).astype(np.float64), smooth)
    lb = _gaussian_blur(lum(b).astype(np.float64), smooth)
    m = lambda x: ndimage.uniform_filter(x, window, mode='nearest')
    ma, mb = m(la), m(lb)
    cov = m(la * lb) - ma * mb
    va = np.maximum(m(la * la) - ma * ma, 1e-12)
    vb = np.maximum(m(lb * lb) - mb * mb, 1e-12)
    return np.clip(cov / np.sqrt(va * vb), 0.0, 1.0).astype(np.float32)


def validate_noise(frames: np.ndarray, frame_noise: Optional[float] = None,
                   block: int = 64) -> Dict[str, Any]:
    """Run the split-half analysis on an (N, H, W, C) stack of aligned frames.

    Returns {'sigma': (H,W,C) noise map of the full stack, 'consistency': (H,W),
    'median_sigma': per-channel median noise, 'n': N, and -- when
    ``frame_noise`` (the median per-frame noise from Phase 1) is given --
    'correlation_factor': measured full-stack noise over ``frame_noise/sqrt(N)``}.
    A factor near 1 means the stack's noise is what the per-frame noise predicts
    at 1/sqrt(N); well below 1 means resampling has correlated neighbouring
    pixels (smoother noise, more of it per real resolution element); above 1
    means the stack is noisier than the per-frame estimate predicts -- typically
    because that estimate is luminance-based and the channels differ, or because
    frames were weighted/rejected unevenly. A gradient or fixed pattern common
    to every frame cancels in A-B and can never raise this number."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required")
    a, b, na, nb = half_means(frames)
    sigma = noise_map(a, b, na, nb, block=block)
    cons = consistency_map(a, b)
    med = [float(np.median(sigma[:, :, c])) for c in range(sigma.shape[2])]
    out = {'sigma': sigma, 'consistency': cons, 'median_sigma': med, 'n': na + nb,
           'half_sizes': (na, nb)}
    if frame_noise and frame_noise > 0:
        expected = frame_noise / np.sqrt(na + nb)
        out['correlation_factor'] = float(np.mean(med) / expected)
        out['expected_sigma'] = float(expected)
    return out


def format_summary(r: Dict[str, Any]) -> str:
    med = r['median_sigma']
    parts = [f"  Noise validation (odd/even halves of {r['n']} frames): stack noise "
             f"R {med[0]:.1f}  G {med[1]:.1f}  B {med[2]:.1f} ADU (measured, per pixel)"]
    if 'correlation_factor' in r:
        f = r['correlation_factor']
        note = ("independent-pixel noise" if 0.85 <= f <= 1.15 else
                "smoother than independent (resampling correlated it)" if f < 0.85 else
                "noisier than the per-frame estimate predicts")
        parts.append(f"    vs per-frame noise / sqrt(N) = {r['expected_sigma']:.1f}: "
                     f"factor {f:.2f} -> {note}")
    c = r['consistency']
    parts.append(f"    structure repeatable between halves over {100 * float((c > 0.5).mean()):.1f}% "
                 f"of the frame (local correlation > 0.5)")
    return "\n".join(parts)
