"""CFA-aware drizzle (``--cfa-drizzle``): combine the *measured* Bayer samples.

A normal OSC stack debayers every sub first, so two thirds of every frame's
colour data (all of R and B's, half of G's) is interpolation, and the stack
then averages those guesses. Drizzling the raw samples directly (Fruchter &
Hook 2002, applied per colour plane as in the HST/Siril "Bayer drizzle"
workflow) never interpolates across the mosaic: each output pixel of each
channel is a weighted mean of samples that channel really measured, and the
gaps in one frame's lattice are filled by the *other* frames, whose sub-pixel
offsets (dither, and the field rotation of an alt-az mount) put their samples
in between.

No Phase 1 change is needed. The debayered frames already in ``mem_rgb``
hold the raw sample untouched at the position where each channel was
measured (Malvar-He-Cutler, the default, copies "its own channel"; bilinear
does too), so the measured lattice is recoverable from the Bayer pattern
alone. Later per-frame steps (white balance, vignette map) are smooth
multiplicative maps applied to samples and interpolation alike; the ~0.2 px
chromatic-aberration shift blends a sample with its neighbours by at most
that fraction, negligible against a ~5 px FWHM.

Each sample is splatted onto the output grid as a square drop of side
``pixfrac * scale`` output pixels (exact area overlap, axis-aligned -- a
drop's rotation is ignored, as it is small at these drop sizes), through the
same affine ``_drizzle_matrix`` the rest of the drizzle path uses.

Rejection is per sample against a reference stack (the normal stack made
moments earlier): a sample is dropped when it differs from the reference by
more than ``reject_sigma`` times the quadrature sum of the frame's own noise
(measured from its residuals, robustly) and 15% of the local signal (seeing
and transparency change star peaks by that much between frames, and a hard
sigma clip would otherwise bias every star low). Frames are weighted by
their measured inverse noise variance.

Where the lattice is still too sparse after all frames (frame edges, small
stacks), the output blends smoothly back to the reference in proportion to
how far the coverage falls short of what a full lattice would give.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.utils import safe_print

try:
    import astro_native as _native
    _HAS_NATIVE = hasattr(_native, 'cfa_drizzle_frame')
except Exception:  # pragma: no cover
    _native = None
    _HAS_NATIVE = False

_CH = {'R': 0, 'G': 1, 'B': 2}
_DENSITY = (0.25, 0.5, 0.25)        # fraction of sensor pixels measuring R, G, B
_SIGNAL_TOLERANCE = 0.15
_COVER_LOW, _COVER_HIGH = 0.35, 0.70   # fraction of full-lattice coverage: blend ramp
_SIGMA_STRIDE = 7    # per-frame noise is measured on every 7th lattice site


def cfa_lattice(pattern: str, H: int, W: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per channel (R, G, B): the (iy, ix) integer sensor positions it measured."""
    pat = pattern.strip().upper()
    if len(pat) != 4 or any(ch not in _CH for ch in pat):
        raise ValueError(f"unusable Bayer pattern {pattern!r}")
    out = []
    for c in range(3):
        ys, xs = [], []
        for k, letter in enumerate(pat):
            if _CH[letter] == c:
                py, px = divmod(k, 2)
                yy, xx = np.meshgrid(np.arange(py, H, 2), np.arange(px, W, 2), indexing='ij')
                ys.append(yy.ravel())
                xs.append(xx.ravel())
        out.append((np.concatenate(ys).astype(np.int32), np.concatenate(xs).astype(np.int32)))
    return out


def _splat_channel(num, den, cov, c, oy, ox, val, frame_w, h, out_h, out_w) -> None:
    """Deposit square drops (half-side ``h``) centred at (oy, ox) into channel
    ``c`` of the (out_h, out_w, 3) accumulators: weighted value, weight, and
    unweighted coverage. numpy reference for ``cfa_drizzle_frame``."""
    k = int(np.floor(2.0 * h)) + 2
    y0 = np.floor(oy - h + 0.5).astype(np.int64)
    x0 = np.floor(ox - h + 0.5).astype(np.int64)
    norm = 1.0 / (4.0 * h * h)
    n = out_h * out_w
    nf, df, cf = num.reshape(-1, 3), den.reshape(-1, 3), cov.reshape(-1, 3)
    for a in range(k):
        qy = y0 + a
        ovy = np.minimum(oy + h, qy + 0.5) - np.maximum(oy - h, qy - 0.5)
        oky = (ovy > 0) & (qy >= 0) & (qy < out_h)
        for b in range(k):
            qx = x0 + b
            ovx = np.minimum(ox + h, qx + 0.5) - np.maximum(ox - h, qx - 0.5)
            ok = oky & (ovx > 0) & (qx >= 0) & (qx < out_w)
            if not ok.any():
                continue
            wgt = (ovy[ok] * ovx[ok]) * norm
            idx = qy[ok] * out_w + qx[ok]
            nf[:, c] += np.bincount(idx, weights=wgt * val[ok] * frame_w, minlength=n)
            df[:, c] += np.bincount(idx, weights=wgt * frame_w, minlength=n)
            cf[:, c] += np.bincount(idx, weights=wgt, minlength=n)


def _bilinear(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinear lookup of ``img`` at fractional (y, x), edge-clamped. Nearest
    pixel is not good enough for the rejection model: across a star's flank
    the reference changes by thousands of ADU per pixel, and half a pixel of
    lookup error alone would reject genuine samples there."""
    h, w = img.shape
    yc = np.clip(y, 0.0, h - 1.0)
    xc = np.clip(x, 0.0, w - 1.0)
    y0 = np.minimum(np.floor(yc).astype(np.int64), h - 2)
    x0 = np.minimum(np.floor(xc).astype(np.int64), w - 2)
    fy = yc - y0
    fx = xc - x0
    return ((1 - fy) * ((1 - fx) * img[y0, x0] + fx * img[y0, x0 + 1])
            + fy * ((1 - fx) * img[y0 + 1, x0] + fx * img[y0 + 1, x0 + 1]))


def _map_sites(iy, ix, Minv, off):
    dy = iy.astype(np.float64) - off[0]
    dx = ix.astype(np.float64) - off[1]
    return (Minv[0, 0] * dy + Minv[0, 1] * dx, Minv[1, 0] * dy + Minv[1, 1] * dx)


def _inside(oy, ox, h, out_h, out_w):
    return ((oy > -h - 0.5) & (oy < out_h + h - 0.5)
            & (ox > -h - 0.5) & (ox < out_w + h - 0.5))


def _frame_sigmas(rgb, reference, lattice, Minv, off, h) -> np.ndarray:
    """Per-channel robust noise of one frame against the reference, measured
    on every ``_SIGMA_STRIDE``-th lattice site (a MAD needs a few 10^4
    samples, not 10^6). Shared by the numpy and native paths so both use the
    identical sigma -- NaN for a channel with no usable samples."""
    out_h, out_w = reference.shape[:2]
    sig = np.full(3, np.nan)
    for c in range(3):
        iy, ix = lattice[c]
        iy, ix = iy[::_SIGMA_STRIDE], ix[::_SIGMA_STRIDE]
        oy, ox = _map_sites(iy, ix, Minv, off)
        ok = _inside(oy, ox, h, out_h, out_w)
        if ok.sum() < 100:
            continue
        val = rgb[iy[ok], ix[ok], c].astype(np.float64)
        resid = val - _bilinear(reference[:, :, c], oy[ok], ox[ok])
        s = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        if np.isfinite(s) and s > 0:
            sig[c] = s
    return sig


def _frame_splat_numpy(rgb, reference, lattice, Minv, off, h, sig, sky,
                       reject_sigma, num, den, cov) -> Tuple[int, int]:
    """numpy reference for ``astro_native.cfa_drizzle_frame``."""
    out_h, out_w = reference.shape[:2]
    n_samples = n_rejected = 0
    for c in range(3):
        if not (np.isfinite(sig[c]) and sig[c] > 0):
            continue
        iy, ix = lattice[c]
        oy, ox = _map_sites(iy, ix, Minv, off)
        inside = _inside(oy, ox, h, out_h, out_w)
        if not inside.any():
            continue
        oy, ox = oy[inside], ox[inside]
        val = rgb[iy[inside], ix[inside], c].astype(np.float64)
        ref_at = _bilinear(reference[:, :, c], oy, ox)
        resid = val - ref_at
        tol = reject_sigma * np.sqrt(sig[c] ** 2 + (_SIGNAL_TOLERANCE
                                                    * np.maximum(ref_at - sky[c], 0.0)) ** 2)
        keep = np.abs(resid) <= tol
        n_samples += int(val.size)
        n_rejected += int(val.size - keep.sum())
        _splat_channel(num, den, cov, c, oy[keep], ox[keep], val[keep],
                       1.0 / sig[c] ** 2, h, out_h, out_w)
    return n_samples, n_rejected


def _chan_table(pattern: str) -> np.ndarray:
    """Channel index measured at sensor parity (y&1)*2 + (x&1)."""
    pat = pattern.strip().upper()
    return np.array([_CH[ch] for ch in pat], dtype=np.uint8)


def cfa_drizzle_combine(mem_rgb: np.ndarray, final_indices: Sequence[int],
                        shifts: Sequence[Optional[Tuple[float, float]]],
                        transforms: Sequence[Optional[Any]],
                        reference: np.ndarray, pattern: str,
                        top: float, left: float, scale: float = 1.0,
                        pixfrac: float = 1.0, reject_sigma: float = 4.0,
                        use_native: Optional[bool] = None,
                        ) -> Tuple[np.ndarray, dict]:
    """Combine every frame's measured Bayer samples onto ``reference``'s grid.

    ``reference`` is the normal (debayered) stack, (out_h, out_w, 3) float32:
    the rejection model and the fallback where the lattice is too sparse.
    ``use_native`` forces (True) or forbids (False) the Rust kernel; default
    is native when built. Returns (cfa_stack float32 same shape, stats dict).
    """
    from src.stacking import _drizzle_matrix

    if use_native is None:
        use_native = _HAS_NATIVE
    out_h, out_w = reference.shape[:2]
    n_frames = len(final_indices)
    H, W = mem_rgb.shape[1], mem_rgb.shape[2]
    lattice = cfa_lattice(pattern, H, W)
    ctab = _chan_table(pattern)
    h = max(0.5 * float(pixfrac) * float(scale), 0.05)
    inv_scale = 1.0 / float(scale)
    ref32 = np.ascontiguousarray(reference, dtype=np.float32)

    num = np.zeros((out_h, out_w, 3))
    den = np.zeros((out_h, out_w, 3))
    cov = np.zeros((out_h, out_w, 3))
    sky = np.array([float(np.median(reference[:, :, c])) for c in range(3)])
    n_used = n_rejected = n_samples = 0

    for j in range(n_frames):
        rgb = np.ascontiguousarray(mem_rgb[final_indices[j]], dtype=np.float32)
        M, off = _drizzle_matrix(transforms[j], shifts[j], top, left, inv_scale)
        Minv = np.linalg.inv(M)
        sig = _frame_sigmas(rgb, ref32, lattice, Minv, off, h)
        if use_native:
            ns, nr = _native.cfa_drizzle_frame(
                rgb, ref32, ctab, np.ascontiguousarray(Minv.ravel(), dtype=np.float64),
                np.asarray(off, dtype=np.float64), h, sig, sky,
                float(reject_sigma), _SIGNAL_TOLERANCE, num, den, cov)
        else:
            ns, nr = _frame_splat_numpy(rgb, ref32, lattice, Minv, off, h, sig, sky,
                                        reject_sigma, num, den, cov)
        n_samples += int(ns)
        n_rejected += int(nr)
        n_used += 1

    out = reference.astype(np.float32).copy()
    coverage = np.zeros(3)
    for c in range(3):
        expected = n_used * _DENSITY[c] * inv_scale ** 2
        d = den[:, :, c].ravel()
        ok = d > 0
        cfa = np.where(ok, num[:, :, c].ravel() / np.where(ok, d, 1.0), 0.0)
        frac = cov[:, :, c].ravel() / max(expected, 1e-9)
        a = np.clip((frac - _COVER_LOW) / (_COVER_HIGH - _COVER_LOW), 0.0, 1.0) * ok
        ref_c = reference[:, :, c].ravel().astype(np.float64)
        out[:, :, c] = (a * cfa + (1.0 - a) * ref_c).reshape(out_h, out_w)
        coverage[c] = float(np.mean(a))
    stats = {'frames': n_used, 'samples': n_samples,
             'rejected_frac': n_rejected / max(n_samples, 1),
             'cfa_fraction': coverage.tolist(),
             'native': bool(use_native)}
    return out, stats


def apply_cfa_drizzle(stacked: np.ndarray, mem_rgb: np.ndarray, final_indices,
                      shifts, transforms, top: float, left: float, args,
                      displacement_fields=None) -> np.ndarray:
    """Pipeline hook: returns the CFA-drizzled stack, or ``stacked`` unchanged
    (with a printed reason) when the input is not a Bayer mosaic we can model."""
    pattern = getattr(args, '_session_bayer', None)
    if mem_rgb.shape[-1] != 3 or stacked.ndim != 3 or stacked.shape[2] != 3:
        safe_print("  --cfa-drizzle skipped: needs 3-channel OSC frames")
        return stacked
    if not pattern:
        safe_print("  --cfa-drizzle skipped: no Bayer pattern known for this session "
                   "(pre-debayered or mono input)")
        return stacked
    method = getattr(args, 'debayer_method', 'malvar')
    if method != 'malvar':
        safe_print(f"  --cfa-drizzle skipped: needs --debayer-method malvar (its output keeps "
                   f"each channel's raw sample; '{method}' does not guarantee that)")
        return stacked
    if displacement_fields is not None and any(f is not None for f in displacement_fields):
        safe_print("  --cfa-drizzle skipped: not supported with --elastic-registration")
        return stacked
    try:
        import time
        t0 = time.time()
        scale = float(getattr(args, 'drizzle_scale', 1.0) or 1.0)
        pixfrac = float(getattr(args, 'drizzle_pixfrac', 1.0) or 1.0)
        safe_print(f"\n  CFA drizzle: {len(final_indices)} frames, pattern {pattern.upper()}, "
                   f"scale {scale:g}, pixfrac {pixfrac:g}...")
        out, st = cfa_drizzle_combine(mem_rgb, final_indices, shifts, transforms,
                                      stacked, pattern, top, left,
                                      scale=scale, pixfrac=pixfrac)
        cf = st['cfa_fraction']
        safe_print(f"  ✓ CFA drizzle ({time.time() - t0:.1f}s): "
                   f"{st['rejected_frac'] * 100:.2f}% of samples rejected, "
                   f"measured-sample coverage R {cf[0] * 100:.0f}% "
                   f"G {cf[1] * 100:.0f}% B {cf[2] * 100:.0f}%")
        return out
    except Exception as exc:
        safe_print(f"  ⚠ CFA drizzle failed ({exc}) — keeping the standard stack")
        return stacked
