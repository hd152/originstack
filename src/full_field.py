"""Full-field stacking (``--full-field``): keep the area outside the common crop.

Phase 3 crops the stack to the rectangle *every* accepted frame covers. On an
alt-az mount the field rotates through a session, so that rectangle shrinks
with the rotation: 1497x2705 of 2048x3056 (65% of the pixels) on a real
148-frame Fireworks Galaxy session. Everything outside it was observed by
most frames, just not all of them.

This extends the normal stack, after Phase 3 has built it, to the largest
rectangle in which every pixel is covered by at least ``frac`` of the frames:

  * The core (the normal common crop) is the normal stack, untouched -- proper
    coadd, patch weighting and every rejection method stay as they were.
  * Outside it each pixel combines only the frames that cover it. When the core
    is a proper coadd (``--proper-coadd``, the default) the outside is one too:
    the same per-frame PSF / transparency / noise model (measured here on star
    stamps and sky crops warped from each frame, ``measure_frames``), the same
    per-frequency weights, written as a *normalised convolution* so that partial
    coverage works --

        out = IFFT(sum_j w_j G_j FFT(m_j N_j)) / IFFT(sum_j w_j G_j FFT(m_j))

    with N_j = (M_j - sky_j) / F_j, w_j = F_j^2 / s_j^2, m_j the frame's coverage
    mask and G_j = conj(P_j) / sqrt(sum_k w_k |P_k|^2 / sum_k w_k). With every
    frame covering a pixel this is exactly proper coadd's ``R / sqrt(sum w)``.
    A plain clipped mean there was ~5-7% softer than the core and, with the
    gain fit below as it was, half as noisy in R and B: a visible texture step
    at the old crop edge after the stretch (see ``_coadd_window``).
    Otherwise (no proper coadd, or the PSF could not be measured) it is a
    weighted sigma-clipped mean of the covering frames (``sigma_clip_combine``'s
    native kernel treats NaN samples as rejected from the start).
  * The outer combine is matched to the core per channel (gain + offset, fitted
    on a band inside the core where both exist, on smoothed pixels) and
    feathered in over ``FEATHER_PX``.

Depth is no longer uniform: an edge pixel covered by half the frames is ~1.4x
noisier. ``<output>_coverage.fits`` records the per-pixel frame count.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np

from src.utils import format_time, get_logger, safe_print

_log = get_logger()

# Source pixels this close to a frame's edge are treated as uncovered: the
# Lanczos-3 warp reaches 3 px and reads zeros beyond the edge.
EDGE_MARGIN_PX = 4
# Width of the band inside the core where the outer combine is blended in, and
# over which its gain/offset against the core is fitted.
FEATHER_PX = 32
# Coverage-grid spacing for choosing the rectangle.
GRID_STEP = 8
# Output rows per combine band (memory: N x band x width x 3 float32).
_BAND_BUDGET_BYTES = 768 * 1024 * 1024
# Coadd path: context around each tile (>= proper coadd's PSF kernel radius, so a
# tile's kept pixels see every frame sample their filters reach), and the zero
# padding of the FFT grid (linear, not circular, convolution).
_TILE_MARGIN = 32
_FFT_PAD = 32
# Sky/noise crops per frame for the coadd path: a _SKY_GRID x _SKY_GRID grid of
# _SKY_CROP px squares inside the core.
_SKY_GRID = 4
_SKY_CROP = 128


def _src_mapping(shift, transform) -> Tuple[np.ndarray, np.ndarray]:
    """(mat, off) with src_rc = mat @ out_rc + off -- the native warp's own
    convention in ``registration.apply_transform``."""
    if transform is not None:
        R = np.asarray(transform.params[:2, :2], dtype=np.float64)
        t_xy = np.asarray(transform.params[:2, 2], dtype=np.float64)
        return R, -R @ np.array([t_xy[1], t_xy[0]])
    sy, sx = (0.0, 0.0) if shift is None else shift
    return np.eye(2), np.array([-float(sy), -float(sx)])


def frame_coverage(shift, transform, rows: np.ndarray, cols: np.ndarray,
                   H: int, W: int, margin: float = EDGE_MARGIN_PX) -> np.ndarray:
    """Boolean (len(rows), len(cols)): output pixel maps inside the source frame."""
    mat, off = _src_mapping(shift, transform)
    r = rows[:, None].astype(np.float64)
    c = cols[None, :].astype(np.float64)
    sr = mat[0, 0] * r + mat[0, 1] * c + off[0]
    sc = mat[1, 0] * r + mat[1, 1] * c + off[1]
    return (sr >= margin) & (sr <= H - 1 - margin) & (sc >= margin) & (sc <= W - 1 - margin)


def coverage_count(shifts, transforms, rows, cols, H, W) -> np.ndarray:
    cnt = np.zeros((len(rows), len(cols)), dtype=np.int32)
    for s, t in zip(shifts, transforms):
        cnt += frame_coverage(s, t, rows, cols, H, W)
    return cnt


def grow_rectangle(ok: np.ndarray, core: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Grow the grid-index rectangle ``core`` (i0, i1, j0, j1, inclusive) one
    step per side, round-robin, while the new strip is entirely ``ok``."""
    i0, i1, j0, j1 = core
    gh, gw = ok.shape
    grew = True
    while grew:
        grew = False
        if i0 > 0 and ok[i0 - 1, j0:j1 + 1].all():
            i0 -= 1; grew = True
        if i1 < gh - 1 and ok[i1 + 1, j0:j1 + 1].all():
            i1 += 1; grew = True
        if j0 > 0 and ok[i0:i1 + 1, j0 - 1].all():
            j0 -= 1; grew = True
        if j1 < gw - 1 and ok[i0:i1 + 1, j1 + 1].all():
            j1 += 1; grew = True
    return i0, i1, j0, j1


def choose_rectangle(shifts, transforms, H: int, W: int, core: Tuple[int, int, int, int],
                     frac: float, step: int = GRID_STEP) -> Tuple[int, int, int, int]:
    """Largest rectangle (grown outward from the core crop) whose every pixel is
    covered by >= max(3, frac * N) frames. Returns (top, bottom, left, right)."""
    n = len(shifts)
    need = max(3, int(np.ceil(frac * n)))
    rows = np.arange(0, H, step)
    cols = np.arange(0, W, step)
    rows = np.unique(np.append(rows, H - 1))
    cols = np.unique(np.append(cols, W - 1))
    ok = coverage_count(shifts, transforms, rows, cols, H, W) >= need
    top, bottom, left, right = core
    # grid indices strictly inside the core
    i0 = int(np.searchsorted(rows, top)); i1 = int(np.searchsorted(rows, bottom - 1, 'right') - 1)
    j0 = int(np.searchsorted(cols, left)); j1 = int(np.searchsorted(cols, right - 1, 'right') - 1)
    if i0 > i1 or j0 > j1:
        return core
    i0, i1, j0, j1 = grow_rectangle(ok, (i0, i1, j0, j1))
    t, b = min(int(rows[i0]), top), max(int(rows[i1]) + 1, bottom)
    lft, rgt = min(int(cols[j0]), left), max(int(cols[j1]) + 1, right)
    return t, b, lft, rgt


def _windows(rect, core, feather) -> List[Tuple[int, int, int, int]]:
    """Output windows (absolute coords) covering rect minus the core shrunk by
    ``feather``: full-width row bands above/below, side strips beside it."""
    T, B, L, R = rect
    ct, cb, cl, cr = core
    it, ib, il, ir = ct + feather, cb - feather, cl + feather, cr - feather
    if it >= ib or il >= ir:
        return [(T, B, L, R)]
    out = []
    if it > T:
        out.append((T, it, L, R))
    if B > ib:
        out.append((ib, B, L, R))
    if il > L:
        out.append((it, ib, L, il))
    if R > ir:
        out.append((it, ib, ir, R))
    return out


def _combine_window(win, mem_rgb, final_indices, shifts, transforms, H, W, C,
                    weights, sigma, iters):
    """Weighted sigma-clipped mean of the frames covering ``win``, plus the
    per-pixel frame count. Uncovered samples are NaN."""
    from src.registration import apply_transform
    from src.stacking import sigma_clip_combine
    t, b, lft, rgt = win
    h, w = b - t, rgt - lft
    rows = np.arange(t, b)
    cols = np.arange(lft, rgt)
    covs = [frame_coverage(shifts[j], transforms[j], rows, cols, H, W)
            for j in range(len(final_indices))]
    use = [j for j, cv in enumerate(covs) if cv.any()]
    out = np.zeros((h, w, C), np.float32)
    count = np.zeros((h, w), np.int32)
    if not use:
        return out, count
    per_row = len(use) * w * C * 4
    band = max(8, int(_BAND_BUDGET_BYTES // max(per_row, 1)))
    for r0 in range(0, h, band):
        r1 = min(h, r0 + band)
        stack = np.empty((len(use), r1 - r0, w, C), np.float32)
        for k, j in enumerate(use):
            rgb = np.asarray(mem_rgb[final_indices[j]])
            if rgb.dtype != np.float32 or not rgb.flags['C_CONTIGUOUS']:
                rgb = np.ascontiguousarray(rgb, dtype=np.float32)
            apply_transform(rgb, shift=shifts[j], transform=transforms[j],
                            crop=(t + r0, t + r1, lft, rgt), out=stack[k])
            stack[k][~covs[j][r0:r1]] = np.nan
        out[r0:r1] = sigma_clip_combine(stack, sigma=sigma, max_iters=iters,
                                        weights=np.asarray(weights, np.float32)[use],
                                        use_mad=True)
        count[r0:r1] = np.sum([covs[j][r0:r1] for j in use], axis=0)
    return out, count


def _lum(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) if a.shape[-1] == 3 else a[..., 0]


def _frame_rgb(mem_rgb, idx) -> np.ndarray:
    rgb = np.asarray(mem_rgb[idx])
    if rgb.dtype != np.float32 or not rgb.flags['C_CONTIGUOUS']:
        rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    return rgb


def measure_frames(stacked: np.ndarray, mem_rgb, final_indices, shifts, transforms,
                   core: Tuple[int, int, int, int], fwhm: float = 5.0) -> Optional[dict]:
    """Proper coadd's per-frame model, measured without warping whole frames.

    PSF stars are picked on the core stack (``proper_coadd.select_psf_stars``); for
    every frame only their 29x29 stamps are warped (into a one-row mosaic that
    ``fit_psf`` reads like a frame), plus a grid of sky crops for the sky level and
    lag-4 noise. Transparency is each star's flux over its median across frames,
    exactly as ``proper_coadd`` does, so the result is on the median frame's flux
    scale like the core. Returns {'use', 'psf', 'F', 'sky' (n, C), 'sig' (n, C)} or
    None when the PSF cannot be measured."""
    from src import proper_coadd as pc
    from src.registration import apply_transform
    ct, cb, cl, cr = core
    C = stacked.shape[-1]
    lum = _lum(stacked)
    stars = pc.select_psf_stars(lum, fwhm)
    if len(stars) < 5:
        return None
    p_ref, _ = pc.fit_psf(lum, stars)
    if p_ref is None:
        return None
    r = pc._STAMP_R
    sw = 2 * r + 1
    iy = np.round(stars[:, 0]).astype(int)
    ix = np.round(stars[:, 1]).astype(int)
    mstars = np.stack([r + (stars[:, 0] - iy),
                       np.arange(len(stars)) * sw + r + (stars[:, 1] - ix)], axis=1)
    hh, ww = cb - ct, cr - cl
    cs = min(_SKY_CROP, hh // (_SKY_GRID + 1), ww // (_SKY_GRID + 1))
    sky_at = [(ct + int((i + 0.5) * hh / _SKY_GRID) - cs // 2,
               cl + int((k + 0.5) * ww / _SKY_GRID) - cs // 2)
              for i in range(_SKY_GRID) for k in range(_SKY_GRID)]

    def one(j):
        rgb = _frame_rgb(mem_rgb, final_indices[j])
        mosaic = np.empty((sw, sw * len(stars), C), np.float32)
        for s_, (y, x) in enumerate(zip(iy, ix)):
            apply_transform(rgb, shift=shifts[j], transform=transforms[j],
                            crop=(ct + y - r, ct + y + r + 1, cl + x - r, cl + x + r + 1),
                            out=mosaic[:, s_ * sw:(s_ + 1) * sw])
        p, flux = pc.fit_psf(mosaic, mstars, p0=p_ref)
        crops = np.empty((len(sky_at), cs, cs, C), np.float32)
        for q, (y0, x0) in enumerate(sky_at):
            apply_transform(rgb, shift=shifts[j], transform=transforms[j],
                            crop=(y0, y0 + cs, x0, x0 + cs), out=crops[q])
        sky = np.array([float(np.median(crops[..., c])) for c in range(C)])
        d = (crops[:, :, 4:] - crops[:, :, :-4]).reshape(-1, C)
        sig = np.array([1.4826 * float(np.median(np.abs(d[:, c] - np.median(d[:, c])))) / np.sqrt(2.0)
                        for c in range(C)])
        return p, flux, sky, sig

    with ThreadPoolExecutor(max_workers=2) as ex:      # GIL-bound fits, as in proper_coadd
        raw = list(ex.map(one, range(len(final_indices))))
    n = len(raw)
    fit_ok = [j for j in range(n) if raw[j][0] is not None]
    if len(fit_ok) < 3:
        return None
    FL = np.array([raw[j][1] for j in fit_ok])
    with np.errstate(invalid='ignore'):
        norm = np.nanmedian(np.where(FL > 0, FL, np.nan), axis=0)
    F = np.full(n, np.nan)
    for i, j in enumerate(fit_ok):
        ok = np.isfinite(FL[i]) & (FL[i] > 0) & (norm > 0)
        if ok.sum() >= 5:
            F[j] = float(np.median(FL[i, ok] / norm[ok]))
    sky = np.array([raw[j][2] for j in range(n)])
    sig = np.array([raw[j][3] for j in range(n)])
    use = [j for j in range(n) if np.isfinite(F[j]) and F[j] > 0.05
           and np.all(np.isfinite(sig[j])) and np.all(sig[j] > 0)]
    if len(use) < 3:
        return None
    return {'use': use, 'psf': [raw[j][0] for j in range(n)], 'F': F, 'sky': sky, 'sig': sig}


def _tiles(win, H, W, budget_px: int):
    """Split ``win`` along its long axis into tiles; each is (kept, padded), where
    ``padded`` adds ``_TILE_MARGIN`` px of context (clipped to the frame)."""
    t, b, lft, rgt = win
    m = _TILE_MARGIN
    out = []
    if (rgt - lft) >= (b - t):
        rows = min(b + m, H) - max(t - m, 0)
        step = max(64, budget_px // max(rows, 1) - 2 * m)
        for c0 in range(lft, rgt, step):
            c1 = min(rgt, c0 + step)
            out.append(((t, b, c0, c1), (max(t - m, 0), min(b + m, H), max(c0 - m, 0), min(c1 + m, W))))
    else:
        cols = min(rgt + m, W) - max(lft - m, 0)
        step = max(64, budget_px // max(cols, 1) - 2 * m)
        for r0 in range(t, b, step):
            r1 = min(b, r0 + step)
            out.append(((r0, r1, lft, rgt), (max(r0 - m, 0), min(r1 + m, H), max(lft - m, 0), min(rgt + m, W))))
    return out


def _coadd_window(win, mem_rgb, final_indices, shifts, transforms, H, W, C, model,
                  sigma, iters):
    """Proper coadd of ``win`` from the frames covering it, as a normalised convolution
    (see the module docstring), plus the per-pixel count of frames used.

    Each frame's samples are flux-normalised and sky-subtracted, cleaned against a
    clipped mean of the same samples exactly as proper coadd cleans against the
    normal stack (``k * sqrt(s^2 + (0.15 * signal)^2)``), zeroed where the frame does
    not cover, and filtered by w_j conj(P_j); the frame's coverage mask goes through
    the same filter, and the ratio of the two sums renormalises every pixel by the
    weight that actually reached it."""
    import scipy.fft as sfft

    from src import proper_coadd as pc
    from src.registration import apply_transform
    from src.stacking import sigma_clip_combine
    use = model['use']
    F, sky, sig, psf = model['F'], model['sky'], model['sig'], model['psf']
    t, b, lft, rgt = win
    out = np.zeros((b - t, rgt - lft, C), np.float32)
    count = np.zeros((b - t, rgt - lft), np.int32)
    budget_px = int(_BAND_BUDGET_BYTES // max(len(use) * C * 4, 1))
    wts = np.array([[F[j] ** 2 / sig[j, c] ** 2 for c in range(C)] for j in use])     # (n_use, C)
    w_ref = (F[use] ** 2 / np.mean(sig[use] ** 2, axis=1)).astype(np.float32)
    pad = _FFT_PAD
    n_thr = max(1, min(6, (os.cpu_count() or 8) // 2, len(use)))
    for kept, padded in _tiles(win, H, W, budget_px):
        pt, pb, pl, pr = padded
        h, w = pb - pt, pr - pl
        rows, cols = np.arange(pt, pb), np.arange(pl, pr)
        covs = [frame_coverage(shifts[j], transforms[j], rows, cols, H, W) for j in use]
        here = [k for k, cv in enumerate(covs) if cv.any()]
        if not here:
            continue
        stack = np.empty((len(here), h, w, C), np.float32)
        for i, k in enumerate(here):
            j = use[k]
            apply_transform(_frame_rgb(mem_rgb, final_indices[j]), shift=shifts[j],
                            transform=transforms[j], crop=padded, out=stack[i])
            stack[i] -= sky[j].astype(np.float32)
            stack[i] *= np.float32(1.0 / F[j])
            stack[i][~covs[k]] = np.nan
        # outlier reference: clipped mean of the flux-normalised samples
        ref = sigma_clip_combine(stack, sigma=sigma, max_iters=iters, weights=w_ref[here], use_mad=True)
        ref_tol = (np.float32(pc._SIGNAL_FRAC) * np.maximum(ref, 0)) ** 2
        PH = sfft.next_fast_len(h + 2 * pad, real=True)
        PW = sfft.next_fast_len(w + 2 * pad, real=True)
        fshape = (PH, PW // 2 + 1)
        ones_spec = None
        if any(covs[k].all() for k in here):
            buf = np.zeros((PH, PW), np.float32)
            buf[pad:pad + h, pad:pad + w] = 1.0
            ones_spec = sfft.rfft2(buf, workers=1)
        pos = {k: i for i, k in enumerate(here)}

        def worker(part):
            num = np.zeros((C,) + fshape, np.complex64)
            den = np.zeros((C,) + fshape, np.complex64)
            dt2 = np.zeros((C,) + fshape, np.float32)
            ppl = np.zeros((PH, PW), np.float32)
            buf = np.zeros((PH, PW), np.float32)
            inner = buf[pad:pad + h, pad:pad + w]
            for k in part:
                ppl[:] = 0
                pc.render_psf_into(psf[use[k]], ppl)
                Ph = sfft.rfft2(ppl, workers=1)
                p2 = Ph.real ** 2 + Ph.imag ** 2
                for c in range(C):
                    dt2[c] += np.float32(wts[k, c]) * p2
                if k not in pos:          # the global PSF norm counts every frame
                    continue
                cPh = np.conj(Ph)
                fr = stack[pos[k]]
                cov = covs[k]
                if cov.all():
                    ms = ones_spec
                else:
                    buf[:] = 0
                    inner[...] = cov
                    ms = sfft.rfft2(buf, workers=1)
                ms = cPh * ms
                s2 = (sig[use[k]] / F[use[k]]) ** 2
                for c in range(C):
                    x = fr[..., c]
                    rc = ref[..., c]
                    with np.errstate(invalid='ignore'):
                        bad = np.abs(x - rc) > np.float32(pc._REJECT_K) * np.sqrt(ref_tol[..., c] + np.float32(s2[c]))
                    buf[:] = 0
                    np.copyto(inner, x)
                    inner[bad] = rc[bad]          # NaN compares False: covered samples only
                    inner[~cov] = 0
                    wk = np.float32(wts[k, c])
                    num[c] += wk * (cPh * sfft.rfft2(buf, workers=1))
                    den[c] += wk * ms
            return num, den, dt2

        with ThreadPoolExecutor(max_workers=n_thr) as ex:
            parts = list(ex.map(worker, [list(range(len(use)))[i::n_thr] for i in range(n_thr)]))
        num, den, dt2 = parts[0]
        for p_num, p_den, p_dt2 in parts[1:]:                 # fixed order: deterministic
            num += p_num
            den += p_den
            dt2 += p_dt2
        del parts
        stack = None          # free the tile stack before the inverse FFTs
        kt, kb, kl, kr = kept
        ks = (slice(kt - pt, kb - pt), slice(kl - pl, kr - pl))
        for c in range(C):
            dt = np.sqrt(np.maximum(dt2[c] / np.float32(wts[:, c].sum()), np.float32(1e-20)))
            a = sfft.irfft2(num[c] / dt, s=(PH, PW), workers=-1)[pad:pad + h, pad:pad + w][ks]
            d = sfft.irfft2(den[c] / dt, s=(PH, PW), workers=-1)[pad:pad + h, pad:pad + w][ks]
            skyw = float(np.sum(wts[:, c] * sky[use, c] / F[use]) / wts[:, c].sum())
            wsum_here = float(wts[here, c].sum())
            # where hardly any weight reaches the pixel the ratio is unstable: use the
            # clipped mean there (inside the chosen rectangle, >= frac of the frames
            # cover every pixel, so this is a guard, not a path)
            good = d > 0.1 * wsum_here
            val = np.where(good, a / np.where(good, d, 1.0), ref[..., c][ks])
            out[kt - t:kb - t, kl - lft:kr - lft, c] = (val + skyw).astype(np.float32)
        count[kt - t:kb - t, kl - lft:kr - lft] = np.sum([covs[k][ks] for k in here], axis=0)
    return out, count


def _fit_gain_offset(src: np.ndarray, ref: np.ndarray, smooth: float = 3.0) -> Tuple[float, float]:
    """Robust ref ~ g*src + o on the overlap, fitted on Gaussian-smoothed pixels.

    Both images are noisy, and a least-squares slope with noise in ``src`` is
    biased towards zero by var(signal) / (var(signal) + var(noise)): on the sky
    pixels this fitted (previously unsmoothed, brightest 1% cut) it read
    g = 0.52 / 1.00 / 0.56 for R/G/B on a real M101 stack whose true gain was
    1.00 / 0.99 / 0.98 -- the extension's R and B came out at half the core's
    contrast and noise. Smoothing first (sigma 3 px cuts the noise variance
    ~100x) and keeping the stars (they give the slope its leverage) removes the
    bias; a slope outside (0.67, 1.5) is not trusted (offset only)."""
    from src.background import _gaussian_blur
    ok = np.isfinite(src) & np.isfinite(ref)
    if ok.sum() < 500:
        return 1.0, 0.0
    fill_s = float(np.median(src[ok]))
    fill_r = float(np.median(ref[ok]))
    xs = _gaussian_blur(np.where(ok, src, fill_s).astype(np.float32), smooth)
    ys = _gaussian_blur(np.where(ok, ref, fill_r).astype(np.float32), smooth)
    # pixels whose smoothing window saw no filled value
    ok = _gaussian_blur(ok.astype(np.float32), smooth) > 0.999
    x, y = xs[ok].astype(np.float64), ys[ok].astype(np.float64)
    if x.size < 500:
        return 1.0, fill_r - fill_s
    g, o = 1.0, float(np.median(y - x))
    for _ in range(5):
        A = np.vstack([x, np.ones_like(x)]).T
        (g, o), *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - (g * x + o)
        s = 1.4826 * np.median(np.abs(res - np.median(res))) or 1.0
        m = np.abs(res) < 4 * s
        if m.all():
            break
        x, y = x[m], y[m]
    if not (0.67 < g < 1.5):
        return 1.0, float(np.median(ys[ok] - xs[ok]))
    return float(g), float(o)


def extend_full_field(stacked: np.ndarray, mem_rgb, final_indices, shifts, transforms,
                      H: int, W: int, C: int, core: Tuple[int, int, int, int], weights,
                      args, frac: float = 0.5, fwhm: float = 5.0):
    """Return (stacked, (top, bottom, left, right), coverage) on the extended
    rectangle, or None when there is nothing to extend.

    With ``args.proper_coadd`` the outside is a proper coadd too (``_coadd_window``);
    ``fwhm`` (the frames' median, px) seeds its PSF-star detection."""
    import time
    t0 = time.time()
    transforms = list(transforms) if transforms is not None else [None] * len(shifts)
    rect = choose_rectangle(shifts, transforms, H, W, core, frac)
    T, B, L, R = rect
    ct, cb, cl, cr = core
    gain_px = (B - T) * (R - L) - (cb - ct) * (cr - cl)
    if gain_px <= 0.02 * (cb - ct) * (cr - cl):
        safe_print("  Full field: the common crop already covers the field -- nothing to add")
        return None
    feather = min(FEATHER_PX, (cb - ct) // 4, (cr - cl) // 4)
    sigma = float(getattr(args, 'rejection_sigma', 3.0) or 3.0)
    iters = int(getattr(args, 'rejection_iters', 3) or 3)

    model = None
    if getattr(args, 'proper_coadd', False):
        t_m = time.time()
        try:
            model = measure_frames(stacked, mem_rgb, final_indices, shifts, transforms, core, fwhm)
        except Exception as exc:
            _log.debug("full field: frame model failed (%s)", exc, exc_info=True)
            model = None
        if model is None:
            safe_print("  Full field: PSF could not be measured -- the outside is a clipped mean "
                       "(softer and less noisy than the proper-coadded core)")
        else:
            _log.debug("full field: frame model for %d/%d frames in %.1fs",
                       len(model['use']), len(final_indices), time.time() - t_m)

    out = np.zeros((B - T, R - L, C), np.float32)
    have = np.zeros((B - T, R - L), bool)
    coverage = np.full((B - T, R - L), len(final_indices), np.int32)
    for win in _windows(rect, core, feather):
        if model is not None:
            part, cnt = _coadd_window(win, mem_rgb, final_indices, shifts, transforms,
                                      H, W, C, model, sigma, iters)
        else:
            part, cnt = _combine_window(win, mem_rgb, final_indices, shifts, transforms,
                                        H, W, C, weights, sigma, iters)
        sl = (slice(win[0] - T, win[1] - T), slice(win[2] - L, win[3] - L))
        out[sl] = part
        have[sl] = True
        coverage[sl] = cnt

    # Match the outer combine to the core on the feather band, then blend.
    cs = (slice(ct - T, cb - T), slice(cl - L, cr - L))
    # every core pixel is the normal stack's, i.e. all frames (the feather band's
    # count above used this module's wider warp-edge margin)
    coverage[cs] = len(final_indices)
    core_have = have[cs]
    for c in range(C):
        g, o = _fit_gain_offset(out[cs][..., c], np.where(core_have, stacked[..., c], np.nan))
        _log.debug("full field: channel %d matched to the core with gain %.4f, offset %.2f", c, g, o)
        out[..., c][have] = out[..., c][have] * g + o
    # blend weight inside the core: 0 at the core edge -> 1 at `feather` px in
    hh, ww = cb - ct, cr - cl
    dy = np.minimum(np.arange(hh), np.arange(hh)[::-1])[:, None]
    dx = np.minimum(np.arange(ww), np.arange(ww)[::-1])[None, :]
    wcore = np.clip(np.minimum(dy, dx) / max(feather, 1), 0.0, 1.0).astype(np.float32)
    wcore = 0.5 - 0.5 * np.cos(np.pi * wcore)
    region = out[cs]
    region[...] = np.where(core_have[..., None],
                           wcore[..., None] * stacked + (1 - wcore[..., None]) * region,
                           stacked)
    out = np.clip(out, 0.0, None) if np.nanmin(stacked) >= 0 else out
    safe_print(f"  Full field: {cr - cl}x{cb - ct} -> {R - L}x{B - T} px "
               f"(+{100.0 * gain_px / ((cb - ct) * (cr - cl)):.0f}% area; every pixel in >= "
               f"{max(3, int(np.ceil(frac * len(final_indices))))} of {len(final_indices)} "
               f"frames; {'proper coadd' if model is not None else 'clipped mean'}; "
               f"{format_time(time.time() - t0)})")
    return out, rect, coverage
