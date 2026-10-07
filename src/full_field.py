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
  * Outside it each pixel is a weighted sigma-clipped mean of only the frames
    that cover it (``sigma_clip_combine``'s native kernel already treats NaN
    samples as rejected from the start, so uncovered samples are NaN).
  * The outer combine is matched to the core per channel (gain + offset, fitted
    on smoothed pixels of a band inside the core where both exist -- proper
    coadd puts the core on the median frame's flux scale) and feathered in
    over ``FEATHER_PX``.

**Texture at the old crop edge (measured 2026-10, two-night M101, 240 frames,
core = proper coadd).** The visible step was mostly a gain-fit bug, not the
combine: an unsmoothed least-squares slope on sky pixels is diluted by the
noise in the extension (``_fit_gain_offset``), g = 0.52 / 1.00 / 0.56 for R/G/B
against a true ~1.0, so the outside came out at ~0.6-0.7x the core's pixel
noise (and contrast). With the fit on smoothed pixels the clipped mean's pixel
noise matches the core's (lag-1 0.95-0.98x, the same lag-4/lag-1 ratio) and its
stars are ~5% wider (joint Moffat FWHM 5.90 vs 5.62 px on the same-brightness
stars just inside; same-star FWHM 1.05-1.07x on an in-core band). Tried to close
that 5%, none kept:

  * Proper coadd outside as a normalised convolution (per-frame conj(P_j) filters
    applied to coverage-masked frames, divided by the same filters applied to
    the masks). Exact where every frame covers (in-core band: FWHM ratio 1.004,
    noise identical, rms difference 21 ADU against 95 ADU pixel noise), but its
    per-frequency normalisation belongs to the contributing set, and the high
    frequencies are carried by a few sharp frames (PSF FWHM 2.9-8.1 px): where
    5% of the frames were missing the pixel noise halved (0.3-0.6x in corners).
    Interpolating between norms of a few real covering sets went the other way
    (1.6x). Cost +55 s on top of the 14 s plain extension.
  * Proper coadd per 64 px tile from the frames covering the whole tile
    (exact per tile, blended): no smoothing, FWHM matched, but dropping the
    partially covering frames raised the noise 10-20% where coverage changes
    (they carry 4% of the weight on average, 24% at worst), and ~4x the FFT
    work of the core's proper coadd per output pixel (~25 s per strip).
  * One filter on the clipped mean, the MTF ratio sqrt(sum w|P|^2 / sum w) /
    (sum w P / sum w): noise x2.4 -- proper coadd reweights frames per
    frequency, which no filter on the mean reproduces. A Wiener-regularised
    Gaussian deconvolution sized to the FWHM gap: FWHM ratio 1.018 at noise
    x1.75, 0.986 at x1.47.

Depth is no longer uniform: an edge pixel covered by half the frames is ~1.4x
noisier. ``<output>_coverage.fits`` records the per-pixel frame count.
"""
from __future__ import annotations

from typing import List, Tuple

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
                      args, frac: float = 0.5):
    """Return (stacked, (top, bottom, left, right), coverage) on the extended
    rectangle, or None when there is nothing to extend."""
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

    out = np.zeros((B - T, R - L, C), np.float32)
    have = np.zeros((B - T, R - L), bool)
    coverage = np.full((B - T, R - L), len(final_indices), np.int32)
    for win in _windows(rect, core, feather):
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
               f"frames; {format_time(time.time() - t0)})")
    return out, rect, coverage
