"""Incremental stacking: merge previously saved linear stacks into this run.

A previous run's main output FITS is the linear, pre-post-processing stack
(``RAWSTACK=True``) with ``NFRAMES``/``TOTEXP``/date metadata in its header.
``--merge PREV.fits [...]`` registers each such stack onto the current
session's pixel grid (star-match affine with a translation fallback — nights
on an alt-az mount differ by arbitrary field rotation) and combines them as a
per-pixel weighted mean, weight = that stack's inverse noise variance inside
its warped footprint and 0 outside, so zero-filled borders never dilute the
result:

    merged(x) = sum_i w_i(x) * S_i(x) / sum_i w_i(x)

Sessions are rarely comparable in raw ADU: a 25 s ISO 500 stack sits at a
different flux scale than a 10 s ISO 200 one, and averaging them as-is mixes
two brightness scales (real Fireworks Galaxy data, five sessions). Each
previous stack is therefore first mapped onto the current stack's flux scale
(``_match_flux_scale``: a robust per-channel gain + sky offset fit on the
smoothed, above-noise pixels both stacks share), and the weights are
1/sigma^2 of the *scaled* stack's measured pixel noise (``_pixel_noise``) --
the Gauss-Markov optimum -- not its frame count, which is only a proxy when
every session has the same per-frame exposure and noise. If any stack's noise
cannot be measured the whole merge falls back to frame-count weights (gains
are still applied).

By default there is no cross-session outlier rejection — each session already
rejected outliers internally when it was stacked (the same tradeoff
hierarchical mode makes). ``--merge-consistency`` adds it where it is still
possible, between whole nights (``consistency_combine``): per starlet scale, a
night is left out of a coefficient when it shows a significant feature the
other nights, combined, show nothing of (a satellite glint, an unrejected
hot-pixel cluster, a transient, a reflection). ``--merge-proper`` replaces the
mean with a proper coadd of the nights (``proper_merge``): each weighted per
spatial frequency by its own PSF, so a sharp night is not diluted by softer ones.
The merged output is itself a linear stack with summed headers, so merges chain
night after night.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.debayer import luminance
from src.models import Config
from src.utils import diff_mad_sigma, get_logger, safe_print
from src.utils import embed_to_shape as _embed_to_shape

_log = get_logger()


def _lum(rgb: np.ndarray) -> np.ndarray:
    return luminance(rgb).astype(np.float32)


def _detect_stars(lum: np.ndarray) -> Optional[Any]:
    """Star catalog for affine matching; robust noise estimate from MAD."""
    from src.quality import detect_stars_auto
    med = float(np.median(lum))
    noise = 1.4826 * float(np.median(np.abs(lum - med)))
    try:
        return detect_stars_auto(lum, max(noise, 1e-3), background=med)
    except Exception:
        return None


def _register_stack(new_lum: np.ndarray, prev_lum: np.ndarray,
                    new_stars: Optional[Any],
                    prev_shape: Optional[Tuple[int, int]] = None) -> Tuple[Optional[Any],
                                                                          Optional[Tuple[float, float]]]:
    """Transform aligning ``prev`` onto ``new``: blind (unknown-rotation)
    star-pattern match first (rotation between nights is arbitrary),
    translation-seeded star-match affine and translation-only fallbacks.
    Returns (transform, shift) — exactly one is non-None on success,
    both None on failure.

    ``prev_shape``: the (h, w) of ``prev``'s own pixels when it was zero-padded
    (``embed_to_shape``, top-left) to the current stack's shape. Stars are
    detected inside it only: the drop from sky to the zero padding read as a
    row of bright "stars" along the edge, which outnumbered the real ones among
    the brightest 40 the blind match uses (a 48-minute Whirlpool session,
    1840x2902 padded to 1992x2949, was refused for it)."""
    from src.registration import _blind_match_transform, calculate_shift, match_stars_affine

    if prev_shape is not None:
        prev_stars = _detect_stars(prev_lum[:prev_shape[0], :prev_shape[1]])
    else:
        prev_stars = _detect_stars(prev_lum)

    # Blind match first: it makes no assumption about rotation, handling the
    # arbitrary field rotation between nights. The nearest-neighbour+RANSAC
    # star matcher (match_stars_affine, below) assumes rotation ~ 0 after
    # translation (an in-session tool) and can return a confidently wrong
    # model on rotated fields.
    if new_stars is not None and prev_stars is not None:
        tf = _blind_match_transform(new_stars, prev_stars)
        if tf is not None:
            return tf, None

    shift = None
    try:
        sy, sx = calculate_shift(new_lum, prev_lum, skip_phase_cc=True,
                                 use_pyramid=True, masked_correlation=False,
                                 corr_downsample=2)
        if np.isfinite(sy) and np.isfinite(sx):
            shift = (float(sy), float(sx))
    except Exception as exc:
        _log.debug("merge: translation estimate failed: %s", exc)

    if new_stars is not None and prev_stars is not None:
        tf = match_stars_affine(new_stars, prev_stars,
                                initial_shift=shift or (0.0, 0.0))
        if tf is not None:
            return tf, None
    if shift is not None:
        return None, shift
    return None, None


def _meta_from_header(path: str, hdr) -> Dict[str, Any]:
    return {
        'path': path,
        'nframes': int(hdr.get('NFRAMES', 0) or 0),
        'intgtime': float(hdr.get('INTGTIME', 0.0) or 0.0),
        'totexp': float(hdr.get('TOTEXP', 0.0) or 0.0),
        'datefrst': hdr.get('DATEFRST'),
        'datelast': hdr.get('DATELAST'),
    }


def read_merge_meta(path: str) -> Dict[str, Any]:
    """The same metadata ``load_merge_stack`` returns, read from the header only.

    For callers that need a stack's depth or dates to *choose* one -- picking
    the reference grid of a hierarchical combine, say -- and would otherwise
    pull in and float32-copy the whole image (~144 MB at 6 MP, per stack) to
    read one integer. Deliberately does not check ``RAWSTACK``: that is
    ``load_merge_stack``'s job, and it runs on whichever stack is chosen.
    """
    from astropy.io import fits as _fits
    return _meta_from_header(path, _fits.getheader(path))


def load_merge_stack(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load and validate one ``--merge`` input. Returns (HWC float32, meta)."""
    from astropy.io import fits as _fits

    with _fits.open(path) as hdul:
        hdr = hdul[0].header
        data = np.asarray(hdul[0].data)
    if not bool(hdr.get('RAWSTACK', False)):
        raise ValueError(
            f"{os.path.basename(path)} is not a linear (pre-post-processing) "
            f"stack: header RAWSTACK is missing or False. Pass the main "
            f"output FITS of a previous run, not the _processed one.")
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"{os.path.basename(path)}: expected an RGB stack, "
                         f"got shape {data.shape}")
    return np.ascontiguousarray(data, dtype=np.float32), _meta_from_header(path, hdr)


_SCALE_SMOOTH_SIGMA = 4.0     # px; equalises seeing/PSF differences between sessions
_SCALE_MIN_SNR = 8.0          # smoothed-pixel S/N (in both stacks) to enter the fit
_SCALE_MIN_PIXELS = 500
_SCALE_SNAP = 0.015           # gains within 1 +/- this are treated as exactly 1
_SCALE_BOUNDS = (0.2, 5.0)
_FOOTPRINT_TRIM_PX = 3        # px trimmed off each warped footprint edge
_NOISE_LAG = 4                # px; differencing at a lag clears resampling correlation


def _pixel_noise(ch: np.ndarray, valid: np.ndarray) -> float:
    """Robust per-pixel noise sigma of one channel inside ``valid``.

    MAD of differences between pixels ``_NOISE_LAG`` apart, / sqrt(2): stars
    and nebulosity are a small tail the MAD ignores, and the lag keeps the
    estimate from being fooled by the short-range correlation debayering and
    resampling leave in a stack (it still sees the variance those steps really
    removed, which is what the weights should reflect). NaN if there are too
    few valid pairs.
    """
    L = _NOISE_LAG
    pair = valid[:, L:] & valid[:, :-L]
    if int(pair.sum()) < 1000:
        return float('nan')
    d = (ch[:, L:].astype(np.float64) - ch[:, :-L])[pair]
    return diff_mad_sigma(d)


def _match_flux_scale(ref: np.ndarray, img: np.ndarray, valid: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per-channel (gain, offset) mapping ``img`` onto ``ref``'s flux scale.

    ``ref ~= offset + gain * img``. Both are Gaussian-smoothed first so a
    seeing difference between sessions does not read as a flux difference,
    each is reduced to its own sky (median over ``valid``), and the gain is a
    Tukey-IRLS regression through the origin on pixels bright in *both*
    (smoothed S/N >= ``_SCALE_MIN_SNR``). Unmeasurable channels (too few
    shared bright pixels, or a gain outside ``_SCALE_BOUNDS``) get gain 1 and
    offset 0; a gain within ``_SCALE_SNAP`` of 1 snaps to 1 so a same-settings
    merge is not perturbed by estimator noise. Returns (gain[3], offset[3],
    pixels used in the worst-measured channel).
    """
    from src.background import _gaussian_blur
    gains = np.ones(3)
    offsets = np.zeros(3)
    n_min = None
    for c in range(3):
        sr = _gaussian_blur(ref[:, :, c].astype(np.float32), _SCALE_SMOOTH_SIGMA)
        si = _gaussian_blur(img[:, :, c].astype(np.float32), _SCALE_SMOOTH_SIGMA)
        vr, vi = sr[valid].astype(np.float64), si[valid].astype(np.float64)
        if vr.size < _SCALE_MIN_PIXELS:
            n_min = 0
            continue
        mr, mi = float(np.median(vr)), float(np.median(vi))
        sd_r = 1.4826 * float(np.median(np.abs(vr - mr)))
        sd_i = 1.4826 * float(np.median(np.abs(vi - mi)))
        x = vi - mi
        y = vr - mr
        sel = (x > _SCALE_MIN_SNR * sd_i) & (y > _SCALE_MIN_SNR * sd_r)
        n = int(sel.sum())
        n_min = n if n_min is None else min(n_min, n)
        if n < _SCALE_MIN_PIXELS:
            continue
        x, y = x[sel], y[sel]
        if n > 200_000:
            k = np.random.default_rng(0).choice(n, 200_000, replace=False)
            x, y = x[k], y[k]
        g = float(np.sum(x * y) / np.sum(x * x))
        for _ in range(8):
            r = y - g * x
            scale = 1.4826 * float(np.median(np.abs(r))) + 1e-12
            u = r / (4.685 * scale)
            w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
            if w.sum() <= 0:
                break
            g = float(np.sum(w * x * y) / np.sum(w * x * x))
        if not np.isfinite(g) or not (_SCALE_BOUNDS[0] <= g <= _SCALE_BOUNDS[1]):
            continue
        if abs(g - 1.0) < _SCALE_SNAP:
            g = 1.0
        gains[c] = g
        offsets[c] = mr - g * mi
    return gains, offsets, int(n_min or 0)


# Clipped-core neutralisation reaches this far past each stack's plateau: a
# wing clipped in some sessions and not others, scaled by their per-channel
# gains, left a blue ring (B/R 1.4-1.7 at 5-9 px) around neutral Lagoon cores.
_CLIP_GROW_PX = 8.0
_OFFSET_BLOCK_PX = 48         # block-median cell for the sky offset surface
_OFFSET_MIN_BLOCKS = 30


def _sky_offset_surface(ref: np.ndarray, img: np.ndarray, valid: np.ndarray,
                        gains: np.ndarray):
    """Per-channel offset surface (H, W, 3) mapping ``gains * img`` onto ``ref``.

    ``_match_flux_scale``'s single offset (median of ``ref`` - gain x median of
    ``img`` over the overlap) assumes the two stacks' skies differ by a
    constant. Each session's linear stack carries its own gradient (light
    pollution, moon, the altitude it was taken at), and on a nebula-filled field
    the overlap's median is nebula, not sky: on a five-session Lagoon combine the
    red sky of the merged sessions came out ~1200 ADU off the reference's, and
    the inverse-variance mean then stepped at every footprint edge -- diagonal
    seams and a red rim where fewer sessions covered. With the gain applied, the
    sources cancel in ``ref - gain * img``; what is left is the sky difference.
    Its block medians over the overlap are fitted with a robust (Tukey) quadratic
    in x, y. None when too few blocks are fully inside the overlap.
    """
    H, W = ref.shape[:2]
    b = _OFFSET_BLOCK_PX
    nby, nbx = H // b, W // b
    if nby < 3 or nbx < 3:
        return None
    vb = valid[:nby * b, :nbx * b].reshape(nby, b, nbx, b).mean(axis=(1, 3)) > 0.95
    if int(vb.sum()) < _OFFSET_MIN_BLOCKS:
        return None
    yy, xx = np.nonzero(vb)
    cy = ((yy + 0.5) * b - H / 2) / (W / 2)
    cx = ((xx + 0.5) * b - W / 2) / (W / 2)
    A = np.stack([np.ones_like(cx), cx, cy, cx * cx, cx * cy, cy * cy], axis=1)
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float64)
    gy = (gy - H / 2) / (W / 2)
    gx = (gx - W / 2) / (W / 2)
    out = np.empty((H, W, 3), dtype=np.float64)
    for c in range(3):
        d = (ref[:nby * b, :nbx * b, c].astype(np.float64)
             - gains[c] * img[:nby * b, :nbx * b, c].astype(np.float64))
        med = np.median(d.reshape(nby, b, nbx, b).transpose(0, 2, 1, 3).reshape(nby, nbx, -1),
                        axis=2)[vb]
        w = np.ones_like(med)
        coef = None
        for _ in range(10):
            sw = np.sqrt(w)
            coef = np.linalg.lstsq(A * sw[:, None], med * sw, rcond=None)[0]
            r = med - A @ coef
            s = 1.4826 * float(np.median(np.abs(r))) + 1e-9
            u = r / (4.685 * s)
            w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
            if w.sum() < A.shape[1] * 3:
                return None
        out[:, :, c] = (coef[0] + coef[1] * gx + coef[2] * gy + coef[3] * gx * gx
                        + coef[4] * gx * gy + coef[5] * gy * gy)
    return out


class _UnusableStack(ValueError):
    """A stack that cannot be merged (no registration, too little overlap, or it
    does not look like the current stack after alignment)."""


_B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16.0
CONSISTENCY_LEVELS = 5
CONSISTENCY_K = 3.0          # disagreement threshold, in combined noise
CONSISTENCY_ABSENT = 3.0     # the other nights must be within this many sigma of no signal
CONSISTENCY_TOL = 0.3        # fractional tolerance on structure all show (calibration, seeing)
CONSISTENCY_CARRY_K = 5.0    # only flags this strong carry to coarser scales (chance: ~1e-6)


def _starlet_step(c: np.ndarray, j: int) -> np.ndarray:
    """One a-trous B3-spline smoothing at scale ``j`` (holes of 2^j - 1)."""
    from scipy import ndimage as ndi
    k = np.zeros(4 * 2 ** j + 1, np.float32)
    k[:: 2 ** j] = _B3
    return ndi.convolve1d(ndi.convolve1d(c, k, axis=0, mode='reflect'), k, axis=1, mode='reflect')


def _scale_noise(d: List[np.ndarray], wts: List[np.ndarray], wd: np.ndarray, wsum: np.ndarray,
                 w: List[float]) -> List[float]:
    """Per-night noise of one starlet scale. The MAD of a night's own coefficients counts
    real structure as noise (on a nebula, 5-8x the true value at coarse scales, which made
    the test lenient exactly there); the difference from the other nights' mean cancels it.
    var(d_i - m_others) = a^2 (1/w_i + 1/W_others) with noise_l = a / sqrt(w_l), the stacks'
    own inverse-variance weights, so one scale factor ``a`` is fitted (median over nights)."""
    est = []
    for i, di in enumerate(d):
        w_o = wsum - wts[i]
        sel = (wts[i] > 0) & (w_o > 1e-20)
        if sel.sum() < 100:
            continue
        e = (di - (wd - wts[i] * di) / np.maximum(w_o, 1e-30))[sel]
        mad = 1.4826 * float(np.median(np.abs(e - np.median(e))))
        w_others = float(np.median(w_o[sel]))
        est.append(mad / np.sqrt(1.0 / w[i] + 1.0 / w_others))
    a = float(np.median(est)) if est else 0.0
    return [a / np.sqrt(wi) for wi in w]


def _starlet_details(planes: List[np.ndarray], levels: int):
    """Per scale, the starlet detail plane of each image and the final coarse planes."""
    details, cur = [], [p.astype(np.float32) for p in planes]
    for j in range(levels):
        nxt = [_starlet_step(x, j) for x in cur]
        details.append([x - s for x, s in zip(cur, nxt)])
        cur = nxt
    return details, cur


def _consistency_drops(lum: List[np.ndarray], wts: List[np.ndarray], wsum: np.ndarray,
                       w: List[float], protect: np.ndarray, k: float, absent: float, tol: float,
                       levels: int) -> List[List[np.ndarray]]:
    """drops[j][i]: where night i is left out at starlet scale j, decided on luminance."""
    from scipy import ndimage as ndi
    details, _ = _starlet_details(lum, levels)
    wds = [np.sum([wi * di for wi, di in zip(wts, d)], axis=0) for d in details]
    noises = [_scale_noise(d, wts, wd, wsum, w) for d, wd in zip(details, wds)]
    carry = [np.zeros(wsum.shape, bool) for _ in lum]
    drops = []
    for j in range(levels):
        d, wd, noise = details[j], wds[j], noises[j]
        m_all = wd / wsum
        r = 2 ** (j + 1)
        disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
        row = []
        for i, di in enumerate(d):
            w_o = wsum - wts[i]
            has_o = w_o > 1e-20
            w_o = np.maximum(w_o, 1e-30)
            m_o = (wd - wts[i] * di) / w_o
            n_o = np.sqrt(np.sum([(wl * nl) ** 2 for idx, (wl, nl) in enumerate(zip(wts, noise))
                                  if idx != i], axis=0)) / w_o
            thr = k * np.sqrt(noise[i] ** 2 + n_o ** 2 + (tol * m_all) ** 2)
            cand = (has_o & ~protect & (np.abs(m_o) < absent * n_o) & (np.abs(di) > np.abs(m_o))
                    & (wts[i] > 0))
            if j + 1 < levels:
                # absent one scale coarser too: a star the softer night shows below 3 sigma
                # at this scale is strong at the next; a glint is absent at every scale
                d1, wd1, nz1 = details[j + 1], wds[j + 1], noises[j + 1]
                m_o1 = (wd1 - wts[i] * d1[i]) / w_o
                n_o1 = np.sqrt(np.sum([(wl * nl) ** 2 for idx, (wl, nl) in enumerate(zip(wts, nz1))
                                       if idx != i], axis=0)) / w_o
                cand &= np.abs(m_o1) < absent * n_o1
            excess = np.abs(di - m_o)
            flag = cand & (excess > thr)
            row.append((flag | carry[i]) & has_o)
            strong = flag & (excess > thr * (CONSISTENCY_CARRY_K / k))
            if strong.any():
                carry[i] |= ndi.binary_dilation(strong, structure=disk)
        drops.append(row)
    return drops


def consistency_combine(members: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                        k: float = CONSISTENCY_K, absent: float = CONSISTENCY_ABSENT,
                        tol: float = CONSISTENCY_TOL,
                        levels: int = CONSISTENCY_LEVELS) -> Tuple[np.ndarray, float]:
    """Inverse-variance mean of registered, flux-matched stacks with cross-night rejection.

    ``members``: (image HWC, valid HW bool, weight per channel (3,)). Decided on luminance
    and applied to all three channels alike, per starlet scale: night i is left out of a
    coefficient when (a) it differs from the weighted mean of the other nights by more than
    ``k`` x sqrt(its noise^2 + theirs^2 + (tol x the mean)^2), (b) the others are consistent
    with no signal there (within ``absent`` sigma), at this scale and the next coarser
    one, and (c) night i is the stronger. Flagged at ``CONSISTENCY_CARRY_K`` or more, it
    stays left out at that place (grown by the scale's footprint) on every coarser detail
    scale. Structure all nights show, at any strength, keeps the plain mean; so do pixels
    only one night covers and the coarse plane. Returns (merged HWC float32, share of
    coefficients where a night was left out).

    Each rule came from a failure on real merges (dev-notes/cross-night-agreement.md):

    - (b) at one scale only flattened star cores where the nights' seeing differed (the
      softer night's core under 3 sigma at the finest scales; Pleiades stars 40% wider).
    - Deciding per channel dropped stars from one colour: a night's noisier blue after flux
      scaling (Pleiades, x1.46) showed real stars under 3 sigma, the other night's blue
      star was dropped, and half the stars lost 3-50% of their blue (colour calibration
      B x0.904 -> x0.980). Luminance has the signal; glints, hot-pixel clusters and
      transients show in it too.
    - Without the carry, a point artefact's coarse scales stayed behind as a ~3-sigma ring;
      carrying the 3-sigma flags too dropped 12% of a pure-noise field's coefficients.
    - Leaving a night out of the coarse plane swapped in the other night's sky level: a
      background step under every removed artefact and signal error -51 to +14% (Orion).
    - Around any night's clipped plateau, the plain mean: nights clip at different levels
      after flux scaling, and on Orion's M42 core the selection stitched them together in
      wavelet-sized blocks (up to 500 sigma). A plateau, not a brightness cut: a satellite
      glint is often a night's brightest feature and must stay rejectable.
    - Noise per scale from the difference to the other nights (``_scale_noise``).
    - Outside a night's footprint its image is replaced by the plain mean first, so the
      zero fill never reads as a feature."""
    from scipy import ndimage as ndi

    from src.color_calibrate import _clipped_weight
    H, W = members[0][0].shape[:2]
    sat = np.zeros((H, W), bool)
    for img, v, _ in members:
        cw = _clipped_weight(np.where(v[:, :, None], img, 0.0).astype(np.float32))
        if cw is not None:
            sat |= v & (cw > 0)
    protect = ndi.binary_dilation(sat, iterations=2 ** (levels + 1)) if sat.any() else sat

    def filled(c):
        """Per night, channel c (None: luminance) with the plain mean outside its footprint,
        and the per-pixel weights."""
        if c is None:
            # inverse variance of the channel mean: 9 / sum(1 / w_c)
            ws = [9.0 / float(np.sum(1.0 / np.asarray(w, np.float64))) for _, _, w in members]
            planes = [img.mean(axis=2) for img, _, _ in members]
        else:
            ws = [float(w[c]) for _, _, w in members]
            planes = [img[:, :, c] for img, _, _ in members]
        wts = [np.where(v, wi, 0.0).astype(np.float32) for wi, (_, v, _) in zip(ws, members)]
        wsum = np.maximum(np.sum(wts, axis=0), 1e-30)
        plain = np.sum([wi * np.where(v, p, 0.0) for wi, p, (_, v, _) in zip(wts, planes, members)],
                       axis=0) / wsum
        return [np.where(v, p, plain) for p, (_, v, _) in zip(planes, members)], wts, wsum, ws

    lum, wts_l, wsum_l, ws_l = filled(None)
    drops = _consistency_drops(lum, wts_l, wsum_l, ws_l, protect, k, absent, tol, levels)
    out = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        planes, wts, wsum, _ = filled(c)
        details, coarse = _starlet_details(planes, levels)
        acc = np.zeros((H, W), np.float32)
        for j in range(levels):
            wk_sum = np.zeros((H, W), np.float32)
            wk_d = np.zeros((H, W), np.float32)
            for i, di in enumerate(details[j]):
                wk = np.where(drops[j][i], 0.0, wts[i])
                wk_sum += wk
                wk_d += wk * di
            m_all = np.sum([wi * di for wi, di in zip(wts, details[j])], axis=0) / wsum
            acc += np.where(wk_sum > 0, wk_d / np.maximum(wk_sum, 1e-30), m_all)
        acc += np.sum([wi * x for wi, x in zip(wts, coarse)], axis=0) / wsum
        out[:, :, c] = acc
    n_flag = sum(int(m.sum()) for row in drops for m in row)
    return out, n_flag / max(levels * H * W, 1)


_PROPER_FEATHER_PX = 48      # proper coadd -> weighted mean ramp inside the common footprint


def proper_merge(members: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                 fallback: np.ndarray, clip_w: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Proper coadd (Zackay & Ofek 2017) of flux-matched stacks of different nights,
    ``--merge-proper``.

    The inverse-variance mean gives each night one weight for every spatial frequency, so a
    long soft night dilutes the fine detail a sharp one carries. Here each night's stack is
    weighted per frequency by its own PSF (an elliptical Moffat fitted on shared stars,
    ``proper_coadd.fit_psf``) and noise:

        R_hat = sum_j w_j conj(P_hat_j) M_hat_j / sqrt(sum_j w_j |P_hat_j|^2) / sqrt(sum_j w_j)

    with w_j = 1 / sigma_j^2 per channel (``members``' weights, on the common flux scale).
    At zero frequency it is exactly the inverse-variance mean; the noise stays white at the
    mean's level; the PSF is the weighted RMS of the nights' MTFs, never softer than the
    mean's. On 24 two-night targets: stars a median 2.6% narrower than the mean (15% at
    p10, 42% on Eyes Galaxies, 3.2 vs 9.4 px nights) at the same noise (median 1.006x).

    It runs where every night covers the field, ramping into ``fallback`` (the plain mean)
    over ``_PROPER_FEATHER_PX`` towards the footprint edge and on clipped star cores
    (``clip_w``: a clipped plateau has no PSF). ``members``: (stack HWC, valid HW bool,
    weight per channel). None when a night's PSF cannot be measured."""
    from scipy import fft as _fft
    from scipy import ndimage as _ndi

    from src.proper_coadd import _PAD, _STAMP_R, fit_psf, psf_fwhm, render_psf_into, select_psf_stars

    if len(members) < 2:
        return None
    H, W, C = fallback.shape
    inter = np.logical_and.reduce([v for _, v, _ in members])
    if inter.mean() < 0.2:
        return None
    # stars from the mean itself, then kept inside the common footprint: zeroing the rest
    # first made its edge the brightest "source" and left 4 of 12 isolated stars
    stars = select_psf_stars(_lum(np.ascontiguousarray(fallback, dtype=np.float32)), 4.0)
    inner = _ndi.binary_erosion(inter, iterations=_STAMP_R + 2)
    if len(stars):
        stars = stars[inner[np.round(stars[:, 0]).astype(int), np.round(stars[:, 1]).astype(int)]]
    if len(stars) < 8:
        return None
    psfs = []
    for img, _, _ in members:
        p, _ = fit_psf(_lum(img), stars)
        if p is None or not np.all(np.isfinite(p)):
            return None
        psfs.append(p)
    safe_print("  Proper merge: night PSF FWHM " + ", ".join(f"{psf_fwhm(p):.2f}" for p in psfs)
               + f" px ({len(stars)} stars)")

    PH = _fft.next_fast_len(H + 2 * _PAD, real=True)
    PW = _fft.next_fast_len(W + 2 * _PAD, real=True)
    P_hat = []
    for p in psfs:
        plane = np.zeros((PH, PW), np.float32)
        render_psf_into(p, plane)
        P_hat.append(_fft.rfft2(plane))
    out = np.empty((H, W, C), np.float32)
    plane = np.zeros((PH, PW), np.float32)
    for c in range(C):
        num = np.zeros((PH, PW // 2 + 1), np.complex128)
        den = np.zeros((PH, PW // 2 + 1), np.float64)
        wsum = sky_acc = 0.0
        for (img, valid, w_c), Ph in zip(members, P_hat):
            w = float(w_c[c])
            ch = np.where(valid, img[..., c], fallback[..., c])     # no edge inside the FFT
            sky = float(np.median(ch[inter]))
            plane[_PAD:_PAD + H, _PAD:_PAD + W] = ch - sky
            num += w * np.conj(Ph) * _fft.rfft2(plane)
            den += w * (Ph.real ** 2 + Ph.imag ** 2)
            wsum += w
            sky_acc += w * sky
        R = _fft.irfft2(num / np.sqrt(np.maximum(den, 1e-30)), s=(PH, PW))
        out[..., c] = R[_PAD:_PAD + H, _PAD:_PAD + W] / np.sqrt(wsum) + sky_acc / wsum
    if not np.all(np.isfinite(out)):
        return None
    alpha = np.clip(_ndi.distance_transform_edt(inter) / _PROPER_FEATHER_PX, 0.0, 1.0)
    if clip_w is not None:
        alpha *= 1.0 - np.clip(_ndi.maximum_filter(clip_w, size=2 * _STAMP_R + 1), 0.0, 1.0)
    a = alpha[..., None].astype(np.float32)
    return (a * out + (1.0 - a) * fallback).astype(np.float32)


def merge_previous_stacks(stacked: np.ndarray, new_frame_count: int,
                          merge_paths: List[str],
                          verbose: bool = False,
                          skip_unusable: bool = False,
                          consistency: bool = False,
                          proper: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Register and weighted-merge previous linear stacks into ``stacked``.

    Returns (merged HWC float32, merge_info). Raises ValueError on
    unusable inputs or failed registration — a silently wrong merge is worse
    than an error. ``skip_unusable`` (the hierarchical combine, whose inputs are
    every session in a folder rather than stacks the user named) leaves such a
    stack out with a warning instead, listed in ``merge_info['skipped']``: one
    mis-aimed 9-frame Whirlpool session (correlation -0.03) used to abort the
    combine of eight good ones. ``consistency`` (``--merge-consistency``) combines with
    ``consistency_combine`` instead of the plain inverse-variance mean. ``proper``
    (``--merge-proper``) replaces the inverse-variance mean with ``proper_merge``, which
    weights each night per spatial frequency by its PSF (not with ``consistency``: that
    selection is per wavelet coefficient and has no proper-coadd form yet).
    """
    from src.registration import apply_transform

    H, W = stacked.shape[:2]
    new_w = float(max(new_frame_count, 1))

    # Two accumulators run side by side: inverse-variance weights (preferred)
    # and frame-count weights (fallback when any stack's noise is unmeasurable).
    new_sigma = np.array([_pixel_noise(stacked[:, :, c], np.ones((H, W), dtype=bool))
                          for c in range(3)])
    noise_ok = bool(np.all(np.isfinite(new_sigma)) and np.all(new_sigma > 0))
    w_new = 1.0 / new_sigma ** 2 if noise_ok else np.ones(3)
    acc_n = stacked.astype(np.float64) * w_new
    if proper and consistency:
        safe_print("  NOTE: --merge-proper is not combined with --merge-consistency; "
                   "using the consistency combine")
        proper = False
    members = ([(stacked.astype(np.float32), np.ones((H, W), dtype=bool), w_new)]
               if consistency or proper else None)
    wsum_n = np.broadcast_to(w_new, (H, W, 3)).astype(np.float64).copy()
    acc = stacked.astype(np.float64) * new_w
    wsum = np.full((H, W), new_w, dtype=np.float64)
    # Saturated cores: each stack's own clipped plateau, on the common grid.
    # A clipped core is neutral in its own stack, but the per-channel gains and
    # inverse-variance weights below scale it per channel -- on a five-session
    # Lagoon combine (gains up to B x2.24) every bright star came out a blue or
    # cyan disc. They are made neutral again after the merge, as
    # color_calibrate.apply_scales_inplace does for one stack.
    from src.color_calibrate import _clipped_weight
    clip_w = _clipped_weight(stacked, grow_px=_CLIP_GROW_PX)

    new_lum = _lum(stacked)
    new_stars = _detect_stars(new_lum)

    info: Dict[str, Any] = {
        'n_sources': 0, 'sources': [],
        'total_frames': int(new_frame_count),
        'total_intgtime': 0.0, 'total_totexp': 0.0,
        'datefrst': None, 'datelast': None,
        'skipped': [],
    }

    for path in merge_paths:
        try:
            prev, meta = load_merge_stack(path)
            w_prev = float(meta['nframes']) if meta['nframes'] > 0 else 1.0
            if meta['nframes'] <= 0:
                safe_print(f"  WARNING: {os.path.basename(path)} has no NFRAMES "
                           f"header — merging with weight 1.0")

            prev_h, prev_w = prev.shape[:2]
            prev = _embed_to_shape(prev, H, W)
            prev_lum = _lum(prev)
            embed_mask = _embed_to_shape(
                np.ones((prev_h, prev_w), dtype=np.float32), H, W)

            tf, shift = _register_stack(new_lum, prev_lum, new_stars,
                                        prev_shape=(prev_h, prev_w))
            if tf is None and shift is None:
                raise _UnusableStack(
                    f"Could not register {os.path.basename(path)} to the current "
                    f"stack (no star match and no correlation peak) — is it the "
                    f"same target?")

            # Negligible transform (already-aligned grids, registration jitter
            # only): skip the warp — resampling costs more than a <0.05px
            # correction is worth (same principle as the CA sub-threshold skip),
            # and (measured) an unnecessary Lanczos-3 resample of a near-identity
            # transform can itself introduce more error near sharp star cores
            # than the correction it applies. A rotation's worst-case pixel
            # displacement scales with distance from the rotation origin, so the
            # bound is evaluated at the frame's farthest corner rather than as a
            # fixed radian constant -- a fixed-radian threshold is either too
            # loose on large frames or (as measured) tighter than real
            # star-centroid noise can reliably clear on any frame size.
            if shift is not None:
                negligible = max(abs(shift[0]), abs(shift[1])) < 0.05
            else:
                _rot = float(np.arctan2(tf.params[1, 0], tf.params[0, 0]))
                _t = np.abs(tf.params[:2, 2]).max()
                _rot_px = abs(_rot) * float(np.hypot(H, W))
                negligible = _rot_px < 0.05 and _t < 0.05

            if negligible:
                aligned = prev
                footprint = embed_mask
            else:
                aligned = apply_transform(prev, shift=shift, transform=tf)
                footprint = apply_transform(embed_mask, shift=shift, transform=tf)
            valid = footprint > 0.5
            # Trim the footprint's rim. A warped stack's outermost pixels are a
            # blend of image and the zero fill outside it (and Lanczos rings against
            # that edge), so they sit off the true sky level; averaged in, they drew
            # a thin bright/dark outline around every merged session's footprint
            # (visible as tilted rectangles across a real five-session combine).
            from scipy import ndimage as _ndi
            if _FOOTPRINT_TRIM_PX > 0:   # scipy: iterations<1 erodes until nothing is left
                valid = _ndi.binary_erosion(valid, iterations=_FOOTPRINT_TRIM_PX,
                                            border_value=1)

            overlap = float(np.mean(valid))
            if overlap < Config.MERGE_MIN_OVERLAP:
                raise _UnusableStack(
                    f"{os.path.basename(path)} overlaps only {overlap * 100:.0f}% "
                    f"of the current frame (< {Config.MERGE_MIN_OVERLAP * 100:.0f}%) "
                    f"— refusing to merge (wrong target or failed registration?)")

            # Post-alignment sanity: the aligned stack must actually look like
            # the new one (same stars). Catches wrong-target inputs and silently
            # failed registrations that still produced a plausible footprint.
            al = _lum(aligned)[::4, ::4]
            nl = new_lum[::4, ::4]
            vm = valid[::4, ::4]
            if vm.sum() > 100:
                a_v, n_v = al[vm].astype(np.float64), nl[vm].astype(np.float64)
                if a_v.std() > 1e-9 and n_v.std() > 1e-9:
                    corr = float(np.corrcoef(a_v, n_v)[0, 1])
                else:
                    corr = 0.0
                if corr < Config.MERGE_MIN_CORRELATION:
                    raise _UnusableStack(
                        f"{os.path.basename(path)}: aligned stack correlates only "
                        f"{corr:.2f} with the current stack "
                        f"(< {Config.MERGE_MIN_CORRELATION}) — wrong target or "
                        f"failed registration; refusing to merge")

            cw = _clipped_weight(aligned, grow_px=_CLIP_GROW_PX)
            if cw is not None:
                cw = np.where(valid, cw, 0.0)
                clip_w = cw if clip_w is None else np.maximum(clip_w, cw)
            gains, offsets, n_fit = _match_flux_scale(stacked, aligned, valid)
            surface = _sky_offset_surface(stacked, aligned, valid, gains)
            scaled = aligned.astype(np.float64) * gains + (surface if surface is not None else offsets)
            if np.any(gains != 1.0):
                safe_print(f"    flux scale vs current stack: R x{gains[0]:.3f}  "
                           f"G x{gains[1]:.3f}  B x{gains[2]:.3f} ({n_fit} px in fit)")

            w_map = np.where(valid, w_prev, 0.0)
            acc += scaled * w_map[:, :, np.newaxis]
            wsum += w_map

            if noise_ok:
                prev_sigma = np.array([_pixel_noise(aligned[:, :, c], valid)
                                       for c in range(3)]) * gains
                if np.all(np.isfinite(prev_sigma)) and np.all(prev_sigma > 0):
                    w_c = 1.0 / prev_sigma ** 2
                    if members is not None:
                        members.append((scaled.astype(np.float32), valid, w_c))
                    acc_n +=scaled * np.where(valid[:, :, np.newaxis], w_c, 0.0)
                    wsum_n += np.where(valid[:, :, np.newaxis], w_c, 0.0)
                    info.setdefault('weights', []).append(
                        (os.path.basename(path), float(np.mean(w_c / w_new))))
                else:
                    noise_ok = False

            if tf is not None:
                t_xy = tf.params[:2, 2]
                sy, sx = float(t_xy[1]), float(t_xy[0])
                rot = float(np.degrees(np.arctan2(tf.params[1, 0], tf.params[0, 0])))
            else:
                sy, sx = shift
                rot = 0.0
            safe_print(f"  Merged {os.path.basename(path)}: {meta['nframes']} frames, "
                       f"shift=({sx:+.1f}, {sy:+.1f})px rot={rot:+.2f}deg, "
                       f"overlap {overlap * 100:.0f}%")

            info['n_sources'] += 1
            info['sources'].append(os.path.basename(path))
            info['total_frames'] += meta['nframes']
            info['total_intgtime'] += meta['intgtime']
            info['total_totexp'] += meta['totexp']
            for key, pick in (('datefrst', min), ('datelast', max)):
                v = meta[key]
                if v:
                    info[key] = pick(info[key], str(v)) if info[key] else str(v)
        except _UnusableStack as exc:
            if not skip_unusable:
                raise
            safe_print(f"  WARNING: leaving out {exc}")
            info['skipped'].append(os.path.basename(path))

    if noise_ok and consistency and members is not None and len(members) >= 2:
        merged, flagged = consistency_combine(members)
        info['weighting'] = 'inverse-variance, cross-night consistency'
        info['consistency_flagged'] = flagged
        safe_print(f"  Cross-night consistency: a night left out of {flagged * 100:.3f}% of "
                   f"wavelet coefficients")
    elif noise_ok:
        merged = (acc_n / np.maximum(wsum_n, 1e-30)).astype(np.float32)
        info['weighting'] = 'inverse-variance'
        if proper and members is not None and len(members) >= 2:
            pm = proper_merge(members, merged, clip_w)
            if pm is not None:
                merged = pm
                info['weighting'] = 'proper coadd (per-frequency PSF weights)'
            else:
                safe_print("  Proper merge: a night's PSF could not be measured; "
                           "kept the inverse-variance mean")
    else:
        merged = (acc / np.maximum(wsum[:, :, np.newaxis], 1e-12)).astype(np.float32)
        info['weighting'] = 'frame-count'
    if clip_w is not None:
        sel = clip_w > 0
        w = clip_w[sel][:, None].astype(np.float32)
        px = merged[sel]
        merged[sel] = px * (1 - w) + px.max(axis=1, keepdims=True) * w
        info['neutral_clipped_px'] = int((clip_w >= 1).sum())
    safe_print(f"  Merge weighting: {info['weighting']}")
    return merged, info


def seed_reference_header(header, meta: Dict[str, Any]) -> None:
    """Start a fresh output header from the reference stack's own aggregates.

    ``apply_merge_header`` is written for ``--merge``, where the output header
    already holds *this* session's totals and only the previous stacks are
    added on top -- so it guards every aggregate with "already present". A
    hierarchical combine builds its header from scratch, where nothing is
    present, so every guard skipped: the combined output lost its integration
    time and dates, and -- never having been marked ``RAWSTACK`` -- could not be
    fed back into ``--merge`` or ``--transient-detect``. Both were silent.

    Seeds the reference's own totals (``merge_info`` covers only the *other*
    stacks) and marks the result a linear stack, which it is: a weighted mean
    of linear stacks, before any Phase 4 step.
    """
    header['RAWSTACK'] = (True, 'Linear pre-post-processing stack')
    # Marks the file as this pipeline's output, so pointing a later run at the
    # folder holding it does not pick it up as a light (classify_frame).
    header['CREATOR'] = ('originstack.py', 'Software that created this file')
    header['COMBINED'] = (True, 'Image is a stacked combination')
    if meta.get('intgtime', 0.0) > 0:
        header['INTGTIME'] = (meta['intgtime'],
                              'Total integration time (s)')
        header['INTGMIN'] = (meta['intgtime'] / 60.0, 'Total integration time (minutes)')
    if meta.get('totexp', 0.0) > 0:
        header['TOTEXP'] = (meta['totexp'], 'Total exposure time (s)')
    if meta.get('datefrst'):
        header['DATEFRST'] = str(meta['datefrst'])
    if meta.get('datelast'):
        header['DATELAST'] = str(meta['datelast'])


def copy_celestial_wcs(header, ref_header) -> bool:
    """Copy the reference stack's celestial WCS into ``header``.

    A hierarchical combine lands every session on the reference stack's pixel
    grid, so the reference's WCS describes the combined image exactly. The
    combined header was built from scratch and carried none: astropy then reads
    a 1 deg/px identity WCS, and photometry, annotation and --transient-detect
    on a multi-night stack had no sky coordinates. Returns whether one was copied.
    """
    try:
        from astropy.wcs import WCS
        w = WCS(ref_header, naxis=2)
    except Exception:
        return False
    if not w.has_celestial:
        return False
    for card in w.to_header(relax=True).cards:
        header[card.keyword] = (card.value, card.comment)
    return True


def apply_merge_header(header, info: Dict[str, Any]) -> None:
    """Override the session header aggregates with merged totals."""
    header['NFRAMES'] = (info['total_frames'], 'Number of stacked frames (all merged sessions)')
    header['MERGED'] = (info['n_sources'], 'Number of previous stacks merged in')
    for i, src in enumerate(info['sources'][:20]):
        header[f'MRGSRC{i + 1}'] = (src[:68], 'Merged source stack')
    if info['total_intgtime'] > 0 and 'INTGTIME' in header:
        total = float(header['INTGTIME']) + info['total_intgtime']
        header['INTGTIME'] = (total, 'Total integration time (s)')
        if 'INTGMIN' in header:
            header['INTGMIN'] = (total / 60.0, 'Total integration time (minutes)')
    if info['total_totexp'] > 0 and 'TOTEXP' in header:
        header['TOTEXP'] = (float(header['TOTEXP']) + info['total_totexp'],
                            'Total exposure time (s)')
    if info['datefrst'] and 'DATEFRST' in header:
        header['DATEFRST'] = min(str(header['DATEFRST']), info['datefrst'])
    if info['datelast'] and 'DATELAST' in header:
        header['DATELAST'] = max(str(header['DATELAST']), info['datelast'])
