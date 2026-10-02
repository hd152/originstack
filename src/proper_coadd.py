"""Proper image coaddition (Zackay & Ofek 2017, ApJ 836, 188) -- ``--proper-coadd``.

A weighted mean gives every frame one number for all spatial frequencies, so the
softest frames dilute the fine detail the sharpest ones carry. The proper coadd
weights each frame *per frequency* by its own PSF and noise:

    R_hat = sum_j (F_j / s_j^2) conj(P_hat_j) M_hat_j / sqrt(sum_j (F_j^2 / s_j^2) |P_hat_j|^2)

with M_j the sky-subtracted frame, P_j its PSF (unit sum), F_j its flux scale
(transparency) and s_j its background noise. R is the sufficient statistic for
the stack: white noise, PSF ``P_hat_R = sqrt(sum w_j |P_hat_j|^2 / sum w_j)``
(w_j = F_j^2 / s_j^2) -- the weighted RMS of the frame MTFs, which is never below
their weighted mean (the weighted mean's MTF), so stars come out at least as sharp
as a weighted mean of the same frames. This module returns ``R / F_R`` with
F_R = sqrt(sum w_j): the same flux scale as the frames (F = 1), and at zero
frequency exactly the w-weighted mean of M_j / F_j, so flux is preserved whatever
the PSF model.

Per-frame inputs are measured here, not assumed:

* PSF: an elliptical Moffat with one shape per frame (sigma_x, sigma_y,
  correlation, beta), fitted jointly to ~30 bright, isolated, unsaturated stars at
  the reference stack's centroids (registration has already put every frame on
  that grid); each star's amplitude and background are solved linearly inside
  the fit. One PSF per frame for all three channels.
* F_j: median ratio of the stars' integrated model fluxes to the reference's.
* s_j: per channel, the lag-difference MAD noise of ``merge._pixel_noise``.

Outliers (cosmic rays, satellite trails) cannot be rejected in Fourier space, so
each frame is first cleaned against the normal (sigma-clipped) stack, scaled by
F_j: a sample further than ``k * sqrt(s_j^2 + (0.15 * signal)^2)`` from it is
replaced by that reference value. The signal term is the one ``cfa_drizzle`` uses
(seeing moves star peaks ~15% between frames). Backgrounds are subtracted per frame
and channel and their weighted mean is added back; frames are zero-padded so the
matched filters do not wrap around the edges.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from src.utils import safe_print

_log = logging.getLogger("originstack")

try:
    import astro_native as _native
except Exception:
    _native = None

try:
    import scipy.fft as sfft
    from scipy.optimize import least_squares
    HAS_SCIPY = True
except Exception:          # pragma: no cover - scipy is a hard dependency in practice
    HAS_SCIPY = False

_STAMP_R = 14            # stamp half-size for the PSF fit and aperture flux (29x29)
_MAX_STARS = 30
_PSF_R = 24              # half-size of the PSF kernel rendered for the FFT
_REJECT_K = 5.0
_SIGNAL_FRAC = 0.15
_PAD = 32                # >= _PSF_R: linear, not circular, convolution at the edges


_BETA_MIN, _BETA_MAX = 1.5, 20.0
_FWHM_MAX = 15.0


def _beta(p) -> float:
    return _BETA_MIN + (_BETA_MAX - _BETA_MIN) / (1.0 + np.exp(-p[3]))


def moffat_params(fwhm_x: float, fwhm_y: float, rho: float = 0.0, beta: float = 3.0) -> np.ndarray:
    """Parameter vector for ``_moffat_grid``: (log FWHM_x, log FWHM_y, atanh rho, logit of beta
    within [_BETA_MIN, _BETA_MAX])."""
    t = (beta - _BETA_MIN) / (_BETA_MAX - _BETA_MIN)
    return np.array([np.log(fwhm_x), np.log(fwhm_y), np.arctanh(rho), np.log(t / (1 - t))])


def psf_fwhm(p) -> float:
    """Geometric-mean FWHM of a fitted PSF, px."""
    return float(np.exp(0.5 * (p[0] + p[1])))


def _moffat_grid(dy: np.ndarray, dx: np.ndarray, p) -> np.ndarray:
    """Elliptical Moffat (unnormalised, peak 1).

    Parametrised by FWHM, not the Moffat alpha: near the Gaussian limit (large beta)
    alpha and beta trade off along a long ridge (a real Omega Nebula stack fitted
    alpha 10-12 px at beta 14 for a 4.5 px FWHM), and a bound on alpha then threw
    out good fits; FWHM stays put along that ridge. beta is held to (1.5, 20)."""
    beta = _beta(p)
    k = 2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0)
    sx, sy = np.exp(p[0]) / k, np.exp(p[1]) / k
    rho = np.tanh(p[2])
    u, v = dx / sx, dy / sy
    q = (u * u - 2.0 * rho * u * v + v * v) / max(1.0 - rho * rho, 1e-6)
    return (1.0 + q) ** (-beta)


def select_psf_stars(ref_lum: np.ndarray, fwhm: float) -> np.ndarray:
    """(y, x) centroids of bright, isolated, unsaturated stars in the reference stack."""
    from src.star_detect import detect_stars_matched_filter
    st = detect_stars_matched_filter(ref_lum, fwhm=max(fwhm, 2.0))
    if st is None or len(st) == 0:
        return np.zeros((0, 2))
    H, W = ref_lum.shape
    ys, xs = np.asarray(st['ycentroid'], float), np.asarray(st['xcentroid'], float)
    peak = np.asarray(st['peak'], float)
    m = _STAMP_R + 2
    inside = (ys > m) & (ys < H - m - 1) & (xs > m) & (xs < W - m - 1)
    # unsaturated: well below the frame's own top end
    sat = np.nanpercentile(ref_lum, 99.99)
    ok = inside & (peak < 0.5 * sat)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(-peak[idx])]
    keep = []
    for i in idx:                                       # isolated: nothing bright within 2 stamp radii
        d2 = (ys - ys[i]) ** 2 + (xs - xs[i]) ** 2
        near = (d2 < (2 * _STAMP_R) ** 2) & (d2 > 0) & (peak > 0.2 * peak[i])
        if not near.any():
            keep.append(i)
        if len(keep) >= _MAX_STARS:
            break
    return np.stack([ys[keep], xs[keep]], axis=1) if keep else np.zeros((0, 2))


def star_stamps(img: np.ndarray, stars: np.ndarray):
    """Luminance stamps (n, (2r+1)^2) around ``stars`` from an (H, W) or (H, W, 3) image,
    plus the pixel offsets from each star's centroid -- only the stamps are read, never
    the whole frame."""
    r = _STAMP_R
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    stamps, dys, dxs = [], [], []
    for y, x in stars:
        iy, ix = int(round(y)), int(round(x))
        st = np.asarray(img[iy - r:iy + r + 1, ix - r:ix + r + 1], np.float64)
        if st.ndim == 3:
            st = 0.299 * st[..., 0] + 0.587 * st[..., 1] + 0.114 * st[..., 2]
        stamps.append(st.ravel())
        dys.append((yy - (y - iy)).ravel())
        dxs.append((xx - (x - ix)).ravel())
    return np.array(stamps), np.array(dys), np.array(dxs)


def fit_psf(img: np.ndarray, stars: np.ndarray, p0=None):
    """Joint elliptical-Moffat fit to stamps around ``stars`` (``img`` (H, W) luminance
    or (H, W, 3)).

    Returns (params, fluxes) -- fluxes are per-star aperture sums over the stamp, less
    the fitted background -- or (None, None) when the fit fails or too few stars remain."""
    if len(stars) < 5:
        return None, None
    r = _STAMP_R
    S, DY, DX = star_stamps(img, stars)
    ones = np.ones(S.shape[1])

    def solve(p):
        G = _moffat_grid(DY, DX, p)                     # (n_stars, n_pix)
        # per star: [amp, bg] by least squares on [G, 1]
        gg = (G * G).sum(1)
        g1 = G.sum(1)
        n = S.shape[1]
        gs = (G * S).sum(1)
        s1 = S.sum(1)
        det = gg * n - g1 * g1
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        amp = (gs * n - g1 * s1) / det
        bg = (gg * s1 - g1 * gs) / det
        return G, amp, bg

    def resid(p):
        G, amp, bg = solve(p)
        return (S - amp[:, None] * G - bg[:, None] * ones[None, :]).ravel()

    if p0 is None:
        p0 = moffat_params(4.0, 4.0)
    try:
        res = least_squares(resid, p0, method='lm', max_nfev=200)
    except Exception as exc:
        _log.debug("PSF fit failed: %s", exc)
        return None, None
    p = res.x
    if not np.all(np.isfinite(p)) or max(np.exp(p[0]), np.exp(p[1])) > _FWHM_MAX:
        return None, None
    # flux by aperture (the stamp, less the fitted background), not the model's
    # integral: a smooth model fitted to a trailed star grows heavy wings, and their
    # integral read 2x the real flux on the trailed frames of a real Omega Nebula
    # session -- which, through w = F^2 / s^2, weighted those worst frames 4x
    _, _, bg = solve(p)
    return p, (S - bg[:, None]).sum(1)


def render_psf_into(p, plane: np.ndarray) -> None:
    """The unit-sum PSF written into ``plane``'s four corners, centred on index [0, 0] (the FFT
    origin); the rest of ``plane`` must stay zero."""
    R = _PSF_R
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    k = _moffat_grid(yy, xx, p)
    k /= k.sum()
    plane[:R + 1, :R + 1] = k[R:, R:]
    plane[:R + 1, -R:] = k[R:, :R]
    plane[-R:, :R + 1] = k[:R, R:]
    plane[-R:, -R:] = k[:R, :R]


def _native_ok(aligned) -> bool:
    return (_native is not None and hasattr(_native, 'proper_coadd_prep')
            and hasattr(_native, 'proper_coadd_accum')
            and getattr(aligned, 'dtype', None) == np.float32)


def _prep_numpy(fr, ref_sig, bufs, F, sky, sig, H, W) -> int:
    """numpy mirror of ``proper_coadd_prep`` (bit-identical buffers)."""
    n = 0
    for c in range(bufs.shape[0]):
        ch = np.asarray(fr[..., c], np.float32)
        exp_c = ref_sig[c] * np.float32(F)
        resid = ch - np.float32(sky[c]) - exp_c
        t = np.maximum(exp_c, np.float32(0))
        t *= np.float32(_SIGNAL_FRAC)
        np.square(t, out=t)
        t += np.float32(sig[c] ** 2)
        bad = np.abs(resid) > np.float32(_REJECT_K) * np.sqrt(t)
        inner = bufs[c, _PAD:_PAD + H, _PAD:_PAD + W]
        np.subtract(ch, np.float32(sky[c]), out=inner)
        nb = int(np.count_nonzero(bad))
        if nb:
            inner[bad] = exp_c[bad]
        n += nb
    return n


def _sky(ch: np.ndarray) -> float:
    s = np.asarray(ch[::4, ::4], np.float32)
    return float(np.median(s[np.isfinite(s)]))


def _noise(ch: np.ndarray) -> float:
    """Lag-4 difference MAD noise (as ``merge._pixel_noise``) on every 4th row: it is a
    weight, and a quarter of the rows pins it to well under 1%."""
    rows = np.asarray(ch[::4], np.float32)
    d = (rows[:, 4:] - rows[:, :-4]).ravel()
    d = d[np.isfinite(d)]
    if d.size < 1000:
        return float('nan')
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def proper_coadd(aligned, reference: np.ndarray, fwhm: float = 5.0,
                 workers: Optional[int] = None, verbose: bool = True) -> Optional[np.ndarray]:
    """Proper coadd of ``aligned`` (N, H, W, C) float32 onto ``reference``'s grid.

    ``reference`` is the normal stack of the same frames (rejection reference and PSF
    star positions). Returns float32 (H, W, C), or None when the PSF could not be
    measured (the caller keeps the normal stack)."""
    if not HAS_SCIPY:
        return None
    t0 = time.time()
    N, H, W, C = aligned.shape
    ref = np.asarray(reference, np.float32)
    ref_lum = (0.299 * ref[..., 0] + 0.587 * ref[..., 1] + 0.114 * ref[..., 2]) if C == 3 else ref[..., 0]
    stars = select_psf_stars(ref_lum, fwhm)
    p_ref, flux_ref = fit_psf(ref_lum, stars)
    if p_ref is None:
        _log.info("proper coadd: PSF fit on the reference stack failed")
        return None
    ref_sky = np.array([_sky(ref[..., c]) for c in range(C)])

    # --- per-frame measurements ------------------------------------------------
    def measure(j):
        fr = aligned[j]                                  # a view: only stamps and samples are read
        p, flux = fit_psf(fr if C == 3 else fr[..., 0], stars, p0=p_ref)
        if p is None:
            return None
        sky = np.array([_sky(fr[..., c]) for c in range(C)])
        sig = np.array([_noise(fr[..., c]) for c in range(C)])
        return p, flux, sky, sig

    nw = workers or 8
    with ThreadPoolExecutor(max_workers=nw) as ex:
        raw = list(ex.map(measure, range(N)))
    # Transparency from the frames' own fits: each star's flux over its median across
    # frames, median over stars. Not against the reference stack's fit -- that stack
    # blends frames of different seeing, is no Moffat, and its model flux was biased
    # by ~2% (synthetic field), which went straight into the output's flux scale. The
    # output is therefore on the median frame's scale.
    fit_ok = [j for j, m in enumerate(raw) if m is not None and np.all(np.isfinite(m[3]))
              and np.all(m[3] > 0)]
    meas = [None] * N
    if len(fit_ok) >= 3:
        FL = np.array([raw[j][1] for j in fit_ok])                 # (frames, stars)
        norm = np.nanmedian(np.where(FL > 0, FL, np.nan), axis=0)
        for i, j in enumerate(fit_ok):
            ok = np.isfinite(FL[i]) & (FL[i] > 0) & (norm > 0)
            if ok.sum() >= 5:
                meas[j] = (raw[j][0], float(np.median(FL[i, ok] / norm[ok])), raw[j][2], raw[j][3])
        # the reference stack only feeds the outlier test: map it onto that scale
        okr = (flux_ref > 0) & (norm > 0) & np.isfinite(norm)
        ref_to_frame = float(np.median(norm[okr] / flux_ref[okr])) if okr.sum() >= 5 else 1.0
    else:
        ref_to_frame = 1.0
    use = [j for j, m in enumerate(meas) if m is not None and np.isfinite(m[1]) and m[1] > 0.05]
    if len(use) < 3:
        _log.info("proper coadd: only %d frames with a usable PSF", len(use))
        return None
    if verbose:
        fw = [psf_fwhm(meas[j][0]) for j in use]
        Fs = [meas[j][1] for j in use]
        safe_print(f"    proper coadd: {len(use)}/{N} frames, PSF FWHM {np.min(fw):.2f}-{np.max(fw):.2f} px "
              f"(median {np.median(fw):.2f}), transparency {np.min(Fs):.2f}-{np.max(Fs):.2f}, "
              f"{len(stars)} PSF stars ({time.time() - t0:.1f}s)")

    # --- accumulate in Fourier space ---------------------------------------------
    PH = sfft.next_fast_len(H + 2 * _PAD, real=True)
    PW = sfft.next_fast_len(W + 2 * _PAD, real=True)
    fshape = (PH, PW // 2 + 1)
    num = np.zeros((C,) + fshape, np.complex128)
    den = np.zeros((C,) + fshape, np.float64)
    wsum = np.zeros(C)
    sky_acc = np.zeros(C)
    n_rep = 0
    ref_sig = np.ascontiguousarray(np.stack([(ref[..., c] - np.float32(ref_sky[c]))
                                            * np.float32(ref_to_frame) for c in range(C)]))
    # planes 0..C-1: the frame's channels, zero-padded; plane C: its PSF (FFT origin
    # at [0, 0]). One batched transform per frame.
    planes = np.zeros((C + 1, PH, PW), np.float32)
    native = _native_ok(aligned)
    # one transform per thread: pocketfft's own `workers` did not parallelise these
    # (4 planes 144 ms with any setting; 4 threads 47 ms)
    fft_pool = ThreadPoolExecutor(max_workers=C + 1)
    t1 = time.time()
    for j in use:
        p, F, sky, sig = meas[j]
        fr = np.asarray(aligned[j])
        render_psf_into(p, planes[C])
        sky32 = [float(np.float32(v)) for v in sky]
        sig2 = [float(np.float32(v ** 2)) for v in sig]
        if native:
            n_rep += _native.proper_coadd_prep(fr, ref_sig, planes[:C], float(np.float32(F)), sky32,
                                               sig2, _REJECT_K, _SIGNAL_FRAC, _PAD)
        else:
            n_rep += _prep_numpy(fr, ref_sig, planes[:C], F, sky, sig, H, W)
        spec = list(fft_pool.map(lambda i: sfft.rfft2(planes[i], workers=1), range(C + 1)))
        Ph = spec[C]
        for c in range(C):
            wn, wf = F / sig[c] ** 2, F * F / sig[c] ** 2
            if native:
                _native.proper_coadd_accum(num[c].view(np.float64), den[c], spec[c].view(np.float32),
                                           Ph.view(np.float32), wn, wf)
            else:
                num[c] += wn * (spec[c] * np.conj(Ph))
                den[c] += wf * (Ph.real.astype(np.float64) ** 2 + Ph.imag.astype(np.float64) ** 2)
            wsum[c] += wf
            sky_acc[c] += wf * sky[c] / F
    fft_pool.shutdown()
    out = np.empty((H, W, C), np.float32)
    for c in range(C):
        R = num[c] / np.sqrt(np.maximum(den[c], 1e-300))
        img = sfft.irfft2(R, s=(PH, PW), workers=-1)[_PAD:_PAD + H, _PAD:_PAD + W]
        out[..., c] = (img / np.sqrt(wsum[c]) + sky_acc[c] / wsum[c]).astype(np.float32)
    if verbose:
        safe_print(f"    proper coadd: combined in {time.time() - t1:.1f}s, "
              f"{n_rep / max(len(use) * H * W * C, 1) * 100:.3f}% of samples replaced as outliers")
    return out
