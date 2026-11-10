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

import collections
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from src.utils import diff_mad_sigma, safe_print

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

try:
    # scipy.fft's own pocketfft binding: the same transforms as sfft.rfft2 (which calls
    # exactly this), but it takes ``out`` -- the combine pass FFTs into preallocated
    # buffers instead of allocating (and page-faulting in) ~86 MB of spectra per frame.
    # Private API, so the combine pass falls back to sfft.rfft2 when it is missing.
    from scipy.fft._pocketfft import pypocketfft as _pfft
except Exception:          # pragma: no cover
    _pfft = None

_STAMP_R = 14            # stamp half-size for the aperture flux and background (29x29)
_FIT_R = 8               # core half-size the shape is fitted on (17x17)
_FIT_STARS = 15          # brightest stars used for the shape fit
_MAX_STARS = 30
_PSF_R = 24              # half-size of the PSF kernel rendered for the FFT
_REJECT_K = 5.0
_SIGNAL_FRAC = 0.15
_PAD = 32                # >= _PSF_R: linear, not circular, convolution at the edges
# combine pass, tiled path: frames per accumulator pass, data rows per prep/row-FFT band,
# spectrum columns per tile (see _combine_tiled)
_BATCH = 1
_ROW_BAND = 8
_COL_TILE = 16


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
    S_all, DY_all, DX_all = star_stamps(img, stars)
    # the shape is fitted on the brightest stars' cores only (select_psf_stars returns
    # them brightest first): 6x fewer pixels per model evaluation; the faint stars and
    # the outer stamp add little to four shape parameters but cost most of the time.
    # Fluxes below still use every star's full stamp.
    nf = min(len(stars), _FIT_STARS)
    w = 2 * r + 1
    core = np.zeros((w, w), bool)
    core[r - _FIT_R:r + _FIT_R + 1, r - _FIT_R:r + _FIT_R + 1] = True
    core = core.ravel()
    S, DY, DX = S_all[:nf][:, core], DY_all[:nf][:, core], DX_all[:nf][:, core]
    ones = np.ones(S.shape[1])

    def solve(p, S=S, DY=DY, DX=DX):
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
    _, _, bg = solve(p, S_all, DY_all, DX_all)
    return p, (S_all - bg[:, None]).sum(1)


# Floor under every frame's PSF spectrum (a delta of this weight added at the PSF's centre;
# the spectrum's value at zero frequency is 1). The PSF plane is transformed in float32, whose
# rounding leaves |P| ~1e-8 of noise: for a 5.8 px FWHM star the true spectrum is below that
# from ~0.4 cycles/px on (8.5e-9 at 0.45 against a float32 FFT's 1.0e-8, 7.8e-10 at Nyquist
# against 6.9e-9), so there the coadd weighted its frames by rounding noise. Pure noise either
# way (no signal reaches those frequencies), but a different, equally good noise realization
# for any change in the last digits of the input: 0.2 ADU of added noise moved a 60-frame
# Whirlpool stack by 28 ADU rms, a fifth of its noise, all of it above 0.35 cycles/px. With the
# floor, frequencies the PSF does not reach combine as a whitened inverse-variance mean of the
# frames, and frequencies it does change by at most floor / |P|. research-2026-11.md.
_PSF_FLOOR = 1e-5


def psf_kernel(p) -> np.ndarray:
    """The (2R+1)^2 PSF kernel the combine uses: the unit-sum Moffat plus ``_PSF_FLOOR`` at
    its centre (float64)."""
    R = _PSF_R
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    k = _moffat_grid(yy, xx, p)
    k /= k.sum()
    k[R, R] += _PSF_FLOOR
    return k


def render_psf_into(p, plane: np.ndarray) -> None:
    """``psf_kernel(p)`` written into ``plane``'s four corners, centred on index [0, 0] (the
    FFT origin); the rest of ``plane`` must stay zero."""
    R = _PSF_R
    k = psf_kernel(p)
    plane[:R + 1, :R + 1] = k[R:, R:]
    plane[:R + 1, -R:] = k[R:, :R]
    plane[-R:, :R + 1] = k[:R, R:]
    plane[-R:, -R:] = k[:R, :R]


def _native_ok(aligned) -> bool:
    return (_native is not None and hasattr(_native, 'proper_coadd_prep')
            and hasattr(_native, 'proper_coadd_accum32')
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
    return diff_mad_sigma(d)


def _tiled_ok(aligned) -> bool:
    return (_pfft is not None and _native_ok(aligned)
            and all(hasattr(_native, f) for f in ('proper_coadd_prep_rows', 'proper_coadd_scatter_tiles',
                                                   'proper_coadd_accum32_tile')))


def _combine_tiled(aligned, frames, meas, ref_sig, PH: int, PW: int, tick=None):
    """One combine thread's share of the proper coadd: ``frames`` (in order) accumulated
    into its own float32 sums. Returns (num complex64 (C, PH, NF), den float32, s_w,
    s_w2f, s_sky, n_bad, n_rep) -- bit-identical to the per-frame path in
    ``proper_coadd`` (every FFT is pocketfft's own, every accumulator element takes the
    same f64-add-then-round updates in the same frame order).

    The per-frame path is memory-bound: six threads stopped scaling at ~43 ms per frame
    on a 532-frame session against ~70 ms of single-thread work. Here the traffic is cut
    instead of the work:

    * prep + row FFT in bands of ``_ROW_BAND`` data rows into a small buffer (in cache),
      for ``_BATCH`` frames at once so each reference row is read once per batch;
    * the row spectra are stored column-tiled, (tile, row, ``_COL_TILE`` columns), so
      the column FFT reads contiguous 144 KB blocks, not one 128-byte piece per row;
    * the column FFT of a tile goes to a small buffer and is accumulated from there, B
      frames per pass over the accumulators (also tiled, converted back at the end);
    * all FFT output goes into preallocated buffers (pocketfft's ``out``);
    * a frame in RAM is read in place; a memmapped one is still copied once (the
      sequential read is what made a disk-backed stack fast) into a reused buffer.
    """
    N, H, W, C = aligned.shape
    B, RB, CB, R = _BATCH, _ROW_BAND, _COL_TILE, _PSF_R
    NF = PW // 2 + 1
    NT = -(-NF // CB)
    copy_in = isinstance(aligned, np.memmap)
    psf = np.zeros((PH, PW), np.float32)
    psf_spec = np.empty((2, R + 1, NF), np.complex64)
    tiles = np.zeros((B * (C + 1), NT, PH, CB), np.complex64)   # rows never written stay zero
    tiles_f = tiles.view(np.float32)
    tiles5 = tiles.reshape(B, C + 1, NT, PH, CB)
    a_num = np.zeros((C, NT, PH, CB), np.complex64)
    a_den = np.zeros((C, NT, PH, CB), np.float32)
    s_w = np.zeros(C); s_w2f = np.zeros(C); s_sky = np.zeros(C)
    frbuf = [None] * B
    bufs = {}
    nbad = nrep = 0

    def flush(batch):
        nonlocal nrep
        nbat = len(batch)
        frs = [b[0] for b in batch]
        Fl = [float(np.float32(b[1])) for b in batch]
        skyl = [float(np.float32(v)) for b in batch for v in b[2]]
        s2l = [float(np.float32(v ** 2)) for b in batch for v in b[3]]
        wn = [b[1] / b[3][c] ** 2 for b in batch for c in range(C)]
        wf = [b[1] * b[1] / b[3][c] ** 2 for b in batch for c in range(C)]
        dplanes = [bi * (C + 1) + c for bi in range(nbat) for c in range(C)]
        for y0 in range(0, H, RB):
            nb = min(RB, H - y0)
            key = ('row', nbat, nb)
            if key not in bufs:
                bufs[key] = (np.empty((nbat, C, nb, PW), np.float32),
                             np.empty((nbat * C, nb, NF), np.complex64))
            band, rspec = bufs[key]
            nrep += int(_native.proper_coadd_prep_rows(frs, ref_sig, band, y0, Fl, skyl, s2l,
                                                       _REJECT_K, _SIGNAL_FRAC, _PAD))
            _pfft.r2c(band.reshape(nbat * C, nb, PW), (2,), True, 0, rspec, 1)
            _native.proper_coadd_scatter_tiles(rspec.view(np.float32), tiles_f, dplanes, _PAD + y0)
        for t in range(NT):
            nb = min(CB, NF - t * CB)
            key = ('col', nbat, nb)
            if key not in bufs:
                bufs[key] = np.empty((nbat, C + 1, PH, nb), np.complex64)
            cspec = bufs[key]
            _pfft.c2c(tiles5[:nbat, :, t, :, :nb], (2,), True, 0, cspec, 1)
            _native.proper_coadd_accum32_tile(a_num.view(np.float32), a_den, cspec.view(np.float32), t, wn, wf)

    batch = []
    for j in frames:
        if tick is not None:
            tick()
        p, F = meas[j]
        fr = aligned[j]
        if copy_in or not fr.flags.c_contiguous:
            slot = len(batch)
            if frbuf[slot] is None:
                frbuf[slot] = np.empty((H, W, C), np.float32)
            np.copyto(frbuf[slot], fr)
            fr = frbuf[slot]
        sky, sig = _sky_noise(fr)
        if not (np.all(np.isfinite(sig)) and np.all(sig > 0)):
            nbad += 1
            continue
        # this frame's PSF spectrum rows: only the first R+1 and last R rows are non-zero
        render_psf_into(p, psf)
        q = len(batch) * (C + 1) + C
        _pfft.r2c(psf[:R + 1], (1,), True, 0, psf_spec[0], 1)
        _pfft.r2c(psf[PH - R:], (1,), True, 0, psf_spec[1, :R], 1)
        _native.proper_coadd_scatter_tiles(psf_spec[0:1].view(np.float32), tiles_f, [q], 0)
        _native.proper_coadd_scatter_tiles(psf_spec[1:2, :R].view(np.float32), tiles_f, [q], PH - R)
        for c in range(C):
            wf = F * F / sig[c] ** 2
            s_w[c] += wf
            s_w2f[c] += wf * wf / F
            s_sky[c] += wf * sky[c] / F
        batch.append((fr, F, sky, sig))
        if len(batch) == B:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    num = a_num.transpose(0, 2, 1, 3).reshape(C, PH, NT * CB)[:, :, :NF]
    den = a_den.transpose(0, 2, 1, 3).reshape(C, PH, NT * CB)[:, :, :NF]
    return num, den, s_w, s_w2f, s_sky, nbad, nrep


def _sky_noise(fr: np.ndarray):
    """(sky, noise) per channel of an (H, W, C) frame: ``_sky`` / ``_noise`` per channel, or
    the bit-identical native kernel (numpy's ~8 temporaries per channel were ~40 ms of a
    ~108 ms frame in the combine pass)."""
    if (_native is not None and hasattr(_native, 'frame_sky_noise')
            and fr.dtype == np.float32 and fr.ndim == 3 and fr.flags.c_contiguous):
        sky, sig = _native.frame_sky_noise(fr)
        return np.array(sky), np.array(sig)
    C = fr.shape[-1]
    return (np.array([_sky(fr[..., c]) for c in range(C)]),
            np.array([_noise(fr[..., c]) for c in range(C)]))


class FrameMeasurer:
    """Per-frame PSF / flux / sky / noise measurement on fixed PSF stars, so that it can
    run inside the alignment loop while each freshly warped frame is still in memory
    (re-reading the aligned stack for it cost 22 s on a 531-frame session whose aligned
    stack had gone to disk). Stars come from one aligned frame -- the brightest isolated
    ones are plenty -- and transparency is relative to the median frame, so nothing here
    needs the finished stack."""

    def __init__(self, stars: np.ndarray, p0):
        self.stars = stars
        self.p0 = p0

    def measure(self, frame):
        """(psf params, star fluxes) -- star stamps only, a few MB of a frame. Sky and noise
        need the whole frame and are measured in the combine pass, which reads it anyway
        (on a 531-frame session whose aligned stack was on disk, the strided reads they
        made here cost more than the PSF fits)."""
        C = frame.shape[-1]
        p, flux = fit_psf(frame if C == 3 else frame[..., 0], self.stars, p0=self.p0)
        if p is None:
            return None
        return p, flux


def _progress_ticker(label: str, total: int, cancel_event=None, verbose: bool = True):
    """A thread-safe per-frame callback: app progress, a log line about every 10%
    on long runs, and ``RunCancelled`` once ``cancel_event`` is set. An 849-frame
    session spent 2.4 min here with no output, which on a slower machine looks
    like a hang."""
    import threading
    lock = threading.Lock()
    state = {'n': 0, 't0': time.time(), 'next': max(1, total // 10)}
    from src.ui_events import get_ui_events
    ui = get_ui_events()

    def tick():
        if cancel_event is not None and cancel_event.is_set():
            from src.models import RunCancelled
            raise RunCancelled("cancelled during proper coadd")
        with lock:
            state['n'] += 1
            n = state['n']
            ui.progress(label, n, total)
            if verbose and total >= 100 and n >= state['next']:
                state['next'] += max(1, total // 10)
                el = time.time() - state['t0']
                safe_print(f"    {label}: {n}/{total} frames ({el:.0f}s, "
                           f"~{el / n * (total - n):.0f}s left)")
    return tick


_GPU_MAX_PLANE = 4096    # the GPU kernels transform power-of-two planes of at most this size


def _gpu_plane(n: int) -> int:
    """Smallest power of two holding ``n`` pixels plus the padding (>= 16)."""
    return 1 << int(np.ceil(np.log2(max(n + _PAD, 16))))


def _gpu_unusable(aligned, C: int, H: int, W: int) -> Optional[str]:
    """Why the GPU coadd cannot run on this stack, or None when it can."""
    if _native is None or not hasattr(_native, 'gpu_coadd_begin'):
        return "this astro_native build has no GPU support (needs astro_native 0.66+, built with its default features)"
    if not _native.gpu_available():
        return "no usable GPU adapter"
    # An integrated GPU (shared memory) measured slower than the CPU: 3x on the combine pass of
    # a real session (AMD Radeon, dev-notes/gpu-coadd.md). Naming an adapter or setting
    # ORIGINSTACK_GPU_ALLOW_INTEGRATED=1 opts in.
    info = _native.gpu_info() if hasattr(_native, 'gpu_info') else None
    if (info and 'IntegratedGpu' in str(info) and not os.environ.get('ORIGINSTACK_GPU_ADAPTER')
            and os.environ.get('ORIGINSTACK_GPU_ALLOW_INTEGRATED', '0') in ('', '0')):
        return ("the only usable GPU is an integrated one (%s), which is slower than the CPU here; "
                "set ORIGINSTACK_GPU_ALLOW_INTEGRATED=1 to use it anyway" % info)
    if C != 3:
        return "only 3-channel frames are supported"
    if (getattr(aligned, 'dtype', None) != np.float32 or not isinstance(aligned, np.ndarray)
            or not aligned.flags.c_contiguous):
        return "the aligned stack is not a contiguous float32 array"
    if max(_gpu_plane(H), _gpu_plane(W)) > _GPU_MAX_PLANE:
        return "frames larger than %d px are not supported" % (_GPU_MAX_PLANE - _PAD)
    return None


_GPU_SUSPECT_MS = 100.0         # device ms per frame above which the CPU is timed for comparison
_GPU_SLOW_FACTOR = 1.0          # ... and the GPU is dropped when it is slower than that (the probe reads ~10-15% high)
_SELFCHECK_MIN_FRAMES = 48      # below this the pass is too short to be worth checking


def _print_gpu_profile(tm: dict, st: list, memmap: bool) -> None:
    """ORIGINSTACK_GPU_PROFILE=1: where the GPU combine's time went. The device stages ran one
    by one (each submitted and waited for), so their sum is longer than an unprofiled pass; the
    host numbers are wall time on the calling thread."""
    names = ('upload', 'prep', 'row FFTs', 'transposes', 'column FFTs', 'accumulate')
    n = max(st[6], 1.0)
    dev = sum(st[:6])
    safe_print(f"    GPU profile ({int(st[6])} frames, {'disk' if memmap else 'RAM'}-backed stack): device "
               f"{dev:.1f}s serialised = " + ", ".join(f"{k} {v:.1f}s ({v / n * 1000:.0f} ms/frame)"
                                                       for k, v in zip(names, st[:6])))
    safe_print(f"    GPU profile: host add calls {tm['add']:.1f}s, waiting on host prep "
               f"{tm['wait_host']:.1f}s, finish {tm['finish']:.1f}s; host prep threads (summed): "
               f"frame reads {tm['read']:.1f}s, sky/noise {tm['noise']:.1f}s")


def _combine_gpu(aligned, ref_sig, use, meas, n_thr: int, tick, verbose: bool, cpu_probe=None):
    """The combine pass of ``proper_coadd`` on the GPU (``--gpu-coadd``, src/gpu_coadd.rs).

    Sky and noise per frame (CPU, in a thread pool that runs ahead) overlap the device, which
    rejects, transforms and accumulates the frames in chunks. Returns ``(img (3, H, W),
    wsum, w2_over_f, sky_acc, n_bad)`` -- ``img`` is the raw inverse transform, the caller adds
    the weight scale and the sky -- or None when the GPU cannot be used or fails (then nothing
    has been added to the caller's sums and the CPU path runs). ``RunCancelled`` propagates.
    The sums are local until the end, so a failure part-way leaves no trace.

    ``cpu_probe`` (a callable returning the CPU combine's ms per frame, or None) is the self-check:
    when the device takes more than ``_GPU_SUSPECT_MS`` per frame over frames 9-24 the CPU is timed,
    and the pass returns None (so the CPU runs) if the GPU is clearly slower. A GPU held at a low
    clock by the power plan was 2.5x slower than the CPU, and nothing else notices that."""
    from src.models import RunCancelled
    N, H, W, C = aligned.shape
    why = _gpu_unusable(aligned, C, H, W)
    if why is not None:
        safe_print(f"    --gpu-coadd: {why}; using the CPU")
        return None
    R = _PSF_R
    wsum, w2f, sky_acc = np.zeros(C), np.zeros(C), np.zeros(C)
    used = 0
    add_log = []                        # (frames, seconds) of each add call
    checked = cpu_probe is None or len(use) < _SELFCHECK_MIN_FRAMES
    memmap = isinstance(aligned, np.memmap)
    prof = os.environ.get('ORIGINSTACK_GPU_PROFILE', '0') not in ('', '0')
    tm = {'wait_host': 0.0, 'add': 0.0, 'finish': 0.0, 'read': 0.0, 'noise': 0.0}
    tm_lock = threading.Lock()

    def host_prep(j):
        p, F = meas[j]
        # an in-RAM frame is read in place; a disk-backed one is read once, sequentially, and
        # that copy is what the device gets (``flush``): with the stack larger than the free RAM
        # the page cache cannot keep it for a second read from the memmap
        t_r = time.time()
        fr = np.array(aligned[j]) if memmap else None
        t_n = time.time()
        sky, sig = _sky_noise(fr if memmap else aligned[j])
        if prof:
            with tm_lock:
                tm['read'] += t_n - t_r
                tm['noise'] += time.time() - t_n
        if not (np.all(np.isfinite(sig)) and np.all(sig > 0)):
            return None
        k = psf_kernel(p)
        return j, F, sky, sig, k, fr

    def flush(batch):
        nonlocal used
        if not batch:
            return
        ids, fs, sks, s2s, wns, wfs, sts = [], [], [], [], [], [], []
        for j, F, sky, sig, k, _fr in batch:
            ids.append(j)
            fs.append(float(np.float32(F)))
            sks.append([float(np.float32(v)) for v in sky])
            s2s.append([float(np.float32(v ** 2)) for v in sig])
            wns.append([F / sig[c] ** 2 for c in range(C)])
            wfs.append([F * F / sig[c] ** 2 for c in range(C)])
            sts.append(k.astype(np.float32))
            for c in range(C):
                wsum[c] += F * F / sig[c] ** 2
                w2f[c] += (F * F / sig[c] ** 2) ** 2 / F
                sky_acc[c] += (F * F / sig[c] ** 2) * sky[c] / F
        stamps = np.ascontiguousarray(np.stack(sts))
        t_add = time.time()
        if memmap:      # the frames already read for sky and noise, not a second read of the memmap
            if hasattr(_native, 'gpu_coadd_add_list'):
                _native.gpu_coadd_add_list([b[5] for b in batch], fs, sks, s2s, wns, wfs, stamps)
            else:       # astro_native older than 0.67: stack the frames first (one more copy)
                _native.gpu_coadd_add(np.stack([b[5] for b in batch]), list(range(len(batch))),
                                      fs, sks, s2s, wns, wfs, stamps)
        else:
            _native.gpu_coadd_add(aligned, ids, fs, sks, s2s, wns, wfs, stamps)
        dt = time.time() - t_add
        tm['add'] += dt
        add_log.append((len(batch), dt))
        used += len(batch)

    ex = ThreadPoolExecutor(max_workers=n_thr)
    try:
        _native.gpu_coadd_begin(ref_sig, float(_REJECT_K), float(_SIGNAL_FRAC), _PAD, R,
                                _gpu_plane(H), _gpu_plane(W))
        if verbose:
            safe_print(f"    proper coadd: on the GPU ({_native.gpu_info()})")
        batch = []
        # host prep runs ahead of the device; for a disk-backed stack each result carries a
        # frame (~70 MB), so the read-ahead is bounded
        ahead = n_thr + 8 if memmap else len(use)
        todo = iter(use)
        pending = collections.deque()

        def refill():
            while len(pending) < ahead:
                j = next(todo, None)
                if j is None:
                    return
                pending.append(ex.submit(host_prep, j))

        refill()
        while pending:
            t_w = time.time()
            hp = pending.popleft().result()
            tm['wait_host'] += time.time() - t_w
            refill()
            tick()                                            # progress, and RunCancelled
            if hp is not None:
                batch.append(hp)
            if len(batch) >= 8:
                flush(batch)
                batch = []
                if not checked and len(add_log) >= 3:
                    checked = True
                    n_f = sum(n for n, _ in add_log[1:3])      # the first call includes warm-up
                    g_ms = sum(d for _, d in add_log[1:3]) / max(n_f, 1) * 1000
                    if g_ms > _GPU_SUSPECT_MS:
                        c_ms = cpu_probe()
                        if verbose and c_ms is not None:
                            safe_print(f"    proper coadd: GPU self-check, GPU {g_ms:.0f} ms per frame, "
                                       f"CPU {c_ms:.0f} ms")
                        if c_ms is not None and g_ms > _GPU_SLOW_FACTOR * c_ms:
                            _native.gpu_coadd_abort()
                            safe_print(f"    --gpu-coadd: the GPU takes {g_ms:.0f} ms per frame, the CPU "
                                       f"{c_ms:.0f} ms; using the CPU (on a laptop, a high-performance "
                                       f"power plan keeps the GPU at full clock)")
                            return None
        flush(batch)
        if used == 0:
            raise RuntimeError("no frame had a usable noise level")
        t_f = time.time()
        img = np.asarray(_native.gpu_coadd_finish())
        tm['finish'] = time.time() - t_f
        if prof and hasattr(_native, 'gpu_coadd_profile'):
            _print_gpu_profile(tm, _native.gpu_coadd_profile(), memmap)
    except RunCancelled:
        _native.gpu_coadd_abort()
        raise
    except BaseException as exc:                              # any GPU or driver failure,
        _native.gpu_coadd_abort()                             # a driver panic (pyo3 PanicException) included
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _log.info("proper coadd: the GPU path failed (%s); using the CPU", exc)
        safe_print(f"    --gpu-coadd: failed ({exc}); using the CPU")
        return None
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return img, wsum, w2f, sky_acc, len(use) - used


def proper_coadd(aligned, reference: np.ndarray, fwhm: float = 5.0,
                 workers: Optional[int] = None, verbose: bool = True,
                 stats: Optional[dict] = None, premeasured=None,
                 cancel_event=None, gpu: bool = False) -> Optional[np.ndarray]:
    """Proper coadd of ``aligned`` (N, H, W, C) float32 onto ``reference``'s grid.

    ``reference`` is the normal stack of the same frames (rejection reference and PSF
    star positions). ``premeasured`` = (FrameMeasurer, [measure() result per frame])
    skips the measuring pass (done during alignment). Returns float32 (H, W, C), or
    None when the PSF could not be measured (the caller keeps the normal stack).
    ``cancel_event``: a set ``threading.Event`` stops it with ``RunCancelled``.
    ``gpu``: run the combine pass on the GPU (``--gpu-coadd``) when the native build and the
    stack allow it; any GPU failure falls back to the CPU path below."""
    if not HAS_SCIPY:
        return None
    t0 = time.time()
    N, H, W, C = aligned.shape
    tick_measure = _progress_ticker('Proper coadd: measuring', N, cancel_event, verbose=False)
    ref = np.asarray(reference, np.float32)
    ref_lum = (0.299 * ref[..., 0] + 0.587 * ref[..., 1] + 0.114 * ref[..., 2]) if C == 3 else ref[..., 0]
    if premeasured is not None:
        measurer, raw = premeasured
        stars = measurer.stars
        _, flux_ref = fit_psf(ref_lum, stars, p0=measurer.p0)
        if flux_ref is None:
            flux_ref = np.full(len(stars), np.nan)
    else:
        stars = select_psf_stars(ref_lum, fwhm)
        p_ref, flux_ref = fit_psf(ref_lum, stars)
        if p_ref is None:
            _log.info("proper coadd: PSF fit on the reference stack failed")
            return None
    ref_sky = np.array([_sky(ref[..., c]) for c in range(C)])

    # --- per-frame measurements ------------------------------------------------
    def measure(j):
        tick_measure()
        # stamps and samples only (a float16 stack decodes just those: lazy_frame)
        lazy = getattr(aligned, 'lazy_frame', None)
        return FrameMeasurer(stars, p_ref).measure(lazy(j) if lazy is not None else aligned[j])

    # 2 threads: the fit is ~6 ms of small-array numpy/scipy calls per frame and holds
    # the GIL most of the time -- 8 threads took 9.8 ms per frame, 2 take 5.4 (532-frame
    # session: 8.9 -> ~3 s). Results do not depend on the thread count (ex.map keeps order).
    nw = workers or 2
    if premeasured is None:
        with ThreadPoolExecutor(max_workers=nw) as ex:
            raw = list(ex.map(measure, range(N)))
    # Transparency from the frames' own fits: each star's flux over its median across
    # frames, median over stars. Not against the reference stack's fit -- that stack
    # blends frames of different seeing, is no Moffat, and its model flux was biased
    # by ~2% (synthetic field), which went straight into the output's flux scale. The
    # output is therefore on the median frame's scale.
    fit_ok = [j for j, m in enumerate(raw) if m is not None]
    meas = [None] * N
    if len(fit_ok) >= 3:
        FL = np.array([raw[j][1] for j in fit_ok])                 # (frames, stars)
        norm = np.nanmedian(np.where(FL > 0, FL, np.nan), axis=0)
        for i, j in enumerate(fit_ok):
            ok = np.isfinite(FL[i]) & (FL[i] > 0) & (norm > 0)
            if ok.sum() >= 5:
                meas[j] = (raw[j][0], float(np.median(FL[i, ok] / norm[ok])))
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
    wsum = np.zeros(C)
    w2_over_f = np.zeros(C)     # for the Poisson factor: sum w_j^2 / F_j
    sky_acc = np.zeros(C)
    ref_sig = np.ascontiguousarray(np.stack([(ref[..., c] - np.float32(ref_sky[c]))
                                            * np.float32(ref_to_frame) for c in range(C)]))
    native = _native_ok(aligned)
    # Each thread takes every k-th frame (in order) and does the whole job for it --
    # sky/noise, reject + sky-subtract into zero-padded planes (0..C-1 the channels,
    # C the PSF with its centre on the FFT origin), one FFT per plane -- and adds it into
    # its own float32 accumulators; the per-thread sums are added in a fixed order at
    # the end, so the result does not depend on scheduling. One shared float64
    # accumulator on the main thread used to serialise ~18 ms of every frame.
    import threading
    n_thr = max(2, min(6, (os.cpu_count() or 8) // 2, len(use)))
    lock = threading.Lock()
    n_rep_box = [0]
    acc32 = native

    tick = _progress_ticker('Proper coadd', len(use), cancel_event, verbose=verbose)

    t1 = time.time()

    def worker(k, frames=None, quiet=False):
        planes = np.zeros((C + 1, PH, PW), np.float32)
        a_num = np.zeros((C,) + fshape, np.complex64 if acc32 else np.complex128)
        a_den = np.zeros((C,) + fshape, np.float32 if acc32 else np.float64)
        s_w = np.zeros(C); s_w2f = np.zeros(C); s_sky = np.zeros(C)
        nbad = nrep = 0
        for j in (use[k::n_thr] if frames is None else frames):
            if not quiet:
                tick()
            p, F = meas[j]
            # one sequential read into RAM: the strided sky/noise samples below touched
            # nearly every page of a disk-backed aligned frame once per channel (34 s of
            # a 531-frame session's 37 s proper coadd)
            # (a float16 stack's read is already a private decoded copy)
            fr = aligned[j] if getattr(aligned, 'decodes', False) else np.array(aligned[j])
            sky, sig = _sky_noise(fr)
            if not (np.all(np.isfinite(sig)) and np.all(sig > 0)):
                nbad += 1
                continue
            render_psf_into(p, planes[C])
            if native:
                nrep += int(_native.proper_coadd_prep(fr, ref_sig, planes[:C], float(np.float32(F)),
                                                      [float(np.float32(v)) for v in sky],
                                                      [float(np.float32(v ** 2)) for v in sig],
                                                      _REJECT_K, _SIGNAL_FRAC, _PAD))
            else:
                nrep += _prep_numpy(fr, ref_sig, planes[:C], F, sky, sig, H, W)
            spec = [sfft.rfft2(planes[i], workers=1) for i in range(C + 1)]
            Ph = spec[C]
            for c in range(C):
                wn, wf = F / sig[c] ** 2, F * F / sig[c] ** 2
                if acc32:
                    _native.proper_coadd_accum32(a_num[c].view(np.float32), a_den[c], spec[c].view(np.float32),
                                                 Ph.view(np.float32), wn, wf)
                else:
                    a_num[c] += wn * (spec[c] * np.conj(Ph))
                    a_den[c] += wf * (Ph.real.astype(np.float64) ** 2 + Ph.imag.astype(np.float64) ** 2)
                s_w[c] += wf
                s_w2f[c] += wf * wf / F
                s_sky[c] += wf * sky[c] / F
        del planes
        if not quiet:
            with lock:
                n_rep_box[0] += nrep
        return a_num, a_den, s_w, s_w2f, s_sky, nbad

    def worker_tiled(k):
        *part, nrep = _combine_tiled(aligned, use[k::n_thr], meas, ref_sig, PH, PW, tick=tick)
        with lock:
            n_rep_box[0] += nrep
        return tuple(part)

    probe = None
    if gpu and acc32:
        tiled = _tiled_ok(aligned)

        def probe():
            """ms per frame of the CPU combine (the path that would run), from a warm-up round and a timed
            round of 2 frames per thread."""
            sample = list(use[-4 * n_thr:])
            rounds = [sample[:len(sample) // 2], sample[len(sample) // 2:]]
            t_round = 0.0
            for rnd in rounds:                  # the first round only warms the buffers and FFT plans
                chunks = [c for c in (rnd[i::n_thr] for i in range(n_thr)) if c]
                t0 = time.time()
                with ThreadPoolExecutor(max_workers=len(chunks)) as pex:
                    list(pex.map(lambda c: (_combine_tiled(aligned, c, meas, ref_sig, PH, PW) if tiled
                                            else worker(0, c, quiet=True)), chunks))
                t_round = time.time() - t0
            return t_round / max(len(rounds[-1]), 1) * 1000
    gpu_res = (_combine_gpu(aligned, ref_sig, use, meas, n_thr, tick, verbose, cpu_probe=probe)
               if gpu else None)

    if gpu_res is not None:
        gpu_img, g_wsum, g_w2f, g_sky, n_bad_noise = gpu_res
        wsum += g_wsum
        w2_over_f += g_w2f
        sky_acc += g_sky
        n_rep = None                        # the device does not count replaced samples
    else:
        num = np.zeros((C,) + fshape, np.complex128)
        den = np.zeros((C,) + fshape, np.float64)
        with ThreadPoolExecutor(max_workers=n_thr) as ex:
            parts = list(ex.map(worker_tiled if acc32 and _tiled_ok(aligned) else worker, range(n_thr)))
        n_bad_noise = 0
        for a_num, a_den, s_w, s_w2f, s_sky, nbad in parts:          # fixed order: deterministic
            num += a_num
            den += a_den
            wsum += s_w
            w2_over_f += s_w2f
            sky_acc += s_sky
            n_bad_noise += nbad
        del parts
        n_rep = n_rep_box[0]
    out = np.empty((H, W, C), np.float32)
    for c in range(C):
        if gpu_res is not None:
            img = gpu_img[c]
        else:
            R = num[c] / np.sqrt(np.maximum(den[c], 1e-300))
            img = sfft.irfft2(R, s=(PH, PW), workers=-1)[_PAD:_PAD + H, _PAD:_PAD + W]
        out[..., c] = (img / np.sqrt(wsum[c]) + sky_acc[c] / wsum[c]).astype(np.float32)
    if verbose:
        if n_rep is None:
            safe_print(f"    proper coadd: combined on the GPU in {time.time() - t1:.1f}s")
        else:
            safe_print(f"    proper coadd: combined in {time.time() - t1:.1f}s, "
                       f"{n_rep / max(len(use) * H * W * C, 1) * 100:.3f}% of samples replaced as outliers")
    if stats is not None:
        # Poisson variance of a source of flux f in the output, per unit of the frames'
        # own Poisson coefficient: at zero frequency the output is sum w_j (M_j / F_j) /
        # sum w_j with var(M_j / F_j) = f k / F_j, so var = f k * sum(w^2 / F) / (sum w)^2
        stats['poisson_factor'] = (w2_over_f / np.maximum(wsum, 1e-300) ** 2).tolist()
        stats['frames'] = len(use) - n_bad_noise
    return out
