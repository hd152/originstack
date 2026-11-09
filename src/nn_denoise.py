"""Learned denoiser (``--denoiser nn``): a small U-Net trained with Noise2Noise on real
Origin sessions.

Training needs no clean images. ``tools/gen_n2n_pairs.py`` stacks each session's frames
``i % 4 == q`` into four quarter-stacks: the same sky with independent noise. A network
trained to predict one from another with an L2 loss converges to the noise-free mean,
because the target's own noise is zero-mean and independent of the input
(Lehtinen et al. 2018). ``tools/train_nn_denoise.py`` trains it; the weights ship as
``src/data/nn_denoise.npz`` and run here in plain numpy (no torch at run time).

The network works in noise units: each channel has a smooth background removed and is
divided by its own measured noise sigma (``normalise``, shared with the training tool so
both see identical inputs). That makes it independent of exposure, gain and stack depth,
so one model trained on quarter-stacks applies to a full stack. Everything is undone
exactly after the forward pass; ``strength`` blends towards the input.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
    import astro_native as _native
    _HAS_NATIVE = hasattr(_native, 'nn_conv3x3')
    _HAS_FORWARD = hasattr(_native, 'nn_forward')
except Exception:
    _native = None
    _HAS_NATIVE = False
    _HAS_FORWARD = False

import logging

_log = logging.getLogger("originstack")

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nn_denoise.npz')
# The same network trained on seeing pairs (--deconvolve nn, dev-notes/nn-deconv-runbook.md):
# sharpest-30% stack as target, so it sharpens as well as denoises. Run with nn_denoise().
DECONV_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nn_deconv.npz')

BG_BLOCK = 64          # background mesh cell (px); the surface is smooth on this scale
ASINH_SOFT = 4.0       # network input = asinh(z / ASINH_SOFT): linear near the sky, compressed on stars
TILE = 384             # forward-pass tile (px), a multiple of 4 (two 2x poolings)
TILE_WORKERS = 2          # tiles in flight (denoise_z); the convolutions already use every core
MARGIN = 24            # context on each side of a tile: the network's receptive-field radius (20 px changes the output)
MARGIN_REQUIRED = MARGIN   # what tests/test_nn_denoise.py proves sufficient; MARGIN may not go below it
LEAK = 0.1             # LeakyReLU slope
# Fused multiply-add convolutions (1.4x faster; rounding then differs in the last bits, and on a CPU
# without FMA the plain kernel runs). ORIGINSTACK_NN_FMA=0 selects the plain kernel everywhere.
USE_FMA = os.environ.get('ORIGINSTACK_NN_FMA', '1') != '0'
RESTORE_SIGMA = 4.0    # px; low-frequency restore of faint extended light (nn_denoise)
# Star FWHM (px, circular Gaussian fit) range of the Origin sessions the deconvolver was trained
# on (2.9-5.1 px; the lower bound is lower: an S50 stack at 2.6 px sharpens fine unresampled). Outside it the network sharpens too much (undersampled) or too little
# (oversampled): ``nn_denoise(..., fwhm_range=...)`` resamples so the stars land inside.
TRAIN_FWHM_RANGE = (2.3, 5.2)
SCALE_MAX_UP = 2.5     # largest upsampling factor (cost grows with its square)
STAR_RESTORE_FWHM = (3.6, 4.0)   # px: star-aware restore at full strength below the first, off above the second
FIT_HALF = 8           # star cutout half-width (px) for the FWHM fit

_weights_cache: Dict[str, Dict[str, np.ndarray]] = {}


# ── Normalisation (shared with tools/train_nn_denoise.py) ─────────────────────

def noise_sigma(img: np.ndarray) -> np.ndarray:
    """Per-channel noise sigma of an (H, W, C) image: MAD of lag-4 row differences on every
    2nd row. Lag 4 skips the short-range correlation the debayer and warp put into the noise,
    and differences cancel the signal on all but the sharpest structure."""
    out = np.empty(img.shape[2], np.float64)
    for c in range(img.shape[2]):
        x = img[::2, :, c]
        d = (x[:, 4:] - x[:, :-4]).ravel()
        d = d[np.isfinite(d)]
        out[c] = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0) if d.size else 1.0
    return np.maximum(out, 1e-12)


def background_grid(img: np.ndarray) -> np.ndarray:
    """Per-channel smooth background of an (H, W, C) image on a ``BG_BLOCK`` mesh:
    block medians, lightly smoothed. Evaluate it on any window with ``background_at``."""
    h, w, c = img.shape
    gh, gw = max(1, h // BG_BLOCK), max(1, w // BG_BLOCK)
    crop = img[:gh * BG_BLOCK, :gw * BG_BLOCK]
    blocks = crop.reshape(gh, BG_BLOCK, gw, BG_BLOCK, c).transpose(0, 2, 4, 1, 3).reshape(gh, gw, c, -1)
    grid = np.nanmedian(blocks, axis=-1)
    for k in range(c):
        grid[:, :, k] = ndi.median_filter(grid[:, :, k], size=3, mode='nearest')
        grid[:, :, k] = ndi.gaussian_filter(grid[:, :, k], 1.0, mode='nearest')
    return grid.astype(np.float32)


def background_at(grid: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The background surface over rows ``y0:y0+h``, cols ``x0:x0+w`` (bilinear on the mesh,
    block centres at ``(i + 0.5) * BG_BLOCK``)."""
    gy = (np.arange(y0, y0 + h, dtype=np.float64) + 0.5) / BG_BLOCK - 0.5
    gx = (np.arange(x0, x0 + w, dtype=np.float64) + 0.5) / BG_BLOCK - 0.5
    yy, xx = np.meshgrid(gy, gx, indexing='ij')
    out = np.empty((h, w, grid.shape[2]), np.float32)
    for k in range(grid.shape[2]):
        out[:, :, k] = ndi.map_coordinates(grid[:, :, k], [yy, xx], order=1, mode='nearest')
    return out


def normalise(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(background grid, sigma)`` for an (H, W, C) image; ``z = (img - bg) / sigma``."""
    return background_grid(img), noise_sigma(img)


def to_input(z: np.ndarray) -> np.ndarray:
    """Network input from noise-unit data."""
    return np.arcsinh(z / ASINH_SOFT)


def output_scale(z: np.ndarray) -> np.ndarray:
    """The network predicts its correction in units of the expected local noise,
    ~ sqrt(1 + signal) in sky-sigma units (sky-limited at 1, Poisson above)."""
    return np.sqrt(1.0 + np.maximum(z, 0.0))


# ── Network ──────────────────────────────────────────────────────────────────

def load_weights(path: Optional[str] = None) -> Dict[str, np.ndarray]:
    path = path or _MODEL_PATH
    w = _weights_cache.get(path)
    if w is None:
        with np.load(path) as f:
            w = {k: f[k].astype(np.float32) for k in f.files}
        _weights_cache[path] = w
    return w


def model_available(path: Optional[str] = None) -> bool:
    return os.path.exists(path or _MODEL_PATH)


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray, act: bool = True) -> np.ndarray:
    """'same' 3x3 convolution with zero padding (+ bias, LeakyReLU when ``act``)."""
    if _HAS_NATIVE:
        return _native.nn_conv3x3(np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(w),
                                  np.ascontiguousarray(b), float(LEAK), bool(act))
    return _conv3x3_numpy(x, w, b, act)


def _conv3x3_numpy(x: np.ndarray, w: np.ndarray, b: np.ndarray, act: bool = True) -> np.ndarray:
    """'same' 3x3 convolution with zero padding, as torch.nn.Conv2d(padding=1).
    x: (C, H, W); w: (O, C, 3, 3). One (O, 9C) @ (9C, H*W) product over an im2col buffer:
    2.7x faster than nine (O, C) products, whose inner dimension is too small for BLAS."""
    c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.empty((9, c, h, wd), np.float32)
    for ky in range(3):
        for kx in range(3):
            cols[ky * 3 + kx] = xp[:, ky:ky + h, kx:kx + wd]
    out = w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1) @ cols.reshape(9 * c, -1)
    out += b[:, None]
    out = out.reshape(-1, h, wd)
    if act:
        np.maximum(out, LEAK * out, out=out)
    return out


def _pool2(x: np.ndarray) -> np.ndarray:
    """2x2 average pool (torch AvgPool2d(2)); four strided adds beat a strided mean 3x."""
    s = x[:, ::2, ::2] + x[:, 1::2, ::2]
    s += x[:, ::2, 1::2]
    s += x[:, 1::2, 1::2]
    s *= np.float32(0.25)
    return s


def _up2_cat(x: np.ndarray, skip: np.ndarray) -> np.ndarray:
    """``concat([2x nearest-neighbour upsample of x, skip])`` (torch Upsample(scale_factor=2)
    then torch.cat), written straight into one buffer: four strided copies, no temporaries."""
    c = x.shape[0]
    out = np.empty((c + skip.shape[0],) + skip.shape[1:], np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            out[:c, dy::2, dx::2] = x
    out[c:] = skip
    return out


def forward(x: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """U-Net forward pass on (C, H, W) float32, H and W multiples of 4. Mirrors
    ``NnDenoiseNet`` in tools/train_nn_denoise.py layer for layer."""
    if _HAS_NATIVE and _HAS_FORWARD:
        return _native.nn_forward(np.ascontiguousarray(x, dtype=np.float32), w, float(LEAK), USE_FMA)

    def conv(name, t, act=True):
        return _conv3x3(t, w[name + '.weight'], w[name + '.bias'], act)

    s0 = conv('e0b', conv('e0a', x))
    s1 = conv('e1b', conv('e1a', _pool2(s0)))
    m = conv('mb', conv('ma', _pool2(s1)))
    d1 = conv('d1b', conv('d1a', _up2_cat(m, s1)))
    d0 = conv('d0b', conv('d0a', _up2_cat(d1, s0)))
    out = np.einsum('oc,chw->ohw', w['out.weight'][:, :, 0, 0], d0, optimize=True)
    return out + w['out.bias'][:, None, None]


def denoise_z(z: np.ndarray, w: Dict[str, np.ndarray], workers: Optional[int] = None) -> np.ndarray:
    """Denoise noise-unit data (H, W, C) tile by tile; returns the same shape.

    Tiles are independent, so ``workers`` of them run at once (default ``TILE_WORKERS``):
    the native convolutions release the GIL and spread over all cores, but the
    pool/upsample/concat glue between them is single-threaded, and a second tile in
    flight fills those gaps. Each tile's arithmetic is unchanged, so the result does not
    depend on ``workers``."""
    h, wd, _ = z.shape
    ph, pw = (-h) % TILE, (-wd) % TILE
    zp = np.pad(z, ((MARGIN, MARGIN + ph), (MARGIN, MARGIN + pw), (0, 0)), mode='reflect')
    out = np.empty_like(z)

    def one(pos):
        y0, x0 = pos
        t = zp[y0:y0 + TILE + 2 * MARGIN, x0:x0 + TILE + 2 * MARGIN]
        pred = forward(to_input(t).transpose(2, 0, 1).astype(np.float32), w).transpose(1, 2, 0)
        res = (t + pred * output_scale(t))[MARGIN:MARGIN + TILE, MARGIN:MARGIN + TILE]
        hh, ww = min(TILE, h - y0), min(TILE, wd - x0)
        out[y0:y0 + hh, x0:x0 + ww] = res[:hh, :ww]    # tiles write disjoint blocks

    tiles = [(y0, x0) for y0 in range(0, h, TILE) for x0 in range(0, wd, TILE)]
    n = max(1, min(int(workers or TILE_WORKERS), len(tiles)))
    if n == 1:
        for pos in tiles:
            one(pos)
    else:
        with ThreadPoolExecutor(max_workers=n, thread_name_prefix='nn-tile') as pool:
            for _ in pool.map(one, tiles):    # re-raises a tile's exception
                pass
    if _HAS_NATIVE and _HAS_FORWARD:
        _native.nn_release_buffers()    # the pass kept its activation buffers for the next tile
    return out


_FIT_LO = np.array([0.0, -3.0, -3.0, 0.5, -np.inf])
_FIT_HI = np.array([np.inf, 3.0, 3.0, 8.0, np.inf])
_FIT_CHUNK = 64        # stars fitted per batch: one chunk usually yields the 60 accepted stars


def _fit_gaussians(cuts: np.ndarray, amp0: np.ndarray, bg0: np.ndarray,
                   max_iter: int = 80, rtol: float = 1e-9) -> np.ndarray:
    """Batched bounded Levenberg-Marquardt fit of ``bg + a exp(-((x-dx)^2 + (y-dy)^2) / (2 s^2))``
    to (n, 2 FIT_HALF + 1, 2 FIT_HALF + 1) stamps. Same start ([amp0, 0, 0, 1.8, bg0]) and bounds
    (a >= 0, |dx|, |dy| <= 3, 0.5 <= s <= 8) as the scipy ``least_squares`` fit it replaces, with
    an analytic Jacobian; a parameter pinned at a bound with the gradient pushing outward is held.
    Only the stars still moving are touched each iteration. Returns (n, 5): a, dx, dy, s, bg."""
    n = len(cuts)
    yy, xx = np.mgrid[-FIT_HALF:FIT_HALF + 1, -FIT_HALF:FIT_HALF + 1]
    gx = xx.ravel().astype(np.float64)[None, :]
    gy = yy.ravel().astype(np.float64)[None, :]
    z = np.asarray(cuts, np.float64).reshape(n, -1)
    p = np.empty((n, 5))
    p[:, 0], p[:, 1:3], p[:, 3], p[:, 4] = amp0, 0.0, 1.8, bg0
    lam = np.full(n, 1e-3)
    eye = np.eye(5)

    def model(q):
        ddx, ddy = gx - q[:, 1:2], gy - q[:, 2:3]
        d2 = ddx * ddx + ddy * ddy
        return np.exp(-d2 / (2.0 * q[:, 3:4] ** 2)), ddx, ddy, d2

    act = np.arange(n)
    for _ in range(max_iter):
        if act.size == 0:
            break
        pa, za, la = p[act], z[act], lam[act]
        e, ddx, ddy, d2 = model(pa)
        a, s = pa[:, 0:1], pa[:, 3:4]
        r = pa[:, 4:5] + a * e - za
        cost = 0.5 * np.einsum('ij,ij->i', r, r)
        ae, s2 = a * e, s * s
        jac = np.empty((act.size, r.shape[1], 5))
        jac[:, :, 0] = e
        jac[:, :, 1] = ae * ddx / s2
        jac[:, :, 2] = ae * ddy / s2
        jac[:, :, 3] = ae * d2 / (s2 * s)
        jac[:, :, 4] = 1.0
        jt = jac.transpose(0, 2, 1)
        g = np.matmul(jt, r[:, :, None])[:, :, 0]
        hess = np.matmul(jt, jac)
        frozen = ((pa <= _FIT_LO) & (g > 0)) | ((pa >= _FIT_HI) & (g < 0))
        mat = hess + la[:, None, None] * (np.maximum(np.einsum('ikk->ik', hess), 1e-12)[:, :, None] * eye)
        mat = np.where(frozen[:, :, None] | frozen[:, None, :], eye[None], mat)
        step = np.linalg.solve(mat, np.where(frozen, 0.0, -g)[:, :, None])[:, :, 0]
        pn = np.clip(pa + step, _FIT_LO, _FIT_HI)
        en = model(pn)[0]
        rn = pn[:, 4:5] + pn[:, 0:1] * en - za
        costn = 0.5 * np.einsum('ij,ij->i', rn, rn)
        better = costn < cost
        small = better & ((cost - costn) <= rtol * np.maximum(cost, 1e-30))
        p[act] = np.where(better[:, None], pn, pa)
        lam[act] = np.where(better, np.maximum(la * 0.3, 1e-12), np.minimum(la * 6.0, 1e12))
        act = act[~(small | (lam[act] >= 1e12) | (np.abs(step).max(axis=1) < 1e-10))]
    return p


def _fwhm_lum(img: np.ndarray) -> np.ndarray:
    """The image ``measure_star_fwhm`` fits: the channel mean with NaNs zeroed, minus its median.
    Three float32 channels are summed in the order ``ndarray.mean(axis=2)`` uses (bit-identical,
    3x faster on a 3000 x 2000 stack)."""
    img = np.asarray(img, np.float32) if img.ndim == 3 else img
    if img.ndim == 3 and img.shape[2] == 3:
        lum = img[..., 0] + img[..., 1]
        lum += img[..., 2]
        lum /= np.float32(3)
        if not np.isfinite(lum).all():
            lum = np.nan_to_num(lum, copy=False)
    else:
        lum = np.nan_to_num(img.mean(axis=2) if img.ndim == 3 else img)
    return lum - np.float32(np.median(lum))


def _fwhm_candidates(lum: np.ndarray, max_stars: int):
    """Stars to fit, brightest first, as ``(cuts, amp0, bg0, min_amp)``: isolated (no neighbour
    within 12 px), clear of the border, unsaturated, with positive peak above their edge
    median; at most 600 detections are considered. ``None`` when fewer than 12 are detected."""
    from src.star_detect import detect_stars_matched_filter
    try:
        src = detect_stars_matched_filter(lum)
    except Exception:
        return None
    if src is None or len(src) < 12:
        return None
    src = src[np.argsort(-np.asarray(src['flux']))]
    xs, ys = np.asarray(src['xcentroid'], float), np.asarray(src['ycentroid'], float)
    sat = np.percentile(lum, 99.99) * 0.6
    min_amp = 8.0 * float(noise_sigma(lum[..., None])[0])
    h, w = lum.shape
    cuts, amp0, bg0 = [], [], []
    for i in range(min(len(xs), 600)):
        d2 = (xs - xs[i]) ** 2 + (ys - ys[i]) ** 2
        d2[i] = np.inf
        xi, yi = int(round(xs[i])), int(round(ys[i]))
        if (d2.min() < 12 ** 2 or xi < FIT_HALF + 1 or yi < FIT_HALF + 1
                or xi >= w - FIT_HALF - 1 or yi >= h - FIT_HALF - 1):
            continue
        cut = lum[yi - FIT_HALF:yi + FIT_HALF + 1, xi - FIT_HALF:xi + FIT_HALF + 1].astype(np.float64)
        edge = np.concatenate([cut[0], cut[-1], cut[1:-1, 0], cut[1:-1, -1]])
        bg = float(np.median(edge))
        amp = float(cut.max()) - bg
        if amp <= 0 or cut.max() >= sat:
            continue
        cuts.append(cut)
        amp0.append(amp)
        bg0.append(bg)
    if not cuts:
        return np.empty((0, 2 * FIT_HALF + 1, 2 * FIT_HALF + 1)), np.empty(0), np.empty(0), min_amp
    return np.array(cuts), np.array(amp0), np.array(bg0), min_amp


def measure_star_fwhm(img: np.ndarray, max_stars: int = 60) -> Optional[float]:
    """Median Gaussian FWHM (px) of bright, unsaturated, isolated stars of an (H, W, C)
    image, or None when fewer than 12 usable stars fit. ``quality.measure_fwhm`` (half-max
    area) reads 7.4 px on stacks whose Gaussian width is 4.4 px, so the network's trained
    range is defined in this measure.

    The stars are fitted in batches (``_fit_gaussians``) in brightness order until ``max_stars``
    are accepted, so the same stars are used as by the one-at-a-time scipy fit it replaces
    (``_measure_star_fwhm_scipy``); on five real stacks every accept/reject decision agreed
    and the median FWHM matched to 4 decimals."""
    cand = _fwhm_candidates(_fwhm_lum(img), max_stars)
    if cand is None:
        return None
    cuts, amp0, bg0, min_amp = cand
    out = []
    for i in range(0, len(cuts), _FIT_CHUNK):
        sol = _fit_gaussians(cuts[i:i + _FIT_CHUNK], amp0[i:i + _FIT_CHUNK], bg0[i:i + _FIT_CHUNK])
        ok = (sol[:, 3] < 8.0 - 1e-6) & (sol[:, 0] > min_amp)
        out.extend(2.3548 * sol[ok, 3])
        if len(out) >= max_stars:
            out = out[:max_stars]
            break
    return float(np.median(out)) if len(out) >= 12 else None


def _measure_star_fwhm_scipy(img: np.ndarray, max_stars: int = 60) -> Optional[float]:
    """The original ``measure_star_fwhm``: one scipy ``least_squares`` fit per star (~5 ms each).
    Kept as the reference the batched fitter is tested against."""
    from scipy.optimize import least_squares

    from src.star_detect import detect_stars_matched_filter
    lum = np.nan_to_num(np.asarray(img, np.float32).mean(axis=2) if img.ndim == 3 else img)
    lum = lum - np.float32(np.median(lum))
    try:
        src = detect_stars_matched_filter(lum)
    except Exception:
        return None
    if src is None or len(src) < 12:
        return None
    src = src[np.argsort(-np.asarray(src['flux']))]
    xs, ys = np.asarray(src['xcentroid'], float), np.asarray(src['ycentroid'], float)
    sat = np.percentile(lum, 99.99) * 0.6
    min_amp = 8.0 * float(noise_sigma(lum[..., None])[0])
    yy, xx = np.mgrid[-FIT_HALF:FIT_HALF + 1, -FIT_HALF:FIT_HALF + 1]
    h, w = lum.shape
    out = []
    for i in range(min(len(xs), 600)):
        d2 = (xs - xs[i]) ** 2 + (ys - ys[i]) ** 2
        d2[i] = np.inf
        xi, yi = int(round(xs[i])), int(round(ys[i]))
        if (d2.min() < 12 ** 2 or xi < FIT_HALF + 1 or yi < FIT_HALF + 1
                or xi >= w - FIT_HALF - 1 or yi >= h - FIT_HALF - 1):
            continue
        cut = lum[yi - FIT_HALF:yi + FIT_HALF + 1, xi - FIT_HALF:xi + FIT_HALF + 1].astype(np.float64)
        edge = np.concatenate([cut[0], cut[-1], cut[1:-1, 0], cut[1:-1, -1]])
        bg0 = float(np.median(edge))
        amp0 = float(cut.max()) - bg0
        if amp0 <= 0 or cut.max() >= sat:
            continue

        def resid(p):
            a, dx, dy, sig, bg = p
            return (bg + a * np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * sig ** 2)) - cut).ravel()
        try:
            sol = least_squares(resid, [amp0, 0.0, 0.0, 1.8, bg0], max_nfev=60,
                                bounds=([0, -3, -3, 0.5, -np.inf], [np.inf, 3, 3, 8.0, np.inf]))
        except Exception:
            continue
        if sol.x[3] < 8.0 - 1e-6 and sol.x[0] > min_amp:
            out.append(2.3548 * sol.x[3])
        if len(out) >= max_stars:
            break
    return float(np.median(out)) if len(out) >= 12 else None


def star_mask(img: np.ndarray, fwhm: float) -> Optional[np.ndarray]:
    """(H, W) float32 weight, 1 within ``fwhm`` of a detected star and fading to 0 at 3 x ``fwhm``,
    or None when no stars are found. Marks where ``nn_denoise`` restores with a tight blur."""
    from src.star_detect import detect_stars_matched_filter
    lum = np.nan_to_num(np.asarray(img, np.float32).mean(axis=2))
    lum = lum - np.float32(np.median(lum))
    h, w = lum.shape
    pts = np.zeros((h, w), bool)

    def _mark(src, off=0, band=0):
        if src is None or len(src) == 0:
            return
        y = np.rint(np.asarray(src['ycentroid'])).astype(int) - off
        x = np.rint(np.asarray(src['xcentroid'])).astype(int) - off
        if band:       # padded pass: only stars inside the frame and the detector's border
            keep = (y >= 0) & (y < h) & (x >= 0) & (x < w)
            keep &= (np.minimum(y, h - 1 - y) < band) | (np.minimum(x, w - 1 - x) < band)
            y, x = y[keep], x[keep]
        pts[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)] = True

    try:
        _mark(detect_stars_matched_filter(lum))
    except Exception:
        return None
    # The detector ignores a border of max(cell // 2, 2 * ceil(3 * 5.5 / 2.3548)) px (its
    # default fwhm), so stars there got the plain 4 px restore. Padded second passes find
    # them; only detections inside the frame and that border are added, so the interior (and
    # every first-pass detection) is unchanged. Reflection misses stars 2-5 px from the edge
    # (the star and its mirror image merge into one elongated blob the roundness cut rejects)
    # and replication misses those at 0-1 px, so both run; zero padding found 20x more
    # "stars" on Sculptor (382 vs 17 in the band: a sky step at the edge). Mirror images land
    # outside the frame and are dropped.
    cell = max(8, min(64, min(h, w) // 8))
    border = max(cell // 2, 2 * int(np.ceil(3.0 * 5.5 / 2.3548)))
    pad = border + 8
    if min(h, w) > pad + 1:
        for mode in ('reflect', 'edge'):
            try:
                _mark(detect_stars_matched_filter(np.pad(lum, pad, mode=mode)), pad, border)
            except Exception:
                _log.debug("star_mask: ignored error", exc_info=True)
    if not pts.any():
        return None
    dist = ndi.distance_transform_edt(~pts)
    return np.clip((3.0 * fwhm - dist) / fwhm, 0.0, 1.0).astype(np.float32)


def scale_factor(fwhm: Optional[float], fwhm_range: Tuple[float, float]) -> float:
    """Resampling factor that brings ``fwhm`` into ``fwhm_range`` (the nearest bound), 1.0
    inside it, when unknown, or when it would need more than ``SCALE_MAX_UP`` upsampling."""
    if fwhm is None or not np.isfinite(fwhm) or fwhm <= 0:
        return 1.0
    f = float(np.clip(fwhm, *fwhm_range)) / fwhm
    return 1.0 if f > SCALE_MAX_UP else f


def _per_channel(fn, n: int) -> list:
    """``[fn(0), ..., fn(n-1)]`` with the channels in threads: scipy's ndimage filters and
    resamplers release the GIL, and each channel is independent, so the result is the same
    as the serial loop."""
    if n <= 1:
        return [fn(c) for c in range(n)]
    with ThreadPoolExecutor(max_workers=n, thread_name_prefix='nn-chan') as pool:
        return list(pool.map(fn, range(n)))


def _rescale(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Cubic-spline resample of an (H, W, C) array to ``shape`` (H', W')."""
    zy, zx = shape[0] / a.shape[0], shape[1] / a.shape[1]
    return np.stack(_per_channel(lambda c: ndi.zoom(a[..., c], (zy, zx), order=3, mode='reflect',
                                                   grid_mode=True), a.shape[2]),
                    axis=-1).astype(np.float32)


def nn_denoise(img: np.ndarray, strength: float = 1.0, model_path: Optional[str] = None,
               restore_sigma: float = RESTORE_SIGMA,
               fwhm_range: Optional[Tuple[float, float]] = None,
               star_restore_frac: float = 0.0) -> np.ndarray:
    """Denoise a linear (H, W, 3) image with the bundled Noise2Noise U-Net.

    ``restore_sigma`` (px, 0 = off) adds back the Gaussian low-pass of what the network
    removed: pixel-scale noise stays removed, faint extended light it flattened returns.
    A minimum-MSE denoiser shrinks faint isolated sources toward the sky (they are rare
    in its training data); injected exponential discs of 0.5-3 sky sigma kept 48-79% of
    their flux without it, 86-97% with 4 px, at roughly twice the (still ~20x smaller
    than the wavelet denoiser's) residual error. ``strength`` 0-1 then blends between
    the input (0) and that result (1).

    ``fwhm_range`` (px; ``TRAIN_FWHM_RANGE`` for the deconvolver) makes the step independent
    of pixel scale: when the stars' measured FWHM is outside it the image is resampled so
    they land on the nearest bound, run, and resampled back to the original grid. Inside
    it (all the Origin sessions), or when no FWHM can be measured, nothing is resampled.

    ``star_restore_frac`` (0 = off) restores with a tighter blur, ``frac`` x the stars' FWHM,
    within ~3 FWHM of detected stars. The 4 px restore moves a star's flux from its core into
    an 8-12 px halo (aperture flux at 3 FWHM 0.94-0.96 of the input on 2.7-3 px stars, 1.00 at
    12 px); a blur of ~0.4 FWHM keeps it in the aperture without giving the sharpening back.
    A fixed 1 px did that on 2.6 px stars but undid it on 4.4 px ones (width ratio 0.74 ->
    0.90), and 0.5 px (0.19 FWHM) undid it on 2.6 px ones, so it is a fraction of the width."""
    img = np.asarray(img, dtype=np.float32)
    fw_img = None            # the stars' FWHM measured on ``img`` itself (reused below when ``src`` is ``img``)
    if fwhm_range is not None:
        fw_img = measure_star_fwhm(img)
        f = scale_factor(fw_img, fwhm_range)
        if f != 1.0:
            shape = (max(8, int(round(img.shape[0] * f))), max(8, int(round(img.shape[1] * f))))
            safe = np.where(np.isfinite(img), img, 0.0).astype(np.float32)
            out = _rescale(nn_denoise(_rescale(safe, shape), strength, model_path, restore_sigma,
                                      star_restore_frac=star_restore_frac), img.shape[:2])
            return np.where(np.isfinite(img), out, img).astype(np.float32)
    w = load_weights(model_path)
    finite = np.isfinite(img)
    src = np.where(finite, img, 0.0).astype(np.float32)
    grid, sigma = normalise(src)
    bg = background_at(grid, 0, 0, *src.shape[:2])
    sig = sigma.astype(np.float32)
    z = (src - bg) / sig
    den = denoise_z(z, w) * sig + bg
    if restore_sigma > 0:
        res = src - den
        if star_restore_frac <= 0:
            fw = None
        elif fwhm_range is not None and finite.all():
            fw = fw_img         # ``src`` is ``img`` then: the same measurement, not repeated
        else:
            fw = measure_star_fwhm(src)
        # wide stars (>= ~4 px) lose no flux to the 4 px restore and would lose sharpness
        # to a tight one (Omega width ratio 0.74 -> 0.82): fade the tight restore out. The
        # fade scales the mask WEIGHT, not the blur width: a width of ramp x frac x fw (0.07 px
        # on Pleiades, 4.37 px) puts the residual back unblurred and the stars return to the
        # input (fwhm_ratio 0.73 -> 1.03, Phase 4 bench)
        lo, hi = STAR_RESTORE_FWHM
        ramp = float(np.clip((hi - fw) / (hi - lo), 0.0, 1.0)) if fw else 0.0
        smask = star_mask(src, fw) * np.float32(ramp) if ramp > 0 else None
        def restored(c):
            if smask is None:
                return ndi.gaussian_filter(res[..., c], restore_sigma, mode='reflect')
            return (ndi.gaussian_filter(res[..., c] * (1 - smask), restore_sigma, mode='reflect')
                    + ndi.gaussian_filter(res[..., c] * smask, star_restore_frac * fw, mode='reflect'))
        for c, low in enumerate(_per_channel(restored, res.shape[2])):
            den[..., c] += low
    if src.shape[2] == 3 and np.array_equal(src[..., 0], src[..., 1]) and np.array_equal(src[..., 0], src[..., 2]):
        # a monochrome stack (replicated planes): the network's three outputs differ (up to
        # ~200 ADU on LCO M51 stars) and would put false colour on a grey image
        den[...] = den.mean(axis=2, keepdims=True)
    out = src + np.float32(strength) * (den - src)
    return np.where(finite, out, img).astype(np.float32)
