"""Which luma denoisers earn their place? A ground-truth bake-off.

Runs every ``--denoiser`` backend (plus the chain ``--auto`` actually
produces) on synthetic astro scenes whose clean image is known exactly, and
scores each against that truth. Two numbers per denoiser:

  * **default**  -- called exactly as ``postprocess.py`` calls it with the
    stock arguments (what a user gets without tuning);
  * **tuned**    -- the best result over a sweep of its main strength knob,
    chosen by true faint-region RMSE among settings that keep >= half the
    fine structure (``DETAIL_FLOOR``) -- each method's ceiling, so a weak
    default doesn't get a method dropped unfairly.

NLM, MMT and BM3D were benchmarked with this script and then removed (2026-09):
NLM halved star peaks (11x input error in star cores, 17 s/MP), MMT kept ~7% of
fine structure at its default, BM3D was best but 14 s/MP and licence-encumbered.
To re-test one, restore it from git history and add it back to ``DENOISERS``.

Scenes are built the way a real stack's noise is: clean RGB -> Bayer mosaic
-> signal-dependent noise -> Malvar debayer. The truth is the debayered
*clean* mosaic, so the only difference between input and truth is noise --
and that noise is spatially correlated by the demosaic, as in real stacks
(on the repo's real stacks the pixel-difference sigma is 30-70% of the
global sigma; white noise would make every denoiser look better than it is).

Images are in the post-pedestal domain most denoisers see in Phase 4: sky at
~8 sigma, star mask from the same ``detect_stars_auto``/``generate_star_mask``
calls ``postprocess.py`` makes, passed only to the denoisers the pipeline
passes it to.

Metrics (lower is better unless stated):
  rmse      RMSE(out - truth) / RMSE(in - truth), all pixels + channels.
            1.00 = no improvement; >1 = made it worse.
  sky       same ratio on empty sky -- raw noise suppression.
  faint     same ratio on faint extended signal (0.3-5 sigma) -- the region
            that decides whether a denoiser is any good for astro.
  detail    high-pass retention on faint signal: regression slope of the
            output's fine structure on the truth's (1.00 = filaments kept,
            0.5 = half their contrast smoothed away). Higher is better.
  chroma    chroma-error ratio on sky (colour mottle).
  star_err  error ratio inside bright-star footprints (1.0 = as noisy as input).
  peak      median peak-brightness retention of the 20 brightest stars (1.0 = intact).
  s/MP      runtime per megapixel on this machine.

Usage:
  python tools/bench_denoise_quality.py              # full run
  python tools/bench_denoise_quality.py --quick      # one scene, no sweeps
  python tools/bench_denoise_quality.py --real-dir . --out-dir bench_out
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.denoising as dn  # noqa: E402
from src.debayer import debayer_malvar  # noqa: E402
from src.quality import detect_stars_auto, generate_star_mask  # noqa: E402

try:
    from astropy.stats import sigma_clipped_stats
except ImportError:  # pragma: no cover - astropy is a core dependency
    sigma_clipped_stats = None

LUMA = np.array([0.299, 0.587, 0.114])
FWHM = 3.5
SIZE = 512
DETAIL_FLOOR = 0.5   # tuned pick must keep >= this fraction of fine structure


# ---------------------------------------------------------------------------
# Scene synthesis
# ---------------------------------------------------------------------------

def _mosaic(rgb: np.ndarray) -> np.ndarray:
    raw = np.empty(rgb.shape[:2], dtype=np.float64)
    raw[0::2, 0::2] = rgb[0::2, 0::2, 0]
    raw[0::2, 1::2] = rgb[0::2, 1::2, 1]
    raw[1::2, 0::2] = rgb[1::2, 0::2, 1]
    raw[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return raw


def _moffat_stars(h: int, w: int, n: int, rng, peak_lo: float, peak_hi: float
                  ) -> np.ndarray:
    """Moffat (beta=3) stars, power-law peaks, blackbody-ish colours."""
    beta = 3.0
    alpha = FWHM / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    out = np.zeros((h, w, 3))
    ys = rng.uniform(8, h - 8, n)
    xs = rng.uniform(8, w - 8, n)
    # Pareto-ish: many faint, few bright
    u = rng.uniform(0, 1, n)
    peaks = peak_lo * (peak_hi / peak_lo) ** (u ** 3)
    colours = np.array([[1.0, 0.85, 0.6], [1.0, 0.95, 0.9], [0.8, 0.9, 1.0],
                        [1.0, 0.7, 0.45], [0.7, 0.85, 1.0]])
    r = 15
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    for y, x, p in zip(ys, xs, peaks):
        iy, ix = int(y), int(x)
        dy, dx = y - iy, x - ix
        prof = p * (1.0 + ((yy - dy) ** 2 + (xx - dx) ** 2) / alpha ** 2) ** -beta
        y0, y1 = max(iy - r, 0), min(iy + r + 1, h)
        x0, x1 = max(ix - r, 0), min(ix + r + 1, w)
        sub = prof[y0 - (iy - r):y1 - (iy - r), x0 - (ix - r):x1 - (ix - r)]
        c = colours[rng.integers(len(colours))]
        out[y0:y1, x0:x1] += sub[..., None] * c
    return out


def _filaments(h: int, w: int, n: int, rng, amp: float) -> np.ndarray:
    """Thin curved filaments: smoothed random walks, rasterised then blurred."""
    img = np.zeros((h, w))
    for _ in range(n):
        steps = 600
        ang = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, 0.05, steps))
        y = rng.uniform(0, h) + np.cumsum(np.sin(ang))
        x = rng.uniform(0, w) + np.cumsum(np.cos(ang))
        ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        a = amp * rng.uniform(0.4, 1.0)
        np.add.at(img, (y[ok].astype(int), x[ok].astype(int)), a)
    return ndimage.gaussian_filter(img, 1.1) * (2 * np.pi * 1.1 ** 2) / 2.0


def _scene_nebula(rng, s: float) -> Tuple[np.ndarray, np.ndarray]:
    h = w = SIZE
    field = ndimage.gaussian_filter(rng.normal(size=(h, w)), 28)
    field = (field - field.mean()) / field.std()
    diffuse = 2.2 * s * np.exp(0.9 * field) * (field > -1.2)
    diffuse = ndimage.gaussian_filter(diffuse, 3)
    fil = _filaments(h, w, 14, rng, 2.5 * s)
    oiii = ndimage.gaussian_filter(rng.normal(size=(h, w)), 40)
    oiii = np.clip((oiii - oiii.mean()) / oiii.std(), 0, None) * 1.2 * s
    ext = (diffuse[..., None] * [1.0, 0.25, 0.18] + fil[..., None] * [1.0, 0.3, 0.2]
           + oiii[..., None] * [0.1, 0.65, 0.75])
    stars = _moffat_stars(h, w, 260, rng, 1.5 * s, 250 * s)
    return ext, stars


def _scene_galaxy(rng, s: float) -> Tuple[np.ndarray, np.ndarray]:
    h = w = SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cy, cx, inc, pa = h / 2 + 7, w / 2 - 11, 0.55, 0.6
    dy, dx = yy - cy, xx - cx
    u = dx * np.cos(pa) + dy * np.sin(pa)
    v = (-dx * np.sin(pa) + dy * np.cos(pa)) / inc
    r = np.hypot(u, v) + 1e-6
    th = np.arctan2(v, u)
    disk = 14 * s * np.exp(-r / 55.0)
    arms = 1.0 + 0.7 * np.cos(2 * (th - np.log(r) / np.tan(np.radians(18))))
    bulge = 60 * s * np.exp(-7.67 * ((r / 14.0) ** 0.25 - 1)) / np.exp(7.67)
    halo = 0.8 * s * np.exp(-r / 140.0)
    lane = 1.0 - 0.55 * np.exp(-((v - 9) / 3.0) ** 2) * (np.abs(u) < 150)
    knots = _moffat_stars(h, w, 90, rng, 0.8 * s, 4 * s) * (disk[..., None] > 1.5 * s)
    lum = (disk * arms + halo) * lane
    ext = (lum[..., None] * [0.75, 0.85, 1.0] + (bulge * lane)[..., None] * [1.0, 0.85, 0.6]
           + knots * [0.8, 0.9, 1.3])
    stars = _moffat_stars(h, w, 150, rng, 1.5 * s, 250 * s)
    return ext, stars


def _scene_starfield(rng, s: float) -> Tuple[np.ndarray, np.ndarray]:
    h = w = SIZE
    field = ndimage.gaussian_filter(rng.normal(size=(h, w)), 60)
    ext = np.clip(field / field.std(), 0, None)[..., None] * 0.6 * s * [0.9, 0.85, 0.8]
    stars = _moffat_stars(h, w, 900, rng, 1.0 * s, 300 * s)
    return ext, stars


SCENES = {'nebula': _scene_nebula, 'galaxy': _scene_galaxy, 'starfield': _scene_starfield}


def make_case(scene: str, sigma_mosaic: float, seed: int, scale: float = 60.0,
              n_subs: int = 6) -> dict:
    """Build (noisy, truth, masks...) for one scene at one noise level.

    Scene amplitudes are fixed in ADU (``scale``, ~ a typical stack sky
    sigma) so a higher ``sigma_mosaic`` is a genuinely lower-SNR stack,
    and ADU-denominated defaults (aniso's kappa=30) are exercised at a
    realistic absolute scale."""
    rng = np.random.default_rng(seed)
    ext, stars = SCENES[scene](rng, scale)
    sky = 8.0 * sigma_mosaic
    clean = sky + ext + stars
    raw_clean = _mosaic(clean)
    obj = np.clip(raw_clean - sky, 0, None)
    gain = 20.0 / sigma_mosaic      # object at 20 sigma doubles the variance
    truth = debayer_malvar(raw_clean.astype(np.float32)).astype(np.float64)
    # Stack of n_subs dithered subs: each sub's noise is demosaiced, then
    # resampled by a sub-pixel registration shift (cubic), then averaged.
    # Scaled by sqrt(n_subs) so sigma_mosaic stays the stack's noise level.
    # This lands the pixel-difference/global sigma ratio near the real
    # stacks' (0.3-0.7); demosaic alone gives ~0.86 and flatters denoisers.
    std = np.sqrt(sigma_mosaic ** 2 + obj / gain)
    acc = np.zeros_like(truth)
    for _ in range(n_subs):
        sub = debayer_malvar((rng.normal(size=raw_clean.shape) * std).astype(np.float32))
        acc += ndimage.shift(sub.astype(np.float64), (*rng.uniform(-1, 1, 2), 0),
                             order=3, mode='reflect')
    noisy = (truth + acc / np.sqrt(n_subs)).astype(np.float32)

    ext_l = (ext @ LUMA)
    star_l = (stars @ LUMA)
    err_l = (noisy.astype(np.float64) - truth) @ LUMA
    sigma_in = float(np.std(err_l[(ext_l < 0.05 * scale) & (star_l < 0.05 * scale)]))
    star_region = ndimage.binary_dilation(star_l > 0.3 * sigma_in, iterations=2)
    sky_m = (ext_l < 0.2 * sigma_in) & ~star_region
    faint_m = (ext_l >= 0.3 * sigma_in) & (ext_l <= 5.0 * sigma_in) & ~star_region
    # Stars for photometry: isolated, clearly detected
    lab, n = ndimage.label(star_l > 0.3 * sigma_in)
    return dict(scene=scene, sigma_mosaic=sigma_mosaic, noisy=noisy, truth=truth,
                ext_l=ext_l, star_l=star_l, sigma_in=sigma_in, sky=sky_m,
                faint=faint_m, star_lab=lab, star_n=n,
                star_mask=pipeline_star_mask(noisy))


def pipeline_star_mask(img: np.ndarray) -> Optional[np.ndarray]:
    """Same star-mask construction as postprocess.py's Phase 4 setup."""
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    try:
        _, med, std = sigma_clipped_stats(lum, sigma=3.0, maxiters=5)
        src = detect_stars_auto(lum, float(std), background=float(med))
        if src is not None and len(src) > 0:
            return generate_star_mask(lum.shape, src, fwhm=4.0)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _chroma(x: np.ndarray) -> np.ndarray:
    cb = -0.16875 * x[..., 0] - 0.33126 * x[..., 1] + 0.5 * x[..., 2]
    cr = 0.5 * x[..., 0] - 0.41869 * x[..., 1] - 0.08131 * x[..., 2]
    return np.stack([cb, cr], -1)


def score(out: np.ndarray, case: dict) -> Dict[str, float]:
    truth, noisy = case['truth'], case['noisy'].astype(np.float64)
    out = out.astype(np.float64)
    e_out, e_in = out - truth, noisy - truth

    def ratio(m):
        return float(np.sqrt(np.mean(e_out[m] ** 2)) / np.sqrt(np.mean(e_in[m] ** 2)))

    allm = np.ones(truth.shape[:2], bool)
    res = dict(rmse=ratio(allm), sky=ratio(case['sky']), faint=ratio(case['faint']))

    # Fine-structure retention on faint signal
    tl, ol = truth @ LUMA, out @ LUMA
    hp_t = tl - ndimage.gaussian_filter(tl, 3.0)
    hp_o = ol - ndimage.gaussian_filter(ol, 3.0)
    f = case['faint']
    den = float(np.sum(hp_t[f] ** 2))
    res['detail'] = float(np.sum(hp_o[f] * hp_t[f]) / den) if den > 0 else float('nan')

    s = case['sky']
    co, ci = _chroma(e_out)[s], _chroma(e_in)[s]
    res['chroma'] = float(np.sqrt(np.mean(co ** 2)) / np.sqrt(np.mean(ci ** 2)))

    # Star damage: error inside bright-star footprints (ratio to input error) and
    # median peak retention of the 20 brightest stars (1.0 = untouched).
    lab, n = case['star_lab'], case['star_n']
    core = ndimage.binary_dilation(case['star_l'] > 5 * case['sigma_in'], iterations=2)
    res['star_err'] = float(np.sqrt(np.mean(e_out[core] ** 2))
                            / np.sqrt(np.mean(e_in[core] ** 2))) if core.any() else float('nan')
    if n:
        idx = np.arange(1, n + 1)
        top = np.argsort(-ndimage.maximum(tl, lab, idx))[:20]
        res['peak'] = float(np.median([ol[lab == k + 1].max() / tl[lab == k + 1].max()
                                       for k in top]))
    else:
        res['peak'] = float('nan')
    return res


# ---------------------------------------------------------------------------
# Denoisers, called the way postprocess.py calls them
# ---------------------------------------------------------------------------

def _quiet(fn: Callable, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _aniso_after(first: Callable) -> Callable:
    """--auto nebula/PN preset chain: primary, then aniso(option=2, 15 it)."""
    def run(img, mask, p):
        x = first(img, mask, p)
        return dn.anisotropic_diffusion(x, iterations=15, kappa=30.0, gamma=0.1,
                                        option=2, star_mask=mask)
    return run


# name -> (callable(img, mask, param), default_param, sweep grid, reachable-as)
DENOISERS: Dict[str, Tuple[Callable, object, Sequence, str]] = {
    'none': (lambda im, m, p: im, None, [], 'baseline'),
    'wavelet': (
        lambda im, m, p: dn.directional_wavelet_denoise(
            im, star_mask=m, protect_strength=0.0, variance_stabilize=p),
        False, [False, True], '--denoiser wavelet'),
    'curvelet': (
        lambda im, m, p: dn.directional_wavelet_denoise(
            im, star_mask=m, protect_strength=p[0], variance_stabilize=p[1]),
        (0.6, False), [(s, v) for s in (0.0, 0.3, 0.6, 0.9) for v in (False, True)],
        '--denoiser curvelet (DEFAULT)'),
    'acdnr': (
        lambda im, m, p: dn.acdnr_denoise(im, smoothing_sigma=p[0], contrast_k=p[1],
                                          star_mask=m),
        (1.5, 3.0), [(s, k) for s in (1.0, 1.5, 2.5) for k in (1.5, 3.0, 5.0)],
        '--denoiser acdnr'),
    'bilateral': (lambda im, m, p: dn.bilateral_denoise(
        im, sigma_color=(None if p[0] is None else p[0] * dn._estimate_sky_sigma(im)),
        sigma_space=p[1]), (None, 3.0),
        [(c, s) for c in (1.0, 2.0, 4.0) for s in (1.5, 3.0)], '--denoiser bilateral'),
    'aniso': (lambda im, m, p: dn.anisotropic_diffusion(
        im, iterations=p[0], kappa=(30.0 if p[1] is None else p[1] * dn._estimate_sky_sigma(im)),
        gamma=0.1, option=p[2], star_mask=m), (20, None, 1),
        [(it, k, o) for it in (10, 20, 40) for k in (1.0, 2.0, 4.0) for o in (1, 2)],
        '--denoiser aniso'),
    # What --auto runs for emission/reflection nebulae (rule 14 keeps
    # preset-enabled aniso on top of the primary)
    'chain:curvelet+aniso': (_aniso_after(
        lambda im, m, p: dn.directional_wavelet_denoise(im, star_mask=m)), None, [],
        '--auto emission/reflection nebula'),
}


def run_one(name: str, case: dict, param) -> Tuple[Dict[str, float], float, np.ndarray]:
    fn = DENOISERS[name][0]
    t0 = time.perf_counter()
    out = _quiet(fn, case['noisy'], case['star_mask'], param)
    dt = time.perf_counter() - t0
    out = np.nan_to_num(np.asarray(out, dtype=np.float64))
    mp = case['noisy'].shape[0] * case['noisy'].shape[1] / 1e6
    return score(out, case), dt / mp, out


# ---------------------------------------------------------------------------
# Real-data visual comparison
# ---------------------------------------------------------------------------

def _load_real_crop(path: str, cy: float, cx: float, size: int = SIZE) -> np.ndarray:
    from astropy.io import fits
    d = np.moveaxis(fits.getdata(path).astype(np.float32), 0, -1)
    h, w = d.shape[:2]
    y0 = int(np.clip(cy * h - size / 2, 0, h - size))
    x0 = int(np.clip(cx * w - size / 2, 0, w - size))
    c = d[y0:y0 + size, x0:x0 + size].copy()
    # Approximate Phase 4's pre-denoise state: sky removed, then the +8 sigma pedestal
    for ch in range(3):
        _, med, _ = sigma_clipped_stats(c[..., ch], sigma=3.0, maxiters=5)
        c[..., ch] -= med
    lum = c @ LUMA.astype(np.float32)
    _, med, std = sigma_clipped_stats(lum, sigma=3.0, maxiters=5)
    return c + np.float32(max(8 * std - med, 0.0))


def _stretch(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    y = np.arcsinh(y * 40) / np.arcsinh(40)
    return (y * 255).astype(np.uint8)


def real_grid(real_dir: str, out_dir: str, names: Sequence[str]) -> List[str]:
    from PIL import Image, ImageDraw
    crops = [
        ('Eagle Nebula_stacked.fits', 0.45, 0.5, 'eagle'),
        ('Fireworks_Galaxy_2025-09-24_19-39-48_stacked.fits', 0.5, 0.5, 'fireworks'),
        ('flamingstar2b.fits', 0.5, 0.5, 'flamingstar'),
    ]
    written = []
    for fname, cy, cx, tag in crops:
        path = os.path.join(real_dir, fname)
        if not os.path.exists(path):
            continue
        img = _load_real_crop(path, cy, cx)
        mask = pipeline_star_mask(img)
        lum = img @ LUMA.astype(np.float32)
        lo = float(np.percentile(lum, 1))
        hi = float(np.percentile(lum, 99.7))
        tiles = [('input', img)]
        for n in names:
            fn, dflt = DENOISERS[n][0], DENOISERS[n][1]
            tiles.append((n, np.asarray(_quiet(fn, img, mask, dflt), np.float32)))
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        tile = 360
        sheet = Image.new('RGB', (cols * tile, rows * (tile + 22)), (18, 18, 18))
        for i, (n, x) in enumerate(tiles):
            c0 = SIZE // 2 - tile // 2
            im = Image.fromarray(_stretch(x[c0:c0 + tile, c0:c0 + tile], lo, hi))
            r, c = divmod(i, cols)
            sheet.paste(im, (c * tile, r * (tile + 22) + 22))
            ImageDraw.Draw(sheet).text((c * tile + 6, r * (tile + 22) + 5), n,
                                       fill=(230, 230, 230))
        p = os.path.join(out_dir, f'real_{tag}.png')
        sheet.save(p)
        written.append(p)
    return written


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--quick', action='store_true', help='one case, defaults only')
    ap.add_argument('--only', default=None, help='comma-separated denoiser names')
    ap.add_argument('--out-dir', default='bench_denoise_out')
    ap.add_argument('--real-dir', default=None,
                    help='directory holding the repo sample stacks for visual crops')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    names = list(DENOISERS) if not a.only else a.only.split(',')
    levels = [('clean', 30.0), ('noisy', 100.0)]
    scenes = ['nebula'] if a.quick else list(SCENES)
    if a.quick:
        levels = levels[1:]

    print(f"native={dn._HAS_NATIVE}")
    results = []
    for scene in scenes:
        for lvl, sig in levels:
            case = make_case(scene, sig, seed=hash((scene, lvl)) % 2 ** 31)
            print(f"\n=== {scene}/{lvl}  input sky sigma(luma)={case['sigma_in']:.1f} ADU  "
                  f"stars={case['star_n']}  faint px={int(case['faint'].sum())}")
            print(f"{'denoiser':24s} {'mode':7s} {'param':>16s} {'rmse':>6s} {'sky':>6s} "
                  f"{'faint':>6s} {'detail':>7s} {'chroma':>7s} {'star_err':>8s} {'peak':>5s} {'s/MP':>7s}")
            for n in names:
                _, dflt, grid, _ = DENOISERS[n]
                runs = [('default', dflt)]
                if not a.quick:
                    runs += [('sweep', p) for p in grid]
                best = None
                for mode, p in runs:
                    try:
                        m, spm, _ = run_one(n, case, p)
                    except Exception as exc:  # report, don't abort the bake-off
                        print(f"{n:24s} {mode:7s} {str(p):>16s}  FAILED: {exc}")
                        continue
                    row = dict(scene=scene, level=lvl, denoiser=n, mode=mode,
                               param=str(p), spmp=spm, **m)
                    results.append(row)
                    if mode == 'default':
                        _print_row(row)
                    # Tuned pick: lowest faint-region error among settings that
                    # keep >= half the fine structure. Unconstrained MSE on a
                    # low-SNR faint region rewards erasing filaments outright
                    # (their energy is small next to the noise), which no
                    # imager would accept.
                    elif m['detail'] >= DETAIL_FLOOR and (
                            best is None or m['faint'] < best['faint']):
                        best = row
                if best is not None:
                    b = dict(best, mode='tuned')
                    results.append(b)
                    _print_row(b)

    with open(os.path.join(a.out_dir, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=1)
    _summary(results)

    if a.real_dir:
        vis = list(names)
        for p in real_grid(a.real_dir, a.out_dir, vis):
            print(f"wrote {p}")


def _print_row(r: dict) -> None:
    print(f"{r['denoiser']:24s} {r['mode']:7s} {r['param'][:16]:>16s} {r['rmse']:6.3f} "
          f"{r['sky']:6.3f} {r['faint']:6.3f} {r['detail']:7.3f} {r['chroma']:7.3f} "
          f"{r['star_err']:8.3f} {r['peak']:5.2f} {r['spmp']:7.2f}")


def _summary(results: List[dict]) -> None:
    print("\n=== Mean over all scenes/levels ===")
    print(f"{'denoiser':24s} {'mode':7s} {'rmse':>6s} {'sky':>6s} {'faint':>6s} "
          f"{'detail':>7s} {'chroma':>7s} {'star_err':>8s} {'peak':>5s} {'s/MP':>7s}")
    keys = ['rmse', 'sky', 'faint', 'detail', 'chroma', 'star_err', 'peak', 'spmp']
    for mode in ('default', 'tuned'):
        rows = []
        for n in DENOISERS:
            sel = [r for r in results if r['denoiser'] == n and r['mode'] == mode]
            if sel:
                rows.append((n, {k: float(np.nanmean([r[k] for r in sel])) for k in keys}))
        for n, m in sorted(rows, key=lambda t: t[1]['faint']):
            print(f"{n:24s} {mode:7s} {m['rmse']:6.3f} {m['sky']:6.3f} {m['faint']:6.3f} "
                  f"{m['detail']:7.3f} {m['chroma']:7.3f} {m['star_err']:8.3f} {m['peak']:5.2f} {m['spmp']:7.2f}")


if __name__ == '__main__':
    main()
