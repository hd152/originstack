"""Faint-source survival, ringing and crops for the learned denoiser/deconvolver on held-out pairs.

The go/no-go checks that the error and star-width scores of ``eval_nn_deconv.py`` do not
cover (dev-notes/nn-deconv-runbook.md, step 5):

* **Faint sources.** Exponential discs (scale 1.5 and 3 px) and point sources (a Gaussian
  of the input stack's own FWHM) are added to sky-only places of each held-out input stack,
  neutral in colour, with peaks of 0.5-4 sky sigma (points: 1-8). Each method runs on the
  stack with and without them; the difference, summed in an aperture, over the flux that
  was added is the fraction of the source that survives (1 = all of it).
* **Ringing.** Around the brightest isolated, unsaturated stars: the deepest dip of the
  azimuthally averaged profile below the local sky between 1.5 and 6 FWHM, in sky sigma.
  Compared with the input's own value (its noise and real neighbours set the floor).
* **Invented sources.** Faint peaks on the sky of each output, and the fraction the
  independent sharp stack confirms (``false_sources``); ``none`` is the input smoothed by
  its PSF, the matched-filter baseline.
* **Crops** (``--crops DIR``): input / method / sharp target side by side around the
  brightest stars and the brightest extended region, asinh-stretched alike, for a look.

Usage:
    python tools/eval_nn_faint.py --pairs C:/nn_work/nn_deconv/pairs --model C:/nn_work/model_deconv_v3.npz \
        --json C:/nn_work/faint_v3.json --crops C:/nn_work/crops_v3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src import nn_denoise as nd  # noqa: E402

CROP = (1024, 1536)          # evaluate on a central window: the network is fully convolutional
SPACING = 72                 # px between injected sources
DISC_SCALES = (1.5, 3.0)
DISC_PEAKS = (0.5, 1.0, 2.0, 4.0)
POINT_PEAKS = (1.0, 2.0, 4.0, 8.0)


def central(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    ch, cw = min(CROP[0], h), min(CROP[1], w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return a[y0:y0 + ch, x0:x0 + cw]


def sky_positions(a: np.ndarray, sigma: np.ndarray, half: int) -> list:
    """Grid positions whose whole (2*half+1)^2 box is sky: no pixel of the smoothed
    noise-unit image above 0.5 sigma, nothing above 3 sigma within 12 px."""
    grid = nd.background_grid(a)
    bg = nd.background_at(grid, 0, 0, *a.shape[:2])
    z = ((a - bg) / sigma).mean(axis=2)
    busy = ndi.gaussian_filter(z, 2.0) > 0.5
    busy |= ndi.maximum_filter(z > 3.0, size=25)
    busy = ndi.maximum_filter(busy, size=2 * half + 1)
    out = []
    for y in range(half + 8, a.shape[0] - half - 8, SPACING):
        for x in range(half + 8, a.shape[1] - half - 8, SPACING):
            if not busy[y, x]:
                out.append((y, x))
    return out


def stamp(kind: str, scale: float, half: int) -> np.ndarray:
    r = np.hypot(*np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64))
    s = np.exp(-r / scale) if kind == "disc" else np.exp(-0.5 * (r / scale) ** 2)
    return s / s.max()


def inject(a, sigma, positions, kind, scale, peaks, half):
    """Add sources (cycling through ``peaks``) at ``positions``; returns the image and,
    per source, (y, x, peak, injected aperture flux summed over channels)."""
    out = a.copy()
    s = stamp(kind, scale, half)
    rad = (3.0 * scale + 3.0) if kind == "disc" else 2.5 * scale * 2.3548
    ap = np.hypot(*np.mgrid[-half:half + 1, -half:half + 1]) <= rad
    rows = []
    for i, (y, x) in enumerate(positions):
        p = peaks[i % len(peaks)]
        src = s[:, :, None] * (p * sigma[None, None, :])
        out[y - half:y + half + 1, x - half:x + half + 1] += src.astype(np.float32)
        rows.append((y, x, p, float((src * ap[:, :, None]).sum())))
    return out, rows, ap


def recovered(d_inj, d_base, rows, ap, half):
    diff = d_inj - d_base
    by_peak = {}
    for y, x, p, f in rows:
        got = float((diff[y - half:y + half + 1, x - half:x + half + 1] * ap[:, :, None]).sum())
        by_peak.setdefault(p, []).append(got / f)
    return {str(p): [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), len(v)]
            for p, v in sorted(by_peak.items())}


def bright_stars(a: np.ndarray, sigma: np.ndarray, fwhm: float, n: int = 15) -> list:
    """Brightest isolated stars well below saturation: (y, x)."""
    lum = a.mean(axis=2)
    z = (lum - np.median(lum)) / sigma.mean()
    peaks = (z == ndi.maximum_filter(z, size=9)) & (z > 30)
    ys, xs = np.nonzero(peaks)
    cap = 0.6 * np.nanmax(lum)
    r = int(6 * fwhm) + 4
    cand = []
    for y, x in zip(ys, xs):
        if y < r or x < r or y >= lum.shape[0] - r or x >= lum.shape[1] - r or lum[y, x] > cap:
            continue
        box = z[y - r:y + r + 1, x - r:x + r + 1]
        others = (box == ndi.maximum_filter(box, size=9)) & (box > 0.05 * z[y, x])
        if others.sum() <= 1:
            cand.append((z[y, x], y, x))
    return [(y, x) for _, y, x in sorted(cand, reverse=True)[:n]]


def ring_depth(img: np.ndarray, sigma: np.ndarray, stars: list, fwhm: float) -> float:
    """Median over stars of the deepest dip (sky sigma) of the radial profile below the
    local sky (median of the 6-8 FWHM annulus), between 1.5 and 6 FWHM."""
    lum = img.mean(axis=2)
    s = float(sigma.mean())
    r_out = int(8 * fwhm) + 2
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    rr = np.hypot(yy, xx)
    dips = []
    for y, x in stars:
        if y < r_out or x < r_out or y + r_out >= lum.shape[0] or x + r_out >= lum.shape[1]:
            continue
        box = lum[y - r_out:y + r_out + 1, x - r_out:x + r_out + 1]
        sky = np.median(box[(rr >= 6 * fwhm) & (rr <= 8 * fwhm)])
        prof = [box[(rr >= r0) & (rr < r0 + 1)].mean() for r0 in np.arange(1.5 * fwhm, 6 * fwhm, 1.0)]
        if prof:
            dips.append(min(0.0, (min(prof) - sky) / s))
    return float(np.median(dips)) if dips else float("nan")


def false_sources(out, a, b, sigma, fwhm, thresholds=(2.0, 3.0, 5.0)):
    """Faint peaks a method shows on the sky, and how many the independent sharp stack
    confirms. Peaks: local maxima of the method's sky-subtracted luminance, in the input's
    sky sigma, away from anything above 10 sigma in the input. Confirmed: the target,
    smoothed by its own PSF (a matched filter), is > 2.5 of its own sigma within 2 px.
    A real source is in both stacks; one made out of noise is not. Returns, per threshold,
    (peaks per megapixel, confirmed fraction)."""
    s = float(sigma.mean())
    grid = nd.background_grid(out)
    lum = (out - nd.background_at(grid, 0, 0, *out.shape[:2])).mean(axis=2) / s
    lum = ndi.gaussian_filter(lum, 0.7)
    za = ((a - nd.background_at(nd.background_grid(a), 0, 0, *a.shape[:2])) / sigma).mean(axis=2)
    keep_out = ndi.maximum_filter(za > 10, size=int(8 * fwhm) | 1)
    edge = np.zeros_like(keep_out)
    edge[16:-16, 16:-16] = True
    peaks = (lum == ndi.maximum_filter(lum, size=7)) & edge & ~keep_out
    bs = ndi.gaussian_filter((b - nd.background_at(nd.background_grid(b), 0, 0, *b.shape[:2])).mean(axis=2),
                             fwhm / 2.3548)
    d = (bs[:, 4:] - bs[:, :-4])[::2].ravel()
    bsig = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    confirmed_map = ndi.maximum_filter(bs / bsig, size=5) > 2.5
    mpx = keep_out.size / 1e6
    sky = edge & ~keep_out
    res = {"chance": float(confirmed_map[sky].mean())}   # confirmation rate of a random sky pixel
    for t in thresholds:
        sel = peaks & (lum > t)
        n = int(sel.sum())
        res[str(t)] = [n / mpx, float(confirmed_map[sel].mean()) if n else float("nan")]
    return res


def save_crops(path, panels, labels, centres, size=96):
    from PIL import Image, ImageDraw
    ref = panels[0].mean(axis=2)
    lo, hi = np.percentile(ref, 20), np.percentile(ref, 99.7)
    soft = (hi - lo) / 40.0
    rows = []
    for cy, cx in centres:
        y0 = int(np.clip(cy - size // 2, 0, ref.shape[0] - size))
        x0 = int(np.clip(cx - size // 2, 0, ref.shape[1] - size))
        row = []
        for p in panels:
            c = p[y0:y0 + size, x0:x0 + size]
            v = np.arcsinh(np.maximum(c - lo, 0) / soft) / np.arcsinh((hi - lo) / soft)
            row.append((np.clip(v, 0, 1) * 255).astype(np.uint8))
        rows.append(np.concatenate(row, axis=1))
    im = Image.fromarray(np.concatenate(rows, axis=0)).resize(
        (size * len(panels) * 3, size * len(centres) * 3), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for i, lab in enumerate(labels):
        d.rectangle([i * size * 3, 0, i * size * 3 + 8 * len(lab) + 8, 16], fill=(0, 0, 0))
        d.text((i * size * 3 + 4, 2), lab, fill=(255, 255, 255))
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True, help="deconvolver weights (.npz)")
    ap.add_argument("--denoise-model", default=None, help="denoiser weights (default: the bundled one)")
    ap.add_argument("--restore", type=float, nargs="*", default=[0.0, 4.0])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--json")
    ap.add_argument("--crops", help="folder for input/output/target crop PNGs")
    args = ap.parse_args()
    from train_nn_denoise import is_holdout
    dirs = sorted(d for d in glob.glob(os.path.join(args.pairs, "*"))
                  if os.path.exists(os.path.join(d, "meta.json")) and is_holdout(os.path.basename(d)))
    if args.only:
        dirs = [d for d in dirs if os.path.basename(d).startswith(tuple(args.only))]
    methods = {"none": lambda x: x,
               "nn_denoise": lambda x: nd.nn_denoise(x, model_path=args.denoise_model)}
    for r in args.restore:
        methods[f"nn_deconv_r{r:g}"] = (lambda x, r=r: nd.nn_denoise(x, model_path=args.model, restore_sigma=r))
    if args.crops:
        os.makedirs(args.crops, exist_ok=True)
    results = {}
    for d in dirs:
        name = os.path.basename(d)
        t0 = time.time()
        meta = json.load(open(os.path.join(d, "meta.json")))
        fwhm = float(meta.get("fwhm_input") or 3.0)
        a = central(np.load(os.path.join(d, "input.npy")).astype(np.float32))
        b = central(np.load(os.path.join(d, "target.npy")).astype(np.float32))
        sigma = nd.noise_sigma(a).astype(np.float32)
        half = 20
        pos = sky_positions(a, sigma, half)
        injected = {}
        for scale in DISC_SCALES:
            injected[f"disc{scale:g}"] = inject(a, sigma, pos, "disc", scale, DISC_PEAKS, half)
        injected["point"] = inject(a, sigma, pos, "point", fwhm / 2.3548, POINT_PEAKS, half)
        stars = bright_stars(a, sigma, fwhm)
        row = {"n_positions": len(pos), "n_ring_stars": len(stars)}
        outs = {}
        for k, fn in methods.items():
            base = fn(a).astype(np.float32)
            outs[k] = base
            row[k] = {"ring": ring_depth(base, sigma, stars, fwhm),
                      "peaks": false_sources(ndi.gaussian_filter(base, (fwhm / 2.3548, fwhm / 2.3548, 0))
                                             if k == "none" else base, a, b, sigma, fwhm)}
            for kind, (img, rows, apm) in injected.items():
                row[k][kind] = recovered(fn(img).astype(np.float32), base, rows, apm, half)
        results[name] = row
        print(f"{name}  ({len(pos)} sky positions, {len(stars)} ring stars, {time.time() - t0:.0f}s)", flush=True)
        for k in methods:
            v = row[k]
            cells = []
            for kind in injected:
                cells.append(kind + " " + " ".join(f"{p}:{m[0]:.2f}" for p, m in v[kind].items()))
            pk = "  ".join(f">{t}:{nc[0]:5.0f}/Mpx {nc[1]:4.0%}" for t, nc in v["peaks"].items() if t != "chance")
            pk += f"  (chance {v['peaks']['chance']:.0%})"
            print(f"   {k:14s} ring {v['ring']:6.2f}   peaks(confirmed) {pk}\n"
                  f"   {'':14s} " + "   ".join(cells), flush=True)
        if args.crops:
            lum = ndi.gaussian_filter(a.mean(axis=2), 8)
            ey, ex = np.unravel_index(np.argmax(lum), lum.shape)
            centres = stars[:3] + [(int(ey), int(ex))]
            deconv_key = next(k for k in methods if k.startswith("nn_deconv"))
            save_crops(os.path.join(args.crops, f"{name}.png"),
                       [a, outs["nn_denoise"], outs[deconv_key], b],
                       ["input", "nn_denoise", deconv_key, "sharp target"], centres)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
