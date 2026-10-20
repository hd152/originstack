"""Score denoisers on held-out sessions without ground truth (Noise2Noise evaluation).

For each session from tools/gen_n2n_pairs.py: input A = half-stack q0+q1, reference B =
q2+q3 (independent noise, same sky). For any denoiser D,

    E[(D(A) - B)^2] = E[(D(A) - clean)^2] + var(B)

so ``mean(w * (D(A) - B)^2) - 1`` (in noise units, w = 2 / var(A - B) so var(B) weighs 1) is
D's squared error against the clean sky relative to the input's noise variance: 1.0 = no
denoising, 0 = perfect. Reported for all pixels, sky (z < 3) and signal (z >= 3), plus the
star width change on the same stars (tools/common_star_fwhm.py) and run time, and a
side-by-side stretched crop per session.

Usage:
    python tools/eval_nn_denoise.py --pairs G:/output/nn_denoise/pairs --out eval/ [--holdout-only]
        [--model n2nr=C:/nn_work/n2nr/model_n2nr_v1.npz]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src import nn_denoise as nd  # noqa: E402


def _wavelet(img):
    from src.denoising import directional_wavelet_denoise
    grid, _ = nd.normalise(img)
    bg = nd.background_at(grid, 0, 0, *img.shape[:2])
    return directional_wavelet_denoise(img - bg) + bg      # Phase 4 denoises sky-centred data


DENOISERS = {
    "none": lambda img: img,
    "wavelet": _wavelet,
    "nn": lambda img: nd.nn_denoise(img),
}


def score(a, b, out):
    grid, sigma = nd.normalise(a)
    bg = nd.background_at(grid, 0, 0, *a.shape[:2])
    s = sigma.astype(np.float32)
    za, zb, zo = (a - bg) / s, (b - bg) / s, (out - bg) / s
    var = np.stack([ndi.gaussian_filter((za[..., c] - zb[..., c]) ** 2, 4.0) for c in range(3)], -1)
    w = 2.0 / np.maximum(var, 1e-3)
    e = w * (zo - zb) ** 2
    m = 32
    e, zs = e[m:-m, m:-m], ndi.gaussian_filter(za.mean(-1), 2.0)[m:-m, m:-m]
    sky = zs < 3
    return {"all": float(e.mean() - 1), "sky": float(e[sky].mean() - 1), "signal": float(e[~sky].mean() - 1),
            "per_channel": [float(e[..., c].mean() - 1) for c in range(3)]}


def _stretch(x, lo, hi):
    y = np.clip((x - lo) / (hi - lo), 0, None)
    return np.clip(np.arcsinh(30 * y) / np.arcsinh(30), 0, 1)


def save_crops(name, a, outs, path, size=512):
    from PIL import Image
    h, w, _ = a.shape
    lum = ndi.uniform_filter(a.mean(-1), 64)[size:-size, size:-size]
    y, x = np.unravel_index(np.argmax(lum), lum.shape)       # the brightest extended region
    y, x = min(max(y + size - size // 2, 0), h - size), min(max(x + size - size // 2, 0), w - size)
    lo = np.percentile(a, 20, axis=(0, 1))
    hi = np.percentile(a, 99.7, axis=(0, 1))
    tiles = [_stretch(o[y:y + size, x:x + size], lo, hi) for o in outs.values()]
    img = (np.concatenate(tiles, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=92)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout-only", action="store_true")
    ap.add_argument("--only", nargs="*", help="session name prefixes to evaluate")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                    help="also score these weights with nn_denoise (repeatable)")
    args = ap.parse_args()
    for spec in args.model:
        name, path = spec.split("=", 1)
        DENOISERS[name] = lambda img, p=path: nd.nn_denoise(img, model_path=p)
    from common_star_fwhm import common_star_fwhm
    from train_nn_denoise import is_holdout
    os.makedirs(args.out, exist_ok=True)
    dirs = sorted(d for d in glob.glob(os.path.join(args.pairs, "*")) if os.path.exists(os.path.join(d, "meta.json")))
    if args.holdout_only:
        dirs = [d for d in dirs if is_holdout(os.path.basename(d))]
    if args.only:
        dirs = [d for d in dirs if os.path.basename(d).startswith(tuple(args.only))]
    results = {}
    for d in dirs:
        name = os.path.basename(d)
        q = [np.load(os.path.join(d, f"q{i}.npy")) for i in range(4)]
        a, b = (q[0] + q[1]) / 2, (q[2] + q[3]) / 2
        outs, row = {}, {}
        for k, fn in DENOISERS.items():
            t = time.time()
            outs[k] = fn(a).astype(np.float32)
            row[k] = score(a, b, outs[k])
            row[k]["seconds"] = time.time() - t
            if k != "none":
                fw = common_star_fwhm(np.moveaxis(outs[k], -1, 0), np.moveaxis(a, -1, 0))
                row[k]["star_width_ratio"] = fw["ratio_a_over_b"] if fw else None
        results[name] = row
        save_crops(name, a, outs, os.path.join(args.out, f"{name}.jpg"))
        print(f"{name}: " + "   ".join(
            f"{k} all {v['all']:.3f} sky {v['sky']:.3f} sig {v['signal']:.3f}"
            + (f" width x{v['star_width_ratio']:.3f}" if v.get('star_width_ratio') else "")
            + f" {v['seconds']:.0f}s" for k, v in row.items()), flush=True)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    if results:
        print("\nmean over sessions (error / input noise variance; 1 = untouched, 0 = clean):")
        for k in DENOISERS:
            v = [r[k] for r in results.values()]
            print(f"  {k:8s} all {np.mean([x['all'] for x in v]):.3f}  sky {np.mean([x['sky'] for x in v]):.3f}  "
                  f"signal {np.mean([x['signal'] for x in v]):.3f}")


if __name__ == "__main__":
    main()
