"""Score the learned deconvolver on held-out seeing pairs (tools/gen_n2n_pairs.py --mode seeing).

Input A = stack of all but the sharpest frames, target B = stack of the sharpest: the same
sky, a sharper PSF, independent noise. For any method D, with w = 1 / B's noise variance
(train_nn_denoise.target_variance),

    mean(w * (D(A) - B)^2) - 1  ~  D's squared error against the noise-free *sharp* sky,

in units of B's noise variance (lower is better, 0 = the sharp sky exactly). Reported with
the star width of D(A) on the same stars as B (``width / target``: 1.0 = as sharp as the
sharp frames) and as A (``width / input``), and run time.

Usage:
    python tools/eval_nn_deconv.py --pairs G:/output/nn_deconv/pairs --model nn_deconv.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src import nn_denoise as nd  # noqa: E402


def score(a, b, out):
    from train_nn_denoise import target_variance
    grid, sigma = nd.normalise(a)
    bg = nd.background_at(grid, 0, 0, *a.shape[:2])
    s = sigma.astype(np.float32)
    zb, zo = (b - bg) / s, (out - bg) / s
    w = 1.0 / target_variance(zb)
    m = 32
    e = (w * (zo - zb) ** 2)[m:-m, m:-m]
    lum = ndi_smooth(zb.mean(-1))[m:-m, m:-m]
    sky = lum < 3
    return {"all": float(e.mean() - 1), "sky": float(e[sky].mean() - 1), "signal": float(e[~sky].mean() - 1)}


def ndi_smooth(x):
    from scipy import ndimage as ndi
    return ndi.gaussian_filter(x, 2.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True, help="deconvolver weights (.npz)")
    ap.add_argument("--denoise-model", default=None, help="denoiser weights (default: the bundled one)")
    ap.add_argument("--restore", type=float, nargs="*", default=[0.0], help="restore sigmas to try")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--json")
    args = ap.parse_args()
    from common_star_fwhm import common_star_fwhm
    from train_nn_denoise import is_holdout
    dirs = sorted(d for d in glob.glob(os.path.join(args.pairs, "*"))
                  if os.path.exists(os.path.join(d, "meta.json")) and is_holdout(os.path.basename(d)))
    if args.only:
        dirs = [d for d in dirs if os.path.basename(d).startswith(tuple(args.only))]
    methods = {"none": lambda x: x,
               "nn_denoise": lambda x: nd.nn_denoise(x, model_path=args.denoise_model)}
    for r in args.restore:
        methods[f"nn_deconv_r{r:g}"] = (lambda x, r=r: nd.nn_denoise(x, model_path=args.model, restore_sigma=r))
    results = {}
    for d in dirs:
        name = os.path.basename(d)
        a, b = np.load(os.path.join(d, "input.npy")), np.load(os.path.join(d, "target.npy"))
        row = {}
        for k, fn in methods.items():
            t = time.time()
            out = fn(a).astype(np.float32)
            row[k] = score(a, b, out)
            row[k]["seconds"] = time.time() - t
            ft = common_star_fwhm(np.moveaxis(out, -1, 0), np.moveaxis(b, -1, 0))
            fi = common_star_fwhm(np.moveaxis(out, -1, 0), np.moveaxis(a, -1, 0))
            row[k]["width_vs_target"] = ft["ratio_a_over_b"] if ft else None
            row[k]["width_vs_input"] = fi["ratio_a_over_b"] if fi else None
        results[name] = row
        print(name, flush=True)
        for k, v in row.items():
            print(f"   {k:16s} error all {v['all']:7.3f} sky {v['sky']:7.3f} signal {v['signal']:7.3f}   "
                  f"width/target {v['width_vs_target'] or float('nan'):.3f}  "
                  f"width/input {v['width_vs_input'] or float('nan'):.3f}  {v['seconds']:.0f}s", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
