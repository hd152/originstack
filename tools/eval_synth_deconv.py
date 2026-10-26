"""Score deconvolvers on held-out synthetic-blur fields (tools/synth_blur.py), where the truth
is known exactly -- the one check that can judge sharpening beyond the seeing pairs' sharp
stack, which ``eval_nn_deconv.py``'s ``width/target`` cannot.

Fields are drawn from the held-out ingredients (``gen_synth_deconv.py`` -> holdout.npz:
held-out targets' backgrounds and noise), with a fixed seed so every model sees the same
fields. Per method:

- ``error``: mean(w * (out - truth)^2), w = 1 / the input's noise variance; sky / signal split.
- ``width/truth``, ``width/input``: Gaussian-fit FWHM on isolated catalogue stars (truth peak
  >= 20 sigma, no neighbour within 10 px), median ratio.
- ``flux``: aperture (r 7, sky annulus 10-14) flux / the truth's, median, same stars.
- ``invented``: output peaks (> 3 sigma on luminance) more than 3 px from every catalogue
  star and on a truth that has no peak there, per Mpx; ``split``: share of isolated stars
  with a second output peak within 2.5-6 px.

Usage:
    python tools/eval_synth_deconv.py --synth C:/nn_work/synth/holdout.npz \
        --model v3=src/data/nn_deconv.npz --model new=C:/nn_work/model_deconv_s1.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src import nn_denoise as nd  # noqa: E402


def run_model(z: np.ndarray, path: str, restore: float) -> np.ndarray:
    den = nd.denoise_z(z, nd.load_weights(path))
    if restore > 0:
        for c in range(3):
            den[..., c] += ndi.gaussian_filter(z[..., c] - den[..., c], restore, mode="reflect")
    return den


def isolated(cat: np.ndarray, size: int, truth_lum: np.ndarray) -> np.ndarray:
    keep = []
    for i, (y, x, f) in enumerate(cat):
        if not (16 <= y < size - 16 and 16 <= x < size - 16):
            continue
        d = np.hypot(cat[:, 0] - y, cat[:, 1] - x)
        near = (d > 0) & (d < 10) & (cat[:, 2] > 0.01 * f)
        if near.any():
            continue
        if truth_lum[int(round(y)), int(round(x))] < 20:
            continue
        keep.append(i)
    return np.array(keep, int)


def ap_flux(img: np.ndarray, y: float, x: float) -> float:
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    r = np.hypot(yy - y, xx - x)
    sky = np.median(img[(r >= 10) & (r < 14)])
    return float((img[r < 7] - sky).sum())


def peaks(lum: np.ndarray, thr: float) -> np.ndarray:
    sm = ndi.gaussian_filter(lum, 1.0)
    mx = (sm == ndi.maximum_filter(sm, 5)) & (sm > thr)
    mx[:6], mx[-6:], mx[:, :6], mx[:, -6:] = False, False, False, False
    return np.argwhere(mx)


def main() -> None:
    from common_star_fwhm import _gauss_fit
    from synth_blur import SynthSource
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--synth", required=True)
    ap.add_argument("--model", action="append", default=[], help="NAME=weights.npz (repeatable)")
    ap.add_argument("--restore", type=float, default=4.0)
    ap.add_argument("--fields", type=int, default=240)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json")
    args = ap.parse_args()
    src = SynthSource(args.synth, "cpu")
    gen = np.random.default_rng(args.seed)
    methods = {"none": None, "nn_denoise": nd._MODEL_PATH}
    methods.update(dict(m.split("=", 1) for m in args.model))
    acc = {k: {"err_sky": [], "err_sig": [], "w_truth": [], "w_input": [], "flux": [], "invented": 0,
               "split": 0, "stars": 0} for k in methods}
    area = 0
    for f0 in range(0, args.fields, 8):
        zin, truth, w, meta = src.batch(8, gen, size=args.size, return_meta=True)
        for i in range(zin.shape[0]):
            z = zin[i].permute(1, 2, 0).numpy().copy()
            t = truth[i].permute(1, 2, 0).numpy()
            ww = w[i].permute(1, 2, 0).numpy()
            cat = meta["cat"][i]
            tl = t.mean(-1)
            iso = isolated(cat, args.size, tl) if len(cat) else np.array([], int)
            # a truth peak counts as real wherever the truth itself has one (background structure)
            tpk = peaks(tl, 0.5)
            m = 8
            area += (args.size - 2 * m) ** 2
            sky = ndi.gaussian_filter(tl, 2.0)[m:-m, m:-m] < 3
            for k, path in methods.items():
                out = z if path is None else run_model(z, path, args.restore)
                e = (ww * (out - t) ** 2)[m:-m, m:-m]
                a = acc[k]
                a["err_sky"].append(float(e[sky].mean()) if sky.any() else np.nan)
                a["err_sig"].append(float(e[~sky].mean()) if (~sky).any() else np.nan)
                ol, il = out.mean(-1), z.mean(-1)
                for j in iso:
                    y, x = cat[j, 0], cat[j, 1]
                    fo, ft, fi = _gauss_fit(ol, x, y), _gauss_fit(tl, x, y), _gauss_fit(il, x, y)
                    if fo and ft and fi:
                        a["w_truth"].append(fo[0] / ft[0])
                        a["w_input"].append(fo[0] / fi[0])
                    a["flux"].append(ap_flux(ol, y, x) / max(ap_flux(tl, y, x), 1e-6))
                    pk = peaks(ol, 3.0)
                    d = np.hypot(pk[:, 0] - y, pk[:, 1] - x) if len(pk) else np.array([])
                    a["split"] += int(((d > 2.5) & (d < 6)).any())
                    a["stars"] += 1
                pk = peaks(ol, 3.0)
                for py, px in pk:
                    if len(cat) and np.hypot(cat[:, 0] - py, cat[:, 1] - px).min() <= 3:
                        continue
                    if len(tpk) and np.hypot(tpk[:, 0] - py, tpk[:, 1] - px).min() <= 3:
                        continue
                    a["invented"] += 1
        print(f"{min(f0 + 8, args.fields)}/{args.fields} fields", flush=True)
    res = {}
    for k, a in acc.items():
        res[k] = {"error_sky": float(np.nanmean(a["err_sky"])), "error_signal": float(np.nanmean(a["err_sig"])),
                  "width_vs_truth": float(np.median(a["w_truth"])) if a["w_truth"] else None,
                  "width_vs_input": float(np.median(a["w_input"])) if a["w_input"] else None,
                  "flux": float(np.median(a["flux"])) if a["flux"] else None,
                  "flux_p10_p90": [float(np.percentile(a["flux"], q)) for q in (10, 90)] if a["flux"] else None,
                  "invented_per_mpx": a["invented"] / area * 1e6,
                  "split": a["split"] / max(a["stars"], 1), "stars": a["stars"]}
        r = res[k]
        print(f"{k:12s} err sky {r['error_sky']:8.3f} signal {r['error_signal']:9.3f}  "
              f"width/truth {r['width_vs_truth'] or np.nan:.3f}  width/input {r['width_vs_input'] or np.nan:.3f}  "
              f"flux {r['flux'] or np.nan:.3f}  invented/Mpx {r['invented_per_mpx']:.0f}  "
              f"split {r['split']:.3f} ({r['stars']} stars)", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
