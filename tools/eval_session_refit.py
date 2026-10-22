"""Per-session refit of the learned deconvolver's last layer, scored on held-out pixels.

The network's output layer is a 1x1 convolution: the prediction is linear in the last
feature map d0 (24 channels). So on a session's own seeing pair (input = all but the
sharpest frames, target = the sharpest; tools/gen_n2n_pairs.py --mode seeing) the training
loss ``w * (zA + pred * s - zB)^2`` is a weighted least-squares problem in that layer's
3 x (24 + 1) weights, solved in closed form, ridge-regularised toward the shipped weights.
It adapts the network to this night's seeing and this unit's optics without any training at
run time -- the synthetic-to-real gap the synthetic-blur experiment stopped at
(dev-notes/nn-deconv-runbook.md).

Fit on the left half of each held-out pair, score on the right half (the fit never sees the
scored pixels' noise), with ``eval_nn_deconv.score`` and ``common_star_fwhm``.

Usage:
    python tools/eval_session_refit.py --pairs C:/nn_work/nn_deconv/pairs \
        --model v3=src/data/nn_deconv.npz --model s4=C:/nn_work/model_deconv_s4.npz --lam 0.01 1 100
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

GAP = 48           # px left out between the fit half and the scored half (receptive field ~24)


def features(x, w):
    """d0 of the network (src/nn_denoise.features)."""
    return nd.features(x, w)


def normal_equations(za, zb, wt, w, cols):
    """Per output channel, sum of wt*s^2 [d0;1][d0;1]^T and wt*s^2 [d0;1] r over pixels whose
    column is in ``cols`` (slice), r = (zB - zA) / s the correction the layer should predict."""
    h, wd, _ = za.shape
    T, M = nd.TILE, nd.MARGIN
    ph, pw = (-h) % T, (-wd) % T
    zp = np.pad(za, ((M, M + ph), (M, M + pw), (0, 0)), mode='reflect')
    nf = w['out.weight'].shape[1] + 1
    A = np.zeros((3, nf, nf))
    B = np.zeros((3, nf))
    n = 0
    for y0 in range(0, h, T):
        for x0 in range(0, wd, T):
            x_lo, x_hi = max(x0, cols.start), min(x0 + T, wd, cols.stop)
            if x_hi <= x_lo:
                continue
            t = zp[y0:y0 + T + 2 * M, x0:x0 + T + 2 * M]
            d0 = features(nd.to_input(t).transpose(2, 0, 1).astype(np.float32), w)[:, M:M + T, M:M + T]
            hh = min(T, h - y0)
            d0 = d0[:, :hh, x_lo - x0:x_hi - x0]
            X = np.concatenate([d0.reshape(d0.shape[0], -1), np.ones((1, d0[0].size), np.float32)]).astype(np.float64)
            a_t = za[y0:y0 + hh, x_lo:x_hi]
            s = nd.output_scale(a_t)
            r = (zb[y0:y0 + hh, x_lo:x_hi] - a_t) / s
            ww = wt[y0:y0 + hh, x_lo:x_hi] * s ** 2
            for c in range(3):
                q = ww[..., c].ravel().astype(np.float64)
                A[c] += (X * q) @ X.T
                B[c] += X @ (q * r[..., c].ravel())
            n += X.shape[1]
    return A, B, n


def refit(w, A, B, n, lam):
    """Weights with the output layer solved under ridge ``lam`` (per pixel) toward the shipped one."""
    out = dict(w)
    W0, b0 = w['out.weight'][:, :, 0, 0].astype(np.float64), w['out.bias'].astype(np.float64)
    nf = W0.shape[1] + 1
    Wn, bn = np.empty_like(W0), np.empty_like(b0)
    for c in range(3):
        th0 = np.r_[W0[c], b0[c]]
        reg = lam * np.trace(A[c]) / nf       # scale-free: lam relative to the mean diagonal
        th = np.linalg.solve(A[c] + reg * np.eye(nf), B[c] + reg * th0)
        Wn[c], bn[c] = th[:-1], th[-1]
    out['out.weight'] = Wn.astype(np.float32)[:, :, None, None]
    out['out.bias'] = bn.astype(np.float32)
    return out


def main() -> None:
    from common_star_fwhm import common_star_fwhm
    from eval_nn_deconv import score
    from train_nn_denoise import SKY_MATCH_SIGMA, is_holdout, sky_level_z, target_variance
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", action="append", default=[], help="NAME=weights.npz (repeatable)")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.01, 1.0, 100.0])
    ap.add_argument("--restore", type=float, default=4.0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--json")
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.model) or {"shipped": nd.DECONV_MODEL_PATH}
    dirs = sorted(d for d in glob.glob(os.path.join(args.pairs, "*"))
                  if os.path.exists(os.path.join(d, "meta.json")) and is_holdout(os.path.basename(d)))
    if args.only:
        dirs = [d for d in dirs if os.path.basename(d).startswith(tuple(args.only))]
    results = {}
    for d in dirs:
        name = os.path.basename(d)
        a, b = np.load(os.path.join(d, "input.npy")), np.load(os.path.join(d, "target.npy"))
        h, wd, _ = a.shape
        half = wd // 2
        fit_cols, score_cols = slice(16, half - GAP // 2), slice(half + GAP // 2, wd - 16)
        bm = b - np.stack([ndi.gaussian_filter(b[..., c] - a[..., c], SKY_MATCH_SIGMA) for c in range(3)], -1)
        grid, sigma = nd.normalise(a)
        bg = nd.background_at(grid, 0, 0, h, wd)
        s = sigma.astype(np.float32)
        za, zb = (a - bg) / s, (bm - bg) / s
        wt = (1.0 / target_variance(zb, sky_level_z(bg, sigma))).astype(np.float32)
        ar, br = a[:, score_cols], b[:, score_cols]
        row = {}
        print(name, flush=True)
        for mname, path in models.items():
            w = nd.load_weights(path)
            t0 = time.time()
            A, B, n = normal_equations(za, zb, wt, w, fit_cols)
            variants = {mname: w}
            for lam in args.lam:
                variants[f"{mname}+refit{lam:g}"] = refit(w, A, B, n, lam)
            for vname, vw in variants.items():
                key = f"__refit__/{name}/{vname}"
                nd._weights_cache[key] = vw
                out = nd.nn_denoise(a, model_path=key, restore_sigma=args.restore)[:, score_cols]
                nd._weights_cache.pop(key, None)
                sc = score(ar, br, out)
                ft = common_star_fwhm(np.moveaxis(out, -1, 0), np.moveaxis(br, -1, 0))
                fi = common_star_fwhm(np.moveaxis(out, -1, 0), np.moveaxis(ar, -1, 0))
                sc["width_vs_target"] = ft["ratio_a_over_b"] if ft else None
                sc["width_vs_input"] = fi["ratio_a_over_b"] if fi else None
                row[vname] = sc
                print(f"   {vname:22s} error all {sc['all']:7.3f} sky {sc['sky']:7.3f} signal {sc['signal']:8.3f}  "
                      f"width/target {sc['width_vs_target'] or float('nan'):.3f}  "
                      f"width/input {sc['width_vs_input'] or float('nan'):.3f}", flush=True)
            print(f"   ({mname}: fit {time.time() - t0:.0f} s incl. scoring)", flush=True)
        results[name] = row
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
