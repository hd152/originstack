"""Score background extraction against a second night, on the held-out targets' captured Phase 4
inputs (tools/gen_background_bank.py ``<name>_full.pkl``): DBE as Phase 4 calls it, and the
other ``--method``s. A new background method is added to ``extract``.

For each target with two nights A, B: both nights' DBE inputs go through each method; B's
result is registered onto A's grid and matched by per-channel *gain only*. On 64 px block
medians of the luminance difference (stars and footprint edge masked), a scaled copy of the
target's own shape is fitted out (a gain mismatch leaves one); the residual RMS, in A's raw
per-pixel sky sigma, is ``mismatch`` -- both nights' background errors together. ``kept`` is the
slope of A's extracted block medians on A's raw ones: how much of the target's own large-scale
light survives (a method that removes the nebula on both nights "agrees" perfectly; this
catches it).

Usage:
    python tools/eval_background_nights.py --bank C:/nn_work/nnbg/bank --method dbe wavelet
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import pickle
import re
import sys

import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LUMA = np.array([0.299, 0.587, 0.114], np.float32)
BLOCK = 64
_DATE = re.compile(r"_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_full$")


def block_medians(img, ok):
    H, W = img.shape
    nby, nbx = H // BLOCK, W // BLOCK
    v = img[:nby * BLOCK, :nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK).transpose(0, 2, 1, 3).reshape(nby, nbx, -1)
    m = ok[:nby * BLOCK, :nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK).transpose(0, 2, 1, 3).reshape(nby, nbx, -1)
    out = np.full((nby, nbx), np.nan)
    for i in range(nby):
        for j in range(nbx):
            if m[i, j].mean() >= 0.5:
                out[i, j] = np.median(v[i, j][m[i, j]])
    return out


def robust_line(x, y):
    X = np.c_[x - np.median(x), np.ones_like(x)]
    w = np.ones_like(y)
    for _ in range(8):
        coef = np.linalg.lstsq(X * w[:, None] ** 0.5, y * w ** 0.5, rcond=None)[0]
        res = y - X @ coef
        s = 1.4826 * np.median(np.abs(res)) + 1e-12
        u = res / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
    return coef, res


def extract(cap, method):
    """``cap``'s DBE input with its background removed by ``method``, with Phase 4's settings."""
    import src.background as bg
    kw = dict(cap["kw"])
    kw.pop("verbose", None)
    with contextlib.redirect_stdout(io.StringIO()):
        if method == "dbe":
            return bg.dynamic_background_extraction(cap["rgb"].copy(), verbose=False, **kw)
        if method == "wavelet":
            for k in ("masked_frac_thresh", "outlier_sigma", "outlier_iters"):
                kw.pop(k, None)
            return bg.wavelet_background_extraction(cap["rgb"].copy(), verbose=False, **kw)
    raise ValueError(f"unknown method {method!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--method", nargs="+", default=["dbe"], choices=["dbe", "wavelet"])
    ap.add_argument("--json")
    args = ap.parse_args()
    from src.merge import _detect_stars, _lum, _match_flux_scale, _pixel_noise, _register_stack
    from src.registration import apply_transform
    from src.utils import embed_to_shape
    methods = {m: m for m in args.method}
    groups = {}
    for p in sorted(glob.glob(os.path.join(args.bank, "*_full.pkl"))):
        groups.setdefault(_DATE.sub("", os.path.basename(p)[:-4]), []).append(p)
    results = {}
    for t, ps in sorted(groups.items()):
        if len(ps) < 2:
            continue
        ca, cb = (pickle.load(open(p, "rb")) for p in ps[:2])
        A, B = ca["rgb"], cb["rgb"]
        H, W = A.shape[:2]
        bh, bw = B.shape[:2]
        HH, WW = max(H, bh), max(W, bw)
        la = _lum(embed_to_shape(A, HH, WW))
        tf, shift = _register_stack(la, _lum(embed_to_shape(B, HH, WW)), _detect_stars(la), prev_shape=(bh, bw))
        if tf is None and shift is None:
            print(f"{t}: does not register")
            continue

        def warp(x):
            return apply_transform(embed_to_shape(x, HH, WW), shift=shift, transform=tf)[:H, :W]
        foot = ndi.binary_erosion(warp(np.ones((bh, bw), np.float32)) > 0.99, iterations=48)
        lraw = A @ LUMA
        st = _detect_stars(lraw)
        starm = np.zeros((H, W), bool)
        for s in (st if st is not None else []):
            y, x = int(round(s["ycentroid"])), int(round(s["xcentroid"]))
            if 0 <= y < H and 0 <= x < W:
                starm[y, x] = True
        ok = foot & ~ndi.binary_dilation(starm, iterations=12)
        sig = _pixel_noise(lraw, ok)
        T = block_medians(lraw, ok)
        results[t] = {}
        for name, m in methods.items():
            Ab, Bb = extract(ca, m), extract(cb, m)
            Bw = warp(Bb)
            g, _, _ = _match_flux_scale(Ab, Bw, foot)
            Db = block_medians((Ab - Bw * g[None, None, :].astype(np.float32)) @ LUMA, ok)
            Ab_b = block_medians(Ab @ LUMA, ok)
            k = np.isfinite(Db) & np.isfinite(T) & np.isfinite(Ab_b)
            _, res = robust_line(T[k], Db[k])
            kc, _ = robust_line(T[k], Ab_b[k])
            results[t][name] = {"mismatch": float(np.sqrt(np.mean(res ** 2)) / sig), "kept": float(kc[0])}
        print(f"{t:22s} " + "  ".join(f"{n}: mismatch {r['mismatch']:6.3f} kept {r['kept']:6.3f}"
                                     for n, r in results[t].items()), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
