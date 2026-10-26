"""Rank stars by the colour blob the nn star remover leaves.

For each detected star, the starless output's per-channel deviation from its own luminance
inside the star's footprint, against the same in a ring just outside it, in local noise
units. Prints the worst stars with their flux, saturation, donor status and background, for
each ``--tta`` setting. ``--no-colour`` turns off the per-footprint colour constraint.

    python tools/diag_star_blobs.py C:/nn_work/heldout_stacks/orion.fits --tta 2 8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import nn_denoise as nd  # noqa: E402
from src import star_n2n as sn  # noqa: E402
from tools.eval_star_n2n import load_stack  # noqa: E402


def blob_scores(x, z, field, ys, xs, rs):
    h, w, c = x.shape
    out = np.full(len(ys), np.nan)
    for i, (y, xc, q) in enumerate(zip(ys, xs, rs)):
        R = q + 10
        y0, y1, x0, x1 = max(0, y - R), min(h, y + R + 1), max(0, xc - R), min(w, xc + R + 1)
        yy, xx = np.mgrid[y0 - y:y1 - y, x0 - xc:x1 - xc]
        rr = np.hypot(yy, xx)
        disk, ring = rr <= q, (rr > q) & (rr <= R)
        if ring.sum() < 20:
            continue
        cut = x[y0:y1, x0:x1]
        chroma = cut - cut.mean(-1, keepdims=True)
        ref = np.median(chroma[ring], axis=0)
        noise = 1.4826 * np.median(np.abs(chroma[ring] - ref), axis=0) + 1e-3
        dev = np.abs(chroma[disk] - ref) / noise
        out[i] = float(np.percentile(dev.max(-1), 95))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stacks", nargs="+")
    ap.add_argument("--model", default=None)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--tta", type=int, nargs="+", default=[2])
    ap.add_argument("--no-colour", action="store_true", help="no colour forcing inside footprints")
    a = ap.parse_args()
    if a.no_colour:
        sn.COLOUR_SNR = (1e9, 2e9)
    w = nd.load_weights(a.model or sn.MODEL_PATH)
    for p in a.stacks:
        img = load_stack(p)
        img = np.where(np.isfinite(img), img, 0.0).astype(np.float32)
        z, _, _, B = sn.normalise_image(img)
        f = sn.StarField(z, B)
        ys, xs = f.det_y, f.det_x
        rs = np.full(len(ys), sn.COPY_R_MIN)
        rs[f.saturated] = sn.COPY_R_MAX
        dmap = {(y, x): q for y, x, q in zip(f.donor_y, f.donor_x, f.donor_r)}
        is_donor = np.array([(y, x) in dmap for y, x in zip(ys, xs)])
        rs = np.array([dmap.get((y, x), r) for y, x, r in zip(ys, xs, rs)])
        flux = sn.aperture_flux(z, ys, xs).mean(1)
        variants = {f"tta{t}": sn.starless_z(z, w, 1.0, f, t) for t in a.tta}
        sc = {k: blob_scores(v, z, f, ys, xs, rs) for k, v in variants.items()}
        bg = np.array([sn._ring_level(z, y, x) for y, x in zip(ys, xs)])
        inp = blob_scores(z, z, f, ys, xs, rs)
        name = os.path.basename(p)
        print(f"== {name}: {len(ys)} stars, {f.saturated.sum()} saturated")
        for k, v in sc.items():
            print(f"  {k:6s} median {np.nanmedian(v):.2f}  p90 {np.nanpercentile(v, 90):.2f}  "
                  f"n>8 {int(np.sum(v > 8))}  n>15 {int(np.sum(v > 15))}")
        order = np.argsort(-np.nan_to_num(sc[list(sc)[0]], nan=-1))[:a.top]
        print("   y     x    flux     sat donor  r   bg     input  " + "  ".join(f"{k:>6s}" for k in sc))
        for i in order:
            print(f"  {ys[i]:4d} {xs[i]:5d} {flux[i]:9.0f}  {int(f.saturated[i]):3d} {int(is_donor[i]):4d} "
                  f"{rs[i]:3d} {bg[i]:6.1f}  {inp[i]:5.1f}  " + "  ".join(f"{v[i]:6.1f}" for v in sc.values()))


if __name__ == "__main__":
    main()
