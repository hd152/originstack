"""Gaia close-pair check for the learned deconvolver on real linear stacks.

Synthetic training loses the seeing pairs' guarantee that the network cannot invent detail
(their target is real sky). This is the guard: on real held-out stacks with a WCS,

- **pairs**: Gaia stars 1.5-4.5 px apart (below the input's resolution, FWHM ~3.3 px), both
  G <= ``--gmax``, within ``--dmag``, nothing else within 8 px. Along the line through both
  positions, the luminance profile should show two peaks and a dip between them. Reported:
  share resolved (two maxima, valley < 0.9 x the fainter peak), median valley/peak, and the
  measured peak separation against Gaia's.
- **singles**: Gaia stars in the same magnitude range with no Gaia neighbour within 8 px
  (to G + 3). Profiles at four angles through the star must *not* split: share split is
  the invented-detail rate.

The header WCS is refined by the offset of matched detected stars first.

Usage:
    python tools/eval_gaia_pairs.py C:/nn_work/heldout_stacks/*.fits \
        --model v3=src/data/nn_deconv.npz --model new=C:/nn_work/model_deconv_s1.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import nn_denoise as nd  # noqa: E402

STEP = 0.1
MIN_SNR = 15.0                       # fainter pair member's peak in the input, sigma
SEP_BINS = (1.5, 2.5, 3.5, 4.5)      # px


def gaia_xy(path: str, shape):
    from astropy.io import fits
    from astropy.wcs import WCS

    from src.net_query import gaia_cone_search
    hdr = fits.getheader(path)
    wcs = WCS(hdr, naxis=2)
    h, w = shape
    ra0, dec0 = wcs.all_pix2world([[w / 2, h / 2]], 0)[0]
    corner = wcs.all_pix2world([[0, 0]], 0)[0]
    rad = float(np.hypot((corner[0] - ra0) * np.cos(np.radians(dec0)), corner[1] - dec0)) * 1.02
    tab = gaia_cone_search(ra0, dec0, rad, ["ra", "dec", "phot_g_mean_mag"], max_rows=20000,
                           require_not_null=["phot_g_mean_mag"])
    ra, dec, g = (np.asarray(tab[c], float) for c in ("ra", "dec", "phot_g_mean_mag"))
    x, y = wcs.all_world2pix(np.c_[ra, dec], 0).T
    ok = (x > 20) & (x < w - 20) & (y > 20) & (y < h - 20)
    return y[ok], x[ok], g[ok]


def refine(lum, y, x, g):
    from src.star_detect import detect_stars_matched_filter
    st = detect_stars_matched_filter(lum - np.median(lum))
    dy, dx = np.asarray(st["ycentroid"]), np.asarray(st["xcentroid"])
    from scipy.spatial import cKDTree
    bright = np.argsort(g)[:3000]
    tree = cKDTree(np.c_[dy, dx])
    # coarse: the peak of all detection - Gaia offsets within 60 px (Pleiades' header was
    # 8 px off), then tighten around it
    near = tree.query_ball_point(np.c_[y[bright], x[bright]], 60.0)
    d = np.array([(dy[k] - y[i], dx[k] - x[i]) for i, ks in zip(bright, near) for k in ks]).reshape(-1, 2)
    hist, ye, xe = np.histogram2d(d[:, 0], d[:, 1], bins=120, range=[[-60, 60], [-60, 60]])
    ky, kx = np.unravel_index(hist.argmax(), hist.shape)
    oy, ox = ye[ky] + 0.5, xe[kx] + 0.5
    for radius in (3.0, 2.0):          # coarse, then tight around the first estimate
        d, j = tree.query(np.c_[y[bright] + oy, x[bright] + ox], distance_upper_bound=radius)
        ok = np.isfinite(d)
        if ok.sum() < 20:
            raise SystemExit(f"WCS check failed: {int(ok.sum())} of {len(bright)} Gaia stars matched")
        offs = np.c_[dy[j[ok]] - y[bright][ok], dx[j[ok]] - x[bright][ok]]
        oy, ox = np.median(offs, 0)
    rms = float(np.sqrt(np.median(((offs - [oy, ox]) ** 2).sum(1))))
    return float(oy), float(ox), rms, int(ok.sum())


def profile(coef, p1, p2, ext=3.0):
    """Cubic-spline profile from ``coef = ndi.spline_filter(lum)`` (prefiltered once)."""
    v = np.asarray(p2, float) - np.asarray(p1, float)
    sep = float(np.hypot(*v))
    u = v / sep
    t = np.arange(-ext, sep + ext + 1e-9, STEP)
    pts = np.asarray(p1, float)[:, None] + u[:, None] * t[None, :]
    return t, ndi.map_coordinates(coef, pts, order=3, mode="nearest", prefilter=False), sep


def local_sky(lum, y, x):
    yi, xi = int(round(y)), int(round(x))
    cut = lum[yi - 14:yi + 15, xi - 14:xi + 15]
    yy, xx = np.mgrid[-14:15, -14:15]
    r = np.hypot(yy, xx)
    return float(np.median(cut[(r > 10) & (r < 14)]))


def two_peaks(t, p, sep):
    """(resolved, valley/fainter peak, measured separation) for a profile through a pair at
    0 and ``sep``."""
    near = lambda c: (t > c - 0.75) & (t < c + 0.75)  # noqa: E731
    i1 = np.flatnonzero(near(0.0))[np.argmax(p[near(0.0)])]
    i2 = np.flatnonzero(near(sep))[np.argmax(p[near(sep)])]
    lo, hi = sorted((i1, i2))
    valley = float(p[lo:hi + 1].min())
    fainter = float(min(p[i1], p[i2]))
    is_max = lambda i: 0 < i < len(p) - 1 and p[i] >= p[i - 1] and p[i] >= p[i + 1]  # noqa: E731
    ratio = valley / fainter if fainter > 0 else 1.0
    resolved = bool(is_max(i1) and is_max(i2) and hi > lo and ratio < 0.9)
    return resolved, ratio, abs(t[i2] - t[i1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stacks", nargs="+")
    ap.add_argument("--model", action="append", default=[], help="NAME=weights.npz (repeatable)")
    ap.add_argument("--restore", type=float, default=4.0)
    ap.add_argument("--gmax", type=float, default=18.0)
    ap.add_argument("--dmag", type=float, default=2.0, help="max magnitude difference within a pair")
    ap.add_argument("--json")
    args = ap.parse_args()
    from astropy.io import fits
    methods = {"none": None}
    methods.update(dict(m.split("=", 1) for m in args.model))
    results = {}
    for path in args.stacks:
        name = os.path.splitext(os.path.basename(path))[0]
        img = np.moveaxis(fits.getdata(path).astype(np.float32), 0, -1)
        img = np.where(np.isfinite(img), img, 0.0).astype(np.float32)
        sigma = nd.noise_sigma(img)
        lum0 = (img / sigma.astype(np.float32)).mean(-1)
        y, x, g = gaia_xy(path, img.shape[:2])
        oy, ox, rms, nm = refine(lum0, y, x, g)
        y, x = y + oy, x + ox
        from scipy.spatial import cKDTree
        tree = cKDTree(np.c_[y, x])
        cand = np.flatnonzero(g <= args.gmax)
        pairs, singles = [], []
        for i in cand:
            nb = [j for j in tree.query_ball_point([y[i], x[i]], 8.0) if j != i and g[j] <= g[i] + 3]
            if not nb:
                singles.append(i)
                continue
            if len(nb) != 1:
                continue
            j = nb[0]
            if j < i or g[j] > args.gmax or abs(g[i] - g[j]) > args.dmag:
                continue
            sep = np.hypot(y[i] - y[j], x[i] - x[j])
            if not 1.5 <= sep <= 4.5:
                continue
            mid = [(y[i] + y[j]) / 2, (x[i] + x[j]) / 2]
            others = [k for k in tree.query_ball_point(mid, 8.0) if k not in (i, j) and g[k] <= max(g[i], g[j]) + 2.5]
            if not others:
                pairs.append((i, j))
        rng = np.random.default_rng(0)
        singles = list(rng.permutation(singles)[:400])
        print(f"{name}: WCS offset ({oy:.2f}, {ox:.2f}) px, rms {rms:.2f} px on {nm} stars; "
              f"{len(pairs)} pairs, {len(singles)} singles (G <= {args.gmax})", flush=True)
        # only pairs whose fainter star peaks >= MIN_SNR sigma in the input: on fainter
        # ones noise alone makes two maxima, so the noisy input "resolves" more than a
        # denoised output
        lum_in = lum0 - np.median(lum0)
        def snr_ok(i):  # noqa: E306
            return lum_in[int(round(y[i])), int(round(x[i]))] - local_sky(lum_in, y[i], x[i]) >= MIN_SNR
        pairs = [(i, j) for i, j in pairs if snr_ok(i) and snr_ok(j)]
        seps_gaia = np.array([np.hypot(y[i] - y[j], x[i] - x[j]) for i, j in pairs])
        bins = np.digitize(seps_gaia, SEP_BINS[1:-1]) if len(pairs) else np.array([], int)
        print(f"   {len(pairs)} pairs above {MIN_SNR:g} sigma, per separation bin {SEP_BINS}: "
              f"{np.bincount(bins, minlength=len(SEP_BINS) - 1).tolist()}", flush=True)
        row = {"wcs_offset": [oy, ox], "wcs_rms": rms, "pairs": len(pairs), "singles": len(singles),
               "pairs_per_bin": np.bincount(bins, minlength=len(SEP_BINS) - 1).tolist()}
        for k, mp in methods.items():
            t0 = time.time()
            out = img if mp is None else nd.nn_denoise(img, model_path=mp, restore_sigma=args.restore)
            lum = (out / sigma.astype(np.float32)).mean(-1)
            coef = ndi.spline_filter(lum.astype(np.float64), order=3, mode="nearest")
            res, ratios, seps = [], [], []
            for i, j in pairs:
                sky = local_sky(lum, (y[i] + y[j]) / 2, (x[i] + x[j]) / 2)
                t, p, sep = profile(coef, (y[i], x[i]), (y[j], x[j]))
                p = p - sky
                r, ratio, ms = two_peaks(t, p, sep)
                res.append(r)
                ratios.append(ratio)
                if r:
                    seps.append(ms / sep)
            split = []
            for i in singles:
                sky = local_sky(lum, y[i], x[i])
                if lum[int(round(y[i])), int(round(x[i]))] - sky < 30:
                    continue
                s = False
                for ang in np.arange(4) * np.pi / 4:
                    d = np.array([np.sin(ang), np.cos(ang)]) * 1.5
                    t, p, _ = profile(coef, (y[i] - d[0], x[i] - d[1]), (y[i] + d[0], x[i] + d[1]), ext=1.5)
                    p = p - sky
                    c = np.flatnonzero((t > -1.5) & (t < 4.5))
                    pc = p[c]
                    mx = [m for m in range(1, len(pc) - 1) if pc[m] >= pc[m - 1] and pc[m] > pc[m + 1]]
                    if len(mx) >= 2:
                        a, b = mx[0], mx[-1]
                        if pc[a:b + 1].min() < 0.9 * min(pc[a], pc[b]):
                            s = True
                split.append(s)
            res_bins = [float(np.mean(np.array(res)[bins == b])) if (bins == b).any() else None
                        for b in range(len(SEP_BINS) - 1)]
            row[k] = {"resolved": float(np.mean(res)) if res else None, "resolved_per_bin": res_bins,
                      "valley_ratio_median": float(np.median(ratios)) if ratios else None,
                      "sep_ratio_median": float(np.median(seps)) if seps else None,
                      "single_split": float(np.mean(split)) if split else None,
                      "singles_tested": len(split), "seconds": time.time() - t0}
            v = row[k]
            rb = " ".join("  - " if r is None else f"{r:.2f}" for r in res_bins)
            print(f"   {k:12s} resolved {v['resolved'] if v['resolved'] is not None else float('nan'):.2f} [{rb}]  "
                  f"valley/peak {v['valley_ratio_median'] or float('nan'):.2f}  "
                  f"sep/gaia {v['sep_ratio_median'] or float('nan'):.2f}  "
                  f"single split {v['single_split'] if v['single_split'] is not None else float('nan'):.3f} "
                  f"({len(split)})  {v['seconds']:.0f}s", flush=True)
        results[name] = row
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
