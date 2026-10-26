"""Cross-night agreement filtering: two nights of one target as two independent half-maps.

Cryo-EM filters a reconstruction by how well two independent half-maps agree (Fourier
shell correlation): noise does not correlate between independent maps, real structure does.
Two nights of the same target are such a pair. Per starlet (undecimated wavelet) scale and
neighbourhood:

    signal power  S = smooth(a * b)          (noise cancels in expectation; >= 0)
    noise power   N = k * smooth((a - b)^2)  (k = the combined stack's share of it)
    combined      c * S / (S + N)

so the shrinkage is measured from the data, not assumed, and anything only one night shows
(noise, a satellite glint, a transient, a hot pixel the rejection missed, mismatched
gradients) is suppressed because the nights disagree on it.

Held-out seeing pairs (tools/gen_n2n_pairs.py --mode seeing) give an independent reference:
for a target with nights A and B, the inputs are night A's *input* stack (all but its
sharpest frames) and night B's full stack registered onto it; the reference is night A's
*sharp* stack, which shares no frames with either. Scored, per method on the combined stack:

- error vs the reference (``eval_nn_deconv.score``: sky / signal, in the reference's noise units)
- faint sources injected into *both* nights: flux kept (disc 1.5 / 3 px at 0.5-2 sigma)
- bright sources injected into *one* night only (a satellite glint, an unrejected hot-pixel
  cluster, a transient): flux left (should be ~0; a plain mean keeps that night's weight)

Usage:
    python tools/eval_cross_night.py --pairs C:/nn_work/nn_deconv/pairs --json out.json
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

B3 = np.array([1, 4, 6, 4, 1], np.float64) / 16.0
LEVELS = 5


def starlet(img: np.ndarray, levels: int = LEVELS):
    """Undecimated B3-spline wavelet (a trous): [w_1, ..., w_J, c_J], img = sum."""
    out, c = [], img.astype(np.float64)
    for j in range(levels):
        k = np.zeros(4 * 2 ** j + 1)
        k[:: 2 ** j] = B3
        s = ndi.convolve1d(ndi.convolve1d(c, k, axis=0, mode="reflect"), k, axis=1, mode="reflect")
        out.append(c - s)
        c = s
    out.append(c)
    return out


def cross_night_filter(a: np.ndarray, b: np.ndarray, wa: float, wb: float, window: float = 2.0,
                       levels: int = LEVELS) -> np.ndarray:
    """Agreement-filtered weighted mean of two registered stacks (H, W) of one channel.
    ``wa``, ``wb``: inverse noise variances (the combined stack is their weighted mean)."""
    fa, fb = wa / (wa + wb), wb / (wa + wb)
    # share of E[(a-b)^2] that is the combined stack's noise: (fa^2 sa^2 + fb^2 sb^2)/(sa^2 + sb^2)
    sa2, sb2 = 1.0 / wa, 1.0 / wb
    k = (fa * fa * sa2 + fb * fb * sb2) / (sa2 + sb2)
    A, Bw = starlet(a, levels), starlet(b, levels)
    out = fa * A[-1] + fb * Bw[-1]
    for j in range(levels):
        aj, bj = A[j], Bw[j]
        sig = window * 2 ** j
        S = np.maximum(ndi.gaussian_filter(aj * bj, sig), 0.0)
        N = k * ndi.gaussian_filter((aj - bj) ** 2, sig)
        out += (fa * aj + fb * bj) * (S / np.maximum(S + N, 1e-30))
    return out.astype(np.float32)


def cross_night_consistency(a: np.ndarray, b: np.ndarray, wa: float, wb: float, k: float = 4.0,
                            tol: float = 0.3, levels: int = LEVELS, blur_a: float = 0.0,
                            blur_b: float = 0.0, absent: float = 0.0) -> np.ndarray:
    """Weighted mean of two registered stacks (H, W) of one channel, except where the nights
    disagree significantly: per starlet coefficient, |a - b| > k * sqrt(noise_a^2 + noise_b^2
    + (tol * |mean|)^2) (``tol`` absorbs seeing differences on real structure). There the
    smaller-magnitude coefficient -- the night that does not show the feature -- is used.
    Agreeing structure, faint or not, passes untouched (the Wiener-gain version,
    ``cross_night_filter``, shrank faint sources both nights show to 30-67%).

    ``blur_a`` / ``blur_b`` (Gaussian sigma, px): the decision is taken on copies blurred to
    a common resolution (the sharper night blurred to the other's seeing), so stars seen at
    different seeing do not count as disagreement; the output still uses the originals.

    ``absent`` > 0 adds the artefact signature: the weaker night's coefficient must be
    consistent with no signal (|x| < absent x its own noise). Structure both nights show at
    different strengths (calibration, colour, seeing) then keeps the weighted mean; without
    it the smaller coefficient biased bright structure (signal error worse than the mean on
    4-5 of 8 directions, with or without PSF matching)."""
    fa, fb = wa / (wa + wb), wb / (wa + wb)
    A, Bw = starlet(a, levels), starlet(b, levels)
    da = starlet(ndi.gaussian_filter(a, blur_a), levels) if blur_a > 0 else A
    db = starlet(ndi.gaussian_filter(b, blur_b), levels) if blur_b > 0 else Bw
    out = fa * A[-1] + fb * Bw[-1]
    for j in range(levels):
        aj, bj, xa, xb = A[j], Bw[j], da[j], db[j]
        na = 1.4826 * np.median(np.abs(xa - np.median(xa)))
        nb = 1.4826 * np.median(np.abs(xb - np.median(xb)))
        m = fa * aj + fb * bj
        thr = k * np.sqrt(na * na + nb * nb + (tol * (fa * xa + fb * xb)) ** 2)
        bad = np.abs(xa - xb) > thr
        a_weaker = np.abs(xa) < np.abs(xb)
        if absent > 0:
            bad &= np.where(a_weaker, np.abs(xa) < absent * na, np.abs(xb) < absent * nb)
        out += np.where(bad, np.where(a_weaker, aj, bj), m)
    return out.astype(np.float32)


def register(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """b resampled onto a's grid (affine from matched stars, cubic)."""
    from scipy.spatial import cKDTree

    from src.blind_match import match_rigid_unknown_rotation
    from src.star_detect import detect_stars_matched_filter as detect

    def stars(img):
        lum = img.mean(-1)
        st = detect(lum - np.median(lum))
        return st[np.argsort(-st["flux"])]
    sa, sb = stars(a), stars(b)
    rigid = match_rigid_unknown_rotation(sa, sb, max_stars=60, pixel_tol=2.0)      # a -> b
    p = np.asarray(rigid.params, float)[:2]
    xa, ya = np.asarray(sa["xcentroid"], float), np.asarray(sa["ycentroid"], float)
    xb, yb = np.asarray(sb["xcentroid"], float), np.asarray(sb["ycentroid"], float)
    tree = cKDTree(np.c_[xb, yb])
    for tol in (3.0, 1.5, 1.0):
        px = p[0, 0] * xa + p[0, 1] * ya + p[0, 2]
        py = p[1, 0] * xa + p[1, 1] * ya + p[1, 2]
        d, j = tree.query(np.c_[px, py], distance_upper_bound=tol)
        ok = np.isfinite(d)
        A = np.c_[xa[ok], ya[ok], np.ones(ok.sum())]
        sol, *_ = np.linalg.lstsq(A, np.c_[xb[j[ok]], yb[j[ok]]], rcond=None)
        p = sol.T
    # output (row, col) of a -> input (row, col) of b
    matrix = np.array([[p[1, 1], p[1, 0]], [p[0, 1], p[0, 0]]])
    offset = [p[1, 2], p[0, 2]]
    return np.stack([ndi.affine_transform(b[..., c], matrix, offset=offset, output_shape=a.shape[:2],
                                          order=3, mode="constant", cval=np.nan) for c in range(3)], -1)


def disc(scale: float, half: int = 20) -> np.ndarray:
    r = np.hypot(*np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64))
    return np.exp(-r / scale)


def sky_positions(z_lum: np.ndarray, half: int = 20, spacing: int = 72):
    busy = ndi.gaussian_filter(z_lum, 2.0) > 0.5
    busy |= ndi.maximum_filter(z_lum > 3.0, size=25)
    busy = ndi.maximum_filter(busy, size=2 * half + 1)
    return [(y, x) for y in range(half + 40, z_lum.shape[0] - half - 40, spacing)
            for x in range(half + 40, z_lum.shape[1] - half - 40, spacing) if not busy[y, x]]


def inject(img, sigma, positions, peaks, scales, half=20):
    out = img.copy()
    rows = []
    for i, (y, x) in enumerate(positions):
        p, sc = peaks[i % len(peaks)], scales[(i // len(peaks)) % len(scales)]
        st = disc(sc, half)[:, :, None] * (p * sigma[None, None, :])
        out[y - half:y + half + 1, x - half:x + half + 1] += st.astype(np.float32)
        ap = np.hypot(*np.mgrid[-half:half + 1, -half:half + 1]) <= 3 * sc + 3
        rows.append((y, x, p, sc, float((st * ap[:, :, None]).sum()), ap))
    return out, rows


def kept(d_inj, d_base, rows, half=20):
    res = {}
    for y, x, p, sc, f, ap in rows:
        diff = (d_inj - d_base)[y - half:y + half + 1, x - half:x + half + 1]
        res.setdefault(f"disc{sc:g}@{p:g}", []).append(float((diff * ap[:, :, None]).sum()) / f)
    return {k: float(np.median(v)) for k, v in sorted(res.items())}


def main() -> None:
    from eval_nn_deconv import score
    from train_nn_denoise import is_holdout
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--window", type=float, nargs="*", default=[])
    ap.add_argument("--k", type=float, nargs="*", default=[3.0])
    ap.add_argument("--tol", type=float, nargs="*", default=[0.3, 0.15],
                    help="tolerances for the PSF-matched decision")
    ap.add_argument("--one-peaks", type=float, nargs="+", default=[10.0, 30.0, 100.0],
                    help="peaks (sigma) of the sources put into one night only")
    ap.add_argument("--only", nargs="*", help="target name prefixes")
    ap.add_argument("--json")
    args = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(args.pairs, "*"))
                  if os.path.exists(os.path.join(d, "meta.json")) and is_holdout(os.path.basename(d)))
    by_t = {}
    for d in dirs:
        by_t.setdefault(os.path.basename(d).rsplit("_", 2)[0], []).append(d)
    results = {}
    rng = np.random.default_rng(0)
    for tgt, ds in sorted(by_t.items()):
        if len(ds) < 2 or (args.only and not tgt.startswith(tuple(args.only))):
            continue
        for da, db in ((ds[0], ds[1]), (ds[1], ds[0])):
            t0 = time.time()
            name = f"{os.path.basename(da)} + {os.path.basename(db)}"
            a1 = np.load(os.path.join(da, "input.npy")).astype(np.float32)
            a2 = np.load(os.path.join(da, "target.npy")).astype(np.float32)
            mb = json.load(open(os.path.join(db, "meta.json")))
            bf = (np.load(os.path.join(db, "input.npy")) * mb["n_input"]
                  + np.load(os.path.join(db, "target.npy")) * mb["n_target"]) / (mb["n_input"] + mb["n_target"])
            b = register(a1, bf.astype(np.float32))
            valid = np.isfinite(b).all(-1)
            ys, xs = np.nonzero(valid)
            y0, y1, x0, x1 = ys.min() + 24, ys.max() - 24, xs.min() + 24, xs.max() - 24
            a1, a2, b = (x[y0:y1, x0:x1].copy() for x in (a1, a2, b))
            if not np.isfinite(b).all():
                b = np.where(np.isfinite(b), b, a1)
            # flux scale and sky of b matched to a1 (different nights: transparency, sky)
            sa, sb = nd.noise_sigma(a1), nd.noise_sigma(b)
            ga, gb = nd.background_at(nd.background_grid(a1), 0, 0, *a1.shape[:2]), \
                nd.background_at(nd.background_grid(b), 0, 0, *b.shape[:2])
            bright = (a1 - ga).mean(-1) > 20 * sa.mean()
            gain = np.array([np.median(((a1 - ga)[..., c][bright]) / np.maximum((b - gb)[..., c][bright], 1e-6))
                             for c in range(3)]) if bright.sum() > 100 else np.ones(3)
            b = (b - gb) * gain + ga
            sb = sb * gain
            wa, wb = 1.0 / sa ** 2, 1.0 / sb ** 2
            from common_star_fwhm import common_star_fwhm
            fw = common_star_fwhm(np.moveaxis(a1, -1, 0), np.moveaxis(b, -1, 0))
            fwa, fwb = (fw["a"], fw["b"]) if fw else (0.0, 0.0)
            blur_a = np.sqrt(max(fwb ** 2 - fwa ** 2, 0.0)) / 2.3548
            blur_b = np.sqrt(max(fwa ** 2 - fwb ** 2, 0.0)) / 2.3548
            print(f"   star FWHM a {fwa:.2f} b {fwb:.2f} px -> decision blur a {blur_a:.2f} b {blur_b:.2f}", flush=True)
            print(f"{name}: {a1.shape[1]}x{a1.shape[0]}, noise a {sa.mean():.1f} b {sb.mean():.1f}, "
                  f"gain {np.round(gain, 3)}", flush=True)

            def combos(x_a, x_b):
                mean = (wa * x_a + wb * x_b) / (wa + wb)
                out = {"input_a": x_a, "mean": mean, "mean+nn_denoise": nd.nn_denoise(mean)}
                for win in args.window:
                    out[f"cross_w{win:g}"] = np.stack([cross_night_filter(x_a[..., c], x_b[..., c], wa[c], wb[c], win)
                                                       for c in range(3)], -1)
                from src.merge import consistency_combine
                ones = np.ones(x_a.shape[:2], bool)
                out["merge_consistency"] = consistency_combine([(x_a, ones, wa), (x_b, ones, wb)])[0]
                for kk in args.k:
                    out[f"consist_k{kk:g}"] = np.stack([cross_night_consistency(
                        x_a[..., c], x_b[..., c], wa[c], wb[c], kk) for c in range(3)], -1)
                    out[f"consist_k{kk:g}_absent"] = np.stack([cross_night_consistency(
                        x_a[..., c], x_b[..., c], wa[c], wb[c], kk, absent=3.0) for c in range(3)], -1)
                    out[f"consist_k{kk:g}_psf_absent"] = np.stack([cross_night_consistency(
                        x_a[..., c], x_b[..., c], wa[c], wb[c], kk, blur_a=blur_a, blur_b=blur_b,
                        absent=3.0) for c in range(3)], -1)
                    for tol in args.tol:
                        out[f"consist_k{kk:g}_psf_t{tol:g}"] = np.stack([cross_night_consistency(
                            x_a[..., c], x_b[..., c], wa[c], wb[c], kk, tol, blur_a=blur_a, blur_b=blur_b)
                            for c in range(3)], -1)
                return out

            base = combos(a1, b)
            z = ((a1 - ga) / sa).mean(-1)
            pos = sky_positions(z)
            rng.shuffle(pos)
            pos = pos[:120]
            half_n = len(pos) // 2
            peaks, scales = (0.5, 1.0, 2.0), (1.5, 3.0)
            a_both, rows_both = inject(a1, sa, pos[:half_n], peaks, scales)
            b_both, _ = inject(b, sa, pos[:half_n], peaks, scales)
            a_one, rows_one = inject(a_both, sa, pos[half_n:], tuple(args.one_peaks), (1.0,))
            inj = combos(a_one, b_both)
            row = {"n_both": half_n, "n_one": len(pos) - half_n}
            for k in base:
                sc = score(a1, a2, base[k]) if k != "input_a" else score(a1, a2, a1)
                row[k] = {"score": sc, "both": kept(inj[k], base[k], rows_both),
                          "one_night": kept(inj[k], base[k], rows_one)}
                both = row[k]["both"]
                one = row[k]["one_night"]
                print(f"   {k:16s} error sky {sc['sky']:7.3f} signal {sc['signal']:8.3f}   kept (both nights) "
                      + " ".join(f"{kk} {v:.2f}" for kk, v in both.items())
                      + "   one night " + " ".join(f"{kk} {v:.2f}" for kk, v in one.items()), flush=True)
            print(f"   ({time.time() - t0:.0f} s)", flush=True)
            results[name] = row
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
