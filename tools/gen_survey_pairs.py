"""Training bank for the learned deconvolver with Pan-STARRS as truth (tools/fetch_ps1_truth.py
crops), in the format ``train_nn_denoise.py train`` reads (za, zb, w, sess).

The seeing-pair target (a session's sharpest 30% of frames) is only ~11% sharper than its
input, and every variant trained on it (v3, synthetic blur, cross-night pairs, wider nets)
ends on one width-noise curve. Here the target is a deep survey image of the same sky on the
same grid: ~1" seeing against Origin's 3-7", ~8x less noise per pixel.

Per crop:
  1. colour: Origin channel c ~ s_k * sum_b M[c, b] * PS1_b (b = g, r, i). M is one 3x3 matrix
     for all crops, fitted on unsaturated isolated stars' aperture fluxes; s_k a per-crop
     scale (exposure, transparency), alternating robust fits;
  2. sharpness: the mapped survey image convolved to a Gaussian of ``--target-fwhm`` px
     (from its own measured star width), so the network is asked for a believable step;
  3. large scales: the target's structure above ``--keep-sigma`` px is replaced by the Origin
     image's own (``T - G(T) + G(A)``). PS1 is sky-subtracted on arcminute scales and would
     otherwise teach the network to remove nebulosity; the network only learns fine detail;
  4. masks (weight 0): pixels PS1 does not cover, stars saturated in PS1 (their core falls
     short of what the Origin flux predicts), saturated Origin cores;
  5. input and target are normalised as ``train_nn_denoise.deconv_pair`` does (the input's
     background and noise), weight 1 / Origin noise variance (the target's own is far lower).

Two changes after the first model (v1; dev-notes/nn-deconv-runbook.md, "Survey truth"):
  - S/N gate (``snr_gate``, off with ``--no-gate``): the survey target only where the Origin
    stack itself shows the source, the plain denoised input elsewhere;
  - bright stars (off with ``--no-bright-v3``): where PS1 is saturated the target is the
    previous deconvolver's output (``--bright-model``), not a hole in the mask.

Usage:
    python tools/gen_survey_pairs.py --crops C:/nn_work/survey/ps1 --bank C:/nn_work/survey/bank_ps1.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src import nn_denoise as nd  # noqa: E402

PATCH = 128
LUMA = np.array([0.299, 0.587, 0.114])
_R = 6
_yy, _xx = np.mgrid[-_R:_R + 1, -_R:_R + 1]
_rad = np.hypot(_yy, _xx)
_AP, _ANN = _rad <= 4, (_rad > 5) & (_rad <= 6)
_SESSION = re.compile(r"_c\d+$")


def target_name(crop: str) -> str:
    return re.sub(r"_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d$", "", _SESSION.sub("", crop))


def star_fluxes(A, P):
    """Aperture fluxes (n, 3) in Origin B, G, R and (n, 3) in PS1 g, r, i, plus positions, of
    isolated stars unsaturated in Origin and fully covered by PS1."""
    from src.star_detect import detect_stars_matched_filter as detect
    lum = A @ LUMA
    st = detect((lum - np.median(lum)).astype(np.float32))
    if st is None or len(st) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2), int)
    H, W = lum.shape
    cov = np.isfinite(P).all(-1)
    top = 0.7 * float(lum.max())
    ys, xs = np.asarray(st["ycentroid"]), np.asarray(st["xcentroid"])
    fo, fp, pos = [], [], []
    for y, x in zip(np.round(ys).astype(int), np.round(xs).astype(int)):
        if not (_R + 2 < y < H - _R - 2 and _R + 2 < x < W - _R - 2) or lum[y, x] > top:
            continue
        if not cov[y - _R:y + _R + 1, x - _R:x + _R + 1].all():
            continue
        o = [(A[y - _R:y + _R + 1, x - _R:x + _R + 1, c] - np.median(A[y - _R:y + _R + 1, x - _R:x + _R + 1, c][_ANN]))[_AP].sum()
             for c in range(3)]
        q = [(P[y - _R:y + _R + 1, x - _R:x + _R + 1, b] - np.median(P[y - _R:y + _R + 1, x - _R:x + _R + 1, b][_ANN]))[_AP].sum()
             for b in range(3)]
        if min(o) > 0 and min(q) > 0:
            fo.append(o), fp.append(q), pos.append((y, x))
    return np.array(fo, float), np.array(fp, float), np.array(pos, int)


def fit_colour(stars, iters: int = 8):
    """Colour model O_c ~ s[k][c] * (M[c] . P): M (3x3) a non-negative blend of g, r, i per
    Origin channel (each row summing to 1: the spectral shape), s[k] a per-crop, per-channel
    scale (exposure, transparency and that session's own white balance -- stacks made offline
    skip the Gaia colour calibration). Alternating fits: s from median log ratios, M by
    non-negative least squares on relative errors with Tukey reweighting. An unconstrained M
    traded huge positive and negative weights on the strongly correlated g, r, i fluxes."""
    from scipy.optimize import nnls
    M = np.array([[0.7, 0.3, 0.0], [0.3, 0.7, 0.0], [0.0, 0.6, 0.4]])
    keys = [k for k in stars if len(stars[k][0]) >= 3]
    s = {k: crop_scale(stars[k][0], stars[k][1], M) for k in keys}
    for _ in range(iters):
        X = np.concatenate([stars[k][1] for k in keys])
        Y = np.concatenate([stars[k][0] / s[k] for k in keys])
        for c in range(3):
            w = np.ones(len(Y))
            for _ in range(5):
                sw = np.sqrt(w) / np.maximum(Y[:, c], 1e-12)                  # relative errors
                coef, _ = nnls(X * sw[:, None], Y[:, c] * sw)
                r = np.log(np.maximum(X @ coef, 1e-12) / Y[:, c])
                sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
                u = (r - np.median(r)) / (4.685 * sc)
                w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
            M[c] = coef / max(coef.sum(), 1e-12)
        s = {k: crop_scale(stars[k][0], stars[k][1], M) for k in keys}
    return M, s


def crop_scale(fo, fp, M) -> np.ndarray:
    """A crop's per-channel scale given M: median log ratio of its stars' Origin and mapped
    PS1 fluxes, channel by channel."""
    return np.exp(np.median(np.log(np.maximum(fo, 1e-12) / np.maximum(fp @ M.T, 1e-12)), axis=0))


def gauss_fwhm(img, pos):
    """Median Gaussian FWHM (second moments) of the stars at ``pos`` in a single-channel image."""
    out = []
    for y, x in pos:
        c = img[y - _R:y + _R + 1, x - _R:x + _R + 1]
        e = np.clip(c - np.median(c[_ANN]), 0, None) * (_rad <= 5)
        if e.sum() <= 0:
            continue
        cy, cx = (e * _yy).sum() / e.sum(), (e * _xx).sum() / e.sum()
        v = (e * ((_yy - cy) ** 2 + (_xx - cx) ** 2)).sum() / e.sum() / 2
        out.append(2.3548 * np.sqrt(max(v, 1e-6)))
    return float(np.median(out)) if out else float("nan")


def nan_gauss(x, sigma):
    ok = np.isfinite(x)
    num = ndi.gaussian_filter(np.where(ok, x, 0.0), sigma)
    den = ndi.gaussian_filter(ok.astype(float), sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan)


def make_target(A, P, M, s, fwhm_target, keep_sigma, pos, return_bad: bool = False):
    """(target (h, w, 3), weight mask (h, w)) for one crop."""
    T = s * np.einsum("cb,hwb->hwc", M, P)
    fw = gauss_fwhm(T @ LUMA, pos)
    if not np.isfinite(fw):
        fw = 1.2
    sig = np.sqrt(max((fwhm_target / 2.3548) ** 2 - (fw / 2.3548) ** 2, 0.0))
    mask = np.isfinite(T).all(-1)
    T = np.stack([nan_gauss(T[..., c], sig) if sig > 0 else T[..., c] for c in range(3)], -1)
    T = np.stack([T[..., c] - nan_gauss(T[..., c], keep_sigma) + ndi.gaussian_filter(A[..., c], keep_sigma)
                  for c in range(3)], -1)
    mask &= np.isfinite(T).all(-1)
    # saturated in PS1: the core falls short of what the Origin flux predicts
    lumA, lumT = A @ LUMA, np.nan_to_num(T) @ LUMA
    bad = np.zeros_like(mask)
    from src.star_detect import detect_stars_matched_filter as detect
    st = detect((lumA - np.median(lumA)).astype(np.float32))
    H, W = lumA.shape
    top = 0.9 * float(lumA.max())
    for y, x in zip(np.round(np.asarray(st["ycentroid"])).astype(int), np.round(np.asarray(st["xcentroid"])).astype(int)):
        if not (_R < y < H - _R and _R < x < W - _R):
            continue
        ca, ct = lumA[y - _R:y + _R + 1, x - _R:x + _R + 1], lumT[y - _R:y + _R + 1, x - _R:x + _R + 1]
        fa = (ca - np.median(ca[_ANN]))[_AP].sum()
        ft = (ct - np.median(ct[_ANN]))[_AP].sum()
        if lumA[y, x] > top or (fa > 0 and ft < 0.6 * fa):
            r = int(np.ceil(4 * fwhm_target + 6))
            bad[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] |= (
                np.hypot(*np.mgrid[max(0, y - r) - y:min(H, y + r + 1) - y, max(0, x - r) - x:min(W, x + r + 1) - x]) <= r)
    if return_bad:
        return np.nan_to_num(T).astype(np.float32), mask & ~bad, fw, bad
    return np.nan_to_num(T).astype(np.float32), mask & ~bad, fw


def snr_gate(A, T, lo: float = 3.0, hi: float = 6.0):
    """The target only where the Origin stack itself shows signal: per pixel, a blend of ``T``
    (survey truth) and the plain denoised input, weight ``alpha`` = 0 below ``lo`` and 1 above
    ``hi`` in a Gaussian matched filter (sigma 1.5 px) of the input's luminance in its own
    noise units, grown by 2 px. PS1 is ~8x deeper than an Origin stack, so the raw target holds
    thousands of real faint stars the input cannot show; a network trained on it learnt to put
    one on any noise bump (invented peaks 1.7x v3's against the truth)."""
    from src.nn_denoise import nn_denoise
    D = nn_denoise(A.astype(np.float32), restore_sigma=4.0).astype(np.float64)
    lum = A @ LUMA
    hp = lum - ndi.gaussian_filter(lum, 16.0)
    mf = ndi.gaussian_filter(hp, 1.5)
    sm = ndi.gaussian_filter(lum, 3.0)
    sky = sm < np.quantile(sm, 0.4)
    noise = 1.4826 * np.median(np.abs(mf[sky] - np.median(mf[sky]))) + 1e-12
    snr = mf / noise
    alpha = np.clip((snr - lo) / (hi - lo), 0.0, 1.0)
    alpha = ndi.gaussian_filter(ndi.maximum_filter(alpha, size=5), 1.0)
    return (alpha[..., None] * T + (1.0 - alpha[..., None]) * D).astype(np.float32), alpha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--crops", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--target-fwhm", type=float, default=2.0)
    ap.add_argument("--keep-sigma", type=float, default=16.0)
    ap.add_argument("--per-crop", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bright-v3", action="store_true",
                    help="mask stars saturated in PS1 instead of training them toward --bright-model's output")
    ap.add_argument("--bright-model", default=None,
                    help="deconvolver whose output is the target on saturated stars (default: the bundled "
                         "src/data/nn_deconv.npz). The shipped v4 was built with v3: "
                         "git show v2.15.0:src/data/nn_deconv.npz > v3.npz")
    ap.add_argument("--no-gate", action="store_true",
                    help="use the raw survey target everywhere (no S/N gate, see snr_gate)")
    ap.add_argument("--max-colour-scatter", type=float, default=0.25,
                    help="dex: crops whose stars fit the colour model worse are left out "
                         "(nebula-dominated photometry, a poor match)")
    args = ap.parse_args()
    from train_nn_denoise import is_holdout
    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(os.path.join(args.crops, "*.npz")))
    crops, stars = {}, {}
    for p in files:
        k = os.path.basename(p)[:-4]
        f = np.load(p)
        A, P = f["origin"].astype(np.float64), f["ps1"].astype(np.float64)
        crops[k] = (A, P)
        stars[k] = star_fluxes(A, P)
    M, s = fit_colour({k: v for k, v in stars.items() if not is_holdout(target_name(k))})
    s = {k: crop_scale(v[0], v[1], M) for k, v in stars.items() if len(v[0]) >= 3}    # held-out crops too
    print("colour matrix (rows Origin R, G, B -- the stacks' channel order; columns PS1 g, r, i):\n",
          np.round(M, 4), flush=True)
    za_l, zb_l, w_l, sess, meta = [], [], [], [], {}
    names = sorted(crops)
    for i, k in enumerate(names):
        A, P = crops[k]
        if len(stars[k][0]) < 3:
            print(f"{k}: too few stars for a scale, skipped", flush=True)
            continue
        T, mask, fw, bright = make_target(A, P, M, s[k], args.target_fwhm, args.keep_sigma, stars[k][2],
                                          return_bad=True)
        fo, fp, _ = stars[k]
        resid = np.log(np.maximum(fo, 1e-9) / np.maximum(s[k] * fp @ M.T, 1e-9))
        meta[k] = {"ps1_fwhm": fw, "scale": [float(v) for v in s[k]], "stars": int(len(fo)), "masked": float(1 - mask.mean()),
                   "colour_scatter_dex": [float(np.std(resid[:, c]) / np.log(10)) for c in range(3)]}
        if is_holdout(target_name(k)):
            continue                                       # evaluation crops: never in the bank
        if max(meta[k]["colour_scatter_dex"]) > args.max_colour_scatter:
            print(f"{k}: colour scatter {np.round(meta[k]['colour_scatter_dex'], 2)} dex, left out", flush=True)
            continue
        if not args.no_gate:
            T, alpha = snr_gate(A, T)
            meta[k]["gated_fraction"] = float(1 - alpha.mean())
        if not args.no_bright_v3:
            # Stars saturated in PS1 (or Origin) have no truth. Masked out, the network never saw
            # a bright star and v1 improvised on them -- donut rings (Sunflower), flat bloated
            # discs (Sculptor), soft blobs (Orion). There the target is the shipped deconvolver's
            # own output, so bright stars come out as v3 renders them.
            from src.nn_denoise import DECONV_MODEL_PATH, nn_denoise
            V = nn_denoise(A.astype(np.float32), model_path=args.bright_model or DECONV_MODEL_PATH,
                           restore_sigma=4.0)
            T = np.where(bright[..., None], V, T).astype(np.float32)
            mask = mask | bright
        Af, Tf = A.astype(np.float32), T
        grid, sigma = nd.normalise(Af)
        bg = nd.background_at(grid, 0, 0, *Af.shape[:2])
        za = (Af - bg) / sigma.astype(np.float32)
        zb = (Tf - bg) / sigma.astype(np.float32)
        # weight 1 / the input's noise variance (sky + Poisson, train_nn_denoise.target_variance):
        # with weight 1, a few bright star cores carried the loss (train 1277 against val 33)
        from train_nn_denoise import sky_level_z, target_variance
        # floored at 0.25 (the sky's variance is ~1 in these units): its 1e-6 floor gave a few
        # pixels weights near 1e6 and a validation loss of 20180 against 7.8 in training
        var = np.maximum(target_variance(za, sky_level_z(bg, sigma)), 0.25)
        # plus a systematic term, (5% of the signal)^2: colour and PSF mismatches between the
        # two telescopes grow with brightness, and PS1's taller cores (the same flux in a
        # narrower star) left squared differences of ~1e8 that swamped the loss (val 20180)
        var = var + (0.05 * np.maximum(np.abs(za), np.abs(zb))) ** 2
        w = mask.astype(np.float32)[..., None] / var.astype(np.float32)
        # PS1's sharper star cores peak far higher than the input's; beyond float16's range
        # they became inf and the training loss NaN from the first step. Leave them out.
        big = ~np.isfinite(zb) | (np.abs(zb) > 3e4)
        w[big] = 0.0
        zb = np.clip(np.nan_to_num(zb), -3e4, 3e4)
        H, W = za.shape[:2]
        pos = stars[k][2]
        for j in range(args.per_crop):
            if len(pos) and j % 2 == 0:
                cy, cx = pos[rng.integers(len(pos))]
                y = int(np.clip(cy - rng.integers(16, PATCH - 16), 0, H - PATCH))
                x = int(np.clip(cx - rng.integers(16, PATCH - 16), 0, W - PATCH))
            else:
                y, x = rng.integers(0, H - PATCH + 1), rng.integers(0, W - PATCH + 1)
            if w[y:y + PATCH, x:x + PATCH, 0].mean() < 0.7:
                continue
            za_l.append(za[y:y + PATCH, x:x + PATCH].astype(np.float16))
            zb_l.append(zb[y:y + PATCH, x:x + PATCH].astype(np.float16))
            w_l.append(w[y:y + PATCH, x:x + PATCH].astype(np.float16))
            sess.append(i)
        print(f"[{i + 1}/{len(names)}] {k}: PS1 FWHM {fw:.2f} px, colour scatter "
              f"{np.round(meta[k]['colour_scatter_dex'], 3)} dex, masked {meta[k]['masked']:.2f}", flush=True)
    np.savez(args.bank, za=np.stack(za_l), zb=np.stack(zb_l), w=np.stack(w_l), sess=np.array(sess))
    with open(os.path.splitext(args.bank)[0] + "_meta.json", "w") as fh:
        json.dump({"M": M.tolist(), "target_fwhm": args.target_fwhm, "keep_sigma": args.keep_sigma, "crops": meta}, fh, indent=1)
    print(f"bank: {len(za_l)} patches -> {args.bank}")


if __name__ == "__main__":
    main()
