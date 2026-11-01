"""Evaluate a Noisier2Noise star-removal model (src/star_n2n.py) on held-out linear stacks.

None of the checks needs starless ground truth. All of them work in noise units, with one
fixed normalisation per stack, and compare the model's output on the stack (o0) against its
output on the stack plus something known:

- ``real_left``: fraction of each detected, unsaturated real star's flux still in o0
  (aperture flux of o0 / of the input). Ideal 0. Reported per flux tertile.
- ``inject_left``: a fresh, independently seeded star layer (``--test-alpha`` x the real
  count) added to the stack; fraction of its flux in (o1 - o0). Ideal 0; an identity model
  scores 1. Per flux tertile.
- ``faint_kept``: faint extended discs (exponential, 2-8 px scale, 0.5-5 sky sigma peak, away
  from stars) added; fraction of their flux in (o2 - o0). Ideal 1 (the StarNet failure mode is
  eating small galaxies).
- ``sky_shift`` / ``sky_mad``: median and MAD of (o0 - input) on star-free pixels, in sky
  sigma. Ideal 0: removal must not move or denoise the sky.
- ``hole``: median over the 20 brightest removed stars of the deepest dip of o0's azimuthal
  mean below the local sky within the stamp, in sky sigma. Negative = over-subtraction.
- ``fill_noise``: median over donors of o0's pixel noise (MAD of lag-1 differences) inside
  the star's footprint / in the ring just outside. Ideal 1; a smooth inpaint fill is ~0.
- ``sat_left``: like ``real_left``, for the saturated stars (the network leaves them alone).

``--method inpaint`` scores the existing remover (``star_removal.starless_from_image``, the
normalised-convolution inpaint behind ``--remove-stars``) with the same checks.

Usage:
    python tools/eval_star_n2n.py C:/nn_work/heldout_stacks/*.fits --model C:/nn/model_starless_v1.npz \
        --alpha 1.0 --json C:/nn/eval_starless_v1.json --crops C:/nn/crops_starless_v1
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
from src import star_n2n as sn  # noqa: E402

NO_CORES = False   # --no-cores: score without the residual-core step


def load_stack(path: str) -> np.ndarray:
    """Linear stack as (H, W, 3) float32: a FITS cube (3, H, W) or an (H, W, 3) .npy."""
    if path.lower().endswith(".npy"):
        return np.load(path).astype(np.float32)
    from astropy.io import fits
    d = np.asarray(fits.getdata(path), np.float32)
    return np.transpose(d, (1, 2, 0)) if d.ndim == 3 and d.shape[0] == 3 else d


def faint_discs(shape, avoid: np.ndarray, rng: np.random.Generator, n: int = 150):
    """(layer, ys, xs) of elliptical exponential discs placed away from ``avoid`` (bool)."""
    h, w, c = shape
    g = np.zeros(shape, np.float32)
    ys, xs = [], []
    tries = 0
    while len(ys) < n and tries < 50 * n:
        tries += 1
        y, x = int(rng.integers(24, h - 24)), int(rng.integers(24, w - 24))
        if avoid[y - 12:y + 13, x - 12:x + 13].any():
            continue
        sa = rng.uniform(2.0, 8.0)
        sb = sa * rng.uniform(0.4, 1.0)
        th = rng.uniform(0, np.pi)
        rr = int(np.ceil(4 * sa))
        y0, y1, x0, x1 = max(0, y - rr), min(h, y + rr + 1), max(0, x - rr), min(w, x + rr + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        u = ((xx - x) * np.cos(th) + (yy - y) * np.sin(th)) / sa
        v = (-(xx - x) * np.sin(th) + (yy - y) * np.cos(th)) / sb
        prof = np.exp(-np.sqrt(u * u + v * v) * 1.6783)
        peak = np.exp(rng.uniform(np.log(0.5), np.log(5.0)))
        col = np.clip(1.0 + 0.3 * rng.standard_normal(c), 0.3, 2.0)
        g[y0:y1, x0:x1] += (peak * prof[:, :, None] * col[None, None]).astype(np.float32)
        ys.append(y)
        xs.append(x)
    return g, np.array(ys), np.array(xs)


def tertiles(frac: np.ndarray, flux: np.ndarray) -> dict:
    ok = np.isfinite(frac) & np.isfinite(flux)
    frac, flux = frac[ok], flux[ok]
    if len(frac) < 3:
        return {"n": int(len(frac))}
    q = np.quantile(flux, [1 / 3, 2 / 3])
    bins = {"faint": flux < q[0], "mid": (flux >= q[0]) & (flux < q[1]), "bright": flux >= q[1]}
    out = {k: float(np.median(frac[m])) for k, m in bins.items() if m.any()}
    out.update(all=float(np.median(frac)), n=int(len(frac)))
    return out


def hole_depth(o0: np.ndarray, ys, xs, r: int = sn.STAMP_R) -> float:
    """Median over stars of the deepest azimuthal-mean dip of o0 (luminance) below the
    stamp-border sky, in sky sigma."""
    lum = o0.mean(-1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    rad = np.rint(np.hypot(yy, xx)).astype(int)
    out = []
    for y, x in zip(ys, xs):
        if not (r <= y < lum.shape[0] - r and r <= x < lum.shape[1] - r):
            continue
        cut = lum[y - r:y + r + 1, x - r:x + r + 1]
        border = np.concatenate([cut[0], cut[-1], cut[1:-1, 0], cut[1:-1, -1]])
        prof = ndi.mean(cut - np.median(border), rad, index=np.arange(r))
        out.append(float(np.min(prof)))
    return float(np.median(out)) if out else float("nan")


def save_crops(stem: str, y: np.ndarray, o0: np.ndarray, ys, xs, out_dir: str, size: int = 384) -> None:
    """Input | starless | stars, asinh-stretched on the input's scale, around the three
    densest-star windows."""
    try:
        from PIL import Image
    except Exception:
        return
    os.makedirs(out_dir, exist_ok=True)
    h, w, _ = y.shape
    cnt = np.zeros((h // size + 1, w // size + 1))
    np.add.at(cnt, (np.asarray(ys) // size, np.asarray(xs) // size), 1)
    order = np.argsort(-cnt.ravel())[:3]
    hi = float(np.quantile(y, 0.999))

    def stretch(a):
        return (np.clip(np.arcsinh(np.maximum(a, 0) / 2.0) / np.arcsinh(max(hi, 1.0) / 2.0), 0, 1) * 255).astype(np.uint8)

    for k, cell in enumerate(order):
        cy, cx = divmod(int(cell), cnt.shape[1])
        y0, x0 = min(cy * size, max(0, h - size)), min(cx * size, max(0, w - size))
        sl = (slice(y0, y0 + size), slice(x0, x0 + size))
        row = np.concatenate([stretch(y[sl]), stretch(o0[sl]), stretch(y[sl] - o0[sl])], axis=1)
        Image.fromarray(row).save(os.path.join(out_dir, f"{stem}_{k}.jpg"), quality=92)


def stamp_fwhm(field) -> float:
    """Median FWHM (px) of the field's bright isolated stamps, from second moments."""
    r = sn.STAMP_R
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    out = []
    for st in field.stamps:
        p = np.clip(st.mean(-1), 0.0, None)
        t = p.sum()
        if t > 0:
            out.append(2.3548 * np.sqrt(((yy ** 2 + xx ** 2) * p).sum() / t / 2.0))
    return float(np.median(out)) if out else 4.0


def fill_noise(o0: np.ndarray, field) -> float:
    """Median over donors of the output's lag-1 MAD noise inside the footprint / in the ring."""
    lum = o0.mean(-1)
    h, w = lum.shape
    vals = []
    for y, x, q in zip(field.donor_y, field.donor_x, field.donor_r):
        R = q + 6
        if not (R <= y < h - R - 1 and R <= x < w - R - 1):
            continue
        cut = lum[y - R:y + R + 1, x - R:x + R + 2]
        d = (cut[:, 1:] - cut[:, :-1])
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        rr = np.hypot(yy, xx)
        a, b = d[rr <= q], d[(rr > q + 1) & (rr <= R)]
        sa = np.median(np.abs(a - np.median(a)))
        sb = np.median(np.abs(b - np.median(b)))
        if sb > 0:
            vals.append(sa / sb)
    return float(np.median(vals)) if vals else float("nan")


def evaluate(path: str, w, alpha: float, test_alpha: float, seed: int, crops: str = "", tta: int = 2,
             method: str = "nn", post: bool = True) -> dict:
    """The checks above for one stack. ``post`` False scores the network alone (no star-layer
    constraint, hole guard or saturated fill)."""
    t = time.time()
    rng = np.random.default_rng(seed)
    img = load_stack(path)
    img = np.where(np.isfinite(img), img, 0.0).astype(np.float32)
    z, _, _, B = sn.normalise_image(img)
    field = sn.StarField(z, B)
    if method == "inpaint":
        from src.star_removal import starless_from_image
        fwhm = stamp_fwhm(field)

        def remover(zz):
            out = starless_from_image(zz, fwhm)
            return zz.copy() if out is None else out.astype(np.float32)
    else:
        def remover(zz):
            # the star field of the remover's own input, as remove_stars_nn builds it: with the
            # clean image's field, injected test stars had no footprint, and the star-layer
            # constraint (outside footprints: undetected faint stars only) kept them
            return sn.starless_z(zz, w, alpha, field if zz is z else sn.StarField(zz, B), tta,
                                 guard=post, saturated=post, cores=post and not NO_CORES)
    o0 = remover(z)
    res = {"stack": os.path.basename(path), "method": method, "donors": field.n_donors,
           "stamps": int(len(field.stamps)), "saturated": int(field.saturated.sum())}

    # real stars
    fin = sn.aperture_flux(z, field.donor_y, field.donor_x).mean(1)
    fout = sn.aperture_flux(o0, field.donor_y, field.donor_x).mean(1)
    res["real_left"] = tertiles(fout / np.where(fin > 0, fin, np.nan), fin)

    # injected stars (independent seed from training)
    s, ys, xs, flux = field.inject(rng, test_alpha)
    o1 = remover(z + s)
    left = sn.aperture_flux(o1 - o0, ys, xs).mean(1)
    fl = flux.mean(1) if len(flux) else np.zeros(0)
    res["inject_left"] = tertiles(left / np.where(fl > 0, fl, np.nan), fl)

    # faint extended sources, away from stars
    star_mask = np.zeros(z.shape[:2], bool)
    star_mask[field.det_y, field.det_x] = True
    star_mask = ndi.binary_dilation(star_mask, iterations=sn.STAMP_R)
    g, gy, gx = faint_discs(z.shape, star_mask, rng)
    o2 = remover(z + g)
    gin = sn.aperture_flux(g, gy, gx).mean(1)
    gout = sn.aperture_flux(o2 - o0, gy, gx).mean(1)
    res["faint_kept"] = float(np.nanmedian(gout / np.where(gin > 0, gin, np.nan))) if len(gy) else None

    # sky and holes
    d = (o0 - z)[~star_mask]
    res["sky_shift"] = float(np.median(d))
    res["sky_mad"] = float(1.4826 * np.median(np.abs(d - np.median(d))))
    bright = np.argsort(-np.nan_to_num(fin, nan=-np.inf))[:20]
    res["hole"] = hole_depth(o0, field.donor_y[bright], field.donor_x[bright])
    res["fill_noise"] = fill_noise(o0, field)
    sy, sx = field.det_y[field.saturated], field.det_x[field.saturated]
    sin = sn.aperture_flux(z, sy, sx).mean(1)
    sout = sn.aperture_flux(o0, sy, sx).mean(1)
    res["sat_left"] = float(np.nanmedian(sout / np.where(sin > 0, sin, np.nan))) if len(sy) else None
    res["seconds"] = round(time.time() - t, 1)
    if crops:
        save_crops(os.path.splitext(os.path.basename(path))[0], z, o0, field.donor_y, field.donor_x, crops)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stacks", nargs="+", help="linear stacks (FITS cube or .npy)")
    ap.add_argument("--model", default=sn.MODEL_PATH)
    ap.add_argument("--alpha", type=float, default=1.0, help="the alpha the model's bank was built with")
    ap.add_argument("--test-alpha", type=float, default=0.3, help="density of the injection-recovery layer")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--method", choices=["nn", "inpaint"], default="nn",
                    help="nn: the Noisier2Noise model; inpaint: the existing --remove-stars")
    ap.add_argument("--tta", type=int, default=2,
                    help="disc realisations averaged per output (starless_z; 0 = plain forward pass)")
    ap.add_argument("--no-cores", action="store_true", help="skip remove_residual_cores (the before column)")
    ap.add_argument("--json")
    ap.add_argument("--crops", default="")
    args = ap.parse_args()
    global NO_CORES
    NO_CORES = args.no_cores
    w = nd.load_weights(args.model) if args.method == "nn" else None
    rows = []
    for p in args.stacks:
        r = evaluate(p, w, args.alpha, args.test_alpha, args.seed, args.crops, args.tta, args.method)
        rows.append(r)
        print(json.dumps(r), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"model": args.model, "alpha": args.alpha, "test_alpha": args.test_alpha,
                       "results": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
