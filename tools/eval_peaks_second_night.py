"""Confirm a deconvolver's faint peaks against a second night of the same target.

``eval_nn_faint.py`` confirms faint output peaks against the session's sharp stack (its
sharpest 30% of frames). Synthetic-blur models show about twice v3's unconfirmed peaks there
(dev-notes/nn-deconv-runbook.md); either they invent them, or the shallow sharp stack cannot
confirm the faintest real sources such a model recovers. A second night separates the two:
its full stack (every frame of that session: independent noise, ~3x the sharp stack's
frames) is a deeper reference. Invented peaks stay unconfirmed; real faint ones get
confirmed.

Per session S with a partner session R of the same target (held-out pairs only):
peaks are found exactly as in ``eval_nn_faint.false_sources`` on S's central crop; their
positions are mapped into R's grid (blind rigid match, then an affine fit on all matched
stars -- no resampling) and confirmed when R's matched-filtered luminance is > 2.5 of its
own sigma within 2 px. Also reported: the same peaks against S's sharp stack, and the
confirmation rate of random sky positions (chance) for both references.

Usage:
    python tools/eval_peaks_second_night.py --pairs C:/nn_work/nn_deconv/pairs \
        --model v3=src/data/nn_deconv.npz --model s4=C:/nn_work/model_deconv_s4.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from src import nn_denoise as nd  # noqa: E402

THRESHOLDS = (2.0, 3.0)


def target_of(name: str) -> str:
    return name.rsplit("_", 2)[0]


def central_offset(shape, crop):
    h, w = shape[:2]
    ch, cw = min(crop[0], h), min(crop[1], w)
    return (h - ch) // 2, (w - cw) // 2


def full_stack(d: str) -> np.ndarray:
    with open(os.path.join(d, "meta.json")) as fh:
        meta = json.load(fh)
    a, b = np.load(os.path.join(d, "input.npy")), np.load(os.path.join(d, "target.npy"))
    ni, nt = meta["n_input"], meta["n_target"]
    return ((a * ni + b * nt) / (ni + nt)).astype(np.float32), meta


def detect_xy(lum):
    from src.star_detect import detect_stars_matched_filter as detect
    st = detect(lum - np.median(lum))
    return st[np.argsort(-st["flux"])]


def map_s_to_r(lum_s, lum_r):
    """Affine (2, 3) mapping S pixel (x, y) to R, and its residual rms on matched stars."""
    from scipy.spatial import cKDTree

    from src.blind_match import match_rigid_unknown_rotation
    ss, sr = detect_xy(lum_s), detect_xy(lum_r)
    rigid = match_rigid_unknown_rotation(ss, sr, max_stars=60, pixel_tol=2.0)
    if rigid is None:
        return None, None, 0
    p = np.asarray(rigid.params, float)[:2]
    xs, ys = np.asarray(ss["xcentroid"], float), np.asarray(ss["ycentroid"], float)
    xr, yr = np.asarray(sr["xcentroid"], float), np.asarray(sr["ycentroid"], float)
    tree = cKDTree(np.c_[xr, yr])
    for tol in (3.0, 1.5, 1.0):
        px = p[0, 0] * xs + p[0, 1] * ys + p[0, 2]
        py = p[1, 0] * xs + p[1, 1] * ys + p[1, 2]
        d, j = tree.query(np.c_[px, py], distance_upper_bound=tol)
        ok = np.isfinite(d)
        A = np.c_[xs[ok], ys[ok], np.ones(ok.sum())]
        sol, *_ = np.linalg.lstsq(A, np.c_[xr[j[ok]], yr[j[ok]]], rcond=None)
        p = sol.T
    res = A @ sol - np.c_[xr[j[ok]], yr[j[ok]]]
    return p, float(np.sqrt(np.median((res ** 2).sum(1)))), int(ok.sum())


def confirm_map(img, fwhm):
    """True where a source is present: the matched-filtered luminance > 2.5 of its own
    sigma within 2 px (as eval_nn_faint.false_sources)."""
    bs = ndi.gaussian_filter((img - nd.background_at(nd.background_grid(img), 0, 0, *img.shape[:2])).mean(axis=2),
                             fwhm / 2.3548)
    d = (bs[:, 4:] - bs[:, :-4])[::2].ravel()
    bsig = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    return ndi.maximum_filter(bs / bsig, size=5) > 2.5


def peak_map(out, a, sigma, fwhm):
    """(peaks bool, lum, candidate-sky mask) exactly as eval_nn_faint.false_sources."""
    s = float(sigma.mean())
    lum = (out - nd.background_at(nd.background_grid(out), 0, 0, *out.shape[:2])).mean(axis=2) / s
    lum = ndi.gaussian_filter(lum, 0.7)
    za = ((a - nd.background_at(nd.background_grid(a), 0, 0, *a.shape[:2])) / sigma).mean(axis=2)
    keep_out = ndi.maximum_filter(za > 10, size=int(8 * fwhm) | 1)
    edge = np.zeros_like(keep_out)
    edge[16:-16, 16:-16] = True
    peaks = (lum == ndi.maximum_filter(lum, size=7)) & edge & ~keep_out
    return peaks, lum, edge & ~keep_out


def main() -> None:
    from eval_nn_faint import CROP, central
    from train_nn_denoise import is_holdout
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", action="append", default=[], help="NAME=weights.npz (repeatable)")
    ap.add_argument("--restore", type=float, default=4.0)
    ap.add_argument("--json")
    args = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(args.pairs, "*"))
                  if os.path.exists(os.path.join(d, "meta.json")) and is_holdout(os.path.basename(d)))
    by_target = {}
    for d in dirs:
        by_target.setdefault(target_of(os.path.basename(d)), []).append(d)
    methods = {"none": None, "nn_denoise": nd._MODEL_PATH}
    methods.update(dict(m.split("=", 1) for m in args.model))
    results = {}
    for tgt, ds in sorted(by_target.items()):
        if len(ds) < 2:
            print(f"{tgt}: one session only, skipped", flush=True)
            continue
        for s_dir, r_dir in ((ds[0], ds[1]), (ds[1], ds[0])):
            name = os.path.basename(s_dir)
            with open(os.path.join(s_dir, "meta.json")) as fh:
                meta_s = json.load(fh)
            fwhm = float(meta_s.get("fwhm_input") or 3.0)
            a_full = np.load(os.path.join(s_dir, "input.npy")).astype(np.float32)
            a = central(a_full)
            b = central(np.load(os.path.join(s_dir, "target.npy")).astype(np.float32))
            y0, x0 = central_offset(a_full.shape, CROP)
            ref, meta_r = full_stack(r_dir)
            p, rms, nmatch = map_s_to_r(a_full.mean(-1) - np.median(a_full), ref.mean(-1) - np.median(ref))
            if p is None:
                print(f"{name}: no star match with {os.path.basename(r_dir)}", flush=True)
                continue
            n_ref = meta_r["n_input"] + meta_r["n_target"]
            print(f"{name} -> {os.path.basename(r_dir)}: {nmatch} stars, rms {rms:.2f} px; reference "
                  f"{n_ref} frames vs sharp stack {meta_s['n_target']}", flush=True)
            sigma = nd.noise_sigma(a).astype(np.float32)
            conf_r = confirm_map(ref, float(meta_r.get("fwhm_input") or fwhm))
            conf_b = confirm_map(b, fwhm)
            hr, wr = conf_r.shape

            def in_ref(ys, xs):
                X, Y = xs + x0, ys + y0
                rx = np.rint(p[0, 0] * X + p[0, 1] * Y + p[0, 2]).astype(int)
                ry = np.rint(p[1, 0] * X + p[1, 1] * Y + p[1, 2]).astype(int)
                ok = (rx >= 16) & (rx < wr - 16) & (ry >= 16) & (ry < hr - 16)
                return ok, ry, rx

            row = {"reference": os.path.basename(r_dir), "match_rms": rms, "ref_frames": n_ref,
                   "sharp_frames": meta_s["n_target"]}
            for k, mp in methods.items():
                out = a if mp is None else nd.nn_denoise(a, model_path=mp, restore_sigma=args.restore)
                peaks, lum, sky = peak_map(out, a, sigma, fwhm)
                sy, sx = np.nonzero(sky)
                ok, ry, rx = in_ref(sy, sx)
                v = {"chance_ref": float(conf_r[ry[ok], rx[ok]].mean()),
                     "chance_sharp": float(conf_b[sy[ok], sx[ok]].mean())}
                for t in THRESHOLDS:
                    py, px = np.nonzero(peaks & (lum > t))
                    okp, qy, qx = in_ref(py, px)
                    py, px, qy, qx = py[okp], px[okp], qy[okp], qx[okp]
                    area = ok.sum() / max(sky.sum(), 1) * sky.size / 1e6
                    cr, cb = conf_r[qy, qx], conf_b[py, px]
                    v[str(t)] = {"per_mpx": len(py) / area, "conf_ref": float(cr.mean()) if len(py) else None,
                                 "conf_sharp": float(cb.mean()) if len(py) else None,
                                 "unconf_ref_per_mpx": float((~cr).sum() / area),
                                 "unconf_sharp_per_mpx": float((~cb).sum() / area),
                                 "unconf_both_per_mpx": float((~cr & ~cb).sum() / area)}
                row[k] = v
                t2 = v["2.0"]
                print(f"   {k:12s} >2 sigma {t2['per_mpx']:6.0f}/Mpx  unconfirmed: sharp {t2['unconf_sharp_per_mpx']:5.1f}"
                      f"  2nd night {t2['unconf_ref_per_mpx']:5.1f}  both {t2['unconf_both_per_mpx']:5.1f}   "
                      f"(chance sharp {v['chance_sharp']:.2f}, 2nd night {v['chance_ref']:.2f})", flush=True)
            results[name] = row
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
