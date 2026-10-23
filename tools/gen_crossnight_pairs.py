"""Cross-night training pairs for the learned deconvolver, from ``gen_n2n_pairs.py --mode seeing`` output.

A seeing pair (one session's sharpest 30% of frames against the rest) only teaches the
network to sharpen as far as that night's best frames: the median width ratio is 0.89. Two
nights of the same target differ far more (median 0.61, 27 targets on the first machine), so
the sharper night's stack is a sharper target for the softer night's -- the same sky,
independent frames and noise, a real PSF difference. This writes such pairs in the seeing
pairs' own format, so ``train_nn_denoise.py bank --mode deconv`` takes them unchanged.

Per target with two or more sessions under ``--seeing``: the sharpest night's ``target.npy``
(its sharp-30% stack) is the target; the input is the softest night's ``input.npy`` or, when
that gap is wider than ``--min-ratio`` allows, its own ``target.npy``. The target is aligned
onto the input with ``--merge``'s registration (blind star match, any rotation), flux-matched
to it with ``--merge``'s per-channel gain and quadratic sky-offset surface, and both are cut to
the largest rectangle the two cover; a pair whose stars land more than ``--max-offset`` px
apart after the warp is skipped. The width ratio is then measured on the same stars in
both (``common_star_fwhm``); a pair outside ``--min-ratio`` .. ``--max-ratio`` is skipped:
a 2x gap is no longer a gentle deconvolution, and a network trained on it would over-sharpen
sessions that are already good.

Output per pair, under ``--out``: ``<target>_X_<sharp date>_on_<soft date>/input.npy``,
``target.npy`` and ``meta.json`` (``mode: crossnight``; names start with the target, so
``train_nn_denoise.HOLDOUT`` still excludes the held-out targets).

Usage:
    python tools/gen_crossnight_pairs.py --seeing C:/nn_work/nn_deconv/pairs --out C:/nn_work/nn_deconv/crossnight
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

_DATE = re.compile(r"_(\d{4}-\d\d-\d\d)_\d\d-\d\d-\d\d$")
_TRIM_PX = 12          # footprint rim dropped: the warp's edge and each stack's ragged border


def target_name(session_dir: str) -> str:
    return _DATE.sub("", os.path.basename(os.path.normpath(session_dir)))


def largest_rectangle(mask: np.ndarray, step: int = 8):
    """(y0, y1, x0, x1) of the largest axis-aligned rectangle inside ``mask``, found on a
    ``step``-px grid (a cell counts only when all of it is inside)."""
    h, w = mask.shape
    gh, gw = h // step, w // step
    cells = mask[:gh * step, :gw * step].reshape(gh, step, gw, step).all(axis=(1, 3))
    heights = np.zeros(gw, int)
    best, box = 0, None
    for r in range(gh):
        heights = np.where(cells[r], heights + 1, 0)
        stack = []                                  # largest rectangle in a histogram
        for c in range(gw + 1):
            hc = heights[c] if c < gw else 0
            start = c
            while stack and stack[-1][1] >= hc:
                s, hs = stack.pop()
                if hs * (c - s) > best:
                    best, box = hs * (c - s), (r - hs + 1, r + 1, s, c)
                start = s
            stack.append((start, hc))
    if box is None:
        return None
    r0, r1, c0, c1 = box
    return r0 * step, r1 * step, c0 * step, c1 * step


def align_onto(soft: dict, tgt: np.ndarray):
    """``tgt`` warped onto the soft night's grid and cut, with the soft night's stacks, to
    the rectangle all of them cover: (soft stacks cut, tgt warped, info) or None.

    The soft night's two stacks share one pixel grid, so the transform is found once, on
    its sharp stack (the better star positions): registering each separately gave Flame
    Nebula -6.9 deg for one and 0.0 for the other."""
    from scipy import ndimage as ndi

    from src.merge import _detect_stars, _lum, _register_stack
    from src.registration import apply_transform
    from src.utils import embed_to_shape

    ref = soft["target"]
    H, W = max(ref.shape[0], tgt.shape[0]), max(ref.shape[1], tgt.shape[1])
    soft = {k: embed_to_shape(v, H, W) for k, v in soft.items()}
    b = embed_to_shape(tgt, H, W)
    foot_a = embed_to_shape(np.ones(ref.shape[:2], np.float32), H, W)
    foot_b = embed_to_shape(np.ones(tgt.shape[:2], np.float32), H, W)
    la = _lum(soft["target"])
    tf, shift = _register_stack(la, _lum(b), _detect_stars(la), prev_shape=tgt.shape[:2])
    if tf is None and shift is None:
        return None
    b = apply_transform(b, shift=shift, transform=tf).astype(np.float32)
    foot_b = apply_transform(foot_b, shift=shift, transform=tf)
    valid = ndi.binary_erosion((foot_a > 0.5) & (foot_b > 0.5), iterations=_TRIM_PX)
    box = largest_rectangle(valid)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    soft = {k: np.ascontiguousarray(v[y0:y1, x0:x1]) for k, v in soft.items()}
    b = np.ascontiguousarray(b[y0:y1, x0:x1])
    rot = float(np.degrees(np.arctan2(tf.params[1, 0], tf.params[0, 0]))) if tf is not None else 0.0
    info = {"box": [int(v) for v in box], "rotation_deg": round(rot, 2),
            "registration": "transform" if tf is not None else "shift",
            "star_offset_px": star_offset(soft["target"], b)}
    return soft, b, info


def star_offset(a: np.ndarray, b: np.ndarray, n: int = 150) -> float:
    """Median distance (px) from ``b``'s brightest stars to their nearest star in ``a``; inf
    when fewer than half have one within 4 px -- the check that the warp put the same stars
    on the same pixels."""
    from scipy.spatial import cKDTree

    from src.merge import _lum
    from src.star_detect import detect_stars_matched_filter as detect

    def xy(img):
        lum = _lum(img)
        st = detect(lum - np.median(lum))
        st = st[np.argsort(-st["flux"])][:n]
        return np.c_[np.asarray(st["xcentroid"]), np.asarray(st["ycentroid"])]
    pa, pb = xy(a), xy(b)
    if len(pa) < 20 or len(pb) < 20:
        return float("inf")
    d, _ = cKDTree(pa).query(pb, distance_upper_bound=4.0)
    d = d[np.isfinite(d)]
    return round(float(np.median(d)), 3) if len(d) >= len(pb) // 2 else float("inf")


def flux_match(a: np.ndarray, b: np.ndarray):
    """``b`` on ``a``'s flux scale: ``--merge``'s per-channel gain and quadratic sky-offset
    surface (a constant offset when the surface cannot be fitted)."""
    from src.merge import _match_flux_scale, _sky_offset_surface
    valid = np.ones(a.shape[:2], bool)
    gains, offsets, n_fit = _match_flux_scale(a, b, valid)
    surface = _sky_offset_surface(a, b, valid, gains)
    b = (gains * b + (surface if surface is not None else offsets)).astype(np.float32)
    return b, {"gains": [round(float(g), 4) for g in gains], "flux_pixels": n_fit,
               "sky_surface": surface is not None}


def measured_ratio(inp: np.ndarray, tgt: np.ndarray):
    """Median per-star FWHM ratio target / input on the same stars, or None."""
    from common_star_fwhm import common_star_fwhm
    r = common_star_fwhm(np.moveaxis(tgt, -1, 0), np.moveaxis(inp, -1, 0))
    return None if r is None else (r["ratio_a_over_b"], r["a"], r["b"], r["n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeing", required=True, help="gen_n2n_pairs.py --mode seeing output folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ratio", type=float, default=0.65,
                    help="narrowest accepted target / input star width (measured)")
    ap.add_argument("--max-ratio", type=float, default=0.85,
                    help="widest accepted ratio: above it the pair teaches less than a seeing pair")
    ap.add_argument("--max-offset", type=float, default=1.0,
                    help="largest accepted median star offset (px) between the aligned stacks")
    args = ap.parse_args()

    by = collections.defaultdict(list)
    for m in sorted(glob.glob(os.path.join(args.seeing, "*", "meta.json"))):
        d = os.path.dirname(m)
        with open(m) as fh:
            meta = json.load(fh)
        if meta.get("mode") == "seeing" and min(meta.get("n_target", 0), meta.get("n_input", 0)) >= 3:
            by[target_name(d)].append((meta["fwhm_target"], d, meta))
    os.makedirs(args.out, exist_ok=True)
    summary = []
    for t, sess in sorted(by.items()):
        if len(sess) < 2:
            continue
        sess.sort(key=lambda s: s[0])
        (_, sharp_d, sharp_m), (_, soft_d, soft_m) = sess[0], sess[-1]
        date = lambda d: _DATE.search(os.path.basename(d)).group(1)      # noqa: E731
        name = f"{t}_X_{date(sharp_d)}_on_{date(soft_d)}"
        dst = os.path.join(args.out, name)
        if os.path.exists(os.path.join(dst, "meta.json")):
            print(f"{name}: done already", flush=True)
            continue
        t0 = time.time()
        tgt = np.load(os.path.join(sharp_d, "target.npy"))
        res = align_onto({k: np.load(os.path.join(soft_d, f"{k}.npy")) for k in ("input", "target")}, tgt)
        if res is None or res[2]["star_offset_px"] > args.max_offset:
            why = "does not register" if res is None else f"stars {res[2]['star_offset_px']} px apart after the warp"
            print(f"{name}: skipped, {why} ({time.time() - t0:.0f} s)", flush=True)
            summary.append((name, None))
            continue
        soft, b_aligned, info = res
        # the softer night's rest-of-frames stack, or its sharp stack when the gap is too wide
        done = None
        for which, n_in in (("input", soft_m["n_input"]), ("target", soft_m["n_target"])):
            a = soft[which]
            b, flux = flux_match(a, b_aligned)
            r = measured_ratio(a, b)
            if r is None:
                print(f"{name}: soft {which}: too few shared stars to measure", flush=True)
                continue
            print(f"{name}: soft {which}: measured ratio {r[0]:.3f} ({r[3]} stars, {r[1]:.2f} / {r[2]:.2f} px), "
                  f"crop {a.shape[1]}x{a.shape[0]}, rot {info['rotation_deg']} deg, "
                  f"offset {info['star_offset_px']} px, gains {flux['gains']}", flush=True)
            if r[0] < args.min_ratio:
                continue                                # try the sharper soft stack
            if r[0] <= args.max_ratio:
                done = (which, n_in, a, b, {**info, **flux}, r)
            break
        if done is None:
            summary.append((name, None))
            print(f"{name}: skipped ({time.time() - t0:.0f} s)", flush=True)
            continue
        which, n_in, a, b, info, r = done
        os.makedirs(dst, exist_ok=True)
        np.save(os.path.join(dst, "input.npy"), a)
        np.save(os.path.join(dst, "target.npy"), b)
        meta = {"mode": "crossnight", "target_session": sharp_m["session"], "input_session": soft_m["session"],
                "input_stack": which, "n_target": sharp_m["n_target"], "n_input": n_in,
                "shape": list(a.shape), "ratio": r[0], "fwhm_target": r[1], "fwhm_input": r[2],
                "ratio_stars": r[3], **info}
        with open(os.path.join(dst, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        summary.append((name, r[0]))
        print(f"{name}: ok ({time.time() - t0:.0f} s)", flush=True)
    kept = [s for s in summary if s[1] is not None]
    print(f"\n{len(kept)} pairs written, {len(summary) - len(kept)} skipped")


if __name__ == "__main__":
    main()
