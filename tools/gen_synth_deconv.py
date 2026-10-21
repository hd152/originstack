"""Ingredients for synthetic-blur training of the learned deconvolver (``--deconvolve nn``,
dev-notes/nn-deconv-runbook.md, "synthetic-blur training").

The seeing pairs (tools/gen_n2n_pairs.py --mode seeing) cap sharpening at a session's
sharpest-30% stack. Synthetic pairs need a truth sharper than any real data; they are built
on the fly during training (tools/train_nn_denoise.py ``--synth``) from two ingredients this
script cuts out of the seeing pairs, both in the input stack's noise units
(``nn_denoise.normalise``, as the network sees them):

- ``bg``: clean, star-free extended background -- the sharp (target) stack denoised with the
  bundled denoiser (no low-frequency restore, which would put noise back) and its stars
  inpainted (``star_removal.remove_stars``). Synthetic stars with a narrow PSF go on top.
- ``noise``: real stack noise, *with* its spatial correlation (debayer + warp): the input
  stack minus its denoised self, cut where the sky is empty. White noise would not transfer.
- ``sky``: the sky level B per channel (noise units) of each patch's session, for the
  Poisson term ``sqrt(1 + signal / B)``.

Held-out targets (``train_nn_denoise.HOLDOUT``) go to a separate file for evaluation.

Usage:
    python tools/gen_synth_deconv.py --pairs C:/nn_work/nn_deconv/pairs --out C:/nn_work/synth
      -> synth/train.npz, synth/holdout.npz
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

BG_SIZE = 192          # background crop: the 128 px patch plus room for the blur kernel
NOISE_SIZE = 128
EDGE = 48              # skip the stack's ragged edge


def starless(z: np.ndarray) -> np.ndarray:
    """``z`` (noise units, denoised) with its stars inpainted."""
    from src.star_detect import detect_stars_matched_filter
    from src.star_removal import remove_stars
    lum = z.mean(-1)
    st = detect_stars_matched_filter(lum - np.median(lum), fwhm=3.5, k_confirm=8.0)
    if st is None or len(st) == 0:
        return z
    out, _ = remove_stars(z.astype(np.float32), st, fwhm=3.5, radius_scale=2.2)
    return out


def session_parts(d: str, n_bg: int, n_noise: int, rng):
    from train_nn_denoise import SKY_MATCH_SIGMA, sky_level_z
    a = np.load(os.path.join(d, "input.npy"))
    b = np.load(os.path.join(d, "target.npy"))
    b = b - np.stack([ndi.gaussian_filter(b[..., c] - a[..., c], SKY_MATCH_SIGMA) for c in range(3)], -1)
    grid, sigma = nd.normalise(a)
    bg = nd.background_at(grid, 0, 0, *a.shape[:2])
    s = sigma.astype(np.float32)
    sky = sky_level_z(bg, sigma).astype(np.float32)
    za = (a - bg) / s
    w = nd.load_weights()
    clean_a = nd.denoise_z(za, w)
    noise = za - clean_a
    clean_b = starless(nd.denoise_z((b - bg) / s, w))
    h, wd, _ = za.shape
    bgs = []
    for _ in range(n_bg):
        y = rng.integers(EDGE, h - BG_SIZE - EDGE)
        x = rng.integers(EDGE, wd - BG_SIZE - EDGE)
        bgs.append(clean_b[y:y + BG_SIZE, x:x + BG_SIZE].astype(np.float16))
    # noise where the sky is empty: no star or nebula residue mistaken for noise
    lum = ndi.gaussian_filter(clean_a.mean(-1), 2.0)
    empty = ndi.uniform_filter((lum < 1.0).astype(np.float32), NOISE_SIZE) > 0.97
    ys, xs = np.nonzero(empty[EDGE + NOISE_SIZE // 2:h - EDGE - NOISE_SIZE // 2,
                              EDGE + NOISE_SIZE // 2:wd - EDGE - NOISE_SIZE // 2])
    ns = []
    if len(ys):
        for k in rng.choice(len(ys), min(n_noise, len(ys)), replace=False):
            y, x = ys[k] + EDGE, xs[k] + EDGE
            ns.append(noise[y:y + NOISE_SIZE, x:x + NOISE_SIZE].astype(np.float16))
    return bgs, ns, sky


def main() -> None:
    from train_nn_denoise import is_holdout
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg-per-session", type=int, default=60)
    ap.add_argument("--noise-per-session", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    parts = {"train": ([], [], [], [], []), "holdout": ([], [], [], [], [])}
    dirs = sorted(d for d in glob.glob(os.path.join(args.pairs, "*")) if os.path.exists(os.path.join(d, "meta.json")))
    for i, d in enumerate(dirs):
        name = os.path.basename(d)
        t = time.time()
        bgs, ns, sky = session_parts(d, args.bg_per_session, args.noise_per_session, rng)
        bl, nl, bs, ns_s, names = parts["holdout" if is_holdout(name) else "train"]
        bl += bgs
        nl += ns
        bs += [sky] * len(bgs)
        ns_s += [sky] * len(ns)
        names.append(name)
        print(f"[{i + 1}/{len(dirs)}] {name}: {len(bgs)} bg, {len(ns)} noise, sky B {np.round(sky, 1)}, "
              f"{time.time() - t:.0f} s", flush=True)
    for k, (bl, nl, bs, ns_s, names) in parts.items():
        path = os.path.join(args.out, f"{k}.npz")
        np.savez(path, bg=np.stack(bl), noise=np.stack(nl), bg_sky=np.stack(bs), noise_sky=np.stack(ns_s))
        with open(os.path.join(args.out, f"{k}_sessions.json"), "w") as fh:
            json.dump(names, fh)
        print(f"{k}: {len(bl)} bg, {len(nl)} noise -> {path}")


if __name__ == "__main__":
    main()
