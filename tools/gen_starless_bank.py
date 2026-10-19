"""Training bank for the Noisier2Noise star remover (src/star_n2n.py, EXPERIMENTAL).

Per session (a ``gen_n2n_pairs.py --mode seeing`` pair directory -- input.npy + target.npy
together are the full-depth stack), in noise units: input = image + an injected star layer
(``StarField.inject``, ``--alpha`` x the real star count), target = the image as it is. Several
independent injections per session (``--realisations``); half of the patches are centred on
a star, real or injected. Weight = 1 / the input's noise variance (``target_variance``: sky plus the
photon noise of the stars), so bright stars do not dominate the loss.

The output has the keys ``tools/train_nn_denoise.py train`` reads (za, zb, w, sess); train with
it unchanged and without ``--inject``. ``<bank>.json`` records alpha: inference must use the
same one (``remove_stars_nn(..., alpha=)``, ``eval_star_n2n.py --alpha``).

Usage:
    python tools/gen_starless_bank.py --pairs C:/nn_work/nn_deconv/pairs --bank C:/nn/bank_starless.npz
    python tools/train_nn_denoise.py train --bank C:/nn/bank_starless.npz --out C:/nn/model_starless_v1.npz --resume
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from train_nn_denoise import PATCH, is_holdout, target_variance  # noqa: E402

from src import star_n2n as sn  # noqa: E402


def full_stack(pair_dir: str) -> np.ndarray:
    """The session's full-depth stack from a seeing pair: frame-count-weighted mean of
    input.npy (the rest) and target.npy (the sharpest), which hold disjoint frames."""
    with open(os.path.join(pair_dir, "meta.json")) as fh:
        meta = json.load(fh)
    a = np.load(os.path.join(pair_dir, "input.npy"))
    b = np.load(os.path.join(pair_dir, "target.npy"))
    na, nb = float(meta.get("n_input", 1)), float(meta.get("n_target", 1))
    return ((na * a + nb * b) / (na + nb)).astype(np.float32)


def session_patches(img: np.ndarray, rng: np.random.Generator, per_session: int, alpha: float,
                    realisations: int):
    """(za, zb, w) float16 patch lists for one image, or None if it has too few stars."""
    z, _, _, B = sn.normalise_image(img)
    field = sn.StarField(z, B)
    if not field.ok:
        return None, field
    h, wd, _ = z.shape
    m = 16                      # skip the stack's ragged edge
    za_l, zb_l, w_l = [], [], []
    for r in range(realisations):
        s, ys, xs, _ = field.inject(rng, alpha)
        za = z + s
        w = (1.0 / target_variance(za, B)).astype(np.float32)
        n = per_session // realisations + (1 if r < per_session % realisations else 0)
        # centre half the patches on a star drawn from real donors and injected stars alike:
        # centring on injected stars only (v1) raised the share of injected stars the network
        # sees above alpha / (1 + alpha), which the correction formula assumes
        cy = np.concatenate([field.donor_y, ys])
        cx = np.concatenate([field.donor_x, xs])
        for k in range(n):
            if len(cy) and k % 2 == 0:
                j = rng.integers(len(cy))
                y = int(np.clip(cy[j] - rng.integers(16, PATCH - 16), m, h - PATCH - m))
                x = int(np.clip(cx[j] - rng.integers(16, PATCH - 16), m, wd - PATCH - m))
            else:
                y = int(rng.integers(m, h - PATCH - m))
                x = int(rng.integers(m, wd - PATCH - m))
            za_l.append(za[y:y + PATCH, x:x + PATCH].astype(np.float16))
            zb_l.append(z[y:y + PATCH, x:x + PATCH].astype(np.float16))
            w_l.append(w[y:y + PATCH, x:x + PATCH].astype(np.float16))
    return (za_l, zb_l, w_l), field


def build_bank(pairs_dir: str, out: str, per_session: int, alpha: float, realisations: int,
               seed: int = 0, limit: int = 0) -> None:
    rng = np.random.default_rng(seed)
    za_l, zb_l, w_l, sess = [], [], [], []
    dirs = sorted(d for d in glob.glob(os.path.join(pairs_dir, "*"))
                  if os.path.exists(os.path.join(d, "meta.json")))
    used = []
    for i, d in enumerate(dirs):
        name = os.path.basename(d)
        if is_holdout(name):
            continue
        if limit and len(used) >= limit:
            break
        t = time.time()
        got, field = session_patches(full_stack(d), rng, per_session, alpha, realisations)
        if got is None:
            print(f"[{i + 1}/{len(dirs)}] {name}: skipped ({field.n_donors} donors, "
                  f"{len(field.stamps)} stamps)", flush=True)
            continue
        za_l += got[0]
        zb_l += got[1]
        w_l += got[2]
        sess += [i] * len(got[0])
        used.append(name)
        print(f"[{i + 1}/{len(dirs)}] {name}: {field.n_donors} donors, {len(field.stamps)} stamps, "
              f"{int(field.saturated.sum())} saturated, {time.time() - t:.0f} s", flush=True)
    if not za_l:
        raise SystemExit("no usable sessions")
    np.savez(out, za=np.stack(za_l), zb=np.stack(zb_l), w=np.stack(w_l), sess=np.array(sess))
    with open(os.path.splitext(out)[0] + ".json", "w") as fh:
        json.dump({"alpha": alpha, "realisations": realisations, "per_session": per_session,
                   "seed": seed, "sessions": used, "patches": len(za_l)}, fh, indent=1)
    print(f"bank: {len(za_l)} patches from {len(used)} sessions, alpha {alpha} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True, help="gen_n2n_pairs.py --mode seeing output")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--per-session", type=int, default=240)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="injected / real star density; inference must use the same value")
    ap.add_argument("--realisations", type=int, default=4, help="independent injections per session")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many sessions (0 = all)")
    args = ap.parse_args()
    build_bank(args.pairs, args.bank, args.per_session, args.alpha, args.realisations, args.seed, args.limit)


if __name__ == "__main__":
    main()
