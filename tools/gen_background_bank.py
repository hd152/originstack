"""Background-extraction captures from real sessions: what Phase 4 hands DBE, and what DBE
removes. Made as training data for a learned background model (2026-10, not shipped: it learned
to remove faint nebula, see dev-notes/phase4-quality.md); the held-out captures feed
tools/eval_background_nights.py, the two-night yardstick for any background method.

For each session's linear stack, Phase 4 runs up to background extraction (the real
``--from-stack`` path, the run's own ``_config.toml``) and DBE runs on exactly what Phase 4
hands it. Saved per session, block-averaged by ``--factor`` (backgrounds are smooth):

    <name>.npz: image  -- DBE's input (H/f, W/f, 3) float32
                clean  -- DBE's output (the image with its background removed)
                bg     -- image - clean (the background DBE removed: sky glow, gradient,
                          vignetting residual)
                sigma  -- per-channel pixel noise of the full-resolution input
                target -- target name (sessions of one target share it)

Sessions of ``--holdout`` targets also keep the full-resolution DBE inputs
(``<name>_full.pkl``, as tools' background sweep expects) for scoring against a second night.

Usage:
    python tools/gen_background_bank.py --stacks C:/nn_work/nnbg/stacks --out C:/nn_work/nnbg/bank
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import pickle
import re
import sys

import numpy as np
from astropy.io import fits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# the targets background methods are scored on (tools/eval_background_nights.py)
HOLDOUT = ("Flaming_Star_Nebula", "Pleiades", "Hercules_Cluster", "Lagoon_Nebula", "Horsehead_Nebula",
           "Eagle_Nebula", "NGC_2244", "NGC_891", "Trifid_Nebula", "Crab_Nebula", "Messier_101",
           "Sunflower_Galaxy", "Pinwheel_Galaxy", "Andromeda_Galaxy")
_DATE = re.compile(r"_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d$")


def target_of(name: str) -> str:
    return _DATE.sub("", name)


def is_holdout(name: str) -> bool:
    return target_of(name) in HOLDOUT


class _Stop(BaseException):
    pass


def capture(stack_path: str):
    """(method, DBE input HWC float32, DBE kwargs) as Phase 4 would call it."""
    import src.postprocess as pp
    from src import cli
    from src.models import ProcessingStats
    from src.pipeline import prepare_linear_for_phase4
    box = {}
    real = {"dbe": pp.dynamic_background_extraction, "wavelet": pp.wavelet_background_extraction,
            "mesh": pp.apply_background_extraction}

    def recorder(name):
        def rec(rgb, **kw):
            box.update(method=name, rgb=np.array(rgb, np.float32), kw=kw)
            raise _Stop()
        return rec
    for name in real:
        setattr(pp, {"dbe": "dynamic_background_extraction", "wavelet": "wavelet_background_extraction",
                     "mesh": "apply_background_extraction"}[name], recorder(name))
    try:
        data, hdr = fits.getdata(stack_path, header=True)
        d = np.asarray(data, np.float32)
        stacked = np.moveaxis(d, 0, -1).copy() if d.shape[0] == 3 else d.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            args = cli.parse_args(["--from-stack", stack_path, "-o", stack_path + ".tmp.fits"])
            cli.apply_post_parse_setup(args)
            prepare_linear_for_phase4(args, stacked, hdr)
            try:
                pp.postprocess_stack(stacked, args, [], ProcessingStats())
            except _Stop:
                pass
    finally:
        pp.dynamic_background_extraction = real["dbe"]
        pp.wavelet_background_extraction = real["wavelet"]
        pp.apply_background_extraction = real["mesh"]
    return box or None


def block_mean(img: np.ndarray, f: int) -> np.ndarray:
    H, W = img.shape[0] // f * f, img.shape[1] // f * f
    return img[:H, :W].reshape(H // f, f, W // f, f, -1).mean(axis=(1, 3)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stacks", required=True, help="folder of linear stacks (+ _config.toml)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=4)
    args = ap.parse_args()
    import src.background as bg
    from src.merge import _pixel_noise
    os.makedirs(args.out, exist_ok=True)
    stacks = sorted(p for p in glob.glob(os.path.join(args.stacks, "*.fits"))
                    if not p.endswith(".tmp.fits") and "_checkpoint" not in p)
    for i, p in enumerate(stacks, 1):
        name = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(args.out, name + ".npz")
        if os.path.exists(dst):
            continue
        cap = capture(p)
        if cap is None or cap["method"] != "dbe":
            print(f"[{i}/{len(stacks)}] {name}: no DBE step ({cap and cap['method']})", flush=True)
            continue
        kw = dict(cap["kw"])
        kw.pop("verbose", None)
        with contextlib.redirect_stdout(io.StringIO()):
            clean = bg.dynamic_background_extraction(cap["rgb"].copy(), verbose=False, **kw)
        img = cap["rgb"]
        ok = np.ones(img.shape[:2], bool)
        sigma = np.array([_pixel_noise(img[..., c], ok) for c in range(3)], np.float32)
        f = args.factor
        np.savez(dst, image=block_mean(img, f), clean=block_mean(clean, f),
                 bg=block_mean(img - clean, f), sigma=sigma, target=target_of(name), factor=f)
        if is_holdout(name):
            with open(os.path.join(args.out, name + "_full.pkl"), "wb") as fh:
                pickle.dump(cap, fh, protocol=4)
        print(f"[{i}/{len(stacks)}] {name}: ok", flush=True)
        for t in glob.glob(p + ".tmp*"):
            os.remove(t)


if __name__ == "__main__":
    main()
