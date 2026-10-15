"""Noise2Noise training data for the learned denoiser (``--denoiser nn``) from real sessions.

For each session, runs the normal stacker and, while the aligned frames still exist (the
``--noise-validate`` hook in ``stacking.run_stacking_phase``), writes four sigma-clipped
quarter-stacks of them: frames ``i % 4 == q``. Any two quarters see the same sky with
independent noise, which is all Noise2Noise needs -- no clean ground truth. The run then
exits; Phase 4 is not needed.

Output per session, under ``--out``:
    <name>/q0.npy .. q3.npy   (H, W, 3) float32 linear, aligned to each other
    <name>/meta.json          frame count, source folder

``--mode seeing`` (training data for the learned deconvolver) instead measures every
aligned frame's FWHM on the same bright isolated stars and writes the stack of the
sharpest ``--sharp-frac`` of frames (target) and of the rest (input): the same sky, a
sharper PSF, disjoint frames so independent noise.
    <name>/input.npy, <name>/target.npy, <name>/fwhm.npy (per frame), meta.json

Usage:
    python tools/gen_n2n_pairs.py --out G:/output/nn_denoise/pairs SESSION_DIR [SESSION_DIR ...]
    python tools/gen_n2n_pairs.py --out ... --list sessions.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_BAND_ROWS = 256


def clipped_mean(frames: np.ndarray, idx: np.ndarray, sigma: float = 3.0, iters: int = 3) -> np.ndarray:
    """MAD sigma-clipped mean of ``frames[idx]`` (N, H, W, C) -- the stacker's own combine,
    run per row band so a disk-backed stack is read once per band."""
    from src.stacking import sigma_clip_combine
    _, h, w, c = frames.shape
    out = np.empty((h, w, c), np.float32)
    for y0 in range(0, h, _BAND_ROWS):
        band = np.ascontiguousarray(frames[idx, y0:y0 + _BAND_ROWS], dtype=np.float32)
        out[y0:y0 + band.shape[1]] = sigma_clip_combine(band, sigma=sigma, max_iters=iters)
    return out


_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def frame_fwhms(frames: np.ndarray, n_stars: int = 60) -> np.ndarray:
    """Median Gaussian FWHM of each aligned frame, fitted on the same bright, unsaturated,
    isolated stars (picked on a quick reference stack) -- one star list for every frame,
    so the numbers compare across frames."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from common_star_fwhm import _HALF, _gauss_fit

    from src.star_detect import detect_stars_matched_filter as detect
    n, h, w, _ = frames.shape
    ref = clipped_mean(frames, np.arange(0, n, max(1, n // 16))) @ _LUMA
    st = detect(ref - np.median(ref))
    st = st[np.argsort(-st["flux"])]
    sat = np.percentile(ref, 99.99) * 0.6
    xs, ys = np.asarray(st["xcentroid"]), np.asarray(st["ycentroid"])
    keep = []
    m = _HALF + 4
    for i in range(len(st)):
        x, y = int(round(xs[i])), int(round(ys[i]))
        if not (m < x < w - m and m < y < h - m) or ref[y, x] >= sat:
            continue
        if any((xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2 < 20 ** 2 for j in range(len(st)) if j != i):
            continue
        keep.append((x, y))
        if len(keep) == n_stars:
            break
    out = np.full(n, np.nan)
    for f in range(n):
        vals = []
        for x, y in keep:
            cut = np.asarray(frames[f, y - m:y + m + 1, x - m:x + m + 1], np.float32) @ _LUMA
            r = _gauss_fit(cut, m, m)
            if r is not None:
                vals.append(r[0])
        if len(vals) >= 5:
            out[f] = np.median(vals)
    return out


def _worker(session: str, out_dir: str, mode: str = "quarters", sharp_frac: float = 0.3) -> None:
    """Run the stacker in this process with the capture hook installed."""
    sys.path.insert(0, ROOT)
    import src.noise_validation as nv

    def capture(frames, frame_noise=None):
        n = frames.shape[0]
        os.makedirs(out_dir, exist_ok=True)
        t = time.time()
        meta = {"session": session, "frames": int(n), "shape": list(frames.shape[1:]), "mode": mode}
        if mode == "seeing":
            fw = frame_fwhms(frames)
            ok = np.flatnonzero(np.isfinite(fw))
            order = ok[np.argsort(fw[ok])]
            k = max(3, int(round(sharp_frac * len(order))))
            sharp, rest = np.sort(order[:k]), np.sort(order[k:])
            np.save(os.path.join(out_dir, "fwhm.npy"), fw)
            np.save(os.path.join(out_dir, "target.npy"), clipped_mean(frames, sharp))
            np.save(os.path.join(out_dir, "input.npy"), clipped_mean(frames, rest))
            meta.update(sharp_frac=sharp_frac, n_target=int(len(sharp)), n_input=int(len(rest)),
                        fwhm_target=float(np.median(fw[sharp])), fwhm_input=float(np.median(fw[rest])))
        else:
            for q in range(4):
                np.save(os.path.join(out_dir, f"q{q}.npy"), clipped_mean(frames, np.arange(q, n, 4)))
        with open(os.path.join(out_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh)
        print(f"[n2n] {mode}: {n} frames in {time.time() - t:.0f} s", flush=True)
        raise SystemExit(0)          # BaseException: not swallowed by the stacker's fail-soft except

    nv.validate_noise = capture
    tmp = os.path.join(out_dir, "_stack.fits")
    sys.argv = ["originstack", "-d", session, "-o", tmp, "--noise-validate", "--offline"]
    from src.cli import main
    main()


def _name(session: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(os.path.normpath(session)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--list", help="text file, one session folder per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["quarters", "seeing"], default="quarters")
    ap.add_argument("--sharp-frac", type=float, default=0.3,
                    help="--mode seeing: fraction of frames, sharpest first, in the target stack")
    ap.add_argument("--worker", nargs=2, metavar=("SESSION", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        _worker(*args.worker, mode=args.mode, sharp_frac=args.sharp_frac)
        return
    sessions = list(args.sessions)
    if args.list:
        with open(args.list, encoding="utf-8") as fh:
            sessions += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    for i, s in enumerate(sessions, 1):
        dst = os.path.join(args.out, _name(s))
        if os.path.exists(os.path.join(dst, "meta.json")):
            print(f"[{i}/{len(sessions)}] {_name(s)}: done already")
            continue
        t = time.time()
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--out", args.out, "--mode", args.mode,
                            "--sharp-frac", str(args.sharp_frac), "--worker", s, dst],
                           capture_output=True, text=True, errors="replace")
        ok = os.path.exists(os.path.join(dst, "meta.json"))
        print(f"[{i}/{len(sessions)}] {_name(s)}: {'ok' if ok else 'FAILED'} ({time.time() - t:.0f} s)", flush=True)
        if not ok:
            print((r.stdout + r.stderr)[-1500:], flush=True)
        for f in os.listdir(dst) if os.path.isdir(dst) else ():
            if f.startswith("_stack"):
                p = os.path.join(dst, f)
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)


if __name__ == "__main__":
    main()
