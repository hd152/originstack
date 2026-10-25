"""Score learned deconvolvers against Pan-STARRS truth on held-out crops (tools/fetch_ps1_truth.py
crops of train_nn_denoise.HOLDOUT targets), built exactly as gen_survey_pairs.py builds targets.

Per crop and method (input, ``nn_denoise``, each ``--model``), on the pixels the truth covers:
  error      RMS of (output - truth) at fine scales (each minus its 16 px Gaussian), in the
             input's noise sigma; split into sky and signal pixels
  width      median FWHM of the crop's stars in the output / in the truth (1 = as sharp as
             the target; the input is ~1.5-2.5)
  invented   peaks above 3 sigma (output's own sky noise) per Mpx with no truth peak above
             1.5 truth sigma within 2 px -- detail the sky does not have
  noise      output's high-passed sky noise / the input's

Usage:
    python tools/eval_ps1_truth.py --crops C:/nn_work/survey/ps1 --meta C:/nn_work/survey/bank_ps1_meta.json \\
        --model v3=src/data/nn_deconv.npz --model ps1=C:/nn_work/survey/model_ps1_v1.npz
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

from gen_survey_pairs import LUMA, crop_scale, gauss_fwhm, make_target, star_fluxes, target_name  # noqa: E402
from train_nn_denoise import is_holdout  # noqa: E402


def mad(x):
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def peaks(img, thresh):
    mx = ndi.maximum_filter(img, size=5)
    return np.argwhere((img == mx) & (img > thresh))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--crops", required=True)
    ap.add_argument("--meta", required=True, help="gen_survey_pairs.py's _meta.json (colour matrix, target)")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=weights.npz")
    ap.add_argument("--json")
    args = ap.parse_args()
    from src.nn_denoise import nn_denoise
    meta = json.load(open(args.meta))
    M = np.array(meta["M"])
    methods = {"input": None, "nn_denoise": os.path.join(ROOT, "src", "data", "nn_denoise.npz"),
               **dict(m.split("=", 1) for m in args.model)}
    res = {}
    for p in sorted(glob.glob(os.path.join(args.crops, "*.npz"))):
        k = os.path.basename(p)[:-4]
        if not is_holdout(target_name(k)):
            continue
        f = np.load(p)
        A, P = f["origin"].astype(np.float64), f["ps1"].astype(np.float64)
        fo, fp, pos = star_fluxes(A, P)
        if len(fo) < 3:
            continue
        s = crop_scale(fo, fp, M)
        T, mask, _ = make_target(A, P, M, s, meta["target_fwhm"], meta["keep_sigma"], pos)
        lt = T.astype(np.float64) @ LUMA
        la = A @ LUMA
        hp = lambda x: x - ndi.gaussian_filter(x, 16.0)                 # noqa: E731
        sm = ndi.gaussian_filter(la, 3)
        sky = mask & (sm < np.quantile(sm[mask], 0.4))
        sig_in = mad(hp(la)[sky])
        tsig = mad(hp(lt)[sky])
        tpk = peaks(hp(lt), 1.5 * tsig)
        tmap = np.zeros(la.shape, bool)
        tmap[tpk[:, 0], tpk[:, 1]] = True
        tmap = ndi.binary_dilation(tmap, iterations=2)
        fw_t = gauss_fwhm(lt, pos)
        res[k] = {}
        for name, mp in methods.items():
            out = A.astype(np.float32) if mp is None else nn_denoise(A.astype(np.float32), model_path=mp, restore_sigma=4.0)
            lo = out.astype(np.float64) @ LUMA
            d = (hp(lo) - hp(lt)) / sig_in
            osig = mad(hp(lo)[sky])
            pk = peaks(hp(lo), 3 * osig)
            inside = mask[pk[:, 0], pk[:, 1]]
            fake = (~tmap[pk[:, 0], pk[:, 1]] & inside).sum()
            res[k][name] = {"error": float(np.sqrt(np.mean(d[mask] ** 2))),
                            "error_sky": float(np.sqrt(np.mean(d[sky] ** 2))),
                            "error_signal": float(np.sqrt(np.mean(d[mask & ~sky] ** 2))),
                            "width": gauss_fwhm(lo, pos) / fw_t, "invented_per_mpx": float(fake / (mask.sum() / 1e6)),
                            "noise": osig / sig_in}
        print(k + "  " + "  ".join(f"{n}: err {r['error']:.2f} (sky {r['error_sky']:.2f}) width {r['width']:.2f} "
                                    f"inv {r['invented_per_mpx']:.0f} noise {r['noise']:.2f}" for n, r in res[k].items()), flush=True)
    names = list(methods)
    print("\nmedians over", len(res), "held-out crops:")
    for n in names:
        v = {m: np.median([res[k][n][m] for k in res]) for m in ("error", "error_sky", "error_signal", "width", "invented_per_mpx", "noise")}
        print(f"  {n:12s} " + "  ".join(f"{m} {x:.3f}" for m, x in v.items()))
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
