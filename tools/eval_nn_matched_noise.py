"""Star width of learned deconvolvers at equal sky noise, on the held-out seeing-pair inputs.

Each model's output is blended with ``--denoiser nn``'s until its sky noise (robust sigma of the
high-passed luminance on the faintest 40%) equals the shipped deconvolver's; the star width
(same stars, ``common_star_fwhm``, / the input's) is then read at that noise. A model that is
only sharper because it is noisier comes out no better here. Also times each forward pass.

Usage:
    python tools/eval_nn_matched_noise.py v3=src/data/nn_deconv.npz f32=C:/nn_work/model_deconv_f32.npz
"""
import glob
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from common_star_fwhm import common_star_fwhm
from train_nn_denoise import is_holdout

from src.debayer import luminance
from src.nn_denoise import nn_denoise

W = os.path.join(ROOT, "src", "data") + os.sep
PAIRS = os.environ.get("NN_PAIRS", "C:/nn_work/nn_deconv/pairs")
models = dict(a.split("=", 1) for a in sys.argv[1:])
lams = np.linspace(0.2, 1.0, 17)
res = {k: [] for k in models}; secs = {k: [] for k in models}
for d in sorted(glob.glob(os.path.join(PAIRS, "*"))):
    n = os.path.basename(d)
    if not is_holdout(n) or not os.path.exists(d + "/meta.json"): continue
    a = np.load(d + "/input.npy").astype(np.float32)
    h, w = a.shape[:2]; a = np.ascontiguousarray(a[h//2-768:h//2+768, w//2-1152:w//2+1152])
    sm = ndi.gaussian_filter(luminance(a), 3); sky = sm < np.quantile(sm, 0.4)
    def noise(img):
        l = luminance(np.ascontiguousarray(img, dtype=np.float32)); hp = (l - ndi.gaussian_filter(l, 2.0))[sky]
        return 1.4826 * np.median(np.abs(hp - np.median(hp)))
    def width(img):
        r = common_star_fwhm(np.moveaxis(img, -1, 0), np.moveaxis(a, -1, 0)); return r["ratio_a_over_b"] if r else np.nan
    dn = nn_denoise(a, model_path=W + "nn_denoise.npz", restore_sigma=4.0)
    v3 = nn_denoise(a, model_path=W + "nn_deconv.npz", restore_sigma=4.0)
    target = noise(v3)
    line = [n[:30]]
    for k, p in models.items():
        t = time.time(); o = nn_denoise(a, model_path=p, restore_sigma=4.0); secs[k].append(time.time() - t)
        ns = np.array([noise(l * o + (1 - l) * dn) for l in lams])
        lam = float(np.interp(target, ns, lams)) if ns[-1] > target else 1.0   # noise rises with lam
        wm = width((lam * o + (1 - lam) * dn).astype(np.float32))
        res[k].append((width(o), noise(o) / target, lam, wm))
        line.append(f"{k}: raw w {res[k][-1][0]:.3f} n {res[k][-1][1]:.2f}x  @v3-noise lam {lam:.2f} w {wm:.3f}")
    print("  ".join(line), flush=True)
print("\nmodel      raw width  raw noise/v3  width at v3 noise   s per 3.5 Mpx")
for k, r in res.items():
    r = np.array(r); print(f"{k:10s} {np.nanmedian(r[:,0]):.3f}      {np.median(r[:,1]):.2f}x         {np.nanmedian(r[:,3]):.3f}             {np.median(secs[k]):.1f}")
