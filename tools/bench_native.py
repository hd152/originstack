"""Benchmark astro_native kernels. Run once against the installed wheel to get a
baseline, rebuild, run again, compare. Fixed seeds; realistic-ish shapes."""
import json
import sys
import time

import astro_native as nat
import numpy as np

rng = np.random.default_rng(7)


def make_stack(n, h, w, c=3, outliers=True):
    d = rng.normal(1000, 30, (n, h, w, c)).astype(np.float32)
    if outliers:
        idx = rng.integers(0, [n, h, w, c], size=(n * h * w // 25, 4))
        d[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] += rng.choice(
            [-1, 1], len(idx)) * rng.uniform(200, 2000, len(idx)).astype(np.float32)
    return np.ascontiguousarray(d)


def bench(name, fn, warm=1, reps=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    best = min(ts)
    print(f"{name:34s} {best*1000:9.1f} ms")
    return best


results = {}

# --- combines: 200-frame stack (realistic session), 384x512 spatial ---
d200 = make_stack(200, 384, 512)
w200 = rng.uniform(0.5, 1.5, 200).astype(np.float32)
results["sigma_clip_mad"] = bench(
    "sigma_clip (MAD, default)", lambda: nat.sigma_clip_combine(d200, 3.0, 3, None, False, True))
results["sigma_clip_std"] = bench(
    "sigma_clip (std)", lambda: nat.sigma_clip_combine(d200, 3.0, 3, None, False, False))
results["sigma_clip_wt_winsor"] = bench(
    "sigma_clip (wt+winsor)", lambda: nat.sigma_clip_combine(d200, 3.0, 3, w200, True, True))
results["median"] = bench("median", lambda: nat.median_combine(d200))
results["percentile"] = bench(
    "percentile", lambda: nat.percentile_clip_combine(d200, 20.0, 80.0, None))

# --- ESD: small-N use case ---
d60 = make_stack(60, 384, 512)
lut = np.full((61, 15), np.inf)
from scipy import stats as st

for ne in range(3, 61):
    for i in range(min(15, ne - 2)):
        ncur = ne - i
        p = min(max(0.05 / (2 * ncur), 1e-10), 0.4999)
        t = st.t.ppf(1 - p, df=max(ncur - 2, 1))
        den = np.sqrt((ncur - 2 + t * t) * ncur)
        lut[ne, i] = (ncur - 1) * t / den
results["esd"] = bench("esd (60fr, mo=15)", lambda: nat.esd_combine(d60, 15, lut, None))

# --- fused patch-weighted ---
d100 = make_stack(100, 256, 256)
qm = np.ascontiguousarray(rng.uniform(0.2, 1.0, (100, 256, 256)).astype(np.float32))
gw = rng.uniform(0.5, 1.5, 100).astype(np.float32)
results["patch_weighted"] = bench(
    "patch_weighted_sigma", lambda: nat.patch_weighted_sigma_combine(d100, qm, gw, 3.0, 3, True))

# --- warp: full-size frame, shift-only and small rotation ---
img = np.ascontiguousarray(rng.normal(500, 50, (2048, 3056, 3)).astype(np.float32))
I = [1.0, 0.0, 0.0, 1.0]
th = np.deg2rad(0.3)
R = [np.cos(th), -np.sin(th), np.sin(th), np.cos(th)]
results["warp_shift"] = bench(
    "warp (shift-only)", lambda: nat.warp_affine_lanczos3(img, I, [2.4, -1.7], 2048, 3056, 0.0))
results["warp_affine"] = bench(
    "warp (rotation)", lambda: nat.warp_affine_lanczos3(img, R, [3.2, -1.7], 2048, 3056, 0.0))

# --- anisotropic diffusion ---
small = np.ascontiguousarray(np.clip(rng.normal(300, 40, (1024, 1528, 3)), 0, None).astype(np.float32))
results["aniso"] = bench(
    "aniso (15 iters)", lambda: nat.anisotropic_diffusion(small, 15, 30.0, 0.1, 2))

# --- Malvar debayer: full-size raw mosaic ---
raw_mosaic = np.ascontiguousarray(rng.uniform(0, 65535, (2822, 4144)).astype(np.float32))
results["debayer_malvar"] = bench(
    "debayer_malvar (RGGB)", lambda: nat.debayer_malvar(raw_mosaic, "RGGB"))

# --- bilateral filter: full-size RGB stack image ---
rgb_full = np.ascontiguousarray(rng.uniform(0, 1000, (2000, 3000, 3)).astype(np.float32))
results["bilateral_filter"] = bench(
    "bilateral_filter (sigma_space=3)",
    lambda: nat.bilateral_filter(rgb_full, 50.0, 3.0, 9))

# --- batch aperture photometry: time-series shape (300 stars, one frame) ---
if hasattr(nat, "aperture_photometry_batch"):
    apb_img = np.ascontiguousarray(
        rng.uniform(20, 2000, (1500, 2000, 3)).astype(np.float32))
    apb_xs = rng.uniform(30, 1970, 300)
    apb_ys = rng.uniform(30, 1470, 300)
    results["aperture_photometry_batch"] = bench(
        "aperture_phot_batch (300 stars)",
        lambda: nat.aperture_photometry_batch(
            apb_img, apb_xs, apb_ys, 6.0, 9.0, 15.0, 4))

out = sys.argv[1] if len(sys.argv) > 1 else "bench_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print(f"\nsaved -> {out}")
