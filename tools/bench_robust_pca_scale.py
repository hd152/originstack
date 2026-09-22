"""Benchmark src.robust_pca.robust_pca_decompose's wall-clock cost vs. frame count N,
to check whether Config.ROBUST_PCA_AUTO_MAX_FRAMES (currently 10, based on a single
N=20/P=18M anchor documented in CLAUDE.md: 1264s) can be safely widened for --auto.

Running the full N=20, P=18,000,000 (2000x3000x3) case directly takes ~21 minutes per
the existing measurement -- too slow to redo here per candidate N. Instead this measures
the real decomposition (astro_native's Gram-matrix-trick SVD, same code path as
production) at reduced P across a range of N, confirms the O(N^2 x P) scaling the
kernel's docstring claims actually holds on this machine, then extrapolates to the real
P=18M shape -- cross-checked against the documented N=20 anchor.

Usage: python tools/bench_robust_pca_scale.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.robust_pca import robust_pca_decompose

try:
    import astro_native  # noqa: F401
    print(f"native: available ({astro_native.__file__})")
except Exception as e:
    print(f"native: NOT available ({e}) -- this benchmark would be measuring the "
          f"numpy fallback, not the production path. Build astro_native first.")
    raise SystemExit(1)

rng = np.random.default_rng(7)

REAL_P = 2000 * 3000 * 3  # matches the documented N=20 anchor's frame shape
ANCHOR_N, ANCHOR_SECONDS = 20, 1264.0  # from CLAUDE.md / src/robust_pca.py docstring


def make_calib_stack(n: int, p: int) -> np.ndarray:
    """Synthetic (n, p) calibration stack: a shared rank-1 pattern (vignetting/dark
    current) + sparse spikes (dust motes, hot pixels) + read noise -- same qualitative
    structure robust_pca_decompose is built to separate, so convergence behavior (and
    therefore iteration count / wall time) is representative of real calibration data,
    not a degenerate all-zero or pure-noise case that would converge trivially fast.
    """
    shared = rng.normal(1000, 50, p)
    weights = rng.uniform(0.8, 1.2, n)
    D = weights[:, None] * shared[None, :]
    n_spikes = int(n * p * 0.01)
    idx_n = rng.integers(0, n, n_spikes)
    idx_p = rng.integers(0, p, n_spikes)
    D[idx_n, idx_p] += rng.choice([-1, 1], n_spikes) * rng.uniform(200, 2000, n_spikes)
    D += rng.normal(0, 15, (n, p))
    return D.astype(np.float64)


def timed_decompose(n: int, p: int, reps: int = 1) -> float:
    best = None
    for _ in range(reps):
        D = make_calib_stack(n, p)
        t0 = time.perf_counter()
        robust_pca_decompose(D)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


print("\n--- Phase 1: N-scaling at fixed small P (confirm ~N^2 growth) ---")
P_SMALL = 258 * 258 * 3  # ~200k px, keeps each run to a few seconds
n_values = [10, 15, 20, 25, 30, 40]
n_results = {}
for n in n_values:
    dt = timed_decompose(n, P_SMALL)
    n_results[n] = dt
    print(f"  N={n:3d}  P={P_SMALL:>9,d}  {dt:7.3f}s")

print("\n--- Phase 2: P-scaling at fixed N=10 (confirm linear growth) ---")
p_values = [50_000, 100_000, 200_000, 400_000]
p_results = {}
for p in p_values:
    dt = timed_decompose(10, p)
    p_results[p] = dt
    print(f"  N=10   P={p:>9,d}  {dt:7.3f}s")

# Fit growth exponents from the measured points (log-log slope) instead of assuming
# the docstring's O(N^2 x P) claim holds exactly on this machine/data.
import math

log_n = [math.log(n) for n in n_values]
log_t_n = [math.log(max(t, 1e-6)) for t in n_results.values()]
n_slope = np.polyfit(log_n, log_t_n, 1)[0]

log_p = [math.log(p) for p in p_values]
log_t_p = [math.log(max(t, 1e-6)) for t in p_results.values()]
p_slope = np.polyfit(log_p, log_t_p, 1)[0]

print(f"\nMeasured scaling exponents: time ~ N^{n_slope:.2f} x P^{p_slope:.2f} "
      f"(docstring claims O(N^2 x P), i.e. exponents 2.0 and 1.0)")

# Extrapolate each candidate N to the real P using the largest small-P measurement as
# the base point (closest to the regime we care about) and the measured P exponent.
base_n, base_p = 40, P_SMALL
base_t = n_results[base_n]
p_scale = (REAL_P / base_p) ** p_slope

print(f"\n--- Phase 3: extrapolated full-resolution (P={REAL_P:,d}) time by N ---")
print(f"  {'N':>4s}  {'predicted':>10s}  {'vs anchor':>10s}")
candidate_ns = [10, 15, 20, 25, 30, 40]
for n in candidate_ns:
    n_scale = (n / base_n) ** n_slope
    predicted = base_t * n_scale * p_scale
    tag = ""
    if n == ANCHOR_N:
        tag = f"  (documented anchor: {ANCHOR_SECONDS:.0f}s / {ANCHOR_SECONDS/60:.1f}min)"
    print(f"  {n:4d}  {predicted:8.1f}s  {predicted/60:8.2f}min{tag}")

anchor_scale = (ANCHOR_N / base_n) ** n_slope * p_scale
anchor_predicted = base_t * anchor_scale
print(f"\nCross-check: predicted N=20 full-res time {anchor_predicted:.0f}s vs. "
      f"documented anchor {ANCHOR_SECONDS:.0f}s "
      f"(ratio {anchor_predicted / ANCHOR_SECONDS:.2f}x)")
print("A ratio far from 1.0 means this machine/build differs enough from the original "
      "measurement that the extrapolation above should not be trusted for a threshold "
      "change -- rerun the real N=20 P=18M case directly instead.")
