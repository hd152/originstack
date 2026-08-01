"""Prototype: online (single-pass, incremental) sigma-clip combine, as an
alternative to the current batch sigma_clip_combine (src/stacking.py),
which requires the full (N,H,W,C) aligned stack (materialized as a disk
memmap today) before it can reject anything.

online_sigma_clip_combine processes frames one at a time and maintains a
running per-pixel (mean, M2) via Welford's algorithm, testing each new
frame's value against the running estimate *before* folding it in. This
never needs more than one frame + the running state (3x an (H,W,C) array)
resident at once -- no full-stack memmap at all.

Cold-start problem: the running estimate isn't reliable until it has seen
enough samples, so the first `burn_in` frames are combined via a normal
batch median+MAD pass (matching the existing method's spirit) to seed the
running state; only frames after that are tested/streamed.

This is exploratory (see tools/prototype_*.py precedent in this repo) --
not wired into the CLI. Run directly to compare against the real batch
sigma_clip_combine on a synthetic stack with known-location injected
outliers (simulated cosmic ray hits), so rejection quality is measurable
against ground truth rather than eyeballed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def online_sigma_clip_combine(
    frame_iter: Iterable[np.ndarray],
    shape: Tuple[int, int, int],
    sigma: float = 3.0,
    burn_in: int = 10,
) -> Tuple[np.ndarray, int, int]:
    """Single-pass sigma-clip combine. Returns (combined, n_rejected, n_total).

    frame_iter: yields (H, W, C) float32 arrays, one at a time.
    burn_in: number of initial frames combined via a robust (MAD-rejected)
        batch pass (materializes only this many frames at once, not the
        whole stack) to seed the running (mean, M2) state before streaming
        begins.
    """
    H, W, C = shape
    it: Iterator[np.ndarray] = iter(frame_iter)

    burn = []
    for _ in range(burn_in):
        try:
            burn.append(next(it))
        except StopIteration:
            break
    if not burn:
        raise ValueError("no frames supplied")

    burn_arr = np.stack(burn, axis=0).astype(np.float64)  # (K,H,W,C)
    n0 = burn_arr.shape[0]

    # Robust burn-in: one MAD-based rejection pass over the burn-in window
    # itself, so an outlier landing in the first `burn_in` frames doesn't
    # permanently poison the running estimate it seeds -- the whole point
    # of a "combined normally" burn-in is defeated if "normally" means a
    # plain, unprotected mean.
    med = np.median(burn_arr, axis=0)
    mad = np.median(np.abs(burn_arr - med), axis=0)
    robust_sigma = np.maximum(1.4826 * mad, 1e-6)
    burn_accept = np.abs(burn_arr - med) <= (sigma * robust_sigma)  # (K,H,W,C)

    n_acc = np.maximum(burn_accept.sum(axis=0).astype(np.float64), 1.0)
    running_mean = np.where(burn_accept, burn_arr, 0.0).sum(axis=0) / n_acc
    M2 = np.where(burn_accept, (burn_arr - running_mean) ** 2, 0.0).sum(axis=0)

    n_rejected = int(np.size(burn_accept) - burn_accept.sum())
    n_total = n0

    for frame in it:
        x = frame.astype(np.float64)
        n_total += 1

        var_est = M2 / np.maximum(n_acc, 1.0)
        std_est = np.sqrt(np.maximum(var_est, 1e-12))
        accept = np.abs(x - running_mean) <= (sigma * std_est)

        n_acc_new = n_acc + accept
        delta = x - running_mean
        inv_n = 1.0 / np.maximum(n_acc_new, 1.0)
        new_mean = np.where(accept, running_mean + delta * inv_n, running_mean)
        delta2 = x - new_mean
        new_M2 = np.where(accept, M2 + delta * delta2, M2)

        running_mean = new_mean
        M2 = new_M2
        n_acc = n_acc_new
        n_rejected += int(np.size(accept) - np.sum(accept))

    return running_mean.astype(np.float32), n_rejected, n_total


# ---------------------------------------------------------------------------
# Synthetic ground-truth comparison
# ---------------------------------------------------------------------------

def _make_ground_truth(H: int, W: int, C: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    gradient = 100.0 + 30.0 * (xx / W) + 15.0 * (yy / H)
    img = np.repeat(gradient[:, :, None], C, axis=2)
    rng = np.random.default_rng(0)
    for _ in range(8):
        cy, cx = rng.integers(20, H - 20), rng.integers(20, W - 20)
        yy2, xx2 = np.mgrid[0:H, 0:W]
        r2 = (yy2 - cy) ** 2 + (xx2 - cx) ** 2
        star = 3000.0 * np.exp(-r2 / (2 * 3.0 ** 2))
        img += star[:, :, None]
    return img.astype(np.float64)


def run_comparison(H=256, W=256, C=3, n_frames=30, noise_sigma=5.0,
                   n_bad_frames=3, n_hits_per_frame=6, hit_amplitude=6000.0,
                   seed=1, bad_frame_idx=None):
    from src.stacking import sigma_clip_combine

    rng = np.random.default_rng(seed)
    truth = _make_ground_truth(H, W, C)

    frames = np.empty((n_frames, H, W, C), dtype=np.float32)
    outlier_locations = []  # (frame_idx, y, x)
    if bad_frame_idx is None:
        bad_frame_idx = rng.choice(n_frames, size=n_bad_frames, replace=False)
    else:
        bad_frame_idx = np.asarray(bad_frame_idx)
    for i in range(n_frames):
        frame = truth + rng.standard_normal((H, W, C)) * noise_sigma
        if i in bad_frame_idx:
            for _ in range(n_hits_per_frame):
                y = int(rng.integers(10, H - 10))
                x = int(rng.integers(10, W - 10))
                frame[y, x, :] += hit_amplitude
                outlier_locations.append((i, y, x))
        frames[i] = frame.astype(np.float32)

    # --- Plain mean (no rejection) baseline ---
    t0 = time.perf_counter()
    mean_combined = frames.mean(axis=0)
    t_mean = time.perf_counter() - t0

    # --- Existing batch sigma-clip (the real production function) ---
    t0 = time.perf_counter()
    batch_combined = sigma_clip_combine(frames.copy(), sigma=3.0, max_iters=3, use_mad=True)
    t_batch = time.perf_counter() - t0

    # --- New online sigma-clip ---
    t0 = time.perf_counter()
    online_combined, n_rej, n_tot = online_sigma_clip_combine(
        (frames[i] for i in range(n_frames)), (H, W, C),
        sigma=3.0, burn_in=10)
    t_online = time.perf_counter() - t0

    def rmse(img):
        return float(np.sqrt(np.mean((img.astype(np.float64) - truth) ** 2)))

    def outlier_leak(img):
        """Mean absolute residual (vs. truth) at the injected-outlier pixels
        specifically -- how much of the injected spike leaked through."""
        if not outlier_locations:
            return 0.0
        vals = [abs(float(img[y, x, 0]) - float(truth[y, x, 0]))
                for (_, y, x) in outlier_locations]
        return float(np.mean(vals))

    print(f"Frames: {n_frames}, bad frames: {sorted(bad_frame_idx.tolist())}, "
          f"{len(outlier_locations)} injected hits total\n")
    print(f"{'method':<16} {'RMSE':>10} {'outlier leak':>14} {'time (s)':>10}")
    print(f"{'plain mean':<16} {rmse(mean_combined):>10.2f} "
          f"{outlier_leak(mean_combined):>14.2f} {t_mean:>10.4f}")
    print(f"{'batch sigma-clip':<16} {rmse(batch_combined):>10.2f} "
          f"{outlier_leak(batch_combined):>14.2f} {t_batch:>10.4f}")
    print(f"{'online sigma-clip':<16} {rmse(online_combined):>10.2f} "
          f"{outlier_leak(online_combined):>14.2f} {t_online:>10.4f}")
    print(f"\nonline: rejected {n_rej}/{n_tot * H * W * C} pixel-samples "
          f"({100.0 * n_rej / (n_tot * H * W * C):.4f}%) after burn-in")

    return {
        'mean': (rmse(mean_combined), outlier_leak(mean_combined)),
        'batch': (rmse(batch_combined), outlier_leak(batch_combined)),
        'online': (rmse(online_combined), outlier_leak(online_combined)),
    }


if __name__ == '__main__':
    run_comparison()
