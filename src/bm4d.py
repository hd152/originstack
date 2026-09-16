"""BM4D-style collaborative-filter denoising across the raw (aligned) frame
stack, before combine (--bm4d-prefilter).

Unlike ``bm3d_denoise`` (src/denoising.py), which spatially block-matches
*within a single already-combined image* (searching nearby positions for
similar-looking patches -- a heuristic), this exploits genuine *temporal*
redundancy: N independently-noisy realizations of the same signal, already
spatially co-registered by Phase 2/3's own alignment. Unlike generic
(unregistered) BM4D, which needs a real cross-frame motion search, no search
is needed here at all -- alignment already solved correspondence, so the
matching block at spatial position (y,x) across frames simply *is* the block
at (y,x) in every frame. Group formation is "stack the block at (y,x) from
every frame", not a similarity search: a stronger, exact-correspondence
signal than BM3D's spatial self-similarity heuristic, at the cost of no
longer helping with structure that only repeats spatially (not temporally)
within one frame -- this complements bm3d_denoise (which still runs
post-combine), it doesn't replace it.

Runs pre-combine on the aligned stack, in the same pipeline slot as
--cosmic-ray-rejection/--trail-reject: a per-frame cleanup pass before the
existing stack_method combine runs on the result.

The collaborative-filter core (3-D DCT, Step-1 hard threshold, Step-2 Wiener
using the Step-1 result as pilot, weighted aggregation) is the same algorithm
``_bm3d_step12_numpy`` already implements and this codebase already validated
-- only how the group is *formed* differs (fixed cross-frame stack here vs.
a DCT-distance spatial search there).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils import safe_print

try:
    from scipy.fft import dctn, idctn
except ImportError:
    from scipy.fftpack import dctn, idctn


def _estimate_stack_sigma(data: np.ndarray) -> float:
    """Per-pixel noise sigma from a single consecutive-frame difference.

    Aligned frames are independent noisy realizations of the same signal, so
    ``frame[0] - frame[1]`` has (near) zero signal component -- its MAD
    directly gives ``sigma * sqrt(2)``, a cleaner estimate than any
    single-frame spatial heuristic (no assumption about spatial noise
    correlation needed, unlike ``_estimate_sky_sigma``'s adjacent-pixel-diff
    approach on a single image).
    """
    if data.shape[0] < 2:
        return 0.0
    diff = data[0].astype(np.float64) - data[1].astype(np.float64)
    mad = float(np.median(np.abs(diff - np.median(diff))))
    return max(mad * 1.4826 / np.sqrt(2.0), 0.0)


def bm4d_denoise_stack(mem_aligned: np.ndarray, sigma_psd: float = 0.0,
                       block_size: int = 8, stride: Optional[int] = None,
                       verbose: bool = False) -> np.ndarray:
    """Denoise an aligned ``(N, H, W)`` luminance stack via cross-frame BM4D
    collaborative filtering. Returns a denoised ``(N, H, W)`` float32 array
    (each frame denoised individually, written back to its own slot).

    Every frame participates in every block's group -- unlike BM3D's spatial
    search, there's no ranking/truncation here (no search happened, so
    there's nothing to cap): group cost is linear in N with no per-pair
    search, so capping group size the way an earlier version of this
    function did just silently left frames beyond the cap completely
    undenoised (a real bug, caught by a stray RuntimeWarning during
    testing) for no compute-cost benefit worth the risk.
    """
    data = np.asarray(mem_aligned, dtype=np.float64)
    n, H, W = data.shape
    if n < 3:
        return data.astype(np.float32)

    if sigma_psd <= 0.0:
        sigma_psd = _estimate_stack_sigma(data)
    if sigma_psd < 1e-9:
        return data.astype(np.float32)

    bs = block_size
    if stride is None:
        stride = bs if max(H, W) > 1500 else max(bs // 2, 2)

    ref_ys = np.arange(0, H - bs + 1, stride)
    ref_xs = np.arange(0, W - bs + 1, stride)
    if len(ref_ys) == 0 or len(ref_xs) == 0:
        return data.astype(np.float32)

    frame_idx = np.arange(n)
    n_g = n

    if verbose:
        safe_print(f"    BM4D prefilter: {n} frames ({n_g}/group), "
                  f"{len(ref_ys) * len(ref_xs)} blocks, sigma={sigma_psd:.3f}")

    ht_threshold = sigma_psd * np.sqrt(2.0 * np.log(float(bs * bs)))

    # --- Step 1: hard-thresholding in joint 3-D (frame x space) DCT domain ---
    acc1 = np.zeros((n, H, W), dtype=np.float64)
    wgt1 = np.zeros((n, H, W), dtype=np.float64)

    for yr in ref_ys:
        for xr in ref_xs:
            group = data[frame_idx, yr:yr + bs, xr:xr + bs]  # (n_g, bs, bs)
            spec3 = dctn(group, axes=(0, 1, 2), norm='ortho')
            ht = np.where(np.abs(spec3) >= ht_threshold, spec3, 0.0)
            n_nz = max(1, int(np.count_nonzero(ht)))
            w1 = 1.0 / n_nz
            denoised1 = idctn(ht, axes=(0, 1, 2), norm='ortho')

            for k, fi in enumerate(frame_idx):
                acc1[fi, yr:yr + bs, xr:xr + bs] += w1 * denoised1[k]
                wgt1[fi, yr:yr + bs, xr:xr + bs] += w1

    pilot = np.where(wgt1 > 0, acc1 / wgt1, data)

    # --- Step 2: Wiener filter using the Step-1 result as pilot ---
    # Same group membership as Step 1 (fixed by alignment, not re-searched --
    # unlike BM3D's spatial search there is nothing that could change between
    # steps here, so re-forming it identically is both simpler and correct).
    acc2 = np.zeros((n, H, W), dtype=np.float64)
    wgt2 = np.zeros((n, H, W), dtype=np.float64)

    for yr in ref_ys:
        for xr in ref_xs:
            noisy_group = data[frame_idx, yr:yr + bs, xr:xr + bs]
            pilot_group = pilot[frame_idx, yr:yr + bs, xr:xr + bs]

            spec_noisy = dctn(noisy_group, axes=(0, 1, 2), norm='ortho')
            spec_pilot = dctn(pilot_group, axes=(0, 1, 2), norm='ortho')

            pilot_sq = spec_pilot ** 2
            wiener = pilot_sq / (pilot_sq + sigma_psd ** 2 + 1e-30)
            spec_filt = wiener * spec_noisy

            wiener_w = float(np.sum(wiener ** 2)) / max(n_g, 1)
            w2 = 1.0 / max(wiener_w, 1e-12)
            denoised2 = idctn(spec_filt, axes=(0, 1, 2), norm='ortho')

            for k, fi in enumerate(frame_idx):
                acc2[fi, yr:yr + bs, xr:xr + bs] += w2 * denoised2[k]
                wgt2[fi, yr:yr + bs, xr:xr + bs] += w2

    result = np.where(wgt2 > 0, acc2 / wgt2, pilot)
    return result.astype(np.float32)
