"""Tests for src/background.py's remove_sky_residual exclusion_mask param.

Without a caller-supplied exclusion mask, remove_sky_residual only protects
itself against a bright, compact peak (5-sigma threshold, fixed-radius
circle) -- a large, diffuse object well under that threshold (a galaxy's
faint outer disk, in particular) got zero protection even when an earlier
DBE pass correctly excluded it via --galaxy-mode's fitted ellipse. That gap
was a real, reported cause of galaxy detail/extent loss surviving stacking.
"""
from __future__ import annotations

import numpy as np

from src.background import remove_sky_residual


def _flat_sky_with_faint_disk(H=400, W=400, cy=200.0, cx=200.0, radius=120.0,
                              amp=4.0, sky=1000.0, seed=0):
    """A large, low-contrast disk (well under remove_sky_residual's own
    5-sigma auto-detect threshold) on a flat, noisy sky -- the "faint outer
    galaxy disk" failure case."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disk = amp * (r < radius)
    plane = sky + disk + rng.normal(0, 3.0, (H, W))
    return plane.astype(np.float32)


def _to_rgb(plane: np.ndarray) -> np.ndarray:
    return np.stack([plane, plane, plane], axis=2)


def test_unmasked_run_erodes_faint_disk_signal():
    """Baseline: confirm the faint disk is genuinely at risk without a mask
    (below the function's own 5-sigma auto-detect, so its mesh fit samples
    disk pixels as if they were sky and subtracts most of the excess)."""
    H, W = 400, 400
    cy, cx, radius, amp = 200.0, 200.0, 120.0, 4.0
    rgb = _to_rgb(_flat_sky_with_faint_disk(H, W, cy, cx, radius, amp))
    result = remove_sky_residual(rgb, mesh_size=64, filter_size=1)

    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disk_core = r < (radius * 0.5)
    # The disk's excess-over-sky should have been substantially eaten.
    residual_excess = float(np.median(result[:, :, 0][disk_core]))
    assert residual_excess < amp * 0.5


def test_exclusion_mask_protects_faint_disk_signal():
    """With an exclusion mask covering the disk (what --galaxy-mode's fitted
    ellipse now feeds in), the disk's excess signal must survive -- this is
    the actual fix: remove_sky_residual now accepts and honors a
    caller-supplied mask instead of relying solely on its own stricter
    built-in detection."""
    H, W = 400, 400
    cy, cx, radius, amp = 200.0, 200.0, 120.0, 4.0
    rgb = _to_rgb(_flat_sky_with_faint_disk(H, W, cy, cx, radius, amp))

    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = (r < radius * 1.1).astype(np.float32)

    result = remove_sky_residual(rgb, mesh_size=64, filter_size=1, exclusion_mask=mask)

    disk_core = r < (radius * 0.5)
    residual_excess = float(np.median(result[:, :, 0][disk_core]))
    # Sky well outside the mask (past its feather radius -- protection is
    # deliberately smoothed over ~mesh_size*0.5 = 32px so the subtraction
    # doesn't leave a hard seam at the mask edge) should still be flattened.
    sky_region = r > (radius * 1.1 + 130.0)
    sky_residual = float(np.median(np.abs(result[:, :, 0][sky_region])))

    assert residual_excess > amp * 0.7   # disk signal preserved
    assert sky_residual < 3.0            # sky still genuinely flattened (noise sigma=3.0)


def test_exclusion_mask_none_is_backward_compatible():
    """Default (no mask) behaves exactly as before -- existing callers that
    don't pass exclusion_mask see no change."""
    rgb = _to_rgb(_flat_sky_with_faint_disk(seed=1))
    a = remove_sky_residual(rgb.copy(), mesh_size=64, filter_size=1)
    b = remove_sky_residual(rgb.copy(), mesh_size=64, filter_size=1, exclusion_mask=None)
    assert np.array_equal(a, b)
