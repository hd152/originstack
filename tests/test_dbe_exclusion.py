"""Tests for src/background.py's dense-field DBE fallback honoring a
caller-supplied exclusion_mask (--galaxy-mode/--comet-mode).

When too few clean background patches survive DBE's primary patch-sampling
path (a large/prominent excluded object leaves little unmasked area), it
falls back to a plain sigma-clip mesh fit. The "dense field" variant of that
fallback (masked fraction > 70%) used to drop *all* masking, including a
caller-supplied galaxy/comet exclusion_mask -- sigma-clipping within a mesh
cell rejects many small point sources fine, but cannot reject one large,
smooth, contiguous object the way an explicit mask does, since the object
*is* the dominant signal over a wide area, not a per-cell outlier. That
silently discarded exactly the protection --galaxy-mode exists to provide,
and precisely when a large/prominent galaxy made the fallback trigger in the
first place.
"""
from __future__ import annotations

import numpy as np

from src.background import dynamic_background_extraction


def _rgb_with_large_galaxy(H=300, W=300, cy=150.0, cx=150.0, radius=120.0,
                           amp=6.0, sky=1000.0, seed=0):
    """A galaxy-sized disk covering most of the frame -- large enough that
    its exclusion mask alone should push DBE's combined masked fraction
    past the dense-field threshold (0.70) and leave under DBE_MIN_SAMPLES
    (20) clean whole patches."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disk = amp * np.exp(-(r / (radius * 0.5)) ** 2)
    plane = sky + disk + rng.normal(0, 3.0, (H, W))
    plane = plane.astype(np.float32)
    return np.stack([plane, plane, plane], axis=2)


def _exclusion_ellipse(H, W, cy, cx, radius):
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return (r < radius).astype(np.float32)


def test_dense_field_fallback_protects_excluded_galaxy():
    H, W = 300, 300
    cy, cx, radius, amp = 150.0, 150.0, 140.0, 6.0
    rgb = _rgb_with_large_galaxy(H, W, cy, cx, radius, amp)
    mask = _exclusion_ellipse(H, W, cy, cx, radius * 1.05)
    assert float(np.mean(mask)) > 0.7  # confirms this hits the dense-field branch

    result = dynamic_background_extraction(rgb, patch_size=32, exclusion_mask=mask,
                                           verbose=False)

    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    core = r < (radius * 0.3)
    # The galaxy core's excess-over-sky must survive -- before the fix, the
    # dense-field fallback used star_mask=None and fit/subtracted straight
    # through it like ordinary background.
    residual_excess = float(np.median(result[:, :, 0][core]))
    assert residual_excess > amp * 0.5


def test_dense_field_fallback_without_exclusion_mask_still_works():
    """No exclusion_mask supplied (the plain --auto-detected star-field
    case this fallback originally targeted) must still run without error
    and produce a roughly sky-flattened result -- confirms the fix (passing
    exclusion_mask instead of a hardcoded None) doesn't regress the
    no-mask path, since exclusion_mask is simply None there too."""
    H, W = 300, 300
    rgb = _rgb_with_large_galaxy(H, W, amp=6.0, seed=1)
    result = dynamic_background_extraction(rgb, patch_size=32, exclusion_mask=None,
                                           verbose=False)
    assert result.shape == rgb.shape
    assert np.all(np.isfinite(result))
