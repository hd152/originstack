"""Parity tests for the optional native (Rust) kernels.

Skipped entirely when `astro_native` is not built/installed, so the suite
still passes in a pure-Python environment.
"""
import numpy as np
import pytest

import astro_stack as astro

native = pytest.importorskip("astro_native")


def _stack(n=30, h=24, w=28, c=3, seed=0, outliers=True):
    rng = np.random.default_rng(seed)
    d = rng.normal(1000.0, 30.0, (n, h, w, c)).astype(np.float32)
    if outliers:
        for _ in range(n * h * w // 15):
            d[rng.integers(0, n), rng.integers(0, h), rng.integers(0, w),
              rng.integers(0, c)] += rng.choice([-1, 1]) * rng.uniform(200, 2000)
    return d


@pytest.mark.parametrize("use_mad", [True, False])
@pytest.mark.parametrize("winsorize", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_sigma_clip_matches_numpy(use_mad, winsorize, weighted):
    """Native combine must match the numpy reference within float tolerance."""
    d = _stack(seed=hash((use_mad, winsorize, weighted)) & 0xFFFF)
    w = None
    if weighted:
        w = np.random.default_rng(1).uniform(0.5, 1.5, d.shape[0]).astype(np.float32)

    # numpy reference: pass float64 so _native_usable() returns False (dtype
    # guard) and the pure-numpy tiled path runs. Its per-tile float32 cast makes
    # the inputs identical to what the native kernel sees.
    ref = astro.sigma_clip_combine(
        d.astype(np.float64), sigma=3.0, max_iters=3, weights=w,
        winsorize=winsorize, use_mad=use_mad)
    got = native.sigma_clip_combine(d, 3.0, 3, w, winsorize, use_mad)

    assert got.shape == ref.shape
    assert got.dtype == np.float32
    # background ~1000 ADU; sub-ADU agreement is well within stacking tolerance.
    assert float(np.max(np.abs(ref.astype(np.float64) - got))) < 2.0


def test_all_nan_pixel_is_zero():
    d = _stack(n=8, h=4, w=4, c=1, outliers=False)
    d[:, 0, 0, 0] = np.nan
    got = native.sigma_clip_combine(d, 3.0, 3, None, False, True)
    assert np.isfinite(got).all()
    assert got[0, 0, 0] == 0.0
