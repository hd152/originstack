"""Comet mode blends a star-aligned and a comet-aligned stack. Each pass has its
own common crop, so the stacks differ in shape -- (2013, 3031) vs (2022, 3032)
on a real SWAN session made the blend raise a broadcast error and the run fail.
``_shared_crop`` cuts both to the rectangle both crops cover on the reference grid."""
import numpy as np
import pytest

from src.pipeline import _shared_crop
from src.stacking import blend_comet_star_stacks


def _grid(H=60, W=80):
    rng = np.random.default_rng(0)
    return rng.random((H, W, 3)).astype(np.float32)


def test_shared_crop_returns_the_same_grid_pixels():
    g = _grid()
    crop_a, crop_b = (5, 50, 3, 70), (8, 55, 1, 72)
    a = g[crop_a[0]:crop_a[1], crop_a[2]:crop_a[3]]
    b = g[crop_b[0]:crop_b[1], crop_b[2]:crop_b[3]]
    ca, cb = _shared_crop(a, crop_a, b, crop_b)
    assert ca.shape == cb.shape == (50 - 8, 70 - 3, 3)
    np.testing.assert_array_equal(ca, cb)
    np.testing.assert_array_equal(ca, g[8:50, 3:70])


def test_blend_runs_on_mismatched_crops():
    g = _grid()
    crop_a, crop_b = (2, 58, 0, 79), (0, 57, 1, 80)
    a = g[crop_a[0]:crop_a[1], crop_a[2]:crop_a[3]]
    b = g[crop_b[0]:crop_b[1], crop_b[2]:crop_b[3]]
    with pytest.raises(ValueError):
        blend_comet_star_stacks(a, b, b.mean(axis=2))   # the bug: shapes differ
    ca, cb = _shared_crop(a, crop_a, b, crop_b)
    out = blend_comet_star_stacks(ca, cb, cb.mean(axis=2))
    assert out.shape == ca.shape


def test_disjoint_or_mismatched_crops_raise():
    g = _grid()
    with pytest.raises(ValueError):
        _shared_crop(g[0:10, 0:10], (0, 10, 0, 10), g[20:30, 20:30], (20, 30, 20, 30))
    with pytest.raises(ValueError):
        _shared_crop(g[0:10, 0:10], (0, 11, 0, 10), g[0:10, 0:10], (0, 10, 0, 10))


def test_reference_is_recovered_from_a_restored_registration():
    """A session resumed from a phase-2 checkpoint skips reference selection; comet
    mode then needs best_idx / ref_lum (it raised UnboundLocalError on a real SWAN
    folder). The reference is the frame registered onto itself."""
    from types import SimpleNamespace

    from src.pipeline import _reference_from_registration
    mem_lum = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    final = [SimpleNamespace(metrics={'score': s}) for s in (9.0, 1.0, 5.0)]
    final_indices = [0, 2, 4]
    shifts = [(1.5, -2.0), (0.0, 0.0), (0.3, 0.1)]
    best_idx, ref = _reference_from_registration(final, final_indices, shifts,
                                                 [None, None, None], mem_lum)
    assert best_idx == 2
    np.testing.assert_array_equal(ref, mem_lum[2])
    # zero shift but a real transform is not the identity: fall back to the best score
    best_idx, _ = _reference_from_registration(final, final_indices, shifts,
                                               [None, object(), None], mem_lum)
    assert best_idx == 0
