"""Regression tests for src/pipeline.py::_save_tiff.

Two real bugs found via user reports:

1. _save_tiff transposes (H,W,3) data to a planar (3,H,W) byte layout but
   was tagging the file `planarconfig='contig'` (asserts interleaved/chunky
   RGB) -- a metadata lie that made standards-compliant readers either
   decode garbage or refuse the file outright ("More samples per pixel than
   can be decoded"), reported as a "corrupt" TIFF. Fixed by tagging it
   'separate' to match the actual byte layout.

2. Even after (1), the file "opened but didn't render correctly": the
   post-processed stack is raw linear ADU counts (thousands), and the
   primary tifffile write path never normalized it -- leaving ~99.9% of
   pixels above 1.0, which most 32-bit-float TIFF readers clip to solid
   white (the function's own Pillow fallback path already assumed [0,1]
   input via `np.clip(..., 0, 1)`, an inconsistency that was itself a tell).
   Fixed by normalizing to [0,1] by the frame's own peak -- a pure scale
   factor, stays linear -- before either write path.
"""
from __future__ import annotations

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from src.pipeline import _save_tiff


def test_save_tiff_planarconfig_matches_actual_layout(tmp_path):
    H, W = 30, 40
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[:, :, 0] = 1.0
    rgb[:, :, 1] = 0.5
    rgb[:, :, 2] = 0.25

    out_path = str(tmp_path / "stack.fits")
    _save_tiff(rgb, out_path)
    tiff_path = tmp_path / "stack.tiff"
    assert tiff_path.exists()

    with tifffile.TiffFile(str(tiff_path)) as tf:
        page = tf.pages[0]
        # PlanarConfiguration=2 (Separate) is the only tag consistent with
        # the (3,H,W) byte layout _save_tiff actually writes.
        assert page.tags["PlanarConfiguration"].value == 2
        assert page.tags["SamplesPerPixel"].value == 3

    # Round-trip through tifffile AND (if available) an independent reader.
    back = tifffile.imread(str(tiff_path))
    assert back.shape == (3, H, W)
    np.testing.assert_allclose(back[0], 1.0)
    np.testing.assert_allclose(back[1], 0.5)
    np.testing.assert_allclose(back[2], 0.25)

    try:
        import imageio.v3 as iio
    except ImportError:
        return
    arr = iio.imread(str(tiff_path))
    assert arr.shape == (3, H, W)
    np.testing.assert_allclose(arr[0], 1.0)


def test_save_tiff_normalizes_raw_adu_range(tmp_path):
    """Real post-processed stacks are linear ADU counts in the thousands,
    not [0,1] -- the exact case that rendered as solid white/garbage."""
    H, W = 30, 40
    rgb = np.full((H, W, 3), 15000.0, dtype=np.float32)
    rgb[:, :, 1] = 18000.0
    rgb[:, :, 2] = 20000.0
    rgb[5, 5, :] = 79726.0  # a bright star pixel sets the peak

    out_path = str(tmp_path / "stack.fits")
    _save_tiff(rgb, out_path)
    arr = tifffile.imread(str(tmp_path / "stack.tiff"))

    assert arr.max() <= 1.0
    assert arr.min() >= 0.0
    assert (arr > 1.0).mean() == 0.0
    # Pure scale factor -- channel ratios (linear proportionality) preserved.
    np.testing.assert_allclose(arr[0, 10, 10] / arr[1, 10, 10], 15000.0 / 18000.0, rtol=1e-5)
    np.testing.assert_allclose(arr[2, 10, 10] / arr[1, 10, 10], 20000.0 / 18000.0, rtol=1e-5)


def test_save_tiff_handles_all_zero_input(tmp_path):
    """peak <= 0 (e.g. an all-black/failed stack) must not divide by zero."""
    H, W = 20, 20
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    out_path = str(tmp_path / "stack.fits")
    _save_tiff(rgb, out_path)
    arr = tifffile.imread(str(tmp_path / "stack.tiff"))
    assert np.isfinite(arr).all()
    np.testing.assert_allclose(arr, 0.0)
