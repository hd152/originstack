"""Regression test for src/pipeline.py::_save_tiff.

_save_tiff transposes (H,W,3) data to a planar (3,H,W) byte layout but was
tagging the file `planarconfig='contig'` (asserts interleaved/chunky RGB) --
a metadata lie that made standards-compliant readers either decode garbage
or refuse the file outright ("More samples per pixel than can be decoded"),
reported by a user as a "corrupt" TIFF. The fix tags it 'separate' to match
the actual byte layout.
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
