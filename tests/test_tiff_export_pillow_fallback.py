"""Regression test for src/pipeline.py::_save_tiff's Pillow-only fallback
path (used when tifffile isn't installed -- e.g. CI, which installs Pillow
as a required dependency but not the optional tifffile).

Real bug: ``Image.fromarray(arr16)`` on a (H,W,3) uint16 array raises
``TypeError: Cannot handle this data type: (1, 1, 3), <u2`` -- Pillow has no
mode for 16-bit-per-channel RGB via ``fromarray``. This silently turned into
"WARNING: TIFF export failed" and no file, caught by ``--stream``'s own
TIFF-export test only because that test runs on CI where tifffile is absent
-- forced here directly, independent of whether tifffile happens to be
installed on the machine running the test.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from src.pipeline import _save_tiff


def _block_tifffile_import(monkeypatch):
    """Force `import tifffile` to raise ImportError inside _save_tiff,
    regardless of whether tifffile is actually installed: a None entry in
    sys.modules makes the import machinery raise ImportError immediately."""
    monkeypatch.setitem(sys.modules, 'tifffile', None)


def test_save_tiff_pillow_fallback_writes_valid_rgb(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    _block_tifffile_import(monkeypatch)

    H, W = 30, 40
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[:, :, 0] = 15000.0
    rgb[:, :, 1] = 18000.0
    rgb[:, :, 2] = 20000.0
    rgb[5, 5, :] = 79726.0  # bright pixel sets the peak, like a real stack

    out_path = str(tmp_path / "stack.fits")
    _save_tiff(rgb, out_path)

    tiff_path = tmp_path / "stack.tiff"
    assert tiff_path.exists()

    from PIL import Image
    with Image.open(str(tiff_path)) as im:
        assert im.mode == 'RGB'
        assert im.size == (W, H)
        arr = np.asarray(im)

    assert arr.shape == (H, W, 3)
    # Pure scale factor (peak-normalized then quantized to 8-bit) -- channel
    # ratios should still roughly hold.
    assert arr[10, 10, 1] > arr[10, 10, 0] > 0
    assert arr[10, 10, 2] > arr[10, 10, 1]


def test_save_tiff_pillow_fallback_handles_all_zero_input(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    _block_tifffile_import(monkeypatch)

    H, W = 20, 20
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    out_path = str(tmp_path / "stack.fits")
    _save_tiff(rgb, out_path)

    tiff_path = tmp_path / "stack.tiff"
    assert tiff_path.exists()

    from PIL import Image
    with Image.open(str(tiff_path)) as im:
        arr = np.asarray(im)
    assert np.isfinite(arr).all()
    np.testing.assert_array_equal(arr, 0)
