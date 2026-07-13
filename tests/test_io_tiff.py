"""Tests for src/io_tiff.py -- TIFF loading via tifffile."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from src import io_tiff

try:
    import tifffile
    HAS_TIFFFILE = True
except Exception:
    HAS_TIFFFILE = False


class TestIsTiffFile(unittest.TestCase):
    def test_known_extensions(self):
        self.assertTrue(io_tiff.is_tiff_file('frame.tif'))
        self.assertTrue(io_tiff.is_tiff_file('frame.TIFF'))

    def test_non_tiff_extensions(self):
        self.assertFalse(io_tiff.is_tiff_file('frame.fits'))
        self.assertFalse(io_tiff.is_tiff_file('frame.cr2'))


@unittest.skipUnless(HAS_TIFFFILE, "tifffile not installed")
class TestReadTiffRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmpdir.cleanup)

    def test_mono_16bit_roundtrip(self):
        arr = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 100)
        path = os.path.join(self.tmpdir.name, 'mono.tif')
        tifffile.imwrite(path, arr)

        data, hdr = io_tiff.read_tiff(path)
        self.assertEqual(data.ndim, 2)
        self.assertEqual(data.dtype, np.float32)
        np.testing.assert_allclose(data, arr.astype(np.float32))
        self.assertEqual(hdr['NAXIS1'], 8)
        self.assertEqual(hdr['NAXIS2'], 8)
        self.assertEqual(hdr['IMAGETYP'], 'Light Frame')

    def test_rgb_roundtrip_passthrough_shape(self):
        arr = np.random.default_rng(0).integers(0, 65535, size=(6, 10, 3)).astype(np.uint16)
        path = os.path.join(self.tmpdir.name, 'rgb.tif')
        tifffile.imwrite(path, arr)

        data, hdr = io_tiff.read_tiff(path)
        self.assertEqual(data.shape, (6, 10, 3))
        self.assertEqual(data.dtype, np.float32)
        np.testing.assert_allclose(data, arr.astype(np.float32))
        self.assertEqual(hdr['NAXIS3'], 3)

    def test_header_only_matches_full_read_dimensions(self):
        arr = np.zeros((12, 20), dtype=np.uint16)
        path = os.path.join(self.tmpdir.name, 'dims.tif')
        tifffile.imwrite(path, arr)

        hdr = io_tiff.read_tiff_header(path)
        data, _ = io_tiff.read_tiff(path)
        self.assertEqual(hdr['NAXIS1'], data.shape[1])
        self.assertEqual(hdr['NAXIS2'], data.shape[0])

    def test_no_pixel_rescaling(self):
        """Integer TIFF values must be preserved as-is (ADU-count convention),
        not rescaled to [0,1] -- otherwise a TIFF light would silently
        mismatch the scale of FITS/RAW calibration masters."""
        arr = np.full((4, 4), 40000, dtype=np.uint16)
        path = os.path.join(self.tmpdir.name, 'flat_value.tif')
        tifffile.imwrite(path, arr)
        data, _ = io_tiff.read_tiff(path)
        self.assertAlmostEqual(float(data[0, 0]), 40000.0)


class TestHasTifffileFalse(unittest.TestCase):
    def setUp(self):
        self._had = io_tiff.HAS_TIFFFILE
        io_tiff.HAS_TIFFFILE = False

    def tearDown(self):
        io_tiff.HAS_TIFFFILE = self._had

    def test_read_tiff_header_returns_minimal_dict(self):
        hdr = io_tiff.read_tiff_header('fake.tif')
        self.assertTrue(hdr['_TIFF_FILE'])
        self.assertNotIn('NAXIS1', hdr)

    def test_read_tiff_raises_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            io_tiff.read_tiff('fake.tif')
        self.assertIn('tifffile is not installed', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
