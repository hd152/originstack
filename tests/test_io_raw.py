"""Tests for src/io_raw.py -- camera RAW (rawpy-based) loading.

No real CR2/NEF fixture files are used; rawpy itself is mocked throughout,
since the module's job is just orchestrating rawpy's API + doing black-level/
white-level normalization math -- both fully testable without real RAW bytes.
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from src import io_raw


class TestIsRawFile(unittest.TestCase):
    def test_known_extensions(self):
        for ext in ('.cr2', '.CR3', '.nef', '.arw', '.dng', '.orf', '.raf'):
            self.assertTrue(io_raw.is_raw_file(f'frame{ext}'), ext)

    def test_non_raw_extensions(self):
        for ext in ('.fits', '.fit', '.tiff', '.jpg'):
            self.assertFalse(io_raw.is_raw_file(f'frame{ext}'), ext)


class TestBayerPatternStr(unittest.TestCase):
    def test_standard_rggb(self):
        raw = mock.Mock()
        raw.color_desc = b'RGBG'
        raw.raw_pattern = np.array([[0, 1], [3, 2]])
        self.assertEqual(io_raw._bayer_pattern_str(raw), 'RGGB')

    def test_falls_back_on_error(self):
        raw = mock.Mock()
        type(raw).color_desc = mock.PropertyMock(side_effect=RuntimeError('no exif'))
        self.assertEqual(io_raw._bayer_pattern_str(raw), 'RGGB')


def _make_mock_raw(width=4, height=4, black_level=(100, 100, 100, 100),
                   white_level=1000, color_desc=b'RGBG',
                   pattern=((0, 1), (3, 2))):
    """A rawpy.imread(...) context-manager mock with a small synthetic mosaic."""
    raw = mock.MagicMock()
    raw.__enter__.return_value = raw
    raw.__exit__.return_value = False
    raw.sizes = mock.Mock(width=width, height=height)
    raw.color_desc = color_desc
    raw.raw_pattern = np.array(pattern)
    raw.black_level_per_channel = list(black_level)
    raw.white_level = white_level
    # Deterministic mosaic: value = row*width + col + 200 (well above black level)
    yy, xx = np.indices((height, width))
    raw.raw_image_visible = (yy * width + xx + 200).astype(np.uint16)
    return raw


class TestReadRaw(unittest.TestCase):
    def setUp(self):
        self._had_rawpy = io_raw.HAS_RAWPY
        io_raw.HAS_RAWPY = True

    def tearDown(self):
        io_raw.HAS_RAWPY = self._had_rawpy

    def test_normalization_and_shape(self):
        mock_raw = _make_mock_raw()
        with mock.patch.object(io_raw, '_rawpy') as _rawpy_mod:
            _rawpy_mod.imread.return_value = mock_raw
            with mock.patch.object(io_raw, '_exif_from_pillow', return_value={}):
                data, hdr = io_raw.read_raw('fake.cr2')

        self.assertEqual(data.dtype, np.float32)
        self.assertEqual(data.shape, (4, 4))
        # scale = white_level - max(black_level) = 1000 - 100 = 900
        # pixel (0,0) raw value = 200, channel black level = 100 -> (200-100)/900
        self.assertAlmostEqual(float(data[0, 0]), (200.0 - 100.0) / 900.0, places=5)
        self.assertGreaterEqual(float(data.min()), 0.0)
        self.assertLessEqual(float(data.max()), 1.0)

        self.assertEqual(hdr['NAXIS1'], 4)
        self.assertEqual(hdr['NAXIS2'], 4)
        self.assertEqual(hdr['BAYERPAT'], 'RGGB')
        self.assertEqual(hdr['COLORTYP'], 'RGGB')
        self.assertEqual(hdr['IMAGETYP'], 'Light Frame')
        self.assertTrue(hdr['_RAW_FILE'])

    def test_header_only_never_touches_pixel_data(self):
        class _NoPixelAccessRaw:
            """Explicit test double: raises if pixel data is ever read, so the
            header-only fast path's "never decodes pixels" contract is a real,
            enforced assertion rather than an unverifiable MagicMock stand-in."""
            sizes = mock.Mock(width=4, height=4)
            color_desc = b'RGBG'
            raw_pattern = np.array([[0, 1], [3, 2]])

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            @property
            def raw_image_visible(self):
                raise AssertionError('header-only path must not read pixel data')

        with mock.patch.object(io_raw, '_rawpy') as _rawpy_mod:
            _rawpy_mod.imread.return_value = _NoPixelAccessRaw()
            with mock.patch.object(io_raw, '_exif_from_pillow', return_value={}):
                hdr = io_raw.read_raw_header('fake.cr2')

        self.assertEqual(hdr['NAXIS1'], 4)
        self.assertEqual(hdr['NAXIS2'], 4)
        self.assertEqual(hdr['BAYERPAT'], 'RGGB')


class TestHasRawpyFalse(unittest.TestCase):
    def setUp(self):
        self._had_rawpy = io_raw.HAS_RAWPY
        io_raw.HAS_RAWPY = False

    def tearDown(self):
        io_raw.HAS_RAWPY = self._had_rawpy

    def test_read_raw_header_returns_minimal_dict(self):
        hdr = io_raw.read_raw_header('fake.cr2')
        self.assertEqual(hdr, {'_RAW_FILE': True, 'IMAGETYP': 'Light Frame'})

    def test_read_raw_raises_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            io_raw.read_raw('fake.cr2')
        self.assertIn('rawpy is not installed', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
