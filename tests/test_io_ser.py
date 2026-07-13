"""Tests for src/io_ser.py -- SER (planetary video) reading.

Hand-crafts small .ser files with struct.pack (no external fixtures/tools
needed) -- the SER format is simple and fully specified, so this is exact.
"""
from __future__ import annotations

import os
import struct
import tempfile
import unittest

import numpy as np

from src import io_ser
from src.frame_discovery import classify_frame
from src.models import FrameInfo


def _write_ser(path: str, color_id: int, width: int, height: int,
              depth: int, frames: list, little_endian: int = 1) -> None:
    """frames: list of np.ndarray, each already shaped/dtyped for one frame."""
    num_planes = 3 if color_id in (100, 101) else 1
    header = struct.pack(
        '<14s7i40s40s40sqq',
        b'LUCAM-RECORDER',
        0, color_id, little_endian, width, height, depth, len(frames),
        b'\x00' * 40, b'\x00' * 40, b'\x00' * 40,
        0, 0,
    )
    with open(path, 'wb') as fh:
        fh.write(header)
        for fr in frames:
            fh.write(fr.tobytes())


class TestHeaderParse(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmpdir.cleanup)
        io_ser._parse_ser_header.cache_clear()

    def test_mono_uint8_header(self):
        path = os.path.join(self.tmpdir.name, 'mono.ser')
        frames = [np.full((4, 4), i * 10, dtype=np.uint8) for i in range(3)]
        _write_ser(path, color_id=0, width=4, height=4, depth=8, frames=frames)

        info = io_ser._parse_ser_header(path)
        self.assertEqual(info['FrameCount'], 3)
        self.assertEqual(info['Width'], 4)
        self.assertEqual(info['Height'], 4)
        self.assertEqual(info['ColorID'], 0)
        self.assertEqual(info['BytesPerSample'], 1)
        self.assertEqual(info['NumPlanes'], 1)
        self.assertEqual(info['FrameSize'], 16)


class TestExpandSerFiles(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmpdir.cleanup)
        io_ser._parse_ser_header.cache_clear()

    def test_expansion_order_and_count(self):
        path = os.path.join(self.tmpdir.name, 'clip.ser')
        frames = [np.full((2, 2), i, dtype=np.uint8) for i in range(3)]
        _write_ser(path, color_id=0, width=2, height=2, depth=8, frames=frames)

        vpaths = io_ser.expand_ser_files(self.tmpdir.name)
        self.assertEqual(len(vpaths), 3)
        self.assertTrue(vpaths[0].endswith('::0'))
        self.assertTrue(vpaths[1].endswith('::1'))
        self.assertTrue(vpaths[2].endswith('::2'))

    def test_ignores_non_ser_files(self):
        with open(os.path.join(self.tmpdir.name, 'notme.txt'), 'w') as fh:
            fh.write('x')
        self.assertEqual(io_ser.expand_ser_files(self.tmpdir.name), [])


class TestFrameReadCorrectness(unittest.TestCase):
    """The critical test: memmap offset arithmetic must select the RIGHT
    frame, not just A frame -- verify 3 distinct hand-written frames read
    back as 3 distinct, correct pixel arrays."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmpdir.cleanup)
        io_ser._parse_ser_header.cache_clear()

    def test_mono_frames_distinct_and_correct(self):
        path = os.path.join(self.tmpdir.name, 'mono.ser')
        values = [10, 20, 30]
        frames = [np.full((4, 4), v, dtype=np.uint8) for v in values]
        _write_ser(path, color_id=0, width=4, height=4, depth=8, frames=frames)

        for i, v in enumerate(values):
            data, hdr = io_ser.read_ser_frame(f'{path}::{i}')
            self.assertEqual(data.shape, (4, 4, 3))  # mono replicated to 3ch
            self.assertTrue(np.allclose(data, float(v)))
            self.assertEqual(hdr['NAXIS1'], 4)
            self.assertEqual(hdr['NAXIS2'], 4)

        # Cross-check: frame 0 and frame 1 pixel data must differ
        d0, _ = io_ser.read_ser_frame(f'{path}::0')
        d1, _ = io_ser.read_ser_frame(f'{path}::1')
        self.assertFalse(np.allclose(d0, d1))

    def test_bayer_rggb_16bit(self):
        path = os.path.join(self.tmpdir.name, 'bayer.ser')
        frames = [
            (np.arange(16, dtype=np.uint16).reshape(4, 4) * 100),
            (np.arange(16, dtype=np.uint16).reshape(4, 4)[::-1] * 100),
        ]
        _write_ser(path, color_id=8, width=4, height=4, depth=16, frames=frames,
                  little_endian=1)

        hdr = io_ser.read_ser_frame_header(f'{path}::0')
        self.assertEqual(hdr['BAYERPAT'], 'RGGB')
        data0, _ = io_ser.read_ser_frame(f'{path}::0')
        data1, _ = io_ser.read_ser_frame(f'{path}::1')
        self.assertEqual(data0.ndim, 2)  # not replicated -- real Bayer mosaic
        np.testing.assert_allclose(data0, frames[0].astype(np.float32))
        self.assertFalse(np.allclose(data0, data1))

    def test_rgb_and_bgr_channel_order(self):
        h, w = 3, 3
        # Distinguishable per-channel constant colors
        rgb_frame = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_frame[:, :, 0] = 10   # R
        rgb_frame[:, :, 1] = 20   # G
        rgb_frame[:, :, 2] = 30   # B

        rgb_path = os.path.join(self.tmpdir.name, 'rgb.ser')
        _write_ser(rgb_path, color_id=100, width=w, height=h, depth=8, frames=[rgb_frame])
        data, _ = io_ser.read_ser_frame(f'{rgb_path}::0')
        self.assertAlmostEqual(float(data[0, 0, 0]), 10.0, places=5)
        self.assertAlmostEqual(float(data[0, 0, 1]), 20.0, places=5)
        self.assertAlmostEqual(float(data[0, 0, 2]), 30.0, places=5)

        bgr_path = os.path.join(self.tmpdir.name, 'bgr.ser')
        # Same on-disk byte layout as rgb_frame, but ColorID says BGR, so
        # channel 0 on disk is "B" and must be swapped to land in R on read
        _write_ser(bgr_path, color_id=101, width=w, height=h, depth=8, frames=[rgb_frame])
        data_bgr, _ = io_ser.read_ser_frame(f'{bgr_path}::0')
        self.assertAlmostEqual(float(data_bgr[0, 0, 0]), 30.0, places=5)  # R <- disk B
        self.assertAlmostEqual(float(data_bgr[0, 0, 2]), 10.0, places=5)  # B <- disk R

    def test_out_of_range_index_raises(self):
        path = os.path.join(self.tmpdir.name, 'small.ser')
        _write_ser(path, color_id=0, width=2, height=2, depth=8,
                  frames=[np.zeros((2, 2), dtype=np.uint8)])
        with self.assertRaises(IndexError):
            io_ser.read_ser_frame(f'{path}::5')

    def test_unsupported_cmy_variant_raises(self):
        path = os.path.join(self.tmpdir.name, 'cmy.ser')
        _write_ser(path, color_id=16, width=2, height=2, depth=8,
                  frames=[np.zeros((2, 2), dtype=np.uint8)])
        with self.assertRaises(NotImplementedError):
            io_ser.read_ser_frame(f'{path}::0')


class TestIsSerVirtualPath(unittest.TestCase):
    def test_positive(self):
        self.assertTrue(io_ser.is_ser_virtual_path('C:/x/clip.ser::42'))
        self.assertTrue(io_ser.is_ser_virtual_path('clip.SER::0'))

    def test_negative(self):
        self.assertFalse(io_ser.is_ser_virtual_path('clip.ser'))
        self.assertFalse(io_ser.is_ser_virtual_path('frame.fits'))


class TestClassifyFrameOnVirtualPath(unittest.TestCase):
    def test_dark_basename_still_classified(self):
        f = FrameInfo(path='C:/x/dark.ser::5', type='light', header={})
        self.assertEqual(classify_frame(f.path, {}), 'dark')

    def test_light_default(self):
        f = FrameInfo(path='C:/x/Jupiter_2024.ser::100', type='light', header={})
        self.assertEqual(classify_frame(f.path, {}), 'light')


if __name__ == '__main__':
    unittest.main()
