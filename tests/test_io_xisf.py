"""Tests for src/io_xisf.py -- round-trips through this codebase's own
xisf_writer.py (a free, exact fixture) plus explicit error-case coverage."""
from __future__ import annotations

import os
import struct
import tempfile
import unittest

import numpy as np

from src.io_xisf import is_xisf_file, read_xisf, read_xisf_header
from src.xisf_writer import write_xisf


class TestIsXisfFile(unittest.TestCase):
    def test_extension(self):
        self.assertTrue(is_xisf_file('stack.xisf'))
        self.assertFalse(is_xisf_file('stack.fits'))


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmpdir.cleanup)
        self.path = os.path.join(self.tmpdir.name, 'out.xisf')

    def test_roundtrip_pixel_values(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 60000, size=(12, 16, 3)).astype(np.float32)
        write_xisf(img, self.path)

        data, hdr = read_xisf(self.path)
        self.assertEqual(data.shape, (12, 16, 3))
        self.assertEqual(data.dtype, np.float32)
        np.testing.assert_allclose(data, img, rtol=1e-6)
        self.assertEqual(hdr['NAXIS1'], 16)
        self.assertEqual(hdr['NAXIS2'], 12)
        self.assertEqual(hdr['NAXIS3'], 3)

    def test_roundtrip_header_meta(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        write_xisf(img, self.path, header_meta={'EXPTIME': 30.0, 'OBJECT': 'M51'})

        _, hdr = read_xisf(self.path)
        self.assertAlmostEqual(hdr['EXPTIME'], 30.0)
        self.assertEqual(hdr['OBJECT'], 'M51')

    def test_header_only_matches_full_read(self):
        img = np.zeros((5, 7, 3), dtype=np.float32)
        write_xisf(img, self.path)

        hdr = read_xisf_header(self.path)
        data, _ = read_xisf(self.path)
        self.assertEqual(hdr['NAXIS1'], data.shape[1])
        self.assertEqual(hdr['NAXIS2'], data.shape[0])

    def test_large_metadata_offset_still_roundtrips(self):
        """write_xisf grows the data offset past 4096 if metadata is large;
        the reader must follow the location attribute, not assume 4096."""
        img = np.ones((3, 3, 3), dtype=np.float32)
        big_meta = {f'KEY{i}': 'x' * 100 for i in range(60)}
        write_xisf(img, self.path, header_meta=big_meta)
        data, _ = read_xisf(self.path)
        np.testing.assert_allclose(data, img)


class TestErrorCases(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmpdir.cleanup)

    def test_bad_signature_raises(self):
        path = os.path.join(self.tmpdir.name, 'bad.xisf')
        with open(path, 'wb') as fh:
            fh.write(b'NOTXISF0' + struct.pack('<II', 0, 0))
        with self.assertRaises(ValueError):
            read_xisf(path)

    def test_embedded_location_raises(self):
        path = os.path.join(self.tmpdir.name, 'embedded.xisf')
        xml = (
            b'<xisf xmlns="http://www.pixinsight.com/xisf">'
            b'<Image geometry="2:2:1" sampleFormat="Float32" colorSpace="Gray" '
            b'location="embedded:base64"/></xisf>'
        )
        with open(path, 'wb') as fh:
            fh.write(b'XISF0100' + struct.pack('<II', len(xml), 0))
            fh.write(xml)
        with self.assertRaises(NotImplementedError):
            read_xisf(path)

    def test_unsupported_sample_format_raises(self):
        path = os.path.join(self.tmpdir.name, 'weird.xisf')
        xml = (
            b'<xisf xmlns="http://www.pixinsight.com/xisf">'
            b'<Image geometry="2:2:1" sampleFormat="Complex64" colorSpace="Gray" '
            b'location="attachment:64:16"/></xisf>'
        )
        with open(path, 'wb') as fh:
            header_block = b'XISF0100' + struct.pack('<II', len(xml), 0) + xml
            fh.write(header_block.ljust(64, b'\x00'))
            fh.write(b'\x00' * 16)
        with self.assertRaises(ValueError):
            read_xisf(path)

    def test_read_xisf_header_graceful_on_garbage(self):
        path = os.path.join(self.tmpdir.name, 'garbage.xisf')
        with open(path, 'wb') as fh:
            fh.write(b'not an xisf file at all')
        hdr = read_xisf_header(path)
        self.assertTrue(hdr['_XISF_FILE'])
        self.assertNotIn('NAXIS1', hdr)


class TestMonoReplication(unittest.TestCase):
    def test_mono_xisf_replicated_to_three_channels(self):
        tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmpdir.cleanup)
        path = os.path.join(tmpdir.name, 'mono.xisf')
        h, w = 4, 5
        payload = (np.arange(h * w, dtype=np.float32)).tobytes()
        xml = (
            f'<xisf xmlns="http://www.pixinsight.com/xisf">'
            f'<Image geometry="{w}:{h}:1" sampleFormat="Float32" colorSpace="Gray" '
            f'location="attachment:64:{len(payload)}"/></xisf>'
        ).encode('utf-8')
        with open(path, 'wb') as fh:
            header_block = b'XISF0100' + struct.pack('<II', len(xml), 0) + xml
            fh.write(header_block.ljust(64, b'\x00'))
            fh.write(payload)

        data, _ = read_xisf(path)
        self.assertEqual(data.shape, (h, w, 3))
        # All three channels identical (replicated mono)
        np.testing.assert_array_equal(data[:, :, 0], data[:, :, 1])
        np.testing.assert_array_equal(data[:, :, 1], data[:, :, 2])


if __name__ == '__main__':
    unittest.main()
