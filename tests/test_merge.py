"""Tests for incremental stacking (--merge): registration, weighting,
validation, and header aggregation."""
import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from src.merge import apply_merge_header, load_merge_stack, merge_previous_stacks


def _star_field(shift=(0.0, 0.0), rot_deg=0.0, seed=0, H=256, W=320):
    """Synthetic RGB star field; optionally shifted/rotated (same sky)."""
    rng = np.random.default_rng(seed)
    star_rng = np.random.default_rng(99)  # same stars every call
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    # rotate/shift the sampling grid about the image centre
    th = np.deg2rad(rot_deg)
    cy, cx = H / 2.0, W / 2.0
    ry = (yy - cy) * np.cos(th) - (xx - cx) * np.sin(th) + cy - shift[0]
    rx = (yy - cy) * np.sin(th) + (xx - cx) * np.cos(th) + cx - shift[1]
    img = np.full((H, W), 1000.0)
    for _ in range(40):
        sy, sx = star_rng.uniform(20, H - 20), star_rng.uniform(20, W - 20)
        amp = star_rng.uniform(2000, 9000)
        img += amp * np.exp(-((ry - sy) ** 2 + (rx - sx) ** 2) / (2 * 1.8 ** 2))
    img = img + rng.normal(0, 5, (H, W))
    return np.stack([img, img, img], axis=2).astype(np.float32)


def _write_stack(path, rgb, nframes=None, rawstack=True, intgtime=None):
    hdu = fits.PrimaryHDU(data=np.transpose(rgb, (2, 0, 1)).astype(np.float32))
    if rawstack:
        hdu.header['RAWSTACK'] = (True, 'linear stack')
    if nframes is not None:
        hdu.header['NFRAMES'] = nframes
    if intgtime is not None:
        hdu.header['INTGTIME'] = intgtime
    hdu.writeto(path, overwrite=True)


class TestLoadMergeStack(unittest.TestCase):

    def test_rejects_processed(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'proc.fits')
            _write_stack(p, _star_field(), nframes=10, rawstack=False)
            with self.assertRaises(ValueError):
                load_merge_stack(p)

    def test_loads_linear(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'lin.fits')
            _write_stack(p, _star_field(), nframes=42, intgtime=420.0)
            data, meta = load_merge_stack(p)
            self.assertEqual(data.shape, (256, 320, 3))
            self.assertEqual(meta['nframes'], 42)
            self.assertEqual(meta['intgtime'], 420.0)


class TestMergePreviousStacks(unittest.TestCase):

    def test_weighted_mean_identity_alignment(self):
        """Same sky, zero offset: merged = (w1*a + w2*b) / (w1+w2)."""
        with tempfile.TemporaryDirectory() as td:
            a = _star_field(seed=1)
            b = _star_field(seed=2)
            p = os.path.join(td, 'prev.fits')
            _write_stack(p, b, nframes=30)
            merged, info = merge_previous_stacks(a, 10, [p])
            expected = (10.0 * a.astype(np.float64)
                        + 30.0 * b.astype(np.float64)) / 40.0
            # interior only: warp of a zero-offset transform is near-exact
            m = 8
            diff = np.abs(merged[m:-m, m:-m].astype(np.float64)
                          - expected[m:-m, m:-m])
            self.assertLess(float(diff.max()), 2.0)
            self.assertEqual(info['total_frames'], 40)

    def test_rotated_shifted_stack_aligns(self):
        """A shifted + rotated previous stack must land on the new grid."""
        with tempfile.TemporaryDirectory() as td:
            new = _star_field(seed=3)
            prev = _star_field(shift=(6.4, -9.2), rot_deg=1.2, seed=4)
            p = os.path.join(td, 'prev.fits')
            _write_stack(p, prev, nframes=10)
            merged, info = merge_previous_stacks(new, 10, [p])
            # equal weights: merged = (new + aligned_prev) / 2. If alignment
            # worked, merged stars sit where new's stars sit -> the merged
            # image correlates much better with new than the raw prev did.
            m = 30
            lum = lambda x: x[m:-m, m:-m, 1].astype(np.float64).ravel()
            c_prev = np.corrcoef(lum(new), lum(prev))[0, 1]
            c_merged = np.corrcoef(lum(new), lum(merged))[0, 1]
            self.assertGreater(c_merged, 0.98)
            self.assertGreater(c_merged, c_prev + 0.05)

    def test_low_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            new = _star_field(seed=5)
            # different sky entirely -> registration garbage or low overlap
            rng = np.random.default_rng(6)
            junk = np.clip(rng.normal(1000, 5, (256, 320, 3)), 0,
                           None).astype(np.float32)
            p = os.path.join(td, 'junk.fits')
            _write_stack(p, junk, nframes=10)
            with self.assertRaises(ValueError):
                merge_previous_stacks(new, 10, [p])


class TestApplyMergeHeader(unittest.TestCase):

    def test_header_sums(self):
        hdr = fits.Header()
        hdr['NFRAMES'] = 35
        hdr['INTGTIME'] = 350.0
        hdr['INTGMIN'] = 350.0 / 60
        hdr['TOTEXP'] = 350.0
        hdr['DATEFRST'] = '2026-07-01T21:18:43'
        hdr['DATELAST'] = '2026-07-01T22:20:00'
        info = {'n_sources': 1, 'sources': ['old.fits'],
                'total_frames': 268, 'total_intgtime': 2330.0,
                'total_totexp': 2330.0,
                'datefrst': '2026-06-20T20:00:00',
                'datelast': '2026-06-20T21:00:00'}
        apply_merge_header(hdr, info)
        self.assertEqual(hdr['NFRAMES'], 268)
        self.assertEqual(hdr['MERGED'], 1)
        self.assertEqual(hdr['MRGSRC1'], 'old.fits')
        self.assertAlmostEqual(hdr['INTGTIME'], 2680.0)
        self.assertAlmostEqual(hdr['TOTEXP'], 2680.0)
        self.assertEqual(hdr['DATEFRST'], '2026-06-20T20:00:00')
        self.assertEqual(hdr['DATELAST'], '2026-07-01T22:20:00')


if __name__ == '__main__':
    unittest.main()
