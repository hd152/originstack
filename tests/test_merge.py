"""Tests for incremental stacking (--merge): registration, weighting,
validation, and header aggregation."""
import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from src.merge import (
    apply_merge_header,
    load_merge_stack,
    merge_previous_stacks,
    read_merge_meta,
    seed_reference_header,
)


def _star_field(shift=(0.0, 0.0), rot_deg=0.0, seed=0, H=256, W=320, noise=5.0):
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
    img = img + rng.normal(0, noise, (H, W))
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
            # Stack noise falls as 1/sqrt(N), as it does for real stacks, so
            # the inverse-variance weights come out at the frame-count ratio.
            a = _star_field(seed=1, noise=5.0 * np.sqrt(3.0))
            b = _star_field(seed=2, noise=5.0)
            p = os.path.join(td, 'prev.fits')
            _write_stack(p, b, nframes=30)
            merged, info = merge_previous_stacks(a, 10, [p])
            expected = (10.0 * a.astype(np.float64)
                        + 30.0 * b.astype(np.float64)) / 40.0
            # interior only: warp of a zero-offset transform is near-exact
            m = 8
            diff = np.abs(merged[m:-m, m:-m].astype(np.float64)
                          - expected[m:-m, m:-m])
            self.assertLess(float(diff.max()), 3.0)
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

    def test_previous_stack_is_mapped_onto_the_current_flux_scale(self):
        """A stack captured at another exposure/gain must not be averaged in
        raw ADU: 2.5x the signal plus a different sky pedestal lands back on
        the current stack's scale."""
        with tempfile.TemporaryDirectory() as td:
            new = _star_field(seed=1, noise=5.0)
            base = _star_field(seed=2, noise=0.0)
            prev = (base - 1000.0) * 2.5 + 400.0
            prev = prev + np.random.default_rng(7).normal(0, 12.5, prev.shape[:2])[:, :, None]
            p = os.path.join(td, 'prev.fits')
            _write_stack(p, prev.astype(np.float32), nframes=30)
            merged, info = merge_previous_stacks(new, 30, [p])
            m = 8
            a = merged[m:-m, m:-m, 1].astype(np.float64).ravel()
            b = new[m:-m, m:-m, 1].astype(np.float64).ravel()
            slope = np.polyfit(b, a, 1)[0]
            self.assertAlmostEqual(slope, 1.0, delta=0.03)
            self.assertAlmostEqual(float(np.median(a)), float(np.median(b)), delta=3.0)
            self.assertEqual(info['weighting'], 'inverse-variance')

    def test_same_settings_stack_is_not_rescaled(self):
        with tempfile.TemporaryDirectory() as td:
            new = _star_field(seed=1)
            p = os.path.join(td, 'prev.fits')
            _write_stack(p, _star_field(seed=2), nframes=10)
            from src.merge import _match_flux_scale
            gains, offsets, _ = _match_flux_scale(
                new, _star_field(seed=2), np.ones(new.shape[:2], dtype=bool))
            np.testing.assert_array_equal(gains, np.ones(3))
            np.testing.assert_allclose(offsets, 0.0, atol=1.0)

    def test_weights_follow_measured_noise_not_frame_count(self):
        """Equal frame counts but 3x the noise in one stack: the merge must
        beat a plain mean and approach the inverse-variance optimum."""
        with tempfile.TemporaryDirectory() as td:
            truth = _star_field(seed=1, noise=0.0)
            good = _star_field(seed=1, noise=5.0) - truth
            noisy = _star_field(seed=1, noise=15.0) - truth
            rng = np.random.default_rng(3)
            good = rng.normal(0, 5.0, truth.shape[:2])[:, :, None] + truth
            noisy = rng.normal(0, 15.0, truth.shape[:2])[:, :, None] + truth
            p = os.path.join(td, 'prev.fits')
            _write_stack(p, noisy.astype(np.float32), nframes=10)
            merged, _ = merge_previous_stacks(good.astype(np.float32), 10, [p])
            m = 8
            err = (merged - truth)[m:-m, m:-m, 1].std()
            plain = np.sqrt(5.0 ** 2 + 15.0 ** 2) / 2.0      # 7.9
            optimum = 1.0 / np.sqrt(1 / 25.0 + 1 / 225.0)    # 4.74
            self.assertLess(err, plain - 1.5)
            self.assertLess(err, optimum + 0.8)

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


class TestReadMergeMeta(unittest.TestCase):
    def test_matches_load_merge_stack_without_touching_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'lin.fits')
            _write_stack(p, _star_field(), nframes=42, intgtime=420.0)
            _, full = load_merge_stack(p)
            self.assertEqual(read_merge_meta(p), full)

    def test_does_not_validate_rawstack(self):
        """Choosing a reference by depth must not fail on a stack that
        load_merge_stack would refuse -- that check runs on the chosen one."""
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'proc.fits')
            _write_stack(p, _star_field(), nframes=10, rawstack=False)
            self.assertEqual(read_merge_meta(p)['nframes'], 10)


class TestHierarchicalHeaderChains(unittest.TestCase):
    """A hierarchical combine builds its header from scratch.

    ``apply_merge_header`` was written for ``--merge``, where the header already
    holds this session's totals and it only adds previous stacks -- so it guards
    every aggregate with "already present". On a fresh header every guard
    skipped, silently: the combined output lost its integration time and dates,
    and, never marked RAWSTACK, could not be fed back into ``--merge`` or
    ``--transient-detect``.
    """

    REF = {'nframes': 100, 'intgtime': 1000.0, 'totexp': 1000.0,
           'datefrst': '2026-06-01T21:00:00', 'datelast': '2026-06-01T23:00:00'}
    INFO = {'n_sources': 1, 'sources': ['other.fits'], 'total_frames': 160,
            'total_intgtime': 600.0, 'total_totexp': 600.0,
            'datefrst': '2026-05-20T20:00:00', 'datelast': '2026-05-20T21:00:00'}

    def _combined_header(self):
        hdr = fits.Header()
        seed_reference_header(hdr, self.REF)
        apply_merge_header(hdr, self.INFO)
        return hdr

    def test_totals_include_the_reference_and_the_others(self):
        hdr = self._combined_header()
        self.assertEqual(hdr['NFRAMES'], 160)
        self.assertAlmostEqual(hdr['INTGTIME'], 1600.0)
        self.assertAlmostEqual(hdr['INTGMIN'], 1600.0 / 60.0)
        self.assertAlmostEqual(hdr['TOTEXP'], 1600.0)

    def test_dates_span_every_stack(self):
        hdr = self._combined_header()
        self.assertEqual(hdr['DATEFRST'], '2026-05-20T20:00:00')
        self.assertEqual(hdr['DATELAST'], '2026-06-01T23:00:00')

    def test_the_result_is_marked_a_linear_stack(self):
        self.assertTrue(self._combined_header()['RAWSTACK'])

    def test_without_seeding_the_aggregates_are_silently_lost(self):
        """The bug, reproduced: apply_merge_header alone on a fresh header."""
        hdr = fits.Header()
        apply_merge_header(hdr, self.INFO)
        self.assertNotIn('INTGTIME', hdr)
        self.assertNotIn('RAWSTACK', hdr)

    def test_the_combined_output_can_be_merged_again(self):
        """The property that matters: write it, and load_merge_stack accepts it
        with the right totals."""
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'combined.fits')
            hdu = fits.PrimaryHDU(data=np.transpose(_star_field(), (2, 0, 1)))
            seed_reference_header(hdu.header, self.REF)
            apply_merge_header(hdu.header, self.INFO)
            hdu.writeto(p)

            _, meta = load_merge_stack(p)          # raises without RAWSTACK

            self.assertEqual(meta['nframes'], 160)
            self.assertAlmostEqual(meta['intgtime'], 1600.0)

    def test_a_reference_with_no_totals_leaves_them_absent_not_zero(self):
        hdr = fits.Header()
        seed_reference_header(hdr, {'nframes': 0, 'intgtime': 0.0, 'totexp': 0.0,
                                    'datefrst': None, 'datelast': None})
        self.assertNotIn('INTGTIME', hdr)
        self.assertNotIn('DATEFRST', hdr)
        self.assertTrue(hdr['RAWSTACK'])


if __name__ == '__main__':
    unittest.main()


class TestFootprintRimIsTrimmed(unittest.TestCase):

    def _merge_with_bright_rim(self, trim):
        import src.merge as m
        old = m._FOOTPRINT_TRIM_PX
        m._FOOTPRINT_TRIM_PX = trim
        try:
            with tempfile.TemporaryDirectory() as td:
                new = _star_field(seed=3, noise=1.0)
                base = _star_field(shift=(-6.0, 0.0), seed=4, noise=1.0)   # sits 6 px lower
                prev = base.copy()
                prev[:2, :, :] += 500.0            # the rim of the previous stack
                p = os.path.join(td, 'prev.fits')
                _write_stack(p, prev, nframes=10)
                merged, _ = m.merge_previous_stacks(new, 10, [p])
            return merged
        finally:
            m._FOOTPRINT_TRIM_PX = old

    def test_bright_rim_of_a_warped_stack_does_not_leak(self):
        # rim lands on rows ~6-7 of the new grid; sky there is ~1000
        leak_off = float(np.median(self._merge_with_bright_rim(0)[6:8, 40:280, 1])) - 1000.0
        leak_on = float(np.median(self._merge_with_bright_rim(3)[6:8, 40:280, 1])) - 1000.0
        self.assertGreater(leak_off, 100.0)      # the artifact is real without the trim
        self.assertLess(abs(leak_on), 10.0)      # and gone with it
