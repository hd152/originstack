"""Tests for robust-PCA (Principal Component Pursuit) master calibration frames.

robust_pca_decompose splits a stack matrix into a low-rank component (the true
shared pattern) plus a sparse component (outliers) -- these tests check the
decomposition recovers a known synthetic low-rank+sparse matrix, that
robust_pca_master produces a finite correctly-shaped master from a synthetic
calibration stack, and that make_master(method='robust_pca') falls back to
median gracefully below the minimum frame count.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from src.models import Config, FrameInfo
from src.robust_pca import robust_pca_decompose, robust_pca_master


def _write_fits(path: str, data: np.ndarray) -> None:
    fits.writeto(path, data.astype(np.float32), overwrite=True)


class TestRobustPcaDecompose(unittest.TestCase):
    """Algorithmic correctness against a known synthetic low-rank + sparse matrix."""

    def test_recovers_rank1_plus_sparse(self):
        rng = np.random.default_rng(0)
        n, p = 12, 400
        # Rank-1 "shared pattern" component (e.g. a single flat-field shape
        # scaled per frame), well-conditioned magnitude.
        u = rng.uniform(0.5, 1.5, n)
        v = rng.uniform(0.5, 1.5, p)
        L_true = np.outer(u, v)

        # Sparse large-magnitude outliers (~2% of entries).
        S_true = np.zeros((n, p))
        n_outliers = int(0.02 * n * p)
        rows = rng.integers(0, n, n_outliers)
        cols = rng.integers(0, p, n_outliers)
        S_true[rows, cols] = rng.uniform(5.0, 10.0, n_outliers) * L_true.mean()

        D = L_true + S_true
        L, S = robust_pca_decompose(D, max_iters=100, tol=1e-8)

        rel_err = np.linalg.norm(L - L_true, 'fro') / np.linalg.norm(L_true, 'fro')
        self.assertLess(rel_err, 0.05)

        # Sparse component should be (near-)zero away from injected outliers.
        clean_mask = (S_true == 0)
        self.assertLess(float(np.abs(S[clean_mask]).mean()),
                         float(np.abs(L_true).mean()) * 0.05)

    def test_zero_matrix_returns_zero(self):
        D = np.zeros((5, 20))
        L, S = robust_pca_decompose(D)
        np.testing.assert_allclose(L, 0.0)
        np.testing.assert_allclose(S, 0.0)


class TestRobustPcaMaster(unittest.TestCase):

    def _write_frame(self, tmpdir: str, name: str, data: np.ndarray) -> FrameInfo:
        path = os.path.join(tmpdir, name)
        _write_fits(path, data)
        return FrameInfo(path=path, type='flat', header={})

    def test_master_shape_and_finite(self):
        rng = np.random.default_rng(1)
        shape = (24, 24)
        # Common vignetting-like pattern shared by every frame.
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        pattern = 1000.0 - 0.3 * ((yy - 12) ** 2 + (xx - 12) ** 2)

        with tempfile.TemporaryDirectory() as d:
            frames = []
            for i in range(8):
                frame = pattern + rng.normal(0, 2.0, shape)
                if i == 3:
                    # One frame has a localized dust-donut-like anomaly.
                    frame[5:9, 5:9] -= 300.0
                frames.append(self._write_frame(d, f'f{i}.fits', frame.astype(np.float32)))

            master = robust_pca_master(frames, shape)
            self.assertIsNotNone(master)
            self.assertEqual(master.shape, shape)
            self.assertTrue(np.all(np.isfinite(master)))
            # Recovered master should be reasonably close to the shared pattern.
            rel_err = np.abs(master - pattern).mean() / np.abs(pattern).mean()
            self.assertLess(rel_err, 0.1)

    def test_returns_none_below_min_frames(self):
        shape = (16, 16)
        with tempfile.TemporaryDirectory() as d:
            frames = [self._write_frame(d, f'f{i}.fits',
                                        np.full(shape, 100.0, dtype=np.float32))
                     for i in range(Config.ROBUST_PCA_MIN_FRAMES - 1)]
            self.assertIsNone(robust_pca_master(frames, shape))

    def test_skips_shape_mismatched_frame_instead_of_crashing(self):
        # A mixed-binning/ROI calibration set can reach robust_pca_master with
        # one frame whose shape doesn't match the rest (the caller's
        # homogeneity fast-path doesn't check dimensions) -- must not crash
        # np.stack, just skip the offending frame.
        shape = (24, 24)
        with tempfile.TemporaryDirectory() as d:
            frames = [self._write_frame(d, f'f{i}.fits',
                                        np.full(shape, float(100 + i), dtype=np.float32))
                     for i in range(Config.ROBUST_PCA_MIN_FRAMES + 1)]
            frames.append(self._write_frame(d, 'bad_shape.fits',
                                             np.full((12, 12), 999.0, dtype=np.float32)))
            master = robust_pca_master(frames, shape)
            self.assertIsNotNone(master)
            self.assertEqual(master.shape, shape)

    def test_falls_back_on_non_finite_values(self):
        shape = (24, 24)
        with tempfile.TemporaryDirectory() as d:
            frames = []
            for i in range(Config.ROBUST_PCA_MIN_FRAMES + 1):
                data = np.full(shape, float(100 + i), dtype=np.float32)
                if i == 0:
                    data[0, 0] = np.nan
                frames.append(self._write_frame(d, f'f{i}.fits', data))
            self.assertIsNone(robust_pca_master(frames, shape))

    def test_falls_back_when_memory_insufficient(self):
        from unittest import mock
        shape = (24, 24)
        with tempfile.TemporaryDirectory() as d:
            frames = [self._write_frame(d, f'f{i}.fits',
                                        np.full(shape, float(100 + i), dtype=np.float32))
                     for i in range(Config.ROBUST_PCA_MIN_FRAMES + 1)]
            fake_mem = mock.MagicMock()
            fake_mem.available = 1  # forces the memory guard to trip
            # create=True: mock.patch normally requires the target attribute
            # to already exist. GitHub Actions' ubuntu-latest runners have a
            # psutil install that's missing virtual_memory (a real, normally
            # always-present psutil API) -- without create=True this test
            # fails there at the patch itself, before robust_pca_master's
            # own except-Exception fallback (the thing under test) ever runs.
            with mock.patch('psutil.virtual_memory', return_value=fake_mem, create=True):
                self.assertIsNone(robust_pca_master(frames, shape))


class TestBayerBlockDownsampleUpsample(unittest.TestCase):
    """CFA-respecting down/upsample used by robust_pca_master's downsample=
    parameter (only --flat-from-lights passes it -- see FLAT_FROM_LIGHTS_
    DOWNSAMPLE's Config docstring for why real dark/bias/flat masters never do)."""

    def test_does_not_mix_bayer_channels(self):
        # Four constant Bayer sub-planes at very different levels -- a
        # cross-channel-mixing bug (averaging the raw mosaic directly instead
        # of each sub-plane independently) would blend these together.
        from src.robust_pca import bayer_block_downsample
        img = np.zeros((16, 16))
        img[0::2, 0::2] = 100.0   # R
        img[0::2, 1::2] = 500.0   # G1
        img[1::2, 0::2] = 500.0   # G2
        img[1::2, 1::2] = 900.0   # B
        small = bayer_block_downsample(img, 2)
        self.assertTrue(np.all(small[0::2, 0::2] == 100.0))
        self.assertTrue(np.all(small[0::2, 1::2] == 500.0))
        self.assertTrue(np.all(small[1::2, 0::2] == 500.0))
        self.assertTrue(np.all(small[1::2, 1::2] == 900.0))

    def test_downsample_shrinks_by_the_requested_factor(self):
        from src.robust_pca import bayer_block_downsample
        img = np.zeros((64, 96))
        small = bayer_block_downsample(img, 4)
        # each sub-plane is (32,48) -> downsampled by 4 -> (8,12) -> reinterleaved (16,24)
        self.assertEqual(small.shape, (16, 24))

    def test_roundtrip_preserves_a_smooth_pattern(self):
        # The point of the downsample is to survive exactly this kind of
        # content (smooth vignetting), not preserve it exactly.
        from src.robust_pca import bayer_block_downsample, bayer_block_upsample
        H, W = 200, 300
        yy, xx = np.mgrid[0:H, 0:W]
        pattern = 1000.0 - 0.01 * ((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
        small = bayer_block_downsample(pattern, 4)
        back = bayer_block_upsample(small, (H, W))
        self.assertEqual(back.shape, (H, W))
        rel_err = np.abs(back - pattern).mean() / np.abs(pattern).mean()
        self.assertLess(rel_err, 0.02)

    def test_downsample_is_a_noop_at_factor_one(self):
        from src.robust_pca import bayer_block_downsample
        img = np.random.default_rng(0).normal(size=(20, 20))
        np.testing.assert_array_equal(bayer_block_downsample(img, 1), img)


class TestRobustPcaMasterDownsample(unittest.TestCase):
    """End-to-end: robust_pca_master(downsample=N) still recovers a real
    vignetting pattern at full output resolution, not just a smaller one."""

    def _write_frame(self, tmpdir: str, name: str, data: np.ndarray) -> FrameInfo:
        path = os.path.join(tmpdir, name)
        _write_fits(path, data)
        return FrameInfo(path=path, type='light', header={})

    def test_recovers_vignette_at_full_resolution(self):
        rng = np.random.default_rng(2)
        shape = (200, 300)  # even dims, real-mosaic-shaped
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        # A smooth vignetting-like pattern, per-Bayer-position gain baked in
        # (R/G1/G2/B different absolute levels) -- a channel-mixing bug in
        # the downsample would show up as a wrong recovered pattern here,
        # not just a blurrier one.
        base = 1000.0 - 0.02 * ((yy - 100) ** 2 + (xx - 150) ** 2)
        gain = np.ones(shape)
        gain[0::2, 0::2] = 1.0
        gain[0::2, 1::2] = 1.9
        gain[1::2, 0::2] = 1.9
        gain[1::2, 1::2] = 1.6
        pattern = base * gain

        with tempfile.TemporaryDirectory() as d:
            frames = []
            for i in range(10):
                frame = pattern + rng.normal(0, 3.0, shape)
                frames.append(self._write_frame(d, f'f{i}.fits', frame.astype(np.float32)))

            master = robust_pca_master(frames, shape, downsample=Config.FLAT_FROM_LIGHTS_DOWNSAMPLE)
            self.assertIsNotNone(master)
            self.assertEqual(master.shape, shape)  # upsampled back to full res
            self.assertTrue(np.all(np.isfinite(master)))
            rel_err = np.abs(master - pattern).mean() / np.abs(pattern).mean()
            self.assertLess(rel_err, 0.15)  # looser than the full-res test -- it's lossy by design

    def test_downsample_one_matches_undownsampled_call(self):
        # downsample=1 must be exactly the pre-existing code path (no
        # upsample step, no behavior change for real bias/dark/flat masters
        # that never pass downsample at all).
        rng = np.random.default_rng(3)
        shape = (24, 24)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        pattern = 1000.0 - 0.3 * ((yy - 12) ** 2 + (xx - 12) ** 2)
        with tempfile.TemporaryDirectory() as d:
            frames = []
            for i in range(8):
                frame = pattern + rng.normal(0, 2.0, shape)
                frames.append(self._write_frame(d, f'f{i}.fits', frame.astype(np.float32)))
            master_plain = robust_pca_master(frames, shape)
            master_ds1 = robust_pca_master(frames, shape, downsample=1)
            np.testing.assert_array_equal(master_plain, master_ds1)


class TestMakeMasterRobustPcaFallback(unittest.TestCase):
    """make_master(method='robust_pca') dispatch and graceful fallback."""

    def _write_frame(self, tmpdir: str, name: str, data: np.ndarray) -> FrameInfo:
        path = os.path.join(tmpdir, name)
        _write_fits(path, data)
        return FrameInfo(path=path, type='dark', header={})

    def test_falls_back_to_median_below_min_frames(self):
        from src.io_fits import make_master
        with tempfile.TemporaryDirectory() as d:
            frames = [self._write_frame(d, f'f{i}.fits',
                                        np.full((16, 16), float(100 + i), dtype=np.float32))
                     for i in range(3)]
            rpca_master = make_master(frames, method='robust_pca')
            median_master = make_master(frames, method='median')
            self.assertIsNotNone(rpca_master)
            np.testing.assert_allclose(rpca_master, median_master, atol=1e-3)

    def test_returns_correct_shape_at_min_frames(self):
        from src.io_fits import make_master
        shape = (16, 16)
        with tempfile.TemporaryDirectory() as d:
            frames = [self._write_frame(d, f'f{i}.fits',
                                        np.full(shape, float(100 + i), dtype=np.float32))
                     for i in range(Config.ROBUST_PCA_MIN_FRAMES + 2)]
            master = make_master(frames, method='robust_pca')
            self.assertIsNotNone(master)
            self.assertEqual(master.shape, shape)


if __name__ == '__main__':
    unittest.main()
