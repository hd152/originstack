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
            with mock.patch('psutil.virtual_memory', return_value=fake_mem):
                self.assertIsNone(robust_pca_master(frames, shape))


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
