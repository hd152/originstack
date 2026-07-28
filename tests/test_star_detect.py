"""Tests for src/star_detect.py -- matched-filter star detection.

See the module docstring for the validation history: this went through
several rounds against SEP/DAOStarFinder on real archive data before
landing at these defaults. These tests cover correctness properties
(output format, no-detection-on-noise, roundness filtering, dtype) rather
than re-deriving the F1/precision numbers, which were validated
interactively against real FITS data outside the test suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.star_detect import (
    detect_stars_matched_filter, _detect_stars_matched_filter_numpy,
    _bilinear_upsample, _SOURCES_DTYPE,
)


def _synthetic_starfield(h=300, w=400, n_stars=20, seed=0, sky=1000.0, sky_std=15.0):
    rng = np.random.default_rng(seed)
    img = rng.normal(sky, sky_std, (h, w)).astype(np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    stars = []
    for _ in range(n_stars):
        cy = rng.uniform(30, h - 30)
        cx = rng.uniform(30, w - 30)
        amp = rng.uniform(500, 5000)
        sigma = rng.uniform(1.8, 2.5)
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        stars.append((cx, cy))
    return img, stars


class TestDetectStarsMatchedFilter:
    def test_returns_sources_dtype(self):
        img, _ = _synthetic_starfield()
        out = detect_stars_matched_filter(img)
        assert out.dtype == _SOURCES_DTYPE

    def test_finds_synthetic_stars(self):
        img, stars = _synthetic_starfield(n_stars=15, seed=2)
        out = detect_stars_matched_filter(img)
        assert len(out) > 0
        from scipy.spatial import cKDTree
        tree = cKDTree(np.array(stars))
        pts = np.column_stack([out['xcentroid'], out['ycentroid']])
        dist, _ = tree.query(pts, k=1)
        # every detection should land near an injected star
        assert np.mean(dist < 2.0) > 0.8

    def test_no_detections_on_pure_noise(self):
        rng = np.random.default_rng(3)
        img = rng.normal(1000.0, 15.0, (200, 250))
        out = detect_stars_matched_filter(img)
        assert len(out) == 0

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            detect_stars_matched_filter(np.zeros((10, 10, 3)))

    def test_centroid_accuracy(self):
        h, w = 200, 200
        cy, cx = 100.3, 80.7
        yy, xx = np.mgrid[0:h, 0:w]
        rng = np.random.default_rng(4)
        img = rng.normal(1000.0, 5.0, (h, w))
        img += 8000.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.2 ** 2))
        out = detect_stars_matched_filter(img, fwhm=5.5, k_confirm=10.0)
        assert len(out) == 1
        assert abs(out['xcentroid'][0] - cx) < 0.3
        assert abs(out['ycentroid'][0] - cy) < 0.3

    def test_roundness_filter_rejects_elongated_source(self):
        # An elongated (trail-like) bright feature should fail the roundness
        # gate even though its peak SNR clears threshold easily.
        h, w = 200, 200
        rng = np.random.default_rng(5)
        img = rng.normal(1000.0, 5.0, (h, w))
        img[95:105, 40:160] += 6000.0  # a bright horizontal streak
        out = detect_stars_matched_filter(img, fwhm=5.5, k_confirm=10.0, roundness_max=0.3)
        # a thin streak has roundness near 1.0 -- must not pass a strict gate
        assert len(out) == 0


class TestNativeNumpyParity:
    def test_native_matches_numpy_when_available(self):
        from src.star_detect import HAS_NATIVE
        if not HAS_NATIVE:
            pytest.skip("astro_native not built")
        img, _ = _synthetic_starfield(n_stars=15, seed=6)
        native_out = detect_stars_matched_filter(img)
        numpy_out = _detect_stars_matched_filter_numpy(img, 5.5, 22.0, 64, 0.5, 2)
        assert len(native_out) == len(numpy_out)
        if len(native_out):
            a = np.sort(native_out, order='xcentroid')
            b = np.sort(numpy_out, order='xcentroid')
            np.testing.assert_allclose(a['xcentroid'], b['xcentroid'], atol=1e-5)
            np.testing.assert_allclose(a['ycentroid'], b['ycentroid'], atol=1e-5)


class TestBilinearUpsample:
    def test_constant_grid_upsamples_to_constant(self):
        grid = np.full((4, 5), 3.0)
        out = _bilinear_upsample(grid, 40, 50, cell=10)
        assert out.shape == (40, 50)
        np.testing.assert_allclose(out, 3.0)

    def test_no_edge_blowup(self):
        # Regression check for the real bug this replaced (scipy.ndimage.zoom
        # producing anomalous values right at the image border).
        rng = np.random.default_rng(7)
        grid = rng.normal(100.0, 5.0, (6, 8))
        out = _bilinear_upsample(grid, 64, 80, cell=10)
        interior = out[10:-10, 10:-10]
        border = np.concatenate([out[0, :], out[-1, :], out[:, 0], out[:, -1]])
        # border values must stay within the same order of magnitude as the
        # interior -- not a spurious spike/collapse from edge extrapolation
        assert border.max() < interior.max() + 3 * grid.std()
        assert border.min() > interior.min() - 3 * grid.std()


class TestDetectStarsAutoDispatcher:
    """The dispatcher in src/quality.py that every detection call site
    (Phase 1 quality analysis, registration reference/residual detection,
    --merge, post-process star masking) routes through -- see
    configure_star_detector's docstring for why a single dispatch point
    matters here (mixing backends across stages would silently corrupt the
    affine-matching / residual-RMS machinery that compares catalogs from
    different stages against each other)."""

    def teardown_method(self):
        from src.quality import configure_star_detector
        configure_star_detector('matched-filter')  # don't leak state across tests

    def test_rejects_unknown_method(self):
        from src.quality import configure_star_detector
        with pytest.raises(ValueError):
            configure_star_detector('not-a-real-method')
        with pytest.raises(ValueError):
            configure_star_detector('auto')  # removed -- no more SEP->DAO chain

    def test_matched_filter_is_default(self):
        from src.quality import _STAR_DETECTOR_METHOD
        assert _STAR_DETECTOR_METHOD == 'matched-filter'

    def test_configure_changes_global_default(self):
        from src.quality import configure_star_detector
        import src.quality as quality_mod
        configure_star_detector('sep')
        assert quality_mod._STAR_DETECTOR_METHOD == 'sep'

    def test_explicit_method_overrides_global_default(self):
        from src.quality import configure_star_detector, detect_stars_auto
        configure_star_detector('sep')
        img, _ = _synthetic_starfield_for_dispatch()
        # explicit method='matched-filter' should win over the global 'sep' setting
        out = detect_stars_auto(img, noise=15.0, method='matched-filter')
        assert out is not None

    def test_sep_method_falls_through_to_matched_filter(self):
        # 'sep' without the sep package installed (or that finds nothing)
        # must still detect via matched-filter, not silently return None --
        # sep is optional now, never a hard requirement.
        from src.quality import detect_stars_auto
        img, _ = _synthetic_starfield_for_dispatch()
        out = detect_stars_auto(img, noise=15.0, method='sep')
        assert out is not None
        assert len(out) > 0

    def test_matched_filter_dispatch_finds_synthetic_stars(self):
        from src.quality import detect_stars_auto
        img, stars = _synthetic_starfield_for_dispatch()
        out = detect_stars_auto(img, noise=15.0, method='matched-filter')
        assert out is not None
        assert len(out) > 0

    def test_returns_none_on_pure_noise(self):
        from src.quality import detect_stars_auto
        rng = np.random.default_rng(9)
        img = rng.normal(1000.0, 15.0, (200, 250))
        out = detect_stars_auto(img, noise=15.0, method='matched-filter')
        assert out is None


def _synthetic_starfield_for_dispatch(h=300, w=400, n_stars=15, seed=8):
    rng = np.random.default_rng(seed)
    img = rng.normal(1000.0, 15.0, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    stars = []
    for _ in range(n_stars):
        cy = rng.uniform(30, h - 30)
        cx = rng.uniform(30, w - 30)
        amp = rng.uniform(1000, 5000)
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.2 ** 2))
        stars.append((cx, cy))
    return img, stars
