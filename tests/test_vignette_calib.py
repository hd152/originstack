"""Tests for src/vignette_calib.py -- the apply-side of the per-instrument
vignetting calibration map (see tools/build_vignette_map.py for the offline
builder)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.vignette_calib import load_vignette_map, apply_vignette_correction


class TestApplyVignetteCorrection:
    def test_subtracts_matching_shape_map(self):
        rgb = np.full((20, 30, 3), 100.0, dtype=np.float32)
        vmap = np.full((20, 30, 3), 5.0, dtype=np.float32)
        out = apply_vignette_correction(rgb, vmap)
        assert np.allclose(out, 95.0)

    def test_output_dtype_is_float32(self):
        rgb = np.full((20, 30, 3), 100.0, dtype=np.float32)
        vmap = np.zeros((20, 30, 3), dtype=np.float32)
        out = apply_vignette_correction(rgb, vmap)
        assert out.dtype == np.float32

    def test_resizes_mismatched_map(self):
        rgb = np.full((64, 96, 3), 100.0, dtype=np.float32)
        vmap = np.full((8, 12, 3), 5.0, dtype=np.float32)   # coarse, needs upsampling
        out = apply_vignette_correction(rgb, vmap)
        assert out.shape == rgb.shape
        # smooth constant map should resize to ~constant too
        assert abs(float(out.mean()) - 95.0) < 0.5

    def test_does_not_mutate_input(self):
        rgb = np.full((10, 10, 3), 100.0, dtype=np.float32)
        vmap = np.full((10, 10, 3), 5.0, dtype=np.float32)
        rgb_copy = rgb.copy()
        apply_vignette_correction(rgb, vmap)
        assert np.array_equal(rgb, rgb_copy)

    def test_2d_input_passthrough(self):
        rgb = np.zeros((10, 10), dtype=np.float32)
        vmap = np.zeros((10, 10, 3), dtype=np.float32)
        out = apply_vignette_correction(rgb, vmap)
        assert out.shape == rgb.shape

    def test_preserves_spatial_pattern_not_just_pedestal(self):
        rgb = np.zeros((20, 20, 3), dtype=np.float32)
        vmap = np.zeros((20, 20, 3), dtype=np.float32)
        vmap[:10, :, :] = 10.0   # top half brighter vignette estimate
        out = apply_vignette_correction(rgb, vmap)
        assert out[0, 0, 0] < out[19, 0, 0]   # top got subtracted more -> darker


class TestLoadVignetteMap:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_vignette_map(str(tmp_path / "nope.fits")) is None

    def test_empty_path_returns_none(self):
        assert load_vignette_map("") is None
        assert load_vignette_map(None) is None

    def test_roundtrip_chw_to_hwc(self, tmp_path):
        from astropy.io import fits
        data_chw = np.random.default_rng(0).normal(size=(3, 12, 16)).astype(np.float32)
        path = tmp_path / "vmap.fits"
        fits.PrimaryHDU(data_chw).writeto(str(path))
        loaded = load_vignette_map(str(path))
        assert loaded.shape == (12, 16, 3)
        assert np.allclose(loaded[..., 0], data_chw[0])
        assert np.allclose(loaded[..., 2], data_chw[2])

    def test_wrong_shape_returns_none(self, tmp_path):
        from astropy.io import fits
        path = tmp_path / "bad.fits"
        fits.PrimaryHDU(np.zeros((10, 10), dtype=np.float32)).writeto(str(path))
        assert load_vignette_map(str(path)) is None


class TestBuildMastersVignetteWiring:
    def test_build_masters_sets_none_without_flag(self):
        from src.cli import _build_masters
        masters = _build_masters({'light': [], 'dark': [], 'flat': [], 'bias': []})
        assert masters['vignette'] is None

    def test_build_masters_loads_map_from_args(self, tmp_path):
        from astropy.io import fits
        from types import SimpleNamespace
        from src.cli import _build_masters
        path = tmp_path / "vmap.fits"
        fits.PrimaryHDU(np.zeros((3, 8, 8), dtype=np.float32)).writeto(str(path))
        args = SimpleNamespace(vignette_map=str(path))
        masters = _build_masters({'light': [], 'dark': [], 'flat': [], 'bias': []}, args=args)
        assert masters['vignette'] is not None
        assert masters['vignette'].shape == (8, 8, 3)

    def test_build_masters_missing_map_path_stays_none(self, tmp_path):
        from types import SimpleNamespace
        from src.cli import _build_masters
        args = SimpleNamespace(vignette_map=str(tmp_path / "missing.fits"))
        masters = _build_masters({'light': [], 'dark': [], 'flat': [], 'bias': []}, args=args)
        assert masters['vignette'] is None
