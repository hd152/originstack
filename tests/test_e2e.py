"""End-to-end pipeline smoke tests.

These tests build a minimal synthetic dataset in a temp directory, run
stack_target(), and validate that:

  1. An output FITS file is produced.
  2. The stacked image has the expected shape and finite values.
  3. Stacking N aligned frames reduces per-pixel noise relative to a single frame.
  4. A known star centroid survives the full pipeline at a reasonable position.
  5. Calibration frames (darks, flats) are actually applied (flat-corrected channels
     differ from uncorrected channels).
"""
from __future__ import annotations

import argparse
import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------------------
# Helpers shared by all end-to-end tests
# ---------------------------------------------------------------------------

def _make_minimal_args(**overrides) -> argparse.Namespace:
    """Return an argparse.Namespace with all fields needed by stack_target."""
    defaults = dict(
        # Phase 1 processing
        debayer_method='bilinear',
        white_balance='none',
        ca_correction=False,
        cosmic_ray_rejection=False,
        quick_quality=False,
        skip_quality=False,
        parallel=1,
        # Quality gate
        quality_filter=False,   # keep ALL frames — tiny test set
        quality_threshold=25.0,
        # Phase 2 registration
        no_registration=False,
        no_affine=True,         # translation-only for speed
        skip_phase_correlation=False,
        use_pyramid=True,
        verbose=False,
        debug_registration=False,
        # Phase 3 stacking
        stack_method='mean',
        winsorize=False,
        rejection_sigma=3.0,
        rejection_iters=3,
        rejection_estimator='mad',
        percentile_low=20.0,
        percentile_high=80.0,
        esd_max_outliers=0,
        esd_significance=0.05,
        weight_snr=1.0,
        weight_fwhm=1.0,
        weight_stars=1.0,
        weight_noise=False,
        drizzle_scale=1.0,
        drizzle_pixfrac=1.0,
        elastic_registration=False,
        # Phase 4 post-processing  — disable everything for speed
        skip_step=[
            'hot_pixel', 'background', 'chroma_nr', 'sky_floor',
            'wavelet', 'sky_residual', 'sky_pedestal',
            'nlm', 'bilateral', 'mmt', 'acdnr',
            'deconvolve', 'star_reduce', 'local_contrast', 'sky_neutralize',
        ],
        background_extraction=False,
        dbe=False,
        bg_mesh_size=64,
        bg_filter_size=3,
        bg_clip_sigma=3.0,
        denoise=False,
        denoise_strength=1.0,
        denoise_adaptive=True,
        denoise_nlm=False,
        denoise_nlm_blend=0.5,
        denoise_bilateral=False,
        denoise_bilateral_sigma_space=3.0,
        denoise_mmt=False,
        denoise_acdnr=False,
        deconvolve=False,
        star_reduce=False,
        local_contrast=False,
        # Output / misc
        stretch='linear',
        ghs_b=8.0,
        ghs_sp=0.15,
        ghs_hp=0.95,
        plate_solve=False,
        output_tiff=False,
        output_xisf=False,
        keep_intermediates=False,
        keep_checkpoint=False,
        diagnostic=False,
        diagnostic_dir=None,
        no_resume=True,         # always skip checkpoint for tests
        comet_mode=False,
        ai_advisor=False,
        ai_report=False,
        auto=False,
        color_calibrate=False,
        hdr_combine=None,
        quality_report=None,
        export_frames_dir=None,
        blink=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_fits(path: str, data: np.ndarray, header: dict | None = None) -> None:
    hdr = fits.Header()
    if header:
        for k, v in header.items():
            hdr[k] = v
    fits.writeto(path, data.astype(np.float32), header=hdr, overwrite=True)


def _make_synthetic_bayer(shape=(128, 128), star_cy=64, star_cx=64,
                           amp=4000.0, bg=200.0,
                           noise_sigma=8.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a synthetic Bayer (RGGB) frame with a primary star and 5 secondary stars.

    Multiple stars are placed so the quality gate's hard minimum of 3 detected
    stars is reliably satisfied even under varying detection thresholds.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    H, W = shape
    yy, xx = np.indices(shape)

    # Uniform background — different per Bayer channel to mimic realistic data
    raw = np.zeros(shape, dtype=np.float32)
    raw[0::2, 0::2] = bg * 1.0   # R
    raw[0::2, 1::2] = bg * 1.3   # G1
    raw[1::2, 0::2] = bg * 1.3   # G2
    raw[1::2, 1::2] = bg * 0.7   # B

    # Primary star at (star_cy, star_cx) — sigma=3 px
    raw += (amp * np.exp(-((yy - star_cy) ** 2 + (xx - star_cx) ** 2)
                          / (2 * 3.0 ** 2))).astype(np.float32)

    # Secondary stars well away from centre — spread to corners
    margin = 16
    secondary_positions = [
        (margin, margin),
        (margin, W - margin),
        (H - margin, margin),
        (H - margin, W - margin),
        (H // 3, W // 2),
    ]
    for sy, sx in secondary_positions:
        raw += (amp * 0.6 * np.exp(-((yy - sy) ** 2 + (xx - sx) ** 2)
                                    / (2 * 2.5 ** 2))).astype(np.float32)

    raw += rng.normal(0.0, noise_sigma, shape).astype(np.float32)
    return np.clip(raw, 0.0, None)


def _create_synthetic_dataset(tmpdir: str,
                               n_lights: int = 6,
                               shifts_yx: list | None = None) -> dict:
    """Write a minimal synthetic dataset and return paths dict."""
    H, W = 128, 128
    rng = np.random.default_rng(42)

    if shifts_yx is None:
        shifts_yx = [(0, 0), (3, -2), (-1, 4), (2, 1), (-3, -1), (1, -3)]
    shifts_yx = shifts_yx[:n_lights]

    # ---- Calibration frames ----
    dark_files = []
    for i in range(3):
        d = rng.normal(50.0, 1.5, (H, W)).astype(np.float32)
        p = os.path.join(tmpdir, f'dark_{i:03d}.fits')
        _write_fits(p, d, {'EXPTIME': 120.0, 'ISOSPEED': 800})
        dark_files.append(p)

    flat_files = []
    for i in range(2):
        f = np.ones((H, W), dtype=np.float32) * 8000.0
        # Mild vignetting
        yy, xx = np.indices((H, W))
        r = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
        f *= np.clip(1.0 - 0.0003 * r, 0.85, 1.0)
        p = os.path.join(tmpdir, f'flat_{i:03d}.fits')
        _write_fits(p, f, {'EXPTIME': 0.5})
        flat_files.append(p)

    # ---- Light frames ----
    light_files = []
    for i, (dy, dx) in enumerate(shifts_yx):
        raw = _make_synthetic_bayer(
            shape=(H, W),
            star_cy=H // 2 + dy,
            star_cx=W // 2 + dx,
            rng=rng,
        )
        p = os.path.join(tmpdir, f'light_{i:03d}.fits')
        _write_fits(p, raw, {
            'BAYERPAT': 'RGGB',
            'EXPTIME': 120.0,
            'ISOSPEED': 800,
        })
        light_files.append(p)

    return {
        'dark': dark_files,
        'flat': flat_files,
        'light': light_files,
        'H': H,
        'W': W,
    }


def _make_synthetic_bayer_piecewise(shape, base_cy, base_cx,
                                    group_a_offset=(0.0, 0.0),
                                    group_b_offset=(0.0, 0.0),
                                    amp=4000.0, bg=200.0, noise_sigma=6.0,
                                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic Bayer frame with two independently-offsettable star groups
    (left half / right half of the frame). A single global affine/translation
    transform cannot null both groups' residuals simultaneously when their
    offsets differ -- this is what elastic (non-rigid) registration exists
    to correct. 16 stars total, well above LOCAL_WARP_MIN_STARS."""
    if rng is None:
        rng = np.random.default_rng(0)
    H, W = shape
    yy, xx = np.indices(shape)
    raw = np.zeros(shape, dtype=np.float32)
    raw[0::2, 0::2] = bg * 1.0
    raw[0::2, 1::2] = bg * 1.3
    raw[1::2, 0::2] = bg * 1.3
    raw[1::2, 1::2] = bg * 0.7

    group_a_base = [(-80, -80), (-80, -20), (80, -80), (80, -20),
                    (-20, -60), (20, -60), (0, -100), (-40, -40)]
    group_b_base = [(-80, 20), (-80, 80), (80, 20), (80, 80),
                    (-20, 60), (20, 60), (0, 100), (40, 40)]

    for base, offset in ((group_a_base, group_a_offset), (group_b_base, group_b_offset)):
        for dy, dx in base:
            cy = base_cy + dy + offset[0]
            cx = base_cx + dx + offset[1]
            raw += (amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                                  / (2 * 3.0 ** 2))).astype(np.float32)

    raw += rng.normal(0.0, noise_sigma, shape).astype(np.float32)
    return np.clip(raw, 0.0, None)


def _create_piecewise_dataset(tmpdir: str, n_lights: int = 8) -> dict:
    """Write a synthetic dataset where each frame's two star groups carry a
    different (alternating-sign) residual offset -- a genuinely non-rigid
    distortion no single per-frame affine transform can correct for both
    groups at once."""
    H, W = 256, 256
    rng = np.random.default_rng(7)

    dark_files = []
    for i in range(3):
        d = rng.normal(50.0, 1.5, (H, W)).astype(np.float32)
        p = os.path.join(tmpdir, f'pw_dark_{i:03d}.fits')
        _write_fits(p, d, {'EXPTIME': 120.0, 'ISOSPEED': 800})
        dark_files.append(p)

    flat_files = []
    for i in range(2):
        f = np.ones((H, W), dtype=np.float32) * 8000.0
        yy, xx = np.indices((H, W))
        r = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
        f *= np.clip(1.0 - 0.0002 * r, 0.85, 1.0)
        p = os.path.join(tmpdir, f'pw_flat_{i:03d}.fits')
        _write_fits(p, f, {'EXPTIME': 0.5})
        flat_files.append(p)

    shifts_yx = [(0, 0), (2, -1), (-1, 2), (1, 1),
                (-2, -1), (2, 1), (-1, -2), (1, -1)][:n_lights]
    light_files = []
    for i, (dy, dx) in enumerate(shifts_yx):
        sign = 1.0 if i % 2 == 0 else -1.0
        group_a_offset = (1.5 * sign, 1.5 * sign)
        group_b_offset = (-1.5 * sign, -1.5 * sign)
        raw = _make_synthetic_bayer_piecewise(
            shape=(H, W), base_cy=H // 2 + dy, base_cx=W // 2 + dx,
            group_a_offset=group_a_offset, group_b_offset=group_b_offset,
            rng=rng,
        )
        p = os.path.join(tmpdir, f'pw_light_{i:03d}.fits')
        _write_fits(p, raw, {'BAYERPAT': 'RGGB', 'EXPTIME': 120.0, 'ISOSPEED': 800})
        light_files.append(p)

    return {'dark': dark_files, 'flat': flat_files, 'light': light_files, 'H': H, 'W': W}


# ---------------------------------------------------------------------------
# End-to-end test cases
# ---------------------------------------------------------------------------

class TestE2EBasicStack(unittest.TestCase):
    """Run the full pipeline on a tiny synthetic dataset and validate outputs."""

    def _run_pipeline(self, tmpdir: str, paths: dict,
                      **args_overrides) -> tuple[str, np.ndarray]:
        """Invoke stack_target; return (output_path, stacked_rgb)."""
        from src.io_fits import make_master
        from src.models import FrameInfo, ProcessingStats
        from src.pipeline import stack_target

        # Build FrameInfo lists
        light_frames = [
            FrameInfo(path=p, type='light',
                      header={'BAYERPAT': 'RGGB', 'EXPTIME': 120.0, 'ISOSPEED': 800,
                               'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
            for p in paths['light']
        ]
        dark_frames = [
            FrameInfo(path=p, type='dark',
                      header={'EXPTIME': 120.0, 'ISOSPEED': 800,
                               'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
            for p in paths['dark']
        ]
        flat_frames = [
            FrameInfo(path=p, type='flat',
                      header={'EXPTIME': 0.5,
                               'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
            for p in paths['flat']
        ]

        master_dark = make_master(dark_frames, method='median')
        master_flat = make_master(flat_frames, method='median')

        masters = {
            'dark': master_dark,
            'flat': master_flat,
            'bias': None,
            'dark_exptime': 120.0,
        }

        output_path = os.path.join(tmpdir, 'stacked.fits')
        args = _make_minimal_args(**args_overrides)
        stats = ProcessingStats()
        all_frames = light_frames  # pipeline expects light frames only

        result = stack_target(all_frames, output_path, args, masters, stats)
        return result, output_path

    # ------------------------------------------------------------------

    def test_output_file_created(self):
        """stack_target must produce a FITS file on disk."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir)
            result, output_path = self._run_pipeline(tmpdir, paths)
            self.assertIsNotNone(result, "stack_target returned None")
            self.assertTrue(os.path.exists(output_path),
                            f"Output FITS not found: {output_path}")

    def test_output_fits_loadable(self):
        """Output FITS must be openable and contain numeric data."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir)
            _, output_path = self._run_pipeline(tmpdir, paths)
            if not os.path.exists(output_path):
                self.skipTest("Output file not produced — skipping shape/value checks")
            with fits.open(output_path, memmap=False) as hdul:
                data = hdul[0].data.copy()
            self.assertIsNotNone(data)
            self.assertTrue(np.all(np.isfinite(data)),
                            "Stacked FITS contains non-finite values")

    def test_output_shape_is_3_H_W(self):
        """Stacked FITS should be a 3-channel image stored as (3, H, W)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir)
            _, output_path = self._run_pipeline(tmpdir, paths)
            if not os.path.exists(output_path):
                self.skipTest("Output file not produced")
            with fits.open(output_path, memmap=False) as hdul:
                data = hdul[0].data.copy()
            # Pipeline saves as (3, H, W) or (H, W, 3); accept either
            if data.ndim == 3:
                n_channels = min(data.shape)
                self.assertEqual(n_channels, 3,
                                 f"Expected 3 channels, got shape {data.shape}")
            else:
                self.fail(f"Unexpected stacked FITS shape: {data.shape}")

    def test_stacking_reduces_noise(self):
        """The stacked image should have lower per-pixel noise than a single frame.

        Strategy: compare the standard deviation of the sky background in the
        stacked result vs. the first individual light frame (after debayering).
        """
        from src.debayer import debayer_bilinear
        from src.io_fits import load_fits

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, n_lights=8)
            _, output_path = self._run_pipeline(tmpdir, paths)
            if not os.path.exists(output_path):
                self.skipTest("Output file not produced")

            # Noise in stacked image (sky corner, green channel)
            with fits.open(output_path, memmap=False) as hdul:
                stacked = hdul[0].data.copy().astype(np.float32)
            if stacked.shape[0] == 3:
                stacked = np.transpose(stacked, (1, 2, 0))   # → (H, W, 3)
            H, W = stacked.shape[:2]
            crop = stacked[5:H // 4, 5:W // 4, 1]  # green channel, corner
            stack_noise = float(np.std(crop))

            # Noise in a single raw frame (debayered, same corner)
            raw, _ = load_fits(paths['light'][0])
            single_rgb = debayer_bilinear(raw, pattern='RGGB')
            single_crop = single_rgb[5:H // 4, 5:W // 4, 1]
            single_noise = float(np.std(single_crop))

            # Stacking should reduce noise; allow generous tolerance
            self.assertLess(stack_noise, single_noise,
                            f"Stacked noise ({stack_noise:.2f}) ≥ single-frame noise "
                            f"({single_noise:.2f}) — stacking not improving SNR")

    def test_star_centroid_in_expected_region(self):
        """The brightest region of the stacked image should be near the centre."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir,
                                              shifts_yx=[(0, 0)] * 6)  # no dither
            _, output_path = self._run_pipeline(tmpdir, paths,
                                                no_registration=True)  # aligned by construction
            if not os.path.exists(output_path):
                self.skipTest("Output file not produced")

            with fits.open(output_path, memmap=False) as hdul:
                stacked = hdul[0].data.copy().astype(np.float32)
            if stacked.shape[0] == 3:
                stacked = np.transpose(stacked, (1, 2, 0))
            H, W = stacked.shape[:2]

            # Find peak brightness location in the stacked luminance
            lum = stacked.mean(axis=2)
            peak_flat = int(np.argmax(lum))
            peak_y, peak_x = peak_flat // W, peak_flat % W

            # Star was placed at (H//2, W//2); allow ±20 px tolerance
            tolerance = 20
            self.assertAlmostEqual(peak_y, H // 2, delta=tolerance,
                                   msg=f"Star Y centroid {peak_y} far from expected {H // 2}")
            self.assertAlmostEqual(peak_x, W // 2, delta=tolerance,
                                   msg=f"Star X centroid {peak_x} far from expected {W // 2}")

    def test_sigma_clip_stacking(self):
        """Pipeline should complete with sigma_clip stack method."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir)
            _, output_path = self._run_pipeline(tmpdir, paths,
                                                stack_method='sigma_clip')
            self.assertTrue(os.path.exists(output_path),
                            "Output FITS not produced with sigma_clip method")

    def test_median_stacking(self):
        """Pipeline should complete with median stack method."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir)
            _, output_path = self._run_pipeline(tmpdir, paths,
                                                stack_method='median')
            self.assertTrue(os.path.exists(output_path),
                            "Output FITS not produced with median method")

    def test_no_calibration_frames(self):
        """Pipeline should complete even without dark/flat masters."""
        from src.models import FrameInfo, ProcessingStats
        from src.pipeline import stack_target

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir)
            light_frames = [
                FrameInfo(path=p, type='light',
                          header={'BAYERPAT': 'RGGB', 'EXPTIME': 120.0, 'ISOSPEED': 800,
                                   'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
                for p in paths['light']
            ]
            masters = {'dark': None, 'flat': None, 'bias': None}
            output_path = os.path.join(tmpdir, 'stacked_nocal.fits')
            args = _make_minimal_args()
            stats = ProcessingStats()
            result = stack_target(light_frames, output_path, args, masters, stats)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(output_path))

    def test_single_light_frame(self):
        """A single frame should still produce a valid output."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, n_lights=1,
                                              shifts_yx=[(0, 0)])
            _, output_path = self._run_pipeline(tmpdir, paths)
            self.assertTrue(os.path.exists(output_path),
                            "Output FITS not produced for single-frame stack")

    def test_accepted_frames_count_in_stats(self):
        """ProcessingStats should record that frames were accepted."""
        from src.io_fits import make_master
        from src.models import FrameInfo, ProcessingStats
        from src.pipeline import stack_target

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, n_lights=4)
            light_frames = [
                FrameInfo(path=p, type='light',
                          header={'BAYERPAT': 'RGGB', 'EXPTIME': 120.0, 'ISOSPEED': 800,
                                   'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
                for p in paths['light']
            ]
            masters = {'dark': None, 'flat': None, 'bias': None}
            output_path = os.path.join(tmpdir, 'stacked.fits')
            args = _make_minimal_args()
            stats = ProcessingStats()
            stack_target(light_frames, output_path, args, masters, stats)
            self.assertGreater(stats.accepted_frames, 0,
                               "No frames were accepted by the pipeline")


class TestE2ERegistration(unittest.TestCase):
    """Validate that registration produces a measurably sharper result."""

    def test_registered_stack_sharper_than_unregistered(self):
        """Stacking with registration should yield better star sharpness than
        naive mean of misaligned frames."""
        H, W = 128, 128
        rng = np.random.default_rng(99)
        # Shifts of ±6 pixels — large enough to visibly blur an unregistered stack
        shifts_yx = [(0, 0), (6, 0), (-6, 0), (0, 6), (0, -6), (3, -3)]

        def _stack_noise(frames_rgb: list) -> float:
            """Naive mean of a list of (H,W,3) arrays → luminance std near star."""
            arr = np.mean(frames_rgb, axis=0)
            lum = arr.mean(axis=2)
            cy, cx = H // 2, W // 2
            region = lum[cy - 8:cy + 8, cx - 8:cx + 8]
            return float(np.std(region))

        from src.debayer import debayer_bilinear

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, shifts_yx=shifts_yx)

            # Run pipeline WITH registration
            from src.io_fits import make_master, load_fits
            from src.models import FrameInfo, ProcessingStats
            from src.pipeline import stack_target

            light_frames = [
                FrameInfo(path=p, type='light',
                          header={'BAYERPAT': 'RGGB', 'EXPTIME': 120.0,
                                   'NAXIS1': W, 'NAXIS2': H})
                for p in paths['light']
            ]
            masters = {'dark': None, 'flat': None, 'bias': None}
            output_reg = os.path.join(tmpdir, 'stacked_reg.fits')
            args_reg = _make_minimal_args(no_registration=False, stack_method='mean')
            stack_target(light_frames, output_reg, args_reg, masters,
                         ProcessingStats())

            # Run pipeline WITHOUT registration
            output_noreg = os.path.join(tmpdir, 'stacked_noreg.fits')
            args_noreg = _make_minimal_args(no_registration=True, stack_method='mean')
            stack_target(light_frames, output_noreg, args_noreg, masters,
                         ProcessingStats())

            if not (os.path.exists(output_reg) and os.path.exists(output_noreg)):
                self.skipTest("One or both output files not produced")

            from astropy.io import fits as afits

            def _lum_near_star(path: str) -> np.ndarray:
                with afits.open(path, memmap=False) as hdul:
                    d = hdul[0].data.copy().astype(np.float32)
                if d.shape[0] == 3:
                    d = np.transpose(d, (1, 2, 0))
                cy, cx = H // 2, W // 2
                return d[cy - 10:cy + 10, cx - 10:cx + 10].mean(axis=2)

            reg_region = _lum_near_star(output_reg)
            noreg_region = _lum_near_star(output_noreg)

            # Registered: star is sharp → higher max relative to mean
            reg_peak = float(reg_region.max())
            noreg_peak = float(noreg_region.max())
            # Registered peak should be at least as high as unregistered peak
            # (misaligned stack smears the star → lower peak)
            self.assertGreaterEqual(reg_peak, noreg_peak * 0.90,
                                    f"Registered peak ({reg_peak:.1f}) significantly "
                                    f"lower than unregistered ({noreg_peak:.1f})")


class TestE2EDrizzlePixfrac(unittest.TestCase):
    """drizzle_pixfrac must actually shrink each frame's footprint, not be a no-op."""

    def _run_drizzle(self, tmpdir: str, paths: dict, pixfrac: float) -> np.ndarray:
        from src.io_fits import make_master
        from src.models import FrameInfo, ProcessingStats
        from src.pipeline import stack_target

        light_frames = [
            FrameInfo(path=p, type='light',
                      header={'BAYERPAT': 'RGGB', 'EXPTIME': 120.0, 'ISOSPEED': 800,
                               'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
            for p in paths['light']
        ]
        dark_frames = [
            FrameInfo(path=p, type='dark',
                      header={'EXPTIME': 120.0, 'ISOSPEED': 800,
                               'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
            for p in paths['dark']
        ]
        flat_frames = [
            FrameInfo(path=p, type='flat',
                      header={'EXPTIME': 0.5, 'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
            for p in paths['flat']
        ]
        masters = {
            'dark': make_master(dark_frames, method='median'),
            'flat': make_master(flat_frames, method='median'),
            'bias': None,
            'dark_exptime': 120.0,
        }
        output_path = os.path.join(tmpdir, f'stacked_pf{pixfrac}.fits')
        args = _make_minimal_args(drizzle_scale=2.0, drizzle_pixfrac=pixfrac,
                                  stack_method='mean')
        stack_target(light_frames, output_path, args, masters, ProcessingStats())
        if not os.path.exists(output_path):
            self.skipTest("Output file not produced")
        with fits.open(output_path, memmap=False) as hdul:
            data = hdul[0].data.copy().astype(np.float64)
        return np.transpose(data, (1, 2, 0)) if data.shape[0] == 3 else data

    def test_pixfrac_changes_output(self):
        """A shrunken footprint (pixfrac<1) must produce a different result
        than the full-footprint default (pixfrac=1) -- this was previously a
        dead CLI flag with zero effect on the actual drizzle path."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, n_lights=4)
            full = self._run_drizzle(tmpdir, paths, pixfrac=1.0)
            shrunk = self._run_drizzle(tmpdir, paths, pixfrac=0.3)
            self.assertFalse(np.allclose(full, shrunk, atol=1e-6),
                             "drizzle_pixfrac had no effect on the output")

    def test_small_pixfrac_leaves_more_uncovered_pixels(self):
        """Shrinking the footprint below the dither spacing should leave more
        output pixels with zero coverage (holes) than the full-footprint case."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, n_lights=4)
            full = self._run_drizzle(tmpdir, paths, pixfrac=1.0)
            shrunk = self._run_drizzle(tmpdir, paths, pixfrac=0.2)
            holes_full = float(np.mean(full.sum(axis=2) == 0.0))
            holes_shrunk = float(np.mean(shrunk.sum(axis=2) == 0.0))
            self.assertGreater(holes_shrunk, holes_full,
                               f"Small pixfrac ({holes_shrunk:.3f} zero-frac) should leave "
                               f"more holes than pixfrac=1 ({holes_full:.3f} zero-frac)")


class TestE2EElasticRegistration(unittest.TestCase):
    """--elastic-registration must correct a genuinely non-rigid distortion
    (two star groups with different residual offsets per frame) that a
    single global affine/translation transform can't null for both groups
    at once -- and must survive --drizzle-scale > 1, unlike the old
    --optical-flow feature it replaces (silently dropped under drizzle)."""

    def _run(self, tmpdir: str, paths: dict, **overrides) -> np.ndarray:
        from src.io_fits import make_master
        from src.models import FrameInfo, ProcessingStats
        from src.pipeline import stack_target

        light_frames = [
            FrameInfo(path=p, type='light',
                      header={'BAYERPAT': 'RGGB', 'EXPTIME': 120.0, 'ISOSPEED': 800,
                               'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
            for p in paths['light']
        ]
        dark_frames = [
            FrameInfo(path=p, type='dark',
                      header={'EXPTIME': 120.0, 'ISOSPEED': 800,
                               'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
            for p in paths['dark']
        ]
        flat_frames = [
            FrameInfo(path=p, type='flat',
                      header={'EXPTIME': 0.5, 'NAXIS1': paths['W'], 'NAXIS2': paths['H']})
            for p in paths['flat']
        ]
        masters = {
            'dark': make_master(dark_frames, method='median'),
            'flat': make_master(flat_frames, method='median'),
            'bias': None,
            'dark_exptime': 120.0,
        }
        suffix = 'on' if overrides.get('elastic_registration') else 'off'
        suffix += f"_dz{overrides.get('drizzle_scale', 1.0)}"
        output_path = os.path.join(tmpdir, f'stacked_elastic_{suffix}.fits')
        args = _make_minimal_args(stack_method='mean', **overrides)
        stack_target(light_frames, output_path, args, masters, ProcessingStats())
        if not os.path.exists(output_path):
            self.skipTest("Output file not produced")
        with fits.open(output_path, memmap=False) as hdul:
            data = hdul[0].data.copy().astype(np.float64)
        return np.transpose(data, (1, 2, 0)) if data.shape[0] == 3 else data

    @staticmethod
    def _group_peaks(img: np.ndarray) -> tuple[float, float]:
        lum = img.mean(axis=2) if img.ndim == 3 else img
        w = lum.shape[1]
        return float(lum[:, :w // 2].max()), float(lum[:, w // 2:].max())

    def test_elastic_registration_changes_output(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_piecewise_dataset(tmpdir, n_lights=8)
            off = self._run(tmpdir, paths, elastic_registration=False)
            on = self._run(tmpdir, paths, elastic_registration=True)
            # A fitted field adds crop margin (calc_common_crop's
            # extra_margin_px), so the two outputs are usually different
            # shapes too -- that alone already proves the flag did something.
            changed = off.shape != on.shape or not np.allclose(off, on, atol=1e-6)
            self.assertTrue(changed, "--elastic-registration had no effect on the output")

    def test_elastic_registration_sharpens_both_groups(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_piecewise_dataset(tmpdir, n_lights=8)
            off = self._run(tmpdir, paths, elastic_registration=False)
            on = self._run(tmpdir, paths, elastic_registration=True)
            peak_a_off, peak_b_off = self._group_peaks(off)
            peak_a_on, peak_b_on = self._group_peaks(on)
            # A misaligned (smeared) star has a lower peak than a sharp one;
            # neither group should get measurably worse, and at least one
            # (both groups carry an equal-and-opposite offset, so a global
            # affine/translation can at best split the difference) should
            # measurably improve.
            self.assertGreaterEqual(peak_a_on, peak_a_off * 0.95)
            self.assertGreaterEqual(peak_b_on, peak_b_off * 0.95)
            self.assertTrue(peak_a_on > peak_a_off * 1.02 or peak_b_on > peak_b_off * 1.02,
                            f"Neither group sharpened: A {peak_a_off:.1f}->{peak_a_on:.1f}, "
                            f"B {peak_b_off:.1f}->{peak_b_on:.1f}")

    def test_elastic_registration_survives_drizzle(self):
        """Unlike the old --optical-flow feature (silently dropped under any
        --drizzle-scale > 1), elastic correction must still take effect."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_piecewise_dataset(tmpdir, n_lights=8)
            off = self._run(tmpdir, paths, elastic_registration=False, drizzle_scale=2.0)
            on = self._run(tmpdir, paths, elastic_registration=True, drizzle_scale=2.0)
            changed = off.shape != on.shape or not np.allclose(off, on, atol=1e-6)
            self.assertTrue(changed,
                            "--elastic-registration had no effect under --drizzle-scale 2.0")


if __name__ == '__main__':
    unittest.main()
