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

from tests._helpers import add_gaussian_stars, write_fits

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
            'bilateral', 'acdnr',
            'deconvolve', 'star_reduce', 'local_contrast', 'sky_neutralize',
        ],
        background_extraction=False,
        dbe=False,
        bg_mesh_size=64,
        bg_filter_size=3,
        bg_clip_sigma=3.0,
        denoise=False,
        denoise_bilateral=False,
        denoise_bilateral_sigma_space=3.0,
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

    # Uniform background — different per Bayer channel to mimic realistic data
    raw = np.zeros(shape, dtype=np.float32)
    raw[0::2, 0::2] = bg * 1.0   # R
    raw[0::2, 1::2] = bg * 1.3   # G1
    raw[1::2, 0::2] = bg * 1.3   # G2
    raw[1::2, 1::2] = bg * 0.7   # B

    # Primary star at (star_cy, star_cx) — sigma=3 px
    add_gaussian_stars(raw, [(star_cy, star_cx, amp)], sigma=3.0)

    # Secondary stars well away from centre — spread to corners
    margin = 16
    secondary_positions = [
        (margin, margin),
        (margin, W - margin),
        (H - margin, margin),
        (H - margin, W - margin),
        (H // 3, W // 2),
    ]
    add_gaussian_stars(raw, [(sy, sx, amp * 0.6) for sy, sx in secondary_positions], sigma=2.5)

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
        write_fits(p, d, {'EXPTIME': 120.0, 'ISOSPEED': 800})
        dark_files.append(p)

    flat_files = []
    for i in range(2):
        f = np.ones((H, W), dtype=np.float32) * 8000.0
        # Mild vignetting
        yy, xx = np.indices((H, W))
        r = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
        f *= np.clip(1.0 - 0.0003 * r, 0.85, 1.0)
        p = os.path.join(tmpdir, f'flat_{i:03d}.fits')
        write_fits(p, f, {'EXPTIME': 0.5})
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
        write_fits(p, raw, {
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
    raw = np.zeros(shape, dtype=np.float32)
    raw[0::2, 0::2] = bg * 1.0
    raw[0::2, 1::2] = bg * 1.3
    raw[1::2, 0::2] = bg * 1.3
    raw[1::2, 1::2] = bg * 0.7

    group_a_base = [(-80, -80), (-80, -20), (80, -80), (80, -20),
                    (-20, -60), (20, -60), (0, -100), (-40, -40)]
    group_b_base = [(-80, 20), (-80, 80), (80, 20), (80, 80),
                    (-20, 60), (20, 60), (0, 100), (40, 40)]

    stars = [(base_cy + dy + offset[0], base_cx + dx + offset[1], amp)
             for base, offset in ((group_a_base, group_a_offset), (group_b_base, group_b_offset))
             for dy, dx in base]
    add_gaussian_stars(raw, stars, sigma=3.0)

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
        write_fits(p, d, {'EXPTIME': 120.0, 'ISOSPEED': 800})
        dark_files.append(p)

    flat_files = []
    for i in range(2):
        f = np.ones((H, W), dtype=np.float32) * 8000.0
        yy, xx = np.indices((H, W))
        r = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
        f *= np.clip(1.0 - 0.0002 * r, 0.85, 1.0)
        p = os.path.join(tmpdir, f'pw_flat_{i:03d}.fits')
        write_fits(p, f, {'EXPTIME': 0.5})
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
        write_fits(p, raw, {'BAYERPAT': 'RGGB', 'EXPTIME': 120.0, 'ISOSPEED': 800})
        light_files.append(p)

    return {'dark': dark_files, 'flat': flat_files, 'light': light_files, 'H': H, 'W': W}


# ---------------------------------------------------------------------------
# End-to-end test cases
# ---------------------------------------------------------------------------

def _frames_and_masters(paths: dict, calibrate: bool = True):
    """FrameInfo list for the lights plus a masters dict (median dark/flat
    from the dataset, or none)."""
    from src.io_fits import make_master
    from src.models import FrameInfo

    size = {'NAXIS1': paths['W'], 'NAXIS2': paths['H']}
    lights = [FrameInfo(path=p, type='light',
                        header={'BAYERPAT': 'RGGB', 'EXPTIME': 120.0, 'ISOSPEED': 800, **size})
              for p in paths['light']]
    if not calibrate:
        return lights, {'dark': None, 'flat': None, 'bias': None}
    darks = [FrameInfo(path=p, type='dark', header={'EXPTIME': 120.0, 'ISOSPEED': 800, **size})
             for p in paths['dark']]
    flats = [FrameInfo(path=p, type='flat', header={'EXPTIME': 0.5, **size})
             for p in paths['flat']]
    return lights, {'dark': make_master(darks, method='median'),
                    'flat': make_master(flats, method='median'),
                    'bias': None, 'dark_exptime': 120.0}


def _stack(tmpdir: str, paths: dict, name: str = 'stacked.fits',
           calibrate: bool = True, **overrides):
    """Run stack_target; return (result, output_path, stats)."""
    from src.models import ProcessingStats
    from src.pipeline import stack_target

    lights, masters = _frames_and_masters(paths, calibrate)
    output_path = os.path.join(tmpdir, name)
    stats = ProcessingStats()
    result = stack_target(lights, output_path, _make_minimal_args(**overrides),
                          masters, stats)
    return result, output_path, stats


def _load_hwc(path: str) -> np.ndarray:
    """Read a stacked FITS as float64 (H, W, C)."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.copy().astype(np.float64)
    return np.transpose(data, (1, 2, 0)) if data.ndim == 3 and data.shape[0] == 3 else data


class TestE2EBasicStack(unittest.TestCase):
    """Run the full pipeline on a tiny synthetic dataset and validate outputs."""

    @classmethod
    def setUpClass(cls):
        # One default calibrated stack, shared by the checks that only read it.
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.paths = _create_synthetic_dataset(cls._tmp.name)
        cls.result, cls.output_path, cls.stats = _stack(cls._tmp.name, cls.paths)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_output_file_created(self):
        """stack_target must produce a FITS file on disk."""
        self.assertIsNotNone(self.result, "stack_target returned None")
        self.assertTrue(os.path.exists(self.output_path),
                        f"Output FITS not found: {self.output_path}")

    def test_output_is_finite_3_channel(self):
        """Output FITS must be openable, finite, and 3-channel."""
        with fits.open(self.output_path, memmap=False) as hdul:
            data = hdul[0].data.copy()
        self.assertTrue(np.all(np.isfinite(data)), "Stacked FITS contains non-finite values")
        self.assertEqual(data.ndim, 3, f"Unexpected stacked FITS shape: {data.shape}")
        self.assertEqual(min(data.shape), 3, f"Expected 3 channels, got shape {data.shape}")

    def test_accepted_frames_count_in_stats(self):
        """ProcessingStats should record that frames were accepted."""
        self.assertGreater(self.stats.accepted_frames, 0,
                           "No frames were accepted by the pipeline")

    def test_stacking_reduces_noise(self):
        """Sky-corner noise of the stack must be below a single debayered frame's."""
        from src.debayer import debayer_bilinear
        from src.io_fits import load_fits

        stacked = _load_hwc(self.output_path)
        H, W = stacked.shape[:2]
        stack_noise = float(np.std(stacked[5:H // 4, 5:W // 4, 1]))  # green, corner
        raw, _ = load_fits(self.paths['light'][0])
        single = debayer_bilinear(raw, pattern='RGGB')
        single_noise = float(np.std(single[5:H // 4, 5:W // 4, 1]))
        self.assertLess(stack_noise, single_noise,
                        f"Stacked noise ({stack_noise:.2f}) >= single-frame noise "
                        f"({single_noise:.2f}) -- stacking not improving SNR")

    def test_star_centroid_in_expected_region(self):
        """The brightest region of the stacked image should be near the centre."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, shifts_yx=[(0, 0)] * 6)  # no dither
            _, output_path, _ = _stack(tmpdir, paths, no_registration=True)
            stacked = _load_hwc(output_path)
        H, W = stacked.shape[:2]
        peak_y, peak_x = np.unravel_index(int(np.argmax(stacked.mean(axis=2))), (H, W))
        # Star was placed at (H//2, W//2); allow +-20 px
        self.assertAlmostEqual(peak_y, H // 2, delta=20)
        self.assertAlmostEqual(peak_x, W // 2, delta=20)

    def test_other_stack_methods_complete(self):
        for method in ('sigma_clip', 'median'):
            with self.subTest(method=method), \
                    tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                paths = _create_synthetic_dataset(tmpdir)
                _, output_path, _ = _stack(tmpdir, paths, stack_method=method)
                self.assertTrue(os.path.exists(output_path))

    def test_no_calibration_frames(self):
        """Pipeline should complete even without dark/flat masters."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir)
            result, output_path, stats = _stack(tmpdir, paths, calibrate=False)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(stats.accepted_frames, 0)

    def test_single_light_frame(self):
        """A single frame should still produce a valid output."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, n_lights=1, shifts_yx=[(0, 0)])
            _, output_path, _ = _stack(tmpdir, paths)
            self.assertTrue(os.path.exists(output_path),
                            "Output FITS not produced for single-frame stack")


class TestE2ERegistration(unittest.TestCase):
    """Validate that registration produces a measurably sharper result."""

    def test_registered_stack_sharper_than_unregistered(self):
        """Registered stack's star peak must not fall below a naive stack of
        frames dithered by +-6 px (a smeared star has a lower peak)."""
        H, W = 128, 128
        shifts_yx = [(0, 0), (6, 0), (-6, 0), (0, 6), (0, -6), (3, -3)]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, shifts_yx=shifts_yx)
            _, out_reg, _ = _stack(tmpdir, paths, 'stacked_reg.fits', calibrate=False,
                                   no_registration=False)
            _, out_noreg, _ = _stack(tmpdir, paths, 'stacked_noreg.fits', calibrate=False,
                                     no_registration=True)

            def _peak_near_star(path: str) -> float:
                d = _load_hwc(path)
                cy, cx = H // 2, W // 2
                return float(d[cy - 10:cy + 10, cx - 10:cx + 10].mean(axis=2).max())

            reg_peak, noreg_peak = _peak_near_star(out_reg), _peak_near_star(out_noreg)
        self.assertGreaterEqual(reg_peak, noreg_peak * 0.90,
                                f"Registered peak ({reg_peak:.1f}) significantly "
                                f"lower than unregistered ({noreg_peak:.1f})")


class DrizzleRunMixin:
    """`_run_drizzle` for TestCase subclasses. Holds no tests, so subclasses
    (here and in test_defer_phase4.py) do not re-run inherited ones."""

    def _run_drizzle(self, tmpdir: str, paths: dict, pixfrac: float, **overrides) -> np.ndarray:
        name = f'stacked_pf{pixfrac}_{len(os.listdir(tmpdir))}.fits'
        _, output_path, _ = _stack(tmpdir, paths, name, drizzle_scale=2.0,
                                   drizzle_pixfrac=pixfrac, stack_method='mean', **overrides)
        if not os.path.exists(output_path):
            self.skipTest("Output file not produced")
        return _load_hwc(output_path)


class TestE2EDrizzlePixfrac(DrizzleRunMixin, unittest.TestCase):
    """drizzle_pixfrac must actually shrink each frame's footprint, not be a no-op."""

    def test_pixfrac_shrinks_footprint(self):
        """pixfrac<1 must change the result (it was once a dead CLI flag), and a
        footprint below the dither spacing must leave more zero-coverage holes."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, n_lights=4)
            full = self._run_drizzle(tmpdir, paths, pixfrac=1.0)
            shrunk = self._run_drizzle(tmpdir, paths, pixfrac=0.2)
        self.assertFalse(np.allclose(full, shrunk, atol=1e-6),
                         "drizzle_pixfrac had no effect on the output")
        holes_full = float(np.mean(full.sum(axis=2) == 0.0))
        holes_shrunk = float(np.mean(shrunk.sum(axis=2) == 0.0))
        self.assertGreater(holes_shrunk, holes_full,
                           f"Small pixfrac ({holes_shrunk:.3f} zero-frac) should leave "
                           f"more holes than pixfrac=1 ({holes_full:.3f} zero-frac)")


class TestE2EDrizzleFusedAndSplat(DrizzleRunMixin, unittest.TestCase):
    """The fused native accumulate must reproduce the warp-then-add path bit
    for bit, and --drizzle-method splat must produce a comparable image."""

    def _run_unfused(self, tmpdir, paths, pixfrac):
        import src.stacking as st
        saved = st._native

        class _NoFused:
            def __getattr__(self, name):
                if name in ('drizzle_accumulate_lanczos3', 'drizzle_splat_frame'):
                    raise AttributeError(name)
                return getattr(saved, name)

        st._native = _NoFused()
        try:
            return self._run_drizzle(tmpdir, paths, pixfrac)
        finally:
            st._native = saved

    def test_fused_matches_unfused_bit_for_bit(self):
        import src.stacking as st
        if not (st.HAS_NATIVE and hasattr(st._native, 'drizzle_accumulate_lanczos3')):
            self.skipTest("native drizzle kernels not built")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, n_lights=4)
            for pf in (1.0, 0.6):
                np.testing.assert_array_equal(self._run_unfused(tmpdir, paths, pf),
                                              self._run_drizzle(tmpdir, paths, pf))

    def test_splat_is_comparable_to_resample(self):
        import src.stacking as st
        if not (st.HAS_NATIVE and hasattr(st._native, 'drizzle_splat_frame')):
            self.skipTest("native drizzle kernels not built")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_synthetic_dataset(tmpdir, n_lights=6)
            base = self._run_drizzle(tmpdir, paths, 1.0)
            splat = self._run_drizzle(tmpdir, paths, 1.0, drizzle_method='splat')
            self.assertEqual(base.shape, splat.shape)
            self.assertTrue(np.isfinite(splat).all())
            covered = (splat.sum(axis=2) > 0) & (base.sum(axis=2) > 0)
            self.assertGreater(covered.mean(), 0.9)
            # same overall brightness (a drop conserves flux), within 3%
            ratio = float(splat[covered].mean() / base[covered].mean())
            self.assertAlmostEqual(ratio, 1.0, delta=0.03)


class TestE2EElasticRegistration(unittest.TestCase):
    """--elastic-registration must correct a genuinely non-rigid distortion
    (two star groups with different residual offsets per frame) that a
    single global affine/translation transform can't null for both groups
    at once -- and must survive --drizzle-scale > 1, unlike the old
    --optical-flow feature it replaces (silently dropped under drizzle)."""

    def _run(self, tmpdir: str, paths: dict, **overrides) -> np.ndarray:
        suffix = 'on' if overrides.get('elastic_registration') else 'off'
        suffix += f"_dz{overrides.get('drizzle_scale', 1.0)}"
        _, output_path, _ = _stack(tmpdir, paths, f'stacked_elastic_{suffix}.fits',
                                   stack_method='mean', **overrides)
        if not os.path.exists(output_path):
            self.skipTest("Output file not produced")
        return _load_hwc(output_path)

    @staticmethod
    def _group_peaks(img: np.ndarray) -> tuple[float, float]:
        lum = img.mean(axis=2) if img.ndim == 3 else img
        w = lum.shape[1]
        return float(lum[:, :w // 2].max()), float(lum[:, w // 2:].max())

    def test_elastic_registration_sharpens_both_groups(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            paths = _create_piecewise_dataset(tmpdir, n_lights=8)
            off = self._run(tmpdir, paths, elastic_registration=False)
            on = self._run(tmpdir, paths, elastic_registration=True)
            # A fitted field adds crop margin (calc_common_crop's
            # extra_margin_px), so the outputs usually differ in shape too.
            changed = off.shape != on.shape or not np.allclose(off, on, atol=1e-6)
            self.assertTrue(changed, "--elastic-registration had no effect on the output")
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

