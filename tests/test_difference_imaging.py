"""Proper image subtraction (ZOGY) and transient detection.

The scenes here are synthetic but not toy: a star field with a *different PSF
per epoch*, which is the situation that makes naive subtraction useless and is
the entire reason ZOGY exists. Two properties are asserted against ground
truth rather than against a golden array:

1. Stellar residuals cancel even when the seeing differs, and measurably
   better than a plain ``new - ref`` on the same scene.
2. An injected transient is recovered at the right position, and a pair with
   nothing new in it produces no detections.
"""

import math
import unittest

import numpy as np
import pytest

import src.transient_triage as _tt_mod
from src.difference_imaging import (
    Transient,
    _prepare_psf,
    _to_luminance,
    detect_transients,
    estimate_background_sigma,
    estimate_flux_ratio,
    write_transient_catalog,
    zogy,
)


def _gaussian_psf(size: int, fwhm: float) -> np.ndarray:
    """Normalised Gaussian kernel, odd-sized and centred."""
    if size % 2 == 0:
        size += 1
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    g = np.exp(-(((yy - c) ** 2 + (xx - c) ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def _render_field(shape, stars, fwhm, sky=100.0, noise=1.0, seed=0):
    """Star field convolved to a given seeing, with Poisson-ish noise."""
    from scipy.signal import fftconvolve

    h, w = shape
    img = np.zeros((h, w), dtype=np.float64)
    for y, x, flux in stars:
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < h and 0 <= ix < w:
            img[iy, ix] += flux

    img = fftconvolve(img, _gaussian_psf(21, fwhm), mode='same')
    rng = np.random.default_rng(seed)
    return img + sky + rng.normal(0.0, noise, (h, w))


_STARS = [(20.0, 25.0, 4000.0), (60.0, 70.0, 9000.0), (40.0, 55.0, 2500.0),
          (75.0, 20.0, 6000.0), (30.0, 80.0, 3200.0), (85.0, 60.0, 5000.0),
          (15.0, 50.0, 2800.0), (55.0, 15.0, 7000.0)]
_SHAPE = (100, 100)


class TestHelpers(unittest.TestCase):
    def test_luminance_uses_rec601_weights(self):
        img = np.zeros((2, 2, 3), dtype=np.float32)
        img[..., 1] = 100.0
        self.assertAlmostEqual(float(_to_luminance(img)[0, 0]), 58.7, places=4)

    def test_luminance_passes_through_a_2d_image(self):
        img = np.arange(9, dtype=np.float32).reshape(3, 3)
        np.testing.assert_allclose(_to_luminance(img), img)

    def test_prepare_psf_normalises_and_centres_at_the_origin(self):
        psf = _gaussian_psf(9, 3.0) * 7.0        # deliberately not unit sum
        prepared = _prepare_psf(psf, (64, 64))

        self.assertEqual(prepared.shape, (64, 64))
        self.assertAlmostEqual(float(prepared.sum()), 1.0, places=9)
        # The peak must land on index [0, 0]: the FFT's origin. Getting this
        # wrong shifts every output by half the frame.
        self.assertEqual(np.unravel_index(int(np.argmax(prepared)), prepared.shape),
                         (0, 0))

    def test_prepare_psf_rejects_a_degenerate_or_oversized_kernel(self):
        with self.assertRaises(ValueError):
            _prepare_psf(np.zeros((5, 5)), (32, 32))
        with self.assertRaises(ValueError):
            _prepare_psf(_gaussian_psf(65, 3.0), (32, 32))

    def test_background_sigma_recovers_injected_noise(self):
        rng = np.random.default_rng(3)
        img = 500.0 + rng.normal(0.0, 7.0, (128, 128))
        self.assertAlmostEqual(estimate_background_sigma(img), 7.0, delta=1.0)

    def test_background_sigma_ignores_bright_stars(self):
        rng = np.random.default_rng(4)
        img = 500.0 + rng.normal(0.0, 5.0, (128, 128))
        img[::16, ::16] = 50000.0               # a grid of bright stars
        self.assertAlmostEqual(estimate_background_sigma(img), 5.0, delta=1.0)

    def test_flux_ratio_recovers_a_known_scaling(self):
        ref = _render_field(_SHAPE, _STARS, fwhm=3.0, seed=1)
        new = ref * 1.7
        self.assertAlmostEqual(estimate_flux_ratio(new, ref), 1.7, delta=0.1)

    def test_flux_ratio_is_unity_for_identical_images(self):
        ref = _render_field(_SHAPE, _STARS, fwhm=3.0, seed=2)
        self.assertAlmostEqual(estimate_flux_ratio(ref, ref), 1.0, delta=0.05)


class TestZogySubtraction(unittest.TestCase):
    """The core claim: stellar residuals cancel across differing seeing."""

    def setUp(self):
        self.psf_ref = _gaussian_psf(21, 3.0)
        self.psf_new = _gaussian_psf(21, 4.5)      # noticeably worse seeing
        self.ref = _render_field(_SHAPE, _STARS, fwhm=3.0, sky=0.0, noise=1.0, seed=10)
        self.new = _render_field(_SHAPE, _STARS, fwhm=4.5, sky=0.0, noise=1.0, seed=11)

    def _star_region_peak(self, img):
        """Largest absolute residual within a few px of any input star."""
        worst = 0.0
        for y, x, _ in _STARS:
            iy, ix = int(y), int(x)
            patch = img[max(0, iy - 4):iy + 5, max(0, ix - 4):ix + 5]
            worst = max(worst, float(np.abs(patch).max()))
        return worst

    def test_beats_naive_subtraction_at_star_positions(self):
        """The money test. Naive subtraction dipoles; ZOGY does not."""
        result = zogy(self.new, self.ref, self.psf_new, self.psf_ref)

        naive = self.new - self.ref
        naive_resid = self._star_region_peak(naive) / estimate_background_sigma(naive)
        zogy_resid = self._star_region_peak(result.difference) / \
            estimate_background_sigma(result.difference)

        self.assertLess(zogy_resid, naive_resid,
                        "ZOGY should leave smaller stellar residuals than a "
                        "plain subtraction when the PSFs differ")
        # Naive subtraction of a 1.5x seeing change leaves enormous dipoles;
        # this guards the test scene itself from becoming trivially easy.
        self.assertGreater(naive_resid, 20.0,
                           "test scene is not exercising a real PSF mismatch")

    def test_identical_epochs_produce_no_detections(self):
        img = _render_field(_SHAPE, _STARS, fwhm=3.0, sky=0.0, noise=1.0, seed=12)
        psf = _gaussian_psf(21, 3.0)

        result = zogy(img, img, psf, psf, astrometric_sigma=(0.3, 0.3))

        self.assertEqual(detect_transients(result.score_corr, threshold=5.0), [])

    def test_outputs_have_the_input_shape_and_are_finite(self):
        result = zogy(self.new, self.ref, self.psf_new, self.psf_ref)
        for name, arr in (('difference', result.difference),
                          ('score', result.score),
                          ('score_corr', result.score_corr)):
            self.assertEqual(arr.shape, _SHAPE, name)
            self.assertTrue(np.isfinite(arr).all(), f"{name} has non-finite pixels")
        self.assertGreater(result.flux_difference, 0.0)

    def test_accepts_rgb_input_by_reducing_to_luminance(self):
        rgb_new = np.stack([self.new] * 3, axis=-1)
        rgb_ref = np.stack([self.ref] * 3, axis=-1)
        result = zogy(rgb_new, rgb_ref, self.psf_new, self.psf_ref)
        self.assertEqual(result.difference.shape, _SHAPE)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            zogy(self.new, self.ref[:-1], self.psf_new, self.psf_ref)


class TestTransientRecovery(unittest.TestCase):
    def test_injected_new_source_is_found_at_the_right_place(self):
        psf_ref = _gaussian_psf(21, 3.0)
        psf_new = _gaussian_psf(21, 3.4)
        ref = _render_field(_SHAPE, _STARS, fwhm=3.0, sky=0.0, noise=1.0, seed=20)

        ty, tx = 48.0, 33.0
        new = _render_field(_SHAPE, _STARS + [(ty, tx, 3000.0)],
                            fwhm=3.4, sky=0.0, noise=1.0, seed=21)

        result = zogy(new, ref, psf_new, psf_ref, astrometric_sigma=(0.3, 0.3))
        found = detect_transients(result.score_corr, threshold=5.0)

        self.assertTrue(found, "injected transient was not detected")
        best = max(found, key=lambda t: t.significance)
        self.assertLess(math.hypot(best.y - ty, best.x - tx), 3.0,
                        f"detected at ({best.y}, {best.x}), injected at ({ty}, {tx})")
        self.assertEqual(best.kind, 'brightening')

    def test_a_source_that_disappears_is_reported_as_fading(self):
        psf = _gaussian_psf(21, 3.0)
        ty, tx = 44.0, 66.0
        ref = _render_field(_SHAPE, _STARS + [(ty, tx, 3000.0)],
                            fwhm=3.0, sky=0.0, noise=1.0, seed=30)
        new = _render_field(_SHAPE, _STARS, fwhm=3.0, sky=0.0, noise=1.0, seed=31)

        result = zogy(new, ref, psf, psf, astrometric_sigma=(0.3, 0.3))
        found = detect_transients(result.score_corr, threshold=5.0)

        self.assertTrue(found)
        best = max(found, key=lambda t: t.significance)
        self.assertEqual(best.kind, 'fading')
        self.assertLess(math.hypot(best.y - ty, best.x - tx), 3.0)

    def test_astrometric_term_suppresses_misregistration_false_positives(self):
        """Without it, a sub-pixel shift lights up every bright star."""
        from scipy.ndimage import shift as ndshift

        psf = _gaussian_psf(21, 3.0)
        ref = _render_field(_SHAPE, _STARS, fwhm=3.0, sky=0.0, noise=1.0, seed=40)
        new = ndshift(ref, (0.4, 0.25), order=3, mode='nearest')

        without = zogy(new, ref, psf, psf, astrometric_sigma=(0.0, 0.0))
        with_ast = zogy(new, ref, psf, psf, astrometric_sigma=(0.4, 0.4))

        n_without = len(detect_transients(without.score_corr, threshold=5.0))
        n_with = len(detect_transients(with_ast.score_corr, threshold=5.0))

        # Measured on this scene: every one of the 8 stars lights up (16
        # detections, both lobes of each dipole) without the term, and none
        # with it. Assert the property, not merely a direction -- "fewer" would
        # still pass at 15 of 16, i.e. with the false positives essentially
        # intact.
        self.assertGreaterEqual(n_without, len(_STARS),
                                "scene should light up every star without the term")
        self.assertEqual(n_with, 0,
                         "astrometric term should remove all shift artefacts")


class TestDetectTransients(unittest.TestCase):
    def test_threshold_is_respected_and_peaks_are_ranked(self):
        s = np.zeros((50, 50), dtype=np.float32)
        s[10, 10] = 9.0
        s[30, 30] = 6.0
        s[40, 40] = 3.0                      # below threshold

        found = detect_transients(s, threshold=5.0)

        self.assertEqual(len(found), 2)
        self.assertAlmostEqual(found[0].significance, 9.0, places=4)
        self.assertAlmostEqual(found[1].significance, 6.0, places=4)

    def test_negative_peaks_are_reported_as_fading(self):
        s = np.zeros((20, 20), dtype=np.float32)
        s[5, 5] = -8.0
        found = detect_transients(s, threshold=5.0)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].kind, 'fading')
        self.assertAlmostEqual(found[0].significance, 8.0, places=4)

    def test_exclusion_radius_prevents_duplicate_peaks(self):
        s = np.zeros((40, 40), dtype=np.float32)
        s[20, 20] = 10.0
        s[20, 21] = 9.5                      # same source, adjacent pixel
        found = detect_transients(s, threshold=5.0, min_separation=5)
        self.assertEqual(len(found), 1)

    def test_non_finite_pixels_are_ignored(self):
        s = np.zeros((20, 20), dtype=np.float32)
        s[3, 3] = np.nan
        s[9, 9] = np.inf
        s[5, 5] = 7.0
        found = detect_transients(s, threshold=5.0)
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].y, found[0].x), (5.0, 5.0))

    def test_respects_the_candidate_cap(self):
        rng = np.random.default_rng(5)
        s = rng.uniform(20.0, 30.0, (60, 60)).astype(np.float32)
        found = detect_transients(s, threshold=5.0, min_separation=1,
                                  max_candidates=7)
        self.assertEqual(len(found), 7)


class TestCatalogOutput(unittest.TestCase):
    def test_csv_has_a_header_and_one_row_per_detection(self):
        import csv
        import tempfile

        transients = [Transient(y=10.0, x=20.0, significance=7.5, kind='brightening'),
                      Transient(y=30.0, x=40.0, significance=5.5, kind='fading')]
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/cat.csv"
            write_transient_catalog(path, transients)
            with open(path, newline='', encoding='utf-8') as fh:
                rows = list(csv.reader(fh))

        self.assertEqual(rows[0], ['x', 'y', 'significance_sigma', 'kind'])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][3], 'brightening')
        self.assertEqual(rows[2][3], 'fading')

    def test_wcs_adds_sky_coordinates(self):
        import csv
        import tempfile
        try:
            from astropy.wcs import WCS
        except Exception:
            self.skipTest("astropy required")

        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [50, 50]
        wcs.wcs.cdelt = [-0.001, 0.001]
        wcs.wcs.crval = [120.0, 30.0]
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/cat.csv"
            write_transient_catalog(
                path, [Transient(y=50.0, x=50.0, significance=8.0, kind='brightening')],
                wcs=wcs)
            with open(path, newline='', encoding='utf-8') as fh:
                rows = list(csv.reader(fh))

        self.assertIn('ra_deg', rows[0])
        self.assertAlmostEqual(float(rows[1][4]), 120.0, delta=0.3)
        self.assertAlmostEqual(float(rows[1][5]), 30.0, delta=0.3)


if __name__ == '__main__':
    unittest.main()


def _rgb(lum):
    """Replicate a luminance plane to the pipeline's (H, W, 3) layout."""
    return np.repeat(np.asarray(lum, dtype=np.float32)[:, :, None], 3, axis=2)


def _write_reference(path, lum, rawstack=True):
    """Write a reference the way the pipeline does: (3, H, W), RAWSTACK set."""
    from astropy.io import fits
    hdu = fits.PrimaryHDU(data=np.transpose(_rgb(lum), (2, 0, 1)))
    if rawstack:
        hdu.header['RAWSTACK'] = True
    hdu.writeto(path, overwrite=True)


def _wide_field(n_stars=40, seed=7, shape=(220, 240)):
    """A star field with enough well-separated stars to register and fit a PSF."""
    rng = np.random.default_rng(seed)
    h, w = shape
    stars = []
    while len(stars) < n_stars:
        y, x = rng.uniform(25, h - 25), rng.uniform(25, w - 25)
        if all(math.hypot(y - sy, x - sx) > 20 for sy, sx, _ in stars):
            stars.append((y, x, float(rng.uniform(3000, 12000))))
    return stars


class TestRunTransientDetection(unittest.TestCase):
    """The ``--transient-detect`` entry point, end to end.

    ``zogy`` and ``detect_transients`` were covered; the orchestration around
    them was not, and that is where the failures documented in the code itself
    live: the sky-pedestal mismatch, the (C, H, W) axis order, the reference
    header, and the footprint of a rotated reference.
    """

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ref_path = f"{self.tmp.name}/ref.fits"
        self.out_path = f"{self.tmp.name}/new.fits"
        self.stars = _wide_field()

    def _epochs(self, transient=None, ref_pedestal=0.0, ref_fwhm=3.4, new_fwhm=2.8):
        ref = _render_field((220, 240), self.stars, fwhm=ref_fwhm,
                            sky=ref_pedestal, noise=1.0, seed=50)
        new_stars = self.stars + ([transient] if transient else [])
        new = _render_field((220, 240), new_stars, fwhm=new_fwhm,
                            sky=0.0, noise=1.0, seed=51)
        return new, ref

    def _run(self, new, ref, **kw):
        from src.difference_imaging import run_transient_detection
        _write_reference(self.ref_path, ref, rawstack=kw.pop('rawstack', True))
        return run_transient_detection(_rgb(new), self.ref_path, self.out_path, **kw)

    def test_finds_an_injected_transient_across_a_seeing_change(self):
        ty, tx = 110.0, 120.0
        new, ref = self._epochs(transient=(ty, tx, 14000.0))

        summary = self._run(new, ref)

        self.assertIsNotNone(summary, "a well-formed pair must produce a result")
        best = max(summary['transients'], key=lambda t: t.significance)
        self.assertEqual(best.kind, 'brightening')
        self.assertLess(math.hypot(best.y - ty, best.x - tx), 3.0)

    def test_a_pair_with_nothing_new_reports_nothing(self):
        new, ref = self._epochs()
        summary = self._run(new, ref)
        self.assertIsNotNone(summary)
        self.assertEqual(summary['transients'], [],
                         "identical fields must not produce candidates")

    def test_a_sky_pedestal_difference_does_not_flood_the_catalogue(self):
        """Regression: the in-memory stack has had its pedestal removed, while
        a reference read from disk still carries one -- medians of 0.7 against
        952 were observed for the same field. Without per-epoch background
        subtraction every pixel reads as 'fading'."""
        new, ref = self._epochs(ref_pedestal=950.0)
        summary = self._run(new, ref)
        self.assertIsNotNone(summary)
        self.assertLess(len(summary['transients']), 5)

    def test_reference_stored_channels_first_is_read_in_the_right_order(self):
        """A wrong-axis read yields garbage rather than an exception; the
        planted transient surviving at the right pixel proves the order."""
        ty, tx = 60.0, 180.0
        new, ref = self._epochs(transient=(ty, tx, 14000.0))
        summary = self._run(new, ref)
        self.assertIsNotNone(summary)
        best = max(summary['transients'], key=lambda t: t.significance)
        self.assertLess(math.hypot(best.y - ty, best.x - tx), 3.0)

    def test_a_non_linear_reference_is_refused(self):
        """A post-processed reference mismatches the flux scale, and every star
        would report as a transient. --merge refuses the same file."""
        import os
        new, ref = self._epochs()
        summary = self._run(new, ref, rawstack=False)
        self.assertIsNone(summary)
        self.assertFalse(os.path.exists(self.out_path.replace('.fits', '_transients.csv')),
                         "a refused reference must not leave a catalogue behind")

    def test_a_missing_reference_returns_none(self):
        from src.difference_imaging import run_transient_detection
        new, _ = self._epochs()
        self.assertIsNone(run_transient_detection(
            _rgb(new), f"{self.tmp.name}/nope.fits", self.out_path))

    def test_mismatched_frame_sizes_are_reconciled_not_rejected(self):
        """Two independently-stacked sessions of the same target routinely
        differ in pixel dimensions (different dither pattern, different
        Phase 3 common-crop) even though they cover the same field -- this
        used to be a hard failure. `_align_reference` now embeds the smaller
        epoch onto the other's grid (`src.utils.embed_to_shape`, the same
        trick `merge.py` uses for a previous stack's own mismatched shape)
        instead of refusing, and a real transient inside the overlap is
        still found afterward."""
        ty, tx = 80.0, 90.0  # inside the 200x200 crop below
        new, ref = self._epochs(transient=(ty, tx, 14000.0))
        summary = self._run(new[:200, :200], ref)
        self.assertIsNotNone(summary, "a shape mismatch must not be a hard failure")
        best = max(summary['transients'], key=lambda t: t.significance)
        self.assertLess(math.hypot(best.y - ty, best.x - tx), 3.0)

    def test_the_registration_sigma_is_measured_and_floored_not_hardcoded(self):
        """The old code returned a literal 0.3 and reported it as a measured
        residual. The summary must carry the value actually used, and it must
        never fall below the documented floor."""
        from src.difference_imaging import _ASTROMETRIC_SIGMA_FLOOR_PX
        new, ref = self._epochs()
        summary = self._run(new, ref)
        self.assertIsNotNone(summary)
        self.assertGreaterEqual(summary['astrometric_sigma_px'],
                                _ASTROMETRIC_SIGMA_FLOOR_PX)
        measured = summary['registration_residual_px']
        self.assertTrue(measured is None or measured >= 0.0)

    def test_stars_in_the_uncovered_wedges_of_a_rotated_reference_are_not_transients(self):
        """A cross-night pair differs by field rotation on an alt-az mount, so
        the warped reference has empty corners. A star in `new` there has
        nothing to subtract against and used to come out as a confident
        'brightening'. Measured on this scene: 6 false candidates (exactly the
        six planted corner/edge stars) without the footprint mask, 0 with it.

        The stars are placed deliberately: a first version of this test used a
        field with a 25 px margin, put none of its 60 stars in a wedge, and so
        passed with the mask disabled.
        """
        from scipy.ndimage import rotate

        corner = [(12.0, 12.0, 9000.0), (14.0, 226.0, 9000.0),
                  (206.0, 14.0, 9000.0), (208.0, 228.0, 9000.0),
                  (110.0, 8.0, 9000.0), (112.0, 232.0, 9000.0)]
        stars = _wide_field(n_stars=40, seed=3)
        new = _render_field((220, 240), stars + corner, fwhm=3.0,
                            sky=0.0, noise=1.0, seed=51)
        # The reference has the same stars *except* the corner ones (they are
        # outside its coverage once rotated), rotated by 9 degrees.
        ref = rotate(_render_field((220, 240), stars, fwhm=3.0, sky=0.0,
                                   noise=1.0, seed=50),
                     9.0, reshape=False, order=1, mode='constant', cval=0.0)

        summary = self._run(new, ref)

        self.assertIsNotNone(summary, "a 9 degree rotation must still register")
        self.assertLess(summary['covered_fraction'], 0.95,
                        "the rotated reference leaves empty wedges")
        self.assertGreater(summary['covered_fraction'], 0.5)
        self.assertEqual(
            [t for t in summary['transients'] if t.kind == 'brightening'], [],
            "stars where the reference has no coverage are not transients")

    def test_a_fully_covered_pair_reports_full_coverage(self):
        new, ref = self._epochs()
        summary = self._run(new, ref)
        self.assertIsNotNone(summary)
        self.assertGreater(summary['covered_fraction'], 0.99)

    def test_stars_beyond_a_smaller_references_extent_are_not_transients(self):
        """A genuinely smaller reference (a different session's own Phase 3
        crop, not just a slice of the same array) gets zero-padded onto the
        new stack's grid before registration (`_align_reference` /
        `src.utils.embed_to_shape`). Stars in `new` beyond the reference's
        real extent have nothing to subtract against there -- same failure
        mode as a rotated reference's empty wedges, just from padding instead
        of rotation -- and must not be reported as 'brightening' either."""
        # n_stars kept low relative to the small shape: _wide_field's
        # rejection sampling (each star >20px from every other) needs real
        # headroom -- 30 stars in this shape's margins is near the packing
        # limit and made the sampling loop pathologically slow.
        stars = _wide_field(n_stars=12, seed=9, shape=(140, 150))
        edge_stars = [(180.0, 210.0, 9000.0), (190.0, 30.0, 9000.0),
                     (30.0, 220.0, 9000.0)]
        new = _render_field((220, 240), stars + edge_stars, fwhm=3.0,
                            sky=0.0, noise=1.0, seed=51)
        # A genuinely smaller array -- the reference's own (unpadded) shape,
        # not new[:140, :150] -- so embedding must zero-pad it, not just crop.
        ref = _render_field((140, 150), stars, fwhm=3.0, sky=0.0, noise=1.0, seed=50)

        summary = self._run(new, ref)

        self.assertIsNotNone(summary, "a smaller reference must not be a hard failure")
        self.assertLess(summary['covered_fraction'], 0.95,
                        "the padded exterior is not reference coverage")
        self.assertEqual(
            [t for t in summary['transients'] if t.kind == 'brightening'], [],
            "stars beyond the reference's real extent are not transients")

    def test_outputs_are_written_next_to_the_output_path(self):
        import os
        new, ref = self._epochs(transient=(110.0, 120.0, 14000.0))
        summary = self._run(new, ref)
        self.assertIsNotNone(summary)
        stem = self.out_path[:-len('.fits')]
        for suffix in ('_difference.fits', '_scorr.fits', '_transients.csv'):
            with self.subTest(suffix=suffix):
                self.assertTrue(os.path.exists(stem + suffix))

    def test_triage_disabled_leaves_real_probability_none_and_off_the_csv(self):
        new, ref = self._epochs(transient=(110.0, 120.0, 14000.0))
        summary = self._run(new, ref, triage=False)
        self.assertIsNotNone(summary)
        self.assertTrue(all(t.real_probability is None for t in summary['transients']))
        with open(summary['catalog']) as fh:
            header = fh.readline()
        self.assertNotIn('real_probability', header)

    def test_triage_requested_but_unavailable_does_not_crash(self):
        """--transient-triage without a native backend/model self-disables
        with a warning (checked via the returned real_probability, not the
        log) rather than raising -- mirrors --originvision's own gate."""
        import src.transient_triage as tt_mod
        had = tt_mod._HAS_NATIVE_TRIAGE
        tt_mod._HAS_NATIVE_TRIAGE = False
        try:
            new, ref = self._epochs(transient=(110.0, 120.0, 14000.0))
            summary = self._run(new, ref, triage=True)
        finally:
            tt_mod._HAS_NATIVE_TRIAGE = had
        self.assertIsNotNone(summary)
        self.assertTrue(all(t.real_probability is None for t in summary['transients']))

    @pytest.mark.skipif(
        not _tt_mod.scoring_backend_available() or _tt_mod.resolve_model_path(None) is None,
        reason='native transient_triage_score / bundled model absent -- run '
              'tools/gen_transient_triage_data.py + tools/train_transient_triage.py')
    def test_triage_populates_real_probability_and_csv_column(self):
        new, ref = self._epochs(transient=(110.0, 120.0, 14000.0))
        summary = self._run(new, ref, triage=True)
        self.assertIsNotNone(summary)
        self.assertTrue(summary['transients'], "expected at least the injected transient")
        self.assertTrue(all(t.real_probability is not None for t in summary['transients']))
        self.assertTrue(all(0.0 <= t.real_probability <= 1.0 for t in summary['transients']))
        with open(summary['catalog']) as fh:
            header = fh.readline()
        self.assertIn('real_probability', header)


class TestErodeFootprint(unittest.TestCase):
    """``_erode`` shrinks the covered region away from *uncovered* pixels only.

    Tested directly because it cannot be reached through a fully covered pair:
    that short-circuits before erosion runs, so an end-to-end test of the
    frame-edge behaviour passes whatever the border setting is (a mutation
    that flipped it survived the first version of this suite).
    """

    def test_the_frame_edge_is_not_treated_as_uncovered(self):
        from src.difference_imaging import _erode
        mask = np.ones((40, 40), dtype=bool)
        mask[:5, :5] = False                      # an empty wedge in one corner

        out = _erode(mask, 3)

        # Edges far from the wedge keep their coverage. With the scipy default
        # (border_value=0) these all erode away, discarding a PSF-width strip
        # off every side of the frame.
        self.assertTrue(out[20, 39], "right edge should survive")
        self.assertTrue(out[39, 20], "bottom edge should survive")
        self.assertTrue(out[39, 39], "far corner should survive")

    def test_pixels_near_an_uncovered_region_are_eroded(self):
        from src.difference_imaging import _erode
        mask = np.ones((40, 40), dtype=bool)
        mask[:5, :5] = False

        out = _erode(mask, 3)

        self.assertFalse(out[2, 6], "within the margin of the wedge")
        self.assertFalse(out[6, 2])
        self.assertTrue(out[20, 20], "well clear of the wedge")

    def test_a_fully_covered_mask_is_returned_unchanged(self):
        from src.difference_imaging import _erode
        mask = np.ones((10, 10), dtype=bool)
        self.assertTrue(_erode(mask, 4).all())


class TestCatalogSkyCoordinates(unittest.TestCase):
    """The catalogue's RA/Dec column is what makes a candidate checkable."""

    def _wcs(self, naxis):
        from astropy.wcs import WCS
        w = WCS(naxis=naxis)
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'LINEAR'][:naxis]
        w.wcs.crval = [270.9, -24.4, 0.0][:naxis]
        w.wcs.crpix = [50.0, 50.0, 1.0][:naxis]
        w.wcs.cdelt = [-0.0005, 0.0005, 1.0][:naxis]
        return w

    def _write(self, wcs):
        import csv
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/t.csv"
            write_transient_catalog(path, [Transient(50.0, 50.0, 9.0, 'brightening')],
                                    wcs=wcs)
            with open(path, newline='') as fh:
                return list(csv.DictReader(fh))

    def test_a_celestial_wcs_fills_ra_and_dec(self):
        row = self._write(self._wcs(2))[0]
        self.assertAlmostEqual(float(row['ra_deg']), 270.9, delta=0.01)
        self.assertAlmostEqual(float(row['dec_deg']), -24.4, delta=0.01)

    def test_a_three_axis_wcs_is_the_failure_the_pipeline_must_avoid(self):
        """Documents why pipeline.py builds WCS(header, naxis=2): a 3-axis WCS
        (what a bare WCS(hdu.header) yields for the (3, H, W) cube) still
        reports has_celestial, but cannot convert 2D pixel coordinates."""
        wcs3 = self._wcs(3)
        self.assertTrue(wcs3.has_celestial)
        row = self._write(wcs3)[0]
        self.assertEqual(row['ra_deg'], '')

    def test_a_failed_conversion_is_announced_not_silent(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self._write(self._wcs(3))
        self.assertIn('no sky coordinates', buf.getvalue())


class TestDetectTransientsBehaviour(unittest.TestCase):
    def test_a_non_positive_threshold_is_rejected(self):
        """Zero admits every pixel: max_candidates of pure noise as detections."""
        with self.assertRaises(ValueError):
            detect_transients(np.zeros((10, 10), dtype=np.float32), threshold=0.0)

    def test_matches_the_reference_greedy_algorithm_on_ties_and_nans(self):
        """The one-pass rewrite must give the same detections, in the same
        order, as the per-candidate argmax loop it replaced -- including when
        many pixels tie and some are NaN."""
        rng = np.random.default_rng(1)
        for trial in range(40):
            base = rng.normal(0, 2.5, (60, 70))
            s = np.round(base) if trial % 2 else base
            s[rng.random(s.shape) < 0.02] = np.nan
            sep, cap = int(rng.integers(1, 8)), int(rng.choice([5, 50, 500]))

            work = np.where(np.isfinite(s), np.abs(s), 0.0)
            expected = []
            while len(expected) < cap:
                idx = int(np.argmax(work))
                if float(work.flat[idx]) < 4.0:
                    break
                y, x = np.unravel_index(idx, work.shape)
                expected.append((float(y), float(x)))
                work[max(0, y - sep):y + sep + 1, max(0, x - sep):x + sep + 1] = 0.0

            got = [(t.y, t.x) for t in detect_transients(s, 4.0, sep, cap)]
            self.assertEqual(got, expected, f"trial {trial}")
