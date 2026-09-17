"""Instrumental terms, the field-of-view gate, and multi-frame separation.

These three go together. The single-frame physical sky model was measured to
fail on a real 1-degree field: the components are nearly collinear there
(condition ~1e5), it could not fit the real gradient, and the gradient at that
scale is mostly *instrumental* anyway. The gate declines those fields outright,
the instrumental terms model what is actually there, and the multi-frame fit is
what makes any of it identifiable -- sky components move between exposures
while detector ones do not.
"""

import unittest
import warnings

import numpy as np

from src.sky_model import (
    _field_of_view_deg,
    amp_glow_basis,
    basis_condition,
    build_basis,
    build_geometry,
    fit_sky_model,
    fit_sky_model_multi,
    remove_physical_sky,
)

try:
    from astropy.wcs import WCS
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False

_SITE = (33.83, -117.79)
_LAGOON = (271.0, -24.4)


def _wcs(fov_deg, shape, crval=_LAGOON):
    h, w = shape
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2, h / 2]
    wcs.wcs.cdelt = [-fov_deg / w, fov_deg / w]
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


class TestInstrumentalTerms(unittest.TestCase):
    """Detector-fixed basis terms -- and the one deliberately left out."""

    def test_each_corner_term_peaks_at_its_own_corner(self):
        terms, names = amp_glow_basis((40, 60))
        corners = {'amp_glow_tl': (0, 0), 'amp_glow_tr': (0, 59),
                   'amp_glow_bl': (39, 0), 'amp_glow_br': (39, 59)}
        for term, name in zip(terms, names):
            with self.subTest(name=name):
                peak = np.unravel_index(int(np.argmax(term)), term.shape)
                self.assertEqual(tuple(int(v) for v in peak), corners[name])

    def test_corner_terms_cannot_absorb_a_centred_object(self):
        """Why there is no vignetting term.

        Amp glow peaks at an edge, so a non-negative sum of corner terms has
        no interior maximum and cannot swallow a centred nebula. A radial
        vignetting term would peak at the centre -- exactly where observers
        frame their target -- and ``_corner_gradient`` could not catch it,
        because a radial pattern leaves all four corners equal by
        construction. Flats already divide vignetting out in any case.
        """
        terms, _ = amp_glow_basis((51, 51))
        total = np.sum(terms, axis=0)          # worst case: every corner lit
        self.assertLess(float(total[25, 25]), float(total.max()),
                        "a corner basis must not peak in the middle")

    @unittest.skipUnless(HAS_ASTROPY, "astropy required")
    def test_instrumental_terms_are_opt_in(self):
        geom = build_geometry(_wcs(8.0, (60, 60)), (60, 60), *_SITE,
                              '2026-09-26T06:00:00Z')
        _, plain = build_basis(geom)
        _, with_inst = build_basis(geom, instrumental=True)
        self.assertFalse(any(n.startswith('amp_glow') for n in plain))
        self.assertEqual(len(with_inst), len(plain) + 4)


@unittest.skipUnless(HAS_ASTROPY, "astropy required")
class TestFieldOfViewGate(unittest.TestCase):
    """Measured: the components only separate around 5 degrees."""

    def test_field_of_view_is_measured_from_the_wcs(self):
        self.assertAlmostEqual(
            _field_of_view_deg(_wcs(8.0, (60, 60)), (60, 60)), 8.0, places=6)

    def test_longest_axis_wins_on_a_non_square_sensor(self):
        wcs = _wcs(4.0, (100, 200))
        self.assertAlmostEqual(_field_of_view_deg(wcs, (100, 200)), 4.0, places=6)

    def test_field_of_view_reads_a_cd_matrix_wcs(self):
        """Regression: a real session WCS leaves cdelt at [1, 1].

        A Celestron Origin solve carries its scale in a CD matrix. Reading
        wcs.wcs.cdelt directly then returns the image size in *pixels* as
        though it were degrees -- 2958 "degrees" for a 2958-pixel frame,
        which sails past the gate instead of tripping it.
        """
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1528.5, 1024.5]
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.cd = [[-4.104e-4, -6.636e-6], [-6.636e-6, 4.104e-4]]
        with warnings.catch_warnings():
            # astropy warns that cd wins over cdelt -- which is precisely the
            # trap being demonstrated, so the warning is the expected result.
            warnings.simplefilter('ignore')
            self.assertEqual(list(wcs.wcs.cdelt), [1.0, 1.0])

        fov = _field_of_view_deg(wcs, (1927, 2958))
        self.assertLess(fov, 5.0, "a telescope field must not read as 2958 deg")
        self.assertAlmostEqual(fov, 2958 * 4.104e-4, delta=0.05)

    def test_unusable_wcs_reports_no_field_of_view(self):
        self.assertIsNone(_field_of_view_deg(None, (60, 60)))

    def test_narrow_field_is_declined_before_any_fitting(self):
        yy, _ = np.mgrid[0:60, 0:60]
        img = np.repeat((1000.0 + yy)[:, :, None], 3, axis=2).astype(np.float32)
        self.assertIsNone(remove_physical_sky(
            img, _wcs(1.0, (60, 60)), *_SITE, '2026-08-31T20:40:32-0700'))

    def test_condition_number_falls_as_the_field_widens(self):
        """The measurement the 5-degree threshold came from."""
        conds = []
        for fov in (1.0, 5.0, 20.0):
            geom = build_geometry(_wcs(fov, (60, 60)), (60, 60), *_SITE,
                                  '2026-08-31T20:40:32-0700')
            basis, _ = build_basis(geom)
            conds.append(basis_condition(basis))
        self.assertGreater(conds[0], conds[1])
        self.assertGreater(conds[1], conds[2])
        self.assertGreater(conds[0], 1e4, "a 1-degree field should be degenerate")


@unittest.skipUnless(HAS_ASTROPY, "astropy required")
class TestMultiFrameSeparation(unittest.TestCase):
    """Sky components move between exposures; the detector does not."""

    TRUTH = {'constant': 800.0, 'airglow': 140.0, 'amp_glow_br': 90.0}
    TIMES = ['2026-09-26T04:00:00Z', '2026-09-26T05:30:00Z',
             '2026-09-26T07:00:00Z', '2026-09-26T08:30:00Z']

    def _scene(self, fov_deg, noise, shape=(60, 60), seed=0):
        rng = np.random.default_rng(seed)
        geoms = [build_geometry(_wcs(fov_deg, shape), shape, *_SITE, t)
                 for t in self.TIMES]
        frames, names = [], None
        for g in geoms:
            basis, names = build_basis(g, instrumental=True)
            coeffs = np.array([self.TRUTH.get(n, 0.0) for n in names])
            clean = np.tensordot(coeffs, basis, axes=(0, 0))
            frames.append((clean + rng.normal(0, noise, shape)).astype(np.float32))
        return geoms, frames, names

    def _error(self, names, coeffs):
        return sum(abs(self.TRUTH.get(n, 0.0) - float(v))
                   for n, v in zip(names, coeffs))

    def test_joint_fit_beats_a_single_frame_under_noise(self):
        """The whole point.

        Noiseless synthetic data proves nothing here -- it lies exactly in the
        basis span, so any least-squares fit recovers it perfectly. The
        advantage only appears once the ill-conditioning has noise to amplify:
        measured ~8x lower coefficient error on a noisy 1-degree field.
        """
        geoms, frames, names = self._scene(1.0, noise=8.0)

        basis0, _ = build_basis(geoms[0], instrumental=True)
        single, _ = fit_sky_model(frames[0], basis0)
        joint, joint_names, _ = fit_sky_model_multi(frames, geoms)

        self.assertLess(self._error(joint_names, joint),
                        0.5 * self._error(names, single),
                        "sharing coefficients across epochs should resolve a "
                        "degeneracy a single frame cannot")

    def test_detector_term_is_recovered(self):
        geoms, frames, _ = self._scene(5.0, noise=4.0)
        coeffs, names, _ = fit_sky_model_multi(frames, geoms)
        self.assertAlmostEqual(float(coeffs[names.index('amp_glow_br')]),
                               self.TRUTH['amp_glow_br'], delta=8.0)

    def test_unlit_corners_stay_near_zero(self):
        geoms, frames, _ = self._scene(5.0, noise=4.0)
        coeffs, names, _ = fit_sky_model_multi(frames, geoms)
        for corner in ('amp_glow_tl', 'amp_glow_tr', 'amp_glow_bl'):
            with self.subTest(corner=corner):
                self.assertLess(float(coeffs[names.index(corner)]), 20.0)

    def test_returns_one_model_per_exposure(self):
        geoms, frames, _ = self._scene(5.0, noise=2.0)
        _, _, models = fit_sky_model_multi(frames, geoms)
        self.assertEqual(len(models), len(frames))
        for model in models:
            self.assertEqual(model.shape, (60, 60))

    def test_masked_pixels_are_excluded_from_the_fit(self):
        geoms, frames, names = self._scene(5.0, noise=2.0)
        # Drop a bright blob into every frame and mask it out; the fit must
        # be unmoved by it.
        masks = []
        for f in frames:
            m = np.zeros(f.shape, dtype=bool)
            m[20:40, 20:40] = True
            f[20:40, 20:40] += 5000.0
            masks.append(m)

        coeffs, joint_names, _ = fit_sky_model_multi(frames, geoms, masks=masks)

        self.assertLess(self._error(joint_names, coeffs), 30.0)

    def test_needs_at_least_two_exposures(self):
        geoms, frames, _ = self._scene(5.0, noise=1.0)
        with self.assertRaises(ValueError):
            fit_sky_model_multi(frames[:1], geoms[:1])

    def test_mismatched_input_lengths_raise(self):
        geoms, frames, _ = self._scene(5.0, noise=1.0)
        with self.assertRaises(ValueError):
            fit_sky_model_multi(frames, geoms[:2])


if __name__ == '__main__':
    unittest.main()
