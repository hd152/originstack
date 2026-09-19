"""The physical sky model's field-of-view gate, and what it guards.

The single-frame physical sky model was measured to fail on a real 1-degree
field: the components are nearly collinear there (condition ~1e5), it could not
fit the real gradient, and the gradient at that scale is mostly *instrumental*
anyway. The gate declines those fields outright, before doing any work.

(This file used to also cover instrumental detector terms and a multi-frame
joint fit. Neither was reachable from ``src/`` -- only from these tests -- and
both were removed before release, so they have no changelog entry.)
"""

import unittest
import warnings
from unittest import mock

import numpy as np

from src import sky_model
from src.sky_model import (
    _field_of_view_deg,
    _fit_stride,
    basis_condition,
    build_basis,
    build_geometry,
    fit_sky_model,
    gmst_deg,
    julian_date,
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
class TestGateRunsBeforeAnyGeometryWork(unittest.TestCase):
    def test_a_narrow_field_never_builds_geometry_or_a_basis(self):
        """The gate is pure WCS metadata. Geometry built for a field that is
        then declined cost ~1 s and ~580 MB at 6 MP -- and a telescope field
        is essentially always declined, so the common case paid for a result it
        always discarded."""
        img = np.ones((60, 60, 3), dtype=np.float32)
        with mock.patch.object(sky_model, 'build_geometry') as geometry, \
                mock.patch.object(sky_model, 'build_basis') as basis:
            result = remove_physical_sky(
                img, _wcs(1.0, (60, 60)), *_SITE, '2026-08-31T20:40:32-0700')

        self.assertIsNone(result)
        geometry.assert_not_called()
        basis.assert_not_called()


@unittest.skipUnless(HAS_ASTROPY, "astropy required")
class TestAzimuthWrap(unittest.TestCase):
    """Azimuth and helio-ecliptic longitude live on a circle.

    Bilinear interpolation across the 360 -> 0 seam ramps the long way round:
    measured at a 22.5 deg step per pixel (= 360 / the 16 px cell) where the
    true gradient is ~0.05 deg per pixel. Reachable on exactly the fields the
    size gate admits -- a field near due north straddles az = 0 by construction.
    """

    def _north_field(self, dec=70.0):
        lat, lon, when = 45.0, -100.0, '2026-06-15T05:00:00'
        lst = (gmst_deg(julian_date(when)) + lon) % 360.0
        h, w = 400, 600
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crval = [lst, dec]                       # on the meridian, north of zenith
        wcs.wcs.crpix = [w / 2, h / 2]
        wcs.wcs.cdelt = [-0.02, 0.02]                    # ~12 x 8 deg
        return build_geometry(wcs, (h, w), lat, lon, when), (h, w)

    @staticmethod
    def _max_step_on_circle(a):
        return max(float(np.abs((np.diff(a, axis=ax) + 180.0) % 360.0 - 180.0).max())
                   for ax in (0, 1))

    def test_the_field_really_straddles_the_seam(self):
        geom, _ = self._north_field()
        self.assertLess(float(geom.azimuth.min()), 5.0)
        self.assertGreater(float(geom.azimuth.max()), 355.0,
                           "precondition: the test field must cross az = 0")

    def test_azimuth_is_continuous_across_the_seam(self):
        geom, _ = self._north_field()
        self.assertLess(self._max_step_on_circle(geom.azimuth), 1.0,
                        "azimuth ramped the long way round the 360/0 seam")

    def test_helio_longitude_is_continuous_across_the_seam(self):
        # Force the seam: aim at the anti-solar point, where helio_lon ~ 180,
        # then look at the point opposite it, where it wraps through 0.
        lat, lon, when = 45.0, -100.0, '2026-06-15T05:00:00'
        from src.sky_model import sun_position
        sun_ra, sun_dec, _ = sun_position(julian_date(when))
        h, w = 200, 300
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crval = [sun_ra, sun_dec]                # helio_lon ~ 0 here
        wcs.wcs.crpix = [w / 2, h / 2]
        wcs.wcs.cdelt = [-0.04, 0.04]
        geom = build_geometry(wcs, (h, w), lat, lon, when)
        self.assertLess(self._max_step_on_circle(geom.helio_lon), 1.0)

    def test_unwrapped_quantities_are_untouched(self):
        """Zenith angle does not wrap, so it must still interpolate plainly."""
        geom, _ = self._north_field()
        self.assertTrue(np.all(np.diff(geom.zenith_angle, axis=1) < 1.0))


class TestFitOnASample(unittest.TestCase):
    def test_small_frames_are_not_decimated(self):
        self.assertEqual(_fit_stride((60, 60)), 1)
        self.assertEqual(_fit_stride((300, 300)), 1)      # 90k < 100k

    def test_large_frames_are_decimated_to_about_the_budget(self):
        stride = _fit_stride((2000, 3000))                 # 6 MP
        self.assertGreater(stride, 1)
        samples = (2000 // stride) * (3000 // stride)
        self.assertLess(samples, 150_000)
        self.assertGreater(samples, 50_000)

    def test_a_sampled_fit_matches_the_full_fit(self):
        """Measured 65x faster on a 6 MP frame with the dominant coefficient
        agreeing to 0.25%; this pins that the answer does not depend on how
        many pixels the parameters were fit from."""
        rng = np.random.default_rng(0)
        h, w = 500, 700
        yy, xx = np.mgrid[0:h, 0:w]
        ramp_y = 1.0 + 0.4 * yy / h
        ramp_x = 1.0 + 0.2 * xx / w
        basis = np.stack([np.ones((h, w)), ramp_y, ramp_x])
        truth = np.array([50.0, 300.0, 120.0])
        data = np.tensordot(truth, basis, axes=(0, 0)) + rng.normal(0, 1.0, (h, w))

        c_full, m_full = fit_sky_model(data, basis, max_samples=10 ** 9)
        c_samp, m_samp = fit_sky_model(data, basis, max_samples=20_000)

        self.assertEqual(m_samp.shape, (h, w), "the model must stay full-resolution")
        np.testing.assert_allclose(m_samp, m_full, atol=0.5)
        np.testing.assert_allclose(c_samp, c_full, rtol=0.05, atol=2.0)

    def test_the_solver_is_actually_handed_a_sample_not_every_pixel(self):
        """Comparing sampled against full results passes trivially when
        nothing is sampled at all (a mutation that disabled decimation
        survived that test), so assert the thing that matters: how many rows
        reach the non-negative solver."""
        h, w = 500, 700                                    # 350k pixels
        yy, xx = np.mgrid[0:h, 0:w]
        basis = np.stack([np.ones((h, w)), 1 + 0.4 * yy / h, 1 + 0.2 * xx / w])
        data = np.tensordot(np.array([50.0, 300.0, 120.0]), basis, axes=(0, 0))

        seen = []
        real = sky_model._nnls

        def spy(design, target):
            seen.append(design.shape[0])
            return real(design, target)

        with mock.patch.object(sky_model, '_nnls', side_effect=spy):
            fit_sky_model(data, basis, max_samples=20_000)

        self.assertTrue(seen, "the solver should have been called")
        self.assertLess(max(seen), 40_000,
                        f"solver was given {max(seen)} rows of a 350k-pixel frame")

    def test_condition_number_is_stable_under_sampling(self):
        h, w = 400, 500
        yy, xx = np.mgrid[0:h, 0:w]
        basis = np.stack([np.ones((h, w)), 1 + 0.05 * yy / h, 1 + 0.04 * xx / w])
        full = basis_condition(basis, max_samples=10 ** 9)
        sampled = basis_condition(basis, max_samples=10_000)
        self.assertAlmostEqual(np.log10(sampled), np.log10(full), delta=0.15)


@unittest.skipUnless(HAS_ASTROPY, "astropy required")
class TestNnlsFailureDeclines(unittest.TestCase):
    """NNLS is what stops the fit eating a nebula. There must be no quiet
    substitute for it."""

    def _wide_setup(self):
        h = w = 60
        yy, _ = np.mgrid[0:h, 0:w]
        img = np.repeat((1000.0 + yy)[:, :, None], 3, axis=2).astype(np.float32)
        return img, _wcs(20.0, (h, w))

    def test_an_nnls_failure_is_not_replaced_by_an_unbounded_fit(self):
        rows = np.random.default_rng(0).normal(size=(30, 3))
        with mock.patch('scipy.optimize.nnls', side_effect=RuntimeError("did not converge")), \
                mock.patch('numpy.linalg.lstsq') as lstsq:
            with self.assertRaises(RuntimeError):
                sky_model._nnls(rows, rows[:, 0])
        lstsq.assert_not_called()

    def test_remove_physical_sky_declines_and_warns_when_the_fit_fails(self):
        img, wcs = self._wide_setup()
        with mock.patch.object(sky_model, '_nnls', side_effect=RuntimeError("boom")), \
                self.assertLogs('originstack', level='WARNING') as logs:
            result = remove_physical_sky(img, wcs, *_SITE, '2026-08-31T20:40:32-0700')

        self.assertIsNone(result, "a failed fit must hand back to DBE, not raise")
        self.assertTrue(any('fit failed' in m for m in logs.output),
                        "unlike a narrow field this was unexpected: it must be visible")
