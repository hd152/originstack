"""Physics-based sky background model.

Two classes of check here:

1. **Ephemeris accuracy against astropy.** These skip when astropy is absent.
   They exist because the first implementation was wrong by 19.6 deg (Schlyter's
   lunar elements are epoched at 1999-12-31.0, not J2000.0) and nothing else in
   the pipeline would have noticed -- a sky model fits *something* either way.
   Note the frame: this module returns mean-equinox-of-date, so the comparison
   precesses to J2000 first. Skipping that step produces a ~0.35 deg residual
   that looks like an ephemeris bug and is not one.

2. **The property the whole module exists for**: a physical basis has too few
   degrees of freedom to absorb astrophysical signal, so it must remove a
   gradient without eating a frame-filling nebula. A blind surface fit eats it,
   which is the failure this is designed to avoid, and that contrast is
   asserted directly.
"""

import math
import unittest

import numpy as np

from src.sky_model import (
    SkyGeometry,
    _airmass_array,
    altaz_from_equatorial,
    angular_separation_deg,
    basis_condition,
    build_basis,
    describe_fit,
    ecliptic_to_equatorial,
    equatorial_to_ecliptic,
    fit_sky_model,
    gmst_deg,
    julian_date,
    light_pollution_brightness,
    moon_illuminated_fraction,
    moon_position,
    moonlight_brightness,
    sun_position,
    van_rhijn,
    zodiacal_brightness,
)

try:
    import astropy.units as u
    from astropy.coordinates import FK5, ICRS, SkyCoord, get_body, get_sun
    from astropy.time import Time
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False

_DATES = ['2024-01-05 03:00:00', '2025-05-11 21:30:00',
          '2026-12-01 06:15:00', '2027-07-04 01:00:00']


class TestTimeConversions(unittest.TestCase):
    def test_julian_date_at_the_j2000_epoch(self):
        self.assertAlmostEqual(julian_date('2000-01-01 12:00:00'), 2451545.0, places=6)

    def test_julian_date_accepts_iso_t_and_z(self):
        self.assertAlmostEqual(julian_date('2000-01-01T12:00:00Z'), 2451545.0, places=6)

    def test_julian_date_advances_by_one_per_day(self):
        a = julian_date('2026-03-15 00:00:00')
        b = julian_date('2026-03-16 00:00:00')
        self.assertAlmostEqual(b - a, 1.0, places=9)

    def test_julian_date_returns_none_on_garbage(self):
        self.assertIsNone(julian_date('not a timestamp'))
        self.assertIsNone(julian_date(''))

    def test_gmst_is_in_range_and_advances_slightly_faster_than_solar_time(self):
        jd = julian_date('2026-03-15 00:00:00')
        self.assertTrue(0.0 <= gmst_deg(jd) < 360.0)
        # A sidereal day is ~3m56s short of a solar day -> ~0.9856 deg/day drift.
        drift = (gmst_deg(jd + 1.0) - gmst_deg(jd)) % 360.0
        self.assertAlmostEqual(drift, 0.9856, places=2)


class TestCoordinateTransforms(unittest.TestCase):
    def test_ecliptic_equatorial_roundtrip(self):
        eps = 23.4393
        for lon, lat in [(0.0, 0.0), (90.0, 5.0), (200.0, -12.0), (310.0, 3.5)]:
            ra, dec = ecliptic_to_equatorial(lon, lat, eps)
            back_lon, back_lat = equatorial_to_ecliptic(ra, dec, eps)
            self.assertAlmostEqual(float(back_lon) % 360.0, lon % 360.0, places=6)
            self.assertAlmostEqual(float(back_lat), lat, places=6)

    def test_vernal_equinox_maps_to_the_origin(self):
        ra, dec = ecliptic_to_equatorial(0.0, 0.0, 23.4393)
        self.assertAlmostEqual(ra, 0.0, places=6)
        self.assertAlmostEqual(dec, 0.0, places=6)

    def test_object_on_the_meridian_is_due_south_from_mid_northern_latitudes(self):
        # Hour angle 0, dec below the observer's latitude -> due south (az 180).
        alt, az = altaz_from_equatorial(np.array([100.0]), np.array([10.0]),
                                        lat_deg=45.0, lst_deg=100.0)
        self.assertAlmostEqual(float(az[0]), 180.0, places=4)
        self.assertAlmostEqual(float(alt[0]), 90.0 - 45.0 + 10.0, places=4)

    def test_pole_star_sits_at_the_observer_latitude_due_north(self):
        alt, az = altaz_from_equatorial(np.array([0.0]), np.array([90.0]),
                                        lat_deg=47.6, lst_deg=123.0)
        self.assertAlmostEqual(float(alt[0]), 47.6, places=4)
        # Azimuth is degenerate exactly at the pole and comes back as either
        # ~0 or ~360; compare on the circle, not on the real line.
        self.assertAlmostEqual(min(float(az[0]), 360.0 - float(az[0])), 0.0, places=3)

    def test_angular_separation_basics(self):
        def sep(ra1, dec1, ra2, dec2):
            return float(angular_separation_deg(np.array([ra1]), np.array([dec1]),
                                                ra2, dec2)[0])

        self.assertAlmostEqual(sep(10.0, 0.0, 10.0, 0.0), 0.0, places=9)
        self.assertAlmostEqual(sep(0.0, 0.0, 90.0, 0.0), 90.0, places=6)
        self.assertAlmostEqual(sep(0.0, -90.0, 0.0, 90.0), 180.0, places=6)
        # Small separations must stay precise -- the acos form loses ~half the
        # significant digits here, which is why this uses the haversine form.
        self.assertAlmostEqual(sep(0.0, 0.0, 0.001, 0.0), 0.001, places=9)


@unittest.skipUnless(HAS_ASTROPY, "astropy required for ephemeris validation")
class TestEphemerisAgainstAstropy(unittest.TestCase):
    """Ground truth. See the module docstring on the of-date frame."""

    def _to_j2000(self, ra_deg, dec_deg, t):
        return SkyCoord(ra_deg * u.deg, dec_deg * u.deg,
                        frame=FK5(equinox=t)).transform_to(ICRS())

    def test_sun_position_within_a_hundredth_of_a_degree(self):
        for when in _DATES:
            with self.subTest(when=when):
                t = Time(when)
                ra, dec, _ = sun_position(julian_date(when))
                truth = get_sun(t)
                sep = self._to_j2000(ra, dec, t).separation(
                    SkyCoord(truth.ra, truth.dec)).deg
                self.assertLess(sep, 0.02, f"sun off by {sep:.4f} deg at {when}")

    def test_moon_position_within_a_tenth_of_a_degree(self):
        """Regression guard for the 19.6 deg element-epoch bug."""
        for when in _DATES:
            with self.subTest(when=when):
                t = Time(when)
                ra, dec, _, _ = moon_position(julian_date(when))
                truth = get_body('moon', t)
                sep = self._to_j2000(ra, dec, t).separation(
                    SkyCoord(truth.ra, truth.dec)).deg
                self.assertLess(sep, 0.1, f"moon off by {sep:.4f} deg at {when}")

    def test_altaz_matches_astropy(self):
        from astropy.coordinates import AltAz, EarthLocation
        lat, lon = 47.6, -122.3
        when = '2026-03-15 04:30:00'
        t = Time(when)
        loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=50 * u.m)
        ra, dec = 83.82, -5.39                      # M42
        lst = (gmst_deg(julian_date(when)) + lon) % 360.0

        alt, az = altaz_from_equatorial(np.array([ra]), np.array([dec]), lat, lst)
        truth = SkyCoord(ra * u.deg, dec * u.deg).transform_to(
            AltAz(obstime=t, location=loc))

        # ~0.35 deg tolerance: J2000 catalogue coords treated as of-date (see
        # the module docstring) -- irrelevant for 10-degree-scale gradients.
        self.assertLess(abs(float(alt[0]) - truth.alt.deg), 0.4)
        self.assertLess(abs(float(az[0]) - truth.az.deg), 0.4)

    def test_moon_illumination_tracks_the_synodic_month(self):
        """Full at opposition, new at conjunction, one cycle per ~29.53 d."""
        jd0 = julian_date('2026-01-03 10:00:00')     # near full moon
        fracs = [moon_illuminated_fraction(jd0 + d) for d in range(0, 30)]
        self.assertGreater(max(fracs), 0.95)
        self.assertLess(min(fracs), 0.1)


class TestBrightnessComponents(unittest.TestCase):
    def test_van_rhijn_is_unity_at_zenith_and_rises_toward_the_horizon(self):
        self.assertAlmostEqual(float(van_rhijn(0.0)), 1.0, places=6)
        self.assertGreater(float(van_rhijn(60.0)), 1.0)
        self.assertGreater(float(van_rhijn(80.0)), float(van_rhijn(60.0)))
        # A finite emitting layer must stay below the plane-parallel sec(z)
        # limit at every zenith angle -- that bounded growth is the whole
        # difference between this and a naive slab model.
        for z in (30.0, 60.0, 80.0, 85.0):
            self.assertLess(float(van_rhijn(z)), 1.0 / math.cos(math.radians(z)),
                            f"van Rhijn exceeded sec(z) at z={z}")

    def test_airmass_is_one_at_zenith_and_finite_at_the_horizon(self):
        self.assertAlmostEqual(float(_airmass_array(0.0)), 1.0, places=3)
        self.assertGreater(float(_airmass_array(60.0)), 1.9)
        self.assertLess(float(_airmass_array(89.9)), 100.0)

    def test_moonlight_is_zero_when_the_moon_is_down(self):
        rho = np.full((4, 4), 60.0)
        z = np.full((4, 4), 30.0)
        out = moonlight_brightness(rho, moon_alt_deg=-5.0,
                                   target_zenith_deg=z, phase_angle_deg=0.0)
        self.assertTrue(np.all(out == 0.0))

    def test_moonlight_falls_off_with_separation(self):
        z = np.full((3,), 40.0)
        near = moonlight_brightness(np.full((3,), 15.0), 45.0, z, 0.0)
        far = moonlight_brightness(np.full((3,), 110.0), 45.0, z, 0.0)
        self.assertGreater(float(near[0]), float(far[0]))

    def test_full_moon_is_far_brighter_than_a_thin_crescent(self):
        rho, z = np.full((3,), 60.0), np.full((3,), 40.0)
        full = moonlight_brightness(rho, 45.0, z, phase_angle_deg=0.0)
        crescent = moonlight_brightness(rho, 45.0, z, phase_angle_deg=150.0)
        self.assertGreater(float(full[0]), 10.0 * float(crescent[0]))

    def test_zodiacal_light_brightens_toward_the_sun_and_the_ecliptic(self):
        near_sun = float(zodiacal_brightness(np.array([20.0]), np.array([0.0]))[0])
        anti_sun = float(zodiacal_brightness(np.array([170.0]), np.array([0.0]))[0])
        off_plane = float(zodiacal_brightness(np.array([20.0]), np.array([60.0]))[0])
        self.assertGreater(near_sun, anti_sun)
        self.assertGreater(near_sun, off_plane)

    def test_light_pollution_is_brighter_low_and_toward_the_source(self):
        z_low, z_high = np.array([70.0]), np.array([10.0])
        toward = float(light_pollution_brightness(np.array([0.0]), z_low, 0.0)[0])
        away = float(light_pollution_brightness(np.array([180.0]), z_low, 0.0)[0])
        overhead = float(light_pollution_brightness(np.array([0.0]), z_high, 0.0)[0])
        self.assertGreater(toward, away)
        self.assertGreater(toward, overhead)


def _fake_geometry(h=64, w=64, moon_alt=40.0, fov_deg=1.5):
    """Geometry with smooth, physically-shaped gradients across the frame.

    ``fov_deg`` is the angular width of the field. It defaults to 1.5 deg
    because that is what real astrophotography fields look like, and it is
    load-bearing for the "cannot eat a nebula" tests: across a degree or two
    the physical components are nearly linear ramps, which is exactly why
    they cannot reproduce a centrally-peaked bump. An unrealistically wide
    field gives the basis spurious curvature and lets it absorb far more
    signal than it could in practice.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    fy = yy / max(h - 1, 1)
    fx = xx / max(w - 1, 1)
    return SkyGeometry(
        zenith_angle=40.0 + fov_deg * fy,
        azimuth=90.0 + fov_deg * fx,
        moon_sep=60.0 + fov_deg * fx,
        moon_alt=moon_alt,
        helio_lon=60.0 + fov_deg * fx,
        ecl_lat=10.0 + fov_deg * fy,
        jd=julian_date('2026-03-15 04:30:00'),
        phase_angle=30.0)


class TestBasisAndFit(unittest.TestCase):
    def test_basis_terms_are_normalised_to_unit_mean(self):
        basis, names = build_basis(_fake_geometry())
        self.assertIn('constant', names)
        self.assertIn('airglow', names)
        for i, name in enumerate(names):
            self.assertAlmostEqual(float(basis[i].mean()), 1.0, places=6,
                                   msg=f"{name} not unit-mean")

    def test_moonlight_term_is_dropped_when_the_moon_is_down(self):
        _, names = build_basis(_fake_geometry(moon_alt=-10.0))
        self.assertNotIn('moonlight', names)

    def test_fit_recovers_known_coefficients(self):
        geom = _fake_geometry()
        basis, names = build_basis(geom)
        truth = np.array([100.0, 30.0, 12.0, 5.0, 8.0][:len(names)], dtype=np.float64)
        synthetic = np.tensordot(truth, basis, axes=(0, 0))

        coeffs, model = fit_sky_model(synthetic.astype(np.float32), basis)

        np.testing.assert_allclose(coeffs, truth, rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(model, synthetic, rtol=1e-4, atol=1e-3)

    def test_fit_is_robust_to_stars_sitting_above_the_sky(self):
        geom = _fake_geometry()
        basis, names = build_basis(geom)
        truth = np.array([100.0, 30.0, 12.0, 5.0, 8.0][:len(names)], dtype=np.float64)
        sky = np.tensordot(truth, basis, axes=(0, 0))

        contaminated = sky.copy()
        rng = np.random.default_rng(0)
        ys = rng.integers(0, sky.shape[0], 60)
        xs = rng.integers(0, sky.shape[1], 60)
        contaminated[ys, xs] += 5000.0            # stars

        coeffs, _ = fit_sky_model(contaminated.astype(np.float32), basis)

        # Upward-only clipping must reject the stars rather than be dragged up.
        np.testing.assert_allclose(coeffs, truth, rtol=0.05, atol=1.0)

    def test_fit_raises_when_almost_everything_is_masked(self):
        geom = _fake_geometry(h=16, w=16)
        basis, _ = build_basis(geom)
        data = np.ones((16, 16), dtype=np.float32)
        mask = np.ones((16, 16), dtype=bool)
        mask[0, 0] = False
        with self.assertRaises(ValueError):
            fit_sky_model(data, basis, mask=mask)

    def test_describe_fit_names_the_dominant_component(self):
        names = ['constant', 'airglow', 'moonlight', 'zodiacal']
        text = describe_fit(np.array([100.0, 1.0, 90.0, 2.0]), names)
        self.assertIn('moonlight', text)
        self.assertTrue(text.index('moonlight') < text.index('airglow'),
                        "components should be ordered by contribution")
        self.assertNotIn('constant', text)

    def test_describe_fit_refuses_to_attribute_an_ill_conditioned_basis(self):
        """Over a small field the components are near-collinear.

        The fit is still well-posed, but the *split* between components is
        not -- NNLS can pile the whole gradient onto whichever term it likes
        (observed: 'zodiacal 100%' on a 0.04 deg field). Reporting that as a
        confident breakdown would be overclaiming.
        """
        names = ['constant', 'airglow', 'moonlight', 'zodiacal']
        coeffs = np.array([1000.0, 0.0, 0.0, 1133.0])

        text = describe_fit(coeffs, names, condition=1.0e5)

        self.assertIn('not identifiable', text)
        self.assertNotIn('100%', text)

    def test_describe_fit_attributes_freely_when_well_conditioned(self):
        names = ['constant', 'airglow', 'moonlight']
        text = describe_fit(np.array([100.0, 10.0, 90.0]), names, condition=5.0)
        self.assertIn('moonlight', text)
        self.assertNotIn('not identifiable', text)

    def test_realistic_field_is_reported_as_unidentifiable(self):
        """The gate must actually fire on a real field, not just in theory."""
        basis, names = build_basis(_fake_geometry(fov_deg=1.5))
        self.assertGreater(basis_condition(basis), 1.0e3)


class TestDoesNotEatExtendedSignal(unittest.TestCase):
    """The property the module exists for.

    A physical basis has ~5 degrees of freedom whose shapes are fixed by
    geometry. A nebula is not in that span, so the fit cannot absorb one --
    unlike a free-form surface, which is exactly what removed half the
    nebulosity from a real Lagoon session.
    """

    def _scene(self, h=96, w=96):
        geom = _fake_geometry(h, w)
        basis, names = build_basis(geom)
        coeffs = np.array([1000.0, 200.0, 150.0, 40.0, 90.0][:len(names)])
        gradient = np.tensordot(coeffs, basis, axes=(0, 0))

        yy, xx = np.mgrid[0:h, 0:w]
        nebula = 300.0 * np.exp(-(((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
                                  / (2.0 * (h / 3.5) ** 2)))
        return geom, basis, gradient, nebula

    def test_frame_filling_nebula_survives_the_subtraction(self):
        geom, basis, gradient, nebula = self._scene()
        image = (gradient + nebula).astype(np.float32)

        _, model = fit_sky_model(image, basis)
        residual = image - (model - float(np.median(model)))

        # The nebula's peak-to-edge contrast must survive. A blind fitter
        # flattens this toward zero.
        centre = float(residual[residual.shape[0] // 2, residual.shape[1] // 2])
        corner = float(residual[2, 2])
        recovered_contrast = centre - corner
        true_contrast = float(nebula[nebula.shape[0] // 2, nebula.shape[1] // 2]
                              - nebula[2, 2])
        # Measured at ~98% with the non-negativity constraint in place. The
        # threshold is set well above what an unbounded fit achieves (-25%,
        # i.e. it inverts the nebula) so a regression that drops the NNLS
        # constraint fails here loudly.
        self.assertGreater(recovered_contrast, 0.90 * true_contrast,
                           "physical model absorbed the nebula")

    def test_a_blind_surface_fit_eats_what_the_physical_model_keeps(self):
        """Contrast check: the same scene, fitted with a flexible surface."""
        geom, basis, gradient, nebula = self._scene()
        image = (gradient + nebula).astype(np.float32)
        h, w = image.shape

        # A modest free-form polynomial surface -- far fewer degrees of freedom
        # than DBE's mesh, and it still swallows most of the nebula.
        yy, xx = np.mgrid[0:h, 0:w]
        ny, nx = yy / h, xx / w
        cols = [np.ones_like(ny), ny, nx, ny * nx, ny ** 2, nx ** 2,
                ny ** 2 * nx, nx ** 2 * ny, ny ** 3, nx ** 3]
        design = np.stack([c.ravel() for c in cols], axis=1)
        beta, *_ = np.linalg.lstsq(design, image.ravel().astype(np.float64), rcond=None)
        blind_model = (design @ beta).reshape(h, w)
        blind_residual = image - (blind_model - float(np.median(blind_model)))

        blind_contrast = float(blind_residual[h // 2, w // 2] - blind_residual[2, 2])
        true_contrast = float(nebula[h // 2, w // 2] - nebula[2, 2])

        _, phys_model = fit_sky_model(image, basis)
        phys_residual = image - (phys_model - float(np.median(phys_model)))
        phys_contrast = float(phys_residual[h // 2, w // 2] - phys_residual[2, 2])

        self.assertLess(blind_contrast, 0.5 * true_contrast,
                        "blind surface was expected to absorb the nebula")
        self.assertGreater(phys_contrast, 1.5 * blind_contrast,
                           "physical model should preserve far more than the blind fit")

    def test_fitted_coefficients_are_never_negative(self):
        """The constraint that makes the rest of this class work.

        Across a real field the component maps are nearly collinear
        (condition number ~1e5), so an unbounded fit can synthesise an
        interior maximum from large cancelling coefficients. Non-negative
        combinations of monotonic ramps stay monotonic and cannot.
        """
        geom, basis, gradient, nebula = self._scene()
        image = (gradient + nebula).astype(np.float32)

        coeffs, _ = fit_sky_model(image, basis)

        self.assertTrue(np.all(coeffs >= 0.0),
                        f"negative sky component: {coeffs}")

    def test_the_model_never_has_an_interior_maximum(self):
        """A non-negative sum of monotonic ramps has its extremes on the edge."""
        geom, basis, gradient, nebula = self._scene()
        image = (gradient + nebula).astype(np.float32)

        _, model = fit_sky_model(image, basis)

        interior = model[1:-1, 1:-1]
        self.assertLessEqual(float(interior.max()), float(model.max()) + 1e-6)
        # The global max must lie on the border, not in the middle.
        iy, ix = np.unravel_index(int(np.argmax(model)), model.shape)
        on_border = (iy in (0, model.shape[0] - 1)) or (ix in (0, model.shape[1] - 1))
        self.assertTrue(on_border, f"model peaked at interior pixel ({iy}, {ix})")

    def test_pure_gradient_with_no_signal_is_removed_almost_completely(self):
        geom, basis, gradient, _ = self._scene()
        image = gradient.astype(np.float32)

        _, model = fit_sky_model(image, basis)
        residual = image - (model - float(np.median(model)))

        self.assertLess(float(np.ptp(residual)), 0.01 * float(np.ptp(image)))


if __name__ == '__main__':
    unittest.main()
