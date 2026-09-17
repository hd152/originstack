"""Local contrast must not carve a dark collar around bright stars.

Reported from a real Lagoon run: stars that previously sat cleanly on
nebulosity had acquired black surrounds somewhere between photometric
calibration and the final image. Of the steps in that window only
``multiscale_local_contrast`` runs by default for an emission nebula
(``--auto`` turns star reduction *off* for that target and sets
``local_contrast_strength`` to 0.749).

Mechanism: blurring at the enhancement scales smears a star's core outward,
so in the annulus just beyond the protected core the blurred luminance far
exceeds the original, ``detail`` goes strongly negative, and the enhancement
subtracts real nebulosity. The core is protected; its immediate surroundings
are not, and that discontinuity is the ring.
"""

import unittest

import numpy as np

from src.denoising import multiscale_local_contrast

_STRENGTH = 0.749          # what --auto selects for an emission nebula


def _star_on_nebula(size=201, star_flux=9000.0, fwhm=3.0):
    """A bright star centred on smooth nebulosity, plus a realistic star mask.

    The mask is built the way ``quality.generate_star_mask`` builds it -- a
    narrow Gaussian at the centroid, fwhm ~3 px -- **not** a hard disk. That
    distinction is the point: a first attempt at fixing this artifact worked
    by infilling masked pixels, measured beautifully against a hard-disk mask,
    and changed real output by 0.01% because the real mask is soft and covers
    only ~0.4% of pixels, far narrower than the stellar wings that actually
    pollute a sigma-12 blur. A fixture that cannot reproduce that failure
    cannot protect against it.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2.0
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    nebula = 300.0 + 120.0 * np.exp(-(r2 / (2.0 * 55.0 ** 2)))
    star = star_flux * np.exp(-(r2 / (2.0 * 2.2 ** 2)))
    img = np.stack([nebula + star] * 3, axis=-1).astype(np.float32)

    sigma = fwhm / 2.3548
    mask = np.exp(-(r2 / (2.0 * sigma ** 2))).astype(np.float32)
    return img, mask, np.sqrt(r2)


class TestNoDarkHaloAroundStars(unittest.TestCase):

    def setUp(self):
        self.img, self.mask, self.radius = _star_on_nebula()
        self.before = self.img[:, :, 1].astype(np.float64)

    def _worst_darkening_fraction(self, after, lo=8, hi=30):
        """Deepest relative darkening in the annulus where the ring forms."""
        zone = (self.radius >= lo) & (self.radius <= hi)
        delta = after[zone] - self.before[zone]
        return float(delta.min()) / float(self.before[zone].mean())

    def test_ring_darkening_stays_small(self):
        out = multiscale_local_contrast(
            self.img, strength=_STRENGTH, star_mask=self.mask)[:, :, 1].astype(np.float64)

        worst = self._worst_darkening_fraction(out)
        # Measured 13.6% before the fix, 2.6% after. 6% leaves headroom for
        # the exact fill radius while still failing loudly on a regression.
        self.assertGreater(worst, -0.06,
                           f"dark collar around star: {100 * worst:.1f}% darkening")

    def test_nebulosity_just_outside_the_core_is_preserved(self):
        out = multiscale_local_contrast(
            self.img, strength=_STRENGTH, star_mask=self.mask)[:, :, 1].astype(np.float64)

        for r in (10, 14, 20):
            ring = np.abs(self.radius - r) < 0.5
            kept = float(out[ring].mean()) / float(self.before[ring].mean())
            with self.subTest(radius=r):
                self.assertGreater(kept, 0.94,
                                   f"r={r}px lost {100 * (1 - kept):.1f}% of its flux")

    def test_it_still_enhances_structure_away_from_stars(self):
        """The fix must not buy its win by simply doing less."""
        out = multiscale_local_contrast(
            self.img, strength=_STRENGTH, star_mask=self.mask)[:, :, 1].astype(np.float64)

        far = (self.radius > 60) & (self.radius < 90)
        self.assertGreater(float(out[far].std()), float(self.before[far].std()),
                           "local contrast should still add structure far from stars")

    def test_star_core_itself_is_untouched(self):
        out = multiscale_local_contrast(
            self.img, strength=_STRENGTH, star_mask=self.mask)[:, :, 1].astype(np.float64)
        core = self.radius <= 3
        np.testing.assert_allclose(out[core], self.before[core], rtol=1e-5)

    def test_without_a_star_mask_it_still_runs(self):
        out = multiscale_local_contrast(self.img, strength=_STRENGTH, star_mask=None)
        self.assertEqual(out.shape, self.img.shape)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_zero_strength_is_a_no_op(self):
        out = multiscale_local_contrast(self.img, strength=0.0, star_mask=self.mask)
        np.testing.assert_allclose(out, self.img, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
