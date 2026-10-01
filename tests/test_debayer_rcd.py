"""RCD (Ratio Corrected Demosaicing) -- ``--debayer-method rcd``.

No reference implementation is vendored, so it is checked against ground
truth instead: on synthetic scenes it must beat Malvar's interpolation error,
and on a pure-noise mosaic it must keep less of the per-pixel noise (the
reason it was added -- Malvar keeps ~0.99 of it in R/B, which on real Origin
data was most of the noise gap to Siril, whose default is RCD). The native
kernel must match the numpy mirror bit for bit.
"""
import numpy as np
import pytest

from src import debayer as db


def _mosaic(rgb, pattern='RGGB'):
    (ry, rx), (g1y, g1x), (g2y, g2x), (by, bx) = db._PATTERN_OFFSETS[pattern]
    m = np.empty(rgb.shape[:2], np.float32)
    m[ry::2, rx::2] = rgb[ry::2, rx::2, 0]
    m[g1y::2, g1x::2] = rgb[g1y::2, g1x::2, 1]
    m[g2y::2, g2x::2] = rgb[g2y::2, g2x::2, 1]
    m[by::2, bx::2] = rgb[by::2, bx::2, 2]
    return m


def _stars(fwhm, n=25, size=160, seed=2):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    s = fwhm / 2.3548
    t = np.zeros((size, size, 3))
    for _ in range(n):
        cy, cx = rng.uniform(20, size - 20, 2)
        g = 5000.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
        t += np.stack([0.8 * g, g, 1.2 * g], -1)
    return (t + 1000.0).astype(np.float32)


def _mae(out, truth, m=12):
    return float(np.mean(np.abs(out[m:-m, m:-m] - truth[m:-m, m:-m])))


@pytest.mark.parametrize('fwhm', [2.5, 3.5, 4.5])
def test_more_accurate_than_malvar_on_stars(fwhm):
    truth = _stars(fwhm)
    raw = _mosaic(truth)
    assert _mae(db.debayer_rcd(raw, 'RGGB'), truth) < 0.7 * _mae(db.debayer_malvar(raw, 'RGGB'), truth)


def test_keeps_less_noise_than_malvar():
    rng = np.random.default_rng(0)
    raw = (1000 + rng.normal(0, 10, (300, 400))).astype(np.float32)

    def sig(x):
        x = x[20:-20, 20:-20]
        return 1.4826 * np.median(np.abs(x - np.median(x)))
    rcd, mal = db.debayer_rcd(raw, 'RGGB'), db.debayer_malvar(raw, 'RGGB')
    for c in range(3):
        assert sig(rcd[..., c]) < 0.95 * sig(mal[..., c])
    np.testing.assert_allclose(rcd[20:-20, 20:-20].mean(axis=(0, 1)), 1000.0, atol=1.0)


@pytest.mark.parametrize('pattern', ['RGGB', 'BGGR', 'GRBG', 'GBRG'])
def test_patterns_recover_a_flat_colour(pattern):
    truth = np.empty((64, 80, 3), np.float32)
    truth[...] = [300.0, 900.0, 500.0]
    out = db.debayer_rcd(_mosaic(truth, pattern), pattern)
    np.testing.assert_allclose(out, truth, rtol=1e-5)


@pytest.mark.skipif(not (db._HAS_NATIVE and hasattr(db._native, 'debayer_rcd_native')),
                    reason='astro_native debayer_rcd_native not built')
def test_native_debayer_rcd_native_matches_numpy_bit_for_bit():
    rng = np.random.default_rng(4)
    for pattern in ('RGGB', 'BGGR', 'GRBG', 'GBRG'):
        for shape in ((16, 16), (17, 23), (41, 34), (96, 128)):
            raw = rng.uniform(0, 4000, shape).astype(np.float32)
            raw[3, 5] += 30000.0
            if shape == (41, 34):
                raw[7, 9] = np.nan
            a = db._rcd_raw(raw, pattern)
            scale = float(np.nanmax(raw.astype(np.float64)))
            b = db._debayer_rcd_numpy(raw.astype(np.float64), db._PATTERN_OFFSETS[pattern],
                                      scale, raw, pattern)
            np.testing.assert_array_equal(a, b)


def test_small_frames_fall_back_to_malvar():
    raw = np.random.default_rng(1).uniform(0, 100, (10, 12)).astype(np.float32)
    np.testing.assert_array_equal(db.debayer_rcd(raw, 'RGGB'), db.debayer_malvar(raw, 'RGGB'))


def test_grid_correction_applied_like_malvar():
    """RCD's interpolated greens are biased by 2x2 position too; debayer_rcd removes
    it the way debayer_malvar does, leaving no checkerboard on a flat field."""
    rng = np.random.default_rng(6)
    truth = np.full((96, 128, 3), 1000.0, np.float32)
    raw = _mosaic(truth) + rng.normal(0, 3, (96, 128)).astype(np.float32)
    raw[0::2, 1::2] += 6.0                      # G1/G2 imbalance of a real sensor
    out = db.debayer_rcd(raw, 'RGGB')[8:-8, 8:-8, 1]
    q = [float(np.median(out[a::2, b::2])) for a in (0, 1) for b in (0, 1)]
    assert max(q) - min(q) < 1.0


def test_dispatch_and_cli():
    from src.cli import parse_args
    raw = np.random.default_rng(1).uniform(0, 100, (32, 32)).astype(np.float32)
    np.testing.assert_array_equal(db.debayer(raw, 'RGGB', method='rcd'), db.debayer_rcd(raw, 'RGGB'))
    assert parse_args(['-d', 'x', '--debayer-method', 'rcd']).debayer_method == 'rcd'
