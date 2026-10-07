import numpy as np
from scipy.signal import fftconvolve

from originstack import (
    FrameInfo,
    calculate_shift,
    compute_quality_metrics,
    estimate_psf,
    make_synthetic_psf,
    richardson_lucy_deconvolve,
    select_matching_darks,
)
from tests._helpers import add_gaussian_stars, star_field


def make_star_image(shape=(64, 64), centers=((32, 32),), amp=1000.0):
    im = np.zeros(shape, dtype=np.float32)
    add_gaussian_stars(im, [(y, x, amp) for y, x in centers], sigma=2.0)
    im += 100.0
    return im


def test_calculate_shift_recovery():
    im = make_star_image((64, 64), centers=[(20, 20)])
    im2 = np.roll(np.roll(im, 3, axis=0), -2, axis=1)
    sy, sx = calculate_shift(im, im2, upsample=1)
    # im2 was rolled down 3 and left 2; expected shift to align is (-3, +2)
    # Allow 1.5 pixel tolerance for upsample=1 with cross-correlation fallback
    assert abs(sy + 3) <= 1.5
    assert abs(sx - 2) <= 1.5


def test_calculate_shift_fft_subpixel_refines_imperfect_seed():
    # Regression test: the manual FFT cross-correlation branch's parabolic
    # sub-pixel refinement (registration.py, sub_y/sub_x) once had a sign
    # error that returned the *negative* of the correct sub-pixel offset.
    # It was invisible when the seed was already near-exact (offset ~0, so
    # the sign of a near-zero correction doesn't matter) but corrupted
    # accuracy whenever the coarse seed was off by any real sub-pixel
    # amount -- which calculate_shift_pyramid always is, since it only
    # returns integer-pixel shifts. This is the sole source of sub-pixel
    # accuracy on the production default path (skip_phase_cc=True).
    from scipy import ndimage
    rng = np.random.default_rng(42)
    shape = (128, 160)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    ref = np.full(shape, 200.0)
    for _ in range(25):
        cy, cx = rng.uniform(15, shape[0] - 15), rng.uniform(15, shape[1] - 15)
        amp = rng.uniform(500, 3000)
        ref += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2))
    ref = ref.astype(np.float32)

    true_shift = (2.4, -3.7)
    want = (-true_shift[0], -true_shift[1])
    img = ndimage.shift(ref, shift=true_shift, order=3, mode='constant', cval=0.0).astype(np.float32)

    # Deliberately imperfect seed (mimics an integer-only pyramid seed).
    seed = (round(want[0]), round(want[1]))
    sy, sx = calculate_shift(ref, img, skip_phase_cc=True, seed_shift=seed,
                              masked_correlation=False)
    assert abs(sy - want[0]) < 0.1
    assert abs(sx - want[1]) < 0.1


def test_quality_metrics_counts_stars():
    im = make_star_image((64, 64), centers=[(20, 20), (40, 10)])
    metrics = compute_quality_metrics(im)
    assert metrics['brightness'] > 0
    assert metrics['contrast'] > 0
    assert metrics['score'] > 0


# ---------------------------------------------------------------------------
# PSF Estimation Tests
# ---------------------------------------------------------------------------

def _make_star_field_rgb(shape=(128, 128), n_stars=15, sigma=2.0, amp=500.0, bg=100.0):
    """Create an RGB star field with known Gaussian PSF for testing."""
    rng = np.random.RandomState(42)
    margin = 20
    positions = [(rng.randint(margin, shape[0] - margin), rng.randint(margin, shape[1] - margin))
                 for _ in range(n_stars)]
    lum = star_field(shape, [(y, x, amp) for y, x in positions], sigma, bg=bg, dtype=np.float64)
    rgb = np.stack([lum, lum, lum], axis=2).astype(np.float32)
    return rgb, positions


def _make_mock_sources(positions, fluxes=None):
    """Create a mock star source table compatible with estimate_psf."""
    from astropy.table import Table
    n = len(positions)
    if fluxes is None:
        fluxes = [500.0] * n
    return Table({
        'xcentroid': [float(p[1]) for p in positions],
        'ycentroid': [float(p[0]) for p in positions],
        'flux': fluxes,
    })


def test_estimate_psf_gaussian_recovery():
    """Verify that estimate_psf recovers known Gaussian FWHM within 20%."""
    true_sigma = 2.5
    true_fwhm = 2.355 * true_sigma  # ~5.89 px
    rgb, positions = _make_star_field_rgb(shape=(128, 128), n_stars=12,
                                          sigma=true_sigma, amp=800.0)
    sources = _make_mock_sources(positions)
    psf, fwhm = estimate_psf(rgb, sources, model='gaussian')
    assert psf is not None, "PSF estimation returned None"
    assert psf.shape[0] == psf.shape[1], "PSF kernel must be square"
    assert abs(psf.sum() - 1.0) < 1e-6, "PSF must be normalized"
    assert abs(fwhm - true_fwhm) / true_fwhm < 0.20, \
        f"Recovered FWHM {fwhm:.2f} vs true {true_fwhm:.2f} (>20% error)"


def test_estimate_psf_moffat():
    """Verify Moffat PSF estimation returns a valid kernel."""
    rgb, positions = _make_star_field_rgb(shape=(128, 128), n_stars=12, sigma=2.0)
    sources = _make_mock_sources(positions)
    psf, fwhm = estimate_psf(rgb, sources, model='moffat')
    assert psf is not None, "Moffat PSF estimation returned None"
    assert fwhm > 0, "FWHM must be positive"
    assert abs(psf.sum() - 1.0) < 1e-6, "PSF must be normalized"


def test_estimate_psf_insufficient_stars():
    """With fewer than RL_PSF_MIN_STARS, estimate_psf should return None."""
    rgb, positions = _make_star_field_rgb(shape=(128, 128), n_stars=2, sigma=2.0)
    sources = _make_mock_sources(positions[:2])
    psf, fwhm = estimate_psf(rgb, sources, model='gaussian')
    assert psf is None
    assert fwhm == 0.0


def test_make_synthetic_psf():
    """Synthetic PSF should be normalized and symmetric."""
    psf = make_synthetic_psf(fwhm=4.0, psf_size=21, model='gaussian')
    assert psf.shape == (21, 21)
    assert abs(psf.sum() - 1.0) < 1e-6
    # Should be symmetric
    assert np.allclose(psf, psf[::-1, :], atol=1e-10)
    assert np.allclose(psf, psf[:, ::-1], atol=1e-10)

    psf_m = make_synthetic_psf(fwhm=4.0, psf_size=21, model='moffat')
    assert psf_m.shape == (21, 21)
    assert abs(psf_m.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Richardson-Lucy Deconvolution Tests
# ---------------------------------------------------------------------------

def test_richardson_lucy_sharpens():
    """RL deconvolution should increase image sharpness (Laplacian variance)."""
    from scipy.ndimage import laplace
    # Create a test image and blur it
    rgb, _ = _make_star_field_rgb(shape=(64, 64), n_stars=5, sigma=1.5, amp=500.0)
    psf = make_synthetic_psf(fwhm=4.0, psf_size=15, model='gaussian')
    blurred = np.empty_like(rgb)
    for c in range(3):
        blurred[:, :, c] = fftconvolve(rgb[:, :, c], psf, mode='same')

    # Deconvolve
    recovered = richardson_lucy_deconvolve(blurred, psf, iterations=15)

    # Sharpness measured by Laplacian variance — should increase after deconvolution
    def sharpness(img):
        lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return float(np.var(laplace(lum)))

    sharp_blurred = sharpness(blurred)
    sharp_recovered = sharpness(recovered)
    assert sharp_recovered > sharp_blurred, \
        f"RL should increase sharpness: blurred={sharp_blurred:.2f}, recovered={sharp_recovered:.2f}"


def test_richardson_lucy_preserves_flux():
    """Total flux should be approximately preserved by RL deconvolution."""
    rgb, _ = _make_star_field_rgb(shape=(64, 64), n_stars=5, sigma=2.0)
    psf = make_synthetic_psf(fwhm=4.0, psf_size=15)
    result = richardson_lucy_deconvolve(rgb, psf, iterations=10)
    flux_in = float(rgb.sum())
    flux_out = float(result.sum())
    # Allow 5% tolerance
    assert abs(flux_out - flux_in) / flux_in < 0.05, \
        f"Flux not preserved: {flux_in:.0f} -> {flux_out:.0f}"


def test_richardson_lucy_star_mask():
    """Star-masked regions should remain close to input."""
    rgb, positions = _make_star_field_rgb(shape=(64, 64), n_stars=3, sigma=2.0)
    psf = make_synthetic_psf(fwhm=4.0, psf_size=15)
    # Create a mask that's 1.0 everywhere (all "star")
    mask = np.ones(rgb.shape[:2], dtype=np.float64)
    result = richardson_lucy_deconvolve(rgb, psf, iterations=10, star_mask=mask)
    # With full mask, luminance is preserved but YCbCr round-trip introduces
    # small floating-point differences in chrominance channels
    np.testing.assert_allclose(result, rgb, atol=0.05)


# ---------- select_matching_darks tests ----------

def _make_frame(ftype, iso=None, exptime=None, naxis1=100, naxis2=100):
    """Helper to create a FrameInfo with given header properties."""
    hdr = {'NAXIS1': naxis1, 'NAXIS2': naxis2}
    if iso is not None:
        hdr['ISOSPEED'] = iso
    if exptime is not None:
        hdr['EXPTIME'] = exptime
    return FrameInfo(path=f'fake_{ftype}.fits', type=ftype, header=hdr)


def test_select_matching_darks_filters_by_iso():
    """When darks have mixed ISOs, select only those matching the lights."""
    lights = [_make_frame('light', iso=800, exptime=30.0) for _ in range(5)]
    darks_800 = [_make_frame('dark', iso=800, exptime=30.0) for _ in range(3)]
    darks_1600 = [_make_frame('dark', iso=1600, exptime=30.0) for _ in range(2)]
    all_darks = darks_800 + darks_1600

    selected = select_matching_darks(lights, all_darks)
    assert len(selected) == 3
    for d in selected:
        assert d.header['ISOSPEED'] == 800


def test_select_matching_darks_prefers_matching_exposure():
    """When ISO is the same but exposures differ, prefer matching exposure."""
    lights = [_make_frame('light', iso=800, exptime=60.0) for _ in range(5)]
    darks_60s = [_make_frame('dark', iso=800, exptime=60.0) for _ in range(2)]
    darks_30s = [_make_frame('dark', iso=800, exptime=30.0) for _ in range(3)]
    all_darks = darks_60s + darks_30s

    selected = select_matching_darks(lights, all_darks)
    assert len(selected) == 2
    for d in selected:
        assert d.header['EXPTIME'] == 60.0


def test_select_matching_darks_returns_all_when_uniform():
    """When all darks have the same properties, return them all."""
    lights = [_make_frame('light', iso=800, exptime=30.0) for _ in range(5)]
    darks = [_make_frame('dark', iso=800, exptime=30.0) for _ in range(4)]

    selected = select_matching_darks(lights, darks)
    assert len(selected) == 4


def test_select_matching_darks_no_lights():
    """With no lights, return all darks unchanged."""
    darks = [_make_frame('dark', iso=800), _make_frame('dark', iso=1600)]
    selected = select_matching_darks([], darks)
    assert len(selected) == 2


def test_select_matching_darks_iso_over_exposure():
    """ISO match should be prioritized over exposure match."""
    lights = [_make_frame('light', iso=800, exptime=60.0) for _ in range(5)]
    # Wrong ISO but right exposure
    darks_wrong_iso = [_make_frame('dark', iso=1600, exptime=60.0) for _ in range(3)]
    # Right ISO but wrong exposure
    darks_right_iso = [_make_frame('dark', iso=800, exptime=30.0) for _ in range(2)]
    all_darks = darks_wrong_iso + darks_right_iso

    selected = select_matching_darks(lights, all_darks)
    assert len(selected) == 2
    for d in selected:
        assert d.header['ISOSPEED'] == 800
