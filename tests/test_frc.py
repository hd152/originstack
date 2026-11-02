"""Split-half measures (src/frc.py) and their use in --noise-validate."""
import numpy as np
from scipy import ndimage

from src.frc import (
    fourier_ring_correlation,
    method_noise,
    resolution_period,
    scale_risk,
    ssnr_from_frc,
    wiener_from_halves,
)
from src.noise_validation import format_summary, validate_noise

N = 256


def _truth(rng, smooth):
    t = ndimage.gaussian_filter(rng.standard_normal((N, N)), smooth, mode='wrap')
    return t / t.std()


def _halves(rng, truth, noise):
    return truth + rng.normal(0, noise, truth.shape), truth + rng.normal(0, noise, truth.shape)


def test_frc_high_where_signal_dominates_and_low_for_noise_only():
    rng = np.random.default_rng(0)
    a, b = _halves(rng, _truth(rng, 3.0), 0.5)
    freq, frc = fourier_ring_correlation(a, b)
    assert frc[2] > 0.9 and frc[-5:].mean() < 0.15
    n1, n2 = rng.standard_normal((N, N)), rng.standard_normal((N, N))
    _f, frc0 = fourier_ring_correlation(n1, n2)
    assert abs(frc0[5:].mean()) < 0.1


def test_smoother_truth_means_coarser_resolution_period():
    rng = np.random.default_rng(1)
    periods = []
    for smooth in (1.5, 3.0, 6.0):
        a, b = _halves(rng, _truth(rng, smooth), 1.0)
        f, c = fourier_ring_correlation(a, b)
        periods.append(resolution_period(f, c))
    assert all(p is not None for p in periods)
    assert periods[0] < periods[1] < periods[2]


def test_resolution_period_none_when_halves_agree_everywhere():
    rng = np.random.default_rng(2)
    t = rng.standard_normal((N, N))
    f, c = fourier_ring_correlation(t, t + 1e-6 * rng.standard_normal((N, N)))
    assert resolution_period(f, c) is None


def test_ssnr_from_frc():
    s = ssnr_from_frc(np.array([0.5, 0.0, 1.0]))
    assert np.allclose(s['half'][:2], [1.0, 0.0]) and np.allclose(s['full'][:2], [2.0, 0.0])
    assert np.isfinite(s['full']).all()


def test_wiener_from_halves_beats_the_plain_mean():
    rng = np.random.default_rng(3)
    t = _truth(rng, 4.0)
    a, b = _halves(rng, t, 1.0)
    w = wiener_from_halves(a, b)
    mean = 0.5 * (a + b)
    assert np.sqrt(np.mean((w - t) ** 2)) < 0.85 * np.sqrt(np.mean((mean - t) ** 2))
    assert abs(float(w.mean()) - float(mean.mean())) < 1e-3


def test_scale_risk_flags_oversmoothing_and_credits_a_good_denoiser():
    rng = np.random.default_rng(4)
    t = _truth(rng, 6.0)
    a, b = _halves(rng, t, 1.0)
    good = scale_risk(a, b, ndimage.gaussian_filter(a, 2.0))
    over = scale_risk(a, b, ndimage.gaussian_filter(a, 14.0))
    assert good['ratio'][0] < 0.5                 # 1-2 px noise: removed
    assert over['ratio'][3] > 1.0                 # 8-16 px structure: destroyed
    assert over['ratio'][3] > good['ratio'][3]


def test_method_noise_tells_noise_from_structure():
    rng = np.random.default_rng(5)
    t = _truth(rng, 6.0)
    noisy = t + rng.standard_normal(t.shape)
    perfect = method_noise(noisy, t)
    smeared = method_noise(noisy, ndimage.gaussian_filter(noisy, 14.0))
    assert abs(perfect['autocorr'][1]) < 0.05 and abs(perfect['corr_with_output']) < 0.1
    assert smeared['autocorr'][1] > 0.3


def test_noise_validation_reports_split_half_resolution():
    rng = np.random.default_rng(6)
    truth = _truth(rng, 2.5)[:128, :128]
    frames = np.stack([np.repeat((truth + rng.normal(0, 1.5, truth.shape))[..., None], 3, axis=2)
                       for _ in range(10)]).astype(np.float32)
    r = validate_noise(frames)
    assert 'frc' in r and r['frc']['period'] is not None
    assert 'split-half resolution' in format_summary(r)
