"""Split-half noise measurement and structure confirmation."""
import numpy as np
import pytest
from scipy import ndimage

from src.noise_validation import consistency_map, format_summary, half_means, noise_map, validate_noise


def _stack(n=40, H=128, W=160, sigma=20.0, seed=0, blob=True, corr=0.0):
    """N frames of one sky (flat + a faint blob) with independent gaussian noise;
    ``corr`` smooths each frame's noise to mimic resampling correlation."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    sky = np.full((H, W), 1000.0)
    if blob:
        sky += 60.0 * np.exp(-((yy - 64) ** 2 + (xx - 90) ** 2) / (2 * 12.0 ** 2))
    frames = np.empty((n, H, W, 3), np.float32)
    for j in range(n):
        for c in range(3):
            noise = rng.normal(0, sigma, (H, W))
            if corr:
                noise = ndimage.gaussian_filter(noise, corr)
                noise *= sigma / noise.std()
            frames[j, :, :, c] = sky + noise * (1.0 + 0.1 * c)
    return frames


def test_half_means_split_by_parity():
    f = np.zeros((5, 4, 4, 3), np.float32)
    f[0::2] = 3.0
    f[1::2] = 9.0
    a, b, na, nb = half_means(f)
    assert (na, nb) == (3, 2) and np.allclose(a, 3.0) and np.allclose(b, 9.0)
    with pytest.raises(ValueError):
        half_means(f[:1])


def test_measured_stack_noise_matches_the_truth_per_channel():
    n, sigma = 40, 20.0
    r = validate_noise(_stack(n=n, sigma=sigma))
    for c, mult in enumerate((1.0, 1.1, 1.2)):
        truth = sigma * mult / np.sqrt(n)
        assert abs(r['median_sigma'][c] - truth) / truth < 0.06
    assert r['sigma'].shape == (128, 160, 3)


def test_unequal_halves_are_handled():
    n, sigma = 41, 20.0                                    # 21 vs 20
    r = validate_noise(_stack(n=n, sigma=sigma, seed=1))
    truth = sigma / np.sqrt(n)
    assert abs(r['median_sigma'][0] - truth) / truth < 0.07


def test_local_noise_variation_is_tracked():
    f = _stack(n=40, sigma=10.0, blob=False)
    extra = np.random.default_rng(3).normal(0, 30.0, f[:, :, 80:, :].shape)
    f[:, :, 80:, :] += extra.astype(np.float32)
    r = validate_noise(f, block=32)
    left, right = r['sigma'][:, :50, 0].mean(), r['sigma'][:, 110:, 0].mean()
    assert right > 2.5 * left


def test_correlation_factor_and_common_gradients():
    n, sigma = 40, 20.0
    white = validate_noise(_stack(n=n, sigma=sigma), frame_noise=sigma)
    assert 0.9 < white['correlation_factor'] < 1.25        # channels scale noise 1.0-1.2x
    # a gradient common to all frames cancels in A-B, so it never inflates the noise
    grad = np.linspace(0, 300, 160, dtype=np.float32)[None, None, :, None]
    g = validate_noise(_stack(n=n, sigma=sigma) + grad)
    assert abs(g['median_sigma'][0] - white['median_sigma'][0]) < 0.4


def test_consistency_is_high_on_real_structure_and_low_on_noise():
    cons = validate_noise(_stack(n=60, sigma=25.0))['consistency']
    blob = cons[54:75, 75:106].mean()
    background = cons[5:35, 5:50].mean()
    assert blob > 0.6 and background < 0.35 and blob > background + 0.3
    only_noise = validate_noise(_stack(n=60, sigma=25.0, blob=False, seed=5))['consistency']
    assert only_noise.mean() < 0.35


def test_summary_text_mentions_the_key_numbers():
    s = format_summary(validate_noise(_stack(n=20), frame_noise=20.0))
    assert 'odd/even' in s and 'factor' in s and 'repeatable' in s
    assert 'factor' not in format_summary(validate_noise(_stack(n=20)))


def test_maps_are_finite_and_in_range():
    r = validate_noise(_stack(n=12, H=64, W=64))
    assert np.isfinite(r['sigma']).all() and (r['sigma'] > 0).all()
    assert 0.0 <= float(r['consistency'].min()) and float(r['consistency'].max()) <= 1.0
    a, b, na, nb = half_means(_stack(n=8, H=40, W=40))
    assert consistency_map(a, b).shape == (40, 40)
    assert noise_map(a, b, na, nb).shape == (40, 40, 3)
