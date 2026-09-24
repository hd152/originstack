"""The streaming (--stream) sigma-clip accumulator must not freeze a pixel on
its first sample. With one accepted sample, M2 = 0 and the spread estimate
floored at 1e-6, so every later sample was rejected forever: a border pixel
only one burn-in frame covered stayed at that single noisy value for the whole
stream. Samples are now accepted unconditionally until a pixel holds
ONLINE_CLIP_MIN_SAMPLES of them. Checked on the numpy path and, when built,
the native kernel (which must agree)."""
import numpy as np
import pytest

from src import stacking as st


def _paths():
    yield 'numpy'
    if st.HAS_NATIVE:
        yield 'native'


def _run(path, fn, *a, **kw):
    had = st.HAS_NATIVE
    st.HAS_NATIVE = had and path == 'native'
    try:
        return fn(*a, **kw)
    finally:
        st.HAS_NATIVE = had


@pytest.mark.parametrize('path', list(_paths()))
def test_pixel_seeded_by_one_frame_keeps_accumulating(path):
    rng = np.random.default_rng(0)
    k, h, w, c = 10, 8, 8, 3
    burn = (100 + rng.normal(0, 5, (k, h, w, c))).astype(np.float32)
    cov = np.ones((k, h, w), np.float32)
    cov[1:, 0, 0] = 0.0                  # pixel (0,0): only the first burn-in frame
    mean, m2, n_acc, _ = _run(path, st.online_sigma_clip_seed_burnin, burn, cov, sigma=3.0)
    assert n_acc[0, 0, 0] == 1.0

    full = np.ones((h, w), np.float32)
    for _ in range(50):
        frame = (100 + rng.normal(0, 5, (h, w, c))).astype(np.float32)
        _run(path, st.online_sigma_clip_fold_frame, mean, m2, n_acc, frame, full, sigma=3.0)
    # Was stuck at 1 for good. Not ~51: three samples give a noisy spread, so
    # some early rejections are expected (40 seeds: mean 47, worst 36 of 51
    # at a 3-sample warm-up, vs 59 of 60 for a burn-in-seeded pixel).
    assert n_acc[0, 0, 0] > 15
    assert abs(mean[0, 0, 0] - 100.0) < 3.0


@pytest.mark.parametrize('path', list(_paths()))
def test_outliers_are_still_rejected_once_warmed_up(path):
    rng = np.random.default_rng(1)
    k, h, w, c = 10, 8, 8, 3
    burn = (100 + rng.normal(0, 5, (k, h, w, c))).astype(np.float32)
    mean, m2, n_acc, _ = _run(path, st.online_sigma_clip_seed_burnin, burn,
                              np.ones((k, h, w), np.float32), sigma=3.0)
    frame = (100 + rng.normal(0, 5, (h, w, c))).astype(np.float32)
    frame[4, 4, :] = 5000.0              # cosmic ray
    before = n_acc[4, 4, 0]
    _run(path, st.online_sigma_clip_fold_frame, mean, m2, n_acc, frame,
         np.ones((h, w), np.float32), sigma=3.0)
    assert n_acc[4, 4, 0] == before
    assert abs(mean[4, 4, 0] - 100.0) < 10.0
