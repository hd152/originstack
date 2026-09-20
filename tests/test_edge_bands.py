"""remove_edge_bands: localised edge strips go, everything else stays."""
import numpy as np

from src.background import remove_edge_bands


def _sky(H=1000, W=500, sigma=20.0, seed=0):
    rng = np.random.default_rng(seed)
    return (1000 + rng.normal(0, sigma, (H, W, 3))).astype(np.float32), np.ones((H, W), np.float32)


def _band_offset(img, rows):
    return [float(np.median(img[rows][..., c])) for c in range(3)]


def test_removes_a_bottom_strip_and_leaves_the_interior_alone():
    img, sky = _sky()
    strip = img.copy()
    strip[-45:, :, 0] += 13.0
    strip[-45:, :, 1] -= 5.0
    out = remove_edge_bands(strip, sky)
    mid = _band_offset(out, slice(400, 600))
    bot = _band_offset(out, slice(-40, None))
    assert max(abs(b - m) for b, m in zip(bot, mid)) < 2.5        # was 13
    before = _band_offset(strip, slice(-40, None))
    assert abs(before[0] - _band_offset(strip, slice(400, 600))[0]) > 10
    np.testing.assert_array_equal(out[200:700], strip[200:700])    # interior untouched


def test_all_four_edges():
    img, sky = _sky()
    bad = img.copy()
    bad[:30, :, 2] += 15.0
    bad[:, :30, 0] -= 15.0
    bad[:, -30:, 1] += 15.0
    out = remove_edge_bands(bad, sky)
    ref = _band_offset(out, (slice(400, 600), slice(150, 350)))
    assert abs(float(np.median(out[:20, 100:400, 2])) - ref[2]) < 3.0
    assert abs(float(np.median(out[300:700, :20, 0])) - ref[0]) < 3.0
    assert abs(float(np.median(out[300:700, -20:, 1])) - ref[1]) < 3.0


def test_clean_image_is_essentially_unchanged():
    img, sky = _sky()
    out = remove_edge_bands(img, sky)
    assert float(np.abs(out - img).max()) < 3.0


def test_a_smooth_gradient_is_reduced_never_overcorrected():
    """The step runs after background extraction, so a residual ramp is unwanted
    and may be partly flattened near the edges -- but it must never be inverted,
    overshoot, or alter the interior."""
    img, sky = _sky()
    ramp = np.linspace(0, 40, img.shape[0], dtype=np.float32)[:, None, None]
    src = img + ramp
    out = remove_edge_bands(src, sky)
    d_in = float(np.median(src[-20:]) - np.median(src[:20]))
    d_out = float(np.median(out[-20:]) - np.median(out[:20]))
    assert 0.0 < d_out < d_in                          # reduced, same direction
    assert abs(float(np.median(out[400:600])) - float(np.median(src[400:600]))) < 1.0


def test_masked_objects_at_the_edge_do_not_drive_the_correction():
    img, sky = _sky()
    img[-45:, 100:300, :] += 500.0            # a bright object on the edge...
    mask = sky.copy()
    mask[-45:, 100:300] = 0.0                 # ...that the sky mask excludes
    out = remove_edge_bands(img, mask)
    # sky columns beside it are already clean and must stay that way
    assert abs(float(np.median(out[-40:, 350:450])) - float(np.median(out[400:600, 350:450]))) < 3.0


def test_edge_mostly_masked_is_skipped():
    img, sky = _sky()
    img[-45:, :, 0] += 13.0
    mask = sky.copy()
    mask[-45:, :] = 0.0                       # nothing left to measure there
    out = remove_edge_bands(img, mask)
    np.testing.assert_array_equal(out[-45:, :, 0], img[-45:, :, 0])
