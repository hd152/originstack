"""Per-frame transparency from a fixed star ensemble."""
import numpy as np

from src.affine_fit import RigidTransform
from src.transparency import match_flux_ratio, measure_transparency, to_aligned_yx, transparency_keep_mask

_DT = np.dtype([('xcentroid', 'f8'), ('ycentroid', 'f8'), ('flux', 'f8')])


def _field(n=120, H=800, W=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, x = rng.uniform(30, H - 30, n), rng.uniform(30, W - 30, n)
    flux = 10 ** rng.uniform(3.0, 4.6, n)
    return y, x, flux


def _stars(y, x, flux):
    a = np.zeros(len(y), _DT)
    a['ycentroid'], a['xcentroid'], a['flux'] = y, x, flux
    return a


class _F:
    def __init__(self, stars, exptime=None):
        self.metrics = {'_star_sources': stars}
        self.header = {'EXPTIME': exptime} if exptime else {}


def test_to_aligned_inverts_apply_transform_convention():
    tf = RigidTransform.from_rotation_translation(np.deg2rad(3.0), (11.0, -7.0))
    from src.registration import apply_transform
    img = np.zeros((300, 400), np.float32)
    img[120:123, 210:213] = 1000.0                    # a blob at native (121, 211)
    out = apply_transform(img, transform=tf)
    yy, xx = np.nonzero(out > 100)
    seen = np.array([[yy.mean(), xx.mean()]])
    pred = to_aligned_yx(np.array([[121.0, 211.0]]), tf, None)
    assert np.allclose(seen, pred, atol=0.6)


def test_shift_convention():
    from src.registration import apply_transform
    img = np.zeros((200, 200), np.float32)
    img[50:53, 60:63] = 1000.0
    out = apply_transform(img, shift=(7.0, -4.0))
    yy, xx = np.nonzero(out > 100)
    assert np.allclose([[yy.mean(), xx.mean()]], to_aligned_yx(np.array([[51.0, 61.0]]), None, (7.0, -4.0)), atol=0.6)


def test_recovers_the_flux_ratio_under_rotation_and_shift():
    y, x, flux = _field()
    ref = _stars(y, x, flux)
    tf = RigidTransform.from_rotation_translation(np.deg2rad(2.0), (9.0, -5.0))
    # native positions of the same stars in a rotated/shifted frame: invert the map
    R = tf.params[:2, :2]
    t_rc = np.array([tf.params[1, 2], tf.params[0, 2]])
    native = (np.column_stack([y, x]) - t_rc) @ R.T
    rng = np.random.default_rng(1)
    frame = _stars(native[:, 0] + rng.normal(0, 0.2, len(y)),
                   native[:, 1] + rng.normal(0, 0.2, len(y)), flux * 0.62)
    r, k = match_flux_ratio(ref, frame, tf, None)
    assert k > 40 and abs(r - 0.62) < 0.02


def test_session_median_is_normalised_and_a_cloudy_frame_stands_out():
    y, x, flux = _field()
    ref = _stars(y, x, flux)
    rng = np.random.default_rng(2)
    levels = [1.0, 1.02, 0.98, 1.01, 0.55, 1.0, 0.99]
    frames = []
    for lv in levels:
        frames.append(_F(_stars(y + rng.normal(0, .2, len(y)), x + rng.normal(0, .2, len(y)),
                                flux * lv * rng.normal(1, 0.03, len(y)))))
    summary = measure_transparency(frames, [(0.0, 0.0)] * len(frames), [None] * len(frames), ref)
    v = summary['values']
    assert abs(np.median(v) - 1.0) < 0.03
    assert v[4] < 0.6 and all(x > 0.9 for i, x in enumerate(v) if i != 4)
    assert frames[4].metrics['transparency'] < 0.6 and frames[0].metrics['transparency_n'] > 40
    assert transparency_keep_mask(v, 0.8) == [True, True, True, True, False, True, True]
    assert all(transparency_keep_mask(v, 0.0))


def test_exposure_time_is_normalised():
    y, x, flux = _field()
    ref = _stars(y, x, flux)
    frames = [_F(_stars(y, x, flux * 2.0), exptime=20.0)]      # 2x flux from 2x exposure
    frames += [_F(_stars(y, x, flux * 1.0), exptime=10.0) for _ in range(3)]
    s = measure_transparency(frames, [(0.0, 0.0)] * 4, [None] * 4, ref, ref_frame=_F(ref, 10.0))
    assert np.allclose(s['values'], 1.0, atol=0.02)


def test_missing_catalogues_and_tiny_fields_are_left_alone():
    y, x, flux = _field()
    ref = _stars(y, x, flux)
    empty = _F(None)
    tiny = _F(_stars(y[:5], x[:5], flux[:5]))
    s = measure_transparency([empty, tiny], [(0.0, 0.0)] * 2, [None] * 2, ref)
    assert s['measured'] == 0 and np.isnan(s['values']).all()
    assert transparency_keep_mask(s['values'], 0.8) == [True, True]     # NaN never gated
