"""--starless-process: denoise / local-contrast the starless layer, add stars back."""
import argparse

import numpy as np

from src.denoising import multiscale_local_contrast
from src.postprocess import _split_starless


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    img = (300 + rng.normal(0, 10, (96, 96, 3))).astype(np.float32)
    yy, xx = np.mgrid[:96, :96]
    src = []
    for cy, cx in ((20, 20), (60, 70), (30, 80)):
        img += (4000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0))[..., None].astype(np.float32)
        src.append((cy, cx))
    dt = np.dtype([('xcentroid', 'f8'), ('ycentroid', 'f8'), ('peak', 'f8'), ('flux', 'f8')])
    sources = np.array([(cx, cy, 4300.0, 9000.0) for cy, cx in src], dtype=dt)
    return img, sources


def _args(**kw):
    return argparse.Namespace(**{'starless_process': True, 'skip_step': [], **kw})


def test_layers_sum_back_exactly():
    img, sources = _scene()
    starless, stars = _split_starless(img, sources, [], _args())
    assert starless is not None
    np.testing.assert_allclose(starless + stars, img, atol=1e-3)
    # stars really came out of the starless layer
    assert starless[20, 20].max() < 0.2 * img[20, 20].max()


def test_off_or_skipped_or_no_sources_is_a_noop():
    img, sources = _scene()
    assert _split_starless(img, sources, [], _args(starless_process=False)) == (None, None)
    assert _split_starless(img, sources, [], _args(skip_step=['starless_process'])) == (None, None)
    assert _split_starless(img, None, [], _args()) == (None, None)
    assert _split_starless(img, sources[:0], [], _args()) == (None, None)


def test_local_contrast_clip_can_be_disabled():
    img, _ = _scene()
    default = multiscale_local_contrast(img, strength=0.7)
    explicit = multiscale_local_contrast(img, strength=0.7, detail_clip_percentile=99.0)
    np.testing.assert_array_equal(default, explicit)   # default unchanged
    unclipped = multiscale_local_contrast(img, strength=0.7, detail_clip_percentile=None)
    assert not np.array_equal(default, unclipped)
