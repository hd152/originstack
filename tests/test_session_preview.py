"""The preview of a session stack whose Phase 4 is deferred to the multi-night combine
is shown linear; a sky that sits higher in one channel tinted the whole preview (bright
blue at the end of every session on real light-polluted data)."""
import numpy as np

from src.pipeline import _sky_neutral_preview


def test_sky_levels_are_equalised_for_display():
    rng = np.random.default_rng(0)
    sky = np.array([4139.0, 4504.0, 4652.0], dtype=np.float32)     # a real M15 night
    img = (sky + rng.normal(0, 20, (120, 160, 3))).astype(np.float32)
    img[60, 80] += 5000.0                                          # a star keeps its colour offset
    out = _sky_neutral_preview(img)
    med = np.median(out.reshape(-1, 3), axis=0)
    assert np.ptp(med) < 2.0                                       # neutral sky
    assert abs(float(med.mean()) - float(sky.mean())) < 5.0        # same overall level
    np.testing.assert_allclose(out[60, 80] - med, img[60, 80] - np.median(img.reshape(-1, 3), axis=0),
                               atol=3.0)                           # signal above sky unchanged
    assert out.dtype == np.float32 and img[0, 0, 2] > img[0, 0, 0]  # input untouched
