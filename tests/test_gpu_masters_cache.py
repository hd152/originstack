"""With --use-gpu, the GPU-side copies of the calibration masters were uploaded
only the first time a frame shape was seen and keyed on (shape, dtype), so the
next target in the same process -- a hierarchical run, or a second desktop-app
job -- with the same sensor was calibrated with the previous target's dark and
flat, even when it had no dark at all. A fake GPU (numpy as xp) drives the real
_process_single_frame GPU branch."""
import numpy as np
import pytest
from astropy.io import fits

import src.frame_processor as fp


class _FakeGpu:
    active = True
    xp = np

    def is_oom(self, exc):
        return False

    def free_pool(self):
        pass

    def to_host(self, arr):
        return arr


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(fp, 'get_gpu', lambda: _FakeGpu())
    monkeypatch.setattr(fp, '_probe_gpu_calibration', lambda h, w, g, m: True)
    monkeypatch.setattr(fp, '_gpu_calib_cache', {})
    monkeypatch.setattr(fp, '_gpu_masters', {})
    monkeypatch.setattr(fp, '_gpu_masters_sig', None)


def test_next_target_does_not_reuse_previous_gpu_masters(tmp_path, fake_gpu):
    h = w = 64
    light = (1000 + np.random.default_rng(1).normal(0, 5, (h, w))).astype(np.float32)
    path = str(tmp_path / 'l.fits')
    fits.writeto(path, light, fits.Header({'EXPTIME': 10.0, 'BAYERPAT': 'RGGB'}))

    with_dark = {'bias': None, 'dark': np.full((h, w), 300.0, np.float32), 'flat': None,
                 'dark_exptime': 10.0}
    no_dark = {'bias': None, 'dark': None, 'flat': None, 'dark_exptime': None}
    other_dark = {'bias': None, 'dark': np.full((h, w), 100.0, np.float32), 'flat': None,
                  'dark_exptime': 10.0}

    means = [float(fp._process_single_frame(path, {}, m, 'malvar', 'none',
                                            skip_quality=True)['rgb'].mean())
             for m in (with_dark, no_dark, other_dark)]
    assert means[0] == pytest.approx(700.0, abs=5.0)
    assert means[1] == pytest.approx(1000.0, abs=5.0)   # was 700: target A's dark
    assert means[2] == pytest.approx(900.0, abs=5.0)    # same shape/dtype as A's dark
