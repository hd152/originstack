"""Pool workers must see the same masters as the sequential path, scalars
included. Only numpy arrays used to be copied into shared memory, so
``dark_exptime`` never reached a worker and a dark of a different exposure
was subtracted unscaled on the default (ProcessPool) path, silently."""
import numpy as np
from astropy.io import fits

import src.frame_processor as fp


def test_scalar_masters_reach_worker_and_scale_the_dark(tmp_path):
    H = W = 64
    rng = np.random.default_rng(0)
    # sky 1000 + 30 s of dark current (300); the master dark is 10 s (100)
    light = (1300 + rng.normal(0, 5, (H, W))).astype(np.float32)
    path = str(tmp_path / 'l.fits')
    fits.writeto(path, light, fits.Header({'EXPTIME': 30.0, 'BAYERPAT': 'RGGB'}))
    masters = {'bias': None, 'dark': np.full((H, W), 100.0, np.float32),
               'flat': None, 'hot_pixel_map': None, 'dark_exptime': 10.0}

    blocks, specs, scalars = fp._share_masters(masters)
    saved = fp._worker_masters
    try:
        assert scalars == {'dark_exptime': 10.0}
        fp._init_worker_shm(specs, scalar_masters=scalars)
        assert fp._worker_masters['dark_exptime'] == 10.0
        pooled = fp._process_single_frame(path, {}, fp._worker_masters, 'malvar', 'none',
                                          skip_quality=True)
        seq = fp._process_single_frame(path, {'EXPTIME': 30.0}, masters, 'malvar', 'none',
                                       skip_quality=True)
        assert pooled['rgb'][..., 1].mean() == np.float32(seq['rgb'][..., 1].mean())
        assert abs(float(pooled['rgb'][..., 1].mean()) - 1000.0) < 5.0
    finally:
        fp._worker_masters = saved
        for shm in blocks:
            shm.close()
            shm.unlink()
