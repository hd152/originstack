"""tools/bench_vs_siril.load_external: another stacker's colour FITS, any channel order."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

fits = pytest.importorskip("astropy.io.fits")


def _write(path, data, extra_hdu_first=False):
    hdus = [fits.PrimaryHDU()] if extra_hdu_first else []
    hdus.append(fits.ImageHDU(data) if hdus else fits.PrimaryHDU(data))
    fits.HDUList(hdus).writeto(path)


def test_channel_last_normalised_is_rescaled(tmp_path):
    from bench_vs_siril import load_external
    rgb = np.random.default_rng(0).uniform(0.0, 0.8, (20, 30, 3)).astype(np.float32)
    p = str(tmp_path / "app.fits")
    _write(p, rgb)
    out = load_external(p)
    assert out.shape == (3, 20, 30)
    np.testing.assert_allclose(out[1], rgb[..., 1].astype(np.float64) * 65535.0, rtol=1e-6)


def test_channel_first_adu_unchanged_in_extension(tmp_path):
    from bench_vs_siril import load_external
    cube = np.random.default_rng(1).uniform(100.0, 4000.0, (3, 16, 24)).astype(np.float32)
    p = str(tmp_path / "dss.fits")
    _write(p, cube, extra_hdu_first=True)       # empty primary, data in the first extension
    out = load_external(p)
    np.testing.assert_allclose(out, cube.astype(np.float64))


def test_mono_stack_refused(tmp_path):
    from bench_vs_siril import load_external
    p = str(tmp_path / "mono.fits")
    _write(p, np.zeros((16, 24), np.float32))
    with pytest.raises(SystemExit):
        load_external(p)
