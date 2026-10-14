"""tools/bench_vs_siril.py --reference-stack: loading another tool's linear master (e.g. WBPP).

The comparison code expects a (3, H, W) cube on OriginStack's 0-65535 scale, in the same
handedness as OriginStack's stack (its star match is rigid and cannot undo a mirror).
"""
import os
import sys

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import bench_vs_siril as bench  # noqa: E402

from src.xisf_writer import write_xisf  # noqa: E402
from tests._helpers import star_field  # noqa: E402


def _cube(shape=(400, 500)):
    rng = np.random.default_rng(3)
    stars = [(rng.uniform(30, shape[0] - 30), rng.uniform(30, shape[1] - 30), rng.uniform(800, 5000))
             for _ in range(60)]
    img = star_field(shape, stars, 1.6, bg=500.0, noise=6.0, rng=np.random.default_rng(4), dtype=np.float64)
    return np.stack([img, img * 0.9, img * 1.1])


def test_xisf_hwc_normalised_master_becomes_chw_on_16bit_scale(tmp_path):
    cube = _cube()
    path = str(tmp_path / "master.xisf")
    write_xisf(np.moveaxis(cube / 65535.0, 0, -1), path)
    out = bench.load_reference(path)
    assert out.shape == cube.shape
    np.testing.assert_allclose(out, cube, rtol=1e-5)


def test_fits_chw_on_adu_scale_is_left_alone(tmp_path):
    cube = _cube()
    path = str(tmp_path / "master.fits")
    fits.writeto(path, cube.astype(np.float32))
    np.testing.assert_allclose(bench.load_reference(path), cube, rtol=1e-6)


def test_mono_master_is_replicated(tmp_path):
    img = _cube()[0]
    path = str(tmp_path / "mono.fits")
    fits.writeto(path, img.astype(np.float32))
    out = bench.load_reference(path)
    assert out.shape == (3,) + img.shape
    np.testing.assert_allclose(out[2], img, rtol=1e-6)


@pytest.mark.parametrize("mirrored", [False, True])
def test_orient_like_undoes_a_vertical_mirror(mirrored):
    cube = _cube()
    ref = np.ascontiguousarray(cube[:, ::-1]) if mirrored else cube.copy()
    np.testing.assert_array_equal(bench.orient_like(cube, ref), cube)
