"""Noise flattening by the coverage map before the wavelet / NN denoisers."""
import argparse

import numpy as np
from scipy import ndimage

from src.denoising import (
    directional_wavelet_denoise,
    noise_flatten,
    noise_rel_from_coverage,
    noise_restore,
)
from src.postprocess import _noise_rel_map

N = 256


def test_rel_from_coverage():
    cov = np.full((8, 8), 20)
    cov[:, 4:] = 5
    cov[0, 0] = 0
    rel = noise_rel_from_coverage(cov)
    assert rel.dtype == np.float32
    assert np.allclose(rel[1:, :4], 1.0) and np.allclose(rel[1:, 4:], 2.0)
    assert np.isfinite(rel).all() and abs(rel[0, 0] - np.sqrt(20.0)) < 1e-5       # count 0 is treated as 1


def test_flatten_restore_roundtrip_and_flat_noise():
    rng = np.random.default_rng(0)
    rel = np.ones((N, N), np.float32)
    rel[:, N // 2:] = 1.5
    img = (1000 + rng.standard_normal((N, N, 3)) * 5.0 * rel[..., None]).astype(np.float32)
    flat, ped = noise_flatten(img, rel)
    assert abs(flat[:, :N // 2].std() - flat[:, N // 2:].std()) < 0.3
    assert np.allclose(noise_restore(flat, rel, ped), img, atol=1e-3)


def test_wavelet_with_flattening_is_less_speckled_in_the_shallow_half():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    truth = 40.0 * np.exp(-((yy - 128) ** 2 + (xx - 128) ** 2) / (2 * 30.0 ** 2))
    rel = np.ones((N, N), np.float32)
    rel[:, N // 2:] = 1.6
    noise = ndimage.gaussian_filter(rng.standard_normal((N, N, 3)), (0.5, 0.5, 0)) * 5.0
    img = (1000 + truth[..., None] + noise * rel[..., None]).astype(np.float32)
    sky = np.zeros((N, N), bool)
    sky[:60, :] = True
    sky[-60:, :] = True
    right = sky.copy()
    right[:, :N // 2] = False

    def err(out):
        return float(np.sqrt(np.mean(((out.mean(axis=2) - 1000) - truth)[right] ** 2)))

    plain = directional_wavelet_denoise(img)
    flat, ped = noise_flatten(img, rel)
    fixed = noise_restore(directional_wavelet_denoise(flat), rel, ped)
    assert err(fixed) < err(plain)


def test_noise_rel_map_gate():
    img = np.zeros((16, 16, 3), np.float32)
    args = argparse.Namespace(_noise_rel=np.ones((16, 16), np.float32))
    assert _noise_rel_map(args, img) is None                       # uniform depth: untouched
    args._noise_rel = np.full((16, 16), 1.4, np.float32)
    assert _noise_rel_map(args, img) is not None
    args._noise_rel = np.full((8, 8), 1.4, np.float32)
    assert _noise_rel_map(args, img) is None                       # wrong shape: ignored
    args._noise_rel = np.full((16, 16), 1.4, np.float32)
    args.noise_map_denoise = False
    assert _noise_rel_map(args, img) is None
    assert _noise_rel_map(argparse.Namespace(), img) is None


def test_from_stack_reads_the_coverage_file_beside_it(tmp_path):
    from astropy.io import fits

    from src.pipeline import _load_noise_rel
    cov = np.full((12, 20), 30, np.int16)
    cov[:, :4] = 10
    fits.PrimaryHDU(cov).writeto(tmp_path / 'run_coverage.fits')
    args = argparse.Namespace()
    _load_noise_rel(args, str(tmp_path / 'run.fits'), (12, 20))
    assert args._noise_rel.shape == (12, 20) and abs(float(args._noise_rel.max()) - np.sqrt(3.0)) < 1e-5
    other = argparse.Namespace()
    _load_noise_rel(other, str(tmp_path / 'run.fits'), (9, 9))          # not the same grid
    assert not hasattr(other, '_noise_rel')
    _load_noise_rel(other, str(tmp_path / 'absent.fits'), (12, 20))
    assert not hasattr(other, '_noise_rel')
