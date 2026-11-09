"""Compare the CPU and the GPU proper coadd (``--gpu-coadd``) on the same aligned stack.

Needs an astro_native built with the ``gpu`` feature (default since 0.66.0) (see dev-notes/gpu-coadd.md). The stack is
either synthetic (a star field with per-frame seeing, transparency and a few cosmic rays) or a
float32 ``.npy`` of shape (N, H, W, 3) -- the aligned frames -- with an optional ``--reference``
.npy (H, W, 3), the normal stack the coadd rejects against (default: the mean of the frames).

    python tools/ab_gpu_coadd.py --synthetic 96
    python tools/ab_gpu_coadd.py --aligned aligned.npy --reference stack.npy

Prints both timings and how far the two outputs are apart relative to the output's noise.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import proper_coadd as pc  # noqa: E402


def synthetic(n: int, h: int = 1990, w: int = 2945, stars: int = 300, seed: int = 11):
    rng = np.random.default_rng(seed)
    sy, sx = rng.uniform(60, h - 60, stars), rng.uniform(60, w - 60, stars)
    flux = 10 ** rng.uniform(3.5, 5.2, stars)
    col = np.array([1.0, 0.8, 0.6])
    out = np.empty((n, h, w, 3), np.float32)
    r = 20
    for j in range(n):
        fwhm, trans = rng.uniform(2.6, 4.4), rng.uniform(0.7, 1.1)
        img = np.full((h, w, 3), 1000.0, np.float32)
        a = fwhm / (2 * np.sqrt(2 ** (1 / 3.0) - 1))
        for y0, x0, f in zip(sy, sx, flux):
            iy, ix = int(y0), int(x0)
            yy, xx = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
            k = (1 + ((yy - y0) ** 2 + (xx - x0) ** 2) / a ** 2) ** -3.0
            k = k / k.sum() * f * trans
            for c in range(3):
                img[iy - r:iy + r + 1, ix - r:ix + r + 1, c] += (k * col[c]).astype(np.float32)
        img += rng.normal(0, 30, img.shape).astype(np.float32) * np.sqrt(np.maximum(img, 1000) / 1000)
        for _ in range(40):
            img[rng.integers(0, h), rng.integers(0, w), rng.integers(0, 3)] += rng.uniform(2000, 20000)
        out[j] = img
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--synthetic', type=int, metavar='N', help='generate N synthetic frames')
    src.add_argument('--aligned', metavar='NPY', help='float32 (N, H, W, 3) aligned frames')
    ap.add_argument('--reference', metavar='NPY', help='(H, W, 3) normal stack (default: frame mean)')
    ap.add_argument('--fwhm', type=float, default=3.5, help='median star FWHM, px (default 3.5)')
    a = ap.parse_args()

    native = pc._native
    if native is None or not hasattr(native, 'gpu_coadd_begin'):
        print('astro_native has no GPU support: build it with `maturin build --release` (astro_native 0.66+)')
        return 2
    if not native.gpu_available():
        print('no usable GPU adapter')
        return 2
    print('adapter:', native.gpu_info())

    aligned = synthetic(a.synthetic) if a.synthetic else np.load(a.aligned, mmap_mode='r')
    if a.synthetic is None and not aligned.flags.c_contiguous:
        aligned = np.ascontiguousarray(aligned)
    ref = np.load(a.reference) if a.reference else np.asarray(aligned.mean(axis=0), np.float32)
    print('frames:', aligned.shape)

    t = time.time()
    cpu = pc.proper_coadd(aligned, ref, a.fwhm, verbose=False)
    t_cpu = time.time() - t
    pc.proper_coadd(aligned[:6], ref, a.fwhm, verbose=False, gpu=True)           # shader compile
    t = time.time()
    gpu = pc.proper_coadd(aligned, ref, a.fwhm, verbose=False, gpu=True)
    t_gpu = time.time() - t
    if cpu is None or gpu is None:
        print('the coadd declined (PSF could not be measured)')
        return 1

    d = (gpu - cpu).astype(np.float64)
    g = cpu[..., 1].astype(np.float64)
    lag = (g[:, 4:] - g[:, :-4]).ravel()
    noise = 1.4826 * np.median(np.abs(lag - np.median(lag))) / np.sqrt(2)
    above = cpu - float(np.median(cpu))
    bright = above > 0.5 * float(np.percentile(above, 99.99))      # the stars, not the sky
    print(f'time: CPU {t_cpu:.1f} s, GPU {t_gpu:.1f} s (whole proper_coadd call, incl. PSF measurement)')
    print(f'output noise {noise:.3f}; difference rms {np.sqrt(np.mean(d ** 2)):.5f} '
          f'({np.sqrt(np.mean(d ** 2)) / noise:.1e} of noise), max {np.abs(d).max():.4f}, '
          f'p99.99 {np.percentile(np.abs(d), 99.99):.4f}, mean {d.mean():.2e}')
    print(f'bright pixels ({int(bright.sum())}): max relative difference {np.abs(d[bright] / cpu[bright]).max():.2e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
