"""Synthetic training-data generator for the ZOGY transient-triage model
(``--transient-triage``, ``src/transient_triage.py``).

No labelled real transients exist yet, so this bootstraps a training set the
same way this codebase already validates ZOGY itself
(``tests/test_difference_imaging.py``): synthetic star fields, rendered at two
different seeings, run through the *real* ``zogy()`` + ``detect_transients()``
so the stamps a model trains on match what ``run_transient_detection`` actually
produces in production -- not a shortcut simulation of what a candidate stamp
"should" look like.

Four scene kinds, chosen at random per pair:

- ``real``       -- an extra star present only in the new epoch (genuine
                    brightening). Positive label.
- ``cosmic_ray``  -- a single-pixel spike added post-hoc to the new epoch
                    only, with no PSF. Negative label.
- ``dipole``      -- the new epoch's star field is rendered with a small
                    sub-pixel ``(dy, dx)`` offset from the reference, and
                    ``zogy()`` is deliberately given a smaller
                    ``astrometric_sigma`` than the true offset -- an
                    *undersuppressed* registration slip, i.e. a hard negative
                    of exactly the artefact ``astrometric_sigma`` exists to
                    catch. Negative label.
- ``hot_pixel``   -- a fixed-position single-pixel spike in the new epoch's
                    noise realization only (not present in the reference, not
                    aligned with any star). Negative label.

For a ``real`` pair, every OTHER candidate the detector turns up (there can be
more than one, e.g. noise peaks) is also a hard negative -- only the injected
position is positive.

Usage:
    python tools/gen_transient_triage_data.py --n-pairs 4000 --out transient_triage_data.npz
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.difference_imaging import (  # noqa: E402
    detect_transients,
    estimate_background_sigma,
    zogy,
)
from src.transient_triage import DEFAULT_STAMP_SIZE, build_stamps  # noqa: E402

_MATCH_RADIUS_PX = 4.0  # a candidate this close to the injected star is the positive


def _gaussian_psf(size: int, fwhm: float) -> np.ndarray:
    """Normalised Gaussian kernel, odd-sized and centred -- same construction
    as tests/test_difference_imaging.py's own PSF helper, kept independent
    here rather than importing from a test module."""
    if size % 2 == 0:
        size += 1
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    g = np.exp(-(((yy - c) ** 2 + (xx - c) ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def _render_field(shape, stars, fwhm: float, sky: float, noise: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Star field convolved to a given seeing, with Gaussian read noise --
    same shape as tests/test_difference_imaging.py's ``_render_field``."""
    h, w = shape
    img = np.zeros((h, w), dtype=np.float64)
    for y, x, flux in stars:
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < h and 0 <= ix < w:
            img[iy, ix] += flux
    img = fftconvolve(img, _gaussian_psf(21, fwhm), mode='same')
    return img + sky + rng.normal(0.0, noise, (h, w))


def _random_star_field(rng: np.random.Generator, shape, n_stars: int):
    h, w = shape
    stars = []
    while len(stars) < n_stars:
        y, x = rng.uniform(20, h - 20), rng.uniform(20, w - 20)
        if all(math.hypot(y - sy, x - sx) > 15 for sy, sx, _ in stars):
            stars.append((y, x, float(rng.uniform(2000, 12000))))
    return stars


def make_pair(rng: np.random.Generator, size: int = DEFAULT_STAMP_SIZE,
             shape=(160, 180), n_stars: int = 30, max_candidates: int = 6):
    """Build one synthetic (new, ref) pair, run it through the real ZOGY path,
    and return ``(stamps, labels)`` for whatever candidates were detected --
    zero or more per pair, since a bogus-kind pair can turn up nothing and a
    noisy one can turn up spurious hard negatives alongside the label."""
    kind = rng.choice(['real', 'cosmic_ray', 'dipole', 'hot_pixel'])
    stars = _random_star_field(rng, shape, n_stars)
    ref_fwhm, new_fwhm = rng.uniform(2.5, 4.5), rng.uniform(2.5, 4.5)

    ref = _render_field(shape, stars, fwhm=ref_fwhm, sky=0.0, noise=1.0, rng=rng)

    inject_yx = None
    new_stars = list(stars)
    astro_sigma = 0.3

    if kind == 'real':
        h, w = shape
        iy, ix = rng.uniform(20, h - 20), rng.uniform(20, w - 20)
        new_stars.append((iy, ix, float(rng.uniform(3000, 20000))))
        inject_yx = (iy, ix)
        new = _render_field(shape, new_stars, fwhm=new_fwhm, sky=0.0, noise=1.0, rng=rng)
    elif kind == 'dipole':
        dy, dx = rng.uniform(0.4, 1.2) * rng.choice([-1, 1]), rng.uniform(0.4, 1.2) * rng.choice([-1, 1])
        shifted = [(y + dy, x + dx, f) for y, x, f in stars]
        new = _render_field(shape, shifted, fwhm=new_fwhm, sky=0.0, noise=1.0, rng=rng)
        # Undersuppressed on purpose: the true offset is ~0.4-1.2 px/axis,
        # this is the floor ZOGY normally applies when nothing better is
        # measured -- exactly the case that leaves a residual dipole.
        astro_sigma = 0.3
    else:
        new = _render_field(shape, new_stars, fwhm=new_fwhm, sky=0.0, noise=1.0, rng=rng)
        h, w = shape
        py, px = int(rng.uniform(15, h - 15)), int(rng.uniform(15, w - 15))
        spike = float(rng.uniform(4000, 15000))
        if kind == 'cosmic_ray':
            new[py, px] += spike  # no PSF -- a single raw pixel, unlike a star
        else:  # hot_pixel
            new[py, px] += spike * 0.6
            new[py, px] += rng.normal(0.0, 1.0)

    psf_new, psf_ref = _gaussian_psf(21, new_fwhm), _gaussian_psf(21, ref_fwhm)
    result = zogy(new, ref, psf_new, psf_ref, astrometric_sigma=(astro_sigma, astro_sigma))
    candidates = detect_transients(result.score_corr, threshold=5.0,
                                   max_candidates=max_candidates)
    if not candidates:
        return np.zeros((0, 3, size, size), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    labels = np.zeros(len(candidates), dtype=np.float32)
    if inject_yx is not None:
        iy, ix = inject_yx
        for i, c in enumerate(candidates):
            if math.hypot(c.y - iy, c.x - ix) <= _MATCH_RADIUS_PX:
                labels[i] = 1.0

    sigma_new = estimate_background_sigma(new)
    sigma_ref = estimate_background_sigma(ref)
    sigma_diff = estimate_background_sigma(result.difference)
    stamps = build_stamps(new.astype(np.float32), ref.astype(np.float32),
                          result.difference, [(c.y, c.x) for c in candidates],
                          sigma_new, sigma_ref, sigma_diff, size=size)
    return stamps, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-pairs', type=int, default=4000,
                        help='Number of synthetic (new, ref) epoch pairs to generate (default: 4000)')
    parser.add_argument('--size', type=int, default=DEFAULT_STAMP_SIZE,
                        help=f'Stamp size (default: {DEFAULT_STAMP_SIZE}, must match training/inference)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None,
                        help='Output .npz path (default: tools/../transient_triage_data.npz)')
    args = parser.parse_args()

    out = args.out or os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'transient_triage_data.npz'))

    rng = np.random.default_rng(args.seed)
    all_stamps, all_labels = [], []
    for i in range(args.n_pairs):
        stamps, labels = make_pair(rng, size=args.size)
        if len(labels):
            all_stamps.append(stamps)
            all_labels.append(labels)
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{args.n_pairs} pairs '
                 f'({sum(len(l) for l in all_labels)} candidates so far)')

    X = np.concatenate(all_stamps, axis=0) if all_stamps else np.zeros((0, 3, args.size, args.size), dtype=np.float32)
    y = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.float32)
    np.savez(out, X=X, y=y, size=args.size)
    n_pos = int(y.sum())
    print(f'Wrote {len(y)} labelled stamps ({n_pos} real, {len(y) - n_pos} bogus) to {out}')


if __name__ == '__main__':
    main()
