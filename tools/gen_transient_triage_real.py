"""Training stamps for the transient-triage model from REAL two-night pairs.

``tools/gen_transient_triage_data.py`` renders synthetic Gaussian star fields;
the model trained on those alone barely separates anything on real stacks
(its probabilities sat in 0.43-0.47 on most real stamps). This builds stamps
from two linear stacks of the same target taken on different nights:

- positives: point sources injected into the new epoch with the new epoch's
  own empirical PSF (median of bright isolated stars), at random positions
  -- including over galaxies, nebulae and star wings -- and fluxes from
  about the detection threshold up;
- negatives: every other candidate ``_compare_epochs`` reports on that pair
  -- real subtraction residuals of real stacks (bright-star and core
  residuals, registration and PSF-mismatch dipoles, edge artefacts, noise).
  A real variable star or asteroid between the nights would be mislabelled
  as bogus; on a handful of nights that is rare enough to accept.

Each pair is run in both directions and over several injection rounds.
The stamps go through the real ``_compare_epochs`` + ``build_stamps``, so they
are what ``--transient-triage`` scores in production.

Usage:
    python tools/gen_transient_triage_real.py --pair NEW.fits REF.fits [--pair ...]
        [--rounds 6] [--per-round 40] --out triage_real.npz
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.difference_imaging import _compare_epochs, estimate_background_sigma  # noqa: E402
from src.merge import load_merge_stack  # noqa: E402
from src.star_detect import detect_stars_matched_filter  # noqa: E402
from src.transient_triage import DEFAULT_STAMP_SIZE, build_stamps  # noqa: E402

_MATCH_RADIUS_PX = 3.0
_PSF_HALF = 12


def empirical_psf(lum: np.ndarray, n: int = 40) -> np.ndarray:
    """Normalised median stamp of bright, isolated, unsaturated stars."""
    s = detect_stars_matched_filter(lum)
    x = np.asarray(s['xcentroid'], float)
    y = np.asarray(s['ycentroid'], float)
    f = np.asarray(s['flux'], float)
    pk = np.asarray(s['peak'], float)
    H, W = lum.shape
    h = _PSF_HALF
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2)
    ok = (d[:, 1] > 4 * h) & (x > 2 * h) & (x < W - 2 * h) & (y > 2 * h) & (y < H - 2 * h) \
        & (pk < 0.5 * np.nanmax(lum))
    order = np.flatnonzero(ok)[np.argsort(-f[ok])][:n]
    stamps = []
    for i in order:
        cy, cx = y[i], x[i]
        iy, ix = int(round(cy)), int(round(cx))
        cut = lum[iy - h - 1:iy + h + 2, ix - h - 1:ix + h + 2].astype(np.float64)
        cut = ndimage.shift(cut, (iy - cy, ix - cx), order=3, mode='nearest')[1:-1, 1:-1]
        cut = cut - np.median(np.r_[cut[0], cut[-1], cut[:, 0], cut[:, -1]])
        tot = cut.sum()
        if tot > 0:
            stamps.append(cut / tot)
    psf = np.median(np.stack(stamps), axis=0)
    psf = np.clip(psf, 0, None)
    return psf / psf.sum()


def inject(rgb: np.ndarray, psf: np.ndarray, rng, n: int, sigma: float):
    """Add ``n`` PSF-shaped sources; returns (image, [(y, x), ...])."""
    out = rgb.copy()
    H, W = rgb.shape[:2]
    h = psf.shape[0] // 2
    pos = []
    peak_unit = float(psf.max())
    for _ in range(n):
        y, x = rng.uniform(60, H - 60), rng.uniform(60, W - 60)
        snr_peak = float(np.exp(rng.uniform(np.log(4.0), np.log(60.0))))
        flux = snr_peak * sigma / peak_unit
        iy, ix = int(round(y)), int(round(x))
        p = ndimage.shift(np.pad(psf, 1), (y - iy, x - ix), order=3, mode='constant')[1:-1, 1:-1]
        p = np.clip(p, 0, None)
        colour = np.exp(rng.normal(0, 0.25, 3))
        colour /= colour @ np.array([0.299, 0.587, 0.114])  # _to_luminance's weights
        out[iy - h:iy + h + 1, ix - h:ix + h + 1] += (flux * p)[..., None] * colour[None, None, :]
        pos.append((y, x))
    return out.astype(np.float32), pos


def pair_stamps(new_path: str, ref_path: str, rounds: int, per_round: int, rng, size: int):
    new = load_merge_stack(new_path)[0].astype(np.float32)
    ref = load_merge_stack(ref_path)[0].astype(np.float32)
    lum = new.astype(np.float64).mean(axis=2)
    psf = empirical_psf(lum - np.median(lum))
    sigma = estimate_background_sigma(lum - np.median(lum))
    X, y = [], []
    for _ in range(rounds):
        img, pos = inject(new, psf, rng, per_round, sigma)
        with contextlib.redirect_stdout(io.StringIO()):
            comp = _compare_epochs(img, ref)
        if comp is None or not comp.transients:
            continue
        cands = comp.transients
        P = np.array(pos)
        labels = np.zeros(len(cands), np.float32)
        for i, c in enumerate(cands):
            if np.min(np.hypot(P[:, 0] - c.y, P[:, 1] - c.x)) <= _MATCH_RADIUS_PX:
                labels[i] = 1.0
        st = build_stamps(comp.new_lum.astype(np.float32), comp.ref_lum.astype(np.float32),
                          comp.difference, [(c.y, c.x) for c in cands],
                          estimate_background_sigma(comp.new_lum),
                          estimate_background_sigma(comp.ref_lum),
                          estimate_background_sigma(comp.difference), size=size)
        X.append(st)
        y.append(labels)
        print(f'  {os.path.basename(new_path)}: {len(cands)} candidates, '
              f'{int(labels.sum())}/{len(pos)} injected recovered', flush=True)
    if not X:
        return np.zeros((0, 3, size, size), np.float32), np.zeros(0, np.float32)
    return np.concatenate(X), np.concatenate(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pair', nargs=2, action='append', required=True, metavar=('NEW', 'REF'))
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--per-round', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--size', type=int, default=DEFAULT_STAMP_SIZE)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    Xs, ys, groups = [], [], []
    for g, (n, r) in enumerate(a.pair):
        for new, ref in ((n, r), (r, n)):
            X, y = pair_stamps(new, ref, a.rounds, a.per_round, rng, a.size)
            Xs.append(X); ys.append(y); groups.append(np.full(len(y), g, np.int16))
    X, y, grp = np.concatenate(Xs), np.concatenate(ys), np.concatenate(groups)
    np.savez_compressed(a.out, X=X, y=y, group=grp, size=a.size)
    print(f'Wrote {a.out}: {len(y)} stamps, {int(y.sum())} positive, '
          f'{len(a.pair)} target pair(s)')


if __name__ == '__main__':
    main()
