"""Learned star removal by Noisier2Noise (EXPERIMENTAL: no shipped model, no CLI flag yet).

On linear data stars are an additive layer: ``y = x + s1`` (x the starless sky). Inject a
second layer ``s2`` drawn from the same distribution as the real one, ``z = y + s2``, and
train the U-Net of ``nn_denoise`` to predict ``y`` from ``z`` (L2, so it learns E[y|z]).
If the stars are a marked Poisson process independent of x and the injected one is the real
one thinned by ``alpha`` (same PSFs, fluxes, colours, local density), then given z each star
is injected with probability alpha / (1 + alpha), so

    E[x|z] = z - (1 + alpha) / alpha * (z - E[y|z])        (alpha = 1: 2 E[y|z] - z)

(Moran et al. 2020, "Noisier2Noise", for additive noise; here the "noise" is the star layer).
No starless ground truth is needed: every session trains on itself. At run time the network
sees y rather than z (a 1/(1 + alpha) lower star density than in training); smaller alpha
narrows that gap at the cost of amplifying the network's error by (1 + alpha) / alpha.

``StarField`` builds the injection model from the image itself: clean per-channel PSFs
(``_psf_from``: median of the bright isolated unsaturated stars, centroid-aligned; the core
from the nearest stars of each field zone, the wings azimuthally averaged over all of them;
placed at a random sub-pixel phase), donor fluxes and colours (the
detected stars' own luminosity function), and a smoothed detection density. Injected stars
get photon noise (variance s / B in noise units). v1 pasted the raw star cutouts instead:
each carried its own sky noise, and the network learned to recognise and fully remove those
stars (bright injected stars kept 4% vs real 60-90%, dev-notes/star-n2n-runbook.md). What it
cannot match is listed there (saturated stars, stars below detection, correlated noise).

Training: tools/gen_starless_bank.py, then tools/train_nn_denoise.py train (unchanged).
Evaluation: tools/eval_star_n2n.py.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from src import nn_denoise as nd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nn_starless.npz')

STAMP_R = 12            # stamp half-size (px); also the aperture radius for star fluxes
SAT_FRAC = 0.8          # a star whose stamp peaks above this x the image maximum is saturated
ISOLATION_FRAC = 0.05   # a neighbour within 2 R at >= this x the flux spoils a stamp
DENSITY_BLOCK = 16      # px, cell of the injection density grid
DENSITY_SIGMA = 4.0     # cells (64 px), smoothing of the detection density
DENSITY_FLOOR = 0.05    # x mean density, so star-free regions still get the odd star
N_STAMPS = 60
PSF_R = 24              # injected PSF half-size (px): bright stars' wings reach past STAMP_R
PSF_CORE_R = 6.0        # px; outside, the PSF is its azimuthal mean (averages the wing noise away)
N_PHASE = 4             # sub-pixel phases per axis the PSF is pre-shifted to
PSF_ZONES = 4           # PSF cores on a PSF_ZONES x PSF_ZONES grid of field zones...
PSF_LOCAL_K = 15        # ...each the median of the nearest this many stars


def sky_level_z(bg: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Sky level B per channel in noise units (median background / sigma); the photon-noise
    variance of a signal s in noise units is s / B. Same as train_nn_denoise.sky_level_z."""
    return np.median(bg.reshape(-1, bg.shape[-1]), axis=0) / sigma


def normalise_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(z, bg, sigma, B)``: the run-time normalisation of ``nn_denoise`` (z in sky-noise
    units) plus the sky level B per channel."""
    grid, sigma = nd.normalise(img)
    bg = nd.background_at(grid, 0, 0, *img.shape[:2])
    sig = sigma.astype(np.float32)
    return ((img - bg) / sig).astype(np.float32), bg, sig, sky_level_z(bg, sigma)


def _shift(a: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Cubic-spline sub-pixel shift of an (h, w, C) array, per channel."""
    return np.stack([ndi.shift(a[..., c], (dy, dx), order=3, mode='constant')
                     for c in range(a.shape[2])], -1).astype(np.float32)


def _centroid(sp: np.ndarray, r: int) -> Tuple[float, float]:
    """First moment of the luminance within 3 px of a cutout's centre pixel ``(r, r)``."""
    cut = np.clip(sp.mean(-1)[r - 3:r + 4, r - 3:r + 4], 0.0, None)
    t = float(cut.sum())
    if t <= 0:
        return 0.0, 0.0
    yy, xx = np.mgrid[-3:4, -3:4]
    return float((yy * cut).sum() / t), float((xx * cut).sum() / t)


def _aligned_cutouts(z: np.ndarray, ys: np.ndarray, xs: np.ndarray, r: int = PSF_R
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """``(cuts, keep)``: (n, 2r+1, 2r+1, C) cutouts of the stars at integer positions
    ``ys, xs`` with the border-median background removed, centroid-aligned and scaled to
    unit flux in the ``aperture_flux`` box (so an injected flux f measures as f); ``keep``
    indexes the stars used (inside the image, positive flux in every channel)."""
    h, w, c = z.shape
    a = r - STAMP_R
    cuts, keep = [], []
    for i, (y, x) in enumerate(zip(np.asarray(ys, int), np.asarray(xs, int))):
        if not (r <= y < h - r and r <= x < w - r):
            continue
        cut = z[y - r:y + r + 1, x - r:x + r + 1]
        border = np.concatenate([cut[0], cut[-1], cut[1:-1, 0], cut[1:-1, -1]])
        sp = cut - np.median(border, axis=0)
        dy, dx = _centroid(sp, r)
        sp = _shift(sp, -dy, -dx)
        core = sp[a:a + 2 * STAMP_R + 1, a:a + 2 * STAMP_R + 1].sum((0, 1))
        if np.all(core > 0):
            cuts.append(sp / core)
            keep.append(i)
    return np.array(cuts, np.float32).reshape(-1, 2 * r + 1, 2 * r + 1, c), np.array(keep, int)


def _psf_from(core_cuts: np.ndarray, wing_cuts: np.ndarray, r: int = PSF_R) -> np.ndarray:
    """A clean PSF: per-pixel median of ``core_cuts`` inside ``PSF_CORE_R`` (keeps the local
    shape: elongation, orientation), the azimuthal mean of the median of ``wing_cuts``
    outside (the wings carry most of the pixels, so most of the cutouts' own sky noise:
    v1 pasted raw cutouts and the network learned to spot that noise). Medians reject a
    neighbour in any one cutout. Clipped at 0, tapered to 0 at r, unit box flux."""
    c = core_cuts.shape[-1]
    a = r - STAMP_R
    core = np.median(core_cuts, axis=0)
    wingm = np.median(wing_cuts, axis=0)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    rad = np.hypot(yy, xx)
    ri = np.rint(rad).astype(int)
    wing = np.clip(rad - PSF_CORE_R, 0.0, 1.0)
    m = np.empty_like(core)
    for ch in range(c):
        prof = ndi.mean(wingm[..., ch], ri, index=np.arange(ri.max() + 1))
        m[..., ch] = (1.0 - wing) * core[..., ch] + wing * prof[ri]
    m = np.clip(m, 0.0, None) * _window(r)[:, :, None]
    tot = m[a:a + 2 * STAMP_R + 1, a:a + 2 * STAMP_R + 1].sum((0, 1))
    return (m / tot).astype(np.float32)


def _window(r: int = STAMP_R) -> np.ndarray:
    """Radial taper: 1 inside r - 3, cosine to 0 at r (no square stamp edges)."""
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    t = np.clip((r - np.hypot(yy, xx)) / 3.0, 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def aperture_flux(img: np.ndarray, ys: np.ndarray, xs: np.ndarray, r: int = STAMP_R) -> np.ndarray:
    """(n, C) background-subtracted flux in a (2r+1)^2 box at each integer position, the
    background being the per-channel median of the box border; NaN where the box leaves the
    image."""
    h, w, c = img.shape
    out = np.full((len(ys), c), np.nan, np.float64)
    for i, (y, x) in enumerate(zip(np.asarray(ys, int), np.asarray(xs, int))):
        if r <= y < h - r and r <= x < w - r:
            cut = img[y - r:y + r + 1, x - r:x + r + 1]
            border = np.concatenate([cut[0], cut[-1], cut[1:-1, 0], cut[1:-1, -1]])
            out[i] = (cut - np.median(border, axis=0)).sum((0, 1))
    return out


class StarField:
    """Injection model of one image's star layer (see the module docstring).

    ``z`` is the image in noise units (``normalise_image``), ``B`` its sky level per channel.
    ``ok`` is False when the image has too few usable stars to model (< 10 donors or no
    stamp); such an image cannot be used for training.
    """

    def __init__(self, z: np.ndarray, B: np.ndarray, n_stamps: int = N_STAMPS):
        self.shape = z.shape
        self.B = np.asarray(B, np.float64)
        h, w, _ = z.shape
        r = STAMP_R
        lum = z.mean(-1)
        st = nd_detect(lum - np.median(lum))
        ys = np.rint(np.asarray(st['ycentroid'], np.float64)).astype(int)
        xs = np.rint(np.asarray(st['xcentroid'], np.float64)).astype(int)
        inb = (ys >= r) & (ys < h - r) & (xs >= r) & (xs < w - r)
        ys, xs = ys[inb], xs[inb]
        self.det_y, self.det_x = ys, xs
        flux = aperture_flux(z, ys, xs)
        flum = flux.mean(1)
        sat_level = SAT_FRAC * float(np.nanmax(lum))
        peak = np.array([lum[y - 2:y + 3, x - 2:x + 3].max() for y, x in zip(ys, xs)]) if len(ys) else np.zeros(0)
        self.saturated = peak >= sat_level
        donor = ~self.saturated & np.isfinite(flum) & (flum > 0)

        # stamps: the brightest isolated unsaturated stars with positive flux in every channel
        win = _window()
        stamps, sy, sx, sflux = [], [], [], []
        for i in np.argsort(-np.where(donor, flum, -np.inf)):
            if not donor[i] or len(stamps) >= n_stamps:
                break
            if np.any(flux[i] <= 0):
                continue
            d2 = (ys - ys[i]) ** 2 + (xs - xs[i]) ** 2
            near = (d2 < (2 * r) ** 2) & (np.nan_to_num(flum, nan=0.0) >= ISOLATION_FRAC * flum[i])
            near[i] = False
            if near.any():
                continue
            cut = z[ys[i] - r:ys[i] + r + 1, xs[i] - r:xs[i] + r + 1]
            border = np.concatenate([cut[0], cut[-1], cut[1:-1, 0], cut[1:-1, -1]])
            sp = (cut - np.median(border, axis=0)) * win[:, :, None]
            tot = sp.sum((0, 1))
            if np.any(tot <= 0):
                continue
            stamps.append((sp / tot).astype(np.float32))
            sy.append(ys[i])
            sx.append(xs[i])
            sflux.append(flum[i])
        self.stamps = np.array(stamps, np.float32).reshape(-1, 2 * r + 1, 2 * r + 1, z.shape[2])
        self.stamp_y, self.stamp_x = np.array(sy, float), np.array(sx, float)

        # the injected PSFs: one per field zone (local core, global wings), each pre-shifted
        # to N_PHASE x N_PHASE sub-pixel phases. One PSF for the whole field (v2) made
        # injected stars rounder than real ones, most at the edges (ellipticity 0.11 vs 0.17)
        cuts, keep = _aligned_cutouts(z, np.array(sy, int), np.array(sx, int))
        self.psf_phases = None
        if len(cuts):
            cy, cx = np.array(sy, float)[keep], np.array(sx, float)[keep]
            off = (np.arange(N_PHASE) + 0.5) / N_PHASE - 0.5
            zh, zw = h / PSF_ZONES, w / PSF_ZONES
            k = min(PSF_LOCAL_K, len(cuts))
            phases = []
            for gy in range(PSF_ZONES):
                for gx in range(PSF_ZONES):
                    d2 = (cy - (gy + 0.5) * zh) ** 2 + (cx - (gx + 0.5) * zw) ** 2
                    psf = _psf_from(cuts[np.argsort(d2)[:k]], cuts)
                    phases.append([[_shift(psf, dy, dx) for dx in off] for dy in off])
            self.psf_phases = np.array(phases, np.float32)     # (zones, N_PHASE, N_PHASE, ...)

        # donors: the luminosity function and colours to inject, capped at the brightest stamp
        # so no stamp is scaled up (that would amplify its noise)
        cap = max(sflux) if sflux else 0.0
        donor &= np.nan_to_num(flum, nan=np.inf) <= cap
        self.donor_flux = np.clip(flux[donor], 0.0, None)
        self.donor_y, self.donor_x = ys[donor], xs[donor]

        # injection intensity proportional to the local density of real (donor) stars
        gh, gw = max(1, h // DENSITY_BLOCK), max(1, w // DENSITY_BLOCK)
        cnt = np.zeros((gh, gw), np.float64)
        np.add.at(cnt, (np.minimum(self.donor_y // DENSITY_BLOCK, gh - 1),
                        np.minimum(self.donor_x // DENSITY_BLOCK, gw - 1)), 1.0)
        dens = ndi.gaussian_filter(cnt, DENSITY_SIGMA, mode='nearest')
        dens = np.maximum(dens, DENSITY_FLOOR * max(dens.mean(), 1e-12))
        self.density = dens / dens.sum()
        self.ok = len(self.donor_flux) >= 10 and self.psf_phases is not None

    @property
    def n_donors(self) -> int:
        return len(self.donor_flux)

    def inject(self, rng: np.random.Generator, alpha: float = 1.0
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """A new star layer, ``alpha`` x the real donor count on average (Poisson):
        ``(s, ys, xs, flux)`` with s (H, W, C) in noise units (photon noise included) and the
        injected positions and per-channel fluxes."""
        h, w, c = self.shape
        r = PSF_R
        s = np.zeros(self.shape, np.float32)
        n = int(rng.poisson(alpha * self.n_donors)) if self.ok else 0
        if n == 0:
            return s, np.zeros(0, int), np.zeros(0, int), np.zeros((0, c))
        gh, gw = self.density.shape
        cell = rng.choice(gh * gw, size=n, p=self.density.ravel())
        ys = np.minimum((cell // gw) * DENSITY_BLOCK + rng.integers(0, DENSITY_BLOCK, n), h - 1)
        xs = np.minimum((cell % gw) * DENSITY_BLOCK + rng.integers(0, DENSITY_BLOCK, n), w - 1)
        flux = self.donor_flux[rng.integers(0, self.n_donors, n)]
        # the clean PSF of the star's field zone at a random sub-pixel phase: v1 pasted the
        # raw cutouts, whose own sky noise the network learned to spot
        ph = rng.integers(0, N_PHASE, (n, 2))
        zone = (np.minimum(ys * PSF_ZONES // h, PSF_ZONES - 1) * PSF_ZONES
                + np.minimum(xs * PSF_ZONES // w, PSF_ZONES - 1))
        for y, x, f, (py, px), zi in zip(ys, xs, flux, ph, zone):
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            st = self.psf_phases[zi, py, px, y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)]
            s[y0:y1, x0:x1] += st * f.astype(np.float32)
        for ch in range(c):
            if self.B[ch] > 0:
                s[..., ch] += (rng.standard_normal((h, w)).astype(np.float32)
                               * np.sqrt(np.maximum(s[..., ch], 0.0) / np.float32(self.B[ch])))
        return s, ys, xs, flux


def nd_detect(lum: np.ndarray):
    """The pipeline's star detector (module-level so tests can shim it)."""
    from src.star_detect import detect_stars_matched_filter
    return detect_stars_matched_filter(np.ascontiguousarray(lum, np.float32))


def starless_z(z: np.ndarray, w, alpha: float = 1.0) -> np.ndarray:
    """Noisier2Noise estimate of the starless sky from noise-unit data z (H, W, C):
    ``z - (1 + alpha) / alpha * (z - f(z))`` with f the trained network."""
    f = nd.denoise_z(z, w)
    return (z - np.float32((1.0 + alpha) / alpha) * (z - f)).astype(np.float32)


def model_available(path: Optional[str] = None) -> bool:
    return os.path.exists(path or MODEL_PATH)


def remove_stars_nn(img: np.ndarray, model_path: Optional[str] = None, alpha: float = 1.0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """``(starless, stars)`` for a linear (H, W, 3) image; ``alpha`` must be the one the
    model's bank was built with. ``stars = img - starless`` (for screen recombination)."""
    w = nd.load_weights(model_path or MODEL_PATH)
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    src = np.where(finite, img, 0.0).astype(np.float32)
    z, bg, sig, _ = normalise_image(src)
    out = starless_z(z, w, alpha) * sig + bg
    out = np.where(finite, out, img).astype(np.float32)
    return out, np.where(finite, src - out, 0.0).astype(np.float32)
