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

``StarField`` builds the injection model from the image itself: injected stars are copies of
the image's own unsaturated stars (donors, picked uniformly, so the luminosity function,
colours, shapes and sub-pixel phases match by construction), pasted 64-192 px from their donor
(same background and PSF zone), out to the radius where the star's wing drops below
``COPY_WING_LEVEL``. A copy carries its donor's photon noise and also a disc of its donor's
sky noise; ``real_discs`` gives every real donor such a disc too (a tapered cutout of
star-free sky), added to the training input only (zero-mean, so the target is unaffected),
and ``starless_z`` adds them at run time, so real and injected stars look alike.

Four rounds of *modelled* injected stars failed (dev-notes/star-n2n-runbook.md): raw stamp
cutouts scaled to new fluxes (their sky noise gave them away), then clean PSFs, per zone,
per flux bin -- the network always told them from real stars by shape (validation loss far
below the ~half-of-identity floor of indistinguishable stars; pasted real cutouts were kept
like real stars). Saturated stars and stars below detection are never injected, so they stay.

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
SAT_FRAC = 0.7          # a star peaking above this x its channel's maximum, in any channel, is saturated
ISOLATION_FRAC = 0.05   # a neighbour within 2 R at >= this x the flux spoils a stamp
DENSITY_BLOCK = 16      # px, cell of the injection density grid
DENSITY_SIGMA = 4.0     # cells (64 px), smoothing of the detection density
DENSITY_FLOOR = 0.05    # x mean density, so star-free regions still get the odd star
N_STAMPS = 60
PSF_R = 40              # px; the wing profile's reach (bright stars' halos go past STAMP_R)
COPY_R_MIN, COPY_R_MAX = 6, 40   # px; radius of a copied star: where its wing drops below...
COPY_WING_LEVEL = 0.05  # ...this (noise units), plus the taper; brighter stars carry wider copies.
                        # 0.2 lost a third of a faint star's flux in the wings it left out
SKY_CLEAR_R = 10        # px; disc sources: sky positions with no detection this close
DISC_QUIET_FRAC = 0.3   # ...and no more structured than this quantile of the candidates,
DISC_QUIET_X = 1.5      # ...or this x their 5th percentile, whichever is higher (see StarField)
STRUCT_BLOCK = 8        # px; block medians for the structure measure (stars drop out)
INJECT_OFFSET = (64.0, 192.0)   # px; a copy lands this far from its donor
GUARD_SIGMA = 1.0       # starless floor inside a star: ring's 10th percentile minus this (noise units)
OUTSIDE_CLIP = (-1.0, 5.0)   # star layer outside every footprint, in local-noise sigma: undetected stars
COLOUR_SNR = (3.0, 8.0)      # star-layer luminance S/N over which a footprint's colour is forced
SAT_R_MIN, SAT_R_MAX = 6, 40  # px; a saturated star's core fill radius: where its halo is above...
SAT_CORE_LEVEL = 20.0         # ...this many sky sigma (the clipped, non-linear part)
HALO_R = 150                  # px; a saturated star's radial halo is subtracted out to here...
HALO_FLOOR = 0.3              # ...or to where it first drops below this (sky sigma)
# A saturated star whose 45-60 px ring is this far above the sky (noise units) sits in
# extended emission and is left alone: its ring medians are mostly nebula, and the halo
# subtraction and core fill cut discs and coloured rings out of it -- Orion's core (ring
# levels up to 1343), a dark teal hole in M43 (46). Isolated saturated stars on the five
# held-out stacks measure -21..+1.3; the nebula-embedded ones 22 (Pleiades) and up.
SAT_SKIP_BG = 10.0
SAT_FEATHER = 0.5             # the core fill is blended into the data over r..(1 + this) r
SPIKE_RADIAL_PX = 21          # spikes: median along the radius over this many px (drops stars, noise)...
SPIKE_ANGLE_DEG = 30.0        # ...minus the median over this wide an angle (keeps what is narrow in angle)
SPIKE_FLOOR = 0.5             # soft threshold on the spike model (sky sigma): noise is not subtracted
CORE_R = 12                   # px; residual cores: a donor's core is searched and refilled within this radius...
CORE_RING = 5                 # ...against the ring this wide just outside it (median level, MAD noise)
CORE_GATE = 8.0               # a star has a residual core when its 1.5 px-smoothed luminance peak is this many ring sigmas up
CORE_K = 3.0                  # pixels more than this many ring sigmas above the ring are the core (grown by CORE_GROW px)
CORE_GROW = 1
CORE_FILL_SIGMA = 3.0         # px; normalised-convolution fill of the core from the pixels around it
MONO_BLUR_SIGMA = 0.6         # px; mono input is blurred to the noise correlation the network was trained on (starless_z)

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


def _sky_structure(lum: np.ndarray) -> np.ndarray:
    """Extended structure around each ``STRUCT_BLOCK`` block of ``lum`` (noise units): the
    mean absolute band-pass (16-64 px) of the block medians over a 40 px neighbourhood, plus
    a tenth of the smoothed level. Block medians drop the stars, whose wings otherwise made
    every crowded patch look structured."""
    b = STRUCT_BLOCK
    h, w = (lum.shape[0] // b) * b, (lum.shape[1] // b) * b
    blk = np.median(lum[:h, :w].reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(h // b, w // b, -1), -1)
    blk = np.pad(blk, ((0, lum.shape[0] // b + 1 - h // b), (0, lum.shape[1] // b + 1 - w // b)), mode='edge')
    smooth = ndi.gaussian_filter(blk, 2.0)
    band = ndi.gaussian_filter(blk, 0.5) - smooth
    return ndi.uniform_filter(np.abs(band), 5) + 0.1 * np.abs(smooth)


def _plane_subtract(cut: np.ndarray) -> np.ndarray:
    """``cut`` (h, w, C) minus a per-channel plane fitted to its border pixels, so a copied
    star does not carry its donor's local background (a nebula gradient) along. Border pixels
    more than 2.5 sigma above the fit (a neighbour's wing) are dropped and the fit repeated:
    with them a crowded field's planes sat too high, and copies lost a third of their flux."""
    h, w, c = cut.shape
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.zeros((h, w), bool)
    edge[0], edge[-1], edge[:, 0], edge[:, -1] = True, True, True, True
    a = np.c_[np.ones(edge.sum()), yy[edge], xx[edge]]
    out = np.empty(cut.shape, np.float32)
    for ch in range(c):
        v = cut[..., ch][edge].astype(np.float64)
        keep = np.ones(len(v), bool)
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(a[keep], v[keep], rcond=None)
            res = v - a @ coef
            sd = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
            new = res < 2.5 * max(sd, 1e-6)
            if new.sum() < 3 or (new == keep).all():
                break
            keep = new
        out[..., ch] = cut[..., ch] - (coef[0] + yy * coef[1] + xx * coef[2])
    return out


def _wing_profile(z: np.ndarray, ys: np.ndarray, xs: np.ndarray, r: int = PSF_R) -> Optional[np.ndarray]:
    """Azimuthal luminance profile (index = radius in px, 0..r) of the stars at ``ys, xs``,
    per unit ``aperture_flux``: median of their background-subtracted cutouts."""
    h, w, _ = z.shape
    a = r - STAMP_R
    cuts = []
    for y, x in zip(np.asarray(ys, int), np.asarray(xs, int)):
        if not (r <= y < h - r and r <= x < w - r):
            continue
        sp = _plane_subtract(z[y - r:y + r + 1, x - r:x + r + 1]).mean(-1)
        tot = sp[a:a + 2 * STAMP_R + 1, a:a + 2 * STAMP_R + 1].sum()
        if tot > 0:
            cuts.append(sp / tot)
    if not cuts:
        return None
    m = np.median(np.array(cuts), axis=0)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    ri = np.rint(np.hypot(yy, xx)).astype(int)
    return ndi.mean(m, ri, index=np.arange(r + 1))


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
        # per channel: on the Pleiades the 9 brightest "unsaturated" stars (by luminance)
        # all had red peaks at 0.79-0.85 of the red maximum, a soft-saturation plateau (some
        # subframes clipped), with green and blue at 0.6-0.7. Not additive: copied, they
        # left coloured blobs and inverted or kept cores
        sat_level = SAT_FRAC * np.nanmax(z.reshape(-1, z.shape[2]), axis=0)
        peak = (np.array([z[y - 2:y + 3, x - 2:x + 3].reshape(-1, z.shape[2]).max(0) for y, x in zip(ys, xs)])
                if len(ys) else np.zeros((0, z.shape[2])))
        self.saturated = (peak >= sat_level).any(1)
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

        # donors: every unsaturated star with positive flux; its copy radius from the bright
        # stars' wing profile (copied out to where the wing falls below COPY_WING_LEVEL)
        prof = _wing_profile(z, np.array(sy, int), np.array(sx, int))
        rad = np.full(len(ys), COPY_R_MIN)
        if prof is not None:
            for i in np.flatnonzero(donor):
                below = np.flatnonzero(flum[i] * prof < COPY_WING_LEVEL)
                rad[i] = (below[0] if len(below) else PSF_R) + 3
        rad = np.clip(rad, COPY_R_MIN, COPY_R_MAX)
        donor &= (ys >= rad) & (ys < h - rad) & (xs >= rad) & (xs < w - rad)
        self.donor_flux = np.clip(flux[donor], 0.0, None)
        self.donor_y, self.donor_x, self.donor_r = ys[donor], xs[donor], rad[donor]
        self._copies = [_plane_subtract(z[y - q:y + q + 1, x - q:x + q + 1]) * _window(q)[:, :, None]
                        for y, x, q in zip(self.donor_y, self.donor_x, self.donor_r)]
        # a copy's own flux: donor_flux (a 25 x 25 box) also counts neighbours in a crowded field
        self.copy_flux = np.array([cp.sum((0, 1)) for cp in self._copies], np.float64).reshape(-1, z.shape[2])

        # disc sources: star-free points 2-3 copy radii from each donor, most clear first. A
        # copy brings its donor's surroundings (sky noise, nebula texture, a halo) along, so
        # the real stars' discs come from the same kind of place; star-free points anywhere
        # (first try) were mostly bright nebula on the Pleiades (disc sigma 15.7 vs 0.9).
        free = np.ones((h, w), bool)
        free[ys, xs] = False
        dist = ndi.distance_transform_edt(free)
        m = COPY_R_MAX
        ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        py, px = [], []
        for k in (2, 3):
            ry = np.rint(self.donor_y[:, None] + k * self.donor_r[:, None] * np.sin(ang)).astype(int).ravel()
            rx = np.rint(self.donor_x[:, None] + k * self.donor_r[:, None] * np.cos(ang)).astype(int).ravel()
            py.append(ry)
            px.append(rx)
        py, px = np.concatenate(py), np.concatenate(px)
        inb = (py >= m) & (py < h - m) & (px >= m) & (px < w - m)
        py, px = py[inb], px[inb]
        cl = dist[py, px]
        ok = cl >= SKY_CLEAR_R
        # the quietest candidates only: a disc carries its source's structure, and the network
        # does not remove all of it, so (1 + alpha) / alpha later doubles what is left. Orion's
        # discs, cut from bright nebula 2-3 radii away, came out as red and green discs up to
        # 80 px across around mid-bright stars on dark sky
        if ok.sum() > 1:
            struct = _sky_structure(lum)[py // STRUCT_BLOCK, px // STRUCT_BLOCK]
            # an absolute bar as well: on plain sky the quantile alone kept the lowest-noise
            # spots, discs smoother than the network was trained with, and the synthetic
            # test's bright stars kept 0.40 of their flux instead of 0.21
            ok &= struct <= max(np.quantile(struct[ok], DISC_QUIET_FRAC),
                                DISC_QUIET_X * np.quantile(struct[ok], 0.05))
        order = np.argsort(-cl[ok])
        self._sky_y, self._sky_x, self._sky_dist = py[ok][order], px[ok][order], cl[ok][order]
        self._z = z

        # injection intensity proportional to the local density of real (donor) stars
        gh, gw = max(1, h // DENSITY_BLOCK), max(1, w // DENSITY_BLOCK)
        cnt = np.zeros((gh, gw), np.float64)
        np.add.at(cnt, (np.minimum(self.donor_y // DENSITY_BLOCK, gh - 1),
                        np.minimum(self.donor_x // DENSITY_BLOCK, gw - 1)), 1.0)
        dens = ndi.gaussian_filter(cnt, DENSITY_SIGMA, mode='nearest')
        dens = np.maximum(dens, DENSITY_FLOOR * max(dens.mean(), 1e-12))
        self.density = dens / dens.sum()
        self.ok = len(self.donor_flux) >= 10 and len(self._sky_y) > 0

    @property
    def n_donors(self) -> int:
        return len(self.donor_flux)

    @staticmethod
    def _paste(s: np.ndarray, y: int, x: int, st: np.ndarray) -> None:
        h, w, _ = s.shape
        q = st.shape[0] // 2
        y0, y1 = max(0, y - q), min(h, y + q + 1)
        x0, x1 = max(0, x - q), min(w, x + q + 1)
        s[y0:y1, x0:x1] += st[y0 - (y - q):y1 - (y - q), x0 - (x - q):x1 - (x - q)]

    def inject(self, rng: np.random.Generator, alpha: float = 1.0
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """A new star layer, ``alpha`` x the donor count on average (Poisson), of copies of
        uniformly drawn donors: ``(s, ys, xs, flux)`` with s (H, W, C) in noise units (the
        copies carry their donors' photon and sky noise), the injected positions and the
        copies' per-channel fluxes."""
        h, w, c = self.shape
        s = np.zeros(self.shape, np.float32)
        n = int(rng.poisson(alpha * self.n_donors)) if self.ok else 0
        if n == 0:
            return s, np.zeros(0, int), np.zeros(0, int), np.zeros((0, c))
        # each copy lands INJECT_OFFSET px from its own donor (random angle): on the same
        # kind of background (nebula or dark sky) and in the same PSF zone, but beyond the
        # network's ~24 px view of its twin. Placed by the overall star density instead (v6),
        # bright copies landed mostly on dark sky while Orion's bright stars all sit in
        # nebula, and the network kept every bright star there (left +1.0)
        pick = rng.integers(0, self.n_donors, n)
        ys = np.zeros(n, int)
        xs = np.zeros(n, int)
        todo = np.arange(n)
        for _ in range(8):
            if not len(todo):
                break
            ang = rng.uniform(0, 2 * np.pi, len(todo))
            rad = rng.uniform(*INJECT_OFFSET, len(todo))
            ty = np.rint(self.donor_y[pick[todo]] + rad * np.sin(ang)).astype(int)
            tx = np.rint(self.donor_x[pick[todo]] + rad * np.cos(ang)).astype(int)
            inb = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
            ys[todo[inb]], xs[todo[inb]] = ty[inb], tx[inb]
            todo = todo[~inb]
        if len(todo):                       # a small image: fall back to the star density
            gh, gw = self.density.shape
            cell = rng.choice(gh * gw, size=len(todo), p=self.density.ravel())
            ys[todo] = np.minimum((cell // gw) * DENSITY_BLOCK + rng.integers(0, DENSITY_BLOCK, len(todo)), h - 1)
            xs[todo] = np.minimum((cell % gw) * DENSITY_BLOCK + rng.integers(0, DENSITY_BLOCK, len(todo)), w - 1)
        for y, x, j in zip(ys, xs, pick):
            self._paste(s, y, x, self._copies[j])
        return s, ys, xs, self.copy_flux[pick]

    def real_discs(self, rng: np.random.Generator) -> np.ndarray:
        """Zero-mean noise layer: at every donor, a tapered cutout of star-free sky of the
        donor's copy radius -- the disc of extra sky noise a copy brings, given to the real
        stars too so it cannot tell injected from real ones."""
        d = np.zeros(self.shape, np.float32)
        if not self.ok:
            return d
        # positions with no detection within q (all of them if none is that clear)
        n_ok = np.maximum(np.searchsorted(-self._sky_dist, -self.donor_r, side='left'), 1)
        k = (rng.random(self.n_donors) * n_ok).astype(int)
        for y, x, q, j in zip(self.donor_y, self.donor_x, self.donor_r, k):
            sy, sx = self._sky_y[j], self._sky_x[j]
            sky = _plane_subtract(self._z[sy - q:sy + q + 1, sx - q:sx + q + 1]) * _window(q)[:, :, None]
            self._paste(d, y, x, sky)
        return d


def nd_detect(lum: np.ndarray):
    """The pipeline's star detector (module-level so tests can shim it)."""
    from src.star_detect import detect_stars_matched_filter
    return detect_stars_matched_filter(np.ascontiguousarray(lum, np.float32))


def guard_holes(x: np.ndarray, field: 'StarField', floor_sigma: float = GUARD_SIGMA) -> np.ndarray:
    """Floor the starless estimate ``x`` (noise units) inside each donor's footprint at the
    10th percentile of a ring just outside it, minus ``floor_sigma``, per channel. The
    (1 + alpha) / alpha extrapolation turns any over-removal into a dip below the sky; v7
    still cut the cores of the Pleiades' brightest stars to -529 sigma. A real starless sky
    does not sit that far under its own surroundings, and a nebula gradient across a
    footprint mostly stays above the ring's low percentile."""
    out = x.copy()
    h, w, _ = x.shape
    for y, xc, q in zip(field.donor_y, field.donor_x, field.donor_r):
        R = q + 4
        y0, y1, x0, x1 = max(0, y - R), min(h, y + R + 1), max(0, xc - R), min(w, xc + R + 1)
        yy, xx = np.mgrid[y0 - y:y1 - y, x0 - xc:x1 - xc]
        rr = np.hypot(yy, xx)
        inside, ring = rr <= q, (rr > q) & (rr <= R)
        if ring.sum() < 8:
            continue
        cut = out[y0:y1, x0:x1]
        floor = np.percentile(cut[ring], 10, axis=0) - np.float32(floor_sigma)
        cut[inside] = np.maximum(cut[inside], floor)
    return out


def remove_residual_cores(x: np.ndarray, field: 'StarField', seed: int = 0) -> np.ndarray:
    """Refill the small core the network leaves at its brightest unsaturated stars.

    The (1 + alpha) / alpha extrapolation under-removes the very brightest stars (Orion:
    ``real_left`` of the top flux tertile 0.20, single stars 0.5-1.1; their halos reach past
    the copies the network trained on). The leftover is a lopsided crescent of the PSF core, so
    a radial profile subtraction is wrong (it digs a hole where the star's true centre is, and
    the Pleiades' ``hole`` went to -1.8); the clipped pixels have to be replaced.

    Per donor: level and noise of the ``CORE_RING`` px ring outside ``CORE_R`` (median, MAD).
    A star has a core only if the 1.5 px-smoothed luminance peak inside ``CORE_R`` is
    ``CORE_GATE`` ring sigmas above the ring (faint stars' tails are noise, which this would
    otherwise bias to a deficit: faint ``real_left`` -0.2..-0.3 without the gate). Pixels more
    than ``CORE_K`` sigma above the ring in any channel, grown ``CORE_GROW`` px, are replaced
    by normalised convolution from the pixels around them (not below the ring's level: the
    network's over-subtracted annulus would otherwise pull the core into a dip) plus the ring's noise, like
    ``fill_saturated``'s core, so the texture stays. Negative pixels (an inverted core) are
    left to ``guard_holes``."""
    out = x.copy()
    h, w, c = x.shape
    rng = np.random.default_rng(seed)
    for y, xc, q in zip(field.donor_y, field.donor_x, field.donor_r):
        rm = min(int(q), CORE_R)
        R = rm + CORE_RING + 2 * int(CORE_FILL_SIGMA)
        y0, y1, x0, x1 = max(0, y - R), min(h, y + R + 1), max(0, xc - R), min(w, xc + R + 1)
        cut = out[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0 - y:y1 - y, x0 - xc:x1 - xc]
        rr = np.hypot(yy, xx)
        ring = (rr > rm) & (rr <= rm + CORE_RING)
        if ring.sum() < 8:
            continue
        inside = rr <= rm
        bg = np.median(cut[ring], axis=0)
        sg = 1.4826 * np.median(np.abs(cut[ring] - bg), axis=0) + 1e-3
        sm = ndi.gaussian_filter((cut - bg).mean(-1), 1.5, mode='nearest')
        rl = sm[ring]
        sl = 1.4826 * np.median(np.abs(rl - np.median(rl))) + 1e-3
        if (sm - np.median(rl))[inside].max() / sl < CORE_GATE:
            continue
        hot = ((cut - bg) / sg > CORE_K).any(-1) & inside
        mask = ndi.binary_dilation(hot, iterations=CORE_GROW) & inside
        keep = (~mask).astype(np.float32)
        den = np.maximum(ndi.gaussian_filter(keep, CORE_FILL_SIGMA, mode='nearest'), 1e-6)
        noise = _ring_noise(cut - bg, ring, rng)
        for k in range(c):
            v = cut[..., k]
            fill = ndi.gaussian_filter(v * keep, CORE_FILL_SIGMA, mode='nearest') / den
            v[mask] = (np.maximum(fill, bg[k]) + noise[..., k])[mask]
    return out


def _ring_noise(res: np.ndarray, ring: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noise with the ring's statistics, shaped like ``res`` (h, w, C): per-channel sigma (MAD),
    the channels' correlation, and the lag-1 pixel correlation (debayer, warp). Independent
    white noise per channel showed as a speckled colour disc in the grey, smooth surroundings."""
    h, w, c = res.shape
    r = res[ring]
    sg = 1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0) + 1e-3
    pair = ring[:, :-1] & ring[:, 1:]
    a, b = res[:, :-1][pair] / sg, res[:, 1:][pair] / sg
    rho = float(np.clip(np.mean(a * b), 0.0, 0.9))
    sp = 1.0 / (2.0 * np.sqrt(-np.log(rho))) if rho > 0.05 else 0.0
    cor = np.corrcoef((r / sg).T) if len(r) > 2 and c > 1 else np.eye(c)
    try:
        chol = np.linalg.cholesky(np.nan_to_num(cor) + 1e-3 * np.eye(c))
    except np.linalg.LinAlgError:
        chol = np.eye(c)
    n = rng.standard_normal((h, w, c)).astype(np.float32)
    if sp > 0:
        n = ndi.gaussian_filter(n, (sp, sp, 0), mode='nearest')
        n /= np.float32(max(n.std(), 1e-6))
    return ((n @ chol.T.astype(np.float32)) * sg.astype(np.float32)).astype(np.float32)


def constrain_star_layer(z: np.ndarray, x: np.ndarray, field: 'StarField') -> np.ndarray:
    """Hold the star layer ``z - x`` (noise units) to what stars can be.

    Outside every detection's footprint it is clipped to ``OUTSIDE_CLIP`` x the local noise
    (sqrt(1 + light / B), Poisson above the sky): room for undetected faint stars and no more.
    v7 changed bright nebula between stars by up to 113 sigma, with a colour deviation of
    6.5 sigma at the median there (Omega's core: magenta / blue / green blotches), which the
    (1 + alpha) / alpha extrapolation doubles. Inside a donor's footprint, where the layer's
    luminance S/N rises through ``COLOUR_SNR``, its colour is faded to the donor's own (one
    colour per star); the faint rim keeps the network's per-channel output and its noise
    texture."""
    h, w, c = z.shape
    s = z - x
    foot = np.zeros((h, w), bool)
    yy, xx = np.ogrid[:h, :w]
    rads = np.full(len(field.det_y), COPY_R_MIN)
    rads[field.saturated] = COPY_R_MAX
    for y, xc, q in zip(np.r_[field.donor_y, field.det_y], np.r_[field.donor_x, field.det_x],
                        np.r_[field.donor_r, rads]):
        y0, y1, x0, x1 = max(0, y - q), min(h, y + q + 1), max(0, xc - q), min(w, xc + q + 1)
        foot[y0:y1, x0:x1] |= (yy[y0:y1] - y) ** 2 + (xx[:, x0:x1] - xc) ** 2 <= q * q
    light = np.stack([ndi.gaussian_filter(np.maximum(z[..., k], 0.0), 2.0) for k in range(c)], -1)
    loc = np.sqrt(1.0 + light / np.maximum(field.B, 1e-6).astype(np.float32))
    out = ~foot
    s[out] = np.clip(s[out], OUTSIDE_CLIP[0] * loc[out], OUTSIDE_CLIP[1] * loc[out])

    lo, hi = COLOUR_SNR
    for y, xc, q, cf in zip(field.donor_y, field.donor_x, field.donor_r, field.copy_flux):
        if np.any(cf <= 0):
            continue
        y0, y1, x0, x1 = max(0, y - q), min(h, y + q + 1), max(0, xc - q), min(w, xc + q + 1)
        r2 = (yy[y0:y1] - y) ** 2 + (xx[:, x0:x1] - xc) ** 2
        disk = r2 <= q * q
        col = _radial_colour(z[y0:y1, x0:x1], np.sqrt(r2), q, cf)
        cut = s[y0:y1, x0:x1]
        tot = cut.sum(-1)
        wgt = np.clip((tot / np.sqrt(c) - lo) / (hi - lo), 0.0, 1.0) * disk
        cut += wgt[..., None] * (tot[..., None] * col - cut)
    return (z - s).astype(np.float32)


def _radial_colour(cut: np.ndarray, rr: np.ndarray, q: int, cf: np.ndarray) -> np.ndarray:
    """Per-pixel colour (channel shares summing to 1) of the star in ``cut`` (noise units),
    from its own per-channel profile in 2 px rings, background = median of the ring just
    outside ``q``. Where a ring's star light is too faint to say (luminance < 1 sigma, or a
    negative channel), the copy's integrated colour ``cf``. One colour for the whole star
    turned a chromatic PSF (red halo, bluer core) into blue rims and red-yellow cores on the
    Pleiades' bright stars."""
    c = cut.shape[2]
    ring = (rr > q) & (rr <= q + 4)
    bg = np.median(cut[ring], axis=0) if ring.sum() >= 8 else np.zeros(c, np.float32)
    ri = np.minimum((rr / 2).astype(int), q // 2 + 1)
    nb = q // 2 + 2
    inside = rr <= q
    prof = np.stack([np.bincount(ri[inside], (cut[..., k] - bg[k])[inside], nb) for k in range(c)], -1)
    cnt = np.maximum(np.bincount(ri[inside], None, nb), 1)[:, None]
    prof = prof / cnt
    full = (cf / cf.sum()).astype(np.float32)
    tot = prof.sum(-1, keepdims=True)
    good = (tot[:, 0] / np.sqrt(c) >= 1.0) & (prof > 0).all(-1)
    col = np.where(good[:, None], prof / np.where(tot > 0, tot, 1.0), full).astype(np.float32)
    return col[ri]


def _subtract_halo(out: np.ndarray, y: int, x: int) -> np.ndarray:
    """Subtract a saturated star's radial halo from ``out`` (in place) out to ``HALO_R``:
    per channel, the median of each 2 px ring minus the median of the outermost 10 px,
    clipped at 0, made non-increasing, zero from the first ring below ``HALO_FLOOR`` sky sigma.
    Ring medians follow the star's own symmetric light and cancel a linear nebula gradient.
    Returns the luminance halo profile (per 2 px ring), used to size the core."""
    h, w, c = out.shape
    R = HALO_R
    y0, y1, x0, x1 = max(0, y - R), min(h, y + R + 1), max(0, x - R), min(w, x + R + 1)
    yy, xx = np.mgrid[y0 - y:y1 - y, x0 - x:x1 - x]
    rr = np.hypot(yy, xx)
    ri = (rr // 2).astype(int)
    nb = R // 2 + 1
    inside = rr <= R
    cut = out[y0:y1, x0:x1]
    lum = np.zeros(nb)
    for k in range(c):
        v = cut[..., k]
        prof = np.array([np.median(v[inside & (ri == i)]) if np.any(inside & (ri == i)) else np.nan
                         for i in range(nb)])
        bg = np.nanmedian(prof[-5:])
        # a halo only falls with radius: the running minimum keeps a nebula's bright
        # rim or dark pocket further out from being subtracted as "halo" (concentric
        # rings next to a dark pocket in Omega's nebula with --starless-process)
        halo = np.minimum.accumulate(np.nan_to_num(np.clip(prof - bg, 0.0, None)))
        low = np.flatnonzero(halo < HALO_FLOOR)
        if len(low):
            halo[low[0]:] = 0.0
        v -= np.where(inside, halo[np.minimum(ri, nb - 1)], 0.0).astype(np.float32)
        lum += halo / c
    return lum


def _subtract_spikes(out: np.ndarray, y: int, x: int, r_in: float, r_out: float,
                     others: Optional[np.ndarray] = None) -> None:
    """Subtract a saturated star's diffraction spikes from ``out`` (in place) between
    ``r_in`` and ``r_out``: in polar coordinates around the star, a median along the radius
    (``SPIKE_RADIAL_PX``) keeps what is long radially and drops noise and the stars that
    were not removed, a median over ``SPIKE_ANGLE_DEG`` of angle is the smooth light
    around (nebula, what is left of the halo), and their positive difference above
    ``SPIKE_FLOOR`` is the spike model. The radial halo subtraction leaves spikes in place:
    they are a few degrees wide, so they barely move a ring's median. ``others`` (N, 2)
    are the other saturated stars: within ``SAT_R_MAX`` of them nothing is modelled or
    subtracted -- seen from a neighbour, a saturated star's own halo is long and narrow
    like a spike (bright streaks and dark blobs at the Pleiades' companions otherwise)."""
    h, w, c = out.shape
    r_out = float(min(r_out, HALO_R))
    if r_out - r_in < 2 * SPIKE_RADIAL_PX:
        return
    n_ang = 720
    rad = np.arange(r_in, r_out + 1.0)
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    rr, aa = np.meshgrid(rad, ang, indexing='ij')
    py, px = y + rr * np.sin(aa), x + rr * np.cos(aa)
    ok = (py >= 0) & (py <= h - 1) & (px >= 0) & (px <= w - 1)
    if others is not None and len(others):
        for oy, ox in others:
            ok &= np.hypot(py - oy, px - ox) > SAT_R_MAX
    R = int(np.ceil(r_out)) + 1
    y0, y1, x0, x1 = max(0, y - R), min(h, y + R + 1), max(0, x - R), min(w, x + R + 1)
    yy, xx = np.mgrid[y0 - y:y1 - y, x0 - x:x1 - x]
    pr = np.hypot(yy, xx)
    pa = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    ri = pr - r_in
    ai = pa / (2 * np.pi) * n_ang
    taper = np.clip(np.minimum(pr - r_in, r_out - pr) / 5.0, 0.0, 1.0)
    if others is not None and len(others):
        for oy, ox in others:
            taper = taper * (np.hypot(yy + y - oy, xx + x - ox) > SAT_R_MAX)
    win = max(3, int(round(SPIKE_ANGLE_DEG / 360.0 * n_ang)) | 1)
    for k in range(c):
        pol = ndi.map_coordinates(out[..., k], [py, px], order=1, mode='nearest')
        # masked samples take their radius's median over the valid angles; a radius with
        # none valid (edge, or ringed by other saturated stars: Omega) models no spike
        any_ok = ok.any(axis=1, keepdims=True)
        fill = np.zeros((pol.shape[0], 1), pol.dtype)
        if any_ok.any():
            rows = any_ok[:, 0]
            fill[rows, 0] = [np.median(pol[i][ok[i]]) for i in np.flatnonzero(rows)]
        pol = np.where(ok, pol, np.broadcast_to(fill, pol.shape))
        along = ndi.median_filter(pol, size=(SPIKE_RADIAL_PX, 1), mode='nearest')
        smooth = ndi.median_filter(along, size=(1, win), mode='wrap')
        spike = np.clip(along - smooth - SPIKE_FLOOR, 0.0, None) * any_ok
        spike = np.concatenate([spike, spike[:, :1]], axis=1)      # wrap for interpolation
        model = ndi.map_coordinates(spike, [ri, ai], order=1, mode='constant', cval=0.0)
        out[y0:y1, x0:x1, k] -= (model * taper).astype(np.float32)


def fill_saturated(x: np.ndarray, z: np.ndarray, field: 'StarField', seed: int = 0) -> np.ndarray:
    """Remove each saturated star (never injected, so the network leaves it) from the
    starless estimate ``x``, whose other stars are already gone: subtract its radial halo
    (``_subtract_halo``), then fill the core -- where the halo was above ``SAT_CORE_LEVEL``
    sky sigma, the clipped, non-linear part -- by normalised convolution from the pixels
    around it, plus sky noise of the local level, feathered into the data over
    ``SAT_FEATHER``. Filling out to where the halo stops falling (60 px at most) left a flat
    disc inside a still-bright halo ring on the Pleiades and Orion; so did the
    ``--remove-stars`` inpaint, whose fills also had 7-39% of the noise. A star in bright
    extended emission (``SAT_SKIP_BG``) is left as it is."""
    out = x.copy()
    h, w, c = x.shape
    rng = np.random.default_rng(seed)
    sat = np.c_[field.det_y[field.saturated], field.det_x[field.saturated]]
    for j, (y, xc) in enumerate(sat):
        if _ring_level(out, y, xc) > SAT_SKIP_BG:
            continue
        lum = _subtract_halo(out, y, xc)
        hi = np.flatnonzero(lum >= SAT_CORE_LEVEL)
        r = int(np.clip(2 * (hi[-1] + 1) + 2 if len(hi) else SAT_R_MIN, SAT_R_MIN, SAT_R_MAX))
        lit = np.flatnonzero(lum >= HALO_FLOOR)
        _subtract_spikes(out, y, xc, r, 2.0 * 2 * (lit[-1] + 1) if len(lit) else 0.0,
                         np.delete(sat, j, axis=0))
        sig = max(4.0, 0.5 * r)
        rf = r * (1.0 + SAT_FEATHER)
        R = int(rf + 3 * sig)
        y0, y1, x0, x1 = max(0, y - R), min(h, y + R + 1), max(0, xc - R), min(w, xc + R + 1)
        yy, xx = np.mgrid[y0 - y:y1 - y, x0 - xc:x1 - xc]
        rr = np.hypot(yy, xx)
        disk = rr <= r
        keep = (~disk).astype(np.float32)
        den = np.maximum(ndi.gaussian_filter(keep, sig, mode='nearest'), 1e-6)
        # 1 at r, 0 at rf: the fill's mean blended into the data across the band. A hard
        # edge showed as a smooth disc set into textured surroundings, ringed by whatever
        # halo residue the ring medians left, in a different colour per channel (Pleiades).
        band = (rr > r) & (rr < rf)
        wb = np.clip((rf - rr) / max(rf - r, 1e-6), 0.0, 1.0)
        cut = out[y0:y1, x0:x1]
        for k in range(c):
            fill = ndi.gaussian_filter(cut[..., k] * keep, sig, mode='nearest') / den
            noise = rng.standard_normal(fill.shape).astype(np.float32) * np.sqrt(
                1.0 + np.maximum(fill, 0.0) / np.float32(max(field.B[k], 1e-6)))
            v = cut[..., k]
            v[band] = (wb * fill + (1.0 - wb) * v)[band]
            v[disk] = (fill + noise)[disk]
    return out


def _ring_level(out: np.ndarray, y: int, x: int, r_in: int = 45, r_out: int = 60) -> float:
    """Median luminance of ``out`` (noise units) in the ``r_in``..``r_out`` px ring around
    (y, x): how bright the surroundings a saturated star's halo is measured against are."""
    h, w = out.shape[:2]
    y0, y1, x0, x1 = max(0, y - r_out), min(h, y + r_out + 1), max(0, x - r_out), min(w, x + r_out + 1)
    yy, xx = np.mgrid[y0 - y:y1 - y, x0 - x:x1 - x]
    rr = np.hypot(yy, xx)
    ring = (rr > r_in) & (rr <= r_out)
    if not ring.any():
        return 0.0
    return float(np.median(out[y0:y1, x0:x1].mean(axis=2)[ring]))


def is_grey(z: np.ndarray) -> bool:
    """True when every channel is the same image (a mono stack replicated to 3 channels)."""
    return z.ndim == 3 and z.shape[2] > 1 and all(np.array_equal(z[..., 0], z[..., c]) for c in range(1, z.shape[2]))


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    return ndi.gaussian_filter(a, (sigma, sigma, 0)).astype(np.float32)


def _blur_gain(z: np.ndarray, zg: np.ndarray) -> np.float32:
    """Sky-noise sigma of the blurred image / of the plain one (MAD of the residual to a 9 px
    box mean, so stars and nebulae do not count)."""
    def mad(a):
        d = a[..., 0] - ndi.uniform_filter(a[..., 0], 9)
        return 1.4826 * np.median(np.abs(d - np.median(d)))
    return np.float32(mad(zg) / mad(z))


def starless_z(z: np.ndarray, w, alpha: float = 1.0, field: Optional['StarField'] = None,
               tta: int = 2, seed: int = 0, guard: bool = True, saturated: bool = True,
               cores: bool = True) -> np.ndarray:
    """Noisier2Noise estimate of the starless sky from noise-unit data z (H, W, C):
    ``z - (1 + alpha) / alpha * (z - f)`` with f the trained network's output. The network
    was trained with a sky-noise disc on every star (``StarField.real_discs``); with a
    ``field`` (built on z), f is its mean over ``tta`` inputs ``z + discs`` with independent
    discs, and the known discs stay out of the formula's z. ``guard`` then holds the star
    layer to what stars can be (``constrain_star_layer``) and floors the star footprints
    (``guard_holes``); ``cores`` refills the cores left at the brightest stars
    (``remove_residual_cores``, with ``guard``); ``saturated`` fills the saturated stars
    (``fill_saturated``)."""
    mono = MONO_BLUR_SIGMA > 0 and is_grey(z)
    if mono:
        zg = _blur(z, MONO_BLUR_SIGMA)
        k = _blur_gain(z, zg)
        net = lambda a: nd.denoise_z(_blur(a, MONO_BLUR_SIGMA) / k, w) * k   # noqa: E731
        base = zg
    else:
        net = lambda a: nd.denoise_z(a, w)   # noqa: E731
        base = z
    if field is None or tta <= 0 or not field.ok:
        f = net(z)
    else:
        rng = np.random.default_rng(seed)
        f = np.zeros_like(z)
        for _ in range(tta):
            f += net(z + field.real_discs(rng))
        f /= np.float32(tta)
    x = (base - np.float32((1.0 + alpha) / alpha) * (base - f)).astype(np.float32)
    if mono:
        x = (x + (z - zg)).astype(np.float32)
    if guard and field is not None and field.ok:
        x = guard_holes(constrain_star_layer(z, x, field), field)
        if cores:
            x = guard_holes(remove_residual_cores(x, field, seed), field)
    if saturated and field is not None and field.ok and field.saturated.any():
        x = fill_saturated(x, z, field, seed)
    if mono:   # the fills' per-channel noise and colour fades are chroma a mono image cannot have
        x = np.repeat(x[..., :1], x.shape[-1], axis=-1)
    return x


def model_available(path: Optional[str] = None) -> bool:
    return os.path.exists(path or MODEL_PATH)


def linear_sky_level(img: np.ndarray) -> np.ndarray:
    """The sky level B per channel in noise units of a linear (H, W, 3) image, for
    ``remove_stars_nn(sky_level=)`` on a copy whose sky has since been subtracted."""
    src = np.where(np.isfinite(img), img, 0.0).astype(np.float32)
    return normalise_image(src)[3]


def remove_stars_nn(img: np.ndarray, model_path: Optional[str] = None, alpha: float = 1.0, tta: int = 2,
                    guard: bool = True, saturated: bool = True, sky_level: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """``(starless, stars)`` for a linear (H, W, 3) image; ``alpha`` must be the one the
    model's bank was built with. ``stars = img - starless`` (for screen recombination).

    ``sky_level`` is B per channel (``linear_sky_level`` of the image before its sky was
    removed): after background extraction the image's own sky reads ~0, and B sets the
    photon noise the star-layer constraint and the saturated fill use."""
    w = nd.load_weights(model_path or MODEL_PATH)
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    src = np.where(finite, img, 0.0).astype(np.float32)
    z, bg, sig, B = normalise_image(src)
    if sky_level is not None:
        B = np.asarray(sky_level, np.float64)
    out = starless_z(z, w, alpha, StarField(z, B), tta, guard=guard, saturated=saturated) * sig + bg
    out = np.where(finite, out, img).astype(np.float32)
    return out, np.where(finite, src - out, 0.0).astype(np.float32)
