"""A survey H-alpha reference for background extraction (``--config dbe_halpha_reference=true``).

DBE cannot tell a faint nebula that fills the frame from a sky gradient: both are large and
smooth, and the surface it fits takes both. The all-sky H-alpha composite of Finkbeiner (2003;
VTSS + SHASSA + WHAM, 6 arcmin FWHM, Rayleigh) knows where the emission is. Here the map is
resampled onto the stack's grid, its scale to each channel is fitted on the image's own block
medians together with a quadratic surface (the gradient), and the scaled map is taken out of
the image before DBE runs and put back afterwards, so DBE never sees -- and never removes --
that part of the nebula. This is the idea of PixInsight's MultiscaleGradientCorrection with a
public reference instead of the MARS survey.

The map is coarse (6.9 arcmin pixels: ~280 px of an Origin frame), so it only carries the
nebula's low-order shape; finer structure is still DBE's own masks' job. It declines (plain
DBE) when the field has no emission to speak of, when the map's shape over the field is itself
close to a quadratic surface (the scale is then not measurable: Flaming Star), or when the
image does not follow the map.

Measured on two nights of NGC 2244 (tools/eval_background_nights.py ``--method dbe dbe_ref``):
night-to-night mismatch 0.351 -> 0.199 sky sigma, target light kept 0.21 -> 0.42; the Eagle
Nebula 0.339 -> 0.221, kept 0.36 -> 0.76.

The map (12.6 MB, NASA LAMBDA) is downloaded once into the star index folder;
``$ORIGINSTACK_HALPHA_MAP`` points at a copy elsewhere.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

_log = logging.getLogger('originstack')

MAP_NAME = 'lambda_halpha_fwhm06_0512.fits'
MAP_URL = 'https://lambda.gsfc.nasa.gov/data/foregrounds/halpha/' + MAP_NAME
MAP_BYTES = 12_594_240
HEALPIX_PIXEL_ARCMIN = 6.87          # nside 512
BLOCK = 64                           # px; the fit runs on block medians
MIN_CONTRAST_R = 20.0                # Rayleigh, 5th-95th percentile over the field: below, no nebula
MIN_INDEPENDENT = 0.25               # share of the map's spread a quadratic surface cannot follow
MIN_PARTIAL_CORR = 0.30              # image vs map, both with the quadratic surface fitted out

# ICRS (J2000) -> Galactic rotation (Hipparcos definition, ESA 1997 vol. 1 eq. 1.5.11)
_TO_GALACTIC = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669]])

_map_cache: dict = {}


def _spread_bits(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.int64) & 0xFFFFFFFF
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & 0x3333333333333333
    v = (v | (v << 1)) & 0x5555555555555555
    return v


def ang2pix_nest(nside: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """HEALPix NESTED pixel of colatitude ``theta`` and longitude ``phi`` (radians); ``nside``
    a power of two. Gorski et al. 2005, as healpy's ``ang2pix(nest=True)``."""
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    z = np.cos(theta)
    za = np.abs(z)
    tt = np.mod(phi, 2.0 * np.pi) / (0.5 * np.pi)            # [0, 4)
    # equatorial belt
    t1 = nside * (0.5 + tt)
    t2 = nside * z * 0.75
    jp = np.floor(t1 - t2).astype(np.int64)
    jm = np.floor(t1 + t2).astype(np.int64)
    ifp, ifm = jp // nside, jm // nside
    f_eq = np.where(ifp == ifm, (ifp & 3) + 4, np.where(ifp < ifm, ifp & 3, (ifm & 3) + 8))
    ix_eq = jm & (nside - 1)
    iy_eq = nside - (jp & (nside - 1)) - 1
    # polar caps
    ntt = np.minimum(tt.astype(np.int64), 3)
    tp = tt - ntt
    tmp = nside * np.sqrt(3.0 * (1.0 - za))
    jp2 = np.minimum((tp * tmp).astype(np.int64), nside - 1)
    jm2 = np.minimum(((1.0 - tp) * tmp).astype(np.int64), nside - 1)
    north = z >= 0
    f_pol = np.where(north, ntt, ntt + 8)
    ix_pol = np.where(north, nside - jm2 - 1, jp2)
    iy_pol = np.where(north, nside - jp2 - 1, jm2)
    eq = za <= 2.0 / 3.0
    face = np.where(eq, f_eq, f_pol)
    ix = np.where(eq, ix_eq, ix_pol)
    iy = np.where(eq, iy_eq, iy_pol)
    return face * nside * nside + _spread_bits(ix) + 2 * _spread_bits(iy)


def icrs_to_galactic(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Galactic ``(l, b)`` in radians of ICRS positions in degrees (a fixed rotation)."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    v = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)
    g = v @ _TO_GALACTIC.T
    return np.arctan2(g[..., 1], g[..., 0]), np.arcsin(np.clip(g[..., 2], -1.0, 1.0))


def map_path() -> str:
    env = os.environ.get('ORIGINSTACK_HALPHA_MAP')
    if env:
        return env
    from src.local_solve import star_index_dir
    return os.path.join(star_index_dir(), 'halpha', MAP_NAME)


def load_map(path: Optional[str] = None, download: bool = True) -> Optional[np.ndarray]:
    """The map as a float32 NESTED HEALPix array (Rayleigh, Galactic), or None when it is
    not on disk and cannot be fetched (``--offline``, no network). Read once per process."""
    path = path or map_path()
    m = _map_cache.get(path)
    if m is not None:
        return m
    if not os.path.exists(path):
        if not download:
            return None
        try:
            from src.net_query import fetch_halpha_map
            fetch_halpha_map(MAP_URL, path, MAP_BYTES)
        except Exception as e:
            _log.debug("H-alpha map download failed", exc_info=True)
            from src.utils import safe_print
            safe_print(f"    H-alpha reference map not available ({e})")
            return None
    try:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdul:
            hdr = hdul[1].header
            if str(hdr.get('ORDERING', '')).strip().upper() != 'NESTED':
                raise ValueError('not a NESTED HEALPix table')
            m = np.asarray(hdul[1].data.field(0), dtype=np.float32).ravel()
        nside = int(round(np.sqrt(m.size / 12.0)))
        if 12 * nside * nside != m.size or nside & (nside - 1):
            raise ValueError(f'{m.size} pixels is not a HEALPix map')
    except Exception as e:
        _log.debug("H-alpha map unreadable", exc_info=True)
        from src.utils import safe_print
        safe_print(f"    H-alpha reference map unreadable ({e})")
        return None
    _map_cache[path] = m
    return m


def halpha_at(hmap: np.ndarray, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Map value (Rayleigh) at ICRS positions: the HEALPix pixel each falls in."""
    nside = int(round(np.sqrt(hmap.size / 12.0)))
    lon, lat = icrs_to_galactic(ra_deg, dec_deg)
    return hmap[ang2pix_nest(nside, 0.5 * np.pi - lat, lon)]


def field_map(hmap: np.ndarray, header, shape: Tuple[int, int], step: int = 32) -> Optional[np.ndarray]:
    """The map on the stack's pixel grid (H, W) float32, or None without a celestial WCS.
    Sampled every ``step`` px, smoothed by about half a map pixel (the lookup is
    nearest-pixel) and interpolated linearly."""
    from scipy import ndimage

    from src.photometry_core import _celestial_wcs
    try:
        w = _celestial_wcs(header)
        if w is None or not w.has_celestial:
            return None
        from astropy.wcs.utils import proj_plane_pixel_scales
        scale_deg = float(np.mean(np.abs(proj_plane_pixel_scales(w))))
    except Exception:
        _log.debug("H-alpha reference: no usable WCS", exc_info=True)
        return None
    if not np.isfinite(scale_deg) or scale_deg <= 0:
        return None
    H, W = int(shape[0]), int(shape[1])
    ys = np.arange(-2 * step, H + 3 * step, step, dtype=np.float64)
    xs = np.arange(-2 * step, W + 3 * step, step, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    ra, dec = w.all_pix2world(X, Y, 0)
    if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))):
        return None
    grid = halpha_at(hmap, ra, dec).astype(np.float64)
    map_px = HEALPIX_PIXEL_ARCMIN / 60.0 / scale_deg          # stack px per map pixel
    grid = ndimage.gaussian_filter(grid, 0.45 * map_px / step, mode='nearest')
    zy = (np.arange(H, dtype=np.float64) + 2 * step) / step
    zx = (np.arange(W, dtype=np.float64) + 2 * step) / step
    out = ndimage.map_coordinates(grid, np.meshgrid(zy, zx, indexing='ij'), order=1, mode='nearest')
    return out.astype(np.float32)


def _block_medians(a: np.ndarray) -> np.ndarray:
    nby, nbx = a.shape[0] // BLOCK, a.shape[1] // BLOCK
    v = a[:nby * BLOCK, :nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    # every 2nd pixel of a block: 1024 samples, plenty for a median
    return np.median(v[:, ::2, :, ::2].transpose(0, 2, 1, 3).reshape(nby, nbx, -1), axis=2)


def _robust_fit(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = np.ones_like(y)
    co = np.zeros(A.shape[1])
    for _ in range(10):
        sw = np.sqrt(w)
        co = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = y - A @ co
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        u = r / (4.685 * s)
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
    return co


def fit_scales(rgb: np.ndarray, hfield: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """Per-channel scale (image units per Rayleigh) of the map in ``rgb`` (H, W, 3), from block
    medians fitted as ``scale * map + quadratic surface``. Returns ``(scales, note)``; scales
    is None when the reference is declined, and ``note`` says why (or what was found)."""
    Hb = _block_medians(hfield).astype(np.float64)
    nby, nbx = Hb.shape
    if nby < 6 or nbx < 6:
        return None, "frame too small"
    lo, hi = np.percentile(Hb, [5, 95])
    if hi - lo < MIN_CONTRAST_R:
        return None, f"no H-alpha structure in the field ({lo:.0f}-{hi:.0f} R)"
    yy, xx = np.mgrid[0:nby, 0:nbx]
    yy = (yy / nby - 0.5).ravel()
    xx = (xx / nbx - 0.5).ravel()
    P = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], axis=1)
    h = Hb.ravel()
    h_res = h - P @ np.linalg.lstsq(P, h, rcond=None)[0]
    independent = float(np.std(h_res) / (np.std(h) + 1e-12))
    if independent < MIN_INDEPENDENT:
        return None, (f"the map's shape over this field is close to a smooth gradient "
                      f"({independent:.2f} of its spread is not): its scale cannot be measured")
    A = np.column_stack([h, P])
    scales = np.zeros(3, dtype=np.float64)
    corr = np.zeros(3, dtype=np.float64)
    for c in range(3):
        y = _block_medians(rgb[:, :, c]).astype(np.float64).ravel()
        ok = np.isfinite(y)
        if ok.sum() < 30:
            continue
        co = _robust_fit(A[ok], y[ok])
        y_res = y[ok] - P[ok] @ np.linalg.lstsq(P[ok], y[ok], rcond=None)[0]
        hr = h_res[ok]
        # rank correlation: stars and the bright core must not carry it
        ry, rh = np.argsort(np.argsort(y_res)), np.argsort(np.argsort(hr))
        corr[c] = float(np.corrcoef(ry, rh)[0, 1])
        if co[0] > 0 and corr[c] >= MIN_PARTIAL_CORR:
            scales[c] = co[0]
    if scales[0] <= 0:
        return None, f"the image does not follow the map (red correlation {corr[0]:+.2f})"
    return scales, (f"scale R/G/B {scales[0]:.3g}/{scales[1]:.3g}/{scales[2]:.3g} per Rayleigh, "
                    f"correlation {corr[0]:.2f}/{corr[1]:.2f}/{corr[2]:.2f}, field {lo:.0f}-{hi:.0f} R")


def reference_model(rgb: np.ndarray, header, download: bool = True) -> Tuple[Optional[np.ndarray], str]:
    """The nebula's large-scale light as the survey sees it, in the image's units: an
    (H, W, 3) float32 array to subtract before background extraction and add back after, or
    None with the reason. Zero at the field's faintest part (5th percentile of the map), so the
    sky DBE leaves still sits at zero there."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None, "needs a colour image"
    if header is None:
        return None, "no stack WCS"
    hmap = load_map(download=download)
    if hmap is None:
        return None, "map not available"
    hfield = field_map(hmap, header, rgb.shape[:2])
    if hfield is None:
        return None, "no usable WCS"
    scales, note = fit_scales(rgb, hfield)
    if scales is None:
        return None, note
    base = float(np.percentile(hfield[::8, ::8], 5))
    shape = np.maximum(hfield - np.float32(base), np.float32(0.0))
    return shape[:, :, None] * scales.astype(np.float32)[None, None, :], note
