"""Low-level photometry primitives shared by the photometry, time-series
photometry and colour-calibration paths.

Kept dependency-light on purpose: numpy + stdlib + the optional
``astro_native`` kernel, with lazy ``astropy`` imports inside the functions
that need a WCS. Higher-level orchestration (Gaia cross-match, zero-point
fitting, the CLI-facing ``run_*`` entry points) lives in ``photometry.py``
and ``photometry_timeseries.py``.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import astro_native as _NATIVE
except Exception:  # pragma: no cover - native module is optional
    _NATIVE = None

_APB_SUBPIX = 4  # aperture-edge supersampling factor (native + numpy paths)


# ---------------------------------------------------------------------------
# WCS helpers
# ---------------------------------------------------------------------------

def _field_centre_and_radius(header, shape):
    """(ra_deg, dec_deg, search_radius_deg, plate_scale_arcsec) from the WCS,
    or None when the header carries no usable celestial WCS."""
    try:
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
    except Exception:
        return None
    try:
        w = WCS(header).celestial
        if not w.has_celestial:
            return None
        H, W = shape[:2]
        sky = w.pixel_to_world(W / 2.0, H / 2.0)
        ra0 = float(sky.ra.deg)
        dec0 = float(sky.dec.deg)
        scales_deg = proj_plane_pixel_scales(w)  # deg/pixel, (x, y)
        scale_arcsec = float(np.mean(scales_deg)) * 3600.0
        half_diag_deg = 0.5 * math.hypot(W * scales_deg[0], H * scales_deg[1])
        return ra0, dec0, half_diag_deg * 1.15, scale_arcsec
    except Exception:
        return None


def _pixel_coords(table, header) -> Optional[np.ndarray]:
    """Project a catalogue's RA/Dec onto image pixels via the header WCS.

    Accepts either lowercase ``ra``/``dec`` (Gaia TAP) or ``RAJ2000``/
    ``DEJ2000`` (VizieR) columns. Returns an ``(N, 2)`` array of ``(x, y)``
    0-based pixel coordinates, or None.
    """
    try:
        from astropy.wcs import WCS
    except Exception:
        return None
    try:
        wcs = WCS(header)
        if "ra" in table.colnames:
            ra = np.array(table["ra"], dtype=float)
            dec = np.array(table["dec"], dtype=float)
        else:
            ra = np.array(table["RAJ2000"], dtype=float)
            dec = np.array(table["DEJ2000"], dtype=float)
        x, y = wcs.all_world2pix(ra, dec, 0)
        return np.column_stack([x, y])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Small array helpers
# ---------------------------------------------------------------------------

def _id_str(v):
    """Best-effort int cast of a catalogue id, falling back to str."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


def row_nanmax(a, fallback):
    """Per-row max ignoring NaN, without numpy's All-NaN-slice warning;
    rows that are entirely NaN take the matching ``fallback`` value."""
    a = np.asarray(a, dtype=np.float64)
    all_nan = ~np.any(np.isfinite(a), axis=1)
    out = np.where(np.isfinite(a), a, -np.inf).max(axis=1)
    return np.where(all_nan, np.asarray(fallback, dtype=np.float64), out)


# ---------------------------------------------------------------------------
# Batch aperture photometry (native Rust kernel + numpy mirror)
# ---------------------------------------------------------------------------

def aperture_photometry_batch(img, xs, ys, r_ap, r_in, r_out,
                              subpix: int = _APB_SUBPIX):
    """Partial-pixel circular-aperture photometry for many centres at once.

    Returns ``(flux, sky, sky_sigma, peak, area)`` -- ``flux``/``sky``/
    ``sky_sigma``/``peak`` are ``(N, C)`` (background-subtracted flux,
    robust sky median, ``1.4826*MAD`` sky sigma, and the raw max pixel
    value inside the aperture per channel), ``area`` is ``(N,)`` (effective
    aperture pixel area). A star whose full ``r_out`` disk is not inside
    the frame gets an all-NaN row. Integer pixel coordinates are pixel
    centres (matches ``star_detect``).

    Dispatches to the native ``aperture_photometry_batch`` kernel when
    built; ``_aperture_photometry_batch_numpy`` otherwise.
    """
    img_c = np.ascontiguousarray(img, dtype=np.float32)
    xs_c = np.ascontiguousarray(xs, dtype=np.float64)
    ys_c = np.ascontiguousarray(ys, dtype=np.float64)
    if _NATIVE is not None and hasattr(_NATIVE, "aperture_photometry_batch"):
        return _NATIVE.aperture_photometry_batch(
            img_c, xs_c, ys_c, float(r_ap), float(r_in), float(r_out),
            int(subpix))
    return _aperture_photometry_batch_numpy(
        img_c, xs_c, ys_c, float(r_ap), float(r_in), float(r_out), int(subpix))


def _aperture_photometry_batch_numpy(img, xs, ys, r_ap, r_in, r_out,
                                     subpix=_APB_SUBPIX):
    """Numpy reference for :func:`aperture_photometry_batch` (parity-tested
    against the native kernel in ``tests/test_native.py``)."""
    a = np.asarray(img, dtype=np.float64)
    H, W, C = a.shape
    N = len(xs)
    flux = np.full((N, C), np.nan)
    sky = np.full((N, C), np.nan)
    sig = np.full((N, C), np.nan)
    peak = np.full((N, C), np.nan)
    area = np.full(N, np.nan)

    sub_off = (np.arange(subpix) + 0.5) / subpix - 0.5
    oy, ox = np.meshgrid(sub_off, sub_off, indexing="ij")
    half_diag = math.sqrt(0.5)
    full_in = (r_ap - half_diag) ** 2 if r_ap > half_diag else -1.0
    full_out = (r_ap + half_diag) ** 2
    r_ap2, r_in2, r_out2 = r_ap ** 2, r_in ** 2, r_out ** 2

    for i in range(N):
        cx, cy = float(xs[i]), float(ys[i])
        if (not np.isfinite(cx) or not np.isfinite(cy)
                or cx - r_out < 0 or cx + r_out >= W - 1
                or cy - r_out < 0 or cy + r_out >= H - 1):
            continue
        x0 = max(int(math.floor(cx - r_out)) - 1, 0)
        y0 = max(int(math.floor(cy - r_out)) - 1, 0)
        x1 = min(int(math.ceil(cx + r_out)) + 1, W - 1)
        y1 = min(int(math.ceil(cy + r_out)) + 1, H - 1)
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        dy = yy - cy
        dx = xx - cx
        d2 = dx * dx + dy * dy

        frac = np.zeros(d2.shape, dtype=np.float64)
        if full_in > 0:
            frac[d2 <= full_in] = 1.0
        edge = (d2 > (full_in if full_in > 0 else -1.0)) & (d2 < full_out)
        if np.any(edge):
            de_y = dy[edge][:, None, None] + oy[None]
            de_x = dx[edge][:, None, None] + ox[None]
            frac[edge] = (de_x ** 2 + de_y ** 2 <= r_ap2).mean(axis=(1, 2))

        patch = a[y0:y1 + 1, x0:x1 + 1, :]
        ap_area = float(frac.sum())
        area[i] = ap_area
        ap_sum = (patch * frac[..., None]).sum(axis=(0, 1))
        ap_mask = frac > 0.0
        ann = (d2 > r_in2) & (d2 <= r_out2)
        for c in range(C):
            if np.any(ap_mask):
                peak[i, c] = float(patch[ap_mask, c].max())
            vals = patch[ann, c]
            if vals.size < 4:
                continue
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            sky[i, c] = med
            sig[i, c] = 1.4826 * mad
            flux[i, c] = ap_sum[c] - med * ap_area
    return flux, sky, sig, peak, area
