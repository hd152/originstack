"""Gaia-anchored extended-PSF halo subtraction (``--config halo_removal_method=psf``).

A bright star's wings fall as roughly ``r^-2.5`` and stay above the sky noise for hundreds of
pixels (Pleiades: 100 sigma at 20 px, 20 sigma at 40 px, 2-3 sigma at 80 px). The Gaussian
``denoising.remove_star_halos`` models none of that. This follows Infante-Sainz, Trujillo & Roman
(2020, arXiv:1911.01430), reduced to what runs on a linear stack:

1. **Flux scale from Gaia.** Aperture fluxes of unsaturated Gaia stars give, per channel,
   ``log10 F = a - 0.4 G + b (BP - RP)``. It predicts the flux of every Gaia star -- the
   saturated ones whose flux cannot be measured, and the ones just off the frame whose wings
   reach in.
2. **An empirical radial PSF.** For every reference star, the annular median profile (minus its
   local sky) divided by its predicted flux is that star's PSF estimate; the median over stars,
   radius by radius, is the PSF. Faint stars supply the core, bright stars the wings (a star is
   used at a radius only where it is well above the noise and unsaturated), so the regimes the
   paper splices by hand fall out of the signal-to-noise cut.
3. **Subtract** ``F_i * PSF(r)`` for each halo-bearing star beyond ``r_in`` (a few FWHM; the
   core is left alone), tapering in smoothly.

The amplitude comes from the catalogue, not from the pixels around the star, so a nebula behind
a star is not mistaken for halo (the failure of any fit to the local profile). Limits: Gaia is
incomplete for G < ~7, so the very brightest stars are only modelled when they have a catalogue
entry; the PSF is one radial profile per channel (no spikes, no reflection rings); stars whose
colour term is unknown use the median colour.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

_log = logging.getLogger("originstack")

R_MAX = 150            # px: profile window and subtraction radius
BIN = 2                # px per radial bin
MIN_REF_STARS = 8      # Gaia stars needed for the flux scale
AMOUNT = 0.85          # share of the model removed: the rest is margin for the halo's own asymmetry


def radial_profile(ch: np.ndarray, cy: float, cx: float, radius: int = R_MAX, step: int = BIN
                   ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Annular ``(median, max, count)`` of ``ch`` around (cy, cx), ``radius // step`` bins.

    None when the star is too close to the frame edge for at least half the window."""
    H, W = ch.shape
    R = int(radius) // step * step
    y0, y1 = max(0, int(cy) - R), min(H, int(cy) + R + 1)
    x0, x1 = max(0, int(cx) - R), min(W, int(cx) + R + 1)
    if y1 - y0 < R or x1 - x0 < R:
        return None
    r = np.hypot(np.arange(y0, y1)[:, None] - cy, np.arange(x0, x1)[None, :] - cx)
    sel = r < R
    rb = (r[sel] / step).astype(np.int64)
    v = np.asarray(ch[y0:y1, x0:x1], dtype=np.float64)[sel]
    nb = R // step
    counts = np.bincount(rb, minlength=nb + 1)[:nb]
    order = np.lexsort((v, rb))
    vs = v[order]
    starts = np.cumsum(np.bincount(rb, minlength=nb + 1))[:nb] - counts
    ok = counts >= 8
    med = np.full(nb, np.nan)
    mx = np.full(nb, np.nan)
    med[ok] = vs[(starts + counts // 2)[ok]]
    mx[ok] = vs[(starts + counts - 1)[ok]]
    return med, mx, counts


def fit_flux_scale(flux: np.ndarray, gmag: np.ndarray, colour: np.ndarray,
                   iters: int = 6) -> Optional[Tuple[float, float, float]]:
    """Robust fit of ``log10 F = a - 0.4 G + b * colour``; returns ``(a, b, scatter_dex)`` or None.

    ``colour`` is BP-RP with NaN where unknown (those stars enter with b * 0 after centring)."""
    ok = np.isfinite(flux) & (flux > 0) & np.isfinite(gmag)
    if ok.sum() < MIN_REF_STARS:
        return None
    y = np.log10(flux[ok]) + 0.4 * gmag[ok]
    c = colour[ok]
    has_c = np.isfinite(c)
    c0 = float(np.median(c[has_c])) if has_c.sum() >= MIN_REF_STARS else 0.0
    cc = np.where(has_c, c - c0, 0.0)
    keep = np.ones(y.shape, bool)
    a, b = float(np.median(y)), 0.0
    for _ in range(iters):
        use = keep & has_c if has_c.sum() >= MIN_REF_STARS else keep
        if use.sum() < MIN_REF_STARS:
            break
        if has_c.sum() >= MIN_REF_STARS:
            A = np.column_stack([np.ones(use.sum()), cc[use]])
            coef, *_r = np.linalg.lstsq(A, y[use], rcond=None)
            a, b = float(coef[0]), float(coef[1])
        else:
            a, b = float(np.median(y[use])), 0.0
        res = y - (a + b * cc)
        sg = max(1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep])))), 0.01)
        keep = np.abs(res) < 3.0 * sg           # 0.01 dex floor: flux errors are never below ~2%
    res = y[keep] - (a + b * cc[keep])
    if keep.sum() < MIN_REF_STARS:
        return None
    # report a as the intercept at colour 0 (not c0): a - b * c0
    return float(a - b * c0), float(b), float(1.4826 * np.median(np.abs(res - np.median(res))))


def predicted_flux(a: float, b: float, gmag: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """``10 ** (a - 0.4 G + b * colour)``; stars without colour use colour 0.8 (solar-ish BP-RP)."""
    col = np.where(np.isfinite(colour), colour, 0.8)
    return 10.0 ** (a - 0.4 * gmag + b * col)


def build_psf_profile(ch: np.ndarray, xs: np.ndarray, ys: np.ndarray, flux: np.ndarray,
                      sigma: float, sat_level: float) -> Optional[Dict[str, np.ndarray]]:
    """Radial PSF (per unit flux, per pixel) from reference stars of known flux.

    In every radial bin the stars' annular excesses ``e_i`` are modelled as ``F_i * PSF``; the
    PSF is the inverse-variance weighted least-squares slope ``sum(F e / s^2) / sum(F^2 / s^2)``
    over the stars whose pixels at and beyond that radius are unsaturated, with one 3-sigma
    clipping pass for neighbours. Every star enters whatever its excess, so the estimate is not
    biased upward by keeping only the bins that came out positive. Bins whose slope is not
    3 sigma significant are filled from the neighbours (inside) or a power law (outside).

    Returns ``{'r': bin centres (px), 'psf': per-pixel PSF, 'n': stars behind each bin}`` or None
    when too few bins are measured."""
    nb = R_MAX // BIN
    e = np.full((len(xs), nb), np.nan)
    var = np.full((len(xs), nb), np.nan)
    F = np.asarray(flux, dtype=np.float64)
    for i, (x, y) in enumerate(zip(xs, ys)):
        prof = radial_profile(ch, y, x)
        if prof is None or not F[i] > 0:
            continue
        med, mx, cnt = prof
        valid = np.isfinite(med)
        if valid.sum() < nb // 2:
            continue
        sky = float(np.median(med[valid][-max(3, valid.sum() // 6):]))
        clipped = np.where(np.isfinite(mx), mx >= sat_level, True)
        last_clipped = np.flatnonzero(clipped)
        r_ok = (last_clipped.max() + 1) if last_clipped.size else 0
        use = valid & (np.arange(nb) >= r_ok)
        e[i, use] = (med - sky)[use]
        var[i, use] = (1.2533 * sigma / np.sqrt(np.maximum(cnt, 1)))[use] ** 2
    ok = np.isfinite(e)
    n = ok.sum(axis=0)
    psf = np.full(nb, np.nan)
    snr = np.zeros(nb)
    for b in range(nb):
        u = ok[:, b]
        if u.sum() < 3:
            continue
        f, ee, vv = F[u], e[u, b], var[u, b]
        keep = np.ones(f.shape, bool)
        for _ in range(2):
            w = 1.0 / vv[keep]
            den = float(np.sum(w * f[keep] ** 2))
            if den <= 0:
                break
            slope = float(np.sum(w * f[keep] * ee[keep]) / den)
            res = (ee - slope * f) / np.sqrt(vv)
            keep = np.abs(res) < 3.0
            if keep.sum() < 3:
                keep = np.ones(f.shape, bool)
                break
        w = 1.0 / vv[keep]
        den = float(np.sum(w * f[keep] ** 2))
        slope = float(np.sum(w * f[keep] * ee[keep]) / den)
        psf[b] = slope
        snr[b] = slope * np.sqrt(den)
    r = (np.arange(nb) + 0.5) * BIN
    sig_bins = np.isfinite(psf) & (psf > 0) & (snr >= 3.0)
    if sig_bins.sum() < 3:
        return None
    first = int(np.argmax(sig_bins))
    last = int(len(sig_bins) - 1 - np.argmax(sig_bins[::-1]))
    out = np.full(nb, np.nan)
    out[sig_bins] = psf[sig_bins]
    out[:first] = out[first]                      # saturated cores: hold the first measured value
    # beyond the last significant bin the wing is below what any reference star can show: extend
    # the last decade as a power law (slope clipped to the physical -4..-1.5), no further
    if last < nb - 1:
        lo = max(first, last - 8)
        seg = sig_bins[lo:last + 1]
        if seg.sum() >= 3:
            slope_p, _ic = np.polyfit(np.log(r[lo:last + 1][seg]), np.log(out[lo:last + 1][seg]), 1)
            slope_p = float(np.clip(slope_p, -4.0, -1.5))
            out[last + 1:] = np.exp(np.log(out[last]) + slope_p * (np.log(r[last + 1:]) - np.log(r[last])))
        else:
            out[last + 1:] = 1e-30
    gaps = ~np.isfinite(out)
    if gaps.any() and (~gaps).sum() >= 3:
        out[gaps] = np.exp(np.interp(np.log(r[gaps]), np.log(r[~gaps]), np.log(out[~gaps])))
    out = np.minimum.accumulate(np.maximum(out, 0.0))     # a PSF only falls; bumps are not halo
    out = np.maximum(out, 1e-30)
    return {'r': r, 'psf': out, 'n': n}


N_SECTORS = 12


def _sector_excess(ch: np.ndarray, y: float, x: float, sigma: float
                   ) -> Optional[np.ndarray]:
    """Light above the local sky in ``N_SECTORS`` angular sectors x radial bins around (y, x).

    The star's own halo is not radially symmetric (flare, internal reflections, a neighbour, a
    nebula on one side), so a radial model that is right on average digs a moat where the real halo
    is weaker. This is what is there to remove, per sector: sector median minus the star's local sky
    (outer-ring median), plus one standard error of the median, floored at 0. None near an edge."""
    H, W = ch.shape
    R = R_MAX // BIN * BIN
    y0, y1 = max(0, int(y) - R), min(H, int(y) + R + 1)
    x0, x1 = max(0, int(x) - R), min(W, int(x) + R + 1)
    if y1 - y0 < R or x1 - x0 < R:
        return None
    dy = np.arange(y0, y1)[:, None] - y
    dx = np.arange(x0, x1)[None, :] - x
    r = np.hypot(dy, dx)
    sel = r < R
    rb = (r[sel] / BIN).astype(np.int64)
    sec = ((np.arctan2(dy, dx)[sel] + np.pi) / (2 * np.pi) * N_SECTORS).astype(np.int64) % N_SECTORS
    key = sec * (R // BIN) + rb
    v = np.asarray(ch[y0:y1, x0:x1], dtype=np.float64)[sel]
    nk = N_SECTORS * (R // BIN)
    cnt = np.bincount(key, minlength=nk + 1)[:nk]
    order = np.lexsort((v, key))
    vs = v[order]
    starts = np.cumsum(np.bincount(key, minlength=nk + 1))[:nk] - cnt
    med = np.full(nk, np.nan)
    ok = cnt >= 6
    med[ok] = vs[(starts + cnt // 2)[ok]]
    med = med.reshape(N_SECTORS, R // BIN)
    cnt = cnt.reshape(N_SECTORS, R // BIN)
    outer = np.nanmedian(med[:, -max(3, (R // BIN) // 6):])
    se = 1.2533 * sigma / np.sqrt(np.maximum(cnt, 1))
    cap = np.maximum(np.where(np.isfinite(med), med - outer + 1.5 * se, 0.0), 0.0)
    # the lowest of each sector and its two neighbours, smoothed: a cap should follow the weak
    # side of the halo, not one noisy sector's dip
    low = np.minimum(np.minimum(cap, np.roll(cap, 1, axis=0)), np.roll(cap, -1, axis=0))
    return 0.5 * (low + 0.25 * (np.roll(low, 1, axis=0) + 2.0 * low + np.roll(low, -1, axis=0)))


def render_halos(shape: Tuple[int, int], xs: np.ndarray, ys: np.ndarray, flux: np.ndarray,
                 profile: Dict[str, np.ndarray], r_in: float,
                 ch: Optional[np.ndarray] = None, sigma: float = 0.0) -> np.ndarray:
    """Sum of ``flux_i * PSF(r)`` for the given centres (off-frame ones included), beyond ``r_in``.

    The profile is lowered by its value at the window edge so the model ends at zero instead of
    with a step, and tapers in over ``[r_in, 1.5 r_in]``. With ``ch`` (the image channel) each
    star's model is also capped, sector by sector, at the light actually present there
    (``_sector_excess``), so an asymmetric halo is not over-subtracted into a moat."""
    H, W = shape
    r_grid, psf = profile['r'], profile['psf']
    edge = float(psf[-1])
    out = np.zeros(shape, dtype=np.float64)
    R = int(R_MAX)
    nb = R // BIN
    for x, y, f in zip(xs, ys, flux):
        if not f > 0:
            continue
        y0, y1 = max(0, int(y) - R), min(H, int(y) + R + 1)
        x0, x1 = max(0, int(x) - R), min(W, int(x) + R + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        dy = np.arange(y0, y1)[:, None] - y
        dx = np.arange(x0, x1)[None, :] - x
        r = np.hypot(dy, dx)
        p = np.exp(np.interp(np.log(np.maximum(r, r_grid[0])), np.log(r_grid), np.log(psf)))
        p = np.where(r < R, np.maximum(p - edge, 0.0), 0.0)
        t = np.clip((r - r_in) / (0.5 * r_in), 0.0, 1.0)
        model = AMOUNT * f * p * (t * t * (3.0 - 2.0 * t))
        if ch is not None:
            cap = _sector_excess(ch, y, x, sigma)
            if cap is not None:
                rb = np.minimum((r / BIN).astype(np.int64), nb - 1)
                # linear in angle between sector centres (periodic): no seams between sectors
                u = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * N_SECTORS - 0.5
                i0 = np.floor(u).astype(np.int64)
                fr = u - i0
                cap_here = (1.0 - fr) * cap[i0 % N_SECTORS, rb] + fr * cap[(i0 + 1) % N_SECTORS, rb]
                model = np.minimum(model, cap_here)
        out[y0:y1, x0:x1] += model
    return out


def subtract_halos(img: np.ndarray, header, fwhm: float, verbose: bool = False
                   ) -> Tuple[np.ndarray, Optional[Dict[str, object]]]:
    """Subtract the Gaia-anchored halo model from every channel of ``img`` (H, W, 3).

    Returns ``(image, info)``; ``info`` is None when the step declined (no WCS, no network or
    catalogue, too few reference stars), and ``image`` is then the input."""
    from src import net_query
    from src.background import _pixel_noise_sigma
    from src.photometry_core import _celestial_wcs, _field_centre_and_radius, aperture_photometry_batch
    H, W = img.shape[:2]
    centre = _field_centre_and_radius(header, img.shape)
    if centre is None:
        return img, {'declined': 'no WCS'}
    ra0, dec0, rad_deg, scale_as = centre
    rad = min(float(rad_deg) + R_MAX * float(scale_as) / 3600.0, 2.5)
    try:
        cat = net_query.gaia_cone_search(
            ra0, dec0, rad, ["ra", "dec", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
            max_rows=1500, require_not_null=["phot_g_mean_mag"])
    except Exception as exc:
        _log.debug("halo model: Gaia query failed: %s", exc)
        cat = None
    if cat is None or len(cat) < MIN_REF_STARS:
        return img, {'declined': 'no Gaia catalogue'}
    w = _celestial_wcs(header)
    px, py = w.all_world2pix(np.asarray(cat['ra'], float), np.asarray(cat['dec'], float), 0)
    g = np.asarray(cat['phot_g_mean_mag'], float)
    bp = np.ma.filled(np.ma.asarray(cat['phot_bp_mean_mag']).astype(float), np.nan)
    rp = np.ma.filled(np.ma.asarray(cat['phot_rp_mean_mag']).astype(float), np.nan)
    colour = bp - rp
    inside = (px > -R_MAX) & (px < W + R_MAX) & (py > -R_MAX) & (py < H + R_MAX) & np.isfinite(px)
    px, py, g, colour = px[inside], py[inside], g[inside], colour[inside]
    in_frame = (px > 12) & (px < W - 12) & (py > 12) & (py < H - 12)

    out = img.astype(np.float32, copy=True)
    # the measured stack FWHM is biased wide where bright halos dominate the sample (9 px on a
    # 4.5 px Pleiades stack): cap it, or the untouched core grows to a visible disc
    fw = min(float(fwhm), 6.0)
    r_in = max(6.0, 2.5 * fw)
    ap = max(3.0, 1.6 * fw)
    totals = {}
    idx = np.flatnonzero(in_frame)
    flux_all, _sky, _sg, peak_all, _area = aperture_photometry_batch(
        np.ascontiguousarray(img, dtype=np.float32), px[idx], py[idx], ap, ap + 3.0,
        ap + 3.0 + max(5.0, fw * 2.0))
    for c in range(img.shape[2]):
        ch = np.ascontiguousarray(img[:, :, c], dtype=np.float64)
        sigma = _pixel_noise_sigma(ch.astype(np.float32))
        sat = float(np.percentile(ch, 99.995)) * 0.9
        fl, pk = flux_all[:, c], peak_all[:, c]
        usable = np.isfinite(fl) & (fl > 0) & (pk < sat) & (pk > 10.0 * sigma)
        fit = fit_flux_scale(np.where(usable, fl, np.nan), g[idx], colour[idx])
        if fit is None:
            return img, {'declined': f'too few unsaturated Gaia stars in channel {c}'}
        a, b, scatter = fit
        pred = predicted_flux(a, b, g, colour)
        # reference stars for the PSF: in the frame, flux from the catalogue scale (so saturated
        # stars carry their true flux), brightest first
        ref = idx[np.argsort(-pred[idx])][:90]
        prof = build_psf_profile(ch, px[ref], py[ref], pred[ref], sigma, sat)
        if prof is None:
            return img, {'declined': f'could not build a PSF profile in channel {c}'}
        # halo-bearing stars: wing at 2 r_in above 0.2 sigma
        p_at = float(np.exp(np.interp(np.log(2.0 * r_in), np.log(prof['r']), np.log(prof['psf']))))
        bearing = pred * p_at > 0.2 * sigma
        model = render_halos((H, W), px[bearing], py[bearing], pred[bearing], prof, r_in, ch, sigma)
        out[:, :, c] = (ch - model).astype(np.float32)
        totals[c] = {'a': a, 'b': b, 'scatter_dex': scatter, 'stars': int(bearing.sum()),
                     'model_peak_sigma': float(model.max() / sigma),
                     'psf_wing_40px': float(np.exp(np.interp(np.log(40.0), np.log(prof['r']), np.log(prof['psf']))))}
        if verbose:
            _log.debug("halo model ch%d: %s", c, totals[c])
    return out, {'channels': totals, 'r_in': r_in}

