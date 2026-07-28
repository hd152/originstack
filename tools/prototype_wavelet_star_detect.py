"""Prototype (NOT wired into the pipeline): multiscale/wavelet (starlet)
star detection, as a candidate replacement for SEP/DAOStarFinder.

Round 2, after round 1 exposed a real precision problem (70-90% false
positives with a single global MAD threshold on a summed 2-scale band):

  - Local (mesh-based) per-scale noise sigma instead of one global MAD --
    real frames have spatially-varying noise/background (nebula, vignetting,
    sky gradient); a single global sigma is either too loose in low-noise
    regions or too tight in high-variance ones.
  - Interscale coincidence: a candidate must be significant at TWO adjacent
    wavelet scales, spatially coincident (not just amplitude-summed) --
    single-scale noise fluctuations rarely reproduce at the next scale up,
    real PSF-shaped sources do by construction.
  - Roundness filter on the second-moment shape (matches SEP wrapper's own
    convention: reject roundness > 0.5) -- correlated-noise blobs are often
    elongated/irregular, real stars are round.
  - Vectorized per-blob measurement via scipy.ndimage batch ops
    (center_of_mass/sum/maximum over labeled arrays) instead of a Python
    loop over thousands of candidates -- fixes both the false-positive-driven
    slowdown and makes iteration on the algorithm itself fast.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from astropy.io import fits
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.background import _starlet_transform
from src.quality import _sep_detect_stars, _SOURCES_DTYPE, _ensure_photutils, _detect_stars_multi_fwhm


def _local_sigma_map(band: np.ndarray, cell: int = 64) -> np.ndarray:
    H, W = band.shape
    ny, nx = max(1, H // cell), max(1, W // cell)
    sig_grid = np.zeros((ny, nx))
    for iy in range(ny):
        y0 = iy * cell
        y1 = H if iy == ny - 1 else (iy + 1) * cell
        for ix in range(nx):
            x0 = ix * cell
            x1 = W if ix == nx - 1 else (ix + 1) * cell
            c = band[y0:y1, x0:x1]
            med = np.median(c)
            mad = np.median(np.abs(c - med))
            sig_grid[iy, ix] = 1.4826 * max(mad, 1e-9)
    sig_full = ndimage.zoom(sig_grid, (H / ny, W / nx), order=1)[:H, :W]
    return ndimage.gaussian_filter(sig_full, sigma=cell * 0.3)


def _local_median_map(img: np.ndarray, cell: int = 64) -> np.ndarray:
    """Same mesh-based construction as _local_sigma_map, but the local
    median (background level) instead of the local MAD sigma."""
    H, W = img.shape
    ny, nx = max(1, H // cell), max(1, W // cell)
    med_grid = np.zeros((ny, nx))
    for iy in range(ny):
        y0 = iy * cell
        y1 = H if iy == ny - 1 else (iy + 1) * cell
        for ix in range(nx):
            x0 = ix * cell
            x1 = W if ix == nx - 1 else (ix + 1) * cell
            med_grid[iy, ix] = np.median(img[y0:y1, x0:x1])
    med_full = ndimage.zoom(med_grid, (H / ny, W / nx), order=1)[:H, :W]
    return ndimage.gaussian_filter(med_full, sigma=cell * 0.3)


def _gaussian_kernel(fwhm: float) -> np.ndarray:
    sigma = fwhm / 2.3548
    radius = max(3, int(np.ceil(3.0 * sigma)))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def matched_filter_detect_stars(lum: np.ndarray, fwhm: float = 5.0, k_confirm: float = 8.0,
                                cell: int = 64, min_pixels: int = 2,
                                roundness_max: float = 0.5) -> np.ndarray:
    """Matched-filter detection: convolve the background-subtracted image
    with a Gaussian kernel shaped like the expected stellar PSF. This is the
    theoretically SNR-optimal linear statistic for detecting a known-shape
    signal in noise (the matched filter theorem) -- conceptually what
    DAOStarFinder does (it convolves with a Gaussian of the given FWHM
    before finding local maxima), tried here as an alternative/complement
    to the wavelet-interscale approach rather than a blunt per-pixel
    threshold on raw or wavelet-transformed pixel values.

    Noise propagates through the (unit-sum) kernel as
    sigma_out = sigma_in * sqrt(sum(kernel**2)) (variance of a weighted sum
    of independent per-pixel noise), giving a proper per-pixel SNR map
    rather than a re-thresholded amplitude.
    """
    lum64 = lum.astype(np.float64)
    H, W = lum64.shape
    bg_map = _local_median_map(lum64, cell=cell)
    resid = lum64 - bg_map

    kernel = _gaussian_kernel(fwhm)
    filtered = ndimage.convolve(resid, kernel, mode='reflect')
    kernel_norm = float(np.sqrt((kernel ** 2).sum()))
    sigma_map = _local_sigma_map(lum64, cell=cell)
    snr_map = filtered / np.maximum(sigma_map * kernel_norm, 1e-9)

    footprint = max(3, int(round(fwhm)))
    is_local_max = snr_map == ndimage.maximum_filter(snr_map, size=footprint)
    detect_mask = is_local_max & (snr_map > k_confirm)
    # The mesh bg/sigma maps (plain ndimage.zoom, no edge-aware extrapolation
    # -- unlike background.py's own gaussian_filter_ds, which documents this
    # exact corner-alignment failure mode) are unreliable right at the image
    # boundary. Exclude a margin rather than fix the interpolation here.
    border = max(cell // 2, 2 * int(np.ceil(3.0 * fwhm / 2.3548)))
    edge_mask = np.zeros((H, W), dtype=bool)
    edge_mask[border:H - border, border:W - border] = True
    detect_mask &= edge_mask
    ys, xs = np.where(detect_mask)
    if len(ys) == 0:
        return np.zeros(0, dtype=_SOURCES_DTYPE)

    rows = []
    r = max(3, int(round(1.5 * fwhm)))
    for py, px in zip(ys, xs):
        y0, y1 = max(py - r, 0), min(py + r + 1, H)
        x0, x1 = max(px - r, 0), min(px + r + 1, W)
        cut = lum64[y0:y1, x0:x1]
        local_bg = float(bg_map[py, px])
        w = np.clip(cut - local_bg, 0, None)
        if w.sum() <= 0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cy = float((w * yy).sum() / w.sum())
        cx = float((w * xx).sum() / w.sum())

        # Refinement pass: re-centroid with a tighter window centred on the
        # first-pass estimate. The initial window is sized for detection
        # (1.5*fwhm, wide enough to catch the whole PSF) but that width also
        # lets in more neighbouring background/faint-source asymmetry than a
        # centroid computation wants -- a second, smaller pass measurably
        # tightened this same gap for the wavelet detector.
        rr = max(2, int(round(0.7 * fwhm)))
        ry0, ry1 = max(int(round(cy)) - rr, 0), min(int(round(cy)) + rr + 1, H)
        rx0, rx1 = max(int(round(cx)) - rr, 0), min(int(round(cx)) + rr + 1, W)
        rcut = lum64[ry0:ry1, rx0:rx1]
        rw = np.clip(rcut - local_bg, 0, None)
        if rw.sum() > 0:
            ryy, rxx = np.mgrid[ry0:ry1, rx0:rx1]
            cy = float((rw * ryy).sum() / rw.sum())
            cx = float((rw * rxx).sum() / rw.sum())

        dy, dx = yy - cy, xx - cx
        wsum = w.sum()
        ixx = float((w * dy * dy).sum() / wsum)
        iyy = float((w * dx * dx).sum() / wsum)
        ixy = float((w * dy * dx).sum() / wsum)
        evals = np.clip(np.linalg.eigvalsh(np.array([[ixx, ixy], [ixy, iyy]])), 1e-6, None)
        a, b = float(np.sqrt(evals[1])), float(np.sqrt(evals[0]))
        roundness = 1.0 - min(a, b) / max(a, b, 1e-6)
        if roundness >= roundness_max:
            continue
        if w.astype(bool).sum() < min_pixels:
            continue
        flux = float(w.sum())
        peak = float(cut.max())
        rows.append((cx, cy, flux, peak, roundness, roundness,
                    float(np.clip(snr_map[py, px] / 20.0, 0, 1)), a, b, 0.0))

    if not rows:
        return np.zeros(0, dtype=_SOURCES_DTYPE)
    out = np.zeros(len(rows), dtype=_SOURCES_DTYPE)
    for i, row in enumerate(rows):
        out[i] = row
    return out


def _interscale_chain_mask(details: list, scales_chain: tuple, cell: int,
                           k_sigma: float, dilate_iters: int = 1) -> tuple:
    """Multiscale-Vision-Model-style interscale connectivity (Starck &
    Murtagh): label connected significant structures independently at each
    scale in the chain, then keep only finest-scale structures whose
    footprint overlaps a significant structure at *every* coarser scale in
    the chain. This is real component-overlap connectivity (a structure is
    "the same object" across scales if their footprints intersect), not a
    blunt pixel-AND of exactly two dilated masks -- longer chains give more
    confirmation stages almost for free, and it's what makes coarse-scale
    blends that split into multiple finer-scale structures fall out as
    separate objects (deblending) rather than needing a dedicated pass.

    Overlap testing is vectorized via `ndimage.sum` batched over all
    candidate ids at once (sums how many of a coarser scale's significant
    pixels fall inside each finest-scale label) instead of a Python loop
    over structures.

    Returns (mask, labels, valid_ids) at the finest (first) scale in the chain.
    """
    j0 = scales_chain[0]
    sig0 = _local_sigma_map(details[j0], cell=cell)
    mask0 = details[j0] > k_sigma * sig0
    labels0, n0 = ndimage.label(mask0, structure=np.ones((3, 3)))
    if n0 == 0:
        return mask0, labels0, np.zeros(0, dtype=int)

    valid_ids = np.arange(1, n0 + 1)
    for j in scales_chain[1:]:
        sig_j = _local_sigma_map(details[j], cell=cell)
        mask_j = details[j] > k_sigma * sig_j
        if dilate_iters > 0:
            # Coarser scales are smoother/broader by construction (bigger
            # a-trous kernel support) -- their significant region for "the
            # same object" doesn't line up pixel-for-pixel with the finer
            # scale's, it's centered nearby. Some dilation slack here turned
            # out to matter more than chain length itself (measured).
            mask_j = ndimage.binary_dilation(mask_j, iterations=dilate_iters)
        overlap = ndimage.sum(mask_j.astype(np.float64), labels0, index=valid_ids)
        valid_ids = valid_ids[overlap > 0]
        if len(valid_ids) == 0:
            break
    chain_mask = np.isin(labels0, valid_ids)
    return chain_mask, labels0, valid_ids


def wavelet_detect_stars(lum: np.ndarray, scales_chain=(2, 3), k_sigma: float = 4.5,
                         min_pixels: int = 2, roundness_max: float = 0.5,
                         cell: int = 64, n_total_scales: int = 5,
                         k_confirm: float = 8.0, dilate_iters: int = 1) -> np.ndarray:
    """Detect stars via interscale-persistent significant structure in the
    starlet decomposition (see `_interscale_chain_mask`), with a second,
    independent peak-SNR confirmation stage measured against the *original
    image's* local noise (not the wavelet coefficients' noise) -- a
    physically-grounded photometric significance test that empirically
    doesn't need per-field retuning the way pure wavelet-domain thresholds
    did (dense cluster core vs. sparse field wanted very different k_sigma
    values; peak SNR against the real per-pixel noise floor doesn't care how
    many other stars are nearby). Returns a _SOURCES_DTYPE structured array.

    scales_chain defaults to just 2 scales: measured (see conversation/sweep
    results) that longer MVM-style chains (3-4 scales) consistently do
    *worse*, not better, for point sources specifically. Deep interscale
    persistence trees are built for objects with genuine extended
    multiscale structure (small galaxies, nebula clumps); a star's wavelet
    signature concentrates in the 1-2 scales matching its actual FWHM, so
    demanding persistence at a much coarser scale just bleeds recall
    testing for structure a point source never had in the first place.
    """
    lum64 = lum.astype(np.float64)
    H, W = lum64.shape
    details, _coarse = _starlet_transform(lum64, n_total_scales)
    image_sigma_map = _local_sigma_map(lum64, cell=cell)

    mask, _labels0, _valid_ids0 = _interscale_chain_mask(details, scales_chain, cell, k_sigma,
                                                         dilate_iters=dilate_iters)

    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(0, dtype=_SOURCES_DTYPE)

    ids = np.arange(1, n + 1)
    sizes = ndimage.sum(mask, labels, index=ids)
    keep = sizes >= min_pixels
    ids = ids[keep]
    if len(ids) == 0:
        return np.zeros(0, dtype=_SOURCES_DTYPE)

    # Local background per blob: median in a slightly-dilated annulus.
    # Only this needs a small per-blob loop (cheap slices); everything else
    # below runs as batched ndimage ops over all blobs at once.
    objs = ndimage.find_objects(labels, max_label=int(ids.max()))

    local_bg = np.zeros(len(ids))
    for j, lid in enumerate(ids):
        sl = objs[lid - 1]
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, H)
        x0, x1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, W)
        blob_local = labels[y0:y1, x0:x1] == lid
        ann = ndimage.binary_dilation(blob_local, iterations=2) & ~blob_local
        cut = lum64[y0:y1, x0:x1]
        local_bg[j] = float(np.median(cut[ann])) if ann.any() else float(np.median(cut))

    # Background-subtract each blob's own footprint by its own local_bg,
    # via a label->background lookup table (single vectorized gather).
    lut = np.zeros(int(ids.max()) + 1)
    lut[ids] = local_bg
    bg_field = lut[np.clip(labels, 0, len(lut) - 1)]
    weights = np.clip(lum64 - bg_field, 0, None)
    weights[~np.isin(labels, ids)] = 0.0

    flux = ndimage.sum(weights, labels, index=ids)
    # A blob whose every pixel sat at/below its own local background (flat
    # noise fluctuation, not a real bump) has zero weight everywhere ->
    # center_of_mass would divide by zero. Drop those before centroiding.
    nonzero = flux > 0
    ids = ids[nonzero]
    local_bg = local_bg[nonzero]
    flux = flux[nonzero]
    if len(ids) == 0:
        return np.zeros(0, dtype=_SOURCES_DTYPE)

    coms = ndimage.center_of_mass(weights, labels, index=ids)
    coms = np.array(coms)  # (N, 2) -> (row, col)
    peak = ndimage.maximum(lum64, labels, index=ids)

    # Stage-2 confirmation: peak significance against the *original image's*
    # local noise at each blob's centroid (bilinear-sampled from the mesh
    # sigma map), independent of whatever noise the wavelet coefficients
    # themselves have.
    cy_i = np.clip(coms[:, 0].round().astype(int), 0, H - 1)
    cx_i = np.clip(coms[:, 1].round().astype(int), 0, W - 1)
    local_noise = image_sigma_map[cy_i, cx_i]
    peak_snr = (peak - local_bg) / np.maximum(local_noise, 1e-9)
    confirmed = peak_snr > k_confirm

    # second moments per blob (small per-blob loop -- shape only, cheap slices)
    a_arr = np.ones(len(ids))
    b_arr = np.ones(len(ids))
    for j, lid in enumerate(ids):
        sl = objs[lid - 1]
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, H)
        x0, x1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, W)
        w = weights[y0:y1, x0:x1]
        wsum = w.sum()
        if wsum <= 0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cy_j, cx_j = coms[j]
        dy, dx = yy - cy_j, xx - cx_j
        ixx = float((w * dy * dy).sum() / wsum)
        iyy = float((w * dx * dx).sum() / wsum)
        ixy = float((w * dy * dx).sum() / wsum)
        evals = np.linalg.eigvalsh(np.array([[ixx, ixy], [ixy, iyy]]))
        evals = np.clip(evals, 1e-6, None)
        a_arr[j] = float(np.sqrt(evals[1]))
        b_arr[j] = float(np.sqrt(evals[0]))

    denom = np.maximum(a_arr, b_arr)
    denom[denom <= 0] = 1e-6
    roundness = 1.0 - np.minimum(a_arr, b_arr) / denom
    shape_ok = (roundness < roundness_max) & confirmed

    out = np.zeros(int(shape_ok.sum()), dtype=_SOURCES_DTYPE)
    sel = np.where(shape_ok)[0]
    out['xcentroid'] = coms[sel, 1]
    out['ycentroid'] = coms[sel, 0]
    out['flux'] = flux[sel]
    out['peak'] = peak[sel]
    out['roundness1'] = roundness[sel]
    out['roundness2'] = roundness[sel]
    out['sharpness'] = np.clip(peak_snr[sel] / 20.0, 0.0, 1.0)
    out['a'] = a_arr[sel]
    out['b'] = b_arr[sel]
    return out


def match_sources(a: np.ndarray, b: np.ndarray, tol_px: float = 3.0):
    if len(a) == 0 or len(b) == 0:
        return 0, float('nan')
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([b['xcentroid'], b['ycentroid']]))
    pts = np.column_stack([a['xcentroid'], a['ycentroid']])
    dist, idx = tree.query(pts, k=1)
    matched = dist < tol_px
    n = int(matched.sum())
    rms = float(np.sqrt(np.mean(dist[matched] ** 2))) if n else float('nan')
    return n, rms


def load_calibrated_lum(path: str) -> np.ndarray:
    with fits.open(path) as h:
        data = np.asarray(h[0].data, dtype=np.float64)
    if data.ndim == 3:
        if data.shape[0] == 3:
            data = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
        else:
            data = 0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]
    return data


def run_comparison(label: str, lum: np.ndarray, noise: float, **kw):
    print(f"\n=== {label} ({lum.shape[1]}x{lum.shape[0]}) ===")

    t0 = time.time()
    sep_sources = _sep_detect_stars(lum.astype(np.float32), noise)
    t_sep = time.time() - t0
    n_sep = 0 if sep_sources is None else len(sep_sources)
    print(f"SEP:      {n_sep:5d} sources  ({t_sep*1000:.0f} ms)")

    t0 = time.time()
    _ensure_photutils()
    from src.quality import DAOStarFinder
    dao_sources = None
    if DAOStarFinder is not None:
        bg = float(np.median(lum))
        std = max(noise, 1e-6)
        dao_sources = _detect_stars_multi_fwhm(lum - bg, threshold=5.0 * std)
    t_dao = time.time() - t0
    n_dao = 0 if dao_sources is None else len(dao_sources)
    print(f"DAO:      {n_dao:5d} sources  ({t_dao*1000:.0f} ms)")

    t0 = time.time()
    wav_sources = wavelet_detect_stars(lum, **kw)
    t_wav = time.time() - t0
    print(f"Wavelet:  {len(wav_sources):5d} sources  ({t_wav*1000:.0f} ms)")

    if sep_sources is not None and len(wav_sources):
        n_match, rms = match_sources(wav_sources, sep_sources, tol_px=3.0)
        recall = n_match / max(n_sep, 1)
        precision = n_match / max(len(wav_sources), 1)
        print(f"  vs SEP: matched={n_match}  recall={recall:.2f}  precision={precision:.2f}  "
             f"centroid_rms={rms:.3f}px")
    if dao_sources is not None and len(wav_sources):
        n_match, rms = match_sources(wav_sources, dao_sources, tol_px=3.0)
        recall = n_match / max(n_dao, 1)
        precision = n_match / max(len(wav_sources), 1)
        print(f"  vs DAO: matched={n_match}  recall={recall:.2f}  precision={precision:.2f}  "
             f"centroid_rms={rms:.3f}px")

    return sep_sources, dao_sources, wav_sources


def main():
    targets = [
        ("M3 single sub (dense globular core)", r"C:\source\originstack\tools\_test_m3_sub.fits"),
        ("Arcturus single sub (sparse field)", r"C:\source\originstack\tools\_test_sparse_sub.fits"),
        ("Virgo A single sub (held-out, galaxy+field stars)", r"C:\source\originstack\tools\_test_virgo_sub.fits"),
    ]
    for label, path in targets:
        if not os.path.isfile(path):
            print(f"missing: {path}")
            continue
        lum = load_calibrated_lum(path)
        med = float(np.median(lum))
        mad = float(np.median(np.abs(lum - med)))
        noise = 1.4826 * mad
        run_comparison(label, lum, noise, scales_chain=(2, 3), k_sigma=4.5,
                       roundness_max=0.5, k_confirm=8.0)


if __name__ == '__main__':
    main()
