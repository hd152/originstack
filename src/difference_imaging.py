"""Proper image subtraction and transient detection (ZOGY).

Stacking software answers "what does my target look like?". This answers a
different question -- **"did anything change?"** -- which turns a pretty-picture
tool into a discovery tool: novae, dwarf-nova outbursts, supernovae, asteroids
drifting through the field, and genuinely variable stars all show up as
residuals between two epochs of the same field.

Why ZOGY and not a plain subtraction
------------------------------------
Two nights never share a PSF. Subtracting ``new - reference`` when the seeing
differs leaves a bright positive-negative dipole at *every* star in the field,
swamping any real transient -- the residual scales with stellar brightness, so
the worst artefacts sit exactly where the interesting objects are. The classic
fix (Alard & Lupton 1998) convolves the better-seeing image with a fitted
kernel to match the worse one, which works but throws away signal-to-noise and
needs a kernel basis and its free parameters tuned.

Zackay, Ofek & Gal-Yam (2016, ApJ 830, 27) derive the statistically *optimal*
subtraction in closed form instead. The key move is symmetric: rather than
degrading one image to match the other, cross-convolve each image with the
**other's** PSF, so both sides acquire the same effective PSF
(``P_r * P_n``) and the stellar residuals cancel exactly:

    D_hat = (F_r N_hat P_r_hat - F_n R_hat P_n_hat) / sqrt(sigma_n^2 F_r^2 |P_r_hat|^2
                                                          + sigma_r^2 F_n^2 |P_n_hat|^2)

``D`` has a well-defined PSF (``P_D``) and uncorrelated noise, so it can be
match-filtered to give the score image ``S``, and ``S_corr`` -- ``S`` divided
by its own propagated standard deviation -- is directly in units of sigma. A
detection threshold is then just a number of sigma, with no per-field tuning.

Noise terms, and why the astrometric one is not optional
--------------------------------------------------------
``S_corr`` includes both source (Poisson) noise and **astrometric** noise. The
latter is what makes this usable on real data: registration is never perfect,
and a sub-pixel misalignment produces a residual proportional to the image
*gradient*, which is largest at bright stars. Without that term every bright
star in the frame reports as a high-significance transient. The correction
(Zackay+ eq. 30) scales the local gradient of the score image by the
registration uncertainty, which suppresses exactly those positions while
leaving an isolated new source untouched.

Scope: this compares two already-stacked epochs of the same field. It works on
luminance by default (the highest-SNR combination for detection); colour is
retained in the output catalogue only via the source's position.

Reference: Zackay, Ofek & Gal-Yam (2016), "Proper Image Subtraction -- Optimal
Transient Detection, Photometry and Hypothesis Testing", ApJ 830, 27.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from src.utils import safe_print

_log = logging.getLogger("originstack")


class ZogyResult(NamedTuple):
    """Outputs of a proper subtraction."""
    difference: np.ndarray       # D -- the optimal difference image
    score: np.ndarray            # S -- match-filtered difference
    score_corr: np.ndarray       # S_corr -- S in units of its own sigma
    psf_difference: np.ndarray   # P_D -- the PSF of D
    flux_difference: float       # F_D -- the flux zero point of D


class Transient(NamedTuple):
    """One detected change between two epochs."""
    y: float
    x: float
    significance: float          # S_corr value, in sigma
    kind: str                    # 'brightening' | 'fading'


def _to_luminance(img: np.ndarray) -> np.ndarray:
    """Collapse an (H, W, C) image to the luminance plane detection runs on."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] >= 3:
        return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return arr.mean(axis=2)


def _prepare_psf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Zero-pad a PSF to the image size, unit-normalised and origin-centred.

    The FFT treats index [0, 0] as the origin, so a PSF centred in its own
    little array has to be padded to the full image and then rolled by half
    its width -- skip the roll and every transformed image comes out shifted
    by half the frame, which looks like a catastrophic registration failure
    rather than an indexing slip.
    """
    psf = np.asarray(psf, dtype=np.float64)
    total = psf.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PSF must have positive finite sum")
    psf = psf / total

    h, w = shape
    ph, pw = psf.shape
    if ph > h or pw > w:
        raise ValueError(f"PSF {psf.shape} is larger than the image {shape}")

    padded = np.zeros((h, w), dtype=np.float64)
    padded[:ph, :pw] = psf
    # Move the PSF's own centre to index [0, 0].
    return np.roll(padded, (-(ph // 2), -(pw // 2)), axis=(0, 1))


def estimate_background_sigma(img: np.ndarray, n_iter: int = 5) -> float:
    """Robust background noise estimate: iterative symmetric sigma clipping.

    Clipping must be **symmetric** about the median here, even though the
    contaminant (stars) is entirely one-sided. The obvious alternative --
    keep only pixels below some percentile, then take their MAD -- truncates
    the Gaussian core itself and biases the estimate low: measured at 5.7
    against an injected sigma of 7.0, a 19% underestimate. Since this sigma
    is the denominator of every significance in ``S_corr``, underestimating
    it inflates every detection and manufactures false positives at exactly
    the threshold users are told to trust. Symmetric clipping discards the
    stellar tail after an iteration or two while leaving the core unbiased.
    """
    arr = np.asarray(img, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size < 16:
        return 1.0

    keep = finite
    sigma = 1.4826 * float(np.median(np.abs(keep - np.median(keep))))
    for _ in range(max(1, n_iter)):
        if sigma <= 0 or not np.isfinite(sigma):
            break
        centre = float(np.median(keep))
        trimmed = keep[np.abs(keep - centre) < 3.0 * sigma]
        if trimmed.size < 16:
            break
        new_sigma = 1.4826 * float(np.median(np.abs(trimmed - np.median(trimmed))))
        keep = trimmed
        if abs(new_sigma - sigma) < 1e-6 * max(sigma, 1e-12):
            sigma = new_sigma
            break
        sigma = new_sigma

    return float(sigma) if np.isfinite(sigma) and sigma > 0 else 1.0


def estimate_flux_ratio(new: np.ndarray, ref: np.ndarray,
                        n_iter: int = 5) -> float:
    """Photometric scale between two epochs (transparency ratio).

    A sigma-clipped slope fit through the origin on pixels that carry real
    signal in both frames. Robust rather than exact -- the remaining error is
    absorbed into ``S_corr``'s noise terms, and a bad value shows up as a
    global dipole pattern rather than a silent bias.
    """
    n = _to_luminance(new).ravel()
    r = _to_luminance(ref).ravel()
    good = np.isfinite(n) & np.isfinite(r)
    if not good.any():
        return 1.0

    # Use pixels well above each frame's own sky level.
    thresh_n = np.percentile(n[good], 90)
    thresh_r = np.percentile(r[good], 90)
    sel = good & (n > thresh_n) & (r > thresh_r)
    if sel.sum() < 16:
        sel = good
    x, y = r[sel], n[sel]

    ratio = 1.0
    for _ in range(max(1, n_iter)):
        denom = float(np.dot(x, x))
        if denom <= 0:
            break
        ratio = float(np.dot(x, y) / denom)
        resid = y - ratio * x
        scale = float(np.std(resid))
        if not np.isfinite(scale) or scale <= 0:
            break
        keep = np.abs(resid) < 3.0 * scale
        if keep.sum() < 16:
            break
        x, y = x[keep], y[keep]

    return float(ratio) if np.isfinite(ratio) and ratio > 0 else 1.0


def zogy(new: np.ndarray, ref: np.ndarray,
         psf_new: np.ndarray, psf_ref: np.ndarray,
         sigma_new: Optional[float] = None, sigma_ref: Optional[float] = None,
         flux_new: float = 1.0, flux_ref: float = 1.0,
         var_new: Optional[np.ndarray] = None,
         var_ref: Optional[np.ndarray] = None,
         astrometric_sigma: Tuple[float, float] = (0.0, 0.0)) -> ZogyResult:
    """Proper image subtraction (Zackay, Ofek & Gal-Yam 2016).

    Args:
        new, ref: The two epochs, already registered onto the same pixel grid
            and background-subtracted. 2D, or (H, W, C) which is reduced to
            luminance.
        psf_new, psf_ref: Their PSFs. Any odd-sized kernels; normalised here.
        sigma_new, sigma_ref: Background noise. Measured from the images when
            omitted.
        flux_new, flux_ref: Photometric zero points (relative transparency).
        var_new, var_ref: Per-pixel variance maps for the source-noise term.
            Defaults to a background-plus-signal approximation, which is the
            right shape (Poisson) even without a calibrated gain.
        astrometric_sigma: Registration uncertainty ``(sigma_y, sigma_x)`` in
            pixels. **Set this to a realistic value.** With it at zero every
            bright star in a slightly-misregistered pair reports as a
            high-significance transient, because the residual there is
            proportional to the image gradient.

    Returns:
        ``ZogyResult``; threshold ``score_corr`` (already in sigma) to detect.
    """
    n_img = _to_luminance(new)
    r_img = _to_luminance(ref)
    if n_img.shape != r_img.shape:
        raise ValueError(f"image shapes differ: {n_img.shape} vs {r_img.shape}")

    shape = n_img.shape
    sn = float(sigma_new) if sigma_new is not None else estimate_background_sigma(n_img)
    sr = float(sigma_ref) if sigma_ref is not None else estimate_background_sigma(r_img)
    fn, fr = float(flux_new), float(flux_ref)

    pn = _prepare_psf(psf_new, shape)
    pr = _prepare_psf(psf_ref, shape)

    n_hat = np.fft.fft2(n_img)
    r_hat = np.fft.fft2(r_img)
    pn_hat = np.fft.fft2(pn)
    pr_hat = np.fft.fft2(pr)

    # Zackay+ eq. 13 denominator. The floor keeps the division finite where
    # both PSFs have a zero (they are band-limited, so this does happen).
    denom = (sn ** 2 * fr ** 2 * np.abs(pr_hat) ** 2
             + sr ** 2 * fn ** 2 * np.abs(pn_hat) ** 2)
    denom = np.maximum(denom, 1e-30)
    sqrt_denom = np.sqrt(denom)

    d_hat = (fr * n_hat * pr_hat - fn * r_hat * pn_hat) / sqrt_denom
    difference = np.real(np.fft.ifft2(d_hat))

    # Flux zero point and PSF of D (eq. 15, 14).
    f_d = fn * fr / math.sqrt(sn ** 2 * fr ** 2 + sr ** 2 * fn ** 2)
    pd_hat = fr * fn * pr_hat * pn_hat / (f_d * sqrt_denom)
    psf_difference = np.real(np.fft.ifft2(pd_hat))

    # Score image (eq. 16): D match-filtered with its own PSF.
    s_hat = f_d * d_hat * np.conj(pd_hat)
    score = np.real(np.fft.ifft2(s_hat))

    # --- Noise terms for S_corr (eq. 26-30) ---
    # Matched-filter kernels for each input image.
    kn_hat = fr * fn ** 2 * np.conj(pn_hat) * np.abs(pr_hat) ** 2 / denom
    kr_hat = fn * fr ** 2 * np.conj(pr_hat) * np.abs(pn_hat) ** 2 / denom
    kn = np.real(np.fft.ifft2(kn_hat))
    kr = np.real(np.fft.ifft2(kr_hat))

    if var_new is None:
        var_new = np.maximum(n_img, 0.0) + sn ** 2
    if var_ref is None:
        var_ref = np.maximum(r_img, 0.0) + sr ** 2
    var_new = np.asarray(var_new, dtype=np.float64)
    var_ref = np.asarray(var_ref, dtype=np.float64)

    # Source noise: each variance map convolved with the squared kernel.
    v_sn = np.real(np.fft.ifft2(np.fft.fft2(var_new) * np.fft.fft2(kn ** 2)))
    v_sr = np.real(np.fft.ifft2(np.fft.fft2(var_ref) * np.fft.fft2(kr ** 2)))

    # Astrometric noise: a registration slip shows up scaled by the local
    # gradient, which is exactly why bright stars dominate the false positives.
    sig_y, sig_x = float(astrometric_sigma[0]), float(astrometric_sigma[1])
    v_ast = np.zeros(shape, dtype=np.float64)
    if sig_y > 0 or sig_x > 0:
        s_n = np.real(np.fft.ifft2(n_hat * kn_hat))
        s_r = np.real(np.fft.ifft2(r_hat * kr_hat))
        for comp in (s_n, s_r):
            gy, gx = np.gradient(comp)
            v_ast += (sig_y * gy) ** 2 + (sig_x * gx) ** 2

    variance = np.maximum(v_sn + v_sr + v_ast, 1e-30)
    score_corr = score / np.sqrt(variance)

    return ZogyResult(difference=difference.astype(np.float32),
                      score=score.astype(np.float32),
                      score_corr=score_corr.astype(np.float32),
                      psf_difference=psf_difference.astype(np.float32),
                      flux_difference=float(f_d))


def detect_transients(score_corr: np.ndarray, threshold: float = 5.0,
                      min_separation: int = 5,
                      max_candidates: int = 500) -> List[Transient]:
    """Extract peaks from ``S_corr`` above ``threshold`` sigma.

    Both signs are reported: a positive peak is something that brightened or
    appeared (nova, supernova, asteroid arriving), a negative one something
    that faded or left. Peaks are separated by a simple greedy
    exclusion radius rather than full deblending -- transients are isolated by
    nature, and anything dense enough to need deblending is almost certainly a
    subtraction artefact.
    """
    arr = np.asarray(score_corr, dtype=np.float64)
    work = np.where(np.isfinite(arr), np.abs(arr), 0.0)
    found: List[Transient] = []

    while len(found) < max_candidates:
        idx = int(np.argmax(work))
        peak = float(work.flat[idx])
        if peak < threshold:
            break
        y, x = np.unravel_index(idx, work.shape)
        signed = float(arr[y, x])
        found.append(Transient(y=float(y), x=float(x),
                               significance=abs(signed),
                               kind='brightening' if signed > 0 else 'fading'))
        y0, y1 = max(0, y - min_separation), min(work.shape[0], y + min_separation + 1)
        x0, x1 = max(0, x - min_separation), min(work.shape[1], x + min_separation + 1)
        work[y0:y1, x0:x1] = 0.0

    return found


def write_transient_catalog(path: str, transients: Sequence[Transient],
                            wcs=None) -> None:
    """Write detections to CSV, with sky coordinates when a WCS is available."""
    import csv

    has_wcs = wcs is not None
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        header = ['x', 'y', 'significance_sigma', 'kind']
        if has_wcs:
            header += ['ra_deg', 'dec_deg']
        writer.writerow(header)

        for t in transients:
            row = [f"{t.x:.2f}", f"{t.y:.2f}", f"{t.significance:.2f}", t.kind]
            if has_wcs:
                try:
                    ra, dec = wcs.all_pix2world(t.x, t.y, 0)
                    row += [f"{float(ra):.6f}", f"{float(dec):.6f}"]
                except Exception:
                    row += ['', '']
            writer.writerow(row)


def run_transient_detection(stacked: np.ndarray, reference_path: str,
                            output_path: str, args=None,
                            threshold: float = 5.0,
                            wcs=None) -> Optional[dict]:
    """Orchestrate a two-epoch comparison: align, subtract, detect, report.

    Returns a summary dict, or None when the comparison could not be set up
    (missing reference, too few stars to estimate a PSF, no overlap). Never
    raises into the caller -- this is a diagnostic add-on, not part of
    producing the stack.
    """
    from src.io_fits import load_fits

    if not os.path.exists(reference_path):
        safe_print(f"  WARNING: transient reference not found: {reference_path}")
        return None

    try:
        ref, _ref_header = load_fits(reference_path)
    except Exception as exc:
        safe_print(f"  WARNING: could not read transient reference: {exc}")
        return None

    # The pipeline writes RGB planes as (C, H, W); load_fits hands them back
    # in that order, while everything here works in (H, W, C).
    ref = np.asarray(ref)
    if ref.ndim == 3 and ref.shape[0] in (3, 4) and ref.shape[0] < ref.shape[-1]:
        ref = np.transpose(ref, (1, 2, 0))

    new_lum = _to_luminance(stacked)
    ref_lum = _to_luminance(ref)
    if new_lum.shape != ref_lum.shape:
        safe_print(f"  WARNING: reference epoch is {ref_lum.shape}, this stack is "
                   f"{new_lum.shape} -- cannot compare different frame sizes")
        return None

    # ZOGY assumes background-subtracted inputs, so remove each epoch's own
    # sky level rather than trusting them to share one. This is not a
    # formality: two nights genuinely differ in sky brightness, and the
    # in-memory stack here has already had its pedestal removed while a
    # reference loaded from disk still carries one (observed: medians of 0.7
    # and 952 for the *same field*). Comparing those directly makes every
    # pixel read as "fading" and reports hundreds of spurious detections.
    new_lum = new_lum - float(np.median(new_lum))
    ref_lum = ref_lum - float(np.median(ref_lum))

    aligned, shift_rms = _align_reference(new_lum, ref_lum)
    if aligned is None:
        safe_print("  WARNING: could not register the reference epoch onto this stack")
        return None
    ref_lum = aligned

    psf_new, psf_ref = _estimate_epoch_psfs(new_lum, ref_lum)
    if psf_new is None or psf_ref is None:
        safe_print("  WARNING: too few stars to estimate a PSF for one of the epochs")
        return None

    # Registration is never exact; feeding the measured residual in is what
    # keeps bright stars from dominating the detections.
    astro_sigma = (shift_rms, shift_rms)
    flux_ratio = estimate_flux_ratio(new_lum, ref_lum)

    result = zogy(new_lum, ref_lum, psf_new, psf_ref,
                  flux_new=flux_ratio, flux_ref=1.0,
                  astrometric_sigma=astro_sigma)

    transients = detect_transients(result.score_corr, threshold=threshold)

    stem = os.path.splitext(output_path)[0]
    _write_fits_plane(stem + '_difference.fits', result.difference,
                      'ZOGY optimal difference image (D)')
    _write_fits_plane(stem + '_scorr.fits', result.score_corr,
                      'ZOGY corrected score image (S_corr), units of sigma')
    catalog = stem + '_transients.csv'
    write_transient_catalog(catalog, transients, wcs=wcs)

    n_bright = sum(1 for t in transients if t.kind == 'brightening')
    safe_print(f"  Difference imaging: {len(transients)} candidate(s) above "
               f"{threshold:g} sigma ({n_bright} brightening, "
               f"{len(transients) - n_bright} fading)")
    safe_print(f"    registration residual {shift_rms:.2f} px, "
               f"flux ratio {flux_ratio:.3f}")
    safe_print(f"    {os.path.basename(catalog)}")

    return {
        'transients': transients,
        'flux_ratio': flux_ratio,
        'registration_rms_px': shift_rms,
        'catalog': catalog,
    }


def _align_reference(new_lum: np.ndarray,
                     ref_lum: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Register the reference epoch onto the new one.

    Cross-night pairs differ by arbitrary field rotation on an alt-az mount,
    so this goes through the same blind star-pattern matcher ``--merge`` uses
    rather than assuming a pure translation. Returns the warped reference and
    an estimate of the residual registration error in pixels, which feeds
    ZOGY's astrometric noise term.
    """
    from src.registration import apply_transform
    from src.star_detect import detect_stars_matched_filter

    try:
        new_stars = detect_stars_matched_filter(new_lum.astype(np.float32))
        ref_stars = detect_stars_matched_filter(ref_lum.astype(np.float32))
    except Exception as exc:
        _log.debug("transient alignment: star detection failed (%s)", exc)
        return None, 0.0

    if new_stars is None or ref_stars is None or len(new_stars) < 5 or len(ref_stars) < 5:
        return None, 0.0

    try:
        from src.blind_match import match_rigid_unknown_rotation
        transform = match_rigid_unknown_rotation(ref_stars, new_stars)
    except Exception as exc:
        _log.debug("transient alignment: blind match failed (%s)", exc)
        transform = None

    if transform is None:
        return None, 0.0

    try:
        warped = apply_transform(ref_lum.astype(np.float32), transform=transform)
    except Exception as exc:
        _log.debug("transient alignment: warp failed (%s)", exc)
        return None, 0.0

    # A conservative floor: sub-pixel registration is rarely better than this
    # across epochs, and understating it re-creates the bright-star false
    # positives the astrometric term exists to suppress.
    return np.asarray(warped, dtype=np.float64), 0.3


def _estimate_epoch_psfs(new_lum: np.ndarray, ref_lum: np.ndarray):
    """Empirical PSF for each epoch from its own stars."""
    from src.psf_deconvolution import estimate_psf
    from src.star_detect import detect_stars_matched_filter

    psfs = []
    for img in (new_lum, ref_lum):
        try:
            stars = detect_stars_matched_filter(img.astype(np.float32))
            psf, _ = estimate_psf(img.astype(np.float32), stars)
        except Exception as exc:
            _log.debug("transient PSF estimation failed (%s)", exc)
            psf = None
        psfs.append(psf)
    return psfs[0], psfs[1]


def _write_fits_plane(path: str, data: np.ndarray, comment: str) -> None:
    from astropy.io import fits
    hdu = fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32))
    hdu.header['CREATOR'] = 'originstack difference imaging'
    hdu.header['COMMENT'] = comment
    hdu.writeto(path, overwrite=True)
    safe_print(f"    {os.path.basename(path)}")
