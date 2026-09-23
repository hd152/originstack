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
from scipy import fft as sp_fft

from src.utils import safe_print

_log = logging.getLogger("originstack")


class ZogyResult(NamedTuple):
    """Outputs of a proper subtraction."""
    difference: np.ndarray       # D -- the optimal difference image
    score: np.ndarray            # S -- match-filtered difference
    score_corr: np.ndarray       # S_corr -- S in units of its own sigma
    flux_difference: float       # F_D -- the flux zero point of D


class Transient(NamedTuple):
    """One detected change between two epochs."""
    y: float
    x: float
    significance: float          # S_corr value, in sigma
    kind: str                    # 'brightening' | 'fading'
    real_probability: Optional[float] = None  # --transient-triage; None if not scored


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

    h, w = n_img.shape
    # Noise is measured on the real pixels, before any padding is added.
    sn = float(sigma_new) if sigma_new is not None else estimate_background_sigma(n_img)
    sr = float(sigma_ref) if sigma_ref is not None else estimate_background_sigma(r_img)
    fn, fr = float(flux_new), float(flux_ref)

    # Pad to an FFT-friendly size. The Celestron Origin's own 1096-px axis is
    # 8 x 137, and a large prime factor drops pocketfft onto its slow Bluestein
    # path: measured 2.1x faster padded to 1100, and this function runs ~17
    # full-frame transforms. It also moves the circular wraparound every FFT
    # convolution carries out into padding that is cropped away. Zeros are
    # the right fill because both epochs arrive background-subtracted, so zero
    # *is* sky.
    fh, fw = sp_fft.next_fast_len(h, real=True), sp_fft.next_fast_len(w, real=True)
    shape = (fh, fw)
    n_pad = np.zeros(shape, dtype=np.float64)
    r_pad = np.zeros(shape, dtype=np.float64)
    n_pad[:h, :w] = n_img
    r_pad[:h, :w] = r_img

    pn_hat = sp_fft.fft2(_prepare_psf(psf_new, shape))
    pr_hat = sp_fft.fft2(_prepare_psf(psf_ref, shape))
    n_hat = sp_fft.fft2(n_pad)
    r_hat = sp_fft.fft2(r_pad)
    del n_pad, r_pad

    # Zackay+ eq. 13 denominator. The floor keeps the division finite where
    # both PSFs have a zero (they are band-limited, so this does happen).
    abs_pn2 = np.abs(pn_hat) ** 2
    abs_pr2 = np.abs(pr_hat) ** 2
    denom = np.maximum(sn ** 2 * fr ** 2 * abs_pr2 + sr ** 2 * fn ** 2 * abs_pn2, 1e-30)
    sqrt_denom = np.sqrt(denom)

    d_hat = (fr * n_hat * pr_hat - fn * r_hat * pn_hat) / sqrt_denom
    difference = np.real(sp_fft.ifft2(d_hat))[:h, :w]

    # Flux zero point and PSF of D (eq. 15, 14). Only P_D's transform is
    # needed -- it builds the score below -- so it is never inverted.
    f_d = fn * fr / math.sqrt(sn ** 2 * fr ** 2 + sr ** 2 * fn ** 2)
    pd_hat = fr * fn * pr_hat * pn_hat / (f_d * sqrt_denom)
    del sqrt_denom

    # Score image (eq. 16): D match-filtered with its own PSF.
    score = np.real(sp_fft.ifft2(f_d * d_hat * np.conj(pd_hat)))[:h, :w]
    del d_hat, pd_hat

    # --- Noise terms for S_corr (eq. 26-30) ---
    # Matched-filter kernels for each input image.
    kn_hat = fr * fn ** 2 * np.conj(pn_hat) * abs_pr2 / denom
    kr_hat = fn * fr ** 2 * np.conj(pr_hat) * abs_pn2 / denom
    del abs_pn2, abs_pr2, denom, pn_hat, pr_hat

    # Variance maps live on the padded grid too. The padding is sky, so it
    # carries sky variance -- not zero, which would understate the noise the
    # kernels smear in from the edges.
    def _padded_var(var, img, sky_sigma):
        out = np.full(shape, sky_sigma ** 2, dtype=np.float64)
        out[:h, :w] = (np.maximum(img, 0.0) + sky_sigma ** 2 if var is None
                       else np.asarray(var, dtype=np.float64))
        return out

    # Source noise: each variance map convolved with the squared kernel.
    kn_sq_hat = sp_fft.fft2(np.real(sp_fft.ifft2(kn_hat)) ** 2)
    v_total = np.real(sp_fft.ifft2(sp_fft.fft2(_padded_var(var_new, n_img, sn)) * kn_sq_hat))
    del kn_sq_hat
    kr_sq_hat = sp_fft.fft2(np.real(sp_fft.ifft2(kr_hat)) ** 2)
    v_total += np.real(sp_fft.ifft2(sp_fft.fft2(_padded_var(var_ref, r_img, sr)) * kr_sq_hat))
    del kr_sq_hat

    # Astrometric noise: a registration slip shows up scaled by the local
    # gradient, which is exactly why bright stars dominate the false positives.
    sig_y, sig_x = float(astrometric_sigma[0]), float(astrometric_sigma[1])
    if sig_y > 0 or sig_x > 0:
        for img_hat, k_hat in ((n_hat, kn_hat), (r_hat, kr_hat)):
            gy, gx = np.gradient(np.real(sp_fft.ifft2(img_hat * k_hat)))
            v_total += (sig_y * gy) ** 2 + (sig_x * gx) ** 2
    del n_hat, r_hat, kn_hat, kr_hat

    variance = np.maximum(v_total[:h, :w], 1e-30)
    score_corr = score / np.sqrt(variance)

    return ZogyResult(difference=difference.astype(np.float32),
                      score=score.astype(np.float32),
                      score_corr=score_corr.astype(np.float32),
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
    if not threshold > 0:
        # A threshold at or below zero admits every pixel, and the result is
        # max_candidates of pure noise presented as detections.
        raise ValueError(f"threshold must be positive, got {threshold}")

    arr = np.asarray(score_corr, dtype=np.float64)
    work = np.where(np.isfinite(arr), np.abs(arr), 0.0)
    height, width = work.shape

    # Collect every pixel over threshold once, brightest first, then walk that
    # short list. The previous form re-ran argmax over the *whole* frame per
    # candidate -- up to max_candidates full-frame scans, costliest precisely
    # on a failed subtraction, which is when it hits the cap. A stable sort
    # breaks ties toward the lowest flat index, as argmax did, so the
    # detections and their order are unchanged.
    cand = np.flatnonzero(work >= threshold)
    order = cand[np.argsort(-work.ravel()[cand], kind='stable')]

    suppressed = np.zeros(work.shape, dtype=bool)
    found: List[Transient] = []
    for idx in order:
        y, x = divmod(int(idx), width)
        if suppressed[y, x]:
            continue
        signed = float(arr[y, x])
        found.append(Transient(y=float(y), x=float(x),
                               significance=abs(signed),
                               kind='brightening' if signed > 0 else 'fading'))
        if len(found) >= max_candidates:
            break
        y0, y1 = max(0, y - min_separation), min(height, y + min_separation + 1)
        x0, x1 = max(0, x - min_separation), min(width, x + min_separation + 1)
        suppressed[y0:y1, x0:x1] = True

    return found


def write_transient_catalog(path: str, transients: Sequence[Transient],
                            wcs=None) -> None:
    """Write detections to CSV, with sky coordinates when a WCS is available."""
    import csv

    has_wcs = wcs is not None
    n_failed = 0
    first_error = None
    has_triage = any(t.real_probability is not None for t in transients)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        header = ['x', 'y', 'significance_sigma', 'kind']
        if has_triage:
            header += ['real_probability']
        if has_wcs:
            header += ['ra_deg', 'dec_deg']
        writer.writerow(header)

        for t in transients:
            row = [f"{t.x:.2f}", f"{t.y:.2f}", f"{t.significance:.2f}", t.kind]
            if has_triage:
                row += [f"{t.real_probability:.3f}" if t.real_probability is not None else '']
            if has_wcs:
                try:
                    ra, dec = wcs.all_pix2world(t.x, t.y, 0)
                    row += [f"{float(ra):.6f}", f"{float(dec):.6f}"]
                except Exception as exc:
                    row += ['', '']
                    n_failed += 1
                    first_error = first_error or exc
            writer.writerow(row)

    # A blank RA/Dec column is easy to miss and makes every candidate
    # uncheckable, so say so -- once, not once per row. This is how a 3-axis
    # WCS built from the (C, H, W) cube blanked the whole catalogue silently.
    if n_failed:
        safe_print(f"    WARNING: no sky coordinates for {n_failed} of "
                   f"{len(transients)} candidate(s): {first_error}")


# Registration is rarely better than this across epochs. The measured
# residual is used when it is larger; this is the floor, never the value.
# Understating it re-creates the bright-star false positives ZOGY's
# astrometric noise term exists to suppress.
_ASTROMETRIC_SIGMA_FLOOR_PX = 0.3

# Stars the residual measurement pairs up must lie this close after the
# transform -- the blind matcher's own inlier tolerance.
_RESIDUAL_MATCH_TOL_PX = 3.0


class EpochComparison(NamedTuple):
    """The core two-epoch ZOGY comparison result -- everything
    ``run_transient_detection`` needs to write its outputs, and everything
    ``tools/gen_transient_triage_data.py``'s real-data mining mode needs to
    build training stamps, without going through that function's file I/O."""
    transients: List[Transient]
    difference: np.ndarray       # D, NaN outside the reference footprint
    score_corr: np.ndarray       # S_corr, NaN outside the reference footprint
    new_lum: np.ndarray          # pedestal-subtracted, not yet warped (it's the reference frame)
    ref_lum: np.ndarray          # pedestal-subtracted AND warped onto new_lum's grid
    covered: float
    measured_axis: Optional[float]
    astro_sigma_px: float
    flux_ratio: float


def _compare_epochs(stacked: np.ndarray, ref: np.ndarray,
                    threshold: float = 5.0) -> Optional[EpochComparison]:
    """Align two RGB (or 2D) epochs, run ZOGY, and detect candidates.

    ``stacked``/``ref`` are raw pixel arrays (RGB or luminance), not yet
    reduced to luminance or background-subtracted -- this does both, then
    registration, PSF estimation, ``zogy()`` and ``detect_transients``.
    Returns ``None`` on any setup failure (logged here), same conditions
    ``run_transient_detection`` always reported at this call site.
    """
    new_lum = _to_luminance(stacked)
    ref_lum = _to_luminance(ref)
    if new_lum.shape != ref_lum.shape:
        # Two independently-stacked sessions of the same target routinely
        # differ in pixel dimensions -- different dither pattern, different
        # Phase 3 common-crop -- even though they're the same field. This
        # used to be a hard failure; `_align_reference` now embeds the
        # reference onto this stack's own grid (same trick `merge.py` uses
        # for its differently-shaped previous stacks) before the blind
        # star-pattern match, which doesn't need or assume equal shapes or
        # any positional correspondence between the two canvases anyway.
        safe_print(f"  reference epoch is {ref_lum.shape}, this stack is "
                   f"{new_lum.shape} -- reconciling onto a common grid")

    # ZOGY assumes background-subtracted inputs, so remove each epoch's own
    # sky level rather than trusting them to share one. This is not a
    # formality: two nights genuinely differ in sky brightness, and the
    # in-memory stack here has already had its pedestal removed while a
    # reference loaded from disk still carries one (observed: medians of 0.7
    # and 952 for the *same field*). Comparing those directly makes every
    # pixel read as "fading" and reports hundreds of spurious detections.
    new_lum = new_lum - float(np.median(new_lum))
    ref_lum = ref_lum - float(np.median(ref_lum))

    alignment = _align_reference(new_lum, ref_lum)
    if alignment is None:
        safe_print("  WARNING: could not register the reference epoch onto this stack")
        return None
    ref_lum, footprint, residual_px, new_stars = alignment

    psf_new, psf_ref = _estimate_epoch_psfs(new_lum, ref_lum, new_stars=new_stars)
    if psf_new is None or psf_ref is None:
        safe_print("  WARNING: too few stars to estimate a PSF for one of the epochs")
        return None

    # The measured residual is 2D; ZOGY wants it per axis. The floor applies
    # when the measurement comes back smaller -- or could not be made.
    measured_axis = residual_px / math.sqrt(2.0) if residual_px is not None else None
    astro_sigma_px = max(measured_axis or 0.0, _ASTROMETRIC_SIGMA_FLOOR_PX)
    flux_ratio = estimate_flux_ratio(new_lum, ref_lum)

    result = zogy(new_lum, ref_lum, psf_new, psf_ref,
                  flux_new=flux_ratio, flux_ref=1.0,
                  astrometric_sigma=(astro_sigma_px, astro_sigma_px))

    # Outside the warped reference's footprint there IS no reference: apply
    # the transform to a cross-night pair with any field rotation and the
    # corners fill with zeros. Every star in `new` there has nothing to
    # subtract against, so each one came out as a high-significance
    # 'brightening' -- and the astrometric term cannot help, since it models a
    # small slip, not a missing image. Phase 3 crops to the common region for
    # this reason; here the uncovered area, eroded by the PSF's reach (a star
    # just inside the edge is still half-cut), is masked out of the results.
    margin = max(max(np.shape(psf_new)), max(np.shape(psf_ref))) // 2 + 2
    valid = _erode(footprint > 0.99, margin)
    score_corr = np.where(valid, result.score_corr, np.nan).astype(np.float32)
    difference = np.where(valid, result.difference, np.nan).astype(np.float32)
    covered = float(valid.mean())

    transients = detect_transients(score_corr, threshold=threshold)
    return EpochComparison(transients=transients, difference=difference,
                           score_corr=score_corr, new_lum=new_lum, ref_lum=ref_lum,
                           covered=covered, measured_axis=measured_axis,
                           astro_sigma_px=astro_sigma_px, flux_ratio=flux_ratio)


def run_transient_detection(stacked: np.ndarray, reference_path: str,
                            output_path: str, threshold: float = 5.0,
                            wcs=None, triage: bool = False,
                            triage_model_path: Optional[str] = None) -> Optional[dict]:
    """Orchestrate a two-epoch comparison: align, subtract, detect, report.

    ``triage`` (``--transient-triage``) additionally scores each candidate
    with a small CNN (``src/transient_triage.py``) for a ``real_probability``
    -- advisory only, never drops a candidate.

    Returns a summary dict, or None when the comparison could not be set up
    (missing or non-linear reference, too few stars to estimate a PSF, no
    overlap). Setup failures are reported and return None; an I/O failure
    writing the outputs (disk full, read-only directory) *does* raise, so the
    caller must still guard this -- ``pipeline.py`` does.
    """
    from src.io_fits import load_fits

    if not os.path.exists(reference_path):
        safe_print(f"  WARNING: transient reference not found: {reference_path}")
        return None

    try:
        ref, ref_header = load_fits(reference_path)
    except Exception as exc:
        safe_print(f"  WARNING: could not read transient reference: {exc}")
        return None

    # The comparison is only meaningful between two LINEAR stacks. Phase 4's
    # stretches, denoisers and local contrast break photometric linearity, so
    # a post-processed reference mismatches the flux scale by a fraction of a
    # percent -- several sigma on a bright star -- and every star in the field
    # reports as a confident transient. --merge refuses the same file for the
    # same reason.
    if not bool((ref_header or {}).get('RAWSTACK', False)):
        safe_print(f"  WARNING: {os.path.basename(reference_path)} is not a linear "
                   f"(pre-post-processing) stack: header RAWSTACK is missing or "
                   f"False. Pass the main output FITS of a previous run, not the "
                   f"_processed one -- skipping difference imaging.")
        return None

    # The pipeline writes RGB planes as (C, H, W); load_fits hands them back
    # in that order, while everything here works in (H, W, C).
    ref = np.asarray(ref)
    if ref.ndim == 3 and ref.shape[0] in (3, 4) and ref.shape[0] < ref.shape[-1]:
        ref = np.transpose(ref, (1, 2, 0))

    comparison = _compare_epochs(stacked, ref, threshold=threshold)
    if comparison is None:
        return None
    transients = comparison.transients
    difference = comparison.difference
    score_corr = comparison.score_corr
    new_lum = comparison.new_lum
    ref_lum = comparison.ref_lum
    covered = comparison.covered
    measured_axis = comparison.measured_axis
    astro_sigma_px = comparison.astro_sigma_px
    flux_ratio = comparison.flux_ratio

    n_triaged = 0
    if triage and transients:
        from src.transient_triage import score_candidates
        # Recomputed rather than threaded out of `zogy()`'s ZogyResult --
        # cheap (same robust-sigma estimator, run on arrays already in hand)
        # and avoids widening that return type for an opt-in feature.
        sigma_new = estimate_background_sigma(new_lum)
        sigma_ref = estimate_background_sigma(ref_lum)
        sigma_diff = estimate_background_sigma(difference)
        probs = score_candidates(new_lum, ref_lum, difference, transients,
                                 sigma_new, sigma_ref, sigma_diff,
                                 model_path=triage_model_path)
        transients = [t._replace(real_probability=p) for t, p in zip(transients, probs)]
        n_triaged = sum(1 for p in probs if p is not None)

    stem = os.path.splitext(output_path)[0]
    _write_fits_plane(stem + '_difference.fits', difference,
                      'ZOGY optimal difference image (D); NaN outside the '
                      'reference footprint')
    _write_fits_plane(stem + '_scorr.fits', score_corr,
                      'ZOGY corrected score image (S_corr), units of sigma; '
                      'NaN outside the reference footprint')
    catalog = stem + '_transients.csv'
    write_transient_catalog(catalog, transients, wcs=wcs)

    n_bright = sum(1 for t in transients if t.kind == 'brightening')
    safe_print(f"  Difference imaging: {len(transients)} candidate(s) above "
               f"{threshold:g} sigma ({n_bright} brightening, "
               f"{len(transients) - n_bright} fading)")
    if triage:
        if n_triaged:
            n_likely = sum(1 for t in transients
                          if t.real_probability is not None and t.real_probability > 0.5)
            safe_print(f"    triage: {n_triaged}/{len(transients)} scored, "
                       f"{n_likely} likely real (real_probability > 0.5) -- "
                       f"advisory only, nothing was dropped")
        else:
            safe_print("    triage: requested but unavailable (see warning above) "
                       "-- candidates left unscored")
    if measured_axis is None:
        reg = (f"registration residual not measurable -- assumed "
               f"{astro_sigma_px:.2f} px/axis")
    elif measured_axis < _ASTROMETRIC_SIGMA_FLOOR_PX:
        reg = (f"registration residual {measured_axis:.2f} px/axis measured, "
               f"{astro_sigma_px:.2f} px floor applied")
    else:
        reg = f"registration residual {measured_axis:.2f} px/axis measured"
    safe_print(f"    {reg}, flux ratio {flux_ratio:.3f}, "
               f"{100.0 * covered:.0f}% of frame covered by the reference")
    safe_print(f"    {os.path.basename(catalog)}")

    return {
        'transients': transients,
        'flux_ratio': flux_ratio,
        'registration_residual_px': measured_axis,
        'astrometric_sigma_px': astro_sigma_px,
        'covered_fraction': covered,
        'catalog': catalog,
    }


def _erode(mask: np.ndarray, margin: int) -> np.ndarray:
    """Shrink a boolean mask by ``margin`` pixels away from its uncovered parts.

    ``border_value=1`` is deliberate: the frame's own outer edge is not
    uncovered sky. Treating it as such (the scipy default) erodes a PSF-width
    strip off all four sides -- measured 66% "covered" for a frame that was
    93% covered -- and would silently discard a genuine transient near the
    edge. Only the boundary against the warped reference's empty wedges
    should cost margin.
    """
    if margin <= 0 or mask.all() or not mask.any():
        return mask
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=int(margin), border_value=1)


def _align_reference(new_lum: np.ndarray, ref_lum: np.ndarray):
    """Register the reference epoch onto the new one.

    Cross-night pairs differ by arbitrary field rotation on an alt-az mount,
    so this goes through the same blind star-pattern matcher ``--merge`` uses
    rather than assuming a pure translation. The matcher itself needs no
    positional correspondence between ``new_lum``/``ref_lum`` -- it matches on
    relative star geometry -- so unequal shapes are reconciled first by
    embedding ``ref_lum`` onto ``new_lum``'s grid (top-left, zero-padded;
    ``src.utils.embed_to_shape``, the same trick ``merge.py`` uses for a
    previous stack whose own shape rarely matches the current run's): a
    no-op when the shapes already match.

    Returns ``(warped_ref, footprint, residual_px, new_stars)`` or None:

    - ``footprint`` is the warped reference's coverage in [0, 1] -- the same
      transform applied to a mask of where ``ref_lum`` had real data (ones
      only inside its own original extent, before any embedding). Outside it
      the reference is fill, not data -- and that now covers both the warp's
      own uncovered wedges (field rotation) and any embed-padding border, so
      neither reads as a bogus "transient" the way an all-ones mask would.
    - ``residual_px`` is the RMS 2D distance between matched star pairs after
      the transform: a real measurement of how well the epochs line up, which
      feeds ZOGY's astrometric noise term. None when too few pairs match to
      measure it.
    - ``new_stars`` is returned so the PSF estimate can reuse it rather than
      detect the same stars on the same unchanged array a second time.
    """
    from src.registration import apply_transform
    from src.star_detect import detect_stars_matched_filter
    from src.utils import embed_to_shape

    ref_valid_mask = np.ones_like(ref_lum, dtype=np.float32)
    if ref_lum.shape != new_lum.shape:
        H, W = new_lum.shape
        ref_valid_mask = embed_to_shape(ref_valid_mask, H, W)
        ref_lum = embed_to_shape(ref_lum, H, W)

    try:
        new_stars = detect_stars_matched_filter(new_lum.astype(np.float32))
        ref_stars = detect_stars_matched_filter(ref_lum.astype(np.float32))
    except Exception as exc:
        _log.debug("transient alignment: star detection failed (%s)", exc)
        return None

    if new_stars is None or ref_stars is None or len(new_stars) < 5 or len(ref_stars) < 5:
        return None

    try:
        from src.blind_match import match_rigid_unknown_rotation
        transform = match_rigid_unknown_rotation(ref_stars, new_stars)
    except Exception as exc:
        _log.debug("transient alignment: blind match failed (%s)", exc)
        transform = None

    if transform is None:
        return None

    try:
        warped = apply_transform(ref_lum.astype(np.float32), transform=transform)
        footprint = apply_transform(ref_valid_mask, transform=transform)
    except Exception as exc:
        _log.debug("transient alignment: warp failed (%s)", exc)
        return None

    footprint = _footprint_with_pixel_tolerance(np.asarray(footprint))
    residual = _match_residual_px(ref_stars, new_stars, transform)
    return (np.asarray(warped, dtype=np.float64), footprint,
            residual, new_stars)


def _footprint_with_pixel_tolerance(footprint: np.ndarray) -> np.ndarray:
    """A footprint whose only uncovered pixels are a 1 px rim on the frame's border
    is a fully covered one.

    A pixel is covered when its source position lies within half a pixel of the
    reference's edge, but a spline warp only fills positions strictly inside the
    frame. A sub-pixel shift therefore left the first row/column at zero, two
    *identical* epochs reported a 1 px uncovered rim, and the erosion in ``_erode``
    widened it into a 17% loss of coverage (numpy fallback only: the native Lanczos
    warp of a pure translation is exact). Anything that reaches further in than
    that -- a rotation's wedges -- is left exactly as warped.
    """
    if footprint.ndim != 2 or min(footprint.shape) <= 4:
        return footprint
    if bool((footprint[1:-1, 1:-1] > 0.99).all()):
        return np.ones_like(footprint)
    from src import registration
    if not registration.HAS_NATIVE:
        # The native Lanczos warp of anything but a pure translation loses its 3-tap
        # support within 3 px of the reference's edge, so its footprint has a ~3 px rim
        # there (and `_erode` then masks the PSF reach beyond it); the scipy spline warp is
        # exact right up to the edge. Zero the same rim so both backends drop the same
        # untrustworthy strip -- without it, a star 8 px from the edge of a rotated
        # reference was reported as a transient by the numpy path only.
        rim = footprint.copy()
        rim[:3, :] = 0
        rim[-3:, :] = 0
        rim[:, :3] = 0
        rim[:, -3:] = 0
        return rim
    return footprint


def _match_residual_px(src_stars, dst_stars, transform) -> Optional[float]:
    """RMS distance between matched star pairs after ``transform``, in px.

    Maps each source star through the transform, pairs it with its nearest
    destination star, and keeps pairs within the matcher's own inlier
    tolerance. Uses the matcher's own src->dst convention.
    """
    try:
        from scipy.spatial import cKDTree

        def _xy(stars):
            return np.column_stack([np.asarray(stars['xcentroid'], dtype=np.float64),
                                    np.asarray(stars['ycentroid'], dtype=np.float64)])

        src, dst = _xy(src_stars), _xy(dst_stars)
        R = transform.params[:2, :2]
        t = transform.params[:2, 2]
        dists, _ = cKDTree(dst).query(src @ R.T + t, k=1)
        good = dists[np.isfinite(dists) & (dists < _RESIDUAL_MATCH_TOL_PX)]
        if good.size < 5:
            return None
        return float(np.sqrt(np.mean(good ** 2)))
    except Exception as exc:
        _log.debug("transient alignment: residual measurement failed (%s)", exc)
        return None


def _estimate_epoch_psfs(new_lum: np.ndarray, ref_lum: np.ndarray, new_stars=None):
    """Empirical PSF for each epoch from its own stars.

    ``new_stars`` may be passed in from alignment: ``new_lum`` is unchanged
    in between, so detecting again is pure repetition. The reference has been
    warped since, so its stars are always re-detected.
    """
    from src.psf_deconvolution import estimate_psf
    from src.star_detect import detect_stars_matched_filter

    psfs = []
    for img, stars in ((new_lum, new_stars), (ref_lum, None)):
        try:
            if stars is None:
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
