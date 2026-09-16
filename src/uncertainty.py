"""End-to-end uncertainty propagation and confidence mapping.

Phase 3 can already emit a statistically exact per-pixel standard error for
the stacked image: ``ivw_combine(..., return_sigma=True)`` (``--uncertainty-map``)
returns the Gauss-Markov estimator's own ``1/sqrt(sum_i 1/var_i)``. That map
describes the *linear* stack, and Phase 4 then discards it -- every
background extraction, denoise, deconvolution and stretch reshapes the noise
field, so the pre-post-processing sigma no longer describes the image the
user actually looks at.

This module carries the error bars through Phase 4 so the delivered image has
one, and turns them into the thing an observer actually wants to know: **is
that faint wisp real, or is it noise the denoiser smoothed into a shape?**

Why Monte Carlo rather than analytic propagation
------------------------------------------------
``postprocess_stack`` is ~20 heterogeneous steps, of which nine are nonlinear
denoisers and four are deconvolvers. Analytic variance propagation would mean
deriving (and maintaining) a Jacobian approximation for every one of them, and
for the iterative ones -- Richardson-Lucy, FISTA, anisotropic diffusion -- the
honest answer is that no closed form exists. Several are also spatially
adaptive (BayesShrink thresholds, the structure-tensor coherence map in
``directional_wavelet_denoise``), so the effective gain at a pixel depends on
its neighbourhood content, not just on the operator.

So this instead treats the whole chain as a black box and measures it: draw
``K`` noise realizations consistent with the Phase 3 sigma map, push each one
through the *unmodified* post-processing chain, and take the per-pixel spread
of the outputs. That is exact (up to Monte Carlo error ~ ``1/sqrt(2K)`` on the
standard deviation) for an arbitrarily nonlinear chain, needs no changes to
any post-processing step, and stays correct automatically when a new denoiser
is added. The cost is ``K`` extra post-processing passes, which is why it is
opt-in (``--uncertainty-propagate``).

The realizations are run with a *quieted* copy of ``args`` (see
``_quiet_args``): file-writing side effects, diagnostics and network-dependent
steps are disabled so K passes don't produce K sets of sidecars or K Gaia
queries. Everything that actually shapes the noise field -- background
extraction, every denoiser, deconvolution, star reduction, local contrast --
is left exactly as the real pass ran it.
"""

from __future__ import annotations

import copy
import io
import logging
import os
from contextlib import redirect_stdout
from typing import Callable, Optional, Tuple

import numpy as np

from src.utils import safe_print

_log = logging.getLogger("originstack")

# Post-processing settings that must not run K extra times: they write files,
# hit the network, or produce sidecar outputs that would be overwritten K
# times with realization noise instead of the real image. None of them
# meaningfully reshapes the noise field, so dropping them does not bias the
# propagated sigma.
_QUIET_OFF = (
    'remove_stars',            # writes <output>_starless.fits
    'nmf_separate',            # writes _star_component/_nebula_component.fits
    'photometric_calibration',  # Gaia/VizieR network queries, per-channel scalar
    'annotate',
    'verbose',
)


def _quiet_args(args):
    """A shallow copy of ``args`` with side-effecting Phase 4 steps disabled.

    Shallow (``copy.copy``) rather than ``deepcopy`` deliberately: an
    argparse Namespace can carry non-copyable entries stashed by earlier
    phases (open handles, loaded masters), and every attribute this function
    changes is a plain bool/str being *rebound* on the copy's own ``__dict__``,
    so the original is untouched either way.
    """
    quiet = copy.copy(args)
    for name in _QUIET_OFF:
        if hasattr(quiet, name):
            setattr(quiet, name, False)
    quiet._diagnostic_dir = None
    quiet._uncertainty_realization = True
    return quiet


def flat_sigma_from_image(image: np.ndarray) -> float:
    """Fallback scalar sigma when Phase 3 produced no uncertainty map.

    ``--uncertainty-map`` only yields a real per-pixel standard error for
    ``--stack-method ivw`` (it is that estimator's own variance). For every
    other combine this measures the residual sky noise of the stacked image
    itself and uses it as a spatially flat sigma. That is an approximation --
    it has no per-pixel shot-noise structure, so it understates the error on
    bright pixels and the propagated map inherits that -- and callers say so
    in the log rather than presenting it as equivalent.
    """
    from src.background import _estimate_sky_sigma
    return float(_estimate_sky_sigma(np.asarray(image, dtype=np.float32)))


def propagate_uncertainty(
    stacked: np.ndarray,
    sigma: np.ndarray,
    args,
    final,
    stats,
    postprocess_fn: Callable,
    n_realizations: int = 8,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Push ``K`` noise realizations through Phase 4 and measure the spread.

    Args:
        stacked: The linear pre-post-processing stack, ``(H, W, C)``.
        sigma: Per-pixel standard error of ``stacked``, ``(H, W)`` (broadcast
            across channels) or ``(H, W, C)``.
        args, final, stats: Passed straight through to ``postprocess_fn``.
        postprocess_fn: ``postprocess_stack``, injected rather than imported
            so tests can substitute a chain with known analytic behaviour.
        n_realizations: ``K``. The relative error on the returned standard
            deviation is roughly ``1/sqrt(2K)`` -- ~25% at K=8, ~18% at K=16.
            Low K is still useful here because the *interpretation* is
            bucketed (below 3 sigma / marginal / solid), not read to 3 digits.
        seed: RNG seed, so a run is reproducible.

    Returns:
        ``(sigma_post, mean_post)`` -- the per-pixel standard deviation across
        realizations ``(H, W)``, collapsed across channels by averaging the
        per-channel variances, and the per-pixel mean of the realizations
        ``(H, W, C)``. The mean is returned because for a nonlinear chain it
        is *not* equal to the chain applied to the unperturbed input, and the
        difference is itself diagnostic (a large gap means the chain is
        strongly nonlinear at that pixel, i.e. the error bar is skewed).
    """
    stacked = np.asarray(stacked, dtype=np.float32)
    H, W = stacked.shape[:2]
    C = stacked.shape[2] if stacked.ndim == 3 else 1

    sigma = np.asarray(sigma, dtype=np.float32)
    if sigma.ndim == 2:
        sigma_b = sigma[:, :, np.newaxis]
    else:
        sigma_b = sigma
    sigma_b = np.broadcast_to(sigma_b, (H, W, C))

    K = max(2, int(n_realizations))
    rng = np.random.default_rng(seed)
    quiet = _quiet_args(args)

    # Welford accumulation over realizations: K full (H,W,C) outputs would be
    # K x the stack's own footprint, and this pipeline's whole memory model is
    # built on not materialising K copies of anything.
    mean = np.zeros((H, W, C), dtype=np.float64)
    m2 = np.zeros((H, W, C), dtype=np.float64)
    n_ok = 0

    for k in range(K):
        noisy = stacked + rng.normal(0.0, 1.0, size=(H, W, C)).astype(np.float32) * sigma_b
        try:
            # Phase 4 mutates its input in place in several steps, so each
            # realization gets its own copy -- and stdout is swallowed so K
            # passes don't produce K copies of the whole Phase 4 log.
            buf = io.StringIO()
            with redirect_stdout(buf):
                out = postprocess_fn(noisy.copy(), quiet, final, stats)
        except Exception as exc:
            _log.debug("uncertainty realization %d failed (%s); skipping", k, exc)
            continue

        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 2:
            out = out[:, :, np.newaxis]
        if out.shape[:2] != (H, W):
            # A Phase 4 step that changes geometry (e.g. a crop) makes the
            # realizations non-comparable; bail rather than silently compare
            # misaligned pixels.
            raise ValueError(
                f"post-processing changed image shape {stacked.shape} -> {out.shape}; "
                "uncertainty propagation needs a geometry-preserving chain")

        n_ok += 1
        delta = out - mean
        mean += delta / n_ok
        m2 += delta * (out - mean)

        if verbose:
            safe_print(f"    realization {k + 1}/{K}")

    if n_ok < 2:
        raise RuntimeError(
            f"only {n_ok} of {K} uncertainty realizations completed; "
            "cannot estimate a spread")
    if n_ok < K:
        safe_print(f"  WARNING: {K - n_ok} of {K} uncertainty realizations failed "
                   f"(propagated sigma uses the {n_ok} that completed)")

    var = m2 / (n_ok - 1)
    # One confidence map, not three: average the per-channel variances,
    # matching how ivw_combine collapses its own per-channel weight sums.
    sigma_post = np.sqrt(np.maximum(var.mean(axis=-1), 0.0)).astype(np.float32)
    return sigma_post, mean.astype(np.float32)


def confidence_map(image: np.ndarray, sigma_post: np.ndarray,
                   background: Optional[float] = None) -> np.ndarray:
    """Per-pixel signal-to-noise of the delivered image.

    ``(pixel - background) / sigma_post`` -- how many propagated standard
    errors a pixel sits above the sky level. This is the number that answers
    "is that faint arc real?", and it is only meaningful *after*
    ``propagate_uncertainty`` because the denoisers have by then changed both
    the numerator (smoothing) and the denominator (noise suppression), often
    by very different factors.

    ``background`` defaults to the image's own robust sky level (median of
    the luminance), which is the right reference for "above the sky" on a
    background-extracted image.

    Pixels with **zero** propagated sigma come back as ``NaN``, not as a huge
    SNR. A zero there is real -- every realization produced a bit-identical
    value -- but it means the chain *clamped* that pixel to a constant (the
    sky pedestal lift and the non-negativity clips both do this), so its
    value is a post-processing artifact rather than a measurement, and
    dividing by it would report near-infinite confidence for exactly the
    pixels that carry no information. ``NaN`` says "this method cannot tell
    you", which is the honest answer; both ``summarize_confidence`` and
    ``error_aware_black_point`` already filter on ``np.isfinite``.
    """
    image = np.asarray(image, dtype=np.float32)
    lum = image.mean(axis=-1) if image.ndim == 3 else image
    if background is None:
        background = float(np.median(lum))
    sigma_post = np.asarray(sigma_post, dtype=np.float32)

    snr = np.full(lum.shape, np.nan, dtype=np.float32)
    measured = sigma_post > 0
    np.divide(lum - background, sigma_post, out=snr, where=measured)
    return snr


def summarize_confidence(snr: np.ndarray, thresholds=(3.0, 5.0)) -> str:
    """One-line summary of how much of the frame carries real signal.

    Percentages are of the *measured* pixels. The clamped fraction (NaN --
    post-processing pinned them to a constant, see ``confidence_map``) is
    reported separately rather than being silently folded into the
    denominator, since a large clamped fraction is itself a warning that the
    chain is flattening the image rather than measuring it.
    """
    snr = np.asarray(snr, dtype=np.float32)
    finite_mask = np.isfinite(snr)
    finite = snr[finite_mask]
    if finite.size == 0:
        return "confidence map: no measurable pixels (post-processing clamped every pixel)"
    total = float(finite.size)
    parts = [f">{t:g}sigma {100.0 * float((finite > t).sum()) / total:.1f}%"
             for t in thresholds]
    clamped = 100.0 * float((~finite_mask).sum()) / float(snr.size)
    if clamped > 0.05:
        parts.append(f"{clamped:.1f}% clamped by post-processing")
    return "confidence map: " + ", ".join(parts)


def save_uncertainty_outputs(output_path: str, sigma_post: np.ndarray,
                             snr: Optional[np.ndarray] = None) -> None:
    """Write the propagated sigma (and confidence map) as FITS sidecars."""
    from astropy.io import fits

    stem = os.path.splitext(output_path)[0]

    sigma_path = stem + '_sigma_final.fits'
    hdu = fits.PrimaryHDU(data=np.asarray(sigma_post, dtype=np.float32))
    hdu.header['CREATOR'] = 'originstack uncertainty'
    hdu.header['COMMENT'] = ('Per-pixel standard error AFTER post-processing, '
                             'from Monte Carlo propagation of the Phase 3 '
                             'uncertainty map through the Phase 4 chain')
    hdu.writeto(sigma_path, overwrite=True)
    safe_print(f"  Propagated uncertainty: {os.path.basename(sigma_path)}")

    if snr is not None:
        snr_path = stem + '_snr.fits'
        snr_hdu = fits.PrimaryHDU(data=np.asarray(snr, dtype=np.float32))
        snr_hdu.header['CREATOR'] = 'originstack uncertainty'
        snr_hdu.header['COMMENT'] = ('Per-pixel signal-to-noise above sky, using '
                                     'the propagated post-processing sigma')
        snr_hdu.writeto(snr_path, overwrite=True)
        safe_print(f"  Confidence map:         {os.path.basename(snr_path)}")


def error_aware_black_point(snr: np.ndarray, image: np.ndarray,
                            n_sigma: float = 3.0) -> Optional[float]:
    """Pixel value corresponding to the ``n_sigma`` confidence contour.

    Feeding this to the preview stretch as a black point clips everything the
    propagated error bars cannot distinguish from sky to black, instead of
    lifting it into visible "structure" -- the classic over-stretch failure
    where amplified correlated noise reads as nebulosity. Returns ``None``
    when no pixel clears the threshold (nothing to anchor on).
    """
    image = np.asarray(image, dtype=np.float32)
    lum = image.mean(axis=-1) if image.ndim == 3 else image
    mask = np.isfinite(snr) & (snr >= float(n_sigma))
    if not mask.any():
        return None
    return float(np.min(lum[mask]))
