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
of the outputs. That needs no changes to any post-processing step and keeps
working automatically when a new denoiser is added. The cost is ``K`` extra
post-processing passes, which is why it is opt-in
(``--uncertainty-propagate``).

**Accuracy, stated honestly, with the caveat measured rather than asserted.**
For a chain whose behaviour does not depend on its input's noise level this is
exact up to Monte Carlo error (~``1/sqrt(2K)`` on the standard deviation,
~25% at the default K=8). For the adaptive parts of this pipeline it is
slightly biased, in a known direction, by a known mechanism.

The realizations are built as ``stacked + N(0, sigma)``, but ``stacked``
already carries about ``sigma`` of noise of its own -- that is what ``sigma``
measures. Each realization therefore presents Phase 4 with roughly
``sqrt(2)*sigma``. A linear step does not care. A step that *estimates its own
parameters from the data* does: BayesShrink reads its threshold off each
subband's measured noise, DBE and the sky-floor passes measure sky sigma,
``estimate_denoise_strength`` keys off SNR. Each denoises the inflated
realization slightly harder than it denoised the real image, so the spread
that comes back understates the chain's true output noise.

**How much, measured against this project's own ``wavelet_denoise`` rather
than a toy:** on a structured synthetic field (sky gradient + gaussian blobs)
swept over sigma in {1, 4, 12} ADU and ``threshold_factor`` in {2, 3, 5}, the
ratio of propagated to true output sigma lands in **0.93-1.03** -- a few
percent low in most configurations, occasionally a hair high. That is well
inside the ~25% Monte Carlo error the default K already carries, so it is a
caveat on interpretation, not a reason to distrust the number.

Worth recording explicitly, because the size of this effect is easy to
overestimate from first principles: a plausible-sounding argument says a
self-estimating soft-threshold handed sqrt(2)x the noise should understate by
~2x. It does not, for these denoisers. The shrinkage and the threshold move
together, and the errors largely cancel. The bias is real; it is small.

Because the direction is nonetheless the dangerous one -- understatement reads
*more* significance than the data supports, and
``confidence_map``/``error_aware_black_point`` inherit it --
``propagate_uncertainty`` also returns an ``adaptivity`` ratio measuring how
far from noise-scale-invariant the chain behaved *on this image*, so a future
denoiser with a stronger adaptive response is detected rather than assumed
away. ``tests/test_uncertainty_propagation.py`` pins both the linear cases and
the adaptive one.

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
#
# Getting this list wrong is silent in both directions: every realization runs
# under redirect_stdout, so a step that writes a sidecar K times still prints
# its own "Saved:" line into a swallowed buffer. The file left on disk is then
# the *last noise realization*, not the real image. Anything in Phase 4 that
# writes a file or calls out to the network belongs here.
_QUIET_OFF = (
    'remove_stars',            # writes <output>_starless.fits
    'nmf_separate',            # writes _star_component/_nebula_component.fits
    'photometric_calibration',  # Gaia/VizieR network queries, per-channel scalar
    'annotate',
    'verbose',
    'aberration_report',       # writes <output>_aberration.png
    'diagnostic',              # ditto, via the same block
    'export_masks',            # writes <output>_star_mask.fits
    'keep_intermediates',      # writes <output>_background.jpg
    'comet_radial_renorm',     # writes <output>_comet_renorm.fits
    'comet_larson_sekanina',   # writes <output>_comet_ls.fits
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
    probe_realizations: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
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
        probe_realizations: extra passes run at *half* noise amplitude to
            measure how noise-scale-dependent the chain is (see
            ``_measure_adaptivity``). 0 disables the probe.

    Returns:
        ``(sigma_post, mean_post, adaptivity)``.

        ``sigma_post`` is the per-pixel standard deviation across realizations
        ``(H, W)``, collapsed across channels by averaging the per-channel
        variances. ``mean_post`` is the per-pixel mean of the realizations
        ``(H, W, C)`` -- returned because for a nonlinear chain it is *not*
        equal to the chain applied to the unperturbed input, and the
        difference is itself diagnostic (a large gap means the chain is
        strongly nonlinear at that pixel, i.e. the error bar is skewed).

        ``adaptivity`` is ``sigma(full amplitude) / sigma(half amplitude)``,
        or None if the probe was disabled or failed. 2.0 means the chain
        treated noise scale-invariantly and ``sigma_post`` can be read at face
        value; below 2.0 means the chain suppressed the inflated realization
        noise and ``sigma_post`` is a *lower bound* by roughly that shortfall.
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

    def _spread(amplitude: float, n: int, label: str):
        """Welford spread of ``n`` realizations at ``amplitude * sigma``.

        Welford rather than stacking the outputs: K full (H,W,C) results would
        be K x the stack's own footprint, and this pipeline's whole memory
        model is built on not materialising K copies of anything.
        """
        mean = np.zeros((H, W, C), dtype=np.float64)
        m2 = np.zeros((H, W, C), dtype=np.float64)
        n_ok = 0

        for k in range(n):
            # standard_normal(dtype=float32) draws f32 directly; rng.normal()
            # would build an (H,W,C) float64 array and then copy it down.
            noise = rng.standard_normal(size=(H, W, C), dtype=np.float32)
            noise *= sigma_b * np.float32(amplitude)
            noisy = stacked + noise
            del noise
            try:
                # `noisy` is freshly allocated every iteration, so Phase 4 may
                # mutate it in place; stdout is swallowed so K passes don't
                # produce K copies of the whole Phase 4 log.
                buf = io.StringIO()
                with redirect_stdout(buf):
                    out = postprocess_fn(noisy, quiet, final, stats)
            except Exception as exc:
                _log.debug("uncertainty realization %d (%s) failed (%s); skipping",
                           k, label, exc)
                continue

            out = np.asarray(out, dtype=np.float64)
            if out.ndim == 2:
                out = out[:, :, np.newaxis]
            if out.shape[:2] != (H, W):
                # A Phase 4 step that changes geometry (e.g. a crop) makes the
                # realizations non-comparable; bail rather than silently
                # compare misaligned pixels.
                raise ValueError(
                    f"post-processing changed image shape {stacked.shape} -> "
                    f"{out.shape}; uncertainty propagation needs a "
                    "geometry-preserving chain")

            n_ok += 1
            delta = out - mean
            mean += delta / n_ok
            m2 += delta * (out - mean)

            if verbose:
                safe_print(f"    realization {k + 1}/{n} ({label})")

        return mean, m2, n_ok

    mean, m2, n_ok = _spread(1.0, K, "full")

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

    adaptivity = _measure_adaptivity(sigma_post, _spread, probe_realizations)
    return sigma_post, mean.astype(np.float32), adaptivity


def _measure_adaptivity(sigma_post, spread_fn, probe_realizations: int):
    """How far from noise-scale-invariant the chain behaved, as a ratio.

    The module docstring explains why the propagated sigma is a lower bound
    wherever a Phase 4 step estimates its own parameters from the data. This
    turns that caveat from a warning into a number: repeat a couple of
    realizations at *half* the noise amplitude and compare spreads.

    A chain whose behaviour does not depend on its input's noise level
    produces a spread proportional to the amplitude, so
    ``sigma(full) / sigma(half)`` is 2.0. A chain that denoises harder when
    handed more noise returns less than 2.0, and the shortfall is the bias:
    a ratio of 1.0 means the chain suppressed the extra noise entirely and the
    propagated sigma says nothing about the real error bar.

    Returns None when the probe is disabled or could not run -- a missing
    diagnostic must not take the propagated sigma down with it.
    """
    if probe_realizations < 2:
        return None
    try:
        _, m2_half, n_half = spread_fn(0.5, probe_realizations, "half-amplitude probe")
        if n_half < 2:
            return None
        var_half = m2_half / (n_half - 1)
        sigma_half = np.sqrt(np.maximum(var_half.mean(axis=-1), 0.0))
        # Compare where there is signal to compare: pixels whose half-amplitude
        # spread is a meaningful fraction of the median, so near-zero pixels
        # (clamped by the chain) don't dominate the ratio.
        med = float(np.median(sigma_half))
        if not np.isfinite(med) or med <= 0.0:
            return None
        live = sigma_half > (0.5 * med)
        if not np.any(live):
            return None
        return float(np.median(sigma_post[live] / sigma_half[live]))
    except Exception as exc:
        _log.debug("adaptivity probe failed (%s); reporting no ratio", exc)
        return None


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
