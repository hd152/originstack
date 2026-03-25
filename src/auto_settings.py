"""Pure-heuristic auto parameter advisor.

No API calls required.  Runs after Phase 1 quality analysis, classifies the
likely target type from per-frame metrics, and applies optimised processing
settings to the args namespace in-place.

Public API
----------
apply_auto_settings(final, args) -> (target_type, label, signals, changes)
    final   : list[FrameInfo]  — accepted frames with .metrics populated
    args    : argparse.Namespace  — modified in-place
    returns : (str, str, dict, list[str])
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    import argparse

# ---------------------------------------------------------------------------
# Target type labels
# ---------------------------------------------------------------------------

TARGET_LABELS: Dict[str, str] = {
    'emission_nebula':   'Emission Nebula',
    'galaxy':            'Galaxy',
    'reflection_nebula': 'Reflection Nebula',
    'star_field':        'Star Field',
    'wide_field':        'Wide Field / Milky Way',
    'unknown':           'Unknown / Generic Deep Sky',
}


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _aggregate(final: list) -> dict:
    """Compute aggregate statistics across accepted frames."""
    def _med(key: str, default: float = 0.0) -> float:
        vals = [f.metrics.get(key, default) for f in final if f.metrics]
        return float(np.median(vals)) if vals else default

    def _mean(key: str, default: float = 0.0) -> float:
        vals = [f.metrics.get(key, default) for f in final if f.metrics]
        return float(np.mean(vals)) if vals else default

    # fwhm: exclude zero values (frames with no stars)
    fwhm_vals = [f.metrics.get('fwhm', 0.0) for f in final
                 if f.metrics and f.metrics.get('fwhm', 0.0) > 0]

    # brightness == median pixel value == p50
    return {
        'median_background':  _med('background'),
        'median_noise':       _med('noise'),
        'median_p50':         _med('p50', _med('brightness')),  # frame median pixel
        'median_p75':         _med('p75'),
        'median_p95':         _med('p95'),
        'mean_star_count':    _mean('star_count'),
        'mean_dynamic_range': _mean('dynamic_range'),
        'mean_snr':           _mean('snr'),
        'mean_fwhm':          float(np.mean(fwhm_vals)) if fwhm_vals else 0.0,
        'n_frames':           len(final),
    }


# ---------------------------------------------------------------------------
# Classification signals
# ---------------------------------------------------------------------------

def _compute_signals(agg: dict) -> dict:
    bg    = agg['median_background']
    noise = max(agg['median_noise'], 1e-6)
    p50   = agg['median_p50']
    p75   = agg['median_p75']
    p95   = agg['median_p95']

    # How much of the frame is filled with emission (signal at the median pixel).
    # High (>1) → extended object covers >50% of the FOV (e.g. emission nebula).
    # Near zero → most pixels are sky background (galaxy, star field).
    median_filling = (p50 - bg) / noise

    # Signal present in the upper quartile above sky background.
    diffuse_excess = (p75 - bg) / noise

    # How concentrated the brightest signal is relative to the mid-signal.
    # High → compact bright source (galaxy nucleus, bright star).
    # Low  → emission is spread uniformly across the frame.
    peak_excess = (p95 - bg) / noise  # ≈ frame SNR

    return {
        'median_filling': round(median_filling, 2),
        'diffuse_excess': round(diffuse_excess, 2),
        'peak_excess':    round(peak_excess, 2),
        'star_count':     agg['mean_star_count'],
        'dynamic_range':  agg['mean_dynamic_range'],
        'snr':            agg['mean_snr'],
        'fwhm':           agg['mean_fwhm'],
        'n_frames':       agg['n_frames'],
    }


# ---------------------------------------------------------------------------
# Target classification
# ---------------------------------------------------------------------------

def _classify(sig: dict) -> str:
    mf  = sig['median_filling']   # signal filling the frame at p50
    de  = sig['diffuse_excess']   # signal at p75 above sky
    pe  = sig['peak_excess']      # signal at p95 above sky  (≈ SNR)
    sc  = sig['star_count']
    dr  = sig['dynamic_range']

    # Wide field (Milky Way): very dense star field
    if sc > 200:
        return 'wide_field'

    # Star field: many stars, sky fills most pixels (mf low)
    if sc > 80 and mf < 1:
        return 'star_field'

    # Emission nebula: emission fills >50% of the frame (mf high), few stars
    if mf > 1 and sc < 100:
        return 'emission_nebula'

    # Galaxy: compact bright source (p95 >> p75), but frame is mostly sky (mf low)
    if pe > 8 and mf < 0.5 and dr > 100:
        return 'galaxy'

    # Reflection / bright nebula: moderate extended emission, fewer stars
    if de > 2 and sc < 100:
        return 'reflection_nebula'

    return 'unknown'


# ---------------------------------------------------------------------------
# Per-target settings table
# ---------------------------------------------------------------------------

# Each list holds (attr_name, value) pairs applied unconditionally for that type.
# Conditional logic (e.g. FWHM-gated deconvolution for star_field) is handled
# inline in _apply_target_settings.

_TARGET_SETTINGS: Dict[str, List[Tuple[str, object]]] = {
    'emission_nebula': [
        ('deconvolve',              True),
        ('deconvolve_iterations',   20),
        ('star_reduce',             False),   # keep stars crisp
        ('local_contrast_strength', 0.75),
        ('ghs_b',                   7.0),
        ('ghs_sp',                  0.18),
        ('dbe_patch_size',          48),
        # Denoising: MMT's edge-preserving median cascade protects thin Ha
        # filaments better than the DWT; ACDNR then adaptively smooths any
        # residual sky noise without touching the filament edges.
        ('denoise',                 False),   # wavelet replaced by MMT
        ('denoise_mmt',             True),
        ('denoise_acdnr',           True),
    ],
    'galaxy': [
        ('deconvolve',              True),
        ('deconvolve_iterations',   15),
        ('star_reduce',             True),
        ('star_reduce_factor',      0.5),     # softer stars vs galaxy
        ('local_contrast_strength', 0.85),    # strong for dust lanes
        ('ghs_b',                   10.0),    # stretch faint halo
        ('ghs_sp',                  0.12),
        ('dbe_patch_size',          48),
        # Denoising: MMT handles the non-Gaussian noise near the bright core
        # without smearing the faint outer halo; ACDNR cleans up sky between
        # the spiral arms without blurring the arm edges.
        ('denoise',                 False),   # wavelet replaced by MMT
        ('denoise_mmt',             True),
        ('denoise_acdnr',           True),
    ],
    'reflection_nebula': [
        ('deconvolve',              True),
        ('deconvolve_iterations',   15),
        ('star_reduce',             False),
        ('local_contrast_strength', 0.70),
        ('ghs_b',                   8.0),
        ('ghs_sp',                  0.15),
        # Denoising: same reasoning as emission nebula; reflection nebulae
        # have fine dust structure that median-based MMT handles well.
        ('denoise',                 False),   # wavelet replaced by MMT
        ('denoise_mmt',             True),
        ('denoise_acdnr',           True),
    ],
    'star_field': [
        # deconvolve is conditional — set in _apply_target_settings
        ('star_reduce',             False),
        ('local_contrast_strength', 0.5),
        ('ghs_b',                   6.0),
        # Denoising: wavelet (default) handles uniform background well;
        # ACDNR added conservatively to clean sky patches between stars
        # without touching star halos (high k → only flat-sky pixels smoothed).
        ('denoise_acdnr',           True),
        ('denoise_acdnr_k',         4.0),
    ],
    'wide_field': [
        ('deconvolve',              False),
        ('star_reduce',             False),
        ('local_contrast_strength', 0.6),
        ('ghs_b',                   5.0),
        ('ghs_sp',                  0.20),
        # Denoising: wavelet (default) is fine for dense star fields;
        # ACDNR removes sky graininess conservatively.
        ('denoise_acdnr',           True),
        ('denoise_acdnr_k',         4.0),
    ],
    'unknown': [],
}


def _apply_target_settings(
    target_type: str, sig: dict, args
) -> List[str]:
    """Apply per-target settings to args.  Returns list of change strings."""
    changes: List[str] = []

    def _set(attr: str, val: object) -> None:
        old = getattr(args, attr, None)
        if old != val:
            setattr(args, attr, val)
            changes.append(f"{attr}  {old!r} -> {val!r}")

    # Star field: deconvolution only when seeing is poor
    if target_type == 'star_field':
        fwhm = sig['fwhm']
        if fwhm > 4.0:
            _set('deconvolve', True)
            _set('deconvolve_iterations', 10)
        else:
            _set('deconvolve', False)

    for attr, val in _TARGET_SETTINGS.get(target_type, []):
        _set(attr, val)

    return changes


# ---------------------------------------------------------------------------
# Quality-based adjustments (always applied, independent of target type)
# ---------------------------------------------------------------------------

def _apply_quality_settings(sig: dict, args) -> List[str]:
    """Apply frame-count, SNR, FWHM, and debayer quality adjustments."""
    changes: List[str] = []

    def _set(attr: str, val: object) -> None:
        old = getattr(args, attr, None)
        if old != val:
            setattr(args, attr, val)
            changes.append(f"{attr}  {old!r} -> {val!r}")

    n   = int(sig['n_frames'])
    snr = sig['snr']
    fwhm = sig['fwhm']

    # 1. Frame-count-based stacking method (only when still at default 'auto')
    if getattr(args, 'stack_method', 'auto') == 'auto':
        if n < 8:
            _set('stack_method', 'percentile')
        elif n < 20:
            _set('stack_method', 'sigma_clip')
            _set('rejection_sigma', 3.0)
        else:
            _set('stack_method', 'sigma_clip')
            _set('rejection_sigma', 2.8)
            _set('rejection_iters', 4)

    # 2. SNR-based denoising (only when auto-tuning is explicitly disabled)
    if not getattr(args, 'auto_denoise_strength', True):
        if snr > 0:
            if snr < 5:
                _set('denoise_strength', 4.5)
            elif snr < 10:
                _set('denoise_strength', 3.5)
            elif snr > 20:
                _set('denoise_strength', 2.0)

    # 3. FWHM-scaled deconvolution iterations (applied after target-type set them)
    if getattr(args, 'deconvolve', False) and fwhm > 4.0:
        current_iters = getattr(args, 'deconvolve_iterations', 15)
        extra = int((fwhm - 3.0) * 3)
        new_iters = min(current_iters + extra, 30)
        if new_iters != current_iters:
            _set('deconvolve_iterations', new_iters)

    # 4. Debayer upgrade: bilinear -> malvar when OpenCV is available
    if getattr(args, 'debayer_method', 'bilinear') == 'bilinear':
        try:
            import cv2 as _cv2  # noqa: F401
            _set('debayer_method', 'malvar')
        except ImportError:
            pass

    # 5. MMT strength scaled to SNR (only when MMT was selected by target type)
    #    Wavelet strength is handled separately by estimate_denoise_strength().
    if getattr(args, 'denoise_mmt', False) and snr > 0:
        if snr < 5:
            _set('denoise_mmt_strength', 4.0)   # heavy for very noisy stacks
        elif snr < 10:
            _set('denoise_mmt_strength', 3.5)
        elif snr > 20:
            _set('denoise_mmt_strength', 2.0)   # gentle; clean data needs less

    # 6. For unknown/unclassified targets: add ACDNR when the stack is very
    #    noisy (SNR < 5) and nothing else has already been set.  At this noise
    #    level the adaptive sky smoothing always helps regardless of target type.
    if snr < 5 and snr > 0 and not getattr(args, 'denoise_acdnr', False):
        _set('denoise_acdnr', True)

    return changes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_auto_settings(
    final: list,
    args,
) -> Tuple[str, str, dict, List[str]]:
    """Classify the target and apply heuristic settings to args in-place.

    Parameters
    ----------
    final : list[FrameInfo]
        Accepted frames after Phase 1, each with .metrics populated.
    args : argparse.Namespace
        Modified in-place.

    Returns
    -------
    target_type : str
        Internal type key (e.g. ``'emission_nebula'``).
    label : str
        Human-readable label (e.g. ``'Emission Nebula'``).
    signals : dict
        Classification signals, useful for diagnostics.
    changes : list[str]
        Every setting that was changed, as ``"attr  old -> new"`` strings.
    """
    if not final:
        return 'unknown', TARGET_LABELS['unknown'], {}, []

    agg = _aggregate(final)
    sig = _compute_signals(agg)
    target_type = _classify(sig)
    label = TARGET_LABELS[target_type]

    changes: List[str] = []
    changes.extend(_apply_target_settings(target_type, sig, args))
    changes.extend(_apply_quality_settings(sig, args))

    return target_type, label, sig, changes
