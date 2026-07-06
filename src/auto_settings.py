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

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from src.models import Config
from src.utils import safe_print

try:
    from src.denoising import HAS_BM3D_PKG
except Exception:
    HAS_BM3D_PKG = False

if TYPE_CHECKING:
    import argparse

# ---------------------------------------------------------------------------
# Target type labels
# ---------------------------------------------------------------------------

TARGET_LABELS: Dict[str, str] = {
    'emission_nebula':   'Emission Nebula',
    'galaxy':            'Galaxy',
    'globular_cluster':  'Globular Cluster',
    'planetary_nebula':  'Planetary Nebula',
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

    # Exclude zeros for metrics that are only valid when measured successfully
    def _nz(key: str) -> list:
        return [f.metrics[key] for f in final if f.metrics and f.metrics.get(key, 0.0) > 0]

    fwhm_vals   = _nz('fwhm')
    strehl_vals = _nz('strehl')
    ellip_vals  = _nz('ellipticity')
    return {
        'median_background':  _med('background'),
        'median_noise':       _med('noise'),
        'median_p50':         _med('p50', _med('brightness')),
        'median_p75':         _med('p75'),
        'median_p95':         _med('p95'),
        'mean_star_count':    _mean('star_count'),
        'mean_dynamic_range': _mean('dynamic_range'),
        'mean_snr':           _mean('snr'),
        'mean_fwhm':          float(np.mean(fwhm_vals)) if fwhm_vals else 0.0,
        'mean_strehl':        float(np.mean(strehl_vals)) if strehl_vals else 0.0,
        'mean_dispersion':    _mean('dispersion_px'),
        'n_frames':           len(final),
        'median_ellipticity': float(np.median(ellip_vals)) if ellip_vals else 0.0,
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

    # How much of the frame is filled with emission at the median pixel.
    # High (>1) → extended object covers >50% of the FOV.
    median_filling = (p50 - bg) / noise

    # Signal present in the upper quartile above sky background.
    diffuse_excess = (p75 - bg) / noise

    # How concentrated the brightest signal is (≈ frame SNR).
    peak_excess = (p95 - bg) / noise

    # Ratio of peak brightness to diffuse brightness: high for compact objects
    # (globular cores, planetary nebula disks, galaxy nuclei), low for extended
    # emission that fills the frame uniformly.
    concentration = peak_excess / max(diffuse_excess, 0.1)

    return {
        'median_filling':    round(median_filling, 2),
        'diffuse_excess':    round(diffuse_excess, 2),
        'peak_excess':       round(peak_excess, 2),
        'concentration':     round(concentration, 2),
        'star_count':        agg['mean_star_count'],
        'dynamic_range':     agg['mean_dynamic_range'],
        'snr':               agg['mean_snr'],
        'fwhm':              agg['mean_fwhm'],
        'strehl':            round(agg['mean_strehl'], 3),
        'dispersion':        round(agg['mean_dispersion'], 3),
        'n_frames':          agg['n_frames'],
        'median_ellipticity': round(agg['median_ellipticity'], 3),
    }


# ---------------------------------------------------------------------------
# Target classification
# ---------------------------------------------------------------------------

def _classify(sig: dict) -> str:
    mf   = sig['median_filling']
    de   = sig['diffuse_excess']
    pe   = sig['peak_excess']
    sc   = sig['star_count']
    dr   = sig['dynamic_range']
    conc = sig['concentration']

    # Globular cluster: many stars with a strongly concentrated, very bright
    # core.  Checked before wide_field because rich globulars have sc > 200.
    # High dynamic range separates them from flat star fields.
    if sc > 60 and pe > 15 and conc > 2.5 and dr > 150:
        return 'globular_cluster'

    # Wide field (Milky Way): very dense star field, but not a concentrated blob
    if sc > 200:
        return 'wide_field'

    # Planetary nebula: compact bright extended source with few background stars.
    # High concentration distinguishes it from a galaxy (which also has low mf
    # and high pe but typically more foreground stars and lower concentration).
    if pe > 10 and mf < 0.3 and sc < 30 and conc > 2.0 and dr > 60:
        return 'planetary_nebula'

    # Star field: many stars, sky fills most pixels
    if sc > 80 and mf < 1:
        return 'star_field'

    # Emission nebula: emission fills >50% of the frame
    if mf > 1 and sc < 100:
        return 'emission_nebula'

    # Galaxy: compact bright source, but frame is mostly sky
    if pe > 8 and mf < 0.5 and dr > 100:
        return 'galaxy'

    # Reflection / bright nebula: moderate extended emission
    if de > 2 and sc < 100:
        return 'reflection_nebula'

    return 'unknown'


# ---------------------------------------------------------------------------
# Per-target settings table
# ---------------------------------------------------------------------------
#
# Each list holds (attr_name, value) pairs applied unconditionally for that
# target type.  Conditional logic (SNR gates, Strehl gates, frame-count
# thresholds) is handled in _apply_quality_settings.

_TARGET_SETTINGS: Dict[str, List[Tuple[str, object]]] = {
    'emission_nebula': [
        ('masked_correlation',      True),
        ('pre_gradient_removal',    True),
        ('deconvolve',              True),
        ('deconvolve_iterations',   20),
        # Empirical PSF captures asymmetric shapes from atmospheric turbulence
        # better than a parametric Gaussian/Moffat for filamentary nebulae.
        ('deconvolve_blind_psf',    True),
        ('star_reduce',             False),
        ('local_contrast_strength', 0.75),
        ('ghs_b',                   7.0),
        ('ghs_sp',                  0.18),
        # Emission nebula fills the frame — keep the black point below the sky
        # so faint outer nebulosity is not clipped to black.
        ('preview_black_sigma',    -0.5),
        ('dbe_patch_size',          48),
        # Emission patches have high entropy; rejecting them gives a cleaner
        # background model on narrowband data.
        ('entropy_bg',              True),
        # OSC sensors have 2:1 green-to-R/B Bayer ratio; SCNR removes the
        # residual green cast that survives flat-field correction.
        ('scnr',                    True),
        ('photometric_calibration', True),
        # Denoising: MMT edge-preserving median cascade for Ha filaments,
        # ACDNR for residual sky noise, then a gentle Perona-Malik pass
        # with option=2 (soft roll-off) to further protect filament edges.
        ('denoise',                 False),
        ('denoise_mmt',             True),
        ('denoise_acdnr',           True),
        ('denoise_aniso',           True),
        ('aniso_option',            2),
        ('aniso_iterations',        15),
    ],
    'galaxy': [
        ('pre_gradient_removal',    True),
        # Deconvolution OFF by default. On wide-field OSC data (undersampled
        # stars, small galaxy) Richardson-Lucy redistributes flux and lowers the
        # local background, so every bright star gets a dark ring/moat — and any
        # star-protection blend just moves the ring to the mask seam. The
        # marginal sharpening is not worth ringing every star. The galaxy is
        # already sharp from stacking. Re-enable with --deconvolve for
        # well-sampled data. (Apodized-PSF + wide-mask paths in postprocess still
        # apply when explicitly enabled.)
        ('deconvolve',              False),
        ('deconvolve_iterations',   15),
        ('deconvolve_blind_psf',    True),
        ('deconvolve_tv',           False),
        ('star_reduce',             True),
        ('star_reduce_factor',      0.5),
        ('local_contrast_strength', 0.85),
        ('ghs_b',                   10.0),
        ('ghs_sp',                  0.12),
        # Galaxy is a small target on empty sky — clip the sky noise to black
        # so the background does not become a colour-noise storm under stretch.
        ('preview_black_sigma',     2.0),
        # Smooth medium-scale colour blotches in the empty sky around the small
        # galaxy (object-masked, so galaxy/star colour is preserved).
        ('chroma_nr_large_sigma',   50.0),
        ('chroma_nr_large_strength', 0.7),
        ('dbe_patch_size',          48),
        # Outer globular cluster halos and H-II regions push patch entropy up,
        # biasing the background model toward the galaxy — reject them.
        ('entropy_bg',              True),
        ('scnr',                    True),
        ('photometric_calibration', True),
        # Denoising: MMT handles non-Gaussian noise near the bright core;
        # ACDNR cleans sky between arms; BM3D added conditionally when
        # SNR and frame count are sufficient (see _apply_quality_settings).
        ('denoise',                 False),
        ('denoise_mmt',             True),
        ('denoise_acdnr',           True),
    ],
    'globular_cluster': [
        # Deconvolution OFF: a dense star field is the worst case for RL
        # ringing — every one of hundreds of stars gets a dark moat. Not worth
        # it. Re-enable with --deconvolve for well-sampled data.
        ('deconvolve',              False),
        ('deconvolve_iterations',   20),
        ('deconvolve_blind_psf',    True),
        ('deconvolve_tv',           False),
        # Stars are the target — never soften them.
        ('star_reduce',             False),
        ('local_contrast',          True),
        ('local_contrast_strength', 0.80),
        # Higher SP lifts faint outer halo stars relative to the saturated core.
        ('ghs_b',                   7.0),
        ('ghs_sp',                  0.25),
        # Cluster on empty sky — clip sky noise to black.
        ('preview_black_sigma',     2.0),
        ('dbe_patch_size',          48),
        ('entropy_bg',              True),
        ('scnr',                    True),
        ('photometric_calibration', True),
        ('denoise',                 False),
        ('denoise_mmt',             True),
        ('denoise_acdnr',           True),
    ],
    'planetary_nebula': [
        # More iterations than other targets: resolving a compact disk or ring
        # requires many RL/TV cycles to recover sub-seeing shell structure.
        ('deconvolve',              True),
        ('deconvolve_iterations',   25),
        ('deconvolve_blind_psf',    True),
        # TV staircasing boxes the compact bright shell/disk — use RL instead.
        ('deconvolve_tv',           False),
        # Reduce background stars so the compact nebula is not dominated by halos.
        ('star_reduce',             True),
        ('star_reduce_factor',      0.5),
        ('local_contrast',          True),
        ('local_contrast_strength', 0.80),
        # Aggressive stretch to lift the faint outer halo and jets.
        ('ghs_b',                   9.0),
        ('ghs_sp',                  0.10),
        # Protect the bright central star from blowout.
        ('ghs_hp',                  0.92),
        # Compact nebula on empty sky, but keep the faint outer halo — mild clip.
        ('preview_black_sigma',     1.0),
        # Smaller patches → denser background sampling around the compact disk.
        ('dbe_patch_size',          32),
        ('entropy_bg',              True),
        ('scnr',                    True),
        ('photometric_calibration', True),
        ('denoise',                 False),
        ('denoise_mmt',             True),
        ('denoise_acdnr',           True),
        ('denoise_aniso',           True),
        ('aniso_option',            2),
        ('aniso_iterations',        15),
    ],
    'reflection_nebula': [
        ('masked_correlation',      True),
        ('pre_gradient_removal',    True),
        ('deconvolve',              True),
        ('deconvolve_iterations',   15),
        ('deconvolve_blind_psf',    True),
        ('star_reduce',             False),
        ('local_contrast_strength', 0.70),
        ('ghs_b',                   8.0),
        ('ghs_sp',                  0.15),
        # Reflection nebula fills the frame — keep faint dust visible.
        ('preview_black_sigma',    -0.5),
        ('entropy_bg',              True),
        ('scnr',                    True),
        ('photometric_calibration', True),
        # Same MMT + ACDNR + anisotropic diffusion combination as emission
        # nebulae; dust structure in reflection nebulae responds well to the
        # PDE edge preservation.
        ('denoise',                 False),
        ('denoise_mmt',             True),
        ('denoise_acdnr',           True),
        ('denoise_aniso',           True),
        ('aniso_option',            2),
        ('aniso_iterations',        15),
    ],
    'star_field': [
        # deconvolve is conditional — set in _apply_target_settings
        ('star_reduce',             False),
        ('local_contrast_strength', 0.5),
        ('ghs_b',                   6.0),
        # Stars on empty sky — clip sky noise to black.
        ('preview_black_sigma',     2.0),
        ('scnr',                    True),
        ('photometric_calibration', True),
        ('denoise_acdnr',           True),
        ('denoise_acdnr_k',         4.0),   # conservative: only flat-sky pixels
    ],
    'wide_field': [
        ('deconvolve',              False),
        ('star_reduce',             False),
        ('local_contrast_strength', 0.6),
        ('ghs_b',                   5.0),
        ('ghs_sp',                  0.20),
        # Wide field usually has scattered nebulosity — mild clip only.
        ('preview_black_sigma',     0.5),
        ('scnr',                    True),
        ('photometric_calibration', True),
        ('denoise_acdnr',           True),
        ('denoise_acdnr_k',         4.0),
    ],
    'unknown': [
        ('scnr',                    True),
        ('photometric_calibration', True),
    ],
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

def _apply_quality_settings(
    sig: dict, args, target_type: str = 'unknown'
) -> List[str]:
    """Apply frame-count, SNR, FWHM, Strehl, and dispersion adjustments."""
    changes: List[str] = []

    def _set(attr: str, val: object) -> None:
        old = getattr(args, attr, None)
        if old != val:
            setattr(args, attr, val)
            changes.append(f"{attr}  {old!r} -> {val!r}")

    def _disable_deconvolve() -> None:
        if getattr(args, 'deconvolve', False):
            _set('deconvolve', False)
        if getattr(args, 'deconvolve_tv', False):
            _set('deconvolve_tv', False)

    n           = int(sig['n_frames'])
    snr         = sig['snr']
    fwhm        = sig['fwhm']
    sc          = sig['star_count']
    strehl      = sig.get('strehl', 0.0)
    dispersion  = sig.get('dispersion', 0.0)
    med_ellip   = sig.get('median_ellipticity', 0.0)

    # 1. Frame-count-based stacking method (only when still at default 'auto')
    if getattr(args, 'stack_method', 'auto') == 'auto':
        if n < 8:
            _set('stack_method', 'percentile')
        elif n < 15:
            # Trimmed mean: simple, robust alternative to ESD for moderate N.
            _set('stack_method', 'trimmed_mean')
        elif n < 20:
            _set('stack_method', 'sigma_clip')
            _set('rejection_sigma', 3.0)
        else:
            _set('stack_method', 'sigma_clip')
            _set('rejection_sigma', 2.8)
            _set('rejection_iters', 4)

    # Consensus reference frame: enable for large frame counts (≥20)
    if n >= 20 and not getattr(args, 'consensus_ref', False):
        _set('consensus_ref', True)

    # 2. SNR-based denoising (only when auto-tuning is explicitly disabled)
    if not getattr(args, 'auto_denoise_strength', True):
        if snr > 0:
            if snr < 5:
                _set('denoise_strength', 4.5)
            elif snr < 10:
                _set('denoise_strength', 3.5)
            elif snr > 20:
                _set('denoise_strength', 2.0)

    # Ellipticity warning: poor tracking produces elongated stars
    if fwhm > 4.0 and med_ellip > 0.3:
        safe_print(f"  NOTE: median star ellipticity={med_ellip:.3f} > 0.3 with FWHM={fwhm:.1f}px "
                   f"— possible tracking error or optical issue")

    # 3. FWHM-scaled deconvolution iterations (applied after target type sets them)
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

    # 5. MMT strength scaled to SNR
    if getattr(args, 'denoise_mmt', False) and snr > 0:
        if snr < 5:
            _set('denoise_mmt_strength', 4.0)
        elif snr < 10:
            _set('denoise_mmt_strength', 3.5)
        elif snr > 20:
            _set('denoise_mmt_strength', 2.0)

    # 6. Baseline ACDNR for unknown/noisy targets not already using it
    if snr < 5 and snr > 0 and not getattr(args, 'denoise_acdnr', False):
        _set('denoise_acdnr', True)

    # 7. BM3D: enable when the bm3d package is installed and the stack has enough
    #    SNR for block-matching to outperform median-based methods.  Thresholds are
    #    lower when the package is available (hardware-accelerated C backend) vs
    #    absent (pure-scipy DCT fallback which is ~5-10x slower and less effective).
    _bm3d_targets = ('galaxy', 'globular_cluster', 'emission_nebula', 'reflection_nebula')
    _bm3d_snr_min = 8 if HAS_BM3D_PKG else 12
    _bm3d_n_min   = 12 if HAS_BM3D_PKG else 20
    if (target_type in _bm3d_targets
            and snr >= _bm3d_snr_min
            and n >= _bm3d_n_min
            and not getattr(args, 'denoise_bm3d', False)):
        _set('denoise_bm3d', True)
        if target_type == 'globular_cluster' or snr > 20:
            _set('bm3d_stride', 4)

    # 8. TV deconvolution iteration count scaled to stack SNR.
    #    Low SNR: fewer iterations to avoid converging toward noise.
    #    High SNR: more iterations converge to a sharper solution.
    if getattr(args, 'deconvolve_tv', False):
        base = Config.TV_ITERATIONS
        if snr > 0 and snr < 8:
            _set('tv_iterations', max(30, base - 10))
        elif snr > 20:
            _set('tv_iterations', min(80, base + 20))

    # 10. Strehl-based deconvolution gate.
    #     Very low Strehl means PSF varies significantly across frames after
    #     stacking; deconvolution amplifies mis-modelled PSF artefacts.
    #     - Strehl < 0.15: PSF too variable — disable deconvolution entirely.
    #     - Strehl 0.15–0.25: cut iterations in half as a safety margin.
    if strehl > 0:
        if strehl < 0.15:
            _disable_deconvolve()
        elif strehl < 0.25:
            current_iters = getattr(args, 'deconvolve_iterations', 15)
            _set('deconvolve_iterations', max(8, current_iters // 2))

    # 11. Atmospheric dispersion gate.
    #     High dispersion makes the PSF strongly chromatic — a single PSF
    #     estimate will fit R/G/B channels poorly, producing colour fringes.
    #     - Dispersion > 3.0 px: disable deconvolution.
    #     - Dispersion 1.5–3.0 px: reduce iterations by 40%.
    if dispersion > 0:
        if dispersion > 3.0:
            _disable_deconvolve()
        elif dispersion > 1.5:
            current_iters = getattr(args, 'deconvolve_iterations', 15)
            _set('deconvolve_iterations', max(5, int(current_iters * 0.6)))

    # 12. Patch-weighted stacking: enable when there are enough frames for the
    #     quality-weighted mean to be statistically robust and quality maps will
    #     have signal.  Requires fwhm > 0 (stars/structure detected) so that
    #     Brenner sharpness has something to discriminate against.
    #     With ≥15 frames each contributes ≤6.7% at full weight, so a single
    #     bad-seeing patch is diluted sufficiently without hard rejection.
    #     Only activates when the user has not already set --patch-weighted.
    if (n >= 15
            and fwhm > 0.0
            and not getattr(args, 'patch_registration', False)):
        _set('patch_registration', True)

    return changes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_auto_settings(
    final: list,
    args,
    prior_type: Optional[str] = None,
    prior_confidence: float = 0.0,
) -> Tuple[str, str, dict, List[str]]:
    """Classify the target and apply heuristic settings to args in-place.

    Parameters
    ----------
    final : list[FrameInfo]
        Accepted frames after Phase 1, each with .metrics populated.
    args : argparse.Namespace
        Modified in-place.
    prior_type : str or None
        Object type inferred from metadata (folder/header/Simbad).  When
        confidence is high enough this overrides the pixel-signal classifier.
    prior_confidence : float
        Confidence of *prior_type* in the range 0–1.
        ≥ 0.85 → prior overrides the heuristic classifier outright.
        ≥ 0.70 → prior wins when heuristic returns 'unknown'.

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
    heuristic_type = _classify(sig)

    # Resolve final classification
    valid_prior = (prior_type and prior_type in TARGET_LABELS
                   and prior_type != 'unknown')
    if valid_prior and prior_confidence >= 0.85:
        target_type = prior_type
    elif valid_prior and prior_confidence >= 0.70 and heuristic_type == 'unknown':
        target_type = prior_type
    else:
        target_type = heuristic_type

    label = TARGET_LABELS[target_type]

    changes: List[str] = []
    changes.extend(_apply_target_settings(target_type, sig, args))
    changes.extend(_apply_quality_settings(sig, args, target_type=target_type))

    return target_type, label, sig, changes
