"""Pure-heuristic auto parameter advisor.

No API calls required.  Runs after Phase 1 quality analysis, classifies the
likely target type from per-frame metrics (for display), and applies
processing settings to the args namespace in-place -- continuously blended
across 8 target-type presets by signal-space proximity (_blend_weights /
_apply_dynamic_settings), not a single bucket lookup. See the comment block
above _TYPE_ANCHORS for the design.

Public API
----------
apply_auto_settings(final, args) -> (target_type, label, signals, changes, weights)
    final   : list[FrameInfo]  — accepted frames with .metrics populated
    args    : argparse.Namespace  — modified in-place
    returns : (str, str, dict, list[str], dict[str, float])
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
        # Deconvolution OFF by default. Same RL-ringing problem documented
        # for galaxy/globular_cluster below, confirmed on a real Trifid
        # Nebula session: unregularized RL with a blind-PSF estimate rings
        # every star and most nebula edges (iteration count gets pushed
        # well past this list's default 20 by the FWHM-based bump in
        # _apply_quality_settings on soft-seeing data, making it worse).
        # Star-core blending only protects the tight core, not the
        # extended ring radius. Re-enable with --deconvolve for
        # well-sampled data / a target that tolerates the tradeoff.
        ('deconvolve',              False),
        ('deconvolve_iterations',   20),
        # Empirical PSF captures asymmetric shapes from atmospheric turbulence
        # better than a parametric Gaussian/Moffat for filamentary nebulae.
        ('deconvolve_blind_psf',    True),
        ('star_reduce',             False),
        ('remove_stars',            True),
        ('galaxy_mode',             False),
        ('local_contrast_strength', 0.75),
        # Toned down from (7.0, 0.18): a real Trifid Nebula render at the old
        # values looked oversaturated/punchy (deep magenta emission region,
        # not hard-clipped -- only ~0.4% of nebula pixels hit 255 -- just a
        # much more vivid curve than the source data needs). (5.0, 0.10)
        # verified side-by-side on the same stack: visibly softer/more
        # natural color, nebula structure and star field equally clear.
        ('ghs_b',                   5.0),
        ('ghs_sp',                  0.10),
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
        # Denoising: directional (curvelet-inspired) wavelet denoising
        # instead of MMT as primary -- its structure-tensor coherence map
        # is a more targeted mechanism for exactly what this target has a
        # lot of (elongated Ha filaments) than MMT's general median
        # cascade. ACDNR for residual sky noise, then a gentle Perona-Malik
        # pass with option=2 (soft roll-off) to further protect filament
        # edges.
        ('denoise',                 False),
        ('denoise_mmt',             False),
        ('denoise_curvelet',        True),
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
        ('remove_stars',            True),
        # The galaxy's own broad, smooth halo tapers with no hard edge, so a
        # per-pixel significance mask can't reliably tell it apart from
        # residual gradient at the object's edge -- the smooth background fit
        # still tracks (and subtracts) much of it. A generous fixed exclusion
        # disk around the detected nucleus (same mechanism --comet-mode uses
        # for the coma) keeps the fit's boundary out in real sky instead.
        ('galaxy_mode',             True),
        ('local_contrast_strength', 0.85),
        ('ghs_b',                   10.0),
        ('ghs_sp',                  0.12),
        # Galaxy is a small target on empty sky — clip the sky noise tail to
        # black (median + 3*sigma) so the vast empty background renders clean,
        # not grainy, under stretch. Faint arms/companion sit well above this.
        ('preview_black_sigma',     3.0),
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
        # Stars are the target — never soften them, and never produce a
        # starless sidecar (it would erase the target itself).
        ('star_reduce',             False),
        ('remove_stars',            False),
        ('galaxy_mode',             False),
        ('local_contrast',          True),
        ('local_contrast_strength', 0.80),
        # Higher SP lifts faint outer halo stars relative to the saturated core.
        ('ghs_b',                   7.0),
        ('ghs_sp',                  0.25),
        # Cluster on empty sky — clip sky noise tail to black (stars sit above).
        ('preview_black_sigma',     3.0),
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
        ('remove_stars',            True),
        ('galaxy_mode',             False),
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
        # Deconvolution OFF by default -- see emission_nebula above (same
        # RL-ringing issue, confirmed on a real Trifid Nebula session,
        # which has a strong reflection component).
        ('deconvolve',              False),
        ('deconvolve_iterations',   15),
        ('deconvolve_blind_psf',    True),
        ('star_reduce',             False),
        ('remove_stars',            True),
        ('galaxy_mode',             False),
        ('local_contrast_strength', 0.70),
        ('ghs_b',                   8.0),
        ('ghs_sp',                  0.15),
        # Reflection nebula fills the frame — keep faint dust visible.
        ('preview_black_sigma',    -0.5),
        ('entropy_bg',              True),
        ('scnr',                    True),
        ('photometric_calibration', True),
        # Same directional (curvelet-inspired) + ACDNR + anisotropic
        # diffusion combination as emission nebulae; dust structure in
        # reflection nebulae is elongated/filamentary too and responds
        # well to both the coherence-protected thresholding and the PDE
        # edge preservation.
        ('denoise',                 False),
        ('denoise_mmt',             False),
        ('denoise_curvelet',        True),
        ('denoise_acdnr',           True),
        ('denoise_aniso',           True),
        ('aniso_option',            2),
        ('aniso_iterations',        15),
    ],
    'star_field': [
        # deconvolve is conditional — set in _apply_dynamic_settings
        ('star_reduce',             False),
        ('remove_stars',            False),
        ('galaxy_mode',             False),
        ('local_contrast_strength', 0.5),
        ('ghs_b',                   6.0),
        # Stars on empty sky — clip sky noise tail to black (stars sit above).
        ('preview_black_sigma',     3.0),
        ('scnr',                    True),
        ('photometric_calibration', True),
        ('denoise_acdnr',           True),
        ('denoise_acdnr_k',         4.0),   # conservative: only flat-sky pixels
    ],
    'wide_field': [
        ('deconvolve',              False),
        ('star_reduce',             False),
        ('remove_stars',            False),
        ('galaxy_mode',             False),
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


# ---------------------------------------------------------------------------
# Continuous blend: replaces _classify()-then-lookup with an inverse-distance
# blend across the 8 preset rows above, so a target that's genuinely between
# two types (e.g. a nebula mixing emission and reflection character) gets a
# smooth mix of both types' tuning instead of being forced into one bucket
# with a hard jump in every parameter at the classification boundary.
# ---------------------------------------------------------------------------

# One representative interior point per type, in the same signal space
# _classify() partitions. Not arbitrary: each point satisfies that type's
# own _classify() inequality while failing every higher-priority type's
# inequality too (_classify() is a sequential if/elif chain, so a type's
# true region excludes any earlier type's overlap) -- these are literal
# points inside _classify()'s partition for that type, not threshold edges.
_TYPE_ANCHORS: Dict[str, Dict[str, float]] = {
    # sc>60, pe>15, conc>2.5, dr>150
    'globular_cluster':  {'median_filling': 0.05, 'diffuse_excess': 7.1,
                          'peak_excess': 25.0, 'concentration': 3.5,
                          'star_count': 120.0, 'dynamic_range': 220.0},
    # sc>200 (and not globular: pe kept low)
    'wide_field':        {'median_filling': 0.8, 'diffuse_excess': 5.0,
                          'peak_excess': 5.0, 'concentration': 1.0,
                          'star_count': 350.0, 'dynamic_range': 80.0},
    # pe>10, mf<0.3, sc<30, conc>2.0, dr>60
    'planetary_nebula':  {'median_filling': 0.1, 'diffuse_excess': 5.0,
                          'peak_excess': 15.0, 'concentration': 3.0,
                          'star_count': 10.0, 'dynamic_range': 100.0},
    # sc>80, mf<1 (and not globular: dr/pe kept below its thresholds)
    'star_field':        {'median_filling': 0.5, 'diffuse_excess': 5.3,
                          'peak_excess': 8.0, 'concentration': 1.5,
                          'star_count': 120.0, 'dynamic_range': 100.0},
    # mf>1, sc<100
    'emission_nebula':   {'median_filling': 1.5, 'diffuse_excess': 3.0,
                          'peak_excess': 3.5, 'concentration': 1.17,
                          'star_count': 50.0, 'dynamic_range': 80.0},
    # pe>8, mf<0.5, dr>100 (and not planetary: mf kept >=0.3)
    'galaxy':            {'median_filling': 0.35, 'diffuse_excess': 6.7,
                          'peak_excess': 12.0, 'concentration': 1.8,
                          'star_count': 15.0, 'dynamic_range': 140.0},
    # de>2, sc<100 (and not galaxy/planetary: pe/conc kept at their edges)
    'reflection_nebula': {'median_filling': 0.5, 'diffuse_excess': 3.0,
                          'peak_excess': 6.0, 'concentration': 2.0,
                          'star_count': 40.0, 'dynamic_range': 90.0},
    # fails every rule above -- genuinely weak/ambiguous signal
    'unknown':           {'median_filling': 0.1, 'diffuse_excess': 1.0,
                          'peak_excess': 3.0, 'concentration': 3.0,
                          'star_count': 5.0, 'dynamic_range': 40.0},
}

_BLEND_AXES = ('median_filling', 'diffuse_excess', 'peak_excess',
              'concentration', 'star_count', 'dynamic_range')

# Per-axis normalization so star_count (0-hundreds) doesn't dominate
# concentration/median_filling (0-20ish) in the distance metric purely by
# units -- scaled by each axis's spread across the 8 anchors above.
_AXIS_SCALES: Dict[str, float] = {
    axis: max(
        max(a[axis] for a in _TYPE_ANCHORS.values())
        - min(a[axis] for a in _TYPE_ANCHORS.values()),
        1e-6,
    )
    for axis in _BLEND_AXES
}


def _blend_weights(sig: dict, prior_type: Optional[str] = None,
                   prior_confidence: float = 0.0) -> Dict[str, float]:
    """Continuous replacement for _classify(): an inverse-distance weight
    per type based on how close the measured signals are to that type's
    anchor point, instead of picking exactly one bucket. Normalized to sum
    to 1. A high-confidence prior_type (SIMBAD/header, via
    infer_target_from_metadata) boosts that type's raw weight before
    normalizing -- same "prior wins when confident" behavior
    apply_auto_settings already had (>=0.85 hard override), applied
    continuously instead of a hard switch.
    """
    raw: Dict[str, float] = {}
    for t, anchor in _TYPE_ANCHORS.items():
        d2 = 0.0
        for axis in _BLEND_AXES:
            diff = (sig.get(axis, 0.0) - anchor[axis]) / _AXIS_SCALES[axis]
            d2 += diff * diff
        dist = d2 ** 0.5
        # eps=0.05 caps the weight at a near-exact anchor match instead of
        # diverging to infinity.
        raw[t] = 1.0 / (dist + 0.05) ** 2

    if prior_type in raw and prior_confidence > 0:
        boost = 1.0 + 20.0 * max(prior_confidence, 0.0) ** 2
        raw[prior_type] *= boost

    total = sum(raw.values()) or 1.0
    return {t: w / total for t, w in raw.items()}


def _apply_dynamic_settings(
    sig: dict, weights: Dict[str, float], args
) -> List[str]:
    """Continuous replacement for _apply_target_settings(): every parameter
    is blended across all 8 presets weighted by _blend_weights, instead of
    looking up one bucket's fixed table. Numeric parameters get a weighted
    average (renormalized over just the presets that define that
    parameter); boolean/string parameters -- which can't be fractionally
    blended -- take the value from whichever contributing preset has the
    single highest weight (nearest-neighbor-in-signal-space), so the
    decision still shifts continuously with the signals even though the
    final choice at any instant is binary.
    """
    changes: List[str] = []
    _explicit = getattr(args, '_explicit_cli_dests', set())

    def _set(attr: str, val: object) -> None:
        if attr in _explicit:
            return  # user set this flag explicitly on the CLI — it wins over auto
        old = getattr(args, attr, None)
        if old != val:
            setattr(args, attr, val)
            changes.append(f"{attr}  {old!r} -> {val!r}")

    # Union of every attr any preset defines, first-seen order for a stable,
    # readable change-list.
    all_attrs: List[str] = []
    seen = set()
    for rows in _TARGET_SETTINGS.values():
        for attr, _ in rows:
            if attr not in seen:
                seen.add(attr)
                all_attrs.append(attr)

    for attr in all_attrs:
        contributors = [(t, val) for t, rows in _TARGET_SETTINGS.items()
                        for a, val in rows if a == attr]
        w_sum = sum(weights.get(t, 0.0) for t, _ in contributors)
        if w_sum <= 0:
            continue
        sample_val = contributors[0][1]
        if isinstance(sample_val, (bool, str)):
            best_t, best_val = max(contributors, key=lambda tv: weights.get(tv[0], 0.0))
            _set(attr, best_val)
        else:
            blended = sum(weights.get(t, 0.0) * val for t, val in contributors) / w_sum
            if isinstance(sample_val, int):
                blended = int(round(blended))
            else:
                blended = round(float(blended), 4)
            _set(attr, blended)

    # star_field's poor-seeing deconvolve exception: previously gated on
    # target_type == 'star_field'; now a direct threshold on how
    # star-field-like the blend is, weighted rather than a bucket check.
    if weights.get('star_field', 0.0) > 0.3 and sig['fwhm'] > 4.0:
        _set('deconvolve', True)
        _set('deconvolve_iterations', 10)

    # Galaxy targets skip the sky-residual correction passes entirely
    # (--skip-step sky_residual): unlike DBE (which honors --galaxy-mode's
    # fitted exclusion ellipse), remove_sky_residual's own extended-source
    # detection is a much stricter, cruder fallback -- even with the ellipse
    # now also threaded through to it, real (patchy, irregular) galaxy arms
    # extending past a symmetric ellipse fit, or spanning a mesh cell only
    # partially, still get partially fit away as "background" across 3
    # residual passes. Confirmed on real data: skipping the step entirely
    # measurably improves output quality for a galaxy -- DBE's own single
    # protected pass already does the main background-flattening job for
    # this target type, so the extra passes are net-negative here even
    # though they help other targets. `skip_step` is a plain list (not
    # blendable/settable via _set() above), and explicit-dest opt-out is
    # honored manually since a straight append can't reuse _set()'s
    # overwrite-if-different equality check.
    current_skip = list(getattr(args, 'skip_step', None) or [])
    if (weights.get('galaxy', 0.0) > 0.3
            and 'skip_step' not in _explicit
            and 'sky_residual' not in current_skip):
        current_skip.append('sky_residual')
        setattr(args, 'skip_step', current_skip)
        changes.append("skip_step  += 'sky_residual' (galaxy target)")

    return changes


# ---------------------------------------------------------------------------
# Quality-based adjustments (always applied, independent of target type)
# ---------------------------------------------------------------------------

def _apply_quality_settings(
    sig: dict, args, target_type: str = 'unknown',
    weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Apply frame-count, SNR, FWHM, Strehl, and dispersion adjustments."""
    changes: List[str] = []
    _explicit = getattr(args, '_explicit_cli_dests', set())

    def _set(attr: str, val: object) -> None:
        if attr in _explicit:
            return  # user set this flag explicitly on the CLI — it wins over auto
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
        if n < 15:
            # percentile: simple, robust for small-to-moderate N (also
            # subsumes what used to be a separate 'trimmed_mean' method --
            # same reject-tails-then-average operation).
            _set('stack_method', 'percentile')
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

    # 5. MMT strength scaled to SNR
    if getattr(args, 'denoise_mmt', False) and snr > 0:
        if snr < 5:
            _set('denoise_mmt_strength', 4.0)
        elif snr < 10:
            _set('denoise_mmt_strength', 3.5)
        elif snr > 20:
            _set('denoise_mmt_strength', 2.0)

    # 6. Baseline ACDNR for unknown/noisy targets with no other luma denoiser
    #    (rule 14 enforces a single primary; don't add one just to remove it).
    if (snr < 5 and snr > 0
            and not getattr(args, 'denoise_acdnr', False)
            and not getattr(args, 'denoise_mmt', False)
            and not getattr(args, 'denoise', False)
            and not getattr(args, 'denoise_bm3d', False)):
        _set('denoise_acdnr', True)

    # 7. BM3D: enable when the bm3d package is installed and the stack has enough
    #    SNR for block-matching to outperform median-based methods.  Thresholds are
    #    lower when the package is available (hardware-accelerated C backend) vs
    #    absent (pure-scipy DCT fallback which is ~5-10x slower and less effective).
    #    BM3D-suitable-ness is now a weighted sum over these 4 types instead of
    #    exact bucket membership -- a session that's mostly (but not entirely)
    #    one of these types can still qualify.
    _bm3d_targets = ('galaxy', 'globular_cluster', 'emission_nebula', 'reflection_nebula')
    _bm3d_snr_min = 8 if HAS_BM3D_PKG else 12
    _bm3d_n_min   = 12 if HAS_BM3D_PKG else 20
    _w = weights or {}
    _bm3d_weight = sum(_w.get(t, 0.0) for t in _bm3d_targets)
    if (_bm3d_weight > 0.5
            and snr >= _bm3d_snr_min
            and n >= _bm3d_n_min
            and not getattr(args, 'denoise_bm3d', False)):
        _set('denoise_bm3d', True)
        if _w.get('globular_cluster', 0.0) > 0.5 or snr > 20:
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

    # 13. Preview black point vs integration depth.
    #     preview_black_sigma clips the sky-noise tail to black in the preview
    #     JPEG. On deep stacks the sky noise is well averaged and the clip
    #     yields clean black sky; on shallow stacks, residual mid-scale
    #     structure (a few ADU at 20-60px scales) modulates which pixels cross
    #     the clip threshold and renders as soft black splotches. Soften the
    #     clip when the stack is shallow — only ever reducing the preset value,
    #     so lower user/preset choices pass through untouched.
    pbs = float(getattr(args, 'preview_black_sigma', 0.0) or 0.0)
    if pbs > 1.0:
        if n < 20:
            setattr(args, '_preview_black_sigma_precap', pbs)
            _set('preview_black_sigma', 1.0)
        elif n < 60 and pbs > 2.0:
            # Stash the preset value so --merge can restore it when the
            # merged frame count justifies the deeper-stack clip again.
            setattr(args, '_preview_black_sigma_precap', pbs)
            _set('preview_black_sigma', 2.0)

    # 14. Single primary luma denoiser. Target presets and the SNR rules
    #     above can each enable a denoiser; layering several full-frame
    #     smoothers (wavelet + MMT + curvelet + ACDNR + BM3D) compounds
    #     smoothing — each pass erodes the faint structure the previous one
    #     preserved — without adding selectivity, and pays for every pass.
    #     Precedence: BM3D (enabled only when its SNR/frame-count
    #     conditions hold) > MMT (robust to the non-Gaussian residual noise
    #     of stacked OSC data) > curvelet (directional/coherence-adaptive
    #     wavelet -- ranked below MMT since MMT's median cascade is the
    #     more battle-tested default when both would otherwise apply) >
    #     wavelet > ACDNR (fallback sky smoother). Chroma-only cleanup
    #     (chroma_nr, SCNR) is orthogonal and unaffected; explicit extras
    #     like --denoise-aniso/-nlm/-bilateral are user intent and also
    #     left alone.
    if getattr(args, 'denoise_bm3d', False):
        for attr in ('denoise_mmt', 'denoise_curvelet', 'denoise', 'denoise_acdnr'):
            if getattr(args, attr, False):
                _set(attr, False)
    elif getattr(args, 'denoise_mmt', False):
        for attr in ('denoise_curvelet', 'denoise', 'denoise_acdnr'):
            if getattr(args, attr, False):
                _set(attr, False)
    elif getattr(args, 'denoise_curvelet', False):
        for attr in ('denoise', 'denoise_acdnr'):
            if getattr(args, attr, False):
                _set(attr, False)
    elif getattr(args, 'denoise', False):
        if getattr(args, 'denoise_acdnr', False):
            _set('denoise_acdnr', False)

    # 15. Variance-stabilize the luma plane (generalized Anscombe transform)
    #     ahead of whichever wavelet-family denoiser ended up primary --
    #     wavelet and curvelet both threshold via a single per-subband
    #     noise estimate that's only strictly valid under uniform Gaussian
    #     noise; the transform makes that assumption closer to true
    #     everywhere in the frame, not just near the sky background level.
    #     A closer match to the true noise model, not a target-type
    #     tradeoff, so applied whenever one of those two denoisers is
    #     active rather than blended per preset.
    if getattr(args, 'denoise', False) or getattr(args, 'denoise_curvelet', False):
        if not getattr(args, 'variance_stabilize', False):
            _set('variance_stabilize', True)

    # 16. Prefer the ringing-free Magic Kernel over the default Lanczos-3
    #     drizzle resample kernel once drizzling is actually active --
    #     Lanczos-3's negative sidelobes are a real, visible artifact on
    #     bright star cores; the Magic Kernel is softer by construction but
    #     provably non-negative. Only when the PSF-matched kernel wasn't
    #     already explicitly requested.
    if (float(getattr(args, 'drizzle_scale', 1.0) or 1.0) > 1.0
            and getattr(args, 'drizzle_kernel', 'lanczos3') == 'lanczos3'):
        _set('drizzle_kernel', 'magic')

    # 17. Prefer Mertens exposure fusion over the original sigmoid-threshold
    #     HDR blend once the user has supplied a short-exposure stack to
    #     blend in -- avoids the seam a hard/sigmoid threshold can leave at
    #     the transition band. --auto can't conjure the short stack itself
    #     (--hdr-combine still needs an explicit path), only upgrade the
    #     blend method once one's been given.
    if (getattr(args, 'hdr_combine', None)
            and getattr(args, 'hdr_blend_mode', 'threshold') == 'threshold'):
        _set('hdr_blend_mode', 'fusion')

    # 18. Prefer spectrophotometric (blackbody-Teff) colour calibration over
    #     the fixed colour-index formula once the user has enabled
    #     --color-calibrate -- same "auto can upgrade a method, not turn on
    #     a feature that needs its own prerequisite (--plate-solve)" shape
    #     as rule 17.
    if (getattr(args, 'color_calibrate', False)
            and getattr(args, 'color_calibrate_method', 'colorindex') == 'colorindex'):
        _set('color_calibrate_method', 'spcc')

    return changes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_auto_settings(
    final: list,
    args,
    prior_type: Optional[str] = None,
    prior_confidence: float = 0.0,
) -> Tuple[str, str, dict, List[str], Dict[str, float]]:
    """Classify the target (for display) and apply continuously-blended
    heuristic settings to args in-place (see _blend_weights /
    _apply_dynamic_settings module docs above _TYPE_ANCHORS).

    Parameters
    ----------
    final : list[FrameInfo]
        Accepted frames after Phase 1, each with .metrics populated.
    args : argparse.Namespace
        Modified in-place.
    prior_type : str or None
        Object type inferred from metadata (folder/header/Simbad). Used both
        for the displayed label (>=0.85 confidence overrides the
        pixel-signal classifier outright, >=0.70 wins when the heuristic
        returns 'unknown' -- unchanged from before) and to boost that type's
        weight in the continuous blend that actually drives settings.
    prior_confidence : float
        Confidence of *prior_type* in the range 0-1.

    Returns
    -------
    target_type : str
        Internal type key (e.g. ``'emission_nebula'``) -- the single
        best-match label for display; NOT what drives settings anymore
        (see ``weights``).
    label : str
        Human-readable label (e.g. ``'Emission Nebula'``).
    signals : dict
        Classification signals, useful for diagnostics.
    changes : list[str]
        Every setting that was changed, as ``"attr  old -> new"`` strings.
    weights : dict[str, float]
        The continuous blend weight per type (sums to 1) that actually
        produced ``changes`` -- e.g. ``{'emission_nebula': 0.72,
        'reflection_nebula': 0.21, ...}``. Useful for a "72% Emission
        Nebula, 21% Reflection Nebula" style diagnostic print.
    """
    if not final:
        return 'unknown', TARGET_LABELS['unknown'], {}, [], {}

    agg = _aggregate(final)
    sig = _compute_signals(agg)
    heuristic_type = _classify(sig)

    # Resolve the DISPLAYED classification (unchanged from before this
    # module went continuous -- this is a label, not a settings lookup key).
    valid_prior = (prior_type and prior_type in TARGET_LABELS
                   and prior_type != 'unknown')
    if valid_prior and prior_confidence >= 0.85:
        target_type = prior_type
    elif valid_prior and prior_confidence >= 0.70 and heuristic_type == 'unknown':
        target_type = prior_type
    else:
        target_type = heuristic_type

    label = TARGET_LABELS[target_type]

    weights = _blend_weights(sig, prior_type=prior_type, prior_confidence=prior_confidence)

    changes: List[str] = []
    changes.extend(_apply_dynamic_settings(sig, weights, args))
    changes.extend(_apply_quality_settings(sig, args, target_type=target_type, weights=weights))

    return target_type, label, sig, changes, weights
