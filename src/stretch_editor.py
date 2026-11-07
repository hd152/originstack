"""Numbers behind the interactive stretch editor (desktop app, Qt window).

Everything here is plain numpy and testable without a window: which controls a stretch
uses, their ranges, the histogram of the linear stack in sky-sigma units, the tone curve a
rendering applied, and the conversions between a marker on the histogram and a stretch
setting (black point in sigma, white point as a luminance percentile).

The editor renders a point-sampled reduction of the final image (every f-th pixel, not an
average): the stretch takes its black and white points from the noise and the brightest
percentiles, and averaging would shrink both, so the reduction must keep the full image's
statistics for the preview to match what *Apply* writes.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Settings the editor changes, in the order the panel lists them.
PARAM_KEYS = ('ghs_b', 'ghs_sp', 'ghs_hp', 'black_sigma', 'white_pct')
# Keys of the stretch kwargs the user can change (the editor's, plus which stretch and colour mode).
STRETCH_KEYS = ('stretch', 'color', 'nebula_auto') + PARAM_KEYS

# name: (minimum, maximum, step, decimals, label); labels are translated where shown.
PARAM_RANGES: Dict[str, Tuple[float, float, float, int]] = {
    'ghs_b': (0.0, 25.0, 0.1, 1),
    'ghs_sp': (0.0, 0.6, 0.005, 3),
    'ghs_hp': (0.5, 1.0, 0.005, 3),
    'black_sigma': (-5.0, 10.0, 0.1, 1),
    'white_pct': (95.0, 99.99, 0.01, 2),
}
PARAM_DEFAULTS = {'ghs_b': 8.0, 'ghs_sp': 0.15, 'ghs_hp': 0.95, 'black_sigma': 0.0, 'white_pct': 99.5,
                  'nebula_auto': True}

GHS_FAMILY = ('ghs', 'masked', 'multiscale', 'oklab', 'clahe', 'object')

# Which settings each stretch reads (io_fits.render_preview_float). 'noisefloor' solves its
# own strength and black point; 'seti' and 'linear' have no strength either.
_CONTROLS = {
    'arcsinh': ('black_sigma', 'white_pct'),
    'noisefloor': ('ghs_hp', 'white_pct'),
    'seti': ('white_pct',),
    'linear': (),
}


def controls_for(stretch: str) -> Tuple[str, ...]:
    """The settings the ``stretch`` responds to (the others would change nothing)."""
    if stretch in GHS_FAMILY:
        return PARAM_KEYS
    return _CONTROLS.get(stretch, ())


def clamp_params(params: Dict[str, float]) -> Dict[str, float]:
    """Only known settings, each held inside its range (``nebula_auto`` is a switch)."""
    out = {}
    for k, v in params.items():
        if k == 'nebula_auto':
            out[k] = bool(v)
        elif k in PARAM_RANGES:
            lo, hi, _step, _dec = PARAM_RANGES[k]
            out[k] = float(min(hi, max(lo, float(v))))
    return out


# ── histogram of the linear image, in sky sigmas ──────────────────────

AXIS_SCALE = 3.0      # asinh(x / AXIS_SCALE): linear near the sky peak, logarithmic far above it
AXIS_MIN_SIGMA = -6.0


def to_axis(x_sigma):
    """Sigmas above the sky -> position on the histogram's axis (squeezes the long bright tail)."""
    return np.arcsinh(np.asarray(x_sigma, dtype=np.float64) / AXIS_SCALE)


def from_axis(u):
    return np.sinh(np.asarray(u, dtype=np.float64)) * AXIS_SCALE


def sky_of(lum: np.ndarray, noise_scale: float = 1.0) -> Tuple[float, float]:
    """(median, raw sigma) of the sky in ``lum``: what the stretch measures."""
    from src.io_fits import _sky_stats
    med, scaled = _sky_stats(lum, noise_scale)
    return med, scaled / float(noise_scale)


def subsample(rgb: np.ndarray, max_dim: int) -> np.ndarray:
    """Every f-th pixel so the long side is at most ``max_dim`` (see the module note on why
    not an average). Contiguous copy; the array itself when no reduction is needed."""
    a = np.asarray(rgb)
    f = -(-max(a.shape[0], a.shape[1]) // max(1, int(max_dim)))
    if f <= 1:
        return a
    return np.ascontiguousarray(a[::f, ::f])


def linear_histogram(rgb: np.ndarray, lum: np.ndarray, med: float, sigma: float,
                     bins: int = 192) -> dict:
    """Counts of the red, green, blue and luminance values of the linear image on the axis
    of ``to_axis`` (sky sigmas above the median): ``{'edges': (bins+1,) axis positions,
    'counts': (4, bins) as log1p of the counts, scaled so the tallest is 1, 'linear': the same
    on a linear scale}``. The range runs
    from a few sigma below the sky to the brightest pixel."""
    sig = sigma if sigma > 0 else 1.0
    lo = to_axis(AXIS_MIN_SIGMA)
    hi = float(to_axis((float(np.max(lum)) - med) / sig))
    hi = max(hi, float(lo) + 1.0)
    edges = np.linspace(lo, hi, bins + 1)
    counts = np.zeros((4, bins))
    planes = (rgb[..., 0], rgb[..., 1], rgb[..., 2], lum)
    for i, plane in enumerate(planes):
        u = to_axis((np.asarray(plane, dtype=np.float64).ravel() - med) / sig)
        counts[i] = np.histogram(u, bins=edges)[0]
    lin_peak = counts.max()
    linear = counts / lin_peak if lin_peak > 0 else counts
    counts = np.log1p(counts)
    peak = counts.max()
    if peak > 0:
        counts /= peak
    return {'edges': edges, 'counts': counts, 'linear': linear, 'med': med, 'sigma': sig}


def tone_curve(lum_in: np.ndarray, lum_out: np.ndarray, med: float, sigma: float,
               bins: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """What the stretch did, as the picture shows it: the median output luminance (0..1) for
    pixels of each input level, input on the histogram's axis. Works for any stretch and
    colour mode, because it reads the rendering, not a formula. Returns (axis positions of
    the bin centres, output) for the bins that hold pixels."""
    sig = sigma if sigma > 0 else 1.0
    u = to_axis((np.asarray(lum_in, dtype=np.float64).ravel() - med) / sig)
    out = np.asarray(lum_out, dtype=np.float64).ravel()
    lo, hi = float(to_axis(AXIS_MIN_SIGMA)), float(u.max())
    edges = np.linspace(lo, max(hi, lo + 1.0), bins + 1)
    idx = np.clip(np.digitize(u, edges) - 1, 0, bins - 1)
    order = np.argsort(idx, kind='stable')
    idx_sorted, out_sorted = idx[order], out[order]
    starts = np.searchsorted(idx_sorted, np.arange(bins))
    ends = np.searchsorted(idx_sorted, np.arange(bins), side='right')
    xs, ys = [], []
    for b in range(bins):
        if ends[b] > starts[b]:
            xs.append(0.5 * (edges[b] + edges[b + 1]))
            ys.append(float(np.median(out_sorted[starts[b]:ends[b]])))
    return np.asarray(xs), np.asarray(ys)


# ── markers <-> settings ──────────────────────────────────────────────


def white_value(lum: np.ndarray, med: float, sigma: float, white_pct: float,
                black_sigma: float) -> float:
    """The luminance that becomes white for ``white_pct`` (the stretch's own rule, including
    its floor above the sky)."""
    from src.io_fits import _preview_white
    black = med + black_sigma * sigma
    white, _sp = _preview_white(lum, med, sigma, float(white_pct), black)
    return float(white)


def marker_axis(value: float, med: float, sigma: float) -> float:
    """Axis position of a linear value."""
    return float(to_axis((value - med) / (sigma if sigma > 0 else 1.0)))


def black_sigma_at(u: float) -> float:
    """Black point (sigmas above the sky median) for a marker dragged to axis position ``u``."""
    lo, hi, _s, _d = PARAM_RANGES['black_sigma']
    return float(min(hi, max(lo, float(from_axis(u)))))


def white_pct_at(u: float, lum: np.ndarray, med: float, sigma: float) -> float:
    """White-point percentile for a marker dragged to axis position ``u``: the share of pixels
    darker than that level."""
    value = med + float(from_axis(u)) * (sigma if sigma > 0 else 1.0)
    pct = 100.0 * float(np.mean(np.asarray(lum) < value))
    lo, hi, _s, _d = PARAM_RANGES['white_pct']
    return float(min(hi, max(lo, pct)))


def kw_for_render(kw: dict, params: Optional[Dict[str, float]] = None,
                  stretch: Optional[str] = None, color: Optional[str] = None) -> dict:
    """``kw`` (a ``preview_stretch_kw``) with the editor's changes applied; settings the
    stretch does not use keep their values."""
    out = dict(kw)
    if stretch:
        out['stretch'] = stretch
    if color:
        out['color'] = color
    if params:
        out.update(clamp_params(params))
    out.setdefault('white_pct', PARAM_DEFAULTS['white_pct'])
    return out
