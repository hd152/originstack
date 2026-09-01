"""Channel combination for LRGB and narrowband imaging.

Combines separately stacked FITS files into a colour image:
  - LRGB:  luminance-layering (L replaces lightness, RGB provides colour)
  - SHO:   Hubble palette  — SII→R, Ha→G, OIII→B
  - HOO:   Ha→R, OIII→G, OIII→B
  - HOS:   Ha→R, OIII→G, SII→B
  - HSO:   Ha→R, SII→G, OIII→B   (uncommon but supported)
  - Custom: any per-channel mapping via --mapping R=<file> G=<file> B=<file>

CLI entry point: ``python originstack.py combine ...``
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np

try:
    import astro_native as _native
    _HAS_NATIVE = True
except Exception:
    _native = None
    _HAS_NATIVE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_channel(path: str) -> np.ndarray:
    """Load a FITS file and return a 2-D float32 luminance array.

    Accepts:
      * (H, W)      mono
      * (3, H, W)   colour cube  → converted to luminance
      * (H, W, 3)   HWC colour   → converted to luminance
    """
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)

    if data.ndim == 2:
        return data
    if data.ndim == 3:
        if data.shape[0] == 3:          # (3, H, W) — FITS convention
            return (0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2])
        if data.shape[2] == 3:          # (H, W, 3)
            return (0.299 * data[:, :, 0] + 0.587 * data[:, :, 1]
                    + 0.114 * data[:, :, 2])
    raise ValueError(f"Unsupported FITS shape {data.shape} in {path}")


def _load_rgb(path: str) -> np.ndarray:
    """Load a FITS file and return a (H, W, 3) float32 RGB array."""
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)

    if data.ndim == 2:
        return np.stack([data, data, data], axis=2)
    if data.ndim == 3:
        if data.shape[0] == 3:
            return np.transpose(data, (1, 2, 0))
        if data.shape[2] == 3:
            return data
    raise ValueError(f"Unsupported FITS shape {data.shape} in {path}")


def _resize_to(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize arr to (H, W) using bilinear zoom if needed."""
    if arr.shape[0] == H and arr.shape[1] == W:
        return arr
    from scipy.ndimage import zoom
    zy = H / arr.shape[0]
    zx = W / arr.shape[1]
    if arr.ndim == 2:
        return zoom(arr, (zy, zx), order=1).astype(np.float32)
    return zoom(arr, (zy, zx, 1), order=1).astype(np.float32)


def _normalise(arr: np.ndarray, pct_lo: float = 0.1, pct_hi: float = 99.9) -> np.ndarray:
    """Stretch array to roughly [0, 1] using robust percentile clipping."""
    lo = float(np.percentile(arr, pct_lo))
    hi = float(np.percentile(arr, pct_hi))
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Public combination functions
# ---------------------------------------------------------------------------

def lrgb_combine(L: np.ndarray, R: np.ndarray,
                 G: np.ndarray, B: np.ndarray,
                 saturation_boost: float = 1.0) -> np.ndarray:
    """Combine luminance + colour channels via luminance-layering.

    The L frame supplies the lightness of every pixel; R/G/B supply the hue
    and saturation.  This is the standard LRGB workflow used by PixInsight's
    LRGBCombination process.

    Algorithm:
      1. Normalise each input to [0, 1].
      2. Compute the colour luminance:  L_rgb = 0.299R + 0.587G + 0.114B
      3. Per pixel, scale R/G/B by  L / max(L_rgb, ε)  to transfer the L
         channel's luminance onto the colour image while preserving hue.
      4. Optional saturation boost applied in a hue-preserving way.

    Args:
        L:  Luminance frame (H, W) or (H, W, 1).
        R, G, B:  Colour channel frames (H, W).
        saturation_boost: Multiply chroma by this factor (1.0 = no change).

    Returns:
        (H, W, 3) float32 RGB array in [0, 1].
    """
    L = _normalise(np.squeeze(L))
    R = _normalise(R)
    G = _normalise(G)
    B = _normalise(B)

    H, W = L.shape
    R = _resize_to(R, H, W)
    G = _resize_to(G, H, W)
    B = _resize_to(B, H, W)

    L_rgb = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.float32)
    eps = 1e-7
    scale = L / np.maximum(L_rgb, eps)

    R_out = np.clip(R * scale, 0.0, 1.0)
    G_out = np.clip(G * scale, 0.0, 1.0)
    B_out = np.clip(B * scale, 0.0, 1.0)

    if saturation_boost != 1.0:
        L_out = (0.299 * R_out + 0.587 * G_out + 0.114 * B_out)
        R_out = np.clip(L_out + saturation_boost * (R_out - L_out), 0.0, 1.0)
        G_out = np.clip(L_out + saturation_boost * (G_out - L_out), 0.0, 1.0)
        B_out = np.clip(L_out + saturation_boost * (B_out - L_out), 0.0, 1.0)

    return np.stack([R_out, G_out, B_out], axis=2)


def _continuum_scale_moments_numpy(a: np.ndarray, b: np.ndarray) -> tuple:
    """Numpy fallback / reference for the fused single-pass moment
    computation ``optimal_continuum_scale`` needs -- see its docstring.
    Returns ``(n, mu20, mu11, mu02, mu30, mu21, mu12, mu03)``, the sample
    size and the central second/third-order (co)moments of ``a``/``b``.
    """
    n = a.size
    mean_a, mean_b = float(a.mean()), float(b.mean())
    ap, bp = a - mean_a, b - mean_b
    mu20 = float(np.mean(ap * ap))
    mu11 = float(np.mean(ap * bp))
    mu02 = float(np.mean(bp * bp))
    mu30 = float(np.mean(ap * ap * ap))
    mu21 = float(np.mean(ap * ap * bp))
    mu12 = float(np.mean(ap * bp * bp))
    mu03 = float(np.mean(bp * bp * bp))
    return (n, mu20, mu11, mu02, mu30, mu21, mu12, mu03)


def _continuum_scale_moments(a: np.ndarray, b: np.ndarray) -> tuple:
    """Dispatch to the native single-pass kernel (fuses what the numpy
    fallback computes as ~9 separate full-array passes/temporaries --
    ap*bp, ap*ap*bp, ap*bp*bp, etc. -- into one), falling back to numpy on
    any failure (module absent, bad input)."""
    if _HAS_NATIVE:
        try:
            return _native.continuum_scale_moments(
                np.ascontiguousarray(a, dtype=np.float64),
                np.ascontiguousarray(b, dtype=np.float64))
        except Exception:
            pass
    return _continuum_scale_moments_numpy(a, b)


def _skewness_from_moments(moments: tuple, scales: np.ndarray) -> np.ndarray:
    """Evaluate the bias-corrected skewness of ``narrowband - s*continuum``
    at every ``s`` in ``scales`` from the 7 central moments
    ``_continuum_scale_moments`` returns -- a closed-form polynomial in
    ``s`` (see ``optimal_continuum_scale``'s docstring for the derivation),
    not a re-scan of the pixel data per scale.
    """
    n, mu20, mu11, mu02, mu30, mu21, mu12, mu03 = moments
    if n <= 2:
        return np.full(scales.shape, np.nan)
    s = scales
    m2 = mu20 - 2.0 * s * mu11 + s ** 2 * mu02
    m3 = mu30 - 3.0 * s * mu21 + 3.0 * s ** 2 * mu12 - s ** 3 * mu03
    with np.errstate(invalid='ignore', divide='ignore'):
        g1 = m3 / np.power(np.maximum(m2, 0.0), 1.5)
        bias_correction = np.sqrt(n * (n - 1)) / (n - 2)
        g1_corrected = bias_correction * g1
    # m2 <= 0 means a degenerate/near-constant residual at that scale --
    # skewness is undefined there, matching scipy.stats.skew's own NaN.
    return np.where(m2 > 1e-300, g1_corrected, np.nan)


def optimal_continuum_scale(narrowband: np.ndarray, continuum: np.ndarray,
                            scale_range: Tuple[float, float] = (0.0, 3.0),
                            n_steps: int = 60,
                            sky_percentile: float = 50.0):
    """Quantitative narrowband continuum-subtraction scale factor: sweep
    candidate scale factors, compute the residual (``narrowband -
    scale*continuum``) pixel skewness restricted to background-dominated
    pixels, and find the scale that **maximises** that skewness.

    Verified against synthetic ground truth (star field shared between both
    images at a known scale, plus a smooth positive-only "emission" signal
    only the narrowband image has -- see ``tests/test_continuum_subtraction
    .py``): skewness peaks sharply at the true subtraction scale, not at a
    zero-crossing. At the correct scale the shared star-continuum signal
    (whose over/under-subtraction residual is small and roughly symmetric
    around each star) cancels out of the selected background-ish pixels,
    leaving only the genuinely one-sided (positive-only) emission structure
    to dominate the skewness measurement; any other scale mixes in a
    partially-cancelled star-residual term that dilutes that one-sidedness
    in both directions. (Named after, and in the same spirit as, published
    "skewness transition" methods for this problem -- this implementation
    was derived and validated independently against synthetic ground truth
    rather than reproduced from a specific paper's exact algorithm.)

    Restricted to pixels at or below the narrowband image's own
    ``sky_percentile`` (default: median) so bright saturated cores/hot
    pixels -- which have nothing to do with continuum matching -- don't
    dominate the skewness estimate.

    Sub-step precision comes from a parabolic fit to the three skewness
    samples around the coarse-grid peak (the same refinement idiom this
    codebase's own FFT registration residual uses), not just the raw grid
    step.

    Implementation note: ``residual(s) = narrowband - s*continuum`` is
    *linear* in ``s``, so its skewness at every candidate scale is a fixed
    rational function of just 7 scalar central moments of the (masked)
    ``(narrowband, continuum)`` pair -- computed once, not once per scale.
    ``_continuum_scale_moments`` (native Rust with a numpy fallback) does
    that single pass; ``_skewness_from_moments`` evaluates the resulting
    polynomial-in-``s`` skewness formula at every swept scale for
    approximately free. This is an algorithmic win (O(pixels) once instead
    of O(pixels x n_steps)), not just a faster loop -- verified to
    reproduce ``scipy.stats.skew(residual, bias=False)`` exactly (to
    float64 precision) for a directly-computed residual at several scales
    before replacing the original per-scale loop.

    Returns ``(best_scale, diagnostics)`` where diagnostics carries the
    swept ``scales``/``skewness`` arrays.
    """
    nb = np.asarray(narrowband, dtype=np.float64)
    cont = np.asarray(continuum, dtype=np.float64)
    if nb.shape != cont.shape:
        raise ValueError("narrowband and continuum must share the same shape")

    thresh = np.percentile(nb, sky_percentile)
    mask = nb <= thresh
    a = np.ascontiguousarray(nb[mask])
    b = np.ascontiguousarray(cont[mask])

    scales = np.linspace(scale_range[0], scale_range[1], n_steps)
    if a.size > 10:
        moments = _continuum_scale_moments(a, b)
        skews = _skewness_from_moments(moments, scales)
    else:
        skews = np.full(n_steps, np.nan)

    valid = np.isfinite(skews)
    if valid.sum() < 2:
        mid = float(np.mean(scale_range))
        return mid, {'scales': scales, 'skewness': skews}

    scales_v, skews_v = scales[valid], skews[valid]
    best_idx = int(np.argmax(skews_v))
    if 0 < best_idx < len(scales_v) - 1:
        y0, y1, y2 = skews_v[best_idx - 1], skews_v[best_idx], skews_v[best_idx + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0)) if abs(denom) > 1e-12 else 0.0
        step = scales_v[best_idx + 1] - scales_v[best_idx]
        best_scale = float(scales_v[best_idx] + delta * step)
    else:
        best_scale = float(scales_v[best_idx])

    return float(best_scale), {'scales': scales, 'skewness': skews}


def subtract_continuum(narrowband: np.ndarray, continuum: np.ndarray,
                       scale: Optional[float] = None) -> Tuple[np.ndarray, float, dict]:
    """Subtract a scaled continuum reference from a narrowband image.
    ``scale=None`` (default) fits it via ``optimal_continuum_scale``.

    Returns ``(result, scale_used, diagnostics)``; ``result`` is clipped at
    zero (continuum subtraction shouldn't drive real signal negative;
    over-subtracted sky noise clipping to zero is the expected, harmless
    failure mode of picking too-large a scale).
    """
    diagnostics: dict = {}
    if scale is None:
        scale, diagnostics = optimal_continuum_scale(narrowband, continuum)
    result = np.clip(narrowband.astype(np.float64) - scale * continuum.astype(np.float64),
                     0.0, None)
    return result.astype(np.float32), float(scale), diagnostics


# Narrowband palette mappings: mode → (R_src, G_src, B_src)
# Each value is the key name as passed in the kwargs dict
_NARROWBAND_MODES = {
    "sho": ("sii", "ha",   "oiii"),   # Hubble palette
    "hoo": ("ha",  "oiii", "oiii"),
    "hos": ("ha",  "oiii", "sii"),
    "hso": ("ha",  "sii",  "oiii"),
    "oho": ("oiii","ha",   "oiii"),
    "shs": ("sii", "ha",   "sii"),
}


def narrowband_combine(mode: str = "sho",
                       ha: Optional[np.ndarray] = None,
                       oiii: Optional[np.ndarray] = None,
                       sii: Optional[np.ndarray] = None) -> np.ndarray:
    """Map narrowband channels to an RGB image.

    Args:
        mode: One of 'sho', 'hoo', 'hos', 'hso'.
        ha:   Hα frame (H, W).
        oiii: OIII frame (H, W).
        sii:  SII frame (H, W).  May be None for modes that don't use it.

    Returns:
        (H, W, 3) float32 RGB in [0, 1].
    """
    mode = mode.lower()
    if mode not in _NARROWBAND_MODES:
        raise ValueError(f"Unknown narrowband mode '{mode}'. "
                         f"Choose from: {', '.join(_NARROWBAND_MODES)}")

    channels = {"ha": ha, "oiii": oiii, "sii": sii}
    r_key, g_key, b_key = _NARROWBAND_MODES[mode]

    for key in (r_key, g_key, b_key):
        if channels[key] is None:
            raise ValueError(
                f"Mode '{mode}' requires '{key}' channel but it was not provided."
            )

    R = _normalise(channels[r_key])
    G = _normalise(channels[g_key])
    B = _normalise(channels[b_key])

    # Ensure consistent shape
    H, W = R.shape
    G = _resize_to(G, H, W)
    B = _resize_to(B, H, W)

    return np.stack([R, G, B], axis=2).astype(np.float32)


def scnr_green(rgb: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Subtractive chromatic noise reduction on green (average-neutral).

    The SHO/Hubble palette maps Hα (the brightest line) to green, leaving a
    heavy green cast over the whole field. SCNR clips green to at most the
    average of red and blue: ``G = G - amount * max(0, G - (R+B)/2)`` — the
    standard PixInsight "average neutral" protection. Applied to the RGB image
    in [0, 1]."""
    out = rgb.astype(np.float32, copy=True)
    r, g, b = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    neutral = 0.5 * (r + b)
    excess = np.maximum(0.0, g - neutral)
    out[:, :, 1] = g - float(np.clip(amount, 0.0, 1.0)) * excess
    return np.clip(out, 0.0, 1.0)


def _peak_sources(lum: np.ndarray, k: float = 5.0):
    """Numpy fallback star detector: local maxima above median + k*MAD.
    Returns a structured array with xcentroid/ycentroid/flux, or None."""
    try:
        from scipy.ndimage import maximum_filter
    except Exception:
        return None
    med = float(np.median(lum))
    sig = 1.4826 * float(np.median(np.abs(lum - med)))
    if sig <= 0:
        sig = float(np.std(lum)) or 1e-6
    thresh = med + k * sig
    peaks = (lum >= maximum_filter(lum, size=5)) & (lum > thresh)
    ys, xs = np.nonzero(peaks)
    if ys.size == 0:
        return None
    dt = np.dtype([('xcentroid', np.float64), ('ycentroid', np.float64),
                   ('flux', np.float64)])
    out = np.zeros(ys.size, dtype=dt)
    out['xcentroid'] = xs
    out['ycentroid'] = ys
    out['flux'] = lum[ys, xs]
    return out


def _star_mask_for(rgb: np.ndarray, fwhm: float = 3.5) -> Optional[np.ndarray]:
    """Detect stars on the RGB luminance and return a soft [0,1] star mask.
    Uses the matched-filter detector (src/star_detect.py) when it finds
    sources, else a crude numpy local-maxima fallback."""
    try:
        from src.quality import detect_stars_auto, generate_star_mask
    except Exception:
        return None
    lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
           + 0.114 * rgb[:, :, 2]).astype(np.float32)
    med = float(np.median(lum))
    noise = 1.4826 * float(np.median(np.abs(lum - med))) or 1e-3
    try:
        src = detect_stars_auto(lum, noise, background=med)
    except Exception:
        src = None
    if src is None or len(src) == 0:
        src = _peak_sources(lum)
    if src is None or len(src) == 0:
        return None
    return generate_star_mask(lum.shape, src, fwhm=fwhm)


def fix_narrowband_stars(rgb: np.ndarray, mode: str = "desaturate",
                         rgb_stars: Optional[np.ndarray] = None,
                         strength: float = 1.0) -> np.ndarray:
    """Fix the magenta/technicolour stars typical of SHO narrowband combines.

    ``mode='desaturate'``: blend star pixels toward their own luminance
    (removes the colour cast, keeps brightness). ``mode='rgb'``: transplant the
    *chrominance* of a real broadband RGB-star image (``rgb_stars``) onto the
    narrowband star luminance, so stars show natural colours while nebulosity
    keeps the narrowband palette. Only star-core pixels are altered."""
    mask = _star_mask_for(rgb)
    if mask is None:
        return rgb
    m = np.clip(mask, 0.0, 1.0)[:, :, np.newaxis] * float(np.clip(strength, 0.0, 1.0))
    out = rgb.astype(np.float32, copy=True)
    lum = (0.299 * out[:, :, 0] + 0.587 * out[:, :, 1]
           + 0.114 * out[:, :, 2])[:, :, np.newaxis]

    if mode == "rgb" and rgb_stars is not None:
        ref = _resize_to(_normalise(rgb_stars), rgb.shape[0], rgb.shape[1]) \
            if rgb_stars.shape[:2] != rgb.shape[:2] else _normalise(rgb_stars)
        ref_lum = (0.299 * ref[:, :, 0] + 0.587 * ref[:, :, 1]
                   + 0.114 * ref[:, :, 2])[:, :, np.newaxis]
        # NB luminance * broadband colour ratio -> natural star colour.
        recolored = np.clip(lum * (ref / np.maximum(ref_lum, 1e-4)), 0.0, 1.0)
        out = out * (1.0 - m) + recolored * m
    else:  # desaturate toward luminance
        neutral = np.repeat(lum, 3, axis=2)
        out = out * (1.0 - m) + neutral * m
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_combined(img: np.ndarray, output_path: str,
                   preview_stretch: str = "ghs") -> None:
    """Save combined image as FITS + JPEG preview."""
    from astropy.io import fits

    data_out = np.transpose(img.astype(np.float32), (2, 0, 1))
    hdu = fits.PrimaryHDU(data=data_out)
    hdu.header["COMBINED"] = (True, "Channel-combined image")
    hdu.header["CREATOR"] = "originstack.py channel_combine"
    hdu.writeto(output_path, overwrite=True)
    print(f"  Saved: {output_path}")

    try:
        from src.io_fits import save_preview_rgb
        preview_path = os.path.splitext(output_path)[0] + ".jpg"
        save_preview_rgb(img, preview_path, stretch=preview_stretch,
                         ghs_b=8.0, ghs_sp=0.15, ghs_hp=0.95)
        print(f"  Preview: {preview_path}")
    except Exception as e:
        print(f"  WARNING: could not save preview: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_combine_cli(argv=None) -> None:
    """Parse argv and run channel combination.  Called from main()."""
    p = argparse.ArgumentParser(
        prog="originstack.py combine",
        description="Combine separately stacked channel FITS files into a colour image.",
    )

    # LRGB inputs
    lrgb = p.add_argument_group("LRGB inputs")
    lrgb.add_argument("--lum", metavar="L.fits",
                      help="Luminance (L) stack for LRGB combination")
    lrgb.add_argument("--red", metavar="R.fits",
                      help="Red channel stack")
    lrgb.add_argument("--green", metavar="G.fits",
                      help="Green channel stack")
    lrgb.add_argument("--blue", metavar="B.fits",
                      help="Blue channel stack")

    # Narrowband inputs
    nb = p.add_argument_group("Narrowband inputs")
    nb.add_argument("--ha", metavar="Ha.fits",
                    help="Ha (hydrogen-alpha) stack")
    nb.add_argument("--oiii", metavar="OIII.fits",
                    help="OIII stack")
    nb.add_argument("--sii", metavar="SII.fits",
                    help="SII stack")

    # Mode & output
    p.add_argument("--mode",
                   choices=["lrgb", "sho", "hoo", "hos", "hso", "oho", "shs"],
                   default="lrgb",
                   help="Combination mode (default: lrgb). "
                        "lrgb: luminance-layering. "
                        "sho: Hubble palette (SII->R, Ha->G, OIII->B). "
                        "hoo: Ha->R, OIII->G+B. "
                        "hos: Ha->R, OIII->G, SII->B.")
    p.add_argument("--saturation-boost", type=float, default=1.0,
                   help="Saturation multiplier applied after LRGB combination (default: 1.0)")
    p.add_argument("--continuum", metavar="CONTINUUM.fits",
                   help="Broadband/OSC continuum reference to subtract from "
                        "--continuum-target before narrowband combination (e.g. an OSC "
                        "stack, for a hybrid narrowband+broadband workflow).")
    p.add_argument("--continuum-target", choices=["ha", "oiii", "sii"], default="ha",
                   help="Which narrowband channel --continuum is subtracted from "
                        "(default: ha).")
    p.add_argument("--continuum-scale", type=float, default=None, metavar="SCALE",
                   help="Manual continuum subtraction scale factor. Default: fit "
                        "automatically (sweeps candidate scales, picks the one that "
                        "maximises the background residual's pixel-value skewness -- "
                        "verified against synthetic ground truth, see "
                        "optimal_continuum_scale's docstring) instead of eyeballing a "
                        "slider.")
    p.add_argument("--scnr", type=float, default=0.0, metavar="AMOUNT",
                   help="Average-neutral SCNR green removal for narrowband palettes "
                        "(0=off, 1=full; recommended ~1.0 for SHO to kill the green cast)")
    p.add_argument("--star-recolor", choices=["none", "desaturate", "rgb"], default="none",
                   help="Fix magenta narrowband stars: 'desaturate' neutralises star "
                        "colour; 'rgb' transplants colours from --rgb-stars.")
    p.add_argument("--rgb-stars", metavar="RGB.fits",
                   help="Broadband RGB image whose star colours are transplanted onto "
                        "the narrowband stars when --star-recolor rgb is used.")
    p.add_argument("--stretch", choices=["linear", "arcsinh", "ghs"], default="ghs",
                   help="Preview JPEG stretch (default: ghs)")
    p.add_argument("-o", "--output", required=True,
                   help="Output FITS path")

    args = p.parse_args(argv)

    print(f"\n{'='*60}")
    print(f"Channel Combination — mode: {args.mode.upper()}")
    print(f"{'='*60}")

    if args.mode == "lrgb":
        missing = []
        for name, val in [("--lum", args.lum), ("--red", args.red),
                          ("--green", args.green), ("--blue", args.blue)]:
            if not val:
                missing.append(name)
        if missing:
            p.error(f"LRGB mode requires: {', '.join(missing)}")

        print(f"  Loading L: {args.lum}")
        L = _load_channel(args.lum)
        print(f"  Loading R: {args.red}")
        R = _load_channel(args.red)
        print(f"  Loading G: {args.green}")
        G = _load_channel(args.green)
        print(f"  Loading B: {args.blue}")
        B = _load_channel(args.blue)

        print(f"  Combining LRGB (saturation_boost={args.saturation_boost:.2f})...")
        combined = lrgb_combine(L, R, G, B,
                                saturation_boost=args.saturation_boost)

    else:
        # Narrowband mode
        ha_arr = _load_channel(args.ha) if args.ha else None
        oiii_arr = _load_channel(args.oiii) if args.oiii else None
        sii_arr = _load_channel(args.sii) if args.sii else None

        if args.continuum:
            print(f"  Loading continuum reference: {args.continuum}")
            continuum_arr = _load_channel(args.continuum)
            targets = {'ha': ha_arr, 'oiii': oiii_arr, 'sii': sii_arr}
            target_arr = targets.get(args.continuum_target)
            if target_arr is None:
                p.error(f"--continuum-target {args.continuum_target} was not supplied "
                        f"(need --{args.continuum_target})")
            if target_arr.shape != continuum_arr.shape:
                p.error(f"--continuum shape {continuum_arr.shape} doesn't match "
                        f"--{args.continuum_target} shape {target_arr.shape}")
            subtracted, scale_used, _diag = subtract_continuum(
                target_arr, continuum_arr, scale=args.continuum_scale)
            method = "manual" if args.continuum_scale is not None else "skewness-transition fit"
            print(f"  Continuum-subtracted {args.continuum_target.upper()}: "
                  f"scale={scale_used:.4f} ({method})")
            targets[args.continuum_target] = subtracted
            ha_arr, oiii_arr, sii_arr = targets['ha'], targets['oiii'], targets['sii']

        print(f"  Combining {args.mode.upper()} narrowband...")
        combined = narrowband_combine(
            mode=args.mode, ha=ha_arr, oiii=oiii_arr, sii=sii_arr
        )

        if args.scnr > 0.0:
            print(f"  SCNR green removal (amount={args.scnr:.2f})...")
            combined = scnr_green(combined, amount=args.scnr)

        if args.star_recolor != "none":
            rgb_stars = _load_rgb(args.rgb_stars) if args.rgb_stars else None
            if args.star_recolor == "rgb" and rgb_stars is None:
                p.error("--star-recolor rgb requires --rgb-stars")
            print(f"  Fixing narrowband star colours ({args.star_recolor})...")
            combined = fix_narrowband_stars(combined, mode=args.star_recolor,
                                            rgb_stars=rgb_stars)

    _save_combined(combined, args.output, preview_stretch=args.stretch)
    print("\n  Done!")
