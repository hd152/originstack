"""Channel combination for LRGB and narrowband imaging.

Combines separately stacked FITS files into a colour image:
  - LRGB:  luminance-layering (L replaces lightness, RGB provides colour)
  - SHO:   Hubble palette  — SII→R, Ha→G, OIII→B
  - HOO:   Ha→R, OIII→G, OIII→B
  - HOS:   Ha→R, OIII→G, SII→B
  - HSO:   Ha→R, SII→G, OIII→B   (uncommon but supported)
  - Custom: any per-channel mapping via --mapping R=<file> G=<file> B=<file>

CLI entry point: ``python astro_stack.py combine ...``
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np


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
    hdu.header["CREATOR"] = "astro_stack.py channel_combine"
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
        prog="astro_stack.py combine",
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

        print(f"  Combining {args.mode.upper()} narrowband...")
        combined = narrowband_combine(
            mode=args.mode, ha=ha_arr, oiii=oiii_arr, sii=sii_arr
        )

    _save_combined(combined, args.output, preview_stretch=args.stretch)
    print(f"\n  Done!")
