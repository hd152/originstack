"""
reprocess_bg.py — Subtract the sky floor from an already-stacked FITS.

Usage:
    python tools/reprocess_bg.py sample/whirl47.fits
    python tools/reprocess_bg.py sample/whirl47.fits -o out.fits -v

The stacker's background extraction removes the *spatial gradient* across the
image but may leave a non-zero constant pedestal (sky floor).  This script
applies only the sky floor subtraction step so the background becomes truly
black, without re-running the full background extraction which can damage
extended emission (galaxy halos, IFN).

Algorithm:
  1. Build a source mask: every pixel that is clearly not pure sky
     (stars detected by DAOStarFinder + any pixel above 2× sky noise in
     the smoothed luminance).
  2. Estimate the per-channel sky floor as the sigma-clipped median of the
     remaining unmasked sky pixels.
  3. Subtract the floor from the entire image.  Stars, galaxy, and IFN
     all remain intact — only the constant offset is removed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from originstack import generate_star_mask, safe_print, sky_floor_normalize

try:
    from astropy.stats import sigma_clipped_stats
except Exception:
    sigma_clipped_stats = None

try:
    from photutils.detection import DAOStarFinder
except Exception:
    DAOStarFinder = None


def detect_stars(rgb: np.ndarray, verbose: bool = False):
    """Return a (H, W) float32 star mask, or None if photutils unavailable."""
    if DAOStarFinder is None or sigma_clipped_stats is None:
        return None
    try:
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        _, bg_med, bg_std = sigma_clipped_stats(lum, sigma=3.0, maxiters=5)
        finder = DAOStarFinder(fwhm=3.0,
                               threshold=float(bg_med) + 5.0 * float(bg_std))
        sources = finder(lum - float(bg_med))
        if sources is not None and len(sources) > 0:
            mask = generate_star_mask(lum.shape, sources, fwhm=4.0)
            if verbose:
                safe_print(f"  Stars detected: {len(sources)}")
            return mask
    except Exception as e:
        if verbose:
            safe_print(f"  Star detection skipped: {e}")
    return None


def main():
    p = argparse.ArgumentParser(
        description="Subtract sky floor from a stacked FITS to make background black")
    p.add_argument("input", help="Input stacked FITS file")
    p.add_argument("-o", "--output",
                   help="Output FITS path (default: <input>_sky0.fits)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"File not found: {inp}")

    out = Path(args.output) if args.output else inp.with_stem(inp.stem + "_sky0")
    if out.suffix.lower() not in (".fits", ".fit"):
        out = out.with_suffix(".fits")

    print(f"Loading {inp} ...")
    with fits.open(inp) as hdul:
        hdr = hdul[0].header.copy()
        raw = hdul[0].data.astype(np.float32)

    # Normalise to (H, W, 3)
    if raw.ndim == 3 and raw.shape[0] == 3:
        rgb = np.moveaxis(raw, 0, -1).copy()
    elif raw.ndim == 3 and raw.shape[2] == 3:
        rgb = raw.copy()
    else:
        sys.exit(f"Unexpected FITS shape {raw.shape}; expected (3,H,W) or (H,W,3)")

    H, W = rgb.shape[:2]
    print(f"Image size: {W}×{H}")

    # Measure sky BEFORE
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[:H, :W]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    outer = dist > 0.75 * min(cy, cx)
    print("\nSky floor before (outer 25% annulus):")
    for i, ch in enumerate("RGB"):
        pix = rgb[:, :, i][outer]
        med = float(np.median(pix))
        print(f"  {ch}: median={med:.2f} ADU")

    # Detect stars for a better source mask
    print("\nDetecting stars ...")
    star_mask = detect_stars(rgb, verbose=args.verbose)

    # Apply sky floor normalisation
    print("\nApplying sky floor normalisation ...")
    rgb_out = sky_floor_normalize(rgb, star_mask=star_mask, verbose=True)

    # Measure sky AFTER
    print("\nSky floor after:")
    for i, ch in enumerate("RGB"):
        pix = rgb_out[:, :, i][outer]
        med = float(np.median(pix))
        print(f"  {ch}: median={med:.2f} ADU")

    # Save FITS
    out_data = np.moveaxis(rgb_out, -1, 0).astype(np.float32)
    hdr["SKYFLOOR"] = (True, "Sky floor normalised to zero by reprocess_bg.py")
    fits.writeto(str(out), out_data, hdr, overwrite=True)
    print(f"\nSaved: {out}")

    # Optional JPEG preview
    try:
        from PIL import Image
        lum_out = (0.299 * rgb_out[:, :, 0] + 0.587 * rgb_out[:, :, 1]
                   + 0.114 * rgb_out[:, :, 2])
        lo = float(np.percentile(lum_out, 0.1))
        hi = float(np.percentile(lum_out, 99.8))
        rng = max(hi - lo, 1e-6)

        def stretch(ch):
            c = np.clip((ch - lo) / rng, 0, 1)
            return (np.arcsinh(c * 5) / np.arcsinh(5) * 255).astype(np.uint8)

        preview = np.stack([stretch(rgb_out[:, :, i]) for i in range(3)], axis=-1)
        jpg = out.with_suffix(".jpg")
        Image.fromarray(preview).save(str(jpg), quality=92)
        print(f"Preview:  {jpg}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
