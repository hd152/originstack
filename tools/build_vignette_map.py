#!/usr/bin/env python
"""Build a per-instrument vignetting/background calibration map from many
past sessions' individual calibrated light frames.

Not part of the per-run stacking pipeline -- an occasional offline step, run
once (or re-run as you accumulate more sessions) to produce a FITS map that
``--vignette-map`` then applies per-frame during normal stacking.

Method: for each sampled light, calibrate (dark-subtract, flat-divide),
debayer, block-downsample, and estimate a per-channel background (masking
stars/nebula the same way DBE does), then pedestal-normalize (subtract its
own median) so only the *shape* remains. Robust-median-combining that shape
across many sessions pointed in different sky directions reinforces whatever
is spatially fixed (optical vignetting, a repeatable flat-calibration
residual, fixed local light-pollution direction) and averages away what
varies session to session (each night's own sky-glow tilt). See
src/vignette_calib.py for why this must be built from individual frames
(native sensor pixel space) rather than final stacks (already warped to each
session's own registration reference).

Usage:
    python tools/build_vignette_map.py --root G:\\astro -o vignette_map.fits
    python tools/build_vignette_map.py --sessions DIR1 DIR2 DIR3 -o vignette_map.fits
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io_fits import load_frame
from src.debayer import green_equalize, debayer
from src.background import (_build_emission_mask, gaussian_filter_ds,
                            _border_pixels, _sigma_sky, extract_background)


def block_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape[:2]
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    img = img[:h2, :w2]
    if img.ndim == 3:
        c = img.shape[2]
        return img.reshape(h2 // factor, factor, w2 // factor, factor, c).mean(axis=(1, 3))
    return img.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def calibrate_and_debayer(light_path: str, dark: np.ndarray, flat_norm: np.ndarray,
                          dark_exptime: float) -> np.ndarray:
    data, hdr = load_frame(light_path)
    if data.ndim != 2:
        return np.asarray(data, dtype=np.float32)
    data = data.astype(np.float32)
    if dark is not None and dark.shape == data.shape:
        light_exptime = float(hdr.get('EXPTIME', 0) or 0) or None
        scale = (light_exptime / dark_exptime) if (light_exptime and dark_exptime) else 1.0
        data -= dark * scale
    if flat_norm is not None and flat_norm.shape == data.shape:
        data /= flat_norm
    data = np.clip(data, 0, None)
    bayer_pat = str(hdr.get('BAYERPAT', 'RGGB'))
    data = green_equalize(data, pattern=bayer_pat)
    return debayer(data, pattern=bayer_pat, method='bilinear')


def per_frame_normalized_background(rgb_small: np.ndarray, mesh_div: int) -> np.ndarray:
    """Per-channel background estimate, pedestal-normalized to zero median."""
    lum = 0.299 * rgb_small[..., 0] + 0.587 * rgb_small[..., 1] + 0.114 * rgb_small[..., 2]
    H, W = lum.shape
    smooth_sigma = max(10.0, min(H, W) / 20.0)
    lum_smooth = gaussian_filter_ds(lum.astype(np.float64), sigma=smooth_sigma)
    sky_med, sky_std = _sigma_sky(_border_pixels(lum_smooth))
    mask = _build_emission_mask(lum, None, lum_smooth, sky_med, sky_std)

    out = np.empty_like(rgb_small, dtype=np.float64)
    for c in range(3):
        bg = extract_background(rgb_small[..., c].astype(np.float64),
                                mesh_size=max(8, W // mesh_div),
                                clip_sigma=3.0, star_mask=mask)
        out[..., c] = bg - np.median(bg)
    return out


def discover_sessions(root: str) -> list:
    sessions = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if (glob.glob(os.path.join(d, 'Light*.fits')) and
                glob.glob(os.path.join(d, 'dark*.fits')) and
                glob.glob(os.path.join(d, 'flat*.fits'))):
            sessions.append(d)
    return sessions


def build(session_dirs: list, out_path: str, max_frames_per_session: int,
         downsample: int, mesh_div: int, expected_fl_mm: float, expected_pxsz_um: float,
         out_h: int, out_w: int, verbose: bool = True) -> None:
    samples = []
    n_sessions_used = 0
    for d in session_dirs:
        sess = os.path.basename(d.rstrip('\\/'))
        lights = sorted(glob.glob(os.path.join(d, 'Light*.fits')))
        darks = sorted(glob.glob(os.path.join(d, 'dark*.fits')))
        flats = sorted(glob.glob(os.path.join(d, 'flat*.fits')))
        if not lights or not darks or not flats:
            if verbose:
                print(f"  skip (missing cal frames): {sess}")
            continue

        first_hdr = fits.getheader(lights[0])
        fl = float(first_hdr.get('FOCALLEN', 0) or 0)
        pxsz = float(first_hdr.get('XPIXSZ', 0) or 0)
        if expected_fl_mm and abs(fl - expected_fl_mm) > 1.0:
            if verbose:
                print(f"  skip (focal length {fl} != {expected_fl_mm}): {sess}")
            continue
        if expected_pxsz_um and abs(pxsz - expected_pxsz_um) > 0.05:
            if verbose:
                print(f"  skip (pixel size {pxsz} != {expected_pxsz_um}): {sess}")
            continue

        dark_data, dark_hdr = load_frame(darks[0])
        dark_exptime = float(dark_hdr.get('EXPTIME', 0) or 0) or 1.0
        flat_data, _ = load_frame(flats[0])
        flat_med = float(np.median(flat_data))
        flat_norm = np.clip(flat_data / flat_med, 0.4, 2.5) if flat_med > 1e-6 else None

        chosen = lights[:max_frames_per_session]
        n_ok = 0
        for lp in chosen:
            try:
                rgb = calibrate_and_debayer(lp, dark_data, flat_norm, dark_exptime)
                small = block_downsample(rgb, downsample)
                bg = per_frame_normalized_background(small, mesh_div)
                samples.append(bg)
                n_ok += 1
            except Exception as e:
                if verbose:
                    print(f"    frame failed ({os.path.basename(lp)}): {e}")
        if verbose:
            print(f"  {sess}: {n_ok}/{len(chosen)} frames sampled")
        if n_ok:
            n_sessions_used += 1

    print(f"\nTotal sessions used: {n_sessions_used}, total frame samples: {len(samples)}")
    if len(samples) < 10:
        raise SystemExit("Too few samples for a meaningful combine (need >= 10).")

    stack = np.stack(samples, axis=0)
    calib_small = np.median(stack, axis=0).astype(np.float32)

    # Upsample the (smooth, low-frequency by construction) map to full sensor
    # resolution with a light final smooth, same finishing pattern DBE uses
    # for its own coarse-grid-to-full-res upsample.
    from scipy import ndimage as _ndimage
    zy, zx = out_h / calib_small.shape[0], out_w / calib_small.shape[1]
    calib_full = _ndimage.zoom(calib_small, (zy, zx, 1.0), order=3)[:out_h, :out_w]
    calib_full = _ndimage.gaussian_filter(calib_full, sigma=(2.0, 2.0, 0))

    hdu = fits.PrimaryHDU(np.moveaxis(calib_full, -1, 0).astype(np.float32))
    hdu.header['NSAMPLES'] = len(samples)
    hdu.header['NSESS'] = n_sessions_used
    hdu.header['FOCALLEN'] = expected_fl_mm
    hdu.header['XPIXSZ'] = expected_pxsz_um
    hdu.writeto(out_path, overwrite=True)
    print(f"Saved: {out_path}  shape={calib_full.shape}")
    for c, name in enumerate(['R', 'G', 'B']):
        ch = calib_full[..., c]
        print(f"  {name}: std={ch.std():.2f}  range=[{ch.min():.2f}, {ch.max():.2f}]")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=None,
                   help='Auto-discover sessions (dirs with Light*/dark*/flat*.fits) under this root')
    p.add_argument('--sessions', nargs='*', default=None,
                   help='Explicit list of session directories (alternative to --root)')
    p.add_argument('-o', '--output', default='vignette_map.fits')
    p.add_argument('--max-frames-per-session', type=int, default=15)
    p.add_argument('--downsample', type=int, default=8,
                   help='Block-average factor before background estimation (default: 8)')
    p.add_argument('--mesh-div', type=int, default=12,
                   help='Downsampled width / this = mesh cell size for background extraction')
    p.add_argument('--focal-length', type=float, default=None,
                   help='Expected FOCALLEN (mm); sessions that don\'t match are skipped. '
                        'Auto-detected from the first usable session if omitted.')
    p.add_argument('--pixel-size', type=float, default=None,
                   help='Expected XPIXSZ (um); sessions that don\'t match are skipped. '
                        'Auto-detected from the first usable session if omitted.')
    p.add_argument('--out-height', type=int, default=None,
                   help='Output map height (default: first session light frame height)')
    p.add_argument('--out-width', type=int, default=None,
                   help='Output map width (default: first session light frame width)')
    args = p.parse_args()

    if not args.root and not args.sessions:
        p.error('Provide --root or --sessions')

    session_dirs = list(args.sessions) if args.sessions else discover_sessions(args.root)
    if not session_dirs:
        raise SystemExit('No candidate sessions found (need Light*/dark*/flat*.fits in a directory)')
    print(f"Candidate sessions: {len(session_dirs)}")

    fl = args.focal_length
    pxsz = args.pixel_size
    out_h, out_w = args.out_height, args.out_width
    if fl is None or pxsz is None or out_h is None or out_w is None:
        for d in session_dirs:
            lights = sorted(glob.glob(os.path.join(d, 'Light*.fits')))
            if not lights:
                continue
            hdr = fits.getheader(lights[0])
            if fl is None:
                fl = float(hdr.get('FOCALLEN', 0) or 0) or None
            if pxsz is None:
                pxsz = float(hdr.get('XPIXSZ', 0) or 0) or None
            if out_h is None:
                out_h = int(hdr.get('NAXIS2', 0) or 0) or None
            if out_w is None:
                out_w = int(hdr.get('NAXIS1', 0) or 0) or None
            if fl and pxsz and out_h and out_w:
                break
    if not (fl and pxsz and out_h and out_w):
        raise SystemExit('Could not determine instrument config / frame size from any session '
                         '-- pass --focal-length/--pixel-size/--out-height/--out-width explicitly')
    print(f"Instrument filter: FOCALLEN={fl}mm XPIXSZ={pxsz}um  output size: {out_w}x{out_h}")

    build(session_dirs, args.output, args.max_frames_per_session, args.downsample,
         args.mesh_div, fl, pxsz, out_h, out_w)


if __name__ == '__main__':
    main()
