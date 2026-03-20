"""Command-line interface: process_directory, parse_args, main."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np
from astropy.io import fits
from scipy import ndimage

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

from src.gpu_context import GpuContext, get_gpu
from src.models import Config, ProcessingStats
from src.utils import safe_print, print_header, format_time, setup_logging
from src.io_fits import make_master, save_preview_rgb
from src.frame_discovery import discover_frames, select_matching_darks
from src.debayer import build_hot_pixel_map
from src.registration import calculate_shift, apply_transform
from src.pipeline import stack_target
from src.health_check import run_health_check


def process_directory(directory: str, output: str, args: argparse.Namespace):
    # Print banner
    print_header("Astrophotography FITS Stacker", "=")
    print(f"Input:  {directory}")
    print(f"Output: {output}")
    get_gpu().print_status()

    # Detect hierarchical mode
    if not os.path.isdir(directory):
        print(f'\n  ERROR: Input directory {directory} does not exist', file=sys.stderr)
        raise SystemExit(1)

    overall_start = time.time()
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    targets = []

    print("\nDiscovering frames...")
    if any(os.listdir(directory)) and any(f.lower().endswith(('.fit', '.fits')) for f in os.listdir(directory)):
        # single folder
        targets = [(directory, output)]
        print(f"  Mode: Single folder")
    elif subdirs:
        # hierarchical: produce per-subfolder stacks then combine
        tmp_stacks = []
        for d in sorted(subdirs):
            name = os.path.basename(d)
            outp = os.path.join(tempfile.gettempdir(), f'{name}_stack.fits')
            targets.append((d, outp))
            tmp_stacks.append(outp)
        print(f"  Mode: Hierarchical ({len(targets)} subfolders)")
        # final combined output will be combined from tmp_stacks
    else:
        print('  ERROR: No FITS files found', file=sys.stderr)
        raise SystemExit('No FITS files found')

    # Process each target
    produced = []
    for target_idx, (d, outp) in enumerate(targets, 1):
        if len(targets) > 1:
            print(f'\n{"=" * 70}')
            print(f'TARGET {target_idx}/{len(targets)}: {os.path.basename(d)}')
            print(f'{"=" * 70}')
        else:
            print()

        # Create stats object for this target
        stats = ProcessingStats()

        frames = discover_frames(d)
        nfiles = sum(len(v) for v in frames.values())
        print(f'  Found {nfiles} FITS files: {len(frames["light"])} lights, {len(frames["dark"])} darks, {len(frames["flat"])} flats, {len(frames["bias"])} bias')

        if getattr(args, 'health_check', False):
            print_header("HEALTH CHECK", "=")
            safe_print(f"  Directory: {os.path.abspath(d)}")

        # Create master calibration frames
        if frames['dark'] or frames['flat'] or frames['bias']:
            print("\nCreating master calibration frames...")
            cal_start = time.time()

        masters = {}
        if frames['bias']:
            masters['bias'] = make_master(frames['bias'], method='median')
            if masters['bias'] is not None:
                safe_print(f"  ✓ Master bias:  {len(frames['bias'])} frames → {masters['bias'].shape[0]}×{masters['bias'].shape[1]}")
        else:
            masters['bias'] = None

        if frames['dark']:
            # Select darks that best match the light frames (ISO, exposure, dimensions)
            frames['dark'] = select_matching_darks(frames.get('light', []), frames['dark'])
            masters['dark'] = make_master(frames['dark'], method='median')
            if masters['dark'] is not None:
                safe_print(f"  ✓ Master dark:  {len(frames['dark'])} frames → {masters['dark'].shape[0]}×{masters['dark'].shape[1]}")
        else:
            masters['dark'] = None

        if frames['flat']:
            masters['flat'] = make_master(frames['flat'], method='median')
            if masters['flat'] is not None:
                safe_print(f"  ✓ Master flat:  {len(frames['flat'])} frames → {masters['flat'].shape[0]}×{masters['flat'].shape[1]}")
        else:
            masters['flat'] = None

        # Build hot pixel map from unsmoothed dark BEFORE smoothing.
        # Dark smoothing destroys per-pixel hot pixel information, so we
        # capture it first for Bayer-level correction in each light frame.
        masters['hot_pixel_map'] = None
        if masters.get('dark') is not None:
            hot_map = build_hot_pixel_map(masters['dark'])
            n_hot = int(np.sum(hot_map))
            if n_hot > 0:
                masters['hot_pixel_map'] = hot_map
                safe_print(f"  ✓ Hot pixel map: {n_hot} pixels from dark frame")

        # Smooth master calibration frames to reduce pixel-level noise.
        # With few calibration frames (especially 1), per-pixel noise is as high
        # as a single light frame — this noise is correlated across all lights
        # and does NOT stack out.  Calibration corrects large-scale effects
        # (bias pedestal, thermal gradient, vignetting/dust), so heavy smoothing
        # preserves the correction while eliminating the noise penalty.
        if masters.get('bias') is not None:
            n_bias = len(frames['bias'])
            # Bias is nearly constant; smooth aggressively
            sigma_b = max(1, 30 // max(1, int(np.sqrt(n_bias))))
            masters['bias'] = ndimage.gaussian_filter(masters['bias'].astype(np.float32), sigma=sigma_b)
        if masters.get('dark') is not None:
            n_dark = len(frames['dark'])
            # Dark has amp-glow gradients (>100px scale); moderate smoothing
            sigma_d = max(1, 20 // max(1, int(np.sqrt(n_dark))))
            masters['dark'] = ndimage.gaussian_filter(masters['dark'].astype(np.float32), sigma=sigma_d)
        if masters.get('flat') is not None:
            n_flat = len(frames['flat'])
            # Flat has vignetting + dust donuts (>30px); preserve those.
            # CRITICAL: smooth each Bayer colour channel independently.
            # A whole-image Gaussian with sigma > ~2 px averages adjacent R/G/B
            # Bayer pixels together, making flat_norm identical for all channels
            # (~0.888) and completely disabling per-channel QE correction.
            # Per-channel smoothing keeps the correct flat_norm values
            # (R≈0.39, G≈1.00, B≈1.16 for this camera) so the flat field
            # simultaneously corrects vignetting AND camera spectral response.
            sigma_f = max(1, 15 // max(1, int(np.sqrt(n_flat))))
            flat_raw = masters['flat'].astype(np.float32)
            for r_off, c_off in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                ch = flat_raw[r_off::2, c_off::2]
                flat_raw[r_off::2, c_off::2] = ndimage.gaussian_filter(ch, sigma=sigma_f)
            masters['flat'] = flat_raw

        # Store dark exposure time so _process_single_frame can scale correctly
        masters['dark_exptime'] = None
        if frames.get('dark'):
            try:
                masters['dark_exptime'] = float(
                    frames['dark'][0].header.get('EXPTIME', 0) or 0) or None
            except Exception:
                pass

        # --- Calibration frame analysis ---
        if frames['dark'] or frames['flat'] or frames['bias']:
            try:
                if masters.get('bias') is not None:
                    b = masters['bias']
                    b_med = float(np.median(b))
                    b_std = float(np.std(b))
                    if b_std < 20:
                        b_quality = "Good (low read noise)"
                    elif b_std < 60:
                        b_quality = "OK"
                    else:
                        b_quality = "Poor (noisy — stack more bias frames)"
                    safe_print(f"    Bias:  pedestal={b_med:.1f} ADU  "
                               f"noise={b_std:.1f} ADU  → {b_quality}")
                if masters.get('dark') is not None:
                    d = masters['dark']
                    dark_med = float(np.median(d))
                    dark_peak = float(d.max())
                    dark_et = masters.get('dark_exptime')
                    dark_hdr = frames['dark'][0].header if frames.get('dark') else {}
                    dark_temp_c = dark_hdr.get('CCD-TEMP')
                    dark_iso = dark_hdr.get('ISOSPEED') or dark_hdr.get('ISO') or dark_hdr.get('GAIN')
                    if dark_et and dark_et > 0:
                        rate = dark_med / dark_et
                        if rate < 0.02:
                            d_quality = "Good (low thermal current)"
                        elif rate < 0.1:
                            d_quality = "OK (moderate thermal current)"
                        else:
                            d_quality = "Poor (warm sensor — cool camera or use shorter darks)"
                        rate_str = f"  ({rate:.4f} ADU/s)"
                    else:
                        rate_str = ''
                        d_quality = "OK" if dark_med < 500 else "High dark current"
                    temp_str = ''
                    if dark_temp_c is not None:
                        temp_f = dark_temp_c * 9.0 / 5.0 + 32.0
                        temp_str = f"  temp={dark_temp_c:.1f}°C/{temp_f:.1f}°F"
                    exp_str = f"  exp={dark_et:.1f}s" if dark_et else ''
                    iso_str = f"  ISO={dark_iso}" if dark_iso is not None else ''
                    safe_print(f"    Dark:  median={dark_med:.1f} ADU{rate_str}"
                               f"{temp_str}{exp_str}{iso_str}  peak={dark_peak:.0f} ADU  → {d_quality}")
                    # Warn if dark ISO doesn't match the majority of light frames
                    if dark_iso is not None and frames.get('light'):
                        light_isos = []
                        for lf in frames['light']:
                            liso = lf.header.get('ISOSPEED') or lf.header.get('ISO') or lf.header.get('GAIN')
                            if liso is not None:
                                light_isos.append(liso)
                        if light_isos:
                            majority_iso = max(set(light_isos), key=light_isos.count)
                            if str(dark_iso) != str(majority_iso):
                                safe_print(f"    ⚠ ISO mismatch: dark ISO={dark_iso}, "
                                           f"lights ISO={majority_iso} — dark may not cancel sensor noise correctly")
                if masters.get('flat') is not None:
                    flat = masters['flat']
                    flat_med = float(np.median(flat))
                    if flat_med > 0:
                        bayer_labels = ['R', 'G1', 'G2', 'B']
                        bayer_ratios = []
                        for r_off, c_off in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                            ch_med = float(np.median(flat[r_off::2, c_off::2]))
                            bayer_ratios.append(ch_med / flat_med)
                        ratio_str = '/'.join(
                            f'{lbl}={r:.3f}'
                            for lbl, r in zip(bayer_labels, bayer_ratios))
                        H_f, W_f = flat.shape
                        qh, qw = H_f // 8, W_f // 8
                        center_med = float(np.median(
                            flat[H_f // 2 - qh:H_f // 2 + qh,
                                 W_f // 2 - qw:W_f // 2 + qw]))
                        cs = max(10, min(100, H_f // 12, W_f // 12))
                        corner_med = float(np.median(np.concatenate([
                            flat[:cs, :cs].ravel(), flat[:cs, -cs:].ravel(),
                            flat[-cs:, :cs].ravel(), flat[-cs:, -cs:].ravel()])))
                        vign = (1.0 - corner_med / center_med) * 100.0 if center_med > 0 else 0.0
                        if vign < 20:
                            f_quality = "Good (low vignetting)"
                        elif vign < 40:
                            f_quality = "OK (moderate vignetting)"
                        elif vign < 60:
                            f_quality = "Heavy vignetting — flat correction important"
                        else:
                            f_quality = "Severe vignetting — check flat exposure/optics"
                        safe_print(f"    Flat:  {ratio_str}  vignetting={vign:.1f}%  → {f_quality}")
            except Exception:
                pass

        if frames['dark'] or frames['flat'] or frames['bias']:
            stats.calibration_time = time.time() - cal_start

        if getattr(args, 'health_check', False):
            run_health_check(frames, masters, d)
            continue  # skip stacking

        # Validation warnings
        if len(frames['light']) < Config.MIN_RECOMMENDED_FRAMES:
            warning = f"Only {len(frames['light'])} light frames found (recommended: {Config.MIN_RECOMMENDED_FRAMES}+)"
            stats.add_warning(warning)
            safe_print(f"\n  ⚠ WARNING: {warning}")

        res = stack_target([f for t in frames.values() for f in t], outp, args, masters, stats)
        if res:
            produced.append(res)
    # If hierarchical combine
    if len(produced) > 1:
        print_header("HIERARCHICAL COMBINING", "=")
        print(f"  Combining {len(produced)} target stacks into final output...")

        # Load all stacks, crop to minimum common dimensions
        def _fits_shape(p):
            with fits.open(p, memmap=True) as hd:
                return hd[0].data.shape
        shapes = [_fits_shape(p) for p in produced]
        # shapes are (3,H,W)
        mins = np.min([[s[1], s[2]] for s in shapes], axis=0)
        Hm, Wm = int(mins[0]), int(mins[1])
        print(f"  Cropping all stacks to minimum dimensions: {Hm}×{Wm}")

        stacks = []
        for p in tqdm(produced, desc="  Loading", unit="target", disable=args.verbose):
            with fits.open(p, memmap=True) as hd:
                d = np.transpose(hd[0].data, (1, 2, 0)).astype(np.float32)
                stacks.append(d[:Hm, :Wm, :])

        # Register each stack against the first (reference) stack
        print(f"  Registering {len(stacks) - 1} stack(s) against reference...")
        ref_lum = np.mean(stacks[0], axis=2)
        shifts = [(0.0, 0.0)]
        for i in range(1, len(stacks)):
            lum = np.mean(stacks[i], axis=2)
            sy, sx = calculate_shift(ref_lum, lum, verbose=getattr(args, 'verbose', False))
            shifts.append((sy, sx))
            safe_print(f"    Stack {i + 1}/{len(stacks)}: shift=({sx:.2f}, {sy:.2f}) px")

        # Apply shifts (zero-pad edges)
        aligned = []
        for stack, (sy, sx) in zip(stacks, shifts):
            if sy == 0.0 and sx == 0.0:
                aligned.append(stack)
            else:
                aligned.append(apply_transform(stack, shift=(sy, sx)))

        # Crop to the valid (non-padded) overlap region across all aligned stacks
        all_dy = [s[0] for s in shifts]
        all_dx = [s[1] for s in shifts]
        y0 = int(np.ceil(max(0.0, max(all_dy))))
        y1 = int(np.floor(Hm + min(0.0, min(all_dy))))
        x0 = int(np.ceil(max(0.0, max(all_dx))))
        x1 = int(np.floor(Wm + min(0.0, min(all_dx))))

        if y0 >= y1 or x0 >= x1:
            safe_print("  Warning: shifts exceed image overlap — combining without registration")
            y0, y1, x0, x1 = 0, Hm, 0, Wm

        Hf, Wf = y1 - y0, x1 - x0
        print(f"  Valid overlap after registration: {Hf}×{Wf}")

        acc = np.zeros((Hf, Wf, 3), dtype=np.float64)
        for img in aligned:
            acc += img[y0:y1, x0:x1, :]
        combined = (acc / len(aligned)).astype(np.float32)

        out_hdu = fits.PrimaryHDU()
        out_hdu.data = np.transpose(combined, (2, 0, 1))
        out_hdu.header['NTARGETS'] = len(produced)
        out_hdu.writeto(output, overwrite=True)
        preview_path = os.path.splitext(output)[0] + '.jpg'
        save_preview_rgb(combined, preview_path, stretch=getattr(args, 'stretch', 'linear'))

        safe_print(f"  ✓ Combined output: {os.path.basename(output)} ({Hf}×{Wf}×3)")
        safe_print(f"  ✓ Preview: {os.path.basename(preview_path)}")

    # Overall summary
    total_time = time.time() - overall_start
    print_header("OVERALL SUMMARY", "=")
    if len(produced) > 1:
        print(f"  Targets processed: {len(produced)}")
    print(f"  Total time: {format_time(total_time)}")
    safe_print(f"\n  ✓ All processing complete!")
    print(f"{'=' * 70}\n")


def parse_args():
    p = argparse.ArgumentParser(description='Streaming FITS stacker')
    p.add_argument('-d', '--directory', required=True)
    p.add_argument('-o', '--output', default=None,
                   help='Output FITS path (required unless --health-check)')
    p.add_argument('--health-check', action='store_true',
                   help='Analyse input frames and calibration quality without stacking')
    p.add_argument('--no-registration', action='store_true')
    p.add_argument('--skip-phase-correlation', action='store_true',
                   help='Skip phase correlation, use only fallback methods (debug)')
    p.add_argument('--no-affine', action='store_true',
                   help='Disable affine (rotation+translation) registration; use translation-only')
    p.add_argument('--affine', action='store_true',
                   help='(Legacy, now default) affine registration is on unless --no-affine is set')
    p.add_argument('--no-quality-filter', action='store_false', dest='quality_filter',
                   default=True,
                   help='Disable automatic rejection of the lowest-quality frames')
    p.add_argument('--quality-threshold', type=float, default=25.0,
                   help='Reject frames below this quality percentile (default: 25 = '
                        'keep the best 75%% of frames). Use --no-quality-filter to keep all.')
    p.add_argument('--keep-intermediates', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--debug-registration', action='store_true',
                   help='Detailed registration diagnostics (implies -v)')
    p.add_argument('--stack-method', choices=['mean', 'median', 'sigma_clip'], default=None,
                   help='Stacking method (default: sigma_clip for dithered data, mean otherwise)')
    p.add_argument('--rejection-sigma', type=float, default=3.0,
                   help='Sigma threshold for pixel rejection in sigma_clip stacking (default: 3.0)')
    p.add_argument('--rejection-iters', type=int, default=3,
                   help='Number of clipping iterations for sigma_clip stacking (default: 3)')
    p.add_argument('--winsorize', action='store_true',
                   help='Winsorized sigma-clip: clip outliers to boundary instead of rejecting')
    p.add_argument('--debayer-method', choices=['bilinear', 'malvar', 'vng'], default='bilinear',
                   help='Debayering method (default: bilinear; malvar/vng require OpenCV)')
    p.add_argument('--white-balance', choices=['none', 'grayworld', 'whitepatch'], default='grayworld')
    p.add_argument('--drizzle-scale', type=float, default=1.0,
                   help='Drizzle scale factor (e.g. 2.0 for 2x super-resolution, 1.0 = disabled)')
    p.add_argument('--drizzle-drop-size', type=float, default=0.7,
                   help='Drizzle pixfrac / drop size (0.5-1.0, default: 0.7). '
                        'Smaller values yield sharper results at the cost of noise.')
    p.add_argument('--use-gpu', action='store_true',
                   help='Use CuPy for available operations (experimental)')
    p.add_argument('--plate-solve', action='store_true',
                   help='Enable plate solving via astrometry.net (requires astroquery and ASTROMETRY_API_KEY)')
    p.add_argument('--background-extraction', action='store_true', default=True,
                   help='Enable intelligent background removal for darker sky (default: on)')
    p.add_argument('--no-background-extraction', dest='background_extraction',
                   action='store_false',
                   help='Disable background extraction')
    p.add_argument('--bg-mesh-size', type=int, default=64,
                   help='Grid cell size in pixels for background estimation (default: 64)')
    p.add_argument('--bg-filter-size', type=int, default=3,
                   help='Median filter size for background grid smoothing (default: 3, must be odd)')
    p.add_argument('--bg-clip-sigma', type=float, default=3.0,
                   help='Sigma for star rejection in background estimation (default: 3.0)')
    p.add_argument('--denoise', action='store_true', default=True,
                   help='Enable wavelet denoising post-stack (default: on; requires pywt)')
    p.add_argument('--no-denoise', dest='denoise', action='store_false',
                   help='Disable wavelet denoising')
    p.add_argument('--denoise-strength', type=float, default=3.0,
                   help='Wavelet luma denoise threshold factor (default: 3.0)')
    p.add_argument('--denoise-chroma-boost', type=float, default=2.0,
                   help='Chroma threshold multiplier relative to luma (default: 2.0)')
    p.add_argument('--denoise-nlm', action='store_true',
                   help='Enable non-local means denoising after wavelet (requires skimage or cv2)')
    p.add_argument('--denoise-nlm-strength', type=float, default=1.0,
                   help='NLM filter strength multiplier relative to auto-estimated sigma (default: 1.0)')
    p.add_argument('--denoise-nlm-blend', type=float, default=0.5,
                   help='Blend fraction of NLM result with original (0=no NLM, 1=full NLM, default: 0.5). '
                        'Lower values prevent the non-uniform smoothing ("leopard print") artifact by '
                        'letting the original noise dominate. 0.5 reduces noise by ~30%% with <3%% '
                        'spatial variation. Increase to 0.7–1.0 for heavier denoising if no pattern appears.')
    p.add_argument('--denoise-bilateral', action='store_true',
                   help='Enable bilateral filter denoising after wavelet (requires cv2). '
                        'Spatially uniform by construction — no leopard-print artifact.')
    p.add_argument('--denoise-bilateral-sigma-color', type=float, default=None,
                   help='Bilateral value-similarity scale in ADU (default: auto from sky noise). '
                        'Pixels differing by more than ~2× this value are not mixed. '
                        'Try 1–5× the expected sky noise level.')
    p.add_argument('--denoise-bilateral-sigma-space', type=float, default=3.0,
                   help='Bilateral spatial smoothing radius in pixels (default: 3.0).')
    p.add_argument('--deconvolve', action='store_true',
                   help='Enable Richardson-Lucy deconvolution for sharpening (requires scikit-image)')
    p.add_argument('--deconvolve-iterations', type=int, default=Config.RL_DEFAULT_ITERATIONS,
                   help=f'Number of Richardson-Lucy iterations (default: {Config.RL_DEFAULT_ITERATIONS}). '
                        'More iterations = sharper but may amplify noise.')
    p.add_argument('--deconvolve-fwhm', type=float, default=None,
                   help='Override auto-estimated PSF FWHM in pixels. If set, a synthetic '
                        'Gaussian PSF is used instead of fitting star profiles.')
    p.add_argument('--deconvolve-psf-model', choices=['moffat', 'gaussian'], default='moffat',
                   help='PSF model for auto-estimation (default: moffat). '
                        'Moffat is preferred for seeing-limited images.')
    p.add_argument('--local-normalize', action='store_true',
                   help='Enable local normalization to remove vignetting residuals')
    p.add_argument('--local-normalize-sigma', type=float, default=50.0,
                   help='Gaussian sigma for local normalization (default: 50)')
    p.add_argument('--chroma-nr', action='store_true', default=True,
                   help='Enable chroma noise reduction to remove color speckle in sky background (default: on)')
    p.add_argument('--no-chroma-nr', dest='chroma_nr', action='store_false',
                   help='Disable chroma noise reduction')
    p.add_argument('--chroma-nr-sigma', type=float, default=2.0,
                   help='Gaussian sigma for chroma smoothing in pixels (default: 2.0)')
    p.add_argument('--stretch', choices=['linear', 'arcsinh'], default='arcsinh',
                   help='Preview image stretch method (default: arcsinh)')
    p.add_argument('-j', '--parallel', type=int, default=0,
                   help='Parallel workers for frame processing (default: 0=auto, 1=sequential)')
    # --- Chromatic aberration correction ---
    p.add_argument('--ca-correction', action='store_true', default=True,
                   help='Correct lateral chromatic aberration by aligning R/B channels '
                        'to green via phase cross-correlation (default: on; requires skimage)')
    p.add_argument('--no-ca-correction', dest='ca_correction', action='store_false',
                   help='Disable chromatic aberration correction')
    # --- Cosmic ray rejection ---
    p.add_argument('--cosmic-ray-rejection', action='store_true', default=True,
                   help='Apply L.A.Cosmic-style Laplacian cosmic ray rejection to each '
                        'light frame before stacking (default: on)')
    p.add_argument('--no-cosmic-ray-rejection', dest='cosmic_ray_rejection',
                   action='store_false',
                   help='Disable cosmic ray rejection')
    p.add_argument('--cr-sigclip', type=float, default=4.5,
                   help='L.A.Cosmic detection threshold in noise sigma units (default: 4.5)')
    p.add_argument('--cr-objlim', type=float, default=5.0,
                   help='L.A.Cosmic object-rejection ratio — prevents flagging star cores '
                        '(default: 5.0, increase to be more conservative)')
    # --- Adaptive denoising ---
    p.add_argument('--denoise-adaptive', action='store_true', default=True,
                   help='Use BayesShrink per-subband thresholds for wavelet denoising '
                        'instead of a fixed global threshold factor (default: on). '
                        'Adapts automatically to local noise — preserves faint nebulosity '
                        'better. Requires --denoise.')
    p.add_argument('--no-denoise-adaptive', dest='denoise_adaptive', action='store_false',
                   help='Use fixed global threshold factor instead of BayesShrink')
    # --- Structured logging ---
    p.add_argument('--log-level',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='WARNING',
                   help='Minimum log severity printed to stderr (default: WARNING). '
                        'Use DEBUG for verbose diagnostic output from all modules.')
    p.add_argument('--log-file', default=None, metavar='PATH',
                   help='Write a full DEBUG-level log to this file in addition to '
                        'console output.')
    return p.parse_args()


def main():
    args = parse_args()
    if not args.health_check and not args.output:
        print("ERROR: -o/--output is required unless --health-check is specified", file=sys.stderr)
        raise SystemExit(1)
    # debug_registration implies verbose
    if args.debug_registration:
        args.verbose = True
    # Initialise structured logging before anything else
    setup_logging(level=getattr(args, 'log_level', 'WARNING'),
                  log_file=getattr(args, 'log_file', None))
    # Initialise GPU context (module-level singleton)
    from src import gpu_context as _gpu_mod
    _gpu_mod._gpu = GpuContext(use_gpu=args.use_gpu)
    try:
        process_directory(args.directory, args.output, args)
    except Exception as e:
        print(f'ERROR: {str(e)}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
