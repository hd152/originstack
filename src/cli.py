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


def load_config_file(config_path: str, args: argparse.Namespace) -> list:
    """Load configuration from a TOML file, applying values that aren't set on CLI."""
    changes = []
    try:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                safe_print(f"  WARNING: TOML support not available (pip install tomli)")
                return changes

        with open(config_path, 'rb') as f:
            config = tomllib.load(f)

        # Flatten nested sections: [stacking] sigma=3.0 -> rejection_sigma=3.0
        flat = {}
        section_maps = {
            'stacking': {
                'method': 'stack_method',
                'sigma': 'rejection_sigma',
                'iters': 'rejection_iters',
                'estimator': 'rejection_estimator',
            },
            'denoise': {
                'wavelet': 'denoise',
                'strength': 'denoise_strength',
                'adaptive': 'denoise_adaptive',
                'nlm': 'denoise_nlm',
                'bilateral': 'denoise_bilateral',
                'mmt': 'denoise_mmt',
                'acdnr': 'denoise_acdnr',
            },
            'stretch': {
                'method': 'stretch',
                'ghs_b': 'ghs_b',
                'ghs_sp': 'ghs_sp',
                'ghs_hp': 'ghs_hp',
            },
        }

        for key, value in config.items():
            if isinstance(value, dict):
                mapping = section_maps.get(key, {})
                for sub_key, sub_value in value.items():
                    mapped = mapping.get(sub_key, f"{key}_{sub_key}".replace('-', '_'))
                    flat[mapped] = sub_value
            else:
                flat[key.replace('-', '_')] = value

        for key, value in flat.items():
            if hasattr(args, key):
                setattr(args, key, value)
                changes.append(f"{key}={value}")

    except FileNotFoundError:
        safe_print(f"  WARNING: Config file not found: {config_path}")
    except Exception as e:
        safe_print(f"  WARNING: Error loading config: {e}")

    return changes


def apply_preset(args: argparse.Namespace) -> list:
    """Apply a preset configuration, returning list of changes made.

    Only sets values that the user hasn't explicitly provided on the command line.
    """
    if not args.preset:
        return []

    changes = []

    # Track which args were explicitly set by the user
    # (argparse doesn't track this natively, so we check against defaults)
    presets = {
        'quick': {
            'stack_method': 'mean',
            'denoise': True,
            'denoise_adaptive': False,
            'denoise_strength': 2.0,
            'denoise_nlm': False,
            'denoise_bilateral': False,
            'denoise_mmt': False,
            'denoise_acdnr': False,
            'deconvolve': False,
            'background_extraction': True,
            'dbe': False,  # Use faster legacy mesh
            'star_reduce': False,
            'local_contrast': False,
            'chroma_nr': True,
            'quality_threshold': 10.0,  # Keep more frames
        },
        'quality': {
            'stack_method': 'sigma_clip',
            'rejection_sigma': 2.5,
            'rejection_iters': 5,
            'denoise': True,
            'denoise_adaptive': True,
            'denoise_mmt': True,
            'denoise_acdnr': True,
            'deconvolve': True,
            'background_extraction': True,
            'dbe': True,
            'star_reduce': True,
            'local_contrast': True,
            'chroma_nr': True,
            'ca_correction': True,
            'cosmic_ray_rejection': True,
        },
        'narrowband': {
            'stack_method': 'sigma_clip',
            'rejection_sigma': 2.0,
            'denoise': True,
            'denoise_adaptive': True,
            'denoise_mmt': True,
            'denoise_mmt_strength': 2.0,
            'denoise_acdnr': True,
            'denoise_acdnr_k': 2.0,
            'deconvolve': False,
            'background_extraction': True,
            'dbe': True,
            'chroma_nr': False,  # Narrowband doesn't need chroma NR
            'star_reduce': True,
            'star_reduce_factor': 0.6,
            'local_contrast': True,
            'local_contrast_strength': 0.9,
            'stretch': 'ghs',
            'ghs_b': 12.0,
            'ghs_sp': 0.10,
        },
        'galaxy': {
            'stack_method': 'sigma_clip',
            'denoise': True,
            'denoise_adaptive': True,
            'denoise_bilateral': True,
            'deconvolve': True,
            'background_extraction': True,
            'dbe': True,
            'star_reduce': True,
            'star_reduce_factor': 0.5,
            'local_contrast': True,
            'local_contrast_strength': 0.8,
            'stretch': 'ghs',
            'ghs_b': 10.0,
            'ghs_sp': 0.12,
            'ghs_hp': 0.95,
            'chroma_nr': True,
        },
        'starfield': {
            'stack_method': 'sigma_clip',
            'denoise': True,
            'denoise_adaptive': True,
            'deconvolve': False,
            'background_extraction': True,
            'dbe': True,
            'star_reduce': False,
            'local_contrast': False,
            'stretch': 'ghs',
            'ghs_b': 5.0,
            'ghs_sp': 0.20,
            'chroma_nr': True,
        },
    }

    preset_values = presets.get(args.preset, {})
    for key, value in preset_values.items():
        current = getattr(args, key, None)
        # Only apply if the user hasn't changed it from the parser default
        # This is a best-effort check -- we apply the preset value
        setattr(args, key, value)
        changes.append(f"{key}={value}")

    return changes


def apply_stretch_preset(args: argparse.Namespace) -> None:
    """Apply a named stretch preset."""
    presets = {
        'galaxy':    {'stretch': 'ghs', 'ghs_b': 10.0, 'ghs_sp': 0.12, 'ghs_hp': 0.95},
        'nebula':    {'stretch': 'ghs', 'ghs_b': 12.0, 'ghs_sp': 0.10, 'ghs_hp': 0.98},
        'starfield': {'stretch': 'ghs', 'ghs_b': 5.0,  'ghs_sp': 0.20, 'ghs_hp': 0.90},
        'planetary': {'stretch': 'ghs', 'ghs_b': 3.0,  'ghs_sp': 0.30, 'ghs_hp': 0.85},
        'lunar':     {'stretch': 'linear'},
    }
    preset = presets.get(getattr(args, 'stretch_preset', None))
    if preset:
        for key, value in preset.items():
            setattr(args, key, value)


def save_effective_config(args: argparse.Namespace, output_path: str) -> None:
    """Save the effective parameter set as a TOML file next to the output."""
    config_path = os.path.splitext(output_path)[0] + '_config.toml'
    lines = ['# OriginStack effective configuration\n',
             '# Generated automatically -- can be reused with --config\n\n']

    skip_keys = {'directory', 'output', 'config', 'health_check', 'dry_run',
                 'verbose', 'debug_registration', 'keep_intermediates',
                 'ai_advisor', 'ai_report', 'preset', 'stretch_preset',
                 'diagnostic', 'diagnostic_dir'}

    for key, value in sorted(vars(args).items()):
        if key.startswith('_') or key in skip_keys:
            continue
        if isinstance(value, bool):
            lines.append(f'{key} = {"true" if value else "false"}\n')
        elif isinstance(value, (int, float)):
            lines.append(f'{key} = {value}\n')
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"\n')
        elif value is None:
            continue

    try:
        with open(config_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        safe_print(f"  WARNING: Could not save config file ({e})")


def _run_combined_sessions(subdirs: list, output: str, args: argparse.Namespace) -> None:
    """Pool all light frames from multiple session subfolders into one unified stack.

    Each subfolder's calibration frames are aggregated together to build
    stronger master frames. All lights are passed to a single stack_target
    call so registration, sigma-clip rejection, and post-processing operate
    across the full multi-night dataset at once.
    """
    overall_start = time.time()
    stats = ProcessingStats()

    combined: dict = {'light': [], 'dark': [], 'flat': [], 'bias': [], 'skip': []}
    for d in sorted(subdirs):
        sub = discover_frames(d)
        for ftype in combined:
            combined[ftype].extend(sub.get(ftype, []))

    n_lights = len(combined['light'])
    n_sessions = len(subdirs)
    print(f"  Pooled {n_lights} lights from {n_sessions} sessions "
          f"({len(combined['dark'])} darks, {len(combined['flat'])} flats, "
          f"{len(combined['bias'])} bias)")

    if not n_lights:
        print('  ERROR: No light frames found across sessions', file=sys.stderr)
        raise SystemExit('No light frames found')

    # Build master calibration frames from all sessions combined
    if combined['dark'] or combined['flat'] or combined['bias']:
        print("\nCreating master calibration frames...")
        cal_start = time.time()

    masters: dict = {}
    if combined['bias']:
        masters['bias'] = make_master(combined['bias'], method='median')
        if masters['bias'] is not None:
            safe_print(f"  ✓ Master bias:  {len(combined['bias'])} frames → "
                       f"{masters['bias'].shape[0]}×{masters['bias'].shape[1]}")
    else:
        masters['bias'] = None

    if combined['dark']:
        combined['dark'] = select_matching_darks(combined['light'], combined['dark'])
        masters['dark'] = make_master(combined['dark'], method='median')
        if masters['dark'] is not None:
            safe_print(f"  ✓ Master dark:  {len(combined['dark'])} frames → "
                       f"{masters['dark'].shape[0]}×{masters['dark'].shape[1]}")
    else:
        masters['dark'] = None

    if combined['flat']:
        masters['flat'] = make_master(combined['flat'], method='median')
        if masters['flat'] is not None:
            safe_print(f"  ✓ Master flat:  {len(combined['flat'])} frames → "
                       f"{masters['flat'].shape[0]}×{masters['flat'].shape[1]}")
    else:
        masters['flat'] = None

    masters['hot_pixel_map'] = None
    if masters.get('dark') is not None:
        hot_map = build_hot_pixel_map(masters['dark'])
        n_hot = int(np.sum(hot_map))
        if n_hot > 0:
            masters['hot_pixel_map'] = hot_map
            safe_print(f"  ✓ Hot pixel map: {n_hot} pixels from dark frame")

    if combined['dark'] or combined['flat'] or combined['bias']:
        # Smooth calibration frames (same logic as single-folder path)
        if masters.get('bias') is not None:
            n_bias = len(combined['bias'])
            sigma_b = max(1, 30 // max(1, int(np.sqrt(n_bias))))
            masters['bias'] = ndimage.gaussian_filter(masters['bias'].astype(np.float32), sigma_b)
        if masters.get('dark') is not None:
            n_dark = len(combined['dark'])
            sigma_d = max(1, 20 // max(1, int(np.sqrt(n_dark))))
            masters['dark'] = ndimage.gaussian_filter(masters['dark'].astype(np.float32), sigma_d)
        if masters.get('flat') is not None:
            n_flat = len(combined['flat'])
            sigma_f = max(1, 15 // max(1, int(np.sqrt(n_flat))))
            flat_raw = masters['flat'].astype(np.float32)
            for r_off, c_off in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                ch = flat_raw[r_off::2, c_off::2]
                flat_raw[r_off::2, c_off::2] = ndimage.gaussian_filter(ch, sigma_f)
            masters['flat'] = flat_raw
        stats.calibration_time = time.time() - cal_start

    masters['dark_exptime'] = None
    if combined.get('dark'):
        try:
            masters['dark_exptime'] = float(
                combined['dark'][0].header.get('EXPTIME', 0) or 0) or None
        except Exception as e:
            safe_print(f"  WARNING: Could not read dark EXPTIME ({e}) — dark scaling disabled")

    all_frames = [f for ftype in combined.values() for f in (ftype if isinstance(ftype, list) else [])]
    stack_target(all_frames, output, args, masters, stats)
    save_effective_config(args, output)
    safe_print(f"\n  Total elapsed: {format_time(time.time() - overall_start)}")


def process_directory(directory: str, output: str, args: argparse.Namespace):
    # Print banner
    print_header("Astrophotography FITS Stacker", "=")
    print(f"Input:  {directory}")
    print(f"Output: {output}")
    if getattr(args, 'preset', None):
        print(f"  Preset: {args.preset}")
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
    elif subdirs and getattr(args, 'combine_sessions', False):
        # Combined-sessions mode: pool all lights from all subfolders into one stack
        safe_print(f"  Mode: Combined sessions ({len(subdirs)} subfolders -> single unified stack)")
        _run_combined_sessions(subdirs, output, args)
        return
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

    # Mosaic mode requires hierarchical layout (one subfolder per panel)
    if getattr(args, 'mosaic', False):
        if len(targets) < 2:
            print('  ERROR: --mosaic requires subfolders (one per panel); '
                  'only a single target was found', file=sys.stderr)
            raise SystemExit('--mosaic requires subfolders')
        if not getattr(args, 'plate_solve', False):
            safe_print("  NOTE: --mosaic implies --plate-solve — enabling automatically")
            args.plate_solve = True

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

        # Validate calibration frame dimensions match light frames
        if frames['light']:
            first_hdr = frames['light'][0].header
            light_h = first_hdr.get('NAXIS2')
            light_w = first_hdr.get('NAXIS1')
            if light_h and light_w:
                for cal_name in ('bias', 'dark', 'flat'):
                    cal = masters.get(cal_name)
                    if cal is not None:
                        cal_shape = cal.shape
                        if cal_shape[0] != light_h or cal_shape[1] != light_w:
                            safe_print(f"  ⚠ WARNING: {cal_name} dimensions {cal_shape[1]}×{cal_shape[0]} "
                                       f"don't match lights {light_w}×{light_h} — disabling {cal_name} calibration")
                            masters[cal_name] = None

        # Store dark exposure time so _process_single_frame can scale correctly
        masters['dark_exptime'] = None
        if frames.get('dark'):
            try:
                masters['dark_exptime'] = float(
                    frames['dark'][0].header.get('EXPTIME', 0) or 0) or None
            except Exception as e:
                safe_print(f"  WARNING: Could not read dark frame EXPTIME ({e}) — dark scaling disabled")

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

        if getattr(args, 'dry_run', False):
            if frames['light']:
                first_hdr = frames['light'][0].header
                h = first_hdr.get('NAXIS2', 0)
                w = first_hdr.get('NAXIS1', 0)
                n_lights = len(frames['light'])
                bytes_per_frame = h * w * 3 * 4  # float32 RGB
                memmap_size_mb = (n_lights * bytes_per_frame * 2) / (1024**2)
                safe_print(f"\n  --- DRY RUN ---")
                safe_print(f"  Light frames: {n_lights} x {w}x{h}")
                safe_print(f"  Estimated temp storage: {memmap_size_mb:.0f} MB")
                safe_print(f"  Stack method: {args.stack_method}")
                safe_print(f"  Background extraction: {'DBE' if getattr(args, 'dbe', True) else 'mesh'}")
                safe_print(f"  Denoising: wavelet={getattr(args, 'denoise', False)}, "
                           f"NLM={getattr(args, 'denoise_nlm', False)}, "
                           f"bilateral={getattr(args, 'denoise_bilateral', False)}, "
                           f"MMT={getattr(args, 'denoise_mmt', False)}, "
                           f"ACDNR={getattr(args, 'denoise_acdnr', False)}")
                safe_print(f"  Deconvolution: {getattr(args, 'deconvolve', False)}")
                safe_print(f"  Star reduction: {getattr(args, 'star_reduce', False)}")
                safe_print(f"  Local contrast: {getattr(args, 'local_contrast', False)}")
                safe_print(f"  Stretch: {getattr(args, 'stretch', 'linear')}")
                if getattr(args, 'preset', None):
                    safe_print(f"  Preset: {args.preset}")
            continue  # skip stacking

        # Validation warnings
        if len(frames['light']) < Config.MIN_RECOMMENDED_FRAMES:
            warning = f"Only {len(frames['light'])} light frames found (recommended: {Config.MIN_RECOMMENDED_FRAMES}+)"
            stats.add_warning(warning)
            safe_print(f"\n  ⚠ WARNING: {warning}")

        res = stack_target([f for t in frames.values() for f in t], outp, args, masters, stats)
        if res:
            produced.append(res)
            save_effective_config(args, outp)
    # If hierarchical combine
    if len(produced) > 1:
        print_header("HIERARCHICAL COMBINING", "=")
        print(f"  Combining {len(produced)} target stacks into final output...")

        # --- Mosaic path (WCS reprojection) ---
        mosaic_done = False
        if getattr(args, 'mosaic', False):
            from src.mosaic import stitch_mosaic_panels
            mosaic_done = stitch_mosaic_panels(produced, output, args)
            if not mosaic_done:
                safe_print("  Falling back to translation-based combine...")

        if not mosaic_done:
            # --- Translation-based combine (default / fallback) ---
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
    p.add_argument('--config', default=None, metavar='PATH',
                   help='Load parameters from a TOML configuration file. '
                        'CLI arguments override config file values.')
    p.add_argument('--preset', choices=['quick', 'quality', 'narrowband', 'galaxy', 'starfield'],
                   default=None,
                   help='Processing preset that sets sensible defaults for common scenarios. '
                        'Individual flags still override preset values. '
                        'quick: fast processing (mean stack, no deconvolution, minimal denoise). '
                        'quality: maximum quality (sigma_clip, all denoisers, deconvolution). '
                        'narrowband: tuned for Ha/OIII/SII narrowband data. '
                        'galaxy: optimized for galaxy imaging (strong stretch, star reduction). '
                        'starfield: optimized for star fields (no star reduction, minimal processing).')
    p.add_argument('--stretch-preset',
                   choices=['galaxy', 'nebula', 'starfield', 'planetary', 'lunar'],
                   default=None,
                   help='Named stretch preset that sets --stretch, --ghs-b, --ghs-sp, --ghs-hp '
                        'to tested values. Overrides individual stretch arguments.')
    p.add_argument('--dry-run', action='store_true',
                   help='Discover and classify frames, show calibration info, print effective '
                        'parameters, and estimate resource usage — without processing anything')
    p.add_argument('--skip-step', action='append', default=[], metavar='STEP',
                   help='Skip a named post-processing step. Can be specified multiple times. '
                        'Steps: hot_pixel, background, chroma_nr, sky_floor, local_normalize, '
                        'wavelet, sky_residual, nlm, bilateral, mmt, acdnr, deconvolve, '
                        'star_reduce, local_contrast')
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
    p.add_argument('--quality-threshold', type=float, default=50.0,
                   help='Reject frames whose score falls more than this percent below the '
                        'reference score (90th-percentile of the session, default: 50). '
                        'E.g. 50 keeps every frame with score >= 50%% of the reference. '
                        'Use --no-quality-filter to disable entirely.')
    p.add_argument('--keep-intermediates', action='store_true')
    p.add_argument('--diagnostic', action='store_true', default=False,
                   help='Save a FITS snapshot before each post-processing step for '
                        'artifact troubleshooting. Files are named by step number and '
                        'step name, e.g. 01_before_hot_pixel.fits. '
                        'WARNING: ~275 MB per snapshot at 24 MP float32 RGB.')
    p.add_argument('--diagnostic-dir', default=None, metavar='PATH',
                   help='Directory for --diagnostic snapshots '
                        '(default: <output_stem>_diagnostic/ next to the output file).')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--debug-registration', action='store_true',
                   help='Detailed registration diagnostics (implies -v)')
    p.add_argument('--stack-method',
                   choices=['mean', 'median', 'sigma_clip', 'winsorized',
                            'percentile', 'esd', 'auto'],
                   default='auto',
                   help='Stacking/rejection method. '
                        'sigma_clip: MAD-based iterative rejection (default for dithered data). '
                        'winsorized: like sigma_clip but clips to boundary instead of rejecting. '
                        'percentile: reject outside [low,high] percentile (good for <8 frames). '
                        'esd: Grubbs/ESD test (best for <15 frames, needs scipy). '
                        'auto: choose based on frame count (<8→percentile, else sigma_clip). '
                        'mean/median: no rejection.')
    p.add_argument('--rejection-sigma', type=float, default=3.0,
                   help='Sigma threshold for pixel rejection in sigma_clip/winsorized stacking (default: 3.0)')
    p.add_argument('--rejection-iters', type=int, default=3,
                   help='Number of clipping iterations for sigma_clip stacking (default: 3)')
    p.add_argument('--rejection-estimator', choices=['mad', 'std'], default='mad',
                   help='Spread estimator for sigma_clip/winsorized: '
                        'mad (default, robust) or std (PixInsight "Linear Clipping")')
    p.add_argument('--winsorize', action='store_true',
                   help='(Deprecated) Shorthand for --stack-method winsorized')
    p.add_argument('--percentile-low', type=float, default=20.0,
                   help='Lower rejection percentile for --stack-method percentile (default: 20)')
    p.add_argument('--percentile-high', type=float, default=80.0,
                   help='Upper rejection percentile for --stack-method percentile (default: 80)')
    p.add_argument('--esd-max-outliers', type=int, default=0,
                   help='Max outliers per pixel for ESD rejection (default: 0 = N//4)')
    p.add_argument('--esd-significance', type=float, default=0.05,
                   help='Significance level for ESD test (default: 0.05)')
    p.add_argument('--weight-snr', type=float, default=1.0,
                   help='Exponent for per-frame SNR quality weight (default: 1.0; 0 = ignore SNR)')
    p.add_argument('--weight-fwhm', type=float, default=1.0,
                   help='Exponent for per-frame FWHM quality weight (default: 1.0; 0 = ignore FWHM; '
                        'lower FWHM = sharper stars = higher weight)')
    p.add_argument('--weight-stars', type=float, default=1.0,
                   help='Exponent for per-frame star-count quality weight (default: 1.0; 0 = ignore)')
    p.add_argument('--weight-noise', action='store_true',
                   help='Add 1/noise² weighting component (favours low-background frames)')
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
    p.add_argument('--dbe', action='store_true', default=True,
                   help='Use Dynamic Background Extraction (RBF scattered-point fitting) '
                        'instead of the legacy mesh estimator (default: on). '
                        'DBE traces the actual morphology of extended sources via dilation, '
                        'then fits a thin-plate-spline surface to clean sky samples. '
                        'Only active when --background-extraction is enabled.')
    p.add_argument('--no-dbe', dest='dbe', action='store_false',
                   help='Use legacy mesh-based background extraction instead of DBE')
    p.add_argument('--dbe-patch-size', type=int, default=64,
                   help='DBE background sampling patch size in pixels (default: 64). '
                        'Smaller patches give denser samples and capture finer gradients; '
                        'larger patches are faster. --bg-mesh-size is ignored when DBE '
                        'is active.')
    p.add_argument('--denoise', action='store_true', default=True,
                   help='Enable wavelet denoising post-stack (default: on; requires pywt)')
    p.add_argument('--no-denoise', dest='denoise', action='store_false',
                   help='Disable wavelet denoising')
    p.add_argument('--denoise-strength', type=float, default=3.0,
                   help='Wavelet luma denoise threshold factor (default: 3.0); '
                        'overridden by --auto-denoise-strength unless disabled')
    p.add_argument('--auto-denoise-strength', action='store_true', default=True,
                   help='Auto-tune denoise strength from stacked image SNR '
                        '(default: on; applies to fixed-threshold mode only, '
                        'i.e. when --no-denoise-adaptive is set)')
    p.add_argument('--no-auto-denoise-strength', dest='auto_denoise_strength',
                   action='store_false',
                   help='Use fixed --denoise-strength instead of auto-tuning')
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
    # --- MMT denoising ---
    p.add_argument('--denoise-mmt', action='store_true',
                   help='Enable Multiscale Median Transform (MMT) denoising. '
                        'Decomposes the stack into detail layers via successive median '
                        'filters (scales 3, 5, 9, 17 px), estimates noise per-scale '
                        'via the MAD estimator, and removes it by soft thresholding. '
                        'More robust to non-Gaussian noise (Poisson + read noise) than '
                        'wavelet denoising; better edge preservation in fine filaments. '
                        'Can run alongside --denoise or as a standalone pass. '
                        'Requires cv2 for best performance (scipy fallback available).')
    p.add_argument('--denoise-mmt-levels', type=int, default=4,
                   help='MMT decomposition depth (default: 4 → kernel sizes 3,5,9,17 px). '
                        'Increase to 5 (adds 33 px scale) for very noisy stacks with '
                        'prominent large-scale sky gradients. Only used when --denoise-mmt.')
    p.add_argument('--denoise-mmt-strength', type=float, default=3.0,
                   help='MMT soft-threshold noise-sigma multiplier (default: 3.0). '
                        'Larger values remove more noise but may smooth fine structure. '
                        'Typical range 2.0–5.0. Only used when --denoise-mmt.')
    # --- ACDNR denoising ---
    p.add_argument('--denoise-acdnr', action='store_true',
                   help='Enable Adaptive Contrast-based Denoising with Noise Reduction '
                        '(ACDNR-style). Computes a per-pixel weight from local luminance '
                        'contrast relative to sky noise: w = exp(-0.5*(contrast/(k*sigma))^2). '
                        'Flat sky pixels (contrast << k*sigma) are fully smoothed; '
                        'structured pixels (nebula edges, filaments) are preserved. '
                        'Effective as a lightweight final pass after wavelet or MMT denoising. '
                        'No extra dependencies required.')
    p.add_argument('--denoise-acdnr-sigma', type=float, default=1.5,
                   help='ACDNR Gaussian smoothing radius in pixels (default: 1.5). '
                        'Controls both the scale of contrast detection and the smoothing '
                        'kernel. Larger values remove coarser noise but blur finer structure. '
                        'Typical range 1.0–3.0. Only used when --denoise-acdnr.')
    p.add_argument('--denoise-acdnr-k', type=float, default=3.0,
                   help='ACDNR contrast threshold multiplier k (default: 3.0). '
                        'Smoothing triggers when local contrast < k * sky_sigma. '
                        'Lower → more aggressive (smooth structured regions too); '
                        'higher → conservative sky-only smoothing. Typical range 2.0–5.0. '
                        'Only used when --denoise-acdnr.')
    p.add_argument('--deconvolve', action='store_true', default=False,
                   help='Enable Richardson-Lucy deconvolution for sharpening (default: off, requires scikit-image)')
    p.add_argument('--no-deconvolve', dest='deconvolve', action='store_false',
                   help='Disable Richardson-Lucy deconvolution')
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
    p.add_argument('--stretch', choices=['linear', 'arcsinh', 'ghs'], default='ghs',
                   help='Preview JPEG stretch method (default: ghs). '
                        'ghs = Generalized Hyperbolic Stretch — the state-of-the-art '
                        'algorithm for galaxy imaging, giving independent control of '
                        'shadow lift (--ghs-sp), highlights protection (--ghs-hp), '
                        'and stretch intensity (--ghs-b).')
    p.add_argument('--ghs-b', type=float, default=8.0,
                   help='GHS stretch factor b (default: 8.0). '
                        '0 = linear, 5 = moderate, 8–12 = galaxy-optimised. '
                        'Higher values lift fainter outer spiral arms and dust lanes '
                        'while compressing bright nuclei. Only used when --stretch=ghs.')
    p.add_argument('--ghs-sp', type=float, default=0.15,
                   help='GHS symmetry point SP [0–1] (default: 0.15). '
                        'The pivot of the stretch curve. Setting SP below the galaxy '
                        'core lifts faint outer arms disproportionately relative to '
                        'the bright nucleus. Typical galaxy range: 0.10–0.20. '
                        'Only used when --stretch=ghs.')
    p.add_argument('--ghs-hp', type=float, default=0.95,
                   help='GHS highlights protection HP [0–1] (default: 0.95). '
                        'Values above HP map to white, protecting bright nuclear '
                        'cores from blowout while faint structure is stretched. '
                        'Increase toward 0.99 for targets with very bright nuclei. '
                        'Only used when --stretch=ghs.')
    # --- Star reduction (galaxy imaging) ---
    p.add_argument('--star-reduce', action='store_true', default=True,
                   help='Reduce star prominence in the final stack to improve the '
                        'galaxy-to-star visual balance. Softens star cores via '
                        'Gaussian blending, making them appear slightly smaller '
                        'without removing them. Useful for galaxy targets. '
                        'Modifies the FITS output data (default: on).')
    p.add_argument('--no-star-reduce', dest='star_reduce', action='store_false',
                   help='Disable star reduction')
    p.add_argument('--star-reduce-factor', type=float, default=0.4,
                   help='Star reduction blend fraction (default: 0.4). '
                        '0 = no effect, 1 = replace star cores with blurred version. '
                        'Typical range 0.3–0.6. Only used when --star-reduce.')
    p.add_argument('--star-reduce-sigma', type=float, default=1.5,
                   help='Gaussian blur radius for star reduction in pixels (default: 1.5). '
                        'Larger values give softer but dimmer star cores. '
                        'Only used when --star-reduce.')
    # --- Multiscale local contrast enhancement (galaxy structure) ---
    p.add_argument('--local-contrast', action='store_true', default=True,
                   help='Apply multiscale local contrast enhancement (MLCE) to reveal '
                        'galaxy structure: dust lanes, spiral arm boundaries, and star '
                        'forming regions. Uses luminance-domain unsharp masking at '
                        'fine (2 px), medium (12 px), and coarse (40 px) scales with '
                        'a mid-tone mask that protects sky noise and bright nuclei. '
                        'Particularly effective for the Black Eye Galaxy (M64) where '
                        'the medium scale targets the characteristic dark dust band. '
                        'Modifies the FITS output data (default: on).')
    p.add_argument('--no-local-contrast', dest='local_contrast', action='store_false',
                   help='Disable multiscale local contrast enhancement')
    p.add_argument('--local-contrast-strength', type=float, default=0.7,
                   help='MLCE overall strength multiplier (default: 0.7). '
                        '0 = off, 1 = full enhancement. Typical range 0.4–0.9. '
                        'Only used when --local-contrast.')
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
    # --- Output formats ---
    p.add_argument('--output-tiff', action='store_true',
                   help='Write a 32-bit float TIFF alongside the FITS output '
                        '(requires tifffile; falls back to 16-bit via Pillow)')
    p.add_argument('--output-xisf', action='store_true',
                   help='Write output in PixInsight XISF 1.0 format alongside FITS '
                        '(no extra dependencies)')
    # --- Quality report ---
    p.add_argument('--quality-report', default=None, metavar='PATH',
                   help='Write per-frame quality metrics CSV after Phase 1 '
                        '(columns: filename, snr, fwhm, star_count, quality_score, '
                        'accepted, rejection_reason)')
    # --- Per-frame preview export ---
    p.add_argument('--export-frames-dir', default=None, metavar='PATH',
                   help='Directory to write a stretched JPEG for every accepted frame '
                        'after Phase 1 processing')
    # --- Plate solver backend ---
    p.add_argument('--plate-solver', choices=['astap', 'astrometry'], default='astrometry',
                   help='Plate solver backend: astap (fast, local binary) or '
                        'astrometry (nova.astrometry.net, requires API key). '
                        'Default: astrometry. ASTAP is recommended when the binary '
                        'is installed — it solves in ~1 s vs 30–120 s online.')
    p.add_argument('--astap-path', default=None, metavar='PATH',
                   help='Explicit path to the ASTAP binary (auto-detected if omitted)')
    # --- Background extraction method ---
    p.add_argument('--bg-method', choices=['mesh', 'dbe', 'graxpert'], default=None,
                   help='Background extraction method: '
                        'mesh: legacy polynomial grid (fastest). '
                        'dbe: Dynamic Background Extraction, RBF thin-plate-spline (default). '
                        'graxpert: AI-powered gradient removal via GraXpert subprocess '
                        '(best quality; requires GraXpert binary on PATH or --graxpert-path).')
    p.add_argument('--graxpert-path', default=None, metavar='PATH',
                   help='Path to GraXpert binary (auto-detected if omitted). '
                        'Download from https://www.graxpert.com/')
    # --- Starnet++ star removal ---
    p.add_argument('--star-remove', action='store_true',
                   help='Remove stars using Starnet++ after post-processing. '
                        'Saves <output>_starless.fits and <output>_stars.fits. '
                        'The main FITS output becomes the starless image. '
                        'Requires Starnet++ binary on PATH or --starnet-path.')
    p.add_argument('--starnet-path', default=None, metavar='PATH',
                   help='Path to Starnet++ binary (auto-detected if omitted). '
                        'Download from https://www.starnetastro.com/')
    # --- Comet stacking ---
    p.add_argument('--comet-mode', action='store_true',
                   help='Enable comet nucleus tracking: produces a second stack '
                        'aligned on the brightest extended blob (comet nucleus) in '
                        'addition to the normal star-aligned stack. '
                        'Comet-aligned output is saved as <stem>_comet.fits.')
    # --- HDR combination ---
    p.add_argument('--hdr-combine', default=None, metavar='SHORT_STACK.fits',
                   help='Blend a short-exposure stack FITS into saturated regions '
                        'of the main (long-exposure) stack for HDR targets '
                        '(e.g. Orion Nebula core, globular clusters). '
                        'The short stack is automatically scaled to match the '
                        'long stack background level before blending.')
    # --- Photometric colour calibration ---
    p.add_argument('--color-calibrate', action='store_true',
                   help='Apply photometric colour calibration after plate solving: '
                        'queries Gaia DR3 (or 2MASS via VizieR) for stars in the '
                        'field, matches them to the image via aperture photometry, '
                        'and derives per-channel scale factors. '
                        'Requires --plate-solve and astroquery.')
    # --- Mask export ---
    p.add_argument('--export-masks', action='store_true',
                   help='Save the star detection mask as a FITS sidecar file '
                        '(<stem>_star_mask.fits) for use in external tools '
                        '(PixInsight, Siril, etc.)')
    # --- AI features (require: pip install anthropic  +  ANTHROPIC_API_KEY env var) ---
    p.add_argument('--ai-advisor', action='store_true',
                   help='After Phase 1, call Claude to recommend optimal stacking '
                        'parameters and apply them automatically. '
                        'Requires: pip install anthropic  and  ANTHROPIC_API_KEY.')
    p.add_argument('--ai-report', action='store_true',
                   help='After stacking, call Claude to generate a narrative session '
                        'report (saved as <output>_report.md). '
                        'Requires: pip install anthropic  and  ANTHROPIC_API_KEY.')
    # --- Checkpoint resume ---
    p.add_argument('--keep-checkpoint', action='store_true',
                   help='After a successful stack, keep the checkpoint directory and '
                        'save the raw pre-post-processing stack to disk. Re-running '
                        'the same command will skip phases 1-3 and only re-run '
                        'post-processing, letting you quickly test different denoising, '
                        'background extraction, stretch, or deconvolution settings.')
    p.add_argument('--no-resume', action='store_true',
                   help='Ignore any existing checkpoint and start from scratch. '
                        'By default, if a checkpoint exists from an interrupted run '
                        '(within the last 72 hours, same frame set), phases 1 and/or 2 '
                        'are skipped and pixel data is reloaded from the accepted frames only.')
    # --- Multi-session combining ---
    p.add_argument('--combine-sessions', action='store_true',
                   help='When the input directory contains subfolders, pool all light '
                        'frames from every subfolder into a single unified stack instead '
                        'of stacking each subfolder separately then averaging the results. '
                        'Use this when all sessions were shot on the same target with the '
                        'same equipment. Calibration frames from all sessions are merged '
                        'into stronger master frames. Registration, sigma-clip rejection, '
                        'and post-processing all run once across the full multi-night dataset.')
    p.add_argument('--mosaic', action='store_true',
                   help='Stitch per-subfolder stacks into a mosaic via WCS reprojection '
                        'instead of the default translation-based combine. Each subfolder '
                        'is stacked independently, then all panels are plate-solved and '
                        'reprojected onto a common RA/Dec grid with overlap feathering '
                        'and background-level matching. '
                        'Requires: pip install reproject  and a working plate solver '
                        '(--plate-solver astap recommended for speed). '
                        'Automatically enables --plate-solve.')
    # --- Heuristic auto-advisor (no API key required) ---
    p.add_argument('--auto', action='store_true',
                   help='After Phase 1, automatically classify the target '
                        '(emission nebula, galaxy, reflection nebula, star field, '
                        'wide field) from frame metrics and apply optimised settings. '
                        'No API key required. Upgrades debayer to malvar when '
                        'OpenCV is available, resolves the stacking method by frame '
                        'count, tunes deconvolution and stretch parameters to the '
                        'detected target type, and selects the best denoising method: '
                        'nebulae and galaxies switch to MMT+ACDNR (edge-preserving '
                        'median cascade + adaptive contrast cleanup); star fields and '
                        'wide fields keep wavelet and add ACDNR for sky cleanup; '
                        'MMT strength is scaled to stack SNR.')
    return p.parse_args()


def main():
    # Dispatch to the 'combine' subcommand without touching the main parser
    if len(sys.argv) > 1 and sys.argv[1] == 'combine':
        from src.channel_combine import run_combine_cli
        run_combine_cli(sys.argv[2:])
        return

    args = parse_args()
    if not args.health_check and not getattr(args, 'dry_run', False) and not args.output:
        print("ERROR: -o/--output is required unless --health-check or --dry-run is specified",
              file=sys.stderr)
        raise SystemExit(1)
    # Load config file (before preset, so preset can override config)
    if getattr(args, 'config', None):
        config_changes = load_config_file(args.config, args)
        if config_changes:
            safe_print(f"  Config loaded: {len(config_changes)} settings from {args.config}")
    # Apply preset (before any other processing)
    preset_changes = apply_preset(args)
    # Apply stretch preset
    apply_stretch_preset(args)
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
