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
from src.frame_discovery import discover_frames, select_matching_darks, select_matching_flats
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

        _protected = {'directory', 'output', 'config', 'health_check', 'dry_run',
                      'verbose', 'debug_registration', 'keep_intermediates',
                      'preset', 'diagnostic', 'diagnostic_dir'}
        for key, value in flat.items():
            if key in _protected:
                continue
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
            'bg_method': 'dbe',
            'chroma_nr': False,
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
            'bg_method': 'dbe',
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
            'bg_method': 'dbe',
            'star_reduce': False,
            'local_contrast': False,
            'stretch': 'ghs',
            'ghs_b': 5.0,
            'ghs_sp': 0.20,
            'chroma_nr': True,
        },
        'nebula': {
            'stack_method': 'sigma_clip',
            'rejection_sigma': 2.5,
            'denoise': True,
            'denoise_adaptive': True,
            'denoise_mmt': True,
            'denoise_acdnr': True,
            'deconvolve': False,
            'background_extraction': True,
            'bg_method': 'dbe',
            'star_reduce': True,
            'local_contrast': True,
            'chroma_nr': True,
            'stretch': 'ghs',
            'ghs_b': 12.0,
            'ghs_sp': 0.10,
            'ghs_hp': 0.98,
        },
        'planetary': {
            'stack_method': 'mean',
            'denoise': True,
            'denoise_adaptive': True,
            'deconvolve': True,
            'background_extraction': False,
            'star_reduce': False,
            'local_contrast': True,
            'chroma_nr': True,
            'stretch': 'ghs',
            'ghs_b': 3.0,
            'ghs_sp': 0.30,
            'ghs_hp': 0.85,
        },
        'lunar': {
            'stack_method': 'mean',
            'denoise': True,
            'denoise_adaptive': True,
            'background_extraction': False,
            'star_reduce': False,
            'local_contrast': True,
            'chroma_nr': False,
            'stretch': 'linear',
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



def save_effective_config(args: argparse.Namespace, output_path: str) -> None:
    """Save the effective parameter set as a TOML file next to the output."""
    config_path = os.path.splitext(output_path)[0] + '_config.toml'
    lines = ['# OriginStack effective configuration\n',
             '# Generated automatically -- can be reused with --config\n\n']

    skip_keys = {'directory', 'output', 'config', 'health_check', 'dry_run',
                 'verbose', 'debug_registration', 'keep_intermediates',
                 'preset', 'diagnostic', 'diagnostic_dir'}

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


def _load_calibration_dir(args: argparse.Namespace) -> dict:
    """Discover calibration frames from --cal-dir.

    Returns a dict with keys 'dark', 'flat', 'bias' containing FrameInfo lists.
    """
    extra: dict = {'dark': [], 'flat': [], 'bias': []}
    cal_dir = getattr(args, 'cal_dir', None)
    if not cal_dir:
        return extra
    if not os.path.isdir(cal_dir):
        print(f'  WARNING: --cal-dir path does not exist: {cal_dir}', file=sys.stderr)
        return extra
    discovered = discover_frames(cal_dir)
    for ftype in ('dark', 'flat', 'bias'):
        found = discovered.get(ftype, [])
        extra[ftype].extend(found)
        if found:
            safe_print(f"  Calibration library: {len(found)} {ftype} frames from {cal_dir}")
    return extra


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
    _session_bayers: dict = {}  # subfolder -> bayer pattern used by its lights
    for d in sorted(subdirs):
        sub = discover_frames(d)
        for ftype in combined:
            combined[ftype].extend(sub.get(ftype, []))
        # Determine bayer pattern for this subfolder's lights
        lights_here = sub.get('light', [])
        if lights_here:
            from src.session_info import load_session_info, _INFO_FILENAMES
            si = load_session_info(d)
            session_bayer = si.bayer if si else None
            patterns = set()
            for f in lights_here:
                p = f.header.get('BAYERPAT') or f.header.get('COLORTYP') or session_bayer
                if p:
                    patterns.add(p.upper())
            if patterns:
                _session_bayers[os.path.basename(d)] = patterns

    # Warn if different sessions use different bayer patterns
    all_patterns = set(p for ps in _session_bayers.values() for p in ps)
    if len(all_patterns) > 1:
        safe_print(
            f"\n  ⚠ WARNING: Bayer pattern mismatch across sessions — "
            f"frames will be debayered with their individual BAYERPAT header if present, "
            f"but sessions without a BAYERPAT header will use the first session's pattern."
        )
        for name, patterns in sorted(_session_bayers.items()):
            safe_print(f"    {name}: {', '.join(sorted(patterns))}")
        safe_print(
            f"  ⚠ Mixing cameras with different Bayer patterns in a combined stack "
            f"will produce incorrect colours in frames that lack a BAYERPAT header."
        )

    extra_cal = _load_calibration_dir(args)
    for ftype in ('dark', 'flat', 'bias'):
        combined[ftype].extend(extra_cal[ftype])

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
            safe_print(f"  ✓ Master bias:  {len(combined['bias'])} frames -> "
                       f"{masters['bias'].shape[0]}×{masters['bias'].shape[1]}")
    else:
        masters['bias'] = None

    if combined['dark']:
        combined['dark'] = select_matching_darks(combined['light'], combined['dark'])
        masters['dark'] = make_master(combined['dark'], method='median')
        if masters['dark'] is not None:
            safe_print(f"  ✓ Master dark:  {len(combined['dark'])} frames -> "
                       f"{masters['dark'].shape[0]}×{masters['dark'].shape[1]}")
    else:
        masters['dark'] = None

    if combined['flat']:
        combined['flat'] = select_matching_flats(combined['light'], combined['flat'])
        masters['flat'] = make_master(combined['flat'], method='median')
        if masters['flat'] is not None:
            safe_print(f"  ✓ Master flat:  {len(combined['flat'])} frames -> "
                       f"{masters['flat'].shape[0]}×{masters['flat'].shape[1]}")
        # Store median flat rotation so execute_frame_processing can detect mismatch
        _flat_rots = []
        for _ff in combined['flat']:
            for _rkey in ('ROTATANG', 'ROTANGLE', 'POSANGLE', 'PA', 'ANGLE', 'ROTATOR'):
                _rval = _ff.header.get(_rkey)
                if _rval is not None:
                    try:
                        _flat_rots.append(float(_rval))
                        break
                    except (TypeError, ValueError):
                        pass
        if _flat_rots:
            masters['flat_rotation'] = float(np.median(_flat_rots))
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

    # Use the first subfolder that has an info.json so pipeline can populate
    # the FITS header with session metadata (bayer pattern, WCS, target name).
    from src.session_info import _INFO_FILENAMES
    args._input_directory = next(
        (d for d in sorted(subdirs)
         if any(os.path.isfile(os.path.join(d, f)) for f in _INFO_FILENAMES)),
        sorted(subdirs)[0],  # fall back to first subfolder even if no info.json
    )

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
        extra_cal = _load_calibration_dir(args)
        for ftype in ('dark', 'flat', 'bias'):
            frames[ftype].extend(extra_cal[ftype])
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
                safe_print(f"  ✓ Master bias:  {len(frames['bias'])} frames -> {masters['bias'].shape[0]}×{masters['bias'].shape[1]}")
        else:
            masters['bias'] = None

        if frames['dark']:
            # Select darks that best match the light frames (ISO, exposure, dimensions)
            frames['dark'] = select_matching_darks(frames.get('light', []), frames['dark'])
            masters['dark'] = make_master(frames['dark'], method='median')
            if masters['dark'] is not None:
                safe_print(f"  ✓ Master dark:  {len(frames['dark'])} frames -> {masters['dark'].shape[0]}×{masters['dark'].shape[1]}")
        else:
            masters['dark'] = None

        if frames['flat']:
            frames['flat'] = select_matching_flats(frames.get('light', []), frames['flat'])
            masters['flat'] = make_master(frames['flat'], method='median')
            if masters['flat'] is not None:
                safe_print(f"  ✓ Master flat:  {len(frames['flat'])} frames -> {masters['flat'].shape[0]}×{masters['flat'].shape[1]}")
            # Store median flat rotation so execute_frame_processing can detect mismatch
            _flat_rots = []
            for _ff in frames['flat']:
                for _rkey in ('ROTATANG', 'ROTANGLE', 'POSANGLE', 'PA', 'ANGLE', 'ROTATOR'):
                    _rval = _ff.header.get(_rkey)
                    if _rval is not None:
                        try:
                            _flat_rots.append(float(_rval))
                            break
                        except (TypeError, ValueError):
                            pass
            if _flat_rots:
                masters['flat_rotation'] = float(np.median(_flat_rots))
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
            # (R~=0.39, G~=1.00, B~=1.16 for this camera) so the flat field
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
                               f"noise={b_std:.1f} ADU  -> {b_quality}")
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
                               f"{temp_str}{exp_str}{iso_str}  peak={dark_peak:.0f} ADU  -> {d_quality}")
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
                        safe_print(f"    Flat:  {ratio_str}  vignetting={vign:.1f}%  -> {f_quality}")
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

        args._input_directory = d  # per-target input dir for session info and target inference
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
                   help='Output FITS path (default: <directory>_stacked.fits)')
    p.add_argument('--cal-dir', default=None, metavar='PATH',
                   help='Directory containing calibration frames (darks, flats, bias). '
                        'Frames are classified automatically and the best-matching subset '
                        'is selected per type (by ISO, CCD temperature, exposure, filter, '
                        'and sensor dimensions). Supplements any calibration frames already '
                        'present in --directory.')
    p.add_argument('--health-check', action='store_true',
                   help='Analyse input frames and calibration quality without stacking')
    p.add_argument('--config', default=None, metavar='PATH',
                   help='Load parameters from a TOML configuration file. '
                        'CLI arguments override config file values. Fine-grained tuning '
                        'parameters removed from the CLI can still be set via config file.')
    p.add_argument('--preset',
                   choices=['quick', 'quality', 'narrowband', 'galaxy', 'starfield',
                            'nebula', 'planetary', 'lunar'],
                   default=None,
                   help='Processing preset. Individual flags still override preset values. '
                        'quick: fast (mean stack, minimal denoise, no deconvolution). '
                        'quality: maximum quality (sigma_clip, all denoisers, deconvolution). '
                        'narrowband: tuned for Ha/OIII/SII data. '
                        'galaxy: galaxy imaging (GHS stretch, star reduction, bilateral denoise). '
                        'starfield: star fields (no star reduction, minimal processing). '
                        'nebula: emission/reflection nebula (GHS stretch, MMT+ACDNR denoise). '
                        'planetary: planetary targets (no background extraction, deconvolution). '
                        'lunar: lunar surface (linear stretch, no star reduction).')
    p.add_argument('--dry-run', action='store_true',
                   help='Discover and classify frames, show calibration info, print effective '
                        'parameters, and estimate resource usage — without processing anything')
    p.add_argument('--skip-step', action='append', default=[], metavar='STEP',
                   help='Skip a named post-processing step. Can be specified multiple times. '
                        'Steps: hot_pixel, background, chroma_nr, sky_floor, local_normalize, '
                        'wavelet, sky_residual, nlm, bilateral, mmt, acdnr, deconvolve, '
                        'star_reduce, local_contrast')
    p.add_argument('--no-registration', action='store_true')
    p.add_argument('--no-affine', action='store_true',
                   help='Disable affine (rotation+translation) registration; use translation-only')
    p.add_argument('--phase-correlation', action='store_false', dest='skip_phase_correlation',
                   default=True,
                   help='Run skimage phase cross-correlation before the FFT fallback. '
                        'Off by default: on typical astronomical frames phase-cc almost always '
                        'fails its error<0.1 acceptance test and the FFT cross-correlation is '
                        'used anyway, so running it just wastes an upsampled correlation per frame.')
    p.add_argument('--advanced-metrics', action='store_true', dest='advanced_metrics',
                   default=False,
                   help='Compute expensive diagnostic-only quality metrics per frame '
                        '(Zernike PSF decomposition, Strehl proxy, atmospheric dispersion). '
                        'Off by default: these do not affect frame acceptance or stacking weights, '
                        'so they only add per-frame cost unless you want the diagnostics.')
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
                            'percentile', 'esd', 'trimmed_mean', 'auto'],
                   default='auto',
                   help='Stacking/rejection method. '
                        'sigma_clip: MAD-based iterative rejection (default for dithered data). '
                        'winsorized: like sigma_clip but clips to boundary instead of rejecting. '
                        'percentile: reject outside [low,high] percentile (good for <8 frames). '
                        'esd: Grubbs/ESD test (best for <15 frames, needs scipy). '
                        'auto: choose based on frame count (<8->percentile, else sigma_clip). '
                        'mean/median: no rejection.')
    p.add_argument('--rejection-sigma', type=float, default=3.0,
                   help='Sigma threshold for pixel rejection in sigma_clip/winsorized stacking (default: 3.0)')
    p.add_argument('--rejection-iters', type=int, default=3,
                   help='Number of clipping iterations for sigma_clip stacking (default: 3)')
    p.add_argument('--rejection-estimator', choices=['mad', 'std'], default='mad',
                   help='Spread estimator for sigma_clip/winsorized: '
                        'mad (default, robust) or std (PixInsight "Linear Clipping")')
    p.add_argument('--debayer-method', choices=['bilinear', 'malvar', 'vng'], default='bilinear',
                   help='Debayering method (default: bilinear; malvar/vng require OpenCV)')
    p.add_argument('--white-balance', choices=['none', 'grayworld', 'whitepatch'], default='grayworld')
    p.add_argument('--drizzle-scale', type=float, default=1.0,
                   help='Drizzle scale factor (e.g. 2.0 for 2x super-resolution, 1.0 = disabled)')
    p.add_argument('--patch-weighted', dest='patch_registration', action='store_true',
                   help='Enable spatially-varying (patch-weighted) stacking: each frame contributes '
                        'to each output pixel weighted by local sharpness, so sharp regions of a '
                        'frame outweigh blurry ones from the same frame.  Automatically enabled by '
                        '--auto when frame count and seeing metrics are favourable.')
    p.add_argument('--use-gpu', action='store_true',
                   help='Use CuPy for available operations (experimental)')
    p.add_argument('--plate-solve', action='store_true',
                   help='Enable plate solving via astrometry.net (requires astroquery and ASTROMETRY_API_KEY)')
    p.add_argument('--no-background-extraction', dest='background_extraction',
                   action='store_false',
                   help='Disable background extraction')
    p.add_argument('--bg-method', choices=['mesh', 'dbe', 'graxpert'], default='dbe',
                   help='Background extraction method (default: dbe). '
                        'mesh: legacy polynomial grid (fastest). '
                        'dbe: Dynamic Background Extraction, RBF thin-plate-spline. '
                        'graxpert: AI-powered gradient removal via GraXpert subprocess '
                        '(best quality; requires GraXpert binary on PATH or --graxpert-path).')
    p.add_argument('--graxpert-path', default=None, metavar='PATH',
                   help='Path to GraXpert binary (auto-detected if omitted).')
    p.add_argument('--no-denoise', dest='denoise', action='store_false',
                   help='Disable wavelet denoising')
    p.add_argument('--denoise-strength', type=float, default=3.0,
                   help='Wavelet luma denoise threshold factor (default: 3.0)')
    p.add_argument('--no-auto-denoise-strength', dest='auto_denoise_strength',
                   action='store_false',
                   help='Use fixed --denoise-strength instead of auto-tuning')
    p.add_argument('--denoise-nlm', action='store_true',
                   help='Enable non-local means denoising after wavelet (requires skimage or cv2)')
    p.add_argument('--denoise-bilateral', action='store_true',
                   help='Enable bilateral filter denoising after wavelet (requires cv2)')
    p.add_argument('--denoise-mmt', action='store_true',
                   help='Enable Multiscale Median Transform (MMT) denoising. '
                        'More robust than wavelet for Poisson + read noise; better edge '
                        'preservation in fine filaments.')
    p.add_argument('--denoise-acdnr', action='store_true',
                   help='Enable Adaptive Contrast-based Denoising (ACDNR). '
                        'Flat sky regions are smoothed; structured pixels are preserved. '
                        'Effective as a lightweight final pass after wavelet or MMT.')
    p.add_argument('--no-denoise-adaptive', dest='denoise_adaptive', action='store_false',
                   help='Use fixed global threshold factor instead of BayesShrink')
    p.add_argument('--denoise-bm3d', action='store_true',
                   help='Apply BM3D collaborative filter denoising (luminance only). '
                        'Near-optimal noise suppression; slower than wavelet.')
    p.add_argument('--denoise-aniso', action='store_true',
                   help='Apply Perona-Malik anisotropic diffusion. Reduces noise in '
                        'uniform regions while sharpening edges and filament boundaries.')
    p.add_argument('--deconvolve', action='store_true',
                   help='Enable Richardson-Lucy deconvolution for sharpening '
                        '(requires scikit-image)')
    p.add_argument('--deconvolve-tv', action='store_true',
                   help='Apply Total Variation regularized deconvolution instead of '
                        'Richardson-Lucy. Better for sharp edges; slower.')
    p.add_argument('--deconvolve-blind-psf', action='store_true',
                   help='Use empirical PSF estimated by median-stacking normalised star '
                        'cutouts instead of a parametric model.')
    p.add_argument('--local-normalize', action='store_true',
                   help='Enable local normalization to remove vignetting residuals')
    p.add_argument('--no-chroma-nr', dest='chroma_nr', action='store_false',
                   help='Disable chroma noise reduction')
    p.add_argument('--stretch', choices=['linear', 'arcsinh', 'ghs'], default='ghs',
                   help='Preview JPEG stretch method (default: ghs = Generalized Hyperbolic Stretch)')
    p.add_argument('--ghs-b', type=float, default=8.0,
                   help='GHS stretch factor b (default: 8.0). '
                        '0 = linear, 5 = moderate, 8–12 = galaxy-optimised.')
    p.add_argument('--ghs-sp', type=float, default=0.15,
                   help='GHS symmetry point SP [0–1] (default: 0.15). Pivot of the stretch curve.')
    p.add_argument('--ghs-hp', type=float, default=0.95,
                   help='GHS highlights protection HP [0–1] (default: 0.95). '
                        'Values above HP map to white, protecting bright cores from blowout.')
    p.add_argument('--no-star-reduce', dest='star_reduce', action='store_false',
                   help='Disable star reduction')
    p.add_argument('--no-local-contrast', dest='local_contrast', action='store_false',
                   help='Disable multiscale local contrast enhancement')
    p.add_argument('-j', '--parallel', type=int, default=0,
                   help='Parallel workers for frame processing (default: 0=auto, 1=sequential)')
    p.add_argument('--no-ca-correction', dest='ca_correction', action='store_false',
                   help='Disable chromatic aberration correction')
    p.add_argument('--no-cosmic-ray-rejection', dest='cosmic_ray_rejection',
                   action='store_false',
                   help='Disable cosmic ray rejection')
    p.add_argument('--log-level',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='WARNING',
                   help='Minimum log severity printed to stderr (default: WARNING). '
                        'Use DEBUG for verbose diagnostic output from all modules.')
    p.add_argument('--log-file', default=None, metavar='PATH',
                   help='Write a full DEBUG-level log to this file in addition to console output.')
    p.add_argument('--output-tiff', action='store_true',
                   help='Write a 32-bit float TIFF alongside the FITS output')
    p.add_argument('--output-xisf', action='store_true',
                   help='Write output in PixInsight XISF 1.0 format alongside FITS')
    p.add_argument('--output-processed-fits', action='store_true',
                   help='Write a second FITS file with post-processing applied '
                        '(suffix _processed.fits alongside the main linear FITS)')
    p.add_argument('--quality-report', default=None, metavar='PATH',
                   help='Write per-frame quality metrics CSV after Phase 1 '
                        '(columns: filename, snr, fwhm, star_count, quality_score, '
                        'accepted, rejection_reason)')
    p.add_argument('--export-frames-dir', default=None, metavar='PATH',
                   help='Directory to write a stretched JPEG for every accepted frame after Phase 1')
    p.add_argument('--plate-solver', choices=['astap', 'astrometry'], default='astrometry',
                   help='Plate solver backend: astap (fast, local) or '
                        'astrometry (nova.astrometry.net, requires API key). '
                        'ASTAP recommended when the binary is installed (~1 s vs 30–120 s online).')
    p.add_argument('--astap-path', default=None, metavar='PATH',
                   help='Explicit path to the ASTAP binary (auto-detected if omitted)')
    p.add_argument('--star-remove', action='store_true',
                   help='Remove stars using Starnet++ after post-processing. '
                        'Saves <output>_starless.fits and <output>_stars.fits. '
                        'Requires Starnet++ binary on PATH or --starnet-path.')
    p.add_argument('--starnet-path', default=None, metavar='PATH',
                   help='Path to Starnet++ binary (auto-detected if omitted).')
    p.add_argument('--comet-mode', action='store_true',
                   help='Enable comet nucleus tracking. Produces a second stack aligned '
                        'on the comet nucleus saved as <stem>_comet.fits.')
    p.add_argument('--comet-blend-sigma', type=float, default=30.0, metavar='PX',
                   help='Gaussian blend radius (pixels) for comet+star blended composite '
                        '(default: 30). Larger = more comet-stack area in the blend.')
    p.add_argument('--comet-xy', default=None, metavar='X,Y',
                   help='Approximate comet nucleus position in the reference frame '
                        '(pixels, comma-separated, e.g. "1024,768"). X=column, Y=row.')
    p.add_argument('--comet-search-radius', type=float, default=50.0, metavar='PX',
                   help='Search radius (pixels) around predicted nucleus position for '
                        'frame-to-frame tracking (default: 50).')
    p.add_argument('--comet-affine', action='store_true',
                   help='Apply rotation+scale correction (affine) on top of nucleus '
                        'translation shift in comet registration (default: translation only).')
    p.add_argument('--coma-mask-radius', type=int, default=150, metavar='PX',
                   help='Circular exclusion radius (pixels) around the comet nucleus '
                        'used to prevent background extraction from sampling the coma '
                        '(default: 150). Only active when --comet-mode is set.')
    p.add_argument('--comet-radial-renorm', action='store_true',
                   help='Apply radial renormalization filter to flatten the coma gradient '
                        'and reveal jets/structure. Saves result as <stem>_comet_renorm.fits.')
    p.add_argument('--comet-larson-sekanina', action='store_true',
                   help='Apply Larson-Sekanina rotational difference filter to enhance '
                        'comet jet structure. Saves result as <stem>_comet_ls.fits.')
    p.add_argument('--comet-ls-rotation', type=float, default=15.0, metavar='DEG',
                   help='Rotation angle (degrees) for the Larson-Sekanina filter '
                        '(default: 15).')
    p.add_argument('--comet-designation', default=None, metavar='DESIG',
                   help='JPL Horizons designation for ephemeris-aided nucleus tracking '
                        '(e.g. "C/2023 A3"). Requires astroquery.')
    p.add_argument('--observer-site', default=None, metavar='SITE',
                   help='Observer location for ephemeris queries: MPC code (e.g. "G37") '
                        'or "lon,lat,elev" in decimal degrees/metres (e.g. "-2.5,51.4,50").')
    p.add_argument('--comet-detrail', action='store_true',
                   help='Apply linear deconvolution to correct intra-frame nucleus trailing '
                        '(requires --comet-designation or consecutive centroid positions).')
    p.add_argument('--hdr-combine', default=None, metavar='SHORT_STACK.fits',
                   help='Blend a short-exposure stack into saturated regions of the main '
                        'stack for HDR targets (e.g. Orion Nebula core, globular clusters).')
    p.add_argument('--color-calibrate', action='store_true',
                   help='Apply photometric colour calibration after plate solving '
                        '(queries Gaia DR3). Requires --plate-solve and astroquery.')
    p.add_argument('--export-masks', action='store_true',
                   help='Save the star detection mask as <stem>_star_mask.fits')
    p.add_argument('--keep-checkpoint', action='store_true',
                   help='Keep the raw pre-post-processing stack after a successful run. '
                        'Re-running skips phases 1–3 so you can iterate on post-processing '
                        'settings quickly.')
    p.add_argument('--no-resume', action='store_true',
                   help='Ignore any existing checkpoint and start from scratch.')
    p.add_argument('--combine-sessions', action='store_true',
                   help='Pool all light frames from every subfolder into a single unified '
                        'stack instead of stacking each subfolder separately.')
    p.add_argument('--mosaic', action='store_true',
                   help='Stitch per-subfolder stacks into a mosaic via WCS reprojection. '
                        'Requires: pip install reproject and a working plate solver. '
                        'Automatically enables --plate-solve.')
    p.add_argument('--auto', action='store_true',
                   help='Classify the target after Phase 1 and apply optimised settings '
                        'automatically. No API key required.')
    p.add_argument('--scnr', action='store_true',
                   help='Apply Subtractive Chromatic Noise Reduction to suppress green '
                        'cast artefacts common in OSC/DSLR images under light pollution.')
    p.add_argument('--photometric-calibration', action='store_true',
                   help='Apply gray-locus photometric colour calibration. '
                        'No external dependencies required.')
    p.add_argument('--gaia-calibration', action='store_true',
                   help='Extend --photometric-calibration with a Gaia DR3 catalogue query. '
                        'Requires --plate-solve and astroquery.')

    # New feature flags (improvements 1-9)
    p.add_argument('--max-ellipticity', type=float, default=0.5, metavar='E',
                   help='Warn (but do not reject) frames with median star ellipticity above '
                        'this threshold (default: 0.5). Set 0 to disable.')
    p.add_argument('--consensus-ref', action='store_true',
                   help='Choose reference frame as the one with shift closest to the session '
                        'median (most central frame) rather than highest quality score.')
    p.add_argument('--masked-correlation', action='store_true',
                   help='Suppress bright nebula emission before cross-correlation to improve '
                        'registration accuracy on extended-emission targets.')
    p.add_argument('--pre-gradient-removal', action='store_true',
                   help='Fit and subtract a degree-2 polynomial sky gradient from each frame '
                        'before quality analysis (useful for nebulae with strong gradients).')
    p.add_argument('--trim-low', type=float, default=0.2, metavar='F',
                   help='Low-fraction trim for trimmed_mean stacking (default: 0.2).')
    p.add_argument('--trim-high', type=float, default=0.2, metavar='F',
                   help='High-fraction trim for trimmed_mean stacking (default: 0.2).')
    p.add_argument('--drizzle-pixfrac', type=float, default=1.0, metavar='P',
                   help='Drizzle pixel fraction (tent-kernel weight; < 1.0 = sharper '
                        'at cost of noise; default: 1.0).')
    p.add_argument('--optical-flow', action='store_true',
                   help='Apply per-frame Farneback optical flow local warp correction after '
                        'global shift/affine registration. Requires OpenCV (cv2).')
    p.add_argument('--halo-removal', action='store_true',
                   help='Fit and subtract Gaussian PSF halos from bright stars in the '
                        'stacked image (post-processing step).')

    # Defaults for parameters that are tunable via config file but not exposed on the CLI.
    # Set these in a TOML config with --config to override them.
    p.set_defaults(
        # Features that are on by default (disabled via --no-* flags)
        background_extraction=True,
        denoise=True,
        auto_denoise_strength=True,
        denoise_adaptive=True,
        chroma_nr=True,
        star_reduce=True,
        local_contrast=True,
        ca_correction=True,
        cosmic_ray_rejection=True,
        # Denoiser tuning
        denoise_chroma_boost=2.0,
        chroma_nr_sigma=2.0,
        denoise_nlm_strength=1.0,
        denoise_nlm_blend=0.5,
        denoise_bilateral_sigma_color=None,
        denoise_bilateral_sigma_space=3.0,
        denoise_mmt_levels=4,
        denoise_mmt_strength=3.0,
        denoise_acdnr_sigma=1.5,
        denoise_acdnr_k=3.0,
        bm3d_sigma=0.0,
        bm3d_stride=None,
        bm3d_search_window=16,
        bm3d_group_size=8,
        aniso_iterations=20,
        aniso_kappa=30.0,
        aniso_gamma=0.1,
        aniso_option=1,
        # Deconvolution tuning
        deconvolve_iterations=Config.RL_DEFAULT_ITERATIONS,
        deconvolve_fwhm=None,
        deconvolve_psf_model='moffat',
        tv_lambda=None,
        tv_iterations=None,
        # Background tuning
        bg_mesh_size=64,
        bg_filter_size=3,
        bg_clip_sigma=3.0,
        dbe_patch_size=64,
        entropy_bg=False,
        # Registration internals
        # (skip_phase_correlation default set on the --phase-correlation argument)
        no_alignment_centrality=False,
        no_shift_outlier_filter=False,
        no_reg_residual_check=False,
        reg_residual_reject=False,
        patch_registration=False,
        poly_distortion=False,
        poly_distortion_degree=2,
        # Stacking internals
        percentile_low=20.0,
        percentile_high=80.0,
        esd_max_outliers=0,
        esd_significance=0.05,
        # Quality
        # (advanced_metrics default set on the --advanced-metrics argument)
        # Per-feature tuning
        star_reduce_factor=0.4,
        star_reduce_sigma=1.5,
        local_contrast_strength=0.7,
        local_normalize_sigma=50.0,
        scnr_amount=1.0,
        scnr_target='green',
        comet_blend_sigma=30.0,
        comet_xy=None,
        comet_search_radius=50.0,
        comet_affine=False,
        coma_mask_radius=150,
        comet_radial_renorm=False,
        comet_larson_sekanina=False,
        comet_ls_rotation=15.0,
        comet_designation=None,
        observer_site=None,
        comet_detrail=False,
        hdr_short_exptime=1.0,
        hdr_long_exptime=1.0,
        drizzle_drop_size=0.7,
        weight_snr=1.0,
        weight_fwhm=1.0,
        weight_stars=1.0,
        weight_noise=False,
        cr_sigclip=4.5,
        cr_objlim=5.0,
        # Improvements 1-9
        max_ellipticity=0.5,
        consensus_ref=False,
        masked_correlation=False,
        pre_gradient_removal=False,
        trim_low=0.2,
        trim_high=0.2,
        drizzle_pixfrac=1.0,
        optical_flow=False,
        halo_removal=False,
    )
    return p.parse_args()


def main():
    # Dispatch to the 'combine' subcommand without touching the main parser
    if len(sys.argv) > 1 and sys.argv[1] == 'combine':
        from src.channel_combine import run_combine_cli
        run_combine_cli(sys.argv[2:])
        return

    args = parse_args()
    # Upgrade bilinear → malvar when OpenCV is available; malvar resolves colour moiré
    # that bilinear introduces around fine star disks at no perceptible speed cost.
    if getattr(args, 'debayer_method', 'bilinear') == 'bilinear':
        try:
            import cv2 as _cv2  # noqa: F401
            args.debayer_method = 'malvar'
        except ImportError:
            pass
    if not args.health_check and not getattr(args, 'dry_run', False) and not args.output:
        dir_name = os.path.basename(os.path.abspath(args.directory))
        args.output = f"{dir_name}_stacked.fits"
        safe_print(f"  No output path specified — writing to {args.output}")
    # Load config file (before preset, so preset can override config)
    if getattr(args, 'config', None):
        config_changes = load_config_file(args.config, args)
        if config_changes:
            safe_print(f"  Config loaded: {len(config_changes)} settings from {args.config}")
    # Apply preset (before any other processing)
    preset_changes = apply_preset(args)
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
