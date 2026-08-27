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

from src.gpu_context import get_gpu, reset_gpu
from src.models import Config, ProcessingStats
from src.utils import safe_print, print_header, format_time, setup_logging
from src.io_fits import make_master, save_preview_rgb
from src.frame_discovery import (discover_frames, select_matching_darks, select_matching_flats,
                                 group_lights_by_filter)
from src.debayer import build_hot_pixel_map
from src.registration import calculate_shift, apply_transform
from src.pipeline import stack_target
from src.health_check import run_health_check
from src.cleanup import register as _cleanup_register, deregister as _cleanup_deregister


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
        _explicit = getattr(args, '_explicit_cli_dests', set())
        for key, value in flat.items():
            if key in _protected or key in _explicit:
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

    explicit = getattr(args, '_explicit_cli_dests', set())
    preset_values = presets.get(args.preset, {})
    for key, value in preset_values.items():
        if key in explicit:
            continue  # user explicitly passed this flag on the command line
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
        safe_print(f'  WARNING: --cal-dir path does not exist: {cal_dir}')
        return extra
    discovered = discover_frames(cal_dir)
    for ftype in ('dark', 'flat', 'bias'):
        found = discovered.get(ftype, [])
        extra[ftype].extend(found)
        if found:
            safe_print(f"  Calibration library: {len(found)} {ftype} frames from {cal_dir}")
    return extra


def _build_masters(frames: dict, stats: "ProcessingStats | None" = None,
                   args: "argparse.Namespace | None" = None) -> dict:
    """Build, filter, and smooth master bias/dark/flat from the frame lists in *frames*.

    Mutates ``frames['dark']`` and ``frames['flat']`` in-place via
    ``select_matching_darks`` / ``select_matching_flats`` so the caller sees
    the filtered counts.  Returns a fully populated ``masters`` dict.

    Also loads ``masters['vignette']`` from ``args.vignette_map`` when given
    (single spot so single-folder, pooled-session, and --live all pick it up).
    """
    lights = frames.get('light', [])
    cal_needed = frames.get('dark') or frames.get('flat') or frames.get('bias')
    cal_start = time.time() if cal_needed else None
    if cal_needed:
        safe_print("\nCreating master calibration frames...")

    # master_method defaults to 'median' and is opt-in only for 'robust_pca' at
    # real scale -- benchmarked robust_pca_decompose directly (2000x3000x3, 20
    # frames): 2910s (~48.5 min) pre-optimization, 1264s (~21 min) after
    # src/robust_pca.py's Gram-matrix-trick SVD (native gram_matrix_wide/
    # small_times_wide kernels, ~9x over a direct SVD call on this shape,
    # ~2.3x end-to-end -- the non-SVD per-iteration ops don't benefit and now
    # dominate more of the budget). Still too slow to silently default for a
    # real-sized calibration library. Below ROBUST_PCA_AUTO_MAX_FRAMES,
    # though, cost scales down enough (~5min at N=10) to be a reasonable
    # --auto default -- gated per calibration type (bias/dark/flat counts
    # often differ) below, not globally, so one small type doesn't drag a
    # large one into robust_pca too.
    master_method = getattr(args, 'master_method', 'median') or 'median'
    _auto_master = (getattr(args, 'auto', False)
                    and master_method == 'median'
                    and 'master_method' not in getattr(args, '_explicit_cli_dests', set()))

    def _method_for(n_frames: int) -> str:
        if (_auto_master and Config.ROBUST_PCA_MIN_FRAMES <= n_frames
                <= Config.ROBUST_PCA_AUTO_MAX_FRAMES):
            return 'robust_pca'
        return master_method

    masters: dict = {}

    def _method_tag(method: str) -> str:
        # Auto-upgrade to robust_pca is a silent behavior change (slower,
        # heavier on memory) unless the run output actually says it happened.
        return f" ({method})" if method != 'median' else ""

    if frames.get('bias'):
        _bias_method = _method_for(len(frames['bias']))
        masters['bias'] = make_master(frames['bias'], method=_bias_method)
        if masters['bias'] is not None:
            safe_print(f"  ✓ Master bias:  {len(frames['bias'])} frames -> "
                       f"{masters['bias'].shape[0]}×{masters['bias'].shape[1]}"
                       f"{_method_tag(_bias_method)}")
    else:
        masters['bias'] = None

    if frames.get('dark'):
        _dark_from_model = False
        _use_dark_model = getattr(args, 'dark_temp_model', False)
        if not _use_dark_model and getattr(args, 'auto', False) and \
                'dark_temp_model' not in getattr(args, '_explicit_cli_dests', set()):
            # Only attempt (and print about) this when the library already
            # has enough temperature spread to plausibly work -- a cheap
            # header-only peek, so the common single-session/single-
            # temperature dark library doesn't get a noisy "unavailable,
            # falling back" message on every --auto run.
            from src.dark_temp_model import _frame_temp as _peek_temp
            _n_distinct = len(set(round(t, 1) for t in
                                  (_peek_temp(f) for f in frames['dark']) if t is not None))
            _use_dark_model = _n_distinct >= 3
        if _use_dark_model:
            from src.dark_temp_model import (build_dark_temperature_model,
                                              sample_dark_at_temperature, _frame_temp)
            light_temps = [t for t in (_frame_temp(f) for f in lights) if t is not None]
            target_temp = sum(light_temps) / len(light_temps) if light_temps else None
            model = build_dark_temperature_model(frames['dark']) if target_temp is not None else None
            if model is not None:
                masters['dark'] = sample_dark_at_temperature(model, target_temp)
                _dark_from_model = True
                safe_print(f"  ✓ Master dark:  temperature model from {model['n_frames']} "
                           f"frames ({model['n_temps']} distinct temps, degree="
                           f"{model['degree']}) evaluated at {target_temp:.1f}°C -> "
                           f"{masters['dark'].shape[0]}×{masters['dark'].shape[1]}")
            else:
                safe_print("  Dark temperature model unavailable (too few distinct "
                           "temperatures/frames) -- falling back to nearest-match selection")
        if not _dark_from_model:
            frames['dark'] = select_matching_darks(lights, frames['dark'])
            _dark_method = _method_for(len(frames['dark']))
            masters['dark'] = make_master(frames['dark'], method=_dark_method)
            if masters['dark'] is not None:
                safe_print(f"  ✓ Master dark:  {len(frames['dark'])} frames -> "
                           f"{masters['dark'].shape[0]}×{masters['dark'].shape[1]}"
                           f"{_method_tag(_dark_method)}")
    else:
        masters['dark'] = None

    if frames.get('flat'):
        frames['flat'] = select_matching_flats(lights, frames['flat'])
        _flat_method = _method_for(len(frames['flat']))
        masters['flat'] = make_master(frames['flat'], method=_flat_method)
        if masters['flat'] is not None:
            safe_print(f"  ✓ Master flat:  {len(frames['flat'])} frames -> "
                       f"{masters['flat'].shape[0]}×{masters['flat'].shape[1]}"
                       f"{_method_tag(_flat_method)}")
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
        _use_flat_from_lights = (
            getattr(args, 'flat_from_lights', False)
            or (getattr(args, 'auto', False)
                and 'flat_from_lights' not in getattr(args, '_explicit_cli_dests', set())))
        if _use_flat_from_lights and len(lights) >= Config.ROBUST_PCA_MIN_FRAMES:
            # Cap the sample: cost is O(N^2 x pixels), and more frames past a
            # modest count buys little extra low-rank/sparse separation quality.
            sample = lights[:: max(1, len(lights) // Config.ROBUST_PCA_AUTO_MAX_FRAMES)]
            sample = sample[:Config.ROBUST_PCA_AUTO_MAX_FRAMES]
            safe_print(f"  No flat frames found -- deriving a synthetic flat from "
                       f"{len(sample)} light frames (--flat-from-lights)...")
            synthetic_flat = make_master(sample, method='robust_pca')
            if synthetic_flat is not None:
                masters['flat'] = synthetic_flat
                safe_print(f"  ✓ Master flat:  {len(sample)} light frames (synthetic) -> "
                           f"{synthetic_flat.shape[0]}×{synthetic_flat.shape[1]} (robust_pca)")
            else:
                safe_print("  Synthetic flat-from-lights failed (too few usable frames) "
                           "-- proceeding without a flat")

    masters['hot_pixel_map'] = None
    if masters.get('dark') is not None:
        hot_map = build_hot_pixel_map(masters['dark'])
        n_hot = int(np.sum(hot_map))
        if n_hot > 0:
            masters['hot_pixel_map'] = hot_map
            safe_print(f"  ✓ Hot pixel map: {n_hot} pixels from dark frame")

    if masters.get('bias') is not None:
        n_bias = len(frames['bias'])
        sigma_b = max(1, 30 // max(1, int(np.sqrt(n_bias))))
        masters['bias'] = ndimage.gaussian_filter(masters['bias'].astype(np.float32), sigma=sigma_b)
    if masters.get('dark') is not None:
        n_dark = len(frames['dark'])
        sigma_d = max(1, 20 // max(1, int(np.sqrt(n_dark))))
        masters['dark'] = ndimage.gaussian_filter(masters['dark'].astype(np.float32), sigma=sigma_d)
    if masters.get('flat') is not None:
        n_flat = len(frames['flat'])
        sigma_f = max(1, 15 // max(1, int(np.sqrt(n_flat))))
        flat_raw = masters['flat'].astype(np.float32)
        for r_off, c_off in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            ch = flat_raw[r_off::2, c_off::2]
            flat_raw[r_off::2, c_off::2] = ndimage.gaussian_filter(ch, sigma=sigma_f)
        masters['flat'] = flat_raw

    masters['dark_exptime'] = None
    if frames.get('dark'):
        try:
            masters['dark_exptime'] = float(
                frames['dark'][0].header.get('EXPTIME', 0) or 0) or None
        except Exception as e:
            safe_print(f"  WARNING: Could not read dark EXPTIME ({e}) — dark scaling disabled")

    if cal_needed and cal_start is not None and stats is not None:
        stats.calibration_time = time.time() - cal_start

    masters['vignette'] = None
    vignette_path = getattr(args, 'vignette_map', None) if args is not None else None
    if vignette_path:
        from src.vignette_calib import load_vignette_map
        vmap = load_vignette_map(vignette_path)
        if vmap is not None:
            masters['vignette'] = vmap
            safe_print(f"  ✓ Vignette map: {os.path.basename(vignette_path)} "
                       f"({vmap.shape[1]}×{vmap.shape[0]})")

    return masters


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
    safe_print(f"  Pooled {n_lights} lights from {n_sessions} sessions "
              f"({len(combined['dark'])} darks, {len(combined['flat'])} flats, "
              f"{len(combined['bias'])} bias)")

    if not n_lights:
        safe_print('  ERROR: No light frames found across sessions')
        raise SystemExit('No light frames found')

    if getattr(args, 'dry_run', False):
        safe_print(f"\n  --- DRY RUN ---")
        safe_print(f"  Light frames: {n_lights} pooled from {n_sessions} sessions")
        safe_print(f"  Stack method: {args.stack_method}")
        return

    masters = _build_masters(combined, stats, args)

    if getattr(args, 'health_check', False):
        print_header("HEALTH CHECK", "=")
        run_health_check(combined, masters, sorted(subdirs)[0])
        return

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


def _want_combine_sessions(args: argparse.Namespace) -> bool:
    """Multiple subfolders default to pooling into one unified stack --
    a multi-night/multi-filter session directory is the far more common
    case than wanting independent per-subfolder stacks stitched together
    afterward. --mosaic needs per-subfolder targets to reproject and stitch,
    so it always wins; otherwise an explicit --combine-sessions or
    --hierarchical wins; absent either, the new default is combine."""
    if getattr(args, 'mosaic', False):
        return False
    explicit = getattr(args, '_explicit_cli_dests', set())
    if 'combine_sessions' in explicit:
        return bool(args.combine_sessions)
    return not getattr(args, 'hierarchical', False)


def process_directory(directory: str, output: str, args: argparse.Namespace):
    # Print banner
    print_header("Astrophotography FITS Stacker", "=")
    safe_print(f"Input:  {directory}")
    safe_print(f"Output: {output}")
    if getattr(args, 'preset', None):
        safe_print(f"  Preset: {args.preset}")
    get_gpu().print_status()
    from src.utils import native_status
    safe_print(native_status())

    # Detect hierarchical mode
    if not os.path.isdir(directory):
        safe_print(f'\n  ERROR: Input directory {directory} does not exist')
        raise SystemExit(1)

    overall_start = time.time()
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    targets = []

    safe_print("\nDiscovering frames...")
    if any(os.listdir(directory)) and any(f.lower().endswith(('.fit', '.fits')) for f in os.listdir(directory)):
        # single folder
        targets = [(directory, output)]
        safe_print(f"  Mode: Single folder")
    elif subdirs and _want_combine_sessions(args):
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
            _cleanup_register(outp)
        safe_print(f"  Mode: Hierarchical ({len(targets)} subfolders)")
        # final combined output will be combined from tmp_stacks
    else:
        safe_print('  ERROR: No FITS files found')
        raise SystemExit('No FITS files found')

    # Mosaic mode requires hierarchical layout (one subfolder per panel)
    if getattr(args, 'mosaic', False):
        if len(targets) < 2:
            safe_print('  ERROR: --mosaic requires subfolders (one per panel); '
                      'only a single target was found')
            raise SystemExit('--mosaic requires subfolders')
        if not getattr(args, 'plate_solve', False):
            safe_print("  NOTE: --mosaic implies --plate-solve — enabling automatically")
            args.plate_solve = True

    # Process each target
    produced = []
    for target_idx, (d, outp) in enumerate(targets, 1):
        if len(targets) > 1:
            safe_print(f'\n{"=" * 70}')
            safe_print(f'TARGET {target_idx}/{len(targets)}: {os.path.basename(d)}')
            safe_print(f'{"=" * 70}')
        else:
            safe_print('')

        # Create stats object for this target
        stats = ProcessingStats()

        frames = discover_frames(d)
        extra_cal = _load_calibration_dir(args)
        for ftype in ('dark', 'flat', 'bias'):
            frames[ftype].extend(extra_cal[ftype])
        nfiles = sum(len(v) for v in frames.values())
        safe_print(f'  Found {nfiles} FITS files: {len(frames["light"])} lights, {len(frames["dark"])} darks, {len(frames["flat"])} flats, {len(frames["bias"])} bias')

        # Lights shot through different filters (per-frame FITS FILTER header,
        # not directory-based -- a session directory can mix filters, e.g. a
        # 'Clear' sub next to a 'Nebula'-filtered one) must not be averaged
        # together: each gets its own masters (flat matching already keys off
        # FILTER) and its own output file.
        filter_groups = group_lights_by_filter(frames['light'])
        if len(filter_groups) > 1:
            safe_print(f"  Mode: Split by filter ({len(filter_groups)} filters: "
                       f"{', '.join(sorted(filter_groups))}) -- stacking each separately")
            if getattr(args, 'health_check', False) or getattr(args, 'dry_run', False):
                for filt_tag, group_lights in sorted(filter_groups.items()):
                    safe_print(f"    {filt_tag}: {len(group_lights)} lights (skipped -- "
                               f"health-check/dry-run not supported per filter group yet)")
                continue
            base, ext = os.path.splitext(outp)
            _orig_output = getattr(args, 'output', None)
            for filt_tag, group_lights in sorted(filter_groups.items()):
                safe_print(f'\n{"=" * 70}')
                safe_print(f'FILTER GROUP: {filt_tag} ({len(group_lights)} lights)')
                safe_print(f'{"=" * 70}')
                group_frames = dict(frames)
                group_frames['light'] = group_lights
                # Fresh copies -- _build_masters mutates frames['dark']/['flat']
                # in place (select_matching_darks/select_matching_flats), and
                # each filter group must match against the full calibration
                # pool independently, not whatever a previous group narrowed it to.
                group_frames['dark'] = list(frames['dark'])
                group_frames['flat'] = list(frames['flat'])
                group_frames['bias'] = list(frames['bias'])
                group_stats = ProcessingStats()
                group_outp = f"{base}_{filt_tag}{ext}"
                group_masters = _build_masters(group_frames, group_stats, args)
                args._input_directory = d
                # postprocess_stack's sidecar writers (e.g. star removal's
                # <output>_starless.fits) key off args.output, not the
                # output_path argument -- without this, every group would
                # collide on the same sidecar filename and overwrite each
                # other's.
                args.output = group_outp
                gres = stack_target([f for t in group_frames.values() for f in t],
                                    group_outp, args, group_masters, group_stats)
                if gres:
                    save_effective_config(args, group_outp)
            args.output = _orig_output
            continue  # filter-split target fully handled -- not part of hierarchical combine

        if getattr(args, 'health_check', False):
            print_header("HEALTH CHECK", "=")
            safe_print(f"  Directory: {os.path.abspath(d)}")

        # Create master calibration frames
        masters = _build_masters(frames, stats, args)

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
        safe_print(f"  Combining {len(produced)} target stacks into final output...")

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
            safe_print(f"  Cropping all stacks to minimum dimensions: {Hm}×{Wm}")

            stacks = []
            for p in tqdm(produced, desc="  Loading", unit="target", disable=args.verbose):
                with fits.open(p, memmap=True) as hd:
                    d = np.transpose(hd[0].data, (1, 2, 0)).astype(np.float32)
                    stacks.append(d[:Hm, :Wm, :])

            # Register each stack against the first (reference) stack
            safe_print(f"  Registering {len(stacks) - 1} stack(s) against reference...")
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
            safe_print(f"  Valid overlap after registration: {Hf}×{Wf}")

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

        # Remove per-target temp stacks now that the combined output is written
        for p in tmp_stacks:
            try:
                os.remove(p)
            except Exception:
                pass
            _cleanup_deregister(p)

    # Overall summary
    total_time = time.time() - overall_start
    print_header("OVERALL SUMMARY", "=")
    if len(produced) > 1:
        safe_print(f"  Targets processed: {len(produced)}")
    safe_print(f"  Total time: {format_time(total_time)}")
    safe_print(f"\n  ✓ All processing complete!")
    safe_print(f"{'=' * 70}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Streaming FITS stacker')
    g_core = p.add_argument_group('Core')
    g_frames = p.add_argument_group('Frames & calibration (Phase 1)')
    g_stack = p.add_argument_group('Registration & stacking (Phases 2-3)')
    g_post = p.add_argument_group('Post-processing (Phase 4)')
    g_out = p.add_argument_group('Output, preview & plate solving')
    g_sessions = p.add_argument_group('Multi-session, merge & checkpoint')
    g_comet = p.add_argument_group('Comet mode')
    g_adv = p.add_argument_group('Advanced (most are managed automatically by --auto)')
    g_debug = p.add_argument_group('Diagnostics & debugging')
    g_astrollm = p.add_argument_group('astrollm scoring (advisory)')
    g_core.add_argument('-d', '--directory', required=True)
    g_core.add_argument('-o', '--output', default=None,
                   help='Output FITS path (default: <directory>_stacked.fits)')
    g_core.add_argument('--cal-dir', default=None, metavar='PATH',
                   help='Directory containing calibration frames (darks, flats, bias). '
                        'Frames are classified automatically and the best-matching subset '
                        'is selected per type (by ISO, CCD temperature, exposure, filter, '
                        'and sensor dimensions). Supplements any calibration frames already '
                        'present in --directory.')
    g_core.add_argument('--vignette-map', default=None, metavar='PATH',
                   help='Per-instrument vignetting/background calibration map (FITS, '
                        'H x W x 3) built offline by tools/build_vignette_map.py from '
                        'many past sessions of this same telescope+camera. Subtracted '
                        'per-frame right after debayer, before registration -- reduces '
                        'the gradient DBE/--bg-method wavelet has to guess at from one '
                        'session\'s sparse sky samples alone.')
    g_core.add_argument('--health-check', action='store_true',
                   help='Analyse input frames and calibration quality without stacking')
    g_core.add_argument('--quality-sweep', action='store_true',
                   help='Walk the folder tree under -d, score every light frame with '
                        'the same quality gate stacking uses (hard failures, '
                        'statistical outliers, score below --quality-threshold%% of '
                        'each folder\'s reference), and report poor frames. Dry run '
                        'by default; add --apply to rename flagged files to '
                        '*.fits.rejected (invisible to stacking, reversible with '
                        '--sweep-undo).')
    g_core.add_argument('--apply', action='store_true',
                   help='With --quality-sweep: actually rename flagged files '
                        '(default is a dry-run report)')
    g_core.add_argument('--sweep-undo', action='store_true',
                   help='Recursively strip the .rejected suffix applied by '
                        '--quality-sweep --apply, restoring all flagged files')
    g_core.add_argument('--live', action='store_true',
                   help='Real-time (live) stacking: watch the directory and fold each new '
                        'sub into a running stack as it lands, pushing the growing result '
                        'and a running SNR to whatever UI is attached (console-only on a '
                        'plain CLI run; the desktop app shows it live). Runs until Ctrl-C.')
    g_core.add_argument('--live-interval', type=float, default=4.0, metavar='SEC',
                   help='Live stacking directory poll interval in seconds (default: 4).')
    g_core.add_argument('--live-duration', type=float, default=None, metavar='MIN',
                   help='Optional live stacking time limit in minutes (default: until Ctrl-C).')
    g_core.add_argument('--stream', action='store_true',
                   help="Two-pass streaming stack of an ALREADY-COMPLETE directory: "
                        "O(1) full-resolution memory via an online (single-pass) "
                        "sigma-clip Welford accumulator, instead of materializing the "
                        "whole (N,H,W,C) aligned stack --stack-method needs. v1 "
                        "limitations vs the default pipeline: reference frame is picked "
                        "by quality score alone (no shift-centrality blend), hard-limit "
                        "quality gating only (no statistical/percentile stages), no "
                        "--elastic-registration, no drizzle, no patch-weighted combine, "
                        "no --merge. Incompatible with --live (one watches a growing "
                        "directory forever, the other expects a closed one).")
    g_core.add_argument('--stream-burnin', type=int, default=10, metavar='N',
                   help='--stream: number of frames MAD-rejected as a batch to seed the '
                        'running sigma-clip state before streaming begins (default: 10).')
    g_core.add_argument('--stream-sigma', type=float, default=None, metavar='SIGMA',
                   help='--stream: sigma-clip rejection threshold (default: '
                        '--rejection-sigma).')
    g_core.add_argument('--ui-frame-every', type=int, default=5, metavar='N',
                   help='Publish a per-frame thumbnail to the desktop app every '
                        'Nth processed light in Phase 1 (default: 5; 0 = off). '
                        'The first frame is always shown. No effect on a plain '
                        'CLI run (nothing is attached to receive it).')
    g_core.add_argument('--config', default=None, metavar='PATH',
                   help='Load parameters from a TOML configuration file. '
                        'CLI arguments override config file values. Fine-grained tuning '
                        'parameters removed from the CLI can still be set via config file.')
    g_core.add_argument('--preset',
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
    g_core.add_argument('--dry-run', action='store_true',
                   help='Discover and classify frames, show calibration info, print effective '
                        'parameters, and estimate resource usage — without processing anything')
    g_post.add_argument('--denoiser',
                   choices=['auto', 'wavelet', 'mmt', 'bm3d', 'acdnr', 'nlm',
                            'bilateral', 'aniso', 'curvelet', 'none'],
                   default='auto',
                   help='Primary luma denoiser (default: auto — curvelet unless a '
                        'preset/--auto selects otherwise). '
                        'wavelet: adaptive BayesShrink DWT. '
                        'mmt: Multiscale Median Transform, robust to Poisson+read noise. '
                        'bm3d: collaborative filtering, near-optimal but slower. '
                        'acdnr: contrast-gated sky smoothing. '
                        'nlm / bilateral / aniso: alternative edge-preserving filters. '
                        'curvelet: adaptive BayesShrink DWT (like wavelet) but with the '
                        'per-subband threshold locally reduced wherever a structure-tensor '
                        'coherence map detects elongated structure (filaments, galaxy arms) '
                        '-- curvelet/shearlet-*inspired*, not an actual ridgelet/shearlet '
                        'transform (tune via --config directional_protect_strength, default '
                        '0.6; 0=identical to wavelet). '
                        'none: disable luma denoising. Chroma noise reduction is '
                        'separate (--no-chroma-nr). Strength via --denoise-strength; '
                        'fine tuning via --config.')
    g_post.add_argument('--deconvolve', choices=['off', 'rl', 'tv', 'rl-sv', 'sparse'],
                   default='off', dest='deconvolve_mode',
                   help='Deconvolution: off (default), rl (Richardson-Lucy), '
                        'tv (Total-Variation regularised; sharper edges, slower), '
                        'rl-sv (spatially-variant RL: a separate PSF per field tile, '
                        'corrects off-axis aberration/tilt the corners suffer), or '
                        'sparse (FISTA, L1-regularised in this project\'s own wavelet '
                        'basis rather than TV\'s spatial-gradient basis -- a different '
                        'sparsity prior, tune via --config deconvolve_sparse_lambda, '
                        'default 0.01). '
                        'PSF options (blind PSF, iterations, model) via --config.')
    g_out.add_argument('--export', default=None, metavar='FMT[,FMT]',
                   help='Extra output formats alongside FITS: tiff (32-bit float), '
                        'xisf (PixInsight). E.g. --export tiff,xisf')
    g_debug.add_argument('--debug', default=None, metavar='KIND[,KIND]',
                   help='Debug artefacts, comma-separated: '
                        'registration (per-frame diagnostics, implies -v), '
                        'diagnostic (FITS snapshot before each post-processing step; '
                        'directory via --config diagnostic_dir), '
                        'intermediates (keep aligned per-frame FITS), '
                        'masks (save the star mask FITS).')
    g_post.add_argument('--skip-step', action='append', default=[], metavar='STEP',
                   help='Skip a named post-processing step. Can be specified multiple times. '
                        'Steps: hot_pixel, background, chroma_nr, sky_floor, '
                        'wavelet, sky_residual, sky_pedestal, nlm, bilateral, mmt, '
                        'acdnr, curvelet, deconvolve, star_reduce, local_contrast, '
                        'sky_neutralize, remove_stars')
    g_stack.add_argument('--no-registration', action='store_true')
    g_stack.add_argument('--no-affine', action='store_true',
                   help='Disable affine (rotation+translation) registration; use translation-only')
    g_stack.add_argument('--no-reg-residual-reject', dest='reg_residual_reject',
                   action='store_false', default=True,
                   help='Disable dropping frames whose post-registration star-position '
                        'RMS residual exceeds an adaptive threshold (%.1f-%.1fpx, scaled to '
                        'the reference frame\'s measured FWHM/SNR; on by default: this '
                        'catches frames a bad affine/FFT registration passed through '
                        'undetected -- an actual measured alignment error, not a '
                        'heuristic). Rejected frames still count in the summary; '
                        'reg_residual_px is recorded in each frame\'s metrics either way.'
                        % (Config.REG_RESIDUAL_MAX_PX, Config.REG_RESIDUAL_MAX_PX_CAP))
    g_stack.add_argument('--no-reg-residual-check', action='store_true',
                   help='Skip the post-registration residual check entirely (no '
                        'diagnostic, no rejection) -- faster but no safety net')
    g_stack.add_argument('--elastic-registration', action='store_true',
                   help='Fit a smooth per-frame local (non-rigid) displacement field '
                        'from matched-star residuals after global affine registration, '
                        'correcting spatially-varying distortion (differential '
                        'atmospheric refraction, field rotation across the frame, tube '
                        'flexure) that a single affine transform per frame cannot fix. '
                        'Composed into the same single resampling pass as the affine '
                        'warp (no extra blur pass), and honoured by both plain and '
                        '--drizzle-scale stacking. Requires >= %d matched stars per '
                        'frame (falls back to affine-only otherwise) and clamps the '
                        'fitted displacement to %.1fpx. Off by default -- opt-in, '
                        'higher-risk correction.' % (Config.LOCAL_WARP_MIN_STARS,
                                                      Config.LOCAL_WARP_MAX_DISPLACEMENT_PX))
    g_frames.add_argument('--master-method', choices=['median', 'mean', 'robust_pca'],
                   default='median',
                   help='Master bias/dark/flat combine method (default: median). '
                        'robust_pca: Principal Component Pursuit low-rank + sparse '
                        'decomposition -- separates the true shared pattern (flat-field '
                        'vignetting, fixed dark current) from sparse outliers (dust motes '
                        'that shifted between sessions, transient hot pixels) more '
                        'explicitly than median\'s per-pixel order statistic. Needs >= %d '
                        'frames per calibration type (falls back to median otherwise); '
                        'loads the full stack into memory and is slower than median/mean.'
                        % Config.ROBUST_PCA_MIN_FRAMES)
    g_frames.add_argument('--dark-temp-model', action='store_true',
                   help='Fit a per-pixel polynomial of dark signal vs. sensor temperature '
                        'across the whole dark library, then evaluate it at the lights\' '
                        'own temperature -- instead of select_matching_darks\' nearest-'
                        'temperature selection. Lets a smaller dark library cover a wider '
                        'range of session temperatures via interpolation rather than '
                        'requiring a near-exact match. Assumes the dark library is already '
                        'homogeneous in ISO/gain and exposure time (does not also model '
                        'those). Needs >= 3 distinct temperatures across the dark library; '
                        'falls back to select_matching_darks otherwise.')
    g_frames.add_argument('--flat-from-lights', action='store_true',
                   help='When no flat frames are available, derive a synthetic flat/'
                        'vignetting map from the light frames themselves via robust-PCA '
                        'low-rank/sparse decomposition (same decomposition --master-method '
                        'robust_pca uses for real calibration frames): the low-rank '
                        'component of the raw light stack is the shared sensor pattern '
                        '(vignetting, dust), while stars and nebula structure -- sparse '
                        'and slightly shifted frame-to-frame by dithering -- are separated '
                        'out as the sparse component. Approximate (a static, undithered '
                        'target can partially leak into the low-rank component) -- opt-in, '
                        'not a substitute for real flats when they exist. Needs >= %d '
                        'light frames; ignored when dedicated flat frames were discovered.'
                        % Config.ROBUST_PCA_MIN_FRAMES)
    g_frames.add_argument('--no-quality-filter', action='store_false', dest='quality_filter',
                   default=True,
                   help='Disable automatic rejection of the lowest-quality frames')
    g_frames.add_argument('--quality-threshold', type=float, default=50.0,
                   help='Reject frames whose score falls more than this percent below the '
                        'reference score (90th-percentile of the session, default: 50). '
                        'E.g. 50 keeps every frame with score >= 50%% of the reference. '
                        'Use --no-quality-filter to disable entirely.')
    g_core.add_argument('-v', '--verbose', action='store_true')
    g_stack.add_argument('--stack-method',
                   choices=['mean', 'median', 'sigma_clip', 'winsorized',
                            'percentile', 'esd', 'linear_fit',
                            'ivw', 'wavelet', 'auto'],
                   default='auto',
                   help='Stacking/rejection method. '
                        'sigma_clip: MAD-based iterative rejection (default for dithered data). '
                        'winsorized: like sigma_clip but clips to boundary instead of rejecting. '
                        'percentile: reject outside [low,high] percentile (good for <8 frames; '
                        'subsumes the removed trimmed_mean method -- same reject-tails-then-'
                        'average operation, just parameterized by percentile bounds instead of '
                        'a trim fraction). '
                        'esd: Grubbs/ESD test (best for <15 frames, needs scipy). '
                        'linear_fit: PixInsight-style Linear Fit Clipping -- sorts each pixel\'s '
                        'per-frame stack ascending and fits a line to value-vs-rank, rejecting '
                        'samples whose residual from that fit exceeds sigma_low/sigma_high; '
                        'more robust to non-Gaussian tails than sigma-clip\'s mean/std test '
                        '(tune via --config: linear_fit_sigma_low/_sigma_high/_iters). '
                        'ivw: inverse-noise-variance-weighted mean (the Gauss-Markov-optimal '
                        'linear combiner) -- weights each frame by 1/noise^2 from its own '
                        'measured Phase 1 background sigma; no rejection, pair with '
                        '--cosmic-ray-rejection/--trail-reject (tune gain via --config '
                        'ivw_gain for an added per-pixel shot-noise term). '
                        'wavelet: decompose every frame and sigma-clip-combine each wavelet '
                        'subband across frames (reuses --rejection-sigma/-iters/-estimator), '
                        'then reconstruct -- preserves faint fine-scale structure a pixel-domain '
                        'outlier test can shave off, at the cost of N wavelet transforms per '
                        'channel (slower than the pixel-domain methods). Not compatible with '
                        '--drizzle-scale > 1 (v1 limitation); tune depth via --config '
                        'wavelet_combine_levels (default 4). '
                        'auto: choose based on frame count (<8->percentile, else sigma_clip). '
                        'mean/median: no rejection.')
    g_stack.add_argument('--uncertainty-map', action='store_true',
                   help='With --stack-method ivw, also write a per-pixel uncertainty '
                        '(standard error) map to <output>_sigma.fits -- the Gauss-Markov '
                        'estimator\'s own 1/sqrt(sum of inverse-variance weights), exposing '
                        'bookkeeping the combine already computes rather than adding new '
                        'analysis. One confidence map (channel weights averaged), not '
                        'per-channel. No effect with other stack methods (only ivw computes '
                        'an exact analytic per-pixel variance).')
    g_stack.add_argument('--rejection-sigma', type=float, default=3.0,
                   help='Sigma threshold for pixel rejection in sigma_clip/winsorized stacking (default: 3.0)')
    g_stack.add_argument('--rejection-iters', type=int, default=3,
                   help='Number of clipping iterations for sigma_clip stacking (default: 3)')
    g_frames.add_argument('--debayer-method', choices=['malvar', 'menon2007'], default='malvar',
                   help='Debayering method (default: malvar; both are native '
                        'Rust kernels, no external dependency). menon2007 '
                        '(DDFAPD directional filtering) is higher fidelity '
                        'than malvar on fine periodic photographic detail in '
                        'general, but on this codebase\'s synthetic astro '
                        'benchmark (tools/bench_debayer_quality.py) the gain '
                        'is modest (~14%% lower MAE on a synthetic starfield) '
                        'and it is meaningfully slower -- most astro frames '
                        'are smooth sky + point sources, not fine texture, so '
                        'malvar remains the default.')
    g_frames.add_argument('--white-balance', choices=['none', 'grayworld', 'whitepatch'], default='grayworld')
    g_frames.add_argument('--no-bayer-autodetect', dest='bayer_autodetect', action='store_false',
                   default=True,
                   help='Disable Bayer row-orientation autodetection (see autodetect_bayer_orientation '
                        'in src/debayer.py): some capture software writes a BAYERPAT header that '
                        'does not match the actual row orientation of the pixel data, and by default '
                        'the session\'s reference frame is checked once and corrected if the G1/G2 '
                        'green sub-pixel imbalance exceeds the sensor-noise range. Disable if this '
                        'heuristic misfires on a legitimately imbalanced sensor/target.')
    g_stack.add_argument('--drizzle-scale', type=float, default=1.0,
                   help='Drizzle scale factor (e.g. 2.0 for 2x super-resolution, 1.0 = disabled)')
    g_stack.add_argument('--super-res-iters', type=int, default=0, metavar='N',
                   help='Iterative back-projection (IBP, Irani & Peleg 1991) super-resolution '
                        'refinement passes after drizzle (default: 0 = off; only meaningful '
                        'with --drizzle-scale > 1). Each iteration forward-simulates what '
                        'every original frame should look like given the current estimate '
                        '(inverse-warp + PSF blur), compares to what was actually observed, '
                        'and back-projects the residual correction. Genuine resolution gain '
                        'beyond drizzle\'s one-shot linear resample -- but like any inverse-'
                        'problem iteration, more isn\'t always better: it converges then '
                        'starts amplifying noise past a scene-dependent point (measured on '
                        'synthetic data: RMSE bottoms out around 5 iterations, then rises '
                        'again by ~10). 5 is a reasonable starting point; watch the output '
                        'rather than maximising N. '
                        'Needs a PSF estimate (same estimator as --drizzle-kernel psf/'
                        '--deconvolve rl) -- skipped with a warning if that fails. Not yet '
                        'compatible with --elastic-registration (tune step size via --config '
                        'ibp_relax, default %.1f).' % Config.IBP_RELAX)
    g_stack.add_argument('--drizzle-kernel', choices=['lanczos3', 'psf', 'magic'], default='lanczos3',
                   help='Drizzle resample kernel (default: lanczos3; only matters when '
                        '--drizzle-scale > 1). psf: build the resample kernel from the '
                        'session\'s own estimated PSF (Moffat fit to reference-frame stars, '
                        'same estimator as --deconvolve rl) via a windowed Wiener inverse '
                        'filter -- mild built-in sharpening during resample (tune '
                        'aggressiveness via --config drizzle_psf_wiener_k, default %.3f; '
                        'lower = sharper but more prone to ringing/noise gain). NOT simply '
                        'the raw PSF shape as the kernel -- that measurably broadens stars '
                        '(convolving an already-blurred profile with itself again), which '
                        'was verified and corrected during development. Cropped to a small '
                        '%dx%d tap radius (a resample kernel, not a full deconvolution pass '
                        '-- pair with --deconvolve for a bigger correction). Falls back to '
                        'lanczos3 if PSF estimation fails (too few reference stars) or with '
                        '--elastic-registration (the elastic warp path doesn\'t use this '
                        'kernel). magic: Costella\'s base Magic Kernel (quadratic B-spline, '
                        'provably non-negative -- ringing-free unlike Lanczos-3\'s negative '
                        'sidelobes, at the cost of being softer by construction). This is '
                        'the base kernel only, not the sharpened "Magic Kernel Sharp" '
                        'variant.' % (Config.DRIZZLE_PSF_WIENER_K, Config.DRIZZLE_PSF_KERNEL_SIZE,
                                      Config.DRIZZLE_PSF_KERNEL_SIZE))
    g_core.add_argument('--use-gpu', action='store_true',
                   help='Use CuPy for available operations (experimental)')
    g_out.add_argument('--plate-solve', action='store_true',
                   help='Enable plate solving via astrometry.net (requires ASTROMETRY_API_KEY)')
    g_out.add_argument('--annotate', action='store_true',
                   help='Circle and label bright stars and named deep-sky objects '
                        '(galaxies, nebulae, clusters) on a copy of the preview, via live '
                        'SIMBAD queries. Needs a WCS solution (--plate-solve, or a session '
                        'info.json that already has one) -- skipped with a message '
                        'otherwise. Writes <output>_annotated.jpg; the main FITS/TIFF/JPG '
                        'are untouched.')
    g_out.add_argument('--annotate-mag-limit', type=float, default=9.0, metavar='MAG',
                   help='--annotate: only label stars brighter than this V magnitude '
                        '(default: 9.0; lower = fewer, brighter-only stars)')
    g_post.add_argument('--no-background-extraction', dest='background_extraction',
                   action='store_false',
                   help='Disable background extraction')
    g_post.add_argument('--bg-method', choices=['mesh', 'dbe', 'wavelet'], default='dbe',
                   help='Background extraction method (default: dbe). '
                        'mesh: legacy polynomial grid (fastest). '
                        'dbe: Dynamic Background Extraction, robust local regression '
                        '(native/Rust accelerated). '
                        'wavelet: starlet (a-trous) low-pass fit on the same sky-patch '
                        'samples as dbe -- a hard pixel-scale cutoff instead of a blur '
                        'radius, better at leaving faint extended nebulosity alone while '
                        'still flattening the sky (see --bg-wavelet-scales).')
    g_post.add_argument('--bg-wavelet-scales', type=int, default=6, metavar='N',
                   help='wavelet bg-method only: starlet scale count -- background is '
                        'the coarsest approximation after N dyadic scales, i.e. structure '
                        'smaller than roughly 2**N sample-grid cells is treated as sky and '
                        'removed, anything larger is left alone (default: 6). Lower = '
                        'more aggressive/smaller-scale gradient removal, risks eating '
                        'broad nebulosity; higher = leaves more large-scale gradient in.')
    g_post.add_argument('--denoise-strength', type=float, default=3.0,
                   help='Wavelet luma denoise threshold factor (default: 3.0)')
    g_post.add_argument('--denoise-strength-calibrate', action='store_true',
                   help='Calibrate --denoise-strength via a Noise2Self-style self-'
                        'supervised sweep on the stack itself instead of the default '
                        'SNR-based heuristic: masks a small random pixel subset, denoises '
                        'the rest, scores each candidate strength by how well it predicts '
                        'the masked pixels\' true values, picks the minimum -- no SNR '
                        'estimate needed, no ground truth needed. Only affects '
                        '--denoiser wavelet (the plain, non-adaptive path); no effect on '
                        'the default adaptive BayesShrink denoiser, which has no single '
                        'strength parameter to calibrate.')
    g_post.add_argument('--variance-stabilize', action='store_true',
                   help='Apply a generalized Anscombe transform to the luma plane before '
                        'wavelet denoising (both --denoiser wavelet and the adaptive '
                        'BayesShrink default), inverting it after. Shot noise on bright '
                        'pixels is Poisson, not Gaussian; BayesShrink\'s single per-subband '
                        'threshold (from the finest detail subband\'s MAD) implicitly '
                        'assumes uniform Gaussian noise, which the transform makes closer '
                        'to true everywhere in the frame rather than just near the sky '
                        'background level it was effectively tuned at. Gain/read-noise are '
                        'estimated from the image\'s own local mean-variance relationship, '
                        'not sensor specs. Chroma channels are untouched (not a photon-count '
                        'quantity). No effect with other --denoiser choices.')
    g_post.add_argument('--no-chroma-nr', dest='chroma_nr', action='store_false',
                   help='Disable chroma noise reduction')
    g_out.add_argument('--stretch', choices=['linear', 'arcsinh', 'ghs'], default='ghs',
                   help='Preview JPEG stretch method (default: ghs = Generalized Hyperbolic Stretch)')
    g_out.add_argument('--preview-black-sigma', type=float, default=0.0,
                   help='Preview black point, in sky-sigma above the sky median '
                        '(default: 0.0). Higher (e.g. 2.0) clips background noise '
                        'to black for a small target on empty sky; negative keeps '
                        'faint frame-filling nebulosity visible. Set per target by --auto.')
    g_post.add_argument('--repair-stars', action='store_true',
                   help='Rebuild saturated (flat-top) star cores from their unsaturated '
                        'wings via a per-channel Moffat fit — restores a natural peak and '
                        'star colour instead of a clipped white disk.')
    g_post.add_argument('--no-star-reduce', dest='star_reduce', action='store_false',
                   help='Disable star reduction')
    g_post.add_argument('--no-local-contrast', dest='local_contrast', action='store_false',
                   help='Disable multiscale local contrast enhancement')
    g_core.add_argument('-j', '--parallel', type=int, default=0,
                   help='Parallel workers for frame processing (default: 0=auto, 1=sequential)')
    g_sessions.add_argument('--merge', nargs='+', default=None, metavar='STACK.fits',
                   help='Incremental stacking: merge previously saved linear stacks '
                        '(the main output FITS of earlier runs, RAWSTACK=True) into '
                        'this run. Each is registered onto the current session\'s '
                        'grid (star-match affine, translation fallback) and combined '
                        'as a per-pixel mean weighted by its NFRAMES header inside '
                        'its warped footprint. Post-processing runs once on the '
                        'merged result, and the output is itself a mergeable linear '
                        'stack. No cross-session outlier rejection (each session '
                        'already rejected internally). Not supported with '
                        '--drizzle-scale > 1.')
    g_frames.add_argument('--no-ca-correction', dest='ca_correction', action='store_false',
                   help='Disable chromatic aberration correction')
    g_frames.add_argument('--trail-reject', action='store_true',
                   help='Detect and erase satellite/aircraft trails per frame before '
                        'stacking (Hough line detection + local-background inpaint). '
                        'Robust even at low frame counts where sigma-clip cannot reject '
                        'a trail seen in only one or two subs.')
    g_stack.add_argument('--local-normalize', action='store_true',
                   help='Additively match every frame background to the per-frame median '
                        'before the rejection combine (removes per-frame gradients from '
                        'moonlight / light-pollution drift / thin cloud, and sharpens '
                        'sigma-clip). Applies to rejection stack methods (not plain mean '
                        'or drizzle).')
    g_frames.add_argument('--no-cosmic-ray-rejection', dest='cosmic_ray_rejection',
                   action='store_false',
                   help='Disable per-frame cosmic ray rejection (L.A.Cosmic)')
    g_frames.add_argument('--cosmic-ray-rejection', dest='cosmic_ray_rejection',
                   action='store_true',
                   help='Force per-frame cosmic ray rejection even on deep stacks '
                        '(default: automatic — skipped when >=20 frames are stacked '
                        'with a rejection method, which removes cosmic rays per-pixel)')
    g_debug.add_argument('--log-level',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='WARNING',
                   help='Minimum log severity printed to stderr (default: WARNING). '
                        'Use DEBUG for verbose diagnostic output from all modules.')
    g_debug.add_argument('--log-file', default=None, metavar='PATH',
                   help='Write a full DEBUG-level log to this file in addition to console output.')
    g_debug.add_argument('--quality-report', default=None, metavar='PATH',
                   help='Write per-frame quality metrics CSV after Phase 1 '
                        '(columns: filename, snr, fwhm, star_count, quality_score, '
                        'accepted, rejection_reason)')
    g_debug.add_argument('--export-frames-dir', default=None, metavar='PATH',
                   help='Directory to write a stretched JPEG for every accepted frame after Phase 1')
    g_astrollm.add_argument('--astrollm', action='store_true',
                   help='Score the final stacked master with astrollm (separately-trained '
                        'defect/quality/category classifier), run as a per-image subprocess. '
                        'When --auto is also active (the default -- pass --no-auto to '
                        'disable), also samples 3 light frames spread through the session '
                        '(fast, ~8s each): the sampled category feeds the same target-'
                        'classification prior SIMBAD/header metadata uses, and a defect flag '
                        'nudges settings defensively (trail-reject, stronger chroma '
                        'denoising) -- never auto-rejects a frame, this model is still '
                        'finishing its first training run. Pair with --astrollm-score-all to '
                        'also score every accepted frame (much slower -- minutes, not '
                        'seconds, on a large session). Needs --astrollm-dir (or the '
                        'individual --astrollm-python/-script/-checkpoint overrides).')
    g_astrollm.add_argument('--astrollm-score-all', action='store_true',
                   help='Also score every accepted light frame with astrollm (not just '
                        'the fast 3-frame sample --astrollm always does), logging advisory '
                        'per-frame defect/stray-light flags and below-average quality_score '
                        'outliers. Meaningfully slower -- ~8s per frame, so minutes on a '
                        'large session -- since each call is a separate subprocess with its '
                        'own Python/torch startup cost, not just per-image compute. Requires '
                        '--astrollm.')
    g_astrollm.add_argument('--astrollm-dir', default=os.environ.get('ASTROLLM_DIR'), metavar='DIR',
                   help='astrollm repo root. Derives --astrollm-python '
                        '(DIR\\.venv\\Scripts\\python.exe), --astrollm-script (DIR\\infer.py), '
                        'and --astrollm-checkpoint (DIR\\checkpoints\\model.pt) from astrollm\'s '
                        'standard layout -- the three overrides below only need to be passed '
                        'individually if your layout differs. Defaults to the ASTROLLM_DIR '
                        'environment variable if set.')
    g_astrollm.add_argument('--astrollm-python', default=None, metavar='PATH',
                   help='Path to the astrollm venv\'s python.exe (override; default: derived '
                        'from --astrollm-dir)')
    g_astrollm.add_argument('--astrollm-script', default=None, metavar='PATH',
                   help='Path to astrollm\'s infer.py (override; default: derived from '
                        '--astrollm-dir)')
    g_astrollm.add_argument('--astrollm-checkpoint', default=None, metavar='PATH',
                   help='Path to the astrollm model checkpoint (override; default: '
                        'DIR\\checkpoints\\model.pt from --astrollm-dir). A relative path is '
                        'resolved against --astrollm-script\'s directory')
    g_astrollm.add_argument('--astrollm-workers', type=int, default=2, metavar='N',
                   help='Thread-pool size for per-frame astrollm scoring calls (default: 2). '
                        'Subprocess-bound (model load + inference in a separate process), '
                        'not CPU-bound, so a thread pool is used rather than ProcessPoolExecutor.')
    g_astrollm.add_argument('--astrollm-timeout', type=float, default=60.0, metavar='SEC',
                   help='Per-call subprocess timeout in seconds (default: 60)')
    g_out.add_argument('--plate-solver', choices=['astap', 'astrometry'], default='astrometry',
                   help='Plate solver backend: astap (fast, local) or '
                        'astrometry (nova.astrometry.net, requires API key). '
                        'ASTAP recommended when the binary is installed (~1 s vs 30–120 s online).')
    g_out.add_argument('--astap-path', default=None, metavar='PATH',
                   help='Explicit path to the ASTAP binary (auto-detected if omitted)')
    g_comet.add_argument('--comet-mode', action='store_true',
                   help='Enable comet nucleus tracking. Produces a second stack aligned '
                        'on the comet nucleus saved as <stem>_comet.fits.')
    g_comet.add_argument('--comet-blend-sigma', type=float, default=30.0, metavar='PX',
                   help='Gaussian blend radius (pixels) for comet+star blended composite '
                        '(default: 30). Larger = more comet-stack area in the blend.')
    g_comet.add_argument('--comet-xy', default=None, metavar='X,Y',
                   help='Approximate comet nucleus position in the reference frame '
                        '(pixels, comma-separated, e.g. "1024,768"). X=column, Y=row.')
    g_comet.add_argument('--comet-search-radius', type=float, default=50.0, metavar='PX',
                   help='Search radius (pixels) around predicted nucleus position for '
                        'frame-to-frame tracking (default: 50).')
    g_comet.add_argument('--comet-affine', action='store_true',
                   help='Apply rotation+scale correction (affine) on top of nucleus '
                        'translation shift in comet registration (default: translation only).')
    g_comet.add_argument('--coma-mask-radius', type=int, default=150, metavar='PX',
                   help='Circular exclusion radius (pixels) around the comet nucleus '
                        'used to prevent background extraction from sampling the coma '
                        '(default: 150). Only active when --comet-mode is set.')
    g_comet.add_argument('--comet-radial-renorm', action='store_true',
                   help='Apply radial renormalization filter to flatten the coma gradient '
                        'and reveal jets/structure. Saves result as <stem>_comet_renorm.fits.')
    g_comet.add_argument('--comet-larson-sekanina', action='store_true',
                   help='Apply Larson-Sekanina rotational difference filter to enhance '
                        'comet jet structure. Saves result as <stem>_comet_ls.fits.')
    g_comet.add_argument('--comet-ls-rotation', type=float, default=15.0, metavar='DEG',
                   help='Rotation angle (degrees) for the Larson-Sekanina filter '
                        '(default: 15).')
    g_comet.add_argument('--comet-designation', default=None, metavar='DESIG',
                   help='JPL Horizons designation for ephemeris-aided nucleus tracking '
                        '(e.g. "C/2023 A3").')
    g_comet.add_argument('--observer-site', default=None, metavar='SITE',
                   help='Observer location for ephemeris queries: MPC code (e.g. "G37") '
                        'or "lon,lat,elev" in decimal degrees/metres (e.g. "-2.5,51.4,50").')
    g_post.add_argument('--hdr-combine', default=None, metavar='SHORT_STACK.fits',
                   help='Blend a short-exposure stack into saturated regions of the main '
                        'stack for HDR targets (e.g. Orion Nebula core, globular clusters).')
    g_post.add_argument('--hdr-blend-mode', choices=['threshold', 'fusion'], default='threshold',
                   help='Blend algorithm for --hdr-combine (default: threshold). threshold: '
                        'a smooth sigmoid mask centred on the long stack\'s 98th-percentile '
                        'value (original behaviour). fusion: Mertens multiresolution exposure '
                        'fusion (src/exposure_fusion.py) -- blends through a Laplacian pyramid '
                        'weighted by local contrast/saturation/well-exposedness instead of a '
                        'single spatial mask, avoiding the seam a threshold blend can leave at '
                        'the transition band; costs more (several pyramid levels of '
                        'Gaussian-filter passes).')
    g_out.add_argument('--color-calibrate', action='store_true',
                   help='Apply photometric colour calibration after plate solving '
                        '(queries Gaia DR3). Requires --plate-solve.')
    g_out.add_argument('--color-calibrate-method', choices=['colorindex', 'spcc'],
                   default='colorindex',
                   help='Colour calibration algorithm for --color-calibrate (default: '
                        'colorindex). colorindex: fixed Gaia BP-RP -> B-V colour-index '
                        'formula. spcc: spectrophotometric-style -- integrates a '
                        'blackbody spectrum at each star\'s Gaia teff_gspphot against '
                        'per-channel response curves (falls back to colorindex per-star '
                        'when a star has no Teff estimate). Uses generic Gaussian R/G/B '
                        'response curves, not a measured curve for your specific camera '
                        '+ filters -- more physically grounded than colorindex, not a '
                        'claim of matching a specific sensor\'s real QE curve.')
    g_out.add_argument('--photometry', action='store_true',
                   help='Aperture-photometer the linear stack against Gaia DR3 and '
                        'write a calibrated stellar catalogue (<output>_photometry.csv) '
                        'plus MAGZP_R/G/B zero-point header keywords. Needs a WCS '
                        '(--plate-solve or a session info.json solve). Uses the site '
                        'GPS + observation time from info.json for an airmass term; '
                        'falls back to folding extinction into the zero point when '
                        'those are absent. OSC channels are mapped coarsely to Gaia '
                        'RP/G/BP -- absolute accuracy ~0.05 mag, differential better.')
    g_out.add_argument('--photometry-extinction-k', type=float, default=None,
                   metavar='MAG/AIRMASS',
                   help='Override the per-band atmospheric extinction coefficient used '
                        'by --photometry (applied to all of R/G/B). Default: nominal '
                        'R=0.09 G=0.15 B=0.23. Only matters when an airmass could be '
                        'derived.')
    g_out.add_argument('--fix-atmospheric-dispersion', action='store_true',
                       help='EXPERIMENTAL: shift R/B channels back toward the green '
                            'channel\'s position to correct chromatic atmospheric '
                            'refraction, derived from first principles (Filippenko 1982\'s '
                            'standard-atmosphere refractive-index formula) -- no established '
                            'software reference implementation exists for this, unlike this '
                            'pipeline\'s other corrections (real ADCs are physical prism '
                            'hardware). Requires --plate-scale, --zenith-angle, and '
                            '--parallactic-angle (all in the units noted below) -- none are '
                            'auto-derived, since getting the geometry wrong shifts colour '
                            'channels the WRONG way and actively worsens the image. Not '
                            'wired into --auto. Most useful for low-altitude targets '
                            '(zenith angle > ~40deg).')
    g_out.add_argument('--plate-scale', type=float, default=None, metavar='ARCSEC/PX',
                       help='Image plate scale for --fix-atmospheric-dispersion.')
    g_out.add_argument('--zenith-angle', type=float, default=None, metavar='DEG',
                       help='Target zenith angle (90 - altitude) at capture time, for '
                            '--fix-atmospheric-dispersion.')
    g_out.add_argument('--parallactic-angle', type=float, default=None, metavar='DEG',
                       help='On-detector direction of "toward zenith" (0=up/+y, 90=right/+x), '
                            'for --fix-atmospheric-dispersion. Depends on site latitude, '
                            'target RA/Dec, and capture time -- not derived by this tool.')
    g_out.add_argument('--matched-filter', action='store_true',
                       help='Write <stem>_matched_filter.fits: the stacked image correlated '
                            'with its own estimated PSF -- the matched filter theorem\'s '
                            'provably SNR-optimal linear filter for detecting a known-shape '
                            '(point) source in white noise. A per-pixel SNR map when a '
                            'background noise estimate is available. Complementary to, not a '
                            'replacement for, --stack-method ivw (that\'s an optimal '
                            'per-frame combine weight; this is a post-stack detection '
                            'filter). Diagnostic/utility output, no effect on the main stack.')
    g_out.add_argument('--nmf-separate', action='store_true',
                       help='Non-negative matrix factorization: split the stacked image '
                            'into a stellar component and a nebular component (2 sources, '
                            'each with its own per-channel spectral signature and a '
                            'spatially-varying activation map), written as '
                            '<stem>_star_component.fits / <stem>_nebula_component.fits. '
                            'An alternative to --no-remove-stars\'s inpainting for use cases '
                            'that want the star signal separated out, not discarded. '
                            'Non-convex optimisation with no convergence guarantee -- '
                            'inspect the output before trusting the split, especially if '
                            'the log notes it as ambiguous.')
    g_out.add_argument('--dither-report', action='store_true',
                   help='Analyse how uniformly the session\'s sub-pixel dither offsets '
                        'sample the output grid, and warn if coverage is significantly '
                        'non-uniform (drizzle output quality is bounded by this, and '
                        'nothing else in the pipeline measures it). Writes '
                        '<stem>_dither.png. Diagnostic only -- no effect on the stack.')
    g_out.add_argument('--aberration-report', action='store_true',
                   help='Analyse star FWHM/elongation across the field and diagnose '
                        'sensor tilt, field curvature, and backfocus (spacing) errors. '
                        'Writes <stem>_aberration.png and FITS header keywords.')
    g_sessions.add_argument('--keep-checkpoint', action='store_true',
                   help='Keep the raw pre-post-processing stack after a successful run. '
                        'Re-running skips phases 1–3 so you can iterate on post-processing '
                        'settings quickly.')
    g_sessions.add_argument('--no-resume', action='store_true',
                   help='Ignore any existing checkpoint and start from scratch.')
    g_sessions.add_argument('--combine-sessions', action='store_true',
                   help='Pool all light frames from every subfolder into a single unified '
                        'stack instead of stacking each subfolder separately. This is now '
                        'the default whenever subfolders are found (unless --mosaic or '
                        '--hierarchical) -- this flag only matters to force it back on '
                        'after an explicit --hierarchical.')
    g_sessions.add_argument('--hierarchical', action='store_true',
                   help='Opt back into the pre-default behavior: stack each subfolder '
                        'separately, then combine the per-subfolder stacks by registration '
                        '(or --mosaic, if given). Has no effect unless the input directory '
                        'contains subfolders.')
    g_sessions.add_argument('--mosaic', action='store_true',
                   help='Stitch per-subfolder stacks into a mosaic via WCS reprojection. '
                        'Requires: pip install reproject and a working plate solver. '
                        'Automatically enables --plate-solve.')
    g_core.add_argument('--no-auto', dest='auto', action='store_false', default=True,
                   help='Disable the auto advisor: classify the target after Phase 1 and '
                        'apply optimised settings automatically (on by default, no API key '
                        'required). Explicit CLI flags you pass still override whatever '
                        'the advisor would have picked.')
    g_post.add_argument('--scnr', action='store_true',
                   help='Apply Subtractive Chromatic Noise Reduction to suppress green '
                        'cast artefacts common in OSC/DSLR images under light pollution.')
    g_post.add_argument('--photometric-calibration', action='store_true',
                   help='Apply gray-locus photometric colour calibration. '
                        'No external dependencies required.')

    # New feature flags (improvements 1-9)
    g_stack.add_argument('--drizzle-pixfrac', type=float, default=1.0, metavar='P',
                   help='Drizzle pixel fraction (tent-kernel weight; < 1.0 = sharper '
                        'at cost of noise; default: 1.0).')
    g_post.add_argument('--halo-removal', action='store_true',
                   help='Fit and subtract Gaussian PSF halos from bright stars in the '
                        'stacked image (post-processing step).')
    g_post.add_argument('--galaxy-mode', action='store_true',
                   help='Exclude a generous ellipse around the detected galaxy/extended '
                        'source from background extraction sampling, so its broad, '
                        'low-contrast halo is not fit and subtracted as a background '
                        'gradient (a smooth-surface model cannot otherwise tell a '
                        'tapering galaxy halo apart from real gradient at its edge). The '
                        'ellipse is fit to the object\'s own second-moment shape, so it '
                        'tracks real elongation instead of assuming a circle. '
                        'Auto-enabled by --auto for galaxy-leaning targets; pass this '
                        'explicitly under --no-auto.')
    g_post.add_argument('--galaxy-mask-radius', type=float, default=None, metavar='PX',
                   help='Overrides the fitted ellipse\'s semi-major axis length (pixels); '
                        'the semi-minor axis and orientation still come from the fit. '
                        'Default: sized automatically from the detected object. Only '
                        'active with --galaxy-mode. Ignored if --galaxy-center is set.')
    g_post.add_argument('--galaxy-center', type=str, default=None, metavar='X,Y',
                   help='Skip automatic extended-source detection entirely and center the '
                        '--galaxy-mode exclusion on this pixel coordinate instead (comma-'
                        'separated, e.g. "1420,930" -- X is the column, Y is the row, same '
                        'orientation as image-viewer pixel coordinates on the stacked '
                        'output). A circle of radius --galaxy-mask-radius (default 200px) '
                        'is used, since there is no fitted shape to take the aspect ratio '
                        'from. For fields where auto-detection keeps finding the wrong '
                        'object (a brighter star, a vignetting corner, etc. -- confirmed on '
                        'a real dense-star-field target) -- open the stacked preview, read '
                        'off the galaxy\'s pixel position, and pin it directly.')
    g_post.add_argument('--no-remove-stars', dest='remove_stars', action='store_false',
                   help='Disable star removal. By default, detected stars are '
                        'inpainted with local background (normalised-convolution '
                        'fill, per-star radius scaled to brightness) and saved as a '
                        '<output>_starless.fits sidecar. The main output is '
                        'untouched; the sidecar is for downstream nebula/background '
                        'work (aggressive stretch, external star recombination). '
                        'Computed last, on the fully post-processed image.')

    # Defaults for parameters that are tunable via config file but not exposed on the CLI.
    # Set these in a TOML config with --config to override them.
    p.set_defaults(
        # Features that are on by default (disabled via --no-* flags)
        background_extraction=True,
        # Primary luma denoiser when --denoiser is left at 'auto' and no
        # preset/--auto overrides it: directional (curvelet-inspired)
        # adaptive wavelet, not plain adaptive_wavelet_denoise. At
        # protect_strength=0.6 it's a strict superset of plain wavelet's
        # protection -- isotropic/noise-only regions get the same BayesShrink
        # threshold either way, coherent/elongated structure (galaxy arms,
        # nebula filaments) gets a reduced one. Plain wavelet erasing real
        # galaxy detail (dust lanes, spiral structure -- low-contrast,
        # sitting close to the noise floor) on a run with no --auto/preset
        # was reported and reproduced; curvelet is the existing, purpose-built
        # fix, just not previously the default. `--denoiser wavelet` still
        # gives the plain (unprotected) behavior explicitly if ever wanted.
        denoise=False,
        denoise_curvelet=True,
        auto_denoise_strength=True,
        denoise_adaptive=True,
        chroma_nr=True,
        star_reduce=True,
        local_contrast=True,
        ca_correction=True,
        cosmic_ray_rejection=None,  # tri-state: None = auto (see pipeline.py)
        # Denoiser tuning
        denoise_chroma_boost=2.0,
        chroma_nr_sigma=2.0,
        chroma_nr_large_sigma=0.0,
        chroma_nr_large_strength=0.7,
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
        # skimage phase cross-correlation before the FFT branch is permanently
        # skipped: on real astro frames it almost always fails its error<0.1
        # acceptance test, wasting an upsampled correlation per frame. Config-
        # file override (skip_phase_correlation=false) remains for experiments.
        skip_phase_correlation=True,
        no_alignment_centrality=False,
        no_shift_outlier_filter=False,
        # no_reg_residual_check / reg_residual_reject are real CLI flags now
        # (--no-reg-residual-check / --no-reg-residual-reject above).
        patch_registration=False,
        # Stacking internals
        percentile_low=20.0,
        percentile_high=80.0,
        esd_max_outliers=0,
        esd_significance=0.05,
        linear_fit_sigma_low=4.0,
        linear_fit_sigma_high=2.0,
        linear_fit_iters=5,
        ivw_gain=None,  # electrons/ADU; None = per-frame-constant noise weight (no shot-noise term)
        drizzle_psf_wiener_k=Config.DRIZZLE_PSF_WIENER_K,  # used only with --drizzle-kernel psf
        wavelet_combine_levels=4,  # used only with --stack-method wavelet
        directional_protect_strength=0.6,  # used only with --denoiser curvelet
        deconvolve_sparse_lambda=0.01,  # used only with --deconvolve sparse
        ibp_relax=Config.IBP_RELAX,  # used only with --super-res-iters
        # Quality
        # (advanced_metrics has no CLI flag -- set via --config only)
        # Per-feature tuning
        star_reduce_factor=0.4,
        star_reduce_sigma=1.5,
        local_contrast_strength=0.7,
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
        weight_snr=1.0,
        weight_fwhm=1.0,
        weight_stars=1.0,
        weight_noise=False,
        # Consolidated under --denoiser / --deconvolve / --export / --debug;
        # individually overridable via --config.
        denoise_nlm=False,
        denoise_bilateral=False,
        denoise_mmt=False,
        denoise_acdnr=False,
        denoise_bm3d=False,
        denoise_aniso=False,
        deconvolve=False,
        deconvolve_tv=False,
        deconvolve_svpsf=False,
        deconvolve_sv_tiles=3,
        deconvolve_blind_psf=False,
        output_tiff=False,
        output_xisf=False,
        keep_intermediates=False,
        diagnostic=False,
        diagnostic_dir=None,
        debug_registration=False,
        export_masks=False,
        ghs_b=8.0,
        ghs_sp=0.15,
        ghs_hp=0.95,
        rejection_estimator='mad',
        advanced_metrics=False,
        # Improvements 1-9
        max_ellipticity=0.5,
        consensus_ref=False,
        masked_correlation=False,
        pre_gradient_removal=False,
        drizzle_pixfrac=1.0,
        halo_removal=False,
        remove_stars=True,
        galaxy_mode=False,
    )
    return p


def parse_args(argv=None):
    """Parse *argv* (default: ``sys.argv[1:]``) into an args Namespace.

    Accepting an explicit *argv* (rather than always reading ``sys.argv``)
    lets a caller other than the CLI entry point -- e.g. the desktop app's
    ``POST /api/start`` handler, building a synthetic argv from a submitted
    form -- construct a real, fully-validated Namespace by going through the
    same parser/preset/config/derived-flags logic every CLI invocation uses,
    instead of re-implementing it.
    """
    p = build_parser()
    _argv = sys.argv[1:] if argv is None else argv
    # Record which dests the user actually typed a flag for, before parsing
    # fills in defaults -- apply_preset/load_config_file need this to honour
    # "explicit CLI flags win over preset/config values" (they can't tell an
    # explicit flag from a default by inspecting the parsed value alone, since
    # a user might explicitly pass the same value as the default).
    def _passed_on_cli(action) -> bool:
        return any(a == opt or a.startswith(opt + '=')
                   for a in _argv for opt in action.option_strings)
    _explicit_dests = {a.dest for a in p._actions if _passed_on_cli(a)}

    args = p.parse_args(_argv)
    args._explicit_cli_dests = _explicit_dests

    # ── Map the consolidated CLI surface onto the internal per-feature flags
    # (presets, --config, and the auto-advisor all operate on the internal
    # attributes, so everything downstream is unchanged). ──
    if args.denoiser != 'auto':
        d = args.denoiser
        args.denoise = (d == 'wavelet')
        args.denoise_mmt = (d == 'mmt')
        args.denoise_bm3d = (d == 'bm3d')
        args.denoise_acdnr = (d == 'acdnr')
        args.denoise_nlm = (d == 'nlm')
        args.denoise_bilateral = (d == 'bilateral')
        args.denoise_aniso = (d == 'aniso')
        args.denoise_curvelet = (d == 'curvelet')
    args.deconvolve = args.deconvolve_mode in ('rl', 'tv', 'rl-sv', 'sparse')
    args.deconvolve_tv = (args.deconvolve_mode == 'tv')
    args.deconvolve_svpsf = (args.deconvolve_mode == 'rl-sv')
    args.deconvolve_sparse = (args.deconvolve_mode == 'sparse')

    # The consolidated flags above (--denoiser, --deconvolve) are user-facing
    # aliases whose dests (denoiser, deconvolve_mode) differ from the internal
    # per-feature attributes that presets / --config / the auto-advisor read.
    # Propagate "explicit" status onto those derived attrs so an explicit
    # --denoiser / --deconvolve wins over preset & auto just like a direct flag.
    if 'denoiser' in _explicit_dests:
        _explicit_dests.update({
            'denoise', 'denoise_mmt', 'denoise_bm3d', 'denoise_acdnr',
            'denoise_nlm', 'denoise_bilateral', 'denoise_aniso', 'denoise_curvelet'})
    if 'deconvolve_mode' in _explicit_dests:
        _explicit_dests.update({'deconvolve', 'deconvolve_tv', 'deconvolve_sparse'})

    _exp = [e.strip() for e in (args.export or '').split(',') if e.strip()]
    for e in _exp:
        if e not in ('tiff', 'xisf'):
            p.error(f"--export: unknown format '{e}' (choose from: tiff, xisf)")
    args.output_tiff = 'tiff' in _exp
    args.output_xisf = 'xisf' in _exp

    _dbg = [d.strip() for d in (args.debug or '').split(',') if d.strip()]
    for d in _dbg:
        if d not in ('registration', 'diagnostic', 'intermediates', 'masks'):
            p.error(f"--debug: unknown kind '{d}' (choose from: registration, "
                    f"diagnostic, intermediates, masks)")
    args.debug_registration = 'registration' in _dbg
    args.diagnostic = 'diagnostic' in _dbg
    args.keep_intermediates = 'intermediates' in _dbg
    args.export_masks = 'masks' in _dbg

    if args.astrollm:
        # --astrollm-dir derives the three individual paths from astrollm's
        # standard repo layout; an explicit --astrollm-python/-script/
        # -checkpoint always wins over the derived value.
        if args.astrollm_dir:
            if not args.astrollm_python:
                args.astrollm_python = os.path.join(args.astrollm_dir, '.venv', 'Scripts', 'python.exe')
            if not args.astrollm_script:
                args.astrollm_script = os.path.join(args.astrollm_dir, 'infer.py')
            if not args.astrollm_checkpoint:
                args.astrollm_checkpoint = os.path.join(args.astrollm_dir, 'checkpoints', 'model.pt')
        if args.astrollm_checkpoint and args.astrollm_script and not os.path.isabs(args.astrollm_checkpoint):
            args.astrollm_checkpoint = os.path.join(
                os.path.dirname(args.astrollm_script), args.astrollm_checkpoint)
        missing = [name for name, val in (
            ('--astrollm-python', args.astrollm_python),
            ('--astrollm-script', args.astrollm_script),
            ('--astrollm-checkpoint', args.astrollm_checkpoint),
        ) if not val]
        bad_paths = [name for name, val in (
            ('--astrollm-python', args.astrollm_python),
            ('--astrollm-script', args.astrollm_script),
            ('--astrollm-checkpoint', args.astrollm_checkpoint),
        ) if val and not os.path.exists(val)]
        if missing:
            safe_print(f"  WARNING: --astrollm requires {', '.join(missing)} -- disabling astrollm scoring")
            args.astrollm = False
        elif bad_paths:
            safe_print(f"  WARNING: --astrollm path(s) not found: {', '.join(bad_paths)} -- disabling astrollm scoring")
            args.astrollm = False

    if getattr(args, 'astrollm_score_all', False) and not args.astrollm:
        # Covers both "never passed --astrollm" and "--astrollm got disabled
        # just above for missing/bad paths" -- either way score_all's own
        # gate (astrollm AND astrollm_score_all, checked in src/astrollm.py
        # too) makes it a silent no-op otherwise, which is easy to mistake
        # for "ran but found nothing" rather than "didn't run at all".
        safe_print("  WARNING: --astrollm-score-all has no effect without --astrollm")

    return args


def apply_post_parse_setup(args: argparse.Namespace) -> None:
    """Everything ``main()`` does between ``parse_args()`` and calling
    ``process_directory()``: default the output path, load ``--config``,
    apply ``--preset``, initialise logging, and (re)initialise the GPU
    context. Shared by ``main()`` (the CLI entry point) and
    ``src.desktop_control.RunManager._run`` (the desktop app's GUI-triggered
    runs), which calls ``process_directory()`` directly and therefore needs
    this same setup without going through ``main()`` itself. Kept as one
    function specifically so the two callers can't drift apart the way they
    already had (the desktop app was silently missing the "no output path
    specified" notice before this was extracted)."""
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
    apply_preset(args)
    # debug_registration implies verbose
    if args.debug_registration:
        args.verbose = True
    # Initialise structured logging before anything else
    setup_logging(level=getattr(args, 'log_level', 'WARNING'),
                  log_file=getattr(args, 'log_file', None))
    # Initialise/reset the GPU context (module-level singleton). reset_gpu
    # frees the outgoing context's cupy pool first -- a no-op for main()'s
    # one-shot process, but load-bearing for a long-lived caller (the
    # desktop app) that can run this more than once per process.
    reset_gpu(use_gpu=args.use_gpu)


def main():
    # Dispatch to the 'combine' subcommand without touching the main parser
    if len(sys.argv) > 1 and sys.argv[1] == 'combine':
        from src.channel_combine import run_combine_cli
        run_combine_cli(sys.argv[2:])
        return

    args = parse_args()

    # Collection maintenance modes: no output path, no stacking.
    if getattr(args, 'sweep_undo', False):
        from src.quality_sweep import undo_quality_sweep
        sys.exit(undo_quality_sweep(args.directory))
    if getattr(args, 'quality_sweep', False):
        from src.quality_sweep import run_quality_sweep
        sys.exit(run_quality_sweep(args.directory, args))

    apply_post_parse_setup(args)

    if getattr(args, 'live', False) and getattr(args, 'stream', False):
        safe_print("  ERROR: --live and --stream are mutually exclusive "
                   "(--live watches a growing directory forever; --stream "
                   "expects an already-complete one)")
        raise SystemExit(1)

    if getattr(args, 'stream', False) and getattr(args, 'merge', None):
        safe_print("  ERROR: --stream does not support --merge "
                   "(the streaming two-pass stacker has no cross-session combine "
                   "step yet; run a normal stack with --merge instead)")
        raise SystemExit(1)

    # Real-time (live) stacking: watch the directory and stack subs as they land.
    if getattr(args, 'live', False):
        from src.live_stack import run_live_stack
        raise SystemExit(run_live_stack(args))

    # Two-pass streaming stack of an already-complete directory.
    if getattr(args, 'stream', False):
        from src.stream_stack import run_stream_stack
        raise SystemExit(run_stream_stack(args))

    try:
        process_directory(args.directory, args.output, args)
    except KeyboardInterrupt:
        safe_print("\n  Interrupted (Ctrl-C).")
        if not getattr(args, 'no_resume', False):
            safe_print("  Progress through the last completed phase was checkpointed "
                      "— rerun the same command to resume.")
        raise SystemExit(130)  # 128 + SIGINT, standard convention
    except Exception as e:
        print(f'ERROR: {str(e)}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
