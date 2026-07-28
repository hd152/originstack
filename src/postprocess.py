"""Post-processing pipeline: hot pixel removal, background extraction, denoising, deconvolution."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from src.models import Config, FrameInfo, ProcessingStats
from src.utils import safe_print, format_time
from src.quality import generate_star_mask, detect_stars_auto
from src.background import (apply_background_extraction, remove_sky_residual,
                            gaussian_filter_ds,
                            sky_floor_normalize, dynamic_background_extraction,
                            wavelet_background_extraction,
                            _border_pixels, _sigma_sky)
from src.denoising import (wavelet_denoise, adaptive_wavelet_denoise, nlm_denoise,
                           bilateral_denoise, reduce_chroma_noise,
                           estimate_denoise_strength, reduce_stars,
                           multiscale_local_contrast, mmt_denoise, acdnr_denoise,
                           bm3d_denoise, anisotropic_diffusion, scnr,
                           remove_star_halos, radial_renormalize, larson_sekanina)
from src.psf_deconvolution import (estimate_psf, make_synthetic_psf,
                                    richardson_lucy_deconvolve,
                                    estimate_psf_blind, tv_regularized_deconvolve)
from src.photometric_calibration import photometric_color_calibrate, try_gaia_calibration

try:
    from astropy.stats import sigma_clipped_stats
except Exception:
    sigma_clipped_stats = None


def _diag_save(img: np.ndarray, diag_dir: Optional[str], counter: list, slug: str) -> None:
    """Save a float32 FITS snapshot to diag_dir if diagnostic mode is active.

    counter is a one-element list [int] so the caller can mutate it across calls
    without a nonlocal statement. Only called when the step actually runs.

    Also publishes a live preview to the web view (independent of diagnostic
    mode; a no-op unless --web-view is active, and throttled internally).
    """
    try:
        from src.webview import get_webview
        wv = get_webview()
        if wv.active:
            wv.preview(img, f"Post-processing: {slug.replace('_', ' ')}",
                       args=None)
    except Exception:
        pass
    if not diag_dir:
        return
    try:
        from astropy.io import fits as _fits
        fname = f"{counter[0]:02d}_{slug}.fits"
        path = os.path.join(diag_dir, fname)
        data_out = np.transpose(img, (2, 0, 1)).astype(np.float32)
        hdu = _fits.PrimaryHDU(data=data_out)
        hdu.header['DIAGSTEP'] = (slug, 'Diagnostic step name')
        hdu.header['DIAGIDX'] = (counter[0], 'Diagnostic step index')
        hdu.writeto(path, overwrite=True)
        safe_print(f"  [diagnostic] saved {fname}")
    except Exception as e:
        safe_print(f"  [diagnostic] WARNING: could not save {slug}: {e}")
    finally:
        counter[0] += 1


# ---------------------------------------------------------------------------
# External tool helpers
# ---------------------------------------------------------------------------

def _find_binary(names: list) -> Optional[str]:
    """Search PATH and common install locations for any of the given binary names."""
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _write_u16_tiff(img: np.ndarray, path: str) -> None:
    """Write (H, W, 3) float32 image as 16-bit RGB TIFF for external tools."""
    arr16 = (np.clip(img, 0.0, 1.0) * 65535).astype(np.uint16)
    try:
        import tifffile
        tifffile.imwrite(path, arr16, photometric="rgb")
    except ImportError:
        try:
            from PIL import Image
            Image.fromarray(arr16).save(path)
        except Exception as e:
            raise RuntimeError(f"Cannot write TIFF: tifffile and Pillow both unavailable ({e})")


def _read_tiff_as_float(path: str) -> np.ndarray:
    """Read a TIFF and return (H, W, 3) float32 in [0, 1]."""
    try:
        import tifffile
        arr = tifffile.imread(path)
    except ImportError:
        from PIL import Image
        arr = np.array(Image.open(path))
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _save_sidecar_fits(img: np.ndarray, output_path: str, suffix: str) -> None:
    """Save a (H, W, 3) or (H, W) float32 array as a sidecar FITS file."""
    from astropy.io import fits as _fits
    path = os.path.splitext(output_path)[0] + suffix + ".fits"
    if img.ndim == 3:
        data_out = np.transpose(img.astype(np.float32), (2, 0, 1))
    else:
        data_out = img.astype(np.float32)
    hdu = _fits.PrimaryHDU(data=data_out)
    hdu.header["CREATOR"] = "astro_stack.py postprocess"
    hdu.header["COMBINED"] = (True, "Sidecar output")
    hdu.writeto(path, overwrite=True)
    safe_print(f"  Saved: {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# GraXpert background extraction
# ---------------------------------------------------------------------------

def _find_graxpert_binary() -> Optional[str]:
    return _find_binary([
        "graxpert", "GraXpert",
        "/usr/local/bin/graxpert",
        r"C:\Program Files\GraXpert\graxpert.exe",
        "/Applications/GraXpert.app/Contents/MacOS/GraXpert",
    ])


def _graxpert_background_extraction(img: np.ndarray,
                                     graxpert_path: Optional[str] = None,
                                     verbose: bool = False) -> np.ndarray:
    """Call the GraXpert CLI for AI-powered gradient removal.

    Falls back to DBE on failure.
    """
    from astropy.io import fits as _fits

    # Prefer the graxpert Python package over the subprocess CLI when available.
    try:
        from graxpert.background_extraction import extract_background as _gx_extract
        img_max = float(img.max()) or 1.0
        img_norm = (img / img_max).astype(np.float32)
        result = _gx_extract(
            img_norm,
            bg_pts=None,
            correction_type='Subtraction',
            smoothing='medium',
            kernel_size=50,
        )
        if verbose:
            safe_print("  [GraXpert] Background extracted via Python API")
        return np.clip(
            np.asarray(result, dtype=np.float32) * img_max, 0.0, None
        )
    except ImportError:
        pass  # Python package not installed — fall through to subprocess
    except Exception as _gx_err:
        if verbose:
            safe_print(f"  [GraXpert] Python API failed ({_gx_err}), trying subprocess")

    binary = graxpert_path or _find_graxpert_binary()
    if binary is None:
        safe_print("  WARNING: GraXpert binary not found — falling back to DBE")
        return dynamic_background_extraction(img, verbose=verbose)

    try:
        with tempfile.TemporaryDirectory() as td:
            in_fits = os.path.join(td, "graxpert_in.fits")
            out_fits = os.path.join(td, "graxpert_out.fits")

            # Write (3, H, W) float32 FITS
            data_out = np.transpose(img.astype(np.float32), (2, 0, 1))
            _fits.PrimaryHDU(data=data_out).writeto(in_fits, overwrite=True)

            cmd = [binary, "-i", in_fits, "-o", out_fits, "background_extraction"]
            if verbose:
                print(f"  [GraXpert] Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                safe_print(f"  WARNING: GraXpert exited with code {result.returncode} "
                           "— falling back to DBE")
                if verbose and result.stderr:
                    print(f"    GraXpert stderr: {result.stderr[:200]}")
                return dynamic_background_extraction(img, verbose=verbose)

            if not os.path.exists(out_fits):
                safe_print("  WARNING: GraXpert produced no output — falling back to DBE")
                return dynamic_background_extraction(img, verbose=verbose)

            with _fits.open(out_fits) as hdul:
                out_data = hdul[0].data.astype(np.float32)
            if out_data.ndim == 3 and out_data.shape[0] == 3:
                return np.transpose(out_data, (1, 2, 0))
            return out_data
    except subprocess.TimeoutExpired:
        safe_print("  WARNING: GraXpert timed out — falling back to DBE")
        return dynamic_background_extraction(img, verbose=verbose)
    except Exception as e:
        safe_print(f"  WARNING: GraXpert error ({e}) — falling back to DBE")
        return dynamic_background_extraction(img, verbose=verbose)


# ---------------------------------------------------------------------------
# Starnet++ star removal
# ---------------------------------------------------------------------------

def _find_starnet_binary(hint: Optional[str] = None) -> Optional[str]:
    if hint and os.path.isfile(hint):
        return hint
    return _find_binary([
        "starnet++", "starnet", "StarNet++",
        "/usr/local/bin/starnet++",
        r"C:\Program Files\StarNet++\starnet++.exe",
        r"C:\Program Files\StarNet\starnet++.exe",
    ])


def apply_starnet(img: np.ndarray,
                  output_path: str,
                  binary_path: Optional[str] = None,
                  verbose: bool = False) -> np.ndarray:
    """Remove stars from the image using the Starnet++ binary.

    Saves:
      <stem>_starless.fits  — star-removed image
      <stem>_stars.fits     — stars-only layer (img − starless)

    Returns the starless image (or the original if Starnet++ fails).
    The caller continues post-processing on the returned starless image.
    Stars can be re-added later with: final = starless + stars_only
    """
    binary = _find_starnet_binary(binary_path)
    if binary is None:
        safe_print(
            "  WARNING: Starnet++ binary not found — star removal skipped.\n"
            "  Install from https://www.starnetastro.com/ and ensure it is on PATH."
        )
        return img

    try:
        with tempfile.TemporaryDirectory() as td:
            in_tiff  = os.path.join(td, "sn_input.tif")
            out_tiff = os.path.join(td, "sn_output.tif")

            # Starnet++ requires 16-bit RGB TIFF
            _write_u16_tiff(img, in_tiff)

            cmd = [binary, in_tiff, out_tiff]
            if verbose:
                print(f"  [Starnet++] Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0 or not os.path.exists(out_tiff):
                safe_print(f"  WARNING: Starnet++ failed (code {result.returncode}) "
                           "— star removal skipped")
                if verbose and result.stderr:
                    print(f"    stderr: {result.stderr[:200]}")
                return img

            starless = _read_tiff_as_float(out_tiff)
            if starless.shape != img.shape:
                safe_print("  WARNING: Starnet++ output shape mismatch — star removal skipped")
                return img

        stars_only = np.clip(img - starless, 0.0, None)
        if output_path:
            _save_sidecar_fits(starless,   output_path, "_starless")
            _save_sidecar_fits(stars_only, output_path, "_stars")

        safe_print(f"  ✓ Star removal complete (starless + stars layers saved)")
        return starless

    except subprocess.TimeoutExpired:
        safe_print("  WARNING: Starnet++ timed out (600 s) — star removal skipped")
        return img
    except Exception as e:
        safe_print(f"  WARNING: Starnet++ error ({e}) — star removal skipped")
        return img


def _sanitize(img: np.ndarray, step_name: str = "") -> np.ndarray:
    """Replace NaN/Inf with zero and clip negatives."""
    if not np.isfinite(img).all():
        n_bad = int(np.sum(~np.isfinite(img)))
        safe_print(f"    ⚠ Sanitized {n_bad} non-finite pixels after {step_name}")
        np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return img


def postprocess_stack(
    stacked: np.ndarray,
    args: argparse.Namespace,
    final: List[FrameInfo],
    stats: ProcessingStats,
) -> np.ndarray:
    """Apply all post-processing steps to the stacked image and return it."""
    skip_steps = set(getattr(args, 'skip_step', []) or [])
    _diag_dir = getattr(args, '_diagnostic_dir', None)
    _diag_counter = [1]

    # Per-channel hot pixel removal
    if 'hot_pixel' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_hot_pixel')
        print("\n  Removing residual hot pixels (per-channel)...")
        _hp_start = time.time()
        _hp_fixed = 0
        _hp_meds = ndimage.median_filter(stacked, size=(5, 5, 1))
        _hp_diffs = stacked - _hp_meds
        _hp_mads = np.median(np.abs(_hp_diffs), axis=(0, 1))
        _hp_sigmas = np.maximum(_hp_mads * 1.4826, 1e-6)
        ch_spikes = _hp_diffs / _hp_sigmas[np.newaxis, np.newaxis, :]
        for c in range(3):
            others = [i for i in range(3) if i != c]
            other_normal = np.all(ch_spikes[:, :, others] < 4.0, axis=2)
            hot = (ch_spikes[:, :, c] > 12.0) & other_normal
            n_hot = int(np.sum(hot))
            if n_hot > 0:
                stacked[:, :, c][hot] = _hp_meds[:, :, c][hot]
                _hp_fixed += n_hot
        safe_print(f"  ✓ Per-channel hot pixel removal: {_hp_fixed} pixels fixed "
                   f"({format_time(time.time() - _hp_start)})")
        stacked = _sanitize(stacked, "hot pixel removal")

    # Detect stars once — reused by background extraction, wavelet, NLM, and deconvolution
    pp_star_mask = None
    _pp_sources = None
    if sigma_clipped_stats is not None:
        try:
            _pp_lum = (0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1]
                       + 0.114 * stacked[:, :, 2])
            _, _bg_med, _bg_std = sigma_clipped_stats(_pp_lum, sigma=3.0, maxiters=5)
            _pp_sources = detect_stars_auto(_pp_lum, float(_bg_std), background=float(_bg_med))
            if _pp_sources is not None and len(_pp_sources) > 0:
                pp_star_mask = generate_star_mask(_pp_lum.shape, _pp_sources, fwhm=4.0)
                if args.verbose:
                    safe_print(f"    Post-processing star mask: {len(_pp_sources)} stars")
        except Exception:
            pass

    # Field aberration / optical-tilt inspector (on the linear stack, before any
    # denoise/deconvolution reshapes the stars). Runs on the shared _pp_sources.
    if (getattr(args, 'aberration_report', False) or getattr(args, 'diagnostic', False)) \
            and _pp_sources is not None and len(_pp_sources) > 0:
        try:
            from src.aberration import analyze_field_aberration
            _ab_out = getattr(args, 'output', None)
            _ab_png = (os.path.splitext(_ab_out)[0] + '_aberration.png') if _ab_out else None
            _ab = analyze_field_aberration(_pp_lum, _pp_sources, output_png=_ab_png,
                                           verbose=args.verbose)
            if _ab is not None:
                args._aberration = _ab  # picked up by the FITS header writer
        except Exception as _abe:
            safe_print(f"  WARNING: aberration report failed: {_abe}")

    # Saturated star core repair (before background/denoise reshape the stars)
    if getattr(args, 'repair_stars', False) and 'repair_stars' not in skip_steps:
        _sr_start = time.time()
        from src.star_repair import repair_saturated_stars
        stacked = repair_saturated_stars(stacked, verbose=args.verbose)
        stacked = _sanitize(stacked, "saturated star repair")
        if args.verbose:
            safe_print(f"    ({format_time(time.time() - _sr_start)})")

    # Star halo removal (before background extraction)
    if getattr(args, 'halo_removal', False) and _pp_sources is not None and len(_pp_sources) > 0:
        _halo_fwhm = float(np.median(
            [f.metrics.get('fwhm', 4.0) for f in final
             if f.metrics and f.metrics.get('fwhm', 0) > 0]) or 4.0)
        print(f"\n  Removing star halos (fwhm={_halo_fwhm:.1f}px)...")
        _halo_start = time.time()
        stacked = remove_star_halos(stacked, _pp_sources, fwhm=_halo_fwhm)
        safe_print(f"  Star halo removal ({format_time(time.time() - _halo_start)})")
        stacked = _sanitize(stacked, "halo removal")

    # Export star/galaxy masks (before any post-processing modifies them)
    if getattr(args, 'export_masks', False) and pp_star_mask is not None:
        _output_path = getattr(args, 'output', None)
        if _output_path:
            _save_sidecar_fits(pp_star_mask.astype(np.float32), _output_path, "_star_mask")

    # 1. Background extraction (GraXpert / DBE / legacy mesh)
    _pre_bg = None
    if getattr(args, 'keep_intermediates', False) and args.background_extraction:
        _pre_bg = stacked.copy()
    if args.background_extraction and 'background' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_background')
        bg_start = time.time()

        bg_method = getattr(args, 'bg_method', 'dbe')

        # Build comet coma exclusion mask to prevent background sampler from
        # subtracting the comet itself as "background".
        _coma_excl_mask = None
        if getattr(args, 'comet_mode', False):
            try:
                from src.registration import find_comet_centroid
                _pp_lum_comet = (0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1]
                                 + 0.114 * stacked[:, :, 2])
                _coma_cy, _coma_cx = find_comet_centroid(_pp_lum_comet)
                _coma_radius = float(getattr(args, 'coma_mask_radius', 150))
                _H_c, _W_c = stacked.shape[:2]
                _yy_c, _xx_c = np.mgrid[:_H_c, :_W_c]
                _coma_dist2 = (_yy_c - _coma_cy) ** 2 + (_xx_c - _coma_cx) ** 2
                _coma_excl_mask = (_coma_dist2 <= _coma_radius ** 2).astype(np.float32)
                if getattr(args, 'verbose', False):
                    safe_print(f"    Coma exclusion mask: nucleus=({_coma_cx:.1f},{_coma_cy:.1f}), "
                               f"radius={_coma_radius:.0f}px")
            except Exception as _cme:
                safe_print(f"  WARNING: coma exclusion mask failed: {_cme}")

        if bg_method == 'graxpert':
            graxpert_bin = getattr(args, 'graxpert_path', None)
            print(f"\n  Applying GraXpert background extraction...")
            stacked = _graxpert_background_extraction(
                stacked, graxpert_path=graxpert_bin, verbose=args.verbose)
            safe_print(f"  ✓ GraXpert background extraction ({format_time(time.time() - bg_start)})")
        elif bg_method == 'dbe':
            dbe_patch = getattr(args, 'dbe_patch_size', Config.DBE_PATCH_SIZE)
            print(f"\n  Applying Dynamic Background Extraction "
                  f"(patch={dbe_patch}px, robust local regression, sigma={args.bg_clip_sigma})...")
            _entropy_bg = getattr(args, 'entropy_bg', False)
            stacked = dynamic_background_extraction(
                stacked, patch_size=dbe_patch, clip_sigma=args.bg_clip_sigma,
                verbose=args.verbose, star_mask=pp_star_mask,
                use_entropy_weights=_entropy_bg,
                exclusion_mask=_coma_excl_mask)
            safe_print(f"  ✓ Dynamic Background Extraction ({format_time(time.time() - bg_start)})")
        elif bg_method == 'wavelet':
            dbe_patch = getattr(args, 'dbe_patch_size', Config.DBE_PATCH_SIZE)
            wavelet_scales = getattr(args, 'bg_wavelet_scales', 6)
            print(f"\n  Applying starlet wavelet-band background extraction "
                  f"(patch={dbe_patch}px, scales={wavelet_scales}, sigma={args.bg_clip_sigma})...")
            _entropy_bg = getattr(args, 'entropy_bg', False)
            stacked = wavelet_background_extraction(
                stacked, patch_size=dbe_patch, clip_sigma=args.bg_clip_sigma,
                n_scales=wavelet_scales, verbose=args.verbose, star_mask=pp_star_mask,
                use_entropy_weights=_entropy_bg,
                exclusion_mask=_coma_excl_mask)
            safe_print(f"  ✓ Wavelet-band background extraction ({format_time(time.time() - bg_start)})")
        else:
            print(f"\n  Applying background extraction "
                  f"(mesh={args.bg_mesh_size}, sigma={args.bg_clip_sigma})...")
            stacked = apply_background_extraction(
                stacked, mesh_size=args.bg_mesh_size, filter_size=args.bg_filter_size,
                clip_sigma=args.bg_clip_sigma, verbose=args.verbose, star_mask=pp_star_mask,
                exclusion_mask=_coma_excl_mask)
            safe_print(f"  ✓ Background extraction ({format_time(time.time() - bg_start)})")
        stacked = _sanitize(stacked, "background extraction")
        # Save background map as diagnostic
        if _pre_bg is not None:
            try:
                output_path = getattr(args, 'output', None)
                if output_path:
                    bg_map = np.clip(_pre_bg - stacked, 0, None)
                    bg_path = os.path.splitext(output_path)[0] + '_background.jpg'
                    from src.io_fits import save_preview_rgb
                    save_preview_rgb(bg_map, bg_path, stretch='linear')
                    safe_print(f"    Saved background map: {os.path.basename(bg_path)}")
            except Exception as e:
                safe_print(f"    WARNING: could not save background map: {e}")
            del _pre_bg

    # 2. Chroma noise reduction
    if getattr(args, 'chroma_nr', True) and 'chroma_nr' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_chroma_nr')
        cnr_sigma = getattr(args, 'chroma_nr_sigma', 2.0)
        cnr_large = float(getattr(args, 'chroma_nr_large_sigma', 0.0))
        cnr_large_str = float(getattr(args, 'chroma_nr_large_strength', 0.7))
        _large_msg = f", large={cnr_large:.0f}" if cnr_large > 0 else ""
        print(f"\n  Applying chroma noise reduction (sigma={cnr_sigma}{_large_msg})...")
        cnr_start = time.time()
        stacked = reduce_chroma_noise(stacked, sigma=cnr_sigma,
                                      sigma_large=cnr_large,
                                      large_strength=cnr_large_str,
                                      star_mask=pp_star_mask)
        safe_print(f"  ✓ Chroma noise reduction ({format_time(time.time() - cnr_start)})")
        stacked = _sanitize(stacked, "chroma noise reduction")

    # Sky floor correction (per-channel pedestal removal after background extraction)
    if args.background_extraction and 'sky_floor' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_sky_floor')
        try:
            H_s, W_s = stacked.shape[:2]
            lum_s = (0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1]
                     + 0.114 * stacked[:, :, 2])
            sky_mask = np.ones((H_s, W_s), dtype=bool)
            try:
                smooth_sigma = max(20.0, min(H_s, W_s) / 50.0)
                lum_smooth = gaussian_filter_ds(lum_s, sigma=smooth_sigma)
                _bp = _border_pixels(lum_smooth)
                sky_med_lum = float(np.median(_bp))
                sky_std_lum = float(np.std(_bp))
                peak_y, peak_x = np.unravel_index(int(np.argmax(lum_smooth)), (H_s, W_s))
                peak_val = float(lum_smooth[peak_y, peak_x])
                detect_thresh = sky_med_lum + max(
                    2.0 * max(sky_std_lum, 1.0), 0.05 * (peak_val - sky_med_lum))
                if peak_val > detect_thresh and float(np.mean(lum_smooth > detect_thresh)) > 0.001:
                    excl_radius = int(min(H_s, W_s) * 0.30)
                    yy, xx = np.mgrid[:H_s, :W_s]
                    remaining_lum = lum_smooth.copy()
                    primary_peak = peak_val
                    for _src_i in range(3):
                        py, px = np.unravel_index(int(np.argmax(remaining_lum)), (H_s, W_s))
                        pv = float(remaining_lum[py, px])
                        if pv <= detect_thresh:
                            break
                        if _src_i > 0:
                            primary_excess = primary_peak - sky_med_lum
                            if primary_excess > 0 and (pv - sky_med_lum) < 0.5 * primary_excess:
                                break
                        dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
                        sky_mask &= (dist >= excl_radius)
                        remaining_lum[dist < excl_radius] = float(np.min(remaining_lum))
            except Exception:
                pass
            try:
                if sigma_clipped_stats is not None:
                    sample = lum_s[sky_mask].ravel() if sky_mask.any() else lum_s.ravel()
                    _, lum_med, lum_std = sigma_clipped_stats(sample, sigma=3.0, maxiters=5)
                    sky_mask &= (lum_s < float(lum_med) + 3.0 * float(lum_std))
            except Exception:
                pass
            if sky_mask.sum() > 1000:
                for c in range(3):
                    col = stacked[:, :, c][sky_mask].ravel()
                    if sigma_clipped_stats is not None:
                        try:
                            _, sky_floor, _ = sigma_clipped_stats(col, sigma=3.0, maxiters=5)
                            sky_floor = float(sky_floor)
                        except Exception:
                            sky_floor = float(np.median(col))
                    else:
                        sky_floor = float(np.median(col))
                    if sky_floor > 0:
                        stacked[:, :, c] -= sky_floor
                        if args.verbose:
                            safe_print(f"    Sky floor correction ch{c}: -{sky_floor:.2f}")
        except Exception:
            pass

    # 4. Wavelet denoising
    if getattr(args, 'denoise', False) and 'wavelet' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_wavelet_denoise')
        chroma_boost = getattr(args, 'denoise_chroma_boost', 2.0)
        dn_start = time.time()
        if getattr(args, 'denoise_adaptive', False):
            print(f"\n  Applying adaptive wavelet denoising "
                  f"(BayesShrink, chroma_factor={chroma_boost:.1f})...")
            stacked = adaptive_wavelet_denoise(stacked, chroma_factor=chroma_boost,
                                               star_mask=pp_star_mask)
        else:
            strength = getattr(args, 'denoise_strength', 3.0)
            if getattr(args, 'auto_denoise_strength', True):
                fwhm_vals = [f.metrics.get('fwhm', 0.0) for f in final
                             if f.metrics and f.metrics.get('fwhm', 0.0) > 0]
                fwhm_mean = float(np.mean(fwhm_vals)) if fwhm_vals else 0.0
                strength = estimate_denoise_strength(stacked, fwhm_mean=fwhm_mean)
                fwhm_note = f', FWHM={fwhm_mean:.1f}px' if fwhm_mean > 0 else ''
                safe_print(f"\n  Auto-denoise strength: {strength:.2f} (from stacked SNR{fwhm_note})")
            print(f"\n  Applying wavelet denoising "
                  f"(luma={strength:.1f}, chroma={strength * chroma_boost:.1f})...")
            stacked = wavelet_denoise(stacked, threshold_factor=strength,
                                      chroma_factor=chroma_boost, star_mask=pp_star_mask)
        safe_print(f"  ✓ Wavelet denoise ({format_time(time.time() - dn_start)})")
        stacked = _sanitize(stacked, "wavelet denoising")

    # 4.5. Sky residual correction (always after background extraction)
    if args.background_extraction and 'sky_residual' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_sky_residual')
        _sr_mesh = max(32, args.bg_mesh_size // 2)
        _H_pp, _W_pp = stacked.shape[:2]
        _sr_broad_mesh = max(args.bg_mesh_size, min(_H_pp, _W_pp) // 6)
        print(f"\n  Correcting sky residuals (broad={_sr_broad_mesh}px, fine={_sr_mesh}px)...")
        _sr_start = time.time()
        stacked = remove_sky_residual(stacked, mesh_size=_sr_broad_mesh, filter_size=1,
                                      clip_sigma=args.bg_clip_sigma,
                                      star_mask=pp_star_mask, verbose=args.verbose)
        for _sr_pass in range(2):
            stacked = remove_sky_residual(stacked, mesh_size=_sr_mesh, filter_size=1,
                                          clip_sigma=args.bg_clip_sigma, star_mask=pp_star_mask,
                                          verbose=(args.verbose and _sr_pass == 0))
        safe_print(f"  ✓ Sky residual correction ({format_time(time.time() - _sr_start)})")
        stacked = sky_floor_normalize(stacked, star_mask=pp_star_mask, verbose=args.verbose)

    # Sky pedestal — lift the background off zero before the downstream
    # non-negativity clips (Richardson-Lucy, star reduction) crush every
    # below-zero sky pixel to a hard zero, which would leave ~half the sky as
    # black holes (unusable linear FITS, mottled autostretch). A scalar lift
    # preserves the noise distribution; colour neutralisation happens at the end
    # (after photometric calibration, which re-scales the channels).
    if args.background_extraction and 'sky_pedestal' not in skip_steps:
        _ped_lum = (0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1]
                    + 0.114 * stacked[:, :, 2])
        _ped_med = float(np.median(_ped_lum))
        _ped_sigma = float(np.median(np.abs(_ped_lum - _ped_med)) * 1.4826)
        pedestal = max(8.0 * _ped_sigma - _ped_med, 0.0)
        if pedestal > 0:
            stacked = stacked + np.float32(pedestal)
            safe_print(f"  ✓ Sky pedestal: +{pedestal:.2f} "
                       f"(sky sigma={_ped_sigma:.2f})")

    # 5. NLM denoising
    if getattr(args, 'denoise_nlm', False) and 'nlm' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_nlm_denoise')
        nlm_h = getattr(args, 'denoise_nlm_strength', 1.0)
        nlm_blend = getattr(args, 'denoise_nlm_blend', 0.5)
        print(f"\n  Applying NLM denoising (h={nlm_h:.1f}, blend={nlm_blend:.2f})...")
        nlm_start = time.time()
        stacked = nlm_denoise(stacked, h=nlm_h, blend=nlm_blend)
        safe_print(f"  ✓ NLM denoise ({format_time(time.time() - nlm_start)})")
        stacked = _sanitize(stacked, "NLM denoising")

    # 6. Bilateral denoising
    if getattr(args, 'denoise_bilateral', False) and 'bilateral' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_bilateral_denoise')
        bil_sigma_color = getattr(args, 'denoise_bilateral_sigma_color', None)
        bil_sigma_space = getattr(args, 'denoise_bilateral_sigma_space', 3.0)
        sc_str = f"{bil_sigma_color:.2f}" if bil_sigma_color is not None else "auto"
        print(f"\n  Applying bilateral denoising "
              f"(sigma_color={sc_str}, sigma_space={bil_sigma_space:.1f})...")
        bil_start = time.time()
        stacked = bilateral_denoise(stacked, sigma_color=bil_sigma_color,
                                    sigma_space=bil_sigma_space)
        safe_print(f"  ✓ Bilateral denoise ({format_time(time.time() - bil_start)})")
        stacked = _sanitize(stacked, "bilateral denoising")

    # 6.5. MMT denoising
    if getattr(args, 'denoise_mmt', False) and 'mmt' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_mmt_denoise')
        mmt_levels = getattr(args, 'denoise_mmt_levels', 4)
        mmt_strength = getattr(args, 'denoise_mmt_strength', 3.0)
        mmt_chroma = getattr(args, 'denoise_chroma_boost', 2.0)
        print(f"\n  Applying MMT denoising "
              f"(levels={mmt_levels}, strength={mmt_strength:.1f}, chroma={mmt_chroma:.1f})...")
        mmt_start = time.time()
        stacked = mmt_denoise(stacked, levels=mmt_levels, threshold_factor=mmt_strength,
                              chroma_factor=mmt_chroma, star_mask=pp_star_mask)
        safe_print(f"  ✓ MMT denoise ({format_time(time.time() - mmt_start)})")

    # 6.6. ACDNR denoising
    if getattr(args, 'denoise_acdnr', False) and 'acdnr' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_acdnr_denoise')
        acdnr_sigma = getattr(args, 'denoise_acdnr_sigma', 1.5)
        acdnr_k = getattr(args, 'denoise_acdnr_k', 3.0)
        acdnr_chroma = getattr(args, 'denoise_chroma_boost', 2.0)
        print(f"\n  Applying ACDNR denoising "
              f"(sigma={acdnr_sigma:.1f}, k={acdnr_k:.1f}, chroma={acdnr_chroma:.1f})...")
        acdnr_start = time.time()
        stacked = acdnr_denoise(stacked, smoothing_sigma=acdnr_sigma, contrast_k=acdnr_k,
                                chroma_factor=acdnr_chroma, star_mask=pp_star_mask)
        safe_print(f"  ✓ ACDNR denoise ({format_time(time.time() - acdnr_start)})")

    # 6.7. BM3D denoising
    if getattr(args, 'denoise_bm3d', False) and 'bm3d' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_bm3d_denoise')
        bm3d_sigma = getattr(args, 'bm3d_sigma', 0.0)
        bm3d_stride = getattr(args, 'bm3d_stride', None)
        bm3d_sw = getattr(args, 'bm3d_search_window', 16)
        bm3d_gs = getattr(args, 'bm3d_group_size', 8)
        sig_str = f"auto" if bm3d_sigma <= 0 else f"{bm3d_sigma:.1f}"
        print(f"\n  Applying BM3D denoising (sigma={sig_str}, stride={bm3d_stride or 'auto'})...")
        bm3d_start = time.time()
        stacked = bm3d_denoise(stacked, sigma_psd=bm3d_sigma,
                               stride=bm3d_stride, search_window=bm3d_sw,
                               group_size=bm3d_gs, star_mask=pp_star_mask)
        safe_print(f"  ✓ BM3D denoise ({format_time(time.time() - bm3d_start)})")
        stacked = _sanitize(stacked, "BM3D denoising")

    # 6.8. Anisotropic diffusion (Perona-Malik)
    if getattr(args, 'denoise_aniso', False) and 'aniso' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_aniso_denoise')
        aniso_iters = getattr(args, 'aniso_iterations', 20)
        aniso_kappa = getattr(args, 'aniso_kappa', 30.0)
        aniso_gamma = getattr(args, 'aniso_gamma', 0.1)
        aniso_opt = getattr(args, 'aniso_option', 1)
        print(f"\n  Applying anisotropic diffusion "
              f"(iters={aniso_iters}, κ={aniso_kappa:.1f}, γ={aniso_gamma:.2f})...")
        aniso_start = time.time()
        stacked = anisotropic_diffusion(stacked, iterations=aniso_iters,
                                        kappa=aniso_kappa, gamma=aniso_gamma,
                                        option=aniso_opt, star_mask=pp_star_mask)
        safe_print(f"  ✓ Anisotropic diffusion ({format_time(time.time() - aniso_start)})")
        stacked = _sanitize(stacked, "anisotropic diffusion")

    # 6.9. SCNR (Subtractive Chromatic Noise Reduction)
    if getattr(args, 'scnr', False) and 'scnr' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_scnr')
        scnr_amount = getattr(args, 'scnr_amount', 1.0)
        scnr_target = getattr(args, 'scnr_target', 'green')
        print(f"\n  Applying SCNR ({scnr_target} channel, amount={scnr_amount:.2f})...")
        scnr_start = time.time()
        stacked = scnr(stacked, amount=scnr_amount, target=scnr_target)
        safe_print(f"  ✓ SCNR ({format_time(time.time() - scnr_start)})")

    # 6.95. Photometric color calibration
    if getattr(args, 'photometric_calibration', False) and 'photo_cal' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_photo_cal')
        print(f"\n  Applying photometric color calibration (gray-locus method)...")
        pc_start = time.time()
        use_gaia = getattr(args, 'gaia_calibration', False)
        wcs_obj = getattr(args, '_wcs', None)
        if use_gaia and wcs_obj is not None:
            stacked, _pc_scales = try_gaia_calibration(
                stacked, _pp_sources, wcs=wcs_obj, verbose=args.verbose)
        else:
            stacked, _pc_scales = photometric_color_calibrate(
                stacked, _pp_sources, verbose=args.verbose)
        if _pc_scales is not None:
            safe_print(f"  ✓ Photometric calibration "
                       f"(R×{_pc_scales[0]:.3f} G×{_pc_scales[1]:.3f} B×{_pc_scales[2]:.3f}, "
                       f"{format_time(time.time() - pc_start)})")
        else:
            safe_print(f"  ⚠ Photometric calibration skipped (insufficient stars)")
        stacked = _sanitize(stacked, "photometric calibration")

    # 7. Deconvolution (RL, blind PSF, or TV)
    if getattr(args, 'deconvolve', False) and 'deconvolve' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_deconvolve')
        rl_iters = getattr(args, 'deconvolve_iterations', Config.RL_DEFAULT_ITERATIONS)
        rl_fwhm_override = getattr(args, 'deconvolve_fwhm', None)
        rl_model = getattr(args, 'deconvolve_psf_model', 'moffat')
        use_blind_psf = getattr(args, 'deconvolve_blind_psf', False)
        use_tv = getattr(args, 'deconvolve_tv', False)
        tv_lambda = getattr(args, 'tv_lambda', None) or Config.TV_LAMBDA
        tv_iters = getattr(args, 'tv_iterations', None) or Config.TV_ITERATIONS
        psf = None

        if rl_fwhm_override is not None:
            psf = make_synthetic_psf(rl_fwhm_override, model=rl_model)
            safe_print(f"\n  Deconvolution: synthetic PSF FWHM={rl_fwhm_override:.2f}px...")
        elif use_blind_psf and _pp_sources is not None and len(_pp_sources) > 0:
            safe_print(f"\n  Estimating PSF (blind star-stacking, no model fitting)...")
            psf, psf_fwhm = estimate_psf_blind(stacked, _pp_sources)
            if psf is not None:
                safe_print(f"    Blind PSF FWHM ≈ {psf_fwhm:.2f} px")
            else:
                safe_print("    Blind PSF estimation failed — skipping deconvolution")
        elif _pp_sources is not None and len(_pp_sources) > 0:
            safe_print(f"\n  Estimating PSF from star profiles ({rl_model} model)...")
            psf, psf_fwhm = estimate_psf(stacked, _pp_sources, model=rl_model)
            if psf is not None:
                safe_print(f"    PSF FWHM: {psf_fwhm:.2f} px")
            else:
                safe_print("    PSF estimation failed — skipping deconvolution")
        else:
            safe_print("\n  No star detections for PSF — use --deconvolve-fwhm to specify")

        if psf is not None:
            # pp_star_mask (fwhm=4.0) only protects star cores from denoising.
            # Deconvolution ringing/staircasing extends out to ~PSF half-size,
            # so bright compact sources (saturated stars, galaxy nuclei) need a
            # wider protection radius here to avoid visible box/ring artefacts.
            # Richardson-Lucy rings a dark moat around every bright point source.
            # A brightness-normalised Gaussian mask (generate_star_mask) barely
            # protects faint stars — their normalised peak is far below 1 — so
            # they ring. Build a flat-topped protection mask instead: mark every
            # star centroid AND every saturated/flat-topped pixel the finder
            # misses, dilate over the whole ringing zone (~PSF radius) so the
            # protection is 1.0 across each star's core and moat regardless of
            # brightness, then feather the outer edge so the blend has no seam.
            _H_dm, _W_dm = stacked.shape[:2]
            _pts = np.zeros((_H_dm, _W_dm), dtype=bool)
            if _pp_sources is not None and len(_pp_sources) > 0:
                _ys = np.round(np.asarray(_pp_sources['ycentroid'],
                                          dtype=np.float64)).astype(int)
                _xs = np.round(np.asarray(_pp_sources['xcentroid'],
                                          dtype=np.float64)).astype(int)
                _ok = ((_ys >= 0) & (_ys < _H_dm) & (_xs >= 0) & (_xs < _W_dm))
                _pts[_ys[_ok], _xs[_ok]] = True
            _dl = (0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1]
                   + 0.114 * stacked[:, :, 2])
            _hi = float(np.percentile(_dl, 99.7))
            if _hi > 0:
                _pts |= (_dl >= _hi)
            if _pts.any():
                # Ring radius scales with source brightness (brighter stars push
                # flux further out), so protect generously — out to a full PSF
                # box — and feather well beyond. Deconvolution then only sharpens
                # the star-free extended structure (galaxy arms), which is where
                # it helps and where it does not ring.
                _ring_r = max(4, int(float(psf.shape[0])))
                _core = ndimage.binary_dilation(_pts, iterations=_ring_r)
                deconv_mask = ndimage.gaussian_filter(
                    _core.astype(np.float32),
                    sigma=max(3.0, float(psf.shape[0]) * 0.45))
                _dmx = float(deconv_mask.max())
                if _dmx > 0:
                    deconv_mask /= _dmx
                np.clip(deconv_mask, 0.0, 1.0, out=deconv_mask)
            else:
                deconv_mask = pp_star_mask

            deconv_start = time.time()
            if getattr(args, 'deconvolve_svpsf', False) and _pp_sources is not None:
                from src.psf_deconvolution import richardson_lucy_svpsf
                _svn = int(getattr(args, 'deconvolve_sv_tiles', 3))
                safe_print(f"  Spatially-variant Richardson-Lucy "
                           f"({_svn}x{_svn} field tiles, iters={rl_iters})...")
                stacked = richardson_lucy_svpsf(stacked, _pp_sources,
                                                iterations=rl_iters, n_tiles=_svn,
                                                model=rl_model, star_mask=deconv_mask,
                                                verbose=args.verbose)
                safe_print(f"  ✓ SV-PSF deconvolution "
                           f"({format_time(time.time() - deconv_start)})")
            elif use_tv:
                safe_print(f"  Total Variation deconvolution "
                           f"(λ={tv_lambda:.4f}, iters={tv_iters})...")
                stacked = tv_regularized_deconvolve(stacked, psf,
                                                     iterations=tv_iters,
                                                     lambda_tv=tv_lambda,
                                                     star_mask=deconv_mask)
                safe_print(f"  ✓ TV deconvolution ({format_time(time.time() - deconv_start)})")
            else:
                safe_print(f"  Richardson-Lucy deconvolution (iters={rl_iters})...")
                stacked = richardson_lucy_deconvolve(stacked, psf, iterations=rl_iters,
                                                      star_mask=deconv_mask)
                safe_print(f"  ✓ Richardson-Lucy deconvolution "
                           f"({format_time(time.time() - deconv_start)})")
            stacked = _sanitize(stacked, "deconvolution")

    # 8. Star reduction
    if getattr(args, 'star_reduce', False) and 'star_reduce' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_star_reduce')
        sr_factor = float(getattr(args, 'star_reduce_factor', 0.4))
        sr_sigma = float(getattr(args, 'star_reduce_sigma', 1.5))
        print(f"\n  Applying star reduction (factor={sr_factor:.2f}, blur_sigma={sr_sigma:.1f})...")
        sr_start = time.time()
        stacked = reduce_stars(stacked, pp_star_mask, reduction_factor=sr_factor,
                               blur_sigma=sr_sigma)
        safe_print(f"  ✓ Star reduction ({format_time(time.time() - sr_start)})")

    # 9. Multiscale local contrast enhancement
    if getattr(args, 'local_contrast', False) and 'local_contrast' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_local_contrast')
        lc_strength = float(getattr(args, 'local_contrast_strength', 0.7))
        print(f"\n  Applying multiscale local contrast enhancement "
              f"(strength={lc_strength:.2f}, scales=2/12/40 px)...")
        lc_start = time.time()
        stacked = multiscale_local_contrast(stacked, strength=lc_strength,
                                            star_mask=pp_star_mask)
        safe_print(f"  ✓ Local contrast enhancement ({format_time(time.time() - lc_start)})")

    # 10. Star removal via Starnet++
    if getattr(args, 'star_remove', False) and 'star_remove' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_star_remove')
        print(f"\n  Applying star removal (Starnet++)...")
        sr_start = time.time()
        _output_path = getattr(args, 'output', None) or ''
        stacked = apply_starnet(
            stacked,
            output_path=_output_path,
            binary_path=getattr(args, 'starnet_path', None),
            verbose=args.verbose,
        )
        safe_print(f"  ✓ Star removal ({format_time(time.time() - sr_start)})")
        stacked = _sanitize(stacked, "star removal")

    # 11. Comet: radial renormalization (reveals jets by flattening coma gradient)
    if (getattr(args, 'comet_mode', False)
            and getattr(args, 'comet_radial_renorm', False)
            and 'comet_radial_renorm' not in skip_steps):
        print(f"\n  Applying comet radial renormalization...")
        _rr_start = time.time()
        try:
            from src.registration import find_comet_centroid
            _rr_lum = (0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1]
                       + 0.114 * stacked[:, :, 2])
            _rr_cy, _rr_cx = find_comet_centroid(_rr_lum)
            renormed = radial_renormalize(stacked, _rr_cy, _rr_cx)
            # Save as sidecar FITS
            _output_path_rr = getattr(args, 'output', None)
            if _output_path_rr:
                _save_sidecar_fits(renormed, _output_path_rr, '_comet_renorm')
            safe_print(f"  ✓ Comet radial renormalization ({format_time(time.time() - _rr_start)})")
        except Exception as _rre:
            safe_print(f"  WARNING: radial renormalization failed: {_rre}")

    # 12. Comet: Larson-Sekanina rotational difference filter (reveals jet structure)
    if (getattr(args, 'comet_mode', False)
            and getattr(args, 'comet_larson_sekanina', False)
            and 'comet_larson_sekanina' not in skip_steps):
        ls_rot = float(getattr(args, 'comet_ls_rotation', 15.0))
        print(f"\n  Applying Larson-Sekanina filter (rotation={ls_rot:.1f} deg)...")
        _ls_start = time.time()
        try:
            from src.registration import find_comet_centroid
            _ls_lum = (0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1]
                       + 0.114 * stacked[:, :, 2])
            _ls_cy, _ls_cx = find_comet_centroid(_ls_lum)
            ls_img = larson_sekanina(stacked, _ls_cy, _ls_cx, rotation_deg=ls_rot)
            # Save as sidecar FITS
            _output_path_ls = getattr(args, 'output', None)
            if _output_path_ls:
                _save_sidecar_fits(ls_img, _output_path_ls, '_comet_ls')
            safe_print(f"  ✓ Larson-Sekanina filter ({format_time(time.time() - _ls_start)})")
        except Exception as _lse:
            safe_print(f"  WARNING: Larson-Sekanina filter failed: {_lse}")

    # Final sky flattening + neutralisation. Runs last so nothing (photometric
    # calibration, deconvolution) can re-introduce a cast afterwards. Two parts:
    #   1. Per-channel large-scale background flattening over MASKED sky —
    #      removes residual low-frequency luminance gradient AND the colour
    #      blotches (walking chroma noise) that a global offset cannot touch.
    #      Only the smooth sky model is subtracted, so stars/galaxy structure
    #      (high frequency) is untouched; bright objects are masked out of the
    #      model and interpolated across, exactly like background extraction.
    #   2. Equalise the per-channel floors to a common target (neutral grey).
    # Add-only final lift keeps every pixel non-negative (no new clipped holes).
    if args.background_extraction and 'sky_neutralize' not in skip_steps:
        _lum_sn = (0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1]
                   + 0.114 * stacked[:, :, 2])
        # Sky mask: exclude bright objects and star cores from the model.
        _obj_hi = float(np.percentile(_lum_sn, 80.0))
        _skym = (_lum_sn < _obj_hi).astype(np.float32)
        if pp_star_mask is not None:
            _skym *= (1.0 - np.clip(pp_star_mask.astype(np.float32), 0.0, 1.0))
        _sig_sn = max(64.0, float(min(stacked.shape[:2])) / 8.0)
        _den = gaussian_filter_ds(_skym, sigma=_sig_sn)
        _gm = []
        for c in range(3):
            _ch = stacked[:, :, c]
            _num = gaussian_filter_ds(_ch * _skym, sigma=_sig_sn)
            _model = _num / (_den + 1e-6)          # smooth sky level per pixel
            _cmed = float(np.median(_ch))
            stacked[:, :, c] = _ch - (_model - _cmed)   # subtract deviation only
            _gm.append(_cmed)
        # Equalise floors to a common neutral target (add-only).
        _target = max(_gm)
        for c in range(3):
            _add = _target - _gm[c]
            if _add > 0:
                stacked[:, :, c] += np.float32(_add)
        np.clip(stacked, 0.0, None, out=stacked)
        safe_print(f"  ✓ Sky flattened + neutralised to grey (floor={_target:.1f}, "
                   f"was R{_gm[0]:.1f} G{_gm[1]:.1f} B{_gm[2]:.1f})")

    stacked = _sanitize(stacked, "final post-processing")
    return stacked
