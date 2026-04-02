"""Post-processing pipeline: hot pixel removal, background extraction, denoising, deconvolution."""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, List, Optional

import numpy as np
from scipy import ndimage

from src.models import Config, FrameInfo, ProcessingStats
from src.utils import safe_print, format_time
from src.quality import generate_star_mask, _detect_stars_multi_fwhm
from src.background import (apply_background_extraction, remove_sky_residual,
                            sky_floor_normalize, dynamic_background_extraction)
from src.denoising import (wavelet_denoise, adaptive_wavelet_denoise, nlm_denoise,
                           bilateral_denoise, local_normalize, reduce_chroma_noise,
                           estimate_denoise_strength, reduce_stars,
                           multiscale_local_contrast, mmt_denoise, acdnr_denoise)
from src.psf_deconvolution import estimate_psf, make_synthetic_psf, richardson_lucy_deconvolve

DAOStarFinder = None


def _ensure_photutils() -> Optional[Any]:
    global DAOStarFinder
    if DAOStarFinder is None:
        try:
            from photutils.detection import DAOStarFinder as _dao
            if callable(_dao):
                DAOStarFinder = _dao
        except Exception:
            pass
    return DAOStarFinder

try:
    from astropy.stats import sigma_clipped_stats
except Exception:
    sigma_clipped_stats = None


def _diag_save(img: np.ndarray, diag_dir: Optional[str], counter: list, slug: str) -> None:
    """Save a float32 FITS snapshot to diag_dir if diagnostic mode is active.

    counter is a one-element list [int] so the caller can mutate it across calls
    without a nonlocal statement. Only called when the step actually runs.
    """
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
    _ensure_photutils()
    if DAOStarFinder is not None and sigma_clipped_stats is not None:
        try:
            _pp_lum = (0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1]
                       + 0.114 * stacked[:, :, 2])
            _, _bg_med, _bg_std = sigma_clipped_stats(_pp_lum, sigma=3.0, maxiters=5)
            _bg_sub = _pp_lum - float(_bg_med)
            _pp_sources = _detect_stars_multi_fwhm(_bg_sub, 5.0 * float(_bg_std))
            if _pp_sources is not None and len(_pp_sources) > 0:
                pp_star_mask = generate_star_mask(_pp_lum.shape, _pp_sources, fwhm=4.0)
                if args.verbose:
                    safe_print(f"    Post-processing star mask: {len(_pp_sources)} stars")
        except Exception:
            pass

    # 1. Background extraction (DBE or legacy mesh)
    _pre_bg = None
    if getattr(args, 'keep_intermediates', False) and args.background_extraction:
        _pre_bg = stacked.copy()
    if args.background_extraction and 'background' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_background')
        use_dbe = getattr(args, 'dbe', True)
        bg_start = time.time()
        if use_dbe:
            dbe_patch = getattr(args, 'dbe_patch_size', Config.DBE_PATCH_SIZE)
            print(f"\n  Applying Dynamic Background Extraction "
                  f"(patch={dbe_patch}px, RBF thin-plate-spline, sigma={args.bg_clip_sigma})...")
            stacked = dynamic_background_extraction(
                stacked, patch_size=dbe_patch, clip_sigma=args.bg_clip_sigma,
                verbose=args.verbose, star_mask=pp_star_mask)
            safe_print(f"  ✓ Dynamic Background Extraction ({format_time(time.time() - bg_start)})")
        else:
            print(f"\n  Applying background extraction "
                  f"(mesh={args.bg_mesh_size}, sigma={args.bg_clip_sigma})...")
            stacked = apply_background_extraction(
                stacked, mesh_size=args.bg_mesh_size, filter_size=args.bg_filter_size,
                clip_sigma=args.bg_clip_sigma, verbose=args.verbose, star_mask=pp_star_mask)
            safe_print(f"  ✓ Background extraction ({format_time(time.time() - bg_start)})")
        stacked = _sanitize(stacked, "background extraction")
        # Save background map as diagnostic
        if _pre_bg is not None:
            try:
                import os
                output_path = getattr(args, 'output', None)
                if output_path:
                    bg_map = np.clip(_pre_bg - stacked, 0, None)
                    bg_path = os.path.splitext(output_path)[0] + '_background.jpg'
                    from src.io_fits import save_preview_rgb
                    save_preview_rgb(bg_map, bg_path, stretch='linear')
                    safe_print(f"    Saved background map: {os.path.basename(bg_path)}")
            except Exception:
                pass
            del _pre_bg

    # 2. Chroma noise reduction
    if getattr(args, 'chroma_nr', True) and 'chroma_nr' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_chroma_nr')
        cnr_sigma = getattr(args, 'chroma_nr_sigma', 2.0)
        print(f"\n  Applying chroma noise reduction (sigma={cnr_sigma})...")
        cnr_start = time.time()
        stacked = reduce_chroma_noise(stacked, sigma=cnr_sigma)
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
                lum_smooth = ndimage.gaussian_filter(lum_s, sigma=smooth_sigma)
                by = max(10, int(H_s * Config.BORDER_FRAC))
                bx = max(10, int(W_s * Config.BORDER_FRAC))
                border_pix = np.concatenate([
                    lum_smooth[:by, :].ravel(), lum_smooth[-by:, :].ravel(),
                    lum_smooth[by:-by, :bx].ravel(), lum_smooth[by:-by, -bx:].ravel(),
                ])
                sky_med_lum = float(np.median(border_pix))
                sky_std_lum = float(np.std(border_pix))
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

    # 3. Local normalization
    if getattr(args, 'local_normalize', False) and 'local_normalize' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_local_normalize')
        ln_sigma = getattr(args, 'local_normalize_sigma', 50.0)
        print(f"\n  Applying local normalization (sigma={ln_sigma})...")
        ln_start = time.time()
        stacked = local_normalize(stacked, sigma=ln_sigma)
        safe_print(f"  ✓ Local normalization ({format_time(time.time() - ln_start)})")

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

    # 7. Richardson-Lucy deconvolution
    if getattr(args, 'deconvolve', False) and 'deconvolve' not in skip_steps:
        _diag_save(stacked, _diag_dir, _diag_counter, 'before_deconvolve')
        rl_iters = getattr(args, 'deconvolve_iterations', Config.RL_DEFAULT_ITERATIONS)
        rl_fwhm_override = getattr(args, 'deconvolve_fwhm', None)
        rl_model = getattr(args, 'deconvolve_psf_model', 'moffat')
        psf = None
        if rl_fwhm_override is not None:
            psf = make_synthetic_psf(rl_fwhm_override, model=rl_model)
            safe_print(f"\n  Richardson-Lucy deconvolution "
                       f"(FWHM={rl_fwhm_override:.2f}px manual, iters={rl_iters})...")
        elif _pp_sources is not None and len(_pp_sources) > 0:
            safe_print(f"\n  Estimating PSF from star profiles ({rl_model} model)...")
            psf, psf_fwhm = estimate_psf(stacked, _pp_sources, model=rl_model)
            if psf is not None:
                safe_print(f"    PSF FWHM: {psf_fwhm:.2f} px")
                safe_print(f"  Richardson-Lucy deconvolution (iters={rl_iters})...")
            else:
                safe_print("    PSF estimation failed — skipping deconvolution")
        else:
            safe_print("\n  No star detections for PSF estimation — use --deconvolve-fwhm "
                       "to specify manually")
        if psf is not None:
            rl_start = time.time()
            stacked = richardson_lucy_deconvolve(stacked, psf, iterations=rl_iters,
                                                  star_mask=pp_star_mask)
            safe_print(f"  ✓ Richardson-Lucy deconvolution ({format_time(time.time() - rl_start)})")
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

    stacked = _sanitize(stacked, "final post-processing")
    return stacked
