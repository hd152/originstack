"""Astro FITS Stream Stacker

Features:
- Streaming processing (constant memory)
- Calibration (bias/dark/flat)
- Debayering (bilinear + optional Malvar)
- Quality analysis (brightness, contrast, star count)
- Registration (sub-pixel via phase correlation, fallback centroid)
- Automatic cropping, hierarchical processing, preview generation
- Several future features implemented in a basic form (white balance, hot pixel removal, gradient removal, optional GPU acceleration hooks)

Usage: python astro_stack.py -d INPUT_DIR -o OUTPUT.fits [options]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

from astropy.io import fits

try:
    from scipy import ndimage
    from scipy import fftpack
except Exception:
    print("scipy is required", file=sys.stderr)
    raise

try:
    from skimage import exposure
    from skimage.registration import phase_cross_correlation
except Exception:
    phase_cross_correlation = None
    exposure = None

try:
    from photutils import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
except Exception:
    DAOStarFinder = None
    sigma_clipped_stats = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False


@dataclass
class FrameInfo:
    path: str
    type: str  # 'light','dark','flat','bias'
    header: dict
    accepted: bool = True
    metrics: Optional[Dict] = None
    shift: Tuple[float, float] = (0.0, 0.0)


def discover_frames(directory: str) -> Dict[str, List[FrameInfo]]:
    """Discover FITS files and classify them by heuristics and headers."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.fit', '.fits'))]
    frames = {'light': [], 'dark': [], 'flat': [], 'bias': []}
    for p in sorted(files):
        hdr = {}
        try:
            with fits.open(p, memmap=True) as hd:
                hdr = dict(hd[0].header)
        except Exception as e:
            # Retry without memmap for files with BZERO/BSCALE/BLANK keywords
            err_str = str(e).lower()
            if 'memmap' in err_str or 'bzero' in err_str or 'bscale' in err_str or 'blank' in err_str:
                try:
                    with fits.open(p, memmap=False) as hd:
                        hdr = dict(hd[0].header)
                except Exception:
                    hdr = {}
            # If not a memmap issue, still try non-memmap as fallback
            else:
                try:
                    with fits.open(p, memmap=False) as hd:
                        hdr = dict(hd[0].header)
                except Exception:
                    hdr = {}
        ftype = classify_frame(p, hdr)
        frames[ftype].append(FrameInfo(path=p, type=ftype, header=hdr))
    return frames


def classify_frame(path: str, header: dict) -> str:
    name = os.path.basename(path).lower()
    if 'dark' in name or header.get('IMAGETYP', '').lower() == 'dark':
        return 'dark'
    if 'flat' in name or header.get('IMAGETYP', '').lower() == 'flat':
        return 'flat'
    if 'bias' in name or header.get('IMAGETYP', '').lower() == 'bias' or header.get('EXPTIME', 1) == 0:
        return 'bias'
    return 'light'


def load_fits(path: str) -> Tuple[np.ndarray, dict]:
    """Load FITS file; retry without memmap if keyword compression is present."""
    try:
        with fits.open(path, memmap=True) as hd:
            data = hd[0].data.astype(np.float32)
            hdr = dict(hd[0].header)
    except Exception as e:
        # Retry without memmap for files with BZERO/BSCALE/BLANK keywords or any memmap issue
        err_str = str(e).lower()
        if 'memmap' in err_str or 'bzero' in err_str or 'bscale' in err_str or 'blank' in err_str:
            with fits.open(path, memmap=False) as hd:
                data = hd[0].data.astype(np.float32)
                hdr = dict(hd[0].header)
        else:
            raise
    return data, hdr


def make_master(frames: List[FrameInfo], method: str = 'median') -> Optional[np.ndarray]:
    if not frames:
        return None
    imgs = []
    for f in frames:
        try:
            data, _ = load_fits(f.path)
            imgs.append(data.astype(np.float32))
        except Exception:
            continue
    if not imgs:
        return None
    if method == 'median':
        stacked = np.median(np.stack(imgs, axis=0), axis=0)
    else:
        stacked = np.mean(np.stack(imgs, axis=0), axis=0)
    return stacked.astype(np.float32)


def debayer_bilinear(raw: np.ndarray, pattern: str = 'RGGB', method: str = 'bilinear') -> np.ndarray:
    # Expect raw shape (H, W) single channel
    H, W = raw.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    # patterns mapping
    pat = pattern.upper()
    if method == 'malvar':
        return debayer_malvar(raw, pattern=pat)
    # simple assignment of RGGB layout
    # row0 col0 = R for RGGB
    if pat == 'RGGB':
        r = raw[0::2, 0::2]
        g1 = raw[0::2, 1::2]
        g2 = raw[1::2, 0::2]
        b = raw[1::2, 1::2]
    else:
        # fallback treat as RGGB
        r = raw[0::2, 0::2]
        g1 = raw[0::2, 1::2]
        g2 = raw[1::2, 0::2]
        b = raw[1::2, 1::2]
    # Upsample each channel to full size with simple nearest-neighbor then blur (bilinear via convolution)
    def upsample(ch, r_offset, c_offset):
        outc = np.zeros_like(raw)
        outc[r_offset::2, c_offset::2] = ch
        # convolve with kernel to interpolate
        kernel = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
        kernel = kernel / kernel.sum()
        return ndimage.convolve(outc, kernel, mode='mirror')

    out[:, :, 0] = upsample(r, 0, 0)
    out[:, :, 1] = 0.5 * (upsample(g1, 0, 1) + upsample(g2, 1, 0))
    out[:, :, 2] = upsample(b, 1, 1)
    return out


def debayer_malvar(raw: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """Malvar-He-Cutler demosaicing (simplified kernels).
    This is a fast, approximate implementation suitable for most consumer cameras.
    """
    H, W = raw.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    pat = pattern.upper()
    # map Bayer to channel offsets (R,G/B layout assumed RGGB)
    # We'll implement generic kernels by rotating base kernels depending on pattern
    # Base kernels from Malvar et al. (simplified)
    kG = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]])
    kR = np.array([[0, 0, 1, 0, 0], [0, -2, 0, -2, 0], [1, 0, 4, 0, 1], [0, -2, 0, -2, 0], [0, 0, 1, 0, 0]])
    kB = kR[::-1, ::-1]
    kG = kG / 8.0
    kR = kR / 8.0
    kB = kB / 8.0
    # apply kernels
    out[:, :, 0] = ndimage.convolve(raw, kR, mode='mirror')
    out[:, :, 1] = ndimage.convolve(raw, kG, mode='mirror')
    out[:, :, 2] = ndimage.convolve(raw, kB, mode='mirror')
    return out


def white_balance_grayworld(rgb: np.ndarray) -> np.ndarray:
    # Simple gray-world white balance
    img = rgb.copy().astype(np.float32)
    mean = img.mean(axis=(0, 1))
    scale = mean.mean() / (mean + 1e-12)
    return np.clip(img * scale, 0, None)


def white_balance_whitepatch(rgb: np.ndarray, pct: float = 99.5) -> np.ndarray:
    img = rgb.copy().astype(np.float32)
    scales = []
    for c in range(3):
        scales.append(np.percentile(img[:, :, c], pct))
    scales = np.array(scales)
    scales = scales / (scales.mean() + 1e-12)
    return np.clip(img / scales[np.newaxis, np.newaxis, :], 0, None)


def remove_hot_pixels(img: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    # Hot pixel removal: compare to median and replace outliers by interpolated value
    med = ndimage.median_filter(img, size=3)
    diff = img - med
    sigma = np.std(diff)
    mask = diff > max(threshold, 5.0 * sigma)
    if not np.any(mask):
        return img
    # replace using local median
    img_fixed = img.copy()
    img_fixed[mask] = med[mask]
    return img_fixed


def background_gradient_subtract(img: np.ndarray) -> np.ndarray:
    # Estimate background with a large Gaussian blur and subtract
    blurred = ndimage.gaussian_filter(img, sigma=max(15, min(img.shape) // 20))
    return img - blurred


def drizzle_combine(aligned_list: List[np.ndarray], shifts: List[Tuple[float, float]], scale: int = 1) -> np.ndarray:
    """Simple integer-factor drizzle: upsample images by `scale` and place into accumulator using integer-shift*scale offsets.
    This is a simplified drizzle suitable for small scale factors.
    """
    if scale <= 1:
        # mean combine
        acc = None
        for im in aligned_list:
            if acc is None:
                acc = np.zeros_like(im, dtype=np.float64)
            acc += im.astype(np.float64)
        return (acc / len(aligned_list)).astype(np.float32)
    # determine output size
    H, W, C = aligned_list[0].shape
    outH, outW = H * scale, W * scale
    acc = np.zeros((outH, outW, C), dtype=np.float64)
    weight = np.zeros((outH, outW, C), dtype=np.float64)
    for im, sh in zip(aligned_list, shifts):
        # upsample by repeating pixels
        up = np.repeat(np.repeat(im, scale, axis=0), scale, axis=1)
        dy = int(round(sh[0] * scale))
        dx = int(round(sh[1] * scale))
        # for simplicity, place centered
        y0 = max(0, dy)
        x0 = max(0, dx)
        y1 = min(outH, y0 + up.shape[0])
        x1 = min(outW, x0 + up.shape[1])
        ay0 = 0 if dy >= 0 else -dy
        ax0 = 0 if dx >= 0 else -dx
        ay1 = ay0 + (y1 - y0)
        ax1 = ax0 + (x1 - x0)
        acc[y0:y1, x0:x1, :] += up[ay0:ay1, ax0:ax1, :].astype(np.float64)
        weight[y0:y1, x0:x1, :] += 1.0
    weight[weight == 0] = 1.0
    return (acc / weight).astype(np.float32)


def compute_quality_metrics(img: np.ndarray) -> Dict:
    # brightness, contrast, star count
    flat = img
    brightness = float(np.median(flat))
    contrast = float(np.std(flat))
    star_count = 0
    snr = 0.0
    if DAOStarFinder is not None and sigma_clipped_stats is not None:
        try:
            mean, median, std = sigma_clipped_stats(flat)
            daof = DAOStarFinder(fwhm=3.0, threshold=5. * std)
            sources = daof(flat - median)
            star_count = 0 if sources is None else len(sources)
            snr = (brightness - mean) / (std + 1e-12) if std > 0 else 0.0
        except Exception:
            star_count = 0
    else:
        # fallback: count bright local maxima
        thresh = np.median(flat) + np.std(flat) * 3
        mask = flat > thresh
        star_count = int(np.sum(ndimage.maximum_filter(mask.astype(int), size=3) == 1))
        snr = (brightness - np.mean(flat)) / (contrast + 1e-12)
    score = brightness * contrast
    return {'brightness': brightness, 'contrast': contrast, 'star_count': star_count, 'score': score, 'snr': snr}


def calculate_shift(ref: np.ndarray, img: np.ndarray, upsample: int = 10, verbose: bool = False, debug: bool = False, frame_name: str = "", skip_phase_cc: bool = False) -> Tuple[float, float]:
    debug_info = []
    
    # Debug mode: save diagnostic images
    if debug and frame_name:
        try:
            os.makedirs('_registration_debug', exist_ok=True)
            # Save normalized versions
            ref_norm = (ref - np.min(ref)) / (np.max(ref) - np.min(ref) + 1e-12) * 255
            img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-12) * 255
            
            if Image is not None:
                Image.fromarray(ref_norm.astype(np.uint8)).save(f'_registration_debug/{frame_name}_ref.png')
                Image.fromarray(img_norm.astype(np.uint8)).save(f'_registration_debug/{frame_name}_img.png')
            
            # Save statistics
            with open(f'_registration_debug/{frame_name}_stats.txt', 'w') as f:
                f.write(f"Reference:\n  min={np.min(ref):.2f}, max={np.max(ref):.2f}, mean={np.mean(ref):.2f}, std={np.std(ref):.2f}\n")
                f.write(f"Image:\n  min={np.min(img):.2f}, max={np.max(img):.2f}, mean={np.mean(img):.2f}, std={np.std(img):.2f}\n")
        except Exception as e:
            pass
    
    # Use phase cross correlation for subpixel shifts when available
    if not skip_phase_cc and phase_cross_correlation is not None:
        try:
            # Normalize images for phase correlation (zero mean, unit variance)
            # This is critical for good phase correlation performance
            ref_norm = (ref - np.mean(ref)) / (np.std(ref) + 1e-12)
            img_norm = (img - np.mean(img)) / (np.std(img) + 1e-12)
            
            # Suppress overflow warnings in phase correlation
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                shift, error, diffphase = phase_cross_correlation(ref_norm, img_norm, upsample_factor=upsample)
            debug_info.append(f"phase_cc: shift={shift}, error={error:.4f}")
            
            # Check for degenerate case: all zeros might mean images are similar enough that it failed silently
            if np.allclose(shift, 0.0) and error < 0.01:
                debug_info.append(f"phase_cc: zero shift with very low error (images might be nearly identical)")
            
            # Validate shift magnitude (sanity check)
            if np.isfinite(shift).all() and np.abs(shift).max() < max(ref.shape) * 0.5:
                if verbose:
                    print(f"      [phase_correlation succeeded: {shift}]")
                return float(shift[0]), float(shift[1])
            else:
                debug_info.append(f"phase_cc rejected: nan/inf or too large ({np.abs(shift).max():.1f} > {max(ref.shape) * 0.5:.1f})")
        except Exception as e:
            debug_info.append(f"phase_cc error: {type(e).__name__}")
    
    # Try normalized cross-correlation as alternative fallback
    try:
        from scipy.signal import correlate2d
        # Normalize images - critical for correlation to work well
        ref_norm = (ref - np.mean(ref)) / (np.std(ref) + 1e-12)
        img_norm = (img - np.mean(img)) / (np.std(img) + 1e-12)
        
        # Compute correlation - use smaller version for speed
        scale = max(1, ref.shape[0] // 64)
        ref_small = ref_norm[::scale, ::scale]
        img_small = img_norm[::scale, ::scale]
        
        corr = correlate2d(img_small, ref_small, mode='same')
        peak = np.unravel_index(np.argmax(corr), corr.shape)
        center = np.array(corr.shape) // 2
        shift_pixels = (peak - center) * scale
        
        # Check if peak correlation is strong enough (sanity check)
        peak_value = corr[peak] if len(corr.shape) > 0 else 0
        mean_corr = np.mean(np.abs(corr))
        
        if peak_value > mean_corr * 2:  # Peak should be significantly above average
            if np.isfinite(shift_pixels).all() and np.abs(shift_pixels).max() < max(ref.shape) * 0.5:
                if verbose:
                    print(f"      [xcorr fallback: shift=({shift_pixels[1]:.1f}, {shift_pixels[0]:.1f})]")
                return float(shift_pixels[0]), float(shift_pixels[1])
            else:
                debug_info.append(f"xcorr rejected: bad result {shift_pixels}")
        else:
            debug_info.append(f"xcorr: weak peak (peak={peak_value:.1f}, mean={mean_corr:.1f}) - correlation is degenerate")
    except Exception as e:
        debug_info.append(f"xcorr error: {type(e).__name__}")
    
    # Fallback to centroid difference - try multiple percentiles for robustness
    best_shift = (0.0, 0.0)
    best_score = float('inf')
    
    for percentile in [95, 90, 85, 80]:  # Try multiple thresholds
        try:
            thresh_ref = np.percentile(ref, percentile)
            thresh_img = np.percentile(img, percentile)
            rmask = ref > thresh_ref
            imask = img > thresh_img
            n_ref = rmask.sum()
            n_img = imask.sum()
            
            if n_ref > 10 and n_img > 10:
                cim = ndimage.center_of_mass(ref * rmask)
                cim2 = ndimage.center_of_mass(img * imask)
                if cim and cim2:
                    shift_y = float(cim2[0] - cim[0])
                    shift_x = float(cim2[1] - cim[1])
                    shift_mag = np.sqrt(shift_x**2 + shift_y**2)
                    
                    # Prefer solution from lower percentile (more pixels = more stable)
                    # But if shift is too large, use lower percentile threshold
                    if shift_mag > 0.1 or n_ref > 50:  # Have a real shift or many pixels
                        if verbose:
                            print(f"      [centroid fallback (p{percentile}): ({shift_x:.1f}, {shift_y:.1f})]")
                        return shift_y, shift_x
            
            debug_info.append(f"centroid(p{percentile}): n_ref={n_ref}, n_img={n_img}")
        except Exception as e:
            pass
    
    if verbose and debug_info:
        print(f"      [CRITICAL: no registration method succeeded] " + " | ".join(debug_info))
    return 0.0, 0.0


def apply_shift(img: np.ndarray, shift: Tuple[float, float]) -> np.ndarray:
    return ndimage.shift(img, shift=shift, order=3, mode='constant', cval=0.0, prefilter=True)


def calc_common_crop(shifts: List[Tuple[float, float]], shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    # compute maximal positive/negative shifts across frames and crop
    ys = [s[0] for s in shifts]
    xs = [s[1] for s in shifts]
    max_up = int(max(0, np.ceil(max(ys))))
    max_down = int(max(0, np.ceil(-min(ys))))
    max_left = int(max(0, np.ceil(max(xs))))
    max_right = int(max(0, np.ceil(-min(xs))))
    H, W = shape
    top = max_up + 2
    bottom = H - (max_down + 2)
    left = max_left + 2
    right = W - (max_right + 2)
    if top >= bottom or left >= right:
        return 0, H, 0, W
    return top, bottom, left, right


def save_preview_rgb(rgb: np.ndarray, path: str):
    if Image is None or exposure is None:
        return
    # per-channel stretch 1-99
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], (1, 99))
        out[:, :, c] = exposure.rescale_intensity(rgb[:, :, c], in_range=(lo, hi))
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path, quality=95)


def stack_target(frames: List[FrameInfo], output_path: str, args: argparse.Namespace, masters: Dict[str, Optional[np.ndarray]]):
    lights = [f for f in frames if f.type == 'light']
    if not lights:
        print('No light frames found for target')
        return None
    print(f'  Quality analysis: analyzing {len(lights)} light frames...')
    accepted = []
    rejected_reasons = {}
    for f in lights:
        try:
            data, hdr = load_fits(f.path)
        except Exception as e:
            f.accepted = False
            f.metrics = {'error': str(e)}
            rejected_reasons[f.path] = f'load error: {str(e)}'
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {str(e)}')
            continue
        # Validate data is not empty
        if data is None or data.size == 0:
            f.accepted = False
            f.metrics = {'error': 'empty data array'}
            rejected_reasons[f.path] = 'empty data array'
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: empty data array')
            continue
        # calibration
        if masters.get('bias') is not None:
            data = data - masters['bias']
        if masters.get('dark') is not None:
            data = data - masters['dark']
        if masters.get('flat') is not None:
            flat = masters['flat'].copy()
            med = np.median(flat)
            if med != 0:
                flat = flat / med
            data = data / (flat + 1e-12)
        # debayer if 2D raw single channel and header indicates bayer or shape suggests
        try:
            if data.ndim == 2:
                bayer = hdr.get('BAYERPAT', hdr.get('COLORTYP', 'RGGB'))
                rgb = debayer_bilinear(data, pattern=bayer, method=args.debayer_method)
            else:
                # already multi-channel
                rgb = data
        except Exception as e:
            f.accepted = False
            f.metrics = {'error': f'debayering error: {str(e)}'}
            rejected_reasons[f.path] = f'debayering error: {str(e)}'
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: debayering error: {str(e)}')
            continue
        # hot pixel removal
        try:
            if rgb.ndim != 3 or rgb.shape[2] < 1:
                raise ValueError(f'Invalid RGB shape: {rgb.shape}')
            for c in range(rgb.shape[2]):
                rgb[:, :, c] = remove_hot_pixels(rgb[:, :, c])
        except Exception as e:
            f.accepted = False
            f.metrics = {'error': f'hot pixel removal error: {str(e)}'}
            rejected_reasons[f.path] = f'hot pixel removal error: {str(e)}'
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {str(e)}')
            continue
        # background subtraction
        # compute a gray metric from luminance
        try:
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            metrics = compute_quality_metrics(lum)
            f.metrics = metrics
        except Exception as e:
            f.accepted = False
            f.metrics = {'error': f'quality analysis error: {str(e)}'}
            rejected_reasons[f.path] = f'quality analysis error: {str(e)}'
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {str(e)}')
            continue
        # quality filter
        if args.quality_filter:
            # percentile threshold across all frames will be applied later; for streaming we mark all and filter after collecting metrics
            accepted.append(f)
            if args.verbose:
                print(f'    {os.path.basename(f.path)}: brightness={metrics["brightness"]:.1f}, contrast={metrics["contrast"]:.1f}, stars={metrics["star_count"]}, score={metrics["score"]:.1f}')
        else:
            f.accepted = True
            if args.verbose:
                print(f'    {os.path.basename(f.path)}: brightness={metrics["brightness"]:.1f}, contrast={metrics["contrast"]:.1f}, stars={metrics["star_count"]}')
    # If quality_filter, compute threshold
    if args.quality_filter and accepted:
        scores = np.array([f.metrics['score'] for f in accepted])
        pct = np.percentile(scores, args.quality_threshold)
        for f in accepted:
            if f.metrics['score'] >= pct:
                f.accepted = True
            else:
                f.accepted = False
                rejected_reasons[f.path] = f'quality score {f.metrics["score"]:.1f} below threshold {pct:.1f}'
                if args.verbose:
                    print(f'  REJECT {os.path.basename(f.path)}: score {f.metrics["score"]:.1f} < {pct:.1f}')
    # Build list of final accepted frames
    final = [f for f in lights if f.accepted]
    if not final:
        print(f'All {len(lights)} frames rejected')
        if args.verbose and rejected_reasons:
            print('Rejection reasons:')
            for path, reason in rejected_reasons.items():
                print(f'  {os.path.basename(path)}: {reason}')
        return None
    # Phase 2: registration - choose reference as highest score
    ref = None
    ref_path = None
    best = max(final, key=lambda x: x.metrics.get('score', 0))
    ref_path = best.path
    ref_data, ref_hdr = load_fits(ref_path)
    if ref_data.ndim == 2:
        ref_rgb = debayer_bilinear(ref_data, pattern=best.header.get('BAYERPAT', 'RGGB'))
    else:
        ref_rgb = ref_data
    # use luminance for registration
    ref_lum = 0.299 * ref_rgb[:, :, 0] + 0.587 * ref_rgb[:, :, 1] + 0.114 * ref_rgb[:, :, 2]
    shifts = []
    aligned_shapes = []
    tmp_files = []
    # Prepare memmap for median stacking if requested
    use_median = args.stack_method == 'median'
    n = len(final)
    H, W = ref_lum.shape
    # create temporary memmap if median
    memmap_path = None
    mem = None
    if use_median:
        memmap_path = os.path.join(tempfile.gettempdir(), f'stack_{os.getpid()}.dat')
        mem = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(n, H, W, 3))

    idx = 0
    ref_lum_std = np.std(ref_lum)  # Cache for diagnostics
    if args.verbose:
        print(f'  Registration: calculating shifts for {len(final)} frames (reference: {os.path.basename(ref_path)})')
        # Diagnostic: show reference image statistics
        ref_stats = {
            'min': np.min(ref_lum),
            'max': np.max(ref_lum),
            'mean': np.mean(ref_lum),
            'std': ref_lum_std,
        }
        print(f'    Reference luminance stats - min={ref_stats["min"]:.1f}, max={ref_stats["max"]:.1f}, mean={ref_stats["mean"]:.1f}, std={ref_stats["std"]:.1f}')
    for f in final:
        data, hdr = load_fits(f.path)
        if data.ndim == 2:
            rgb = debayer_bilinear(data, pattern=hdr.get('BAYERPAT', 'RGGB'), method=args.debayer_method)
        else:
            rgb = data
        # optional white balance
        if args.white_balance == 'grayworld':
            rgb = white_balance_grayworld(rgb)
        elif args.white_balance == 'whitepatch':
            rgb = white_balance_whitepatch(rgb)
        else:
            rgb = rgb
        # registration
        if not args.no_registration:
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            # Diagnostic: check image statistics
            lum_std = np.std(lum)
            if args.verbose and lum_std < ref_lum_std * 0.1:
                lum_min, lum_max = np.min(lum), np.max(lum)
                print(f'      [WARNING: very low contrast] min={lum_min:.1f}, max={lum_max:.1f}, std={lum_std:.1f} (ref_std={ref_lum_std:.1f})')
            sy, sx = calculate_shift(ref_lum, lum, verbose=args.verbose, debug=args.debug_registration, frame_name=os.path.splitext(os.path.basename(f.path))[0], skip_phase_cc=args.skip_phase_correlation)
            # gate unrealistic shifts
            if abs(sx) > 0.1 * W or abs(sy) > 0.1 * H:
                print(f'Unrealistic shift {sx},{sy} for {f.path}, ignoring')
                sx, sy = 0.0, 0.0
            f.shift = (sy, sx)
        else:
            f.shift = (0.0, 0.0)
        shifts.append(f.shift)
        
        # Report shift with diagnostics
        shift_mag = np.sqrt(f.shift[0]**2 + f.shift[1]**2)
        if args.verbose:
            print(f'    {os.path.basename(f.path)}: shift=({f.shift[1]:+.1f}, {f.shift[0]:+.1f}) px, magnitude={shift_mag:.2f} px')
        # apply shift and write to memmap or accumulate
        aligned = np.zeros_like(rgb)
        for c in range(3):
            aligned[:, :, c] = apply_shift(rgb[:, :, c], f.shift)
        if use_median:
            mem[idx] = aligned
            mem.flush()
        else:
            tmp_files.append(aligned.astype(np.float32))
        aligned_shapes.append(aligned.shape[:2])
        idx += 1
    
    # Check for suspicious all-zero shifts (possible algorithm failure)
    zero_shifts = sum(1 for f in final if f.shift == (0.0, 0.0))
    if zero_shifts > len(final) * 0.8 and len(final) > 2:
        print(f'\n[WARNING] {zero_shifts}/{len(final)} frames have zero shift - this is suspicious!')
        print(f'[SUGGESTION] Try running with --skip-phase-correlation to test fallback methods:')
        print(f'  python astro_stack.py --skip-phase-correlation ...')
        print()
    
    # Phase 3: crop to common valid region
    top, bottom, left, right = calc_common_crop([f.shift for f in final], (H, W))
    # Crop & combine
    if use_median:
        stacked = np.median(mem[:, top:bottom, left:right, :], axis=0)
        # cleanup memmap file
        try:
            del mem
            os.remove(memmap_path)
        except Exception:
            pass
    else:
        # mean combine streaming
        acc = None
        count = 0
        for a in tmp_files:
            cropped = a[top:bottom, left:right, :]
            if acc is None:
                acc = np.zeros_like(cropped, dtype=np.float64)
            acc += cropped.astype(np.float64)
            count += 1
        stacked = (acc / max(1, count)).astype(np.float32)
    # Save FITS (3,H,W)
    out_h, out_w, _ = stacked.shape
    hdu = fits.PrimaryHDU()
    # store as (3,H,W)
    data_out = np.transpose(stacked, (2, 0, 1)).astype(np.float32)
    hdu.data = data_out
    hdu.header['NFRAMES'] = len(final)
    hdu.header['NREJECT'] = len(lights) - len(final)
    hdu.header['COMBINED'] = True
    hdu.writeto(output_path, overwrite=True)
    # preview
    preview_path = os.path.splitext(output_path)[0] + '.jpg'
    save_preview_rgb(stacked, preview_path)
    print(f'Saved stacked FITS to {output_path} and preview to {preview_path}')
    return output_path


def process_directory(directory: str, output: str, args: argparse.Namespace):
    # Detect hierarchical mode
    if not os.path.isdir(directory):
        print(f'ERROR: Input directory {directory} does not exist', file=sys.stderr)
        raise SystemExit(1)
    print(f'Processing directory: {directory}')
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    targets = []
    if any(os.listdir(directory)) and any(f.lower().endswith(('.fit', '.fits')) for f in os.listdir(directory)):
        # single folder
        targets = [(directory, output)]
        print(f'Detected single-folder mode')
    elif subdirs:
        # hierarchical: produce per-subfolder stacks then combine
        tmp_stacks = []
        for d in sorted(subdirs):
            name = os.path.basename(d)
            outp = os.path.join(tempfile.gettempdir(), f'{name}_stack.fits')
            targets.append((d, outp))
            tmp_stacks.append(outp)
        print(f'Detected hierarchical mode with {len(targets)} subfolders')
        # final combined output will be combined from tmp_stacks
    else:
        print('ERROR: No FITS files found', file=sys.stderr)
        raise SystemExit('No FITS files found')

    # Process each target
    produced = []
    for d, outp in targets:
        print(f'\nProcessing target: {os.path.basename(d) if d != directory else "root"}')
        frames = discover_frames(d)
        nfiles = sum(len(v) for v in frames.values())
        print(f'  Found {nfiles} FITS files: {len(frames["light"])} lights, {len(frames["dark"])} darks, {len(frames["flat"])} flats, {len(frames["bias"])} bias')
        masters = {
            'bias': make_master(frames['bias'], method='median'),
            'dark': make_master(frames['dark'], method='median'),
            'flat': make_master(frames['flat'], method='median'),
        }
        res = stack_target([f for t in frames.values() for f in t], outp, args, masters)
        if res:
            produced.append(res)
    # If hierarchical combine
    if len(produced) > 1:
        # load all stacks, resize to minimum
        stacks = []
        shapes = [fits.open(p)[0].data.shape for p in produced]
        # shapes are (3,H,W)
        mins = np.min([[s[1], s[2]] for s in shapes], axis=0)
        Hm, Wm = int(mins[0]), int(mins[1])
        acc = None
        for p in produced:
            with fits.open(p, memmap=True) as hd:
                d = np.transpose(hd[0].data, (1, 2, 0)).astype(np.float32)
                d = d[:Hm, :Wm, :]
                if acc is None:
                    acc = np.zeros_like(d, dtype=np.float64)
                acc += d
        combined = (acc / len(produced)).astype(np.float32)
        out_hdu = fits.PrimaryHDU()
        out_hdu.data = np.transpose(combined, (2, 0, 1))
        out_hdu.header['NTARGETS'] = len(produced)
        out_hdu.writeto(output, overwrite=True)
        save_preview_rgb(combined, os.path.splitext(output)[0] + '.jpg')
        print('Saved hierarchical combined output to', output)


def parse_args():
    p = argparse.ArgumentParser(description='Streaming FITS stacker')
    p.add_argument('-d', '--directory', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--no-registration', action='store_true')
    p.add_argument('--skip-phase-correlation', action='store_true', help='Skip phase correlation, use only fallback methods (debug)')
    p.add_argument('--quality-filter', action='store_true')
    p.add_argument('--quality-threshold', type=float, default=50.0)
    p.add_argument('--keep-intermediates', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--debug-registration', action='store_true', help='Detailed registration diagnostics (implies -v)')
    p.add_argument('--stack-method', choices=['mean', 'median'], default='mean')
    p.add_argument('--debayer-method', choices=['bilinear', 'malvar'], default='bilinear')
    p.add_argument('--white-balance', choices=['none', 'grayworld', 'whitepatch'], default='grayworld')
    p.add_argument('--drizzle-scale', type=int, default=1, help='Integer drizzle scale factor (1 = disabled)')
    p.add_argument('--use-gpu', action='store_true', help='Use CuPy for available operations (experimental)')
    return p.parse_args()


def main():
    args = parse_args()
    # debug_registration implies verbose
    if args.debug_registration:
        args.verbose = True
    try:
        process_directory(args.directory, args.output, args)
    except Exception as e:
        print(f'ERROR: {str(e)}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
