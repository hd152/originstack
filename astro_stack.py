"""Astro FITS Stream Stacker

Features:
- Streaming processing (constant memory)
- Calibration (bias/dark/flat)
- Debayering (bilinear + optional Malvar)
- Quality analysis (brightness, contrast, star count)
- Registration (sub-pixel via phase correlation, fallback centroid)
- Automatic cropping, hierarchical processing, preview generation
- Intelligent background extraction (mesh-based sigma-clipped sky removal)
- GPU acceleration via CuPy (--use-gpu) with automatic CPU fallback
- Several future features implemented in a basic form (white balance, hot pixel removal, gradient removal)

Usage: python astro_stack.py -d INPUT_DIR -o OUTPUT.fits [options]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import logging
from dataclasses import dataclass, field
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

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    # Fallback: create a pass-through wrapper
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    from astroquery.astrometry_net import AstrometryNet
    HAS_ASTROMETRY_NET = True
except Exception:
    HAS_ASTROMETRY_NET = False


# ---------------------------------------------------------------------------
# GPU / CPU abstraction layer
# ---------------------------------------------------------------------------
class GpuContext:
    """Array-agnostic computation context.

    Provides ``xp`` (numpy or cupy), ``xndimage`` and ``xsignal`` so that
    every compute function can be written once and dispatched to GPU or CPU.
    """

    def __init__(self, use_gpu: bool = False):
        self.active = False
        self.xp = np
        self.xndimage = ndimage
        self.xsignal = None
        self.device_name = "CPU"
        self.vram_total_mb = 0.0
        self.vram_free_mb = 0.0

        if use_gpu and HAS_CUPY:
            try:
                cp.cuda.Device(0).compute_capability
                self.xp = cp
                import cupyx.scipy.ndimage as _cp_ndimage
                self.xndimage = _cp_ndimage
                try:
                    import cupyx.scipy.signal as _cp_signal
                    self.xsignal = _cp_signal
                except ImportError:
                    from scipy import signal as _signal
                    self.xsignal = _signal
                self.active = True
                dev = cp.cuda.Device(0)
                self.device_name = str(dev)
                mem = dev.mem_info
                self.vram_free_mb = mem[0] / 1024 ** 2
                self.vram_total_mb = mem[1] / 1024 ** 2
            except Exception as exc:
                logging.warning("GPU init failed (%s), falling back to CPU.", exc)
                self.xp = np
                self.xndimage = ndimage
                self.active = False

        if self.xsignal is None:
            from scipy import signal as _signal
            self.xsignal = _signal

    # --- transfer helpers ---------------------------------------------------
    def to_device(self, arr: np.ndarray):
        """Move *arr* to GPU.  No-op when running on CPU."""
        if self.active:
            return cp.asarray(arr)
        return arr

    def to_host(self, arr) -> np.ndarray:
        """Move *arr* to CPU numpy.  No-op when already numpy."""
        if self.active and hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)

    def free_pool(self):
        """Release CuPy's cached GPU memory."""
        if self.active:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def available_vram_mb(self) -> float:
        if self.active:
            return cp.cuda.Device(0).mem_info[0] / 1024 ** 2
        return 0.0

    def print_status(self):
        if self.active:
            safe_print(f"  GPU: {self.device_name}")
            safe_print(f"  VRAM: {self.vram_free_mb:.0f}/{self.vram_total_mb:.0f} MB free")
        else:
            safe_print(f"  Compute: CPU")


_gpu: Optional[GpuContext] = None


def get_gpu() -> GpuContext:
    global _gpu
    if _gpu is None:
        _gpu = GpuContext(use_gpu=False)
    return _gpu


# Configuration constants
class Config:
    """Central configuration for magic numbers and thresholds."""
    HOT_PIXEL_THRESHOLD = 10.0
    WHITE_PATCH_PERCENTILE = 99.5
    MAX_SHIFT_FRACTION = 0.1
    STAR_DETECTION_SIGMA = 5.0
    CROP_MARGIN = 2
    XCORR_DOWNSCALE_TARGET = 256  # Target size for cross-correlation
    CENTROID_PERCENTILES = [95, 90, 85, 80]
    QUALITY_LOW_BRIGHTNESS = 10
    QUALITY_LOW_CONTRAST = 1
    LARGE_SHIFT_WARNING_PX = 20
    MIN_RECOMMENDED_FRAMES = 10
    PREVIEW_JPEG_QUALITY = 95
    PREVIEW_STRETCH_PERCENTILES = (1, 99)


@dataclass
class FrameInfo:
    path: str
    type: str  # 'light','dark','flat','bias'
    header: dict
    accepted: bool = True
    metrics: Optional[Dict] = None
    shift: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ProcessingStats:
    """Track timing and statistics during processing."""
    start_time: float = field(default_factory=time.time)
    discovery_time: float = 0.0
    calibration_time: float = 0.0
    quality_time: float = 0.0
    registration_time: float = 0.0
    stacking_time: float = 0.0
    total_frames: int = 0
    accepted_frames: int = 0
    rejected_frames: int = 0
    errors: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_shape: Optional[Tuple[int, int]] = None
    cropped_pixels: Optional[Tuple[int, int]] = None
    peak_memory_mb: float = 0.0

    def total_time(self) -> float:
        return time.time() - self.start_time

    def add_error(self, path: str, error: str):
        self.errors.append((path, error))

    def add_warning(self, warning: str):
        self.warnings.append(warning)


def safe_print(text: str):
    """Print text with fallback for unicode characters on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: replace unicode symbols with ASCII
        text = text.replace('✓', '[OK]')
        text = text.replace('✗', '[X]')
        text = text.replace('⚠', '[!]')
        text = text.replace('ℹ', '[i]')
        text = text.replace('─', '-')
        text = text.replace('→', '->')
        text = text.replace('×', 'x')
        print(text)


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    safe_print(f"\n{char * 70}")
    safe_print(text)
    safe_print(f"{char * 70}")


def print_quality_table(frames: List[FrameInfo], show_all: bool = False):
    """Print a formatted table of frame quality metrics."""
    if not frames:
        return

    # Filter to only frames with metrics
    frames_with_metrics = [f for f in frames if f.metrics and 'score' in f.metrics]
    if not frames_with_metrics:
        return

    # Header
    safe_print("\n  Frame Quality Details:")
    safe_print("  " + "─" * 100)
    safe_print(f"  {'Frame':<30} {'Bright':>8} {'Contr':>8} {'Stars':>6} {'Score':>10} {'Status':>8}")
    safe_print("  " + "─" * 100)

    # Show first 10, last 10, or all if show_all
    if show_all or len(frames_with_metrics) <= 20:
        frames_to_show = frames_with_metrics
    else:
        frames_to_show = frames_with_metrics[:10] + frames_with_metrics[-10:]
        show_ellipsis = True

    shown_count = 0
    for i, f in enumerate(frames_with_metrics):
        if not show_all and len(frames_with_metrics) > 20 and i == 10:
            print(f"  {'...':<30} {'...':>8} {'...':>8} {'...':>6} {'...':>10} {'...':>8}")
            continue
        elif not show_all and len(frames_with_metrics) > 20 and 10 < i < len(frames_with_metrics) - 10:
            continue

        name = os.path.basename(f.path)
        if len(name) > 30:
            name = name[:27] + "..."

        brightness = f.metrics.get('brightness', 0)
        contrast = f.metrics.get('contrast', 0)
        stars = f.metrics.get('star_count', 0)
        score = f.metrics.get('score', 0)
        status = "✓" if f.accepted else "✗"

        safe_print(f"  {name:<30} {brightness:8.1f} {contrast:8.1f} {stars:6} {score:10.1f} {status:>8}")

    safe_print("  " + "─" * 100)


def print_phase(phase_num: int, title: str):
    """Print a phase header."""
    print(f"\n{'=' * 70}")
    print(f"PHASE {phase_num}: {title.upper()}")
    print(f"{'=' * 70}")


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024**2
    return 0.0


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


def debayer_bilinear(raw, pattern: str = 'RGGB', method: str = 'bilinear'):
    gpu = get_gpu()
    xp = gpu.xp
    raw = gpu.to_device(raw)
    H, W = raw.shape
    pat = pattern.upper()
    if method == 'malvar':
        return debayer_malvar(raw, pattern=pat)
    r = raw[0::2, 0::2]
    g1 = raw[0::2, 1::2]
    g2 = raw[1::2, 0::2]
    b = raw[1::2, 1::2]

    def upsample(ch, r_offset, c_offset):
        outc = xp.zeros_like(raw)
        outc[r_offset::2, c_offset::2] = ch
        kernel = xp.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]],
                          dtype=xp.float32)
        kernel = kernel / kernel.sum()
        return gpu.xndimage.convolve(outc, kernel, mode='mirror')

    out = xp.zeros((H, W, 3), dtype=xp.float32)
    out[:, :, 0] = upsample(r, 0, 0)
    out[:, :, 1] = 0.5 * (upsample(g1, 0, 1) + upsample(g2, 1, 0))
    out[:, :, 2] = upsample(b, 1, 1)
    return out


def debayer_malvar(raw, pattern: str = 'RGGB'):
    """Malvar-He-Cutler demosaicing (simplified kernels)."""
    gpu = get_gpu()
    xp = gpu.xp
    raw = gpu.to_device(raw)
    H, W = raw.shape
    kG = xp.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
                   [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], dtype=xp.float32) / 8.0
    kR = xp.array([[0, 0, 1, 0, 0], [0, -2, 0, -2, 0], [1, 0, 4, 0, 1],
                   [0, -2, 0, -2, 0], [0, 0, 1, 0, 0]], dtype=xp.float32) / 8.0
    kB = kR[::-1, ::-1]
    out = xp.zeros((H, W, 3), dtype=xp.float32)
    out[:, :, 0] = gpu.xndimage.convolve(raw, kR, mode='mirror')
    out[:, :, 1] = gpu.xndimage.convolve(raw, kG, mode='mirror')
    out[:, :, 2] = gpu.xndimage.convolve(raw, kB, mode='mirror')
    return out


def white_balance_grayworld(rgb):
    gpu = get_gpu()
    xp = gpu.xp
    img = xp.array(rgb, dtype=xp.float32, copy=True)
    mean = img.mean(axis=(0, 1))
    scale = mean.mean() / (mean + 1e-12)
    return xp.clip(img * scale, 0, None)


def white_balance_whitepatch(rgb, pct: float = None):
    gpu = get_gpu()
    xp = gpu.xp
    if pct is None:
        pct = Config.WHITE_PATCH_PERCENTILE
    img = xp.array(rgb, dtype=xp.float32, copy=True)
    scales = xp.array([float(xp.percentile(img[:, :, c], pct)) for c in range(3)])
    scales = scales / (scales.mean() + 1e-12)
    return xp.clip(img / scales[None, None, :], 0, None)


def remove_hot_pixels(img, threshold: float = None):
    gpu = get_gpu()
    xp = gpu.xp
    if threshold is None:
        threshold = Config.HOT_PIXEL_THRESHOLD
    med = gpu.xndimage.median_filter(img, size=3)
    diff = img - med
    sigma = float(xp.std(diff))
    mask = diff > max(threshold, 5.0 * sigma)
    if not bool(xp.any(mask)):
        return img
    img_fixed = xp.array(img, copy=True)
    img_fixed[mask] = med[mask]
    return img_fixed


def background_gradient_subtract(img):
    gpu = get_gpu()
    blurred = gpu.xndimage.gaussian_filter(img, sigma=max(15, min(img.shape) // 20))
    return img - blurred


def extract_background(img: np.ndarray, mesh_size: int = 256, filter_size: int = 3,
                       clip_sigma: float = 3.0, clip_iters: int = 5) -> np.ndarray:
    """Estimate smooth sky background using mesh-based sigma-clipped statistics.

    Divides image into a grid, computes sigma-clipped median in each cell
    (rejecting stars), rejects cells contaminated by extended bright objects
    (nebulae), then interpolates the clean grid to a smooth background model.

    Args:
        img: 2D image array (single channel).
        mesh_size: Size of each grid cell in pixels.
        filter_size: Median filter size applied to the mesh grid to reject
            anomalous cells (in grid units, must be odd).
        clip_sigma: Sigma threshold for iterative clipping within each cell.
        clip_iters: Maximum iterations for sigma clipping.

    Returns:
        2D background model array with same shape as input.
    """
    H, W = img.shape
    ny = max(1, H // mesh_size)
    nx = max(1, W // mesh_size)

    cell_h = H / ny
    cell_w = W / nx

    bg_grid = np.zeros((ny, nx), dtype=np.float64)

    for iy in range(ny):
        y0 = int(round(iy * cell_h))
        y1 = min(int(round((iy + 1) * cell_h)), H)
        for ix in range(nx):
            x0 = int(round(ix * cell_w))
            x1 = min(int(round((ix + 1) * cell_w)), W)

            cell = img[y0:y1, x0:x1].ravel()
            if cell.size == 0:
                bg_grid[iy, ix] = 0.0
                continue

            # Iterative sigma clipping to reject stars
            if sigma_clipped_stats is not None:
                try:
                    _, median_val, _ = sigma_clipped_stats(
                        cell, sigma=clip_sigma, maxiters=clip_iters)
                    bg_grid[iy, ix] = float(median_val)
                    continue
                except Exception:
                    pass

            # Manual sigma-clipping fallback
            clipped = cell.copy()
            for _ in range(clip_iters):
                med = np.median(clipped)
                std = np.std(clipped)
                if std < 1e-12:
                    break
                mask = np.abs(clipped - med) < clip_sigma * std
                if not np.any(mask):
                    break
                clipped = clipped[mask]
            bg_grid[iy, ix] = float(np.median(clipped))

    # Reject grid cells contaminated by extended bright objects (nebulae).
    # Cells whose value is well above the overall grid median are replaced
    # with the grid median so the background model stays at the sky level.
    grid_median = float(np.median(bg_grid))
    grid_std = float(np.std(bg_grid))
    if grid_std > 1e-6:
        bright_thresh = grid_median + 2.0 * grid_std
        bright_mask = bg_grid > bright_thresh
        if np.any(bright_mask):
            bg_grid[bright_mask] = grid_median

    # Smooth the grid to reject remaining anomalous cells
    if filter_size > 1 and min(ny, nx) >= filter_size:
        bg_grid = ndimage.median_filter(bg_grid, size=filter_size)

    # Interpolate grid back to full image resolution
    from scipy.interpolate import RectBivariateSpline

    grid_y = np.array([(i + 0.5) * cell_h for i in range(ny)])
    grid_x = np.array([(j + 0.5) * cell_w for j in range(nx)])

    ky = min(3, ny - 1)
    kx = min(3, nx - 1)

    spline = RectBivariateSpline(grid_y, grid_x, bg_grid, kx=kx, ky=ky)
    background = spline(np.arange(H), np.arange(W)).astype(np.float32)

    # Clamp to grid value range to prevent spline overshoot
    np.clip(background, float(bg_grid.min()), float(bg_grid.max()),
            out=background)

    return background


def apply_background_extraction(rgb: np.ndarray, mesh_size: int = 256,
                                filter_size: int = 3, clip_sigma: float = 3.0,
                                verbose: bool = False) -> np.ndarray:
    """Apply intelligent background extraction to an RGB image.

    Estimates the background shape from the luminance channel, then scales
    that single spatial model to each RGB channel proportionally.  Using a
    shared shape ensures colour-neutral subtraction so star and nebula
    colours are preserved.

    Extended bright objects (nebulae) are detected at the grid level and
    excluded from the background estimate so they are not subtracted.

    Args:
        rgb: Image array with shape (H, W, 3).
        mesh_size: Grid cell size for background estimation.
        filter_size: Median filter size for grid smoothing.
        clip_sigma: Sigma for iterative clipping in each cell.
        verbose: Print per-channel statistics.

    Returns:
        Background-subtracted RGB image, clipped to >= 0.
    """
    # 1. Compute luminance and estimate a single background model from it
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    bg_lum = extract_background(lum, mesh_size=mesh_size,
                                filter_size=filter_size,
                                clip_sigma=clip_sigma)
    lum_bg_median = float(np.median(bg_lum))

    if verbose:
        safe_print(f"    Luminance background: median={lum_bg_median:.1f}, "
                   f"range={float(np.max(bg_lum) - np.min(bg_lum)):.1f}")

    # 2. For each channel, scale the luminance model to that channel's
    #    background level so the subtracted amount is proportional.
    result = np.empty_like(rgb)
    channel_names = ['Red', 'Green', 'Blue']

    for c in range(rgb.shape[2]):
        channel = rgb[:, :, c]

        # Estimate this channel's sky level via sigma-clipped stats
        if sigma_clipped_stats is not None:
            try:
                _, ch_bg_median, _ = sigma_clipped_stats(
                    channel, sigma=clip_sigma, maxiters=5)
                ch_bg_median = float(ch_bg_median)
            except Exception:
                ch_bg_median = float(np.median(channel))
        else:
            ch_bg_median = float(np.median(channel))

        # Scale factor: map luminance model amplitude to this channel
        if lum_bg_median > 1e-6:
            scale = ch_bg_median / lum_bg_median
        else:
            scale = 1.0

        bg_channel = bg_lum * scale
        subtracted = channel - bg_channel
        np.clip(subtracted, 0, None, out=subtracted)
        result[:, :, c] = subtracted

        if verbose:
            safe_print(f"    {channel_names[c]}: bg_median={ch_bg_median:.1f}, "
                       f"scale={scale:.3f}, "
                       f"subtracted median={float(np.median(subtracted)):.1f}")

    return result


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


def validate_image_data(img: np.ndarray, name: str = "") -> Tuple[bool, Optional[str]]:
    """Validate image data for common issues. Returns (is_valid, error_message)."""

    # Check for NaN or Inf
    if not np.isfinite(img).all():
        nan_count = np.isnan(img).sum()
        inf_count = np.isinf(img).sum()
        return False, f"contains {nan_count} NaN and {inf_count} Inf values"

    # Check for completely flat/dead image
    if np.std(img) < 0.1:
        return False, f"flat image (std={np.std(img):.3f})"

    # Check for saturated image (>95% of pixels at max value)
    max_val = np.max(img)
    if max_val > 0:
        saturated_fraction = np.sum(img >= max_val * 0.999) / img.size
        if saturated_fraction > 0.95:
            return False, f"saturated ({saturated_fraction*100:.1f}% at max)"

    # Check for mostly zeros (dead/corrupt)
    zero_fraction = np.sum(img == 0) / img.size
    if zero_fraction > 0.5:
        return False, f"mostly zeros ({zero_fraction*100:.1f}%)"

    # Check dynamic range
    p01, p99 = np.percentile(img, [1, 99])
    if p99 - p01 < 10:
        return False, f"insufficient dynamic range ({p99-p01:.1f})"

    return True, None


def compute_quality_metrics(img: np.ndarray) -> Dict:
    """Comprehensive quality analysis with multiple metrics."""

    # Basic statistics
    brightness = float(np.median(img))
    mean = float(np.mean(img))
    contrast = float(np.std(img))

    # Percentiles for outlier detection
    p01, p05, p25, p50, p75, p95, p99 = np.percentile(img, [1, 5, 25, 50, 75, 95, 99])

    # Signal-to-noise estimation
    # Use sigma-clipped statistics if available
    snr = 0.0
    background = mean
    noise = contrast

    if sigma_clipped_stats is not None:
        try:
            bg_mean, bg_median, bg_std = sigma_clipped_stats(img, sigma=3.0, maxiters=5)
            background = float(bg_median)
            noise = float(bg_std)
            snr = (p95 - background) / (noise + 1e-12) if noise > 0 else 0.0
        except:
            snr = (p95 - mean) / (contrast + 1e-12)
    else:
        snr = (p95 - mean) / (contrast + 1e-12)

    # Star detection
    star_count = 0
    star_snr = 0.0

    if DAOStarFinder is not None and sigma_clipped_stats is not None:
        try:
            bg_mean, bg_median, bg_std = sigma_clipped_stats(img, sigma=3.0)
            # Use more aggressive threshold for quality filtering
            threshold = background + 5.0 * noise
            daof = DAOStarFinder(fwhm=3.0, threshold=threshold)
            sources = daof(img - bg_median)

            if sources is not None and len(sources) > 0:
                star_count = len(sources)
                # Calculate median star SNR
                star_peaks = sources['peak']
                star_snr = float(np.median(star_peaks)) / (noise + 1e-12)
        except Exception as e:
            pass

    # Fallback star detection using local maxima
    if star_count == 0:
        try:
            # Find bright local maxima
            threshold = background + 3.0 * noise
            from scipy.ndimage import maximum_filter
            local_max = maximum_filter(img, size=5)
            detected_peaks = (img == local_max) & (img > threshold)
            star_count = int(np.sum(detected_peaks))

            if star_count > 0:
                peak_values = img[detected_peaks]
                star_snr = float(np.median(peak_values)) / (noise + 1e-12)
        except:
            star_count = 0

    # Focus/sharpness metric using Laplacian variance
    try:
        from scipy.ndimage import laplace
        laplacian = laplace(img.astype(np.float32))
        sharpness = float(np.var(laplacian))
    except:
        sharpness = 0.0

    # Composite quality score
    # Penalize low star count heavily
    star_factor = min(star_count / 50.0, 1.0) if star_count > 0 else 0.01
    snr_factor = min(snr / 10.0, 1.0) if snr > 0 else 0.01
    contrast_factor = min(contrast / 100.0, 1.0) if contrast > 0 else 0.01

    score = brightness * contrast * star_factor * snr_factor * 100.0

    return {
        'brightness': brightness,
        'mean': mean,
        'contrast': contrast,
        'snr': snr,
        'star_count': star_count,
        'star_snr': star_snr,
        'sharpness': sharpness,
        'background': background,
        'noise': noise,
        'score': score,
        'p01': p01,
        'p99': p99,
        'dynamic_range': p99 - p01
    }


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
            
            # Validate shift magnitude (sanity check)
            if np.isfinite(shift).all() and np.abs(shift).max() < max(ref.shape) * 0.5:
                # Check error threshold - phase correlation must have low error to be trusted
                # error < 0.01 is excellent, error < 0.1 is acceptable, error >= 0.5 is failure
                if error < 0.1:
                    if verbose:
                        if np.allclose(shift, 0.0):
                            print(f"      [phase_correlation: zero shift (error={error:.4f}, images well-aligned)]")
                        else:
                            print(f"      [phase_correlation succeeded: shift={shift}, error={error:.4f}]")
                    return float(shift[0]), float(shift[1])
                else:
                    debug_info.append(f"phase_cc rejected: high error ({error:.4f} >= 0.1)")
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
        
        # Use less aggressive downscaling - // 8 instead of // 64
        # This preserves small shifts while still being computationally tractable
        scale = max(1, ref.shape[0] // Config.XCORR_DOWNSCALE_TARGET)
        ref_small = ref_norm[::scale, ::scale]
        img_small = img_norm[::scale, ::scale]
        
        # correlate2d(ref, img) slides img over ref
        # Peak at position p means img centered at p in ref gives best match
        corr = correlate2d(ref_small, img_small, mode='same')
        peak = np.unravel_index(np.argmax(corr), corr.shape)
        center = np.array(corr.shape) // 2
        # Shift needed to move img from center to peak position
        shift_pixels = (peak - center) * scale
        
        # Reject if peak is at image corner (degenerate case)
        # Corners are: (0,0), (height-1,0), (0,width-1), (height-1,width-1)
        if peak[0] < 2 or peak[0] >= corr.shape[0] - 2 or peak[1] < 2 or peak[1] >= corr.shape[1] - 2:
            debug_info.append(f"xcorr rejected: degenerate peak at corner {peak}")
        else:
            # Check if peak correlation is strong enough (sanity check)
            peak_value = corr[peak] if len(corr.shape) > 0 else 0
            mean_corr = np.mean(np.abs(corr))
            
            if peak_value > mean_corr * 2 and peak_value > 0:  # Peak should be significantly above average
                if np.isfinite(shift_pixels).all() and np.abs(shift_pixels).max() < max(ref.shape) * 0.5:
                    if verbose:
                        print(f"      [xcorr fallback: shift=({shift_pixels[1]:.1f}, {shift_pixels[0]:.1f})]")
                    return float(shift_pixels[0]), float(shift_pixels[1])
                else:
                    debug_info.append(f"xcorr rejected: bad result {shift_pixels}")
            else:
                debug_info.append(f"xcorr: weak peak (peak={peak_value:.1f}, mean={mean_corr:.1f})")
    except Exception as e:
        debug_info.append(f"xcorr error: {type(e).__name__}")
    
    # Fallback to centroid difference - try multiple percentiles for robustness
    best_shift = (0.0, 0.0)
    best_score = float('inf')

    for percentile in Config.CENTROID_PERCENTILES:
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
                    # Shift needed to align img with ref: ref_center - img_center
                    # If img is down by 5px, img_center.y = ref_center.y + 5
                    # We need to shift img UP by -5, which is ref_center.y - img_center.y
                    shift_y = float(cim[0] - cim2[0])
                    shift_x = float(cim[1] - cim2[1])
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
    top = max_up + Config.CROP_MARGIN
    bottom = H - (max_down + Config.CROP_MARGIN)
    left = max_left + Config.CROP_MARGIN
    right = W - (max_right + Config.CROP_MARGIN)
    if top >= bottom or left >= right:
        return 0, H, 0, W
    return top, bottom, left, right


def sigma_clip_combine(data: np.ndarray, sigma: float = 3.0, max_iters: int = 3,
                       verbose: bool = False) -> np.ndarray:
    """Combine frames along axis 0 using iterative sigma-clipped mean.

    For each output pixel the value from every frame is collected.  Values
    more than *sigma* standard-deviations from the current mean are masked
    and the mean is recomputed.  This effectively removes hot pixels,
    cosmic rays, and satellite trails that appear in only a few frames —
    the key benefit of dithered exposures.

    Args:
        data: Array of shape ``(N, H, W, C)`` (all aligned frames).
        sigma: Rejection threshold in standard-deviations.
        max_iters: Maximum clipping iterations.
        verbose: Print per-iteration rejection statistics.

    Returns:
        Combined image of shape ``(H, W, C)``.
    """
    N = data.shape[0]
    # Start with all pixels included
    mask = np.ones(data.shape, dtype=bool)

    for iteration in range(max_iters):
        # Compute masked mean and std along frame axis
        masked = np.where(mask, data, np.nan)
        with np.errstate(all='ignore'):
            mean = np.nanmean(masked, axis=0)
            std = np.nanstd(masked, axis=0)

        # Reject pixels farther than sigma * std from the mean
        deviation = np.abs(data - mean[np.newaxis])
        new_mask = mask & (deviation <= sigma * std[np.newaxis])

        # Ensure at least 1 frame survives at every pixel
        surviving = new_mask.sum(axis=0)
        # Where all frames would be rejected, keep the original mask
        all_rejected = surviving == 0
        if np.any(all_rejected):
            for frame_idx in range(N):
                new_mask[frame_idx][all_rejected] = mask[frame_idx][all_rejected]

        rejected_this_iter = int(mask.sum() - new_mask.sum())
        mask = new_mask

        if verbose:
            total_pixels = mask.size
            total_rejected = int((~mask).sum())
            safe_print(f"    Iteration {iteration + 1}: rejected {rejected_this_iter} pixels "
                       f"(total {total_rejected}/{total_pixels}, "
                       f"{total_rejected / total_pixels * 100:.2f}%)")

        if rejected_this_iter == 0:
            break

    # Final masked mean
    masked = np.where(mask, data, np.nan)
    with np.errstate(all='ignore'):
        result = np.nanmean(masked, axis=0)

    # Replace any remaining NaN with zero (shouldn't happen with the guard above)
    np.nan_to_num(result, copy=False, nan=0.0)
    return result.astype(np.float32)


def detect_dither(shifts: List[Tuple[float, float]], verbose: bool = False) -> Dict:
    """Analyse registration shifts to detect dithering patterns.

    Returns a dict with:
      - ``is_dithered`` (bool)
      - ``pattern`` (str): 'dithered', 'tracking_drift', or 'aligned'
      - ``mean_magnitude`` (float): mean shift magnitude in pixels
      - ``unique_positions`` (int): approx distinct integer pixel positions
      - ``direction_spread_deg`` (float): std of shift angles in degrees
      - ``autocorrelation`` (float): correlation between consecutive shift vectors
    """
    if len(shifts) < 3:
        return {'is_dithered': False, 'pattern': 'aligned',
                'mean_magnitude': 0.0, 'unique_positions': len(shifts),
                'direction_spread_deg': 0.0, 'autocorrelation': 0.0}

    sy = np.array([s[0] for s in shifts])
    sx = np.array([s[1] for s in shifts])
    mags = np.sqrt(sy ** 2 + sx ** 2)

    mean_mag = float(np.mean(mags))

    # Count approximately unique integer positions
    int_positions = set((int(round(y)), int(round(x))) for y, x in shifts)
    unique_positions = len(int_positions)

    # Direction spread — std of angles (in degrees)
    non_zero = mags > 0.5  # ignore near-zero shifts
    if non_zero.sum() >= 3:
        angles = np.degrees(np.arctan2(sy[non_zero], sx[non_zero]))
        # Circular std: use the Mardia definition
        sin_mean = np.mean(np.sin(np.radians(angles)))
        cos_mean = np.mean(np.cos(np.radians(angles)))
        R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
        direction_spread = float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-12)))))
    else:
        direction_spread = 0.0

    # Autocorrelation of consecutive shift vectors (low = random / dithered)
    if len(shifts) >= 4:
        dx = np.diff(sx)
        dy = np.diff(sy)
        if len(dx) >= 2:
            autocorr_x = float(np.corrcoef(dx[:-1], dx[1:])[0, 1]) if np.std(dx) > 1e-6 else 0.0
            autocorr_y = float(np.corrcoef(dy[:-1], dy[1:])[0, 1]) if np.std(dy) > 1e-6 else 0.0
            autocorrelation = (autocorr_x + autocorr_y) / 2.0
            if not np.isfinite(autocorrelation):
                autocorrelation = 0.0
        else:
            autocorrelation = 0.0
    else:
        autocorrelation = 0.0

    # Classification heuristics
    all_near_zero = mean_mag < 1.0
    is_random_direction = direction_spread > 40.0  # degrees
    is_low_autocorr = abs(autocorrelation) < 0.5
    has_many_positions = unique_positions >= len(shifts) * 0.5

    if all_near_zero:
        pattern = 'aligned'
        is_dithered = False
    elif is_random_direction and is_low_autocorr and has_many_positions:
        pattern = 'dithered'
        is_dithered = True
    elif not is_random_direction or not is_low_autocorr:
        pattern = 'tracking_drift'
        is_dithered = False
    else:
        pattern = 'dithered'
        is_dithered = True

    result = {
        'is_dithered': is_dithered,
        'pattern': pattern,
        'mean_magnitude': mean_mag,
        'unique_positions': unique_positions,
        'direction_spread_deg': direction_spread,
        'autocorrelation': autocorrelation,
    }

    if verbose:
        labels = {'dithered': 'Dithered (random offsets detected)',
                  'tracking_drift': 'Tracking drift (systematic trend)',
                  'aligned': 'Well-aligned (minimal offsets)'}
        safe_print(f"\n  Dither analysis:")
        safe_print(f"    Pattern: {labels.get(pattern, pattern)}")
        safe_print(f"    Mean offset: {mean_mag:.1f} px")
        safe_print(f"    Unique positions: {unique_positions}/{len(shifts)} frames")
        if direction_spread > 0:
            safe_print(f"    Direction spread: {direction_spread:.1f} deg")
        safe_print(f"    Autocorrelation: {autocorrelation:.2f}")

    return result


def save_preview_rgb(rgb: np.ndarray, path: str):
    if Image is None or exposure is None:
        return
    # per-channel stretch
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], Config.PREVIEW_STRETCH_PERCENTILES)
        out[:, :, c] = exposure.rescale_intensity(rgb[:, :, c], in_range=(lo, hi))
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path, quality=Config.PREVIEW_JPEG_QUALITY)


def populate_fits_header(header: fits.Header, frames: List[FrameInfo], stats: ProcessingStats, args: argparse.Namespace, stacked_shape: Tuple[int, int, int], shifts: List[Tuple[float, float]], masters: Dict[str, Optional[np.ndarray]], dither_info: Optional[Dict] = None) -> None:
    """Populate FITS header with comprehensive metadata."""
    from datetime import datetime

    # Basic stacking info
    header['NFRAMES'] = (len(frames), 'Number of stacked frames')
    header['NREJECT'] = (stats.rejected_frames, 'Number of rejected frames')
    header['COMBINED'] = (True, 'Image is a stacked combination')
    header['STACKMTH'] = (args.stack_method.upper(), 'Stacking method (MEAN/MEDIAN/SIGMA_CLIP)')
    if args.stack_method == 'sigma_clip':
        header['REJSIGMA'] = (args.rejection_sigma, 'Sigma-clip rejection threshold')
        header['REJITERS'] = (args.rejection_iters, 'Sigma-clip rejection iterations')

    # Image dimensions
    header['NAXIS'] = 3
    header['NAXIS1'] = stacked_shape[1]  # Width
    header['NAXIS2'] = stacked_shape[0]  # Height
    header['NAXIS3'] = stacked_shape[2]  # Channels (3 for RGB)

    # Processing software and version
    header['CREATOR'] = ('astro_stack.py', 'Software that created this file')
    header['DATE'] = (datetime.now(datetime.timezone.utc).isoformat(), 'UTC date/time of file creation')

    # Calibration info
    header['BIASCAL'] = (masters.get('bias') is not None, 'Bias calibration applied')
    header['DARKCAL'] = (masters.get('dark') is not None, 'Dark calibration applied')
    header['FLATCAL'] = (masters.get('flat') is not None, 'Flat calibration applied')

    # Registration info
    if not args.no_registration and len(shifts) > 0:
        shifts_array = np.array(shifts)
        header['REGISTER'] = (True, 'Image registration applied')
        header['SHIFTX_M'] = (float(np.mean(shifts_array[:, 1])), 'Mean X shift in pixels')
        header['SHIFTY_M'] = (float(np.mean(shifts_array[:, 0])), 'Mean Y shift in pixels')
        header['SHIFTX_S'] = (float(np.std(shifts_array[:, 1])), 'Std dev of X shifts')
        header['SHIFTY_S'] = (float(np.std(shifts_array[:, 0])), 'Std dev of Y shifts')
        shift_mags = np.sqrt(shifts_array[:, 0]**2 + shifts_array[:, 1]**2)
        header['SHIFTMAX'] = (float(np.max(shift_mags)), 'Maximum shift magnitude in pixels')
    else:
        header['REGISTER'] = (False, 'No image registration applied')

    # Processing times
    header['PROCTIME'] = (stats.total_time(), 'Total processing time in seconds')
    header['QUALTIME'] = (stats.quality_time, 'Quality analysis time in seconds')
    header['REGTIME'] = (stats.registration_time, 'Registration time in seconds')
    header['STKTIME'] = (stats.stacking_time, 'Stacking time in seconds')

    # Memory usage
    if HAS_PSUTIL and stats.peak_memory_mb > 0:
        header['PEAKMEM'] = (stats.peak_memory_mb, 'Peak memory usage in MB')

    # Copy relevant metadata from first light frame
    if frames:
        first_header = frames[0].header
        # Copy common FITS keywords if they exist
        copy_keys = ['TELESCOP', 'INSTRUME', 'OBSERVER', 'OBJECT', 'DATE-OBS',
                     'EXPTIME', 'CCD-TEMP', 'GAIN', 'OFFSET', 'XBINNING', 'YBINNING',
                     'BAYERPAT', 'XPIXSZ', 'YPIXSZ', 'FOCALLEN', 'APTDIA']
        for key in copy_keys:
            if key in first_header:
                header[key] = first_header[key]

        # Calculate total exposure time
        if 'EXPTIME' in first_header:
            try:
                total_exp = float(first_header['EXPTIME']) * len(frames)
                header['TOTEXP'] = (total_exp, 'Total integrated exposure time in seconds')
            except (ValueError, TypeError):
                pass

    # Background extraction info
    if args.background_extraction:
        header['BGEXTR'] = (True, 'Background extraction applied')
        header['BGMESH'] = (args.bg_mesh_size, 'Background mesh cell size in pixels')
        header['BGFILTR'] = (args.bg_filter_size, 'Background grid filter size')
        header['BGCLIP'] = (args.bg_clip_sigma, 'Background sigma-clip threshold')
    else:
        header['BGEXTR'] = (False, 'No background extraction applied')

    # Dither analysis info
    if dither_info is not None:
        header['DITHERED'] = (dither_info['is_dithered'], 'Dithering detected in frame shifts')
        header['DITHMAG'] = (round(dither_info['mean_magnitude'], 2), 'Mean dither magnitude in pixels')
        header['DITHPOS'] = (dither_info['unique_positions'], 'Number of unique dither positions')
        header['DITHPAT'] = (dither_info['pattern'], 'Detected shift pattern type')

    # Add quality metrics if available
    if frames and frames[0].metrics:
        avg_metrics = {
            'brightness': np.mean([f.metrics.get('brightness', 0) for f in frames if f.metrics]),
            'contrast': np.mean([f.metrics.get('contrast', 0) for f in frames if f.metrics]),
            'score': np.mean([f.metrics.get('score', 0) for f in frames if f.metrics]),
        }
        header['AVGBRITE'] = (float(avg_metrics['brightness']), 'Average frame brightness')
        header['AVGCONTR'] = (float(avg_metrics['contrast']), 'Average frame contrast')
        header['AVGSCORE'] = (float(avg_metrics['score']), 'Average quality score')


def solve_plate(image_data: np.ndarray, header: fits.Header, output_path: str, verbose: bool = False) -> bool:
    """
    Attempt to plate-solve the stacked image and add WCS + object info to header.

    Args:
        image_data: Stacked image data (H, W, 3) or (3, H, W)
        header: FITS header to update with WCS info
        output_path: Path where FITS file is saved
        verbose: Print detailed progress

    Returns:
        True if plate solving succeeded, False otherwise
    """
    if not HAS_ASTROMETRY_NET:
        if verbose:
            print("  [Plate solving] astroquery not available - skipping")
        return False

    try:
        # Convert to grayscale luminance for plate solving
        if image_data.ndim == 3:
            if image_data.shape[0] == 3:
                # (3, H, W) format
                lum = 0.299 * image_data[0] + 0.587 * image_data[1] + 0.114 * image_data[2]
            else:
                # (H, W, 3) format
                lum = 0.299 * image_data[:, :, 0] + 0.587 * image_data[:, :, 1] + 0.114 * image_data[:, :, 2]
        else:
            lum = image_data

        # Create temporary FITS file with luminance for submission
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save luminance to temporary file
            tmp_hdu = fits.PrimaryHDU(lum.astype(np.float32))
            # Copy relevant keywords that might help plate solving
            for key in ['TELESCOP', 'INSTRUME', 'FOCALLEN', 'XPIXSZ', 'YPIXSZ', 'APTDIA']:
                if key in header:
                    tmp_hdu.header[key] = header[key]
            tmp_hdu.writeto(tmp_path, overwrite=True)

            if verbose:
                print("  [Plate solving] Submitting to astrometry.net...")

            # Initialize astrometry.net client
            ast = AstrometryNet()

            # Check for API key in environment or config
            api_key = os.environ.get('ASTROMETRY_API_KEY')
            if not api_key:
                if verbose:
                    print("  [Plate solving] No API key found. Set ASTROMETRY_API_KEY environment variable.")
                    print("  [Plate solving] Get a key from: https://nova.astrometry.net/api_help")
                return False

            ast.api_key = api_key

            # Estimate scale from header if available
            scale_units = 'arcsecperpix'
            scale_lower = None
            scale_upper = None

            if 'FOCALLEN' in header and 'XPIXSZ' in header:
                try:
                    focal_length_mm = float(header['FOCALLEN'])
                    pixel_size_um = float(header['XPIXSZ'])
                    # Calculate plate scale: pixel_size / focal_length * 206265 arcsec/radian
                    plate_scale = (pixel_size_um / 1000.0) / focal_length_mm * 206265.0
                    scale_lower = plate_scale * 0.9  # 10% tolerance
                    scale_upper = plate_scale * 1.1
                    if verbose:
                        print(f"  [Plate solving] Estimated scale: {plate_scale:.2f} arcsec/pixel")
                except (ValueError, ZeroDivisionError):
                    pass

            # Submit for solving
            wcs_header = ast.solve_from_image(
                tmp_path,
                force_image_upload=True,
                solve_timeout=300,  # 5 minute timeout
                scale_units=scale_units,
                scale_lower=scale_lower,
                scale_upper=scale_upper,
                publicly_visible='n'
            )

            if wcs_header:
                if verbose:
                    print("  [Plate solving] ✓ Success! Adding WCS to header...")

                # Copy WCS keywords to main header
                wcs_keywords = ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                               'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CROTA1', 'CROTA2',
                               'EQUINOX', 'RADECSYS', 'CUNIT1', 'CUNIT2']
                for key in wcs_keywords:
                    if key in wcs_header:
                        header[key] = wcs_header[key]

                # Add plate solving success flag
                header['PLTSOLVD'] = (True, 'Plate solving successful')

                # Try to identify object using SIMBAD
                if 'CRVAL1' in header and 'CRVAL2' in header:
                    try:
                        from astroquery.simbad import Simbad
                        ra = float(header['CRVAL1'])
                        dec = float(header['CRVAL2'])

                        # Query SIMBAD for objects near the center
                        custom_simbad = Simbad()
                        custom_simbad.add_votable_fields('otype')
                        result = custom_simbad.query_region(f"{ra} {dec}", radius='0d30m0s', frame='icrs')

                        if result and len(result) > 0:
                            # Get the brightest/most prominent object
                            obj_name = result[0]['MAIN_ID']
                            obj_type = result[0]['OTYPE']
                            if verbose:
                                print(f"  [Plate solving] Identified object: {obj_name} ({obj_type})")
                            header['OBJECT'] = (str(obj_name), 'Object identified via SIMBAD')
                            header['OBJTYPE'] = (str(obj_type), 'Object type from SIMBAD')
                    except Exception as e:
                        if verbose:
                            print(f"  [Plate solving] Object identification failed: {e}")

                return True
            else:
                if verbose:
                    print("  [Plate solving] Failed to solve plate")
                header['PLTSOLVD'] = (False, 'Plate solving attempted but failed')
                return False

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

    except Exception as e:
        if verbose:
            print(f"  [Plate solving] Error: {e}")
        header['PLTSOLVD'] = (False, f'Plate solving error: {str(e)[:50]}')
        return False


def stack_target(frames: List[FrameInfo], output_path: str, args: argparse.Namespace, masters: Dict[str, Optional[np.ndarray]], stats: ProcessingStats):
    lights = [f for f in frames if f.type == 'light']
    if not lights:
        print('  No light frames found for target')
        return None

    stats.total_frames = len(lights)

    # Phase 1: Quality Analysis
    print_phase(1, "Quality Analysis")
    phase_start = time.time()
    print(f"  Analyzing {len(lights)} light frames...")

    accepted = []
    rejected_reasons = {}

    # Create progress iterator
    lights_iter = tqdm(lights, desc="  Analyzing", unit="frame", disable=args.verbose) if not args.verbose else lights

    for f in lights_iter:
        try:
            data, hdr = load_fits(f.path)
        except Exception as e:
            f.accepted = False
            error_msg = f'load error: {str(e)}'
            f.metrics = {'error': error_msg}
            rejected_reasons[f.path] = error_msg
            stats.add_error(f.path, error_msg)
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {error_msg}')
            continue
        # Validate data is not empty
        if data is None or data.size == 0:
            f.accepted = False
            error_msg = 'empty data array'
            f.metrics = {'error': error_msg}
            rejected_reasons[f.path] = error_msg
            stats.add_error(f.path, error_msg)
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {error_msg}')
            continue
        # Calibration with validation
        try:
            # Apply bias subtraction
            if masters.get('bias') is not None:
                if masters['bias'].shape == data.shape:
                    data = np.clip(data - masters['bias'], 0, None)
                else:
                    if args.verbose:
                        print(f'    WARNING: bias shape mismatch for {os.path.basename(f.path)}, skipping bias')

            # Apply dark subtraction
            if masters.get('dark') is not None:
                if masters['dark'].shape == data.shape:
                    data = np.clip(data - masters['dark'], 0, None)
                else:
                    if args.verbose:
                        print(f'    WARNING: dark shape mismatch for {os.path.basename(f.path)}, skipping dark')

            # Apply flat division
            if masters.get('flat') is not None:
                if masters['flat'].shape == data.shape:
                    flat = masters['flat'].copy()
                    # Normalize flat to median
                    med = np.median(flat)
                    if med > 1e-6:  # Avoid division by zero
                        flat_norm = flat / med
                        # Avoid division by very small numbers
                        flat_norm = np.clip(flat_norm, 0.1, 10.0)
                        data = data / flat_norm
                    else:
                        if args.verbose:
                            print(f'    WARNING: flat field median too low ({med:.6f}), skipping flat')
                else:
                    if args.verbose:
                        print(f'    WARNING: flat shape mismatch for {os.path.basename(f.path)}, skipping flat')

            # Check for calibration artifacts (negative or NaN values)
            if not np.isfinite(data).all():
                raise ValueError(f"calibration produced non-finite values")
            if np.any(data < 0):
                if args.verbose:
                    neg_count = np.sum(data < 0)
                    print(f'    WARNING: {neg_count} negative values after calibration, clipping to zero')
                data = np.clip(data, 0, None)

        except Exception as e:
            f.accepted = False
            error_msg = f'calibration error: {str(e)}'
            f.metrics = {'error': error_msg}
            rejected_reasons[f.path] = error_msg
            stats.add_error(f.path, error_msg)
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {error_msg}')
            continue
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
            error_msg = f'debayering error: {str(e)}'
            f.metrics = {'error': error_msg}
            rejected_reasons[f.path] = error_msg
            stats.add_error(f.path, error_msg)
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {error_msg}')
            continue
        # hot pixel removal
        try:
            if rgb.ndim != 3 or rgb.shape[2] < 1:
                raise ValueError(f'Invalid RGB shape: {rgb.shape}')
            for c in range(rgb.shape[2]):
                rgb[:, :, c] = remove_hot_pixels(rgb[:, :, c])
        except Exception as e:
            f.accepted = False
            error_msg = f'hot pixel removal error: {str(e)}'
            f.metrics = {'error': error_msg}
            rejected_reasons[f.path] = error_msg
            stats.add_error(f.path, error_msg)
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {error_msg}')
            continue
        # Validate image data
        try:
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

            # First pass: validate image is usable
            is_valid, validation_error = validate_image_data(lum, os.path.basename(f.path))
            if not is_valid:
                f.accepted = False
                error_msg = f'validation failed: {validation_error}'
                f.metrics = {'error': error_msg}
                rejected_reasons[f.path] = error_msg
                if args.verbose:
                    print(f'  REJECT {os.path.basename(f.path)}: {error_msg}')
                continue

            # Compute comprehensive quality metrics
            metrics = compute_quality_metrics(lum)
            f.metrics = metrics

            # Hard quality gates (reject obviously bad frames immediately)
            reject_reason = None

            # Gate 1: Minimum star count (very lenient - just need SOME stars)
            if metrics['star_count'] < 3:
                reject_reason = f"insufficient stars ({metrics['star_count']} < 3)"

            # Gate 2: Minimum SNR (relaxed threshold)
            elif metrics['snr'] < 0.5:
                reject_reason = f"extremely low SNR ({metrics['snr']:.2f} < 0.5)"

            # Gate 3: Minimum contrast
            elif metrics['contrast'] < 2.0:
                reject_reason = f"extremely low contrast ({metrics['contrast']:.1f} < 2.0)"

            # Gate 4: Minimum dynamic range
            elif metrics['dynamic_range'] < 20:
                reject_reason = f"extremely low dynamic range ({metrics['dynamic_range']:.1f} < 20)"

            # Gate 5: Check for excessive noise (very lenient)
            elif metrics['noise'] > metrics['brightness'] * 0.8:
                reject_reason = f"excessive noise ({metrics['noise']:.1f} > {metrics['brightness']*0.8:.1f})"

            if reject_reason:
                f.accepted = False
                rejected_reasons[f.path] = reject_reason
                if args.verbose:
                    print(f'  REJECT {os.path.basename(f.path)}: {reject_reason}')
                continue

            # Passed all hard gates - add to potential accepts
            accepted.append(f)
            if args.verbose:
                print(f'    {os.path.basename(f.path)}: SNR={metrics["snr"]:.1f}, stars={metrics["star_count"]}, contrast={metrics["contrast"]:.1f}, score={metrics["score"]:.1f}')

        except Exception as e:
            f.accepted = False
            error_msg = f'quality analysis error: {str(e)}'
            f.metrics = {'error': error_msg}
            rejected_reasons[f.path] = error_msg
            stats.add_error(f.path, error_msg)
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {error_msg}')
            continue

    # Statistical outlier detection
    if len(accepted) > 3:
        # Collect metrics arrays
        snrs = np.array([f.metrics['snr'] for f in accepted])
        star_counts = np.array([f.metrics['star_count'] for f in accepted])
        contrasts = np.array([f.metrics['contrast'] for f in accepted])
        scores = np.array([f.metrics['score'] for f in accepted])

        # Calculate z-scores for each metric
        def reject_outliers(values, threshold=2.5):
            """Return boolean mask of outliers (True = keep, False = reject)."""
            if len(values) < 3:
                return np.ones(len(values), dtype=bool)
            mean = np.mean(values)
            std = np.std(values)
            if std < 1e-6:  # All values identical
                return np.ones(len(values), dtype=bool)
            z_scores = np.abs((values - mean) / std)
            return z_scores < threshold

        # Identify outliers
        snr_ok = reject_outliers(snrs, threshold=2.5)
        star_ok = reject_outliers(star_counts, threshold=2.5)
        contrast_ok = reject_outliers(contrasts, threshold=2.5)

        # Combined outlier detection (reject if outlier in 2+ metrics)
        outlier_count = (~snr_ok).astype(int) + (~star_ok).astype(int) + (~contrast_ok).astype(int)

        for i, f in enumerate(accepted):
            if outlier_count[i] >= 2:
                outlier_reasons = []
                if not snr_ok[i]:
                    outlier_reasons.append(f"SNR={f.metrics['snr']:.1f} (mean={np.mean(snrs):.1f})")
                if not star_ok[i]:
                    outlier_reasons.append(f"stars={f.metrics['star_count']} (mean={np.mean(star_counts):.0f})")
                if not contrast_ok[i]:
                    outlier_reasons.append(f"contrast={f.metrics['contrast']:.1f} (mean={np.mean(contrasts):.1f})")

                reason = f"statistical outlier: " + ", ".join(outlier_reasons)
                rejected_reasons[f.path] = reason
                f.accepted = False
                if args.verbose:
                    print(f'  REJECT {os.path.basename(f.path)}: {reason}')
            else:
                f.accepted = True

    # If quality_filter, apply additional percentile threshold
    if args.quality_filter and accepted:
        # Only consider non-rejected frames
        valid_frames = [f for f in accepted if f.accepted]
        if valid_frames:
            scores = np.array([f.metrics['score'] for f in valid_frames])
            pct = np.percentile(scores, args.quality_threshold)

            for f in valid_frames:
                if f.metrics['score'] < pct:
                    f.accepted = False
                    rejected_reasons[f.path] = f'quality score {f.metrics["score"]:.1f} below threshold {pct:.1f}'
                    if args.verbose:
                        print(f'  REJECT {os.path.basename(f.path)}: score {f.metrics["score"]:.1f} < {pct:.1f}')
    # Build list of final accepted frames
    final = [f for f in lights if f.accepted]
    stats.accepted_frames = len(final)
    stats.rejected_frames = len(lights) - len(final)
    stats.quality_time = time.time() - phase_start

    # Show detailed table in verbose mode
    if args.verbose:
        print_quality_table(lights, show_all=len(lights) <= 50)

    # Summary of quality analysis
    safe_print(f"  ✓ Accepted: {len(final)}/{len(lights)} ({len(final)/len(lights)*100:.1f}%)")
    if stats.rejected_frames > 0:
        # Categorize rejection reasons
        reason_counts = {}
        for reason in rejected_reasons.values():
            # Extract category
            if 'brightness' in reason or 'contrast' in reason:
                category = 'Poor quality'
            elif 'stars' in reason or 'star' in reason:
                category = 'No stars detected'
            elif 'load' in reason or 'empty' in reason:
                category = 'Load/data errors'
            else:
                category = 'Other'
            reason_counts[category] = reason_counts.get(category, 0) + 1

        safe_print(f"  ✗ Rejected: {stats.rejected_frames} ({', '.join(f'{cat}: {cnt}' for cat, cnt in reason_counts.items())})")

    if not final:
        print(f'\n  ERROR: All {len(lights)} frames rejected!')
        if rejected_reasons:
            print('  Rejection reasons:')
            for path, reason in list(rejected_reasons.items())[:10]:  # Show first 10
                print(f'    • {os.path.basename(path)}: {reason}')
            if len(rejected_reasons) > 10:
                print(f'    ... and {len(rejected_reasons) - 10} more')
        return None
    # Phase 2: Registration
    print_phase(2, "Registration")
    phase_start = time.time()

    ref = None
    ref_path = None
    best = max(final, key=lambda x: x.metrics.get('score', 0))
    ref_path = best.path
    print(f"  Reference frame: {os.path.basename(ref_path)} (score={best.metrics.get('score', 0):.1f})")

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
    # Prepare memmap for median/sigma_clip stacking if requested
    use_memmap = args.stack_method in ('median', 'sigma_clip')
    n = len(final)
    H, W = ref_lum.shape
    # create temporary memmap if median or sigma_clip
    memmap_path = None
    mem = None
    if use_memmap:
        memmap_path = os.path.join(tempfile.gettempdir(), f'stack_{os.getpid()}.dat')
        mem = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(n, H, W, 3))

    idx = 0
    ref_lum_std = np.std(ref_lum)  # Cache for diagnostics

    if args.verbose:
        # Diagnostic: show reference image statistics
        ref_stats = {
            'min': np.min(ref_lum),
            'max': np.max(ref_lum),
            'mean': np.mean(ref_lum),
            'std': ref_lum_std,
        }
        print(f'  Reference luminance: min={ref_stats["min"]:.1f}, max={ref_stats["max"]:.1f}, mean={ref_stats["mean"]:.1f}, std={ref_stats["std"]:.1f}')

    # Create progress iterator
    print(f"  Calculating shifts for {len(final)} frames...")
    final_iter = tqdm(final, desc="  Registering", unit="frame", disable=args.verbose) if not args.verbose else final

    for f in final_iter:
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
        if use_memmap:
            mem[idx] = aligned
            mem.flush()
        else:
            tmp_files.append(aligned.astype(np.float32))
        aligned_shapes.append(aligned.shape[:2])
        idx += 1

    stats.registration_time = time.time() - phase_start

    # Calculate shift statistics
    shift_x = [s[1] for s in shifts]
    shift_y = [s[0] for s in shifts]
    shift_mags = [np.sqrt(sx**2 + sy**2) for sx, sy in shifts]

    if not args.no_registration:
        print(f"  Shift statistics:")
        print(f"    X: mean={np.mean(shift_x):+.1f}px, std={np.std(shift_x):.1f}px, range=[{np.min(shift_x):+.1f}, {np.max(shift_x):+.1f}]")
        print(f"    Y: mean={np.mean(shift_y):+.1f}px, std={np.std(shift_y):.1f}px, range=[{np.min(shift_y):+.1f}, {np.max(shift_y):+.1f}]")
        print(f"    Magnitude: mean={np.mean(shift_mags):.1f}px, max={np.max(shift_mags):.1f}px")

        # Check for large shifts
        if np.max(shift_mags) > Config.LARGE_SHIFT_WARNING_PX:
            warning = f"Large shifts detected (max={np.max(shift_mags):.1f}px) - possible tracking issues"
            stats.add_warning(warning)
            safe_print(f"  ⚠ {warning}")

    # Check for suspicious shift patterns (possible algorithm failure)
    shift_set = set(f.shift for f in final)
    zero_shifts = sum(1 for f in final if f.shift == (0.0, 0.0))
    
    # Only warn about identical shifts - all frames having zero shift might be correct
    # if the images are naturally well-aligned
    if len(shift_set) == 1 and len(final) > 2:
        unique_shift = list(shift_set)[0]
        if unique_shift != (0.0, 0.0):
            # All non-zero identical shifts = impossible, algorithm failure
            warning = f'All {len(final)} frames have IDENTICAL shift {unique_shift} - registration algorithm failure!'
            stats.add_warning(warning)
            safe_print(f'\n  ⚠ WARNING: {warning}')
            print(f'  DIAGNOSIS: Registration algorithm is not distinguishing between frames.')
            print(f'  SUGGESTION: Try with --skip-phase-correlation --debug-registration')
            print(f'  Then check PNG images in _registration_debug/ folder.')
        else:
            # All frames have zero shift - this is valid if phase correlation succeeded with low error
            if len(final) <= 3:
                safe_print(f'\n  ℹ INFO: All {len(final)} frames registered with zero shift - images appear well-aligned.')
    elif zero_shifts > len(final) * 0.8 and len(final) > 2:
        # Majority (but not all) have zero shifts - this suggests inconsistency
        warning = f'{zero_shifts}/{len(final)} frames have zero shift - registration may have issues'
        stats.add_warning(warning)
        safe_print(f'\n  ⚠ WARNING: {warning}')
        print(f'  SUGGESTION: Try with --debug-registration to visualize registration')
        print(f'  Check PNG images in _registration_debug/ folder.')
    
    # Dither analysis
    dither_info = detect_dither(shifts, verbose=args.verbose)
    if not args.no_registration and len(shifts) > 2:
        print(f"\n  Dither analysis:")
        print(f"    Pattern: {dither_info['pattern'].replace('_', ' ').title()}")
        print(f"    Mean shift: {dither_info['mean_magnitude']:.1f} px")
        print(f"    Unique positions: {dither_info['unique_positions']}/{len(shifts)} frames")
        print(f"    Direction spread: {dither_info['direction_spread']:.1f}\u00b0")
        if dither_info['is_dithered'] and args.stack_method == 'mean':
            safe_print(f"    \u2139 Recommendation: Use --stack-method sigma_clip for best results with dithered data")

    # Phase 3: Stacking
    print_phase(3, "Stacking")
    phase_start = time.time()
    print(f"  Method: {args.stack_method}")
    print(f"  Combining {len(final)} frames...")

    # Crop to common valid region
    top, bottom, left, right = calc_common_crop([f.shift for f in final], (H, W))
    cropped_h = H - (top + (H - bottom))
    cropped_w = W - (left + (W - right))
    stats.output_shape = (bottom - top, right - left)
    stats.cropped_pixels = (H - (bottom - top), W - (right - left))

    # Crop & combine
    if args.stack_method == 'sigma_clip':
        cropped_data = mem[:, top:bottom, left:right, :]
        print(f"  Sigma-clip rejection: sigma={args.rejection_sigma}, iters={args.rejection_iters}")
        stacked = sigma_clip_combine(
            cropped_data,
            sigma=args.rejection_sigma,
            max_iters=args.rejection_iters,
            verbose=args.verbose
        )
        # cleanup memmap file
        try:
            del mem
            os.remove(memmap_path)
        except Exception:
            pass
    elif args.stack_method == 'median':
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
    stats.stacking_time = time.time() - phase_start

    # Post-stack background extraction
    if args.background_extraction:
        print(f"\n  Applying background extraction (mesh={args.bg_mesh_size}, "
              f"sigma={args.bg_clip_sigma})...")
        bg_start = time.time()
        stacked = apply_background_extraction(
            stacked,
            mesh_size=args.bg_mesh_size,
            filter_size=args.bg_filter_size,
            clip_sigma=args.bg_clip_sigma,
            verbose=args.verbose
        )
        bg_time = time.time() - bg_start
        safe_print(f"  ✓ Background extraction complete ({format_time(bg_time)})")

    # Update memory usage
    if HAS_PSUTIL:
        stats.peak_memory_mb = get_memory_usage_mb()

    # Save FITS (3,H,W)
    out_h, out_w, _ = stacked.shape
    hdu = fits.PrimaryHDU()
    # store as (3,H,W)
    data_out = np.transpose(stacked, (2, 0, 1)).astype(np.float32)
    hdu.data = data_out

    # Populate comprehensive FITS header
    populate_fits_header(
        header=hdu.header,
        frames=final,
        stats=stats,
        args=args,
        stacked_shape=stacked.shape,
        shifts=shifts,
        masters=masters,
        dither_info=dither_info
    )

    # Save FITS file
    hdu.writeto(output_path, overwrite=True)

    # Attempt plate solving to add WCS and object identification
    plate_solved = False
    if not args.skip_plate_solve:
        if args.verbose:
            print("\n  Attempting plate solving...")
        plate_solved = solve_plate(data_out, hdu.header, output_path, verbose=args.verbose)

        # Re-save FITS with updated header if plate solving succeeded
        if plate_solved:
            hdu.writeto(output_path, overwrite=True)
            if args.verbose:
                print("  FITS header updated with WCS and object info")
    elif args.verbose:
        print("\n  Plate solving skipped (--skip-plate-solve)")

    # preview
    preview_path = os.path.splitext(output_path)[0] + '.jpg'
    save_preview_rgb(stacked, preview_path)

    print(f"  Output size: {out_h}×{out_w} (cropped {stats.cropped_pixels[0]}×{stats.cropped_pixels[1]} pixels)")

    # Print summary
    print_header("SUMMARY", "=")
    print(f"  Frames analyzed:  {stats.total_frames}")
    print(f"  Frames stacked:   {stats.accepted_frames} ({stats.accepted_frames/stats.total_frames*100:.1f}%)")
    if stats.rejected_frames > 0:
        print(f"  Frames rejected:  {stats.rejected_frames}")
    print(f"  Output:           {os.path.basename(output_path)} ({out_h}×{out_w}×3)")
    print(f"  Preview:          {os.path.basename(preview_path)}")
    print(f"  Processing time:  {format_time(stats.total_time())}")
    print(f"    Quality:        {format_time(stats.quality_time)}")
    print(f"    Registration:   {format_time(stats.registration_time)}")
    print(f"    Stacking:       {format_time(stats.stacking_time)}")

    if HAS_PSUTIL:
        print(f"  Peak memory:      {stats.peak_memory_mb:.1f} MB")
        frames_per_gb = (stats.accepted_frames / (stats.peak_memory_mb / 1024)) if stats.peak_memory_mb > 0 else 0
        if frames_per_gb > 0:
            print(f"  Memory efficiency: {frames_per_gb:.1f} frames/GB")

    # Print warnings if any
    if stats.warnings:
        safe_print(f"\n  ⚠ Warnings:")
        for warning in stats.warnings[:5]:  # Show first 5
            safe_print(f"    • {warning}")
        if len(stats.warnings) > 5:
            safe_print(f"    ... and {len(stats.warnings) - 5} more")

    # Print errors if any
    if stats.errors:
        safe_print(f"\n  ✗ Errors encountered: {len(stats.errors)}")
        if args.verbose:
            for path, error in stats.errors[:10]:
                safe_print(f"    • {os.path.basename(path)}: {error}")
            if len(stats.errors) > 10:
                safe_print(f"    ... and {len(stats.errors) - 10} more")

    safe_print(f"\n  ✓ Stack complete!")
    return output_path


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

        if frames['dark'] or frames['flat'] or frames['bias']:
            stats.calibration_time = time.time() - cal_start

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

        # load all stacks, resize to minimum
        stacks = []
        shapes = [fits.open(p)[0].data.shape for p in produced]
        # shapes are (3,H,W)
        mins = np.min([[s[1], s[2]] for s in shapes], axis=0)
        Hm, Wm = int(mins[0]), int(mins[1])
        print(f"  Resizing all to minimum dimensions: {Hm}×{Wm}")

        acc = None
        for p in tqdm(produced, desc="  Combining", unit="target", disable=args.verbose):
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
        preview_path = os.path.splitext(output)[0] + '.jpg'
        save_preview_rgb(combined, preview_path)

        safe_print(f"  ✓ Combined output: {os.path.basename(output)} ({Hm}×{Wm}×3)")
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
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--no-registration', action='store_true')
    p.add_argument('--skip-phase-correlation', action='store_true', help='Skip phase correlation, use only fallback methods (debug)')
    p.add_argument('--quality-filter', action='store_true')
    p.add_argument('--quality-threshold', type=float, default=50.0)
    p.add_argument('--keep-intermediates', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--debug-registration', action='store_true', help='Detailed registration diagnostics (implies -v)')
    p.add_argument('--stack-method', choices=['mean', 'median', 'sigma_clip'], default='mean')
    p.add_argument('--rejection-sigma', type=float, default=3.0,
                   help='Sigma threshold for pixel rejection in sigma_clip stacking (default: 3.0)')
    p.add_argument('--rejection-iters', type=int, default=3,
                   help='Number of clipping iterations for sigma_clip stacking (default: 3)')
    p.add_argument('--debayer-method', choices=['bilinear', 'malvar'], default='bilinear')
    p.add_argument('--white-balance', choices=['none', 'grayworld', 'whitepatch'], default='grayworld')
    p.add_argument('--drizzle-scale', type=int, default=1, help='Integer drizzle scale factor (1 = disabled)')
    p.add_argument('--use-gpu', action='store_true', help='Use CuPy for available operations (experimental)')
    p.add_argument('--skip-plate-solve', action='store_true', help='Skip plate solving (astrometry)')
    p.add_argument('--background-extraction', action='store_true',
                   help='Enable intelligent background removal for darker sky')
    p.add_argument('--bg-mesh-size', type=int, default=256,
                   help='Grid cell size in pixels for background estimation (default: 256)')
    p.add_argument('--bg-filter-size', type=int, default=3,
                   help='Median filter size for background grid smoothing (default: 3, must be odd)')
    p.add_argument('--bg-clip-sigma', type=float, default=3.0,
                   help='Sigma for star rejection in background estimation (default: 3.0)')
    return p.parse_args()


def main():
    args = parse_args()
    # debug_registration implies verbose
    if args.debug_registration:
        args.verbose = True
    # Initialise GPU context (module-level singleton)
    global _gpu
    _gpu = GpuContext(use_gpu=args.use_gpu)
    try:
        process_directory(args.directory, args.output, args)
    except Exception as e:
        print(f'ERROR: {str(e)}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
