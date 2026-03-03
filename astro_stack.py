"""Astro FITS Stream Stacker

Features:
- Streaming processing (constant memory)
- Calibration (bias/dark/flat)
- Debayering (bilinear, Malvar, VNG via OpenCV)
- Quality analysis (brightness, contrast, star count, FWHM)
- Registration (sub-pixel phase correlation, FFT cross-correlation, affine/star-matching)
- Automatic cropping, hierarchical processing, preview generation
- Intelligent background extraction (mesh-based sigma-clipped sky removal with star masking)
- GPU acceleration via CuPy (--use-gpu) with automatic CPU fallback
- Parallel frame processing via multiprocessing (-j)
- Quality-weighted stacking, MAD-based sigma clipping, winsorized combine
- Wavelet denoising, local normalization, arcsinh preview stretch
- White balance, hot pixel removal, gradient removal

Usage: python astro_stack.py -d INPUT_DIR -o OUTPUT.fits [options]
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import tempfile
import time
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np

from astropy.io import fits

try:
    from scipy import ndimage
    from scipy import fftpack
    from scipy.signal import fftconvolve
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
    from skimage.transform import EuclideanTransform
    from skimage.measure import ransac
    HAS_SKIMAGE_TRANSFORM = True
except Exception:
    HAS_SKIMAGE_TRANSFORM = False

try:
    from photutils.detection import DAOStarFinder
except Exception:
    DAOStarFinder = None
try:
    from astropy.stats import sigma_clipped_stats
except Exception:
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

try:
    import pywt
    HAS_PYWT = True
except Exception:
    HAS_PYWT = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


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
    HOT_PIXEL_THRESHOLD = 12.0
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
    TILE_SIZE = 256  # Tile size for tiled sigma-clip (pixels)
    FWHM_CUTOUT_RADIUS = 10  # Cutout radius for FWHM measurement
    FWHM_MAX_STARS = 50  # Max stars to measure for FWHM
    ARCSINH_STRETCH_FACTOR = 5.0  # Default arcsinh stretch factor
    STAR_MASK_MAX_STARS = 500  # Max stars for mask generation
    AFFINE_MAX_STARS = 80  # Max stars for affine matching
    AFFINE_MATCH_RADIUS = 10.0  # Max pixel distance for star matching


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
    """Create master calibration frame using streaming (mean) or memmap (median)."""
    if not frames:
        return None
    # Probe first frame for shape
    try:
        first_data, _ = load_fits(frames[0].path)
        shape = first_data.shape
    except Exception:
        return None

    if method != 'median':
        # Streaming mean — O(1) memory per frame
        acc = np.zeros(shape, dtype=np.float64)
        count = 0
        for f in frames:
            try:
                data, _ = load_fits(f.path)
                acc += data.astype(np.float64)
                count += 1
            except Exception:
                continue
        if count == 0:
            return None
        return (acc / count).astype(np.float32)

    # Median — use memmap for large datasets to avoid OOM
    n = len(frames)
    estimated_bytes = n * int(np.prod(shape)) * 4
    if estimated_bytes > 500_000_000:  # > 500 MB → memmap
        mm_path = os.path.join(tempfile.gettempdir(), f'master_{os.getpid()}.dat')
        mem = np.memmap(mm_path, dtype='float32', mode='w+', shape=(n, *shape))
        count = 0
        for i, f in enumerate(frames):
            try:
                data, _ = load_fits(f.path)
                mem[count] = data.astype(np.float32)
                count += 1
            except Exception:
                continue
        if count == 0:
            del mem
            try:
                os.remove(mm_path)
            except Exception:
                pass
            return None
        result = np.median(mem[:count], axis=0).astype(np.float32)
        del mem
        try:
            os.remove(mm_path)
        except Exception:
            pass
        return result
    else:
        # Small enough for in-memory
        imgs = []
        for f in frames:
            try:
                data, _ = load_fits(f.path)
                imgs.append(data.astype(np.float32))
            except Exception:
                continue
        if not imgs:
            return None
        return np.median(np.stack(imgs, axis=0), axis=0).astype(np.float32)


def debayer_bilinear(raw, pattern: str = 'RGGB', method: str = 'bilinear'):
    gpu = get_gpu()
    xp = gpu.xp
    raw = gpu.to_device(raw)
    H, W = raw.shape
    pat = pattern.upper()
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


def debayer_vng(raw, pattern: str = 'RGGB'):
    """VNG (Variable Number of Gradients) debayering via OpenCV."""
    if not HAS_CV2:
        return debayer_malvar(raw, pattern)
    pat_map = {
        'RGGB': cv2.COLOR_BAYER_RG2RGB_VNG,
        'BGGR': cv2.COLOR_BAYER_BG2RGB_VNG,
        'GRBG': cv2.COLOR_BAYER_GR2RGB_VNG,
        'GBRG': cv2.COLOR_BAYER_GB2RGB_VNG,
    }
    code = pat_map.get(pattern.upper())
    if code is None:
        return debayer_malvar(raw, pattern)
    raw_np = np.asarray(raw, dtype=np.float32)
    max_val = raw_np.max()
    if max_val <= 0:
        return np.zeros((*raw_np.shape, 3), dtype=np.float32)
    raw_u16 = np.clip(raw_np / max_val * 65535, 0, 65535).astype(np.uint16)
    rgb = cv2.cvtColor(raw_u16, code)
    return (rgb.astype(np.float32) / 65535.0 * max_val)


def debayer(raw, pattern: str = 'RGGB', method: str = 'bilinear'):
    """Dispatch to the appropriate debayering method."""
    if method == 'vng':
        return debayer_vng(raw, pattern)
    elif method == 'malvar':
        return debayer_malvar(raw, pattern)
    else:
        return debayer_bilinear(raw, pattern, method)


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
    # threshold is a sigma multiplier (e.g. 12.0 = 12-sigma detection)
    mask = diff > threshold * sigma
    if not bool(xp.any(mask)):
        return img
    img_fixed = xp.array(img, copy=True)
    img_fixed[mask] = med[mask]
    return img_fixed


def background_gradient_subtract(img):
    gpu = get_gpu()
    blurred = gpu.xndimage.gaussian_filter(img, sigma=max(15, min(img.shape) // 20))
    return img - blurred


# ---------------------------------------------------------------------------
# New utility functions
# ---------------------------------------------------------------------------

def remove_hot_pixels_rgb(rgb, threshold: float = None):
    """Detect hot pixels on luminance (1 filter pass), fix all 3 channels."""
    gpu = get_gpu()
    xp = gpu.xp
    if threshold is None:
        threshold = Config.HOT_PIXEL_THRESHOLD
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    med_lum = gpu.xndimage.median_filter(lum, size=3)
    diff = lum - med_lum
    sigma = float(xp.std(diff))
    # threshold is a sigma multiplier (e.g. 12.0 = 12-sigma detection)
    mask = diff > threshold * sigma
    if not bool(xp.any(mask)):
        return rgb
    result = xp.array(rgb, copy=True)
    for c in range(rgb.shape[2]):
        ch_med = gpu.xndimage.median_filter(rgb[:, :, c], size=3)
        result[:, :, c][mask] = ch_med[mask]
    return result


def generate_star_mask(shape: Tuple[int, int], star_positions, fwhm: float = 3.0) -> np.ndarray:
    """Generate a float mask with Gaussian PSFs at detected star positions."""
    mask = np.zeros(shape, dtype=np.float32)
    if star_positions is None or len(star_positions) == 0:
        return mask
    sigma = fwhm / 2.355
    radius = int(3 * sigma) + 1
    H, W = shape
    n_stars = min(len(star_positions), Config.STAR_MASK_MAX_STARS)
    for i in range(n_stars):
        star = star_positions[i]
        y = int(round(float(star['ycentroid'])))
        x = int(round(float(star['xcentroid'])))
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        gaussian = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], gaussian)
    return mask


def measure_fwhm(img: np.ndarray, star_positions, cutout_radius: int = None) -> float:
    """Measure median FWHM from star cutouts using half-max area method."""
    if cutout_radius is None:
        cutout_radius = Config.FWHM_CUTOUT_RADIUS
    if star_positions is None or len(star_positions) == 0:
        return 0.0
    H, W = img.shape
    fwhms = []
    n_stars = min(len(star_positions), Config.FWHM_MAX_STARS)
    # Sort by flux (brightest first) for more reliable measurements
    try:
        sorted_idx = np.argsort(star_positions['flux'])[::-1]
    except (KeyError, TypeError):
        sorted_idx = range(n_stars)
    for idx in sorted_idx[:n_stars]:
        star = star_positions[idx]
        y = int(round(float(star['ycentroid'])))
        x = int(round(float(star['xcentroid'])))
        if (y < cutout_radius or y >= H - cutout_radius or
                x < cutout_radius or x >= W - cutout_radius):
            continue
        cutout = img[y - cutout_radius:y + cutout_radius + 1,
                     x - cutout_radius:x + cutout_radius + 1].astype(np.float64)
        peak = np.max(cutout)
        bg = np.percentile(cutout, 25)
        # Skip if star has very low contrast relative to noise
        if (peak - bg) < np.std(cutout) * 2.0:
            continue
        half_max = (peak + bg) / 2.0
        above_half = np.sum(cutout > half_max)
        fwhm_est = 2.0 * np.sqrt(above_half / np.pi)
        if 0.5 <= fwhm_est < cutout_radius * 2.5:
            fwhms.append(fwhm_est)
    return float(np.median(fwhms)) if fwhms else 0.0


def wavelet_denoise(img: np.ndarray, wavelet: str = 'bior1.3',
                    levels: int = 4, threshold_factor: float = 3.0) -> np.ndarray:
    """Multi-scale wavelet denoising (BayesShrink) per channel."""
    if not HAS_PYWT:
        logging.warning("pywt not installed, skipping wavelet denoise")
        return img
    result = np.empty_like(img)
    h, w = img.shape[0], img.shape[1]

    def _denoise_channel(c):
        channel = img[:, :, c].astype(np.float64)
        max_level = pywt.dwt_max_level(min(channel.shape), pywt.Wavelet(wavelet).dec_len)
        use_levels = min(levels, max_level)
        if use_levels < 1:
            return c, channel
        coeffs = pywt.wavedec2(channel, wavelet, level=use_levels)
        detail_hh = coeffs[-1][-1]
        sigma_noise = np.median(np.abs(detail_hh)) / 0.6745
        threshold = threshold_factor * sigma_noise
        new_coeffs = [coeffs[0]]
        for detail_level in coeffs[1:]:
            new_coeffs.append(tuple(
                pywt.threshold(d, threshold, mode='soft') for d in detail_level
            ))
        reconstructed = pywt.waverec2(new_coeffs, wavelet)
        return c, reconstructed[:h, :w]

    with ThreadPoolExecutor(max_workers=img.shape[2]) as executor:
        for c, ch_result in executor.map(_denoise_channel, range(img.shape[2])):
            result[:, :, c] = ch_result
    return np.clip(result, 0, None).astype(np.float32)


def local_normalize(img: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """Local normalization to remove flat-field residuals and vignetting."""
    result = np.empty_like(img)

    def _normalize_channel(c):
        channel = img[:, :, c].astype(np.float64)
        local_mean = ndimage.gaussian_filter(channel, sigma=sigma)
        local_sq_mean = ndimage.gaussian_filter(channel ** 2, sigma=sigma)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
        return c, (channel - local_mean) / (local_std + 1e-12)

    with ThreadPoolExecutor(max_workers=img.shape[2]) as executor:
        for c, ch_result in executor.map(_normalize_channel, range(img.shape[2])):
            result[:, :, c] = ch_result
    # Re-scale to original data range (positive values)
    result = result - result.min()
    orig_max = np.max(img)
    if result.max() > 0 and orig_max > 0:
        result = result / result.max() * orig_max
    return result.astype(np.float32)


def arcsinh_stretch(img: np.ndarray, factor: float = None) -> np.ndarray:
    """Non-linear arcsinh stretch, ideal for astrophotography previews."""
    if factor is None:
        factor = Config.ARCSINH_STRETCH_FACTOR
    lo = np.percentile(img, 0.5)
    hi = np.percentile(img, 99.5)
    norm = np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)
    stretched = np.arcsinh(norm * factor) / np.arcsinh(factor)
    return np.clip(stretched, 0, 1)


def match_stars_affine(ref_positions, img_positions,
                       initial_shift: Tuple[float, float] = (0.0, 0.0)):
    """Match star catalogs and compute a Euclidean (rotation+translation) transform.

    Uses nearest-neighbor matching after applying an initial translation estimate,
    then RANSAC-robust fitting. Returns None if matching fails.
    """
    if not HAS_SKIMAGE_TRANSFORM:
        return None
    from scipy.spatial import cKDTree

    max_stars = Config.AFFINE_MAX_STARS
    if ref_positions is None or img_positions is None:
        return None
    if len(ref_positions) < 3 or len(img_positions) < 3:
        return None

    ref_pts = np.array([(float(s['xcentroid']), float(s['ycentroid']))
                        for s in ref_positions[:max_stars]])
    img_pts = np.array([(float(s['xcentroid']), float(s['ycentroid']))
                        for s in img_positions[:max_stars]])

    # Shift img points by initial estimate for better matching
    img_pts_shifted = img_pts + np.array([initial_shift[1], initial_shift[0]])

    tree = cKDTree(ref_pts)
    distances, indices = tree.query(img_pts_shifted, k=1)

    good = distances < Config.AFFINE_MATCH_RADIUS
    if good.sum() < 3:
        return None

    src = img_pts[good]
    dst = ref_pts[indices[good]]

    try:
        model, inliers = ransac(
            (src, dst), EuclideanTransform,
            min_samples=3, residual_threshold=2.0, max_trials=1000
        )
        if inliers is not None and inliers.sum() >= 3:
            return model
    except Exception:
        pass
    return None


def apply_transform(img: np.ndarray, shift: Tuple[float, float] = None,
                    transform=None) -> np.ndarray:
    """Apply translation or affine transform to a multi-channel image.

    Uses cubic spline interpolation (order=3) for subpixel accuracy.
    With accurate registration (full-resolution FFT cross-correlation),
    subpixel shifts preserve detail better than integer rounding.
    """
    if transform is not None:
        matrix = transform.params
        result = np.zeros_like(img)
        for c in range(img.shape[2]):
            result[:, :, c] = ndimage.affine_transform(
                img[:, :, c], matrix[:2, :2], offset=matrix[:2, 2],
                order=3, mode='constant', cval=0.0
            )
        return result
    elif shift is not None:
        result = np.zeros_like(img)
        for c in range(img.shape[2]):
            result[:, :, c] = ndimage.shift(
                img[:, :, c], shift=shift, order=3,
                mode='constant', cval=0.0, prefilter=True
            )
        return result
    return img


def extract_background(img: np.ndarray, mesh_size: int = 256, filter_size: int = 3,
                       clip_sigma: float = 3.0, clip_iters: int = 5,
                       star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate smooth sky background using mesh-based sigma-clipped statistics.

    Divides image into a grid, computes sigma-clipped median in each cell
    (rejecting stars via both sigma-clipping and optional star mask), rejects
    cells contaminated by extended bright objects (nebulae), then interpolates
    the clean grid to a smooth background model.

    Args:
        img: 2D image array (single channel).
        mesh_size: Size of each grid cell in pixels.
        filter_size: Median filter size applied to the mesh grid.
        clip_sigma: Sigma threshold for iterative clipping within each cell.
        clip_iters: Maximum iterations for sigma clipping.
        star_mask: Optional float mask (0=bg, 1=star) to exclude star pixels.
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

            # Mask out star pixels if star_mask provided
            if star_mask is not None:
                sm = star_mask[y0:y1, x0:x1].ravel()
                bg_pixels = cell[sm < 0.5]
                if bg_pixels.size > 10:
                    cell = bg_pixels

            if cell.size == 0:
                bg_grid[iy, ix] = 0.0
                continue

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
                                verbose: bool = False,
                                star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply intelligent background extraction with star-mask and chromatic gradient support.

    First estimates per-channel backgrounds. If the channel backgrounds are
    spatially correlated (>0.95), uses luminance-based colour-neutral subtraction.
    Otherwise falls back to independent per-channel subtraction to handle
    chromatic light-pollution gradients.
    """
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Estimate per-channel backgrounds in parallel (passing star mask for exclusion)
    bg_channels = [None, None, None]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(extract_background, rgb[:, :, c],
                            mesh_size=mesh_size, filter_size=filter_size,
                            clip_sigma=clip_sigma, star_mask=star_mask): c
            for c in range(3)
        }
        for future in as_completed(futures):
            c = futures[future]
            bg_channels[c] = future.result()

    # Check for chromatic gradient by correlating channel backgrounds
    use_per_channel = False
    if len(bg_channels) == 3:
        rg_flat = bg_channels[0].ravel()
        bg_flat = bg_channels[2].ravel()
        gg_flat = bg_channels[1].ravel()
        corr_rg = np.corrcoef(rg_flat, gg_flat)[0, 1] if np.std(rg_flat) > 1e-6 else 1.0
        corr_rb = np.corrcoef(rg_flat, bg_flat)[0, 1] if np.std(rg_flat) > 1e-6 else 1.0
        if corr_rg < 0.95 or corr_rb < 0.95:
            use_per_channel = True
            if verbose:
                safe_print(f"    Chromatic gradient detected (R-G corr={corr_rg:.3f}, "
                           f"R-B corr={corr_rb:.3f}), using per-channel subtraction")

    result = np.empty_like(rgb)
    channel_names = ['Red', 'Green', 'Blue']

    if use_per_channel:
        # Independent per-channel subtraction for chromatic gradients
        for c in range(3):
            subtracted = rgb[:, :, c] - bg_channels[c]
            np.clip(subtracted, 0, None, out=subtracted)
            result[:, :, c] = subtracted
            if verbose:
                safe_print(f"    {channel_names[c]}: bg_median="
                           f"{float(np.median(bg_channels[c])):.1f}, "
                           f"subtracted median={float(np.median(subtracted)):.1f}")
    else:
        # Colour-neutral: use luminance model scaled per channel
        bg_lum = extract_background(lum, mesh_size=mesh_size,
                                    filter_size=filter_size,
                                    clip_sigma=clip_sigma, star_mask=star_mask)
        lum_bg_median = float(np.median(bg_lum))
        if verbose:
            safe_print(f"    Luminance background: median={lum_bg_median:.1f}, "
                       f"range={float(np.max(bg_lum) - np.min(bg_lum)):.1f}")

        for c in range(3):
            channel = rgb[:, :, c]
            if sigma_clipped_stats is not None:
                try:
                    _, ch_bg_median, _ = sigma_clipped_stats(
                        channel, sigma=clip_sigma, maxiters=5)
                    ch_bg_median = float(ch_bg_median)
                except Exception:
                    ch_bg_median = float(np.median(channel))
            else:
                ch_bg_median = float(np.median(channel))
            scale = ch_bg_median / lum_bg_median if lum_bg_median > 1e-6 else 1.0
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

    sources = None
    if DAOStarFinder is not None and sigma_clipped_stats is not None:
        try:
            bg_mean, bg_median, bg_std = sigma_clipped_stats(img, sigma=3.0)
            # Threshold for background-subtracted image: N * sigma above zero
            threshold = 5.0 * float(bg_std)
            daof = DAOStarFinder(fwhm=3.0, threshold=threshold)
            sources = daof(img - float(bg_median))

            if sources is not None and len(sources) > 0:
                star_count = len(sources)
                # Calculate median star SNR
                star_peaks = sources['peak']
                star_snr = float(np.median(star_peaks)) / (noise + 1e-12)
        except Exception as e:
            logging.debug(f"DAOStarFinder failed: {type(e).__name__}: {e}")
            sources = None

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

    # FWHM measurement from detected stars
    fwhm = 0.0
    if star_count > 0 and sources is not None:
        fwhm = measure_fwhm(img, sources)

    # Composite quality score
    star_factor = min(star_count / 50.0, 1.0) if star_count > 0 else 0.01
    snr_factor = min(snr / 10.0, 1.0) if snr > 0 else 0.01
    # Penalize poor focus (high FWHM) — prefer tighter stars
    fwhm_factor = 1.0
    if fwhm > 0:
        fwhm_factor = max(0.1, 1.0 / (1.0 + max(0, fwhm - 2.0) ** 2 * 0.1))

    score = brightness * contrast * star_factor * snr_factor * fwhm_factor * 100.0

    return {
        'brightness': brightness,
        'mean': mean,
        'contrast': contrast,
        'snr': snr,
        'star_count': star_count,
        'star_snr': star_snr,
        'sharpness': sharpness,
        'fwhm': fwhm,
        'background': background,
        'noise': noise,
        'score': score,
        'p01': p01,
        'p99': p99,
        'dynamic_range': p99 - p01,
        '_star_sources': sources,
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
    
    # FFT cross-correlation at full resolution with zero-padding.
    # No downscaling or windowing — these cause multi-pixel registration
    # errors that broaden stars in the stacked result.
    try:
        ref_norm = (ref - np.mean(ref)).astype(np.float64)
        img_norm = (img - np.mean(img)).astype(np.float64)

        h, w = ref_norm.shape
        pad_h, pad_w = 2 * h, 2 * w
        F_ref = np.fft.rfft2(ref_norm, s=(pad_h, pad_w))
        F_img = np.fft.rfft2(img_norm, s=(pad_h, pad_w))
        corr = np.fft.irfft2(F_ref * np.conj(F_img), s=(pad_h, pad_w))

        peak = np.unravel_index(np.argmax(corr), corr.shape)
        dy = peak[0] if peak[0] < h else peak[0] - pad_h
        dx = peak[1] if peak[1] < w else peak[1] - pad_w

        # Parabolic subpixel refinement (wrap-safe)
        py, px = peak
        sub_y = sub_x = 0.0
        vc = corr[py, px]
        vm = corr[(py - 1) % pad_h, px]
        vp = corr[(py + 1) % pad_h, px]
        denom = 2.0 * (2.0 * vc - vm - vp)
        if abs(denom) > 1e-12:
            sub_y = max(-0.5, min(0.5, (vm - vp) / denom))
        vm = corr[py, (px - 1) % pad_w]
        vp = corr[py, (px + 1) % pad_w]
        denom = 2.0 * (2.0 * vc - vm - vp)
        if abs(denom) > 1e-12:
            sub_x = max(-0.5, min(0.5, (vm - vp) / denom))

        shift_y = float(dy + sub_y)
        shift_x = float(dx + sub_x)

        if np.isfinite(shift_y) and np.isfinite(shift_x) and max(abs(shift_y), abs(shift_x)) < max(h, w) * 0.5:
            if verbose:
                print(f"      [fft_xcorr: shift=({shift_x:.1f}, {shift_y:.1f})]")
            return shift_y, shift_x
        else:
            debug_info.append(f"fft_xcorr rejected: shift ({shift_y:.1f}, {shift_x:.1f}) too large")
    except Exception as e:
        debug_info.append(f"fft_xcorr error: {type(e).__name__}")
    
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


def _sigma_clip_tile(tile: np.ndarray, sigma: float, max_iters: int,
                     weights: Optional[np.ndarray], winsorize: bool) -> np.ndarray:
    """Process a single spatial tile for sigma-clip combine.

    Uses MAD (Median Absolute Deviation) for robust spread estimation
    instead of standard deviation, which is less sensitive to the very
    outliers we are trying to reject.
    """
    N = tile.shape[0]
    mask = np.ones(tile.shape, dtype=bool)

    for iteration in range(max_iters):
        masked = np.where(mask, tile, np.nan)
        with np.errstate(all='ignore'):
            median = np.nanmedian(masked, axis=0)
            # MAD * 1.4826 is a consistent estimator of std for normal data
            mad = np.nanmedian(np.abs(masked - median[np.newaxis]), axis=0) * 1.4826

        # Fallback to std where MAD is zero (constant regions)
        spread = mad.copy()
        zero_mad = spread < 1e-12
        if np.any(zero_mad):
            with np.errstate(all='ignore'):
                std_fallback = np.nanstd(masked, axis=0)
            spread[zero_mad] = std_fallback[zero_mad]

        deviation = np.abs(tile - median[np.newaxis])
        new_mask = mask & (deviation <= sigma * spread[np.newaxis])

        # Ensure at least 1 frame survives at every pixel
        surviving = new_mask.sum(axis=0)
        all_rejected = surviving == 0
        if np.any(all_rejected):
            for frame_idx in range(N):
                new_mask[frame_idx][all_rejected] = mask[frame_idx][all_rejected]

        rejected = int(mask.sum() - new_mask.sum())
        mask = new_mask
        if rejected == 0:
            break

    if winsorize:
        # Replace outliers with clip boundaries instead of masking to NaN
        masked_final = np.where(mask, tile, np.nan)
        with np.errstate(all='ignore'):
            med_final = np.nanmedian(masked_final, axis=0)
            mad_final = np.nanmedian(
                np.abs(masked_final - med_final[np.newaxis]), axis=0) * 1.4826
        mad_final = np.maximum(mad_final, 1e-12)
        upper = med_final + sigma * mad_final
        lower = med_final - sigma * mad_final
        clipped = np.clip(tile, lower[np.newaxis], upper[np.newaxis])
        if weights is not None:
            w = weights[:, np.newaxis, np.newaxis, np.newaxis]
            return (np.sum(clipped * w, axis=0) / np.sum(w)).astype(np.float32)
        return np.mean(clipped, axis=0).astype(np.float32)
    else:
        masked_final = np.where(mask, tile, np.nan)
        if weights is not None:
            w = np.where(mask, weights[:, np.newaxis, np.newaxis, np.newaxis], 0.0)
            with np.errstate(all='ignore'):
                total_w = np.sum(w, axis=0)
                total_w[total_w == 0] = 1.0
                result = np.nansum(masked_final * w, axis=0) / total_w
            np.nan_to_num(result, copy=False, nan=0.0)
            return result.astype(np.float32)
        with np.errstate(all='ignore'):
            result = np.nanmean(masked_final, axis=0)
        np.nan_to_num(result, copy=False, nan=0.0)
        return result.astype(np.float32)


def sigma_clip_combine(data: np.ndarray, sigma: float = 3.0, max_iters: int = 3,
                       weights: Optional[np.ndarray] = None,
                       winsorize: bool = False,
                       verbose: bool = False) -> np.ndarray:
    """Combine frames using tiled, MAD-based, optionally winsorized sigma-clip.

    Processes the image in spatial tiles to keep peak memory low.  Uses
    MAD (Median Absolute Deviation) instead of standard deviation for more
    robust outlier detection.  Optionally supports quality-weighted
    combination and winsorized clipping.

    Args:
        data: Array of shape ``(N, H, W, C)`` (all aligned frames).
        sigma: Rejection threshold in MADs.
        max_iters: Maximum clipping iterations.
        weights: Optional 1-D array of length N with per-frame quality weights.
        winsorize: If True, clip outliers to boundary instead of rejecting.
        verbose: Print per-tile progress.
    """
    N, H, W, C = data.shape
    tile_size = Config.TILE_SIZE
    result = np.zeros((H, W, C), dtype=np.float32)
    total_rejected = 0
    total_pixels = 0

    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles_x = (W + tile_size - 1) // tile_size

    # Build list of tile coordinates, then process in parallel
    tile_coords = []
    for ty_idx in range(n_tiles_y):
        ty = ty_idx * tile_size
        ty_end = min(ty + tile_size, H)
        for tx_idx in range(n_tiles_x):
            tx = tx_idx * tile_size
            tx_end = min(tx + tile_size, W)
            tile_coords.append((ty, ty_end, tx, tx_end))

    def _process_tile(coords):
        ty, ty_end, tx, tx_end = coords
        tile = np.array(data[:, ty:ty_end, tx:tx_end, :], dtype=np.float32)
        return coords, _sigma_clip_tile(tile, sigma, max_iters, weights, winsorize)

    n_tile_workers = min(os.cpu_count() or 4, len(tile_coords))
    with ThreadPoolExecutor(max_workers=n_tile_workers) as executor:
        for coords, tile_result in executor.map(_process_tile, tile_coords):
            ty, ty_end, tx, tx_end = coords
            result[ty:ty_end, tx:tx_end, :] = tile_result

    if verbose:
        safe_print(f"    Tiled sigma-clip: {n_tiles_y * n_tiles_x} tiles of "
                   f"{tile_size}x{tile_size}, mode={'winsorized' if winsorize else 'reject'}")
    return result


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


def save_preview_rgb(rgb: np.ndarray, path: str, stretch: str = 'linear'):
    if Image is None:
        return
    if stretch == 'arcsinh':
        # Arcsinh stretch — preserves faint nebulosity and bright stars
        out = np.zeros_like(rgb)
        for c in range(3):
            out[:, :, c] = arcsinh_stretch(rgb[:, :, c])
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
    else:
        # Linear percentile stretch (original behaviour)
        if exposure is None:
            return
        out = np.zeros_like(rgb)
        for c in range(3):
            lo, hi = np.percentile(rgb[:, :, c], Config.PREVIEW_STRETCH_PERCENTILES)
            out[:, :, c] = exposure.rescale_intensity(rgb[:, :, c], in_range=(lo, hi))
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path, quality=Config.PREVIEW_JPEG_QUALITY)


def populate_fits_header(header: fits.Header, frames: List[FrameInfo], stats: ProcessingStats, args: argparse.Namespace, stacked_shape: Tuple[int, int, int], shifts: List[Tuple[float, float]], masters: Dict[str, Optional[np.ndarray]], dither_info: Optional[Dict] = None) -> None:
    """Populate FITS header with comprehensive metadata."""
    from datetime import datetime, timezone

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
    header['DATE'] = (datetime.now(timezone.utc).isoformat(), 'UTC date/time of file creation')

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

    # Sigma-clip details
    if args.stack_method == 'sigma_clip':
        header['WINSORIZ'] = (getattr(args, 'winsorize', False), 'Winsorized sigma-clip used')

    # Affine registration
    header['AFFINE'] = (getattr(args, 'affine', False), 'Affine registration enabled')

    # Post-processing flags
    header['DENOISE'] = (getattr(args, 'denoise', False), 'Wavelet denoising applied')
    if getattr(args, 'denoise', False):
        header['DNSTRNG'] = (getattr(args, 'denoise_strength', 3.0), 'Denoise threshold factor')
    header['LOCNORM'] = (getattr(args, 'local_normalize', False), 'Local normalization applied')
    header['STRETCH'] = (getattr(args, 'stretch', 'linear'), 'Preview stretch method')
    header['DEBAYER'] = (args.debayer_method, 'Debayering method used')

    # Add quality metrics including FWHM
    if frames and frames[0].metrics:
        frames_with_metrics = [f for f in frames if f.metrics and 'score' in f.metrics]
        if frames_with_metrics:
            header['AVGBRITE'] = (float(np.mean([f.metrics.get('brightness', 0) for f in frames_with_metrics])),
                                  'Average frame brightness')
            header['AVGCONTR'] = (float(np.mean([f.metrics.get('contrast', 0) for f in frames_with_metrics])),
                                  'Average frame contrast')
            header['AVGSCORE'] = (float(np.mean([f.metrics.get('score', 0) for f in frames_with_metrics])),
                                  'Average quality score')
            # FWHM statistics
            fwhms = [f.metrics.get('fwhm', 0) for f in frames_with_metrics if f.metrics.get('fwhm', 0) > 0]
            if fwhms:
                header['AVGFWHM'] = (round(float(np.mean(fwhms)), 2), 'Average star FWHM in pixels')
                header['MINFWHM'] = (round(float(np.min(fwhms)), 2), 'Minimum star FWHM in pixels')
                header['MAXFWHM'] = (round(float(np.max(fwhms)), 2), 'Maximum star FWHM in pixels')


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


def _process_single_frame(path: str, header: dict, masters: Dict[str, Optional[np.ndarray]],
                          debayer_method: str, white_balance: str) -> Dict:
    """Process one frame: load, calibrate, debayer, hot-pixel, quality.

    Returns dict with keys: 'rgb', 'lum', 'metrics', 'error'.
    Used by both sequential and parallel paths.
    """
    try:
        data, hdr = load_fits(path)
    except Exception as e:
        return {'error': f'load error: {e}'}
    if data is None or data.size == 0:
        return {'error': 'empty data array'}

    # Calibration — preserve negative noise through bias/dark subtraction,
    # clip only once after all steps to avoid cumulative truncation of shadow detail
    try:
        if masters.get('bias') is not None and masters['bias'].shape == data.shape:
            data = data - masters['bias']
        if masters.get('dark') is not None and masters['dark'].shape == data.shape:
            data = data - masters['dark']
        if masters.get('flat') is not None and masters['flat'].shape == data.shape:
            flat = masters['flat']
            med = np.median(flat)
            if med > 1e-6:
                flat_norm = np.clip(flat / med, 0.4, 2.5)
                data = data / flat_norm
        if not np.isfinite(data).all():
            return {'error': 'calibration produced non-finite values'}
        data = np.clip(data, 0, None)
    except Exception as e:
        return {'error': f'calibration error: {e}'}

    # Debayer
    try:
        if data.ndim == 2:
            bayer = hdr.get('BAYERPAT', hdr.get('COLORTYP', 'RGGB'))
            rgb = debayer(data, pattern=bayer, method=debayer_method)
        else:
            rgb = data
    except Exception as e:
        return {'error': f'debayering error: {e}'}

    # Hot pixel removal (single-channel detection, 3x faster)
    try:
        if rgb.ndim != 3 or rgb.shape[2] < 1:
            return {'error': f'Invalid RGB shape: {rgb.shape}'}
        rgb = remove_hot_pixels_rgb(rgb)
    except Exception as e:
        return {'error': f'hot pixel removal error: {e}'}

    # White balance
    if white_balance == 'grayworld':
        rgb = white_balance_grayworld(rgb)
    elif white_balance == 'whitepatch':
        rgb = white_balance_whitepatch(rgb)

    # Ensure arrays are on CPU (GPU/CuPy paths return device arrays)
    gpu = get_gpu()
    rgb = gpu.to_host(rgb)

    # Compute luminance & quality
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    is_valid, validation_error = validate_image_data(lum, os.path.basename(path))
    if not is_valid:
        return {'error': f'validation failed: {validation_error}'}

    metrics = compute_quality_metrics(lum)
    return {'rgb': rgb, 'lum': lum, 'metrics': metrics, 'error': None}


# Module-level state for parallel workers
_worker_masters: Dict[str, Optional[np.ndarray]] = {}


def _init_worker(master_paths: Dict[str, str]):
    """Initializer for pool workers — load master calibration arrays from disk."""
    global _worker_masters
    _worker_masters = {}
    for name, p in master_paths.items():
        _worker_masters[name] = np.load(p)


def _parallel_frame_worker(args_tuple):
    """Worker function for ProcessPoolExecutor. Must be module-level for pickling."""
    path, frame_idx, debayer_method, white_balance, mm_rgb_path, mm_lum_path, rgb_shape, lum_shape = args_tuple
    global _worker_masters
    result = _process_single_frame(path, {}, _worker_masters, debayer_method, white_balance)
    if result.get('error'):
        return (frame_idx, None, result['error'])

    rgb = result['rgb']
    lum = result['lum']
    metrics = result['metrics']
    # Remove non-picklable star sources from metrics for IPC
    metrics_clean = {k: v for k, v in metrics.items() if k != '_star_sources'}

    # Write processed data to shared memmap
    try:
        mem_rgb = np.memmap(mm_rgb_path, dtype='float32', mode='r+', shape=rgb_shape)
        mem_lum = np.memmap(mm_lum_path, dtype='float32', mode='r+', shape=lum_shape)
        mem_rgb[frame_idx] = rgb
        mem_lum[frame_idx] = lum
        mem_rgb.flush()
        mem_lum.flush()
        del mem_rgb, mem_lum
    except Exception as e:
        return (frame_idx, None, f'memmap write error: {e}')

    return (frame_idx, metrics_clean, None)


def stack_target(frames: List[FrameInfo], output_path: str, args: argparse.Namespace,
                 masters: Dict[str, Optional[np.ndarray]], stats: ProcessingStats):
    lights = [f for f in frames if f.type == 'light']
    if not lights:
        print('  No light frames found for target')
        return None

    stats.total_frames = len(lights)
    n = len(lights)

    # ======================================================================
    # PHASE 1: Process & Analyse (fused — each frame loaded only ONCE)
    # ======================================================================
    print_phase(1, "Processing & Quality Analysis")
    phase_start = time.time()

    # Probe first frame for dimensions
    first_data, first_hdr = load_fits(lights[0].path)
    if first_data.ndim == 2:
        first_rgb = debayer(first_data,
                            pattern=first_hdr.get('BAYERPAT', first_hdr.get('COLORTYP', 'RGGB')),
                            method=args.debayer_method)
        H_rgb, W_rgb, C = first_rgb.shape
    else:
        H_rgb, W_rgb = first_data.shape[:2]
        C = first_data.shape[2] if first_data.ndim == 3 else 1
    del first_data

    # Create memmaps for ALL frames (unaligned, white-balanced, calibrated)
    mm_rgb_path = os.path.join(tempfile.gettempdir(), f'stack_rgb_{os.getpid()}.dat')
    mm_lum_path = os.path.join(tempfile.gettempdir(), f'stack_lum_{os.getpid()}.dat')
    rgb_shape = (n, H_rgb, W_rgb, C)
    lum_shape = (n, H_rgb, W_rgb)
    mem_rgb = np.memmap(mm_rgb_path, dtype='float32', mode='w+', shape=rgb_shape)
    mem_lum = np.memmap(mm_lum_path, dtype='float32', mode='w+', shape=lum_shape)

    rejected_reasons = {}
    use_parallel = (getattr(args, 'parallel', 1) != 1
                    and not get_gpu().active
                    and n >= 4)

    if use_parallel:
        # --- Parallel path: save masters to disk, use pool ---
        workers = args.parallel if args.parallel > 0 else min(os.cpu_count() or 4, n, 8)
        print(f"  Processing {n} frames in parallel ({workers} workers)...")

        master_paths = {}
        for name, arr in masters.items():
            if arr is not None:
                p = os.path.join(tempfile.gettempdir(), f'master_{name}_{os.getpid()}.npy')
                np.save(p, arr)
                master_paths[name] = p

        tasks = [(lights[i].path, i, args.debayer_method, args.white_balance,
                  mm_rgb_path, mm_lum_path, rgb_shape, lum_shape)
                 for i in range(n)]

        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker,
                                 initargs=(master_paths,)) as pool:
            futures = {pool.submit(_parallel_frame_worker, t): t[1] for t in tasks}
            done_iter = tqdm(as_completed(futures), total=n,
                             desc="  Processing", unit="frame",
                             disable=args.verbose)
            for future in done_iter:
                idx = futures[future]
                frame_idx, metrics, error = future.result()
                f = lights[frame_idx]
                if error:
                    f.accepted = False
                    f.metrics = {'error': error}
                    rejected_reasons[f.path] = error
                    stats.add_error(f.path, error)
                    if args.verbose:
                        print(f'  REJECT {os.path.basename(f.path)}: {error}')
                else:
                    f.metrics = metrics

        # Cleanup master temp files
        for p in master_paths.values():
            try:
                os.remove(p)
            except Exception:
                pass
    else:
        # --- Sequential path ---
        print(f"  Processing {n} frames sequentially...")
        frame_iter = tqdm(enumerate(lights), total=n,
                          desc="  Processing", unit="frame",
                          disable=args.verbose)
        for i, f in frame_iter:
            result = _process_single_frame(
                f.path, f.header, masters, args.debayer_method, args.white_balance)
            if result.get('error'):
                f.accepted = False
                f.metrics = {'error': result['error']}
                rejected_reasons[f.path] = result['error']
                stats.add_error(f.path, result['error'])
                if args.verbose:
                    print(f'  REJECT {os.path.basename(f.path)}: {result["error"]}')
            else:
                mem_rgb[i] = result['rgb']
                mem_lum[i] = result['lum']
                f.metrics = result['metrics']
                if args.verbose:
                    m = f.metrics
                    print(f'    {os.path.basename(f.path)}: SNR={m["snr"]:.1f}, '
                          f'stars={m["star_count"]}, FWHM={m.get("fwhm",0):.1f}, '
                          f'score={m["score"]:.1f}')
        mem_rgb.flush()
        mem_lum.flush()

    # --- Quality gating ---
    accepted = []
    for f in lights:
        if not f.accepted or not f.metrics or 'score' not in f.metrics:
            continue
        metrics = f.metrics
        reject_reason = None
        if metrics['star_count'] < 3:
            reject_reason = f"insufficient stars ({metrics['star_count']} < 3)"
        elif metrics['snr'] < 0.5:
            reject_reason = f"extremely low SNR ({metrics['snr']:.2f} < 0.5)"
        elif metrics['contrast'] < 2.0:
            reject_reason = f"extremely low contrast ({metrics['contrast']:.1f} < 2.0)"
        elif metrics['dynamic_range'] < 20:
            reject_reason = f"extremely low dynamic range ({metrics['dynamic_range']:.1f} < 20)"
        elif metrics['noise'] > metrics['brightness'] * 0.8:
            reject_reason = f"excessive noise ({metrics['noise']:.1f} > {metrics['brightness']*0.8:.1f})"
        if reject_reason:
            f.accepted = False
            rejected_reasons[f.path] = reject_reason
            if args.verbose:
                print(f'  REJECT {os.path.basename(f.path)}: {reject_reason}')
        else:
            accepted.append(f)

    # Statistical outlier detection
    if len(accepted) > 3:
        snrs = np.array([f.metrics['snr'] for f in accepted])
        star_counts = np.array([f.metrics['star_count'] for f in accepted])
        contrasts = np.array([f.metrics['contrast'] for f in accepted])

        def reject_outliers(values, threshold=2.5):
            if len(values) < 3:
                return np.ones(len(values), dtype=bool)
            m, s = np.mean(values), np.std(values)
            if s < 1e-6:
                return np.ones(len(values), dtype=bool)
            return np.abs((values - m) / s) < threshold

        snr_ok = reject_outliers(snrs)
        star_ok = reject_outliers(star_counts)
        contrast_ok = reject_outliers(contrasts)
        outlier_count = (~snr_ok).astype(int) + (~star_ok).astype(int) + (~contrast_ok).astype(int)
        for i, f in enumerate(accepted):
            if outlier_count[i] >= 2:
                parts = []
                if not snr_ok[i]:
                    parts.append(f"SNR={f.metrics['snr']:.1f}")
                if not star_ok[i]:
                    parts.append(f"stars={f.metrics['star_count']}")
                if not contrast_ok[i]:
                    parts.append(f"contrast={f.metrics['contrast']:.1f}")
                reason = "statistical outlier: " + ", ".join(parts)
                rejected_reasons[f.path] = reason
                f.accepted = False
            else:
                f.accepted = True

    if args.quality_filter and accepted:
        valid = [f for f in accepted if f.accepted]
        if valid:
            scores = np.array([f.metrics['score'] for f in valid])
            pct = np.percentile(scores, args.quality_threshold)
            for f in valid:
                if f.metrics['score'] < pct:
                    f.accepted = False
                    rejected_reasons[f.path] = f'score {f.metrics["score"]:.1f} < {pct:.1f}'

    final = [f for f in lights if f.accepted]
    # Build index map: for each final frame, its original index in `lights`
    final_indices = [lights.index(f) for f in final]
    stats.accepted_frames = len(final)
    stats.rejected_frames = n - len(final)
    stats.quality_time = time.time() - phase_start

    if args.verbose:
        print_quality_table(lights, show_all=len(lights) <= 50)
    safe_print(f"  ✓ Accepted: {len(final)}/{n} ({len(final)/n*100:.1f}%)")
    if stats.rejected_frames > 0:
        reason_counts = {}
        for reason in rejected_reasons.values():
            if 'brightness' in reason or 'contrast' in reason:
                cat = 'Poor quality'
            elif 'star' in reason:
                cat = 'No stars detected'
            elif 'load' in reason or 'empty' in reason:
                cat = 'Load/data errors'
            else:
                cat = 'Other'
            reason_counts[cat] = reason_counts.get(cat, 0) + 1
        safe_print(f"  ✗ Rejected: {stats.rejected_frames} "
                   f"({', '.join(f'{c}: {n}' for c, n in reason_counts.items())})")

    if not final:
        print(f'\n  ERROR: All {n} frames rejected!')
        if rejected_reasons:
            print('  Rejection reasons:')
            for path, reason in list(rejected_reasons.items())[:10]:
                print(f'    - {os.path.basename(path)}: {reason}')
        # Cleanup memmaps
        del mem_rgb, mem_lum
        for p in (mm_rgb_path, mm_lum_path):
            try:
                os.remove(p)
            except Exception:
                pass
        return None

    # ======================================================================
    # PHASE 2: Registration (reads from stored memmaps — no re-load)
    # ======================================================================
    print_phase(2, "Registration")
    phase_start = time.time()

    best = max(final, key=lambda x: x.metrics.get('score', 0))
    best_idx = lights.index(best)
    print(f"  Reference frame: {os.path.basename(best.path)} "
          f"(score={best.metrics.get('score', 0):.1f})")

    ref_lum = np.array(mem_lum[best_idx])  # copy from memmap
    H, W = ref_lum.shape
    ref_lum_std = np.std(ref_lum)

    if args.verbose:
        print(f'  Reference luminance: min={np.min(ref_lum):.1f}, max={np.max(ref_lum):.1f}, '
              f'mean={np.mean(ref_lum):.1f}, std={ref_lum_std:.1f}')

    # Get reference star positions for affine registration
    ref_stars = best.metrics.get('_star_sources')

    shifts = [None] * len(final)
    transforms = [None] * len(final)  # For affine mode
    print(f"  Calculating shifts for {len(final)} frames...")

    def _register_one_frame(j, f, orig_idx):
        """Compute registration for a single frame (thread-safe, reads only)."""
        if orig_idx == best_idx or args.no_registration:
            return j, (0.0, 0.0), None

        lum = np.array(mem_lum[orig_idx])

        # Try affine registration first if enabled
        affine_tf = None
        if getattr(args, 'affine', False) and HAS_SKIMAGE_TRANSFORM:
            img_stars = f.metrics.get('_star_sources')
            sy, sx = calculate_shift(ref_lum, lum, verbose=False,
                                     skip_phase_cc=args.skip_phase_correlation)
            affine_tf = match_stars_affine(ref_stars, img_stars,
                                           initial_shift=(sy, sx))
            if affine_tf is not None:
                tx = affine_tf.params[0, 2]
                ty = affine_tf.params[1, 2]
                return j, (ty, tx), affine_tf

        # Translation-only registration
        sy, sx = calculate_shift(
            ref_lum, lum, verbose=args.verbose,
            debug=args.debug_registration,
            frame_name=os.path.splitext(os.path.basename(f.path))[0],
            skip_phase_cc=args.skip_phase_correlation)
        if abs(sx) > 0.1 * W or abs(sy) > 0.1 * H:
            sx, sy = 0.0, 0.0
        return j, (sy, sx), None

    n_reg_workers = min(os.cpu_count() or 4, len(final))
    with ThreadPoolExecutor(max_workers=n_reg_workers) as executor:
        futures = {
            executor.submit(_register_one_frame, j, f, orig_idx): j
            for j, (f, orig_idx) in enumerate(zip(final, final_indices))
        }
        for future in tqdm(as_completed(futures), total=len(final),
                           desc="  Registering", unit="frame",
                           disable=args.verbose):
            j, shift_val, transform_val = future.result()
            shifts[j] = shift_val
            transforms[j] = transform_val
            final[j].shift = shift_val
            if args.verbose:
                f = final[j]
                if transform_val is not None:
                    tx, ty = shift_val[1], shift_val[0]
                    rot_deg = np.degrees(np.arctan2(transform_val.params[1, 0],
                                                    transform_val.params[0, 0]))
                    print(f'    {os.path.basename(f.path)}: affine shift=({tx:+.1f}, '
                          f'{ty:+.1f}) px, rotation={rot_deg:+.3f} deg')
                elif shift_val != (0.0, 0.0):
                    sy, sx = shift_val
                    mag = np.sqrt(sy**2 + sx**2)
                    print(f'    {os.path.basename(f.path)}: shift=({sx:+.1f}, {sy:+.1f}) px, '
                          f'magnitude={mag:.2f} px')

    stats.registration_time = time.time() - phase_start

    # Shift statistics
    shift_x = [s[1] for s in shifts]
    shift_y = [s[0] for s in shifts]
    shift_mags = [np.sqrt(sx**2 + sy**2) for sx, sy in shifts]
    if not args.no_registration:
        print(f"  Shift statistics:")
        print(f"    X: mean={np.mean(shift_x):+.1f}px, std={np.std(shift_x):.1f}px, "
              f"range=[{np.min(shift_x):+.1f}, {np.max(shift_x):+.1f}]")
        print(f"    Y: mean={np.mean(shift_y):+.1f}px, std={np.std(shift_y):.1f}px, "
              f"range=[{np.min(shift_y):+.1f}, {np.max(shift_y):+.1f}]")
        print(f"    Magnitude: mean={np.mean(shift_mags):.1f}px, max={np.max(shift_mags):.1f}px")
        if np.max(shift_mags) > Config.LARGE_SHIFT_WARNING_PX:
            warning = f"Large shifts detected (max={np.max(shift_mags):.1f}px)"
            stats.add_warning(warning)
            safe_print(f"  ⚠ {warning}")

    # Shift pattern warnings
    shift_set = set(f.shift for f in final)
    zero_shifts = sum(1 for f in final if f.shift == (0.0, 0.0))
    if len(shift_set) == 1 and len(final) > 2:
        unique_shift = list(shift_set)[0]
        if unique_shift != (0.0, 0.0):
            warning = f'All {len(final)} frames have IDENTICAL shift — registration failure!'
            stats.add_warning(warning)
            safe_print(f'\n  ⚠ WARNING: {warning}')
        elif len(final) <= 3:
            safe_print(f'\n  INFO: All frames registered with zero shift — well-aligned.')
    elif zero_shifts > len(final) * 0.8 and len(final) > 2:
        warning = f'{zero_shifts}/{len(final)} frames have zero shift'
        stats.add_warning(warning)
        safe_print(f'\n  ⚠ WARNING: {warning}')

    dither_info = detect_dither(shifts, verbose=args.verbose)
    if not args.no_registration and len(shifts) > 2:
        print(f"\n  Dither analysis:")
        print(f"    Pattern: {dither_info['pattern'].replace('_', ' ').title()}")
        print(f"    Mean shift: {dither_info['mean_magnitude']:.1f} px")
        print(f"    Unique positions: {dither_info['unique_positions']}/{len(shifts)} frames")
        if dither_info['is_dithered'] and args.stack_method == 'mean':
            safe_print(f"    Recommendation: Use --stack-method sigma_clip for dithered data")

    # ======================================================================
    # PHASE 3: Stacking (quality-weighted, with post-processing chain)
    # ======================================================================
    print_phase(3, "Stacking")
    phase_start = time.time()
    print(f"  Method: {args.stack_method}")
    print(f"  Combining {len(final)} frames...")

    # Compute quality weights for weighted stacking
    scores = np.array([f.metrics.get('score', 1.0) for f in final])
    max_score = scores.max() if scores.max() > 0 else 1.0
    weights = np.sqrt(scores / max_score)  # Sqrt-compress: preserves ranking, improves effective frame count
    print(f"  Quality weights: min={weights.min():.3f}, max={weights.max():.3f}, "
          f"mean={weights.mean():.3f} (sqrt-compressed)")

    # Crop to common valid region
    top, bottom, left, right = calc_common_crop(shifts, (H, W))
    stats.output_shape = (bottom - top, right - left)
    stats.cropped_pixels = (H - (bottom - top), W - (right - left))

    n_final = len(final)
    use_aligned_memmap = args.stack_method in ('median', 'sigma_clip')

    if use_aligned_memmap:
        # Create aligned memmap for median/sigma_clip
        mm_aligned_path = os.path.join(tempfile.gettempdir(), f'stack_aligned_{os.getpid()}.dat')
        crop_h, crop_w = bottom - top, right - left
        mem_aligned = np.memmap(mm_aligned_path, dtype='float32', mode='w+',
                                shape=(n_final, crop_h, crop_w, C))

        def _align_one_frame(j):
            orig_idx = final_indices[j]
            rgb = np.array(mem_rgb[orig_idx])
            aligned = apply_transform(rgb, shift=shifts[j], transform=transforms[j])
            mem_aligned[j] = aligned[top:bottom, left:right, :]

        n_align_workers = min(os.cpu_count() or 4, n_final)
        with ThreadPoolExecutor(max_workers=n_align_workers) as executor:
            futures = {executor.submit(_align_one_frame, j): j for j in range(n_final)}
            for future in tqdm(as_completed(futures), total=n_final,
                               desc="  Aligning", unit="frame",
                               disable=args.verbose):
                future.result()  # propagate exceptions
        mem_aligned.flush()

        if args.stack_method == 'sigma_clip':
            print(f"  Sigma-clip: sigma={args.rejection_sigma}, iters={args.rejection_iters}, "
                  f"mode={'winsorized' if getattr(args, 'winsorize', False) else 'reject'}")
            stacked = sigma_clip_combine(
                mem_aligned, sigma=args.rejection_sigma,
                max_iters=args.rejection_iters,
                weights=weights,
                winsorize=getattr(args, 'winsorize', False),
                verbose=args.verbose)
        else:
            stacked = np.median(mem_aligned, axis=0).astype(np.float32)

        del mem_aligned
        try:
            os.remove(mm_aligned_path)
        except Exception:
            pass
    else:
        # Weighted mean combine — align in parallel, accumulate as results arrive
        acc = np.zeros((bottom - top, right - left, C), dtype=np.float64)
        total_weight = 0.0

        def _align_and_crop(j):
            orig_idx = final_indices[j]
            rgb = np.array(mem_rgb[orig_idx])
            aligned = apply_transform(rgb, shift=shifts[j], transform=transforms[j])
            return j, aligned[top:bottom, left:right, :]

        n_mean_workers = min(os.cpu_count() or 4, n_final)
        with ThreadPoolExecutor(max_workers=n_mean_workers) as executor:
            futures = {executor.submit(_align_and_crop, j): j for j in range(n_final)}
            for future in tqdm(as_completed(futures), total=n_final,
                               desc="  Stacking", unit="frame",
                               disable=args.verbose):
                j, cropped = future.result()
                w = float(weights[j])
                acc += cropped.astype(np.float64) * w
                total_weight += w

        stacked = (acc / max(total_weight, 1e-12)).astype(np.float32)

    stats.stacking_time = time.time() - phase_start

    # Cleanup input memmaps
    del mem_rgb, mem_lum
    for p in (mm_rgb_path, mm_lum_path):
        try:
            os.remove(p)
        except Exception:
            pass

    # ======================================================================
    # Post-processing chain
    # ======================================================================

    # 1. Background extraction (with star mask for better exclusion)
    if args.background_extraction:
        print(f"\n  Applying background extraction (mesh={args.bg_mesh_size}, "
              f"sigma={args.bg_clip_sigma})...")
        bg_start = time.time()
        # Generate star mask from stacked luminance for better bg estimation
        star_mask = None
        stacked_lum = 0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1] + 0.114 * stacked[:, :, 2]
        if DAOStarFinder is not None and sigma_clipped_stats is not None:
            try:
                _, bg_med, bg_std = sigma_clipped_stats(stacked_lum, sigma=3.0, maxiters=5)
                daof = DAOStarFinder(fwhm=3.0, threshold=float(bg_med) + 5.0 * float(bg_std))
                stacked_sources = daof(stacked_lum - float(bg_med))
                if stacked_sources is not None and len(stacked_sources) > 0:
                    star_mask = generate_star_mask(stacked_lum.shape, stacked_sources, fwhm=4.0)
                    if args.verbose:
                        safe_print(f"    Star mask: {len(stacked_sources)} stars masked")
            except Exception:
                pass

        stacked = apply_background_extraction(
            stacked, mesh_size=args.bg_mesh_size,
            filter_size=args.bg_filter_size,
            clip_sigma=args.bg_clip_sigma,
            verbose=args.verbose,
            star_mask=star_mask)
        safe_print(f"  ✓ Background extraction ({format_time(time.time() - bg_start)})")

    # 2. Local normalization
    if getattr(args, 'local_normalize', False):
        ln_sigma = getattr(args, 'local_normalize_sigma', 50.0)
        print(f"\n  Applying local normalization (sigma={ln_sigma})...")
        ln_start = time.time()
        stacked = local_normalize(stacked, sigma=ln_sigma)
        safe_print(f"  ✓ Local normalization ({format_time(time.time() - ln_start)})")

    # 3. Wavelet denoising
    if getattr(args, 'denoise', False):
        strength = getattr(args, 'denoise_strength', 3.0)
        print(f"\n  Applying wavelet denoising (strength={strength})...")
        dn_start = time.time()
        stacked = wavelet_denoise(stacked, threshold_factor=strength)
        safe_print(f"  ✓ Wavelet denoise ({format_time(time.time() - dn_start)})")

    # Update memory usage
    if HAS_PSUTIL:
        stats.peak_memory_mb = get_memory_usage_mb()

    # Save FITS (3,H,W)
    out_h, out_w, _ = stacked.shape
    hdu = fits.PrimaryHDU()
    data_out = np.transpose(stacked, (2, 0, 1)).astype(np.float32)
    hdu.data = data_out

    populate_fits_header(
        header=hdu.header, frames=final, stats=stats, args=args,
        stacked_shape=stacked.shape, shifts=shifts,
        masters=masters, dither_info=dither_info)
    hdu.writeto(output_path, overwrite=True)

    # Plate solving
    plate_solved = False
    if not args.skip_plate_solve:
        if args.verbose:
            print("\n  Attempting plate solving...")
        plate_solved = solve_plate(data_out, hdu.header, output_path, verbose=args.verbose)
        if plate_solved:
            hdu.writeto(output_path, overwrite=True)
    elif args.verbose:
        print("\n  Plate solving skipped (--skip-plate-solve)")

    # Preview with configurable stretch
    preview_path = os.path.splitext(output_path)[0] + '.jpg'
    stretch_method = getattr(args, 'stretch', 'linear')
    save_preview_rgb(stacked, preview_path, stretch=stretch_method)

    print(f"  Output size: {out_h}x{out_w} "
          f"(cropped {stats.cropped_pixels[0]}x{stats.cropped_pixels[1]} pixels)")

    # Summary
    print_header("SUMMARY", "=")
    print(f"  Frames analyzed:  {stats.total_frames}")
    print(f"  Frames stacked:   {stats.accepted_frames} "
          f"({stats.accepted_frames/stats.total_frames*100:.1f}%)")
    if stats.rejected_frames > 0:
        print(f"  Frames rejected:  {stats.rejected_frames}")
    print(f"  Output:           {os.path.basename(output_path)} ({out_h}x{out_w}x3)")
    print(f"  Preview:          {os.path.basename(preview_path)} ({stretch_method} stretch)")
    print(f"  Processing time:  {format_time(stats.total_time())}")
    print(f"    Quality+Load:   {format_time(stats.quality_time)}")
    print(f"    Registration:   {format_time(stats.registration_time)}")
    print(f"    Stacking:       {format_time(stats.stacking_time)}")
    if HAS_PSUTIL:
        print(f"  Peak memory:      {stats.peak_memory_mb:.1f} MB")

    if stats.warnings:
        safe_print(f"\n  Warnings:")
        for w in stats.warnings[:5]:
            safe_print(f"    - {w}")
    if stats.errors:
        safe_print(f"\n  Errors: {len(stats.errors)}")

    safe_print(f"\n  Stack complete!")
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
            # Flat has vignetting + dust donuts (>30px); preserve those
            sigma_f = max(1, 15 // max(1, int(np.sqrt(n_flat))))
            masters['flat'] = ndimage.gaussian_filter(masters['flat'].astype(np.float32), sigma=sigma_f)

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
        save_preview_rgb(combined, preview_path, stretch=getattr(args, 'stretch', 'linear'))

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
    p.add_argument('--skip-phase-correlation', action='store_true',
                   help='Skip phase correlation, use only fallback methods (debug)')
    p.add_argument('--affine', action='store_true',
                   help='Enable affine (rotation+translation) registration via star matching')
    p.add_argument('--quality-filter', action='store_true')
    p.add_argument('--quality-threshold', type=float, default=50.0)
    p.add_argument('--keep-intermediates', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--debug-registration', action='store_true',
                   help='Detailed registration diagnostics (implies -v)')
    p.add_argument('--stack-method', choices=['mean', 'median', 'sigma_clip'], default='mean')
    p.add_argument('--rejection-sigma', type=float, default=3.0,
                   help='Sigma threshold for pixel rejection in sigma_clip stacking (default: 3.0)')
    p.add_argument('--rejection-iters', type=int, default=3,
                   help='Number of clipping iterations for sigma_clip stacking (default: 3)')
    p.add_argument('--winsorize', action='store_true',
                   help='Winsorized sigma-clip: clip outliers to boundary instead of rejecting')
    p.add_argument('--debayer-method', choices=['bilinear', 'malvar', 'vng'], default='bilinear',
                   help='Debayering method (vng requires OpenCV)')
    p.add_argument('--white-balance', choices=['none', 'grayworld', 'whitepatch'], default='grayworld')
    p.add_argument('--drizzle-scale', type=int, default=1,
                   help='Integer drizzle scale factor (1 = disabled)')
    p.add_argument('--use-gpu', action='store_true',
                   help='Use CuPy for available operations (experimental)')
    p.add_argument('--skip-plate-solve', action='store_true',
                   help='Skip plate solving (astrometry)')
    p.add_argument('--background-extraction', action='store_true', default=True,
                   help='Enable intelligent background removal for darker sky (default: on)')
    p.add_argument('--no-background-extraction', dest='background_extraction',
                   action='store_false',
                   help='Disable background extraction')
    p.add_argument('--bg-mesh-size', type=int, default=256,
                   help='Grid cell size in pixels for background estimation (default: 256)')
    p.add_argument('--bg-filter-size', type=int, default=3,
                   help='Median filter size for background grid smoothing (default: 3, must be odd)')
    p.add_argument('--bg-clip-sigma', type=float, default=3.0,
                   help='Sigma for star rejection in background estimation (default: 3.0)')
    p.add_argument('--denoise', action='store_true',
                   help='Enable wavelet denoising post-stack (requires pywt)')
    p.add_argument('--denoise-strength', type=float, default=3.0,
                   help='Wavelet denoise threshold factor (default: 3.0)')
    p.add_argument('--local-normalize', action='store_true',
                   help='Enable local normalization to remove vignetting residuals')
    p.add_argument('--local-normalize-sigma', type=float, default=50.0,
                   help='Gaussian sigma for local normalization (default: 50)')
    p.add_argument('--stretch', choices=['linear', 'arcsinh'], default='linear',
                   help='Preview image stretch method (default: linear)')
    p.add_argument('-j', '--parallel', type=int, default=1,
                   help='Parallel workers for frame processing (0=auto, 1=sequential)')
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
