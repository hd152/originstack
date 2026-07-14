"""FITS I/O utilities: loading, saving, master frame creation, preview generation."""
from __future__ import annotations

import argparse
import os
import tempfile
from typing import List, Dict, Optional, Tuple

import numpy as np
from astropy.io import fits

from src.models import FrameInfo, ProcessingStats
from src.utils import safe_print

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from skimage import exposure
except Exception:
    exposure = None


def _read_fits_header(path: str) -> dict:
    """Read FITS header only, with memmap fallback on keyword-compression errors."""
    try:
        with fits.open(path, memmap=True) as hd:
            return dict(hd[0].header)
    except Exception:
        try:
            with fits.open(path, memmap=False) as hd:
                return dict(hd[0].header)
        except Exception:
            return {}


def load_frame(path: str) -> Tuple[np.ndarray, dict]:
    """Load a FITS, camera RAW, TIFF, XISF, or SER (virtual-path) file;
    dispatches on file extension (SER's ``path::index`` marker is checked
    first, since a virtual path's extension via splitext is meaningless).

    Only import failures (optional dependency missing) fall through to the
    next format / to FITS -- errors raised while actually reading a matched
    file (corrupt data, unsupported variant, bad frame index, ...) propagate
    to the caller instead of being masked by a confusing downstream FITS-open
    failure on a path that was never a FITS file."""
    try:
        from src.io_ser import is_ser_virtual_path, read_ser_frame
    except ImportError:
        pass
    else:
        if is_ser_virtual_path(path):
            return read_ser_frame(path)
    ext = os.path.splitext(path)[1].lower()
    try:
        from src.io_raw import RAW_EXTENSIONS, read_raw
    except ImportError:
        pass
    else:
        if ext in RAW_EXTENSIONS:
            return read_raw(path)
    try:
        from src.io_tiff import TIFF_EXTENSIONS, read_tiff
    except ImportError:
        pass
    else:
        if ext in TIFF_EXTENSIONS:
            return read_tiff(path)
    try:
        from src.io_xisf import XISF_EXTENSIONS, read_xisf
    except ImportError:
        pass
    else:
        if ext in XISF_EXTENSIONS:
            return read_xisf(path)
    return load_fits(path)


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
        first_data, _ = load_frame(frames[0].path)
        shape = first_data.shape
    except Exception:
        return None

    if method != 'median':
        # Streaming mean — O(1) memory per frame
        acc = np.zeros(shape, dtype=np.float64)
        count = 0
        for f in frames:
            try:
                data, _ = load_frame(f.path)
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
    try:
        import psutil
        _avail = psutil.virtual_memory().available
        _memmap_threshold = max(200_000_000, _avail // 3)
    except Exception:
        _memmap_threshold = 500_000_000
    if estimated_bytes > _memmap_threshold:
        mm_path = os.path.join(tempfile.gettempdir(), f'master_{os.getpid()}.dat')
        mem = np.memmap(mm_path, dtype='float32', mode='w+', shape=(n, *shape))
        count = 0
        for i, f in enumerate(frames):
            try:
                data, _ = load_frame(f.path)
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
                data, _ = load_frame(f.path)
                imgs.append(data.astype(np.float32))
            except Exception:
                continue
        if not imgs:
            return None
        return np.median(np.stack(imgs, axis=0), axis=0).astype(np.float32)


def render_preview_uint8(rgb: np.ndarray, stretch: str = 'linear',
                         ghs_b: float = 8.0, ghs_sp: float = 0.15,
                         ghs_hp: float = 0.95,
                         black_sigma: float = 0.0) -> Optional[np.ndarray]:
    """Stretch an HWC float32 image to display uint8 (the shared core of the
    preview JPEG file writer and the live web view). Returns None when the
    required stretch backend is unavailable."""
    from src.denoising import arcsinh_stretch, generalized_hyperbolic_stretch
    from src.models import Config
    if stretch == 'ghs':
        # Generalized Hyperbolic Stretch — uses unified luminance-based normalization
        # so all three channels share the same black/white reference, preserving
        # cross-channel color ratios that would otherwise be destroyed by independent
        # per-channel sky statistics.
        out = np.zeros_like(rgb)
        lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
        flat_lum = lum.ravel().astype(np.float64)
        _med = float(np.median(flat_lum))
        for _ in range(3):
            _mad = float(np.median(np.abs(flat_lum - _med)))
            _sig = 1.4826 * _mad
            flat_lum = flat_lum[np.abs(flat_lum - _med) < 2.5 * _sig]
            if len(flat_lum) < 100:
                break
            _med = float(np.median(flat_lum))
        _bg_sigma = float(np.std(flat_lum)) if len(flat_lum) > 1 else 1.0
        # Black point relative to the sky median in units of sky sigma.
        # black_sigma < 0 keeps sky noise visible (good for frame-filling faint
        # nebulae); black_sigma > 0 clips the noise floor to black (good for a
        # small target on empty sky, e.g. a galaxy or cluster, where a low black
        # point turns the whole background into a colour-noise storm). The
        # target-type advisor sets an appropriate value per preset.
        unified_black = _med + black_sigma * _bg_sigma
        unified_white = float(np.percentile(lum, 99.9))
        for c in range(3):
            out[:, :, c] = generalized_hyperbolic_stretch(
                rgb[:, :, c], b=ghs_b, SP=ghs_sp, LP=0.0, HP=ghs_hp,
                black_point=unified_black, white_point=unified_white)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
    elif stretch == 'arcsinh':
        # Arcsinh stretch — unified luminance-based normalization to preserve color
        out = np.zeros_like(rgb)
        lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
        flat_lum = lum.ravel().astype(np.float64)
        _med = float(np.median(flat_lum))
        for _ in range(3):
            _mad = float(np.median(np.abs(flat_lum - _med)))
            _sig = 1.4826 * _mad
            flat_lum = flat_lum[np.abs(flat_lum - _med) < 2.5 * _sig]
            if len(flat_lum) < 100:
                break
            _med = float(np.median(flat_lum))
        _bg_sigma = float(np.std(flat_lum)) if len(flat_lum) > 1 else 1.0
        unified_black = _med + black_sigma * _bg_sigma
        unified_white = float(np.percentile(lum, 99.8))
        for c in range(3):
            out[:, :, c] = arcsinh_stretch(rgb[:, :, c],
                                           black_point=unified_black,
                                           white_point=unified_white)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
    else:
        # Linear percentile stretch (original behaviour)
        if exposure is None:
            return None
        out = np.zeros_like(rgb)
        for c in range(3):
            lo, hi = np.percentile(rgb[:, :, c], Config.PREVIEW_STRETCH_PERCENTILES)
            lo = max(lo, 0.0)  # Don't let negative noise expand the display range
            out[:, :, c] = exposure.rescale_intensity(rgb[:, :, c], in_range=(lo, hi))
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out


def _preview_pil_image(out: np.ndarray, max_dim: int):
    """uint8 HWC -> size-capped PIL image (shared by file and bytes paths)."""
    h, w = out.shape[:2]
    if max(h, w) > max_dim:
        # Pre-slice in numpy before PIL to avoid allocating a huge PIL Image object.
        # A full-resolution fromarray() on a large stack can OOM mid-JPEG-write,
        # leaving a corrupt partial file. Stride-slice to ~target size first, then
        # let thumbnail() do a quality LANCZOS pass to the exact limit.
        step = max(1, max(h, w) // max_dim)
        out = np.ascontiguousarray(out[::step, ::step, :])
    img = Image.fromarray(out)
    if img.width > max_dim or img.height > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return img


def save_preview_rgb(rgb: np.ndarray, path: str, stretch: str = 'linear',
                     ghs_b: float = 8.0, ghs_sp: float = 0.15,
                     ghs_hp: float = 0.95, black_sigma: float = 0.0) -> None:
    from src.models import Config
    if Image is None:
        return
    out = render_preview_uint8(rgb, stretch=stretch, ghs_b=ghs_b, ghs_sp=ghs_sp,
                               ghs_hp=ghs_hp, black_sigma=black_sigma)
    if out is None:
        return
    img = _preview_pil_image(out, Config.PREVIEW_MAX_DIMENSION)
    img.save(path, format='JPEG', quality=Config.PREVIEW_JPEG_QUALITY)


def preview_jpeg_bytes(rgb: np.ndarray, stretch: str = 'ghs',
                       ghs_b: float = 8.0, ghs_sp: float = 0.15,
                       ghs_hp: float = 0.95, black_sigma: float = 0.0,
                       max_dim: int = 1024) -> Optional[bytes]:
    """Stretched preview JPEG as bytes (for the live web view)."""
    import io as _io
    if Image is None:
        return None
    out = render_preview_uint8(rgb, stretch=stretch, ghs_b=ghs_b, ghs_sp=ghs_sp,
                               ghs_hp=ghs_hp, black_sigma=black_sigma)
    if out is None:
        return None
    img = _preview_pil_image(out, max_dim)
    buf = _io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def populate_fits_header(header: fits.Header, frames: List[FrameInfo],
                         stats: ProcessingStats, args: argparse.Namespace,
                         stacked_shape: Tuple[int, int, int],
                         shifts: List[Tuple[float, float]],
                         masters: Dict[str, Optional[np.ndarray]],
                         dither_info: Optional[Dict] = None,
                         post_processed: bool = False) -> None:
    """Populate FITS header with comprehensive metadata.

    post_processed: True if the saved data has had post-processing applied
    (background extraction, denoising, etc.).  False means the FITS contains
    the raw linear stacked data before Phase 4.
    """
    from datetime import datetime, timezone

    try:
        import psutil
        HAS_PSUTIL = True
    except Exception:
        HAS_PSUTIL = False

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
        # Copy common FITS keywords if they exist.
        # BAYERPAT is intentionally excluded: the output is a debayered RGB
        # image, not a raw CFA mosaic.  Viewers such as Siril use BAYERPAT to
        # detect raw frames and will attempt to debayer the already-processed
        # image if the keyword is present, producing garbage.
        copy_keys = ['TELESCOP', 'INSTRUME', 'OBSERVER', 'OBJECT', 'DATE-OBS',
                     'EXPTIME', 'CCD-TEMP', 'GAIN', 'OFFSET', 'XBINNING', 'YBINNING',
                     'XPIXSZ', 'YPIXSZ', 'FOCALLEN', 'APTDIA']
        for key in copy_keys:
            if key in first_header:
                header[key] = first_header[key]

        # Inferred target metadata (set by target_inference before this call)
        inferred_name = getattr(args, '_inferred_target', None)
        inferred_type = getattr(args, '_inferred_type', None)
        inferred_conf = getattr(args, '_inferred_confidence', 0.0)
        inferred_src  = getattr(args, '_inferred_source', None)
        if inferred_name and inferred_type and inferred_type != 'unknown':
            # Only write OBJECT if no capture software already filled it in
            if 'OBJECT' not in header:
                header['OBJECT'] = (inferred_name[:68], 'Inferred target name')
            header['OBJTYPE'] = (inferred_type[:68], 'Inferred object type')
            header['INFCONF'] = (round(float(inferred_conf), 3),
                                 'Target inference confidence (0-1)')
            if inferred_src:
                header['INFSRC'] = (inferred_src[:68],
                                    'Target inference source')

        # Session info metadata (from info.json written by capture app)
        si = getattr(args, '_session_info', None)
        if si is not None:
            # Equipment metadata — only fill gaps not already covered by FITS headers
            if si.telescope and 'TELESCOP' not in header:
                header['TELESCOP'] = (si.telescope[:68], 'Telescope from session info')
            if si.mount:
                header['MOUNT'] = (si.mount[:68], 'Mount from session info')
            if si.reducer:
                header['REDUCER'] = (si.reducer[:68], 'Reducer/flattener from session info')
            # Filter — prefer existing FITS keyword
            if si.filter_name and 'FILTER' not in header:
                header['FILTER'] = (si.filter_name[:68], 'Filter from session info')
            # Fallback exposure/ISO from session when FITS headers are missing
            if si.exposure is not None and 'EXPTIME' not in header:
                header['EXPTIME'] = (si.exposure, 'Exposure time from session info (s)')
            if si.iso is not None and 'GAIN' not in header:
                header['ISO'] = (si.iso, 'ISO from session info')
            if si.temperature is not None and 'CCD-TEMP' not in header:
                header['CCD-TEMP'] = (si.temperature, 'Sensor temperature from session info (C)')
            # Observation site GPS coordinates
            if si.has_gps:
                header['SITELAT'] = (round(si.latitude, 6), 'Observatory latitude (degrees)')
                header['SITELONG'] = (round(si.longitude, 6), 'Observatory longitude (degrees)')
                if si.altitude is not None:
                    header['SITEELEV'] = (round(si.altitude, 1), 'Observatory altitude (metres)')
            # Session integration info
            if si.total_duration_ms is not None:
                total_s = si.total_duration_ms / 1000.0
                if 'INTGTIME' not in header:
                    header['INTGTIME'] = (round(total_s, 1),
                                          'Total integration time (session info, seconds)')
            # WCS from celestial + FOV + orientation (only when plate solve has not run)
            if si.has_wcs and 'CTYPE1' not in header:
                from src.session_info import build_wcs_keywords
                wcs = build_wcs_keywords(si)
                for kw, (val, comment) in wcs.items():
                    header[kw] = (val, comment)

        # Mark the output as a debayered RGB image so FITS viewers open it
        # correctly.  Siril and many other tools recognise COLORTYP=SRGB to
        # identify a 3-plane (NAXIS3=3) FITS cube as a colour image rather
        # than three separate science frames.
        header['COLORTYP'] = ('SRGB', 'Colour space of the stacked image')

        # Aggregate exposure info across all frames
        frame_dates = []
        total_integration = 0.0
        iso_values = set()
        for f in frames:
            if f.header.get('DATE-OBS'):
                frame_dates.append(str(f.header['DATE-OBS']))
            if f.header.get('EXPTIME'):
                try:
                    total_integration += float(f.header['EXPTIME'])
                except (ValueError, TypeError):
                    pass
            iso = f.header.get('ISOSPEED') or f.header.get('ISO') or f.header.get('GAIN')
            if iso is not None:
                iso_values.add(str(iso))

        if total_integration > 0:
            header['INTGTIME'] = (round(total_integration, 1),
                                  'Total integration time across all frames (seconds)')
            if total_integration >= 60:
                header['INTGMIN'] = (round(total_integration / 60, 1),
                                     'Total integration time (minutes)')
            header['TOTEXP'] = (round(total_integration, 1),
                                'Total integrated exposure time in seconds')
        elif 'EXPTIME' in first_header:
            try:
                total_exp = float(first_header['EXPTIME']) * len(frames)
                header['TOTEXP'] = (total_exp, 'Total integrated exposure time in seconds')
            except (ValueError, TypeError):
                pass
        if frame_dates:
            header['DATEFRST'] = (min(frame_dates), 'Date of first frame')
            header['DATELAST'] = (max(frame_dates), 'Date of last frame')
        if iso_values:
            header['ISOVALUS'] = (','.join(sorted(iso_values)), 'ISO/gain values used')

    # Whether post-processing was applied to the data in this FITS
    header['RAWSTACK'] = (not post_processed,
                          'True = pre-post-processing linear stack; sky background not subtracted')

    # Background extraction info (reflects what was done to the saved data)
    bg_applied = post_processed and args.background_extraction
    if bg_applied:
        header['BGEXTR'] = (True, 'Background extraction applied to FITS data')
        header['BGMESH'] = (args.bg_mesh_size, 'Background mesh cell size in pixels')
        header['BGFILTR'] = (args.bg_filter_size, 'Background grid filter size')
        header['BGCLIP'] = (args.bg_clip_sigma, 'Background sigma-clip threshold')
    else:
        header['BGEXTR'] = (False, 'Background extraction NOT applied to FITS data')

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
    header['AFFINE'] = (not getattr(args, 'no_affine', False), 'Affine registration enabled')

    # Post-processing flags (reflect what was done to the saved data, not just config)
    denoise_applied = post_processed and getattr(args, 'denoise', False)
    header['DENOISE'] = (denoise_applied, 'Wavelet denoising applied to FITS data')
    if denoise_applied:
        header['DNSTRNG'] = (getattr(args, 'denoise_strength', 3.0), 'Denoise threshold factor')
    header['STRETCH'] = (getattr(args, 'stretch', 'linear'), 'Preview stretch method (JPG only)')
    header['DEBAYER'] = (args.debayer_method, 'Debayering method used')

    # Drizzle
    drizzle_scale = getattr(args, 'drizzle_scale', 1.0)
    header['DRIZZLE'] = (drizzle_scale > 1.0, 'Drizzle upscaling applied')
    if drizzle_scale > 1.0:
        header['DRZSCALE'] = (drizzle_scale, 'Drizzle scale factor')
        header['DRZPIXFR'] = (getattr(args, 'drizzle_pixfrac', 1.0), 'Drizzle pixel fraction')

    # Richardson-Lucy deconvolution
    deconv_applied = post_processed and getattr(args, 'deconvolve', False)
    header['DECONV'] = (deconv_applied, 'Richardson-Lucy deconvolution applied')
    if getattr(args, 'deconvolve', False):
        header['DCITERS'] = (getattr(args, 'deconvolve_iterations', 15), 'Deconvolution iterations')
        if getattr(args, 'deconvolve_fwhm', None):
            header['DCFWHM'] = (args.deconvolve_fwhm, 'Deconvolution PSF FWHM (manual)')
        header['DCMODEL'] = (getattr(args, 'deconvolve_psf_model', 'moffat'), 'PSF model used')

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

    # Field aberration inspector summary (--aberration-report)
    _ab = getattr(args, '_aberration', None)
    if _ab is not None:
        header['ABFWHM'] = (round(float(_ab.get('fwhm_median', 0.0)), 2),
                            'Field-median star FWHM (aberration report)')
        header['ABSPREAD'] = (round(float(_ab.get('fwhm_spread_pct', 0.0)), 1),
                              'FWHM spread across field (percent)')
        header['ABELLIP'] = (round(float(_ab.get('ellipticity_median', 0.0)), 3),
                             'Field-median star ellipticity')
        if _ab.get('tilt_direction'):
            header['ABTILT'] = (str(_ab['tilt_direction'])[:8], 'Sensor-tilt soft-side direction')
            header['ABTILTPX'] = (round(float(_ab.get('tilt_gradient_px', 0.0)), 2),
                                  'FWHM gradient across field (px)')
        header['ABCURV'] = (round(float(_ab.get('curvature_corr', 0.0)), 3),
                            'FWHM-vs-radius correlation (field curvature)')
        _diag = _ab.get('diagnosis') or []
        if _diag:
            header['ABDIAG'] = (str(_diag[0])[:68], 'Aberration diagnosis')
