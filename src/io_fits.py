"""FITS I/O utilities: loading, saving, master frame creation, preview generation."""
from __future__ import annotations

import argparse
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from src.models import FrameInfo, ProcessingStats

try:
    from PIL import Image
except Exception:
    Image = None


def _rescale_intensity(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly rescale ``x`` from [lo, hi] to [0, 1], clipping outside that
    range -- equivalent to skimage.exposure.rescale_intensity(x, in_range=(lo,
    hi)) with its default out_range for a float array."""
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip((x.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)


def _sky_stats(lum: np.ndarray) -> Tuple[float, float]:
    """Robust (median, sigma) of the sky background via 2.5-sigma /
    3-iteration clipping -- shared by the ghs and arcsinh preview stretch
    branches below."""
    _, med, sigma = sigma_clipped_stats(lum, sigma=2.5, maxiters=3)
    med = float(med)
    sigma = float(sigma) if np.isfinite(sigma) and sigma > 0 else 1.0
    return med, sigma


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


def read_fits_be16(path: str):
    """(raw, bzero, header dict) for a plain 2-D BITPIX=16 FITS image (BSCALE 1, BZERO 0
    or 32768, no BLANK), with ``raw`` the data block's bytes as native uint16 (still
    big-endian: ``debayer.calibrate_frame_be16`` swaps them while it calibrates). None
    for anything else -- the caller then uses ``load_frame``. The physical values are
    exactly what ``load_fits`` returns; reading the block directly skips astropy's
    conversion (~30 ms of the ~34 ms it took per Origin frame)."""
    if not str(path).lower().endswith(('.fits', '.fit', '.fts')):
        return None
    try:
        with fits.open(path, memmap=False, lazy_load_hdus=True) as hd:
            h = hd[0].header
            if (h.get('NAXIS') != 2 or h.get('BITPIX') != 16 or 'BLANK' in h
                    or float(h.get('BSCALE', 1.0)) != 1.0
                    or float(h.get('BZERO', 0.0)) not in (0.0, 32768.0)
                    or h.get('XTENSION') is not None):
                return None
            off = hd.fileinfo(0)['datLoc']
            H, W = int(h['NAXIS2']), int(h['NAXIS1'])
            hdr = dict(h)
            bzero = float(h.get('BZERO', 0.0))
        raw = np.fromfile(path, dtype=np.uint16, count=H * W, offset=off)
        if raw.size != H * W:
            return None
        return raw.reshape(H, W), bzero, hdr
    except Exception:
        return None


def make_master(frames: List[FrameInfo], method: str = 'median',
                downsample: int = 1) -> Optional[np.ndarray]:
    """Create master calibration frame using streaming (mean), memmap (median),
    or robust PCA (low-rank + sparse decomposition, ``method='robust_pca'``).

    ``downsample`` only affects the robust_pca path (see
    ``robust_pca_master``'s docstring) -- median/mean ignore it."""
    if not frames:
        return None
    # Probe first frame for shape
    try:
        first_data, _ = load_frame(frames[0].path)
        shape = first_data.shape
    except Exception:
        return None

    if method == 'robust_pca':
        from src.robust_pca import robust_pca_master
        master = robust_pca_master(frames, shape, downsample=downsample)
        if master is not None:
            return master
        method = 'median'  # too few frames for RPCA -- fall back

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


def _preview_white(lum: np.ndarray, sky: float, sigma: float, percentile: float,
                   black: float) -> Tuple[float, float]:
    """Preview white point, and the factor to scale GHS's symmetry point by.

    White is the given luminance percentile, but never closer to the sky than
    ``Config.PREVIEW_WHITE_MIN_SIGMA`` sky sigmas: on a small target in an empty
    field the percentile is barely above sky, which stretched the sky noise across
    the whole display range (see the Config note). Raising white by a factor puts
    the target's light that much lower in the normalised range, below where the
    curve's contrast is focused (SP), so SP follows it down -- by at most 3x:
    with SP / 3 the galaxies' outer disks and arms came back on a still-clean sky
    (Black Eye, Sunflower), while a black point at the sky brought the speckle
    back."""
    from src.models import Config
    pct = float(np.percentile(lum, percentile))
    white = max(pct, float(sky) + Config.PREVIEW_WHITE_MIN_SIGMA * float(sigma))
    span = white - float(black)
    sp_scale = 1.0 if pct >= white or span <= 0 else max((pct - float(black)) / span, 1.0 / 3.0)
    return white, sp_scale


def _extended_highlight_top(lum: np.ndarray, white: float) -> Optional[float]:
    """Top of the highlight roll-off, or None when there is nothing to roll off.

    A white point at the 99.5th percentile clips star cores, which is fine, but
    also any bright *extended* region above it -- a nebula core such as the
    Trapezium region of M42 then renders as a flat white blob. That case is told
    apart from star cores by the largest connected region above white (see
    ``Config.PREVIEW_ROLLOFF_MIN_AREA``); the top is the 99.99th percentile."""
    from scipy import ndimage

    from src.models import Config
    above = lum > white
    if not above.any():
        return None
    labels, n = ndimage.label(above)
    if n == 0 or np.bincount(labels.ravel())[1:].max() < Config.PREVIEW_ROLLOFF_MIN_AREA * lum.size:
        return None
    top = float(np.percentile(lum, 99.99))
    return top if top > 1.2 * float(white) else None


def _rolloff_curve(curve, lum: np.ndarray, white: float, top: float) -> np.ndarray:
    """``curve`` (a luminance -> [0, 1] stretch with white point ``white``) scaled
    into [0, knee], with a log roll-off mapping white..top into [knee, 1]."""
    from src.models import Config
    knee = Config.PREVIEW_ROLLOFF_KNEE
    a = Config.PREVIEW_ROLLOFF_STRENGTH
    below = np.asarray(curve(np.minimum(lum, white).astype(np.float32)), dtype=np.float64)
    over = np.clip((lum.astype(np.float64) - white) / (top - white), 0.0, 1.0)
    return np.where(lum <= white, knee * below,
                    knee + (1.0 - knee) * np.log1p(a * over) / np.log1p(a)).astype(np.float32)


def _preserving_preview(rgb: np.ndarray, lum: np.ndarray, curve, black: float,
                        white: float, sky_sigma: float) -> np.ndarray:
    """Colour-preserving preview of ``rgb`` with the luminance curve ``curve``,
    rolling off an extended highlight above ``white`` instead of clipping it."""
    top = _extended_highlight_top(lum, white)
    if top is None:
        return colour_preserving_stretch(rgb, curve(lum), black, white, sky_sigma)
    return colour_preserving_stretch(rgb, _rolloff_curve(curve, lum, white, top),
                                     black, top, sky_sigma)


def colour_preserving_stretch(rgb: np.ndarray, lum_stretched: np.ndarray,
                              black: float, white: float,
                              sky_sigma: float) -> np.ndarray:
    """Apply a luminance curve to an RGB image without changing its hue.

    A curve applied to each channel separately is not colour-preserving, even
    with one shared black/white point: a nonlinear curve compresses the
    brighter channel more, so every star and bright core is pushed toward
    white and hue drifts with brightness (on the real stacks scored by
    ``tools/bench_phase4.py`` the stars' display colour spread was 17-29% of
    their linear colour spread). Here the curve ``T`` is applied to luminance
    only and every channel is scaled by the same ``T(n) / n`` (Lupton et al.
    2004), so each pixel keeps its RGB ratios and its luminance is ``T``.
    Two corrections keep the result displayable:

    * gamut: a saturated colour whose luminance maps near 1 has a channel
      above 1. Rather than clipping that channel (a hue shift) or
      desaturating toward grey (which turns every bright star core white,
      since a core at the white point has ``T = 1``), the pixel is divided by
      its largest channel (Lupton et al.'s rescale): hue and saturation are
      kept and only that pixel's luminance drops below ``T``.
    * noise floor: near the black point a channel's ratio to luminance is
      mostly noise, which the per-pixel scale would render as colour speckle.
      Chroma fades in over the first 3 sky sigma above black (grey below).

    ``lum_stretched`` is the curve evaluated on the image's luminance with the
    same ``black``/``white`` points; ``sky_sigma`` is the luminance sky sigma.
    """
    span = max(float(white) - float(black), 1e-12)
    w = np.array([0.299, 0.587, 0.114])
    cn = (rgb.astype(np.float64) - float(black)) / span
    n = cn @ w
    T = np.asarray(lum_stretched, dtype=np.float64)
    pos = n > 1e-9
    k = np.where(pos, T / np.where(pos, n, 1.0), 0.0)
    out = cn * k[..., None]
    Tg = T[..., None]
    n_gate = max(3.0 * float(sky_sigma) / span, 1e-9)
    g = np.clip(n / n_gate, 0.0, 1.0)[..., None]
    out = Tg + g * (out - Tg)
    # A negative channel (a pixel bluer/redder than the black point allows):
    # desaturate toward grey at constant luminance until it reaches 0.
    mn = out.min(axis=2, keepdims=True)
    s_lo = np.where(mn < 0.0, Tg / np.maximum(Tg - mn, 1e-12), 1.0)
    out = Tg + np.clip(s_lo, 0.0, 1.0) * (out - Tg)
    # Over-range: rescale by the largest channel (keeps hue and saturation).
    mx = out.max(axis=2, keepdims=True)
    out = out / np.maximum(mx, 1.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def render_preview_uint8(rgb: np.ndarray, stretch: str = 'linear',
                         ghs_b: float = 8.0, ghs_sp: float = 0.15,
                         ghs_hp: float = 0.95,
                         black_sigma: float = 0.0,
                         color: str = 'preserve') -> Optional[np.ndarray]:
    """Stretch an HWC float32 image to display uint8 (the shared core of the
    preview JPEG file writer and the live web view). Returns None when the
    required stretch backend is unavailable."""
    out = render_preview_float(rgb, stretch=stretch, ghs_b=ghs_b, ghs_sp=ghs_sp,
                               ghs_hp=ghs_hp, black_sigma=black_sigma, color=color)
    if out is None:
        return None
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def render_preview_float(rgb: np.ndarray, stretch: str = 'linear',
                         ghs_b: float = 8.0, ghs_sp: float = 0.15,
                         ghs_hp: float = 0.95,
                         black_sigma: float = 0.0,
                         color: str = 'preserve') -> Optional[np.ndarray]:
    """The display stretch of ``render_preview_uint8`` before quantisation:
    HWC float in [0, 1]. ``tools/bench_phase4.py`` scores this.

    ``color`` (ghs/arcsinh only): 'preserve' (default) applies the curve to
    luminance and keeps each pixel's RGB ratios (``colour_preserving_stretch``);
    'channel' applies it to each channel separately (the behaviour before
    2026-10, which whitens stars and bright cores)."""
    from src.denoising import arcsinh_stretch, generalized_hyperbolic_stretch
    from src.models import Config
    if stretch == 'ghs':
        # Generalized Hyperbolic Stretch — uses unified luminance-based normalization
        # so all three channels share the same black/white reference, preserving
        # cross-channel color ratios that would otherwise be destroyed by independent
        # per-channel sky statistics.
        out = np.zeros_like(rgb)
        lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
        _med, _bg_sigma = _sky_stats(lum)
        # Black point relative to the sky median in units of sky sigma.
        # black_sigma < 0 keeps sky noise visible (good for frame-filling faint
        # nebulae); black_sigma > 0 clips the noise floor to black (good for a
        # small target on empty sky, e.g. a galaxy or cluster, where a low black
        # point turns the whole background into a colour-noise storm). The
        # target-type advisor sets an appropriate value per preset.
        unified_black = _med + black_sigma * _bg_sigma
        # 99.5th, not 99.9th: on a starry frame the top 0.1% of pixels is
        # saturated-star cores, which pushes the white point far above any
        # extended structure (nebulosity, galaxy arms). Since GHS's shadow
        # boost operates on the *normalized* (black..white) range, a white
        # point set that high buries genuine low-contrast diffuse signal
        # deep in the heavily-compressed shadow region of the curve --
        # visually indistinguishable from sky even though the pixel data
        # is fine. 99.5 keeps stars comfortably white while giving diffuse
        # signal several times more of the normalized range to live in.
        unified_white, _sp_scale = _preview_white(lum, _med, _bg_sigma, 99.5, unified_black)
        ghs_sp = ghs_sp * _sp_scale
        if color == 'preserve':
            out = _preserving_preview(
                rgb, lum, lambda x: generalized_hyperbolic_stretch(
                    x, b=ghs_b, SP=ghs_sp, LP=0.0, HP=ghs_hp,
                    black_point=unified_black, white_point=unified_white),
                unified_black, unified_white, _bg_sigma)
        else:
            for c in range(3):
                out[:, :, c] = generalized_hyperbolic_stretch(
                    rgb[:, :, c], b=ghs_b, SP=ghs_sp, LP=0.0, HP=ghs_hp,
                    black_point=unified_black, white_point=unified_white)
    elif stretch == 'arcsinh':
        # Arcsinh stretch — unified luminance-based normalization to preserve color
        out = np.zeros_like(rgb)
        lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
        _med, _bg_sigma = _sky_stats(lum)
        unified_black = _med + black_sigma * _bg_sigma
        # See the 'ghs' branch above for why 99.5 rather than a higher
        # percentile: saturated-star pixels otherwise set a white point far
        # above any extended structure, burying diffuse signal near-black.
        unified_white, _ = _preview_white(lum, _med, _bg_sigma, 99.5, unified_black)
        if color == 'preserve':
            out = _preserving_preview(
                rgb, lum, lambda x: arcsinh_stretch(x, black_point=unified_black,
                                                    white_point=unified_white),
                unified_black, unified_white, _bg_sigma)
        else:
            for c in range(3):
                out[:, :, c] = arcsinh_stretch(rgb[:, :, c],
                                               black_point=unified_black,
                                               white_point=unified_white)
    else:
        # Linear percentile stretch (original behaviour)
        out = np.zeros_like(rgb)
        for c in range(3):
            lo, hi = np.percentile(rgb[:, :, c], Config.PREVIEW_STRETCH_PERCENTILES)
            lo = max(lo, 0.0)  # Don't let negative noise expand the display range
            out[:, :, c] = _rescale_intensity(rgb[:, :, c], lo, hi)
    return out


def render_preview_layered_uint8(rgb: np.ndarray, starless: np.ndarray,
                                 ghs_b: float = 8.0, ghs_sp: float = 0.15,
                                 ghs_hp: float = 0.95, black_sigma: float = 0.0,
                                 color: str = 'preserve') -> np.ndarray:
    """GHS preview whose black and white points come from the starless layer.

    One stretch has to pick a single white point, and on a starry frame the
    top of the histogram is stars (see the 99.5 percentile note in
    ``render_preview_uint8``), so faint extended structure -- what the stretch
    is for -- is squeezed into a sliver of the range. Here the points are read
    off the *starless* image, so a faint galaxy disk or nebula owns the range;
    the full image (stars included) is then stretched with those points and
    GHS's highlight protection keeps the star cores from clipping harder than
    they otherwise would.

    An earlier version stretched the starless layer and the stars layer
    separately and screen-blended them. That fails exactly where it matters: a
    bright star over the galaxy is inpainted from its surroundings, the fill
    sits below the true galaxy light there, and the stars layer carries the
    difference through a *different* curve -- a dark disc at every such star,
    measured on a real Fireworks Galaxy stack. Stretching the one full image
    avoids ever recombining two curves.
    """
    from src.denoising import generalized_hyperbolic_stretch
    lum_s = (0.299 * starless[:, :, 0] + 0.587 * starless[:, :, 1]
             + 0.114 * starless[:, :, 2])
    med, sigma = _sky_stats(lum_s)
    black = med + black_sigma * sigma
    # 99.9, not the 99.5 the plain stretch uses: with the stars gone the top of
    # the histogram is the galaxy's own bright body, and 99.5 lands inside the
    # disk (measured on a real stack: white 338 vs 679 at 99.9 vs 1177 for the
    # full image) -- the core blows out and the noise floor is stretched to
    # grain. 99.9 keeps the arms and outer disk visible without either.
    white, _sp_scale = _preview_white(lum_s, med, sigma, 99.9, black)
    ghs_sp = ghs_sp * _sp_scale
    if color == 'preserve':
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        out = colour_preserving_stretch(
            rgb, generalized_hyperbolic_stretch(
                lum, b=ghs_b, SP=ghs_sp, LP=0.0, HP=ghs_hp,
                black_point=black, white_point=white),
            black, white, sigma)
    else:
        out = np.zeros(rgb.shape, dtype=np.float32)
        for c in range(3):
            out[:, :, c] = generalized_hyperbolic_stretch(
                rgb[:, :, c], b=ghs_b, SP=ghs_sp, LP=0.0, HP=ghs_hp,
                black_point=black, white_point=white)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


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
                     ghs_hp: float = 0.95, black_sigma: float = 0.0,
                     starless: Optional[np.ndarray] = None,
                     color: str = 'preserve') -> None:
    """Write the preview JPEG. ``starless`` (same shape as ``rgb``), with
    ``stretch='ghs'``, switches to the layered stretch -- see
    ``render_preview_layered_uint8``."""
    from src.models import Config
    if Image is None:
        return
    if starless is not None and stretch == 'ghs' and starless.shape == rgb.shape:
        out = render_preview_layered_uint8(rgb, starless, ghs_b=ghs_b, ghs_sp=ghs_sp,
                                           ghs_hp=ghs_hp, black_sigma=black_sigma,
                                           color=color)
    else:
        out = render_preview_uint8(rgb, stretch=stretch, ghs_b=ghs_b, ghs_sp=ghs_sp,
                                   ghs_hp=ghs_hp, black_sigma=black_sigma, color=color)
    if out is None:
        return
    img = _preview_pil_image(out, Config.PREVIEW_MAX_DIMENSION)
    img.save(path, format='JPEG', quality=Config.PREVIEW_JPEG_QUALITY)


def _desaturate_preview_uint8(out: np.ndarray, amount: float) -> np.ndarray:
    """Blend a stretched uint8 HWC preview toward its own per-pixel luminance.

    A single unstacked sub (one Phase 1 frame, no rejection-combine averaging
    yet) from a noisy/light-polluted session has real per-pixel photon/read
    noise that differs randomly across R/G/B -- stretching each channel
    independently (as ``render_preview_uint8`` does, to preserve real colour
    ratios) amplifies that into random per-pixel colour speckle. Downsized to
    a small ring thumbnail, that speckle averages into a solid, misleading
    colour cast (a noisy low-SNR sub can render as a near-solid green/blue
    blob) instead of the star field it actually is. Blending toward luminance
    turns the speckle back into visible gray-noise texture without touching
    the shared full-resolution stretch path other previews use."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0 or out.ndim != 3 or out.shape[2] != 3:
        return out
    lum = (0.299 * out[:, :, 0] + 0.587 * out[:, :, 1]
           + 0.114 * out[:, :, 2]).astype(np.float32)
    blended = out.astype(np.float32) * (1.0 - amount) + lum[:, :, None] * amount
    return np.clip(blended, 0, 255).astype(np.uint8)


def preview_jpeg_bytes(rgb: np.ndarray, stretch: str = 'ghs',
                       ghs_b: float = 8.0, ghs_sp: float = 0.15,
                       ghs_hp: float = 0.95, black_sigma: float = 0.0,
                       max_dim: int = 1024, desaturate: float = 0.0,
                       color: str = 'preserve') -> Optional[bytes]:
    """Stretched preview JPEG as bytes (for the live web view).

    ``desaturate`` (0-1) blends the stretched result toward luminance -- see
    ``_desaturate_preview_uint8``. 0 (default) preserves full colour, as
    every non-thumbnail caller wants."""
    import io as _io
    if Image is None:
        return None
    out = render_preview_uint8(rgb, stretch=stretch, ghs_b=ghs_b, ghs_sp=ghs_sp,
                               ghs_hp=ghs_hp, black_sigma=black_sigma, color=color)
    if out is None:
        return None
    out = _desaturate_preview_uint8(out, desaturate)
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
    header['CREATOR'] = ('originstack.py', 'Software that created this file')
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
        # TIMEZONE travels with DATE-OBS: Origin stamps DATE-OBS in local time
        # and readers (utils.obs_time_utc_iso) need the offset to get UTC.
        copy_keys = ['TELESCOP', 'INSTRUME', 'OBSERVER', 'OBJECT', 'DATE-OBS', 'TIMEZONE',
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
            # WCS from celestial + FOV + orientation (only when plate solve has not run).
            # Prefer the one pipeline._settle_stack_wcs mapped onto this stack's
            # grid (and refined against Gaia): the raw session WCS describes the
            # first sub, not the registered, cropped stack.
            _stack_wcs = getattr(args, '_stack_wcs', None)
            if (si.has_wcs or _stack_wcs) and 'CTYPE1' not in header:
                from src.session_info import build_wcs_keywords
                wcs = _stack_wcs or build_wcs_keywords(si)
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
    denoise_applied = post_processed and getattr(args, 'denoise_curvelet', False)
    header['DENOISE'] = (denoise_applied, 'Wavelet denoising applied to FITS data')
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
