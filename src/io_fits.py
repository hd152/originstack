"""FITS I/O utilities: loading, saving, master frame creation, preview generation."""
from __future__ import annotations

import argparse
import os
import tempfile
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

from src.debayer import luminance
from src.models import FrameInfo, ProcessingStats


def sigma_clipped_stats(*args, **kwargs):
    # astropy.stats costs ~0.3 s to import; only the preview/sky paths need it.
    from astropy.stats import sigma_clipped_stats as _scs
    return _scs(*args, **kwargs)

try:
    from PIL import Image
except Exception:
    Image = None


# Every preview JPEG keeps full-resolution colour (4:4:4). Pillow's default at quality >= 50 is
# 4:2:0, which stores colour at half resolution: a 2-4 px star's colour is averaged with the
# black sky around it -- star colour error 3x, saturation kept 61-78% instead of 96-99% on the
# Sunflower / Omega / Pleiades previews, for files 15-32% larger.
JPEG_SAVE_KW = {'subsampling': 0}


def to_uint8(x: np.ndarray) -> np.ndarray:
    """A [0, 1] display image as uint8, rounded. ``astype`` truncates: every pixel came out up to
    one step darker (0.05-0.11 steps on average over a preview's sky)."""
    return np.clip(np.rint(np.asarray(x, np.float32) * np.float32(255.0)), 0, 255).astype(np.uint8)


def _rescale_intensity(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly rescale ``x`` from [lo, hi] to [0, 1], clipping outside that
    range -- equivalent to skimage.exposure.rescale_intensity(x, in_range=(lo,
    hi)) with its default out_range for a float array."""
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip((x.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)


def _sky_stats(lum: np.ndarray, noise_scale: float = 1.0) -> Tuple[float, float]:
    """Robust (median, sigma) of the sky background via 2.5-sigma /
    3-iteration clipping -- shared by the ghs and arcsinh preview stretch
    branches below. ``noise_scale`` multiplies the sigma: the stretch is chosen
    from the sky noise, and after ``--denoiser nn`` (which removes ~90% of it)
    the measured sigma would make it far stronger than the same data without
    denoising -- the pipeline passes the noise the image had before the step."""
    _, med, sigma = sigma_clipped_stats(lum, sigma=2.5, maxiters=3)
    med = float(med)
    sigma = float(sigma) if np.isfinite(sigma) and sigma > 0 else 1.0
    return med, sigma * float(noise_scale)


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


OBJECT_DOWNSAMPLE = 4        # px per block for finding the object (speed; stars are median-filtered out)
OBJECT_STAR_FILTER = 5       # median filter size in blocks: removes stars up to ~10 px radius
OBJECT_SIGMA = 4.0           # object = smoothed luminance this many smoothed-sky sigmas above the sky
OBJECT_MIN_FRAC = 0.0005     # smaller than this share of the frame: a star halo, not an object
OBJECT_MAX_FRAC = 0.35       # larger: the object fills the frame and plain 'ghs' is the right stretch
OBJECT_NEAR_FRAC = 0.15      # an object not under the anchor must lie within this share of the diagonal
OBJECT_MIN_CONTRAST = 15.0   # its core must stand this many smoothed-sky sigmas above the sky


def _object_region(lum: np.ndarray, sky: float = 0.0, anchor: Optional[Tuple[float, float]] = None):
    """The target's extended light, found on the image itself: ``(core, mid, area_frac)``
    in luminance units -- its bright end (99.5th percentile of the star-free, smoothed
    light inside it), its typical level (median) and its share of the frame
    -- or None when there is no separate object (nothing extended, or it fills the frame).

    Stars are median-filtered out at quarter resolution; the object is the connected
    region more than ``OBJECT_SIGMA`` smoothed-sky sigmas above the sky that contains
    ``anchor`` (row, col in full-resolution pixels: where the telescope pointed), or the
    nearest such region within ``OBJECT_NEAR_FRAC`` of the diagonal. Without an anchor the
    frame centre stands in: the Origin centres its target. (``sky`` is unused: the sky
    level is measured on the smoothed image itself.)"""
    from scipy import ndimage as ndi
    h, w = lum.shape
    f = OBJECT_DOWNSAMPLE
    hb, wb = h // f, w // f
    if hb < 16 or wb < 16:
        return None
    small = lum[:hb * f, :wb * f].reshape(hb, f, wb, f).mean(axis=(1, 3)).astype(np.float32)
    smooth = ndi.gaussian_filter(ndi.median_filter(small, size=OBJECT_STAR_FILTER), 1.5)
    # the smoothed sky's own level and noise (clipped, like _sky_stats): the MAD of the
    # lower 60% alone underestimated the noise and found "objects" in empty sky
    _, sky_s, sig = sigma_clipped_stats(smooth, sigma=2.5, maxiters=5)
    sky_s, sig = float(sky_s), float(sig)
    if not np.isfinite(sig) or sig <= 0:
        return None
    resid = smooth - sky_s
    mask = resid > OBJECT_SIGMA * sig
    lab, n = ndi.label(mask)
    if n == 0:
        return None
    ay, ax = anchor if anchor is not None else ((h - 1) / 2.0, (w - 1) / 2.0)
    by, bx = int(np.clip(ay / f, 0, hb - 1)), int(np.clip(ax / f, 0, wb - 1))
    areas = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    k = int(lab[by, bx])
    if k == 0 or areas[k - 1] < OBJECT_MIN_FRAC * hb * wb:
        cents = ndi.center_of_mass(mask, lab, index=np.arange(1, n + 1))
        best, bestd = 0, np.inf
        for i, (cy, cx) in enumerate(cents):
            if areas[i] < OBJECT_MIN_FRAC * hb * wb:
                continue
            d = np.hypot(cy - by, cx - bx)
            if d < bestd:
                best, bestd = i + 1, d
        if best == 0 or bestd > OBJECT_NEAR_FRAC * np.hypot(hb, wb):
            return None
        k = best
    frac = float(areas[k - 1]) / (hb * wb)
    if not OBJECT_MIN_FRAC <= frac <= OBJECT_MAX_FRAC:
        return None
    vals = smooth[lab == k]
    core = float(np.percentile(vals, 99.5))
    if core - sky_s < OBJECT_MIN_CONTRAST * sig:
        return None                          # a noise or halo blob, not a target
    return core, float(np.median(vals)), frac


def _object_stretch_points(lum: np.ndarray, sky: float, sigma: float, black: float,
                           white: float, sp: float,
                           anchor: Optional[Tuple[float, float]] = None
                           ) -> Tuple[float, float]:
    """``(white, SP)`` for ``--stretch object``, starting from the plain 'ghs' ones.

    The whole-frame white point (99.5th percentile) is set by the field: a faint galaxy
    whose core sits below it ends up in the bottom sliver of the display range. Here the
    white point comes down to just above the object's own core, and the symmetry point,
    where the curve puts most of its contrast, moves up to the object's typical (median)
    level. Only ever *down* for white and *up* for SP: raising white to a bright core
    darkened the whole of M42 (bright extended cores are the highlight roll-off's job,
    ``_extended_highlight_top``), and an SP at the object's faint edge, a few sigma above
    the sky, put the strongest contrast on the sky noise (sky noise +55% on the Black Eye,
    blotch +50%). No object, or the plain points already right: the plain 'ghs' ones."""
    reg = _object_region(lum, sky, anchor)
    if reg is None:
        return white, sp
    core, mid, _ = reg
    w_obj = max(sky + 1.15 * (core - sky), black + 20.0 * sigma)
    w_new = min(white, w_obj)
    sp_obj = (mid - black) / max(w_new - black, 1e-12)
    sp_new = float(min(max(sp, sp_obj), 0.25)) if w_new < white else sp
    return w_new, sp_new


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


class _Rolloff(NamedTuple):
    """Highlight roll-off: white..``top`` maps into [``knee``, ``ceiling``]
    (``ceiling`` only in the colour-preserving stretch; 1 per channel). Both
    are a float, or a per-pixel map that differs from the default only on the
    object (``_extended_highlight_top``)."""
    top: float
    knee: Union[float, np.ndarray]
    ceiling: Union[float, np.ndarray]


def _extended_highlight_top(lum: np.ndarray, white: float, sky: Optional[float] = None,
                            sky_sigma: Optional[float] = None,
                            rgb: Optional[np.ndarray] = None,
                            black: Optional[float] = None) -> Optional[_Rolloff]:
    """The highlight roll-off, or None when there is nothing to roll off.

    A white point at the 99.5th percentile clips star cores, which is fine, but
    also any bright *extended* region above it -- a nebula core such as the
    Trapezium region of M42 then renders as a flat white blob. That case is told
    apart from star cores by the largest connected region above white (see
    ``Config.PREVIEW_ROLLOFF_MIN_AREA``); the top is the 99.99th percentile.
    With ``sky``/``sky_sigma`` (and ``rgb``/``black`` for the ceiling) the knee
    and ceiling follow how much of the object is above white
    (``Config.PREVIEW_ROLLOFF_TARGET_KNEE``), on the object only: a lower knee
    everywhere dimmed every star of the M57 frame to make room for the ring."""
    from scipy import ndimage

    from src.models import Config
    above = lum > white
    if not above.any():
        return None
    labels, n = ndimage.label(above)
    min_area = max(Config.PREVIEW_ROLLOFF_MIN_AREA * lum.size, Config.PREVIEW_ROLLOFF_MIN_PIXELS)
    if n == 0:
        return None
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    if areas.max() < min_area:
        return None
    top = float(np.percentile(lum, 99.99))
    if top <= 1.2 * float(white):
        return None
    knee, ceiling = Config.PREVIEW_ROLLOFF_KNEE, 1.0
    if sky is not None and sky_sigma is not None:
        region = labels == int(areas.argmax())
        obj_labels, _ = ndimage.label(lum > float(sky) + 10.0 * float(sky_sigma))
        ids = np.unique(obj_labels[region])
        obj = np.isin(obj_labels, ids[ids > 0])
        frac = float((above & obj).sum()) / max(int(obj.sum()), 1)
        lo, hi = Config.PREVIEW_ROLLOFF_TARGET_FRACTION
        t = float(_smoothstep(np.asarray((frac - lo) / (hi - lo))))
        if t > 0.0:
            # feathered object mask, so the curve changes smoothly into its halo
            weight = t * np.clip(ndimage.gaussian_filter(
                ndimage.binary_dilation(obj, iterations=3).astype(np.float64), 3.0), 0.0, 1.0)
            target = Config.PREVIEW_ROLLOFF_TARGET_KNEE
            top_c = 1.0
            if rgb is not None and black is not None:
                excess = np.maximum(lum[region].astype(np.float64) - float(black), 1e-12)
                ratio = float(np.median((rgb[region].astype(np.float64) - float(black)).max(axis=1)
                                        / excess))
                top_c = max(min(1.0 / max(ratio, 1e-12), 1.0), target + 0.1)
            knee = knee + weight * (target - knee)
            ceiling = 1.0 - weight * (1.0 - top_c)
    return _Rolloff(top, knee, ceiling)


def _rolloff_curve(curve, lum: np.ndarray, white: float, roll: _Rolloff,
                   ceiling: bool = False) -> np.ndarray:
    """``curve`` (a luminance -> [0, 1] stretch with white point ``white``) scaled
    into [0, knee], with a log roll-off mapping white..top into [knee, 1] (into
    [knee, ``roll.ceiling``] with ``ceiling``)."""
    from src.models import Config
    knee = roll.knee
    hi = roll.ceiling if ceiling else 1.0
    a = Config.PREVIEW_ROLLOFF_STRENGTH
    below = np.asarray(curve(np.minimum(lum, white).astype(np.float32)), dtype=np.float64)
    over = np.clip((lum.astype(np.float64) - white) / (roll.top - white), 0.0, 1.0)
    return np.where(lum <= white, knee * below,
                    knee + (hi - knee) * np.log1p(a * over) / np.log1p(a)).astype(np.float32)


def _extended_nebula_ghs(lum: np.ndarray, sky: float, sky_sigma: float, black: float,
                         white: float, b: float, sp: float, hp: float) -> Tuple[float, float]:
    """GHS stretch factor and symmetry point for a frame-filling bright nebula.

    The curve is ``asinh(b (x - SP))``, which lifts the faintest signal only
    ~b / asinh(b) times: about 3x at the presets' b = 6-9. That suits a galaxy,
    whose white point sits a few times above its disk, but on a real Orion
    stack the white point is ~70k against outer wings a few hundred ADU above
    sky -- the median above-sky pixel displayed at 0.09 and the whole outer
    nebula, plain in any auto-stretched view of the FITS, rendered black.

    For such an image (the highlight roll-off case) SP goes to 0 and b rises
    until the lower quartile of the clearly-above-sky pixels displays at
    ``Config.PREVIEW_NEBULA_FAINT_LEVEL``. Never weaker than the given curve."""
    from src.models import Config
    sub = lum[::2, ::2]
    vals = sub[sub > sky + 3.0 * sky_sigma]
    span = float(white) - float(black)
    if vals.size < 100 or span <= 0:
        return b, sp
    q = (float(np.percentile(vals, 25)) - float(black)) / span
    if not 0.0 < q < hp:
        return b, sp

    def level(bb: float, s: float) -> float:
        lo = np.arcsinh(-bb * s)
        return float((np.arcsinh(bb * (q - s)) - lo) / (np.arcsinh(bb * (hp - s)) - lo))

    target = Config.PREVIEW_NEBULA_FAINT_LEVEL / Config.PREVIEW_ROLLOFF_KNEE
    if level(b, sp) >= target:
        return b, sp
    lo_b, hi_b = float(b), Config.PREVIEW_NEBULA_MAX_B
    if level(hi_b, 0.0) < target:
        return hi_b, 0.0
    for _ in range(50):
        mid = 0.5 * (lo_b + hi_b)
        if level(mid, 0.0) < target:
            lo_b = mid
        else:
            hi_b = mid
    return hi_b, 0.0


def _preserving_preview(rgb: np.ndarray, lum: np.ndarray, curve, black: float,
                        white: float, sky_sigma: float, top=False,
                        apply=None) -> np.ndarray:
    """Colour-preserving preview of ``rgb`` with the luminance curve ``curve``,
    rolling off an extended highlight above ``white`` instead of clipping it.
    ``top``: a precomputed ``_extended_highlight_top`` (False: compute it here).
    ``apply``: the colour step (default ``colour_preserving_stretch``)."""
    apply = apply or colour_preserving_stretch
    if top is False:
        top = _extended_highlight_top(lum, white)
    if top is None:
        return apply(rgb, curve(lum), black, white, sky_sigma)
    return apply(rgb, _rolloff_curve(curve, lum, white, top, ceiling=True), black, top.top,
                 sky_sigma)


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _masked_stretch_weight(lum: np.ndarray, sky: float, sky_sigma: float,
                           white: float) -> np.ndarray:
    """Softly select faint, above-sky structure for the masked preview stretch."""
    from scipy import ndimage

    smoothed = ndimage.gaussian_filter(lum.astype(np.float64), sigma=2.0)
    faint_start = sky + 2.0 * sky_sigma
    faint_full = sky + 4.0 * sky_sigma
    bright_start = max(sky + 12.0 * sky_sigma, sky + 0.35 * (white - sky))
    bright_end = max(float(white), bright_start + sky_sigma)

    faint = _smoothstep((smoothed - faint_start) / max(faint_full - faint_start, 1e-12))
    highlights = 1.0 - _smoothstep(
        (smoothed - bright_start) / max(bright_end - bright_start, 1e-12))
    return ndimage.gaussian_filter(faint * highlights, sigma=2.0).astype(np.float32)


def _multiscale_stretch_weights(lum: np.ndarray, sky: float, sky_sigma: float,
                                white: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return diffuse-emission and fine-filament weights for preview enhancement."""
    from scipy import ndimage

    source = lum.astype(np.float64)
    broad = ndimage.gaussian_filter(source, sigma=8.0)
    medium = ndimage.gaussian_filter(source, sigma=5.0)
    fine = ndimage.gaussian_filter(source, sigma=1.5)

    faint_start = sky + 2.0 * sky_sigma
    faint_full = sky + 4.0 * sky_sigma
    bright_start = max(sky + 12.0 * sky_sigma, sky + 0.35 * (white - sky))
    bright_end = max(float(white), bright_start + sky_sigma)

    def emission_weight(signal: np.ndarray) -> np.ndarray:
        faint = _smoothstep((signal - faint_start) / max(faint_full - faint_start, 1e-12))
        highlights = 1.0 - _smoothstep(
            (signal - bright_start) / max(bright_end - bright_start, 1e-12))
        return faint * highlights

    diffuse_weight = emission_weight(broad)
    # Positive fine-minus-medium residuals select compact bright ridges while
    # the diffuse mask prevents noise or unrelated stars from becoming targets.
    detail = np.maximum(fine - medium, 0.0)
    detail_weight = _smoothstep(
        (detail - 1.0 * sky_sigma) / max(3.0 * sky_sigma, 1e-12))
    filament_weight = diffuse_weight * detail_weight
    diffuse_weight = ndimage.gaussian_filter(diffuse_weight, sigma=3.0)
    filament_weight = ndimage.gaussian_filter(filament_weight, sigma=1.0)
    return (np.clip(diffuse_weight, 0.0, 1.0).astype(np.float32),
            np.clip(filament_weight, 0.0, 1.0).astype(np.float32))


def _statistical_stretch(normalized: np.ndarray, image_median: float,
                         target_median: float = 0.2) -> np.ndarray:
    """Map a normalized image median to a chosen display median."""
    values = np.clip(np.asarray(normalized, dtype=np.float64), 0.0, 1.0)
    median = float(np.clip(image_median, 1e-12, 1.0))
    target = float(np.clip(target_median, 0.01, 0.99))
    if median >= 1.0 - 1e-12:
        return values.astype(np.float32)
    denominator = median * (target + values - 1.0) - target * values
    numerator = (median - 1.0) * target * values
    stretched = np.divide(numerator, denominator, out=np.zeros_like(values),
                          where=np.abs(denominator) > 1e-12)
    stretched[values >= 1.0] = 1.0
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def _noise_limited_b(sky: float, sky_sigma: float, black: float, white: float,
                     hp: float) -> float:
    """GHS stretch factor (SP = LP = 0) at which one sky sigma spans
    ``Config.PREVIEW_NOISE_LEVELS`` display levels at the sky.

    The curve is ``asinh(b x) / asinh(b HP)``. Its slope at the sky rises
    with b while ``b x_sky << 1`` and falls once the sky moves into the
    logarithmic part of the curve, so the noise amplitude peaks at a finite b:
    the smallest b reaching the target is taken (the weakest curve that shows
    the noise at that size), or the peak when no b reaches it."""
    from src.models import Config
    span = float(white) - float(black)
    if span <= 0 or sky_sigma <= 0:
        return 0.0
    target = Config.PREVIEW_NOISE_LEVELS / 255.0
    x = (float(sky) - float(black)) / span
    d = float(sky_sigma) / span

    def amplitude(b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=np.float64)
        norm = np.arcsinh(b * hp)
        return 0.5 * (np.arcsinh(b * min(x + d, hp))
                      - np.arcsinh(b * max(x - d, 0.0))) / norm

    linear = 0.5 * (min(x + d, hp) - max(x - d, 0.0)) / hp   # the b -> 0 limit
    if linear >= target:
        return 0.0
    grid = np.logspace(-3.0, np.log10(Config.PREVIEW_NEBULA_MAX_B), 400)
    amp = amplitude(grid)
    above = np.nonzero(amp >= target)[0]
    if above.size == 0:
        return float(grid[int(np.argmax(amp))])
    i = int(above[0])
    lo, hi = (float(grid[i - 1]) if i > 0 else 0.0), float(grid[i])
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if amplitude(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


_SRGB_TO_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                         [0.2119034982, 0.6806995451, 0.1073969566],
                         [0.0883024619, 0.2817188376, 0.6299787005]])
_LMS_TO_OKLAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                          [1.9779984951, -2.4285922050, 0.4505937099],
                          [0.0259040371, 0.7827717662, -0.8086757660]])
_OKLAB_TO_LMS = np.linalg.inv(_LMS_TO_OKLAB)
_LMS_TO_SRGB = np.linalg.inv(_SRGB_TO_LMS)


def _linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Linear sRGB (HWC) -> OKLab (Ottosson 2020). ``np.cbrt`` keeps the sign
    of a slightly negative (sky noise) channel."""
    return np.cbrt(rgb @ _SRGB_TO_LMS.T) @ _LMS_TO_OKLAB.T


def _oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    return ((lab @ _OKLAB_TO_LMS.T) ** 3) @ _LMS_TO_SRGB.T


def _srgb_decode(v: np.ndarray) -> np.ndarray:
    """sRGB display value -> linear, extended beyond [0, 1] (odd, unbounded)
    for colour directions read off an over-range display pixel."""
    a = np.abs(v)
    return np.sign(v) * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def _srgb_encode(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def oklab_stretch(rgb: np.ndarray, lum_stretched: np.ndarray, black: float,
                  white: float, sky_sigma: float) -> np.ndarray:
    """Apply a luminance curve in OKLab: lightness from the curve, hue and
    saturation (chroma / lightness) from the colour the 'ghs' preview shows.

    ``colour_preserving_stretch`` keeps RGB ratios, so a saturated colour
    whose luminance maps near 1 has a channel over range and the whole pixel
    is dimmed to fit (a red star core displays at half the brightness of a
    white one); colours of equal luminance also differ in perceived
    lightness. Here the target lightness is the curve's display value read as
    a grey (so a grey pixel renders exactly as with the 'ghs' stretch). Hue
    and C/L come from that function's display pixel before its gamut fit --
    not from the linear colour: the ratios it keeps are written as display
    (gamma-encoded) values, which is more saturated than the linear colour
    (a synthetic red star's wing: saturation 0.67 vs 0.41), and the existing
    previews' colour is the look to keep. Near white
    the chroma is compressed by up to ``Config.PREVIEW_OKLAB_MAX_DESAT``
    (over ``PREVIEW_OKLAB_KNEE``..1); whatever is still out of gamut is fitted
    by dimming (as Lupton et al.), so star cores keep some colour instead of
    turning white. Chroma fades in over the first 3 sky sigma above black, as
    in ``colour_preserving_stretch``.
    """
    from src.models import Config
    span = max(float(white) - float(black), 1e-12)
    cn = (rgb.astype(np.float64) - float(black)) / span
    n = cn @ np.array([0.299, 0.587, 0.114])
    T = np.clip(np.asarray(lum_stretched, dtype=np.float64), 0.0, 1.0)
    pos = n > 1e-9
    shown = cn * np.where(pos, T / np.where(pos, n, 1.0), 0.0)[..., None]
    lab = _linear_to_oklab(_srgb_decode(shown))
    L0 = lab[..., 0]
    sat = np.hypot(lab[..., 1], lab[..., 2]) / np.maximum(L0, 1e-6)
    sat = np.where(L0 > 1e-6, np.minimum(sat, Config.PREVIEW_OKLAB_MAX_SATURATION), 0.0)
    hue = np.arctan2(lab[..., 2], lab[..., 1])
    L = np.cbrt(_srgb_decode(T))
    knee = Config.PREVIEW_OKLAB_KNEE
    desat = 1.0 - Config.PREVIEW_OKLAB_MAX_DESAT * _smoothstep(
        (T - knee) / max(1.0 - knee, 1e-12))
    gate = np.clip(n / max(3.0 * float(sky_sigma) / span, 1e-9), 0.0, 1.0)
    C = sat * L * desat * gate
    out = _oklab_to_linear(np.stack([L, C * np.cos(hue), C * np.sin(hue)], axis=-1))
    # A negative channel: desaturate toward the grey of the same lightness.
    grey = (L ** 3)[..., None]
    mn = out.min(axis=2, keepdims=True)
    s_lo = np.where(mn < 0.0, grey / np.maximum(grey - mn, 1e-12), 1.0)
    out = grey + np.clip(s_lo, 0.0, 1.0) * (out - grey)
    out = out / np.maximum(out.max(axis=2, keepdims=True), 1.0)
    return _srgb_encode(out).astype(np.float32)


def _clahe(values: np.ndarray, tiles: int, clip: float, bins: int = 256) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation of a [0, 1] plane.

    Each of ``tiles`` x ``tiles`` tiles gets a histogram clipped at ``clip``
    times the uniform bin count (the excess spread over all bins) and its
    CDF as a tone map; every pixel is mapped by bilinear interpolation of the
    four nearest tile maps (Zuiderveld 1994), so there are no tile seams."""
    h, w = values.shape
    ty, tx = max(1, min(tiles, h // 8)), max(1, min(tiles, w // 8))
    idx = np.clip((values * (bins - 1) + 0.5).astype(np.intp), 0, bins - 1)
    ys = np.linspace(0, h, ty + 1).astype(int)
    xs = np.linspace(0, w, tx + 1).astype(int)
    maps = np.empty((ty, tx, bins), dtype=np.float64)
    for i in range(ty):
        for j in range(tx):
            block = idx[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            hist = np.bincount(block.ravel(), minlength=bins).astype(np.float64)
            limit = max(clip * block.size / bins, 1.0)
            excess = np.maximum(hist - limit, 0.0).sum()
            hist = np.minimum(hist, limit) + excess / bins
            cdf = np.cumsum(hist)
            maps[i, j] = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-12)
    # Fractional tile coordinates of each pixel, relative to the tile centres.
    cy = (np.arange(h) + 0.5) / h * ty - 0.5
    cx = (np.arange(w) + 0.5) / w * tx - 0.5
    y0 = np.clip(np.floor(cy).astype(int), 0, ty - 1)
    x0 = np.clip(np.floor(cx).astype(int), 0, tx - 1)
    y1 = np.minimum(y0 + 1, ty - 1)
    x1 = np.minimum(x0 + 1, tx - 1)
    fy = np.clip(cy - y0, 0.0, 1.0)[:, None]
    fx = np.clip(cx - x0, 0.0, 1.0)[None, :]
    Y0, X0 = y0[:, None], x0[None, :]
    Y1, X1 = y1[:, None], x1[None, :]
    top = maps[Y0, X0, idx] * (1.0 - fx) + maps[Y0, X1, idx] * fx
    bottom = maps[Y1, X0, idx] * (1.0 - fx) + maps[Y1, X1, idx] * fx
    return (top * (1.0 - fy) + bottom * fy).astype(np.float32)


def _clahe_weight(lum: np.ndarray, sky: float, sky_sigma: float) -> np.ndarray:
    """Where CLAHE may act: smoothed signal clearly above sky, minus stars.

    Stars are compact sources standing well above their surroundings (fine
    scale minus broad scale above 8 sky sigma), grown by a few pixels so the
    equalisation does not lift their wings into halos."""
    from scipy import ndimage
    src = lum.astype(np.float64)
    smooth = ndimage.gaussian_filter(src, sigma=3.0)
    signal = _smoothstep((smooth - (sky + 2.0 * sky_sigma)) / max(3.0 * sky_sigma, 1e-12))
    compact = (ndimage.gaussian_filter(src, sigma=1.0)
               - ndimage.gaussian_filter(src, sigma=6.0)) > 8.0 * sky_sigma
    stars = ndimage.binary_dilation(compact, iterations=4)
    weight = signal * (1.0 - ndimage.gaussian_filter(stars.astype(np.float64), sigma=2.0))
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def _apply_display_luminance(out: np.ndarray, lum_new: np.ndarray) -> np.ndarray:
    """Give a display image (HWC, [0, 1]) the luminance ``lum_new``: by
    scaling where it is clearly above black, fading to an additive change near
    zero (CLAUDE.md: never ``img * new_lum / lum`` where lum ~ 0)."""
    lum = out @ np.array([0.299, 0.587, 0.114], dtype=out.dtype)
    delta = (lum_new - lum)[..., None]
    ratio = (lum_new / np.maximum(lum, 1e-6))[..., None]
    f = _smoothstep((lum - 0.02) / 0.06)[..., None]
    res = f * (out * ratio) + (1.0 - f) * (out + delta)
    res = np.maximum(res, 0.0)
    res = res / np.maximum(res.max(axis=2, keepdims=True), 1.0)
    return res.astype(np.float32)


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
                         color: str = 'preserve',
                         noise_scale: float = 1.0,
                         anchor: Optional[Tuple[float, float]] = None) -> Optional[np.ndarray]:
    """Stretch an HWC float32 image to display uint8 (the shared core of the
    preview JPEG file writer and the live web view). Returns None when the
    required stretch backend is unavailable."""
    out = render_preview_float(rgb, stretch=stretch, ghs_b=ghs_b, ghs_sp=ghs_sp,
                               ghs_hp=ghs_hp, black_sigma=black_sigma, color=color,
                               noise_scale=noise_scale, anchor=anchor)
    if out is None:
        return None
    return to_uint8(out)


def render_preview_float(rgb: np.ndarray, stretch: str = 'linear',
                         ghs_b: float = 8.0, ghs_sp: float = 0.15,
                         ghs_hp: float = 0.95,
                         black_sigma: float = 0.0,
                         color: str = 'preserve',
                         noise_scale: float = 1.0,
                         anchor: Optional[Tuple[float, float]] = None) -> Optional[np.ndarray]:
    """The display stretch of ``render_preview_uint8`` before quantisation:
    HWC float in [0, 1]. ``tools/bench_phase4.py`` scores this.

    ``color`` ('ghs', 'arcsinh', and 'seti'): 'preserve' (default) applies the
    curve to luminance and keeps each pixel's RGB ratios
    (``colour_preserving_stretch``); 'channel' applies it to each channel
    separately (the behaviour before 2026-10, which whitens stars and bright
    cores). SETI statistical stretch targets a normalized median of 0.20.

    'noisefloor' is GHS with b solved from the sky noise (``_noise_limited_b``)
    and its own black point; 'oklab' is the GHS curve on OKLab lightness
    (``oklab_stretch``, always colour-preserving); 'clahe' is the GHS preview
    plus contrast-limited local equalisation where there is extended signal; 'object' is
    the GHS preview with its white point and symmetry point taken from the target itself
    (``_object_stretch_points``; ``anchor`` = where the telescope pointed), and plain
    'ghs' when there is no separate object."""
    from src.denoising import arcsinh_stretch, generalized_hyperbolic_stretch
    from src.models import Config
    if stretch in ('ghs', 'masked', 'multiscale', 'oklab', 'clahe', 'object'):
        # Generalized Hyperbolic Stretch — uses unified luminance-based normalization
        # so all three channels share the same black/white reference, preserving
        # cross-channel color ratios that would otherwise be destroyed by independent
        # per-channel sky statistics.
        out = np.zeros_like(rgb)
        lum = luminance(rgb)
        _med, _bg_sigma = _sky_stats(lum, noise_scale)
        # Black point relative to the sky median in units of sky sigma -- the noise
        # actually left in the image: after --denoiser nn, black_sigma times the
        # pre-denoising sigma clipped the whole (now clean) sky to black.
        _black_sigma_px = _bg_sigma / noise_scale
        # black_sigma < 0 keeps sky noise visible (good for frame-filling faint
        # nebulae); black_sigma > 0 clips the noise floor to black (good for a
        # small target on empty sky, e.g. a galaxy or cluster, where a low black
        # point turns the whole background into a colour-noise storm). The
        # target-type advisor sets an appropriate value per preset.
        unified_black = _med + black_sigma * _black_sigma_px
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
        if stretch == 'object':
            unified_white, ghs_sp = _object_stretch_points(
                lum, _med, _bg_sigma, unified_black, unified_white, ghs_sp, anchor)
        _top = None
        # Every path that ends in _preserving_preview or the per-channel curve below.
        # The per-channel one was left out while 'preserve' was the default; when the
        # default went back to 'channel' an extended core (M42's Trapezium region, all
        # of M57) clipped to a flat white blob again.
        if stretch != 'multiscale' or color == 'preserve':
            _top = _extended_highlight_top(lum, unified_white, _med, _bg_sigma, rgb,
                                           unified_black)
            # Lift the faint wings only where the preset keeps the sky noise (nebulae,
            # black_sigma <= 0): a cluster or galaxy preset asks for a dark sky, and on
            # M15 the lift brought up red background-residual blotches around the core.
            if _top is not None and black_sigma <= 0:
                ghs_b, ghs_sp = _extended_nebula_ghs(lum, _med, _bg_sigma, unified_black,
                                                     unified_white, ghs_b, ghs_sp, ghs_hp)

        def curve(x: np.ndarray) -> np.ndarray:
            return generalized_hyperbolic_stretch(
                x, b=ghs_b, SP=ghs_sp, LP=0.0, HP=ghs_hp,
                black_point=unified_black, white_point=unified_white)

        if stretch in ('masked', 'multiscale'):
            base = _preserving_preview(rgb, lum, curve, unified_black, unified_white,
                                       _bg_sigma, top=_top)
            span = max(unified_white - unified_black, 1e-12)

            def faint_curve(x: np.ndarray) -> np.ndarray:
                normalized = np.clip((x.astype(np.float64) - unified_black) / span, 0.0, 1.0)
                return np.power(normalized, 0.35).astype(np.float32)

            boosted = _preserving_preview(
                rgb, lum, faint_curve, unified_black, unified_white, _bg_sigma, top=_top)
            if stretch == 'masked':
                mask = _masked_stretch_weight(lum, _med, _bg_sigma, unified_white)
                out = base + (boosted - base) * (0.65 * mask[..., None])
            else:
                diffuse_weight, filament_weight = _multiscale_stretch_weights(
                    lum, _med, _bg_sigma, unified_white)
                out = base + (boosted - base) * (
                    0.45 * diffuse_weight[..., None] + 0.5 * filament_weight[..., None])
        elif stretch == 'oklab' or color == 'preserve':
            out = _preserving_preview(rgb, lum, curve, unified_black, unified_white,
                                      _bg_sigma, top=_top,
                                      apply=oklab_stretch if stretch == 'oklab' else None)
        elif _top is not None:
            for c in range(3):
                out[:, :, c] = _rolloff_curve(curve, rgb[:, :, c], unified_white, _top)
        else:
            for c in range(3):
                out[:, :, c] = curve(rgb[:, :, c])
        if stretch == 'clahe':
            # Local contrast on top of the plain GHS preview: equalise its
            # display luminance per tile and blend that in where there is
            # extended signal (not sky noise, not stars).
            disp = out @ np.array([0.299, 0.587, 0.114], dtype=out.dtype)
            eq = _clahe(np.clip(disp, 0.0, 1.0), Config.PREVIEW_CLAHE_TILES,
                        Config.PREVIEW_CLAHE_CLIP)
            weight = _clahe_weight(lum, _med, _bg_sigma)
            out = _apply_display_luminance(
                out, disp + Config.PREVIEW_CLAHE_STRENGTH * weight * (eq - disp))
    elif stretch == 'noisefloor':
        # GHS whose strength is solved from the sky noise (_noise_limited_b)
        # instead of the preset's b / SP / black point.
        out = np.zeros_like(rgb)
        lum = luminance(rgb)
        _med, _bg_sigma = _sky_stats(lum, noise_scale)
        black = _med - Config.PREVIEW_NOISE_BLACK_SIGMA * _bg_sigma
        white, _ = _preview_white(lum, _med, _bg_sigma, 99.5, black)
        nb = _noise_limited_b(_med, _bg_sigma, black, white, ghs_hp)

        def noise_curve(x: np.ndarray) -> np.ndarray:
            return generalized_hyperbolic_stretch(x, b=nb, SP=0.0, LP=0.0, HP=ghs_hp,
                                                  black_point=black, white_point=white)
        if color == 'preserve':
            out = _preserving_preview(rgb, lum, noise_curve, black, white, _bg_sigma)
        else:
            for c in range(3):
                out[:, :, c] = noise_curve(rgb[:, :, c])
    elif stretch == 'seti':
        lum = luminance(rgb)
        _med, _bg_sigma = _sky_stats(lum, noise_scale)
        black = max(float(np.min(lum)), _med - 2.7 * _bg_sigma)
        white, _ = _preview_white(lum, _med, _bg_sigma, 99.5, black)
        span = max(white - black, 1e-12)
        normalized_lum = np.clip((lum.astype(np.float64) - black) / span, 0.0, 1.0)
        linked_median = float(np.median(normalized_lum))
        top = _extended_highlight_top(lum, white, _med, _bg_sigma, rgb, black)

        def seti_curve(x: np.ndarray, median: float) -> np.ndarray:
            normalized = np.clip((x.astype(np.float64) - black) / span, 0.0, 1.0)
            return _statistical_stretch(normalized, median)

        if color == 'preserve':
            out = _preserving_preview(
                rgb, lum, lambda x: seti_curve(x, linked_median),
                black, white, _bg_sigma, top=top)
        else:
            out = np.zeros_like(rgb)
            for c in range(3):
                normalized = np.clip(
                    (rgb[:, :, c].astype(np.float64) - black) / span, 0.0, 1.0)
                channel_median = float(np.median(normalized))
                out[:, :, c] = seti_curve(rgb[:, :, c], channel_median)
    elif stretch == 'arcsinh':
        # Arcsinh stretch — unified luminance-based normalization to preserve color
        out = np.zeros_like(rgb)
        lum = luminance(rgb)
        _med, _bg_sigma = _sky_stats(lum, noise_scale)
        unified_black = _med + black_sigma * _bg_sigma / noise_scale   # as in 'ghs'
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
    return _feather_clipped_core(out, rgb)


def _feather_clipped_core(out: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Soften the edge of an extended clipped core in the display image.

    Where the sensor clipped over an extended region (M42's Trapezium region in
    every 20 s sub), the stack holds a flat plateau ending in a cliff: the light
    just outside it is a third of the plateau's within a few pixels, and the
    plateau's outline -- set by where each sub happened to clip, not by the
    nebula -- rendered as a jagged white cut-out however the highlights were
    rolled off. The plateau holds no detail, so its outline is blended into a
    blurred copy of the display over a width that scales with it (~8 px on
    M42's 70 px plateau); star cores (< ``Config.PREVIEW_CORE_FEATHER_MIN_PIXELS``)
    and the rest of the image are untouched."""
    from scipy import ndimage

    from src.models import Config
    lum = luminance(rgb)
    peak = float(lum.max())
    if not np.isfinite(peak) or peak <= 0:
        return out
    labels, n = ndimage.label(lum >= Config.PREVIEW_CORE_FEATHER_LEVEL * peak)
    if n == 0:
        return out
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    k = int(areas.argmax())
    if areas[k] < Config.PREVIEW_CORE_FEATHER_MIN_PIXELS:
        return out
    plate = labels == k
    sigma = float(np.clip(Config.PREVIEW_CORE_FEATHER_SCALE * np.sqrt(areas[k] / np.pi), 2.0, 12.0))
    ys, xs = np.nonzero(plate)
    pad = int(np.ceil(4 * sigma)) + 1
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, out.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, out.shape[1])
    win = out[y0:y1, x0:x1]
    w = np.clip(2.0 * ndimage.gaussian_filter(plate[y0:y1, x0:x1].astype(np.float32), sigma),
                0.0, 1.0)[..., None]
    blur = np.stack([ndimage.gaussian_filter(win[..., c], sigma) for c in range(win.shape[2])],
                    axis=2)
    out = np.array(out, copy=True)
    out[y0:y1, x0:x1] = win * (1.0 - w) + blur * w
    return out


def render_preview_layered_float(rgb: np.ndarray, starless: np.ndarray,
                                 ghs_b: float = 8.0, ghs_sp: float = 0.15,
                                 ghs_hp: float = 0.95, black_sigma: float = 0.0,
                                 color: str = 'preserve',
                                 noise_scale: float = 1.0) -> np.ndarray:
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
    lum_s = luminance(starless)
    med, sigma = _sky_stats(lum_s, noise_scale)
    black = med + black_sigma * sigma / noise_scale       # the noise left (see 'ghs')
    # 99.9, not the 99.5 the plain stretch uses: with the stars gone the top of
    # the histogram is the galaxy's own bright body, and 99.5 lands inside the
    # disk (measured on a real stack: white 338 vs 679 at 99.9 vs 1177 for the
    # full image) -- the core blows out and the noise floor is stretched to
    # grain. 99.9 keeps the arms and outer disk visible without either.
    white, _sp_scale = _preview_white(lum_s, med, sigma, 99.9, black)
    ghs_sp = ghs_sp * _sp_scale
    if color == 'preserve':
        lum = luminance(rgb)
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
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def render_preview_layered_uint8(rgb: np.ndarray, starless: np.ndarray, **kw) -> np.ndarray:
    """``render_preview_layered_float`` as display uint8."""
    return to_uint8(render_preview_layered_float(rgb, starless, **kw))


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
                     color: str = 'preserve',
                     noise_scale: float = 1.0,
                     anchor: Optional[Tuple[float, float]] = None,
                     exports=()) -> None:
    """Write the preview JPEG (and the lossless ``exports``, see ``write_final_images``).
    ``starless`` (same shape as ``rgb``), with ``stretch='ghs'``, switches to the layered
    stretch -- see ``render_preview_layered_float``."""
    if Image is None:
        return
    out = render_preview_rgb_float(rgb, stretch=stretch, ghs_b=ghs_b, ghs_sp=ghs_sp,
                                   ghs_hp=ghs_hp, black_sigma=black_sigma,
                                   starless=starless, color=color, noise_scale=noise_scale,
                                   anchor=anchor)
    if out is None:
        return
    write_final_images(out, path, exports)


def render_preview_rgb_uint8(rgb: np.ndarray, stretch: str = 'linear',
                             ghs_b: float = 8.0, ghs_sp: float = 0.15,
                             ghs_hp: float = 0.95, black_sigma: float = 0.0,
                             starless: Optional[np.ndarray] = None,
                             color: str = 'preserve',
                             noise_scale: float = 1.0,
                             anchor: Optional[Tuple[float, float]] = None) -> Optional[np.ndarray]:
    """The full-resolution uint8 image ``save_preview_rgb`` writes:
    ``render_preview_rgb_float`` rounded to 8 bits."""
    out = render_preview_rgb_float(rgb, stretch=stretch, ghs_b=ghs_b, ghs_sp=ghs_sp,
                                   ghs_hp=ghs_hp, black_sigma=black_sigma, starless=starless,
                                   color=color, noise_scale=noise_scale, anchor=anchor)
    return None if out is None else to_uint8(out)


def render_preview_rgb_float(rgb: np.ndarray, stretch: str = 'linear',
                             ghs_b: float = 8.0, ghs_sp: float = 0.15,
                             ghs_hp: float = 0.95, black_sigma: float = 0.0,
                             starless: Optional[np.ndarray] = None,
                             color: str = 'preserve',
                             noise_scale: float = 1.0,
                             anchor: Optional[Tuple[float, float]] = None) -> Optional[np.ndarray]:
    """The final display image in [0, 1], before any quantisation: the layered stretch
    when ``starless`` is given with ``stretch='ghs'``, else ``render_preview_float``.
    The JPEG and the ``--export png`` / ``png16`` files are all made from it."""
    if starless is not None and stretch == 'ghs' and starless.shape == rgb.shape:
        return render_preview_layered_float(rgb, starless, ghs_b=ghs_b, ghs_sp=ghs_sp,
                                            ghs_hp=ghs_hp, black_sigma=black_sigma,
                                            color=color, noise_scale=noise_scale)
    out = render_preview_float(rgb, stretch=stretch, ghs_b=ghs_b, ghs_sp=ghs_sp,
                               ghs_hp=ghs_hp, black_sigma=black_sigma, color=color,
                               noise_scale=noise_scale, anchor=anchor)
    return None if out is None else np.clip(out, 0.0, 1.0).astype(np.float32)


def preview_jpeg_from_uint8(out: np.ndarray, max_dim: int = 1024) -> bytes:
    """A stretched uint8 HWC image as size-capped viewer JPEG bytes."""
    import io as _io
    buf = _io.BytesIO()
    _preview_pil_image(out, max_dim).save(buf, format='JPEG', quality=85, **JPEG_SAVE_KW)
    return buf.getvalue()


def preview_stretch_kw(args) -> dict:
    """The preview stretch settings in ``args`` (defaults where unset, or for
    ``args=None``), as keyword arguments of ``save_preview_rgb``."""
    ns = getattr(args, '_preview_noise_scale', 1.0)    # set by Phase 4's --denoiser nn step
    return dict(
        stretch=getattr(args, 'stretch', 'ghs'),
        ghs_b=float(getattr(args, 'ghs_b', 8.0)),
        ghs_sp=float(getattr(args, 'ghs_sp', 0.15)),
        ghs_hp=float(getattr(args, 'ghs_hp', 0.95)),
        black_sigma=float(getattr(args, 'preview_black_sigma', 0.0) or 0.0),
        color=getattr(args, 'stretch_color', 'channel'),
        noise_scale=float(ns) if isinstance(ns, (int, float)) and ns > 0 else 1.0,
        anchor=object_anchor(args))


def preview_exports(args) -> tuple:
    """The lossless copies of the preview ``--export`` asked for ('png', 'png16')."""
    return tuple(e for e in ('png', 'png16') if getattr(args, f'output_{e}', False))


def object_anchor(args) -> Optional[Tuple[float, float]]:
    """(row, col) on the stack where the telescope pointed -- the session's pointing
    (``info.json``) through the settled stack WCS -- or None (then 'object' starts from
    the frame centre). The Origin centres its target, so this is where it is."""
    si = getattr(args, '_session_info', None)
    cards = getattr(args, '_stack_wcs', None)
    if si is None or not cards or getattr(si, 'ra_rad', None) is None or getattr(si, 'dec_rad', None) is None:
        return None
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        h = fits.Header()
        for key, (val, _c) in cards.items():
            h[key] = val
        x, y = WCS(h, naxis=2).all_world2pix(np.degrees(si.ra_rad), np.degrees(si.dec_rad), 0)
        return float(y), float(x)
    except Exception:
        return None


def write_preview_jpeg(out: np.ndarray, path: str) -> None:
    """Write a stretched uint8 HWC image as the size-capped preview JPEG."""
    from src.models import Config
    img = _preview_pil_image(out, Config.PREVIEW_MAX_DIMENSION)
    img.save(path, format='JPEG', quality=Config.PREVIEW_JPEG_QUALITY, **JPEG_SAVE_KW)


_PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def write_png16(path: str, img01: np.ndarray, level: int = 6) -> None:
    """A [0, 1] RGB image as a 16-bit-per-channel PNG (Pillow cannot write 16-bit RGB):
    rows with the 'Sub' filter (each byte minus the one 6 earlier), zlib-compressed. For
    editing the finished image further: a curves or levels change on 8 bits bands."""
    import struct
    import zlib
    h, w = img01.shape[:2]
    u16 = np.clip(np.rint(np.asarray(img01, np.float32) * np.float32(65535.0)), 0, 65535).astype('>u2')
    raw = np.ascontiguousarray(u16).reshape(h, w * 3).view(np.uint8)     # big-endian, 6 bytes/pixel
    sub = raw.copy()
    sub[:, 6:] = raw[:, 6:] - raw[:, :-6]                                  # uint8 wraps mod 256
    data = np.empty((h, 1 + w * 6), np.uint8)
    data[:, 0] = 1                                                         # filter type: Sub
    data[:, 1:] = sub

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (struct.pack('>I', len(body)) + kind + body
                + struct.pack('>I', zlib.crc32(kind + body) & 0xFFFFFFFF))
    with open(path, 'wb') as fh:
        fh.write(_PNG_SIGNATURE)
        fh.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 16, 2, 0, 0, 0)))
        fh.write(chunk(b'IDAT', zlib.compress(data.tobytes(), level)))
        fh.write(chunk(b'IEND', b''))


def export_paths(jpeg_path: str) -> Dict[str, str]:
    """The ``--export png`` / ``png16`` files written next to a preview JPEG."""
    stem = os.path.splitext(jpeg_path)[0]
    return {'png': stem + '.png', 'png16': stem + '_16bit.png'}


def write_final_images(img01: np.ndarray, jpeg_path: str, exports=()) -> None:
    """The preview JPEG of a [0, 1] display image, plus the lossless ``exports`` ('png':
    8-bit, the JPEG's exact pixels before compression; 'png16': 16-bit) from the same
    render. Each file goes through a temporary name, so a failed write never leaves a
    half-written file."""
    def atomic(path, write):
        tmp = path + '.tmp'
        try:
            write(tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    u8 = to_uint8(img01)
    atomic(jpeg_path, lambda t: write_preview_jpeg(u8, t))
    paths = export_paths(jpeg_path)
    if 'png' in exports and Image is not None:
        atomic(paths['png'], lambda t: Image.fromarray(u8).save(t, format='PNG', compress_level=6))
    if 'png16' in exports:
        atomic(paths['png16'], lambda t: write_png16(t, img01))


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
    lum = luminance(out).astype(np.float32)
    blended = out.astype(np.float32) * (1.0 - amount) + lum[:, :, None] * amount
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def preview_jpeg_bytes(rgb: np.ndarray, stretch: str = 'ghs',
                       ghs_b: float = 8.0, ghs_sp: float = 0.15,
                       ghs_hp: float = 0.95, black_sigma: float = 0.0,
                       max_dim: int = 1024, desaturate: float = 0.0,
                       color: str = 'preserve',
                       noise_scale: float = 1.0,
                       anchor: Optional[Tuple[float, float]] = None) -> Optional[bytes]:
    """Stretched preview JPEG as bytes (for the live web view).

    ``desaturate`` (0-1) blends the stretched result toward luminance -- see
    ``_desaturate_preview_uint8``. 0 (default) preserves full colour, as
    every non-thumbnail caller wants."""
    if Image is None:
        return None
    out = render_preview_uint8(rgb, stretch=stretch, ghs_b=ghs_b, ghs_sp=ghs_sp,
                               ghs_hp=ghs_hp, black_sigma=black_sigma, color=color,
                               noise_scale=noise_scale, anchor=anchor)
    if out is None:
        return None
    return preview_jpeg_from_uint8(_desaturate_preview_uint8(out, desaturate), max_dim)


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
        elif getattr(args, '_stack_wcs', None) and 'CTYPE1' not in header:
            # no info.json: the WCS came from solving the first sub (Seestar, Dwarf)
            for kw, (val, comment) in args._stack_wcs.items():
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
        if 'step_px' in dither_info:
            header['DITHSTEP'] = (round(float(dither_info['step_px']), 2),
                                  'Median frame-to-frame step (px), drift removed')

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
