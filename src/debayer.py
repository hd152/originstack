"""Debayering, white balance, and hot pixel removal."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy import ndimage

from src.gpu_context import get_gpu
from src.models import Config
from src.utils import get_logger, safe_print

_log = get_logger()

# Optional native (Rust) kernels — graceful degradation to numpy if absent.
try:
    import astro_native as _native
    _HAS_NATIVE = True
except Exception:
    _native = None
    _HAS_NATIVE = False

from src.phase_correlate import phase_cross_correlation as _pcc

# FIX #10: Module-level constant — avoids reallocating the kernel on every
# upsample() call inside debayer_bilinear().
_BILINEAR_KERNEL = np.array(
    [[0.25, 0.5, 0.25],
     [0.5,  1.0, 0.5],
     [0.25, 0.5, 0.25]],
    dtype=np.float32,
)
_BILINEAR_KERNEL /= _BILINEAR_KERNEL.sum()
_BILINEAR_KERNEL_GPU: dict = {}   # xp-id → device copy, uploaded once per session

# Bayer pattern → (r_offset, g1_offset, g2_offset, b_offset)
# Each offset is (row, col) into the 2×2 Bayer tile.
_PATTERN_OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    'RGGB': ((0, 0), (0, 1), (1, 0), (1, 1)),  # R G1 G2 B
    'BGGR': ((1, 1), (0, 1), (1, 0), (0, 0)),  # B G1 G2 R  → swap R↔B
    'GRBG': ((0, 1), (0, 0), (1, 1), (1, 0)),  # R G1 G2 B (shifted)
    'GBRG': ((1, 0), (0, 0), (1, 1), (0, 1)),  # R G1 G2 B (shifted)
}

# Each pattern's row-flipped counterpart: swapping which sensor row is "row 0"
# turns RGGB<->GBRG and BGGR<->GRBG (green moves from the anti-diagonal to the
# main diagonal or back -- a column flip or 180 deg rotation would not do this,
# only a single-axis row flip). See autodetect_bayer_orientation below.
_ROW_FLIP_ALTERNATE: dict[str, str] = {
    'RGGB': 'GBRG', 'GBRG': 'RGGB', 'BGGR': 'GRBG', 'GRBG': 'BGGR',
}


def autodetect_bayer_orientation(raw, pattern: str, imbalance_threshold: float = 1.2) -> str:
    """Sanity-check a declared Bayer pattern against the actual raw mosaic and
    correct for a single-axis row-orientation mismatch some capture software
    gets wrong (observed on Celestron Origin FITS: BAYERPAT declares a pattern
    whose green positions don't match the row order the pixel data was
    actually written in).

    A genuine G1/G2 sub-pixel sensitivity mismatch -- what green_equalize
    corrects for -- is capped there at +-20% (`imbalance_threshold`'s default
    matches that same boundary). Real correctly-labeled data measured well
    inside it (Rosette Nebula/RGGB: ~0.4%); the real mislabeled case this
    guards against (Trifid Nebula/Celestron Origin, declared GBRG) measured
    ~25% on the declared pattern, dropping to ~0.4% on the row-flipped
    alternate -- so the declared pattern's G1/G2 ratio exceeding this bound
    is a reliable, non-arbitrary tell that green is actually on the OTHER
    diagonal, not a sensor characteristic. Returns `pattern` unchanged when
    the declared pattern already looks correct (the normal case) or the
    requested pattern isn't a plain 4-letter CFA name.
    """
    pattern = pattern.upper()
    offsets = _PATTERN_OFFSETS.get(pattern)
    alt = _ROW_FLIP_ALTERNATE.get(pattern)
    if offsets is None or alt is None:
        return pattern
    try:
        import cupy as _cp
        xp = _cp.get_array_module(raw)
    except Exception:
        xp = np
    (_, _), (g1_r, g1_c), (g2_r, g2_c), (_, _) = offsets
    g1 = raw[g1_r::2, g1_c::2]
    g2 = raw[g2_r::2, g2_c::2]
    if xp is np:
        m1 = _sigma_clipped_median(g1, xp=xp)
        m2 = _sigma_clipped_median(g2, xp=xp)
    else:
        m1 = float(xp.median(g1))
        m2 = float(xp.median(g2))
    if min(m1, m2) < 1e-6:
        return pattern
    ratio = max(m1, m2) / min(m1, m2)
    if ratio > imbalance_threshold:
        msg = (f"Bayer pattern {pattern}: G1/G2 medians differ {ratio:.2f}x "
               f"(>{imbalance_threshold:.1f}x sensor-noise range) -- using "
               f"row-flipped {alt} instead; the declared BAYERPAT likely "
               f"doesn't match this file's row orientation")
        _log.warning(msg)
        safe_print(f"  NOTE: {msg}")
        return alt
    return pattern


def _sigma_clipped_median(arr, sigma: float = 3.0, iters: int = 3, xp=np) -> float:
    if (xp is np and _HAS_NATIVE and hasattr(_native, 'strided_sigma_clipped_median')
            and isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.dtype == np.float32):
        # gathers the (strided) view itself: no ravel()/ascontiguousarray copy in numpy
        try:
            return float(_native.strided_sigma_clipped_median(arr, float(sigma), int(iters)))
        except Exception:
            pass
    if xp is np and _HAS_NATIVE and hasattr(_native, 'sigma_clipped_median_native'):
        try:
            flat = np.ascontiguousarray(arr.ravel(), dtype=np.float32)
            return float(_native.sigma_clipped_median_native(flat, float(sigma), int(iters)))
        except Exception:
            pass
    x = arr.ravel()
    for _ in range(iters):
        med = float(xp.median(x))
        std = float(xp.std(x))
        if std < 1e-12:
            break
        x = x[xp.abs(x - med) < sigma * std]
        if len(x) == 0:
            break
    return float(xp.median(x)) if len(x) > 0 else float(xp.median(arr))


def green_equalize(raw, pattern: str = 'RGGB', inplace: bool = False):
    """Scale the G2 sub-channel to match G1's sigma-clipped median.

    CMOS sensors have two physically distinct green sub-pixels (G1, G2) per
    Bayer tile that often differ by a few percent in sensitivity.  After
    bilinear debayering, the mismatch propagates as a 2-pixel-period
    checkerboard across the green channel (and therefore luminance).  Scaling
    G2 to match G1 before debayering removes the artifact entirely.

    The correction is capped at ±20 % to guard against bad frames where one
    sub-channel is near zero (saturated sky, very low counts, etc.).
    Accepts both numpy and CuPy arrays; output matches the input type. With
    ``inplace=True`` a writable C-contiguous float32 numpy mosaic is corrected
    where it lies (no 25 MB copy) -- only for a caller that owns it.
    """
    offsets = _PATTERN_OFFSETS.get(pattern.upper())
    if offsets is None:
        return raw
    cfg = _session_cfa
    if (cfg is not None and cfg['pattern'] == pattern.upper()
            and isinstance(raw, np.ndarray) and raw.ndim == 2):
        # session-constant gain (see `combine_cfa_stats`): no per-frame medians
        work = raw
        if not (inplace and raw.dtype == np.float32 and raw.flags['C_CONTIGUOUS']
                and raw.flags['WRITEABLE']):
            work = np.array(raw, dtype=np.float32, order='C', copy=True)
        (_, _), (_, _), (g2_r, g2_c), (_, _) = offsets
        if abs(cfg['gain'] - 1.0) < 0.2:
            work[g2_r::2, g2_c::2] *= np.float32(cfg['gain'])
        return work
    if (_HAS_NATIVE and hasattr(_native, 'green_equalize_inplace') and isinstance(raw, np.ndarray)
            and raw.ndim == 2):
        try:
            (_, _), (g1_r, g1_c), (g2_r, g2_c), (_, _) = offsets
            work = raw
            if not (inplace and raw.dtype == np.float32 and raw.flags['C_CONTIGUOUS']
                    and raw.flags['WRITEABLE']):
                work = np.array(raw, dtype=np.float32, order='C', copy=True)
            # medians of both green planes gathered natively, G2 scaled in place
            _native.green_equalize_inplace(work, g1_r, g1_c, g2_r, g2_c)
            return work
        except Exception:
            pass
    try:
        import cupy as _cp
        xp = _cp.get_array_module(raw)
    except Exception:
        # Broad on purpose: a present-but-broken cupy install (partial,
        # mismatched CUDA build, or -- as originally caught this in CI --
        # a fake/stub module a test left in sys.modules) must fall back to
        # numpy the same as cupy being absent entirely, not crash
        # debayering. Narrower ImportError-only handling let
        # AttributeError (e.g. a stub with no get_array_module) escape
        # uncaught.
        xp = np
    (_, _), (g1_r, g1_c), (g2_r, g2_c), (_, _) = offsets
    raw_f = xp.array(raw, dtype=xp.float32)
    g1 = raw_f[g1_r::2, g1_c::2].ravel()
    g2 = raw_f[g2_r::2, g2_c::2].ravel()
    # On GPU use a direct median (2 GPU→CPU syncs) instead of sigma-clipped
    # (12+ syncs). For typical astrophotography data the difference is <0.1%.
    if xp is np:
        g1_med = _sigma_clipped_median(g1, xp=xp)
        g2_med = _sigma_clipped_median(g2, xp=xp)
    else:
        g1_med = float(xp.median(g1))
        g2_med = float(xp.median(g2))
    if g2_med > 1e-6 and abs(g1_med / g2_med - 1.0) < 0.2:
        raw_f[g2_r::2, g2_c::2] *= g1_med / g2_med
    return raw_f


def _equalize_bayer_grid(rgb: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Remove 2×2 position-dependent green bias introduced by edge-aware CFA interpolation.

    Malvar debayering produces systematically different green values at
    interpolated positions (R and B cells) vs source positions (G1 and G2 cells).
    After stacking many frames, this coherent per-pixel offset survives noise
    averaging and becomes a visible checkerboard in the sky background.

    Uses sigma-clipped medians to estimate the sky-level offset for each of the
    four Bayer-position sub-channels, then subtracts the deviation from the
    per-image mean.  Guards skip corrections outside the plausible 0.01–100 ADU
    range to avoid modifying high-SNR targets or corrupted frames.

    ``inplace=True`` edits a writable C-contiguous float32 array where it lies (the
    native path then touches the green plane once and copies nothing).
    """
    if (_HAS_NATIVE and hasattr(_native, 'bayer_grid_equalize_inplace') and isinstance(rgb, np.ndarray)
            and rgb.dtype == np.float32 and rgb.ndim == 3 and rgb.shape[2] == 3
            and rgb.flags['C_CONTIGUOUS'] and rgb.flags['WRITEABLE']):
        try:
            work = rgb if inplace else rgb.copy()
            return work if _native.bayer_grid_equalize_inplace(work) or inplace else rgb
        except Exception:
            pass
    G = rgb[:, :, 1]
    ee = _sigma_clipped_median(G[::2,  ::2])   # R positions (even row, even col)
    eo = _sigma_clipped_median(G[::2,  1::2])  # G1 positions (even row, odd col)
    oe = _sigma_clipped_median(G[1::2, ::2])   # G2 positions (odd row, even col)
    oo = _sigma_clipped_median(G[1::2, 1::2])  # B positions (odd row, odd col)
    overall = (ee + eo + oe + oo) / 4.0
    spread = max(abs(ee - overall), abs(eo - overall),
                 abs(oe - overall), abs(oo - overall))
    if spread < 0.01 or spread > 100.0:
        return rgb
    result = rgb.copy()
    G_out = result[:, :, 1]
    G_out[::2,  ::2]  = np.clip(G[::2,  ::2]  - float(ee - overall), 0, None)
    G_out[::2,  1::2] = np.clip(G[::2,  1::2] - float(eo - overall), 0, None)
    G_out[1::2, ::2]  = np.clip(G[1::2, ::2]  - float(oe - overall), 0, None)
    G_out[1::2, 1::2] = np.clip(G[1::2, 1::2] - float(oo - overall), 0, None)
    return result


def debayer_bilinear(raw: np.ndarray, pattern: str = 'RGGB', method: str = 'bilinear') -> np.ndarray:
    # FIX #1: honour the `pattern` argument — previously hardcoded to RGGB.
    # FIX #2: `method` param is accepted for API compatibility but not used
    #          (bilinear is the only variant here); document this explicitly.
    gpu = get_gpu()
    xp = gpu.xp
    raw = gpu.to_device(raw)
    H, W = raw.shape

    offsets = _PATTERN_OFFSETS.get(pattern.upper())
    if offsets is None:
        raise ValueError(f"Unknown Bayer pattern '{pattern}'. "
                         f"Expected one of {list(_PATTERN_OFFSETS)}")

    (r_r, r_c), (g1_r, g1_c), (g2_r, g2_c), (b_r, b_c) = offsets

    r  = raw[r_r::2,  r_c::2]
    g1 = raw[g1_r::2, g1_c::2]
    g2 = raw[g2_r::2, g2_c::2]
    b  = raw[b_r::2,  b_c::2]

    # FIX #6 & #10: Use kron for channel expansion instead of a sparse
    # zero-filled array; reuse the pre-allocated module-level kernel.
    _xp_id = id(xp)
    if _xp_id not in _BILINEAR_KERNEL_GPU:
        _BILINEAR_KERNEL_GPU[_xp_id] = xp.array(_BILINEAR_KERNEL)
    kernel = _BILINEAR_KERNEL_GPU[_xp_id]
    # Pre-allocate once; each upsample() call zeros and refills it.
    # convolve() returns a new array so the previous result is safe.
    expanded = xp.zeros((H, W), dtype=xp.float32)

    def upsample(ch, r_offset, c_offset):
        expanded[:] = 0
        expanded[r_offset::2, c_offset::2] = ch
        return gpu.xndimage.convolve(expanded, kernel, mode='mirror')

    # FIX #7: Average the two green upsamples in one expression — same cost,
    # but expressed clearly; a future GPU kernel could fuse these.
    out = xp.zeros((H, W, 3), dtype=xp.float32)
    out[:, :, 0] = upsample(r,  r_r,  r_c)
    out[:, :, 1] = 0.5 * (upsample(g1, g1_r, g1_c) + upsample(g2, g2_r, g2_c))
    out[:, :, 2] = upsample(b,  b_r,  b_c)
    return out


# Malvar-He-Cutler (2004) kernels -- "High-Quality Linear Interpolation for
# Demosaicing of Bayer-Patterned Color Images". Coefficients as published
# (Table 1), normalised by 8; verified against the reference implementation
# in the `colour-demosaicing` package (bit-exact on interior pixels, see
# tests/test_debayer_malvar.py). The old cv2-EA path required 16-bit
# requantization (real precision loss) and only ran when cv2 was installed;
# this operates directly on the native float32 data and has no dependency.
_MALVAR_G_AT_RB = np.array([
    [0.0, 0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0,  2.0, 0.0, 0.0],
    [-1.0, 2.0, 4.0, 2.0, -1.0],
    [0.0, 0.0,  2.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0],
], dtype=np.float64) / 8.0

# R at green in an R row / B column (and B at green in a B row / R column).
_MALVAR_RG_RB_BG_BR = np.array([
    [0.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, -1.0, 0.0, -1.0, 0.0],
    [-1.0, 4.0, 5.0, 4.0, -1.0],
    [0.0, -1.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.0],
], dtype=np.float64) / 8.0

# R at green in a B row / R column (and B at green in an R row / B column) --
# the transpose of the kernel above.
_MALVAR_RG_BR_BG_RB = _MALVAR_RG_RB_BG_BR.T

# R at B (and B at R).
_MALVAR_R_AT_B = np.array([
    [0.0, 0.0, -1.5, 0.0, 0.0],
    [0.0, 2.0, 0.0, 2.0, 0.0],
    [-1.5, 0.0, 6.0, 0.0, -1.5],
    [0.0, 2.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, -1.5, 0.0, 0.0],
], dtype=np.float64) / 8.0


def _debayer_malvar_numpy(raw: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """Malvar-He-Cutler demosaicing, pure numpy (native Rust dispatch happens
    one level up in ``debayer_malvar``)."""
    offsets = _PATTERN_OFFSETS.get(pattern.upper())
    if offsets is None:
        raise ValueError(f"Unknown Bayer pattern '{pattern}'. "
                         f"Expected one of {list(_PATTERN_OFFSETS)}")
    (r_r, r_c), _, _, (b_r, b_c) = offsets
    raw64 = np.asarray(raw, dtype=np.float64)
    H, W = raw64.shape

    row = np.arange(H) % 2
    col = np.arange(W) % 2
    r_row = (row == r_r)[:, None]
    r_col = (col == r_c)[None, :]
    b_row = (row == b_r)[:, None]
    b_col = (col == b_c)[None, :]

    g_at_rb = ndimage.convolve(raw64, _MALVAR_G_AT_RB, mode='mirror')
    rg_rb_bg_br = ndimage.convolve(raw64, _MALVAR_RG_RB_BG_BR, mode='mirror')
    rg_br_bg_rb = ndimage.convolve(raw64, _MALVAR_RG_BR_BG_RB, mode='mirror')
    r_at_b = ndimage.convolve(raw64, _MALVAR_R_AT_B, mode='mirror')

    is_r = r_row & r_col
    is_b = b_row & b_col

    R = np.where(is_r, raw64, 0.0)
    R = np.where(r_row & b_col, rg_rb_bg_br, R)
    R = np.where(b_row & r_col, rg_br_bg_rb, R)
    R = np.where(is_b, r_at_b, R)

    B = np.where(is_b, raw64, 0.0)
    B = np.where(b_row & r_col, rg_rb_bg_br, B)
    B = np.where(r_row & b_col, rg_br_bg_rb, B)
    B = np.where(is_r, r_at_b, B)

    G = np.where(is_r | is_b, g_at_rb, raw64)

    return np.stack([R, G, B], axis=-1).astype(np.float32)


def _malvar_raw(raw: np.ndarray, pattern: str) -> np.ndarray:
    """Malvar demosaic with no grid equalisation (native, numpy fallback)."""
    if _HAS_NATIVE and hasattr(_native, 'debayer_malvar'):
        try:
            return _native.debayer_malvar(np.ascontiguousarray(raw, dtype=np.float32),
                                          pattern.upper())
        except Exception:
            pass
    return _debayer_malvar_numpy(raw, pattern)


def debayer_malvar(raw: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """Malvar-He-Cutler demosaicing (native Rust kernel with a numpy
    fallback -- see ``_debayer_malvar_numpy`` for the algorithm/validation
    notes). No longer depends on cv2."""
    out = _malvar_raw(raw, pattern)      # `out` is ours: equalise it where it lies
    cfg = _session_cfa
    if cfg is not None and cfg['pattern'] == pattern.upper():
        _apply_fixed_grid(out, cfg)
        return out
    return _equalize_bayer_grid(out, inplace=True)


# ── Session-constant CFA equalisation ────────────────────────────────────────
# green_equalize (G1/G2 gain) and _equalize_bayer_grid (2x2 green offsets) each
# re-measure sigma-clipped medians on every frame -- six medians over ~100 MB of
# strided reads, ~70% of the Debayer step (and memory-bandwidth bound, so it does
# not scale across workers). Both quantities are properties of the sensor and the
# Malvar kernel, not of any one frame, and each per-frame estimate is itself noisy
# (~2 ADU against offsets of ~2-3 ADU), so `measure_session_cfa` estimates them
# once from a few frames and every frame applies the session values. Opt-in per
# session: `set_session_cfa(None)` (the default) keeps the per-frame path.
_session_cfa: Optional[dict] = None
_GRID_PARITY = ((0, 0), (0, 1), (1, 0), (1, 1))


def set_session_cfa(cfg: Optional[dict]) -> None:
    global _session_cfa
    _session_cfa = cfg


def get_session_cfa() -> Optional[dict]:
    return _session_cfa


def cfa_frame_stats(mosaic: np.ndarray, pattern: str, method: str = 'malvar') -> Optional[dict]:
    """What the per-frame path would have measured on this calibrated mosaic:
    the G1/G2 gain ``green_equalize`` applies, then the four 2x2 green
    deviations ``_equalize_bayer_grid`` removes from the *method*'s own output
    (malvar or rcd -- the deviations are a property of the interpolator). None
    when the frame fails the same guards (G2 near zero, gain outside +-20%)."""
    offsets = _PATTERN_OFFSETS.get(pattern.upper())
    if offsets is None or getattr(mosaic, 'ndim', 0) != 2:
        return None
    (_, _), (g1_r, g1_c), (g2_r, g2_c), (_, _) = offsets
    work = np.array(mosaic, dtype=np.float32, order='C', copy=True)
    g1 = _sigma_clipped_median(work[g1_r::2, g1_c::2])
    g2 = _sigma_clipped_median(work[g2_r::2, g2_c::2])
    if g2 <= 1e-6 or abs(g1 / g2 - 1.0) >= 0.2:
        return None
    work[g2_r::2, g2_c::2] *= np.float32(g1 / g2)
    green = (_malvar_raw(work, pattern) if method == 'malvar'
             else _rcd_raw(work, pattern))[:, :, 1]
    q = np.array([_sigma_clipped_median(np.ascontiguousarray(green[a::2, b::2]))
                  for a, b in _GRID_PARITY])
    return {'gain': float(g1 / g2), 'grid': (q - q.mean()).tolist()}


def combine_cfa_stats(samples, pattern: str, min_samples: int = 5,
                      max_grid_std: float = 3.0, max_gain_std: float = 0.01) -> Optional[dict]:
    """Session values from per-frame ``cfa_frame_stats``: the median of each. None
    (per-frame fallback) with too few valid samples or when they disagree by more
    than a sensor property should -- a session whose offsets wander is not one a
    single value describes."""
    good = [s for s in samples if s]
    if len(good) < min_samples:
        return None
    gains = np.array([s['gain'] for s in good])
    grid = np.array([s['grid'] for s in good])
    if gains.std() > max_gain_std or grid.std(axis=0).max() > max_grid_std:
        return None
    med = np.median(grid, axis=0)
    med -= med.mean()
    spread = float(np.abs(med).max())
    return {'pattern': pattern.upper(), 'gain': float(np.median(gains)),
            'grid': tuple(float(x) for x in med),
            # the per-frame path's own guard: outside 0.01-100 ADU it leaves the frame alone
            'apply_grid': 0.01 <= spread <= 100.0, 'n': len(good)}


def _apply_fixed_grid(rgb: np.ndarray, cfg: dict) -> None:
    """Subtract the session's 2x2 green offsets in place (clipped at zero, as the
    per-frame path does)."""
    if not cfg['apply_grid']:
        return
    green = rgb[:, :, 1]
    for (a, b), off in zip(_GRID_PARITY, cfg['grid']):
        v = green[a::2, b::2]
        v -= np.float32(off)
        np.maximum(v, 0, out=v)


# Menon (2007) DDFAPD 5x5 gradient-diffusion kernel: weights how far the
# horizontal/vertical colour-difference-gradient signal (D_H/D_V) spreads
# before the two are compared to pick a per-pixel green interpolation
# direction. Exact values from Menon, Andriani & Calvagno 2007 (and the
# colour-demosaicing reference this was validated against).
_MENON_DIFFUSION_K = np.array([
    [0.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 3.0, 0.0, 3.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 1.0],
], dtype=np.float64)

# Green interpolation at R/B sites: h_0 samples the two nearest green
# neighbours, h_1 is a 2nd-derivative correction term using the R/B samples
# two positions further out (same directional-filter pair the reference uses
# for both the horizontal and vertical passes).
_MENON_H0 = np.array([0.0, 0.5, 0.0, 0.5, 0.0], dtype=np.float64)
_MENON_H1 = np.array([-0.25, 0.0, 0.5, 0.0, -0.25], dtype=np.float64)
_MENON_KB = np.array([0.5, 0.0, 0.5], dtype=np.float64)
_MENON_FIR = np.full(3, 1.0 / 3.0, dtype=np.float64)


def _debayer_menon2007_numpy(raw: np.ndarray, pattern: str = 'RGGB',
                             refining_step: bool = True) -> np.ndarray:
    """DDFAPD - Menon (2007) directional-filtering demosaicing, pure numpy.

    Port of Menon, Andriani & Calvagno 2007 ("Demosaicing With Directional
    Filtering and a posteriori Decision"), validated bit-exact (float32
    rounding only) against the `colour-demosaicing` package's reference
    implementation across all 4 Bayer patterns -- see
    tests/test_debayer_menon2007.py. Native Rust dispatch happens one level
    up in ``debayer_menon2007``.

    Green is interpolated in both directions (h_0/h_1 directional filters),
    then a horizontal/vertical colour-difference gradient (diffused through
    ``_MENON_DIFFUSION_K``) decides, per pixel, which direction's green
    estimate to keep. Red/blue are then filled in at green sites and at the
    opposite colour's sites using the same per-pixel direction. An optional
    refining pass (on by default, matching the reference) re-derives green
    from the now-complete R-G/B-G colour differences and touches up R/B at
    green sites and at each other's sites for a final consistency pass.
    """
    offsets = _PATTERN_OFFSETS.get(pattern.upper())
    if offsets is None:
        raise ValueError(f"Unknown Bayer pattern '{pattern}'. "
                         f"Expected one of {list(_PATTERN_OFFSETS)}")
    (r_r, r_c), _, _, (b_r, b_c) = offsets
    raw64 = np.asarray(raw, dtype=np.float64)
    H, W = raw64.shape

    def cnv_h(x, k):
        return ndimage.convolve1d(x, k, axis=1, mode='mirror')

    def cnv_v(x, k):
        return ndimage.convolve1d(x, k, axis=0, mode='mirror')

    row = (np.arange(H) % 2)[:, None]
    col = (np.arange(W) % 2)[None, :]
    R_m = (row == r_r) & (col == r_c)
    B_m = (row == b_r) & (col == b_c)
    G_m = ~R_m & ~B_m
    R_r_rows = np.broadcast_to(row == r_r, (H, W))  # rows containing an R sample
    B_r_rows = np.broadcast_to(row == b_r, (H, W))  # rows containing a B sample

    R = np.where(R_m, raw64, 0.0)
    G = np.where(G_m, raw64, 0.0)
    B = np.where(B_m, raw64, 0.0)

    G_H = np.where(~G_m, cnv_h(raw64, _MENON_H0) + cnv_h(raw64, _MENON_H1), G)
    G_V = np.where(~G_m, cnv_v(raw64, _MENON_H0) + cnv_v(raw64, _MENON_H1), G)

    C_H = np.where(R_m, R - G_H, 0.0)
    C_H = np.where(B_m, B - G_H, C_H)
    C_V = np.where(R_m, R - G_V, 0.0)
    C_V = np.where(B_m, B - G_V, C_V)

    # |C(i) - C(i+2)| along each axis, reflecting at the far boundary --
    # matches np.pad(..., mode='reflect')[..., 2:] in the reference exactly.
    D_H = np.abs(C_H - np.pad(C_H, ((0, 0), (0, 2)), mode='reflect')[:, 2:])
    D_V = np.abs(C_V - np.pad(C_V, ((0, 2), (0, 0)), mode='reflect')[2:, :])

    d_H = ndimage.convolve(D_H, _MENON_DIFFUSION_K, mode='constant', cval=0.0)
    d_V = ndimage.convolve(D_V, _MENON_DIFFUSION_K.T, mode='constant', cval=0.0)

    use_h = d_V >= d_H  # True where the horizontal green estimate wins
    G = np.where(use_h, G_H, G_V)

    R = np.where(G_m & R_r_rows, G + cnv_h(R, _MENON_KB) - cnv_h(G, _MENON_KB), R)
    R = np.where(G_m & B_r_rows, G + cnv_v(R, _MENON_KB) - cnv_v(G, _MENON_KB), R)
    B = np.where(G_m & B_r_rows, G + cnv_h(B, _MENON_KB) - cnv_h(G, _MENON_KB), B)
    B = np.where(G_m & R_r_rows, G + cnv_v(B, _MENON_KB) - cnv_v(G, _MENON_KB), B)

    R = np.where(B_r_rows & B_m,
                np.where(use_h, B + cnv_h(R, _MENON_KB) - cnv_h(B, _MENON_KB),
                                B + cnv_v(R, _MENON_KB) - cnv_v(B, _MENON_KB)), R)
    B = np.where(R_r_rows & R_m,
                np.where(use_h, R + cnv_h(B, _MENON_KB) - cnv_h(R, _MENON_KB),
                                R + cnv_v(B, _MENON_KB) - cnv_v(R, _MENON_KB)), B)

    if refining_step:
        R_G = R - G
        B_G = B - G
        B_G_m = np.where(B_m, np.where(use_h, cnv_h(B_G, _MENON_FIR), cnv_v(B_G, _MENON_FIR)), 0.0)
        R_G_m = np.where(R_m, np.where(use_h, cnv_h(R_G, _MENON_FIR), cnv_v(R_G, _MENON_FIR)), 0.0)
        G = np.where(R_m, R - R_G_m, G)
        G = np.where(B_m, B - B_G_m, G)

        col_b = np.broadcast_to(col == r_c, (H, W))  # columns containing an R sample
        colB_b = np.broadcast_to(col == b_c, (H, W))  # columns containing a B sample

        R_G = R - G
        B_G = B - G
        R_G_m = np.where(G_m & B_r_rows, cnv_v(R_G, _MENON_KB), R_G_m)
        R = np.where(G_m & B_r_rows, G + R_G_m, R)
        R_G_m = np.where(G_m & colB_b, cnv_h(R_G, _MENON_KB), R_G_m)
        R = np.where(G_m & colB_b, G + R_G_m, R)

        B_G_m = np.where(G_m & R_r_rows, cnv_v(B_G, _MENON_KB), B_G_m)
        B = np.where(G_m & R_r_rows, G + B_G_m, B)
        B_G_m = np.where(G_m & col_b, cnv_h(B_G, _MENON_KB), B_G_m)
        B = np.where(G_m & col_b, G + B_G_m, B)

        R_B = R - B
        R_B_m = np.where(B_m, np.where(use_h, cnv_h(R_B, _MENON_FIR), cnv_v(R_B, _MENON_FIR)), 0.0)
        R = np.where(B_m, B + R_B_m, R)
        R_B_m = np.where(R_m, np.where(use_h, cnv_h(R_B, _MENON_FIR), cnv_v(R_B, _MENON_FIR)), 0.0)
        B = np.where(R_m, R - R_B_m, B)

    return np.stack([R, G, B], axis=-1).astype(np.float32)


def debayer_menon2007(raw: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """DDFAPD - Menon (2007) directional-filtering demosaicing (native Rust
    kernel with a numpy fallback -- see ``_debayer_menon2007_numpy`` for the
    algorithm/validation notes).

    Higher fidelity than Malvar on fine periodic photographic detail in
    general, but on this codebase's synthetic astro benchmark
    (tools/bench_debayer_quality.py) the gain over Malvar is modest (~14%
    lower MAE on a synthetic starfield) and it isn't the default -- most
    astro frames are smooth sky + point sources, not fine texture, and
    multi-frame dithered stacking further dilutes any single-frame demosaic
    residual either algorithm leaves behind."""
    if _HAS_NATIVE and hasattr(_native, 'debayer_menon2007'):
        raw_np = np.ascontiguousarray(raw, dtype=np.float32)
        try:
            out = _native.debayer_menon2007(raw_np, pattern.upper())
        except Exception:
            out = None
        if out is not None:
            return _equalize_bayer_grid(out)
    return _equalize_bayer_grid(_debayer_menon2007_numpy(raw, pattern))


def debayer_rcd(raw: np.ndarray, pattern: str = 'RGGB', out: Optional[np.ndarray] = None) -> np.ndarray:
    """RCD demosaic plus the same 2x2 green-grid correction as ``debayer_malvar``
    (session-constant when measured, per frame otherwise): RCD's interpolated
    greens are also biased by position, measured ~+-2.5 ADU on real Origin subs
    (Malvar's ~+-6), which survives stacking as a checkerboard."""
    cfg = _session_cfa
    if cfg is not None and cfg['pattern'] == pattern.upper():
        # session-constant grid: fused into the native kernel's output write when it
        # can (bit-identical, saves four strided passes over the 75 MB output)
        out, fused = _rcd_impl(raw, pattern, out, cfg['grid'] if cfg['apply_grid'] else None)
        if not fused:
            _apply_fixed_grid(out, cfg)
        return out
    out = _rcd_raw(raw, pattern, out=out)
    return _equalize_bayer_grid(out, inplace=True)


def _rcd_into_takes_grid() -> bool:
    """Whether the loaded native ``debayer_rcd_native_into`` has the fused ``grid``
    argument (an older build does not)."""
    f = getattr(_native, 'debayer_rcd_native_into', None) if _HAS_NATIVE else None
    return 'grid' in (getattr(f, '__text_signature__', None) or '')


def _rcd_raw(raw: np.ndarray, pattern: str = 'RGGB', out: Optional[np.ndarray] = None) -> np.ndarray:
    """RCD -- Ratio Corrected Demosaicing (Luis Sanz Rodriguez, 2017), the
    default in Siril, RawTherapee and darktable.

    Green at R/B sites is a gradient-weighted blend of four cardinal estimates,
    each the neighbouring green scaled by a *ratio* of low-pass-filtered CFA
    values (so a star's profile is followed rather than averaged over); the
    horizontal/vertical blend comes from a high-pass colour-difference
    statistic smoothed over three rows/columns. R at B (and B at R) uses
    diagonal colour differences the same way, then R/B at green sites use
    cardinal ones. Unlike Malvar, whose fixed gradient-correction kernels put
    back almost all of the per-pixel noise in R/B (~0.99 of the input sigma on
    a pure-noise mosaic), interpolated samples here are averages, so on smooth
    sky it is quieter at the same star sharpness -- the reason it was added:
    see dev-notes/siril-comparison.md.

    Values are processed scaled to [0, 1] by the frame maximum (the ratio terms
    are scale-free; ``eps`` assumes that range) and clipped below at 0. A 4-px
    border, where the 9-tap statistics do not fit, is taken from Malvar.
    """
    return _rcd_impl(raw, pattern, out, None)[0]


def _rcd_impl(raw: np.ndarray, pattern: str, out: Optional[np.ndarray],
              grid) -> tuple:
    """``_rcd_raw`` with an optional session grid (``_apply_fixed_grid``'s four
    offsets) fused into the native output write. Returns ``(rgb, grid_applied)``;
    ``grid_applied`` is False when the grid was not given or a path without the
    fusion ran (the caller then applies it as before)."""
    offsets = _PATTERN_OFFSETS.get(pattern.upper())
    if offsets is None:
        raise ValueError(f"Unknown Bayer pattern: {pattern!r}. "
                         f"Expected one of {list(_PATTERN_OFFSETS)}")
    if isinstance(raw, np.ndarray) and raw.dtype == np.float32:
        # the frame max straight off the float32 data: the same value the float64 copy
        # gave, without writing 50 MB to read one number (one pass, not three)
        a = None
        H, W = raw.shape
        if H < 16 or W < 16:
            return _malvar_raw(raw, pattern), False
        m = float(np.fmax.reduce(raw, axis=None))        # NaN only if every value is NaN
        scale = m if (np.isfinite(m) or np.isfinite(raw).any()) else 1.0
    else:
        a = np.asarray(raw, dtype=np.float64)
        H, W = a.shape
        if H < 16 or W < 16:
            return _malvar_raw(raw, pattern), False
        scale = float(np.nanmax(a)) if np.isfinite(a).any() else 1.0
    if not scale > 0:
        scale = 1.0
    (ry, rx), _g1, _g2, (by, bx) = offsets
    if _HAS_NATIVE and hasattr(_native, 'debayer_rcd_native') and isinstance(raw, np.ndarray):
        try:
            src = np.ascontiguousarray(raw, dtype=np.float32)
            into_ok = (out is not None and hasattr(_native, 'debayer_rcd_native_into')
                       and isinstance(out, np.ndarray) and out.dtype == np.float32
                       and out.shape == (H, W, 3) and out.flags['C_CONTIGUOUS']
                       and out.flags['WRITEABLE'])
            if grid is not None and _rcd_into_takes_grid():
                # the kernel writes the whole frame with the grid applied; the Malvar
                # border then replaces 4 px all round, so only those get it here
                buf = out if into_ok else np.empty((H, W, 3), dtype=np.float32)
                _native.debayer_rcd_native_into(src, buf, scale, (ry, rx), (by, bx),
                                                grid=tuple(float(g) for g in grid))
                _rcd_border(buf, raw, pattern)
                _apply_fixed_grid_border(buf, grid)
                return buf, True
            if into_ok:
                # straight into the caller's buffer (Phase 1's frame-store slot)
                _native.debayer_rcd_native_into(src, out, scale, (ry, rx), (by, bx))
                return _rcd_border(out, raw, pattern), False
            res = _native.debayer_rcd_native(src, scale, (ry, rx), (by, bx))
            return _rcd_border(res, raw, pattern), False
        except Exception as e:
            _log.debug("native RCD failed (%s); using numpy", e)
    if a is None:
        a = np.asarray(raw, dtype=np.float64)
    return _debayer_rcd_numpy(a, offsets, scale, raw, pattern), False


_RCD_BORDER = 4
_RCD_BORDER_STRIP = 12     # >= border + Malvar's 2-px reach, with room to spare


def _rcd_border(out: np.ndarray, raw: np.ndarray, pattern: str) -> np.ndarray:
    """The 4-px frame border, where RCD's 9-tap statistics do not fit, from Malvar
    (uncorrected: ``debayer_rcd`` equalises the whole frame afterwards).

    Malvar is a 5x5 local kernel, so it runs on four edge strips instead of the
    whole frame: each strip keeps the frame's own edges, starts on an even row/column
    (same Bayer phase) and reaches 8 px past the border, so the border values are
    exactly the full-frame ones -- a full-frame Malvar (75 MB written) was ~30% of
    the RCD step for 4 px of output."""
    b, m = _RCD_BORDER, _RCD_BORDER_STRIP
    H, W = raw.shape[:2]
    if H < 3 * m or W < 3 * m:
        border = _malvar_raw(raw, pattern)
        out[:b], out[-b:], out[:, :b], out[:, -b:] = (border[:b], border[-b:],
                                                      border[:, :b], border[:, -b:])
        return out
    out[:b] = _malvar_raw(raw[:m], pattern)[:b]
    out[-b:] = _malvar_raw(raw[(H - m) & ~1:], pattern)[-b:]
    out[:, :b] = _malvar_raw(raw[:, :m], pattern)[:, :b]
    out[:, -b:] = _malvar_raw(raw[:, (W - m) & ~1:], pattern)[:, -b:]
    return out


def _apply_fixed_grid_border(rgb: np.ndarray, grid) -> None:
    """``_apply_fixed_grid`` on the ``_RCD_BORDER``-px frame border only (four
    non-overlapping strips, each green value by its global 2x2 parity): the rest of
    the frame got the grid inside the native kernel."""
    b = _RCD_BORDER
    H, W = rgb.shape[:2]
    green = rgb[:, :, 1]
    if H <= 2 * b or W <= 2 * b:
        regions = [(0, H, 0, W)]
    else:
        regions = [(0, b, 0, W), (H - b, H, 0, W), (b, H - b, 0, b), (b, H - b, W - b, W)]
    for y0, y1, x0, x1 in regions:
        for (a, c), off in zip(_GRID_PARITY, grid):
            v = green[y0 + (a - y0) % 2:y1:2, x0 + (c - x0) % 2:x1:2]
            v -= np.float32(off)
            np.maximum(v, 0, out=v)


def _debayer_rcd_numpy(a: np.ndarray, offsets, scale: float, raw: np.ndarray,
                       pattern: str) -> np.ndarray:
    """numpy mirror of the native ``debayer_rcd_native`` (bit-identical)."""
    H, W = a.shape
    # f32 from here on (f64 doubled the memory traffic of the native kernel, which
    # is bandwidth bound); Python-float constants stay f32 under NEP 50
    cfa = np.nan_to_num(a / scale).astype(np.float32)
    eps, epssq = 1e-5, 1e-10
    P = 5
    c = np.pad(cfa, P, mode='reflect')

    def s(dy, dx, arr=c):                  # shifted view, aligned with the frame
        return arr[P + dy:P + dy + H, P + dx:P + dx + W]

    (ry, rx), (g1y, g1x), (g2y, g2x), (by, bx) = offsets
    yy, xx = np.mgrid[0:H, 0:W]
    is_r = ((yy % 2) == ry) & ((xx % 2) == rx)
    is_b = ((yy % 2) == by) & ((xx % 2) == bx)
    is_g = ~(is_r | is_b)
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0][is_r] = cfa[is_r]
    rgb[..., 1][is_g] = cfa[is_g]
    rgb[..., 2][is_b] = cfa[is_b]

    # Step 1: vertical/horizontal discrimination
    v_hpf = ((s(-3, 0) - s(-1, 0) - s(1, 0) + s(3, 0)) - 3.0 * (s(-2, 0) + s(2, 0))
             + 6.0 * s(0, 0)) ** 2
    h_hpf = ((s(0, -3) - s(0, -1) - s(0, 1) + s(0, 3)) - 3.0 * (s(0, -2) + s(0, 2))
             + 6.0 * s(0, 0)) ** 2
    vp, hp = np.pad(v_hpf, P, mode='reflect'), np.pad(h_hpf, P, mode='reflect')
    v_stat = np.maximum(epssq, s(-1, 0, vp) + s(0, 0, vp) + s(1, 0, vp))
    h_stat = np.maximum(epssq, s(0, -1, hp) + s(0, 0, hp) + s(0, 1, hp))
    vh_dir = v_stat / (v_stat + h_stat)
    vhp = np.pad(vh_dir, P, mode='reflect')
    vh_nb = 0.25 * (s(-1, -1, vhp) + s(-1, 1, vhp) + s(1, -1, vhp) + s(1, 1, vhp))
    vh_disc = np.where(np.abs(0.5 - vh_dir) < np.abs(0.5 - vh_nb), vh_nb, vh_dir)

    # Step 2: low-pass filter of the CFA (used at R/B sites)
    lpf = (s(0, 0) + 0.5 * (s(-1, 0) + s(1, 0) + s(0, -1) + s(0, 1))
           + 0.25 * (s(-1, -1) + s(-1, 1) + s(1, -1) + s(1, 1)))
    lp = np.pad(lpf, P, mode='reflect')

    # Step 3: green at R and B
    n_grad = eps + np.abs(s(-1, 0) - s(1, 0)) + np.abs(s(0, 0) - s(-2, 0)) \
        + np.abs(s(-1, 0) - s(-3, 0)) + np.abs(s(-2, 0) - s(-4, 0))
    s_grad = eps + np.abs(s(-1, 0) - s(1, 0)) + np.abs(s(0, 0) - s(2, 0)) \
        + np.abs(s(1, 0) - s(3, 0)) + np.abs(s(2, 0) - s(4, 0))
    w_grad = eps + np.abs(s(0, -1) - s(0, 1)) + np.abs(s(0, 0) - s(0, -2)) \
        + np.abs(s(0, -1) - s(0, -3)) + np.abs(s(0, -2) - s(0, -4))
    e_grad = eps + np.abs(s(0, -1) - s(0, 1)) + np.abs(s(0, 0) - s(0, 2)) \
        + np.abs(s(0, 1) - s(0, 3)) + np.abs(s(0, 2) - s(0, 4))
    l0 = lpf

    def est(dy, dx):
        ln = s(2 * dy, 2 * dx, lp)
        return s(dy, dx) * (1.0 + (l0 - ln) / (eps + l0 + ln))
    n_est, s_est, w_est, e_est = est(-1, 0), est(1, 0), est(0, -1), est(0, 1)
    v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad)
    h_est = (w_grad * e_est + e_grad * w_est) / (e_grad + w_grad)
    g_rb = np.clip(vh_disc * h_est + (1.0 - vh_disc) * v_est, 0.0, None)
    rb = ~is_g
    rgb[..., 1][rb] = g_rb[rb]

    G = np.pad(rgb[..., 1], P, mode='reflect')

    # Step 4.1-4.2: diagonal discrimination at R/B sites
    p_hpf = ((s(-3, -3) - s(-1, -1) - s(1, 1) + s(3, 3)) - 3.0 * (s(-2, -2) + s(2, 2))
             + 6.0 * s(0, 0)) ** 2
    q_hpf = ((s(-3, 3) - s(-1, 1) - s(1, -1) + s(3, -3)) - 3.0 * (s(-2, 2) + s(2, -2))
             + 6.0 * s(0, 0)) ** 2
    pp_, qp_ = np.pad(p_hpf, P, mode='reflect'), np.pad(q_hpf, P, mode='reflect')
    p_stat = np.maximum(epssq, s(-1, -1, pp_) + s(0, 0, pp_) + s(1, 1, pp_))
    q_stat = np.maximum(epssq, s(-1, 1, qp_) + s(0, 0, qp_) + s(1, -1, qp_))
    pq_dir = p_stat / (p_stat + q_stat)
    pqp = np.pad(pq_dir, P, mode='reflect')
    pq_nb = 0.25 * (s(-1, -1, pqp) + s(-1, 1, pqp) + s(1, -1, pqp) + s(1, 1, pqp))
    pq_disc = np.where(np.abs(0.5 - pq_dir) < np.abs(0.5 - pq_nb), pq_nb, pq_dir)

    # Step 4.3: R at B sites and B at R sites (diagonal colour differences)
    for ch, sites in ((0, is_b), (2, is_r)):
        C = np.pad(rgb[..., ch], P, mode='reflect')
        nw = eps + np.abs(s(-1, -1, C) - s(1, 1, C)) + np.abs(s(-1, -1, C) - s(-3, -3, C)) \
            + np.abs(s(0, 0, G) - s(-2, -2, G))
        ne = eps + np.abs(s(-1, 1, C) - s(1, -1, C)) + np.abs(s(-1, 1, C) - s(-3, 3, C)) \
            + np.abs(s(0, 0, G) - s(-2, 2, G))
        sw = eps + np.abs(s(1, -1, C) - s(-1, 1, C)) + np.abs(s(1, -1, C) - s(3, -3, C)) \
            + np.abs(s(0, 0, G) - s(2, -2, G))
        se = eps + np.abs(s(1, 1, C) - s(-1, -1, C)) + np.abs(s(1, 1, C) - s(3, 3, C)) \
            + np.abs(s(0, 0, G) - s(2, 2, G))
        nw_e = s(-1, -1, C) - s(-1, -1, G)
        ne_e = s(-1, 1, C) - s(-1, 1, G)
        sw_e = s(1, -1, C) - s(1, -1, G)
        se_e = s(1, 1, C) - s(1, 1, G)
        p_est = (nw * se_e + se * nw_e) / (nw + se)
        q_est = (ne * sw_e + sw * ne_e) / (ne + sw)
        val = np.clip(rgb[..., 1] + (1.0 - pq_disc) * p_est + pq_disc * q_est, 0.0, None)
        rgb[..., ch][sites] = val[sites]

    # Step 4.4: R and B at G sites (cardinal colour differences)
    for ch in (0, 2):
        C = np.pad(rgb[..., ch], P, mode='reflect')
        n_g = eps + np.abs(s(0, 0, G) - s(-2, 0, G)) + np.abs(s(-1, 0, C) - s(1, 0, C)) \
            + np.abs(s(-1, 0, C) - s(-3, 0, C))
        s_g = eps + np.abs(s(0, 0, G) - s(2, 0, G)) + np.abs(s(1, 0, C) - s(-1, 0, C)) \
            + np.abs(s(1, 0, C) - s(3, 0, C))
        w_g = eps + np.abs(s(0, 0, G) - s(0, -2, G)) + np.abs(s(0, -1, C) - s(0, 1, C)) \
            + np.abs(s(0, -1, C) - s(0, -3, C))
        e_g = eps + np.abs(s(0, 0, G) - s(0, 2, G)) + np.abs(s(0, 1, C) - s(0, -1, C)) \
            + np.abs(s(0, 1, C) - s(0, 3, C))
        n_e = s(-1, 0, C) - s(-1, 0, G)
        s_e = s(1, 0, C) - s(1, 0, G)
        w_e = s(0, -1, C) - s(0, -1, G)
        e_e = s(0, 1, C) - s(0, 1, G)
        v_e = (n_g * s_e + s_g * n_e) / (n_g + s_g)
        h_e = (e_g * w_e + w_g * e_e) / (e_g + w_g)
        val = np.clip(rgb[..., 1] + (1.0 - vh_disc) * v_e + vh_disc * h_e, 0.0, None)
        rgb[..., ch][is_g] = val[is_g]

    return _rcd_border((rgb * scale).astype(np.float32), raw, pattern)


def debayer(raw: np.ndarray, pattern: str = 'RGGB', method: str = 'bilinear',
            out: Optional[np.ndarray] = None) -> np.ndarray:
    """Dispatch to the appropriate debayering method. ``out``: an (H, W, 3) float32 buffer
    the native RCD path writes into (others ignore it and return a new array)."""
    if method == 'malvar':
        return debayer_malvar(raw, pattern)
    elif method == 'menon2007':
        return debayer_menon2007(raw, pattern)
    elif method == 'rcd':
        return debayer_rcd(raw, pattern, out=out)
    else:
        return debayer_bilinear(raw, pattern, method)


def _desaturate_near_clipped_highlights(xp, img: np.ndarray, scaled: np.ndarray) -> np.ndarray:
    """Blend WB output back towards neutral grey for pixels near this frame's own peak.

    A genuinely saturated star core is clipped at the sensor's full-well
    ceiling equally across R/G/B (the physical clip doesn't care which
    colour filter sits over the photosite) -- but white balance's
    per-channel gain is fit to the whole-frame *mean*, which is dominated
    by unsaturated background, so it has no notion of this. Applying that
    gain uniformly to an already-clipped pixel scales its (accidentally
    equal, not really representative of colour) channels apart, and since
    typical OSC sensors need a much larger R/B gain than G, this produces a
    magenta/purple halo squarely on bright stars even though the
    background comes out perfectly neutral (observed in the wild on a
    real Pleiades session: bright star cores at R/G ~2.1, B/G ~1.25 while
    the sky background measured ~0.99 on both ratios).

    Ramps smoothly from 0 (no effect) at 80% of the frame's own pre-WB
    peak to full neutralisation at the peak itself, so ordinary bright
    (but not clipped) stars are unaffected -- only pixels actually close
    to this frame's own ceiling get pulled back towards grey.
    """
    pixel_peak = img.max(axis=-1, keepdims=True)
    ceiling = img.max()
    sat_frac = xp.clip((pixel_peak / (ceiling + 1e-12) - 0.8) / 0.2, 0.0, 1.0)
    neutral = xp.broadcast_to(pixel_peak, scaled.shape)
    return scaled * (1 - sat_frac) + neutral * sat_frac


def _white_balance_native(xp, img, factors, divide: bool):
    """Native fused gain + highlight-neutralise, or None to use the array path.

    Bit-identical to ``_desaturate_near_clipped_highlights`` on the numpy path
    (same f32 operations in the same order, no fused multiply-add; the gains are
    still computed by the caller so numpy's reduction order is kept). Only for
    host numpy float32 arrays with float32 gains: anything else (cupy, or gains
    that numpy promoted to float64, whose result dtype the array path would then
    also carry) keeps the original code."""
    if not (xp is np and _HAS_NATIVE and hasattr(_native, 'white_balance_apply')):
        return None
    f = np.asarray(factors)
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.float32 or f.dtype != np.float32:
        return None
    try:
        return _native.white_balance_apply(np.ascontiguousarray(img), np.ascontiguousarray(f), bool(divide))
    except Exception:
        return None


def white_balance_grayworld(rgb: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Gray-world white balance. ``inplace=True`` rewrites a writable C-contiguous float32
    numpy image where it lies and returns it (native path; otherwise as usual)."""
    gpu = get_gpu()
    xp = gpu.xp
    if (inplace and xp is np and _HAS_NATIVE and hasattr(_native, 'white_balance_grayworld_inplace')
            and isinstance(rgb, np.ndarray) and rgb.dtype == np.float32 and rgb.ndim == 3
            and rgb.shape[2] == 3 and rgb.flags['C_CONTIGUOUS'] and rgb.flags['WRITEABLE']):
        try:
            _native.white_balance_grayworld_inplace(rgb)
            return rgb
        except Exception:
            pass
    img = xp.asarray(rgb, dtype=xp.float32)
    if (xp is np and _HAS_NATIVE and hasattr(_native, 'white_balance_grayworld')
            and img.ndim == 3 and img.shape[2] == 3):
        # whole path native, incl. the float64-accumulated channel means
        try:
            return _native.white_balance_grayworld(np.ascontiguousarray(img))
        except Exception:
            pass
    # float64 accumulation: a float32 running sum over ~6M pixels drifts ~1.5% on real frames
    mean = img.mean(axis=(0, 1), dtype=xp.float64).astype(xp.float32)
    scale = mean.mean() / (mean + 1e-12)
    fast = _white_balance_native(xp, img, scale, divide=False)
    if fast is not None:
        return fast
    out = _desaturate_near_clipped_highlights(xp, img, img * scale)
    return xp.clip(out, 0, None)


def white_balance_grayworld_lum(rgb: np.ndarray):
    """``white_balance_grayworld(rgb, inplace=True)`` followed by ``luminance``, as
    ``(rgb, lum)``. The native path balances in place and emits the luminance from
    the same pass (bit-identical to the two calls, one fewer read of the image);
    anything it does not cover runs the two calls."""
    if (_HAS_NATIVE and hasattr(_native, 'white_balance_grayworld_lum_inplace')
            and get_gpu().xp is np and isinstance(rgb, np.ndarray) and rgb.dtype == np.float32
            and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.flags['C_CONTIGUOUS']
            and rgb.flags['WRITEABLE']):
        try:
            return rgb, _native.white_balance_grayworld_lum_inplace(rgb)
        except Exception:
            pass
    rgb = white_balance_grayworld(rgb, inplace=True)
    return rgb, luminance(get_gpu().to_host(rgb))


def white_balance_grayworld_lum_into(rgb: np.ndarray, lum_out: np.ndarray):
    """``white_balance_grayworld_lum`` writing the luminance into ``lum_out`` (e.g. the
    frame store's luminance slot) and returning, from the same pass, what
    ``quality.validate_image_data`` reads from it: ``(lum_out, native_stats, sample)``
    with ``native_stats`` = ``validate_frame_stats(lum)`` and ``sample`` =
    ``lum[::3, ::3]`` (None below 64 px, where validation uses the whole frame).
    Bit-identical to ``white_balance_grayworld_lum``; None when the native kernel
    cannot take these arrays (the caller then runs ``white_balance_grayworld_lum``)."""
    if not (_HAS_NATIVE and hasattr(_native, 'white_balance_grayworld_lum_into')
            and get_gpu().xp is np and isinstance(rgb, np.ndarray) and rgb.dtype == np.float32
            and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.flags['C_CONTIGUOUS']
            and rgb.flags['WRITEABLE'] and isinstance(lum_out, np.ndarray)
            and lum_out.dtype == np.float32 and lum_out.shape == rgb.shape[:2]
            and lum_out.flags['C_CONTIGUOUS'] and lum_out.flags['WRITEABLE']):
        return None
    h, w = rgb.shape[:2]
    sample = np.empty((-(-h // 3), -(-w // 3)), np.float32) if min(h, w) >= 64 else None
    try:
        stats = _native.white_balance_grayworld_lum_into(rgb, lum_out, sample, 3)
    except Exception:
        return None
    return lum_out, stats, sample


def white_balance_whitepatch(rgb: np.ndarray, pct: Optional[float] = None) -> np.ndarray:
    gpu = get_gpu()
    xp = gpu.xp
    if pct is None:
        pct = Config.WHITE_PATCH_PERCENTILE
    img = xp.asarray(rgb, dtype=xp.float32)
    # Compute all per-channel stats on GPU, avoiding per-channel CPU syncs.
    # Guard against saturated channels: fall back to channel mean when the
    # pct-th percentile hits or exceeds the image max (fully clipped channel).
    scales  = xp.stack([xp.percentile(img[:, :, c], pct) for c in range(3)])
    img_max = img.max()
    means   = img.mean(axis=(0, 1))
    bad     = (scales >= img_max * 0.999) | (scales < 1e-12)
    scales  = xp.where(bad, means + 1e-12, scales)
    scales  = scales / (scales.mean() + 1e-12)
    fast = _white_balance_native(xp, img, scales, divide=True)
    if fast is not None:
        return fast
    out = _desaturate_near_clipped_highlights(xp, img, img / scales[None, None, :])
    return xp.clip(out, 0, None)


def build_hot_pixel_map(dark: np.ndarray, sigma_threshold: float = 5.0) -> np.ndarray:
    """Build a boolean hot pixel map from an unsmoothed dark frame.

    Detects pixels with dark current significantly above the background.
    Must be called BEFORE Gaussian smoothing of the master dark.
    """
    dark_f = dark.astype(np.float32)
    med = np.median(dark_f)
    # Use MAD-based sigma for robustness against the hot pixels themselves
    mad = np.median(np.abs(dark_f - med))
    sigma = mad * 1.4826  # MAD to Gaussian sigma conversion
    if sigma < 1e-6:
        sigma = float(np.std(dark_f))
    return dark_f > (med + sigma_threshold * sigma)


_DETECT = object()  # sentinel: use default threshold for statistical detection


def fix_hot_pixels(data: np.ndarray, mode: str = 'auto',
                   threshold: Optional[float] = _DETECT,
                   hot_map: Optional[np.ndarray] = None,
                   inplace: bool = False) -> np.ndarray:
    """Unified hot pixel detection and replacement.

    Modes:
        'bayer'  — Process each 2x2 Bayer sub-channel independently.
                   If *hot_map* is provided, those pixels are replaced.
                   Statistical detection also runs unless *threshold=None*.
        'rgb'    — Detect on luminance, replace all 3 channels.
        'mono'   — Single-channel statistical detection (GPU-accelerated).
        'auto'   — Infer from data shape: 2D → bayer, 3D → rgb.

    Args:
        threshold: Sigma multiplier for statistical detection. Pass *None*
                   to skip detection (useful when only applying a hot_map).
                   Defaults to the per-mode Config value.

    All modes use MAD-based sigma for robust noise estimation. ``inplace`` (bayer mode
    only) lets the statistical pass fix ``data`` itself when it can.
    """
    if mode == 'auto':
        mode = 'bayer' if data.ndim == 2 else 'rgb'

    if mode == 'bayer':
        return _fix_hot_bayer(data, threshold, hot_map, inplace=inplace)
    elif mode == 'rgb':
        rgb_fixed, _lum = _fix_hot_rgb(data, threshold)
        return rgb_fixed
    elif mode == 'mono':
        return _fix_hot_mono(data, threshold)
    else:
        raise ValueError(f"Unknown hot pixel mode: {mode!r}")


def _fix_hot_bayer(data: np.ndarray, threshold: Optional[float] = _DETECT,
                   hot_map: Optional[np.ndarray] = None,
                   star_support: Optional[float] = _DETECT,
                   inplace: bool = False) -> np.ndarray:
    """Bayer-aware hot pixel fix: apply pre-built map and/or statistical detection.

    Merged implementation: median_filter is computed once per sub-channel and
    reused for both hot-map replacement and statistical detection, halving the
    number of median filter calls when both are active.

    ``star_support`` (sigma; default ``Config.HOT_PIXEL_STAR_SUPPORT``, ``None`` turns
    the test off) protects stars from the statistical detector. Each colour plane is
    half resolution, so a star ~4 px wide is ~2 px wide there and its peak pixel stands
    far above its 3x3 plane median: without this the peak of every bright star was
    replaced by that median in every frame, clipping the cores (measured on a real
    session: stars 18% wider and the stack noisier than with the step off). A flagged
    pixel is kept when any adjacent mosaic pixel (one pixel away, in another plane) is
    itself more than ``star_support`` sigma over its own plane's median: a hot pixel is
    a single-sensor-pixel event and leaves its neighbours normal, a star lifts them.

    ``inplace=True`` (statistical pass, no map, a C-contiguous writable float32 array the
    caller owns) fixes ``data`` itself through the native ``hot_pixel_bayer_inplace``:
    the same result bit for bit in two streaming passes and no frame-sized output.
    """
    if data.ndim != 2:
        return data

    has_map = hot_map is not None and hot_map.shape == data.shape and np.any(hot_map)
    do_stat = threshold is not None
    if do_stat and threshold is _DETECT:
        threshold = Config.HOT_PIXEL_BAYER_THRESHOLD
    if star_support is _DETECT:
        star_support = Config.HOT_PIXEL_STAR_SUPPORT

    # Native: one call, no per-sub-plane temporaries; bit-identical (same f32 ops in
    # the same order). Map and statistics are never asked for together by the
    # pipeline, and the numpy path shares one median between them, so that
    # combination stays on numpy.
    if (inplace and do_stat and not has_map and _HAS_NATIVE
            and hasattr(_native, 'hot_pixel_bayer_inplace') and isinstance(data, np.ndarray)
            and data.dtype == np.float32 and data.flags['C_CONTIGUOUS']
            and data.flags['WRITEABLE']):
        try:
            _native.hot_pixel_bayer_inplace(
                data, float(threshold), None if star_support is None else float(star_support))
            return data
        except Exception:
            pass
    if (_HAS_NATIVE and hasattr(_native, 'hot_pixel_bayer') and isinstance(data, np.ndarray)
            and (has_map or do_stat) and not (has_map and do_stat)):
        try:
            src = np.ascontiguousarray(data, dtype=np.float32)
            if has_map:
                return _native.hot_pixel_bayer(
                    src, np.ascontiguousarray(hot_map, dtype=np.uint8), None)
            return _native.hot_pixel_bayer(
                src, None, float(threshold),
                None if star_support is None else float(star_support))
        except Exception:
            pass

    result = data.astype(np.float32, copy=True)
    if not has_map and not do_stat:
        return result

    # Pass 1, per sub-plane: median, map replacement, and (for the statistics) the excess
    # over the median and its sigma. Pass 2 needs every plane's excess at once, because the
    # star test looks at neighbours that live in the other planes.
    planes = []
    for dy in range(2):
        for dx in range(2):
            sub = result[dy::2, dx::2]
            # Compute median once; used for both map application and statistics.
            # sub is a strided view (every other row/col) -- _median_filter3's
            # native fast path requires a C-contiguous array, so route through
            # a contiguous copy rather than falling through to scipy's much
            # slower generic ndimage.median_filter for every sub-channel.
            med = _median_filter3(np.ascontiguousarray(sub), np, ndimage)

            if has_map:
                map_mask = hot_map[dy::2, dx::2]
                if np.any(map_mask):
                    sub[map_mask] = med[map_mask]
                    # sub is a view of result — no need to write back.

            diff = sigma = None
            if do_stat:
                diff = sub - med
                mad = np.median(np.abs(diff))
                sigma = mad * 1.4826
                if sigma < 1e-6:
                    diff = sigma = None
            planes.append((dy, dx, sub, med, diff, sigma))

    if not do_stat:
        return result

    near = None
    if star_support is not None:
        z = np.zeros(result.shape, dtype=np.float32)
        for dy, dx, _sub, _med, diff, sigma in planes:
            if diff is not None and np.isfinite(sigma):
                z[dy::2, dx::2] = diff / sigma
        pad = np.full((result.shape[0] + 2, result.shape[1] + 2), -np.inf, dtype=np.float32)
        pad[1:-1, 1:-1] = z
        near = np.maximum.reduce([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])

    for dy, dx, sub, med, diff, sigma in planes:
        if diff is None:
            continue
        stat_mask = diff > threshold * sigma
        if near is not None:
            stat_mask &= ~(near[dy::2, dx::2] > star_support)
        if np.any(stat_mask):
            sub[stat_mask] = med[stat_mask]

    return result


def _median_filter3(arr, xp, _nd):
    """3x3 median filter, native (Rust) on the CPU/numpy path when available —
    ~10x faster than scipy's generic rank filter for this small, fixed
    footprint. GPU (cupy) path is untouched; falls back to scipy on any error.
    """
    if (xp is np and _HAS_NATIVE and arr.dtype == np.float32
            and arr.ndim == 2 and arr.flags['C_CONTIGUOUS']):
        try:
            return _native.median_filter_native(arr, 3)
        except Exception:
            pass
    return _nd.median_filter(arr, size=3)


def _fix_hot_rgb_impl(rgb, threshold, xp, _nd, inplace=False):
    """Pure implementation of RGB hot-pixel correction; works with numpy or CuPy.

    Detection uses a single luma median filter (1 pass instead of 3).
    Replacement uses uniform_filter (box mean) which is ~5x faster than
    median on GPU and gives equivalent quality for isolated hot pixels.
    """
    if (xp is np and _HAS_NATIVE and hasattr(_native, 'hot_pixel_rgb')
            and isinstance(rgb, np.ndarray) and rgb.dtype == np.float32
            and rgb.ndim == 3 and rgb.shape[2] == 3 and threshold is not None):
        # Fused luma + 3x3 median + MAD + sparse replacement, bit-identical to the
        # numpy steps below; None means a degenerate MAD (numpy falls back to np.std).
        try:
            if (inplace and hasattr(_native, 'hot_pixel_rgb_inplace')
                    and rgb.flags['C_CONTIGUOUS'] and rgb.flags['WRITEABLE']):
                # only the flagged pixels are written: no output array, no copy
                lum_ip = _native.hot_pixel_rgb_inplace(rgb, float(threshold))
                if lum_ip is not None:
                    return rgb, lum_ip
                fused = None
            else:
                fused = _native.hot_pixel_rgb(np.ascontiguousarray(rgb), float(threshold))
        except Exception:
            fused = None
        if fused is not None:
            fixed, lum_out = fused
            return (rgb if fixed is None else fixed), lum_out

    lum     = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    med_lum = _median_filter3(lum, xp, _nd)
    diff    = lum - med_lum
    mad     = float(xp.median(xp.abs(diff)))
    sigma   = mad * 1.4826
    if sigma < 1e-6:
        sigma = float(xp.std(diff))
    mask = diff > threshold * sigma
    if not bool(xp.any(mask)):
        return rgb, lum

    if xp is np and _HAS_NATIVE and hasattr(_native, 'hot_pixel_box_replace_native'):
        try:
            result = _native.hot_pixel_box_replace_native(
                np.ascontiguousarray(rgb, dtype=np.float32),
                np.ascontiguousarray(mask, dtype=np.uint8))
            lum_fixed = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
            return result, lum_fixed
        except Exception:
            pass

    result = xp.empty_like(rgb)
    for c in range(rgb.shape[2]):
        ch = rgb[:, :, c]
        result[:, :, c] = xp.where(mask, _nd.uniform_filter(ch, size=3), ch)
    lum_fixed = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
    return result, lum_fixed


def _fix_hot_rgb(rgb: np.ndarray, threshold: Optional[float] = _DETECT, inplace: bool = False):
    """Detect hot pixels on luminance, fix all 3 channels.

    Computes per-channel medians once (3 passes), reconstructs median
    luminance from them, and reuses medians for replacement.

    Returns (rgb_fixed, lum) so the caller can reuse the luminance array
    instead of recomputing it.  Falls back to CPU scipy on GPU OOM.
    """
    gpu = get_gpu()
    if threshold is _DETECT:
        threshold = Config.HOT_PIXEL_THRESHOLD
    if gpu.active:
        try:
            return _fix_hot_rgb_impl(gpu.xp.asarray(rgb), threshold, gpu.xp, gpu.xndimage)
        except Exception as exc:
            if gpu.is_oom(exc):
                gpu.disable()
            else:
                raise
    return _fix_hot_rgb_impl(np.asarray(rgb), threshold, np, ndimage, inplace=inplace)


def _fix_hot_mono_impl(img, threshold, xp, _nd):
    """Pure implementation of mono hot-pixel correction; works with numpy or CuPy."""
    med   = _median_filter3(img, xp, _nd)
    diff  = img - med
    mad   = float(xp.median(xp.abs(diff)))
    sigma = mad * 1.4826
    if sigma < 1e-6:
        sigma = float(xp.std(diff))
    mask = diff > threshold * sigma
    if not bool(xp.any(mask)):
        return img

    if xp is np and _HAS_NATIVE and hasattr(_native, 'hot_pixel_box_replace_native'):
        try:
            img3 = np.ascontiguousarray(img, dtype=np.float32)[:, :, np.newaxis]
            result = _native.hot_pixel_box_replace_native(
                img3, np.ascontiguousarray(mask, dtype=np.uint8))
            return result[:, :, 0]
        except Exception:
            pass

    return xp.where(mask, _nd.uniform_filter(img, size=3), img)


def _fix_hot_mono(img: np.ndarray, threshold: Optional[float] = _DETECT) -> np.ndarray:
    """Single-channel hot pixel detection and replacement.  Falls back to CPU on GPU OOM."""
    gpu = get_gpu()
    if threshold is _DETECT:
        threshold = Config.HOT_PIXEL_THRESHOLD
    if gpu.active:
        try:
            return _fix_hot_mono_impl(gpu.xp.asarray(img), threshold, gpu.xp, gpu.xndimage)
        except Exception as exc:
            if gpu.is_oom(exc):
                gpu.disable()
            else:
                raise
    return _fix_hot_mono_impl(np.asarray(img), threshold, np, ndimage)


# Legacy aliases for backwards compatibility
remove_hot_pixels = _fix_hot_mono
remove_hot_pixels_bayer = lambda data, threshold=_DETECT, inplace=False: fix_hot_pixels(
    data, mode='bayer', threshold=threshold, inplace=inplace)


def _spike_reject_bayer_numpy(data: np.ndarray, k: float, contrast: float,
                              support_frac: float):
    """numpy mirror of the native ``spike_reject_bayer`` (bit-identical: same
    f32 operations in the same order). See ``remove_spikes_bayer``."""
    d = np.ascontiguousarray(data, dtype=np.float32)
    H, W = d.shape
    out = d.copy()
    k32, c32, f32 = np.float32(k), np.float32(contrast), np.float32(support_frac)
    stage, maxes = {}, {}
    for dy in (0, 1):
        for dx in (0, 1):
            P = d[dy::2, dx::2]
            hh, ww = P.shape
            if hh < 2 or ww < 2:
                stage[(dy, dx)] = (None, np.float32(0), np.float32(0))
                continue
            pad = np.pad(P, 1, mode='symmetric')      # == reflect_idx (edge repeated)
            nb = np.stack([pad[1 + oy:1 + oy + hh, 1 + ox:1 + ox + ww]
                           for oy in (-1, 0, 1) for ox in (-1, 0, 1) if (oy, ox) != (0, 0)])
            nb.sort(axis=0)
            m8 = (nb[3] + nb[4]) * np.float32(0.5)
            maxes[(dy, dx)] = nb[7]
            if np.isnan(P).any():
                stage[(dy, dx)] = (m8, np.float32(0), np.float32(0))
                continue
            # statistics over every other row/column of the plane (as the kernel)
            ad = np.abs(P[::2, ::2] - m8[::2, ::2])
            sigma = np.float32(np.median(ad)) * np.float32(1.4826)
            bg = np.float32(np.median(P[::2, ::2]))
            if not sigma >= 1e-6:
                sigma = np.float32(0)
            stage[(dy, dx)] = (m8, sigma, bg)
    bgmap = np.empty_like(d)
    for (dy, dx), (_, _, bg) in stage.items():
        bgmap[dy::2, dx::2] = bg
    above = d - bgmap                                  # each pixel over its own plane median
    n = 0
    for (dy, dx), (m8, sigma, bg) in stage.items():
        if m8 is None or sigma <= 0:
            continue
        P = d[dy::2, dx::2]
        m8max = maxes[(dy, dx)]
        exc = P - m8
        lift = m8 - bg
        denom = np.where(lift > sigma, lift, sigma)
        cand = (exc > k32 * sigma) & (exc > c32 * denom) & (P > m8max)
        if not cand.any():
            continue
        lim = f32 * (P - bg)
        ys, xs = np.nonzero(cand)
        yy, xx = 2 * ys + dy, 2 * xs + dx
        up = np.zeros(len(ys), np.int32)
        for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = yy + oy, xx + ox
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            val = np.full(len(ys), -np.inf, np.float32)
            val[ok] = above[ny[ok], nx[ok]]
            up += (val > lim[ys, xs]).astype(np.int32)
        fl = up < 3
        out[yy[fl], xx[fl]] = m8[ys[fl], xs[fl]]
        n += int(fl.sum())
    return out, n


def remove_spikes_bayer(data: np.ndarray, k: Optional[float] = None,
                        contrast: Optional[float] = None,
                        support_frac: Optional[float] = None):
    """Replace sharp single/few-pixel spikes (cosmic rays, residual hot pixels)
    in a calibrated 2-D mosaic by their same-plane 8-neighbour median, before
    debayering smears them into their neighbours. Returns ``(cleaned, n)``.
    A candidate must also be brighter than all 8 same-plane neighbours: on a
    steep, undersampled star profile a wing pixel otherwise passes the other
    tests (most of its neighbours are sky), but it always has a brighter
    same-plane neighbour on the side of the core.

    ``--spike-reject``: a cheap, correctly noise-scaled alternative to per-frame
    L.A.Cosmic. Noise is measured per Bayer plane from the frame itself (MAD
    of the pixel-minus-neighbour-median residual over every other row/column),
    so it needs no gain/read-noise model. A candidate must be significant (``k`` sigma over the median of
    its 8 same-plane neighbours), sharp (``contrast`` times the neighbours' own
    lift above the plane median -- a sampled star lifts its 2-px neighbours
    to a large fraction of its peak), and must not lift 3+ of its 4 adjacent
    mosaic pixels (a star, even an undersampled one, lifts nearly all of
    them). Stars survive where ``hot_pixel_bayer``'s star-support rule is
    needed, but a 2-pixel hit (each pixel lifting the other) is still caught.
    Measured on a real 20 s Origin sub: ~200 events per frame, 1-2 px each.
    """
    k = Config.SPIKE_REJECT_SIGMA if k is None else k
    contrast = Config.SPIKE_REJECT_CONTRAST if contrast is None else contrast
    support_frac = Config.SPIKE_REJECT_SUPPORT_FRAC if support_frac is None else support_frac
    if data.ndim != 2:
        return data, 0
    if _HAS_NATIVE and hasattr(_native, 'spike_reject_bayer') and isinstance(data, np.ndarray):
        try:
            return _native.spike_reject_bayer(np.ascontiguousarray(data, dtype=np.float32),
                                              float(k), float(contrast), float(support_frac))
        except Exception:
            pass
    return _spike_reject_bayer_numpy(data, k, contrast, support_frac)
apply_hot_pixel_map_bayer = lambda data, hot_map: fix_hot_pixels(data, mode='bayer', hot_map=hot_map, threshold=None)


def luminance(rgb):
    """0.299 R + 0.587 G + 0.114 B in float32 -- one native pass for a host float32
    image (bit-identical to the numpy expression, which builds three temporaries)."""
    if (_HAS_NATIVE and hasattr(_native, 'luminance_native') and isinstance(rgb, np.ndarray)
            and rgb.dtype == np.float32 and rgb.ndim == 3 and rgb.shape[2] == 3
            and rgb.flags['C_CONTIGUOUS']):
        try:
            return _native.luminance_native(rgb)
        except Exception:
            pass
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def calibrate_frame_be16(raw, bzero, bias, dark, dark_scale, flat_norm, out=None):
    """``calibrate_frame`` on a raw BITPIX=16 block from ``io_fits.read_fits_be16``:
    byte swap, + BZERO, the float32 conversion and the calibration in one native pass
    (the same per-pixel operations, so identical to load_fits + calibrate_frame).

    ``out`` (optional): a C-contiguous float32 array of ``raw``'s shape to write into
    instead of a fresh one -- every element is overwritten, and a reused buffer skips
    the OS zero-filling a new 25 MB allocation on first touch."""
    masters = [m for m in (bias, dark, flat_norm) if m is not None]
    if (_HAS_NATIVE and hasattr(_native, 'calibrate_frame_from_be16')
            and all(isinstance(m, np.ndarray) and m.dtype == np.float32
                    and m.flags['C_CONTIGUOUS'] for m in masters)):
        try:
            if not (isinstance(out, np.ndarray) and out.shape == raw.shape
                    and out.dtype == np.float32 and out.flags['C_CONTIGUOUS']
                    and out.flags['WRITEABLE']):
                out = np.empty(raw.shape, np.float32)
            finite = _native.calibrate_frame_from_be16(
                np.ascontiguousarray(raw).reshape(-1), float(bzero), out.reshape(-1),
                None if bias is None else bias.reshape(-1),
                None if dark is None else dark.reshape(-1),
                float(dark_scale),
                None if flat_norm is None else flat_norm.reshape(-1))
            return out, bool(finite)
        except Exception:
            pass
    data = (raw.byteswap().view(np.int16).astype(np.int32) + int(bzero)).astype(np.float32)
    return calibrate_frame(data, bias, dark, dark_scale, flat_norm)


def calibrate_frame(data, bias, dark, dark_scale, flat_norm):
    """Bias / scaled-dark / flat calibration of one light frame, clipped at zero.

    ``bias``, ``dark`` and ``flat_norm`` are same-shape masters or None. Returns
    ``(calibrated, finite)``; ``finite`` is False when the result held a non-finite
    value (the caller reports that as an error, as before). ``data`` is updated in
    place when it is already a writable float32 array, exactly as the numpy steps
    always did.

    Native (float32 masters): one parallel pass, same per-element operations in the
    same order as the numpy sequence, so bit-identical -- it replaces ~6 full-frame
    passes and two temporaries (``dark * scale``, the clip's copy).
    """
    masters = [m for m in (bias, dark, flat_norm) if m is not None]
    if (_HAS_NATIVE and hasattr(_native, 'calibrate_frame_inplace') and masters
            and isinstance(data, np.ndarray)
            and all(isinstance(m, np.ndarray) and m.dtype == np.float32
                    and m.flags['C_CONTIGUOUS'] for m in masters)):
        try:
            work = data
            if work.dtype != np.float32 or not work.flags['C_CONTIGUOUS'] or not work.flags['WRITEABLE']:
                work = np.array(data, dtype=np.float32, order='C', copy=True)
            finite = _native.calibrate_frame_inplace(
                work.reshape(-1),
                None if bias is None else bias.reshape(-1),
                None if dark is None else dark.reshape(-1),
                float(dark_scale),
                None if flat_norm is None else flat_norm.reshape(-1))
            return work, bool(finite)
        except Exception:
            pass

    if bias is not None:
        data = data.astype(np.float32, copy=False)
        data -= bias
    if dark is not None:
        data = data.astype(np.float32, copy=False)
        np.subtract(data, dark * dark_scale, out=data)
        if bias is not None:
            data += bias * dark_scale
    if flat_norm is not None:
        data = data.astype(np.float32, copy=False)
        data /= flat_norm
    if not np.isfinite(data).all():
        return data, False
    return np.clip(data, 0, None), True


def _block_avg_2x(a: np.ndarray) -> np.ndarray:
    """2x2 block-average downsample (even-cropped). Preserves input dtype —
    call on float32 and cast to float64 afterward (on the now-small array) to
    avoid a wasted full-resolution float64 copy."""
    h2 = (a.shape[0] // 2) * 2
    w2 = (a.shape[1] // 2) * 2
    c = a[:h2, :w2]
    return (c[::2, ::2] + c[1::2, ::2] + c[::2, 1::2] + c[1::2, 1::2]) * 0.25


def correct_chromatic_aberration(rgb: np.ndarray, max_shift_px: float = 5.0,
                                  upsample: int = 10,
                                  downsample: int = 2) -> np.ndarray:
    """Correct lateral chromatic aberration by sub-pixel per-channel registration.

    Registers the red and blue channels against the green channel using phase
    cross-correlation and applies a sub-pixel shift to bring them into alignment.
    Correction is intentionally limited to ``max_shift_px`` pixels to avoid
    over-correcting in frames where phase correlation fails.

    Returns the original image unchanged if registration fails.

    The shift *estimate* runs on a ``downsample``x block-averaged copy of each
    channel — CA is a smooth, near-constant sub-pixel offset across the frame
    (lens dispersion), not fine per-pixel detail, so it survives a 2x
    downsample essentially exactly while cutting the dominant FFT cost by
    ~4x (N log N). Measured on real 233-frame runs this step was 3-4x slower
    under full parallel load than in isolation — consistent with the full-res
    FFTs being memory-bandwidth bound, which downsampling directly reduces.
    The recovered shift is scaled back up and applied to the FULL-resolution
    channel (accuracy of the *applied* correction is unaffected; only the
    *estimation* resolution changes) via the native Lanczos-3 warp when
    available, else scipy's cubic-spline shift.

    Args:
        rgb: Float32 image (H, W, 3), R/G/B order.
        max_shift_px: Maximum plausible CA shift in pixels (default 5).
                      Corrections larger than this are silently suppressed.
        upsample: Sub-pixel upsample factor for phase correlation (default 10),
                  applied at the downsampled scale.
        downsample: Block-average factor for the correlation estimate (default
                    2). Set to 1 to correlate at full resolution (old behaviour).
    """
    shifts = measure_chromatic_aberration(rgb, max_shift_px=max_shift_px,
                                          upsample=upsample,
                                          downsample=downsample)
    return apply_chromatic_aberration(rgb, shifts)


def measure_chromatic_aberration(rgb: np.ndarray, max_shift_px: float = 5.0,
                                 upsample: int = 10,
                                 downsample: int = 2) -> dict:
    """Measure the R/B channel offsets against G (see
    ``correct_chromatic_aberration``). Returns ``{0: (sy, sx) | None,
    2: (sy, sx) | None}`` keyed by channel index; None where measurement
    failed or exceeded ``max_shift_px``. Measurement and application are
    split so the session-constant CA (lens dispersion is fixed in the sensor
    frame for a whole session) can be measured once on a few sample frames
    and applied to every frame."""
    shifts: dict = {0: None, 2: None}
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return shifts

    # Downsample the cheap float32 data FIRST, cast to float64 only the small
    # result. Casting the full-res channel to float64 before downsampling (the
    # original order) wastes a 50MB copy per channel that gets thrown away
    # immediately — exactly the memory traffic this function is trying to cut.
    g_small = (_block_avg_2x(rgb[:, :, 1]) if downsample >= 2
              else rgb[:, :, 1]).astype(np.float64)
    g_std = g_small.std()
    if g_std < 1e-12:
        return shifts
    g_norm = (g_small - g_small.mean()) / g_std

    for c_idx in (0, 2):  # Red, Blue
        ch_small = (_block_avg_2x(rgb[:, :, c_idx]) if downsample >= 2
                   else rgb[:, :, c_idx]).astype(np.float64)
        ch_std = ch_small.std()
        if ch_std < 1e-12:
            continue
        ch_norm = (ch_small - ch_small.mean()) / ch_std
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                shift, error, _ = _pcc(g_norm, ch_norm, upsample_factor=upsample)
            scale = 2.0 if downsample >= 2 else 1.0
            shift = shift * scale
            if (np.isfinite(shift).all()
                    and np.abs(shift[0]) <= max_shift_px
                    and np.abs(shift[1]) <= max_shift_px):
                shifts[c_idx] = (float(shift[0]), float(shift[1]))
                _log.debug("CA measure ch%d: shift=(%.3f, %.3f) err=%.4f",
                           c_idx, float(shift[0]), float(shift[1]), float(error))
        except Exception as exc:
            _log.debug("CA measure ch%d failed: %s", c_idx, exc)

    return shifts


def apply_chromatic_aberration(rgb: np.ndarray, shifts: dict) -> np.ndarray:
    """Apply pre-measured CA channel shifts (see
    ``measure_chromatic_aberration``). Returns the input unchanged when no
    channel has a valid shift."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not shifts:
        return rgb
    if not any(shifts.get(c) is not None for c in (0, 2)):
        return rgb
    result = rgb.copy()
    H, W = rgb.shape[:2]
    for c_idx in (0, 2):
        shift = shifts.get(c_idx)
        if shift is None:
            continue
        if _HAS_NATIVE:
            try:
                off = [-float(shift[0]), -float(shift[1])]
                result[:, :, c_idx] = _native.warp_affine_lanczos3(
                    np.ascontiguousarray(rgb[:, :, c_idx:c_idx + 1]),
                    [1.0, 0.0, 0.0, 1.0], off, H, W, 0.0)[:, :, 0]
                continue
            except Exception:
                pass
        result[:, :, c_idx] = ndimage.shift(
            rgb[:, :, c_idx], shift=shift,
            order=3, mode='reflect').astype(np.float32)
    return result


def remove_hot_pixels_rgb_with_lum(rgb: np.ndarray, threshold: Optional[float] = None,
                                   inplace: bool = False):
    """Detect hot pixels on luminance, fix all 3 channels; returns (rgb_fixed, lum).

    Use this in performance-critical paths to avoid recomputing luminance after
    hot pixel removal. ``inplace=True`` lets the native path write the (few) repaired
    pixels into ``rgb`` itself instead of returning a copy -- for a caller that owns it.
    """
    if threshold is None:
        threshold = Config.HOT_PIXEL_THRESHOLD
    return _fix_hot_rgb(rgb, threshold=threshold, inplace=inplace)
