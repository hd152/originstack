"""2D wavelet transform (bior1.3, db4) -- replaces PyWavelets everywhere it
was used in this codebase: src/denoising.py's wavelet_denoise/
adaptive_wavelet_denoise (bior1.3, forward+inverse) and
src/quality.py's compute_multiscale_entropy (db4, forward-only -- entropy
is computed directly on the decomposition coefficients, nothing is ever
reconstructed, so db4 only needs wavedec2/dwt_max_level, not waverec2).

Native/from-scratch implementation of the standard Mallat-algorithm 2D DWT
(separable: 1D DWT along rows, then along columns), 'symmetric' boundary
extension (pywt's default mode -- whole-point symmetric, i.e. the edge
sample is duplicated, same convention numpy.pad(mode='symmetric') uses),
and soft-threshold shrinkage. Validated bit-exact (see
tests/test_wavelet.py) against real PyWavelets (pywt.wavedec2/waverec2)
across many image sizes (even/odd height and width) and decomposition
depths, including the exact length semantics needed for a correct
multi-level reconstruction on odd-sized inputs (see module docstring notes
below and the test file for how that was reverse-engineered).

The filter coefficients themselves are pywt's own published values
(``pywt.Wavelet(name).filter_bank``) -- not re-derived, just copied
verbatim, since transcription of published filter coefficients from memory
would be needless risk for zero benefit (these are wavelet family
constants, not something to "port" in any meaningful sense). Each family's
convolution-alignment convention (whether the decomposition kernel needs
reversing before a `mode='valid'` correlation, and the output-slice offset)
was found the same way: brute-force search over {reversed, unreversed} x
{offset 0..filter_len-1} until the result matched real pywt.dwt output
exactly, since neither convention is standardized/guessable from theory
alone. bior1.3 needs its low-pass kernel reversed but not its high-pass;
db4 (orthogonal, not biorthogonal like bior1.3 -- no linear-phase symmetry)
needs neither reversed. Both use offset=1.

Length semantics note: a single-level 1D DWT of a length-n signal is
unambiguous (output length = (n + filter_len - 1) // 2), but the inverse
is not — reconstructing from (cA, cD) alone cannot recover whether the
original signal had an odd or even length matching that pair (they differ
by exactly the filter's overlap). PyWavelets resolves this the same way
this codebase's own pre-existing pywt-based code already did (see the
`[:h, :w]` crop after `pywt.waverec2` in the code this module replaces):
every intermediate reconstruction level naturally produces a
`2*len(cA) - (filter_len - 2)`-length result, each subsequent level's
result is cropped to match the true next-level detail-coefficient length
during the reconstruction cascade, and only the very final 2D output needs
a final crop to the original (H, W) -- never an intermediate one beyond
matching the next detail array's shape. (Reconstruction is only
implemented for bior1.3 -- the only family anything in this codebase ever
inverts.)
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

try:
    import astro_native as _native
    _HAS_NATIVE = True
except Exception:
    _native = None
    _HAS_NATIVE = False

# bior1.3 filter bank -- pywt.Wavelet('bior1.3').filter_bank, verbatim.
_DEC_LO = np.array([-0.08838834764831845, 0.08838834764831845, 0.7071067811865476,
                    0.7071067811865476, 0.08838834764831845, -0.08838834764831845])
_DEC_HI = np.array([-0.0, 0.0, -0.7071067811865476, 0.7071067811865476, -0.0, 0.0])
_REC_LO = np.array([0.0, 0.0, 0.7071067811865476, 0.7071067811865476, 0.0, 0.0])
_REC_HI = np.array([-0.08838834764831845, -0.08838834764831845, 0.7071067811865476,
                    -0.7071067811865476, 0.08838834764831845, 0.08838834764831845])
_FLEN = len(_DEC_LO)
# The convolution-alignment offset used by both _dwt_1d and _idwt_1d below
# (found by matching real pywt output exactly -- see module docstring).
_OFFSET = 1
_IDWT_OFFSET = 4

# db4 filter bank -- pywt.Wavelet('db4').filter_bank, verbatim. Decomposition
# only (see module docstring): no _REC_LO/_REC_HI needed.
_DB4_DEC_LO = np.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764,
                        -0.18703481171909309, -0.027983769416859854, 0.6308807679298589,
                        0.7148465705529157, 0.2303778133088965])
_DB4_DEC_HI = np.array([-0.2303778133088965, 0.7148465705529157, -0.6308807679298589,
                        -0.027983769416859854, 0.18703481171909309, 0.030841381835560764,
                        -0.0328830116668852, -0.010597401785069032])


class _FilterBank(NamedTuple):
    lo_kernel: np.ndarray   # correlation kernel for cA (mode='valid'), already oriented
    hi_kernel: np.ndarray   # correlation kernel for cD (mode='valid'), already oriented
    flen: int
    offset: int


_FILTER_BANKS: Dict[str, _FilterBank] = {
    'bior1.3': _FilterBank(_DEC_LO[::-1], _DEC_HI, _FLEN, _OFFSET),
    'db4': _FilterBank(_DB4_DEC_LO, _DB4_DEC_HI, len(_DB4_DEC_LO), 1),
}


def dwt_max_level(data_len: int, wavelet: str = 'bior1.3') -> int:
    """Exact port of pywt.dwt_max_level(data_len, filter_len):
    floor(log2(data_len / (filter_len - 1)))."""
    flen = _FILTER_BANKS[wavelet].flen
    if data_len <= 0:
        return 0
    ratio = data_len / (flen - 1)
    if ratio < 1:
        return 0
    return int(np.floor(np.log2(ratio)))


def _dwt_1d(x: np.ndarray, axis: int, bank: _FilterBank) -> Tuple[np.ndarray, np.ndarray]:
    """Single-level 1D DWT along `axis`. Returns (cA, cD)."""
    n = x.shape[axis]
    flen = bank.flen
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (flen - 1, flen - 1)
    xp = np.pad(x, pad_width, mode='symmetric')
    out_len = (n + flen - 1) // 2

    def _conv(arr: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda m: np.convolve(m, h, mode='valid'), axis, arr)

    cA = _conv(xp, bank.lo_kernel)
    cD = _conv(xp, bank.hi_kernel)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(bank.offset, None, 2)
    cA, cD = cA[tuple(sl)], cD[tuple(sl)]
    crop = [slice(None)] * x.ndim
    crop[axis] = slice(0, out_len)
    return cA[tuple(crop)], cD[tuple(crop)]


def _idwt_1d(cA: np.ndarray, cD: np.ndarray, axis: int,
            out_len: Optional[int] = None) -> np.ndarray:
    """Single-level 1D inverse DWT along `axis`. `out_len=None` gives the
    natural reconstruction length (2*len(cA) - (filter_len - 2)); pass an
    explicit `out_len` when chaining a multi-level reconstruction (crop to
    the next level's true detail-coefficient length)."""
    n = cA.shape[axis]

    def _upsample(c: np.ndarray) -> np.ndarray:
        shape = list(c.shape)
        shape[axis] *= 2
        u = np.zeros(shape, dtype=c.dtype)
        sl = [slice(None)] * c.ndim
        sl[axis] = slice(None, None, 2)
        u[tuple(sl)] = c
        return u

    def _conv(arr: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda m: np.convolve(m, h, mode='full'), axis, arr)

    full = _conv(_upsample(cA), _REC_LO) + _conv(_upsample(cD), _REC_HI)
    natural_len = 2 * n - (_FLEN - 2)
    if out_len is None:
        out_len = natural_len
    sl = [slice(None)] * cA.ndim
    sl[axis] = slice(_IDWT_OFFSET, _IDWT_OFFSET + out_len)
    return full[tuple(sl)]


# 2D coefficient tuple: (cA, (cH, cV, cD)) per level, matching pywt.dwt2's
# subband assignment -- axis=0 (rows) first, then axis=1 (cols): cA=low/low,
# cV=low/high, cH=high/low, cD=high/high.
Coeffs2D = List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]


def _dwt2(img: np.ndarray, bank: _FilterBank) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if _HAS_NATIVE:
        try:
            c_a, c_h, c_v, c_d = _native.dwt2_native(
                np.ascontiguousarray(img, dtype=np.float64),
                bank.lo_kernel, bank.hi_kernel, bank.offset)
            return c_a, (c_h, c_v, c_d)
        except Exception:
            pass
    La, Lh = _dwt_1d(img, axis=0, bank=bank)
    cA, cV = _dwt_1d(La, axis=1, bank=bank)
    cH, cD = _dwt_1d(Lh, axis=1, bank=bank)
    return cA, (cH, cV, cD)


def _idwt2(cA: np.ndarray, detail: Tuple[np.ndarray, np.ndarray, np.ndarray],
          out_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    cH, cV, cD = detail
    # Mirrors _dwt2's order (axis=0/rows first, then axis=1/cols) in reverse:
    # reconstruct along axis=1 (cols) first -- target column count is
    # out_shape[1] -- then axis=0 (rows) -- target row count out_shape[0].
    # Natural (no explicit out_shape) length resolved up front so both the
    # native and numpy paths below get concrete lengths.
    col_out_len = out_shape[1] if out_shape is not None else (2 * cA.shape[1] - (_FLEN - 2))
    row_out_len = out_shape[0] if out_shape is not None else (2 * cA.shape[0] - (_FLEN - 2))

    if _HAS_NATIVE:
        try:
            return _native.idwt2_native(
                np.ascontiguousarray(cA, dtype=np.float64),
                np.ascontiguousarray(cH, dtype=np.float64),
                np.ascontiguousarray(cV, dtype=np.float64),
                np.ascontiguousarray(cD, dtype=np.float64),
                _REC_LO, _REC_HI, _IDWT_OFFSET, row_out_len, col_out_len)
        except Exception:
            pass
    La = _idwt_1d(cA, cV, axis=1, out_len=col_out_len)
    Lh = _idwt_1d(cH, cD, axis=1, out_len=col_out_len)
    return _idwt_1d(La, Lh, axis=0, out_len=row_out_len)


def wavedec2(img: np.ndarray, level: int, wavelet: str = 'bior1.3') -> Coeffs2D:
    """Multi-level 2D DWT -- same coefficient structure as
    pywt.wavedec2(img, wavelet, level=level):
    [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]."""
    bank = _FILTER_BANKS[wavelet]
    coeffs: Coeffs2D = []
    a = img
    for _ in range(level):
        a, detail = _dwt2(a, bank)
        coeffs.append(detail)
    coeffs.append(a)
    return coeffs[::-1]


def waverec2(coeffs: Coeffs2D) -> np.ndarray:
    """Inverse of wavedec2. The result may be 0 or 1 sample larger than the
    original image along each axis (the same odd-length reconstruction
    ambiguity pywt itself has -- see module docstring); crop to the
    original (H, W) same as this codebase's pywt-based code already did."""
    a = coeffs[0]
    n_levels = len(coeffs) - 1
    for i in range(n_levels):
        detail = coeffs[i + 1]
        is_last = (i == n_levels - 1)
        out_shape = None if is_last else _detail_shape(coeffs[i + 2])
        a = _idwt2(a, detail, out_shape=out_shape)
    return a


def _detail_shape(detail_or_array) -> Tuple[int, int]:
    arr = detail_or_array[0] if isinstance(detail_or_array, tuple) else detail_or_array
    return arr.shape


def soft_threshold(data: np.ndarray, value) -> np.ndarray:
    """Soft-threshold shrinkage -- exact equivalent of
    pywt.threshold(data, value, mode='soft'):
    sign(x) * max(|x| - value, 0).

    ``value`` may be a scalar (the original, still the common case) or an
    array broadcastable against ``data`` (e.g. a per-pixel spatially
    adaptive threshold, as ``src.denoising.directional_wavelet_denoise``
    uses) -- ``np.any(... < 0)`` rather than a bare ``value < 0`` so the
    validation itself doesn't choke on a multi-element array (a bare
    comparison there raises "truth value of an array is ambiguous" instead
    of the intended "must be non-negative" error).
    """
    if np.any(np.asarray(value) < 0):
        raise ValueError("threshold value must be non-negative")
    magnitude = np.abs(data)
    return np.sign(data) * np.maximum(magnitude - value, 0.0)
