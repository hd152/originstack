"""2D biorthogonal wavelet transform (bior1.3) -- replaces PyWavelets for
src/denoising.py's wavelet_denoise/adaptive_wavelet_denoise, the only wavelet
family either function ever uses (the `wavelet` parameter defaults to
'bior1.3' and nothing in this codebase overrides it).

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

The bior1.3 filter coefficients themselves are pywt's own published values
(``pywt.Wavelet('bior1.3').filter_bank``) -- not re-derived, just copied
verbatim, since transcription of published filter coefficients from memory
would be needless risk for zero benefit (these are wavelet family
constants, not something to "port" in any meaningful sense).

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
matching the next detail array's shape.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

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


def dwt_max_level(data_len: int) -> int:
    """Exact port of pywt.dwt_max_level(data_len, filter_len) for bior1.3's
    filter length (6): floor(log2(data_len / (filter_len - 1)))."""
    if data_len <= 0:
        return 0
    ratio = data_len / (_FLEN - 1)
    if ratio < 1:
        return 0
    return int(np.floor(np.log2(ratio)))


def _dwt_1d(x: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Single-level 1D DWT along `axis`. Returns (cA, cD)."""
    n = x.shape[axis]
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (_FLEN - 1, _FLEN - 1)
    xp = np.pad(x, pad_width, mode='symmetric')
    out_len = (n + _FLEN - 1) // 2

    def _conv(arr: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda m: np.convolve(m, h, mode='valid'), axis, arr)

    cA = _conv(xp, _DEC_LO[::-1])
    cD = _conv(xp, _DEC_HI)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(_OFFSET, None, 2)
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


def _dwt2(img: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    La, Lh = _dwt_1d(img, axis=0)
    cA, cV = _dwt_1d(La, axis=1)
    cH, cD = _dwt_1d(Lh, axis=1)
    return cA, (cH, cV, cD)


def _idwt2(cA: np.ndarray, detail: Tuple[np.ndarray, np.ndarray, np.ndarray],
          out_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    cH, cV, cD = detail
    # Mirrors _dwt2's order (axis=0/rows first, then axis=1/cols) in reverse:
    # reconstruct along axis=1 (cols) first -- target column count is
    # out_shape[1] -- then axis=0 (rows) -- target row count out_shape[0].
    col_out_len = None if out_shape is None else out_shape[1]
    La = _idwt_1d(cA, cV, axis=1, out_len=col_out_len)
    Lh = _idwt_1d(cH, cD, axis=1, out_len=col_out_len)
    row_out_len = None if out_shape is None else out_shape[0]
    return _idwt_1d(La, Lh, axis=0, out_len=row_out_len)


def wavedec2(img: np.ndarray, level: int) -> Coeffs2D:
    """Multi-level 2D DWT -- same coefficient structure as
    pywt.wavedec2(img, 'bior1.3', level=level):
    [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]."""
    coeffs: Coeffs2D = []
    a = img
    for _ in range(level):
        a, detail = _dwt2(a)
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


def soft_threshold(data: np.ndarray, value: float) -> np.ndarray:
    """Soft-threshold shrinkage -- exact equivalent of
    pywt.threshold(data, value, mode='soft'):
    sign(x) * max(|x| - value, 0)."""
    if value < 0:
        raise ValueError("threshold value must be non-negative")
    magnitude = np.abs(data)
    return np.sign(data) * np.maximum(magnitude - value, 0.0)
