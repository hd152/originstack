//! Native `wavedec2`/`waverec2` building blocks.

use crate::*;

// ---------------------------------------------------------------------------
// 2D separable wavelet transform (bior1.3 / db4)
// ---------------------------------------------------------------------------
//
// Mirrors src/wavelet.py's _dwt_1d/_idwt_1d + _dwt2/_idwt2 exactly, but as a
// direct closed-form tap sum per output sample instead of
// pad-then-np.apply_along_axis-then-np.convolve (which loops in pure Python
// over every row/column -- see that module's docstring for why the exact
// convolve-vs-correlate orientation of each kernel matters and how it was
// determined). No padded array is ever materialised: `reflect_idx` computes
// the symmetric (edge-duplicating -- numpy's mode='symmetric', scipy's
// mode='reflect') boundary index directly, for any overshoot.

// Forward 1D DWT of a length-`n` line (computed inline in `dwt2_native`):
// exact port of `_dwt_1d`, `cA[j] = sum_k lo[k] * x[reflect_idx(offset+2j-k, n)]`,
// `cD` analogous with `hi` -- the direct closed form of
// `np.convolve(padded, kernel, mode='valid')[offset::2]` with no padding
// materialised (convolve flips one operand, so
// `valid[i] = sum_k kernel[k] * padded[i+flen-1-k]`, and substituting the
// padding offset collapses to this form).

/// One tap of the forward DWT's axis-0 pass over a whole row:
/// `a[c] += lk * src[c]; d[c] += hk * src[c]`. Separate multiply and add (no
/// FMA), so the AVX2 build gives the baseline build's bits.
#[inline(always)]
pub(crate) fn dwt_row_accumulate_body(a: &mut [f64], d: &mut [f64], src: &[f64], lk: f64, hk: f64) {
    let n = src.len();
    let (a, d) = (&mut a[..n], &mut d[..n]);
    for ((av, dv), &v) in a.iter_mut().zip(d.iter_mut()).zip(src.iter()) {
        *av += lk * v;
        *dv += hk * v;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dwt_row_accumulate_avx2(a: &mut [f64], d: &mut [f64], src: &[f64], lk: f64, hk: f64) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };
    let n = src.len();
    let vl = _mm256_set1_pd(lk);
    let vh = _mm256_set1_pd(hk);
    let mut i = 0;
    while i + 4 <= n {
        // Deliberately separate multiply and add: using _mm256_fmadd_pd
        // changes rounding and breaks the scalar parity contract.
        let x = _mm256_loadu_pd(src.as_ptr().add(i));
        let av = _mm256_loadu_pd(a.as_ptr().add(i));
        let dv = _mm256_loadu_pd(d.as_ptr().add(i));
        _mm256_storeu_pd(a.as_mut_ptr().add(i), _mm256_add_pd(av, _mm256_mul_pd(vl, x)));
        _mm256_storeu_pd(d.as_mut_ptr().add(i), _mm256_add_pd(dv, _mm256_mul_pd(vh, x)));
        i += 4;
    }
    while i < n {
        *a.get_unchecked_mut(i) += lk * *src.get_unchecked(i);
        *d.get_unchecked_mut(i) += hk * *src.get_unchecked(i);
        i += 1;
    }
}

pub(crate) fn dwt_row_accumulate(a: &mut [f64], d: &mut [f64], src: &[f64], lk: f64, hk: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU reports AVX2.
            unsafe { return dwt_row_accumulate_avx2(a, d, src, lk, hk) }
        }
    }
    dwt_row_accumulate_body(a, d, src, lk, hk)
}

/// Inverse 1D DWT of one line. Exact port of `_idwt_1d`: rather than
/// zero-stuffing `cA`/`cD` to length `2n` and running a full convolution,
/// only the even-position (nonzero) upsampled taps are summed directly --
/// `out[oi] = sum_k rec_lo[k]*cA[j] + rec_hi[k]*cD[j]` where
/// `j = (idwt_offset+oi-k)/2`, only when that quantity is a nonnegative
/// even integer within range.
pub(crate) fn idwt_1d_line(
    ca: &[f64],
    cd: &[f64],
    rec_lo: &[f64],
    rec_hi: &[f64],
    idwt_offset: usize,
    out_len: usize,
    out: &mut [f64],
) {
    let flen = rec_lo.len();
    let n_c = ca.len();
    for oi in 0..out_len {
        let n_pos = idwt_offset as isize + oi as isize;
        let mut acc = 0.0f64;
        for k in 0..flen {
            let m = n_pos - k as isize;
            if m >= 0 && m % 2 == 0 {
                let j = (m / 2) as usize;
                if j < n_c {
                    acc += rec_lo[k] * ca[j] + rec_hi[k] * cd[j];
                }
            }
        }
        out[oi] = acc;
    }
}

#[pyfunction]
pub(crate) fn dwt2_native<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, f64>,
    dec_lo: PyReadonlyArray1<'py, f64>,
    dec_hi: PyReadonlyArray1<'py, f64>,
    offset: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let arr = img.as_array();
    let s = arr.shape();
    let (h, w) = (s[0], s[1]);
    let lo: Vec<f64> = dec_lo.as_array().to_vec();
    let hi: Vec<f64> = dec_hi.as_array().to_vec();
    let flen = lo.len();
    let out_h = (h + flen - 1) / 2;
    let out_w = (w + flen - 1) / 2;
    // Kept in f64 throughout (unlike every other kernel in this file, which
    // downcast to f32): multi-level wavedec2/waverec2 chains many of these
    // calls together, and the numpy reference this is validated bit-exact
    // against (itself validated bit-exact against real pywt) runs entirely
    // in float64 -- rounding to f32 at each level's Python/Rust boundary
    // compounds across levels and breaks that bit-exact parity.
    let owned: Vec<f64>;
    let img64: &[f64] = match arr.as_slice() {
        Some(s) => s,
        None => {
            owned = arr.iter().copied().collect();
            &owned
        }
    };

    // Per-sample tap sums as in the closed form above (start at 0.0, add
    // lo[k]*x[reflect_idx(base-k)] for k = 0..flen in order, no FMA), laid
    // out for speed: pass 1 accumulates whole source rows into whole output
    // rows (contiguous, vectorisable) instead of gathering one strided column
    // at a time and transposing; pass 2 reads each row through a
    // symmetric-padded copy so the tap loop has no per-tap index arithmetic.
    // Bit-identical to the previous per-column/per-tap form.
    let (c_a, c_h, c_v, c_d) = py.detach(|| {
        // Pass 1: axis=0 -> La, Lh, each (out_h, w).
        let mut la = vec![0.0f64; out_h * w];
        let mut lh = vec![0.0f64; out_h * w];
        la.par_chunks_mut(w)
            .zip(lh.par_chunks_mut(w))
            .enumerate()
            .for_each(|(j, (la_row, lh_row))| {
                let base = offset as isize + 2 * j as isize;
                for k in 0..flen {
                    let r = reflect_idx(base - k as isize, h);
                    dwt_row_accumulate(la_row, lh_row, &img64[r * w..(r + 1) * w], lo[k], hi[k]);
                }
            });

        // Pass 2: axis=1 (per-row) on La -> cA, cV; on Lh -> cH, cD.
        let ext_len = offset + 2 * out_w.saturating_sub(1) + flen;
        let row_pass = |src_rows: &[f64], o1: &mut [f64], o2: &mut [f64]| {
            o1.par_chunks_mut(out_w)
                .zip(o2.par_chunks_mut(out_w))
                .enumerate()
                .for_each_init(
                    || Vec::<f64>::with_capacity(ext_len),
                    |ext, (r, (a_row, d_row))| {
                        let line = &src_rows[r * w..(r + 1) * w];
                        // ext[t] = line[reflect_idx(t - (flen - 1))]
                        ext.resize(ext_len, 0.0);
                        for (t, value) in ext.iter_mut().enumerate() {
                            *value = line[reflect_idx(t as isize - (flen as isize - 1), w)];
                        }
                        for j in 0..out_w {
                            // index base-k  ->  ext[base - k + flen - 1]
                            let top = offset + 2 * j + flen - 1;
                            let mut a = 0.0f64;
                            let mut d = 0.0f64;
                            for k in 0..flen {
                                let v = ext[top - k];
                                a += lo[k] * v;
                                d += hi[k] * v;
                            }
                            a_row[j] = a;
                            d_row[j] = d;
                        }
                    },
                );
        };
        let mut c_a = vec![0.0f64; out_h * out_w];
        let mut c_v = vec![0.0f64; out_h * out_w];
        row_pass(&la, &mut c_a, &mut c_v);
        let mut c_h = vec![0.0f64; out_h * out_w];
        let mut c_d = vec![0.0f64; out_h * out_w];
        row_pass(&lh, &mut c_h, &mut c_d);
        (c_a, c_h, c_v, c_d)
    });

    let mk = |v: Vec<f64>| -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(numpy::ndarray::Array2::from_shape_vec((out_h, out_w), v)
            .unwrap()
            .into_pyarray(py))
    };
    Ok((mk(c_a)?, mk(c_h)?, mk(c_v)?, mk(c_d)?))
}

#[pyfunction]
pub(crate) fn idwt2_native<'py>(
    py: Python<'py>,
    ca: PyReadonlyArray2<'py, f64>,
    ch: PyReadonlyArray2<'py, f64>,
    cv: PyReadonlyArray2<'py, f64>,
    cd: PyReadonlyArray2<'py, f64>,
    rec_lo: PyReadonlyArray1<'py, f64>,
    rec_hi: PyReadonlyArray1<'py, f64>,
    idwt_offset: usize,
    out_h: usize,
    out_w: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ca_arr = ca.as_array();
    let ch_arr = ch.as_array();
    let cv_arr = cv.as_array();
    let cd_arr = cd.as_array();
    let s = ca_arr.shape();
    let (in_h, in_w) = (s[0], s[1]);
    let rlo: Vec<f64> = rec_lo.as_array().to_vec();
    let rhi: Vec<f64> = rec_hi.as_array().to_vec();

    let ca64 = match ca_arr.as_slice() {
        Some(s) => std::borrow::Cow::Borrowed(s),
        None => std::borrow::Cow::Owned(ca_arr.iter().copied().collect()),
    };
    let ch64 = match ch_arr.as_slice() {
        Some(s) => std::borrow::Cow::Borrowed(s),
        None => std::borrow::Cow::Owned(ch_arr.iter().copied().collect()),
    };
    let cv64 = match cv_arr.as_slice() {
        Some(s) => std::borrow::Cow::Borrowed(s),
        None => std::borrow::Cow::Owned(cv_arr.iter().copied().collect()),
    };
    let cd64 = match cd_arr.as_slice() {
        Some(s) => std::borrow::Cow::Borrowed(s),
        None => std::borrow::Cow::Owned(cd_arr.iter().copied().collect()),
    };

    let out = py.detach(|| {
        // Pass 1: axis=1 (per-row) reconstruction -- La = idwt(cA,cV), Lh = idwt(cH,cD).
        // Both natural-length inputs have in_h rows, out_w columns.
        let mut la = vec![0.0f64; in_h * out_w];
        let mut lh = vec![0.0f64; in_h * out_w];
        la.par_chunks_mut(out_w)
            .zip(lh.par_chunks_mut(out_w))
            .enumerate()
            .for_each(|(r, (la_row, lh_row))| {
                let ca_row = &ca64[r * in_w..(r + 1) * in_w];
                let cv_row = &cv64[r * in_w..(r + 1) * in_w];
                idwt_1d_line(ca_row, cv_row, &rlo, &rhi, idwt_offset, out_w, la_row);
                let ch_row = &ch64[r * in_w..(r + 1) * in_w];
                let cd_row = &cd64[r * in_w..(r + 1) * in_w];
                idwt_1d_line(ch_row, cd_row, &rlo, &rhi, idwt_offset, out_w, lh_row);
            });

        let mut result = vec![0.0f64; out_h * out_w];
        // Axis=0 reconstruction, calculated row-major to avoid a pair of
        // temporary column vectors and the final column-to-row transpose.
        result
            .par_chunks_mut(out_w)
            .enumerate()
            .for_each(|(oi, row)| {
                let n_pos = idwt_offset as isize + oi as isize;
                for (c, value) in row.iter_mut().enumerate() {
                    let mut acc = 0.0f64;
                    for k in 0..rlo.len() {
                        let m = n_pos - k as isize;
                        if m >= 0 && m % 2 == 0 {
                            let j = (m / 2) as usize;
                            if j < in_h {
                                acc += rlo[k] * la[j * out_w + c] + rhi[k] * lh[j * out_w + c];
                            }
                        }
                    }
                    *value = acc;
                }
            });
        result
    });

    Ok(numpy::ndarray::Array2::from_shape_vec((out_h, out_w), out)
        .unwrap()
        .into_pyarray(py))
}
