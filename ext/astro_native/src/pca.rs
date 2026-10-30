//! Robust-PCA and continuum-subtraction linear algebra.

use crate::*;

/// Gram matrix `D @ D.T` for a wide `(N, P)` matrix (N small, P huge -- the
/// robust-PCA calibration-stack shape: N frames, P = flattened pixels).
/// Parallel over the `N*(N+1)/2` upper-triangle `(i,j)` pairs (rayon), each a
/// length-P dot product. Lower triangle mirrored from the upper (symmetric
/// by construction).
///
/// This exists because `D @ D.T` (thin-SVD-via-Gram-matrix, i.e. eigh on the
/// small N x N Gram matrix instead of SVD-ing the full N x P matrix) is
/// ~2.3x faster than calling `np.linalg.svd` directly on a wide matrix in
/// the first place -- and on measurement, numpy's own `@` for this exact
/// shape got zero benefit from this machine's 16 cores (8.1s with default
/// threading vs 8.9s forced single-threaded), leaving real headroom for a
/// rayon-parallel version.
#[pyfunction]
pub(crate) fn gram_matrix_wide<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = data.as_array();
    let shape = arr.shape();
    let (n, p) = (shape[0], shape[1]);
    let flat: Option<&[f64]> = arr.as_slice();
    let flat = flat.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("data must be C-contiguous")
    })?;

    let pairs: Vec<(usize, usize)> =
        (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();

    let mut out = vec![0f64; n * n];
    py.detach(|| {
        let results: Vec<((usize, usize), f64)> = pairs
            .par_iter()
            .map(|&(i, j)| {
                let row_i = &flat[i * p..(i + 1) * p];
                let row_j = &flat[j * p..(j + 1) * p];
                let dot = row_i.iter().zip(row_j.iter()).fold(0.0f64, |acc, (&a, &b)| {
                    acc + a * b
                });
                ((i, j), dot)
            })
            .collect();
        for ((i, j), dot) in results {
            out[i * n + j] = dot;
            out[j * n + i] = dot;
        }
    });

    let arr2 = numpy::ndarray::Array2::from_shape_vec((n, n), out)
        .expect("shape mismatch building gram_matrix_wide output");
    Ok(arr2.into_pyarray(py))
}

/// `small @ data` for a small `(N, N)` left operand and a wide `(N, P)`
/// right operand -- the `U.T @ D` back-projection step of the Gram-matrix
/// thin-SVD trick (`gram_matrix_wide` above computes the other GEMM in that
/// same trick). Parallel over the N output rows; each row is a
/// length-N-term linear combination of `data`'s N rows.
#[pyfunction]
pub(crate) fn small_times_wide<'py>(
    py: Python<'py>,
    small: PyReadonlyArray2<'py, f64>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let small_arr = small.as_array();
    let data_arr = data.as_array();
    let n = small_arr.shape()[0];
    if small_arr.shape()[1] != n || data_arr.shape()[0] != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "small must be (N,N) and data must be (N,P) with matching N",
        ));
    }
    let p = data_arr.shape()[1];
    let small_flat: Option<&[f64]> = small_arr.as_slice();
    let data_flat: Option<&[f64]> = data_arr.as_slice();
    let (small_flat, data_flat) = match (small_flat, data_flat) {
        (Some(s), Some(d)) => (s, d),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "small and data must both be C-contiguous",
            ))
        }
    };

    // Loop order matters: accumulate axpy-style (out_row += w[i] * data_row_i)
    // so the inner loop over `col` reads each data row sequentially. The
    // naive col-outer/i-inner order would stride by P between consecutive
    // reads -- exactly the huge-stride-read-stream antipattern this file's
    // gather-transpose driver (`row_parallel`) exists to avoid elsewhere.
    //
    // Tried and reverted: nested rayon parallelism (also column-chunking
    // within each row) on the theory that N-way row parallelism leaves cores
    // idle when N (a frame/calibration-stack count, often ~10-20) is well
    // under the machine's core count. Measured no improvement (a real
    // robust-PCA call: 0.41-0.47s/call either way) -- this operation is
    // memory-bandwidth-bound on this machine, same lesson `gram_matrix_wide`
    // above already documents for the sibling GEMM in this same trick
    // (`numpy's own @ got zero benefit from this machine's cores at this
    // shape`). Kept simple since the added chunking bought nothing real.
    let mut out = vec![0f64; n * p];
    py.detach(|| {
        par_rows(&mut out, p, |k, out_row| {
            let weights = &small_flat[k * n..(k + 1) * n];
            for i in 0..n {
                let w = weights[i];
                let data_row = &data_flat[i * p..(i + 1) * p];
                for (o, &d) in out_row.iter_mut().zip(data_row.iter()) {
                    *o += w * d;
                }
            }
        });
    });

    let arr2 = numpy::ndarray::Array2::from_shape_vec((n, p), out)
        .expect("shape mismatch building small_times_wide output");
    Ok(arr2.into_pyarray(py))
}

/// Fused `D - S + Y/mu`: the IALM L-update's per-iteration input to the thin
/// SVD in `_thin_svd_wide` (`src/robust_pca.py::robust_pca_decompose`). The
/// numpy reference builds this as two full-`(N,P)`-array passes (`Y/mu` into
/// a temporary, then `D - S + that` into another); profiling a real
/// `--flat-from-lights` run (N=10, P=6.25M) found `robust_pca_decompose`
/// spending 100s of its 126s total in exactly this kind of elementwise numpy
/// arithmetic around the SVD, not in the SVD itself -- this and
/// `robust_pca_iterate` below fuse that arithmetic into one native pass each,
/// same category of win as `calibrate_frame_inplace`. Same float64 operation
/// order as the numpy reference (divide, then left-to-right subtract/add), so
/// the result is bit-identical.
#[pyfunction]
pub(crate) fn robust_pca_pre_svd_input<'py>(
    py: Python<'py>,
    d: PyReadonlyArray2<'py, f64>,
    s: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    mu: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d_arr = d.as_array();
    let shape = d_arr.shape().to_vec();
    if s.as_array().shape() != d_arr.shape() || y.as_array().shape() != d_arr.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "d, s, y must all have the same shape",
        ));
    }
    let d_flat: &[f64] = d_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("d must be C-contiguous"))?;
    let s_arr = s.as_array();
    let s_flat: &[f64] = s_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("s must be C-contiguous"))?;
    let y_arr = y.as_array();
    let y_flat: &[f64] = y_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("y must be C-contiguous"))?;

    const CH: usize = 1 << 16;
    let mut out = vec![0f64; d_flat.len()];
    py.detach(|| {
        out.par_chunks_mut(CH)
            .zip(d_flat.par_chunks(CH))
            .zip(s_flat.par_chunks(CH))
            .zip(y_flat.par_chunks(CH))
            .for_each(|(((oc, dc), sc), yc)| {
                for i in 0..oc.len() {
                    oc[i] = dc[i] - sc[i] + yc[i] / mu;
                }
            });
    });

    let arr2 = numpy::ndarray::Array2::from_shape_vec((shape[0], shape[1]), out)
        .expect("shape mismatch building robust_pca_pre_svd_input output");
    Ok(arr2.into_pyarray(py))
}

/// Fused IALM S-update + residual + Y-update + residual-norm: the second half
/// of `robust_pca_decompose`'s per-iteration elementwise arithmetic (see
/// `robust_pca_pre_svd_input` above for the profiling context). The numpy
/// reference computes `temp = D - L + Y/mu`, `S = sign(temp) *
/// max(|temp| - lam/mu, 0)`, `residual = D - L - S`, `Y += mu * residual`,
/// then `norm(residual, 'fro')` as roughly a dozen separate full-array passes
/// (each its own temporary); this does all of it in one pass, writing `S` and
/// `Y` in place (`np.zeros_like(D)`-allocated once by the caller, reused
/// every iteration -- no repeated `(N,P)` allocation) and returning the
/// residual's Frobenius norm directly. Per-element arithmetic matches numpy's
/// operation order exactly (`np.sign` semantics: 0 at exactly 0, not
/// `f64::signum`'s +-1), so `S`/`Y` are bit-identical to the numpy reference;
/// the norm is a parallel reduction over chunks and so may differ from
/// numpy's sequential sum by a few ULPs, same as this file's other
/// parallel-reduction kernels -- immaterial here since it only feeds a
/// convergence-tolerance comparison.
#[pyfunction]
pub(crate) fn robust_pca_iterate<'py>(
    py: Python<'py>,
    d: PyReadonlyArray2<'py, f64>,
    l: PyReadonlyArray2<'py, f64>,
    mut s: numpy::PyReadwriteArray2<'py, f64>,
    mut y: numpy::PyReadwriteArray2<'py, f64>,
    lam_over_mu: f64,
    mu: f64,
) -> PyResult<f64> {
    let d_arr = d.as_array();
    let l_arr = l.as_array();
    if l_arr.shape() != d_arr.shape()
        || s.as_array().shape() != d_arr.shape()
        || y.as_array().shape() != d_arr.shape()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "d, l, s, y must all have the same shape",
        ));
    }
    let d_flat: &[f64] = d_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("d must be C-contiguous"))?;
    let l_flat: &[f64] = l_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("l must be C-contiguous"))?;
    let s_flat: &mut [f64] = s
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("s must be C-contiguous"))?;
    let y_flat: &mut [f64] = y
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("y must be C-contiguous"))?;

    const CH: usize = 1 << 16;
    let sq_sum: f64 = py.detach(|| {
        d_flat
            .par_chunks(CH)
            .zip(l_flat.par_chunks(CH))
            .zip(s_flat.par_chunks_mut(CH))
            .zip(y_flat.par_chunks_mut(CH))
            .map(|(((dc, lc), sc), yc)| {
                let mut acc = 0.0f64;
                for i in 0..dc.len() {
                    let temp = dc[i] - lc[i] + yc[i] / mu;
                    let abs_temp = temp.abs();
                    let shrunk = (abs_temp - lam_over_mu).max(0.0);
                    let sign = if temp > 0.0 {
                        1.0
                    } else if temp < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    let sval = sign * shrunk;
                    sc[i] = sval;
                    let resid = dc[i] - lc[i] - sval;
                    yc[i] += mu * resid;
                    acc += resid * resid;
                }
                acc
            })
            .sum()
    });
    Ok(sq_sum.sqrt())
}

/// Single-pass central moments of a masked (narrowband, continuum) pixel
/// pair, for `optimal_continuum_scale`'s closed-form skewness-vs-scale
/// polynomial (`src/channel_combine.py`): the subtraction residual
/// `a - s*b` is linear in `s`, so its skewness at every candidate scale is
/// a fixed rational function of these 7 scalars -- computed once here
/// instead of re-scanning the full pixel array once per swept scale (the
/// real algorithmic win; this kernel is the constant-factor win on top of
/// that). Two passes (mean, then central moments) deliberately mirror the
/// numpy fallback's own two-step logic instead of a single-pass raw-moment
/// shift-formula, so there's no separate algebra to risk transcribing
/// wrong -- the win here is fusing what the numpy path computes as ~7
/// separate full-array elementwise-power passes (each its own temporary
/// array) into one pass per stage. Intentionally not rayon-parallelised:
/// called once per `optimal_continuum_scale` call (not per swept scale),
/// so unlike this file's per-frame/per-pixel hot-path kernels there's no
/// outer loop multiplying its cost.
#[pyfunction]
pub(crate) fn continuum_scale_moments(
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
) -> PyResult<(usize, f64, f64, f64, f64, f64, f64, f64)> {
    let a = a.as_slice()?;
    let b = b.as_slice()?;
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "a and b must have the same length",
        ));
    }
    let n = a.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("a and b must be non-empty"));
    }

    let (sum_a, sum_b) = a.iter().zip(b.iter())
        .fold((0.0f64, 0.0f64), |(sa, sb), (&av, &bv)| (sa + av, sb + bv));
    let nf = n as f64;
    let mean_a = sum_a / nf;
    let mean_b = sum_b / nf;

    let (s20, s11, s02, s30, s21, s12, s03) = a.iter().zip(b.iter()).fold(
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64),
        |(s20, s11, s02, s30, s21, s12, s03), (&av, &bv)| {
            let ap = av - mean_a;
            let bp = bv - mean_b;
            let ap2 = ap * ap;
            let bp2 = bp * bp;
            (
                s20 + ap2,
                s11 + ap * bp,
                s02 + bp2,
                s30 + ap2 * ap,
                s21 + ap2 * bp,
                s12 + ap * bp2,
                s03 + bp2 * bp,
            )
        },
    );

    Ok((n, s20 / nf, s11 / nf, s02 / nf, s30 / nf, s21 / nf, s12 / nf, s03 / nf))
}
