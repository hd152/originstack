//! Satellite-trail helpers: Bresenham lines, Hough accumulator.

use crate::*;

// ---------------------------------------------------------------------------
// Bresenham line rasterization (src/trail_reject.py's `_bresenham_line`)
// ---------------------------------------------------------------------------
//
// Textbook zero-vectorization case: a sequential line-walk with pure scalar
// integer arithmetic, no numpy call it could hide behind. Direct port of
// the existing (skimage-validated) symmetric error-term formulation, so
// results match integer-for-integer.

#[pyfunction]
pub(crate) fn bresenham_line_native<'py>(
    py: Python<'py>,
    r0: i64,
    c0: i64,
    r1: i64,
    c1: i64,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let mut r = r0;
    let mut c = c0;
    let mut dr = (r1 - r0).abs();
    let mut dc = (c1 - c0).abs();
    let mut sr: i64 = if r1 - r0 > 0 { 1 } else { -1 };
    let mut sc: i64 = if c1 - c0 > 0 { 1 } else { -1 };
    let steep = dr > dc;
    if steep {
        std::mem::swap(&mut r, &mut c);
        std::mem::swap(&mut dr, &mut dc);
        std::mem::swap(&mut sr, &mut sc);
    }

    let mut d = 2 * dr - dc;
    let n = (dc + 1) as usize;
    let mut rr = vec![0i64; n];
    let mut cc = vec![0i64; n];
    for i in 0..(dc as usize) {
        if steep {
            rr[i] = c;
            cc[i] = r;
        } else {
            rr[i] = r;
            cc[i] = c;
        }
        while d >= 0 {
            r += sr;
            d -= 2 * dc;
        }
        c += sc;
        d += 2 * dr;
    }
    rr[dc as usize] = r1;
    cc[dc as usize] = c1;
    (rr.into_pyarray(py), cc.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Hough accumulator (src/trail_reject.py's `_detect_line_segments`)
// ---------------------------------------------------------------------------
//
// The numpy implementation materializes both outer products and the resulting
// n_points x n_theta rho-index array.  Accumulating each point directly keeps
// memory bounded by the accumulator while preserving numpy's float32
// arithmetic and ties-to-even rounding.

pub(crate) fn round_ties_even_f32(value: f32) -> i64 {
    let lower = value.floor();
    let fraction = value - lower;
    if fraction < 0.5 {
        lower as i64
    } else if fraction > 0.5 {
        lower as i64 + 1
    } else {
        let lower_i = lower as i64;
        if lower_i % 2 == 0 { lower_i } else { lower_i + 1 }
    }
}

#[pyfunction]
pub(crate) fn hough_accumulator<'py>(
    py: Python<'py>,
    binary: PyReadonlyArray2<'py, bool>,
    cos_t: PyReadonlyArray1<'py, f32>,
    sin_t: PyReadonlyArray1<'py, f32>,
    diag: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let binary = binary.as_array();
    let cos_t = cos_t.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("cos_t must be C-contiguous")
    })?;
    let sin_t = sin_t.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("sin_t must be C-contiguous")
    })?;
    if cos_t.len() != sin_t.len() || cos_t.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cos_t and sin_t must have the same nonzero length",
        ));
    }
    let n_rho = diag.checked_mul(2).and_then(|v| v.checked_add(1)).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("diag is too large")
    })?;
    let points: Vec<(usize, usize)> = binary
        .indexed_iter()
        .filter_map(|((y, x), on)| if *on { Some((y, x)) } else { None })
        .collect();
    let cos_t = cos_t.to_vec();
    let sin_t = sin_t.to_vec();
    let accumulator = py.detach(|| {
        let mut accumulator = vec![0i32; n_rho * cos_t.len()];
        for (y, x) in points {
            let x_f = x as f32;
            let y_f = y as f32;
            for theta in 0..cos_t.len() {
                let rho = x_f * cos_t[theta] + y_f * sin_t[theta];
                let rho_i = round_ties_even_f32(rho) + diag as i64;
                if rho_i >= 0 && rho_i < n_rho as i64 {
                    accumulator[rho_i as usize * cos_t.len() + theta] += 1;
                }
            }
        }
        accumulator
    });
    let arr = numpy::ndarray::Array2::from_shape_vec((n_rho, cos_t.len()), accumulator)
        .expect("shape checked while building accumulator");
    Ok(arr.into_pyarray(py))
}
