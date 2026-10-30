//! Luminance and white-balance kernels.

use crate::*;

/// Luminance `0.299 r + 0.587 g + 0.114 b`, f32 in numpy's operation order
/// (the weak Python scalars round to f32 first), one parallel pass instead of
/// three temporaries.
#[pyfunction]
pub(crate) fn luminance_native<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let arr = rgb.as_array();
    let (h, w) = hot_rgb_check(&arr)?;
    let f: &[f32] = arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("rgb must be contiguous"))?;
    let (kr, kg, kb) = (0.299f64 as f32, 0.587f64 as f32, 0.114f64 as f32);
    let mut lum = vec![0f32; h * w];
    py.detach(|| {
        lum.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let src = &f[y * w * 3..(y + 1) * w * 3];
            for x in 0..w {
                row[x] = kr * src[x * 3] + kg * src[x * 3 + 1] + kb * src[x * 3 + 2];
            }
        });
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), lum).unwrap().into_pyarray(py))
}

// ---------------------------------------------------------------------------
// White balance: per-channel gain + near-clipped-highlight neutralisation
// (src/debayer.py `_desaturate_near_clipped_highlights`)
// ---------------------------------------------------------------------------
//
// The numpy path is: scaled = img * gain (or img / gain for white-patch);
// pixel_peak = img.max(-1); ceiling = img.max(); sat = clip((pixel_peak /
// (ceiling + 1e-12) - 0.8) / 0.2, 0, 1); out = scaled * (1 - sat) +
// pixel_peak * sat; clip(out, 0). Written as array expressions that is ~8 full-size
// temporaries and as many passes over 75 MB (538 ms per 2048x3056 frame on one
// thread, against ~35 ms for the arithmetic itself -- and under 16 workers the
// passes fight for memory bandwidth, so it was 1.5 s/frame in a real run).
//
// Here: one parallel max pass for the ceiling, one parallel pass that does
// everything per pixel. The arithmetic is the same f32 operations in the same
// order with no fused multiply-add (Rust does not contract a*b+c on its own), so
// the result is bit-identical to the numpy path -- the ceiling is an exact max, and
// the per-channel gains stay computed in numpy so its reduction order is kept.

/// Shared body: gain (multiply, or divide for white-patch) + highlight
/// neutralisation over a contiguous (h, w, 3) f32 buffer.
pub(crate) fn white_balance_body(
    py: Python<'_>,
    flat: &[f32],
    h: usize,
    w: usize,
    f: [f32; 3],
    divide: bool,
) -> Vec<f32> {
    let (f0, f1, f2) = (f[0], f[1], f[2]);
    let mut out = vec![0f32; h * w * 3];
    py.detach(|| {
        // exact max (order-independent), NaN propagating like numpy's max
        let ceiling: f32 = flat
            .par_chunks(w * 3)
            .map(|row| {
                let mut m = f32::NEG_INFINITY;
                for &v in row {
                    if v.is_nan() {
                        return f32::NAN;
                    }
                    if v > m {
                        m = v;
                    }
                }
                m
            })
            .reduce(
                || f32::NEG_INFINITY,
                |a, b| {
                    if a.is_nan() || b.is_nan() {
                        f32::NAN
                    } else if a > b {
                        a
                    } else {
                        b
                    }
                },
            );
        let denom = ceiling + 1e-12f32;
        out.par_chunks_mut(w * 3)
            .zip(flat.par_chunks(w * 3))
            .for_each(|(orow, irow)| {
                for x in 0..w {
                    let r = irow[x * 3];
                    let g = irow[x * 3 + 1];
                    let b = irow[x * 3 + 2];
                    // numpy max(axis=-1) propagates NaN
                    let peak = if r.is_nan() || g.is_nan() || b.is_nan() {
                        f32::NAN
                    } else {
                        let mut p = r;
                        if g > p { p = g; }
                        if b > p { p = b; }
                        p
                    };
                    let t = (peak / denom - 0.8f32) / 0.2f32;
                    let sat = if t < 0.0 { 0.0 } else if t > 1.0 { 1.0 } else { t };
                    let one_minus = 1.0f32 - sat;
                    let (s0, s1, s2) = if divide {
                        (r / f0, g / f1, b / f2)
                    } else {
                        (r * f0, g * f1, b * f2)
                    };
                    let o0 = s0 * one_minus + peak * sat;
                    let o1 = s1 * one_minus + peak * sat;
                    let o2 = s2 * one_minus + peak * sat;
                    orow[x * 3] = if o0 < 0.0 { 0.0 } else { o0 };
                    orow[x * 3 + 1] = if o1 < 0.0 { 0.0 } else { o1 };
                    orow[x * 3 + 2] = if o2 < 0.0 { 0.0 } else { o2 };
                }
            });
    });
    out
}

/// White balance with caller-supplied per-channel factors (white-patch).
#[pyfunction]
pub(crate) fn white_balance_apply<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, f32>,
    factors: PyReadonlyArray1<'py, f32>,
    divide: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let shape = img.as_array().shape().to_vec();
    if shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("img must have 3 channels"));
    }
    let f = factors.as_slice()?;
    if f.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("factors must have 3 entries"));
    }
    let flat = img.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("img must be C-contiguous")
    })?;
    let out = white_balance_body(py, flat, shape[0], shape[1], [f[0], f[1], f[2]], divide);
    let arr3 = numpy::ndarray::Array3::from_shape_vec((shape[0], shape[1], 3), out)
        .expect("shape mismatch building white_balance_apply output");
    Ok(arr3.into_pyarray(py))
}

/// Gray-world white balance: the gains come from the per-channel means.
///
/// The means are accumulated in float64 (sequential, row-major) and rounded to
/// float32, matching ``img.mean(axis=(0, 1), dtype=float64).astype(float32)``.
/// A float32 running sum over millions of pixels drifts by up to ~1.5% on real
/// frames, which skewed the gray-world gains. The gain
/// ``mean.mean() / (mean + 1e-12)`` then follows in float32 as numpy computes it.
#[pyfunction]
pub(crate) fn white_balance_grayworld<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let shape = img.as_array().shape().to_vec();
    if shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("img must have 3 channels"));
    }
    let flat = img.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("img must be C-contiguous")
    })?;
    let (h, w) = (shape[0], shape[1]);
    let mut acc = [0f64; 3];
    for px in flat.chunks_exact(3) {
        acc[0] += px[0] as f64;
        acc[1] += px[1] as f64;
        acc[2] += px[2] as f64;
    }
    let n = (h * w) as f64;
    let mean = [(acc[0] / n) as f32, (acc[1] / n) as f32, (acc[2] / n) as f32];
    // numpy: mean.mean() on a 3-element float32 array is (m0 + m1) + m2, then / 3
    let mm = ((mean[0] + mean[1]) + mean[2]) / 3.0f32;
    let scale = [
        mm / (mean[0] + 1e-12f32),
        mm / (mean[1] + 1e-12f32),
        mm / (mean[2] + 1e-12f32),
    ];
    let out = white_balance_body(py, flat, h, w, scale, false);
    let arr3 = numpy::ndarray::Array3::from_shape_vec((h, w, 3), out)
        .expect("shape mismatch building white_balance_grayworld output");
    Ok(arr3.into_pyarray(py))
}

/// Exact max of a contiguous f32 buffer, NaN-propagating like numpy's `max`.
pub(crate) fn wb_ceiling(flat: &[f32], w: usize) -> f32 {
    flat.par_chunks(w * 3)
        .map(|row| {
            let mut m = f32::NEG_INFINITY;
            for &v in row {
                if v.is_nan() {
                    return f32::NAN;
                }
                if v > m {
                    m = v;
                }
            }
            m
        })
        .reduce(
            || f32::NEG_INFINITY,
            |a, b| {
                if a.is_nan() || b.is_nan() {
                    f32::NAN
                } else if a > b {
                    a
                } else {
                    b
                }
            },
        )
}

/// Gray-world gains and the frame ceiling from ONE sequential pass: the
/// per-channel float64 means (row-major, as numpy's `mean(dtype=float64)`)
/// and the exact max (order-independent, so folding it into this pass changes
/// nothing). Returns (scale, ceiling).
pub(crate) fn grayworld_scale_and_ceiling(flat: &[f32], h: usize, w: usize) -> ([f32; 3], f32) {
    let mut acc = [0f64; 3];
    let mut m = f32::NEG_INFINITY;
    let mut nan = false;
    for px in flat.chunks_exact(3) {
        acc[0] += px[0] as f64;
        acc[1] += px[1] as f64;
        acc[2] += px[2] as f64;
        for &v in px {
            if v > m {
                m = v;
            }
            nan |= v.is_nan();
        }
    }
    let n = (h * w) as f64;
    let mean = [(acc[0] / n) as f32, (acc[1] / n) as f32, (acc[2] / n) as f32];
    // numpy: mean.mean() on a 3-element float32 array is (m0 + m1) + m2, then / 3
    let mm = ((mean[0] + mean[1]) + mean[2]) / 3.0f32;
    let scale = [
        mm / (mean[0] + 1e-12f32),
        mm / (mean[1] + 1e-12f32),
        mm / (mean[2] + 1e-12f32),
    ];
    (scale, if nan { f32::NAN } else { m })
}

/// In-place twin of `white_balance_body`: same per-pixel f32 operations in the same
/// order, but each output pixel overwrites its input pixel, so no 75 MB output
/// array is allocated, first-touched and written. `ceiling` skips the max pass
/// when the caller already has it; `lum` (h*w) also receives the luminance of
/// each written pixel (`0.299 r + 0.587 g + 0.114 b`, f32 as `luminance_native`),
/// saving the separate read of the whole image that a recompute costs.
/// One row of `white_balance_body_inplace`: gains, highlight desaturation toward the
/// pixel's peak above 0.8 of `denom`, clip at 0, and (`lrow`) the luminance of the
/// written pixel. Shared by every in-place white-balance kernel so the f32 operations
/// exist once.
#[inline(always)]
pub(crate) fn wb_row(row: &mut [f32], mut lrow: Option<&mut [f32]>, w: usize, f: [f32; 3], divide: bool, denom: f32) {
    let (f0, f1, f2) = (f[0], f[1], f[2]);
    let (kr, kg, kb) = (0.299f64 as f32, 0.587f64 as f32, 0.114f64 as f32);
    for x in 0..w {
        let r = row[x * 3];
        let g = row[x * 3 + 1];
        let b = row[x * 3 + 2];
        let peak = if r.is_nan() || g.is_nan() || b.is_nan() {
            f32::NAN
        } else {
            let mut p = r;
            if g > p { p = g; }
            if b > p { p = b; }
            p
        };
        let t = (peak / denom - 0.8f32) / 0.2f32;
        let sat = if t < 0.0 { 0.0 } else if t > 1.0 { 1.0 } else { t };
        let one_minus = 1.0f32 - sat;
        let (s0, s1, s2) = if divide {
            (r / f0, g / f1, b / f2)
        } else {
            (r * f0, g * f1, b * f2)
        };
        let o0 = s0 * one_minus + peak * sat;
        let o1 = s1 * one_minus + peak * sat;
        let o2 = s2 * one_minus + peak * sat;
        let o0 = if o0 < 0.0 { 0.0 } else { o0 };
        let o1 = if o1 < 0.0 { 0.0 } else { o1 };
        let o2 = if o2 < 0.0 { 0.0 } else { o2 };
        row[x * 3] = o0;
        row[x * 3 + 1] = o1;
        row[x * 3 + 2] = o2;
        if let Some(l) = lrow.as_deref_mut() {
            l[x] = kr * o0 + kg * o1 + kb * o2;
        }
    }
}

pub(crate) fn white_balance_body_inplace(
    py: Python<'_>,
    flat: &mut [f32],
    w: usize,
    f: [f32; 3],
    divide: bool,
    ceiling: Option<f32>,
    lum: Option<&mut [f32]>,
) {
    py.detach(|| {
        let ceiling = match ceiling {
            Some(c) => c,
            None => wb_ceiling(flat, w),
        };
        let denom = ceiling + 1e-12f32;
        let row_fn = |row: &mut [f32], lrow: Option<&mut [f32]>| wb_row(row, lrow, w, f, divide, denom);
        match lum {
            Some(l) => flat
                .par_chunks_mut(w * 3)
                .zip(l.par_chunks_mut(w))
                .for_each(|(row, lrow)| row_fn(row, Some(lrow))),
            None => flat.par_chunks_mut(w * 3).for_each(|row| row_fn(row, None)),
        }
    });
}

/// `white_balance_grayworld` writing into `img` itself (same float64-accumulated means).
#[pyfunction]
pub(crate) fn white_balance_grayworld_inplace<'py>(
    py: Python<'py>,
    mut img: numpy::PyReadwriteArray3<'py, f32>,
) -> PyResult<()> {
    let shape = img.as_array().shape().to_vec();
    if shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("img must have 3 channels"));
    }
    let flat = img
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("img must be C-contiguous"))?;
    let (h, w) = (shape[0], shape[1]);
    let (scale, ceiling) = grayworld_scale_and_ceiling(flat, h, w);
    white_balance_body_inplace(py, flat, w, scale, false, Some(ceiling), None);
    Ok(())
}

/// `white_balance_grayworld_inplace` that also returns the luminance of the
/// balanced image -- what `luminance_native` would compute from it afterwards,
/// bit for bit, without reading the 75 MB image a second time. Two passes over
/// the image in all (statistics, then gain + highlight + luminance) where the
/// separate steps made four.
#[pyfunction]
pub(crate) fn white_balance_grayworld_lum_inplace<'py>(
    py: Python<'py>,
    mut img: numpy::PyReadwriteArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = img.as_array().shape().to_vec();
    if shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("img must have 3 channels"));
    }
    let flat = img
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("img must be C-contiguous"))?;
    let (h, w) = (shape[0], shape[1]);
    let (scale, ceiling) = grayworld_scale_and_ceiling(flat, h, w);
    let mut lum = vec![0f32; h * w];
    white_balance_body_inplace(py, flat, w, scale, false, Some(ceiling), Some(&mut lum));
    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), lum).unwrap().into_pyarray(py))
}

/// `white_balance_grayworld_lum_inplace` writing the luminance into `lum` (h, w) --
/// e.g. the frame store's luminance slot, so no fresh 25 MB array is allocated and
/// then copied there -- and, from each luminance row while it is in cache, what
/// `quality.validate_image_data` reads from the luminance:
/// * `validate_frame_stats(lum)`'s tuple (all finite, max, zero count, saturated
///   count), exactly: the max and counts are order-independent, and the saturated
///   count rescans only the rows whose own max reaches `f32(max * 0.999)`;
/// * with `sample` (ceil(h/step), ceil(w/step)), `lum[::step, ::step]`.
/// The image and luminance are bit-identical to `white_balance_grayworld_lum_inplace`.
#[pyfunction]
#[pyo3(signature = (img, lum, sample=None, step=3))]
pub(crate) fn white_balance_grayworld_lum_into<'py>(
    py: Python<'py>,
    mut img: numpy::PyReadwriteArray3<'py, f32>,
    mut lum: numpy::PyReadwriteArray2<'py, f32>,
    sample: Option<numpy::PyReadwriteArray2<'py, f32>>,
    step: usize,
) -> PyResult<(bool, f32, u64, u64)> {
    let shape = img.as_array().shape().to_vec();
    if shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("img must have 3 channels"));
    }
    let (h, w) = (shape[0], shape[1]);
    if lum.as_array().shape() != [h, w] || step == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("lum must be (h, w)"));
    }
    let (sh, sw) = (h.div_ceil(step), w.div_ceil(step));
    let mut sample = sample;
    if let Some(sm) = sample.as_ref() {
        if sm.as_array().shape() != [sh, sw] {
            return Err(pyo3::exceptions::PyValueError::new_err("sample must be (ceil(h/step), ceil(w/step))"));
        }
    }
    let flat = img
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("img must be C-contiguous"))?;
    let ls = lum
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("lum must be C-contiguous"))?;
    let mut sm_slice: Option<&mut [f32]> = match sample.as_mut() {
        Some(sm) => Some(sm.as_slice_mut().map_err(|_| pyo3::exceptions::PyValueError::new_err("sample must be C-contiguous"))?),
        None => None,
    };
    let (scale, ceiling) = grayworld_scale_and_ceiling(flat, h, w);
    let denom = ceiling + 1e-12f32;
    let mut row_max = vec![0f32; h];
    let mut dummy = vec![0f32; h.div_ceil(step)]; // sample rows when no sample is wanted
    Ok(py.detach(|| {
        // one task per group of `step` rows: balance + luminance, the rows' stats, and
        // the group's sample row (its first row, every step-th column)
        let has_sample = sm_slice.is_some();
        let srows: Vec<&mut [f32]> = match sm_slice.as_deref_mut() {
            Some(sv) => sv.chunks_mut(sw).collect(),
            None => dummy.chunks_mut(1).collect(),
        };
        let (finite, zeros) = flat
            .par_chunks_mut(w * 3 * step)
            .zip(ls.par_chunks_mut(w * step))
            .zip(row_max.par_chunks_mut(step))
            .zip(srows.into_par_iter())
            .map(|(((rows, lrows), maxes), srow)| {
                let mut fin = true;
                let mut z = 0u64;
                for (k, (row, lrow)) in rows.chunks_mut(w * 3).zip(lrows.chunks_mut(w)).enumerate() {
                    wb_row(row, Some(&mut *lrow), w, scale, false, denom);
                    let mut m = f32::NEG_INFINITY;
                    for &v in lrow.iter() {
                        if !v.is_finite() {
                            fin = false;
                        }
                        if v > m {
                            m = v;
                        }
                        if v == 0.0 {
                            z += 1;
                        }
                    }
                    maxes[k] = m;
                    if k == 0 && has_sample {
                        for (j, v) in lrow.iter().step_by(step).enumerate() {
                            srow[j] = *v;
                        }
                    }
                }
                (fin, z)
            })
            .reduce(|| (true, 0u64), |a, b| (a.0 && b.0, a.1 + b.1));
        let mut mx = f32::NEG_INFINITY;
        for &m in row_max.iter() {
            if m > mx {
                mx = m;
            }
        }
        if !finite {
            return (false, mx, zeros, 0u64);
        }
        let thr = (mx as f64 * 0.999) as f32;
        let sat: u64 = ls
            .par_chunks(w)
            .zip(row_max.par_iter())
            .filter(|(_, &m)| m >= thr)
            .map(|(row, _)| row.iter().filter(|&&v| v >= thr).count() as u64)
            .sum();
        (true, mx, zeros, sat)
    }))
}
