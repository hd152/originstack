//! CFA / green equalisation, strided clipped medians, pre-gradient removal.

use crate::*;

// ---------------------------------------------------------------------------
// Debayer-adjacent kernels: sigma-clipped medians without numpy's strided copies,
// green G1/G2 equalisation, the Bayer-position grid equalisation
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn median_f32_as_f64(v: &mut [f32]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    let (_, &mut m, _) =
        v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        m as f64
    } else {
        let mut lo = f32::NEG_INFINITY;
        for &x in v[..mid].iter() {
            if x > lo {
                lo = x;
            }
        }
        0.5 * (lo as f64 + m as f64)
    }
}

/// Iterative sigma-clipped median on f32 storage. Identical result to the f64
/// version (`sigma_clipped_median_native`): f32 -> f64 is exact, the order
/// statistics are the same values, and every sum/compare is still done in f64;
/// only the buffers are half the size, which is what matters under memory contention.
/// (A two-pass radix-histogram selection was tried in place of the quickselect: exact
/// too, but no faster -- ~6 ms per iteration either way, of which the two sequential
/// f64 sums are a good part and cannot be reordered without changing the bits.)
pub(crate) fn sigma_clipped_median_f32(orig: &[f32], sigma: f64, iters: usize) -> f64 {
    let mut x: Vec<f32> = orig.to_vec();
    let mut scratch: Vec<f32> = Vec::with_capacity(x.len());
    for _ in 0..iters {
        if x.is_empty() {
            break;
        }
        scratch.clear();
        scratch.extend_from_slice(&x);
        let med = median_f32_as_f64(&mut scratch);
        let n = x.len() as f64;
        let m: f64 = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let std = (x.iter().map(|&v| { let d = v as f64 - m; d * d }).sum::<f64>() / n).sqrt();
        if std < 1e-12 {
            break;
        }
        let thresh = sigma * std;
        x.retain(|&v| (v as f64 - med).abs() < thresh);
    }
    if !x.is_empty() {
        median_f32_as_f64(&mut x)
    } else {
        let mut o = orig.to_vec();
        median_f32_as_f64(&mut o)
    }
}

/// Gather the (py, px) 2x2 sub-plane of channel `ch` from an interleaved
/// `chans`-channel image, row-major (the order numpy's `ravel` of the strided view has).
pub(crate) fn gather_plane(src: &[f32], w: usize, h: usize, chans: usize, ch: usize, py: usize, px: usize) -> Vec<f32> {
    let hh = (h.saturating_sub(py) + 1) / 2;
    let ww = (w.saturating_sub(px) + 1) / 2;
    let mut p = vec![0f32; hh * ww];
    if ww == 0 {
        return p;
    }
    p.par_chunks_mut(ww).enumerate().for_each(|(i, row)| {
        let base = (2 * i + py) * w + px;
        for j in 0..ww {
            row[j] = src[(base + 2 * j) * chans + ch];
        }
    });
    p
}

/// `_sigma_clipped_median` of any (possibly strided) 2-D float32 view, without
/// numpy's copy of the view.
#[pyfunction]
#[pyo3(signature = (data, sigma=3.0, iters=3))]
pub(crate) fn strided_sigma_clipped_median<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    sigma: f64,
    iters: usize,
) -> f64 {
    let v: Vec<f32> = data.as_array().iter().copied().collect();
    py.detach(|| sigma_clipped_median_f32(&v, sigma, iters))
}

/// `green_equalize` on a float32 mosaic in place: scale the G2 sub-plane so its
/// sigma-clipped median matches G1's, when the two are within 20% of each other.
/// Returns whether a correction was applied.
#[pyfunction]
pub(crate) fn green_equalize_inplace<'py>(
    py: Python<'py>,
    mut raw: numpy::PyReadwriteArray2<'py, f32>,
    g1_r: usize,
    g1_c: usize,
    g2_r: usize,
    g2_c: usize,
) -> PyResult<bool> {
    let (h, w) = (raw.as_array().shape()[0], raw.as_array().shape()[1]);
    let d = raw
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("raw must be contiguous"))?;
    Ok(py.detach(|| {
        let (m1, m2) = {
            let dd: &[f32] = &*d;
            let (a, b) = rayon::join(
                || sigma_clipped_median_f32(&gather_plane(dd, w, h, 1, 0, g1_r, g1_c), 3.0, 3),
                || sigma_clipped_median_f32(&gather_plane(dd, w, h, 1, 0, g2_r, g2_c), 3.0, 3),
            );
            (a, b)
        };
        if m2 > 1e-6 && (m1 / m2 - 1.0).abs() < 0.2 {
            let s = (m1 / m2) as f32;
            d.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y % 2 == g2_r % 2 {
                    let mut x = g2_c;
                    while x < w {
                        row[x] *= s;
                        x += 2;
                    }
                }
            });
            true
        } else {
            false
        }
    }))
}

/// `_equalize_bayer_grid` on an interleaved float32 RGB image in place: subtract
/// each of the four Bayer-position green medians' deviation from their mean (floor
/// at 0), when that spread is within 0.01..100 ADU. Returns whether it applied.
#[pyfunction]
pub(crate) fn bayer_grid_equalize_inplace<'py>(
    py: Python<'py>,
    mut rgb: numpy::PyReadwriteArray3<'py, f32>,
) -> PyResult<bool> {
    let s = rgb.as_array().shape().to_vec();
    let (h, w) = (s[0], s[1]);
    if s[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("expected 3 channels"));
    }
    let f = rgb
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("rgb must be contiguous"))?;
    Ok(py.detach(|| {
        let med: Vec<f64> = (0..4usize)
            .into_par_iter()
            .map(|q| {
                let p = gather_plane(&*f, w, h, 3, 1, q >> 1, q & 1);
                sigma_clipped_median_f32(&p, 3.0, 3)
            })
            .collect();
        let overall = (med[0] + med[1] + med[2] + med[3]) / 4.0;
        let spread = (med[0] - overall)
            .abs()
            .max((med[1] - overall).abs())
            .max((med[2] - overall).abs())
            .max((med[3] - overall).abs());
        if spread < 0.01 || spread > 100.0 {
            return false;
        }
        let off: Vec<f32> = med.iter().map(|m| (m - overall) as f32).collect();
        f.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            let py_ = y & 1;
            for x in 0..w {
                let v = row[x * 3 + 1] - off[py_ * 2 + (x & 1)];
                row[x * 3 + 1] = if v < 0.0 { 0.0 } else { v };
            }
        });
        true
    }))
}

/// Per-frame quadratic background removal (`pre_gradient_removal`), fused. Evaluates
/// the fitted surface `c0 + c1 r + c2 c + c3 r^2 + c4 r c + c5 c^2` (r, c = row and
/// column over the frame size) in f64 in numpy's operation order, rounds it to f32,
/// subtracts it from every channel in place with a floor at zero, and returns the
/// new luminance -- one pass instead of two full-size f64 meshgrids, ~8 f64
/// temporaries and three luma builds. Bit-identical to the numpy sequence.
#[pyfunction]
pub(crate) fn pre_gradient_apply<'py>(
    py: Python<'py>,
    mut rgb: numpy::PyReadwriteArray3<'py, f32>,
    coeffs: Vec<f64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if coeffs.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err("expected 6 coefficients"));
    }
    let s = rgb.as_array().shape().to_vec();
    let (h, w, c) = (s[0], s[1], s[2]);
    if c != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("expected 3 channels"));
    }
    let f = rgb
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("rgb must be contiguous"))?;
    let (kr, kg, kb) = (0.299f64 as f32, 0.587f64 as f32, 0.114f64 as f32);
    let (c0, c1, c2, c3, c4, c5) = (coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]);
    let cols: Vec<f64> = (0..w).map(|x| x as f64 / w as f64).collect();
    let c2cf: Vec<f64> = cols.iter().map(|&cf| c2 * cf).collect();
    let c5cf2: Vec<f64> = cols.iter().map(|&cf| (c5 * cf) * cf).collect();

    let mut lum = vec![0f32; h * w];
    py.detach(|| {
        f.par_chunks_mut(w * 3).zip(lum.par_chunks_mut(w)).enumerate().for_each(|(y, (row, lrow))| {
            let rf = y as f64 / h as f64;
            let t1 = c0 + c1 * rf;
            let t3 = (c3 * rf) * rf;
            let c4rf = c4 * rf;
            for x in 0..w {
                let bg = ((((t1 + c2cf[x]) + t3) + c4rf * cols[x]) + c5cf2[x]) as f32;
                let b = x * 3;
                for k in 0..3 {
                    let v = row[b + k] - bg;
                    row[b + k] = if v < 0.0 { 0.0 } else { v };
                }
                lrow[x] = kr * row[b] + kg * row[b + 1] + kb * row[b + 2];
            }
        });
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), lum).unwrap().into_pyarray(py))
}
