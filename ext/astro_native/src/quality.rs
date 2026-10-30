//! Per-frame noise and statistics validation.

use crate::*;

/// Per-channel sky and noise of one aligned (H, W, C) frame for the proper coadd,
/// bit-identical to `proper_coadd._sky` / `_noise`:
///   sky   = median of the finite samples of ch[::4, ::4]
///   noise = f64(1.4826f32 * MAD(d)) / sqrt(2), d = finite (row[x+4] - row[x]) over
///           rows ch[::4]; NaN when fewer than 1000 differences
/// Medians are exact order statistics (even count: (a + b) / 2 in f32, as numpy).
/// Gathers its own strided samples: numpy built ~8 temporaries per channel
/// (~40 ms of the coadd's ~108 ms per frame).
#[pyfunction]
pub(crate) fn frame_sky_noise<'py>(py: Python<'py>, frame: PyReadonlyArray3<'py, f32>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let fa = frame.as_array();
    let (h, w, c) = (fa.shape()[0], fa.shape()[1], fa.shape()[2]);
    let fr = fa.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("frame must be contiguous"))?;
    let res: Vec<(f64, f64)> = py.detach(|| {
        (0..c).into_par_iter().map(|ch| {
            let mut s: Vec<f32> = Vec::with_capacity(h.div_ceil(4) * w.div_ceil(4));
            for y in (0..h).step_by(4) {
                let row = &fr[y * w * c..(y + 1) * w * c];
                for x in (0..w).step_by(4) {
                    let v = row[x * c + ch];
                    if v.is_finite() {
                        s.push(v);
                    }
                }
            }
            let sky = median_inplace(&mut s) as f64;
            let mut d: Vec<f32> = Vec::with_capacity(h.div_ceil(4) * w.saturating_sub(4));
            if w > 4 {
                for y in (0..h).step_by(4) {
                    let row = &fr[y * w * c..(y + 1) * w * c];
                    for x in 0..w - 4 {
                        let v = row[(x + 4) * c + ch] - row[x * c + ch];
                        if v.is_finite() {
                            d.push(v);
                        }
                    }
                }
            }
            let noise = if d.len() < 1000 {
                f64::NAN
            } else {
                let m = median_inplace(&mut d);
                for v in d.iter_mut() {
                    *v = (*v - m).abs();
                }
                let mad = median_inplace(&mut d);
                (1.4826f32 * mad) as f64 / 2f64.sqrt()
            };
            (sky, noise)
        }).collect()
    });
    Ok((res.iter().map(|r| r.0).collect(), res.iter().map(|r| r.1).collect()))
}

/// `quality.validate_image_data`'s full-frame passes in two parallel sweeps instead of
/// four numpy passes with full-size temporaries: (all finite, max, zero count,
/// saturated count) with saturated = values >= `f32(max * 0.999)` (numpy compares the
/// float32 frame with that Python float rounded to float32). The max and the counts
/// are exact; when a value is not finite, the counts are not computed (the caller
/// reports the NaN/Inf counts itself).
#[pyfunction]
pub(crate) fn validate_frame_stats<'py>(py: Python<'py>, img: PyReadonlyArray2<'py, f32>) -> PyResult<(bool, f32, u64, u64)> {
    let a = img.as_array();
    let w = a.shape()[1].max(1);
    let owned: Vec<f32>;
    let s: &[f32] = match a.as_slice() {
        Some(x) => x,
        None => {
            owned = a.iter().copied().collect();
            &owned
        }
    };
    Ok(py.detach(|| {
        let (finite, mx, zeros) = s
            .par_chunks(w)
            .map(|row| {
                let mut fin = true;
                let mut m = f32::NEG_INFINITY;
                let mut z = 0u64;
                for &v in row {
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
                (fin, m, z)
            })
            .reduce(|| (true, f32::NEG_INFINITY, 0u64), |x, y| (x.0 && y.0, if x.1 > y.1 { x.1 } else { y.1 }, x.2 + y.2));
        if !finite {
            return (false, mx, zeros, 0u64);
        }
        let thr = (mx as f64 * 0.999) as f32;
        let sat: u64 = s.par_chunks(w).map(|row| row.iter().filter(|&&v| v >= thr).count() as u64).sum();
        (true, mx, zeros, sat)
    }))
}
