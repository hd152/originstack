//! Frame calibration (dark/flat/bias) and per-patch Brenner scores.

use crate::*;

// ---------------------------------------------------------------------------
// Fused Phase-1 kernels: calibration and hot-pixel removal
// ---------------------------------------------------------------------------
//
// The numpy reference makes ~6 full-frame passes (and two temporaries) for
// calibration and ~20 for hot-pixel detection, all of it memory-bandwidth
// bound under 16 workers. These kernels do the same arithmetic in the same
// order per element, so the result is bit-identical (no FMA in Rust unless
// asked for), in one or two passes.

/// Calibrate one frame in place: `x -= bias; x -= dark*s; x += bias*s;
/// x /= flat; clip at 0`, all f32, one pass. Arguments are flat 1-D views (the
/// caller reshapes) so the same kernel serves mosaics and RGB frames. Returns
/// false when any value is non-finite after the flat division (the numpy path
/// reports that as a calibration error).
pub(crate) fn opt_master<'a, 'py>(
    a: &'a Option<PyReadonlyArray1<'py, f32>>,
    n: usize,
) -> PyResult<Option<&'a [f32]>> {
    match a {
        None => Ok(None),
        Some(v) => {
            let s = v.as_slice().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("master must be contiguous")
            })?;
            if s.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
            }
            Ok(Some(s))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (data, bias, dark, dark_scale, flat_norm))]
pub(crate) fn calibrate_frame_inplace<'py>(
    py: Python<'py>,
    mut data: numpy::PyReadwriteArray1<'py, f32>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    dark: Option<PyReadonlyArray1<'py, f32>>,
    dark_scale: f64,
    flat_norm: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<bool> {
    let d = data
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("data must be contiguous"))?;
    let n = d.len();
    let b = opt_master(&bias, n)?;
    let dk = opt_master(&dark, n)?;
    let fl = opt_master(&flat_norm, n)?;
    // numpy: `dark_arr * dark_scale` with a Python float scalar is an f32
    // multiply (NEP 50 weak scalar), so the scale is rounded to f32 first.
    let s = dark_scale as f32;
    const CH: usize = 1 << 16;

    let ok = py.detach(|| {
        d.par_chunks_mut(CH)
            .enumerate()
            .map(|(ci, chunk)| {
                let base = ci * CH;
                let mut ok = true;
                for (k, v) in chunk.iter_mut().enumerate() {
                    let i = base + k;
                    let mut x = *v;
                    if let Some(b) = b {
                        x -= b[i];
                    }
                    if let Some(dk) = dk {
                        x -= dk[i] * s;
                        if let Some(b) = b {
                            x += b[i] * s;
                        }
                    }
                    if let Some(fl) = fl {
                        x /= fl[i];
                    }
                    if !x.is_finite() {
                        ok = false;
                    }
                    *v = if x < 0.0 { 0.0 } else { x };
                }
                ok
            })
            .reduce(|| true, |a, b| a && b)
    });
    Ok(ok)
}

/// Load + calibrate fused: `raw` holds a FITS BITPIX=16 data block's bytes read as
/// native u16 (so each value is byte-swapped here); the physical value is the signed
/// 16-bit integer plus `bzero` (0 or 32768 -- exact in f32, as astropy's uint16 path
/// gives). Then exactly `calibrate_frame_inplace`'s per-pixel steps, written to `out`.
/// Replaces astropy's read + conversion (~30 ms) and one full pass of the calibration.
/// Returns False when a value came out non-finite, like `calibrate_frame_inplace`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn calibrate_frame_from_be16<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray1<'py, u16>,
    bzero: f64,
    mut out: numpy::PyReadwriteArray1<'py, f32>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    dark: Option<PyReadonlyArray1<'py, f32>>,
    dark_scale: f64,
    flat_norm: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<bool> {
    let r = raw.as_slice()?;
    let d = out
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("out must be contiguous"))?;
    let n = d.len();
    if r.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err("raw/out size mismatch"));
    }
    let b = opt_master(&bias, n)?;
    let dk = opt_master(&dark, n)?;
    let fl = opt_master(&flat_norm, n)?;
    let s = dark_scale as f32;
    let z = bzero as i32;
    const CH: usize = 1 << 16;
    let ok = py.detach(|| {
        d.par_chunks_mut(CH)
            .enumerate()
            .map(|(ci, chunk)| {
                let base = ci * CH;
                let mut ok = true;
                for (k, v) in chunk.iter_mut().enumerate() {
                    let i = base + k;
                    let mut x = ((r[i].swap_bytes() as i16) as i32 + z) as f32;
                    if let Some(b) = b {
                        x -= b[i];
                    }
                    if let Some(dk) = dk {
                        x -= dk[i] * s;
                        if let Some(b) = b {
                            x += b[i] * s;
                        }
                    }
                    if let Some(fl) = fl {
                        x /= fl[i];
                    }
                    if !x.is_finite() {
                        ok = false;
                    }
                    *v = if x < 0.0 { 0.0 } else { x };
                }
                ok
            })
            .reduce(|| true, |a, b| a && b)
    });
    Ok(ok)
}

/// `registration.compute_patch_scores`: per patch of an (ny, nx) grid (patch ph x pw,
/// the last row/column of patches ending at ny*ph / nx*pw like the numpy loop), the mean of
/// (lum[y, x+2] - lum[y, x])^2 over the patch's own columns, in f64. Reads the float32
/// luminance directly (the numpy version made a float64 copy of the whole frame first);
/// parallel over patch rows. Values agree with numpy up to summation order (~1e-15).
#[pyfunction]
pub(crate) fn patch_brenner_scores<'py>(
    py: Python<'py>,
    lum: PyReadonlyArray2<'py, f32>,
    ph: usize,
    pw: usize,
    ny: usize,
    nx: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let a = lum.as_array();
    let (h, w) = (a.shape()[0], a.shape()[1]);
    let s = a.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("lum must be contiguous"))?;
    let mut out = vec![0f32; ny * nx];
    py.detach(|| {
        par_rows(&mut out, nx, |iy, row| {
            let y0 = iy * ph;
            let y1 = ((iy + 1) * ph).min(h);
            for ix in 0..nx {
                let x0 = ix * pw;
                let x1 = ((ix + 1) * pw).min(w);
                if (y1 - y0) * (x1 - x0) < 4 || x1 - x0 < 3 {
                    continue;
                }
                let mut acc = 0.0f64;
                for y in y0..y1 {
                    let r = &s[y * w..(y + 1) * w];
                    for x in x0..x1 - 2 {
                        let d = r[x + 2] as f64 - r[x] as f64;
                        acc += d * d;
                    }
                }
                row[ix] = (acc / ((y1 - y0) * (x1 - x0 - 2)) as f64) as f32;
            }
        });
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((ny, nx), out).unwrap().into_pyarray(py))
}
