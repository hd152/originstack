//! Native hot-path kernels for OriginStack.
//!
//! Currently exposes `sigma_clip_combine`, a per-pixel iterative sigma-clip /
//! winsorized combine over an `(N, H, W, C)` stack of aligned frames. It mirrors
//! the numpy reference in `src/stacking.py` (`_sigma_clip_tile`) but runs the
//! per-pixel loop in native code, parallelised across image rows with rayon, so
//! there is no per-tile float32 copy and no repeated whole-stack NaN passes.

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Median of a slice, sorting it in place. Matches numpy: even N averages the
/// two middle values. Returns NaN for an empty slice.
#[inline]
fn median_inplace(v: &mut [f32]) -> f32 {
    let n = v.len();
    if n == 0 {
        return f32::NAN;
    }
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Population mean of a slice.
#[inline]
fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let s: f64 = v.iter().map(|&x| x as f64).sum();
    (s / v.len() as f64) as f32
}

/// Population standard deviation (ddof=0), matching numpy's nanstd default.
#[inline]
fn std_pop(v: &[f32]) -> f32 {
    let n = v.len();
    if n == 0 {
        return f32::NAN;
    }
    let m = mean(v) as f64;
    let var: f64 = v.iter().map(|&x| (x as f64 - m) * (x as f64 - m)).sum::<f64>() / n as f64;
    var.sqrt() as f32
}

/// Sample standard deviation (ddof=1), matching numpy `nanstd(..., ddof=1)`.
#[inline]
fn std_sample(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(v) as f64;
    let var = v.iter().map(|&x| (x as f64 - m) * (x as f64 - m)).sum::<f64>() / (n as f64 - 1.0);
    var.sqrt() as f32
}

/// Center + spread for the active values, with the same zero-spread fallback as
/// the numpy reference (MAD -> std, std -> MAD).
#[inline]
fn center_spread(active: &[f32], scratch: &mut Vec<f32>, use_mad: bool) -> (f32, f32) {
    if use_mad {
        scratch.clear();
        scratch.extend_from_slice(active);
        let med = median_inplace(scratch);
        scratch.clear();
        scratch.extend(active.iter().map(|&x| (x - med).abs()));
        let mut spread = median_inplace(scratch) * 1.4826;
        if spread < 1e-12 {
            spread = std_pop(active); // fallback
        }
        (med, spread)
    } else {
        let ctr = mean(active);
        let mut spread = std_pop(active);
        if spread < 1e-12 {
            scratch.clear();
            scratch.extend_from_slice(active);
            let med = median_inplace(scratch);
            scratch.clear();
            scratch.extend(active.iter().map(|&x| (x - med).abs()));
            spread = median_inplace(scratch) * 1.4826; // fallback
        }
        (ctr, spread)
    }
}

/// numpy-`linear` percentile of an already-sorted slice (method='linear',
/// the numpy default): rank = p/100 * (n-1), linear interpolation.
#[inline]
fn percentile_sorted(sorted: &[f32], p: f64) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (n as f64 - 1.0);
    let k = rank.floor() as usize;
    let frac = rank - k as f64;
    if k + 1 >= n {
        sorted[n - 1]
    } else {
        (sorted[k] as f64 + frac * (sorted[k + 1] as f64 - sorted[k] as f64)) as f32
    }
}

/// Row-parallel driver: fills each pixel's N samples into `vals`, then calls
/// `work` with per-thread `state` built once per row via `init`.
fn row_parallel<S, Init, Work>(
    arr: &numpy::ndarray::ArrayView4<'_, f32>,
    h: usize,
    w: usize,
    c: usize,
    n: usize,
    init: Init,
    work: Work,
) -> Vec<f32>
where
    S: Send,
    Init: Fn() -> S + Sync,
    Work: Fn(&mut S, &[f32]) -> f32 + Sync,
{
    let mut out = vec![0f32; h * w * c];
    out.par_chunks_mut(w * c).enumerate().for_each(|(row, out_row)| {
        let mut state = init();
        let mut vals = vec![0f32; n];
        for col in 0..(w * c) {
            let wj = col / c;
            let cj = col % c;
            for k in 0..n {
                vals[k] = arr[[k, row, wj, cj]];
            }
            out_row[col] = work(&mut state, &vals);
        }
    });
    out
}

/// Combine one pixel's N samples. `vals`/`weights` are length N; NaN samples are
/// treated as already-rejected.
#[allow(clippy::too_many_arguments)]
fn combine_pixel(
    vals: &[f32],
    weights: Option<&[f32]>,
    sigma: f32,
    max_iters: usize,
    winsorize: bool,
    use_mad: bool,
    active: &mut [bool],
    gather: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
) -> f32 {
    let n = vals.len();
    for i in 0..n {
        active[i] = !vals[i].is_nan();
    }

    for _ in 0..max_iters {
        gather.clear();
        for i in 0..n {
            if active[i] {
                gather.push(vals[i]);
            }
        }
        if gather.is_empty() {
            break;
        }
        let (center, spread) = center_spread(gather, scratch, use_mad);
        let thresh = sigma * spread;

        // Proposed new mask.
        let mut changed = false;
        let mut survivors = 0usize;
        // First pass: count survivors under the threshold.
        for i in 0..n {
            if active[i] && (vals[i] - center).abs() <= thresh {
                survivors += 1;
            }
        }
        if survivors == 0 {
            // All would be rejected -> keep the current mask, no change, stop.
            break;
        }
        for i in 0..n {
            if active[i] {
                let keep = (vals[i] - center).abs() <= thresh;
                if !keep {
                    active[i] = false;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    if winsorize {
        // Recompute center/spread on the surviving set, clip ALL samples to the
        // boundary, then (weighted) mean over all clipped samples.
        gather.clear();
        for i in 0..n {
            if active[i] {
                gather.push(vals[i]);
            }
        }
        let (center, mut spread) = if gather.is_empty() {
            (0.0, 1e-12)
        } else {
            center_spread(gather, scratch, use_mad)
        };
        if spread < 1e-12 {
            spread = 1e-12;
        }
        let lo = center - sigma * spread;
        let hi = center + sigma * spread;
        let mut acc = 0f64;
        let mut wsum = 0f64;
        for i in 0..n {
            if vals[i].is_nan() {
                continue;
            }
            let clipped = vals[i].clamp(lo, hi);
            let w = weights.map(|w| w[i]).unwrap_or(1.0) as f64;
            acc += clipped as f64 * w;
            wsum += w;
        }
        if wsum == 0.0 {
            0.0
        } else {
            (acc / wsum) as f32
        }
    } else {
        // (Weighted) mean over surviving samples.
        let mut acc = 0f64;
        let mut wsum = 0f64;
        for i in 0..n {
            if active[i] {
                let w = weights.map(|w| w[i]).unwrap_or(1.0) as f64;
                acc += vals[i] as f64 * w;
                wsum += w;
            }
        }
        if wsum == 0.0 {
            0.0
        } else {
            (acc / wsum) as f32
        }
    }
}

/// Sigma-clip / winsorized combine of an `(N, H, W, C)` float32 stack.
///
/// Returns an `(H, W, C)` float32 array. Parallelised across rows.
#[pyfunction]
#[pyo3(signature = (data, sigma=3.0, max_iters=3, weights=None, winsorize=false, use_mad=true))]
fn sigma_clip_combine<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    sigma: f32,
    max_iters: usize,
    weights: Option<PyReadonlyArray1<'py, f32>>,
    winsorize: bool,
    use_mad: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let shape = arr.shape();
    let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);

    let weights_vec: Option<Vec<f32>> = weights.map(|wa| wa.as_array().to_vec());

    // Flat output buffer, filled row-parallel.
    let mut out = vec![0f32; h * w * c];

    // Release the GIL for the compute; arr is a read-only view over the numpy
    // buffer (kept alive by `data`), safe to share across rayon threads.
    py.allow_threads(|| {
        out.par_chunks_mut(w * c).enumerate().for_each(|(row, out_row)| {
            // Thread-local scratch reused across the row.
            let mut vals = vec![0f32; n];
            let mut active = vec![true; n];
            let mut gather: Vec<f32> = Vec::with_capacity(n);
            let mut scratch: Vec<f32> = Vec::with_capacity(n);
            for col in 0..(w * c) {
                let wj = col / c;
                let cj = col % c;
                for k in 0..n {
                    vals[k] = arr[[k, row, wj, cj]];
                }
                out_row[col] = combine_pixel(
                    &vals,
                    weights_vec.as_deref(),
                    sigma,
                    max_iters,
                    winsorize,
                    use_mad,
                    &mut active,
                    &mut gather,
                    &mut scratch,
                );
            }
        });
    });

    let out_arr = numpy::ndarray::Array3::from_shape_vec((h, w, c), out)
        .expect("shape mismatch building output");
    Ok(out_arr.into_pyarray(py))
}

/// Median combine of an `(N,H,W,C)` float32 stack (even N averages the two
/// middle values, matching `np.median`).
#[pyfunction]
fn median_combine<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
    let out = py.allow_threads(|| {
        row_parallel(&arr, h, w, c, n, || Vec::<f32>::with_capacity(n), |buf, vals| {
            buf.clear();
            buf.extend_from_slice(vals);
            median_inplace(buf)
        })
    });
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out).unwrap().into_pyarray(py))
}

/// Percentile-clip combine: reject samples outside [low, high] percentile at
/// each pixel, then (weighted) mean the survivors. Matches
/// `_percentile_clip_tile` (numpy 'linear' percentile).
#[pyfunction]
#[pyo3(signature = (data, low=20.0, high=80.0, weights=None))]
fn percentile_clip_combine<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    low: f64,
    high: f64,
    weights: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
    let wv: Option<Vec<f32>> = weights.map(|x| x.as_array().to_vec());
    let wref = wv.as_deref();
    let out = py.allow_threads(|| {
        row_parallel(&arr, h, w, c, n, || Vec::<f32>::with_capacity(n), |buf, vals| {
            buf.clear();
            buf.extend_from_slice(vals);
            buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo = percentile_sorted(buf, low);
            let hi = percentile_sorted(buf, high);
            // Survivors: lo <= v <= hi; if none, keep all (matches numpy).
            let mut any = false;
            for &v in vals.iter() {
                if v >= lo && v <= hi {
                    any = true;
                    break;
                }
            }
            let mut acc = 0f64;
            let mut wsum = 0f64;
            for (i, &v) in vals.iter().enumerate() {
                let keep = any && v >= lo && v <= hi;
                if keep || !any {
                    let wt = wref.map(|w| w[i]).unwrap_or(1.0) as f64;
                    acc += v as f64 * wt;
                    wsum += wt;
                }
            }
            if wsum == 0.0 { 0.0 } else { (acc / wsum) as f32 }
        })
    });
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out).unwrap().into_pyarray(py))
}

/// Trimmed-mean combine: sort, drop floor(N*trim_low) low and floor(N*trim_high)
/// high samples, mean the rest. Matches `trimmed_mean_combine`.
#[pyfunction]
#[pyo3(signature = (data, trim_low=0.2, trim_high=0.2))]
fn trimmed_mean_combine<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    trim_low: f64,
    trim_high: f64,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
    let mut n_low = (n as f64 * trim_low).floor().max(0.0) as usize;
    let mut n_high = (n as f64 * trim_high).floor().max(0.0) as usize;
    let mut n_keep = n as isize - n_low as isize - n_high as isize;
    if n_keep < 1 {
        n_keep = 1;
        n_low = 0;
        n_high = 0;
    }
    let (n_low, n_keep) = (n_low, n_keep as usize);
    let _ = n_high;
    let out = py.allow_threads(|| {
        row_parallel(&arr, h, w, c, n, || Vec::<f32>::with_capacity(n), |buf, vals| {
            buf.clear();
            buf.extend_from_slice(vals);
            buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let slice = &buf[n_low..n_low + n_keep];
            mean(slice)
        })
    });
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out).unwrap().into_pyarray(py))
}

/// Generalized ESD (Grubbs) combine. The critical-value table `lut` (shape
/// `(N+1, max_outliers)`, +inf where undefined) is precomputed in Python from
/// the Student-t distribution and indexed `[n_active, iteration]`, so no
/// statistics crate is needed here. Matches `_esd_clip_tile`.
#[pyfunction]
#[pyo3(signature = (data, max_outliers, lut, weights=None))]
fn esd_combine<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    max_outliers: usize,
    lut: numpy::PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
    let mo = max_outliers;
    let lut_flat: Vec<f64> = lut.as_array().iter().copied().collect(); // (N+1)*mo row-major
    let wv: Option<Vec<f32>> = weights.map(|x| x.as_array().to_vec());
    let wref = wv.as_deref();

    let out = py.allow_threads(|| {
        row_parallel(
            &arr,
            h,
            w,
            c,
            n,
            || (vec![true; n], Vec::<f32>::with_capacity(n)),
            |(active, gather), vals| {
                for k in 0..n {
                    active[k] = !vals[k].is_nan();
                }
                for i in 0..mo {
                    gather.clear();
                    for k in 0..n {
                        if active[k] {
                            gather.push(vals[k]);
                        }
                    }
                    let m = gather.len();
                    if m < 3 {
                        break; // lambda is +inf for n_active < 3 -> never rejects
                    }
                    let mn = mean(gather);
                    let mut sd = std_sample(gather);
                    if sd < 1e-12 {
                        sd = 1e-12;
                    }
                    let mut max_dev = -1f32;
                    let mut idx = usize::MAX;
                    for k in 0..n {
                        if active[k] {
                            let dv = (vals[k] - mn).abs() / sd;
                            if dv > max_dev {
                                max_dev = dv;
                                idx = k;
                            }
                        }
                    }
                    let lam = lut_flat[m * mo + i];
                    if (max_dev as f64) > lam && idx != usize::MAX {
                        active[idx] = false; // m >= 3 -> m-1 >= 2 survive
                    }
                }
                let mut acc = 0f64;
                let mut wsum = 0f64;
                for k in 0..n {
                    if active[k] {
                        let wt = wref.map(|w| w[k]).unwrap_or(1.0) as f64;
                        acc += vals[k] as f64 * wt;
                        wsum += wt;
                    }
                }
                if wsum == 0.0 {
                    0.0
                } else {
                    (acc / wsum) as f32
                }
            },
        )
    });
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out).unwrap().into_pyarray(py))
}

/// Lanczos-a windowed-sinc kernel weight for offset `x` (a = support radius).
#[inline]
fn lanczos_w(x: f64, a: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else if x.abs() >= a {
        0.0
    } else {
        let px = std::f64::consts::PI * x;
        (a * px.sin() * (px / a).sin()) / (px * px)
    }
}

/// Affine warp with Lanczos-3 resampling, matching scipy's
/// `affine_transform` sampling convention: `out[oy,ox] = in[M @ (oy,ox) + off]`,
/// out-of-bounds -> `cval`. `mat` is row-major 2x2 `[[m00,m01],[m10,m11]]`
/// mapping (row,col); `off` is `[off_row, off_col]`. All channels in one pass,
/// parallel across output rows.
#[pyfunction]
#[pyo3(signature = (data, mat, off, out_h, out_w, cval=0.0))]
fn warp_affine_lanczos3<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f32>,
    mat: [f64; 4],
    off: [f64; 2],
    out_h: usize,
    out_w: usize,
    cval: f32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (h, w, c) = (s[0], s[1], s[2]);
    let a = 3.0_f64;
    let (m00, m01, m10, m11) = (mat[0], mat[1], mat[2], mat[3]);
    let (o0, o1) = (off[0], off[1]);

    let mut out = vec![0f32; out_h * out_w * c];
    py.allow_threads(|| {
        out.par_chunks_mut(out_w * c).enumerate().for_each(|(oy, out_row)| {
            let mut wy = [0f64; 6];
            let mut wx = [0f64; 6];
            for ox in 0..out_w {
                // Inverse map output -> input (scipy convention).
                let iy = m00 * oy as f64 + m01 * ox as f64 + o0;
                let ix = m10 * oy as f64 + m11 * ox as f64 + o1;
                let fy = iy.floor();
                let fx = ix.floor();
                let ry = iy - fy;
                let rx = ix - fx;
                // 6-tap Lanczos-3 weights, taps at floor-2 .. floor+3.
                let mut sy = 0.0;
                let mut sx = 0.0;
                for t in 0..6 {
                    let dy = (t as f64 - 2.0) - ry;
                    let dx = (t as f64 - 2.0) - rx;
                    wy[t] = lanczos_w(dy, a);
                    wx[t] = lanczos_w(dx, a);
                    sy += wy[t];
                    sx += wx[t];
                }
                if sy != 0.0 {
                    for v in wy.iter_mut() {
                        *v /= sy;
                    }
                }
                if sx != 0.0 {
                    for v in wx.iter_mut() {
                        *v /= sx;
                    }
                }
                let base_y = fy as isize - 2;
                let base_x = fx as isize - 2;
                for ch in 0..c {
                    let mut acc = 0.0f64;
                    let mut any = false;
                    for ty in 0..6 {
                        let yy = base_y + ty as isize;
                        if yy < 0 || yy >= h as isize || wy[ty] == 0.0 {
                            continue;
                        }
                        let mut row_acc = 0.0f64;
                        for tx in 0..6 {
                            let xx = base_x + tx as isize;
                            if xx < 0 || xx >= w as isize || wx[tx] == 0.0 {
                                continue;
                            }
                            row_acc += wx[tx] * arr[[yy as usize, xx as usize, ch]] as f64;
                            any = true;
                        }
                        acc += wy[ty] * row_acc;
                    }
                    out_row[ox * c + ch] = if any { acc as f32 } else { cval };
                }
            }
        });
    });
    Ok(numpy::ndarray::Array3::from_shape_vec((out_h, out_w, c), out).unwrap().into_pyarray(py))
}

/// Perona-Malik anisotropic diffusion, `iterations` Jacobi steps with a
/// periodic (np.roll) boundary. Matches the numpy reference; returns the
/// diffused float64 image (H,W,C) BEFORE the Python-side star-mask blend/clip.
#[pyfunction]
#[pyo3(signature = (data, iterations, kappa, gamma, option))]
fn anisotropic_diffusion<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f32>,
    iterations: usize,
    kappa: f64,
    gamma: f64,
    option: i32,
) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (h, w, c) = (s[0], s[1], s[2]);
    let g = gamma.clamp(1e-6, 0.25);

    // Per-channel contiguous buffers for cache-friendly stencils.
    let mut chans: Vec<Vec<f64>> = (0..c)
        .map(|ch| {
            let mut b = vec![0f64; h * w];
            for y in 0..h {
                for x in 0..w {
                    b[y * w + x] = arr[[y, x, ch]] as f64;
                }
            }
            b
        })
        .collect();

    let cond = |d: f64| -> f64 {
        if option == 1 {
            (-(d / kappa) * (d / kappa)).exp()
        } else {
            1.0 / (1.0 + (d / kappa) * (d / kappa))
        }
    };

    py.allow_threads(|| {
        for buf in chans.iter_mut() {
            let mut next = vec![0f64; h * w];
            for _ in 0..iterations {
                next.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
                    let yn = (y + 1) % h; // roll(-1, axis0): north neighbour
                    let ys = (y + h - 1) % h; // roll(1, axis0): south
                    for x in 0..w {
                        let xe = (x + 1) % w; // roll(-1, axis1): east
                        let xw = (x + w - 1) % w; // roll(1, axis1): west
                        let ctr = buf[y * w + x];
                        let dn = buf[yn * w + x] - ctr;
                        let ds = buf[ys * w + x] - ctr;
                        let de = buf[y * w + xe] - ctr;
                        let dw = buf[y * w + xw] - ctr;
                        out_row[x] = ctr
                            + g * (cond(dn) * dn + cond(ds) * ds + cond(de) * de + cond(dw) * dw);
                    }
                });
                std::mem::swap(buf, &mut next);
            }
        }
    });

    let mut out = vec![0f64; h * w * c];
    for ch in 0..c {
        let buf = &chans[ch];
        for y in 0..h {
            for x in 0..w {
                out[(y * w + x) * c + ch] = buf[y * w + x];
            }
        }
    }
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out).unwrap().into_pyarray(py))
}

#[pymodule]
fn astro_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sigma_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(median_combine, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(trimmed_mean_combine, m)?)?;
    m.add_function(wrap_pyfunction!(esd_combine, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine_lanczos3, m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_diffusion, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
