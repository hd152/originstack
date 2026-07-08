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

/// Median of a slice via quickselect (O(n), vs O(n log n) full sort). Exact
/// order statistics, so values match the numpy/sort-based reference bit-for-bit:
/// odd N -> k-th element; even N -> mean of the two middle order statistics
/// (the second is the max of the left partition after select_nth).
#[inline]
fn median_inplace(v: &mut [f32]) -> f32 {
    let n = v.len();
    if n == 0 {
        return f32::NAN;
    }
    let mid = n / 2;
    let (_, &mut m, _) =
        v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        m
    } else {
        // (mid-1)-th order statistic = max of the left partition, using the
        // same comparator ordering as the select (NaN compares Equal).
        let mut lo = f32::NEG_INFINITY;
        for &x in v[..mid].iter() {
            if x.partial_cmp(&lo) == Some(std::cmp::Ordering::Greater) {
                lo = x;
            }
        }
        0.5 * (lo + m)
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

/// Pixels per gather-transpose tile: sized so a tile (`tile * n` floats) stays
/// L2-resident (~128 KB ceiling), with sane bounds.
#[inline]
fn gather_tile(n: usize) -> usize {
    (32768 / n.max(1)).clamp(16, 256)
}

/// Row-parallel driver with a blocked gather-transpose.
///
/// The naive per-pixel gather reads each pixel's N samples with a stride of a
/// whole frame (H*W*C floats — tens of MB): N concurrent read streams, which
/// defeats the hardware prefetcher and thrashes the TLB. Instead, per tile of
/// `T` pixels we copy each frame's contiguous row segment (sequential, one
/// stream at a time) into an L2-resident pixel-major block, then hand `work`
/// contiguous `&block[p*n..(p+1)*n]` slices.
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
    let row_len = w * c;
    let frame_len = h * row_len;
    let tile = gather_tile(n);
    let data: Option<&[f32]> = arr.as_slice();

    let mut out = vec![0f32; h * row_len];
    out.par_chunks_mut(row_len).enumerate().for_each(|(row, out_row)| {
        let mut state = init();
        let mut block = vec![0f32; tile * n];
        match data {
            Some(flat) => {
                let row_base = row * row_len;
                let mut start = 0usize;
                while start < row_len {
                    let t = tile.min(row_len - start);
                    // Gather-transpose: sequential read per frame, L2 write.
                    for k in 0..n {
                        let src = &flat[k * frame_len + row_base + start..][..t];
                        for (p, &v) in src.iter().enumerate() {
                            block[p * n + k] = v;
                        }
                    }
                    for p in 0..t {
                        out_row[start + p] = work(&mut state, &block[p * n..(p + 1) * n]);
                    }
                    start += t;
                }
            }
            None => {
                // Non-contiguous fallback: original indexed gather.
                let vals = &mut block[..n];
                for col in 0..row_len {
                    let wj = col / c;
                    let cj = col % c;
                    for k in 0..n {
                        vals[k] = arr[[k, row, wj, cj]];
                    }
                    out_row[col] = work(&mut state, &vals[..n]);
                }
            }
        }
    });
    out
}

/// Fill `active[i]` = survives sigma-clip (same iteration as the numpy
/// `_sigma_clip_tile` per-pixel logic). NaN samples start rejected.
fn sigma_clip_mask(
    vals: &[f32],
    sigma: f32,
    max_iters: usize,
    use_mad: bool,
    active: &mut [bool],
    gather: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
) {
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
        let mut survivors = 0usize;
        for i in 0..n {
            if active[i] && (vals[i] - center).abs() <= thresh {
                survivors += 1;
            }
        }
        if survivors == 0 {
            break;
        }
        let mut changed = false;
        for i in 0..n {
            if active[i] && (vals[i] - center).abs() > thresh {
                active[i] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
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
    let wref = weights_vec.as_deref();

    // Release the GIL for the compute; arr is a read-only view over the numpy
    // buffer (kept alive by `data`), safe to share across rayon threads.
    let out = py.allow_threads(|| {
        row_parallel(
            &arr,
            h,
            w,
            c,
            n,
            || {
                (
                    vec![true; n],
                    Vec::<f32>::with_capacity(n),
                    Vec::<f32>::with_capacity(n),
                )
            },
            |(active, gather, scratch), vals| {
                combine_pixel(
                    vals, wref, sigma, max_iters, winsorize, use_mad, active, gather, scratch,
                )
            },
        )
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

/// Compute the 6 normalised Lanczos-3 tap weights for fractional offset `r`
/// (taps at floor-2 .. floor+3). Same arithmetic as the original inline loop.
#[inline]
fn lanczos6_weights(r: f64, out: &mut [f64; 6]) {
    let mut s = 0.0;
    for (t, o) in out.iter_mut().enumerate() {
        *o = lanczos_w((t as f64 - 2.0) - r, 3.0);
        s += *o;
    }
    if s != 0.0 {
        for v in out.iter_mut() {
            *v /= s;
        }
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
    let (m00, m01, m10, m11) = (mat[0], mat[1], mat[2], mat[3]);
    let (o0, o1) = (off[0], off[1]);
    // Pure translation: iy depends only on oy and ix only on ox, so the Lanczos
    // weights are separable — one wx table per image, one wy per row. This is
    // the common case (rotation off) and removes ALL sin() calls plus the
    // weight normalisation from the per-pixel loop.
    let is_shift = m00 == 1.0 && m01 == 0.0 && m10 == 0.0 && m11 == 1.0;
    let flat: Option<&[f32]> = arr.as_slice();

    let col_tab: Option<(Vec<[f64; 6]>, Vec<isize>)> = if is_shift && flat.is_some() {
        let mut wxs = vec![[0f64; 6]; out_w];
        let mut bxs = vec![0isize; out_w];
        for ox in 0..out_w {
            let ix = ox as f64 + o1;
            let fx = ix.floor();
            lanczos6_weights(ix - fx, &mut wxs[ox]);
            bxs[ox] = fx as isize - 2;
        }
        Some((wxs, bxs))
    } else {
        None
    };

    let mut out = vec![0f32; out_h * out_w * c];
    py.allow_threads(|| {
        out.par_chunks_mut(out_w * c).enumerate().for_each(|(oy, out_row)| {
            let mut wy = [0f64; 6];
            let mut wx = [0f64; 6];
            match (flat, &col_tab) {
                // ---- fast path: contiguous input ----
                (Some(img), tab) => {
                    let row_stride = w * c;
                    for ox in 0..out_w {
                        let (wyv, wxv, base_y, base_x): (&[f64; 6], &[f64; 6], isize, isize) =
                            if let Some((wxs, bxs)) = tab {
                                if ox == 0 {
                                    let iy = oy as f64 + o0;
                                    let fy = iy.floor();
                                    lanczos6_weights(iy - fy, &mut wy);
                                }
                                let iy = oy as f64 + o0;
                                (&wy, &wxs[ox], iy.floor() as isize - 2, bxs[ox])
                            } else {
                                let iy = m00 * oy as f64 + m01 * ox as f64 + o0;
                                let ix = m10 * oy as f64 + m11 * ox as f64 + o1;
                                let fy = iy.floor();
                                let fx = ix.floor();
                                lanczos6_weights(iy - fy, &mut wy);
                                lanczos6_weights(ix - fx, &mut wx);
                                (&wy, &wx, fy as isize - 2, fx as isize - 2)
                            };
                        let interior = base_y >= 0
                            && base_y + 6 <= h as isize
                            && base_x >= 0
                            && base_x + 6 <= w as isize;
                        if interior {
                            // No bounds checks, no zero-weight branches: weights
                            // sum to 1, zero taps contribute exactly 0.0.
                            let by = base_y as usize;
                            let bx = base_x as usize;
                            for ch in 0..c {
                                let mut acc = 0.0f64;
                                for ty in 0..6 {
                                    let base = (by + ty) * row_stride + bx * c + ch;
                                    let mut ra = 0.0f64;
                                    for tx in 0..6 {
                                        ra += wxv[tx] * img[base + tx * c] as f64;
                                    }
                                    acc += wyv[ty] * ra;
                                }
                                out_row[ox * c + ch] = acc as f32;
                            }
                        } else {
                            for ch in 0..c {
                                let mut acc = 0.0f64;
                                let mut any = false;
                                for ty in 0..6 {
                                    let yy = base_y + ty as isize;
                                    if yy < 0 || yy >= h as isize || wyv[ty] == 0.0 {
                                        continue;
                                    }
                                    let mut ra = 0.0f64;
                                    for tx in 0..6 {
                                        let xx = base_x + tx as isize;
                                        if xx < 0 || xx >= w as isize || wxv[tx] == 0.0 {
                                            continue;
                                        }
                                        ra += wxv[tx]
                                            * img[yy as usize * row_stride + xx as usize * c + ch]
                                                as f64;
                                        any = true;
                                    }
                                    acc += wyv[ty] * ra;
                                }
                                out_row[ox * c + ch] = if any { acc as f32 } else { cval };
                            }
                        }
                    }
                }
                // ---- non-contiguous fallback: original indexed loop ----
                (None, _) => {
                    for ox in 0..out_w {
                        let iy = m00 * oy as f64 + m01 * ox as f64 + o0;
                        let ix = m10 * oy as f64 + m11 * ox as f64 + o1;
                        let fy = iy.floor();
                        let fx = ix.floor();
                        lanczos6_weights(iy - fy, &mut wy);
                        lanczos6_weights(ix - fx, &mut wx);
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
                                let mut ra = 0.0f64;
                                for tx in 0..6 {
                                    let xx = base_x + tx as isize;
                                    if xx < 0 || xx >= w as isize || wx[tx] == 0.0 {
                                        continue;
                                    }
                                    ra += wx[tx] * arr[[yy as usize, xx as usize, ch]] as f64;
                                    any = true;
                                }
                                acc += wy[ty] * ra;
                            }
                            out_row[ox * c + ch] = if any { acc as f32 } else { cval };
                        }
                    }
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
                    let step = |x: usize, xe: usize, xw: usize| -> f64 {
                        let ctr = buf[y * w + x];
                        let dn = buf[yn * w + x] - ctr;
                        let ds = buf[ys * w + x] - ctr;
                        let de = buf[y * w + xe] - ctr;
                        let dw = buf[y * w + xw] - ctr;
                        ctr + g * (cond(dn) * dn + cond(ds) * ds + cond(de) * de + cond(dw) * dw)
                    };
                    // Periodic boundary handled at the two edge columns only —
                    // keeps the interior loop free of `%` operations.
                    if w >= 2 {
                        out_row[0] = step(0, 1, w - 1);
                        for x in 1..w - 1 {
                            out_row[x] = step(x, x + 1, x - 1);
                        }
                        out_row[w - 1] = step(w - 1, 0, w - 2);
                    } else if w == 1 {
                        out_row[0] = step(0, 0, 0);
                    }
                });
                std::mem::swap(buf, &mut next);
            }
        }
    });

    let mut out = vec![0f64; h * w * c];
    out.par_chunks_mut(w * c).enumerate().for_each(|(y, orow)| {
        for (ch, buf) in chans.iter().enumerate() {
            for x in 0..w {
                orow[x * c + ch] = buf[y * w + x];
            }
        }
    });
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out).unwrap().into_pyarray(py))
}

/// Fused patch-weighted + sigma-clip combine — one pass, no rejection-mask
/// array. Matches the numpy two-pass path (sigma_clip_combine(return_mask=True)
/// then patch_weighted_mean_combine): per pixel, sigma-clip each channel to get
/// a per-frame reject fraction over channels, then weighted-mean the frames
/// with weight = qmap * global_weight * (1 - reject_fraction).
#[pyfunction]
#[pyo3(signature = (data, qmaps, gweights=None, sigma=3.0, max_iters=3, use_mad=true))]
fn patch_weighted_sigma_combine<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    qmaps: PyReadonlyArray3<'py, f32>,
    gweights: Option<PyReadonlyArray1<'py, f32>>,
    sigma: f32,
    max_iters: usize,
    use_mad: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let qm = qmaps.as_array();
    let s = arr.shape();
    let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
    let gw: Option<Vec<f32>> = gweights.map(|x| x.as_array().to_vec());
    let gwref = gw.as_deref();

    let flat: Option<&[f32]> = arr.as_slice();
    let row_len = w * c;
    let frame_len = h * row_len;

    let mut out = vec![0f32; h * w * c];
    py.allow_threads(|| {
        out.par_chunks_mut(row_len).enumerate().for_each(|(row, out_row)| {
            let mut active = vec![true; n];
            let mut gather: Vec<f32> = Vec::with_capacity(n);
            let mut scratch: Vec<f32> = Vec::with_capacity(n);
            let mut reject_count = vec![0u32; n]; // per-frame rejected-channel count
            let inv_c = 1.0f64 / c as f64;

            // Per-pixel combine given contiguous per-channel sample slices in
            // `block` (layout [(p*c + ch)*n + f]) — see gather-transpose below.
            let combine_col = |block: &[f32], p: usize, col: usize,
                                   active: &mut [bool], gather: &mut Vec<f32>,
                                   scratch: &mut Vec<f32>, reject_count: &mut [u32],
                                   out_row: &mut [f32]| {
                for rc in reject_count.iter_mut() {
                    *rc = 0;
                }
                for ch in 0..c {
                    let chan = &block[(p * c + ch) * n..][..n];
                    sigma_clip_mask(chan, sigma, max_iters, use_mad, active, gather, scratch);
                    for f in 0..n {
                        if !active[f] {
                            reject_count[f] += 1;
                        }
                    }
                }
                let mut wsum = 0f64;
                let mut accs = [0f64; 8]; // supports up to 8 channels
                for f in 0..n {
                    let rej_frac = reject_count[f] as f64 * inv_c;
                    let qwt = qm[[f, row, col]] as f64;
                    let gwt = gwref.map(|g| g[f] as f64).unwrap_or(1.0);
                    let wt = qwt * gwt * (1.0 - rej_frac);
                    if wt == 0.0 {
                        continue;
                    }
                    for ch in 0..c {
                        accs[ch] += wt * block[(p * c + ch) * n + f] as f64;
                    }
                    wsum += wt;
                }
                let denom = if wsum > 1e-12 { wsum } else { 1e-12 };
                for ch in 0..c {
                    out_row[col * c + ch] = (accs[ch] / denom) as f32;
                }
            };

            match flat {
                Some(data) => {
                    // Blocked gather-transpose (same rationale as row_parallel):
                    // sequential row-segment reads per frame instead of N huge-
                    // stride streams per pixel.
                    let tile_cols = (32768 / (n * c).max(1)).clamp(4, 256);
                    let mut block = vec![0f32; tile_cols * c * n];
                    let row_base = row * row_len;
                    let mut start = 0usize;
                    while start < w {
                        let t = tile_cols.min(w - start);
                        for k in 0..n {
                            let src = &data[k * frame_len + row_base + start * c..][..t * c];
                            for (i, &v) in src.iter().enumerate() {
                                block[i * n + k] = v;
                            }
                        }
                        for p in 0..t {
                            combine_col(&block, p, start + p, &mut active, &mut gather,
                                        &mut scratch, &mut reject_count, out_row);
                        }
                        start += t;
                    }
                }
                None => {
                    // Non-contiguous fallback: single-column "tile".
                    let mut block = vec![0f32; c * n];
                    for col in 0..w {
                        for ch in 0..c {
                            for f in 0..n {
                                block[ch * n + f] = arr[[f, row, col, ch]];
                            }
                        }
                        combine_col(&block, 0, col, &mut active, &mut gather,
                                    &mut scratch, &mut reject_count, out_row);
                    }
                }
            }
        });
    });
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out).unwrap().into_pyarray(py))
}

/// scipy `mode='reflect'` boundary index: (d c b a | a b c d | d c b a) — the
/// edge value is duplicated (index -1 == index 0), not a full mirror without
/// repeat. Looped so it is correct even if an offset exceeds one bounce.
#[inline]
fn reflect_idx(i: isize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let n_i = n as isize;
    let mut idx = i;
    while idx < 0 || idx >= n_i {
        idx = if idx < 0 { -idx - 1 } else { 2 * n_i - 1 - idx };
    }
    idx as usize
}

/// Windowed median filter (odd `size`, reflect boundary), row-parallel.
/// The window is tiny (9 or 25 elements for size 3/5) so a per-pixel gather +
/// insertion-style sort beats scipy's generic rank-filter machinery, which
/// pays per-pixel dispatch overhead for an arbitrary footprint.
fn median_filter_2d_f64(data: &[f64], h: usize, w: usize, size: usize) -> Vec<f64> {
    let half = (size / 2) as isize;
    let mut out = vec![0f64; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        let mut window = vec![0f64; size * size];
        for x in 0..w {
            let mut k = 0usize;
            for dy in -half..=half {
                let yy = reflect_idx(y as isize + dy, h);
                let base = yy * w;
                for dx in -half..=half {
                    let xx = reflect_idx(x as isize + dx, w);
                    window[k] = data[base + xx];
                    k += 1;
                }
            }
            window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            row_out[x] = window[window.len() / 2];
        }
    });
    out
}

/// f32 variant of the above (used by the hot-pixel detector, which operates on
/// float32 luminance).
fn median_filter_2d_f32(data: &[f32], h: usize, w: usize, size: usize) -> Vec<f32> {
    let half = (size / 2) as isize;
    let mut out = vec![0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        let mut window = vec![0f32; size * size];
        for x in 0..w {
            let mut k = 0usize;
            for dy in -half..=half {
                let yy = reflect_idx(y as isize + dy, h);
                let base = yy * w;
                for dx in -half..=half {
                    let xx = reflect_idx(x as isize + dx, w);
                    window[k] = data[base + xx];
                    k += 1;
                }
            }
            window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            row_out[x] = window[window.len() / 2];
        }
    });
    out
}

/// 3x3 Laplacian [[0,-1,0],[-1,4,-1],[0,-1,0]], reflect boundary. The kernel
/// is symmetric under 180-degree rotation, so `convolve` and `correlate`
/// coincide — no flip needed to match `scipy.ndimage.convolve`.
fn laplacian_2d(data: &[f64], h: usize, w: usize) -> Vec<f64> {
    let mut out = vec![0f64; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        let yn = reflect_idx(y as isize - 1, h);
        let ys = reflect_idx(y as isize + 1, h);
        let row_c = y * w;
        let row_n = yn * w;
        let row_s = ys * w;
        for x in 0..w {
            let xw = reflect_idx(x as isize - 1, w);
            let xe = reflect_idx(x as isize + 1, w);
            row_out[x] = 4.0 * data[row_c + x]
                - data[row_n + x]
                - data[row_s + x]
                - data[row_c + xw]
                - data[row_c + xe];
        }
    });
    out
}

/// L.A.Cosmic-style cosmic-ray rejection, matching `lacosmic_reject` in
/// `src/stacking.py`: per channel, Laplacian spike / local-noise-model
/// detection statistic, object-rejection ratio to protect star cores,
/// replace flagged pixels with the 5x5 local median. All f64 internally
/// (matching the numpy reference's dtype) for parity in the threshold test;
/// output f32. Channels are processed sequentially (each internally
/// row-parallel) since S_med depends on the fully-computed S array.
#[pyfunction]
#[pyo3(signature = (data, sigclip=4.5, objlim=5.0, gain=1.0, readnoise=6.5))]
fn lacosmic_reject_native<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f32>,
    sigclip: f64,
    objlim: f64,
    gain: f64,
    readnoise: f64,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (h, w, c) = (s[0], s[1], s[2]);
    let rn_term = (readnoise / gain) * (readnoise / gain);
    let flat: Option<&[f32]> = arr.as_slice();

    let mut out = vec![0f32; h * w * c];
    if c != 3 {
        // Mirror the Python early-return: pass the input through unchanged.
        match flat {
            Some(f) => out.copy_from_slice(f),
            None => {
                for y in 0..h {
                    for x in 0..w {
                        for ch in 0..c {
                            out[(y * w + x) * c + ch] = arr[[y, x, ch]];
                        }
                    }
                }
            }
        }
        return Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out)
            .unwrap()
            .into_pyarray(py));
    }

    py.allow_threads(|| {
        for ch in 0..3usize {
            let mut chd = vec![0f64; h * w];
            match flat {
                Some(f) => {
                    chd.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                        let base = y * w * c + ch;
                        for x in 0..w {
                            row[x] = f[base + x * c] as f64;
                        }
                    });
                }
                None => {
                    for y in 0..h {
                        for x in 0..w {
                            chd[y * w + x] = arr[[y, x, ch]] as f64;
                        }
                    }
                }
            }

            let fine = laplacian_2d(&chd, h, w);
            let med5 = median_filter_2d_f64(&chd, h, w, 5);

            let mut sarr = vec![0f64; h * w];
            sarr.par_iter_mut().enumerate().for_each(|(i, sv)| {
                let f = fine[i].max(0.0);
                let noise = (med5[i].max(0.0) / gain + rn_term).sqrt().max(1e-6);
                *sv = f / (2.0 * noise);
            });
            let smed = median_filter_2d_f64(&sarr, h, w, 3);

            out.par_chunks_mut(w * c).enumerate().for_each(|(y, row_out)| {
                let row_base = y * w;
                for x in 0..w {
                    let i = row_base + x;
                    let sv = sarr[i];
                    let smv = smed[i].max(1e-6);
                    let ratio = sv / smv;
                    let val = if sv > sigclip && ratio > objlim { med5[i] } else { chd[i] };
                    row_out[x * c + ch] = val as f32;
                }
            });
        }
    });

    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out)
        .unwrap()
        .into_pyarray(py))
}

/// Standalone median filter (odd `size`, reflect boundary) over a single
/// float32 2D array. Exposed for reuse by other hot-pixel-style detectors.
#[pyfunction]
#[pyo3(signature = (data, size=3))]
fn median_filter_native<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f32>,
    size: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (h, w) = (s[0], s[1]);
    let flat: Vec<f32> = arr.iter().copied().collect();
    let out = py.allow_threads(|| median_filter_2d_f32(&flat, h, w, size));
    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), out)
        .unwrap()
        .into_pyarray(py))
}

#[pymodule]
fn astro_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sigma_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(patch_weighted_sigma_combine, m)?)?;
    m.add_function(wrap_pyfunction!(median_combine, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(trimmed_mean_combine, m)?)?;
    m.add_function(wrap_pyfunction!(esd_combine, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine_lanczos3, m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_diffusion, m)?)?;
    m.add_function(wrap_pyfunction!(lacosmic_reject_native, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_native, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
