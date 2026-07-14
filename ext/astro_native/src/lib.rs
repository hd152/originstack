//! Native hot-path kernels for OriginStack.
//!
//! Currently exposes `sigma_clip_combine`, a per-pixel iterative sigma-clip /
//! winsorized combine over an `(N, H, W, C)` stack of aligned frames. It mirrors
//! the numpy reference in `src/stacking.py` (`_sigma_clip_tile`) but runs the
//! per-pixel loop in native code, parallelised across image rows with rayon, so
//! there is no per-tile float32 copy and no repeated whole-stack NaN passes.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4,
};
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
                    // NOTE: intentionally no early-break on a non-rejecting
                    // iteration. The numpy reference (`_esd_clip_tile`) breaks
                    // the loop *per tile* when no pixel rejects; in a large
                    // production tile that keeps some pixel active it therefore
                    // re-tests every pixel through all `max_outliers` iterations
                    // — which this per-pixel "run all iterations" loop matches.
                    // A per-pixel break here would instead match numpy's
                    // degenerate single-pixel-tile case and diverge on real
                    // tiles (see tests/test_native.py::test_esd_matches_numpy).
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
    // Diagonal matrix (pure translation or axis-aligned scaling, e.g. the
    // drizzle output grid): iy depends only on oy and ix only on ox, so the
    // Lanczos weights are separable — one wx table per image, one wy per row.
    // This covers the common cases (rotation off) and removes ALL sin() calls
    // plus the weight normalisation from the per-pixel loop.
    let is_sep = m01 == 0.0 && m10 == 0.0;
    let flat: Option<&[f32]> = arr.as_slice();

    let col_tab: Option<(Vec<[f64; 6]>, Vec<isize>)> = if is_sep && flat.is_some() {
        let mut wxs = vec![[0f64; 6]; out_w];
        let mut bxs = vec![0isize; out_w];
        for ox in 0..out_w {
            let ix = m11 * ox as f64 + o1;
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
                                    let iy = m00 * oy as f64 + o0;
                                    let fy = iy.floor();
                                    lanczos6_weights(iy - fy, &mut wy);
                                }
                                let iy = m00 * oy as f64 + o0;
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
///
/// `qmaps` is either (N, H, W) full-resolution weights (grid_geom = None,
/// original behaviour), or (N, gh, gw) coarse patch grids with
/// `grid_geom = (h_full, w_full, top, left)`: the weight at cropped pixel
/// (row, col) is the grid sampled bilinearly at full-frame coordinates
/// ((row+top)*(gh-1)/(h_full-1), (col+left)*(gw-1)/(w_full-1)) — the same
/// corner-aligned mapping scipy `zoom(order=1)` uses, so it matches the old
/// upsample-then-crop path without ever materialising N full-res maps.
#[pyfunction]
#[pyo3(signature = (data, qmaps, gweights=None, sigma=3.0, max_iters=3, use_mad=true, grid_geom=None))]
fn patch_weighted_sigma_combine<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    qmaps: PyReadonlyArray3<'py, f32>,
    gweights: Option<PyReadonlyArray1<'py, f32>>,
    sigma: f32,
    max_iters: usize,
    use_mad: bool,
    grid_geom: Option<(f64, f64, f64, f64)>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let qm = qmaps.as_array();
    let s = arr.shape();
    let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
    let gw: Option<Vec<f32>> = gweights.map(|x| x.as_array().to_vec());
    let gwref = gw.as_deref();

    let (qh, qw) = (qm.shape()[1], qm.shape()[2]);
    // Per-column grid sample tables (index + fraction), computed once.
    let col_tab: Option<(Vec<usize>, Vec<f32>)> = grid_geom.map(|(_, wf, _, left)| {
        let sx = if wf > 1.0 && qw > 1 {
            (qw - 1) as f64 / (wf - 1.0)
        } else {
            0.0
        };
        let mut gx0 = vec![0usize; w];
        let mut fx = vec![0f32; w];
        for col in 0..w {
            let g = ((col as f64 + left) * sx).clamp(0.0, (qw - 1) as f64);
            let g0 = (g.floor() as usize).min(qw.saturating_sub(2).max(0));
            gx0[col] = g0;
            fx[col] = (g - g0 as f64) as f32;
        }
        (gx0, fx)
    });

    let flat: Option<&[f32]> = arr.as_slice();
    let row_len = w * c;
    let frame_len = h * row_len;

    let mut out = vec![0f32; h * w * c];
    py.allow_threads(|| {
        out.par_chunks_mut(row_len).enumerate().for_each(|(row, out_row)| {
            // Per-row grid sample coordinate (index + fraction).
            let row_tab: Option<(usize, f32)> = grid_geom.map(|(hf, _, top, _)| {
                let sy = if hf > 1.0 && qh > 1 {
                    (qh - 1) as f64 / (hf - 1.0)
                } else {
                    0.0
                };
                let g = ((row as f64 + top) * sy).clamp(0.0, (qh - 1) as f64);
                let g0 = (g.floor() as usize).min(qh.saturating_sub(2).max(0));
                (g0, (g - g0 as f64) as f32)
            });
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
                    let qwt = match (&row_tab, &col_tab) {
                        (Some((gy0, fy)), Some((gx0s, fxs))) => {
                            let (gx0, fx) = (gx0s[col], fxs[col]);
                            let gy1 = (gy0 + 1).min(qh - 1);
                            let gx1 = (gx0 + 1).min(qw - 1);
                            let q00 = qm[[f, *gy0, gx0]];
                            let q01 = qm[[f, *gy0, gx1]];
                            let q10 = qm[[f, gy1, gx0]];
                            let q11 = qm[[f, gy1, gx1]];
                            let top_v = q00 + (q01 - q00) * fx;
                            let bot_v = q10 + (q11 - q10) * fx;
                            (top_v + (bot_v - top_v) * fy) as f64
                        }
                        _ => qm[[f, row, col]] as f64,
                    };
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

/// Exact median of 9 via Paeth's 19-op compare-exchange network
/// (Graphics Gems: "Median finding on a 3x3 grid"). Branchless, so it
/// vectorises; ~5x faster than a comparator sort of the window.
#[inline(always)]
fn median9(p: &mut [f32; 9]) -> f32 {
    sort2_idx(p, 1, 2); sort2_idx(p, 4, 5); sort2_idx(p, 7, 8);
    sort2_idx(p, 0, 1); sort2_idx(p, 3, 4); sort2_idx(p, 6, 7);
    sort2_idx(p, 1, 2); sort2_idx(p, 4, 5); sort2_idx(p, 7, 8);
    sort2_idx(p, 0, 3); sort2_idx(p, 5, 8); sort2_idx(p, 4, 7);
    sort2_idx(p, 3, 6); sort2_idx(p, 1, 4); sort2_idx(p, 2, 5);
    sort2_idx(p, 4, 7); sort2_idx(p, 4, 2); sort2_idx(p, 6, 4);
    sort2_idx(p, 4, 2);
    p[4]
}

/// Branchless compare-exchange via hardware minss/maxss.
/// NaN note: f32::min/max return the non-NaN operand, which differs from the
/// sort-based border path's "NaN compares Equal"; inputs here are calibrated
/// frames already validated finite upstream, so the case cannot occur.
#[inline(always)]
fn sort2_idx(p: &mut [f32; 9], i: usize, j: usize) {
    let (a, b) = (p[i], p[j]);
    p[i] = a.min(b);
    p[j] = a.max(b);
}

/// Windowed median filter (odd `size`, reflect boundary), row-parallel.
/// The window is tiny (9 or 25 elements for size 3/5) so a per-pixel gather
/// beats scipy's generic rank-filter machinery. Interior pixels (no boundary
/// reflection possible) take a fast path: contiguous row-segment reads with no
/// per-tap reflect_idx, then a branchless median network (3x3) or quickselect
/// (5x5) instead of a full comparator sort. f32 (not f64): used by
/// lacosmic_reject_native and the hot-pixel detector, both of which run inside
/// many concurrent ProcessPoolExecutor workers, where halving the bytes moved
/// per call directly reduces shared memory-bandwidth contention.
fn median_filter_2d_f32(data: &[f32], h: usize, w: usize, size: usize) -> Vec<f32> {
    let half = (size / 2) as isize;
    let hu = size / 2;
    let mut out = vec![0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        let mut window = vec![0f32; size * size];
        // Border/generic path: reflect boundary + comparator sort (exact
        // median, same as before; only runs on the frame edges).
        let generic = |x: usize, window: &mut [f32], row_out: &mut [f32]| {
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
        };

        let interior_y = y >= hu && y + hu < h;
        if !interior_y || w < size {
            for x in 0..w {
                generic(x, &mut window, row_out);
            }
            return;
        }
        for x in 0..hu {
            generic(x, &mut window, row_out);
        }
        for x in (w - hu)..w {
            generic(x, &mut window, row_out);
        }
        if size != 3 && size != 5 {
            // Any other odd size (MMT uses 9 and 17): contiguous row-segment
            // gather + O(n) quickselect. No per-tap reflect_idx in the
            // interior; borders take the generic path above.
            let mid = (size * size) / 2;
            for x in hu..w - hu {
                for ty in 0..size {
                    let base = (y - hu + ty) * w + x - hu;
                    window[ty * size..(ty + 1) * size]
                        .copy_from_slice(&data[base..base + size]);
                }
                let (_, &mut m, _) = window.select_nth_unstable_by(mid, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                row_out[x] = m;
            }
            return;
        }
        if size == 3 {
            let r0 = (y - 1) * w;
            let r1 = y * w;
            let r2 = (y + 1) * w;
            for x in 1..w - 1 {
                let mut p = [
                    data[r0 + x - 1], data[r0 + x], data[r0 + x + 1],
                    data[r1 + x - 1], data[r1 + x], data[r1 + x + 1],
                    data[r2 + x - 1], data[r2 + x], data[r2 + x + 1],
                ];
                row_out[x] = median9(&mut p);
            }
        } else {
            // size == 5: contiguous 5x5 gather + O(n) quickselect.
            // (f32::total_cmp was tried here and measured slightly slower
            // than the partial_cmp closure — 106ms vs 98ms full-frame.)
            let mut p = [0f32; 25];
            for x in 2..w - 2 {
                for ty in 0..5 {
                    let base = (y - 2 + ty) * w + x - 2;
                    p[ty * 5..ty * 5 + 5].copy_from_slice(&data[base..base + 5]);
                }
                let (_, &mut m, _) = p.select_nth_unstable_by(12, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                row_out[x] = m;
            }
        }
    });
    out
}

/// 3x3 Laplacian [[0,-1,0],[-1,4,-1],[0,-1,0]], reflect boundary. The kernel
/// is symmetric under 180-degree rotation, so `convolve` and `correlate`
/// coincide — no flip needed to match `scipy.ndimage.convolve`. f32 (not
/// f64): see `lacosmic_reject_native` for why this matters more here than
/// raw compute — this kernel's working set (5 full-frame arrays per channel:
/// source, fine, med5, S, S_med) is what gets driven through memory under
/// real multi-process contention, and f32 halves it.
fn laplacian_2d_f32(data: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        let yn = reflect_idx(y as isize - 1, h);
        let ys = reflect_idx(y as isize + 1, h);
        let row_c = y * w;
        let row_n = yn * w;
        let row_s = ys * w;
        let lap = |x: usize, xw: usize, xe: usize| -> f32 {
            4.0 * data[row_c + x]
                - data[row_n + x]
                - data[row_s + x]
                - data[row_c + xw]
                - data[row_c + xe]
        };
        // Reflection only matters at the two edge columns; the interior loop
        // is a pure 5-point stencil the compiler can vectorise.
        if w >= 2 {
            row_out[0] = lap(0, 0, 1); // reflect: index -1 == index 0
            for x in 1..w - 1 {
                row_out[x] = lap(x, x - 1, x + 1);
            }
            row_out[w - 1] = lap(w - 1, w - 2, w - 1); // index w == w-1
        } else if w == 1 {
            row_out[0] = lap(0, 0, 0);
        }
    });
    out
}

/// L.A.Cosmic-style cosmic-ray rejection, matching `lacosmic_reject` in
/// `src/stacking.py`: per channel, Laplacian spike / local-noise-model
/// detection statistic, object-rejection ratio to protect star cores,
/// replace flagged pixels with the 5x5 local median.
///
/// f32 internally (NOT f64, unlike this kernel's first version). Two real
/// 233-frame production runs on 16-core hardware showed this kernel getting
/// SLOWER under real ProcessPoolExecutor contention (~8s/frame) than a naive
/// single-thread estimate would predict, while CA correction's downsample fix
/// (which cuts memory traffic, not thread count) gave its full isolated
/// speedup in production. Same diagnosis applies here: under N concurrent
/// worker processes each running this kernel, the limiter is shared memory
/// bandwidth, not core count — and this kernel moves 5 full-frame arrays
/// through memory per channel (source, fine, med5, S, S_med). f32 halves
/// that traffic. The f64 precision was never required for correctness: this
/// is a threshold test (S > sigclip), not an accumulation sensitive to
/// rounding — see tests/test_native.py for the parity bound now in effect
/// (matches the original numpy f64 reference to <0.5 ADU per pixel, not
/// exact-zero as the f64 Rust version achieved).
/// Output f32. Channels processed sequentially (each internally row-parallel)
/// since S_med depends on the fully-computed S array.
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
    let rn_term = ((readnoise / gain) * (readnoise / gain)) as f32;
    let sigclip = sigclip as f32;
    let objlim = objlim as f32;
    let gain = gain as f32;
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
            let mut chd = vec![0f32; h * w];
            match flat {
                Some(f) => {
                    chd.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                        let base = y * w * c + ch;
                        for x in 0..w {
                            row[x] = f[base + x * c];
                        }
                    });
                }
                None => {
                    for y in 0..h {
                        for x in 0..w {
                            chd[y * w + x] = arr[[y, x, ch]];
                        }
                    }
                }
            }

            let fine = laplacian_2d_f32(&chd, h, w);
            let med5 = median_filter_2d_f32(&chd, h, w, 5);

            let mut sarr = vec![0f32; h * w];
            sarr.par_iter_mut().enumerate().for_each(|(i, sv)| {
                let f = fine[i].max(0.0);
                let noise = (med5[i].max(0.0) / gain + rn_term).sqrt().max(1e-6);
                *sv = f / (2.0 * noise);
            });
            let smed = median_filter_2d_f32(&sarr, h, w, 3);

            out.par_chunks_mut(w * c).enumerate().for_each(|(y, row_out)| {
                let row_base = y * w;
                for x in 0..w {
                    let i = row_base + x;
                    let sv = sarr[i];
                    let smv = smed[i].max(1e-6);
                    let ratio = sv / smv;
                    let val = if sv > sigclip && ratio > objlim { med5[i] } else { chd[i] };
                    row_out[x * c + ch] = val;
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
    // Zero-copy view when the numpy array is contiguous (the normal case);
    // the collect fallback only runs for strided views.
    let owned: Vec<f32>;
    let flat: &[f32] = match arr.as_slice() {
        Some(sl) => sl,
        None => {
            owned = arr.iter().copied().collect();
            &owned
        }
    };
    let out = py.allow_threads(|| median_filter_2d_f32(flat, h, w, size));
    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), out)
        .unwrap()
        .into_pyarray(py))
}

// ---------------------------------------------------------------------------
// DBE robust background-surface fit
// ---------------------------------------------------------------------------

/// Median of an f64 slice (copies + sorts; N is small — DBE patch counts).
fn median_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Gaussian-weighted local accumulators at an evaluation point, for a
/// weighted local-linear (degree-1) fit. Offsets are in units of sigma.
/// Samples beyond `trunc_d2` sigma^2 are skipped (weight < ~4e-6 at 25).
#[allow(clippy::too_many_arguments)]
#[inline]
fn dbe_accum(
    py_: f64,
    px_: f64,
    ys: &[f64],
    xs: &[f64],
    vs: &[f64],
    wr: &[f64],
    inv_sigma: f64,
    trunc_d2: f64,
) -> [f64; 9] {
    let mut acc = [0.0f64; 9]; // sw swy swx swyy swyx swxx swv swyv swxv
    for j in 0..ys.len() {
        let dy = (ys[j] - py_) * inv_sigma;
        let dx = (xs[j] - px_) * inv_sigma;
        let d2 = dy * dy + dx * dx;
        if d2 > trunc_d2 {
            continue;
        }
        let w = wr[j] * (-0.5 * d2).exp();
        let v = vs[j];
        acc[0] += w;
        acc[1] += w * dy;
        acc[2] += w * dx;
        acc[3] += w * dy * dy;
        acc[4] += w * dy * dx;
        acc[5] += w * dx * dx;
        acc[6] += w * v;
        acc[7] += w * dy * v;
        acc[8] += w * dx * v;
    }
    acc
}

/// Solve the ridge-regularised 3x3 local-linear normal equations; returns the
/// fitted value at the expansion center (the constant term). Falls back to
/// the plain weighted mean when ill-conditioned, NaN when there is no weight.
#[inline]
fn dbe_solve(acc: &[f64; 9]) -> f64 {
    let [sw, swy, swx, swyy, swyx, swxx, swv, swyv, swxv] = *acc;
    if sw < 1e-12 {
        return f64::NAN;
    }
    let lam = 1e-3 * sw; // ridge on the slope terms only
    let (a11, a12, a13) = (sw, swy, swx);
    let (a22, a23) = (swyy + lam, swyx);
    let a33 = swxx + lam;
    let det = a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a23 * a13)
        + a13 * (a12 * a23 - a22 * a13);
    let scale = (a11.abs() * a22.abs() * a33.abs()).max(1e-30);
    if det.abs() < 1e-10 * scale {
        return swv / sw; // Nadaraya-Watson fallback
    }
    let det1 = swv * (a22 * a33 - a23 * a23) - a12 * (swyv * a33 - a23 * swxv)
        + a13 * (swyv * a23 - a22 * swxv);
    det1 / det
}

/// Evaluate the robust local-linear fit at one point, widening the truncation
/// radius if the point sits in a large gap with no nearby samples.
#[inline]
fn dbe_fit_at(
    py_: f64,
    px_: f64,
    ys: &[f64],
    xs: &[f64],
    vs: &[f64],
    wr: &[f64],
    inv_sigma: f64,
    global_mean: f64,
) -> f64 {
    let mut acc = dbe_accum(py_, px_, ys, xs, vs, wr, inv_sigma, 25.0);
    if acc[0] < 1e-12 {
        acc = dbe_accum(py_, px_, ys, xs, vs, wr, inv_sigma, f64::INFINITY);
    }
    let v = dbe_solve(&acc);
    if v.is_nan() {
        global_mean
    } else {
        v
    }
}

/// Robust background-surface fit for DBE: Gaussian-weighted local-linear
/// regression with IRLS (Tukey biweight) downweighting of contaminated
/// patch samples. Replaces the former unbounded thin-plate-spline RBF +
/// hard outlier-rejection loop: the local fit stays near the surrounding
/// sample values by construction (no runaway extrapolation into sample
/// gaps near bright stars), and IRLS downweights outliers continuously
/// instead of carving hard gaps into the sample set.
///
/// `coords` are (N,2) normalized (y/H, x/W) patch centers; `values` the
/// patch sky medians; `sigma_px` the Gaussian bandwidth in pixels. The
/// surface is evaluated on a (grid_h, grid_w) grid spanning
/// linspace(0,1)×linspace(0,1) in normalized coordinates (matching the
/// caller's zoom-to-full-res convention). Returns (surface, robust_weights).
#[pyfunction]
#[pyo3(signature = (coords, values, img_h, img_w, grid_h, grid_w, sigma_px, tukey_c=4.685, irls_iters=3))]
#[allow(clippy::too_many_arguments)]
fn dbe_fit_surface<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    img_h: f64,
    img_w: f64,
    grid_h: usize,
    grid_w: usize,
    sigma_px: f64,
    tukey_c: f64,
    irls_iters: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let carr = coords.as_array();
    let vs: Vec<f64> = values.as_array().to_vec();
    let n = vs.len();
    if carr.shape()[0] != n || carr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coords must be (N,2) matching values length",
        ));
    }
    let ys: Vec<f64> = (0..n).map(|i| carr[[i, 0]] * img_h).collect();
    let xs: Vec<f64> = (0..n).map(|i| carr[[i, 1]] * img_w).collect();
    let inv_sigma = 1.0 / sigma_px.max(1e-9);

    let (surface, wrob) = py.allow_threads(|| {
        let mut wr = vec![1.0f64; n];

        // IRLS: refit each sample leave-one-out, reweight by Tukey biweight.
        for _ in 0..irls_iters {
            let residuals: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut acc =
                        dbe_accum(ys[i], xs[i], &ys, &xs, &vs, &wr, inv_sigma, 25.0);
                    // remove self (d2 = 0 -> contributes to sw and swv only)
                    acc[0] -= wr[i];
                    acc[6] -= wr[i] * vs[i];
                    let fit = dbe_solve(&acc);
                    if fit.is_nan() {
                        0.0
                    } else {
                        vs[i] - fit
                    }
                })
                .collect();
            let med_r = median_f64(&residuals);
            let abs_dev: Vec<f64> = residuals.iter().map(|r| (r - med_r).abs()).collect();
            let s = 1.4826 * median_f64(&abs_dev);
            if s < 1e-9 {
                break;
            }
            let cs = tukey_c * s;
            for i in 0..n {
                let u = (residuals[i] - med_r) / cs;
                wr[i] = if u.abs() < 1.0 {
                    let t = 1.0 - u * u;
                    t * t
                } else {
                    0.0
                };
            }
        }

        let wsum: f64 = wr.iter().sum();
        let global_mean = if wsum > 1e-12 {
            wr.iter().zip(&vs).map(|(w, v)| w * v).sum::<f64>() / wsum
        } else {
            median_f64(&vs)
        };

        // Evaluate on the coarse grid (parallel over rows).
        let mut surface = vec![0.0f64; grid_h * grid_w];
        surface
            .par_chunks_mut(grid_w)
            .enumerate()
            .for_each(|(gi, row)| {
                let gy = if grid_h > 1 {
                    gi as f64 / (grid_h - 1) as f64 * img_h
                } else {
                    0.0
                };
                for (gj, out) in row.iter_mut().enumerate() {
                    let gx = if grid_w > 1 {
                        gj as f64 / (grid_w - 1) as f64 * img_w
                    } else {
                        0.0
                    };
                    *out = dbe_fit_at(gy, gx, &ys, &xs, &vs, &wr, inv_sigma, global_mean);
                }
            });
        (surface, wr)
    });

    let out = numpy::ndarray::Array2::from_shape_vec((grid_h, grid_w), surface)
        .expect("shape mismatch building DBE surface");
    Ok((out.into_pyarray(py), wrob.into_pyarray(py)))
}

/// DBE background-patch sampler — the per-patch loop of
/// `_sample_background_patches` in src/background.py: for each grid cell,
/// reject emission-masked/bright patches, sigma-clip, and return the patch
/// centre (normalised), clipped median, and clipped variance. The variance
/// and entropy filters stay in Python (cheap, operate on the small result).
/// Medians are exact order statistics, so f32 input matches the f64
/// reference wherever the values are f32-representable.
#[pyfunction]
#[pyo3(signature = (channel, emission_mask, patch_size, masked_frac_thresh, sky_ref, sky_std))]
fn dbe_sample_patches<'py>(
    py: Python<'py>,
    channel: PyReadonlyArray2<'py, f32>,
    emission_mask: PyReadonlyArray2<'py, f32>,
    patch_size: usize,
    masked_frac_thresh: f64,
    sky_ref: f64,
    sky_std: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let ch = channel.as_array();
    let em = emission_mask.as_array();
    let (h, w) = (ch.shape()[0], ch.shape()[1]);
    if em.shape() != [h, w] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "emission_mask shape must match channel",
        ));
    }
    let ch_owned: Vec<f32>;
    let ch_flat: &[f32] = match ch.as_slice() {
        Some(s) => s,
        None => {
            ch_owned = ch.iter().copied().collect();
            &ch_owned
        }
    };
    let em_owned: Vec<f32>;
    let em_flat: &[f32] = match em.as_slice() {
        Some(s) => s,
        None => {
            em_owned = em.iter().copied().collect();
            &em_owned
        }
    };
    let ny = (h / patch_size.max(1)).max(1);
    let nx = (w / patch_size.max(1)).max(1);
    let cell_h = h as f64 / ny as f64;
    let cell_w = w as f64 / nx as f64;
    let bright_cut = sky_ref + 2.0 * sky_std.max(1.0);

    let rows: Vec<Vec<(f64, f64, f64, f64)>> = py.allow_threads(|| {
        (0..ny)
            .into_par_iter()
            .map(|iy| {
                // round_ties_even matches Python's banker's-rounding round()
                let y0 = (iy as f64 * cell_h).round_ties_even() as usize;
                let y1 = ((((iy + 1) as f64) * cell_h).round_ties_even() as usize).min(h);
                let mut out = Vec::new();
                let mut px: Vec<f32> = Vec::with_capacity(patch_size * patch_size * 2);
                let mut dev: Vec<f32> = Vec::with_capacity(px.capacity());
                for ix in 0..nx {
                    let x0 = (ix as f64 * cell_w).round_ties_even() as usize;
                    let x1 = ((((ix + 1) as f64) * cell_w).round_ties_even() as usize).min(w);
                    if y1 <= y0 || x1 <= x0 {
                        continue;
                    }
                    let total = (y1 - y0) * (x1 - x0);
                    let mut masked = 0usize;
                    px.clear();
                    for y in y0..y1 {
                        let base = y * w;
                        for x in x0..x1 {
                            if em_flat[base + x] >= 0.5 {
                                masked += 1;
                            } else {
                                px.push(ch_flat[base + x]);
                            }
                        }
                    }
                    if masked > 0 && (masked as f64 / total as f64) > masked_frac_thresh {
                        continue;
                    }
                    if px.len() < 10 {
                        continue;
                    }
                    dev.clear();
                    dev.extend_from_slice(&px);
                    let patch_med = median_inplace(&mut dev) as f64;
                    if patch_med > bright_cut {
                        continue;
                    }
                    dev.clear();
                    dev.extend(px.iter().map(|&v| (v as f64 - patch_med).abs() as f32));
                    let mad = median_inplace(&mut dev) as f64;
                    let sig = 1.4826 * mad;
                    if sig > 1e-12 {
                        let cut = (3.0 * sig) as f32;
                        let med32 = patch_med as f32;
                        px.retain(|&v| (v - med32).abs() <= cut);
                    }
                    let med_val = if px.is_empty() {
                        patch_med
                    } else {
                        dev.clear();
                        dev.extend_from_slice(&px);
                        median_inplace(&mut dev) as f64
                    };
                    // Population variance (ddof=0) in f64, matching np.var.
                    let var = if px.is_empty() {
                        0.0
                    } else {
                        let m: f64 =
                            px.iter().map(|&v| v as f64).sum::<f64>() / px.len() as f64;
                        px.iter().map(|&v| (v as f64 - m) * (v as f64 - m)).sum::<f64>()
                            / px.len() as f64
                    };
                    let cy = (y0 as f64 + (y1 - y0) as f64 * 0.5) / h as f64;
                    let cx = (x0 as f64 + (x1 - x0) as f64 * 0.5) / w as f64;
                    out.push((cy, cx, med_val, var));
                }
                out
            })
            .collect()
    });

    let flat: Vec<(f64, f64, f64, f64)> = rows.into_iter().flatten().collect();
    let n = flat.len();
    let mut coords = Vec::with_capacity(n * 2);
    let mut values = Vec::with_capacity(n);
    let mut variances = Vec::with_capacity(n);
    for (cy, cx, v, var) in flat {
        coords.push(cy);
        coords.push(cx);
        values.push(v);
        variances.push(var);
    }
    let carr = numpy::ndarray::Array2::from_shape_vec((n, 2), coords)
        .expect("shape mismatch building patch coords");
    Ok((
        carr.into_pyarray(py),
        values.into_pyarray(py),
        variances.into_pyarray(py),
    ))
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
    m.add_function(wrap_pyfunction!(dbe_fit_surface, m)?)?;
    m.add_function(wrap_pyfunction!(dbe_sample_patches, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
