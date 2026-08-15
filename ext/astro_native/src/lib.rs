//! Native hot-path kernels for OriginStack.
//!
//! Currently exposes `sigma_clip_combine`, a per-pixel iterative sigma-clip /
//! winsorized combine over an `(N, H, W, C)` stack of aligned frames. It mirrors
//! the numpy reference in `src/stacking.py` (`_sigma_clip_tile`) but runs the
//! per-pixel loop in native code, parallelised across image rows with rayon, so
//! there is no per-tile float32 copy and no repeated whole-stack NaN passes.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4, PyReadwriteArray3,
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

/// Burn-in seed math for ONE pixel: MAD-reject the `k`-sample burn-in window
/// and return the persistent Welford state `(running_mean, m2, n_acc,
/// n_rejected)`. `n_acc` is clamped to >=1 and carried forward as state (not
/// just at the point of division) -- matches the numpy reference, where an
/// all-rejected burn-in window still yields a defined (phantom-count)
/// running estimate rather than a divide-by-zero. Shared by
/// `online_sigma_clip_pixel` (whole-array kernel) and
/// `online_sigma_clip_seed_burnin` (streaming kernel) so both stay bit-for-bit
/// identical on the burn-in math.
///
/// `valid[i]` = sample `i` was actually covered by that frame's warp (not a
/// zero-fill pixel from an out-of-frame shift/rotation) -- invalid samples
/// never enter the median/MAD/mean/M2 computation, same idea as
/// `fold_pixel`'s coverage gating but per-sample instead of per-frame,
/// since a burn-in window mixes several frames' warps at once. If every
/// sample at a pixel is invalid (never happens for the whole-array kernel,
/// which always passes all-true), there's no defined signal yet: return the
/// same `n_acc=0` sentinel `fold_pixel` uses for "never seeded" -- NOT a
/// phantom `(mean=0, n_acc=1)` state, which would make the first real sample
/// to arrive fail the accept-test against a fabricated near-zero-variance
/// estimate and get rejected forever (the pixel would stay frozen at 0 for
/// the rest of the stream). `fold_pixel` special-cases `n_acc<=0` to
/// initialize directly from that first real sample instead.
#[inline]
fn burnin_seed_pixel(
    burn: &[f32],
    valid: &[bool],
    sigma: f64,
    scratch: &mut Vec<f32>,
) -> (f64, f64, f64, usize) {
    let k = burn.len();
    scratch.clear();
    for i in 0..k {
        if valid[i] {
            scratch.push(burn[i]);
        }
    }
    if scratch.is_empty() {
        return (0.0, 0.0, 0.0, k);
    }
    let med = median_inplace(scratch) as f64;
    scratch.clear();
    for i in 0..k {
        if valid[i] {
            scratch.push((burn[i] as f64 - med).abs() as f32);
        }
    }
    let mad = median_inplace(scratch) as f64;
    let robust_sigma = (1.4826 * mad).max(1e-6);
    let thresh0 = sigma * robust_sigma;

    let mut accepted_in_burn = 0usize;
    let mut sum_acc = 0f64;
    for i in 0..k {
        if valid[i] && (burn[i] as f64 - med).abs() <= thresh0 {
            accepted_in_burn += 1;
            sum_acc += burn[i] as f64;
        }
    }
    let n_acc = (accepted_in_burn as f64).max(1.0);
    let running_mean = sum_acc / n_acc;
    let mut m2 = 0f64;
    for i in 0..k {
        if valid[i] && (burn[i] as f64 - med).abs() <= thresh0 {
            let d = burn[i] as f64 - running_mean;
            m2 += d * d;
        }
    }
    // Invalid samples are excluded, not "rejected" in the sigma-clip sense,
    // but they still never became part of n_acc -- counted the same way
    // as a MAD-rejected sample for the returned count.
    let n_rejected = k - accepted_in_burn;
    (running_mean, m2, n_acc, n_rejected)
}

/// Single-sample Welford accept-test + update for ONE pixel: given the
/// current running state and one new value, either fold it in (returns the
/// updated state, `true`) or leave the state unchanged (returns it as-is,
/// `false`). Shared by `online_sigma_clip_pixel` and
/// `online_sigma_clip_fold_frame`.
///
/// `n_acc<=0` is the "unseeded" sentinel `burnin_seed_pixel` returns for a
/// pixel no burn-in frame covered (large/drifting dithers can leave 100+ px
/// borders uncovered). Running the accept-test against that fabricated state
/// would reject the first real sample forever; instead initialize directly
/// from it.
#[inline]
fn fold_pixel(mean: f64, m2: f64, n_acc: f64, x: f64, sigma: f64) -> (f64, f64, f64, bool) {
    if n_acc <= 0.0 {
        return (x, 0.0, 1.0, true);
    }
    let var_est = m2 / n_acc;
    let std_est = var_est.max(1e-12).sqrt();
    if (x - mean).abs() <= sigma * std_est {
        let n_acc_new = n_acc + 1.0;
        let delta = x - mean;
        let new_mean = mean + delta / n_acc_new;
        let delta2 = x - new_mean;
        let new_m2 = m2 + delta * delta2;
        (new_mean, new_m2, n_acc_new, true)
    } else {
        (mean, m2, n_acc, false)
    }
}

/// Per-pixel online sigma-clip: a MAD-rejected burn-in window (first `k =
/// min(burn_in, n)` samples) seeds a running (mean, M2) Welford state; each
/// remaining sample is tested against that running estimate before being
/// folded in, done in f64 to match numpy's float64 accumulation. Returns
/// (combined, rejected_sample_count).
#[inline]
fn online_sigma_clip_pixel(
    vals: &[f32],
    sigma: f32,
    burn_in: usize,
    all_valid: &[bool],
    scratch: &mut Vec<f32>,
) -> (f32, usize) {
    let n = vals.len();
    let k = burn_in.min(n).max(1);
    let sigma = sigma as f64;

    let (mut mean, mut m2, mut n_acc, mut n_rejected) =
        burnin_seed_pixel(&vals[..k], &all_valid[..k], sigma, scratch);

    for &v in &vals[k..] {
        let (new_mean, new_m2, new_n_acc, accepted) = fold_pixel(mean, m2, n_acc, v as f64, sigma);
        mean = new_mean;
        m2 = new_m2;
        n_acc = new_n_acc;
        if !accepted {
            n_rejected += 1;
        }
    }

    (mean as f32, n_rejected)
}

/// Online (single-pass) sigma-clip combine of an `(N,H,W,C)` float32 stack,
/// for throughput comparison against the batch `sigma_clip_combine` above.
/// Still takes the whole `(N,H,W,C)` array (reuses the same gather-transpose
/// row-parallel driver as the batch kernels), so this measures compute cost
/// only -- it does not exercise the frame-at-a-time streaming I/O the
/// production `--stream` path (`online_sigma_clip_seed_burnin` +
/// `online_sigma_clip_fold_frame` below) actually uses. Exploratory /
/// benchmark-only; not called from the production stacking path.
///
/// Returns `(combined, n_rejected, n_total)` where `n_rejected` is a
/// pixel-*sample* count (summed over all pixels and frames) and `n_total`
/// is the frame count.
#[pyfunction]
#[pyo3(signature = (data, sigma=3.0, burn_in=10))]
fn online_sigma_clip_combine<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    sigma: f32,
    burn_in: usize,
) -> PyResult<(Bound<'py, PyArray3<f32>>, usize, usize)> {
    let arr = data.as_array();
    let shape = arr.shape();
    let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let n_rejected = std::sync::atomic::AtomicUsize::new(0);

    let out = py.allow_threads(|| {
        row_parallel(
            &arr,
            h,
            w,
            c,
            n,
            || (vec![true; n], Vec::<f32>::with_capacity(n)),
            |(all_valid, scratch), vals| {
                let (combined, rejected) =
                    online_sigma_clip_pixel(vals, sigma, burn_in, all_valid, scratch);
                n_rejected.fetch_add(rejected, std::sync::atomic::Ordering::Relaxed);
                combined
            },
        )
    });

    let out_arr = numpy::ndarray::Array3::from_shape_vec((h, w, c), out)
        .expect("shape mismatch building output");
    Ok((
        out_arr.into_pyarray(py),
        n_rejected.load(std::sync::atomic::Ordering::Relaxed),
        n,
    ))
}

/// Seed a running Welford (mean, M2, n_acc) state from a small `(K,H,W,C)`
/// burn-in stack via one MAD-reject pass -- the burn-in half of
/// `online_sigma_clip_pixel`/`burnin_seed_pixel`, run once over the whole
/// buffered window. `K` is expected small and bounded (e.g. 10 -- the
/// streaming product's burn-in size, not the N-samples-per-pixel gather
/// problem `row_parallel` solves for), so this uses a plain per-pixel gather
/// rather than the L2-tiling gather-transpose.
///
/// `coverage` is `(K,H,W)` float32 (>=0.5 = that frame's warp actually
/// covers this pixel), one mask per burn-in frame -- a burn-in window mixes
/// several frames' warps, each with its own out-of-frame zero-fill region
/// (large dithers/shifts easily reach 100+ px on real sessions), so unlike
/// the whole-array kernel above (which has no coverage concept and always
/// treats every sample as valid) this MUST exclude uncovered samples from
/// the median/MAD/mean/M2 computation per `burnin_seed_pixel`'s `valid` gate
/// -- otherwise zero-fill pixels masquerade as real (very dark) samples at
/// every frame's border.
///
/// Returns `(mean, m2, n_acc)` each `(H, W, C)` float64, plus the
/// rejected-sample count (summed over all pixels; uncovered samples count
/// as rejected too, since they never became part of n_acc).
#[pyfunction]
#[pyo3(signature = (burn_stack, coverage, sigma=3.0))]
fn online_sigma_clip_seed_burnin<'py>(
    py: Python<'py>,
    burn_stack: PyReadonlyArray4<'py, f32>,
    coverage: PyReadonlyArray3<'py, f32>,
    sigma: f32,
) -> PyResult<(
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
    usize,
)> {
    let arr = burn_stack.as_array();
    let cov_arr = coverage.as_array();
    let shape = arr.shape();
    let (k, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let sigma_f64 = sigma as f64;
    let row_len = w * c;
    let frame_len = h * row_len;
    let n_rejected = std::sync::atomic::AtomicUsize::new(0);

    let (mean_out, m2_out, nacc_out) = py.allow_threads(|| {
        let mut mean_flat = vec![0f64; h * row_len];
        let mut m2_flat = vec![0f64; h * row_len];
        let mut nacc_flat = vec![0f64; h * row_len];
        let data: Option<&[f32]> = arr.as_slice();

        mean_flat
            .par_chunks_mut(row_len)
            .zip(m2_flat.par_chunks_mut(row_len))
            .zip(nacc_flat.par_chunks_mut(row_len))
            .enumerate()
            .for_each(|(row, ((mean_row, m2_row), nacc_row))| {
                let mut scratch: Vec<f32> = Vec::with_capacity(k);
                let mut burn: Vec<f32> = vec![0f32; k];
                let mut valid: Vec<bool> = vec![true; k];
                let row_base = row * row_len;
                let mut row_rejected = 0usize;
                for col in 0..row_len {
                    let wj = col / c;
                    match data {
                        Some(flat) => {
                            for kk in 0..k {
                                burn[kk] = flat[kk * frame_len + row_base + col];
                            }
                        }
                        None => {
                            let cj = col % c;
                            for kk in 0..k {
                                burn[kk] = arr[[kk, row, wj, cj]];
                            }
                        }
                    }
                    for kk in 0..k {
                        valid[kk] = cov_arr[[kk, row, wj]] >= 0.5;
                    }
                    let (mean, m2, n_acc, n_rej) =
                        burnin_seed_pixel(&burn, &valid, sigma_f64, &mut scratch);
                    mean_row[col] = mean;
                    m2_row[col] = m2;
                    nacc_row[col] = n_acc;
                    row_rejected += n_rej;
                }
                n_rejected.fetch_add(row_rejected, std::sync::atomic::Ordering::Relaxed);
            });
        (mean_flat, m2_flat, nacc_flat)
    });

    let mean_arr = numpy::ndarray::Array3::from_shape_vec((h, w, c), mean_out)
        .expect("shape mismatch building output");
    let m2_arr = numpy::ndarray::Array3::from_shape_vec((h, w, c), m2_out)
        .expect("shape mismatch building output");
    let nacc_arr = numpy::ndarray::Array3::from_shape_vec((h, w, c), nacc_out)
        .expect("shape mismatch building output");
    Ok((
        mean_arr.into_pyarray(py),
        m2_arr.into_pyarray(py),
        nacc_arr.into_pyarray(py),
        n_rejected.load(std::sync::atomic::Ordering::Relaxed),
    ))
}

/// Elementwise (no N-gather) accept-test + Welford update: given the current
/// running `(mean, m2, n_acc)` state (each `(H,W,C)` float64) and ONE new
/// `(H,W,C)` float32 frame plus an `(H,W)` float32 coverage mask (>=0.5 =
/// covered; matches `LiveStacker`'s shift-coverage mask semantics), test
/// each covered pixel against the running estimate and fold it in if
/// accepted. Uncovered pixels are left untouched -- not treated as a
/// sample, not rejected either. Plain rayon `par_chunks_mut` over rows; no
/// gather-transpose, since there's no "N samples per pixel" axis here, just
/// three same-shaped state arrays and one frame.
///
/// Updates `mean`/`m2`/`n_acc` IN PLACE (unlike every other kernel in this
/// file, which is alloc-and-return) -- this runs once per accepted frame in
/// a `--stream` session, so at full resolution the alloc-and-copy-through
/// version this replaced was ~1.7GB of allocator churn per frame (3 fresh
/// (H,W,C) f64 arrays, including a full copy-through of every untouched
/// uncovered pixel) for work that only actually changes the covered
/// fraction of the image. Caller (`online_sigma_clip_fold_frame` in
/// `src/stacking.py`) must pass arrays it holds no other alias to expecting
/// the old value, and must not rely on numpy `ascontiguousarray` silently
/// falling back to a copy -- see the Python wrapper's contiguity check.
/// Returns just the rejected-pixel count for this frame.
#[pyfunction]
#[pyo3(signature = (mean, m2, n_acc, frame, coverage, sigma=3.0))]
fn online_sigma_clip_fold_frame<'py>(
    py: Python<'py>,
    mut mean: PyReadwriteArray3<'py, f64>,
    mut m2: PyReadwriteArray3<'py, f64>,
    mut n_acc: PyReadwriteArray3<'py, f64>,
    frame: PyReadonlyArray3<'py, f32>,
    coverage: PyReadonlyArray2<'py, f32>,
    sigma: f32,
) -> PyResult<usize> {
    let shape = mean.as_array().shape().to_vec();
    let (w, c) = (shape[1], shape[2]);
    let frame_arr = frame.as_array();
    let cov_arr = coverage.as_array();
    let sigma_f64 = sigma as f64;
    let row_len = w * c;

    let mean_slice = mean.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("mean must be C-contiguous for in-place fold")
    })?;
    let m2_slice = m2.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("m2 must be C-contiguous for in-place fold")
    })?;
    let nacc_slice = n_acc.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("n_acc must be C-contiguous for in-place fold")
    })?;
    let n_rejected = std::sync::atomic::AtomicUsize::new(0);

    py.allow_threads(|| {
        mean_slice
            .par_chunks_mut(row_len)
            .zip(m2_slice.par_chunks_mut(row_len))
            .zip(nacc_slice.par_chunks_mut(row_len))
            .enumerate()
            .for_each(|(y, ((mean_row, m2_row), nacc_row))| {
                let mut row_rejected = 0usize;
                for x in 0..w {
                    if cov_arr[[y, x]] < 0.5 {
                        continue;
                    }
                    for ch in 0..c {
                        let col = x * c + ch;
                        let xf = frame_arr[[y, x, ch]] as f64;
                        let (nm, nv, nn, accepted) =
                            fold_pixel(mean_row[col], m2_row[col], nacc_row[col], xf, sigma_f64);
                        mean_row[col] = nm;
                        m2_row[col] = nv;
                        nacc_row[col] = nn;
                        if !accepted {
                            row_rejected += 1;
                        }
                    }
                }
                n_rejected.fetch_add(row_rejected, std::sync::atomic::Ordering::Relaxed);
            });
    });

    Ok(n_rejected.load(std::sync::atomic::Ordering::Relaxed))
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

// ---------------------------------------------------------------------------
// Linear Fit Clipping (PixInsight's algorithm)
// ---------------------------------------------------------------------------
//
// Mirrors src/stacking.py's `_linear_fit_clip_tile` exactly: sort each
// pixel's per-frame stack ascending, fit a line to value-vs-rank, reject
// samples whose residual from that fit exceeds sigma_low/sigma_high times
// the residual scale, refit on survivors, iterate. See that function's
// docstring for why this is more robust to non-Gaussian tails than
// mean/std-based sigma-clip.

#[pyfunction]
#[pyo3(signature = (data, sigma_low, sigma_high, max_iters, weights=None))]
fn linear_fit_clip_combine<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    sigma_low: f64,
    sigma_high: f64,
    max_iters: usize,
    weights: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
    let wv: Option<Vec<f32>> = weights.map(|x| x.as_array().to_vec());
    let wref = wv.as_deref();
    let iters = max_iters.max(1);

    let out = py.allow_threads(|| {
        row_parallel(
            &arr,
            h,
            w,
            c,
            n,
            || {
                (
                    (0..n).collect::<Vec<usize>>(),      // order (reused scratch)
                    vec![0.0f64; n],                      // y_sorted
                    vec![true; n],                        // accepted
                    Vec::<usize>::with_capacity(n),       // newly-rejected scratch
                )
            },
            |(order, y_sorted, accepted, newly_rejected), vals| {
                for i in 0..n {
                    order[i] = i;
                }
                order.sort_unstable_by(|&a, &b| {
                    vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal)
                });
                for r in 0..n {
                    y_sorted[r] = vals[order[r]] as f64;
                    accepted[r] = !y_sorted[r].is_nan();
                }

                for _ in 0..iters {
                    let mut n_acc = 0usize;
                    let mut sum_x = 0f64;
                    let mut sum_y = 0f64;
                    for r in 0..n {
                        if accepted[r] {
                            sum_x += r as f64;
                            sum_y += y_sorted[r];
                            n_acc += 1;
                        }
                    }
                    if n_acc < 2 {
                        break;
                    }
                    let x_mean = sum_x / n_acc as f64;
                    let y_mean = sum_y / n_acc as f64;
                    let mut sxx = 0f64;
                    let mut sxy = 0f64;
                    for r in 0..n {
                        if accepted[r] {
                            let dx = r as f64 - x_mean;
                            sxx += dx * dx;
                            sxy += dx * (y_sorted[r] - y_mean);
                        }
                    }
                    if sxx <= 1e-12 {
                        sxx = 1.0;
                    }
                    let slope = sxy / sxx;
                    let intercept = y_mean - slope * x_mean;

                    let mut ss = 0f64;
                    for r in 0..n {
                        if accepted[r] {
                            let e = y_sorted[r] - (intercept + slope * r as f64);
                            ss += e * e;
                        }
                    }
                    let sigma = ((ss / (n_acc as f64 - 1.0).max(1.0)).sqrt()).max(1e-6);

                    newly_rejected.clear();
                    for r in 0..n {
                        if accepted[r] {
                            let e = y_sorted[r] - (intercept + slope * r as f64);
                            if e < -sigma_low * sigma || e > sigma_high * sigma {
                                newly_rejected.push(r);
                            }
                        }
                    }
                    if newly_rejected.is_empty() {
                        break;
                    }
                    if n_acc - newly_rejected.len() < 2 {
                        // Applying this iteration's rejections would leave
                        // fewer than 2 survivors -- freeze this pixel's
                        // accepted set instead (matches the numpy reference's
                        // per-pixel `apply` gate).
                        break;
                    }
                    for &r in newly_rejected.iter() {
                        accepted[r] = false;
                    }
                }

                let mut acc = 0f64;
                let mut wsum = 0f64;
                for r in 0..n {
                    if accepted[r] {
                        let orig_idx = order[r];
                        let wt = wref.map(|wv| wv[orig_idx] as f64).unwrap_or(1.0);
                        acc += y_sorted[r] * wt;
                        wsum += wt;
                    }
                }
                if wsum <= 0.0 {
                    0.0
                } else {
                    (acc / wsum) as f32
                }
            },
        )
    });
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out).unwrap().into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Inverse-variance-weighted combine ("Bayesian" / Gauss-Markov optimal mean)
// ---------------------------------------------------------------------------
//
// Mirrors src/stacking.py's `_ivw_tile` exactly. No iteration/sorting needed
// (unlike every other kernel in this file) -- still routed through
// `row_parallel` for the gather-transpose + parallelism, since the per-pixel
// weight is data-dependent (varies with each frame's own pixel value when
// `gain` is supplied) and so isn't reducible to a single static-weight numpy
// broadcast the way a per-frame-constant weighted mean would be.

#[pyfunction]
#[pyo3(signature = (data, noise, sky=None, gain=None, weights=None))]
fn ivw_combine<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    noise: PyReadonlyArray1<'py, f32>,
    sky: Option<PyReadonlyArray1<'py, f32>>,
    gain: Option<f64>,
    weights: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (n, h, w, c) = (s[0], s[1], s[2], s[3]);

    let noise2: Vec<f64> = noise.as_array().iter().map(|&v| (v as f64) * (v as f64)).collect();
    let sky_v: Option<Vec<f64>> = sky.map(|x| x.as_array().iter().map(|&v| v as f64).collect());
    let wv: Option<Vec<f32>> = weights.map(|x| x.as_array().to_vec());
    let wref = wv.as_deref();
    let shot_model = match (&sky_v, gain) {
        (Some(sk), Some(g)) if g > 0.0 => Some((sk, g)),
        _ => None,
    };

    let out = py.allow_threads(|| {
        row_parallel(&arr, h, w, c, n, || (), |_, vals| {
            let mut acc = 0f64;
            let mut wsum = 0f64;
            for k in 0..n {
                let var = match shot_model {
                    Some((sk, g)) => {
                        let shot = ((vals[k] as f64) - sk[k]).max(0.0) / g;
                        noise2[k] + shot
                    }
                    None => noise2[k],
                };
                let var = var.max(1e-12);
                let mut wt = 1.0 / var;
                if let Some(qw) = wref {
                    wt *= qw[k] as f64;
                }
                acc += wt * vals[k] as f64;
                wsum += wt;
            }
            if wsum <= 0.0 {
                0.0
            } else {
                (acc / wsum) as f32
            }
        })
    });
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out).unwrap().into_pyarray(py))
}

// ---------------------------------------------------------------------------
// 2D separable wavelet transform (bior1.3 / db4)
// ---------------------------------------------------------------------------
//
// Mirrors src/wavelet.py's _dwt_1d/_idwt_1d + _dwt2/_idwt2 exactly, but as a
// direct closed-form tap sum per output sample instead of
// pad-then-np.apply_along_axis-then-np.convolve (which loops in pure Python
// over every row/column -- see that module's docstring for why the exact
// convolve-vs-correlate orientation of each kernel matters and how it was
// determined). No padded array is ever materialised: `wavelet_symmetric_idx`
// computes the symmetric ("whole-point", edge-duplicating -- numpy's
// mode='symmetric') boundary index directly.

#[inline]
fn wavelet_symmetric_idx(i: isize, n: usize) -> usize {
    let n_i = n as isize;
    let period = 2 * n_i;
    let mut m = i.rem_euclid(period);
    if m >= n_i {
        m = period - 1 - m;
    }
    m as usize
}

/// Forward 1D DWT of one length-`n` line, accessed via `get`. Exact port of
/// `_dwt_1d`: `cA[j] = sum_k lo[k] * x[symmetric_idx(offset+2j-k, n)]`, `cD`
/// analogous with `hi` -- the direct closed form of
/// `np.convolve(padded, kernel, mode='valid')[offset::2]` with no padding
/// materialised (derivation in the Rust module's own commit notes: convolve
/// flips one operand, so `valid[i] = sum_k kernel[k] * padded[i+flen-1-k]`,
/// and substituting the padding offset collapses to this form).
fn dwt_1d_line(
    get: impl Fn(usize) -> f64,
    n: usize,
    lo: &[f64],
    hi: &[f64],
    offset: usize,
    out_len: usize,
    ca_out: &mut [f64],
    cd_out: &mut [f64],
) {
    let flen = lo.len();
    for j in 0..out_len {
        let base = offset as isize + 2 * j as isize;
        let mut a = 0.0f64;
        let mut d = 0.0f64;
        for k in 0..flen {
            let v = get(wavelet_symmetric_idx(base - k as isize, n));
            a += lo[k] * v;
            d += hi[k] * v;
        }
        ca_out[j] = a;
        cd_out[j] = d;
    }
}

/// Inverse 1D DWT of one line. Exact port of `_idwt_1d`: rather than
/// zero-stuffing `cA`/`cD` to length `2n` and running a full convolution,
/// only the even-position (nonzero) upsampled taps are summed directly --
/// `out[oi] = sum_k rec_lo[k]*cA[j] + rec_hi[k]*cD[j]` where
/// `j = (idwt_offset+oi-k)/2`, only when that quantity is a nonnegative
/// even integer within range.
fn idwt_1d_line(
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
fn dwt2_native<'py>(
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
    let img64: Vec<f64> = arr.iter().copied().collect();

    let (c_a, c_h, c_v, c_d) = py.allow_threads(|| {
        // Pass 1: axis=0 (per-column) -> La, Lh, each (out_h, w).
        let cols: Vec<(Vec<f64>, Vec<f64>)> = (0..w)
            .into_par_iter()
            .map(|c| {
                let get = |i: usize| img64[i * w + c];
                let mut ca_col = vec![0.0f64; out_h];
                let mut cd_col = vec![0.0f64; out_h];
                dwt_1d_line(get, h, &lo, &hi, offset, out_h, &mut ca_col, &mut cd_col);
                (ca_col, cd_col)
            })
            .collect();
        let mut la = vec![0.0f64; out_h * w];
        let mut lh = vec![0.0f64; out_h * w];
        for (c, (ca_col, cd_col)) in cols.into_iter().enumerate() {
            for j in 0..out_h {
                la[j * w + c] = ca_col[j];
                lh[j * w + c] = cd_col[j];
            }
        }

        // Pass 2: axis=1 (per-row) on La -> cA, cV; on Lh -> cH, cD.
        let mut c_a = vec![0.0f64; out_h * out_w];
        let mut c_v = vec![0.0f64; out_h * out_w];
        c_a.par_chunks_mut(out_w)
            .zip(c_v.par_chunks_mut(out_w))
            .enumerate()
            .for_each(|(r, (ca_row, cv_row))| {
                let get = |i: usize| la[r * w + i];
                dwt_1d_line(get, w, &lo, &hi, offset, out_w, ca_row, cv_row);
            });
        let mut c_h = vec![0.0f64; out_h * out_w];
        let mut c_d = vec![0.0f64; out_h * out_w];
        c_h.par_chunks_mut(out_w)
            .zip(c_d.par_chunks_mut(out_w))
            .enumerate()
            .for_each(|(r, (ch_row, cd_row))| {
                let get = |i: usize| lh[r * w + i];
                dwt_1d_line(get, w, &lo, &hi, offset, out_w, ch_row, cd_row);
            });
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
fn idwt2_native<'py>(
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

    let ca64: Vec<f64> = ca_arr.iter().copied().collect();
    let ch64: Vec<f64> = ch_arr.iter().copied().collect();
    let cv64: Vec<f64> = cv_arr.iter().copied().collect();
    let cd64: Vec<f64> = cd_arr.iter().copied().collect();

    let out = py.allow_threads(|| {
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

        // Pass 2: axis=0 (per-column) reconstruction on (La, Lh) -> final (out_h, out_w).
        let cols: Vec<Vec<f64>> = (0..out_w)
            .into_par_iter()
            .map(|c| {
                let la_col: Vec<f64> = (0..in_h).map(|r| la[r * out_w + c]).collect();
                let lh_col: Vec<f64> = (0..in_h).map(|r| lh[r * out_w + c]).collect();
                let mut out_col = vec![0.0f64; out_h];
                idwt_1d_line(&la_col, &lh_col, &rlo, &rhi, idwt_offset, out_h, &mut out_col);
                out_col
            })
            .collect();
        let mut result = vec![0.0f64; out_h * out_w];
        for (c, col) in cols.into_iter().enumerate() {
            for r in 0..out_h {
                result[r * out_w + c] = col[r];
            }
        }
        result
    });

    Ok(numpy::ndarray::Array2::from_shape_vec((out_h, out_w), out)
        .unwrap()
        .into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Iterative sigma-clipped median (src/debayer.py's _sigma_clipped_median)
// ---------------------------------------------------------------------------
//
// Used by green_equalize (G1/G2 sub-channel balance) and _equalize_bayer_grid
// (per-Bayer-position sky level) -- both call this on quarter-resolution
// strided views (every other row/col), several times per frame, and both
// showed up as a real chunk of Quality+Load's sequential-profile cost.
// f64 throughout (this returns a single scalar, no coefficient-chaining
// precision concern like the wavelet kernel), matching numpy's internal
// promotion for median/std of a float32 input.

#[inline]
fn median_f64_scratch(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    let (_, &mut m, _) =
        v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        m
    } else {
        let mut lo = f64::NEG_INFINITY;
        for &x in v[..mid].iter() {
            if x > lo {
                lo = x;
            }
        }
        0.5 * (lo + m)
    }
}

#[inline]
fn std_pop_f64(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let m: f64 = v.iter().sum::<f64>() / n as f64;
    (v.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / n as f64).sqrt()
}

#[pyfunction]
fn sigma_clipped_median_native(data: PyReadonlyArray1<'_, f32>, sigma: f64, iters: usize) -> f64 {
    let orig: Vec<f64> = data.as_array().iter().map(|&v| v as f64).collect();
    let mut x = orig.clone();
    // Reused across iterations instead of `x.clone()`-ing fresh every time:
    // median_f64_scratch's quickselect is destructive, so it still needs its
    // own copy of `x` each iteration, but reusing one buffer (clear + refill)
    // avoids a malloc/free round-trip per iteration on top of the copy that's
    // unavoidable either way -- `x` itself must stay unpartitioned for the
    // std-dev and sigma-clip retain() below.
    let mut scratch: Vec<f64> = Vec::with_capacity(x.len());

    for _ in 0..iters {
        if x.is_empty() {
            break;
        }
        scratch.clear();
        scratch.extend_from_slice(&x);
        let med = median_f64_scratch(&mut scratch);
        let std = std_pop_f64(&x);
        if std < 1e-12 {
            break;
        }
        let thresh = sigma * std;
        x.retain(|&v| (v - med).abs() < thresh);
    }

    if !x.is_empty() {
        median_f64_scratch(&mut x)
    } else {
        median_f64_scratch(&mut orig.clone())
    }
}

// ---------------------------------------------------------------------------
// Sparse hot-pixel box-mean replacement (src/debayer.py's
// _fix_hot_rgb_impl/_fix_hot_mono_impl replacement step)
// ---------------------------------------------------------------------------
//
// The numpy reference computes `scipy.ndimage.uniform_filter(ch, size=3)`
// over the *entire* channel and then keeps only the masked positions via
// `np.where` -- wasteful when hot pixels are a small fraction of the frame
// (the normal case), since the box mean is computed everywhere but used
// almost nowhere. This computes the 3x3 box mean only at masked positions.
// Boundary convention is scipy.ndimage's default `mode='reflect'` for
// uniform_filter/median_filter, which (confusingly) is the *duplicate-edge*
// reflection -- the same convention as `wavelet_symmetric_idx` above (numpy
// `pad(mode='symmetric')`), reused here rather than redefined.

#[pyfunction]
fn hot_pixel_box_replace_native<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f32>,
    mask: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (h, w, c) = (s[0], s[1], s[2]);
    let data_flat: Vec<f32> = arr.iter().copied().collect();
    let mask_flat: Vec<u8> = mask.as_array().iter().copied().collect();

    let out = py.allow_threads(|| {
        let mut result = data_flat.clone();
        result.par_chunks_mut(w * c).enumerate().for_each(|(y, row_out)| {
            for x in 0..w {
                if mask_flat[y * w + x] == 0 {
                    continue;
                }
                for ch in 0..c {
                    let mut acc = 0.0f64;
                    for dy in -1isize..=1 {
                        let yy = wavelet_symmetric_idx(y as isize + dy, h);
                        for dx in -1isize..=1 {
                            let xx = wavelet_symmetric_idx(x as isize + dx, w);
                            acc += data_flat[(yy * w + xx) * c + ch] as f64;
                        }
                    }
                    row_out[x * c + ch] = (acc / 9.0) as f32;
                }
            }
        });
        result
    });

    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out)
        .unwrap()
        .into_pyarray(py))
}

// ---------------------------------------------------------------------------
// BM3D pure-scipy-DCT fallback: block-matching + Step1/Step2 core loop
// ---------------------------------------------------------------------------
//
// Mirrors src/denoising.py's bm3d_denoise pure-scipy-DCT path (the fallback
// used when the `bm3d` pip package -- non-commercial-licensed, see
// THIRD_PARTY_NOTICES.md -- isn't installed) exactly: same reference-block
// grid, same DCT-domain block-matching distance/threshold, same Step1
// hard-threshold + Step2 Wiener passes, same weighted overlap-add. The
// outer per-reference-block loop (ny*nx iterations, each previously calling
// scipy.fft.dctn/idctn on a tiny array) is where the real cost was: Python
// loop + repeated small-FFT call overhead, not the underlying arithmetic,
// which is trivial at block_size~8/group_size~8. Two of the Python
// reference's local variables (`Yp`, `pilot_dcts`) are computed but never
// read again anywhere in that function -- confirmed dead, not ported here.
//
// DCT-II/DCT-III here are direct O(n^2) sums (orthonormal, `norm='ortho'`
// convention, self-inverse pair) -- fine at block_size/group_size's small
// scale, and simpler to get bit-exact right than an FFT-based DCT. Validated
// directly against scipy.fft.dctn/idctn in tests/test_native.py before
// trusting it inside the full block-matching loop.

#[inline]
fn dct1d_ortho(x: &[f64], out: &mut [f64]) {
    let n = x.len();
    let nf = n as f64;
    let s0 = (1.0 / nf).sqrt();
    let sk = (2.0 / nf).sqrt();
    for k in 0..n {
        let scale = if k == 0 { s0 } else { sk };
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += x[i] * (std::f64::consts::PI / nf * (i as f64 + 0.5) * k as f64).cos();
        }
        out[k] = scale * acc;
    }
}

#[inline]
fn idct1d_ortho(x: &[f64], out: &mut [f64]) {
    let n = x.len();
    let nf = n as f64;
    let s0 = (1.0 / nf).sqrt();
    let sk = (2.0 / nf).sqrt();
    for i in 0..n {
        let mut acc = 0.0f64;
        for k in 0..n {
            let scale = if k == 0 { s0 } else { sk };
            acc += scale * x[k] * (std::f64::consts::PI / nf * (i as f64 + 0.5) * k as f64).cos();
        }
        out[i] = acc;
    }
}

/// Separable 2D DCT-II (ortho), in place semantics via fresh output: `data`
/// is `rows x cols` row-major.
fn dct2d_ortho(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut tmp = vec![0.0f64; rows * cols];
    // Pass 1: along columns (axis=1, per-row).
    for r in 0..rows {
        dct1d_ortho(&data[r * cols..(r + 1) * cols], &mut tmp[r * cols..(r + 1) * cols]);
    }
    // Pass 2: along rows (axis=0, per-column).
    let mut out = vec![0.0f64; rows * cols];
    let mut col_in = vec![0.0f64; rows];
    let mut col_out = vec![0.0f64; rows];
    for c in 0..cols {
        for r in 0..rows {
            col_in[r] = tmp[r * cols + c];
        }
        dct1d_ortho(&col_in, &mut col_out);
        for r in 0..rows {
            out[r * cols + c] = col_out[r];
        }
    }
    out
}

fn idct2d_ortho(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut tmp = vec![0.0f64; rows * cols];
    let mut col_in = vec![0.0f64; rows];
    let mut col_out = vec![0.0f64; rows];
    for c in 0..cols {
        for r in 0..rows {
            col_in[r] = data[r * cols + c];
        }
        idct1d_ortho(&col_in, &mut col_out);
        for r in 0..rows {
            tmp[r * cols + c] = col_out[r];
        }
    }
    let mut out = vec![0.0f64; rows * cols];
    for r in 0..rows {
        idct1d_ortho(&tmp[r * cols..(r + 1) * cols], &mut out[r * cols..(r + 1) * cols]);
    }
    out
}

/// 3D separable DCT-II (ortho) over a `(g, bs, bs)` group -- axis 0 (group),
/// then the 2D spatial transform per group-member. Mirrors
/// `scipy.fft.dctn(group, axes=(0,1,2), norm='ortho')`: order among
/// independent separable axes doesn't affect the result.
fn dct3d_ortho(data: &[f64], g: usize, bs: usize) -> Vec<f64> {
    let plane = bs * bs;
    // Axis 0 (group axis): for each spatial position, DCT across the group.
    let mut after_axis0 = vec![0.0f64; g * plane];
    let mut col_in = vec![0.0f64; g];
    let mut col_out = vec![0.0f64; g];
    for p in 0..plane {
        for gi in 0..g {
            col_in[gi] = data[gi * plane + p];
        }
        dct1d_ortho(&col_in, &mut col_out);
        for gi in 0..g {
            after_axis0[gi * plane + p] = col_out[gi];
        }
    }
    // Spatial 2D DCT per group member.
    let mut out = vec![0.0f64; g * plane];
    for gi in 0..g {
        let plane_dct = dct2d_ortho(&after_axis0[gi * plane..(gi + 1) * plane], bs, bs);
        out[gi * plane..(gi + 1) * plane].copy_from_slice(&plane_dct);
    }
    out
}

fn idct3d_ortho(data: &[f64], g: usize, bs: usize) -> Vec<f64> {
    let plane = bs * bs;
    let mut after_spatial = vec![0.0f64; g * plane];
    for gi in 0..g {
        let plane_idct = idct2d_ortho(&data[gi * plane..(gi + 1) * plane], bs, bs);
        after_spatial[gi * plane..(gi + 1) * plane].copy_from_slice(&plane_idct);
    }
    let mut out = vec![0.0f64; g * plane];
    let mut col_in = vec![0.0f64; g];
    let mut col_out = vec![0.0f64; g];
    for p in 0..plane {
        for gi in 0..g {
            col_in[gi] = after_spatial[gi * plane + p];
        }
        idct1d_ortho(&col_in, &mut col_out);
        for gi in 0..g {
            out[gi * plane + p] = col_out[gi];
        }
    }
    out
}

#[pyfunction]
fn dct2_ortho_native<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (rows, cols) = (s[0], s[1]);
    let flat: Vec<f64> = arr.iter().copied().collect();
    let out = dct2d_ortho(&flat, rows, cols);
    Ok(numpy::ndarray::Array2::from_shape_vec((rows, cols), out)
        .unwrap()
        .into_pyarray(py))
}

#[pyfunction]
fn idct2_ortho_native<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (rows, cols) = (s[0], s[1]);
    let flat: Vec<f64> = arr.iter().copied().collect();
    let out = idct2d_ortho(&flat, rows, cols);
    Ok(numpy::ndarray::Array2::from_shape_vec((rows, cols), out)
        .unwrap()
        .into_pyarray(py))
}

/// Find similar blocks to reference `(iy, ix)` within the search window,
/// mirroring the Python reference's block-matching exactly, including its
/// literal "always include self" fallback: when nothing beats
/// `dist_threshold`, it keeps whatever candidate is *first in raster order
/// within the (possibly window-clipped) search box* -- not necessarily the
/// true reference block itself. Replicated as-is for bit-for-bit parity,
/// not "fixed".
fn bm3d_similar_blocks(
    iy: usize,
    ix: usize,
    ny: usize,
    nx: usize,
    max_dy: usize,
    max_dx: usize,
    all_dcts_flat: &[f64],
    plane: usize,
    dist_threshold: f64,
    group_size: usize,
) -> Vec<usize> {
    let iy_lo = iy.saturating_sub(max_dy);
    let iy_hi = (iy + max_dy).min(ny - 1);
    let ix_lo = ix.saturating_sub(max_dx);
    let ix_hi = (ix + max_dx).min(nx - 1);

    let ref_dct = &all_dcts_flat[(iy * nx + ix) * plane..(iy * nx + ix + 1) * plane];

    let mut cand_idx: Vec<usize> = Vec::with_capacity((iy_hi - iy_lo + 1) * (ix_hi - ix_lo + 1));
    for cy in iy_lo..=iy_hi {
        for cx in ix_lo..=ix_hi {
            cand_idx.push(cy * nx + cx);
        }
    }

    let mut scored: Vec<(f64, usize)> = Vec::with_capacity(cand_idx.len());
    for &idx in &cand_idx {
        let cand_dct = &all_dcts_flat[idx * plane..(idx + 1) * plane];
        let mut d2 = 0.0f64;
        for p in 0..plane {
            let diff = cand_dct[p] - ref_dct[p];
            d2 += diff * diff;
        }
        if d2 < dist_threshold {
            scored.push((d2, idx));
        }
    }

    if scored.is_empty() {
        return vec![cand_idx[0]];
    }
    if scored.len() > group_size {
        // Stable sort by distance ascending, matching np.argsort's role
        // here (ties broken by original -- i.e. raster -- order, same as
        // numpy's default stable-enough behaviour for this use since
        // `scored` was built in raster order).
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(group_size);
    }
    scored.into_iter().map(|(_, idx)| idx).collect()
}

#[pyfunction]
fn bm3d_denoise_native<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    block_size: usize,
    stride: usize,
    search_window: usize,
    group_size: usize,
    sigma_psd: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = y.as_array();
    let s = arr.shape();
    let (h, w) = (s[0], s[1]);
    let y_flat: Vec<f64> = arr.iter().copied().collect();
    let bs = block_size;
    let plane = bs * bs;

    let out = py.allow_threads(|| {
        let ref_ys: Vec<usize> = (0..=(h - bs)).step_by(stride).collect();
        let ref_xs: Vec<usize> = (0..=(w - bs)).step_by(stride).collect();
        let ny = ref_ys.len();
        let nx = ref_xs.len();

        let get_block = |yr: usize, xr: usize| -> Vec<f64> {
            let mut b = vec![0.0f64; plane];
            for r in 0..bs {
                for c in 0..bs {
                    b[r * bs + c] = y_flat[(yr + r) * w + (xr + c)];
                }
            }
            b
        };

        let all_ref: Vec<Vec<f64>> = ref_ys
            .iter()
            .flat_map(|&yr| ref_xs.iter().map(move |&xr| (yr, xr)))
            .map(|(yr, xr)| get_block(yr, xr))
            .collect();
        let all_dcts_flat: Vec<f64> = all_ref
            .iter()
            .flat_map(|blk| dct2d_ortho(blk, bs, bs))
            .collect();

        let ht_threshold = sigma_psd * (2.0 * (plane as f64).ln()).sqrt();
        let dist_threshold = (ht_threshold * 0.5).powi(2) * plane as f64;
        let max_dy = (search_window / stride).max(1);
        let max_dx = max_dy;

        let mut acc1 = vec![0.0f64; h * w];
        let mut wgt1 = vec![0.0f64; h * w];
        let mut acc2 = vec![0.0f64; h * w];
        let mut wgt2 = vec![0.0f64; h * w];

        // --- Step 1: hard-thresholding ---
        for iy in 0..ny {
            for ix in 0..nx {
                let similar = bm3d_similar_blocks(
                    iy, ix, ny, nx, max_dy, max_dx, &all_dcts_flat, plane, dist_threshold, group_size,
                );
                let n_g = similar.len();
                let group: Vec<f64> = similar.iter().flat_map(|&idx| all_ref[idx].clone()).collect();
                let spec3 = dct3d_ortho(&group, n_g, bs);
                let mut ht = vec![0.0f64; spec3.len()];
                let mut n_nz = 0usize;
                for (i, &v) in spec3.iter().enumerate() {
                    if v.abs() >= ht_threshold {
                        ht[i] = v;
                        n_nz += 1;
                    }
                }
                let n_nz = n_nz.max(1);
                let w1 = 1.0 / n_nz as f64;
                let denoised1 = idct3d_ortho(&ht, n_g, bs);

                for (k, &idx) in similar.iter().enumerate() {
                    let gi = idx / nx;
                    let gj = idx % nx;
                    let (yr2, xr2) = (ref_ys[gi], ref_xs[gj]);
                    for r in 0..bs {
                        for c in 0..bs {
                            let p = (yr2 + r) * w + (xr2 + c);
                            acc1[p] += w1 * denoised1[k * plane + r * bs + c];
                            wgt1[p] += w1;
                        }
                    }
                }
            }
        }

        let pilot: Vec<f64> = (0..h * w)
            .map(|i| if wgt1[i] > 0.0 { acc1[i] / wgt1[i] } else { y_flat[i] })
            .collect();

        // --- Step 2: Wiener filter using the Step-1 pilot estimate ---
        for iy in 0..ny {
            for ix in 0..nx {
                let similar = bm3d_similar_blocks(
                    iy, ix, ny, nx, max_dy, max_dx, &all_dcts_flat, plane, dist_threshold, group_size,
                );
                let n_g = similar.len();
                let noisy_group: Vec<f64> = similar.iter().flat_map(|&idx| all_ref[idx].clone()).collect();
                let pilot_group: Vec<f64> = similar
                    .iter()
                    .flat_map(|&idx| {
                        let gi = idx / nx;
                        let gj = idx % nx;
                        let (yr2, xr2) = (ref_ys[gi], ref_xs[gj]);
                        (0..bs).flat_map(move |r| (0..bs).map(move |c| (yr2 + r, xr2 + c)))
                            .map(|(rr, cc)| pilot[rr * w + cc])
                            .collect::<Vec<f64>>()
                    })
                    .collect();

                let spec_noisy = dct3d_ortho(&noisy_group, n_g, bs);
                let spec_pilot = dct3d_ortho(&pilot_group, n_g, bs);
                let mut spec_filt = vec![0.0f64; spec_noisy.len()];
                let mut wiener_sq_sum = 0.0f64;
                for i in 0..spec_noisy.len() {
                    let pilot_sq = spec_pilot[i] * spec_pilot[i];
                    let wiener = pilot_sq / (pilot_sq + sigma_psd * sigma_psd + 1e-30);
                    spec_filt[i] = wiener * spec_noisy[i];
                    wiener_sq_sum += wiener * wiener;
                }
                let wiener_w = wiener_sq_sum / (n_g.max(1) as f64);
                let w2 = 1.0 / wiener_w.max(1e-12);
                let denoised2 = idct3d_ortho(&spec_filt, n_g, bs);

                for (k, &idx) in similar.iter().enumerate() {
                    let gi = idx / nx;
                    let gj = idx % nx;
                    let (yr2, xr2) = (ref_ys[gi], ref_xs[gj]);
                    for r in 0..bs {
                        for c in 0..bs {
                            let p = (yr2 + r) * w + (xr2 + c);
                            acc2[p] += w2 * denoised2[k * plane + r * bs + c];
                            wgt2[p] += w2;
                        }
                    }
                }
            }
        }

        (0..h * w)
            .map(|i| if wgt2[i] > 0.0 { acc2[i] / wgt2[i] } else { pilot[i] })
            .collect::<Vec<f64>>()
    });

    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), out)
        .unwrap()
        .into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Blind (unknown-rotation) rigid star-pattern match hypothesis search
// ---------------------------------------------------------------------------
//
// Mirrors src/blind_match.py's match_rigid_unknown_rotation exactly (same
// iteration order over src pairs, same sorted-distance binary search into
// dst pairs, same early-exit conditions) -- see that module's docstring for
// the algorithm rationale (distance-matched pair-of-pairs hypotheses,
// RANSAC-style consensus scoring). n/m are capped small (max_stars, default
// 40) by the caller, so nearest-neighbour queries are brute-force here
// rather than a KDTree -- the cost this kernel actually removes is the
// per-hypothesis Python-level loop + scipy KDTree.query call overhead (up
// to _MAX_HYPOTHESES of them), not the underlying O(n*m) query cost, which
// is trivial at this scale either way. Only searches for the best
// hypothesis and returns its (R, t, inlier_count) -- the final
// all-inliers Umeyama refit stays in Python (src/affine_fit.py's
// _umeyama_2d), called once, not per-hypothesis.

#[pyfunction]
fn blind_match_hypotheses<'py>(
    py: Python<'py>,
    src: numpy::PyReadonlyArray2<'py, f64>,
    dst: numpy::PyReadonlyArray2<'py, f64>,
    pixel_tol: f64,
    dist_rel_tol: f64,
    min_sep: f64,
    target_inliers: usize,
    max_hypotheses: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, usize)> {
    let src_arr = src.as_array();
    let dst_arr = dst.as_array();
    let n = src_arr.shape()[0];
    let m = dst_arr.shape()[0];
    let src_pts: Vec<(f64, f64)> = (0..n).map(|i| (src_arr[[i, 0]], src_arr[[i, 1]])).collect();
    let dst_pts: Vec<(f64, f64)> = (0..m).map(|i| (dst_arr[[i, 0]], dst_arr[[i, 1]])).collect();

    let (best_r, best_t, best_inliers) = py.allow_threads(|| {
        // Sorted dst pairwise distances (ascending), with original indices --
        // mirrors `order = np.argsort(dst_d)` in the Python reference.
        let mut dst_pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(m * m / 2);
        for i in 0..m {
            for j in (i + 1)..m {
                let dx = dst_pts[i].0 - dst_pts[j].0;
                let dy = dst_pts[i].1 - dst_pts[j].1;
                dst_pairs.push(((dx * dx + dy * dy).sqrt(), i, j));
            }
        }
        dst_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let dst_d_sorted: Vec<f64> = dst_pairs.iter().map(|p| p.0).collect();

        let mut best_inliers = 0usize;
        let mut best_r = [[1.0f64, 0.0], [0.0, 1.0]];
        let mut best_t = [0.0f64, 0.0];
        let mut tested = 0usize;

        'outer: for i in 0..n {
            for j in (i + 1)..n {
                if tested >= max_hypotheses {
                    break 'outer;
                }
                let (pix, piy) = src_pts[i];
                let (pjx, pjy) = src_pts[j];
                let dx = pjx - pix;
                let dy = pjy - piy;
                let d = (dx * dx + dy * dy).sqrt();
                if d < min_sep {
                    continue;
                }
                let tol = (pixel_tol * 2.0).max(d * dist_rel_tol);
                let lo = dst_d_sorted.partition_point(|&x| x < d - tol);
                let hi = dst_d_sorted.partition_point(|&x| x <= d + tol);
                if lo >= hi {
                    continue;
                }
                let v_src_angle = dy.atan2(dx);

                for cand in lo..hi {
                    let (_, ci, cj) = dst_pairs[cand];
                    for &(a_idx, b_idx) in &[(ci, cj), (cj, ci)] {
                        tested += 1;
                        let (qax, qay) = dst_pts[a_idx];
                        let (qbx, qby) = dst_pts[b_idx];
                        let vdx = qbx - qax;
                        let vdy = qby - qay;
                        // NOTE: the Python reference (_rigid_from_two_points)
                        // only guards the *src* pair's length (already
                        // enforced by min_sep above, unreachable in
                        // practice) -- it has no dst-side degeneracy check.
                        // A near-zero v_dst just yields atan2(0,0)=0 (same
                        // in Rust as numpy) and a hypothesis that will
                        // naturally lose on inlier count. Replicated as-is,
                        // not "fixed", to stay bit-for-bit faithful.
                        let theta = vdy.atan2(vdx) - v_src_angle;
                        let (c, s) = (theta.cos(), theta.sin());
                        let r = [[c, -s], [s, c]];
                        let t = [qax - (r[0][0] * pix + r[0][1] * piy),
                                qay - (r[1][0] * pix + r[1][1] * piy)];

                        let mut n_in = 0usize;
                        for &(sx, sy) in &src_pts {
                            let px = r[0][0] * sx + r[0][1] * sy + t[0];
                            let py = r[1][0] * sx + r[1][1] * sy + t[1];
                            let mut best_d2 = f64::INFINITY;
                            for &(dxp, dyp) in &dst_pts {
                                let ddx = px - dxp;
                                let ddy = py - dyp;
                                let d2 = ddx * ddx + ddy * ddy;
                                if d2 < best_d2 {
                                    best_d2 = d2;
                                }
                            }
                            if best_d2 < pixel_tol * pixel_tol {
                                n_in += 1;
                            }
                        }
                        if n_in > best_inliers {
                            best_inliers = n_in;
                            best_r = r;
                            best_t = t;
                            if best_inliers >= target_inliers {
                                break;
                            }
                        }
                        if tested >= max_hypotheses {
                            break;
                        }
                    }
                    if tested >= max_hypotheses || best_inliers >= target_inliers {
                        break;
                    }
                }
                if best_inliers >= target_inliers {
                    break 'outer;
                }
            }
        }
        (best_r, best_t, best_inliers)
    });

    let r_flat: Vec<f64> = best_r.iter().flatten().copied().collect();
    let r_arr = numpy::ndarray::Array2::from_shape_vec((2, 2), r_flat).unwrap();
    let t_arr = numpy::ndarray::Array1::from_vec(best_t.to_vec());
    Ok((r_arr.into_pyarray(py), t_arr.into_pyarray(py), best_inliers))
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

/// Affine warp with an arbitrary precomputed tap-weight table instead of the
/// fixed Lanczos-3 formula (`warp_affine_lanczos3` above, left untouched) --
/// lets drizzle resample with a frame's own estimated PSF as a matched
/// filter (`--drizzle-kernel psf`). The table is not assumed separable (a
/// general Moffat/Gaussian PSF isn't X/Y-separable like Lanczos), so there
/// is no fast-path branch: every output pixel gathers the full
/// `(2*halo+1)^2` neighbourhood. `table` is a flat
/// `phases*phases*taps*taps` array (`taps = 2*halo+1`), row-major over
/// `[phase_y, phase_x, tap_y, tap_x]`, built in Python by resampling the
/// discrete PSF kernel at `phases` subpixel offsets per axis (each phase's
/// taps already normalised to sum to 1). Same boundary convention as
/// `warp_affine_lanczos3`: out-of-range taps are skipped (not renormalised);
/// `cval` is only used where *no* tap in the window was in-bounds.
#[pyfunction]
#[pyo3(signature = (data, mat, off, out_h, out_w, table, halo, phases, cval=0.0))]
fn warp_affine_kernel_table<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f32>,
    mat: [f64; 4],
    off: [f64; 2],
    out_h: usize,
    out_w: usize,
    table: PyReadonlyArray1<'py, f64>,
    halo: usize,
    phases: usize,
    cval: f32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (h, w, c) = (s[0], s[1], s[2]);
    let (m00, m01, m10, m11) = (mat[0], mat[1], mat[2], mat[3]);
    let (o0, o1) = (off[0], off[1]);
    if phases == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "phases must be > 0",
        ));
    }
    let taps = 2 * halo + 1;
    let tab = table.as_slice()?;
    if tab.len() != phases * phases * taps * taps {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "table length must be phases*phases*taps*taps",
        ));
    }
    let flat: Option<&[f32]> = arr.as_slice();
    let phases_f = phases as f64;

    let mut out = vec![0f32; out_h * out_w * c];
    py.allow_threads(|| {
        out.par_chunks_mut(out_w * c).enumerate().for_each(|(oy, out_row)| {
            for ox in 0..out_w {
                let iy = m00 * oy as f64 + m01 * ox as f64 + o0;
                let ix = m10 * oy as f64 + m11 * ox as f64 + o1;
                let fy = iy.floor();
                let fx = ix.floor();
                let phase_y = (((iy - fy) * phases_f) as usize).min(phases - 1);
                let phase_x = (((ix - fx) * phases_f) as usize).min(phases - 1);
                let weights = &tab[(phase_y * phases + phase_x) * taps * taps
                    ..(phase_y * phases + phase_x + 1) * taps * taps];
                let base_y = fy as isize - halo as isize;
                let base_x = fx as isize - halo as isize;
                let interior = base_y >= 0
                    && base_y + taps as isize <= h as isize
                    && base_x >= 0
                    && base_x + taps as isize <= w as isize;

                if let Some(img) = flat {
                    let row_stride = w * c;
                    if interior {
                        // All taps are in-bounds here by construction, but a
                        // degenerate (e.g. ill-conditioned Wiener) phase can
                        // still have every weight exactly zero -- match the
                        // numpy mirror's "any valid & nonzero-weight tap"
                        // rule instead of unconditionally emitting acc (which
                        // would silently return 0.0 instead of cval).
                        let by = base_y as usize;
                        let bx = base_x as usize;
                        for ch in 0..c {
                            let mut acc = 0.0f64;
                            let mut any = false;
                            for ty in 0..taps {
                                let row_base = (by + ty) * row_stride + bx * c + ch;
                                for tx in 0..taps {
                                    let wgt = weights[ty * taps + tx];
                                    if wgt == 0.0 {
                                        continue;
                                    }
                                    acc += wgt * img[row_base + tx * c] as f64;
                                    any = true;
                                }
                            }
                            out_row[ox * c + ch] = if any { acc as f32 } else { cval };
                        }
                    } else {
                        for ch in 0..c {
                            let mut acc = 0.0f64;
                            let mut any = false;
                            for ty in 0..taps {
                                let yy = base_y + ty as isize;
                                if yy < 0 || yy >= h as isize {
                                    continue;
                                }
                                for tx in 0..taps {
                                    let xx = base_x + tx as isize;
                                    if xx < 0 || xx >= w as isize {
                                        continue;
                                    }
                                    let wgt = weights[ty * taps + tx];
                                    if wgt == 0.0 {
                                        continue;
                                    }
                                    acc += wgt
                                        * img[yy as usize * row_stride + xx as usize * c + ch]
                                            as f64;
                                    any = true;
                                }
                            }
                            out_row[ox * c + ch] = if any { acc as f32 } else { cval };
                        }
                    }
                } else {
                    for ch in 0..c {
                        let mut acc = 0.0f64;
                        let mut any = false;
                        for ty in 0..taps {
                            let yy = base_y + ty as isize;
                            if yy < 0 || yy >= h as isize {
                                continue;
                            }
                            for tx in 0..taps {
                                let xx = base_x + tx as isize;
                                if xx < 0 || xx >= w as isize {
                                    continue;
                                }
                                let wgt = weights[ty * taps + tx];
                                if wgt == 0.0 {
                                    continue;
                                }
                                acc += wgt * arr[[yy as usize, xx as usize, ch]] as f64;
                                any = true;
                            }
                        }
                        out_row[ox * c + ch] = if any { acc as f32 } else { cval };
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

// ============ Matched-filter star detection ============
//
// Mirrors src/star_detect.py::_detect_stars_matched_filter_numpy exactly
// (same mesh-median/sigma construction, same hand-rolled bilinear upsample,
// same separable Gaussian blur, same matched-filter SNR statistic, same
// two-pass centroid refinement). See that module's docstring for the
// validation history -- this is not a first-draft algorithm.

/// 1D Gaussian kernel, unit sum, radius = ceil(3*sigma).
fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil().max(1.0) as isize;
    let mut k: Vec<f64> = (-radius..=radius)
        .map(|i| (-(i as f64 * i as f64) / (2.0 * sigma * sigma)).exp())
        .collect();
    let s: f64 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    k
}

/// Separable convolution with a 1D kernel along both axes, reflect boundary
/// (matches scipy.ndimage.convolve1d(mode='reflect')), row-parallel on each pass.
fn separable_blur(img: &[f64], h: usize, w: usize, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return img.to_vec();
    }
    let k = gaussian_kernel_1d(sigma);
    let half = (k.len() / 2) as isize;

    // Pass 1: along rows (axis=1 in numpy terms -- columns within a row).
    let mut tmp = vec![0f64; h * w];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        let row = &img[y * w..(y + 1) * w];
        for x in 0..w {
            let mut acc = 0f64;
            for (t, &kv) in k.iter().enumerate() {
                let dx = t as isize - half;
                let xi = reflect_idx(x as isize + dx, w);
                acc += kv * row[xi];
            }
            out_row[x] = acc;
        }
    });

    // Pass 2: along columns (axis=0). Parallelise over output rows; each
    // reads a full column stride from `tmp`, which is fine at this size.
    let mut out = vec![0f64; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        for x in 0..w {
            let mut acc = 0f64;
            for (t, &kv) in k.iter().enumerate() {
                let dy = t as isize - half;
                let yi = reflect_idx(y as isize + dy, h);
                acc += kv * tmp[yi * w + x];
            }
            out_row[x] = acc;
        }
    });
    out
}

/// Cell-center-aligned bilinear upsample of a (ny, nx) mesh to (h, w).
/// See src/star_detect.py::_bilinear_upsample for why this is hand-rolled
/// instead of a generic zoom (corner- vs centre-alignment produced a real
/// false-positive cluster at the image border during validation).
fn bilinear_upsample(grid: &[f64], ny: usize, nx: usize, h: usize, w: usize, cell: usize) -> Vec<f64> {
    let cellf = cell as f64;
    let mut out = vec![0f64; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        let gy = y as f64 / cellf - 0.5;
        let gy0 = gy.floor().max(0.0).min((ny - 1) as f64) as usize;
        let gy1 = (gy0 + 1).min(ny - 1);
        let fy = (gy - gy0 as f64).clamp(0.0, 1.0);
        for x in 0..w {
            let gx = x as f64 / cellf - 0.5;
            let gx0 = gx.floor().max(0.0).min((nx - 1) as f64) as usize;
            let gx1 = (gx0 + 1).min(nx - 1);
            let fx = (gx - gx0 as f64).clamp(0.0, 1.0);
            let v00 = grid[gy0 * nx + gx0];
            let v01 = grid[gy0 * nx + gx1];
            let v10 = grid[gy1 * nx + gx0];
            let v11 = grid[gy1 * nx + gx1];
            let v0 = v00 * (1.0 - fx) + v01 * fx;
            let v1 = v10 * (1.0 - fx) + v11 * fx;
            out_row[x] = v0 * (1.0 - fy) + v1 * fy;
        }
    });
    out
}

/// Per-cell median (use_mad=false) or 1.4826*MAD sigma (use_mad=true),
/// upsampled to full resolution and lightly smoothed (sigma = cell*0.3).
fn local_mesh_stat(img: &[f64], h: usize, w: usize, cell: usize, use_mad: bool) -> Vec<f64> {
    let ny = (h / cell.max(1)).max(1);
    let nx = (w / cell.max(1)).max(1);
    let grid: Vec<f64> = (0..ny * nx)
        .into_par_iter()
        .map(|idx| {
            let iy = idx / nx;
            let ix = idx % nx;
            let y0 = iy * cell;
            let y1 = if iy == ny - 1 { h } else { (iy + 1) * cell };
            let x0 = ix * cell;
            let x1 = if ix == nx - 1 { w } else { (ix + 1) * cell };
            let mut vals: Vec<f32> = Vec::with_capacity((y1 - y0) * (x1 - x0));
            for y in y0..y1 {
                let base = y * w;
                for x in x0..x1 {
                    vals.push(img[base + x] as f32);
                }
            }
            let med = median_inplace(&mut vals) as f64;
            if use_mad {
                let mut dev: Vec<f32> = vals.iter().map(|&v| (v as f64 - med).abs() as f32).collect();
                1.4826 * (median_inplace(&mut dev) as f64).max(1e-9)
            } else {
                med
            }
        })
        .collect();
    // Smooth the small mesh grid (blocky-cell artifacts) before upsampling,
    // not the full-resolution field after: same intent (soften cell-to-cell
    // jumps) at a few thousand times less work -- the grid is ~1500 px, the
    // full field ~6M. sigma=0.3 grid-cells here is the same *relative*
    // smoothing as sigma=cell*0.3 was at full resolution.
    let smoothed_grid = separable_blur(&grid, ny, nx, 0.3);
    bilinear_upsample(&smoothed_grid, ny, nx, h, w, cell)
}

#[pyfunction]
#[pyo3(signature = (image, fwhm, k_confirm, cell, roundness_max, min_pixels))]
fn detect_stars_matched_filter<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f32>,
    fwhm: f64,
    k_confirm: f64,
    cell: usize,
    roundness_max: f64,
    min_pixels: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = image.as_array();
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    let owned: Vec<f64>;
    let lum: &[f64] = match arr.as_slice() {
        Some(s) => {
            owned = s.iter().map(|&v| v as f64).collect();
            &owned
        }
        None => {
            owned = arr.iter().map(|&v| v as f64).collect();
            &owned
        }
    };

    let rows: Vec<[f64; 10]> = py.allow_threads(|| {
        let bg_map = local_mesh_stat(lum, h, w, cell, false);
        let sigma_map = local_mesh_stat(lum, h, w, cell, true);
        let resid: Vec<f64> = lum.iter().zip(&bg_map).map(|(&v, &b)| v - b).collect();

        // A Gaussian is exactly separable: conv2d(img, outer(k1,k1)) ==
        // conv1d_col(conv1d_row(img, k1), k1), same result at O(2k)
        // taps/pixel instead of O(k^2). kernel_norm (the SNR noise
        // normalisation) collapses algebraically too:
        // sqrt(sum(outer(k1,k1)^2)) == sum(k1^2) exactly (sum_ij
        // (k1_i k1_j)^2 = (sum k1^2)^2, sqrt of that = sum k1^2).
        let sigma_k = fwhm / 2.3548;
        let k1 = gaussian_kernel_1d(sigma_k);
        let kernel_norm: f64 = k1.iter().map(|&v| v * v).sum();
        let filtered = separable_blur(&resid, h, w, sigma_k);

        let mut snr_map = vec![0f64; h * w];
        snr_map.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
            for x in 0..w {
                let sig = (sigma_map[y * w + x] * kernel_norm).max(1e-9);
                out_row[x] = filtered[y * w + x] / sig;
            }
        });

        // Local-maxima + threshold + border exclusion, row-parallel.
        let footprint = (fwhm.round() as isize).max(3);
        let fhalf = footprint / 2;
        let border = (cell / 2).max((2.0 * (3.0 * fwhm / 2.3548).ceil()) as usize);

        let candidates: Vec<(usize, usize)> = (0..h)
            .into_par_iter()
            .flat_map_iter(|y| {
                let mut out = Vec::new();
                if y < border || y + border >= h {
                    return out;
                }
                for x in border..w.saturating_sub(border) {
                    let v = snr_map[y * w + x];
                    if v <= k_confirm {
                        continue;
                    }
                    let mut is_max = true;
                    'outer: for dy in -fhalf..=fhalf {
                        let yi = y as isize + dy;
                        if yi < 0 || yi >= h as isize {
                            continue;
                        }
                        let row_base = yi as usize * w;
                        for dx in -fhalf..=fhalf {
                            let xi = x as isize + dx;
                            if xi < 0 || xi >= w as isize {
                                continue;
                            }
                            if snr_map[row_base + xi as usize] > v {
                                is_max = false;
                                break 'outer;
                            }
                        }
                    }
                    if is_max {
                        out.push((y, x));
                    }
                }
                out
            })
            .collect();

        // Per-candidate measurement: local background, two-pass centroid,
        // second moments (shape/roundness). Embarrassingly parallel.
        let r = ((1.5 * fwhm).round() as isize).max(3);
        let rr = ((0.7 * fwhm).round() as isize).max(2);

        candidates
            .into_par_iter()
            .filter_map(|(py_, px_)| {
                let y0 = (py_ as isize - r).max(0) as usize;
                let y1 = ((py_ as isize + r + 1).max(0) as usize).min(h);
                let x0 = (px_ as isize - r).max(0) as usize;
                let x1 = ((px_ as isize + r + 1).max(0) as usize).min(w);
                let local_bg = bg_map[py_ * w + px_];

                let mut wsum = 0f64;
                let mut cy = 0f64;
                let mut cx = 0f64;
                let mut n_positive = 0usize;
                for y in y0..y1 {
                    let row_base = y * w;
                    for x in x0..x1 {
                        let wv = (lum[row_base + x] - local_bg).max(0.0);
                        if wv > 0.0 {
                            n_positive += 1;
                        }
                        wsum += wv;
                        cy += wv * y as f64;
                        cx += wv * x as f64;
                    }
                }
                if wsum <= 0.0 {
                    return None;
                }
                cy /= wsum;
                cx /= wsum;

                // Refinement pass: tighter window centred on first estimate.
                let ry0 = ((cy.round() as isize) - rr).max(0) as usize;
                let ry1 = (((cy.round() as isize) + rr + 1).max(0) as usize).min(h);
                let rx0 = ((cx.round() as isize) - rr).max(0) as usize;
                let rx1 = (((cx.round() as isize) + rr + 1).max(0) as usize).min(w);
                let mut rwsum = 0f64;
                let mut rcy = 0f64;
                let mut rcx = 0f64;
                for y in ry0..ry1 {
                    let row_base = y * w;
                    for x in rx0..rx1 {
                        let wv = (lum[row_base + x] - local_bg).max(0.0);
                        rwsum += wv;
                        rcy += wv * y as f64;
                        rcx += wv * x as f64;
                    }
                }
                if rwsum > 0.0 {
                    cy = rcy / rwsum;
                    cx = rcx / rwsum;
                }

                // Second moments over the ORIGINAL (first-pass) window,
                // centred on the refined centroid -- matches the numpy mirror.
                let mut ixx = 0f64;
                let mut iyy = 0f64;
                let mut ixy = 0f64;
                for y in y0..y1 {
                    let row_base = y * w;
                    let dy = y as f64 - cy;
                    for x in x0..x1 {
                        let wv = (lum[row_base + x] - local_bg).max(0.0);
                        let dx = x as f64 - cx;
                        ixx += wv * dy * dy;
                        iyy += wv * dx * dx;
                        ixy += wv * dy * dx;
                    }
                }
                ixx /= wsum;
                iyy /= wsum;
                ixy /= wsum;
                // 2x2 symmetric eigenvalues (closed form).
                let tr = ixx + iyy;
                let det = ixx * iyy - ixy * ixy;
                let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
                let e1 = (tr / 2.0 + disc).max(1e-6);
                let e0 = (tr / 2.0 - disc).max(1e-6);
                let a = e1.sqrt();
                let b = e0.sqrt();
                let roundness = 1.0 - a.min(b) / a.max(b).max(1e-6);
                if roundness >= roundness_max {
                    return None;
                }
                if n_positive < min_pixels {
                    return None;
                }

                let mut flux = 0f64;
                let mut peak = f64::NEG_INFINITY;
                for y in y0..y1 {
                    let row_base = y * w;
                    for x in x0..x1 {
                        let v = lum[row_base + x];
                        if v > peak {
                            peak = v;
                        }
                        flux += (v - local_bg).max(0.0);
                    }
                }
                let sharpness = (snr_map[py_ * w + px_] / 20.0).clamp(0.0, 1.0);

                Some([cx, cy, flux, peak, roundness, roundness, sharpness, a, b, 0.0])
            })
            .collect()
    });

    let n = rows.len();
    let mut flat = Vec::with_capacity(n * 10);
    for row in rows {
        flat.extend_from_slice(&row);
    }
    let out = numpy::ndarray::Array2::from_shape_vec((n, 10), flat)
        .expect("shape mismatch building star detection rows");
    Ok(out.into_pyarray(py))
}

// ============ RANSAC-robust rigid (Euclidean) transform fit ============
//
// Mirrors src/affine_fit.py exactly -- same Umeyama (1991) closed-form
// rigid-transform solve (via a 2x2 SVD computed here through eigendecomposition
// of A^T*A, algebraically the same operation numpy.linalg.svd performs), same
// RANSAC loop semantics (dynamic max_trials shrinking, more-inliers-then-
// less-residual tie-break, final refit on all inliers of the best trial).
// See that module's docstring for why parity with skimage itself is
// statistical (skimage's own usage here is unseeded) rather than bit-exact,
// while parity between this kernel and the numpy mirror (for a shared seed)
// is exact and is what's actually tested.

/// Minimal splitmix64 PRNG -- no external `rand` crate dependency for what's
/// just "pick k distinct indices from n, many times".
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// k distinct indices from 0..n via partial Fisher-Yates.
    fn choice(&mut self, n: usize, k: usize, pool: &mut Vec<usize>) {
        pool.clear();
        pool.extend(0..n);
        for i in 0..k {
            let span = (n - i) as u64;
            let j = i + (self.next_u64() % span) as usize;
            pool.swap(i, j);
        }
        pool.truncate(k);
    }
}

fn entropy_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    // Mix in a stack address for extra spread between near-simultaneous calls
    // on different threads (cheap ASLR-based entropy, not cryptographic --
    // this only needs to avoid identical RANSAC sample sequences).
    let local = 0u8;
    let addr = &local as *const u8 as u64;
    nanos ^ addr.wrapping_mul(0x9E3779B97F4A7C15)
}

/// 2x2 SVD via eigendecomposition of A^T*A: A = U * diag(S) * V^T, S
/// descending. Returns (U, S, Vt) matching numpy.linalg.svd's convention
/// (Vt is V transposed, i.e. its rows are the right singular vectors).
fn svd_2x2(a: [[f64; 2]; 2]) -> ([[f64; 2]; 2], [f64; 2], [[f64; 2]; 2]) {
    let m00 = a[0][0] * a[0][0] + a[1][0] * a[1][0];
    let m01 = a[0][0] * a[0][1] + a[1][0] * a[1][1];
    let m11 = a[0][1] * a[0][1] + a[1][1] * a[1][1];

    let tr = m00 + m11;
    let det = m00 * m11 - m01 * m01;
    let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
    let l1 = (tr / 2.0 + disc).max(0.0);
    let l2 = (tr / 2.0 - disc).max(0.0);
    let s1 = l1.sqrt();
    let s2 = l2.sqrt();

    let eigvec = |lambda: f64| -> [f64; 2] {
        if m01.abs() > 1e-12 {
            let (x, y) = (1.0, -(m00 - lambda) / m01);
            let n = (x * x + y * y).sqrt();
            [x / n, y / n]
        } else if (m00 - lambda).abs() < 1e-9 {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        }
    };
    let v1 = eigvec(l1);
    let v2 = [-v1[1], v1[0]];

    let apply = |v: [f64; 2]| [a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1]];
    let u1 = if s1 > 1e-12 {
        let raw = apply(v1);
        [raw[0] / s1, raw[1] / s1]
    } else {
        [1.0, 0.0]
    };
    let u2 = if s2 > 1e-12 {
        let raw = apply(v2);
        [raw[0] / s2, raw[1] / s2]
    } else {
        [-u1[1], u1[0]]
    };

    ([[u1[0], u2[0]], [u1[1], u2[1]]], [s1, s2], [[v1[0], v1[1]], [v2[0], v2[1]]])
}

fn mat2_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

/// Umeyama (1991) 2D rigid (no scale) least-squares fit -- exact port of
/// skimage.transform._geometric._umeyama(src, dst, estimate_scale=False).
/// Returns a 3x3 homogeneous matrix, or None if degenerate (rank 0).
fn umeyama_2d(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<[[f64; 3]; 3]> {
    let n = src.len() as f64;
    if n <= 0.0 {
        return None;
    }
    let mut src_mean = [0.0, 0.0];
    let mut dst_mean = [0.0, 0.0];
    for p in src {
        src_mean[0] += p[0];
        src_mean[1] += p[1];
    }
    for p in dst {
        dst_mean[0] += p[0];
        dst_mean[1] += p[1];
    }
    src_mean[0] /= n;
    src_mean[1] /= n;
    dst_mean[0] /= n;
    dst_mean[1] /= n;

    let mut a = [[0.0, 0.0], [0.0, 0.0]];
    for (s, d) in src.iter().zip(dst.iter()) {
        let sx = s[0] - src_mean[0];
        let sy = s[1] - src_mean[1];
        let dx = d[0] - dst_mean[0];
        let dy = d[1] - dst_mean[1];
        a[0][0] += dx * sx;
        a[0][1] += dx * sy;
        a[1][0] += dy * sx;
        a[1][1] += dy * sy;
    }
    a[0][0] /= n;
    a[0][1] /= n;
    a[1][0] /= n;
    a[1][1] /= n;

    let det_a = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let d = [1.0, if det_a < 0.0 { -1.0 } else { 1.0 }];

    let (u, s, vt) = svd_2x2(a);
    let tol = s[0] * 2.0 * f64::EPSILON;
    let rank = s.iter().filter(|&&x| x > tol).count();

    let rot = if rank == 0 {
        return None;
    } else if rank == 1 {
        let det_u = u[0][0] * u[1][1] - u[0][1] * u[1][0];
        let det_vt = vt[0][0] * vt[1][1] - vt[0][1] * vt[1][0];
        if det_u * det_vt > 0.0 {
            mat2_mul(u, vt)
        } else {
            // Python mirror restores d[dim-1] after use (matching skimage's
            // own hygiene, in case d were read again) -- unlike Python's
            // mutable-list semantics, this is provably dead in Rust (d goes
            // out of scope right after), so just build the flipped diag
            // inline instead of mutating d.
            mat2_mul(mat2_mul(u, [[d[0], 0.0], [0.0, -1.0]]), vt)
        }
    } else {
        mat2_mul(mat2_mul(u, [[d[0], 0.0], [0.0, d[1]]]), vt)
    };

    let tx = dst_mean[0] - (rot[0][0] * src_mean[0] + rot[0][1] * src_mean[1]);
    let ty = dst_mean[1] - (rot[1][0] * src_mean[0] + rot[1][1] * src_mean[1]);
    Some([[rot[0][0], rot[0][1], tx], [rot[1][0], rot[1][1], ty], [0.0, 0.0, 1.0]])
}

fn dynamic_max_trials(n_inliers: usize, n_samples: usize, min_samples: usize) -> f64 {
    // probability=1.0 (skimage's default, unchanged by this codebase's caller)
    if n_inliers == 0 {
        return f64::INFINITY;
    }
    let eps = f64::EPSILON;
    let inlier_ratio = n_inliers as f64 / n_samples as f64;
    let nom = eps; // clip(1 - 1.0, eps, 1-eps) == eps
    let denom = (1.0 - inlier_ratio.powi(min_samples as i32)).clamp(eps, 1.0 - eps);
    (nom.ln() / denom.ln()).ceil()
}

#[pyfunction]
#[pyo3(signature = (src, dst, min_samples, residual_threshold, max_trials, seed))]
fn fit_rigid_ransac<'py>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, f64>,
    dst: PyReadonlyArray2<'py, f64>,
    min_samples: usize,
    residual_threshold: f64,
    max_trials: usize,
    seed: i64,
) -> PyResult<(Option<Bound<'py, PyArray2<f64>>>, Option<Bound<'py, PyArray1<bool>>>)> {
    let src_arr = src.as_array();
    let dst_arr = dst.as_array();
    let n = src_arr.shape()[0];
    if dst_arr.shape()[0] != n {
        return Err(pyo3::exceptions::PyValueError::new_err("src/dst length mismatch"));
    }
    if n < min_samples {
        return Ok((None, None));
    }

    let src_pts: Vec<[f64; 2]> = (0..n).map(|i| [src_arr[[i, 0]], src_arr[[i, 1]]]).collect();
    let dst_pts: Vec<[f64; 2]> = (0..n).map(|i| [dst_arr[[i, 0]], dst_arr[[i, 1]]]).collect();

    let (best_params, best_inliers) = py.allow_threads(|| {
        let mut rng = SplitMix64::new(if seed < 0 { entropy_seed() } else { seed as u64 });
        let mut best_inlier_num = 0usize;
        let mut best_inlier_residuals_sum = f64::INFINITY;
        let mut best_inliers: Option<Vec<bool>> = None;

        let mut idx_pool = Vec::with_capacity(n);
        let mut sample_src = vec![[0.0, 0.0]; min_samples];
        let mut sample_dst = vec![[0.0, 0.0]; min_samples];
        let mut trials = 0usize;
        let mut cur_max_trials = max_trials;

        while trials < cur_max_trials {
            trials += 1;
            rng.choice(n, min_samples, &mut idx_pool);
            for (k, &i) in idx_pool.iter().enumerate() {
                sample_src[k] = src_pts[i];
                sample_dst[k] = dst_pts[i];
            }
            let params = match umeyama_2d(&sample_src, &sample_dst) {
                Some(p) => p,
                None => continue,
            };

            let mut inliers = vec![false; n];
            let mut inliers_count = 0usize;
            let mut residuals_sum = 0.0f64;
            for i in 0..n {
                let p = src_pts[i];
                let tx = params[0][0] * p[0] + params[0][1] * p[1] + params[0][2];
                let ty = params[1][0] * p[0] + params[1][1] * p[1] + params[1][2];
                let dx = tx - dst_pts[i][0];
                let dy = ty - dst_pts[i][1];
                let r2 = dx * dx + dy * dy;
                residuals_sum += r2;
                if r2.sqrt() < residual_threshold {
                    inliers[i] = true;
                    inliers_count += 1;
                }
            }

            if inliers_count > best_inlier_num
                || (inliers_count == best_inlier_num && residuals_sum < best_inlier_residuals_sum)
            {
                best_inlier_num = inliers_count;
                best_inlier_residuals_sum = residuals_sum;
                best_inliers = Some(inliers);
                cur_max_trials = cur_max_trials.min(
                    dynamic_max_trials(best_inlier_num, n, min_samples).min(max_trials as f64) as usize,
                );
                if best_inlier_num >= n || best_inlier_residuals_sum <= 0.0 {
                    break;
                }
            }
        }

        let inliers = match best_inliers {
            Some(v) if v.iter().any(|&b| b) => v,
            _ => return (None, None),
        };
        let final_src: Vec<[f64; 2]> = (0..n).filter(|&i| inliers[i]).map(|i| src_pts[i]).collect();
        let final_dst: Vec<[f64; 2]> = (0..n).filter(|&i| inliers[i]).map(|i| dst_pts[i]).collect();
        match umeyama_2d(&final_src, &final_dst) {
            Some(params) => (Some(params), Some(inliers)),
            None => (None, None),
        }
    });

    match (best_params, best_inliers) {
        (Some(params), Some(inliers)) => {
            let flat: Vec<f64> = params.iter().flatten().copied().collect();
            let arr = numpy::ndarray::Array2::from_shape_vec((3, 3), flat)
                .expect("shape mismatch building rigid transform params");
            Ok((Some(arr.into_pyarray(py)), Some(inliers.into_pyarray(py))))
        }
        _ => Ok((None, None)),
    }
}

// ---------------------------------------------------------------------------
// Malvar-He-Cutler (2004) Bayer demosaicing
// ---------------------------------------------------------------------------
//
// Kernels and per-position selection mirror src/debayer.py's
// _debayer_malvar_numpy exactly (validated bit-exact against it there, which
// is itself validated against the `colour-demosaicing` reference package —
// see tests/test_debayer_malvar.py). Expressed as sparse (dy, dx, weight)
// tap lists rather than dense 5x5 convolution passes: each output pixel
// needs at most 2 of the 4 kernels (its own channel is the raw sample), so a
// per-pixel gather is less work than 4 whole-image convolutions, and this
// kernel is already a per-pixel loop (unlike the numpy reference, where 4
// vectorised scipy convolutions is the natural expression). Weights bake in
// the /8 normalisation from the published coefficients.

const MALVAR_G_AT_RB: [(isize, isize, f64); 9] = [
    (-2, 0, -1.0 / 8.0), (-1, 0, 2.0 / 8.0),
    (0, -2, -1.0 / 8.0), (0, -1, 2.0 / 8.0), (0, 0, 4.0 / 8.0), (0, 1, 2.0 / 8.0), (0, 2, -1.0 / 8.0),
    (1, 0, 2.0 / 8.0), (2, 0, -1.0 / 8.0),
];

// R at green in an R row / B column (and B at green in a B row / R column).
const MALVAR_RG_RB_BG_BR: [(isize, isize, f64); 11] = [
    (-2, 0, 0.5 / 8.0),
    (-1, -1, -1.0 / 8.0), (-1, 1, -1.0 / 8.0),
    (0, -2, -1.0 / 8.0), (0, -1, 4.0 / 8.0), (0, 0, 5.0 / 8.0), (0, 1, 4.0 / 8.0), (0, 2, -1.0 / 8.0),
    (1, -1, -1.0 / 8.0), (1, 1, -1.0 / 8.0),
    (2, 0, 0.5 / 8.0),
];

// R at green in a B row / R column (and B at green in an R row / B column) --
// transpose of the kernel above.
const MALVAR_RG_BR_BG_RB: [(isize, isize, f64); 11] = [
    (0, -2, 0.5 / 8.0),
    (-1, -1, -1.0 / 8.0), (1, -1, -1.0 / 8.0),
    (-2, 0, -1.0 / 8.0), (-1, 0, 4.0 / 8.0), (0, 0, 5.0 / 8.0), (1, 0, 4.0 / 8.0), (2, 0, -1.0 / 8.0),
    (-1, 1, -1.0 / 8.0), (1, 1, -1.0 / 8.0),
    (0, 2, 0.5 / 8.0),
];

// R at B (and B at R).
const MALVAR_R_AT_B: [(isize, isize, f64); 9] = [
    (-2, 0, -1.5 / 8.0),
    (-1, -1, 2.0 / 8.0), (-1, 1, 2.0 / 8.0),
    (0, -2, -1.5 / 8.0), (0, 0, 6.0 / 8.0), (0, 2, -1.5 / 8.0),
    (1, -1, 2.0 / 8.0), (1, 1, 2.0 / 8.0),
    (2, 0, -1.5 / 8.0),
];

/// scipy `mode='mirror'` boundary index (reflects without duplicating the
/// edge sample, period `2*(n-1)`) -- matches the mode the numpy counterpart
/// (`_debayer_malvar_numpy`) passes to `scipy.ndimage.convolve`. Distinct
/// from `reflect_idx` above, which implements scipy's `mode='reflect'`
/// (duplicates the edge sample) for the *other* kernels in this file whose
/// own numpy mirrors use that convention instead -- each kernel only needs
/// to agree with its own Python counterpart, not with every other kernel.
#[inline]
fn mirror_idx(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut idx = i.rem_euclid(period);
    if idx >= n as isize {
        idx = period - idx;
    }
    idx as usize
}

#[inline]
fn malvar_tap(data: &[f32], h: usize, w: usize, y: usize, x: usize, taps: &[(isize, isize, f64)]) -> f32 {
    let mut acc = 0.0f64;
    for &(dy, dx, wt) in taps {
        let yy = mirror_idx(y as isize + dy, h);
        let xx = mirror_idx(x as isize + dx, w);
        acc += data[yy * w + xx] as f64 * wt;
    }
    acc as f32
}

/// Same as `malvar_tap` but for interior pixels only (2-pixel margin from
/// every edge already guaranteed by the caller) -- direct indexing, no
/// per-tap `mirror_idx` modulo/branch.
#[inline]
fn malvar_tap_interior(data: &[f32], w: usize, base: usize, taps: &[(isize, isize, f64)]) -> f32 {
    let mut acc = 0.0f64;
    for &(dy, dx, wt) in taps {
        let off = dy * w as isize + dx;
        acc += data[(base as isize + off) as usize] as f64 * wt;
    }
    acc as f32
}

fn malvar_pattern_offsets(pattern: &str) -> PyResult<(usize, usize, usize, usize)> {
    match pattern {
        "RGGB" => Ok((0, 0, 1, 1)),
        "BGGR" => Ok((1, 1, 0, 0)),
        "GRBG" => Ok((0, 1, 1, 0)),
        "GBRG" => Ok((1, 0, 0, 1)),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown Bayer pattern '{pattern}'"
        ))),
    }
}

#[pyfunction]
fn debayer_malvar<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f32>,
    pattern: &str,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (r_r, r_c, b_r, b_c) = malvar_pattern_offsets(pattern)?;
    let arr = data.as_array();
    let s = arr.shape();
    let (h, w) = (s[0], s[1]);
    let owned: Vec<f32>;
    let flat: &[f32] = match arr.as_slice() {
        Some(sl) => sl,
        None => {
            owned = arr.iter().copied().collect();
            &owned
        }
    };

    let mut out = vec![0f32; h * w * 3];
    py.allow_threads(|| {
        out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row_out)| {
            let is_r_row = y % 2 == r_r;
            let is_b_row = y % 2 == b_r;
            let interior_y = y >= 2 && y + 2 < h;
            for x in 0..w {
                let is_r_col = x % 2 == r_c;
                let is_b_col = x % 2 == b_c;
                let raw = flat[y * w + x];
                let (r, g, b) = if interior_y && x >= 2 && x + 2 < w {
                    let base = y * w + x;
                    if is_r_row && is_r_col {
                        (raw, malvar_tap_interior(flat, w, base, &MALVAR_G_AT_RB),
                         malvar_tap_interior(flat, w, base, &MALVAR_R_AT_B))
                    } else if is_b_row && is_b_col {
                        (malvar_tap_interior(flat, w, base, &MALVAR_R_AT_B),
                         malvar_tap_interior(flat, w, base, &MALVAR_G_AT_RB), raw)
                    } else if is_r_row && is_b_col {
                        (malvar_tap_interior(flat, w, base, &MALVAR_RG_RB_BG_BR), raw,
                         malvar_tap_interior(flat, w, base, &MALVAR_RG_BR_BG_RB))
                    } else {
                        (malvar_tap_interior(flat, w, base, &MALVAR_RG_BR_BG_RB), raw,
                         malvar_tap_interior(flat, w, base, &MALVAR_RG_RB_BG_BR))
                    }
                } else if is_r_row && is_r_col {
                    (raw, malvar_tap(flat, h, w, y, x, &MALVAR_G_AT_RB),
                     malvar_tap(flat, h, w, y, x, &MALVAR_R_AT_B))
                } else if is_b_row && is_b_col {
                    (malvar_tap(flat, h, w, y, x, &MALVAR_R_AT_B),
                     malvar_tap(flat, h, w, y, x, &MALVAR_G_AT_RB), raw)
                } else if is_r_row && is_b_col {
                    (malvar_tap(flat, h, w, y, x, &MALVAR_RG_RB_BG_BR), raw,
                     malvar_tap(flat, h, w, y, x, &MALVAR_RG_BR_BG_RB))
                } else {
                    // is_b_row && is_r_col
                    (malvar_tap(flat, h, w, y, x, &MALVAR_RG_BR_BG_RB), raw,
                     malvar_tap(flat, h, w, y, x, &MALVAR_RG_RB_BG_BR))
                };
                row_out[x * 3] = r;
                row_out[x * 3 + 1] = g;
                row_out[x * 3 + 2] = b;
            }
        });
    });

    let arr3 = numpy::ndarray::Array3::from_shape_vec((h, w, 3), out)
        .expect("shape mismatch building debayer_malvar output");
    Ok(arr3.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Menon (2007) DDFAPD Bayer demosaicing
// ---------------------------------------------------------------------------
//
// Direct staged port of src/debayer.py's _debayer_menon2007_numpy (itself
// validated bit-exact against the `colour-demosaicing` reference package --
// see tests/test_debayer_menon2007.py). Unlike debayer_malvar above, whose
// per-pixel output depends only on a fixed set of nearby taps (fusible into
// one gather per pixel), Menon's per-pixel direction decision (`use_h`)
// depends on a 5x5-diffused colour-difference gradient built from a
// shifted-by-2 difference of a *different* directional green estimate --
// genuinely sequential stages, each needing the whole previous stage
// materialized first. Ported as the same staged full-array passes the numpy
// reference uses (each stage parallelised across rows with rayon), rather
// than one fused gather.
//
// f64 throughout (matching the numpy reference's internal precision, since
// scipy promotes to float64 for these convolutions) with f32 in/out only at
// the native/Python boundary.

/// Combined green-at-R/B directional filter (h_0 + h_1 from the paper,
/// pre-summed since both are applied to the same input and immediately
/// added -- see _MENON_H0/_MENON_H1 in src/debayer.py for the two separately).
const MENON_G_TAPS: [(isize, f64); 5] =
    [(-2, -0.25), (-1, 0.5), (0, 0.5), (1, 0.5), (2, -0.25)];

/// R/B <-> G colour-difference averaging filter ([0.5, 0, 0.5]; the centre
/// tap is zero so it's omitted).
const MENON_KB_TAPS: [(isize, f64); 2] = [(-1, 0.5), (1, 0.5)];

/// Refining-step 3-tap box filter (FIR = [1/3, 1/3, 1/3]).
const MENON_FIR_TAPS: [(isize, f64); 3] =
    [(-1, 1.0 / 3.0), (0, 1.0 / 3.0), (1, 1.0 / 3.0)];

/// 5x5 gradient-diffusion kernel (horizontal direction), sparse taps only --
/// see _MENON_DIFFUSION_K in src/debayer.py for the dense matrix this comes
/// from. Zero-padded boundary (scipy `mode='constant'`, cval=0), unlike the
/// mirror boundary every other Menon convolution here uses.
///
/// `_MENON_DIFFUSION_K` is *not* 180-degree symmetric (unlike every other
/// Menon kernel in this file), so unlike those, this one is sensitive to the
/// convolve-vs-correlate distinction: `scipy.ndimage.convolve` flips the
/// kernel before applying it, so a tap at dense-matrix offset (dy, dx) is
/// read from input position (y - dy, x - dx), not (y + dy, x + dx). These
/// taps are pre-negated ((-dy, -dx) of the dense matrix's own offsets) so
/// the direct-gather loop below can read (y + dy, x + dx) as usual.
const MENON_DIFF_K: [(isize, isize, f64); 8] = [
    (2, 0, 1.0), (2, -2, 1.0),
    (1, -1, 1.0),
    (0, 0, 3.0), (0, -2, 3.0),
    (-1, -1, 1.0),
    (-2, 0, 1.0), (-2, -2, 1.0),
];

/// Transpose of MENON_DIFF_K (vertical direction) -- each tap's (dy, dx) swapped.
const MENON_DIFF_KT: [(isize, isize, f64); 8] = [
    (0, 2, 1.0), (-2, 2, 1.0),
    (-1, 1, 1.0),
    (0, 0, 3.0), (-2, 0, 3.0),
    (-1, -1, 1.0),
    (0, -2, 1.0), (-2, -2, 1.0),
];

/// 1D horizontal convolution, scipy `mode='mirror'` boundary (reuses `mirror_idx`).
fn menon_conv_h(data: &[f64], h: usize, w: usize, taps: &[(isize, f64)]) -> Vec<f64> {
    let mut out = vec![0.0f64; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        let base = y * w;
        for x in 0..w {
            let mut acc = 0.0f64;
            for &(dx, wt) in taps {
                acc += data[base + mirror_idx(x as isize + dx, w)] * wt;
            }
            row_out[x] = acc;
        }
    });
    out
}

/// 1D vertical convolution, scipy `mode='mirror'` boundary.
fn menon_conv_v(data: &[f64], h: usize, w: usize, taps: &[(isize, f64)]) -> Vec<f64> {
    let mut out = vec![0.0f64; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        for x in 0..w {
            let mut acc = 0.0f64;
            for &(dy, wt) in taps {
                acc += data[mirror_idx(y as isize + dy, h) * w + x] * wt;
            }
            row_out[x] = acc;
        }
    });
    out
}

/// 2D sparse convolution, scipy `mode='constant'` (cval=0) boundary --
/// out-of-range taps contribute nothing, matching the reference's
/// `scipy.ndimage.convolve(..., mode='constant')` used only for the 5x5
/// gradient-diffusion pass.
fn menon_diffuse(data: &[f64], h: usize, w: usize, taps: &[(isize, isize, f64)]) -> Vec<f64> {
    let mut out = vec![0.0f64; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        for x in 0..w {
            let mut acc = 0.0f64;
            for &(dy, dx, wt) in taps {
                let yy = y as isize + dy;
                let xx = x as isize + dx;
                if yy >= 0 && xx >= 0 && (yy as usize) < h && (xx as usize) < w {
                    acc += data[yy as usize * w + xx as usize] * wt;
                }
            }
            row_out[x] = acc;
        }
    });
    out
}

#[pyfunction]
fn debayer_menon2007<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f32>,
    pattern: &str,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (r_r, r_c, b_r, b_c) = malvar_pattern_offsets(pattern)?;
    let arr = data.as_array();
    let s = arr.shape();
    let (h, w) = (s[0], s[1]);
    let n = h * w;
    let owned: Vec<f32>;
    let flat: &[f32] = match arr.as_slice() {
        Some(sl) => sl,
        None => {
            owned = arr.iter().copied().collect();
            &owned
        }
    };
    let raw: Vec<f64> = flat.iter().map(|&v| v as f64).collect();

    let is_r_row: Vec<bool> = (0..h).map(|y| y % 2 == r_r).collect();
    let is_b_row: Vec<bool> = (0..h).map(|y| y % 2 == b_r).collect();
    let is_r_col: Vec<bool> = (0..w).map(|x| x % 2 == r_c).collect();
    let is_b_col: Vec<bool> = (0..w).map(|x| x % 2 == b_c).collect();

    let (r_final, g_final, b_final) = py.allow_threads(|| {
        let is_r: Vec<bool> = (0..n).map(|i| is_r_row[i / w] && is_r_col[i % w]).collect();
        let is_b: Vec<bool> = (0..n).map(|i| is_b_row[i / w] && is_b_col[i % w]).collect();
        let is_g: Vec<bool> = (0..n).map(|i| !is_r[i] && !is_b[i]).collect();

        let g_h_conv = menon_conv_h(&raw, h, w, &MENON_G_TAPS);
        let g_v_conv = menon_conv_v(&raw, h, w, &MENON_G_TAPS);
        let g_h: Vec<f64> = (0..n).map(|i| if is_g[i] { raw[i] } else { g_h_conv[i] }).collect();
        let g_v: Vec<f64> = (0..n).map(|i| if is_g[i] { raw[i] } else { g_v_conv[i] }).collect();

        let c_h: Vec<f64> = (0..n)
            .map(|i| if is_r[i] || is_b[i] { raw[i] - g_h[i] } else { 0.0 })
            .collect();
        let c_v: Vec<f64> = (0..n)
            .map(|i| if is_r[i] || is_b[i] { raw[i] - g_v[i] } else { 0.0 })
            .collect();

        let mut d_h = vec![0.0f64; n];
        d_h.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let base = y * w;
            for x in 0..w {
                row[x] = (c_h[base + x] - c_h[base + mirror_idx(x as isize + 2, w)]).abs();
            }
        });
        let mut d_v = vec![0.0f64; n];
        d_v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                row[x] = (c_v[y * w + x] - c_v[mirror_idx(y as isize + 2, h) * w + x]).abs();
            }
        });

        let dd_h = menon_diffuse(&d_h, h, w, &MENON_DIFF_K);
        let dd_v = menon_diffuse(&d_v, h, w, &MENON_DIFF_KT);
        let use_h: Vec<bool> = (0..n).map(|i| dd_v[i] >= dd_h[i]).collect();

        let g: Vec<f64> = (0..n)
            .map(|i| if is_g[i] { raw[i] } else if use_h[i] { g_h[i] } else { g_v[i] })
            .collect();

        let mut r: Vec<f64> = (0..n).map(|i| if is_r[i] { raw[i] } else { 0.0 }).collect();
        let mut b: Vec<f64> = (0..n).map(|i| if is_b[i] { raw[i] } else { 0.0 }).collect();

        // Green sites: fill R and B from the just-finalised G, in row-parity
        // order matching the reference exactly (each step reads the *current*
        // r/g/b state, so order and in-place mutation both matter).
        let g_ch = menon_conv_h(&g, h, w, &MENON_KB_TAPS);
        let g_cv = menon_conv_v(&g, h, w, &MENON_KB_TAPS);

        {
            let r_ch = menon_conv_h(&r, h, w, &MENON_KB_TAPS);
            for i in 0..n {
                if is_g[i] && is_r_row[i / w] {
                    r[i] = g[i] + r_ch[i] - g_ch[i];
                }
            }
        }
        {
            let r_cv = menon_conv_v(&r, h, w, &MENON_KB_TAPS);
            for i in 0..n {
                if is_g[i] && is_b_row[i / w] {
                    r[i] = g[i] + r_cv[i] - g_cv[i];
                }
            }
        }
        {
            let b_ch = menon_conv_h(&b, h, w, &MENON_KB_TAPS);
            for i in 0..n {
                if is_g[i] && is_b_row[i / w] {
                    b[i] = g[i] + b_ch[i] - g_ch[i];
                }
            }
        }
        {
            let b_cv = menon_conv_v(&b, h, w, &MENON_KB_TAPS);
            for i in 0..n {
                if is_g[i] && is_r_row[i / w] {
                    b[i] = g[i] + b_cv[i] - g_cv[i];
                }
            }
        }

        // Opposite-colour sites (R at B, B at R), direction-selected.
        {
            let r_ch = menon_conv_h(&r, h, w, &MENON_KB_TAPS);
            let b_ch = menon_conv_h(&b, h, w, &MENON_KB_TAPS);
            let r_cv = menon_conv_v(&r, h, w, &MENON_KB_TAPS);
            let b_cv = menon_conv_v(&b, h, w, &MENON_KB_TAPS);
            for i in 0..n {
                if is_b_row[i / w] && is_b[i] {
                    r[i] = if use_h[i] { b[i] + r_ch[i] - b_ch[i] } else { b[i] + r_cv[i] - b_cv[i] };
                }
            }
        }
        {
            let r_ch = menon_conv_h(&r, h, w, &MENON_KB_TAPS);
            let b_ch = menon_conv_h(&b, h, w, &MENON_KB_TAPS);
            let r_cv = menon_conv_v(&r, h, w, &MENON_KB_TAPS);
            let b_cv = menon_conv_v(&b, h, w, &MENON_KB_TAPS);
            for i in 0..n {
                if is_r_row[i / w] && is_r[i] {
                    b[i] = if use_h[i] { r[i] + b_ch[i] - r_ch[i] } else { r[i] + b_cv[i] - r_cv[i] };
                }
            }
        }

        let mut g = g;
        // Refining step (matches src/debayer.py's refining_step=True default).
        {
            let r_g: Vec<f64> = (0..n).map(|i| r[i] - g[i]).collect();
            let b_g: Vec<f64> = (0..n).map(|i| b[i] - g[i]).collect();
            let r_g_ch = menon_conv_h(&r_g, h, w, &MENON_FIR_TAPS);
            let r_g_cv = menon_conv_v(&r_g, h, w, &MENON_FIR_TAPS);
            let b_g_ch = menon_conv_h(&b_g, h, w, &MENON_FIR_TAPS);
            let b_g_cv = menon_conv_v(&b_g, h, w, &MENON_FIR_TAPS);
            for i in 0..n {
                if is_r[i] {
                    g[i] = r[i] - if use_h[i] { r_g_ch[i] } else { r_g_cv[i] };
                }
            }
            for i in 0..n {
                if is_b[i] {
                    g[i] = b[i] - if use_h[i] { b_g_ch[i] } else { b_g_cv[i] };
                }
            }

            let r_g2: Vec<f64> = (0..n).map(|i| r[i] - g[i]).collect();
            let b_g2: Vec<f64> = (0..n).map(|i| b[i] - g[i]).collect();
            let r_g2_cv = menon_conv_v(&r_g2, h, w, &MENON_KB_TAPS);
            let r_g2_ch = menon_conv_h(&r_g2, h, w, &MENON_KB_TAPS);
            for i in 0..n {
                if is_g[i] && is_b_row[i / w] {
                    r[i] = g[i] + r_g2_cv[i];
                }
            }
            for i in 0..n {
                if is_g[i] && is_b_col[i % w] {
                    r[i] = g[i] + r_g2_ch[i];
                }
            }

            let b_g2_cv = menon_conv_v(&b_g2, h, w, &MENON_KB_TAPS);
            let b_g2_ch = menon_conv_h(&b_g2, h, w, &MENON_KB_TAPS);
            for i in 0..n {
                if is_g[i] && is_r_row[i / w] {
                    b[i] = g[i] + b_g2_cv[i];
                }
            }
            for i in 0..n {
                if is_g[i] && is_r_col[i % w] {
                    b[i] = g[i] + b_g2_ch[i];
                }
            }

            let r_b: Vec<f64> = (0..n).map(|i| r[i] - b[i]).collect();
            let r_b_ch = menon_conv_h(&r_b, h, w, &MENON_FIR_TAPS);
            let r_b_cv = menon_conv_v(&r_b, h, w, &MENON_FIR_TAPS);
            for i in 0..n {
                if is_b[i] {
                    r[i] = b[i] + if use_h[i] { r_b_ch[i] } else { r_b_cv[i] };
                }
            }
            for i in 0..n {
                if is_r[i] {
                    b[i] = r[i] - if use_h[i] { r_b_ch[i] } else { r_b_cv[i] };
                }
            }
        }

        (r, g, b)
    });

    let mut out = vec![0f32; n * 3];
    out.par_chunks_mut(3).enumerate().for_each(|(i, px)| {
        px[0] = r_final[i] as f32;
        px[1] = g_final[i] as f32;
        px[2] = b_final[i] as f32;
    });

    let arr3 = numpy::ndarray::Array3::from_shape_vec((h, w, 3), out)
        .expect("shape mismatch building debayer_menon2007 output");
    Ok(arr3.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Joint (colour-space) bilateral filter -- replaces cv2.bilateralFilter
// ---------------------------------------------------------------------------
//
// Mirrors src/denoising.py's _bilateral_filter_numpy exactly: same
// mirror_idx boundary convention (matches its np.pad(mode='reflect'), which
// -- despite the name -- is numpy's non-edge-duplicating reflection, the
// same convention scipy calls 'mirror') for bit-exact native/numpy parity.
// The colour-similarity weight uses the joint Euclidean distance across all
// 3 channels per neighbour (matching cv2.bilateralFilter's multi-channel
// behaviour), not independent per-channel weights, so it doesn't introduce
// colour fringing at edges.

#[pyfunction]
fn bilateral_filter<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f32>,
    sigma_color: f64,
    sigma_space: f64,
    radius: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (h, w, c) = (s[0], s[1], s[2]);
    let owned: Vec<f32>;
    let flat: &[f32] = match arr.as_slice() {
        Some(sl) => sl,
        None => {
            owned = arr.iter().copied().collect();
            &owned
        }
    };

    let inv_2s2 = 1.0 / (2.0 * sigma_space * sigma_space);
    let inv_2c2 = 1.0 / (2.0 * sigma_color * sigma_color);
    let r = radius as isize;

    // Precompute the spatial weight table (radius is small, <=10) so the
    // per-pixel loop only evaluates the colour-distance exponential.
    let mut spatial_w = vec![0f64; (2 * radius + 1) * (2 * radius + 1)];
    for dy in -r..=r {
        for dx in -r..=r {
            let idx = ((dy + r) as usize) * (2 * radius + 1) + (dx + r) as usize;
            spatial_w[idx] = (-((dy * dy + dx * dx) as f64) * inv_2s2).exp();
        }
    }

    let mut out = vec![0f32; h * w * c];
    py.allow_threads(|| {
        out.par_chunks_mut(w * c).enumerate().for_each(|(y, row_out)| {
            let interior_y = y >= radius && y + radius < h;
            let mut center = [0f64; 8];
            let mut neighbor = [0f64; 8];
            for x in 0..w {
                let base = y * w * c + x * c;
                for ch in 0..c {
                    center[ch] = flat[base + ch] as f64;
                }
                let mut acc = [0f64; 8];
                let mut wsum = 0f64;
                let interior = interior_y && x >= radius && x + radius < w;
                for dy in -r..=r {
                    let yy = if interior { (y as isize + dy) as usize } else { mirror_idx(y as isize + dy, h) };
                    for dx in -r..=r {
                        let xx = if interior { (x as isize + dx) as usize } else { mirror_idx(x as isize + dx, w) };
                        let nbase = yy * w * c + xx * c;
                        let mut color_dist2 = 0f64;
                        for ch in 0..c {
                            let v = flat[nbase + ch] as f64;
                            neighbor[ch] = v;
                            let d = v - center[ch];
                            color_dist2 += d * d;
                        }
                        let sw = spatial_w[((dy + r) as usize) * (2 * radius + 1) + (dx + r) as usize];
                        let w_total = sw * (-color_dist2 * inv_2c2).exp();
                        for ch in 0..c {
                            acc[ch] += neighbor[ch] * w_total;
                        }
                        wsum += w_total;
                    }
                }
                let wsum = wsum.max(1e-12);
                for ch in 0..c {
                    row_out[x * c + ch] = (acc[ch] / wsum) as f32;
                }
            }
        });
    });

    let arr3 = numpy::ndarray::Array3::from_shape_vec((h, w, c), out)
        .expect("shape mismatch building bilateral_filter output");
    Ok(arr3.into_pyarray(py))
}

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
fn gram_matrix_wide<'py>(
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
    py.allow_threads(|| {
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
fn small_times_wide<'py>(
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
    let mut out = vec![0f64; n * p];
    py.allow_threads(|| {
        out.par_chunks_mut(p).enumerate().for_each(|(k, out_row)| {
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

#[pymodule]
fn astro_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sigma_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(online_sigma_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(online_sigma_clip_seed_burnin, m)?)?;
    m.add_function(wrap_pyfunction!(online_sigma_clip_fold_frame, m)?)?;
    m.add_function(wrap_pyfunction!(patch_weighted_sigma_combine, m)?)?;
    m.add_function(wrap_pyfunction!(median_combine, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(trimmed_mean_combine, m)?)?;
    m.add_function(wrap_pyfunction!(esd_combine, m)?)?;
    m.add_function(wrap_pyfunction!(linear_fit_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(ivw_combine, m)?)?;
    m.add_function(wrap_pyfunction!(dwt2_native, m)?)?;
    m.add_function(wrap_pyfunction!(idwt2_native, m)?)?;
    m.add_function(wrap_pyfunction!(sigma_clipped_median_native, m)?)?;
    m.add_function(wrap_pyfunction!(hot_pixel_box_replace_native, m)?)?;
    m.add_function(wrap_pyfunction!(blind_match_hypotheses, m)?)?;
    m.add_function(wrap_pyfunction!(dct2_ortho_native, m)?)?;
    m.add_function(wrap_pyfunction!(idct2_ortho_native, m)?)?;
    m.add_function(wrap_pyfunction!(bm3d_denoise_native, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine_lanczos3, m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_diffusion, m)?)?;
    m.add_function(wrap_pyfunction!(lacosmic_reject_native, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_native, m)?)?;
    m.add_function(wrap_pyfunction!(dbe_fit_surface, m)?)?;
    m.add_function(wrap_pyfunction!(dbe_sample_patches, m)?)?;
    m.add_function(wrap_pyfunction!(detect_stars_matched_filter, m)?)?;
    m.add_function(wrap_pyfunction!(fit_rigid_ransac, m)?)?;
    m.add_function(wrap_pyfunction!(debayer_malvar, m)?)?;
    m.add_function(wrap_pyfunction!(debayer_menon2007, m)?)?;
    m.add_function(wrap_pyfunction!(bilateral_filter, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine_kernel_table, m)?)?;
    m.add_function(wrap_pyfunction!(gram_matrix_wide, m)?)?;
    m.add_function(wrap_pyfunction!(small_times_wide, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
