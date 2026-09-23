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

/// Sibling of `row_parallel` for a `work` closure that produces two
/// per-pixel outputs instead of one (e.g. a combined value plus its own
/// summed weight) -- same gather-transpose driver, same tiling, just two
/// output buffers filled together instead of one. Kept separate from
/// `row_parallel` (rather than adding an output-count generic to it)
/// so every existing single-output caller's signature is untouched.
fn row_parallel_pair<S, Init, Work>(
    arr: &numpy::ndarray::ArrayView4<'_, f32>,
    h: usize,
    w: usize,
    c: usize,
    n: usize,
    init: Init,
    work: Work,
) -> (Vec<f32>, Vec<f32>)
where
    S: Send,
    Init: Fn() -> S + Sync,
    Work: Fn(&mut S, &[f32]) -> (f32, f32) + Sync,
{
    let row_len = w * c;
    let frame_len = h * row_len;
    let tile = gather_tile(n);
    let data: Option<&[f32]> = arr.as_slice();

    let mut out_a = vec![0f32; h * row_len];
    let mut out_b = vec![0f32; h * row_len];
    out_a.par_chunks_mut(row_len)
        .zip(out_b.par_chunks_mut(row_len))
        .enumerate()
        .for_each(|(row, (out_row_a, out_row_b))| {
            let mut state = init();
            let mut block = vec![0f32; tile * n];
            match data {
                Some(flat) => {
                    let row_base = row * row_len;
                    let mut start = 0usize;
                    while start < row_len {
                        let t = tile.min(row_len - start);
                        for k in 0..n {
                            let src = &flat[k * frame_len + row_base + start..][..t];
                            for (p, &v) in src.iter().enumerate() {
                                block[p * n + k] = v;
                            }
                        }
                        for p in 0..t {
                            let (a, b) = work(&mut state, &block[p * n..(p + 1) * n]);
                            out_row_a[start + p] = a;
                            out_row_b[start + p] = b;
                        }
                        start += t;
                    }
                }
                None => {
                    let vals = &mut block[..n];
                    for col in 0..row_len {
                        let wj = col / c;
                        let cj = col % c;
                        for k in 0..n {
                            vals[k] = arr[[k, row, wj, cj]];
                        }
                        let (a, b) = work(&mut state, &vals[..n]);
                        out_row_a[col] = a;
                        out_row_b[col] = b;
                    }
                }
            }
        });
    (out_a, out_b)
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
    let out = py.detach(|| {
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

    let out = py.detach(|| {
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

    let (mean_out, m2_out, nacc_out) = py.detach(|| {
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

    py.detach(|| {
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
    let out = py.detach(|| {
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
    let out = py.detach(|| {
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

    let out = py.detach(|| {
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

    let out = py.detach(|| {
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

    let out = py.detach(|| {
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

/// Sibling of `ivw_combine` that also returns the per-pixel-per-channel
/// summed inverse-variance weight (the Gauss-Markov estimator's own
/// variance is exactly `1/wsum`) -- backs `--uncertainty-map`
/// (`src/stacking.py`'s `ivw_combine(..., return_sigma=True)`), which
/// previously always fell back to the numpy tiled path because this
/// kernel's own `wsum` was never exposed. A separate function rather than
/// adding an output-mode flag to `ivw_combine` itself, so that already-
/// shipped, tested kernel's signature and call sites are untouched by an
/// opt-in diagnostic path. Necessarily duplicates `ivw_combine`'s per-pixel
/// weighting loop (there's no cheap way to share it while also returning a
/// second output through `row_parallel`'s single-output contract) --
/// accepted for the same reason this file keeps native kernels and their
/// numpy mirrors as separate, independently-readable implementations
/// rather than forcing a shared abstraction across a codepath boundary.
#[pyfunction]
#[pyo3(signature = (data, noise, sky=None, gain=None, weights=None))]
fn ivw_combine_with_sigma<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    noise: PyReadonlyArray1<'py, f32>,
    sky: Option<PyReadonlyArray1<'py, f32>>,
    gain: Option<f64>,
    weights: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray3<f32>>)> {
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

    let (result, wsum_out) = py.detach(|| {
        row_parallel_pair(&arr, h, w, c, n, || (), |_, vals| {
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
            let result = if wsum <= 0.0 { 0.0 } else { (acc / wsum) as f32 };
            (result, wsum as f32)
        })
    });
    let result_arr = numpy::ndarray::Array3::from_shape_vec((h, w, c), result)
        .expect("shape mismatch building ivw_combine_with_sigma result");
    let wsum_arr = numpy::ndarray::Array3::from_shape_vec((h, w, c), wsum_out)
        .expect("shape mismatch building ivw_combine_with_sigma wsum");
    Ok((result_arr.into_pyarray(py), wsum_arr.into_pyarray(py)))
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

    let (c_a, c_h, c_v, c_d) = py.detach(|| {
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

    let out = py.detach(|| {
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

#[pyfunction]
fn sigma_clipped_median_native(data: PyReadonlyArray1<'_, f32>, sigma: f64, iters: usize) -> f64 {
    let v: Vec<f32> = data.as_array().iter().copied().collect();
    sigma_clipped_median_f32(&v, sigma, iters)
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

    let out = py.detach(|| {
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

    let (best_r, best_t, best_inliers) = py.detach(|| {
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

/// Compute the 6 normalised Lanczos-3 tap weights for fractional offset `r`
/// (taps at floor-2 .. floor+3). Same arithmetic as the original inline loop.
#[inline]
fn lanczos6_weights(r: f64, out: &mut [f64; 6]) {
    // The taps sit at x_t = k_t - r for k_t = -2..=3, spaced exactly 1 apart, so
    // every sine the windowed-sinc needs follows from three evaluations by angle
    // addition instead of two sin() per tap (12 per call, two calls per output
    // pixel on a rotated warp -- ~24 trig calls a pixel, which was the whole cost:
    // 1.74 s for a 2048x3056x3 frame on one thread):
    //   sin(pi x_t)   = sin(pi k - pi r) = -(-1)^k sin(pi r)
    //   sin(pi x_t/3) = sin(pi k/3) cos(pi r/3) - cos(pi k/3) sin(pi r/3)
    // and lanczos3(x) = 3 sin(pi x) sin(pi x/3) / (pi x)^2 for 0 < |x| < 3 (1 at 0,
    // 0 beyond). That is the direct formula the old per-tap loop evaluated; the two
    // agree to ~1e-15 (verified against the previous kernel on a real rotated frame).
    if r == 0.0 {
        // x_2 == 0 -> weight 1 exactly; every other tap is an integer offset
        *out = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        return;
    }
    const SIN_K3: [f64; 6] = [
        -0.866_025_403_784_438_6, // sin(-2pi/3)
        -0.866_025_403_784_438_6, // sin(-pi/3)
        0.0,                      // sin(0)
        0.866_025_403_784_438_6,  // sin(pi/3)
        0.866_025_403_784_438_6,  // sin(2pi/3)
        0.0,                      // sin(pi)
    ];
    const COS_K3: [f64; 6] = [
        -0.5, // cos(-2pi/3)
        0.5,  // cos(-pi/3)
        1.0,  // cos(0)
        0.5,  // cos(pi/3)
        -0.5, // cos(2pi/3)
        -1.0, // cos(pi)
    ];
    let pi = std::f64::consts::PI;
    let s_r = (pi * r).sin();
    let (s3, c3) = (pi * r / 3.0).sin_cos();
    let mut s = 0.0;
    for t in 0..6 {
        let k = t as f64 - 2.0;
        let x = k - r;
        // (-1)^k for k = -2..=3
        let sign = if t % 2 == 0 { 1.0 } else { -1.0 };
        let sin_px = -sign * s_r;
        let sin_px3 = SIN_K3[t] * c3 - COS_K3[t] * s3;
        let w = if x.abs() >= 3.0 {
            0.0
        } else {
            let px = pi * x;
            (3.0 * sin_px * sin_px3) / (px * px)
        };
        out[t] = w;
        s += w;
    }
    if s != 0.0 {
        for v in out.iter_mut() {
            *v /= s;
        }
    }
}

// Weights from a cubic-interpolated table instead of the closed form (used on the general,
// non-separable path -- a rotation -- where every output pixel needs its own two weight sets,
// ~4 trig calls and ~24 divisions each). 4-point Lagrange over 512 cells: 99.985% of the
// float32 output of a 2048x3056 rotated warp is bit-identical to the closed form, the rest
// differs by one ulp (max abs 4.9e-4 on values up to ~6000), and it is 1.53x faster
// (638 -> 416 ms per frame, single thread). Sums stay exactly 1 (the Lagrange coefficients
// sum to 1). ORIGINSTACK_LANCZOS_EXACT=1 restores the closed form, for A/B checks.
const LUT_N: usize = 512;

fn lanczos_lut() -> &'static Vec<[f64; 6]> {
    static L: std::sync::OnceLock<Vec<[f64; 6]>> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        // nodes r = i/N for i = -1 ..= N+1 (table index i+1)
        (-1..=(LUT_N as isize + 1))
            .map(|i| {
                let mut w = [0f64; 6];
                if i == LUT_N as isize {
                    // r == 1: the tap at k = 1 sits exactly on the sample (x = 0, where the
                    // closed form is 0/0); the weights are a delta there
                    w[3] = 1.0;
                } else {
                    lanczos6_weights(i as f64 / LUT_N as f64, &mut w);
                }
                w
            })
            .collect()
    })
}

fn lut_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("ORIGINSTACK_LANCZOS_EXACT").map(|v| v != "1").unwrap_or(true))
}

#[inline]
fn lanczos6_weights_lut(r: f64, out: &mut [f64; 6]) {
    if r == 0.0 {
        *out = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        return;
    }
    let tab = lanczos_lut();
    let t = r * LUT_N as f64;
    let i = t.floor();
    let f = t - i;
    let j = i as usize; // nodes i-1..i+2 -> table indices j..j+3
    let cm1 = -f * (f - 1.0) * (f - 2.0) * (1.0 / 6.0);
    let c0 = (f + 1.0) * (f - 1.0) * (f - 2.0) * 0.5;
    let c1 = -(f + 1.0) * f * (f - 2.0) * 0.5;
    let c2 = (f + 1.0) * f * (f - 1.0) * (1.0 / 6.0);
    let (a, b, c, d) = (&tab[j], &tab[j + 1], &tab[j + 2], &tab[j + 3]);
    for k in 0..6 {
        out[k] = cm1 * a[k] + c0 * b[k] + c1 * c[k] + c2 * d[k];
    }
}

#[inline]
fn lw(r: f64, out: &mut [f64; 6]) {
    if lut_enabled() {
        lanczos6_weights_lut(r, out);
    } else {
        lanczos6_weights(r, out);
    }
}

/// One output row of the contiguous-input Lanczos-3 warp (the body of
/// `warp_affine_lanczos3`'s fast path, shared with `drizzle_accumulate_lanczos3`
/// so the two are the same arithmetic by construction).
#[allow(clippy::too_many_arguments)]
#[inline]
fn lanczos3_row_flat(
    img: &[f32],
    h: usize,
    w: usize,
    c: usize,
    oy: usize,
    out_row: &mut [f32],
    mat: [f64; 4],
    off: [f64; 2],
    tab: &Option<(Vec<[f64; 6]>, Vec<isize>)>,
    cval: f32,
    ox0: usize,
) {
    let (m00, m01, m10, m11) = (mat[0], mat[1], mat[2], mat[3]);
    let (o0, o1) = (off[0], off[1]);
    let out_w = out_row.len() / c;
    let mut wy = [0f64; 6];
    let mut wx = [0f64; 6];
                    let row_stride = w * c;
                    for ox in 0..out_w {
                        let (wyv, wxv, base_y, base_x): (&[f64; 6], &[f64; 6], isize, isize) =
                            if let Some((wxs, bxs)) = tab.as_ref() {
                                if ox == 0 {
                                    let iy = m00 * oy as f64 + o0;
                                    let fy = iy.floor();
                                    lw(iy - fy, &mut wy);
                                }
                                let iy = m00 * oy as f64 + o0;
                                (&wy, &wxs[ox], iy.floor() as isize - 2, bxs[ox])
                            } else {
                                let oxg = (ox + ox0) as f64;
                                let iy = m00 * oy as f64 + m01 * oxg + o0;
                                let ix = m10 * oy as f64 + m11 * oxg + o1;
                                let fy = iy.floor();
                                let fx = ix.floor();
                                lw(iy - fy, &mut wy);
                                lw(ix - fx, &mut wx);
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
                            if c == 3 {
                                // RGB (the pipeline's case): read each tap row's 18
                                // contiguous floats once and accumulate all three
                                // channels together, instead of three passes with
                                // stride-3 loads. Per channel the summation order is
                                // unchanged (taps in x, then rows in y), so the result
                                // is bit-identical to the generic loop below.
                                let (mut a0, mut a1, mut a2) = (0.0f64, 0.0f64, 0.0f64);
                                for ty in 0..6 {
                                    let base = (by + ty) * row_stride + bx * 3;
                                    let seg = &img[base..base + 18];
                                    let (mut r0, mut r1, mut r2) = (0.0f64, 0.0f64, 0.0f64);
                                    for tx in 0..6 {
                                        let wxt = wxv[tx];
                                        r0 += wxt * seg[tx * 3] as f64;
                                        r1 += wxt * seg[tx * 3 + 1] as f64;
                                        r2 += wxt * seg[tx * 3 + 2] as f64;
                                    }
                                    let wyt = wyv[ty];
                                    a0 += wyt * r0;
                                    a1 += wyt * r1;
                                    a2 += wyt * r2;
                                }
                                out_row[ox * 3] = a0 as f32;
                                out_row[ox * 3 + 1] = a1 as f32;
                                out_row[ox * 3 + 2] = a2 as f32;
                            } else {
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

/// Affine warp with Lanczos-3 resampling, matching scipy's
/// `affine_transform` sampling convention: `out[oy,ox] = in[M @ (oy,ox) + off]`,
/// out-of-bounds -> `cval`. `mat` is row-major 2x2 `[[m00,m01],[m10,m11]]`
/// mapping (row,col); `off` is `[off_row, off_col]`. All channels in one pass,
/// parallel across output rows.
#[pyfunction]
#[pyo3(signature = (data, mat, off, out_h, out_w, cval=0.0, origin=(0, 0)))]
fn warp_affine_lanczos3<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f32>,
    mat: [f64; 4],
    off: [f64; 2],
    out_h: usize,
    out_w: usize,
    cval: f32,
    origin: (usize, usize),
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    // `origin` = (row, col) of the output window's corner in the full output grid: output
    // pixel (oy, ox) is computed as full-grid pixel (oy + row, ox + col) with exactly the
    // arithmetic a full-frame warp uses there (integer offset first, then the f64 maths),
    // so a cropped warp equals the same crop of the full warp bit for bit.
    let (oy0, ox0) = origin;
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
            let ix = m11 * (ox + ox0) as f64 + o1;
            let fx = ix.floor();
            lanczos6_weights(ix - fx, &mut wxs[ox]);
            bxs[ox] = fx as isize - 2;
        }
        Some((wxs, bxs))
    } else {
        None
    };

    let mut out = vec![0f32; out_h * out_w * c];
    py.detach(|| {
        out.par_chunks_mut(out_w * c).enumerate().for_each(|(oy_l, out_row)| {
            let oy = oy_l + oy0;
            let mut wy = [0f64; 6];
            let mut wx = [0f64; 6];
            match (flat, &col_tab) {
                // ---- fast path: contiguous input ----
                (Some(img), tab) => {
                    lanczos3_row_flat(img, h, w, c, oy, out_row, [m00, m01, m10, m11], [o0, o1], tab, cval, ox0);
                }
                // ---- non-contiguous fallback: original indexed loop ----
                (None, _) => {
                    for ox in 0..out_w {
                        let oxg = (ox + ox0) as f64;
                        let iy = m00 * oy as f64 + m01 * oxg + o0;
                        let ix = m10 * oy as f64 + m11 * oxg + o1;
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
    py.detach(|| {
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

    py.detach(|| {
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
    py.detach(|| {
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

/// Interior of one output row of the 3x3 median: `row_out[1..w-1]` from the three input rows
/// starting at `r0`, `r1`, `r2`. The compare-exchange network is branchless min/max, so the
/// compiler vectorises it across `x` -- 4 lanes on baseline x86-64 (SSE2), 8 with AVX2.
#[inline(always)]
fn median3_interior_body(data: &[f32], r0: usize, r1: usize, r2: usize, w: usize, row_out: &mut [f32]) {
    for x in 1..w - 1 {
        let mut p = [
            data[r0 + x - 1], data[r0 + x], data[r0 + x + 1],
            data[r1 + x - 1], data[r1 + x], data[r1 + x + 1],
            data[r2 + x - 1], data[r2 + x], data[r2 + x + 1],
        ];
        row_out[x] = median9(&mut p);
    }
}

/// The same body compiled with AVX2 enabled. `#[target_feature]` applies to this function and to
/// everything inlined into it, so the shared body above is vectorised 8 wide here. Only ever
/// called after a run-time `is_x86_feature_detected!("avx2")`. Min/max are exact, so the result
/// is identical to the baseline build's.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn median3_interior_avx2(data: &[f32], r0: usize, r1: usize, r2: usize, w: usize, row_out: &mut [f32]) {
    median3_interior_body(data, r0, r1, r2, w, row_out)
}

fn median3_interior(r0: usize, r1: usize, r2: usize, w: usize, data: &[f32], row_out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU reports AVX2.
            unsafe { return median3_interior_avx2(data, r0, r1, r2, w, row_out) }
        }
    }
    median3_interior_body(data, r0, r1, r2, w, row_out)
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
            // Any other odd size (e.g. 9 and 17): contiguous row-segment
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
            median3_interior((y - 1) * w, y * w, (y + 1) * w, w, data, row_out);
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

    py.detach(|| {
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
    let out = py.detach(|| median_filter_2d_f32(flat, h, w, size));
    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), out)
        .unwrap()
        .into_pyarray(py))
}

/// `scipy.ndimage.gaussian_filter1d`'s exact 1D kernel: normalized samples of
/// the Gaussian PDF over `[-radius, radius]`, `radius = floor(truncate*sigma
/// + 0.5)` -- same formula scipy uses, order-0 (no derivative).
fn gaussian_kernel1d(sigma: f64, truncate: f64) -> Vec<f64> {
    let radius = (truncate * sigma + 0.5) as isize;
    let sigma2 = sigma * sigma;
    let mut w: Vec<f64> = (-radius..=radius)
        .map(|x| {
            let xf = x as f64;
            (-0.5 * xf * xf / sigma2).exp()
        })
        .collect();
    let sum: f64 = w.iter().sum();
    for v in w.iter_mut() {
        *v /= sum;
    }
    w
}

/// Separable Gaussian blur matching `scipy.ndimage.gaussian_filter(data,
/// sigma, mode='reflect')` (the default mode, and the only one any call site
/// in this codebase uses) for a scalar `sigma` on a 2D array. Two rayon-
/// parallel correlate1d-style passes (row-wise, then column-wise) instead of
/// scipy's generic N-D machinery -- found by profiling a real session:
/// `correlate1d` (the C function `gaussian_filter1d` calls per axis) was the
/// single largest self-time item in every profile taken this session, spread
/// across ~30 call sites project-wide (DBE, chroma denoising, local
/// contrast, structure-tensor coherence, star-mask generation, registration,
/// edge-band correction, and more). This kernel is wired into
/// `background.py`'s `gaussian_filter_ds` (already the single most-reused
/// choke point among those callers) as a first step, not a full sweep of
/// every direct `scipy.ndimage.gaussian_filter` call site -- each of those
/// has its own sigma/shape assumptions worth checking individually before
/// switching it over. Boundary handling reuses `reflect_idx` (scipy's
/// edge-duplicating `mode='reflect'`, not numpy's non-duplicating one).
#[pyfunction]
#[pyo3(signature = (data, sigma, truncate=4.0))]
fn gaussian_filter_native<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    sigma: f64,
    truncate: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = data.as_array();
    let s = arr.shape();
    let (h, w) = (s[0], s[1]);
    let owned: Vec<f64>;
    let flat: &[f64] = match arr.as_slice() {
        Some(sl) => sl,
        None => {
            owned = arr.iter().copied().collect();
            &owned
        }
    };

    if sigma <= 0.0 {
        let arr2 = numpy::ndarray::Array2::from_shape_vec((h, w), flat.to_vec())
            .expect("shape mismatch building gaussian_filter_native passthrough");
        return Ok(arr2.into_pyarray(py));
    }

    let kernel = gaussian_kernel1d(sigma, truncate);
    let radius = (kernel.len() / 2) as isize;

    let out = py.detach(|| {
        // Pass 1: blur along axis 1 (each row independently), parallel over rows.
        let mut tmp = vec![0f64; h * w];
        tmp.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
            let row = &flat[y * w..(y + 1) * w];
            for x in 0..w {
                let mut acc = 0.0f64;
                for (k, &kv) in kernel.iter().enumerate() {
                    let dx = k as isize - radius;
                    let xi = reflect_idx(x as isize + dx, w);
                    acc += kv * row[xi];
                }
                out_row[x] = acc;
            }
        });
        // Pass 2: blur along axis 0 (each column), parallel over output rows.
        let mut out = vec![0f64; h * w];
        out.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
            for x in 0..w {
                let mut acc = 0.0f64;
                for (k, &kv) in kernel.iter().enumerate() {
                    let dy = k as isize - radius;
                    let yi = reflect_idx(y as isize + dy, h);
                    acc += kv * tmp[yi * w + x];
                }
                out_row[x] = acc;
            }
        });
        out
    });

    let arr2 = numpy::ndarray::Array2::from_shape_vec((h, w), out)
        .expect("shape mismatch building gaussian_filter_native output");
    Ok(arr2.into_pyarray(py))
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

    let (surface, wrob) = py.detach(|| {
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

    let rows: Vec<Vec<(f64, f64, f64, f64)>> = py.detach(|| {
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

/// Shannon entropy of each sampled DBE patch's masked pixel values --
/// `_filter_sampled_patches`'s entropy-filter loop (`src/background.py`),
/// one native pass over `coords` (rayon-parallel) instead of a Python loop
/// calling `np.histogram` per patch. `dbe_sample_patches` above documents
/// the entropy filter as staying in Python because it's "cheap, operates on
/// the small per-patch result" -- true for the per-patch histogram call
/// itself, but under `--auto` (which sets `entropy_bg=True` for most target
/// types) it runs on every DBE pass regardless of frame count, and a real
/// profiled run found it costing 4.2s on its own: `DBE_MAX_SAMPLES` (4000)
/// candidate patches is enough Python-loop-plus-per-call-numpy-overhead to
/// add up even though each individual histogram is genuinely small.
///
/// `coords` is `(N, 2)` normalised `(row, col)` fractions, same convention
/// `dbe_sample_patches` returns and `_filter_sampled_patches` consumes:
/// `iy = clip(round(coord[0]*ny_g - 0.5), 0, ny_g-1)` locates the patch's
/// grid cell, whose pixel bounds are then `[round(iy*cell_h), round((iy+1)*
/// cell_h))` -- reproduced here exactly, not re-derived, so a coordinate
/// convention change in the sampler doesn't silently desync the two.
/// Histogram binning uses the same `(value - min) / range * n_bins`
/// uniform-bin formula `np.histogram`'s fast path computes for
/// `range=(mn, mx)` -- but not the edge-correction pass numpy adds after it
/// (comparing each value against the *actual* `linspace` bin edges and
/// nudging the index by one where float rounding put it a bin off), so an
/// occasional value within ~1 ULP of a bin boundary lands one bin over from
/// numpy's answer. Measured against the Python reference on a real-shaped
/// synthetic case: entropy values agree to ~3e-4 absolute (values are
/// O(1), so this is a few parts in 10,000), not bit-exact. Immaterial for
/// what this feeds -- a median+MAD outlier threshold over thousands of
/// patches -- so the edge-correction pass wasn't worth porting.
#[pyfunction]
#[pyo3(signature = (channel, emission_mask, coords, patch_size, n_bins=16))]
fn patch_entropy_batch<'py>(
    py: Python<'py>,
    channel: PyReadonlyArray2<'py, f32>,
    emission_mask: PyReadonlyArray2<'py, f32>,
    coords: PyReadonlyArray2<'py, f64>,
    patch_size: usize,
    n_bins: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ch = channel.as_array();
    let em = emission_mask.as_array();
    let (h, w) = (ch.shape()[0], ch.shape()[1]);
    if em.shape() != [h, w] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "channel and emission_mask must have the same shape",
        ));
    }
    let ch_flat: Option<&[f32]> = ch.as_slice();
    let em_flat: Option<&[f32]> = em.as_slice();
    let (ch_flat, em_flat) = match (ch_flat, em_flat) {
        (Some(c), Some(e)) => (c, e),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "channel and emission_mask must be C-contiguous",
            ))
        }
    };
    let coords_arr = coords.as_array();
    if coords_arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("coords must be (N, 2)"));
    }
    let coords_flat: &[f64] = coords_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("coords must be C-contiguous"))?;
    let n = coords_arr.shape()[0];
    let patch_size = patch_size.max(1);
    let ny_g = (h / patch_size).max(1);
    let nx_g = (w / patch_size).max(1);
    let cell_h = h as f64 / ny_g as f64;
    let cell_w = w as f64 / nx_g as f64;

    let mut out = vec![0f64; n];
    py.detach(|| {
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let cy = coords_flat[i * 2];
            let cx = coords_flat[i * 2 + 1];
            let iy = ((cy * ny_g as f64 - 0.5).round() as isize).clamp(0, ny_g as isize - 1) as usize;
            let ix = ((cx * nx_g as f64 - 0.5).round() as isize).clamp(0, nx_g as isize - 1) as usize;
            let y0 = (iy as f64 * cell_h).round() as usize;
            let y1 = (((iy + 1) as f64 * cell_h).round() as usize).min(h);
            let x0 = (ix as f64 * cell_w).round() as usize;
            let x1 = (((ix + 1) as f64 * cell_w).round() as usize).min(w);

            // f64 throughout the binning arithmetic, matching the numpy
            // reference exactly: _patch_entropy's `mn, mx = float(pixels.min()),
            // float(pixels.max())` casts to Python float (f64) before calling
            // np.histogram, even though `pixels` itself is f32 -- numpy's
            // uniform-bin fast path then promotes the per-element bin-index
            // arithmetic to f64 too. Using f32 here would still be "close" but
            // not actually bit-exact.
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            let mut count = 0usize;
            for y in y0..y1 {
                let row_base = y * w;
                for x in x0..x1 {
                    if em_flat[row_base + x] < 0.5 {
                        let v = ch_flat[row_base + x] as f64;
                        if v < mn {
                            mn = v;
                        }
                        if v > mx {
                            mx = v;
                        }
                        count += 1;
                    }
                }
            }
            if count < 4 {
                *o = 0.0;
                return;
            }
            let range = mx - mn;
            if range < 1e-12 {
                *o = 0.0;
                return;
            }
            let mut counts = vec![0u32; n_bins];
            for y in y0..y1 {
                let row_base = y * w;
                for x in x0..x1 {
                    if em_flat[row_base + x] < 0.5 {
                        let v = ch_flat[row_base + x] as f64;
                        let mut idx = (((v - mn) / range) * n_bins as f64) as usize;
                        if idx >= n_bins {
                            idx = n_bins - 1;
                        }
                        counts[idx] += 1;
                    }
                }
            }
            let total = count as f64;
            let mut ent = 0.0f64;
            for &c in &counts {
                if c > 0 {
                    let p = c as f64 / (total + 1e-12);
                    ent -= p * p.log2();
                }
            }
            *o = ent;
        });
    });

    Ok(out.into_pyarray(py))
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

    let rows: Vec<[f64; 10]> = py.detach(|| {
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

    let (best_params, best_inliers) = py.detach(|| {
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
    py.detach(|| {
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

    let (r_final, g_final, b_final) = py.detach(|| {
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
    py.detach(|| {
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
fn robust_pca_pre_svd_input<'py>(
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
fn robust_pca_iterate<'py>(
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
fn continuum_scale_moments(
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

// ---------------------------------------------------------------------------
// Moffat wing fit (src/star_repair.py's `_fit_moffat_wing`)
// ---------------------------------------------------------------------------
//
// scipy.optimize.curve_fit's bounded trust-region solver calls back into the
// Python Moffat model every iteration -- with up to 800 saturated stars x 3
// channels per postprocess run, that's tens of thousands of tiny Python
// callbacks, not a large-array cost. This is a from-scratch Levenberg-
// Marquardt fit with the Moffat model and its analytic Jacobian both inlined
// (no numerical diff, no callback), box constraints enforced by clamping
// each proposed step into [lower, upper] before evaluating it. Not a port of
// scipy's TRF algorithm -- a different (simpler) bounded LM -- so parity
// against scipy is judged by both recovering the same synthetic
// ground-truth Moffat parameters within tolerance (tests/test_native.py),
// not bit-exact agreement with curve_fit's own iterate path.

/// Moffat value + analytic partials at squared radius `r2`.
/// I(r) = amp * (1 + (r/alpha)^2)^-beta ; returns (I, dI/damp, dI/dalpha, dI/dbeta).
#[inline]
fn moffat_eval(r2: f64, amp: f64, alpha: f64, beta: f64) -> (f64, f64, f64, f64) {
    let u = r2 / (alpha * alpha);
    let base = 1.0 + u;
    let pw = base.powf(-beta);
    let i = amp * pw;
    let d_amp = pw;
    let d_alpha = 2.0 * beta * u / (alpha * base) * i;
    let d_beta = -i * base.ln();
    (i, d_amp, d_alpha, d_beta)
}

/// Cramer's-rule solve of a 3x3 linear system; `None` on a near-singular matrix.
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det3 = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let det = det3(&a);
    if det.abs() < 1e-18 {
        return None;
    }
    let mut x = [0.0f64; 3];
    for k in 0..3 {
        let mut m = a;
        for row in 0..3 {
            m[row][k] = b[row];
        }
        x[k] = det3(&m) / det;
    }
    Some(x)
}

/// Bounded Levenberg-Marquardt fit of the Moffat model to (r, v) samples.
/// Mirrors `_fit_moffat_wing`'s p0/bounds construction exactly; `None` when
/// there are too few samples or the peak is non-positive (same early-outs
/// as the Python reference).
fn fit_moffat_lm(r: &[f64], v: &[f64]) -> Option<(f64, f64, f64)> {
    let n = r.len();
    if n < 6 {
        return None;
    }
    let amp0 = v.iter().cloned().fold(f64::MIN, f64::max);
    if !(amp0 > 0.0) {
        return None;
    }
    let mut rs: Vec<f64> = r.to_vec();
    let alpha0 = median_f64_scratch(&mut rs).max(1.0);

    let lower = [amp0 * 0.5, 0.5, 1.0];
    let upper = [amp0 * 50.0, 50.0, 8.0];
    let mut p = [amp0 * 2.0, alpha0, 2.5];
    for k in 0..3 {
        p[k] = p[k].clamp(lower[k], upper[k]);
    }

    let cost = |p: &[f64; 3]| -> f64 {
        let mut s = 0.0;
        for i in 0..n {
            let (val, ..) = moffat_eval(r[i] * r[i], p[0], p[1], p[2]);
            let res = val - v[i];
            s += res * res;
        }
        s
    };

    let mut lambda = 1e-3;
    let mut c = cost(&p);
    for _outer in 0..100 {
        let mut jtj = [[0.0f64; 3]; 3];
        let mut jtr = [0.0f64; 3];
        for i in 0..n {
            let (val, d_amp, d_alpha, d_beta) = moffat_eval(r[i] * r[i], p[0], p[1], p[2]);
            let res = val - v[i];
            let j = [d_amp, d_alpha, d_beta];
            for a in 0..3 {
                jtr[a] += j[a] * res;
                for b in 0..3 {
                    jtj[a][b] += j[a] * j[b];
                }
            }
        }
        let mut improved = false;
        for _try in 0..8 {
            let mut a = jtj;
            for d in 0..3 {
                a[d][d] += lambda * jtj[d][d].max(1e-12);
            }
            let neg_jtr = [-jtr[0], -jtr[1], -jtr[2]];
            let delta = match solve3(a, neg_jtr) {
                Some(d) => d,
                None => {
                    lambda *= 10.0;
                    continue;
                }
            };
            let mut cand = p;
            for k in 0..3 {
                cand[k] = (p[k] + delta[k]).clamp(lower[k], upper[k]);
            }
            let cand_cost = cost(&cand);
            if cand_cost < c {
                p = cand;
                c = cand_cost;
                lambda = (lambda * 0.3).max(1e-12);
                improved = true;
                break;
            } else {
                lambda *= 10.0;
            }
        }
        if !improved {
            break;
        }
    }
    if p.iter().all(|x| x.is_finite()) {
        Some((p[0], p[1], p[2]))
    } else {
        None
    }
}

#[pyfunction]
fn fit_moffat_native(
    r: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
) -> PyResult<Option<(f64, f64, f64)>> {
    let r = r.as_slice()?;
    let v = v.as_slice()?;
    if r.len() != v.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r and v must have the same length",
        ));
    }
    Ok(fit_moffat_lm(r, v))
}

// ---------------------------------------------------------------------------
// 2D PSF star fit (src/psf_deconvolution.py's `estimate_psf` per-star curve_fit)
// ---------------------------------------------------------------------------
//
// estimate_psf fits a 2D Moffat (default) or Gaussian to each of up to
// RL_PSF_MAX_STARS bright, unsaturated star cutouts (~31x31 px) via
// scipy.optimize.curve_fit -- whose bounded solver evaluates the Python model
// (plus numerical differencing) on ~961 points every trust-region iteration.
// Same pattern and fix as fit_moffat_native above, one dimension up: a
// from-scratch bounded Levenberg-Marquardt with each model and its analytic
// Jacobian inlined, box constraints enforced by clamping the step. Not a port
// of curve_fit's TRF path -- parity is judged by both recovering the same
// synthetic ground-truth params within tolerance (tests/test_native.py).

/// Gaussian elimination with partial pivoting for a small dense NxN system.
/// `None` on a (near-)singular matrix.
fn solve_lin<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for row in (col + 1)..N {
            let m = a[row][col].abs();
            if m > best {
                best = m;
                piv = row;
            }
        }
        if best < 1e-18 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for row in (col + 1)..N {
            let f = a[row][col] / d;
            if f != 0.0 {
                for k in col..N {
                    a[row][k] -= f * a[col][k];
                }
                b[row] -= f * b[col];
            }
        }
    }
    let mut x = [0.0f64; N];
    for i in (0..N).rev() {
        let mut s = b[i];
        for k in (i + 1)..N {
            s -= a[i][k] * x[k];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// Bounded LM shared by the two 2D-PSF models. `resid_jac(p, i)` returns
/// `(model(p, i) - z[i], d model / d p)` for sample `i`. Same outer-loop /
/// lambda schedule as `fit_moffat_lm`.
fn psf_lm<const N: usize, F>(
    n: usize,
    lower: [f64; N],
    upper: [f64; N],
    p0: [f64; N],
    resid_jac: F,
) -> Option<[f64; N]>
where
    F: Fn(&[f64; N], usize) -> (f64, [f64; N]),
{
    let mut p = p0;
    for k in 0..N {
        p[k] = p[k].clamp(lower[k], upper[k]);
    }
    let cost = |p: &[f64; N]| -> f64 {
        let mut s = 0.0;
        for i in 0..n {
            let (r, _) = resid_jac(p, i);
            s += r * r;
        }
        s
    };
    let mut lambda = 1e-3f64;
    let mut c = cost(&p);
    for _outer in 0..100 {
        let mut jtj = [[0.0f64; N]; N];
        let mut jtr = [0.0f64; N];
        for i in 0..n {
            let (r, j) = resid_jac(&p, i);
            for a in 0..N {
                jtr[a] += j[a] * r;
                for b in 0..N {
                    jtj[a][b] += j[a] * j[b];
                }
            }
        }
        let mut improved = false;
        for _try in 0..8 {
            let mut aug = jtj;
            for d in 0..N {
                aug[d][d] += lambda * jtj[d][d].max(1e-12);
            }
            let mut neg = [0.0f64; N];
            for d in 0..N {
                neg[d] = -jtr[d];
            }
            let delta = match solve_lin(aug, neg) {
                Some(d) => d,
                None => {
                    lambda *= 10.0;
                    continue;
                }
            };
            let mut cand = p;
            for k in 0..N {
                cand[k] = (p[k] + delta[k]).clamp(lower[k], upper[k]);
            }
            let cand_cost = cost(&cand);
            if cand_cost < c {
                p = cand;
                c = cand_cost;
                lambda = (lambda * 0.3).max(1e-12);
                improved = true;
                break;
            } else {
                lambda *= 10.0;
            }
        }
        if !improved {
            break;
        }
    }
    if p.iter().all(|x| x.is_finite()) {
        Some(p)
    } else {
        None
    }
}

/// 2D Moffat `A*(1 + ((x-x0)^2+(y-y0)^2)/alpha^2)^-beta + bg`,
/// params `[amp, x0, y0, alpha, beta, bg]`; returns (value, partials).
#[inline]
fn moffat2d_eval(x: f64, y: f64, p: &[f64; 6]) -> (f64, [f64; 6]) {
    let (amp, x0, y0, alpha, beta, bg) = (p[0], p[1], p[2], p[3], p[4], p[5]);
    let dx = x - x0;
    let dy = y - y0;
    let r2 = dx * dx + dy * dy;
    let a2 = alpha * alpha;
    let base = 1.0 + r2 / a2;
    let pw = base.powf(-beta);
    let sig = amp * pw;
    let g = sig * beta / base;
    (
        sig + bg,
        [
            pw,                             // d/d amp
            2.0 * dx / a2 * g,              // d/d x0
            2.0 * dy / a2 * g,              // d/d y0
            2.0 * r2 / (a2 * alpha) * g,    // d/d alpha  (= 2 r2 / alpha^3 * g)
            -sig * base.ln(),               // d/d beta
            1.0,                            // d/d bg
        ],
    )
}

/// 2D Gaussian `A*exp(-((x-x0)^2+(y-y0)^2)/(2 sigma^2)) + bg`,
/// params `[amp, x0, y0, sigma, bg]`; returns (value, partials).
#[inline]
fn gauss2d_eval(x: f64, y: f64, p: &[f64; 5]) -> (f64, [f64; 5]) {
    let (amp, x0, y0, sigma, bg) = (p[0], p[1], p[2], p[3], p[4]);
    let dx = x - x0;
    let dy = y - y0;
    let r2 = dx * dx + dy * dy;
    let s2 = sigma * sigma;
    let e = (-r2 / (2.0 * s2)).exp();
    let sig = amp * e;
    (
        sig + bg,
        [
            e,                        // d/d amp
            sig * dx / s2,            // d/d x0
            sig * dy / s2,            // d/d y0
            sig * r2 / (s2 * sigma),  // d/d sigma  (= r2 / sigma^3 * sig)
            1.0,                      // d/d bg
        ],
    )
}

/// Fit a 2D Moffat to a row-major `sz*sz` star cutout. `peak`/`bg` seed the
/// same p0/bounds `estimate_psf` builds for curve_fit. Returns
/// `(amp, x0, y0, alpha, beta, bg)` or `None`.
#[pyfunction]
fn fit_psf_moffat2d_native(
    z: PyReadonlyArray1<f64>,
    sz: usize,
    peak: f64,
    bg: f64,
) -> PyResult<Option<(f64, f64, f64, f64, f64, f64)>> {
    let z = z.as_slice()?;
    if sz < 3 || z.len() != sz * sz {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "z length must equal sz*sz with sz >= 3",
        ));
    }
    if !(peak > bg) || !peak.is_finite() || !bg.is_finite() {
        return Ok(None);
    }
    let center = sz as f64 / 2.0;
    let lower = [0.0, center - 3.0, center - 3.0, 0.5, 1.0, 0.0];
    let upper = [peak * 2.0, center + 3.0, center + 3.0, 20.0, 10.0, peak];
    let p0 = [peak - bg, center, center, 2.0, 3.0, bg];
    let res = psf_lm::<6, _>(z.len(), lower, upper, p0, |p, i| {
        let x = (i % sz) as f64;
        let y = (i / sz) as f64;
        let (val, j) = moffat2d_eval(x, y, p);
        (val - z[i], j)
    });
    Ok(res.map(|p| (p[0], p[1], p[2], p[3], p[4], p[5])))
}

/// Fit a 2D Gaussian to a row-major `sz*sz` star cutout.
/// Returns `(amp, x0, y0, sigma, bg)` or `None`.
#[pyfunction]
fn fit_psf_gauss2d_native(
    z: PyReadonlyArray1<f64>,
    sz: usize,
    peak: f64,
    bg: f64,
) -> PyResult<Option<(f64, f64, f64, f64, f64)>> {
    let z = z.as_slice()?;
    if sz < 3 || z.len() != sz * sz {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "z length must equal sz*sz with sz >= 3",
        ));
    }
    if !(peak > bg) || !peak.is_finite() || !bg.is_finite() {
        return Ok(None);
    }
    let center = sz as f64 / 2.0;
    let lower = [0.0, center - 3.0, center - 3.0, 0.3, 0.0];
    let upper = [peak * 2.0, center + 3.0, center + 3.0, 20.0, peak];
    let p0 = [peak - bg, center, center, 2.0, bg];
    let res = psf_lm::<5, _>(z.len(), lower, upper, p0, |p, i| {
        let x = (i % sz) as f64;
        let y = (i / sz) as f64;
        let (val, j) = gauss2d_eval(x, y, p);
        (val - z[i], j)
    });
    Ok(res.map(|p| (p[0], p[1], p[2], p[3], p[4])))
}

// ---------------------------------------------------------------------------
// Background mesh median grid (src/background.py's `_process_channel`)
// ---------------------------------------------------------------------------
//
// Unconditional on the default pipeline (background extraction is on by
// default, and this exact loop shape also runs as a DBE/wavelet-BG fallback
// whenever patch sampling comes up short, plus the legacy `--bg-method
// mesh` path) -- called 9x per stack (1 broad + 2 fine passes x 3
// channels). Per-cell `np.median` call overhead dominates at the small cell
// sizes involved (fine pass is order ~700 cells/channel), not per-cell
// compute; this fuses the whole (ny, nx) grid into one native pass,
// parallel over cells.

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn mesh_median_grid<'py>(
    py: Python<'py>,
    channel: PyReadonlyArray2<'py, f64>,
    star_mask: PyReadonlyArray2<'py, f32>,
    has_star_mask: bool,
    cell_excluded: PyReadonlyArray2<'py, u8>,
    ny: usize,
    nx: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let channel = channel.as_array();
    let mask = star_mask.as_array();
    let excluded = cell_excluded.as_array();
    let (h, w) = (channel.shape()[0], channel.shape()[1]);

    let results: Vec<f64> = (0..ny * nx)
        .into_par_iter()
        .map(|idx| {
            let iy = idx / nx;
            let ix = idx % nx;
            if excluded[[iy, ix]] != 0 {
                return f64::NAN;
            }
            let y0 = ((iy as f64) * (h as f64) / (ny as f64)).round() as i64;
            let y1 = ((((iy + 1) as f64) * (h as f64) / (ny as f64)).round() as i64).min(h as i64);
            let x0 = ((ix as f64) * (w as f64) / (nx as f64)).round() as i64;
            let x1 = ((((ix + 1) as f64) * (w as f64) / (nx as f64)).round() as i64).min(w as i64);
            if y1 <= y0 || x1 <= x0 {
                return f64::NAN;
            }
            let (y0, y1, x0, x1) = (y0 as usize, y1 as usize, x0 as usize, x1 as usize);
            let mut cell: Vec<f64> = Vec::with_capacity((y1 - y0) * (x1 - x0));
            for y in y0..y1 {
                for x in x0..x1 {
                    if has_star_mask && mask[[y, x]] >= 0.5 {
                        continue;
                    }
                    cell.push(channel[[y, x]]);
                }
            }
            if cell.is_empty() {
                f64::NAN
            } else {
                median_f64_scratch(&mut cell)
            }
        })
        .collect();

    Ok(numpy::ndarray::Array2::from_shape_vec((ny, nx), results)
        .unwrap()
        .into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Local-normalization coarse background grid (src/local_normalize.py's
// `_coarse_background`)
// ---------------------------------------------------------------------------
//
// Called once per frame (N_frames x grid^2 small `np.percentile` calls in
// the numpy reference -- e.g. a 200-frame session x 24^2 grid = ~115k tiny
// percentile calls per `--local-normalize` run). Fuses the whole (grid,
// grid, C) grid into one native pass per frame, parallel over cells.

#[pyfunction]
fn local_normalize_grid<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray3<'py, f32>,
    grid: usize,
    pct: f64,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let frame = frame.as_array();
    let (h, w, c) = (frame.shape()[0], frame.shape()[1], frame.shape()[2]);

    let ys: Vec<usize> = (0..=grid)
        .map(|i| {
            if i == grid {
                h
            } else {
                (((i as f64) * (h as f64) / (grid as f64)) as usize).min(h)
            }
        })
        .collect();
    let xs: Vec<usize> = (0..=grid)
        .map(|i| {
            if i == grid {
                w
            } else {
                (((i as f64) * (w as f64) / (grid as f64)) as usize).min(w)
            }
        })
        .collect();

    let results: Vec<f32> = (0..grid * grid)
        .into_par_iter()
        .flat_map(|idx| {
            let iy = idx / grid;
            let ix = idx % grid;
            let y0 = ys[iy];
            let y1 = (ys[iy] + 1).max(ys[iy + 1]).min(h);
            let x0 = xs[ix];
            let x1 = (xs[ix] + 1).max(xs[ix + 1]).min(w);
            let mut out = vec![0.0f32; c];
            let mut scratch: Vec<f32> = Vec::with_capacity((y1 - y0) * (x1 - x0));
            for ch in 0..c {
                scratch.clear();
                for y in y0..y1 {
                    for x in x0..x1 {
                        scratch.push(frame[[y, x, ch]]);
                    }
                }
                scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                out[ch] = percentile_sorted(&scratch, pct);
            }
            out
        })
        .collect();

    Ok(numpy::ndarray::Array3::from_shape_vec((grid, grid, c), results)
        .unwrap()
        .into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Star-disk mask stamping (src/star_removal.py's `build_star_mask`)
// ---------------------------------------------------------------------------
//
// On by default (`--no-remove-stars` to disable); rich star fields hit up to
// 4000 small per-star Python iterations building the mask. Parallel over
// output rows rather than stars (a shared mutable mask ruled out per-star
// parallelism) -- each row scans every star's precomputed integer bounding
// box (a cheap O(1) reject before the per-pixel test), then applies the
// exact same disk-membership test the numpy reference uses,
// `(y - cy)^2 + (x - cx)^2 <= r^2`, so results match bit-for-bit. Also
// returns the max radius actually stamped (over stars whose bbox was
// non-empty), matching the Python reference's `max_r_used` bookkeeping.

#[pyfunction]
fn stamp_star_disks<'py>(
    py: Python<'py>,
    h: usize,
    w: usize,
    cy: PyReadonlyArray1<'py, f64>,
    cx: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<u8>>, f64)> {
    let cy = cy.as_slice()?;
    let cx = cx.as_slice()?;
    let r = r.as_slice()?;
    let n = cy.len();

    let mut max_r_used = 0.0f64;
    let bbox: Vec<(usize, usize, usize, usize)> = (0..n)
        .map(|i| {
            let y0f = (cy[i] - r[i]).trunc();
            let y0 = if y0f < 0.0 { 0usize } else { (y0f as usize).min(h) };
            let y1f = (cy[i] + r[i]).trunc() + 1.0;
            let y1 = if y1f < 0.0 { 0usize } else { (y1f as usize).min(h) };
            let x0f = (cx[i] - r[i]).trunc();
            let x0 = if x0f < 0.0 { 0usize } else { (x0f as usize).min(w) };
            let x1f = (cx[i] + r[i]).trunc() + 1.0;
            let x1 = if x1f < 0.0 { 0usize } else { (x1f as usize).min(w) };
            if y1 > y0 && x1 > x0 && r[i] > max_r_used {
                max_r_used = r[i];
            }
            (y0, y1, x0, x1)
        })
        .collect();

    let mut data = vec![0u8; h * w];
    data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for i in 0..n {
            let (y0, y1, x0, x1) = bbox[i];
            if y < y0 || y >= y1 {
                continue;
            }
            let dy = y as f64 - cy[i];
            let dy2 = dy * dy;
            let rr2 = r[i] * r[i];
            if dy2 > rr2 {
                continue;
            }
            for x in x0..x1 {
                let dx = x as f64 - cx[i];
                if dy2 + dx * dx <= rr2 {
                    row[x] = 1;
                }
            }
        }
    });

    let arr = numpy::ndarray::Array2::from_shape_vec((h, w), data)
        .unwrap()
        .into_pyarray(py);
    Ok((arr, max_r_used))
}

// ---------------------------------------------------------------------------
// Bresenham line rasterization (src/trail_reject.py's `_bresenham_line`)
// ---------------------------------------------------------------------------
//
// Textbook zero-vectorization case: a sequential line-walk with pure scalar
// integer arithmetic, no numpy call it could hide behind. Direct port of
// the existing (skimage-validated) symmetric error-term formulation, so
// results match integer-for-integer.

#[pyfunction]
fn bresenham_line_native<'py>(
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
// Radial-bin median profile (src/denoising.py's `radial_renormalize`,
// comet-mode only)
// ---------------------------------------------------------------------------
//
// The numpy reference rebuilds a boolean mask over the *whole* image once
// per bin (n_bins full-image passes) just to select each bin's pixels
// before taking the median -- wasteful, since bin membership only needs one
// O(H*W) bucket-assignment pass. This does exactly that: one pass buckets
// every pixel's channel value by its radial bin (matching the numpy
// reference's half-open `[edge[b], edge[b+1])` binning), then a native
// quickselect median per bucket, parallel over bins.

#[pyfunction]
fn radial_bin_median<'py>(
    py: Python<'py>,
    radii: PyReadonlyArray2<'py, f64>,
    channel: PyReadonlyArray2<'py, f64>,
    max_radius: f64,
    n_bins: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let radii = radii.as_array();
    let channel = channel.as_array();
    let (h, w) = (radii.shape()[0], radii.shape()[1]);
    let bin_width = (max_radius + 1.0) / n_bins as f64;

    let mut buckets: Vec<Vec<f64>> = (0..n_bins).map(|_| Vec::new()).collect();
    for y in 0..h {
        for x in 0..w {
            let rad = radii[[y, x]];
            let b = ((rad / bin_width) as isize).clamp(0, n_bins as isize - 1) as usize;
            buckets[b].push(channel[[y, x]]);
        }
    }

    let profile: Vec<f64> = buckets
        .into_par_iter()
        .map(|mut v| if v.is_empty() { 0.0 } else { median_f64_scratch(&mut v) })
        .collect();

    Ok(profile.into_pyarray(py))
}

/// Batch circular-aperture photometry with partial-pixel apertures.
///
/// For each of N star centres, integrates an image `(H, W, C)` inside a
/// circle of radius `r_ap` pixels, estimates the local sky from a robust
/// median over the `(r_in, r_out]` annulus, and returns per channel the
/// background-subtracted flux, the sky level, the robust sky sigma
/// (`1.4826 * MAD`), the raw max pixel value inside the aperture (for
/// saturation flags), plus the effective aperture pixel area (shared by
/// all channels).
///
/// Aperture-edge pixels are weighted by the fraction of their unit cell
/// inside the circle, estimated by `subpix`^2 supersampling; a pixel more
/// than `sqrt(0.5)` inside/outside the edge is taken as fully in / fully
/// out with no supersampling. Integer pixel coordinates are pixel centres,
/// matching this project's `_aperture_flux` / star-detection convention. A
/// star whose full `r_out` disk is not inside the frame gets an all-NaN
/// row (the caller decides what a missing point means).
///
/// Hot path for time-series photometry: N stars x M frames aperture
/// measurements, each otherwise a Python-level masked reduction. Parallel
/// over stars (each writes its own output rows). Numpy mirror:
/// `_aperture_photometry_batch_numpy` in `src/photometry.py`.
#[pyfunction]
#[pyo3(signature = (img, xs, ys, r_ap, r_in, r_out, subpix=4))]
fn aperture_photometry_batch<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, f32>,
    xs: PyReadonlyArray1<'py, f64>,
    ys: PyReadonlyArray1<'py, f64>,
    r_ap: f64,
    r_in: f64,
    r_out: f64,
    subpix: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let img_arr = img.as_array();
    let (h, w, c) = (img_arr.shape()[0], img_arr.shape()[1], img_arr.shape()[2]);
    let img_flat = img_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("img must be C-contiguous (H, W, C)")
    })?;
    let xs = xs.as_slice()?;
    let ys = ys.as_slice()?;
    let n = xs.len();
    if ys.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xs and ys must match in length",
        ));
    }
    if subpix == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("subpix must be >= 1"));
    }
    if !(r_ap > 0.0) || r_in < r_ap || r_out <= r_in {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "need 0 < r_ap <= r_in < r_out",
        ));
    }

    let sub = subpix;
    let sub_area = 1.0 / (sub * sub) as f64;
    let sub_off: Vec<f64> = (0..sub)
        .map(|k| (k as f64 + 0.5) / sub as f64 - 0.5)
        .collect();
    const HALF_DIAG: f64 = 0.7071067811865476; // sqrt(0.5)
    let r_ap2 = r_ap * r_ap;
    let r_in2 = r_in * r_in;
    let r_out2 = r_out * r_out;
    let full_in = if r_ap > HALF_DIAG {
        (r_ap - HALF_DIAG).powi(2)
    } else {
        -1.0
    };
    let full_out = (r_ap + HALF_DIAG).powi(2);

    let mut flux = vec![f64::NAN; n * c];
    let mut sky = vec![f64::NAN; n * c];
    let mut sky_sig = vec![f64::NAN; n * c];
    let mut peak = vec![f64::NAN; n * c];
    let mut area = vec![f64::NAN; n];

    py.detach(|| {
        flux.par_chunks_mut(c)
            .zip(sky.par_chunks_mut(c))
            .zip(sky_sig.par_chunks_mut(c))
            .zip(peak.par_chunks_mut(c))
            .zip(area.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((((frow, srow), sgrow), prow), arow))| {
                let cx = xs[i];
                let cy = ys[i];
                if !cx.is_finite()
                    || !cy.is_finite()
                    || cx - r_out < 0.0
                    || cx + r_out >= (w - 1) as f64
                    || cy - r_out < 0.0
                    || cy + r_out >= (h - 1) as f64
                {
                    return;
                }
                let x0 = ((cx - r_out).floor() as isize - 1).max(0) as usize;
                let y0 = ((cy - r_out).floor() as isize - 1).max(0) as usize;
                let x1 = ((cx + r_out).ceil() as isize + 1).min(w as isize - 1) as usize;
                let y1 = ((cy + r_out).ceil() as isize + 1).min(h as isize - 1) as usize;

                let mut ap_sum = vec![0f64; c];
                let mut ap_area = 0f64;
                let mut ann: Vec<Vec<f32>> = vec![Vec::new(); c];

                for iy in y0..=y1 {
                    let dy = iy as f64 - cy;
                    for ix in x0..=x1 {
                        let dx = ix as f64 - cx;
                        let d2 = dx * dx + dy * dy;
                        let base = (iy * w + ix) * c;
                        let frac = if full_in > 0.0 && d2 <= full_in {
                            1.0
                        } else if d2 >= full_out {
                            0.0
                        } else {
                            let mut inside = 0usize;
                            for &oy in &sub_off {
                                let sy = dy + oy;
                                for &ox in &sub_off {
                                    let sx = dx + ox;
                                    if sx * sx + sy * sy <= r_ap2 {
                                        inside += 1;
                                    }
                                }
                            }
                            inside as f64 * sub_area
                        };
                        if frac > 0.0 {
                            ap_area += frac;
                            for ch in 0..c {
                                let val = img_flat[base + ch] as f64;
                                ap_sum[ch] += frac * val;
                                if !(val <= prow[ch]) {
                                    prow[ch] = val;
                                }
                            }
                        }
                        if d2 > r_in2 && d2 <= r_out2 {
                            for ch in 0..c {
                                ann[ch].push(img_flat[base + ch]);
                            }
                        }
                    }
                }

                *arow = ap_area;
                for ch in 0..c {
                    let v = &mut ann[ch];
                    if v.len() < 4 {
                        continue;
                    }
                    let med = median_inplace(v) as f64;
                    let mut dev: Vec<f32> =
                        v.iter().map(|&x| (x as f64 - med).abs() as f32).collect();
                    let mad = median_inplace(&mut dev) as f64;
                    srow[ch] = med;
                    sgrow[ch] = 1.4826 * mad;
                    frow[ch] = ap_sum[ch] - med * ap_area;
                }
            });
    });

    let flux2 = numpy::ndarray::Array2::from_shape_vec((n, c), flux).unwrap();
    let sky2 = numpy::ndarray::Array2::from_shape_vec((n, c), sky).unwrap();
    let sig2 = numpy::ndarray::Array2::from_shape_vec((n, c), sky_sig).unwrap();
    let peak2 = numpy::ndarray::Array2::from_shape_vec((n, c), peak).unwrap();
    let area1 = numpy::ndarray::Array1::from_vec(area);
    Ok((
        flux2.into_pyarray(py),
        sky2.into_pyarray(py),
        sig2.into_pyarray(py),
        peak2.into_pyarray(py),
        area1.into_pyarray(py),
    ))
}

// ===========================================================================
// originvision in-process inference (src/originvision_infer.py's score_rgb)
// ===========================================================================
//
// Runs the bundled originvision ONNX model end-to-end in Rust via `tract`
// (pure Rust -- no C++ ONNX Runtime, no extra DLL to ship). Replaces the
// numpy/scipy preprocessing + Python `onnxruntime` InferenceSession that
// src/originvision_infer.py::score_rgb used to do. FITS/TIFF/PNG loading +
// debayering stays in Python (astropy / the pipeline's loaders); this takes
// an already-decoded (H, W, 3) f32 RGB array and returns the same result
// dict `score_rgb` did.
//
// Preprocessing mirrors src/originvision_infer.py:
//   per-channel [0.5, 99.5] percentile stretch -> u8
//   -> gaussian pre-blur (downscale only) + bilinear resize shorter side to
//      `size`, centre-crop to size x size, /255, NCHW
// Every result head is gated on the ONNX metadata `tasks` list, never on
// output presence (the v4 graph builds an untrained `trailing` head).
mod originvision {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    use tract_onnx::prelude::*;

    type Runnable = TypedRunnableModel<TypedModel>;

    pub struct Session {
        model: Runnable,
        head_order: Vec<String>,
        tasks: Vec<String>,
        categories: Vec<String>,
        exposures: Vec<f64>,
        quality_scale: f64,
        stray_light_threshold: f64,
        epoch: Option<i64>,
    }

    fn cache() -> &'static Mutex<HashMap<String, Arc<Session>>> {
        static C: OnceLock<Mutex<HashMap<String, Arc<Session>>>> = OnceLock::new();
        C.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn split_csv(m: &HashMap<String, String>, k: &str) -> Vec<String> {
        m.get(k)
            .map(|s| {
                s.split(',')
                    .map(|x| x.trim().to_string())
                    .filter(|x| !x.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn load(path: &str, size: usize) -> TractResult<Session> {
        let proto = tract_onnx::onnx().proto_model_for_path(path)?;
        let mut meta: HashMap<String, String> = HashMap::new();
        for p in &proto.metadata_props {
            meta.insert(p.key.clone(), p.value.clone());
        }
        let head_order = split_csv(&meta, "head_order");
        let tasks = {
            let t = split_csv(&meta, "tasks");
            if t.is_empty() {
                head_order.clone()
            } else {
                t
            }
        };
        let categories = split_csv(&meta, "categories");
        let exposures = split_csv(&meta, "exposures")
            .iter()
            .filter_map(|x| x.parse::<f64>().ok())
            .collect();
        let quality_scale = meta
            .get("quality_scale")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(400.0);
        let stray_light_threshold = meta
            .get("stray_light_threshold")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(27.0);
        let epoch = meta.get("epoch").and_then(|s| s.trim().parse::<i64>().ok());

        let model = tract_onnx::onnx()
            .model_for_proto_model(&proto)?
            .with_input_fact(0, f32::fact([1, 3, size, size]).into())?
            .into_optimized()?
            .into_runnable()?;

        Ok(Session {
            model,
            head_order,
            tasks,
            categories,
            exposures,
            quality_scale,
            stray_light_threshold,
            epoch,
        })
    }

    fn get_session(path: &str, size: usize) -> TractResult<Arc<Session>> {
        // Key on (path, size): the input size is baked into the compiled graph
        // by `load`'s `with_input_fact` + `into_optimized`, so a second call at
        // a different size for the same path must not reuse the old graph.
        let key = format!("{path}\u{0}{size}");
        {
            let c = cache().lock().unwrap();
            if let Some(s) = c.get(&key) {
                return Ok(Arc::clone(s));
            }
        }
        let s = Arc::new(load(path, size)?);
        cache().lock().unwrap().insert(key, Arc::clone(&s));
        Ok(s)
    }

    // ---- preprocessing (mirror src/originvision_infer.py) -----------------

    /// numpy-style linear-interpolated percentile at fractional rank
    /// `p/100*(n-1)` via quickselect -- O(n), no full sort (this file's
    /// kernel-internals convention: `select_nth_unstable` over `sort`).
    /// Reorders `v` in place.
    fn pctl_select(v: &mut [f32], p: f64) -> f64 {
        let n = v.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return v[0] as f64;
        }
        let rank = p / 100.0 * (n as f64 - 1.0);
        let lo = rank.floor() as usize;
        let frac = rank - lo as f64;
        let (_, kth, right) = v.select_nth_unstable_by(lo, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let vlo = *kth as f64;
        if frac <= 0.0 || right.is_empty() {
            return vlo;
        }
        // element at sorted index lo+1 == min of the right partition
        let vhi = right.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
        vlo * (1.0 - frac) + vhi * frac
    }

    /// Per-channel [0.5, 99.5] percentile clip + linear stretch to u8.
    /// Byte-for-byte `imageops.stretch_to_uint8` (`(clip01 * 255).astype(u8)`
    /// truncates toward zero).
    fn stretch_u8(rgb: &[f32], n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n * 3];
        let mut ch = vec![0f32; n];
        let mut scratch = vec![0f32; n];
        for c in 0..3 {
            for i in 0..n {
                ch[i] = rgb[i * 3 + c];
            }
            scratch.copy_from_slice(&ch);
            let lo = pctl_select(&mut scratch, 0.5);
            scratch.copy_from_slice(&ch);
            let mut hi = pctl_select(&mut scratch, 99.5);
            if hi <= lo {
                hi = lo + 1.0;
            }
            let scale = 255.0 / (hi - lo);
            for i in 0..n {
                let v = ((ch[i] as f64 - lo) * scale).clamp(0.0, 255.0);
                out[i * 3 + c] = v as u8;
            }
        }
        out
    }

    /// Separable Gaussian blur over an (h, w, 3) u8 image -> f32, reflect
    /// boundary, `truncate=4.0` radius (matches scipy.ndimage.gaussian_filter).
    fn gaussian_blur(img: &[u8], h: usize, w: usize, sigma: f64) -> Vec<f32> {
        let radius = (4.0 * sigma + 0.5) as isize;
        let mut kernel = vec![0f64; (2 * radius + 1) as usize];
        let mut ksum = 0.0;
        for (idx, kv) in kernel.iter_mut().enumerate() {
            let x = idx as isize - radius;
            *kv = (-(x * x) as f64 / (2.0 * sigma * sigma)).exp();
            ksum += *kv;
        }
        for kv in kernel.iter_mut() {
            *kv /= ksum;
        }
        let refl = |i: isize, len: isize| -> usize {
            // scipy 'reflect' (a b c | c b a), non-edge-duplicating is 'mirror';
            // gaussian_filter's default is 'reflect' == edge-duplicating.
            let mut i = i;
            let n2 = 2 * len;
            i = ((i % n2) + n2) % n2;
            if i >= len {
                i = n2 - 1 - i;
            }
            i as usize
        };
        let mut tmp = vec![0f32; h * w * 3];
        // horizontal
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut acc = 0.0f64;
                    for (idx, kv) in kernel.iter().enumerate() {
                        let xx = refl(x as isize + idx as isize - radius, w as isize);
                        acc += *kv * img[(y * w + xx) * 3 + c] as f64;
                    }
                    tmp[(y * w + x) * 3 + c] = acc as f32;
                }
            }
        }
        // vertical
        let mut out = vec![0f32; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut acc = 0.0f64;
                    for (idx, kv) in kernel.iter().enumerate() {
                        let yy = refl(y as isize + idx as isize - radius, h as isize);
                        acc += *kv * tmp[(yy * w + x) * 3 + c] as f64;
                    }
                    out[(y * w + x) * 3 + c] = acc as f32;
                }
            }
        }
        out
    }

    /// Resize shorter side to `size`, centre-crop to size x size. Downscale
    /// gets a Gaussian pre-blur (sigma = ((1/scale)-1)/2), then bilinear
    /// resample (input coord = output coord / scale, reflect edges) -- the
    /// numpy/scipy `_resize_center_crop` this ports uses `zoom(order=1)` with
    /// the same pre-blur; a small pixel drift vs that is expected and covered
    /// by the parity test's tolerance.
    fn resize_center_crop(stretched: &[u8], h: usize, w: usize, size: usize) -> Vec<u8> {
        let scale = size as f64 / h.min(w) as f64;
        let f: Vec<f32> = if scale < 1.0 {
            let sigma = (1.0 / scale - 1.0) / 2.0;
            if sigma > 0.01 {
                gaussian_blur(stretched, h, w, sigma)
            } else {
                stretched.iter().map(|&v| v as f32).collect()
            }
        } else {
            stretched.iter().map(|&v| v as f32).collect()
        };
        let nh = (h as f64 * scale).round().max(1.0) as usize;
        let nw = (w as f64 * scale).round().max(1.0) as usize;
        let clampi = |v: isize, n: usize| -> usize {
            if v < 0 {
                0
            } else if v as usize >= n {
                n - 1
            } else {
                v as usize
            }
        };
        let mut res = vec![0f32; nh * nw * 3];
        for oy in 0..nh {
            let iy = oy as f64 / scale;
            let y0 = iy.floor();
            let fy = iy - y0;
            let y0i = clampi(y0 as isize, h);
            let y1i = clampi(y0 as isize + 1, h);
            for ox in 0..nw {
                let ix = ox as f64 / scale;
                let x0 = ix.floor();
                let fx = ix - x0;
                let x0i = clampi(x0 as isize, w);
                let x1i = clampi(x0 as isize + 1, w);
                for c in 0..3 {
                    let v00 = f[(y0i * w + x0i) * 3 + c] as f64;
                    let v01 = f[(y0i * w + x1i) * 3 + c] as f64;
                    let v10 = f[(y1i * w + x0i) * 3 + c] as f64;
                    let v11 = f[(y1i * w + x1i) * 3 + c] as f64;
                    let top = v00 * (1.0 - fx) + v01 * fx;
                    let bot = v10 * (1.0 - fx) + v11 * fx;
                    res[(oy * nw + ox) * 3 + c] = (top * (1.0 - fy) + bot * fy) as f32;
                }
            }
        }
        let top = if nh > size { (nh - size) / 2 } else { 0 };
        let left = if nw > size { (nw - size) / 2 } else { 0 };
        let mut out = vec![0u8; size * size * 3];
        for oy in 0..size {
            let sy = (top + oy).min(nh - 1);
            for ox in 0..size {
                let sx = (left + ox).min(nw - 1);
                for c in 0..3 {
                    out[(oy * size + ox) * 3 + c] =
                        res[(sy * nw + sx) * 3 + c].clamp(0.0, 255.0) as u8;
                }
            }
        }
        out
    }

    fn softmax(x: &[f32]) -> Vec<f64> {
        let m = x.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let e: Vec<f64> = x.iter().map(|&v| (v as f64 - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    // ---- public entry ----------------------------------------------------

    #[derive(Default)]
    struct CategoryOut {
        category: String,
        confidence: f64,
        top: Vec<(String, f64)>,
        shape_gated: bool,
    }

    /// Everything the forward pass produces, as plain Rust -- built off the
    /// GIL, then marshalled into a `PyDict` by the caller.
    #[derive(Default)]
    struct ScoreData {
        epoch: Option<i64>,
        tasks: Vec<String>,
        defect: Option<(f64, bool)>,
        quality_score: Option<f64>,
        category: Option<CategoryOut>,
        exposure: Option<(Option<f64>, f64)>,
        sky_brightness: Option<f64>,
        stray_light: Option<(f64, bool)>,
    }

    /// Pure compute: preprocess + tract forward pass + head decode. No
    /// `Python` token -- runs inside `py.allow_threads`. Errors are strings
    /// (surfaced as `PyRuntimeError` by the caller).
    fn compute(
        flat: &[f32],
        h: usize,
        w: usize,
        model_path: &str,
        size: usize,
        shape_gate: bool,
    ) -> Result<ScoreData, String> {
        let n = h * w;
        let sess = get_session(model_path, size).map_err(|e| e.to_string())?;
        let stretched = stretch_u8(flat, n);
        let model_in = resize_center_crop(&stretched, h, w, size);

        // NCHW, /255
        let mut nchw = vec![0f32; 3 * size * size];
        for yx in 0..(size * size) {
            for c in 0..3 {
                nchw[c * size * size + yx] = model_in[yx * 3 + c] as f32 / 255.0;
            }
        }
        let input = tract_ndarray::Array4::from_shape_vec((1, 3, size, size), nchw)
            .map_err(|e| e.to_string())?
            .into_tensor();
        let outputs = sess
            .model
            .run(tvec!(input.into()))
            .map_err(|e| e.to_string())?;

        // A model whose graph outputs don't line up with its head_order
        // metadata is an export bug -- refuse to guess (mirrors the
        // onnxruntime fallback's `zip(strict=True)` intent).
        if sess.head_order.is_empty() || outputs.len() != sess.head_order.len() {
            return Err(format!(
                "model has {} graph outputs but head_order metadata lists {} -- refusing to guess",
                outputs.len(),
                sess.head_order.len()
            ));
        }

        let mut out: HashMap<&str, Vec<f32>> = HashMap::new();
        for (name, t) in sess.head_order.iter().zip(outputs.iter()) {
            let v = t
                .to_array_view::<f32>()
                .map_err(|e| e.to_string())?
                .iter()
                .cloned()
                .collect();
            out.insert(name.as_str(), v);
        }
        let has = |k: &str| sess.tasks.iter().any(|t| t == k);
        let g = |k: &str| out.get(k);

        let mut sd = ScoreData {
            epoch: sess.epoch,
            tasks: sess.tasks.clone(),
            ..Default::default()
        };

        if has("reject") {
            if let Some(v) = g("reject") {
                let p = sigmoid(v[0] as f64);
                sd.defect = Some((p, p > 0.5));
            }
        }
        if has("quality") {
            if let Some(v) = g("quality") {
                sd.quality_score = Some(v[0] as f64 * sess.quality_scale);
            }
        }
        if has("category") {
            if let Some(v) = g("category") {
                let probs = softmax(v);
                let cats = &sess.categories;
                let mut order: Vec<usize> = (0..probs.len()).collect();
                order.sort_by(|&i, &j| {
                    probs[j].partial_cmp(&probs[i]).unwrap_or(std::cmp::Ordering::Equal)
                });
                let top = order[0];
                // `comet` is suppressed: the current checkpoint's comet class
                // isn't trusted, so a top `comet` pick is demoted to the
                // runner-up rather than run through a hand-tuned shape gate.
                // (`shape_gate=false` disables the suppression.)
                let picked = if shape_gate
                    && order.len() > 1
                    && cats.get(top).map(|c| c == "comet").unwrap_or(false)
                {
                    order[1]
                } else {
                    top
                };
                sd.category = Some(CategoryOut {
                    category: cats.get(picked).cloned().unwrap_or_default(),
                    confidence: probs[picked],
                    top: order
                        .iter()
                        .take(3)
                        .map(|&i| (cats.get(i).cloned().unwrap_or_default(), probs[i]))
                        .collect(),
                    shape_gated: picked != top,
                });
            }
        }
        if has("exposure") {
            if let Some(v) = g("exposure") {
                let probs = softmax(v);
                let i = (0..probs.len())
                    .max_by(|&x, &y| {
                        probs[x].partial_cmp(&probs[y]).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);
                sd.exposure = Some((sess.exposures.get(i).copied(), probs[i]));
            }
        }
        if has("sky_brightness") {
            if let Some(v) = g("sky_brightness") {
                sd.sky_brightness = Some(v[0] as f64 * 255.0);
            }
        }
        if has("stray_light_gradient") {
            if let Some(v) = g("stray_light_gradient") {
                let val = v[0] as f64 * 255.0;
                sd.stray_light = Some((val, val > sess.stray_light_threshold));
            }
        }
        Ok(sd)
    }

    #[pyfunction]
    #[pyo3(signature = (rgb, model_path, size=256, shape_gate=true))]
    pub fn originvision_score(
        py: Python<'_>,
        rgb: numpy::PyReadonlyArray3<f32>,
        model_path: &str,
        size: usize,
        shape_gate: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        let a = rgb.as_array();
        let sh = a.shape();
        if sh.len() != 3 || sh[2] < 3 || sh[0] == 0 || sh[1] == 0 {
            return Ok(None);
        }
        let (h, w) = (sh[0], sh[1]);
        let n = h * w;
        let mut flat = vec![0f32; n * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    flat[(y * w + x) * 3 + c] = a[[y, x, c]];
                }
            }
        }

        // Heavy work off the GIL, panic-guarded: `tract` parses an external
        // .onnx file and a malformed one can panic inside the parser -- that
        // unwinds to a `pyo3_runtime.PanicException` (a `BaseException`,
        // uncatchable by callers' `except Exception`), so convert it to a
        // plain `RuntimeError` here.
        let outcome = py.detach(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compute(&flat, h, w, model_path, size, shape_gate)
            }))
        });

        let sd = match outcome {
            Ok(Ok(sd)) => sd,
            Ok(Err(msg)) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "originvision native inference failed: {msg}"
                )))
            }
            Err(_) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "originvision native inference panicked (malformed model?)".to_string(),
                ))
            }
        };

        let d = PyDict::new(py);
        d.set_item("checkpoint_epoch", sd.epoch)?;
        d.set_item("tasks", sd.tasks.clone())?;
        if let Some((p, is_def)) = sd.defect {
            d.set_item("defect_probability", p)?;
            d.set_item("is_defective", is_def)?;
        }
        if let Some(q) = sd.quality_score {
            d.set_item("quality_score", q)?;
        }
        if let Some(c) = sd.category {
            d.set_item("category", c.category)?;
            d.set_item("category_confidence", c.confidence)?;
            d.set_item("top_categories", c.top)?;
            d.set_item("category_shape_gated", c.shape_gated)?;
        }
        if let Some((exp_s, conf)) = sd.exposure {
            if let Some(e) = exp_s {
                d.set_item("predicted_exposure_s", e)?;
            }
            d.set_item("exposure_confidence", conf)?;
        }
        if let Some(s) = sd.sky_brightness {
            d.set_item("sky_brightness", s)?;
        }
        if let Some((val, flag)) = sd.stray_light {
            d.set_item("stray_light_gradient", val)?;
            d.set_item("stray_light_flag", flag)?;
        }
        Ok(Some(d.into()))
    }
}

// ---------------------------------------------------------------------------
// ZOGY transient triage: real/bogus classification of --transient-detect
// candidates (src/transient_triage.py, --transient-triage)
// ---------------------------------------------------------------------------
//
// A much smaller sibling of `mod originvision` above: no percentile stretch,
// no resize -- stamps arrive already extracted at a fixed size and
// per-channel sigma-normalized in Python (there is no perf-relevant work to
// move into Rust for a few hundred 31x31x3 stamps), and no multi-head
// metadata decode, just one scalar sigmoid probability per candidate.
// Batched: all of a frame's candidates score in a single tract forward pass
// over (N, 3, size, size), not one call per candidate. Advisory only --
// never filters `detect_transients`' output, just attaches a
// `real_probability`.
//
// The bundled model (src/data/transient_triage.onnx) is trained entirely on
// synthetic data (tools/gen_transient_triage_data.py +
// tools/train_transient_triage.py) -- no labelled real transients exist yet
// -- so treat it as a first cut, not a production classifier. Deliberately
// has no numpy/onnxruntime fallback yet, unlike every other native kernel in
// this file: a source checkout without astro_native built simply can't use
// --transient-triage until one is added (self-disables with a warning, see
// src/transient_triage.py).
mod transient_triage {
    use pyo3::prelude::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    use tract_onnx::prelude::*;

    type Runnable = TypedRunnableModel<TypedModel>;

    struct Session {
        model: Runnable,
    }

    fn cache() -> &'static Mutex<HashMap<String, Arc<Session>>> {
        static C: OnceLock<Mutex<HashMap<String, Arc<Session>>>> = OnceLock::new();
        C.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn load(path: &str, _size: usize) -> TractResult<Session> {
        // Unlike `originvision::load`, the batch axis here must stay
        // symbolic: this kernel scores a whole frame's candidates in one
        // batched call (N varies per frame), while originvision always
        // calls with N=1. The channel/H/W axes are already concrete in the
        // exported graph (torch.onnx.export's dynamic_axes only marks axis
        // 0 as dynamic), so no `with_input_fact` override is needed -- one
        // was tried and made every N != 1 call fail with a tract symbol
        // resolution clash against the fixed batch=1 it forced.
        let proto = tract_onnx::onnx().proto_model_for_path(path)?;
        let model = tract_onnx::onnx()
            .model_for_proto_model(&proto)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Session { model })
    }

    fn get_session(path: &str, size: usize) -> TractResult<Arc<Session>> {
        // Key on (path, size): the input size is baked into the compiled
        // graph by `load`'s `with_input_fact` + `into_optimized`, same
        // reasoning as `originvision`'s cache above.
        let key = format!("{path}\u{0}{size}");
        {
            let c = cache().lock().unwrap();
            if let Some(s) = c.get(&key) {
                return Ok(Arc::clone(s));
            }
        }
        let s = Arc::new(load(path, size)?);
        cache().lock().unwrap().insert(key, Arc::clone(&s));
        Ok(s)
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Pure compute: batched forward pass over N pre-normalized stamps, no
    /// `Python` token -- runs inside `py.detach`.
    fn compute(stamps: &[f32], n: usize, size: usize, model_path: &str) -> Result<Vec<f64>, String> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let sess = get_session(model_path, size).map_err(|e| e.to_string())?;
        // `stamps` is already NCHW-ordered per candidate: (n, 3, size, size).
        let input = tract_ndarray::Array4::from_shape_vec((n, 3, size, size), stamps.to_vec())
            .map_err(|e| e.to_string())?
            .into_tensor();
        let outputs = sess.model.run(tvec!(input.into())).map_err(|e| e.to_string())?;
        let raw = outputs
            .first()
            .ok_or_else(|| "model produced no output".to_string())?
            .to_array_view::<f32>()
            .map_err(|e| e.to_string())?;
        if raw.len() != n {
            return Err(format!(
                "model produced {} outputs for {n} candidates -- refusing to guess",
                raw.len()
            ));
        }
        Ok(raw.iter().map(|&v| sigmoid(v as f64)).collect())
    }

    #[pyfunction]
    #[pyo3(signature = (stamps, model_path, size=31))]
    pub fn transient_triage_score(
        py: Python<'_>,
        stamps: numpy::PyReadonlyArray4<f32>,
        model_path: &str,
        size: usize,
    ) -> PyResult<Vec<f64>> {
        let a = stamps.as_array();
        let sh = a.shape();
        if sh.len() != 4 || sh[1] != 3 || sh[2] != size || sh[3] != size {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected stamps shaped (N, 3, {size}, {size}), got {:?}",
                sh
            )));
        }
        let n = sh[0];
        let flat: Vec<f32> = a.iter().cloned().collect();
        let model_path = model_path.to_string();

        // Heavy work off the GIL, panic-guarded -- same reasoning as
        // `originvision_score`: `tract` parses an external .onnx file and a
        // malformed one can panic inside the parser, which would otherwise
        // unwind past callers' `except Exception` as a bare PanicException.
        let outcome = py.detach(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compute(&flat, n, size, &model_path)
            }))
        });

        match outcome {
            Ok(Ok(probs)) => Ok(probs),
            Ok(Err(msg)) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "transient_triage native inference failed: {msg}"
            ))),
            Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "transient_triage native inference panicked (malformed model?)".to_string(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// CFA drizzle: splat one frame's measured Bayer samples (src/cfa_drizzle.py)
// ---------------------------------------------------------------------------
//
// One call folds a whole calibrated frame into the running drizzle
// accumulators. Every sensor pixel measured exactly one colour (its Bayer
// parity picks the channel via `chan_table[(y&1)*2 + (x&1)]`); its sample is
// (1) mapped onto the output grid through the frame's affine, (2) tested
// against the reference stack, and (3) deposited as a square drop of half-side
// `h` by exact area overlap into `num` (weighted value), `den` (weight) and
// `cov` (unweighted coverage), each (out_h, out_w, 3) f64.
//
// The numpy path (`_frame_splat_numpy`) walks the same three steps with
// per-channel bincounts -- ~6 full-size bincount passes per neighbour tap per
// channel, which is what made a 148-frame run take 407 s. Here the work is one
// pass over the samples, parallelised over bands of OUTPUT rows: every thread
// owns its rows exclusively (a drop that straddles a band edge is deposited
// by each band only into its own rows), so there are no atomics and no
// per-thread accumulator copies (three full-size f64 planes per thread would
// be gigabytes at sensor size).

#[inline]
fn bilinear_at(img: &[f32], oh: usize, ow: usize, c: usize, y: f64, x: f64) -> f64 {
    let yc = y.max(0.0).min(oh as f64 - 1.0);
    let xc = x.max(0.0).min(ow as f64 - 1.0);
    let y0 = (yc.floor() as usize).min(oh - 2);
    let x0 = (xc.floor() as usize).min(ow - 2);
    let fy = yc - y0 as f64;
    let fx = xc - x0 as f64;
    let at = |yy: usize, xx: usize| img[(yy * ow + xx) * 3 + c] as f64;
    (1.0 - fy) * ((1.0 - fx) * at(y0, x0) + fx * at(y0, x0 + 1))
        + fy * ((1.0 - fx) * at(y0 + 1, x0) + fx * at(y0 + 1, x0 + 1))
}

/// Returns (samples_considered, samples_rejected) for this frame.
#[pyfunction]
#[pyo3(signature = (rgb, reference, chan_table, minv, off, h, sig, sky, reject_sigma, signal_tol, num, den, cov))]
#[allow(clippy::too_many_arguments)]
fn cfa_drizzle_frame<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f32>,
    reference: PyReadonlyArray3<'py, f32>,
    chan_table: PyReadonlyArray1<'py, u8>,
    minv: PyReadonlyArray1<'py, f64>,
    off: PyReadonlyArray1<'py, f64>,
    h: f64,
    sig: PyReadonlyArray1<'py, f64>,
    sky: PyReadonlyArray1<'py, f64>,
    reject_sigma: f64,
    signal_tol: f64,
    mut num: PyReadwriteArray3<'py, f64>,
    mut den: PyReadwriteArray3<'py, f64>,
    mut cov: PyReadwriteArray3<'py, f64>,
) -> PyResult<(u64, u64)> {
    let rs = rgb.as_array().shape().to_vec();
    let os = reference.as_array().shape().to_vec();
    if rs[2] != 3 || os[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("rgb and reference must have 3 channels"));
    }
    let (sh, sw) = (rs[0], rs[1]);
    let (oh, ow) = (os[0], os[1]);
    if oh < 2 || ow < 2 || num.as_array().shape() != [oh, ow, 3]
        || den.as_array().shape() != [oh, ow, 3] || cov.as_array().shape() != [oh, ow, 3]
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "accumulators must be (out_h, out_w, 3) matching reference",
        ));
    }
    let rgb_s = rgb.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("rgb must be C-contiguous")
    })?;
    let ref_s = reference.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("reference must be C-contiguous")
    })?;
    let ct = chan_table.as_slice()?;
    let mv = minv.as_slice()?;
    let of = off.as_slice()?;
    let sg = sig.as_slice()?;
    let sk = sky.as_slice()?;
    if ct.len() != 4 || mv.len() != 4 || of.len() != 2 || sg.len() != 3 || sk.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("bad parameter vector length"));
    }
    let num_s = num.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("num must be C-contiguous")
    })?;
    let den_s = den.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("den must be C-contiguous")
    })?;
    let cov_s = cov.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("cov must be C-contiguous")
    })?;

    let (m00, m01, m10, m11) = (mv[0], mv[1], mv[2], mv[3]);
    let (off0, off1) = (of[0], of[1]);
    // Forward affine input->output is o = Minv (i - off); its inverse
    // i = M o + off maps an output band back to the input rows that can land in it.
    let det = m00 * m11 - m01 * m10;
    if det.abs() < 1e-12 {
        return Err(pyo3::exceptions::PyValueError::new_err("singular affine"));
    }
    let (i00, i01) = (m11 / det, -m01 / det);

    let k = (2.0 * h).floor() as i64 + 2;
    let norm = 1.0 / (4.0 * h * h);
    let valid_ch: [bool; 3] = [
        sg[0].is_finite() && sg[0] > 0.0,
        sg[1].is_finite() && sg[1] > 0.0,
        sg[2].is_finite() && sg[2] > 0.0,
    ];
    let fw: [f64; 3] = [
        if valid_ch[0] { 1.0 / (sg[0] * sg[0]) } else { 0.0 },
        if valid_ch[1] { 1.0 / (sg[1] * sg[1]) } else { 0.0 },
        if valid_ch[2] { 1.0 / (sg[2] * sg[2]) } else { 0.0 },
    ];

    const BAND: usize = 16;
    let row_len = ow * 3;
    let considered = std::sync::atomic::AtomicU64::new(0);
    let rejected = std::sync::atomic::AtomicU64::new(0);

    py.detach(|| {
        num_s
            .par_chunks_mut(BAND * row_len)
            .zip(den_s.par_chunks_mut(BAND * row_len))
            .zip(cov_s.par_chunks_mut(BAND * row_len))
            .enumerate()
            .for_each(|(band, ((nb, db), cb))| {
                let r0 = band * BAND;
                let r1 = (r0 + nb.len() / row_len).min(oh);
                // Output rows this band may receive (a drop reaches h beyond its centre).
                let lo_o = r0 as f64 - h - 0.5;
                let hi_o = r1 as f64 + h - 0.5;

                // Bounding input rows: map the band's 4 corners back to the sensor.
                let mut imin = f64::INFINITY;
                let mut imax = f64::NEG_INFINITY;
                for &oy in &[lo_o, hi_o] {
                    for &ox in &[-h - 0.5, ow as f64 + h - 0.5] {
                        let iy = i00 * oy + i01 * ox + off0;
                        if iy < imin { imin = iy; }
                        if iy > imax { imax = iy; }
                    }
                }
                let y_start = (imin.floor() as i64 - 1).max(0) as usize;
                let y_end = ((imax.ceil() as i64 + 2).max(0) as usize).min(sh);

                let mut n_cons = 0u64;
                let mut n_rej = 0u64;
                for iy in y_start..y_end {
                    let dy = iy as f64 - off0;
                    // oy(ix) = m00*dy + m01*(ix - off1): solve for the ix span that
                    // lands inside [lo_o, hi_o]; that keeps this pass O(samples in band).
                    let base = m00 * dy;
                    let (x_lo, x_hi) = if m01.abs() < 1e-12 {
                        if base >= lo_o && base <= hi_o { (0usize, sw) } else { continue; }
                    } else {
                        let a = (lo_o - base) / m01 + off1;
                        let b = (hi_o - base) / m01 + off1;
                        let (u, v) = if a < b { (a, b) } else { (b, a) };
                        (
                            (u.floor() as i64 - 1).max(0) as usize,
                            ((v.ceil() as i64 + 2).max(0) as usize).min(sw),
                        )
                    };
                    for ix in x_lo..x_hi {
                        let dx = ix as f64 - off1;
                        let oy = m00 * dy + m01 * dx;
                        let ox = m10 * dy + m11 * dx;
                        // same admission test as the numpy path
                        if !(oy > -h - 0.5 && oy < oh as f64 + h - 0.5
                            && ox > -h - 0.5 && ox < ow as f64 + h - 0.5)
                        {
                            continue;
                        }
                        // band ownership: skip drops that cannot touch our rows
                        if oy + h <= r0 as f64 - 0.5 || oy - h >= r1 as f64 - 0.5 {
                            continue;
                        }
                        let c = ct[(iy & 1) * 2 + (ix & 1)] as usize;
                        if c > 2 {
                            continue;
                        }
                        let v = rgb_s[(iy * sw + ix) * 3 + c] as f64;
                        let r_at = bilinear_at(ref_s, oh, ow, c, oy, ox);
                        let resid = v - r_at;
                        // Count each sample once: only the band that owns its centre row.
                        let owner = {
                            let cy = oy.round().max(0.0).min(oh as f64 - 1.0) as usize;
                            cy >= r0 && cy < r1
                        };
                        if !valid_ch[c] {
                            continue;
                        }
                        let s = sg[c];
                        let sig_sig = (r_at - sk[c]).max(0.0) * signal_tol;
                        let tol = reject_sigma * (s * s + sig_sig * sig_sig).sqrt();
                        let keep = resid.abs() <= tol;
                        if owner {
                            n_cons += 1;
                            if !keep { n_rej += 1; }
                        }
                        if !keep {
                            continue;
                        }
                        let y0 = (oy - h + 0.5).floor() as i64;
                        let x0 = (ox - h + 0.5).floor() as i64;
                        for a in 0..k {
                            let qy = y0 + a;
                            if qy < r0 as i64 || qy >= r1 as i64 { continue; }
                            let qyf = qy as f64;
                            let ovy = (oy + h).min(qyf + 0.5) - (oy - h).max(qyf - 0.5);
                            if ovy <= 0.0 { continue; }
                            for b in 0..k {
                                let qx = x0 + b;
                                if qx < 0 || qx >= ow as i64 { continue; }
                                let qxf = qx as f64;
                                let ovx = (ox + h).min(qxf + 0.5) - (ox - h).max(qxf - 0.5);
                                if ovx <= 0.0 { continue; }
                                let wgt = ovy * ovx * norm;
                                let idx = ((qy as usize - r0) * ow + qx as usize) * 3 + c;
                                nb[idx] += wgt * v * fw[c];
                                db[idx] += wgt * fw[c];
                                cb[idx] += wgt;
                            }
                        }
                    }
                }
                considered.fetch_add(n_cons, std::sync::atomic::Ordering::Relaxed);
                rejected.fetch_add(n_rej, std::sync::atomic::Ordering::Relaxed);
            });
    });

    Ok((
        considered.load(std::sync::atomic::Ordering::Relaxed),
        rejected.load(std::sync::atomic::Ordering::Relaxed),
    ))
}

// ---------------------------------------------------------------------------
// Drizzle: fused warp + weight + accumulate (src/stacking.py run_stacking_phase)
// ---------------------------------------------------------------------------
//
// The Python drizzle loop warped each frame into a fresh (out_h, out_w, 3) f32
// array, multiplied it by the frame weight, and added it into a shared f64
// accumulator under a lock -- three full-size passes and a serialisation point
// per frame (the add alone was ~150 ms of a ~435 ms rotated frame at 2x scale,
// one thread at a time). This kernel does the same arithmetic per output row and
// adds straight into the accumulator: rows are the parallel unit, so each thread
// owns its accumulator rows and no lock or temporary array exists.
//
// Bit-identical to the old path: the row is produced by `lanczos3_row_flat` (the
// very code `warp_affine_lanczos3` runs), then `v * (w as f32)` in f32 (numpy
// in-place `resampled *= w`), then `acc += f64(v)`. With `pixfrac < 1` the tent
// weight numpy built from seven full-size f64 temporaries is computed per pixel
// with the same float64 operations in the same order, and `wmap += pfw`.
#[pyfunction]
#[pyo3(signature = (acc, data, mat, off, weight, pixfrac, wmap=None))]
#[allow(clippy::too_many_arguments)]
fn drizzle_accumulate_lanczos3<'py>(
    py: Python<'py>,
    mut acc: PyReadwriteArray3<'py, f64>,
    data: PyReadonlyArray3<'py, f32>,
    mat: [f64; 4],
    off: [f64; 2],
    weight: f64,
    pixfrac: f64,
    wmap: Option<PyReadwriteArray3<'py, f64>>,
) -> PyResult<()> {
    let s = data.as_array().shape().to_vec();
    let (h, w, c) = (s[0], s[1], s[2]);
    let acs = acc.as_array().shape().to_vec();
    if c != 3 || acs.len() != 3 || acs[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("data and acc must have 3 channels"));
    }
    let (out_h, out_w) = (acs[0], acs[1]);
    let img = data.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("data must be C-contiguous")
    })?;
    let acc_s = acc.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("acc must be C-contiguous")
    })?;
    let use_pf = pixfrac < 1.0 - 1e-9;
    let mut wmap = wmap;
    let wm_s: Option<&mut [f64]> = match wmap.as_mut() {
        Some(m) => {
            if m.as_array().shape() != [out_h, out_w, 1] {
                return Err(pyo3::exceptions::PyValueError::new_err("wmap must be (out_h, out_w, 1)"));
            }
            Some(m.as_slice_mut().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("wmap must be C-contiguous")
            })?)
        }
        None => None,
    };
    if use_pf && wm_s.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err("pixfrac < 1 needs a wmap"));
    }
    let (m00, m01, m10, m11) = (mat[0], mat[1], mat[2], mat[3]);
    let (o0, o1) = (off[0], off[1]);
    let col_tab: Option<(Vec<[f64; 6]>, Vec<isize>)> = if m01 == 0.0 && m10 == 0.0 {
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
    let wf32 = weight as f32;
    let half_drop = (pixfrac / 2.0).max(1e-6);

    py.detach(|| {
        let row = |oy: usize, arow: &mut [f64], mrow: Option<&mut [f64]>, buf: &mut Vec<f32>| {
            buf.resize(out_w * 3, 0.0);
            lanczos3_row_flat(img, h, w, 3, oy, buf, [m00, m01, m10, m11], [o0, o1], &col_tab, 0.0, 0);
            if use_pf {
                let mrow = mrow.unwrap();
                for ox in 0..out_w {
                    let raw_y = m00 * oy as f64 + m01 * ox as f64 + o0;
                    let raw_x = m10 * oy as f64 + m11 * ox as f64 + o1;
                    let fy = (raw_y - raw_y.round_ties_even()).abs();
                    let fx = (raw_x - raw_x.round_ties_even()).abs();
                    let wy_ = (1.0 - fy / half_drop).max(0.0);
                    let wx_ = (1.0 - fx / half_drop).max(0.0);
                    let pfw = wy_ * wx_ * weight;
                    for ch in 0..3 {
                        arow[ox * 3 + ch] += buf[ox * 3 + ch] as f64 * pfw;
                    }
                    mrow[ox] += pfw;
                }
            } else {
                for i in 0..out_w * 3 {
                    arow[i] += (buf[i] * wf32) as f64;
                }
            }
        };
        match wm_s {
            Some(wm) if use_pf => {
                acc_s
                    .par_chunks_mut(out_w * 3)
                    .zip(wm.par_chunks_mut(out_w))
                    .enumerate()
                    .for_each_init(Vec::new, |buf, (oy, (arow, mrow))| {
                        row(oy, arow, Some(mrow), buf)
                    });
            }
            _ => {
                acc_s
                    .par_chunks_mut(out_w * 3)
                    .enumerate()
                    .for_each_init(Vec::new, |buf, (oy, arow)| row(oy, arow, None, buf));
            }
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Drizzle by area-overlap splatting (--drizzle-method splat)
// ---------------------------------------------------------------------------
//
// The resampling path evaluates a 6x6-tap Lanczos gather at every OUTPUT pixel
// (4x as many as input pixels at 2x scale) and does not implement drizzle's
// pixfrac at all (it is a weight applied afterwards). This is the algorithm
// itself: every input pixel is a square drop of side `pixfrac * scale` output
// pixels (half-side `h`), mapped through the frame's affine and deposited by
// exact area overlap into `num` (weight * value, per channel) and `den`
// (weight). Work is one pass over the INPUT (~16x fewer multiply-adds than the
// gather) and parallel over bands of output rows so every thread owns its rows.
// Same drop geometry and band scheme as `cfa_drizzle_frame`; all three channels
// share one drop (a debayered frame carries a value in each).
#[pyfunction]
#[pyo3(signature = (rgb, fwd, off, h, weight, num, den))]
#[allow(clippy::too_many_arguments)]
fn drizzle_splat_frame<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f32>,
    fwd: [f64; 4],
    off: [f64; 2],
    h: f64,
    weight: f64,
    mut num: PyReadwriteArray3<'py, f64>,
    mut den: PyReadwriteArray3<'py, f64>,
) -> PyResult<()> {
    let rs = rgb.as_array().shape().to_vec();
    let ns = num.as_array().shape().to_vec();
    if rs[2] != 3 || ns[2] != 3 || den.as_array().shape() != [ns[0], ns[1], 1] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rgb/num must have 3 channels and den must be (out_h, out_w, 1)",
        ));
    }
    if !(h > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err("h must be > 0"));
    }
    let (sh, sw) = (rs[0], rs[1]);
    let (oh, ow) = (ns[0], ns[1]);
    let rgb_s = rgb.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("rgb must be C-contiguous")
    })?;
    let num_s = num.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("num must be C-contiguous")
    })?;
    let den_s = den.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("den must be C-contiguous")
    })?;
    let (m00, m01, m10, m11) = (fwd[0], fwd[1], fwd[2], fwd[3]);
    let (off0, off1) = (off[0], off[1]);
    let det = m00 * m11 - m01 * m10;
    if det.abs() < 1e-12 {
        return Err(pyo3::exceptions::PyValueError::new_err("singular affine"));
    }
    let (i00, i01) = (m11 / det, -m01 / det);
    let k = (2.0 * h).floor() as i64 + 2;
    let norm = 1.0 / (4.0 * h * h);

    const BAND: usize = 16;
    py.detach(|| {
        num_s
            .par_chunks_mut(BAND * ow * 3)
            .zip(den_s.par_chunks_mut(BAND * ow))
            .enumerate()
            .for_each(|(band, (nb, db))| {
                let r0 = band * BAND;
                let r1 = (r0 + db.len() / ow).min(oh);
                let lo_o = r0 as f64 - h - 0.5;
                let hi_o = r1 as f64 + h - 0.5;
                let mut imin = f64::INFINITY;
                let mut imax = f64::NEG_INFINITY;
                for &oy in &[lo_o, hi_o] {
                    for &ox in &[-h - 0.5, ow as f64 + h - 0.5] {
                        let iy = i00 * oy + i01 * ox + off0;
                        if iy < imin { imin = iy; }
                        if iy > imax { imax = iy; }
                    }
                }
                let y_start = (imin.floor() as i64 - 1).max(0) as usize;
                let y_end = ((imax.ceil() as i64 + 2).max(0) as usize).min(sh);
                for iy in y_start..y_end {
                    let dy = iy as f64 - off0;
                    let base = m00 * dy;
                    let (x_lo, x_hi) = if m01.abs() < 1e-12 {
                        if base >= lo_o && base <= hi_o { (0usize, sw) } else { continue; }
                    } else {
                        let a = (lo_o - base) / m01 + off1;
                        let b = (hi_o - base) / m01 + off1;
                        let (u, v) = if a < b { (a, b) } else { (b, a) };
                        (
                            (u.floor() as i64 - 1).max(0) as usize,
                            ((v.ceil() as i64 + 2).max(0) as usize).min(sw),
                        )
                    };
                    for ix in x_lo..x_hi {
                        let dx = ix as f64 - off1;
                        let oy = m00 * dy + m01 * dx;
                        let ox = m10 * dy + m11 * dx;
                        if !(oy > -h - 0.5 && oy < oh as f64 + h - 0.5
                            && ox > -h - 0.5 && ox < ow as f64 + h - 0.5)
                        {
                            continue;
                        }
                        if oy + h <= r0 as f64 - 0.5 || oy - h >= r1 as f64 - 0.5 {
                            continue;
                        }
                        let px = &rgb_s[(iy * sw + ix) * 3..(iy * sw + ix) * 3 + 3];
                        if !(px[0].is_finite() && px[1].is_finite() && px[2].is_finite()) {
                            continue;
                        }
                        let (v0, v1, v2) = (px[0] as f64, px[1] as f64, px[2] as f64);
                        let y0 = (oy - h + 0.5).floor() as i64;
                        let x0 = (ox - h + 0.5).floor() as i64;
                        for a in 0..k {
                            let qy = y0 + a;
                            if qy < r0 as i64 || qy >= r1 as i64 { continue; }
                            let qyf = qy as f64;
                            let ovy = (oy + h).min(qyf + 0.5) - (oy - h).max(qyf - 0.5);
                            if ovy <= 0.0 { continue; }
                            for b in 0..k {
                                let qx = x0 + b;
                                if qx < 0 || qx >= ow as i64 { continue; }
                                let qxf = qx as f64;
                                let ovx = (ox + h).min(qxf + 0.5) - (ox - h).max(qxf - 0.5);
                                if ovx <= 0.0 { continue; }
                                let wgt = ovy * ovx * norm * weight;
                                let cell = (qy as usize - r0) * ow + qx as usize;
                                nb[cell * 3] += wgt * v0;
                                nb[cell * 3 + 1] += wgt * v1;
                                nb[cell * 3 + 2] += wgt * v2;
                                db[cell] += wgt;
                            }
                        }
                    }
                }
            });
    });
    Ok(())
}

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
fn opt_master<'a, 'py>(
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
fn calibrate_frame_inplace<'py>(
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

/// 3x3 median at plane position (i, j) of the (py, px) Bayer sub-plane of a
/// row-major `w`-wide mosaic, reflect boundary on the sub-plane (scipy
/// `mode='reflect'`), exactly what `median_filter(sub, size=3)` returns there.
#[inline]
fn bayer_plane_median3(data: &[f32], w: usize, hh: usize, ww: usize, py: usize, px: usize,
                       i: usize, j: usize) -> f32 {
    let mut win = [0f32; 9];
    let mut k = 0;
    for di in -1isize..=1 {
        let ii = reflect_idx(i as isize + di, hh);
        for dj in -1isize..=1 {
            let jj = reflect_idx(j as isize + dj, ww);
            win[k] = data[(2 * ii + py) * w + 2 * jj + px];
            k += 1;
        }
    }
    median9(&mut win)
}

/// Bayer-aware hot-pixel fix on a 2-D mosaic, returning a new array.
///
/// `hot_map` (optional, u8): those pixels are replaced by their sub-plane 3x3
/// median -- computed only at the flagged pixels, not over the frame.
/// `threshold` (optional): statistical detection per 2x2 sub-plane -- a pixel
/// whose excess over the plane's 3x3 median exceeds `threshold` times
/// 1.4826 x the plane's median absolute deviation is replaced by that median.
/// `star_support` (optional, with `threshold`): a flagged pixel is *kept* when any of
/// its four adjacent mosaic pixels (one pixel away, in the other colour planes) is
/// itself more than `star_support` sigma above its own plane's median. A hot pixel is a
/// single-sensor-pixel event, so its neighbours stay normal; a star lifts its
/// neighbours with it, and without this test the peak of every bright star was replaced
/// by its plane median (a star is only ~2 pixels wide in a half-resolution plane), which
/// clipped the cores and softened and noised the whole stack.
/// Give one or the other, not both (the numpy path shares one median between
/// the two; the pipeline never asks for both at once). All arithmetic is f32
/// in numpy's order, so the result is bit-identical to `_fix_hot_bayer`.
#[pyfunction]
#[pyo3(signature = (data, hot_map=None, threshold=None, star_support=None))]
fn hot_pixel_bayer<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    hot_map: Option<PyReadonlyArray2<'py, u8>>,
    threshold: Option<f32>,
    star_support: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if hot_map.is_some() && threshold.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "give either hot_map or threshold",
        ));
    }
    let arr = data.as_array();
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    let d: &[f32] = arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("data must be contiguous"))?;
    let map_view = hot_map.as_ref().map(|m| m.as_array());
    let map: Option<&[u8]> = match &map_view {
        Some(v) => Some(v.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("hot_map must be contiguous")
        })?),
        None => None,
    };
    if let Some(m) = map {
        if m.len() != h * w {
            return Err(pyo3::exceptions::PyValueError::new_err("hot_map shape mismatch"));
        }
    }

    let out = py.detach(|| {
        let mut out = d.to_vec();
        if let Some(m) = map {
            out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                let py_ = y & 1;
                let i = y >> 1;
                let hh = (h - py_ + 1) / 2;
                for x in 0..w {
                    if m[y * w + x] == 0 {
                        continue;
                    }
                    let px = x & 1;
                    let ww = (w - px + 1) / 2;
                    row[x] = bayer_plane_median3(d, w, hh, ww, py_, px, i, x >> 1);
                }
            });
        } else if let Some(thr) = threshold {
            let sigma_k = 1.4826f64 as f32;
            // per plane: (values, 3x3 median, sigma); sigma 0 marks a plane that is left alone
            // (empty, a NaN made numpy's MAD NaN, or sigma < 1e-6)
            let stage: Vec<(Vec<f32>, Vec<f32>, f32)> = (0..4usize)
                .into_par_iter()
                .map(|q| {
                    let (py_, px) = (q >> 1, q & 1);
                    let hh = (h.saturating_sub(py_) + 1) / 2;
                    let ww = (w.saturating_sub(px) + 1) / 2;
                    let mut p = vec![0f32; hh * ww];
                    for i in 0..hh {
                        let src = (2 * i + py_) * w + px;
                        for j in 0..ww {
                            p[i * ww + j] = d[src + 2 * j];
                        }
                    }
                    if p.is_empty() {
                        return (p, Vec::new(), 0f32);
                    }
                    let med = median_filter_2d_f32(&p, hh, ww, 3);
                    let mut ad: Vec<f32> =
                        p.iter().zip(&med).map(|(a, m)| (a - m).abs()).collect();
                    if ad.iter().any(|v| v.is_nan()) {
                        return (p, med, 0f32);
                    }
                    let mad = median_inplace(&mut ad);
                    let sigma = mad * sigma_k;
                    if sigma < 1e-6 {
                        return (p, med, 0f32);
                    }
                    (p, med, sigma)
                })
                .collect();
            let wws = [(w + 1) / 2, w / 2];
            // excess over the plane median in units of that plane's sigma, on the mosaic grid
            let zmos: Option<Vec<f32>> = star_support.map(|_| {
                let mut z = vec![0f32; h * w];
                z.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                    let py_ = y & 1;
                    let i = y >> 1;
                    for x in 0..w {
                        let (p, med, sigma) = &stage[py_ * 2 + (x & 1)];
                        if *sigma > 0f32 {
                            let k = i * wws[x & 1] + (x >> 1);
                            row[x] = (p[k] - med[k]) / *sigma;
                        }
                    }
                });
                z
            });
            out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                let py_ = y & 1;
                let i = y >> 1;
                for x in 0..w {
                    let px = x & 1;
                    let (p, med, sigma) = &stage[py_ * 2 + px];
                    let k = i * wws[px] + (x >> 1);
                    let v = p[k];
                    row[x] = v;
                    if *sigma > 0f32 && v - med[k] > thr * *sigma {
                        let star = match (&zmos, star_support) {
                            (Some(z), Some(sup)) => {
                                let mut near = f32::NEG_INFINITY;
                                if y > 0 { near = near.max(z[(y - 1) * w + x]); }
                                if y + 1 < h { near = near.max(z[(y + 1) * w + x]); }
                                if x > 0 { near = near.max(z[y * w + x - 1]); }
                                if x + 1 < w { near = near.max(z[y * w + x + 1]); }
                                near > sup
                            }
                            _ => false,
                        };
                        if !star {
                            row[x] = med[k];
                        }
                    }
                }
            });
        }
        out
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), out)
        .unwrap()
        .into_pyarray(py))
}

/// Everything `hot_pixel_rgb` decides, before any pixel is written: the luminance,
/// and (when something exceeded the cut) each flagged pixel's index and its
/// replacement. f32 arithmetic in numpy's order, box mean in f64.
enum HotRgb {
    /// MAD-based sigma < 1e-6: numpy falls back to `np.std`, which the caller runs.
    Degenerate,
    /// Nothing flagged (or a NaN made the MAD NaN, so nothing passes): the input is the answer.
    Clean(Vec<f32>),
    /// Luminance of the *input*, plus (pixel index, replacement rgb).
    Hot(Vec<f32>, Vec<(usize, [f32; 3])>),
}

fn hot_rgb_core(f: &[f32], h: usize, w: usize, threshold: f64) -> HotRgb {
    let (kr, kg, kb) = (0.299f64 as f32, 0.587f64 as f32, 0.114f64 as f32);
    let mut lum = vec![0f32; h * w];
    lum.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let b = (y * w + x) * 3;
            row[x] = kr * f[b] + kg * f[b + 1] + kb * f[b + 2];
        }
    });
    let med = median_filter_2d_f32(&lum, h, w, 3);
    let mut ad: Vec<f32> = lum.par_iter().zip(med.par_iter()).map(|(a, m)| (a - m).abs()).collect();
    if ad.par_iter().any(|v| v.is_nan()) {
        return HotRgb::Clean(lum);
    }
    let mad = median_inplace(&mut ad);
    drop(ad);
    let sigma = mad as f64 * 1.4826;
    if sigma < 1e-6 {
        return HotRgb::Degenerate;
    }
    let cut = (threshold * sigma) as f32;
    let hot: Vec<usize> = (0..h * w)
        .into_par_iter()
        .filter(|&i| lum[i] - med[i] > cut)
        .collect();
    if hot.is_empty() {
        return HotRgb::Clean(lum);
    }
    let repl: Vec<(usize, [f32; 3])> = hot
        .par_iter()
        .map(|&i| {
            let (y, x) = (i / w, i % w);
            let mut v = [0f32; 3];
            for ch in 0..3usize {
                let mut acc = 0.0f64;
                for dy in -1isize..=1 {
                    let yy = wavelet_symmetric_idx(y as isize + dy, h);
                    for dx in -1isize..=1 {
                        let xx = wavelet_symmetric_idx(x as isize + dx, w);
                        acc += f[(yy * w + xx) * 3 + ch] as f64;
                    }
                }
                v[ch] = (acc / 9.0) as f32;
            }
            (i, v)
        })
        .collect();
    HotRgb::Hot(lum, repl)
}

/// Write the replacements into `buf` and the luminance of each replaced pixel into `lum`.
fn hot_rgb_apply(buf: &mut [f32], lum: &mut [f32], repl: &[(usize, [f32; 3])]) {
    let (kr, kg, kb) = (0.299f64 as f32, 0.587f64 as f32, 0.114f64 as f32);
    for &(i, v) in repl {
        buf[i * 3] = v[0];
        buf[i * 3 + 1] = v[1];
        buf[i * 3 + 2] = v[2];
        lum[i] = kr * v[0] + kg * v[1] + kb * v[2];
    }
}

fn hot_rgb_check(rgb: &numpy::ndarray::ArrayView3<'_, f32>) -> PyResult<(usize, usize)> {
    let s = rgb.shape();
    if s[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("expected 3 channels"));
    }
    Ok((s[0], s[1]))
}

/// RGB hot-pixel removal on luminance, fused: luma, its 3x3 median, the MAD
/// threshold and the sparse 3x3 box-mean replacement (all three channels of a
/// flagged pixel), returning a new array.
///
/// Returns None when the MAD-based sigma is degenerate (< 1e-6: the numpy path
/// then falls back to `np.std`, which the caller runs). Otherwise
/// `(fixed_or_None, lum)`: `fixed` is None when nothing was flagged (the input
/// is already the answer), `lum` is the luminance of the returned image.
/// Bit-identical to `_fix_hot_rgb_impl`.
#[pyfunction]
fn hot_pixel_rgb<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f32>,
    threshold: f64,
) -> PyResult<Option<(Option<Bound<'py, PyArray3<f32>>>, Bound<'py, PyArray2<f32>>)>> {
    let arr = rgb.as_array();
    let (h, w) = hot_rgb_check(&arr)?;
    let f: &[f32] = arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("rgb must be contiguous"))?;
    let res = py.detach(|| match hot_rgb_core(f, h, w, threshold) {
        HotRgb::Degenerate => None,
        HotRgb::Clean(lum) => Some((None, lum)),
        HotRgb::Hot(mut lum, repl) => {
            let mut out = f.to_vec();
            hot_rgb_apply(&mut out, &mut lum, &repl);
            Some((Some(out), lum))
        }
    });
    Ok(res.map(|(out, lum)| {
        (
            out.map(|o| numpy::ndarray::Array3::from_shape_vec((h, w, 3), o).unwrap().into_pyarray(py)),
            numpy::ndarray::Array2::from_shape_vec((h, w), lum).unwrap().into_pyarray(py),
        )
    }))
}

/// `hot_pixel_rgb` that writes the replacements into `rgb` itself: only the few
/// flagged pixels are touched, so no 75 MB output array is allocated and copied.
/// Returns the luminance of the resulting image, or None for a degenerate MAD
/// (nothing is modified then).
#[pyfunction]
fn hot_pixel_rgb_inplace<'py>(
    py: Python<'py>,
    mut rgb: numpy::PyReadwriteArray3<'py, f32>,
    threshold: f64,
) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
    let (h, w) = hot_rgb_check(&rgb.as_array())?;
    let f = rgb
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("rgb must be contiguous"))?;
    let lum = py.detach(|| match hot_rgb_core(&*f, h, w, threshold) {
        HotRgb::Degenerate => None,
        HotRgb::Clean(lum) => Some(lum),
        HotRgb::Hot(mut lum, repl) => {
            hot_rgb_apply(f, &mut lum, &repl);
            Some(lum)
        }
    });
    Ok(lum.map(|l| numpy::ndarray::Array2::from_shape_vec((h, w), l).unwrap().into_pyarray(py)))
}

/// Luminance `0.299 r + 0.587 g + 0.114 b`, f32 in numpy's operation order
/// (the weak Python scalars round to f32 first), one parallel pass instead of
/// three temporaries.
#[pyfunction]
fn luminance_native<'py>(
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
// Debayer-adjacent kernels: sigma-clipped medians without numpy's strided copies,
// green G1/G2 equalisation, the Bayer-position grid equalisation
// ---------------------------------------------------------------------------

#[inline]
fn median_f32_as_f64(v: &mut [f32]) -> f64 {
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
fn sigma_clipped_median_f32(orig: &[f32], sigma: f64, iters: usize) -> f64 {
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
fn gather_plane(src: &[f32], w: usize, h: usize, chans: usize, ch: usize, py: usize, px: usize) -> Vec<f32> {
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
fn strided_sigma_clipped_median<'py>(
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
fn green_equalize_inplace<'py>(
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
fn bayer_grid_equalize_inplace<'py>(
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
fn pre_gradient_apply<'py>(
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
fn white_balance_body(
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
fn white_balance_apply<'py>(
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
fn white_balance_grayworld<'py>(
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

/// In-place twin of `white_balance_body`: same per-pixel f32 operations in the same
/// order, but each output pixel overwrites its input pixel, so no 75 MB output
/// array is allocated, first-touched and written.
fn white_balance_body_inplace(py: Python<'_>, flat: &mut [f32], w: usize, f: [f32; 3], divide: bool) {
    let (f0, f1, f2) = (f[0], f[1], f[2]);
    py.detach(|| {
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
        flat.par_chunks_mut(w * 3).for_each(|row| {
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
                row[x * 3] = if o0 < 0.0 { 0.0 } else { o0 };
                row[x * 3 + 1] = if o1 < 0.0 { 0.0 } else { o1 };
                row[x * 3 + 2] = if o2 < 0.0 { 0.0 } else { o2 };
            }
        });
    });
}

/// `white_balance_apply` writing into `img` itself.
#[pyfunction]
fn white_balance_apply_inplace<'py>(
    py: Python<'py>,
    mut img: numpy::PyReadwriteArray3<'py, f32>,
    factors: PyReadonlyArray1<'py, f32>,
    divide: bool,
) -> PyResult<()> {
    let shape = img.as_array().shape().to_vec();
    if shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("img must have 3 channels"));
    }
    let f = factors.as_slice()?;
    if f.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("factors must have 3 entries"));
    }
    let flat = img
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("img must be C-contiguous"))?;
    white_balance_body_inplace(py, flat, shape[1], [f[0], f[1], f[2]], divide);
    Ok(())
}

/// `white_balance_grayworld` writing into `img` itself (same float64-accumulated means).
#[pyfunction]
fn white_balance_grayworld_inplace<'py>(
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
    let mut acc = [0f64; 3];
    for px in flat.chunks_exact(3) {
        acc[0] += px[0] as f64;
        acc[1] += px[1] as f64;
        acc[2] += px[2] as f64;
    }
    let n = (h * w) as f64;
    let mean = [(acc[0] / n) as f32, (acc[1] / n) as f32, (acc[2] / n) as f32];
    let mm = ((mean[0] + mean[1]) + mean[2]) / 3.0f32;
    let scale = [
        mm / (mean[0] + 1e-12f32),
        mm / (mean[1] + 1e-12f32),
        mm / (mean[2] + 1e-12f32),
    ];
    white_balance_body_inplace(py, flat, w, scale, false);
    Ok(())
}

#[pymodule]
fn astro_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(originvision::originvision_score, m)?)?;
    m.add_function(wrap_pyfunction!(transient_triage::transient_triage_score, m)?)?;
    m.add_function(wrap_pyfunction!(sigma_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(online_sigma_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(online_sigma_clip_seed_burnin, m)?)?;
    m.add_function(wrap_pyfunction!(online_sigma_clip_fold_frame, m)?)?;
    m.add_function(wrap_pyfunction!(patch_weighted_sigma_combine, m)?)?;
    m.add_function(wrap_pyfunction!(median_combine, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(esd_combine, m)?)?;
    m.add_function(wrap_pyfunction!(linear_fit_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(ivw_combine, m)?)?;
    m.add_function(wrap_pyfunction!(ivw_combine_with_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(dwt2_native, m)?)?;
    m.add_function(wrap_pyfunction!(idwt2_native, m)?)?;
    m.add_function(wrap_pyfunction!(sigma_clipped_median_native, m)?)?;
    m.add_function(wrap_pyfunction!(hot_pixel_box_replace_native, m)?)?;
    m.add_function(wrap_pyfunction!(blind_match_hypotheses, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine_lanczos3, m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_diffusion, m)?)?;
    m.add_function(wrap_pyfunction!(lacosmic_reject_native, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_native, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_filter_native, m)?)?;
    m.add_function(wrap_pyfunction!(dbe_fit_surface, m)?)?;
    m.add_function(wrap_pyfunction!(dbe_sample_patches, m)?)?;
    m.add_function(wrap_pyfunction!(patch_entropy_batch, m)?)?;
    m.add_function(wrap_pyfunction!(detect_stars_matched_filter, m)?)?;
    m.add_function(wrap_pyfunction!(fit_rigid_ransac, m)?)?;
    m.add_function(wrap_pyfunction!(debayer_malvar, m)?)?;
    m.add_function(wrap_pyfunction!(debayer_menon2007, m)?)?;
    m.add_function(wrap_pyfunction!(bilateral_filter, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine_kernel_table, m)?)?;
    m.add_function(wrap_pyfunction!(gram_matrix_wide, m)?)?;
    m.add_function(wrap_pyfunction!(small_times_wide, m)?)?;
    m.add_function(wrap_pyfunction!(robust_pca_pre_svd_input, m)?)?;
    m.add_function(wrap_pyfunction!(robust_pca_iterate, m)?)?;
    m.add_function(wrap_pyfunction!(continuum_scale_moments, m)?)?;
    m.add_function(wrap_pyfunction!(fit_moffat_native, m)?)?;
    m.add_function(wrap_pyfunction!(fit_psf_moffat2d_native, m)?)?;
    m.add_function(wrap_pyfunction!(fit_psf_gauss2d_native, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_median_grid, m)?)?;
    m.add_function(wrap_pyfunction!(local_normalize_grid, m)?)?;
    m.add_function(wrap_pyfunction!(stamp_star_disks, m)?)?;
    m.add_function(wrap_pyfunction!(bresenham_line_native, m)?)?;
    m.add_function(wrap_pyfunction!(radial_bin_median, m)?)?;
    m.add_function(wrap_pyfunction!(aperture_photometry_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cfa_drizzle_frame, m)?)?;
    m.add_function(wrap_pyfunction!(drizzle_accumulate_lanczos3, m)?)?;
    m.add_function(wrap_pyfunction!(drizzle_splat_frame, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_frame_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(hot_pixel_bayer, m)?)?;
    m.add_function(wrap_pyfunction!(hot_pixel_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(pre_gradient_apply, m)?)?;
    m.add_function(wrap_pyfunction!(hot_pixel_rgb_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(luminance_native, m)?)?;
    m.add_function(wrap_pyfunction!(strided_sigma_clipped_median, m)?)?;
    m.add_function(wrap_pyfunction!(green_equalize_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(bayer_grid_equalize_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(white_balance_apply_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(white_balance_grayworld_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(white_balance_apply, m)?)?;
    m.add_function(wrap_pyfunction!(white_balance_grayworld, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
