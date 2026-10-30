//! Per-pixel stack combines: sigma-clip, online clip, median, percentile, ESD, linear fit, IVW, patch-weighted.

use crate::*;

/// Bit-packed per-sample activity mask used by the clipping kernels.
#[derive(Clone)]
pub(crate) struct ActiveMask {
    words: Vec<u64>,
}

impl ActiveMask {
    #[inline]
    fn new(len: usize) -> Self { Self { words: vec![0; (len + 63) / 64] } }
    #[inline]
    fn clear(&mut self) { self.words.fill(0); }
    #[inline]
    fn set_all_non_nan(&mut self, vals: &[f32]) {
        self.clear();
        for (i, &v) in vals.iter().enumerate() {
            if !v.is_nan() { self.set(i, true); }
        }
    }
    #[inline(always)]
    fn get(&self, i: usize) -> bool {
        (self.words[i >> 6] & (1u64 << (i & 63))) != 0
    }
    #[inline(always)]
    fn set(&mut self, i: usize, value: bool) {
        let bit = 1u64 << (i & 63);
        if value { self.words[i >> 6] |= bit; } else { self.words[i >> 6] &= !bit; }
    }
}

/// Center + spread for the active values, with the same zero-spread fallback as
/// the numpy reference (MAD -> std, std -> MAD).
#[inline]
pub(crate) fn center_spread(active: &[f32], scratch: &mut Vec<f32>, use_mad: bool) -> (f32, f32) {
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

pub(crate) fn sigma_clip_mask_std_fast(
    vals: &[f32], sigma: f32, max_iters: usize, active: &mut ActiveMask,
    gather: &mut Vec<f32>, scratch: &mut Vec<f32>, rejected: &mut Vec<usize>,
) {
    active.set_all_non_nan(vals);
    if vals.iter().any(|v| !v.is_finite()) {
        sigma_clip_mask(vals, sigma, max_iters, false, active, gather, scratch, rejected);
        return;
    }
    for iter in 0..max_iters {
        if iter != 0 {
            gather.clear();
            for i in 0..vals.len() { if active.get(i) { gather.push(vals[i]); } }
            if gather.is_empty() { break; }
        }
        let source = if iter == 0 { vals } else { gather.as_slice() };
        let center = mean(source);
        let mut spread = std_pop(source);
        if spread < 1e-12 {
            scratch.clear();
            scratch.extend_from_slice(source);
            let med = median_inplace(scratch);
            scratch.clear();
            scratch.extend(source.iter().map(|&x| (x - med).abs()));
            spread = median_inplace(scratch) * 1.4826;
        }
        let thresh = sigma * spread;
        rejected.clear();
        let mut survivors = 0usize;
        for i in 0..vals.len() {
            if active.get(i) {
                let deviation = (vals[i] - center).abs();
                if deviation <= thresh { survivors += 1; }
                else if deviation > thresh { rejected.push(i); }
            }
        }
        if survivors == 0 || rejected.is_empty() { break; }
        for &i in rejected.iter() { active.set(i, false); }
    }
}

/// numpy-`linear` percentile of an already-sorted slice (method='linear',
/// the numpy default): rank = p/100 * (n-1), linear interpolation.
#[inline]
pub(crate) fn percentile_sorted(sorted: &[f32], p: f64) -> f32 {
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
pub(crate) fn gather_tile(n: usize) -> usize {
    (32768 / n.max(1)).clamp(16, 256)
}

/// Row-parallel driver with a blocked gather-transpose, for a `work` closure
/// producing `K` per-pixel outputs (one per output buffer).
///
/// The naive per-pixel gather reads each pixel's N samples with a stride of a
/// whole frame (H*W*C floats — tens of MB): N concurrent read streams, which
/// defeats the hardware prefetcher and thrashes the TLB. Instead, per tile of
/// `T` pixels we copy each frame's contiguous row segment (sequential, one
/// stream at a time) into an L2-resident pixel-major block, then hand `work`
/// contiguous `&block[p*n..(p+1)*n]` slices.
pub(crate) fn row_parallel_k<const K: usize, S, Init, Work>(
    arr: &numpy::ndarray::ArrayView4<'_, f32>,
    h: usize,
    w: usize,
    c: usize,
    n: usize,
    init: Init,
    work: Work,
) -> [Vec<f32>; K]
where
    S: Send,
    Init: Fn() -> S + Sync,
    Work: Fn(&mut S, &[f32]) -> [f32; K] + Sync,
{
    let row_len = w * c;
    let frame_len = h * row_len;
    let tile = gather_tile(n);
    let data: Option<&[f32]> = arr.as_slice();

    let mut outs: [Vec<f32>; K] = std::array::from_fn(|_| vec![0f32; h * row_len]);
    if row_len == 0 {
        return outs;
    }
    // One entry per image row: that row's slice of each of the K outputs.
    let mut chunks: Vec<_> = outs.iter_mut().map(|o| o.chunks_mut(row_len)).collect();
    let rows: Vec<[&mut [f32]; K]> = (0..h)
        .map(|_| std::array::from_fn(|j| chunks[j].next().unwrap()))
        .collect();
    let min_rows = (h / (rayon::current_num_threads().max(1) * 4)).max(1);
    rows.into_par_iter()
        .enumerate()
        .with_min_len(min_rows)
        .for_each_init(
            || (init(), vec![0f32; tile * n]),
            |(state, block), (row, out_rows)| {
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
                                let r = work(state, &block[p * n..(p + 1) * n]);
                                for j in 0..K {
                                    out_rows[j][start + p] = r[j];
                                }
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
                            let r = work(state, &vals[..n]);
                            for j in 0..K {
                                out_rows[j][col] = r[j];
                            }
                        }
                    }
                }
            },
        );
    outs
}

/// `row_parallel_k` with a single output.
pub(crate) fn row_parallel<S, Init, Work>(
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
    let [out] = row_parallel_k::<1, _, _, _>(arr, h, w, c, n, init, |s, v| [work(s, v)]);
    out
}

/// Fill `active[i]` = survives sigma-clip (same iteration as the numpy
/// `_sigma_clip_tile` per-pixel logic). NaN samples start rejected.
pub(crate) fn sigma_clip_mask(
    vals: &[f32],
    sigma: f32,
    max_iters: usize,
    use_mad: bool,
    active: &mut ActiveMask,
    gather: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
    rejected: &mut Vec<usize>,
) {
    let n = vals.len();
    for i in 0..n {
        active.set(i, !vals[i].is_nan());
    }
    for _ in 0..max_iters {
        gather.clear();
        for i in 0..n {
            if active.get(i) {
                gather.push(vals[i]);
            }
        }
        if gather.is_empty() {
            break;
        }
        let (center, spread) = center_spread(gather, scratch, use_mad);
        let thresh = sigma * spread;
        rejected.clear();
        let mut survivors = 0usize;
        for i in 0..n {
            if active.get(i) {
                let deviation = (vals[i] - center).abs();
                if deviation <= thresh {
                    survivors += 1;
                } else if deviation > thresh {
                    rejected.push(i);
                }
            }
        }
        if survivors == 0 {
            break;
        }
        for &i in rejected.iter() {
            active.set(i, false);
        }
        if rejected.is_empty() {
            break;
        }
    }
}

/// `sigma_clip_mask` with `use_mad = true`, with less work per iteration and the same
/// `active` bit for bit: the median is selected in place on `gather` (the active
/// samples in frame order, exactly the copy `center_spread` selects on), the
/// deviations are taken from that reordered buffer (an order statistic of non-NaN,
/// non-negative values does not depend on input order), and the survivor count and
/// the "anything rejected" test are one pass. The `spread < 1e-12` fallback re-gathers
/// the samples in frame order for `std_pop`, whose f64 sum is order-dependent. A sample
/// that is +-inf can make a deviation NaN, which the quickselect orders by position, so
/// such a pixel-channel takes `sigma_clip_mask` itself.
pub(crate) fn sigma_clip_mask_fast(
    vals: &[f32],
    sigma: f32,
    max_iters: usize,
    active: &mut ActiveMask,
    gather: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
    rejected: &mut Vec<usize>,
) {
    let n = vals.len();
    let mut inf = false;
    for i in 0..n {
        let v = vals[i];
        active.set(i, !v.is_nan());
        inf |= v.is_infinite();
    }
    if inf {
        sigma_clip_mask(
            vals, sigma, max_iters, true, active, gather, scratch, rejected,
        );
        return;
    }
    for _ in 0..max_iters {
        gather.clear();
        for i in 0..n {
            if active.get(i) {
                gather.push(vals[i]);
            }
        }
        if gather.is_empty() {
            break;
        }
        let center = median_inplace(gather);
        scratch.clear();
        scratch.extend(gather.iter().map(|&x| (x - center).abs()));
        let mut spread = median_inplace(scratch) * 1.4826;
        if spread < 1e-12 {
            gather.clear();
            for i in 0..n {
                if active.get(i) {
                    gather.push(vals[i]);
                }
            }
            spread = std_pop(gather); // fallback
        }
        let thresh = sigma * spread;
        rejected.clear();
        let mut survivors = 0usize;
        for i in 0..n {
            if active.get(i) {
                let deviation = (vals[i] - center).abs();
                if deviation <= thresh {
                    survivors += 1;
                } else if deviation > thresh {
                    rejected.push(i);
                }
            }
        }
        if survivors == 0 || rejected.is_empty() {
            break;
        }
        for &i in rejected.iter() {
            active.set(i, false);
        }
    }
}

/// Combine one pixel's N samples. `vals`/`weights` are length N; NaN samples are
/// treated as already-rejected.
#[allow(clippy::too_many_arguments)]
pub(crate) fn combine_pixel(
    vals: &[f32],
    weights: Option<&[f32]>,
    sigma: f32,
    max_iters: usize,
    winsorize: bool,
    use_mad: bool,
    active: &mut ActiveMask,
    gather: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
    rejected: &mut Vec<usize>,
) -> f32 {
    let n = vals.len();
    // The unweighted, non-winsorized clipping modes are common stacking paths.
    // Use the fused mask path for those cases: it avoids rebuilding the
    // active sample list in center_spread and preserves the same fallback for
    // infinities and zero-spread pixels.
    if !use_mad && !winsorize && weights.is_none() {
        sigma_clip_mask_std_fast(
            vals, sigma, max_iters, active, gather, scratch, rejected,
        );
        let mut acc = 0.0f64;
        let mut count = 0usize;
        for i in 0..n {
            if active.get(i) {
                acc += vals[i] as f64;
                count += 1;
            }
        }
        return if count == 0 { 0.0 } else { (acc / count as f64) as f32 };
    }
    if use_mad && !winsorize && weights.is_none() {
        sigma_clip_mask_fast(
            vals, sigma, max_iters, active, gather, scratch, rejected,
        );
        let mut acc = 0.0f64;
        let mut count = 0usize;
        for i in 0..n {
            if active.get(i) {
                acc += vals[i] as f64;
                count += 1;
            }
        }
        return if count == 0 {
            0.0
        } else {
            (acc / count as f64) as f32
        };
    }

    for i in 0..n {
        active.set(i, !vals[i].is_nan());
    }

    for _ in 0..max_iters {
        gather.clear();
        for i in 0..n {
            if active.get(i) {
                gather.push(vals[i]);
            }
        }
        if gather.is_empty() {
            break;
        }
        let (center, spread) = center_spread(gather, scratch, use_mad);
        let thresh = sigma * spread;

        // Collect rejects while counting survivors, so the no-survivor case
        // can still retain the current mask without scanning the values twice.
        rejected.clear();
        let mut survivors = 0usize;
        for i in 0..n {
            if active.get(i) {
                let deviation = (vals[i] - center).abs();
                if deviation <= thresh {
                    survivors += 1;
                } else if deviation > thresh {
                    rejected.push(i);
                }
            }
        }
        if survivors == 0 {
            // All would be rejected -> keep the current mask, no change, stop.
            break;
        }
        for &i in rejected.iter() {
            active.set(i, false);
        }
        if rejected.is_empty() {
            break;
        }
    }

    if winsorize {
        // Recompute center/spread on the surviving set, clip ALL samples to the
        // boundary, then (weighted) mean over all clipped samples.
        gather.clear();
        for i in 0..n {
            if active.get(i) {
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
            if active.get(i) {
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
pub(crate) fn sigma_clip_combine<'py>(
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
                    ActiveMask::new(n),
                    Vec::<f32>::with_capacity(n),
                    Vec::<f32>::with_capacity(n),
                    Vec::<usize>::with_capacity(n),
                )
            },
            |(active, gather, scratch, rejected), vals| {
                combine_pixel(
                    vals, wref, sigma, max_iters, winsorize, use_mad, active, gather, scratch,
                    rejected,
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
/// running estimate rather than a divide-by-zero. Used by
/// `online_sigma_clip_seed_burnin` (streaming kernel).
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
pub(crate) fn burnin_seed_pixel(
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
/// `false`). Used by `online_sigma_clip_fold_frame`.
///
/// `n_acc<=0` is the "unseeded" sentinel `burnin_seed_pixel` returns for a
/// pixel no burn-in frame covered (large/drifting dithers can leave 100+ px
/// borders uncovered). Running the accept-test against that fabricated state
/// would reject the first real sample forever; instead initialize directly
/// from it.
///
/// Below `ONLINE_CLIP_MIN_SAMPLES` accepted samples there is no spread to
/// clip against (one sample has M2 = 0, so every later sample failed the
/// 1e-6 test and the pixel froze on a single frame), so samples are accepted
/// unconditionally until the state holds that many.
#[inline]
pub(crate) fn fold_pixel(mean: f64, m2: f64, n_acc: f64, x: f64, sigma: f64) -> (f64, f64, f64, bool) {
    if n_acc <= 0.0 {
        return (x, 0.0, 1.0, true);
    }
    if n_acc >= ONLINE_CLIP_MIN_SAMPLES {
        let std_est = (m2 / n_acc).max(1e-12).sqrt();
        if (x - mean).abs() > sigma * std_est {
            return (mean, m2, n_acc, false);
        }
    }
    let n_acc_new = n_acc + 1.0;
    let delta = x - mean;
    let new_mean = mean + delta / n_acc_new;
    let delta2 = x - new_mean;
    let new_m2 = m2 + delta * delta2;
    (new_mean, new_m2, n_acc_new, true)
}

/// Accepted samples a pixel needs before the online sigma-clip starts
/// rejecting. Mirrored by `ONLINE_CLIP_MIN_SAMPLES` in src/stacking.py.
pub(crate) const ONLINE_CLIP_MIN_SAMPLES: f64 = 3.0;

/// Seed a running Welford (mean, M2, n_acc) state from a small `(K,H,W,C)`
/// burn-in stack via one MAD-reject pass -- `burnin_seed_pixel`, run once over the whole
/// buffered window. `K` is expected small and bounded (e.g. 10 -- the
/// streaming product's burn-in size, not the N-samples-per-pixel gather
/// problem `row_parallel` solves for), so this uses a plain per-pixel gather
/// rather than the L2-tiling gather-transpose.
///
/// `coverage` is `(K,H,W)` float32 (>=0.5 = that frame's warp actually
/// covers this pixel), one mask per burn-in frame -- a burn-in window mixes
/// several frames' warps, each with its own out-of-frame zero-fill region
/// (large dithers/shifts easily reach 100+ px on real sessions), so this MUST exclude uncovered samples from
/// the median/MAD/mean/M2 computation per `burnin_seed_pixel`'s `valid` gate
/// -- otherwise zero-fill pixels masquerade as real (very dark) samples at
/// every frame's border.
///
/// Returns `(mean, m2, n_acc)` each `(H, W, C)` float64, plus the
/// rejected-sample count (summed over all pixels; uncovered samples count
/// as rejected too, since they never became part of n_acc).
#[pyfunction]
#[pyo3(signature = (burn_stack, coverage, sigma=3.0))]
pub(crate) fn online_sigma_clip_seed_burnin<'py>(
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
pub(crate) fn online_sigma_clip_fold_frame<'py>(
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
pub(crate) fn median_combine<'py>(
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
pub(crate) fn percentile_clip_combine<'py>(
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
pub(crate) fn esd_combine<'py>(
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
            || (ActiveMask::new(n), Vec::<f32>::with_capacity(n)),
            |(active, gather), vals| {
                for k in 0..n {
                    active.set(k, !vals[k].is_nan());
                }
                for i in 0..mo {
                    gather.clear();
                    for k in 0..n {
                        if active.get(k) {
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
                        if active.get(k) {
                            let dv = (vals[k] - mn).abs() / sd;
                            if dv > max_dev {
                                max_dev = dv;
                                idx = k;
                            }
                        }
                    }
                    let lam = lut_flat[m * mo + i];
                    if (max_dev as f64) > lam && idx != usize::MAX {
                        active.set(idx, false); // m >= 3 -> m-1 >= 2 survive
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
                    if active.get(k) {
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
pub(crate) fn linear_fit_clip_combine<'py>(
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
                    ActiveMask::new(n),                   // accepted
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
                    accepted.set(r, !y_sorted[r].is_nan());
                }

                for _ in 0..iters {
                    let mut n_acc = 0usize;
                    let mut sum_x = 0f64;
                    let mut sum_y = 0f64;
                    for r in 0..n {
                        if accepted.get(r) {
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
                        if accepted.get(r) {
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
                        if accepted.get(r) {
                            let e = y_sorted[r] - (intercept + slope * r as f64);
                            ss += e * e;
                        }
                    }
                    let sigma = ((ss / (n_acc as f64 - 1.0).max(1.0)).sqrt()).max(1e-6);

                    newly_rejected.clear();
                    for r in 0..n {
                        if accepted.get(r) {
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
                        accepted.set(r, false);
                    }
                }

                let mut acc = 0f64;
                let mut wsum = 0f64;
                for r in 0..n {
                    if accepted.get(r) {
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

/// One pixel's inverse-variance weighting, shared by `ivw_combine` and
/// `ivw_combine_with_sigma`: returns `(sum w*x, sum w)` in f64.
#[inline]
pub(crate) fn ivw_pixel(
    vals: &[f32],
    noise2: &[f64],
    shot_model: Option<(&Vec<f64>, f64)>,
    wref: Option<&[f32]>,
) -> (f64, f64) {
    let mut acc = 0f64;
    let mut wsum = 0f64;
    for k in 0..vals.len() {
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
    (acc, wsum)
}

#[pyfunction]
#[pyo3(signature = (data, noise, sky=None, gain=None, weights=None))]
pub(crate) fn ivw_combine<'py>(
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
            let (acc, wsum) = ivw_pixel(vals, &noise2, shot_model, wref);
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
/// opt-in diagnostic path. The per-pixel weighting is the shared `ivw_pixel`.
#[pyfunction]
#[pyo3(signature = (data, noise, sky=None, gain=None, weights=None))]
pub(crate) fn ivw_combine_with_sigma<'py>(
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

    let [result, wsum_out] = py.detach(|| {
        row_parallel_k::<2, _, _, _>(&arr, h, w, c, n, || (), |_, vals| {
            let (acc, wsum) = ivw_pixel(vals, &noise2, shot_model, wref);
            let result = if wsum <= 0.0 { 0.0 } else { (acc / wsum) as f32 };
            [result, wsum as f32]
        })
    });
    let result_arr = numpy::ndarray::Array3::from_shape_vec((h, w, c), result)
        .expect("shape mismatch building ivw_combine_with_sigma result");
    let wsum_arr = numpy::ndarray::Array3::from_shape_vec((h, w, c), wsum_out)
        .expect("shape mismatch building ivw_combine_with_sigma wsum");
    Ok((result_arr.into_pyarray(py), wsum_arr.into_pyarray(py)))
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
///
/// The MAD sigma-clip uses `sigma_clip_mask_fast` (fewer passes per iteration), and
/// with a patch grid the four grid corners every frame needs are tabulated once per
/// row and grid column (same f32 bilinear expression, same f64 weight product as a
/// direct per-frame lookup). (The `_fast` suffix is historical: the slower original
/// `patch_weighted_sigma_combine` was removed in crate 0.53.0.)
#[pyfunction]
#[pyo3(signature = (data, qmaps, gweights=None, sigma=3.0, max_iters=3, use_mad=true, grid_geom=None))]
pub(crate) fn patch_weighted_sigma_combine_fast<'py>(
    py: Python<'py>,
    data: PyReadonlyArray4<'py, f32>,
    qmaps: PyReadonlyArray3<'py, f32>,
    gweights: Option<PyReadonlyArray1<'py, f32>>,
    sigma: f32,
    max_iters: usize,
    use_mad: bool,
    grid_geom: Option<(f64, f64, f64, f64)>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    patch_weighted_sigma_combine_impl(py, data, qmaps, gweights, sigma, max_iters, use_mad, grid_geom)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn patch_weighted_sigma_combine_impl<'py>(
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
            let mut active = ActiveMask::new(n);
            let mut gather: Vec<f32> = Vec::with_capacity(n);
            let mut scratch: Vec<f32> = Vec::with_capacity(n);
            let mut rejected: Vec<usize> = Vec::with_capacity(n);
            let mut reject_count = vec![0u32; n]; // per-frame rejected-channel count
            // patch grid: corners[(gx0 * n + f)] = this row's four grid
            // values around grid column gx0 for frame f
            let corners: Vec<[f32; 4]> = match &row_tab {
                Some((gy0, _)) => {
                    let gy1 = (gy0 + 1).min(qh - 1);
                    let mut t = Vec::with_capacity(qw * n);
                    for gx0 in 0..qw {
                        let gx1 = (gx0 + 1).min(qw - 1);
                        for f in 0..n {
                            t.push([qm[[f, *gy0, gx0]], qm[[f, *gy0, gx1]],
                                    qm[[f, gy1, gx0]], qm[[f, gy1, gx1]]]);
                        }
                    }
                    t
                }
                _ => Vec::new(),
            };
            let gw64: Vec<f64> = (0..n).map(|f| gwref.map(|g| g[f] as f64).unwrap_or(1.0)).collect();
            let inv_c = 1.0f64 / c as f64;

            // Per-pixel combine given contiguous per-channel sample slices in
            // `block` (layout [(p*c + ch)*n + f]) — see gather-transpose below.
            let mut combine_col = |block: &[f32], p: usize, col: usize,
                                   active: &mut ActiveMask, gather: &mut Vec<f32>,
                                   scratch: &mut Vec<f32>, reject_count: &mut [u32],
                                   out_row: &mut [f32]| {
                for rc in reject_count.iter_mut() {
                    *rc = 0;
                }
                for ch in 0..c {
                    let chan = &block[(p * c + ch) * n..][..n];
                    if use_mad {
                        sigma_clip_mask_fast(chan, sigma, max_iters, active, gather, scratch,
                                             &mut rejected);
                    } else {
                        sigma_clip_mask(chan, sigma, max_iters, use_mad, active, gather, scratch,
                                        &mut rejected);
                    }
                    for f in 0..n {
                        if !active.get(f) {
                            reject_count[f] += 1;
                        }
                    }
                }
                let mut wsum = 0f64;
                let mut accs = [0f64; 8]; // supports up to 8 channels
                if let (false, Some((_, fy)), Some((gx0s, fxs))) = (corners.is_empty(), &row_tab, &col_tab) {
                    let (fx, fy) = (fxs[col], *fy);
                    let cell = &corners[gx0s[col] * n..(gx0s[col] + 1) * n];
                    for f in 0..n {
                        let rej_frac = reject_count[f] as f64 * inv_c;
                        let [q00, q01, q10, q11] = cell[f];
                        let top_v = q00 + (q01 - q00) * fx;
                        let bot_v = q10 + (q11 - q10) * fx;
                        let qwt = (top_v + (bot_v - top_v) * fy) as f64;
                        let wt = qwt * gw64[f] * (1.0 - rej_frac);
                        if wt == 0.0 {
                            continue;
                        }
                        for ch in 0..c {
                            accs[ch] += wt * block[(p * c + ch) * n + f] as f64;
                        }
                        wsum += wt;
                    }
                } else {
                // full-resolution weights (no patch grid)
                for f in 0..n {
                    let rej_frac = reject_count[f] as f64 * inv_c;
                    let qwt = qm[[f, row, col]] as f64;
                    let wt = qwt * gw64[f] * (1.0 - rej_frac);
                    if wt == 0.0 {
                        continue;
                    }
                    for ch in 0..c {
                        accs[ch] += wt * block[(p * c + ch) * n + f] as f64;
                    }
                    wsum += wt;
                }
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
