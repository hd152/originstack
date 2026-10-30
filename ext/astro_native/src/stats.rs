//! Shared order statistics: medians, mean/std, sigma-clipped median.

use crate::*;

/// f32/f64 for `median_inplace`.
pub(crate) trait MedianFloat: Copy + PartialOrd + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> {
    const NAN: Self;
    const NEG_INFINITY: Self;
    const HALF: Self;
}

impl MedianFloat for f32 {
    const NAN: f32 = f32::NAN;
    const NEG_INFINITY: f32 = f32::NEG_INFINITY;
    const HALF: f32 = 0.5;
}

impl MedianFloat for f64 {
    const NAN: f64 = f64::NAN;
    const NEG_INFINITY: f64 = f64::NEG_INFINITY;
    const HALF: f64 = 0.5;
}

/// Median in the slice's own precision (f32 or f64). Tiny finite slices use
/// insertion-sized sorting; larger slices use quickselect (O(n), vs O(n log
/// n) full sort). Exact order statistics, so values match the numpy/sort-based
/// reference: odd N -> k-th element; even N -> mean of the middle pair. Empty
/// -> NaN.
#[inline]
pub(crate) fn median_inplace<T: MedianFloat>(v: &mut [T]) -> T {
    let n = v.len();
    if n == 0 {
        return T::NAN;
    }
    // Selection has appreciable setup overhead for tiny samples. Keep the
    // general quickselect path for larger stacks and NaN-containing slices.
    if n <= 8 && v.iter().all(|x| x.partial_cmp(x).is_some()) {
        for i in 1..n {
            let value = v[i];
            let mut j = i;
            while j > 0 && v[j - 1] > value {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = value;
        }
        if n % 2 == 1 {
            return v[n / 2];
        }
        return T::HALF * (v[n / 2 - 1] + v[n / 2]);
    }
    let (lo, hi) = select_middle(v);
    if n % 2 == 1 {
        hi
    } else {
        T::HALF * (lo + hi)
    }
}

/// Quickselect to the middle of `v` (non-empty): returns `(lo, hi)`, the
/// (n/2 - 1)-th and n/2-th order statistics for even n, and the same element
/// twice for odd n. Only an even count pays for the scan of the left partition
/// (its maximum; NaN never compares greater, as with the select's comparator).
#[inline]
pub(crate) fn select_middle<T: MedianFloat>(v: &mut [T]) -> (T, T) {
    let n = v.len();
    let mid = n / 2;
    let (_, &mut m, _) =
        v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        return (m, m);
    }
    let mut lo = T::NEG_INFINITY;
    for &x in v[..mid].iter() {
        if x > lo {
            lo = x;
        }
    }
    (lo, m)
}

/// Median of an f64 slice that is left untouched; empty -> 0.0 (the caller
/// treats "no samples" as "no offset", unlike `median_inplace`'s NaN).
pub(crate) fn median_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    median_inplace(&mut v.to_vec())
}

/// Median of an f32 slice, returned as f64 and averaged in f64 (the sigma-clip
/// callers compare it against f64 sums, so the even-count mean is not rounded
/// to f32 first, unlike `median_inplace::<f32>`).
#[inline]
pub(crate) fn median_f32_as_f64(v: &mut [f32]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let (lo, hi) = select_middle(v);
    if v.len() % 2 == 1 {
        hi as f64
    } else {
        0.5 * (lo as f64 + hi as f64)
    }
}

/// Population mean of a slice.
#[inline]
pub(crate) fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let s: f64 = v.iter().map(|&x| x as f64).sum();
    (s / v.len() as f64) as f32
}

/// Population standard deviation (ddof=0), matching numpy's nanstd default.
#[inline]
pub(crate) fn std_pop(v: &[f32]) -> f32 {
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
pub(crate) fn std_sample(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(v) as f64;
    let var = v.iter().map(|&x| (x as f64 - m) * (x as f64 - m)).sum::<f64>() / (n as f64 - 1.0);
    var.sqrt() as f32
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

/// Iterative sigma-clipped median on f32 storage: every sum/compare is done in
/// f64 (f32 -> f64 is exact), only the buffers are half the size, which is what
/// matters under memory contention.
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

#[pyfunction]
pub(crate) fn sigma_clipped_median_native(data: PyReadonlyArray1<'_, f32>, sigma: f64, iters: usize) -> f64 {
    let v: Vec<f32> = data.as_array().iter().copied().collect();
    sigma_clipped_median_f32(&v, sigma, iters)
}
