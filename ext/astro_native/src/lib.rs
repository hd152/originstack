//! Native hot-path kernels for OriginStack.
//!
//! Currently exposes `sigma_clip_combine`, a per-pixel iterative sigma-clip /
//! winsorized combine over an `(N, H, W, C)` stack of aligned frames. It mirrors
//! the numpy reference in `src/stacking.py` (`_sigma_clip_tile`) but runs the
//! per-pixel loop in native code, parallelised across image rows with rayon, so
//! there is no per-tile float32 copy and no repeated whole-stack NaN passes.

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray4};
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

#[pymodule]
fn astro_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sigma_clip_combine, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
