//! Image filters: median, gaussian, bilateral, anisotropic diffusion, LA-Cosmic, hot-pixel box replace, harmonic fill.

use crate::*;

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
// reflection -- the same convention as `reflect_idx` (numpy
// `pad(mode='symmetric')`), reused here rather than redefined.

#[pyfunction]
pub(crate) fn hot_pixel_box_replace_native<'py>(
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
        par_rows(&mut result, w * c, |y, row_out| {
            for x in 0..w {
                if mask_flat[y * w + x] == 0 {
                    continue;
                }
                for ch in 0..c {
                    let mut acc = 0.0f64;
                    for dy in -1isize..=1 {
                        let yy = reflect_idx(y as isize + dy, h);
                        for dx in -1isize..=1 {
                            let xx = reflect_idx(x as isize + dx, w);
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

/// Perona-Malik anisotropic diffusion, `iterations` Jacobi steps with a
/// periodic (np.roll) boundary. Matches the numpy reference; returns the
/// diffused float64 image (H,W,C) BEFORE the Python-side star-mask blend/clip.
#[pyfunction]
#[pyo3(signature = (data, iterations, kappa, gamma, option))]
pub(crate) fn anisotropic_diffusion<'py>(
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
                par_rows(&mut next, w, |y, out_row| {
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
    par_rows(&mut out, w * c, |y, orow| {
        for (ch, buf) in chans.iter().enumerate() {
            for x in 0..w {
                orow[x * c + ch] = buf[y * w + x];
            }
        }
    });
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, c), out).unwrap().into_pyarray(py))
}

/// scipy `mode='reflect'` boundary index: (d c b a | a b c d | d c b a) — the
/// edge value is duplicated (index -1 == index 0), not a full mirror without
/// repeat. Looped so it is correct even if an offset exceeds one bounce.
#[inline]
pub(crate) fn reflect_idx(i: isize, n: usize) -> usize {
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
pub(crate) fn median9(p: &mut [f32; 9]) -> f32 {
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
pub(crate) fn sort2_idx(p: &mut [f32; 9], i: usize, j: usize) {
    let (a, b) = (p[i], p[j]);
    p[i] = a.min(b);
    p[j] = a.max(b);
}

/// Interior of one output row of the 3x3 median: `row_out[1..w-1]` from the three input rows
/// starting at `r0`, `r1`, `r2`. The compare-exchange network is branchless min/max, so the
/// compiler vectorises it across `x` -- 4 lanes on baseline x86-64 (SSE2), 8 with AVX2.
#[inline(always)]
pub(crate) fn median3_interior_body(data: &[f32], r0: usize, r1: usize, r2: usize, w: usize, row_out: &mut [f32]) {
    for x in 1..w - 1 {
        let mut p = [
            data[r0 + x - 1], data[r0 + x], data[r0 + x + 1],
            data[r1 + x - 1], data[r1 + x], data[r1 + x + 1],
            data[r2 + x - 1], data[r2 + x], data[r2 + x + 1],
        ];
        row_out[x] = median9(&mut p);
    }
}

// Dispatches `median3_interior_body` to an AVX2 twin at run time; min/max are exact, so the
// result is identical to the baseline build's.
avx2_twin!(pub(crate) fn median3_interior(data: &[f32], r0: usize, r1: usize, r2: usize, w: usize, row_out: &mut [f32]) => median3_interior_body);

/// Windowed median filter (odd `size`, reflect boundary), row-parallel.
/// The window is tiny (9 or 25 elements for size 3/5) so a per-pixel gather
/// beats scipy's generic rank-filter machinery. Interior pixels (no boundary
/// reflection possible) take a fast path: contiguous row-segment reads with no
/// per-tap reflect_idx, then a branchless median network (3x3) or quickselect
/// (5x5) instead of a full comparator sort. f32 (not f64): used by
/// lacosmic_reject_native and the hot-pixel detector, both of which run inside
/// many concurrent ProcessPoolExecutor workers, where halving the bytes moved
/// per call directly reduces shared memory-bandwidth contention.
pub(crate) fn median_filter_2d_f32(data: &[f32], h: usize, w: usize, size: usize) -> Vec<f32> {
    let half = (size / 2) as isize;
    let hu = size / 2;
    let mut out = vec![0f32; h * w];
    par_rows(&mut out, w, |y, row_out| {
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
            median3_interior(data, (y - 1) * w, y * w, (y + 1) * w, w, row_out);
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
pub(crate) fn laplacian_2d_f32(data: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; h * w];
    par_rows(&mut out, w, |y, row_out| {
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
#[pyo3(signature = (data, sigclip=4.5, objlim=5.0, gain=1.0, readnoise=6.5,
                    noise_a=None, noise_b=None))]
pub(crate) fn lacosmic_reject_native<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f32>,
    sigclip: f64,
    objlim: f64,
    gain: f64,
    readnoise: f64,
    noise_a: Option<Vec<f64>>,
    noise_b: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    // Per-channel measured noise model, noise^2 = a[c] + b[c] * max(med5, 0),
    // replacing the gain/readnoise one when given (src/stacking.py).
    let per_ch: Option<([f32; 3], [f32; 3])> = match (&noise_a, &noise_b) {
        (Some(a), Some(b)) if a.len() == 3 && b.len() == 3 => Some((
            [a[0] as f32, a[1] as f32, a[2] as f32],
            [b[0] as f32, b[1] as f32, b[2] as f32],
        )),
        (None, None) => None,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "noise_a and noise_b must both be 3-element sequences",
            ))
        }
    };
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
                    par_rows(&mut chd, w, |y, row| {
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
                let var = match per_ch {
                    Some((a, b)) => a[ch] + b[ch] * med5[i].max(0.0),
                    None => med5[i].max(0.0) / gain + rn_term,
                };
                let noise = var.sqrt().max(1e-6);
                *sv = f / (2.0 * noise);
            });
            let smed = median_filter_2d_f32(&sarr, h, w, 3);

            par_rows(&mut out, w * c, |y, row_out| {
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
pub(crate) fn median_filter_native<'py>(
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
pub(crate) fn gaussian_kernel1d(sigma: f64, truncate: f64) -> Vec<f64> {
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

/// Separable correlation of an `(h, w)` row-major image with an odd-length 1D kernel along both
/// axes (rows, then columns), reflect boundary -- `scipy.ndimage.correlate1d(mode='reflect')`
/// per axis. Parallel over output rows.
///
/// Each output is `0 + k[0]*s0 + k[1]*s1 + ...` accumulated in tap order, which is what a
/// per-pixel loop computes. The interior is therefore done tap-major through `tap_accumulate`
/// (one contiguous multiply-add sweep per tap, AVX2 where available) with the same bits; only
/// the `half` edge columns on each side, where taps reflect, take the per-pixel path.
pub(crate) fn separable_convolve_f64(img: &[f64], h: usize, w: usize, k: &[f64]) -> Vec<f64> {
    debug_assert!(k.len() % 2 == 1);
    let half = k.len() / 2;

    // Pass 1: along axis 1, within each row.
    let mut tmp = vec![0f64; h * w];
    par_rows(&mut tmp, w, |y, out_row| {
        let row = &img[y * w..(y + 1) * w];
        let n_int = w.saturating_sub(2 * half);
        let (lo, hi) = if n_int > 0 { (half, w - half) } else { (w, w) };
        if n_int > 0 {
            // output x reads row[x + t - half]; for x = half + i that is row[t + i]
            tap_accumulate(&mut out_row[half..half + n_int], k, |t| &row[t..]);
        }
        for x in (0..lo).chain(hi..w) {
            let mut acc = 0f64;
            for (t, &kv) in k.iter().enumerate() {
                let xi = reflect_idx(x as isize + t as isize - half as isize, w);
                acc += kv * row[xi];
            }
            out_row[x] = acc;
        }
    });

    // Pass 2: along axis 0; each tap contributes one (reflected) whole row.
    let mut out = vec![0f64; h * w];
    par_rows(&mut out, w, |y, out_row| {
        tap_accumulate(out_row, k, |t| {
            let yi = reflect_idx(y as isize + t as isize - half as isize, h);
            &tmp[yi * w..(yi + 1) * w]
        });
    });
    out
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
pub(crate) fn gaussian_filter_native<'py>(
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
    let out = py.detach(|| separable_convolve_f64(flat, h, w, &kernel));

    let arr2 = numpy::ndarray::Array2::from_shape_vec((h, w), out)
        .expect("shape mismatch building gaussian_filter_native output");
    Ok(arr2.into_pyarray(py))
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
pub(crate) fn bilateral_filter<'py>(
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
        par_rows(&mut out, w * c, |y, row_out| {
            let interior_y = y >= radius && y + radius < h;
            let mut center = [0f64; 8];
            let mut neighbor = [0f64; 8];
            for x in 0..w {
                let base = y * w * c + x * c;
                let interior = interior_y && x >= radius && x + radius < w;
                let mut acc = [0f64; 8];
                let mut wsum = 0f64;

                // RGB is the production path. Keeping its three channels in
                // locals avoids the generic temporary channel array and loop
                // in the innermost neighbour operation.
                if c == 3 {
                    let cr = flat[base] as f64;
                    let cg = flat[base + 1] as f64;
                    let cb = flat[base + 2] as f64;
                    for dy in -r..=r {
                        let yy = if interior { (y as isize + dy) as usize } else { mirror_idx(y as isize + dy, h) };
                        let spatial_row = ((dy + r) as usize) * (2 * radius + 1);
                        for dx in -r..=r {
                            let xx = if interior { (x as isize + dx) as usize } else { mirror_idx(x as isize + dx, w) };
                            let nbase = yy * w * 3 + xx * 3;
                            let nr = flat[nbase] as f64;
                            let ng = flat[nbase + 1] as f64;
                            let nb = flat[nbase + 2] as f64;
                            let dr = nr - cr;
                            let dg = ng - cg;
                            let db = nb - cb;
                            let sw = spatial_w[spatial_row + (dx + r) as usize];
                            let wt = sw * (-(dr * dr + dg * dg + db * db) * inv_2c2).exp();
                            acc[0] += nr * wt;
                            acc[1] += ng * wt;
                            acc[2] += nb * wt;
                            wsum += wt;
                        }
                    }
                    let denom = wsum.max(1e-12);
                    row_out[x * 3] = (acc[0] / denom) as f32;
                    row_out[x * 3 + 1] = (acc[1] / denom) as f32;
                    row_out[x * 3 + 2] = (acc[2] / denom) as f32;
                    continue;
                }

                for ch in 0..c {
                    center[ch] = flat[base + ch] as f64;
                }
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

/// Fill a boolean hole mask by Jacobi iteration on a nearest-valid initial
/// surface. This is the small-mesh counterpart to the image-scale diffusion
/// kernels: the whole operation stays native so Python does not allocate a
/// padded array and four temporaries on every iteration.
#[pyfunction]
#[pyo3(signature = (grid, holes, iters=500, tolerance=1e-3))]
pub(crate) fn harmonic_fill<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<'py, f64>,
    holes: PyReadonlyArray2<'py, bool>,
    iters: usize,
    tolerance: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let g = grid.as_array();
    let h = holes.as_array();
    if g.shape() != h.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("grid and holes shape mismatch"));
    }
    let rows = g.shape()[0];
    let cols = g.shape()[1];
    if rows == 0 || cols == 0 {
        return Ok(numpy::ndarray::Array2::zeros((rows, cols)).into_pyarray(py));
    }
    let gs = g.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("grid must be contiguous"))?;
    let hs = h.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("holes must be contiguous"))?;
    let out = py.detach(|| {
        let mut result = gs.to_vec();
        let mut nearest = vec![usize::MAX; rows * cols];
        for y in 0..rows {
            for x in 0..cols {
                let i = y * cols + x;
                if !hs[i] {
                    nearest[i] = i;
                }
            }
        }
        for y in 0..rows {
            for x in 0..cols {
                let i = y * cols + x;
                if nearest[i] != usize::MAX {
                    continue;
                }
                let mut best = usize::MAX;
                let mut best_d = usize::MAX;
                for yy in 0..rows {
                    for xx in 0..cols {
                        let j = yy * cols + xx;
                        if nearest[j] != j {
                            continue;
                        }
                        let d = y.abs_diff(yy) + x.abs_diff(xx);
                        if d < best_d {
                            best_d = d;
                            best = j;
                        }
                    }
                }
                if best != usize::MAX {
                    nearest[i] = best;
                    result[i] = gs[best];
                } else {
                    result[i] = 0.0;
                }
            }
        }
        let mut next = result.clone();
        for _ in 0..iters {
            let mut step = 0.0_f64;
            for y in 0..rows {
                for x in 0..cols {
                    let i = y * cols + x;
                    if !hs[i] {
                        continue;
                    }
                    let up = result[y.saturating_sub(1) * cols + x];
                    let down = result[y.saturating_add(1).min(rows - 1) * cols + x];
                    let left = result[y * cols + x.saturating_sub(1)];
                    let right = result[y * cols + x.saturating_add(1).min(cols - 1)];
                    let value = 0.25 * (up + down + left + right);
                    step = step.max((value - result[i]).abs());
                    next[i] = value;
                }
            }
            for y in 0..rows {
                for x in 0..cols {
                    let i = y * cols + x;
                    if hs[i] {
                        result[i] = next[i];
                    }
                }
            }
            if step < tolerance {
                break;
            }
        }
        result
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((rows, cols), out).unwrap().into_pyarray(py))
}
