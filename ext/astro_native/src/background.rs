//! Background modelling: DBE surface fit and patch sampling, mesh medians, local normalisation.

use crate::*;

// ---------------------------------------------------------------------------
// DBE robust background-surface fit
// ---------------------------------------------------------------------------

/// Median of an f64 slice (copies + sorts; N is small — DBE patch counts).
pub(crate) fn median_f64(v: &[f64]) -> f64 {
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
pub(crate) fn dbe_accum(
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
pub(crate) fn dbe_solve(acc: &[f64; 9]) -> f64 {
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
pub(crate) fn dbe_fit_at(
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
pub(crate) fn dbe_fit_surface<'py>(
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
pub(crate) fn dbe_sample_patches<'py>(
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
pub(crate) fn patch_entropy_batch<'py>(
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
pub(crate) fn mesh_median_grid<'py>(
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
                median_inplace(&mut cell)
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
pub(crate) fn local_normalize_grid<'py>(
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
