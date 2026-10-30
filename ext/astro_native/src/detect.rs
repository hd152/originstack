//! Star detection (matched filter), mesh helpers and star-disk stamping.

use crate::*;

// ============ Matched-filter star detection ============
//
// Mirrors src/star_detect.py::_detect_stars_matched_filter_numpy exactly
// (same mesh-median/sigma construction, same hand-rolled bilinear upsample,
// same separable Gaussian blur, same matched-filter SNR statistic, same
// two-pass centroid refinement). See that module's docstring for the
// validation history -- this is not a first-draft algorithm.

/// 1D Gaussian kernel, unit sum, radius = ceil(3*sigma).
pub(crate) fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
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
pub(crate) fn separable_blur(img: &[f64], h: usize, w: usize, sigma: f64) -> Vec<f64> {
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

/// Per-cell median (use_mad=false) or 1.4826*MAD sigma (use_mad=true) of an
/// f32 image, as the smoothed (ny, nx) mesh grid. `mesh_bilinear` evaluates the
/// cell-centre-aligned upsample of it at any pixel -- the same arithmetic the
/// numpy mirror's full-resolution `_bilinear_upsample` uses, without
/// materialising a full-frame f64 map.
pub(crate) fn local_mesh_grids(img: &[f32], h: usize, w: usize, cell: usize) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let ny = (h / cell.max(1)).max(1);
    let nx = (w / cell.max(1)).max(1);
    // One gather + selection per cell for both statistics: the median grid
    // and the MAD grid start from the same cell median, so computing it once
    // gives the same two grids the two separate passes did.
    let both: Vec<(f64, f64)> = (0..ny * nx)
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
                vals.extend_from_slice(&img[y * w + x0..y * w + x1]);
            }
            let med = median_inplace(&mut vals) as f64;
            for v in vals.iter_mut() {
                *v = (*v as f64 - med).abs() as f32;
            }
            (med, 1.4826 * (median_inplace(&mut vals) as f64).max(1e-9))
        })
        .collect();
    let med_grid: Vec<f64> = both.iter().map(|p| p.0).collect();
    let mad_grid: Vec<f64> = both.iter().map(|p| p.1).collect();
    // Smooth the small mesh grid (blocky-cell artifacts) before upsampling,
    // not the full-resolution field after: same intent (soften cell-to-cell
    // jumps) at a few thousand times less work.
    (separable_blur(&med_grid, ny, nx, 0.3), separable_blur(&mad_grid, ny, nx, 0.3), ny, nx)
}

/// `mesh_bilinear`'s first (x) interpolation, tabulated for every grid row:
/// `out[gy*w + x] = grid[gy,gx0]*(1-fx) + grid[gy,gx1]*fx`. Combined per pixel
/// as `t[gy0]*(1-fy) + t[gy1]*fy` this is exactly `mesh_bilinear`'s arithmetic.
pub(crate) fn mesh_xinterp_rows(grid: &[f64], ny: usize, nx: usize, xaxis: &[(usize, usize, f64)]) -> Vec<f64> {
    let w = xaxis.len();
    let mut out = vec![0f64; ny * w];
    for gy in 0..ny {
        let g = &grid[gy * nx..(gy + 1) * nx];
        for (x, &(gx0, gx1, fx)) in xaxis.iter().enumerate() {
            out[gy * w + x] = g[gx0] * (1.0 - fx) + g[gx1] * fx;
        }
    }
    out
}

/// `out[x] = sum_t k[t] * row(t)[x]`, accumulated tap by tap from 0.0.
/// Separate multiply and add (Rust never contracts to FMA), so the AVX2
/// build below gives identical bits.
#[inline(always)]
pub(crate) fn tap_accumulate_body<'a>(out: &mut [f64], k: &[f64], row: impl Fn(usize) -> &'a [f64]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for (t, &kv) in k.iter().enumerate() {
        let sl = &row(t)[..out.len()];
        for (o, &s) in out.iter_mut().zip(sl.iter()) {
            *o += kv * s;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn tap_accumulate_avx2<'a>(out: &mut [f64], k: &[f64], row: impl Fn(usize) -> &'a [f64]) {
    tap_accumulate_body(out, k, row)
}

pub(crate) fn tap_accumulate<'a>(out: &mut [f64], k: &[f64], row: impl Fn(usize) -> &'a [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU reports AVX2.
            unsafe { return tap_accumulate_avx2(out, k, row) }
        }
    }
    tap_accumulate_body(out, k, row)
}

/// (lo, hi, frac) of the cell-centre-aligned bilinear upsample along one axis.
#[inline(always)]
pub(crate) fn mesh_axis(i: usize, n: usize, cellf: f64) -> (usize, usize, f64) {
    let g = i as f64 / cellf - 0.5;
    let g0 = g.floor().max(0.0).min((n - 1) as f64) as usize;
    let g1 = (g0 + 1).min(n - 1);
    (g0, g1, (g - g0 as f64).clamp(0.0, 1.0))
}

#[inline(always)]
pub(crate) fn mesh_bilinear(grid: &[f64], nx: usize, ya: (usize, usize, f64), xa: (usize, usize, f64)) -> f64 {
    let (gy0, gy1, fy) = ya;
    let (gx0, gx1, fx) = xa;
    let v00 = grid[gy0 * nx + gx0];
    let v01 = grid[gy0 * nx + gx1];
    let v10 = grid[gy1 * nx + gx0];
    let v11 = grid[gy1 * nx + gx1];
    let v0 = v00 * (1.0 - fx) + v01 * fx;
    let v1 = v10 * (1.0 - fx) + v11 * fx;
    v0 * (1.0 - fy) + v1 * fy
}

/// Output rows per band in `detect_stars_matched_filter`'s blur/SNR/local-max
/// pass (its per-band scratch is ~(band + ~20) rows x width x 8 bytes, twice).
pub(crate) const DETECT_BAND: usize = 32;

#[pyfunction]
#[pyo3(signature = (image, fwhm, k_confirm, cell, roundness_max, min_pixels))]
pub(crate) fn detect_stars_matched_filter<'py>(
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
    let owned: Vec<f32>;
    let src: &[f32] = match arr.as_slice() {
        Some(s) => s,
        None => {
            owned = arr.iter().copied().collect();
            &owned
        }
    };

    // Memory-light, same arithmetic as the numpy mirror: the background and
    // noise maps are evaluated from their small mesh grids where needed instead
    // of being upsampled to full-frame f64 arrays, the residual is formed row
    // by row inside the first blur pass, and the second pass divides by the
    // noise directly; both passes and the local-max test run in row bands with
    // cache-sized scratch (no full-frame arrays at all). Per-pixel sums keep
    // their tap order, so results are identical.
    let rows: Vec<[f64; 10]> = py.detach(|| {
        let (bg_grid, sg_grid, ny, nx) = local_mesh_grids(src, h, w, cell);
        let cellf = cell as f64;
        let xaxis: Vec<(usize, usize, f64)> = (0..w).map(|x| mesh_axis(x, nx, cellf)).collect();
        let bg_at = |y: usize, x: usize| mesh_bilinear(&bg_grid, nx, mesh_axis(y, ny, cellf), xaxis[x]);
        let lum = |i: usize| src[i] as f64;
        // x-interpolated grid rows: the per-pixel maps below are then one
        // lerp between two table rows -- mesh_bilinear's arithmetic exactly.
        let bg_rows = mesh_xinterp_rows(&bg_grid, ny, nx, &xaxis);
        let sg_rows = mesh_xinterp_rows(&sg_grid, ny, nx, &xaxis);

        let sigma_k = fwhm / 2.3548;
        let k1 = gaussian_kernel_1d(sigma_k);
        let kernel_norm: f64 = k1.iter().map(|&v| v * v).sum();
        let half = k1.len() / 2;

        // Local-maxima + threshold + border exclusion.
        let footprint = (fwhm.round() as isize).max(3);
        let fhalf = footprint / 2;
        let fh = fhalf as usize;
        let border = (cell / 2).max((2.0 * (3.0 * fwhm / 2.3548).ceil()) as usize);

        // The blur, SNR and local-maximum test run in bands of output rows,
        // each band computing only the residual-blur rows (pass 1) and SNR rows
        // (pass 2) its own rows need, in per-thread scratch buffers that stay
        // in cache -- instead of two full-frame f64 arrays (h*w*8 bytes each,
        // written once and read back from DRAM). A band recomputes a few
        // pass-1 rows its neighbour also computes; every value is produced by
        // the same per-row code with the same tap order, so the SNR map, and
        // hence the candidate list (in row-major order), is unchanged. Rows
        // outside the candidate region plus the local-max halo are skipped:
        // nothing reads them.
        let y_lo = border;
        let y_hi = h.saturating_sub(border);
        let n_bands = if y_hi > y_lo { (y_hi - y_lo + DETECT_BAND - 1) / DETECT_BAND } else { 0 };
        let pass1_row = |y: usize, out_row: &mut [f64], buf: &mut Vec<f64>| {
            let (gy0, gy1, fy) = mesh_axis(y, ny, cellf);
            let b0 = &bg_rows[gy0 * w..(gy0 + 1) * w];
            let b1 = &bg_rows[gy1 * w..(gy1 + 1) * w];
            let srow = &src[y * w..(y + 1) * w];
            buf.resize(w + 2 * half, 0.0);
            for x in 0..w {
                buf[half + x] = srow[x] as f64 - (b0[x] * (1.0 - fy) + b1[x] * fy);
            }
            for k in 0..half {
                buf[half - 1 - k] = buf[half + reflect_idx(-(k as isize) - 1, w)];
                buf[half + w + k] = buf[half + reflect_idx((w + k) as isize, w)];
            }
            let bref: &[f64] = &buf[..];
            tap_accumulate(out_row, &k1, |t| &bref[t..t + w]);
        };
        let band_candidates: Vec<Vec<(usize, usize, f64)>> = (0..n_bands)
            .into_par_iter()
            .map_init(
                || (Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()),
                |(tmp, snr, buf), bi| {
                    let c0 = y_lo + bi * DETECT_BAND;
                    let c1 = (c0 + DETECT_BAND).min(y_hi);
                    // SNR rows the local-max test reads, and the pass-1 rows
                    // those need (reflect_idx keeps every index in [t0, t1)).
                    let s0 = c0.saturating_sub(fh);
                    let s1 = (c1 + fh).min(h);
                    let t0 = s0.saturating_sub(half);
                    let t1 = (s1 + half).min(h);

                    // pass 1 (along rows) on the residual
                    tmp.resize((t1 - t0) * w, 0.0);
                    for (i, out_row) in tmp.chunks_mut(w).enumerate() {
                        pass1_row(t0 + i, out_row, buf);
                    }

                    // pass 2 (along columns) and the SNR
                    snr.resize((s1 - s0) * w, 0.0);
                    let tmp_ref: &[f64] = tmp;
                    for (i, out_row) in snr.chunks_mut(w).enumerate() {
                        let y = s0 + i;
                        tap_accumulate(out_row, &k1, |t| {
                            let yi = reflect_idx(y as isize + t as isize - half as isize, h) - t0;
                            &tmp_ref[yi * w..(yi + 1) * w]
                        });
                        let (gy0, gy1, fy) = mesh_axis(y, ny, cellf);
                        let s0r = &sg_rows[gy0 * w..(gy0 + 1) * w];
                        let s1r = &sg_rows[gy1 * w..(gy1 + 1) * w];
                        for x in 0..w {
                            let sig = ((s0r[x] * (1.0 - fy) + s1r[x] * fy) * kernel_norm).max(1e-9);
                            out_row[x] /= sig;
                        }
                    }

                    let mut out = Vec::new();
                    for y in c0..c1 {
                        for x in border..w.saturating_sub(border) {
                            let v = snr[(y - s0) * w + x];
                            if v <= k_confirm {
                                continue;
                            }
                            let mut is_max = true;
                            'outer: for dy in -fhalf..=fhalf {
                                let yi = y as isize + dy;
                                if yi < 0 || yi >= h as isize {
                                    continue;
                                }
                                let row_base = (yi as usize - s0) * w;
                                for dx in -fhalf..=fhalf {
                                    let xi = x as isize + dx;
                                    if xi < 0 || xi >= w as isize {
                                        continue;
                                    }
                                    if snr[row_base + xi as usize] > v {
                                        is_max = false;
                                        break 'outer;
                                    }
                                }
                            }
                            if is_max {
                                out.push((y, x, v));
                            }
                        }
                    }
                    out
                },
            )
            .collect();
        let candidates: Vec<(usize, usize, f64)> = band_candidates.into_iter().flatten().collect();

        // Per-candidate measurement: local background, two-pass centroid,
        // second moments (shape/roundness). Embarrassingly parallel.
        let r = ((1.5 * fwhm).round() as isize).max(3);
        let rr = ((0.7 * fwhm).round() as isize).max(2);

        candidates
            .into_par_iter()
            .filter_map(|(py_, px_, snr_v)| {
                let y0 = (py_ as isize - r).max(0) as usize;
                let y1 = ((py_ as isize + r + 1).max(0) as usize).min(h);
                let x0 = (px_ as isize - r).max(0) as usize;
                let x1 = ((px_ as isize + r + 1).max(0) as usize).min(w);
                let local_bg = bg_at(py_, px_);

                let mut wsum = 0f64;
                let mut cy = 0f64;
                let mut cx = 0f64;
                let mut n_positive = 0usize;
                for y in y0..y1 {
                    let row_base = y * w;
                    for x in x0..x1 {
                        let wv = (lum(row_base + x) - local_bg).max(0.0);
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
                        let wv = (lum(row_base + x) - local_bg).max(0.0);
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
                        let wv = (lum(row_base + x) - local_bg).max(0.0);
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
                        let v = lum(row_base + x);
                        if v > peak {
                            peak = v;
                        }
                        flux += (v - local_bg).max(0.0);
                    }
                }
                let sharpness = (snr_v / 20.0).clamp(0.0, 1.0);

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
pub(crate) fn stamp_star_disks<'py>(
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
