//! Registration kernels: blind match, rigid RANSAC, phase-correlation window, pyramid downsample.

use crate::*;

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
pub(crate) fn blind_match_hypotheses<'py>(
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
pub(crate) struct SplitMix64 {
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

pub(crate) fn entropy_seed() -> u64 {
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
pub(crate) fn svd_2x2(a: [[f64; 2]; 2]) -> ([[f64; 2]; 2], [f64; 2], [[f64; 2]; 2]) {
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

pub(crate) fn mat2_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

/// Umeyama (1991) 2D rigid (no scale) least-squares fit -- exact port of
/// skimage.transform._geometric._umeyama(src, dst, estimate_scale=False).
/// Returns a 3x3 homogeneous matrix, or None if degenerate (rank 0).
pub(crate) fn umeyama_2d(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<[[f64; 3]; 3]> {
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

pub(crate) fn dynamic_max_trials(n_inliers: usize, n_samples: usize, min_samples: usize) -> f64 {
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
pub(crate) fn fit_rigid_ransac<'py>(
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

/// 2x box downsample of an (H, W) float32 image, bit-identical to
/// `registration._downsample_half`: even-cropped, out = (((a00 + a10) + a01) + a11) * 0.25
/// in f32 -- numpy's operation order. One read pass instead of numpy's three
/// strided adds and temporaries. Single-threaded: callers run it from many
/// Python threads at once (the pyramid pass), so rayon here would only contend.
#[pyfunction]
pub(crate) fn downsample_half_f32<'py>(py: Python<'py>, img: PyReadonlyArray2<'py, f32>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let a = img.as_array();
    let (h, w) = (a.shape()[0], a.shape()[1]);
    let src = a.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("image must be C-contiguous"))?;
    let (h2, w2) = (h / 2, w / 2);
    let out: Vec<f32> = py.detach(|| {
        let mut o = vec![0f32; h2 * w2];
        for y in 0..h2 {
            let r0 = &src[2 * y * w..2 * y * w + w];
            let r1 = &src[(2 * y + 1) * w..(2 * y + 1) * w + w];
            let orow = &mut o[y * w2..(y + 1) * w2];
            for x in 0..w2 {
                orow[x] = (((r0[2 * x] + r1[2 * x]) + r0[2 * x + 1]) + r1[2 * x + 1]) * 0.25f32;
            }
        }
        o
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((h2, w2), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_pyarray(py))
}

/// Linear cross-correlation `c[dy][dx] = sum_p ref[p + d] * img[p]` (zero outside
/// the frame) for every lag |dy|, |dx| <= radius: the values a zero-padded FFT
/// correlation has at those lags (src/registration.py::calculate_shift_pyramid_pref,
/// below the coarsest pyramid level, where the coarse pass has already found the
/// shift to a few pixels and a full padded FFT -- 2x each axis -- is ~100 ms of
/// work for 25 numbers). f64 accumulation, parallel over lags.
#[pyfunction]
#[pyo3(signature = (reference, image, radius))]
pub(crate) fn xcorr_window<'py>(
    py: Python<'py>,
    reference: PyReadonlyArray2<'py, f32>,
    image: PyReadonlyArray2<'py, f32>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ra = reference.as_array();
    let ia = image.as_array();
    if ra.shape() != ia.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
    }
    let (h, w) = (ra.shape()[0], ra.shape()[1]);
    let rs = ra.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("reference must be contiguous"))?;
    let is = ia.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("image must be contiguous"))?;
    if radius >= h.max(w) {
        return Err(pyo3::exceptions::PyValueError::new_err("radius must be smaller than the frame"));
    }
    let n = 2 * radius + 1;
    let r = radius as isize;
    let vals: Vec<f64> = py.detach(|| {
        (0..n * n)
            .into_par_iter()
            .map(|k| {
                let dy = (k / n) as isize - r;
                let dx = (k % n) as isize - r;
                // p and p + d both inside the frame
                let y0 = (-dy).max(0) as usize;
                let y1 = ((h as isize) - dy.max(0)).max(0) as usize;
                let x0 = (-dx).max(0) as usize;
                let x1 = ((w as isize) - dx.max(0)).max(0) as usize;
                let mut acc = 0f64;
                if x1 > x0 {
                    for y in y0..y1 {
                        let ry = (y as isize + dy) as usize;
                        let a = &rs[ry * w + (x0 as isize + dx) as usize..ry * w + (x1 as isize + dx) as usize];
                        let b = &is[y * w + x0..y * w + x1];
                        let mut row = 0f64;
                        for (u, v) in a.iter().zip(b) {
                            row += *u as f64 * *v as f64;
                        }
                        acc += row;
                    }
                }
                acc
            })
            .collect()
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((n, n), vals).unwrap().into_pyarray(py))
}
