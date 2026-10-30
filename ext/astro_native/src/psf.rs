//! Moffat / Gaussian PSF fitting (Levenberg-Marquardt).

use crate::*;

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
pub(crate) fn moffat_eval(r2: f64, amp: f64, alpha: f64, beta: f64) -> (f64, f64, f64, f64) {
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
pub(crate) fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
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
pub(crate) fn fit_moffat_lm(r: &[f64], v: &[f64]) -> Option<(f64, f64, f64)> {
    let n = r.len();
    if n < 6 {
        return None;
    }
    let amp0 = v.iter().cloned().fold(f64::MIN, f64::max);
    if !(amp0 > 0.0) {
        return None;
    }
    let mut rs: Vec<f64> = r.to_vec();
    let alpha0 = median_inplace(&mut rs).max(1.0);

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
pub(crate) fn fit_moffat_native(
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
pub(crate) fn solve_lin<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
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
pub(crate) fn psf_lm<const N: usize, F>(
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
pub(crate) fn moffat2d_eval(x: f64, y: f64, p: &[f64; 6]) -> (f64, [f64; 6]) {
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
pub(crate) fn gauss2d_eval(x: f64, y: f64, p: &[f64; 5]) -> (f64, [f64; 5]) {
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
pub(crate) fn fit_psf_moffat2d_native(
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
pub(crate) fn fit_psf_gauss2d_native(
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
