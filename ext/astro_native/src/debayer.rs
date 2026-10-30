//! Malvar-He-Cutler and Menon 2007 demosaicing.

use crate::*;

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

pub(crate) const MALVAR_G_AT_RB: [(isize, isize, f64); 9] = [
    (-2, 0, -1.0 / 8.0), (-1, 0, 2.0 / 8.0),
    (0, -2, -1.0 / 8.0), (0, -1, 2.0 / 8.0), (0, 0, 4.0 / 8.0), (0, 1, 2.0 / 8.0), (0, 2, -1.0 / 8.0),
    (1, 0, 2.0 / 8.0), (2, 0, -1.0 / 8.0),
];

// R at green in an R row / B column (and B at green in a B row / R column).
pub(crate) const MALVAR_RG_RB_BG_BR: [(isize, isize, f64); 11] = [
    (-2, 0, 0.5 / 8.0),
    (-1, -1, -1.0 / 8.0), (-1, 1, -1.0 / 8.0),
    (0, -2, -1.0 / 8.0), (0, -1, 4.0 / 8.0), (0, 0, 5.0 / 8.0), (0, 1, 4.0 / 8.0), (0, 2, -1.0 / 8.0),
    (1, -1, -1.0 / 8.0), (1, 1, -1.0 / 8.0),
    (2, 0, 0.5 / 8.0),
];

// R at green in a B row / R column (and B at green in an R row / B column) --
// transpose of the kernel above.
pub(crate) const MALVAR_RG_BR_BG_RB: [(isize, isize, f64); 11] = [
    (0, -2, 0.5 / 8.0),
    (-1, -1, -1.0 / 8.0), (1, -1, -1.0 / 8.0),
    (-2, 0, -1.0 / 8.0), (-1, 0, 4.0 / 8.0), (0, 0, 5.0 / 8.0), (1, 0, 4.0 / 8.0), (2, 0, -1.0 / 8.0),
    (-1, 1, -1.0 / 8.0), (1, 1, -1.0 / 8.0),
    (0, 2, 0.5 / 8.0),
];

// R at B (and B at R).
pub(crate) const MALVAR_R_AT_B: [(isize, isize, f64); 9] = [
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
pub(crate) fn mirror_idx(i: isize, n: usize) -> usize {
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
pub(crate) fn malvar_tap(data: &[f32], h: usize, w: usize, y: usize, x: usize, taps: &[(isize, isize, f64)]) -> f32 {
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
pub(crate) fn malvar_tap_interior(data: &[f32], w: usize, base: usize, taps: &[(isize, isize, f64)]) -> f32 {
    let mut acc = 0.0f64;
    for &(dy, dx, wt) in taps {
        let off = dy * w as isize + dx;
        acc += data[(base as isize + off) as usize] as f64 * wt;
    }
    acc as f32
}

pub(crate) fn malvar_pattern_offsets(pattern: &str) -> PyResult<(usize, usize, usize, usize)> {
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
pub(crate) fn debayer_malvar<'py>(
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
        par_rows(&mut out, w * 3, |y, row_out| {
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

/// Malvar-He-Cutler demosaicing directly into a caller-owned frame-store slot.
/// This mirrors `debayer_malvar` exactly but avoids a full RGB allocation and
/// copy when Phase 1 already has an output buffer.
#[pyfunction]
pub(crate) fn debayer_malvar_into<'py>(
    py: Python<'py>,
    raw: numpy::PyReadonlyArray2<'py, f32>,
    mut out: numpy::PyReadwriteArray3<'py, f32>,
    pattern: &str,
) -> PyResult<()> {
    let (r_r, r_c, b_r, b_c) = malvar_pattern_offsets(pattern)?;
    let arr = raw.as_array();
    let s = arr.shape();
    let (h, w) = (s[0], s[1]);
    if out.as_array().shape() != [h, w, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err("out must be (h, w, 3)"));
    }
    let owned: Vec<f32>;
    let flat: &[f32] = match arr.as_slice() {
        Some(sl) => sl,
        None => {
            owned = arr.iter().copied().collect();
            &owned
        }
    };
    let dst = out.as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("out must be contiguous"))?;
    py.detach(|| {
        par_rows(&mut *dst, w * 3, |y, row_out| {
            let is_r_row = y % 2 == r_r;
            let is_b_row = y % 2 == b_r;
            let interior_y = y >= 2 && y + 2 < h;
            for x in 0..w {
                let is_r_col = x % 2 == r_c;
                let is_b_col = x % 2 == b_c;
                let raw_value = flat[y * w + x];
                let (r, g, b) = if interior_y && x >= 2 && x + 2 < w {
                    let base = y * w + x;
                    if is_r_row && is_r_col {
                        (raw_value, malvar_tap_interior(flat, w, base, &MALVAR_G_AT_RB),
                         malvar_tap_interior(flat, w, base, &MALVAR_R_AT_B))
                    } else if is_b_row && is_b_col {
                        (malvar_tap_interior(flat, w, base, &MALVAR_R_AT_B),
                         malvar_tap_interior(flat, w, base, &MALVAR_G_AT_RB), raw_value)
                    } else if is_r_row && is_b_col {
                        (malvar_tap_interior(flat, w, base, &MALVAR_RG_RB_BG_BR), raw_value,
                         malvar_tap_interior(flat, w, base, &MALVAR_RG_BR_BG_RB))
                    } else {
                        (malvar_tap_interior(flat, w, base, &MALVAR_RG_BR_BG_RB), raw_value,
                         malvar_tap_interior(flat, w, base, &MALVAR_RG_RB_BG_BR))
                    }
                } else if is_r_row && is_r_col {
                    (raw_value, malvar_tap(flat, h, w, y, x, &MALVAR_G_AT_RB),
                     malvar_tap(flat, h, w, y, x, &MALVAR_R_AT_B))
                } else if is_b_row && is_b_col {
                    (malvar_tap(flat, h, w, y, x, &MALVAR_R_AT_B),
                     malvar_tap(flat, h, w, y, x, &MALVAR_G_AT_RB), raw_value)
                } else if is_r_row && is_b_col {
                    (malvar_tap(flat, h, w, y, x, &MALVAR_RG_RB_BG_BR), raw_value,
                     malvar_tap(flat, h, w, y, x, &MALVAR_RG_BR_BG_RB))
                } else {
                    (malvar_tap(flat, h, w, y, x, &MALVAR_RG_BR_BG_RB), raw_value,
                     malvar_tap(flat, h, w, y, x, &MALVAR_RG_RB_BG_BR))
                };
                row_out[x * 3] = r;
                row_out[x * 3 + 1] = g;
                row_out[x * 3 + 2] = b;
            }
        });
    });
    Ok(())
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
pub(crate) const MENON_G_TAPS: [(isize, f64); 5] =
    [(-2, -0.25), (-1, 0.5), (0, 0.5), (1, 0.5), (2, -0.25)];

/// R/B <-> G colour-difference averaging filter ([0.5, 0, 0.5]; the centre
/// tap is zero so it's omitted).
pub(crate) const MENON_KB_TAPS: [(isize, f64); 2] = [(-1, 0.5), (1, 0.5)];

/// Refining-step 3-tap box filter (FIR = [1/3, 1/3, 1/3]).
pub(crate) const MENON_FIR_TAPS: [(isize, f64); 3] =
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
pub(crate) const MENON_DIFF_K: [(isize, isize, f64); 8] = [
    (2, 0, 1.0), (2, -2, 1.0),
    (1, -1, 1.0),
    (0, 0, 3.0), (0, -2, 3.0),
    (-1, -1, 1.0),
    (-2, 0, 1.0), (-2, -2, 1.0),
];

/// Transpose of MENON_DIFF_K (vertical direction) -- each tap's (dy, dx) swapped.
pub(crate) const MENON_DIFF_KT: [(isize, isize, f64); 8] = [
    (0, 2, 1.0), (-2, 2, 1.0),
    (-1, 1, 1.0),
    (0, 0, 3.0), (-2, 0, 3.0),
    (-1, -1, 1.0),
    (0, -2, 1.0), (-2, -2, 1.0),
];

/// 1D horizontal convolution, scipy `mode='mirror'` boundary (reuses `mirror_idx`).
pub(crate) fn menon_conv_h(data: &[f64], h: usize, w: usize, taps: &[(isize, f64)]) -> Vec<f64> {
    let mut out = vec![0.0f64; h * w];
    par_rows(&mut out, w, |y, row_out| {
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
pub(crate) fn menon_conv_v(data: &[f64], h: usize, w: usize, taps: &[(isize, f64)]) -> Vec<f64> {
    let mut out = vec![0.0f64; h * w];
    par_rows(&mut out, w, |y, row_out| {
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
pub(crate) fn menon_diffuse(data: &[f64], h: usize, w: usize, taps: &[(isize, isize, f64)]) -> Vec<f64> {
    let mut out = vec![0.0f64; h * w];
    par_rows(&mut out, w, |y, row_out| {
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
pub(crate) fn debayer_menon2007<'py>(
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
        par_rows(&mut d_h, w, |y, row| {
            let base = y * w;
            for x in 0..w {
                row[x] = (c_h[base + x] - c_h[base + mirror_idx(x as isize + 2, w)]).abs();
            }
        });
        let mut d_v = vec![0.0f64; n];
        par_rows(&mut d_v, w, |y, row| {
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
    par_rows(&mut out, 3, |i, px| {
        px[0] = r_final[i] as f32;
        px[1] = g_final[i] as f32;
        px[2] = b_final[i] as f32;
    });

    let arr3 = numpy::ndarray::Array3::from_shape_vec((h, w, 3), out)
        .expect("shape mismatch building debayer_menon2007 output");
    Ok(arr3.into_pyarray(py))
}
