//! Resampling: Lanczos-3 warp, ndimage order-1 affine/shift, kernel tables.

use crate::*;

/// Compute the 6 normalised Lanczos-3 tap weights for fractional offset `r`
/// (taps at floor-2 .. floor+3). Same arithmetic as the original inline loop.
#[inline]
pub(crate) fn lanczos6_weights(r: f64, out: &mut [f64; 6]) {
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
// sum to 1).
pub(crate) const LUT_N: usize = 512;

pub(crate) fn lanczos_lut() -> &'static Vec<[f64; 6]> {
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

#[inline]
pub(crate) fn lanczos6_weights_lut(r: f64, out: &mut [f64; 6]) {
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

/// One output row of the contiguous-input Lanczos-3 warp (the body of
/// `warp_affine_lanczos3`'s fast path, shared with `drizzle_accumulate_lanczos3`
/// so the two are the same arithmetic by construction).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn lanczos3_row_flat(
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
                                    lanczos6_weights_lut(iy - fy, &mut wy);
                                }
                                let iy = m00 * oy as f64 + o0;
                                (&wy, &wxs[ox], iy.floor() as isize - 2, bxs[ox])
                            } else {
                                let oxg = (ox + ox0) as f64;
                                let iy = m00 * oy as f64 + m01 * oxg + o0;
                                let ix = m10 * oy as f64 + m11 * oxg + o1;
                                let fy = iy.floor();
                                let fx = ix.floor();
                                lanczos6_weights_lut(iy - fy, &mut wy);
                                lanczos6_weights_lut(ix - fx, &mut wx);
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

// ---------------------------------------------------------------------------
// scipy.ndimage order-1 warps, mode='constant' -- bit-identical ports of
// NI_GeometricTransform (affine_transform) and NI_ZoomShift (shift) in
// scipy/ndimage/src/ni_interpolation.c (scipy 1.17), order 1, nprepad 0, 2-D.
// Used by registration._match_frame_stars (the post-registration residual check).
// ---------------------------------------------------------------------------

/// `map_coordinate(in, len, NI_EXTEND_MIRROR)` for an index past the last one (the
/// only side a linear tap can leave once its start is inside the frame).
#[inline]
pub(crate) fn ni_mirror_hi(idx: f64, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let sz2 = 2 * len as i64 - 2;
    let mut v = idx;
    v -= (sz2 * ((v / sz2 as f64) as i64)) as f64;
    if v >= len as f64 {
        v = sz2 as f64 - v;
    }
    v as usize
}

/// One axis of an order-1 sample at input coordinate `c` (mode='constant'): None when
/// scipy takes `cval` (`map_coordinate` sends c < 0 or c > len - 1 to -1, and NaN fails
/// `cc > -1.0`), else the two tap indices -- the second mirrored if it falls past the
/// end, where its weight is 0 -- and `get_spline_interpolation_weights(c, 1)`.
#[inline(always)]
pub(crate) fn ni_axis1(c: f64, len: usize) -> Option<(usize, usize, f64, f64)> {
    let mut cc = c;
    if cc < 0.0 || cc > (len as i64 - 1) as f64 {
        cc = -1.0;
    }
    if !(cc > -1.0) {
        return None;
    }
    let fl = cc.floor();
    let start = fl as i64 as usize;
    let x = cc - fl;
    let w0 = 1.0 - x;
    let w1 = 1.0 - w0;
    let i1 = if start + 1 >= len { ni_mirror_hi((start + 1) as f64, len) } else { start + 1 };
    Some((start, i1, w0, w1))
}

/// `t = 0; t += (v * wy) * wx` over the four taps in scipy's filter order. When the
/// sum is NaN it is redone with x86's NaN rule made explicit -- `t + coeff` returns
/// `t`'s NaN when `t` is NaN, else `coeff`'s -- because two different NaNs can meet
/// there (an input NaN and the default NaN of `inf * 0` from a weight-0 tap) and LLVM
/// may swap the operands of a commutative add, which would pick the other one. Each
/// coeff has at most one NaN operand (the weights are finite), so it needs no such care.
#[inline(always)]
pub(crate) fn ni_sample1(src: &[f32], w: usize, ay: (usize, usize, f64, f64), ax: (usize, usize, f64, f64)) -> f64 {
    let (y0, y1, wy0, wy1) = ay;
    let (x0, x1, wx0, wx1) = ax;
    let c = [
        (src[y0 * w + x0] as f64 * wy0) * wx0,
        (src[y0 * w + x1] as f64 * wy0) * wx1,
        (src[y1 * w + x0] as f64 * wy1) * wx0,
        (src[y1 * w + x1] as f64 * wy1) * wx1,
    ];
    let mut t = 0.0f64;
    t += c[0];
    t += c[1];
    t += c[2];
    t += c[3];
    if t.is_nan() {
        t = 0.0;
        for &k in c.iter() {
            t = if t.is_nan() {
                t
            } else if k.is_nan() {
                k
            } else {
                t + k
            };
        }
    }
    t
}

/// `scipy.ndimage.affine_transform(img.astype(np.float64), matrix, offset=offset,
/// order=1, mode='constant', cval=cval)` for a 2-D float32 `img`, bit for bit (float64
/// output), without the float64 copy of the input (each tap's float32 -> float64
/// conversion is exact, as the copy's). Input coordinate per output pixel (oy, ox):
/// `(offset[i] + oy * m[i][0]) + ox * m[i][1]`, as scipy accumulates it.
#[pyfunction]
#[pyo3(signature = (img, matrix, offset, cval=0.0))]
pub(crate) fn ndimage_affine_order1<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, f32>,
    matrix: PyReadonlyArray2<'py, f64>,
    offset: PyReadonlyArray1<'py, f64>,
    cval: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = img.as_array();
    let (h, w) = (a.shape()[0], a.shape()[1]);
    let m = matrix.as_array();
    let o = offset.as_array();
    if m.shape() != [2, 2] || o.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("matrix must be (2, 2) and offset (2,)"));
    }
    let (m00, m01, m10, m11) = (m[[0, 0]], m[[0, 1]], m[[1, 0]], m[[1, 1]]);
    let (o0, o1) = (o[0], o[1]);
    let owned: Vec<f32>;
    let src: &[f32] = match a.as_slice() {
        Some(x) => x,
        None => {
            owned = a.iter().copied().collect();
            &owned
        }
    };
    let mut out = vec![0f64; h * w];
    py.detach(|| {
        out.par_chunks_mut(w.max(1)).enumerate().for_each(|(oy, orow)| {
            let fy = oy as f64;
            let (r0, r1) = (o0 + fy * m00, o1 + fy * m10);
            for (ox, v) in orow.iter_mut().enumerate() {
                let fx = ox as f64;
                let cy = r0 + fx * m01;
                let cx = r1 + fx * m11;
                *v = match ni_axis1(cy, h) {
                    None => cval,
                    Some(ay) => match ni_axis1(cx, w) {
                        None => cval,
                        Some(ax) => ni_sample1(src, w, ay, ax),
                    },
                };
            }
        });
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// `scipy.ndimage.shift(img, shift, order=1, mode='constant', cval=cval)` for a 2-D
/// float32 `img`, bit for bit (float32 output: the float64 sum rounded once). scipy's
/// zoom_shift is called with the negated shift and maps output index k of each axis to
/// `(double)k + (-shift)`, independently per axis, so the per-axis taps and weights are
/// tabulated once.
#[pyfunction]
#[pyo3(signature = (img, shift, cval=0.0))]
pub(crate) fn ndimage_shift_order1<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, f32>,
    shift: (f64, f64),
    cval: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let a = img.as_array();
    let (h, w) = (a.shape()[0], a.shape()[1]);
    let owned: Vec<f32>;
    let src: &[f32] = match a.as_slice() {
        Some(x) => x,
        None => {
            owned = a.iter().copied().collect();
            &owned
        }
    };
    let (sy, sx) = (-shift.0, -shift.1);
    let ty: Vec<Option<(usize, usize, f64, f64)>> = (0..h).map(|k| ni_axis1(k as f64 + sy, h)).collect();
    let tx: Vec<Option<(usize, usize, f64, f64)>> = (0..w).map(|k| ni_axis1(k as f64 + sx, w)).collect();
    let cv = cval as f32;
    let mut out = vec![0f32; h * w];
    py.detach(|| {
        out.par_chunks_mut(w.max(1)).enumerate().for_each(|(oy, orow)| match ty[oy] {
            None => orow.fill(cv),
            Some(ay) => {
                for (ox, v) in orow.iter_mut().enumerate() {
                    *v = match tx[ox] {
                        None => cv,
                        Some(ax) => ni_sample1(src, w, ay, ax) as f32,
                    };
                }
            }
        });
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// Affine warp with Lanczos-3 resampling, matching scipy's
/// `affine_transform` sampling convention: `out[oy,ox] = in[M @ (oy,ox) + off]`,
/// out-of-bounds -> `cval`. `mat` is row-major 2x2 `[[m00,m01],[m10,m11]]`
/// mapping (row,col); `off` is `[off_row, off_col]`. All channels in one pass,
/// parallel across output rows.
#[pyfunction]
#[pyo3(signature = (data, mat, off, out_h, out_w, cval=0.0, origin=(0, 0)))]
pub(crate) fn warp_affine_lanczos3<'py>(
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

/// `warp_affine_lanczos3` writing into a caller-supplied `(out_h, out_w, c)` array
/// (e.g. one slot of Phase 3's aligned stack) instead of allocating and returning
/// one -- the caller then has no 70 MB copy to make, and no fresh allocation for
/// the OS to zero-fill. Contiguous input only (the path Phase 3 takes); each
/// output row comes from `lanczos3_row_flat`, the same code the allocating
/// kernel runs, so the result is bit-identical to it.
#[pyfunction]
pub(crate) fn warp_affine_lanczos3_into<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f32>,
    mut out: numpy::PyReadwriteArray3<'py, f32>,
    mat: [f64; 4],
    off: [f64; 2],
    cval: f32,
    origin: (usize, usize),
) -> PyResult<()> {
    let (oy0, ox0) = origin;
    let arr = data.as_array();
    let s = arr.shape();
    let (h, w, c) = (s[0], s[1], s[2]);
    let os = out.as_array().shape().to_vec();
    let out_w = os[1];
    if os[2] != c {
        return Err(pyo3::exceptions::PyValueError::new_err("out must have the input's channel count"));
    }
    let img: &[f32] = arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("data must be C-contiguous"))?;
    let dst: &mut [f32] = out
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("out must be C-contiguous"))?;
    let (m00, m01, m10, m11) = (mat[0], mat[1], mat[2], mat[3]);
    let (o0, o1) = (off[0], off[1]);
    let col_tab: Option<(Vec<[f64; 6]>, Vec<isize>)> = if m01 == 0.0 && m10 == 0.0 {
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
    py.detach(|| {
        dst.par_chunks_mut(out_w * c).enumerate().for_each(|(oy_l, out_row)| {
            lanczos3_row_flat(img, h, w, c, oy_l + oy0, out_row, [m00, m01, m10, m11], [o0, o1],
                              &col_tab, cval, ox0);
        });
    });
    Ok(())
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
pub(crate) fn warp_affine_kernel_table<'py>(
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
