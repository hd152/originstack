//! 3x3 convolution kernel for the numpy-forward-pass U-Net (`src/nn_denoise.py`).

use crate::*;

// ── Learned denoiser: 3x3 convolution (src/nn_denoise.py, --denoiser nn) ─────────────────────

/// Output channels per register block (each loaded input vector is reused this many times).
/// On an idle machine every shape from 4x16 to 16x8 ran 18-22 GMAC/s per core (no FMA);
/// 8x8 keeps 8 accumulators with no register pressure (dev-notes/native-kernels.md).
pub(crate) const NN_OB: usize = 8;

/// Pixels per register block (one AVX2 vector): NN_OB x 1 accumulators stay in registers.
pub(crate) const NN_XV: usize = 8;

/// Shapes shared by the two row kernels: input `xp` is (c, hp, wp), zero-padded by one pixel
/// and on the right to a whole number of NN_XV blocks plus 2; weights `wt` are tap-major
/// (c, 9, o_pad), one block's weights for a tap contiguous; `b` is padded to o_pad.
pub(crate) struct NnConvGeom {
    c: usize,
    hp: usize,
    wp: usize,
    o_n: usize,
    o_pad: usize,
    wd: usize,
}

/// Output row `y`, all output channels, into `out_row` (O, wd). Per output value: bias, then
/// `acc = acc + w * x` over the taps in (channel, ky, kx) order -- the same f32 operations in
/// the same order as `nn_conv_row_avx2`, so both paths give identical results. LeakyReLU is
/// applied by the caller.
pub(crate) fn nn_conv_row_portable(xp: &[f32], wt: &[f32], b: &[f32], g: &NnConvGeom, y: usize, out_row: &mut [f32]) {
    let mut x0 = 0;
    while x0 < g.wd {
        let n = (g.wd - x0).min(NN_XV);
        let mut o0 = 0;
        while o0 < g.o_n {
            let mut acc = [[0f32; NN_XV]; NN_OB];
            for k in 0..NN_OB {
                acc[k] = [b[o0 + k]; NN_XV];
            }
            for ci in 0..g.c {
                for ky in 0..3 {
                    let base = (ci * g.hp + y + ky) * g.wp + x0;
                    for kx in 0..3 {
                        let s = &xp[base + kx..base + kx + NN_XV];
                        let wo = (ci * 9 + ky * 3 + kx) * g.o_pad + o0;
                        for k in 0..NN_OB {
                            let wv = wt[wo + k];
                            for l in 0..NN_XV {
                                acc[k][l] += wv * s[l];
                            }
                        }
                    }
                }
            }
            for k in 0..NN_OB.min(g.o_n - o0) {
                out_row[(o0 + k) * g.wd + x0..(o0 + k) * g.wd + x0 + n].copy_from_slice(&acc[k][..n]);
            }
            o0 += NN_OB;
        }
        x0 += NN_XV;
    }
}

/// AVX2 twin of `nn_conv_row_portable` written with intrinsics: left to auto-vectorise, LLVM
/// merged the three overlapping kx loads into one and rebuilt the shifted vectors with
/// shuffles and blends, spilling to the stack. Mul then add, no FMA. Only called after a
/// run-time AVX2 check; `nn_conv3x3` sizes every buffer so the unchecked loads stay inside.
///
/// # Safety
/// The CPU must support AVX2, and `xp`, `wt`, `b` and `out_row` must be sized from `g` as
/// `nn_conv3x3` does (the loads are unchecked).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nn_conv_row_avx2(xp: &[f32], wt: &[f32], b: &[f32], g: &NnConvGeom, y: usize, out_row: &mut [f32]) {
    use std::arch::x86_64::*;
    debug_assert!(xp.len() >= g.c * g.hp * g.wp && wt.len() >= g.c * 9 * g.o_pad && y + 2 < g.hp);
    let (xptr, wptr) = (xp.as_ptr(), wt.as_ptr());
    let mut x0 = 0;
    while x0 < g.wd {
        let n = (g.wd - x0).min(NN_XV);
        let mut o0 = 0;
        while o0 < g.o_n {
            let mut a = [_mm256_setzero_ps(); NN_OB];
            for k in 0..NN_OB {
                a[k] = _mm256_set1_ps(b[o0 + k]);
            }
            for ci in 0..g.c {
                for ky in 0..3 {
                    let row = xptr.add((ci * g.hp + y + ky) * g.wp + x0);
                    for kx in 0..3 {
                        let s = _mm256_loadu_ps(row.add(kx));
                        let wq = wptr.add((ci * 9 + ky * 3 + kx) * g.o_pad + o0);
                        for k in 0..NN_OB {
                            a[k] = _mm256_add_ps(a[k], _mm256_mul_ps(_mm256_set1_ps(*wq.add(k)), s));
                        }
                    }
                }
            }
            let mut tmp = [0f32; NN_XV];
            for k in 0..NN_OB.min(g.o_n - o0) {
                _mm256_storeu_ps(tmp.as_mut_ptr(), a[k]);
                out_row[(o0 + k) * g.wd + x0..(o0 + k) * g.wd + x0 + n].copy_from_slice(&tmp[..n]);
            }
            o0 += NN_OB;
        }
        x0 += NN_XV;
    }
}

/// 3x3 'same' convolution with zero padding (as torch `Conv2d(padding=1)`), plus bias and an
/// optional LeakyReLU (`max(v, leak * v)`, numpy's `np.maximum(out, leak * out)`), for the
/// learned denoiser's U-Net (`src/nn_denoise.py::_conv3x3`). x: (C, H, W); w: (O, C, 3, 3);
/// b: (O,). Returns (O, H, W); each rayon task computes one output row of every channel and
/// copies it into place. Register-blocked: 8 output channels x 8 pixels of
/// accumulators stay in registers across all 9*C taps, each input vector loaded once per 8
/// outputs. Not bit-identical to the numpy fallback, whose BLAS product sums the taps in its
/// own order: tested with a tolerance.
#[pyfunction]
pub(crate) fn nn_conv3x3<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f32>,
    w: PyReadonlyArray4<'py, f32>,
    b: PyReadonlyArray1<'py, f32>,
    leak: f32,
    act: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let xs = x.as_array().shape().to_vec();
    let ws = w.as_array().shape().to_vec();
    let (c, h, wd) = (xs[0], xs[1], xs[2]);
    let o_n = ws[0];
    if ws[1] != c || ws[2] != 3 || ws[3] != 3 || b.as_array().len() != o_n || h == 0 || wd == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
    }
    let xv = x.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("x must be contiguous"))?;
    let wv = w.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("w must be contiguous"))?;
    let bv = b.as_slice()?;
    let g = NnConvGeom {
        c,
        hp: h + 2,
        wp: wd.div_ceil(NN_XV) * NN_XV + 2,
        o_n,
        o_pad: o_n.div_ceil(NN_OB) * NN_OB,
        wd,
    };
    let out = py.detach(|| {
        let mut bp = vec![0f32; g.o_pad];
        bp[..o_n].copy_from_slice(bv);
        let mut wt = vec![0f32; c * 9 * g.o_pad];
        for o in 0..o_n {
            for ci in 0..c {
                for t in 0..9 {
                    wt[(ci * 9 + t) * g.o_pad + o] = wv[(o * c + ci) * 9 + t];
                }
            }
        }
        let mut xp = vec![0f32; c * g.hp * g.wp];
        xp.par_chunks_mut(g.hp * g.wp).zip(xv.par_chunks(h * wd)).for_each(|(dst, src)| {
            for y in 0..h {
                dst[(y + 1) * g.wp + 1..(y + 1) * g.wp + 1 + wd].copy_from_slice(&src[y * wd..(y + 1) * wd]);
            }
        });
        let mut out = vec![0f32; o_n * h * wd];
        let optr = HpPtr(out.as_mut_ptr());
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::is_x86_feature_detected!("avx2");
        (0..h).into_par_iter().for_each_init(|| vec![0f32; o_n * wd], |row, y| {
            let row = &mut row[..];
            #[cfg(target_arch = "x86_64")]
            let done = if avx2 {
                // SAFETY: the CPU reports AVX2; xp, wt and bp are sized from g above.
                unsafe { nn_conv_row_avx2(&xp, &wt, &bp, &g, y, row) };
                true
            } else {
                false
            };
            #[cfg(not(target_arch = "x86_64"))]
            let done = false;
            if !done {
                nn_conv_row_portable(&xp, &wt, &bp, &g, y, row);
            }
            if act {
                for v in row.iter_mut() {
                    *v = v.max(leak * *v);
                }
            }
            let optr = optr;
            for o in 0..o_n {
                // SAFETY: row y of plane o; each task owns a distinct y, all inside `out`.
                unsafe {
                    std::ptr::copy_nonoverlapping(row.as_ptr().add(o * wd), optr.0.add((o * h + y) * wd), wd);
                }
            }
        });
        out
    });
    let arr = numpy::ndarray::Array3::from_shape_vec((o_n, h, wd), out).expect("shape mismatch building output");
    Ok(arr.into_pyarray(py))
}

/// The portable row kernel, exposed so tests can check the AVX2 twin against it
/// (same arguments and result as `nn_conv3x3`).
#[pyfunction]
pub(crate) fn nn_conv3x3_portable<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f32>,
    w: PyReadonlyArray4<'py, f32>,
    b: PyReadonlyArray1<'py, f32>,
    leak: f32,
    act: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let xs = x.as_array().shape().to_vec();
    let ws = w.as_array().shape().to_vec();
    let (c, h, wd) = (xs[0], xs[1], xs[2]);
    let o_n = ws[0];
    if ws[1] != c || ws[2] != 3 || ws[3] != 3 || b.as_array().len() != o_n || h == 0 || wd == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
    }
    let xv = x.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("x must be contiguous"))?;
    let wv = w.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("w must be contiguous"))?;
    let g = NnConvGeom {
        c,
        hp: h + 2,
        wp: wd.div_ceil(NN_XV) * NN_XV + 2,
        o_n,
        o_pad: o_n.div_ceil(NN_OB) * NN_OB,
        wd,
    };
    let mut bp = vec![0f32; g.o_pad];
    bp[..o_n].copy_from_slice(b.as_slice()?);
    let mut wt = vec![0f32; c * 9 * g.o_pad];
    for o in 0..o_n {
        for ci in 0..c {
            for t in 0..9 {
                wt[(ci * 9 + t) * g.o_pad + o] = wv[(o * c + ci) * 9 + t];
            }
        }
    }
    let mut xp = vec![0f32; c * g.hp * g.wp];
    for ci in 0..c {
        for y in 0..h {
            let d = (ci * g.hp + y + 1) * g.wp + 1;
            xp[d..d + wd].copy_from_slice(&xv[(ci * h + y) * wd..(ci * h + y + 1) * wd]);
        }
    }
    let mut out = vec![0f32; o_n * h * wd];
    let mut row = vec![0f32; o_n * wd];
    for y in 0..h {
        nn_conv_row_portable(&xp, &wt, &bp, &g, y, &mut row);
        for o in 0..o_n {
            for x in 0..wd {
                let v = row[o * wd + x];
                out[(o * h + y) * wd + x] = if act { v.max(leak * v) } else { v };
            }
        }
    }
    let arr = numpy::ndarray::Array3::from_shape_vec((o_n, h, wd), out).expect("shape mismatch building output");
    Ok(arr.into_pyarray(py))
}
