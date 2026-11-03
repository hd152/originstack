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

// ── Whole U-Net forward pass in one call (src/nn_denoise.py::forward) ────────────────────────

/// Width of the zero padding block: every activation row is `ceil(w / 16) * 16 + 2` floats, so
/// both conv kernels (8 and 16 pixels per step) read whole vectors without a bounds test.
const NN_WALIGN: usize = 16;

/// Output channels per block of the FMA kernel (6 x 2 vectors = 12 accumulators + 2 input
/// vectors + 1 weight broadcast fit the 16 ymm registers; every channel count of the shipped
/// networks, 24 / 48 / 96, divides by 6).
const NN_OB_FMA: usize = 6;

/// A (c, h, w) activation stored zero-padded by one pixel on every side (and to NN_WALIGN on the
/// right), the layout the conv kernels read. Layers write straight into the next layer's input,
/// so the forward pass has no separate padding or copy passes.
struct NnAct {
    c: usize,
    h: usize,
    w: usize,
    hp: usize,
    wp: usize,
    data: Vec<f32>,
}

/// Activation buffers recycled between layers, tiles and calls; emptied by `nn_release_buffers`.
static NN_POOL: std::sync::Mutex<Vec<Vec<f32>>> = std::sync::Mutex::new(Vec::new());

/// Most buffers kept for reuse (a forward pass holds at most ~6 at once, two tiles in flight).
const NN_POOL_MAX: usize = 24;

impl Drop for NnAct {
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.data);
        let mut pool = NN_POOL.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < NN_POOL_MAX {
            pool.push(v);
        }
    }
}

/// Free the activation buffers `nn_forward` keeps for reuse (call when a run of tiles is done).
#[pyfunction]
pub(crate) fn nn_release_buffers() {
    NN_POOL.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

impl NnAct {
    fn new(c: usize, h: usize, w: usize) -> Self {
        let (hp, wp) = (h + 2, w.div_ceil(NN_WALIGN) * NN_WALIGN + 2);
        let n = c * hp * wp;
        // A recycled buffer is reused as it is: much cheaper than faulting in fresh pages
        // (~300 MB per 448 px tile, and page faults serialise on the process address space).
        let recycled = {
            let mut pool = NN_POOL.lock().unwrap_or_else(|e| e.into_inner());
            let best = (0..pool.len()).filter(|&i| pool[i].capacity() >= n && pool[i].capacity() <= n * 2 + 4096)
                .min_by_key(|&i| pool[i].capacity());
            best.map(|i| pool.swap_remove(i))
        };
        let data = match recycled {
            Some(mut v) => {
                // Every writer overwrites the whole interior, so only the padding ring needs
                // zeros: clearing ~100 MB per tile on one thread cost 15% of a forward pass.
                v.truncate(n);
                v.resize(n, 0.0);
                v.par_chunks_mut(hp * wp).for_each(|plane| {
                    plane[..wp].fill(0.0);
                    plane[(hp - 1) * wp..].fill(0.0);
                    for y in 1..=h {
                        plane[y * wp] = 0.0;
                        plane[y * wp + w + 1..(y + 1) * wp].fill(0.0);
                    }
                });
                v
            }
            None => vec![0f32; n],
        };
        NnAct { c, h, w, hp, wp, data }
    }

    /// Offset of pixel (y, x) of channel `ci` (interior coordinates).
    #[inline(always)]
    fn at(&self, ci: usize, y: usize, x: usize) -> usize {
        (ci * self.hp + y + 1) * self.wp + x + 1
    }
}

/// One conv layer's weights reordered for a kernel: tap-major (c, 9, o_pad), bias padded.
struct NnLayer {
    c: usize,
    o_n: usize,
    o_pad: usize,
    wt: Vec<f32>,
    bp: Vec<f32>,
}

fn nn_layer(w: &numpy::ndarray::ArrayView4<f32>, b: &numpy::ndarray::ArrayView1<f32>, ob: usize) -> PyResult<NnLayer> {
    let (o_n, c) = (w.shape()[0], w.shape()[1]);
    if w.shape()[2] != 3 || w.shape()[3] != 3 || b.len() != o_n {
        return Err(pyo3::exceptions::PyValueError::new_err("conv weight shape mismatch"));
    }
    let o_pad = o_n.div_ceil(ob) * ob;
    let mut wt = vec![0f32; c * 9 * o_pad];
    for o in 0..o_n {
        for ci in 0..c {
            for ky in 0..3 {
                for kx in 0..3 {
                    wt[(ci * 9 + ky * 3 + kx) * o_pad + o] = w[[o, ci, ky, kx]];
                }
            }
        }
    }
    let mut bp = vec![0f32; o_pad];
    for o in 0..o_n {
        bp[o] = b[o];
    }
    Ok(NnLayer { c, o_n, o_pad, wt, bp })
}

/// FMA row kernel: output row `y`, all channels, 6 channels x 16 pixels per register block.
/// Same layout contract as `nn_conv_row_avx2` but the taps are accumulated with fused
/// multiply-add, so results differ from the mul-then-add kernels in the last bits (and from a
/// CPU without FMA); see `nn_forward`.
///
/// With `init` (o_n rows of `wd` floats, plus 16 spare) the accumulators start from those values
/// instead of the bias: the output of `nn_up_row_fma`, which already holds bias + the
/// upsampled channels' share.
///
/// # Safety
/// The CPU must support AVX2 and FMA; `xp` (c, hp, wp with wp >= ceil16(wd) + 2), `wt`
/// (c, 9, o_pad) and `b` (o_pad) must be sized from `g` with `o_pad` a multiple of `NN_OB_FMA`;
/// `init`, when given, holds `o_n * wd + 16` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn nn_conv_row_fma(xp: &[f32], wt: &[f32], b: &[f32], g: &NnConvGeom, y: usize, out_row: &mut [f32],
                          init: Option<&[f32]>) {
    use std::arch::x86_64::*;
    debug_assert!(xp.len() >= g.c * g.hp * g.wp && wt.len() >= g.c * 9 * g.o_pad && y + 2 < g.hp);
    debug_assert!(g.o_pad % NN_OB_FMA == 0 && g.wp >= g.wd.div_ceil(16) * 16 + 2);
    let (xptr, wptr) = (xp.as_ptr(), wt.as_ptr());
    let mut x0 = 0;
    while x0 < g.wd {
        let n = (g.wd - x0).min(16);
        let mut o0 = 0;
        while o0 < g.o_n {
            let mut a = [[_mm256_setzero_ps(); 2]; NN_OB_FMA];
            for k in 0..NN_OB_FMA {
                match init {
                    // rows of channels past o_n (padding) read the next row or the spare tail; unused
                    Some(iv) => {
                        let ip = iv.as_ptr().add((o0 + k).min(g.o_n - 1) * g.wd + x0);
                        a[k] = [_mm256_loadu_ps(ip), _mm256_loadu_ps(ip.add(8))];
                    }
                    None => {
                        let bv = _mm256_set1_ps(b[o0 + k]);
                        a[k] = [bv, bv];
                    }
                }
            }
            for ci in 0..g.c {
                for ky in 0..3 {
                    let row = xptr.add((ci * g.hp + y + ky) * g.wp + x0);
                    for kx in 0..3 {
                        let s0 = _mm256_loadu_ps(row.add(kx));
                        let s1 = _mm256_loadu_ps(row.add(kx + 8));
                        let wq = wptr.add((ci * 9 + ky * 3 + kx) * g.o_pad + o0);
                        for k in 0..NN_OB_FMA {
                            let wv = _mm256_set1_ps(*wq.add(k));
                            a[k][0] = _mm256_fmadd_ps(wv, s0, a[k][0]);
                            a[k][1] = _mm256_fmadd_ps(wv, s1, a[k][1]);
                        }
                    }
                }
            }
            let mut tmp = [0f32; 16];
            for k in 0..NN_OB_FMA.min(g.o_n - o0) {
                _mm256_storeu_ps(tmp.as_mut_ptr(), a[k][0]);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(8), a[k][1]);
                out_row[(o0 + k) * g.wd + x0..(o0 + k) * g.wd + x0 + n].copy_from_slice(&tmp[..n]);
            }
            o0 += NN_OB_FMA;
        }
        x0 += 16;
    }
}

/// Which row kernel a forward pass uses.
#[derive(Clone, Copy, PartialEq)]
enum NnKernel {
    Portable,
    Avx2,
    Fma,
}

/// 3x3 conv + bias (+ LeakyReLU) of `src` into `dst`'s interior, rows in parallel.
fn nn_conv_layer(src: &NnAct, l: &NnLayer, leak: f32, act: bool, kernel: NnKernel, dst: &mut NnAct) {
    debug_assert!(src.c == l.c && dst.c == l.o_n && dst.h == src.h && dst.w == src.w);
    let g = NnConvGeom { c: l.c, hp: src.hp, wp: src.wp, o_n: l.o_n, o_pad: l.o_pad, wd: src.w };
    let (h, wd, o_n) = (src.h, src.w, l.o_n);
    let (dhp, dwp) = (dst.hp, dst.wp);
    let dptr = HpPtr(dst.data.as_mut_ptr());
    (0..h).into_par_iter().for_each_init(|| vec![0f32; o_n * wd], |row, y| {
        let row = &mut row[..];
        match kernel {
            // SAFETY: the kernel was chosen after run-time feature detection; src, wt and bp are
            // sized from g (activation rows are padded to NN_WALIGN + 2, o_pad matches the kernel).
            #[cfg(target_arch = "x86_64")]
            NnKernel::Fma => unsafe { nn_conv_row_fma(&src.data, &l.wt, &l.bp, &g, y, row, None) },
            #[cfg(target_arch = "x86_64")]
            NnKernel::Avx2 => unsafe { nn_conv_row_avx2(&src.data, &l.wt, &l.bp, &g, y, row) },
            _ => nn_conv_row_portable(&src.data, &l.wt, &l.bp, &g, y, row),
        }
        if act {
            for v in row.iter_mut() {
                *v = v.max(leak * *v);
            }
        }
        let dptr = dptr;
        for o in 0..o_n {
            // SAFETY: row y of plane o inside dst's interior; each task owns a distinct y.
            unsafe {
                std::ptr::copy_nonoverlapping(row.as_ptr().add(o * wd), dptr.0.add((o * dhp + y + 1) * dwp + 1), wd);
            }
        }
    });
}

/// A conv layer whose input is `concat([2x nearest-neighbour upsample of x, skip])`, with the
/// upsampled channels folded into half-resolution weights. Nearest upsampling then a 3x3 kernel
/// equals, for each output phase (dy, dx), a 2x2 kernel on `x` whose taps are sums of the 3x3
/// taps that land on the same source pixel (rows: dy 0 -> {0}, {1, 2}; dy 1 -> {0, 1}, {2}).
/// That is 4 taps per upsampled channel instead of 9, and neither the upsampled tensor nor the
/// concatenation is ever written.
struct NnUpLayer {
    c_x: usize,
    c_s: usize,
    o_n: usize,
    o_pad: usize,
    /// (c_x, phase dy*2+dx, tap a*2+b, o_pad)
    wt_up: Vec<f32>,
    /// (c_s, 9, o_pad), the ordinary tap-major layout
    wt_skip: Vec<f32>,
    bp: Vec<f32>,
}

fn nn_up_layer(w: &numpy::ndarray::ArrayView4<f32>, b: &numpy::ndarray::ArrayView1<f32>, c_x: usize) -> PyResult<NnUpLayer> {
    let (o_n, c) = (w.shape()[0], w.shape()[1]);
    if w.shape()[2] != 3 || w.shape()[3] != 3 || b.len() != o_n || c_x == 0 || c_x >= c {
        return Err(pyo3::exceptions::PyValueError::new_err("up-conv weight shape mismatch"));
    }
    let (c_s, o_pad) = (c - c_x, o_n.div_ceil(NN_OB_FMA) * NN_OB_FMA);
    // 3x3 taps that read source row/col offset `a` (0 or 1) for output phase `d`
    let taps = |d: usize, a: usize| -> &'static [usize] {
        match (d, a) {
            (0, 0) => &[0],
            (0, _) => &[1, 2],
            (_, 0) => &[0, 1],
            _ => &[2],
        }
    };
    let mut wt_up = vec![0f32; c_x * 16 * o_pad];
    for o in 0..o_n {
        for ci in 0..c_x {
            for dy in 0..2 {
                for dx in 0..2 {
                    for a in 0..2 {
                        for bb in 0..2 {
                            let mut sum = 0f32;
                            for &ky in taps(dy, a) {
                                for &kx in taps(dx, bb) {
                                    sum += w[[o, ci, ky, kx]];
                                }
                            }
                            wt_up[((ci * 4 + dy * 2 + dx) * 4 + a * 2 + bb) * o_pad + o] = sum;
                        }
                    }
                }
            }
        }
    }
    let mut wt_skip = vec![0f32; c_s * 9 * o_pad];
    for o in 0..o_n {
        for cs in 0..c_s {
            for t in 0..9 {
                wt_skip[(cs * 9 + t) * o_pad + o] = w[[o, c_x + cs, t / 3, t % 3]];
            }
        }
    }
    let mut bp = vec![0f32; o_pad];
    for o in 0..o_n {
        bp[o] = b[o];
    }
    Ok(NnUpLayer { c_x, c_s, o_n, o_pad, wt_up, wt_skip, bp })
}

/// Output row `y` (all channels, `2 * x.w` wide) of the upsampled channels' share of an
/// `NnUpLayer`, plus the bias, into `out_row` (o_n rows of `2 * x.w`). 6 channels x 16 source
/// columns per block, one phase dx at a time, the two phases interleaved on store.
///
/// # Safety
/// The CPU must support AVX2 and FMA; `x` is a padded activation (wp >= ceil16(w) + 2) and `l`
/// was built for it (`o_pad` a multiple of `NN_OB_FMA`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn nn_up_row_fma(x: &NnAct, l: &NnUpLayer, y: usize, out_row: &mut [f32]) {
    use std::arch::x86_64::*;
    let (dy, i, wd) = (y & 1, y >> 1, 2 * x.w);
    debug_assert!(i < x.h && out_row.len() >= l.o_n * wd);
    let (xptr, wptr) = (x.data.as_ptr(), l.wt_up.as_ptr());
    let mut j0 = 0;
    while j0 < x.w {
        let npix = (wd - 2 * j0).min(32);
        let mut o0 = 0;
        while o0 < l.o_n {
            let mut first = [[_mm256_setzero_ps(); 2]; NN_OB_FMA];
            for dx in 0..2 {
                let mut a = [[_mm256_setzero_ps(); 2]; NN_OB_FMA];
                for k in 0..NN_OB_FMA {
                    let bv = _mm256_set1_ps(l.bp[o0 + k]);
                    a[k] = [bv, bv];
                }
                for ci in 0..l.c_x {
                    for ra in 0..2 {
                        let row = xptr.add((ci * x.hp + i + dy + ra) * x.wp + j0 + dx);
                        for cb in 0..2 {
                            let s0 = _mm256_loadu_ps(row.add(cb));
                            let s1 = _mm256_loadu_ps(row.add(cb + 8));
                            let wq = wptr.add(((ci * 4 + dy * 2 + dx) * 4 + ra * 2 + cb) * l.o_pad + o0);
                            for k in 0..NN_OB_FMA {
                                let wv = _mm256_set1_ps(*wq.add(k));
                                a[k][0] = _mm256_fmadd_ps(wv, s0, a[k][0]);
                                a[k][1] = _mm256_fmadd_ps(wv, s1, a[k][1]);
                            }
                        }
                    }
                }
                if dx == 0 {
                    first = a;
                } else {
                    let mut tmp = [0f32; 32];
                    for k in 0..NN_OB_FMA.min(l.o_n - o0) {
                        for v in 0..2 {
                            let lo = _mm256_unpacklo_ps(first[k][v], a[k][v]);
                            let hi = _mm256_unpackhi_ps(first[k][v], a[k][v]);
                            _mm256_storeu_ps(tmp.as_mut_ptr().add(v * 16), _mm256_permute2f128_ps(lo, hi, 0x20));
                            _mm256_storeu_ps(tmp.as_mut_ptr().add(v * 16 + 8), _mm256_permute2f128_ps(lo, hi, 0x31));
                        }
                        out_row[(o0 + k) * wd + 2 * j0..(o0 + k) * wd + 2 * j0 + npix].copy_from_slice(&tmp[..npix]);
                    }
                }
            }
            o0 += NN_OB_FMA;
        }
        j0 += 16;
    }
}

/// `nn_conv_layer` for an `NnUpLayer`: per output row the upsampled channels' share (with the
/// bias) into a scratch row, then the skip channels' 3x3 convolution accumulated on top.
fn nn_conv_layer_up(x: &NnAct, skip: &NnAct, l: &NnUpLayer, leak: f32, dst: &mut NnAct) {
    debug_assert!(x.c == l.c_x && skip.c == l.c_s && x.h * 2 == skip.h && x.w * 2 == skip.w && dst.c == l.o_n);
    let g = NnConvGeom { c: l.c_s, hp: skip.hp, wp: skip.wp, o_n: l.o_n, o_pad: l.o_pad, wd: skip.w };
    let (h, wd, o_n) = (skip.h, skip.w, l.o_n);
    let (dhp, dwp) = (dst.hp, dst.wp);
    let dptr = HpPtr(dst.data.as_mut_ptr());
    (0..h).into_par_iter().for_each_init(
        || (vec![0f32; o_n * wd + 16], vec![0f32; o_n * wd]),
        |(up, row), y| {
            // SAFETY: only called when the CPU has AVX2 + FMA (kernel choice in nn_forward); x and
            // skip are padded activations, wt_up / wt_skip / bp are sized by nn_up_layer, `up`
            // holds o_n * wd + 16 floats.
            #[cfg(target_arch = "x86_64")]
            unsafe {
                nn_up_row_fma(x, l, y, up);
                nn_conv_row_fma(&skip.data, &l.wt_skip, &l.bp, &g, y, row, Some(&up[..]));
            }
            #[cfg(not(target_arch = "x86_64"))]
            let _ = (&g, &up, &row);
            for v in row.iter_mut() {
                *v = v.max(leak * *v);
            }
            let dptr = dptr;
            for o in 0..o_n {
                // SAFETY: row y of plane o inside dst's interior; each task owns a distinct y.
                unsafe {
                    std::ptr::copy_nonoverlapping(row.as_ptr().add(o * wd), dptr.0.add((o * dhp + y + 1) * dwp + 1), wd);
                }
            }
        },
    );
}

/// 2x2 average pool (torch `AvgPool2d(2)`): `((a + b) + c) + d` then `* 0.25`, the order of the
/// numpy version's four strided adds.
fn nn_pool2(src: &NnAct) -> NnAct {
    let mut dst = NnAct::new(src.c, src.h / 2, src.w / 2);
    let (dh, dw, dhp, dwp) = (dst.h, dst.w, dst.hp, dst.wp);
    let dptr = HpPtr(dst.data.as_mut_ptr());
    let q = 0.25f32;
    (0..dh).into_par_iter().for_each(|y| {
        let dptr = dptr;
        for ci in 0..src.c {
            let r0 = &src.data[src.at(ci, 2 * y, 0)..src.at(ci, 2 * y, 0) + 2 * dw];
            let r1 = &src.data[src.at(ci, 2 * y + 1, 0)..src.at(ci, 2 * y + 1, 0) + 2 * dw];
            // SAFETY: row y of plane ci in dst's interior; each task owns a distinct y.
            let out = unsafe { std::slice::from_raw_parts_mut(dptr.0.add((ci * dhp + y + 1) * dwp + 1), dw) };
            for x in 0..dw {
                let mut s = r0[2 * x] + r1[2 * x];
                s += r0[2 * x + 1];
                s += r1[2 * x + 1];
                out[x] = s * q;
            }
        }
    });
    dst
}

/// `concat([2x nearest-neighbour upsample of x, skip])` along channels.
fn nn_up2_cat(x: &NnAct, skip: &NnAct) -> NnAct {
    debug_assert!(x.h * 2 == skip.h && x.w * 2 == skip.w);
    let mut dst = NnAct::new(x.c + skip.c, skip.h, skip.w);
    let (h, w, dhp, dwp) = (dst.h, dst.w, dst.hp, dst.wp);
    let dptr = HpPtr(dst.data.as_mut_ptr());
    (0..h).into_par_iter().for_each(|y| {
        let dptr = dptr;
        for ci in 0..x.c {
            let r = &x.data[x.at(ci, y / 2, 0)..x.at(ci, y / 2, 0) + x.w];
            // SAFETY: row y of plane ci in dst's interior; each task owns a distinct y.
            let out = unsafe { std::slice::from_raw_parts_mut(dptr.0.add((ci * dhp + y + 1) * dwp + 1), w) };
            for xx in 0..x.w {
                out[2 * xx] = r[xx];
                out[2 * xx + 1] = r[xx];
            }
        }
        for ci in 0..skip.c {
            let r = &skip.data[skip.at(ci, y, 0)..skip.at(ci, y, 0) + w];
            // SAFETY: as above, plane x.c + ci.
            let out = unsafe { std::slice::from_raw_parts_mut(dptr.0.add(((x.c + ci) * dhp + y + 1) * dwp + 1), w) };
            out.copy_from_slice(r);
        }
    });
    dst
}

/// The whole U-Net of `src/nn_denoise.py::forward` in one call: x (C, H, W) float32 with H and W
/// multiples of 4; `w` maps the layer names (`e0a`, `e0b`, `e1a`, `e1b`, `ma`, `mb`, `d1a`,
/// `d1b`, `d0a`, `d0b`, `out`) + `.weight` / `.bias` to the arrays. Returns (O, H, W).
///
/// With `fma` false the convolutions are `nn_conv3x3`'s kernels and the pooling / upsampling are
/// the numpy glue's arithmetic, so everything but the last 1x1 layer (summed here in channel
/// order; numpy's einsum uses BLAS) is bit-identical to calling `nn_conv3x3` per layer. With
/// `fma` true (and a CPU that has it) the convolutions use fused multiply-add: faster, but the
/// result differs in the last bits and depends on whether the CPU has FMA.
#[pyfunction]
#[pyo3(signature = (x, w, leak, fma=false))]
pub(crate) fn nn_forward<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f32>,
    w: &Bound<'py, pyo3::types::PyDict>,
    leak: f32,
    fma: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let bad = |m: &str| pyo3::exceptions::PyValueError::new_err(m.to_string());
    let xs = x.as_array().shape().to_vec();
    let (c0, h, wd) = (xs[0], xs[1], xs[2]);
    if h == 0 || wd == 0 || h % 4 != 0 || wd % 4 != 0 {
        return Err(bad("H and W must be positive multiples of 4"));
    }
    #[cfg(target_arch = "x86_64")]
    let kernel = if fma && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        NnKernel::Fma
    } else if std::is_x86_feature_detected!("avx2") {
        NnKernel::Avx2
    } else {
        NnKernel::Portable
    };
    #[cfg(not(target_arch = "x86_64"))]
    let kernel = NnKernel::Portable;
    let ob = if kernel == NnKernel::Fma { NN_OB_FMA } else { NN_OB };
    let get = |name: &str| -> PyResult<(PyReadonlyArray4<'py, f32>, PyReadonlyArray1<'py, f32>)> {
        let wk = w.get_item(format!("{name}.weight"))?.ok_or_else(|| bad(&format!("missing {name}.weight")))?;
        let bk = w.get_item(format!("{name}.bias"))?.ok_or_else(|| bad(&format!("missing {name}.bias")))?;
        Ok((wk.extract()?, bk.extract()?))
    };
    let layer = |name: &str| -> PyResult<NnLayer> {
        let (wa, ba) = get(name)?;
        nn_layer(&wa.as_array(), &ba.as_array(), ob)
    };
    let (e0a, e0b, e1a, e1b) = (layer("e0a")?, layer("e0b")?, layer("e1a")?, layer("e1b")?);
    let (ma, mb, d1b, d0b) = (layer("ma")?, layer("mb")?, layer("d1b")?, layer("d0b")?);
    // The concat layers: folded upsample (FMA kernel) or the plain layer behind a materialised concat.
    let concat_layer = |name: &str, c_x: usize| -> PyResult<(Option<NnUpLayer>, Option<NnLayer>)> {
        if kernel == NnKernel::Fma {
            let (wa, ba) = get(name)?;
            Ok((Some(nn_up_layer(&wa.as_array(), &ba.as_array(), c_x)?), None))
        } else {
            Ok((None, Some(layer(name)?)))
        }
    };
    let (d1a_up, d1a) = concat_layer("d1a", mb.o_n)?;
    let (d0a_up, d0a) = concat_layer("d0a", d1b.o_n)?;
    let (ow, ob_) = get("out")?;
    let (ow, ob_) = (ow.as_array().to_owned(), ob_.as_array().to_owned());
    if e0a.c != c0 || ow.shape()[1] != d0b.o_n || ow.shape()[2] != 1 || ow.shape()[3] != 1 {
        return Err(bad("network shape does not match the input"));
    }
    let o_out = ow.shape()[0];
    let xv = x.as_slice().map_err(|_| bad("x must be contiguous"))?;
    let out = py.detach(|| {
        let mut a0 = NnAct::new(c0, h, wd);
        for ci in 0..c0 {
            for y in 0..h {
                let d = a0.at(ci, y, 0);
                a0.data[d..d + wd].copy_from_slice(&xv[(ci * h + y) * wd..(ci * h + y + 1) * wd]);
            }
        }
        let conv = |src: &NnAct, l: &NnLayer| {
            let mut dst = NnAct::new(l.o_n, src.h, src.w);
            nn_conv_layer(src, l, leak, true, kernel, &mut dst);
            dst
        };
        let t = conv(&a0, &e0a);
        drop(a0);
        let s0 = conv(&t, &e0b);
        drop(t);
        let p0 = nn_pool2(&s0);
        let t = conv(&p0, &e1a);
        drop(p0);
        let s1 = conv(&t, &e1b);
        drop(t);
        let p1 = nn_pool2(&s1);
        let t = conv(&p1, &ma);
        drop(p1);
        let m = conv(&t, &mb);
        drop(t);
        let t = match (&d1a_up, &d1a) {
            (Some(up), _) => {
                let mut dst = NnAct::new(up.o_n, s1.h, s1.w);
                nn_conv_layer_up(&m, &s1, up, leak, &mut dst);
                dst
            }
            (_, Some(l)) => conv(&nn_up2_cat(&m, &s1), l),
            _ => unreachable!(),
        };
        drop((m, s1));
        let d1 = conv(&t, &d1b);
        drop(t);
        let t = match (&d0a_up, &d0a) {
            (Some(up), _) => {
                let mut dst = NnAct::new(up.o_n, s0.h, s0.w);
                nn_conv_layer_up(&d1, &s0, up, leak, &mut dst);
                dst
            }
            (_, Some(l)) => conv(&nn_up2_cat(&d1, &s0), l),
            _ => unreachable!(),
        };
        drop((d1, s0));
        let d0 = conv(&t, &d0b);
        drop(t);
        // 1x1 output layer: per row, channel-order accumulation (vectorises over x).
        let mut out = vec![0f32; o_out * h * wd];
        let optr = HpPtr(out.as_mut_ptr());
        (0..h).into_par_iter().for_each(|y| {
            let optr = optr;
            for o in 0..o_out {
                // SAFETY: row y of plane o of `out`; each task owns a distinct y.
                let dst = unsafe { std::slice::from_raw_parts_mut(optr.0.add((o * h + y) * wd), wd) };
                for ci in 0..d0.c {
                    let wv = ow[[o, ci, 0, 0]];
                    let r = &d0.data[d0.at(ci, y, 0)..d0.at(ci, y, 0) + wd];
                    for xx in 0..wd {
                        dst[xx] += wv * r[xx];
                    }
                }
                let bv = ob_[o];
                for v in dst.iter_mut() {
                    *v += bv;
                }
            }
        });
        out
    });
    let arr = numpy::ndarray::Array3::from_shape_vec((o_out, h, wd), out).expect("shape mismatch building output");
    Ok(arr.into_pyarray(py))
}
