//! RCD demosaicing.

use crate::*;

// ---------------------------------------------------------------------------
// RCD (Ratio Corrected Demosaicing, L. Sanz Rodriguez) -- src/debayer.py::debayer_rcd
// ---------------------------------------------------------------------------
// A line-for-line port of the numpy reference: every stage is computed over the
// whole frame in f32 (as the numpy mirror; f64 doubled the memory traffic of a
// bandwidth-bound kernel), intermediate arrays are read through numpy's
// np.pad(mode='reflect') (== mirror_idx) at the same offsets, and every sum is
// accumulated in the reference's left-to-right order, so the result is
// bit-identical. The caller scales the mosaic by its maximum, pastes the 4-px
// Malvar border and does the dtype work; this returns the interior RGB (H, W, 3)
// already multiplied back by `scale` and rounded to f32.

// Every array is stored reflect-padded by RCD_P on each side (np.pad(mode='reflect')
// == mirror_idx), so a tap is a plain load from a shifted row slice.
pub(crate) const RCD_P: usize = 5;

pub(crate) fn rcd_pad_in_place(a: &mut [f32], h: usize, w: usize) {
    let pw = w + 2 * RCD_P;
    // columns, on the frame rows
    for py in RCD_P..h + RCD_P {
        let row = &mut a[py * pw..(py + 1) * pw];
        for k in 0..RCD_P {
            let xl = mirror_idx(k as isize - RCD_P as isize, w);
            let xr = mirror_idx((w + k) as isize, w);
            row[k] = row[xl + RCD_P];
            row[w + RCD_P + k] = row[xr + RCD_P];
        }
    }
    // whole rows (incl. their padded columns)
    for k in 0..RCD_P {
        let yt = mirror_idx(k as isize - RCD_P as isize, h);
        let yb = mirror_idx((h + k) as isize, h);
        let (src_t, src_b) = ((yt + RCD_P) * pw, (yb + RCD_P) * pw);
        a.copy_within(src_t..src_t + pw, k * pw);
        a.copy_within(src_b..src_b + pw, (h + RCD_P + k) * pw);
    }
}

#[pyfunction]
#[pyo3(signature = (raw, scale, r_off, b_off))]
pub(crate) fn debayer_rcd_native<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray2<'py, f32>,
    scale: f64,
    r_off: (usize, usize),
    b_off: (usize, usize),
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let arr = raw.as_array();
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    if h < 16 || w < 16 {
        return Err(pyo3::exceptions::PyValueError::new_err("frame too small for RCD"));
    }
    let owned: Vec<f32>;
    let src: &[f32] = match arr.as_slice() {
        Some(x) => x,
        None => {
            owned = arr.iter().copied().collect();
            &owned
        }
    };
    let out = py.detach(|| rcd_strips(src, h, w, scale, r_off, b_off));
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, 3), out)
        .unwrap()
        .into_pyarray(py))
}

/// `debayer_rcd_native` writing into `out` (h, w, 3), e.g. a frame-store slot.
/// `grid`: the session's four 2x2 green offsets (`_GRID_PARITY` order), subtracted
/// and clipped as `debayer.py::_apply_fixed_grid` does, fused into the output write.
#[pyfunction]
#[pyo3(signature = (raw, out, scale, r_off, b_off, grid=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn debayer_rcd_native_into<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray2<'py, f32>,
    mut out: numpy::PyReadwriteArray3<'py, f32>,
    scale: f64,
    r_off: (usize, usize),
    b_off: (usize, usize),
    grid: Option<(f64, f64, f64, f64)>,
) -> PyResult<()> {
    let arr = raw.as_array();
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    if h < 16 || w < 16 {
        return Err(pyo3::exceptions::PyValueError::new_err("frame too small for RCD"));
    }
    if out.as_array().shape() != [h, w, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err("out must be (h, w, 3)"));
    }
    let owned: Vec<f32>;
    let src: &[f32] = match arr.as_slice() {
        Some(x) => x,
        None => {
            owned = arr.iter().copied().collect();
            &owned
        }
    };
    let dst = out.as_slice_mut().map_err(|_| pyo3::exceptions::PyValueError::new_err("out must be contiguous"))?;
    // np.float32(off): each offset rounded to f32 once, as numpy does
    let grid = grid.map(|(a, b, c, d)| [a as f32, b as f32, c as f32, d as f32]);
    py.detach(|| rcd_strips_into(src, h, w, scale, r_off, b_off, grid, dst));
    Ok(())
}

/// Scratch buffers reused across RCD strips and calls. Every stage array is fully
/// overwritten (frame rows by the stage, padding by `rcd_pad_in_place`), so a
/// reused buffer needs no clearing. Without this each strip's ~14 arrays were
/// fresh allocations the OS zero-fills on first touch: hundreds of MB of zeroing
/// per frame, which is what made RCD slow under 16 Phase 1 workers.
pub(crate) static RCD_POOL: std::sync::OnceLock<std::sync::Mutex<Vec<Vec<f32>>>> = std::sync::OnceLock::new();

pub(crate) fn rcd_take(n: usize) -> Vec<f32> {
    let pool = RCD_POOL.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    if let Ok(mut p) = pool.lock() {
        if let Some(i) = p.iter().position(|v| v.len() == n) {
            return p.swap_remove(i);
        }
    }
    vec![0f32; n]
}

pub(crate) fn rcd_give(v: Vec<f32>) {
    let pool = RCD_POOL.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    if let Ok(mut p) = pool.lock() {
        if p.len() < 48 {
            p.push(v);
        }
    }
}

/// RCD in tiles of `RCD_TILE` x `RCD_TILE_X` output pixels, each run on its tile plus
/// `RCD_HALO` pixels around it. Whole-frame stages kept ~14 frame-sized arrays in
/// flight and streamed ~1.3 GB through memory per frame, so under Phase 1's
/// workers the kernel was DRAM-bandwidth bound (0.28 s/frame alone, 0.76 s with
/// 8 processes, 1.26 s with 16, where Malvar stayed at 0.08-0.16). A tile's
/// arrays (~1.3 MB together) stay in the core's L2. RCD's dependency reach is ~10
/// px (R/B at G reads R/B 3 px away, which read G 2 px away, which reads the CFA
/// 4 px away), so pixels more than `RCD_HALO` from a tile's cut edge come out
/// exactly as in a whole-frame pass; at the frame's own edges the tile's reflect
/// padding is the frame's. Tiles start on even rows/columns, so the Bayer parity
/// is unchanged. Bit-identical to the numpy mirror. Buffers come from `RCD_POOL`.
pub(crate) const RCD_TILE: usize = 128;

/// Tile width: 128 x 256 tiles (halo overhead 1.30x the output instead of 1.41x for
/// 128 x 128) measured ~15-20% faster on real 2048x3056 frames, alone (2 threads) and
/// with 8 Phase 1 processes x 2 threads; 320-384 wide was the same, 512 slower.
pub(crate) const RCD_TILE_X: usize = 256;

pub(crate) const RCD_HALO: usize = 12;

pub(crate) fn rcd_strips(src: &[f32], h: usize, w: usize, scale: f64,
              r_off: (usize, usize), b_off: (usize, usize)) -> Vec<f32> {
    let mut out = rcd_take(h * w * 3); // fully overwritten
    rcd_strips_into(src, h, w, scale, r_off, b_off, None, &mut out);
    out
}

/// `debayer.py::_apply_fixed_grid` on one green value: `v -= f32(off)` then
/// `np.maximum(v, 0)` (NaN stays NaN, -0.0 and negatives become +0.0).
#[inline(always)]
pub(crate) fn rcd_grid_sub(v: f32, off: f32) -> f32 {
    let v = v - off;
    if v > 0.0 || v.is_nan() { v } else { 0.0 }
}

/// Write `rows` x `cols` output pixels from the padded tile planes (tile rows from
/// `ty0`, columns from `tx0`; `tw` = tile width) into `dst` (row stride `dstride`
/// floats, interleaved RGB): each value times `sc` (numpy: f32 array * Python float
/// is an f32 multiply), then -- with `grid` -- green minus the session's 2x2 offset
/// for its global parity (`_GRID_PARITY` order: (0,0), (0,1), (1,0), (1,1)), clipped
/// as `_apply_fixed_grid` does. `gy0`/`gx0`: global coordinates of the first pixel.
/// The same f32 operations in the same order as debayer + `_apply_fixed_grid`, so
/// fusing the grid here is bit-identical and saves its four strided passes over
/// the 75 MB output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rcd_emit(planes: &(Vec<f32>, Vec<f32>, Vec<f32>), tw: usize, ty0: usize, tx0: usize,
            rows: usize, cols: usize, sc: f32, grid: Option<[f32; 4]>, gy0: usize, gx0: usize,
            dst: &mut [f32], dstride: usize) {
    let (r, g, b) = planes;
    for row in 0..rows {
        let (rr, gg, bb) = (&rcd_row(r, tw, ty0 + row, 0, 0)[tx0..tx0 + cols],
                            &rcd_row(g, tw, ty0 + row, 0, 0)[tx0..tx0 + cols],
                            &rcd_row(b, tw, ty0 + row, 0, 0)[tx0..tx0 + cols]);
        let o = &mut dst[row * dstride..row * dstride + cols * 3];
        match grid {
            None => {
                for x in 0..cols {
                    o[3 * x] = rr[x] * sc;
                    o[3 * x + 1] = gg[x] * sc;
                    o[3 * x + 2] = bb[x] * sc;
                }
            }
            Some(gd) => {
                let py = (gy0 + row) & 1;
                // offsets for even / odd local columns of this row
                let (oe, oo) = if gx0 & 1 == 0 { (gd[2 * py], gd[2 * py + 1]) }
                               else { (gd[2 * py + 1], gd[2 * py]) };
                for x in 0..cols {
                    let off = if x & 1 == 0 { oe } else { oo };
                    o[3 * x] = rr[x] * sc;
                    o[3 * x + 1] = rcd_grid_sub(gg[x] * sc, off);
                    o[3 * x + 2] = bb[x] * sc;
                }
            }
        }
    }
}

pub(crate) fn rcd_give_planes(p: (Vec<f32>, Vec<f32>, Vec<f32>)) {
    rcd_give(p.0);
    rcd_give(p.1);
    rcd_give(p.2);
}

/// `rcd_strips` writing into a caller-supplied (h, w, 3) buffer -- e.g. the frame's own
/// slot of Phase 1's frame store, so neither a fresh 75 MB output (zero-filled by the OS
/// on first touch) nor a copy into the store is needed. Each tile's planes are written
/// straight into `out` (no per-tile interleaved buffer), with the optional fused grid.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rcd_strips_into(src: &[f32], h: usize, w: usize, scale: f64,
                   r_off: (usize, usize), b_off: (usize, usize), grid: Option<[f32; 4]>,
                   out: &mut [f32]) {
    let sc = scale as f32;
    let (tyl, txl) = (RCD_TILE, RCD_TILE_X);
    if h <= tyl + 2 * RCD_HALO && w <= txl + 2 * RCD_HALO {
        let planes = rcd_core(src, h, w, scale, r_off, b_off);
        par_rows(&mut *out, w * 3, |y, row| {
            rcd_emit(&planes, w, y, 0, 1, w, sc, grid, y, 0, row, w * 3);
        });
        rcd_give_planes(planes);
        return;
    }
    // extended span [lo, hi) of tile k along an axis of length n, >= 16 long
    let span = |k: usize, n: usize, t: usize| -> (usize, usize, usize) {
        let s0 = k * t;
        let len = t.min(n - s0);
        let mut lo = s0.saturating_sub(RCD_HALO);
        let mut hi = (s0 + len + RCD_HALO).min(n);
        if hi - lo < 16 {
            lo = hi.saturating_sub(16);
            hi = (lo + 16).min(n);
        }
        (lo, hi, len)
    };
    let tx = w.div_ceil(txl);
    // one band of tile rows per task: each band owns its output rows
    par_rows(&mut *out, tyl * w * 3, |ky, band| {
        let (y0, y1, rows) = span(ky, h, tyl);
        let sy = ky * tyl;
        for kx in 0..tx {
            let (x0, x1, cols) = span(kx, w, txl);
            let sx = kx * txl;
            let (th, tw) = (y1 - y0, x1 - x0);
            let mut tile = rcd_take(th * tw);
            for r in 0..th {
                tile[r * tw..(r + 1) * tw].copy_from_slice(&src[(y0 + r) * w + x0..(y0 + r) * w + x1]);
            }
            let planes = rcd_core(&tile, th, tw, scale, r_off, b_off);
            rcd_give(tile);
            rcd_emit(&planes, tw, sy - y0, sx - x0, rows, cols, sc, grid, sy, sx,
                     &mut band[sx * 3..], w * 3);
            rcd_give_planes(planes);
        }
    });
}

/// A row of a padded array at frame row `y` shifted by (dy, dx): `w` values.
#[inline(always)]
pub(crate) fn rcd_row(a: &[f32], w: usize, y: usize, dy: isize, dx: isize) -> &[f32] {
    let pw = w + 2 * RCD_P;
    let start = ((y + RCD_P) as isize + dy) as usize * pw + (RCD_P as isize + dx) as usize;
    &a[start..start + w]
}

/// Fill a padded array row by row (`f(y, out_row)` writes the `w` frame values),
/// then reflect-pad it. Rows are independent, so they run in parallel.
pub(crate) fn rcd_rows<F: Fn(usize, &mut [f32]) + Sync>(h: usize, w: usize, f: F) -> Vec<f32> {
    let pw = w + 2 * RCD_P;
    let mut out = rcd_take((h + 2 * RCD_P) * pw);
    par_rows(&mut out, pw, |py, row| {
        if py >= RCD_P && py < h + RCD_P {
            f(py - RCD_P, &mut row[RCD_P..RCD_P + w]);
        }
    });
    rcd_pad_in_place(&mut out, h, w);
    out
}

macro_rules! rcd_core_impl {
    ($src:ident, $h:ident, $w:ident, $scale:ident, $r_off:ident, $b_off:ident) => {{
        let (src, h, w, scale, r_off, b_off) = ($src, $h, $w, $scale, $r_off, $b_off);
    const EPS: f32 = 1e-5;
    const EPSSQ: f32 = 1e-10;
    // numpy: np.nan_to_num(a / scale) in f64, then .astype(float32)
    let cfa = rcd_rows(h, w, |y, o| {
        let s = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let q = s[x] as f64 / scale;
            o[x] = (if q.is_nan() {
                0.0
            } else if q == f64::INFINITY {
                f64::MAX
            } else if q == f64::NEG_INFINITY {
                f64::MIN
            } else {
                q
            }) as f32;
        }
    });
    // site class of (y, x): 0 = R, 1 = G, 2 = B
    let row_kind = |y: usize| -> (Option<usize>, u8) {
        if y % 2 == r_off.0 {
            (Some(r_off.1), 0)
        } else if y % 2 == b_off.0 {
            (Some(b_off.1), 2)
        } else {
            (None, 1)
        }
    };
    let c = |y, dy, dx| rcd_row(&cfa, w, y, dy, dx);

    // Step 1: vertical/horizontal discrimination
    let v_hpf = rcd_rows(h, w, |y, o| {
        let (m3, m2, m1, c0, p1, p2, p3) =
            (c(y, -3, 0), c(y, -2, 0), c(y, -1, 0), c(y, 0, 0), c(y, 1, 0), c(y, 2, 0), c(y, 3, 0));
        for x in 0..w {
            let t = (m3[x] - m1[x] - p1[x] + p3[x]) - 3.0 * (m2[x] + p2[x]) + 6.0 * c0[x];
            o[x] = t * t;
        }
    });
    let h_hpf = rcd_rows(h, w, |y, o| {
        let (m3, m2, m1, c0, p1, p2, p3) =
            (c(y, 0, -3), c(y, 0, -2), c(y, 0, -1), c(y, 0, 0), c(y, 0, 1), c(y, 0, 2), c(y, 0, 3));
        for x in 0..w {
            let t = (m3[x] - m1[x] - p1[x] + p3[x]) - 3.0 * (m2[x] + p2[x]) + 6.0 * c0[x];
            o[x] = t * t;
        }
    });
    let vh_dir = rcd_rows(h, w, |y, o| {
        let (va, vb, vc) = (rcd_row(&v_hpf, w, y, -1, 0), rcd_row(&v_hpf, w, y, 0, 0),
                            rcd_row(&v_hpf, w, y, 1, 0));
        let (ha, hb, hc) = (rcd_row(&h_hpf, w, y, 0, -1), rcd_row(&h_hpf, w, y, 0, 0),
                            rcd_row(&h_hpf, w, y, 0, 1));
        for x in 0..w {
            let vs = (va[x] + vb[x] + vc[x]).max(EPSSQ);
            let hs = (ha[x] + hb[x] + hc[x]).max(EPSSQ);
            o[x] = vs / (vs + hs);
        }
    });
    rcd_give(v_hpf);
    rcd_give(h_hpf);
    let disc = |d: &[f32]| -> Vec<f32> {
        rcd_rows(h, w, |y, o| {
            let (a, b, cc, dd, e) = (rcd_row(d, w, y, -1, -1), rcd_row(d, w, y, -1, 1),
                                     rcd_row(d, w, y, 1, -1), rcd_row(d, w, y, 1, 1),
                                     rcd_row(d, w, y, 0, 0));
            for x in 0..w {
                let nb = 0.25 * (a[x] + b[x] + cc[x] + dd[x]);
                let v = e[x];
                o[x] = if (0.5 - v).abs() < (0.5 - nb).abs() { nb } else { v };
            }
        })
    };
    let vh_disc = disc(&vh_dir);
    rcd_give(vh_dir);

    // Step 2: low-pass filter (read only at R/B sites; computed everywhere)
    let lpf = rcd_rows(h, w, |y, o| {
        let (c0, n, s, wv, e) = (c(y, 0, 0), c(y, -1, 0), c(y, 1, 0), c(y, 0, -1), c(y, 0, 1));
        let (nw, ne, sw, se) = (c(y, -1, -1), c(y, -1, 1), c(y, 1, -1), c(y, 1, 1));
        for x in 0..w {
            o[x] = c0[x] + 0.5 * (n[x] + s[x] + wv[x] + e[x])
                + 0.25 * (nw[x] + ne[x] + sw[x] + se[x]);
        }
    });

    // Step 3: green at R and B
    let g = rcd_rows(h, w, |y, o| {
        let cm = |k: isize| c(y, k, 0);
        let ch = |k: isize| c(y, 0, k);
        let (vm4, vm3, vm2, vm1, c0, vp1, vp2, vp3, vp4) =
            (cm(-4), cm(-3), cm(-2), cm(-1), cm(0), cm(1), cm(2), cm(3), cm(4));
        let (hm4, hm3, hm2, hm1, hp1, hp2, hp3, hp4) =
            (ch(-4), ch(-3), ch(-2), ch(-1), ch(1), ch(2), ch(3), ch(4));
        let l0 = rcd_row(&lpf, w, y, 0, 0);
        let (ln, ls, lw, le) = (rcd_row(&lpf, w, y, -2, 0), rcd_row(&lpf, w, y, 2, 0),
                                rcd_row(&lpf, w, y, 0, -2), rcd_row(&lpf, w, y, 0, 2));
        let vd = rcd_row(&vh_disc, w, y, 0, 0);
        let (gx, _) = row_kind(y);
        for x in 0..w {
            let n_grad = EPS + (vm1[x] - vp1[x]).abs() + (c0[x] - vm2[x]).abs()
                + (vm1[x] - vm3[x]).abs() + (vm2[x] - vm4[x]).abs();
            let s_grad = EPS + (vm1[x] - vp1[x]).abs() + (c0[x] - vp2[x]).abs()
                + (vp1[x] - vp3[x]).abs() + (vp2[x] - vp4[x]).abs();
            let w_grad = EPS + (hm1[x] - hp1[x]).abs() + (c0[x] - hm2[x]).abs()
                + (hm1[x] - hm3[x]).abs() + (hm2[x] - hm4[x]).abs();
            let e_grad = EPS + (hm1[x] - hp1[x]).abs() + (c0[x] - hp2[x]).abs()
                + (hp1[x] - hp3[x]).abs() + (hp2[x] - hp4[x]).abs();
            let l = l0[x];
            let n_est = vm1[x] * (1.0 + (l - ln[x]) / (EPS + l + ln[x]));
            let s_est = vp1[x] * (1.0 + (l - ls[x]) / (EPS + l + ls[x]));
            let w_est = hm1[x] * (1.0 + (l - lw[x]) / (EPS + l + lw[x]));
            let e_est = hp1[x] * (1.0 + (l - le[x]) / (EPS + l + le[x]));
            let v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad);
            let h_est = (w_grad * e_est + e_grad * w_est) / (e_grad + w_grad);
            let v = vd[x] * h_est + (1.0 - vd[x]) * v_est;
            let v = if v < 0.0 { 0.0 } else { v };
            let is_rb = matches!(gx, Some(px) if x & 1 == px);
            o[x] = if is_rb { v } else { c0[x] };
        }
    });
    rcd_give(lpf);

    // Step 4.1-4.2: diagonal discrimination
    let p_hpf = rcd_rows(h, w, |y, o| {
        let (m3, m2, m1, c0, p1, p2, p3) = (c(y, -3, -3), c(y, -2, -2), c(y, -1, -1), c(y, 0, 0),
                                            c(y, 1, 1), c(y, 2, 2), c(y, 3, 3));
        for x in 0..w {
            let t = (m3[x] - m1[x] - p1[x] + p3[x]) - 3.0 * (m2[x] + p2[x]) + 6.0 * c0[x];
            o[x] = t * t;
        }
    });
    let q_hpf = rcd_rows(h, w, |y, o| {
        let (m3, m2, m1, c0, p1, p2, p3) = (c(y, -3, 3), c(y, -2, 2), c(y, -1, 1), c(y, 0, 0),
                                            c(y, 1, -1), c(y, 2, -2), c(y, 3, -3));
        for x in 0..w {
            let t = (m3[x] - m1[x] - p1[x] + p3[x]) - 3.0 * (m2[x] + p2[x]) + 6.0 * c0[x];
            o[x] = t * t;
        }
    });
    let pq_dir = rcd_rows(h, w, |y, o| {
        let (pa, pb, pc) = (rcd_row(&p_hpf, w, y, -1, -1), rcd_row(&p_hpf, w, y, 0, 0),
                            rcd_row(&p_hpf, w, y, 1, 1));
        let (qa, qb, qc) = (rcd_row(&q_hpf, w, y, -1, 1), rcd_row(&q_hpf, w, y, 0, 0),
                            rcd_row(&q_hpf, w, y, 1, -1));
        for x in 0..w {
            let ps = (pa[x] + pb[x] + pc[x]).max(EPSSQ);
            let qs = (qa[x] + qb[x] + qc[x]).max(EPSSQ);
            o[x] = ps / (ps + qs);
        }
    });
    rcd_give(p_hpf);
    rcd_give(q_hpf);
    let pq_disc = disc(&pq_dir);
    rcd_give(pq_dir);

    // Step 4.3: R at B sites, B at R sites (from raw R / raw B planes)
    let own = |want: u8| {
        rcd_rows(h, w, |y, o| {
            let c0 = c(y, 0, 0);
            let (px, kind) = row_kind(y);
            for x in 0..w {
                let hit = kind == want && matches!(px, Some(p) if x & 1 == p);
                o[x] = if hit { c0[x] } else { 0.0 };
            }
        })
    };
    let diag = |cc: &[f32], target: u8| -> Vec<f32> {
        rcd_rows(h, w, |y, o| {
            let cv = |dy, dx| rcd_row(cc, w, y, dy, dx);
            let gv = |dy, dx| rcd_row(&g, w, y, dy, dx);
            let (c_nw, c_ne, c_sw, c_se) = (cv(-1, -1), cv(-1, 1), cv(1, -1), cv(1, 1));
            let (c_nw3, c_ne3, c_sw3, c_se3) = (cv(-3, -3), cv(-3, 3), cv(3, -3), cv(3, 3));
            let (g_nw2, g_ne2, g_sw2, g_se2) = (gv(-2, -2), gv(-2, 2), gv(2, -2), gv(2, 2));
            let (g_nw, g_ne, g_sw, g_se) = (gv(-1, -1), gv(-1, 1), gv(1, -1), gv(1, 1));
            let g0 = gv(0, 0);
            let c0 = cv(0, 0);
            let pd = rcd_row(&pq_disc, w, y, 0, 0);
            let (px, kind) = row_kind(y);
            for x in 0..w {
                let nw = EPS + (c_nw[x] - c_se[x]).abs() + (c_nw[x] - c_nw3[x]).abs()
                    + (g0[x] - g_nw2[x]).abs();
                let ne = EPS + (c_ne[x] - c_sw[x]).abs() + (c_ne[x] - c_ne3[x]).abs()
                    + (g0[x] - g_ne2[x]).abs();
                let sw = EPS + (c_sw[x] - c_ne[x]).abs() + (c_sw[x] - c_sw3[x]).abs()
                    + (g0[x] - g_sw2[x]).abs();
                let se = EPS + (c_se[x] - c_nw[x]).abs() + (c_se[x] - c_se3[x]).abs()
                    + (g0[x] - g_se2[x]).abs();
                let nw_e = c_nw[x] - g_nw[x];
                let ne_e = c_ne[x] - g_ne[x];
                let sw_e = c_sw[x] - g_sw[x];
                let se_e = c_se[x] - g_se[x];
                let p_est = (nw * se_e + se * nw_e) / (nw + se);
                let q_est = (ne * sw_e + sw * ne_e) / (ne + sw);
                let v = g0[x] + (1.0 - pd[x]) * p_est + pd[x] * q_est;
                let v = if v < 0.0 { 0.0 } else { v };
                let hit = kind == target && matches!(px, Some(p) if x & 1 == p);
                o[x] = if hit { v } else { c0[x] };
            }
        })
    };
    let o0 = own(0);
    let r1 = diag(&o0, 2);
    rcd_give(o0);
    let o2 = own(2);
    let b1 = diag(&o2, 0);
    rcd_give(o2);
    rcd_give(pq_disc);

    // Step 4.4: R and B at G sites
    let card = |cc: &[f32]| -> Vec<f32> {
        rcd_rows(h, w, |y, o| {
            let cv = |dy, dx| rcd_row(cc, w, y, dy, dx);
            let gv = |dy, dx| rcd_row(&g, w, y, dy, dx);
            let (cn, cs, cw, ce) = (cv(-1, 0), cv(1, 0), cv(0, -1), cv(0, 1));
            let (cn3, cs3, cw3, ce3) = (cv(-3, 0), cv(3, 0), cv(0, -3), cv(0, 3));
            let (gn2, gs2, gw2, ge2) = (gv(-2, 0), gv(2, 0), gv(0, -2), gv(0, 2));
            let (gn, gs, gw, ge) = (gv(-1, 0), gv(1, 0), gv(0, -1), gv(0, 1));
            let g0 = gv(0, 0);
            let c0 = cv(0, 0);
            let vd = rcd_row(&vh_disc, w, y, 0, 0);
            let (px, _) = row_kind(y);
            for x in 0..w {
                let n_g = EPS + (g0[x] - gn2[x]).abs() + (cn[x] - cs[x]).abs()
                    + (cn[x] - cn3[x]).abs();
                let s_g = EPS + (g0[x] - gs2[x]).abs() + (cs[x] - cn[x]).abs()
                    + (cs[x] - cs3[x]).abs();
                let w_g = EPS + (g0[x] - gw2[x]).abs() + (cw[x] - ce[x]).abs()
                    + (cw[x] - cw3[x]).abs();
                let e_g = EPS + (g0[x] - ge2[x]).abs() + (ce[x] - cw[x]).abs()
                    + (ce[x] - ce3[x]).abs();
                let n_e = cn[x] - gn[x];
                let s_e = cs[x] - gs[x];
                let w_e = cw[x] - gw[x];
                let e_e = ce[x] - ge[x];
                let v_e = (n_g * s_e + s_g * n_e) / (n_g + s_g);
                let h_e = (e_g * w_e + w_g * e_e) / (e_g + w_g);
                let v = g0[x] + (1.0 - vd[x]) * v_e + vd[x] * h_e;
                let v = if v < 0.0 { 0.0 } else { v };
                let is_g = !matches!(px, Some(p) if x & 1 == p);
                o[x] = if is_g { v } else { c0[x] };
            }
        })
    };
    let r = card(&r1);
    let b = card(&b1);
    rcd_give(r1);
    rcd_give(b1);
    for v in [cfa, vh_disc] {
        rcd_give(v);
    }
    (r, g, b)
    }};
}

/// RCD over the whole frame, returning the padded R, G, B planes (unscaled; see
/// `rcd_emit_rows`). Every stage is a branch-free loop over contiguous
/// row slices (computed at every pixel, then selected per Bayer site, as the
/// numpy mirror does), so the compiler vectorises it; the per-pixel arithmetic
/// and its order are the mirror's, so the result is bit-identical.
pub(crate) fn rcd_core(src: &[f32], h: usize, w: usize, scale: f64,
            r_off: (usize, usize), b_off: (usize, usize)) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Run-time AVX2 twin, as for the 3x3 median: the crate targets baseline
    // x86-64, and these loops are where a wider vector pays. No FMA is enabled
    // and Rust never contracts a*b+c on its own, so both give identical bits.
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU reports AVX2, the only requirement of `rcd_core_avx2`.
            return unsafe { rcd_core_avx2(src, h, w, scale, r_off, b_off) };
        }
    }
    rcd_core_impl!(src, h, w, scale, r_off, b_off)
}

/// # Safety
/// The CPU must support AVX2 (check `is_x86_feature_detected!("avx2")` first).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rcd_core_avx2(src: &[f32], h: usize, w: usize, scale: f64,
                        r_off: (usize, usize), b_off: (usize, usize)) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // the macro puts every stage closure lexically inside this function, so
    // they inherit the target feature
    rcd_core_impl!(src, h, w, scale, r_off, b_off)
}
