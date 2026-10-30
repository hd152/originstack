//! Hot-pixel and spike rejection for Bayer and RGB frames.

use crate::*;

/// 3x3 median at plane position (i, j) of the (py, px) Bayer sub-plane of a
/// row-major `w`-wide mosaic, reflect boundary on the sub-plane (scipy
/// `mode='reflect'`), exactly what `median_filter(sub, size=3)` returns there.
#[inline]
pub(crate) fn bayer_plane_median3(data: &[f32], w: usize, hh: usize, ww: usize, py: usize, px: usize,
                       i: usize, j: usize) -> f32 {
    let mut win = [0f32; 9];
    let mut k = 0;
    for di in -1isize..=1 {
        let ii = reflect_idx(i as isize + di, hh);
        for dj in -1isize..=1 {
            let jj = reflect_idx(j as isize + dj, ww);
            win[k] = data[(2 * ii + py) * w + 2 * jj + px];
            k += 1;
        }
    }
    median9(&mut win)
}

/// Bayer-aware hot-pixel fix on a 2-D mosaic, returning a new array.
///
/// `hot_map` (optional, u8): those pixels are replaced by their sub-plane 3x3
/// median -- computed only at the flagged pixels, not over the frame.
/// `threshold` (optional): statistical detection per 2x2 sub-plane -- a pixel
/// whose excess over the plane's 3x3 median exceeds `threshold` times
/// 1.4826 x the plane's median absolute deviation is replaced by that median.
/// `star_support` (optional, with `threshold`): a flagged pixel is *kept* when any of
/// its four adjacent mosaic pixels (one pixel away, in the other colour planes) is
/// itself more than `star_support` sigma above its own plane's median. A hot pixel is a
/// single-sensor-pixel event, so its neighbours stay normal; a star lifts its
/// neighbours with it, and without this test the peak of every bright star was replaced
/// by its plane median (a star is only ~2 pixels wide in a half-resolution plane), which
/// clipped the cores and softened and noised the whole stack.
/// Give one or the other, not both (the numpy path shares one median between
/// the two; the pipeline never asks for both at once). All arithmetic is f32
/// in numpy's order, so the result is bit-identical to `_fix_hot_bayer`.
#[pyfunction]
#[pyo3(signature = (data, hot_map=None, threshold=None, star_support=None))]
pub(crate) fn hot_pixel_bayer<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    hot_map: Option<PyReadonlyArray2<'py, u8>>,
    threshold: Option<f32>,
    star_support: Option<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if hot_map.is_some() && threshold.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "give either hot_map or threshold",
        ));
    }
    let arr = data.as_array();
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    let d: &[f32] = arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("data must be contiguous"))?;
    let map_view = hot_map.as_ref().map(|m| m.as_array());
    let map: Option<&[u8]> = match &map_view {
        Some(v) => Some(v.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("hot_map must be contiguous")
        })?),
        None => None,
    };
    if let Some(m) = map {
        if m.len() != h * w {
            return Err(pyo3::exceptions::PyValueError::new_err("hot_map shape mismatch"));
        }
    }

    let out = py.detach(|| {
        let mut out = d.to_vec();
        if let Some(m) = map {
            par_rows(&mut out, w, |y, row| {
                let py_ = y & 1;
                let i = y >> 1;
                let hh = (h - py_ + 1) / 2;
                for x in 0..w {
                    if m[y * w + x] == 0 {
                        continue;
                    }
                    let px = x & 1;
                    let ww = (w - px + 1) / 2;
                    row[x] = bayer_plane_median3(d, w, hh, ww, py_, px, i, x >> 1);
                }
            });
        } else if let Some(thr) = threshold {
            let sigma_k = 1.4826f64 as f32;
            // per plane: (values, 3x3 median, sigma); sigma 0 marks a plane that is left alone
            // (empty, a NaN made numpy's MAD NaN, or sigma < 1e-6)
            let stage: Vec<(Vec<f32>, Vec<f32>, f32)> = (0..4usize)
                .into_par_iter()
                .map(|q| {
                    let (py_, px) = (q >> 1, q & 1);
                    let hh = (h.saturating_sub(py_) + 1) / 2;
                    let ww = (w.saturating_sub(px) + 1) / 2;
                    let mut p = vec![0f32; hh * ww];
                    for i in 0..hh {
                        let src = (2 * i + py_) * w + px;
                        for j in 0..ww {
                            p[i * ww + j] = d[src + 2 * j];
                        }
                    }
                    if p.is_empty() {
                        return (p, Vec::new(), 0f32);
                    }
                    let med = median_filter_2d_f32(&p, hh, ww, 3);
                    let mut ad: Vec<f32> =
                        p.iter().zip(&med).map(|(a, m)| (a - m).abs()).collect();
                    if ad.iter().any(|v| v.is_nan()) {
                        return (p, med, 0f32);
                    }
                    let mad = median_inplace(&mut ad);
                    let sigma = mad * sigma_k;
                    if sigma < 1e-6 {
                        return (p, med, 0f32);
                    }
                    (p, med, sigma)
                })
                .collect();
            let wws = [(w + 1) / 2, w / 2];
            // excess over the plane median in units of that plane's sigma, on the mosaic grid
            let zmos: Option<Vec<f32>> = star_support.map(|_| {
                let mut z = vec![0f32; h * w];
                par_rows(&mut z, w, |y, row| {
                    let py_ = y & 1;
                    let i = y >> 1;
                    for x in 0..w {
                        let (p, med, sigma) = &stage[py_ * 2 + (x & 1)];
                        if *sigma > 0f32 {
                            let k = i * wws[x & 1] + (x >> 1);
                            row[x] = (p[k] - med[k]) / *sigma;
                        }
                    }
                });
                z
            });
            par_rows(&mut out, w, |y, row| {
                let py_ = y & 1;
                let i = y >> 1;
                for x in 0..w {
                    let px = x & 1;
                    let (p, med, sigma) = &stage[py_ * 2 + px];
                    let k = i * wws[px] + (x >> 1);
                    let v = p[k];
                    row[x] = v;
                    if *sigma > 0f32 && v - med[k] > thr * *sigma {
                        let star = match (&zmos, star_support) {
                            (Some(z), Some(sup)) => {
                                let mut near = f32::NEG_INFINITY;
                                if y > 0 { near = near.max(z[(y - 1) * w + x]); }
                                if y + 1 < h { near = near.max(z[(y + 1) * w + x]); }
                                if x > 0 { near = near.max(z[y * w + x - 1]); }
                                if x + 1 < w { near = near.max(z[y * w + x + 1]); }
                                near > sup
                            }
                            _ => false,
                        };
                        if !star {
                            row[x] = med[k];
                        }
                    }
                }
            });
        }
        out
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), out)
        .unwrap()
        .into_pyarray(py))
}

/// Reused diff-plane buffers for `hot_pixel_bayer_inplace` (fully overwritten each
/// call, so a reused one needs no clearing; a fresh 25 MB one is zero-filled by the
/// OS on first touch).
pub(crate) static HOTPIX_POOL: std::sync::OnceLock<std::sync::Mutex<Vec<Vec<f32>>>> = std::sync::OnceLock::new();

pub(crate) fn hotpix_take(n: usize) -> Vec<f32> {
    let pool = HOTPIX_POOL.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    if let Ok(mut p) = pool.lock() {
        if let Some(i) = p.iter().position(|v| v.len() == n) {
            return p.swap_remove(i);
        }
    }
    let mut v = Vec::with_capacity(n);
    // SAFETY: every element is written by pass 1 before it is read
    #[allow(clippy::uninit_vec)]
    unsafe {
        v.set_len(n)
    };
    v
}

pub(crate) fn hotpix_give(v: Vec<f32>) {
    let pool = HOTPIX_POOL.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    if let Ok(mut p) = pool.lock() {
        if p.len() < 4 {
            p.push(v);
        }
    }
}

/// A raw output pointer handed to rayon tasks that each write a disjoint region of the
/// buffer it points into (one row of one plane, or one output row), which the buffer's
/// owner keeps alive until the parallel section ends.
#[derive(Clone, Copy)]
pub(crate) struct HpPtr(pub(crate) *mut f32);

// SAFETY: only used for the disjoint, owner-outlived writes described above.
unsafe impl Send for HpPtr {}

// SAFETY: as for `Send`; no task reads through the pointer or writes outside its own region.
unsafe impl Sync for HpPtr {}

/// |diff| histogram resolution: buckets are the top 12 bits of the non-negative f32
/// pattern (exponent + 3 mantissa bits), so bucket order is value order.
pub(crate) const HP_SHIFT: u32 = 19;

pub(crate) const HP_NB: usize = 1 << (31 - HP_SHIFT);

pub(crate) const HP_COPIES: usize = 4;

/// 3x3 median at plane position (i, j) of the (py, px) Bayer sub-plane, computed exactly
/// as `median_filter_2d_f32(plane, hh, ww, 3)` does there: the median9 network in the
/// interior, reflect + comparator sort on the plane's border rows and columns.
#[inline]
pub(crate) fn bayer_plane_median3_as_filter(data: &[f32], w: usize, hh: usize, ww: usize, py: usize,
                                 px: usize, i: usize, j: usize) -> f32 {
    let mut win = [0f32; 9];
    if i >= 1 && i + 1 < hh && j >= 1 && j + 1 < ww {
        let mut k = 0;
        for ii in i - 1..=i + 1 {
            let r = (2 * ii + py) * w + px;
            for jj in j - 1..=j + 1 {
                win[k] = data[r + 2 * jj];
                k += 1;
            }
        }
        return median9(&mut win);
    }
    let mut k = 0;
    for di in -1isize..=1 {
        let ii = reflect_idx(i as isize + di, hh);
        for dj in -1isize..=1 {
            let jj = reflect_idx(j as isize + dj, ww);
            win[k] = data[(2 * ii + py) * w + 2 * jj + px];
            k += 1;
        }
    }
    win.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    win[4]
}

/// `hot_pixel_bayer(data, threshold=..., star_support=...)` (statistical mode) done in
/// place, with the same result bit for bit, in two streaming passes instead of ~10:
///
/// 1. per plane row (all four planes of a row chunk at once, so each mosaic row is read
///    from DRAM once): the 3x3 plane median (the same median9 network / border sort as
///    `median_filter_2d_f32`), `diff = value - median` (the f32 the old kernel formed for
///    its MAD, its threshold test and its star-support z) into a pooled plane-layout
///    buffer, and a histogram of |diff| by its top f32 bits;
/// 2. the histogram says which bucket(s) hold the median |diff| (the MAD), so one pass
///    over the diffs collects just those buckets' values -- sorted, they give the exact
///    order statistics `median_inplace` returned -- and every pixel whose diff exceeds the
///    threshold at the bucket's lower edge (a superset: f32 multiplication by a positive
///    constant is monotonic).
///
/// The candidates then get the exact test (`diff > thr * sigma`), the star-support test
/// (neighbour z = its diff / its plane's sigma, as before) and, if flagged, their plane
/// median recomputed from the untouched input; the replacements are written last.
/// Returns the number of pixels replaced.
#[pyfunction]
#[pyo3(signature = (data, threshold, star_support=None))]
pub(crate) fn hot_pixel_bayer_inplace<'py>(
    py: Python<'py>,
    mut data: numpy::PyReadwriteArray2<'py, f32>,
    threshold: f32,
    star_support: Option<f32>,
) -> PyResult<usize> {
    let (h, w) = {
        let s = data.as_array();
        if !s.is_standard_layout() {
            return Err(pyo3::exceptions::PyValueError::new_err("data must be C-contiguous"));
        }
        (s.shape()[0], s.shape()[1])
    };
    let d: &mut [f32] = data
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("data must be contiguous"))?;
    let thr = threshold;
    let n_fixed = py.detach(|| {
        let sigma_k = 1.4826f64 as f32;
        let hhs = [(h + 1) / 2, h / 2];
        let wws = [(w + 1) / 2, w / 2];
        let dims: [(usize, usize); 4] = [(hhs[0], wws[0]), (hhs[0], wws[1]), (hhs[1], wws[0]), (hhs[1], wws[1])];
        let mut offs = [0usize; 4];
        for q in 1..4 {
            offs[q] = offs[q - 1] + dims[q - 1].0 * dims[q - 1].1;
        }
        let mut diff = hotpix_take(h * w);
        let dp = HpPtr(diff.as_mut_ptr());
        let src: &[f32] = &*d;

        // ---- pass 1: median, diff, histogram ----
        let nrows = hhs[0];
        let nt = rayon::current_num_threads().max(1);
        let chunk = nrows.div_ceil(4 * nt).max(8);
        let nchunks = nrows.div_ceil(chunk);
        let (hist, nan) = (0..nchunks)
            .into_par_iter()
            .map(|c| {
                let dp = dp;
                // HP_COPIES interleaved sub-histograms per plane: neighbouring pixels mostly
                // fall in the same bucket, and a single counter would serialise on its own
                // increments
                let mut hist = vec![0u32; 4 * HP_COPIES * HP_NB];
                let mut nan = [false; 4];
                let i0 = c * chunk;
                let i1 = (i0 + chunk).min(nrows);
                // per plane, the last three plane rows deinterleaved from the mosaic (slot r % 3)
                let mut ring: [Vec<f32>; 4] = Default::default();
                let mut ring_row = [[usize::MAX; 3]; 4];
                let mut medbuf: Vec<f32> = vec![0f32; wws[0]];
                for i in i0..i1 {
                    for q in 0..4 {
                        let (py_, px) = (q >> 1, q & 1);
                        let (hh, ww) = dims[q];
                        if ww == 0 || i >= hh {
                            continue;
                        }
                        {
                            let rg = &mut ring[q];
                            if rg.len() != 3 * ww {
                                rg.resize(3 * ww, 0f32);
                            }
                            let lo = if i >= 1 { i - 1 } else { i };
                            let hi = (i + 1).min(hh - 1);
                            for r in lo..=hi {
                                let slot = r % 3;
                                if ring_row[q][slot] != r {
                                    let s0 = (2 * r + py_) * w + px;
                                    for (j, v) in rg[slot * ww..(slot + 1) * ww].iter_mut().enumerate() {
                                        *v = src[s0 + 2 * j];
                                    }
                                    ring_row[q][slot] = r;
                                }
                            }
                        }
                        let rg = &ring[q];
                        let cur = (i % 3) * ww;
                        let med = &mut medbuf[..ww];
                        if i >= 1 && i + 1 < hh && ww >= 3 {
                            median3_interior(rg, ((i - 1) % 3) * ww, cur, ((i + 1) % 3) * ww, ww, med);
                            med[0] = bayer_plane_median3_as_filter(src, w, hh, ww, py_, px, i, 0);
                            med[ww - 1] =
                                bayer_plane_median3_as_filter(src, w, hh, ww, py_, px, i, ww - 1);
                        } else {
                            for (j, m) in med.iter_mut().enumerate() {
                                *m = bayer_plane_median3_as_filter(src, w, hh, ww, py_, px, i, j);
                            }
                        }
                        // SAFETY: plane q row i is written only by the chunk owning row i
                        let out = unsafe {
                            std::slice::from_raw_parts_mut(dp.0.add(offs[q] + i * ww), ww)
                        };
                        let p = &rg[cur..cur + ww];
                        let hq = &mut hist[q * HP_COPIES * HP_NB..(q + 1) * HP_COPIES * HP_NB];
                        let mut any_nan = false;
                        for j in 0..ww {
                            let dv = p[j] - med[j];
                            out[j] = dv;
                            any_nan |= dv.is_nan();
                            hq[(j & (HP_COPIES - 1)) * HP_NB
                                + ((dv.to_bits() & 0x7FFF_FFFF) >> HP_SHIFT) as usize] += 1;
                        }
                        nan[q] |= any_nan;
                    }
                }
                let mut h1 = vec![0u32; 4 * HP_NB];
                for q in 0..4 {
                    for k in 0..HP_COPIES {
                        let src_h = &hist[(q * HP_COPIES + k) * HP_NB..(q * HP_COPIES + k + 1) * HP_NB];
                        for (a, b) in h1[q * HP_NB..(q + 1) * HP_NB].iter_mut().zip(src_h) {
                            *a += *b;
                        }
                    }
                }
                (h1, nan)
            })
            .reduce(
                || (vec![0u32; 4 * HP_NB], [false; 4]),
                |(mut a, mut na), (b, nb)| {
                    for (x, y) in a.iter_mut().zip(&b) {
                        *x += *y;
                    }
                    for q in 0..4 {
                        na[q] |= nb[q];
                    }
                    (a, na)
                },
            );
        let diff_ro: &[f32] = &diff;

        // ---- MAD bucket(s) and candidate bound per plane ----
        // (ranks, lowest bucket, highest bucket, count below the lowest, candidate bound)
        let mut plan: [Option<(usize, usize, usize, u64, f32)>; 4] = [None; 4];
        for q in 0..4 {
            let n = dims[q].0 * dims[q].1;
            if n == 0 || nan[q] {
                continue;
            }
            let hq = &hist[q * HP_NB..(q + 1) * HP_NB];
            let mid = n / 2;
            let r_lo = if n % 2 == 1 { mid } else { mid - 1 };
            let (mut b_lo, mut b_hi) = (usize::MAX, usize::MAX);
            let mut below_lo = 0u64;
            let mut cum = 0u64;
            for (b, &c) in hq.iter().enumerate() {
                let next = cum + c as u64;
                if b_lo == usize::MAX && (r_lo as u64) < next {
                    b_lo = b;
                    below_lo = cum;
                }
                if (mid as u64) < next {
                    b_hi = b;
                    break;
                }
                cum = next;
            }
            // the largest value bucket b_hi can hold: a plane whose sigma is < 1e-6 even
            // then is left alone, as before
            let top = f32::from_bits((((b_hi as u32) + 1) << HP_SHIFT) - 1);
            if b_hi + 1 < HP_NB && !(top * sigma_k >= 1e-6) {
                continue;
            }
            let edge = f32::from_bits((b_lo as u32) << HP_SHIFT);
            // superset bound; a non-positive threshold would reverse the monotonicity
            let cthr = if thr > 0f32 { thr * (edge * sigma_k) } else { f32::NEG_INFINITY };
            plan[q] = Some((mid, b_lo, b_hi, below_lo, cthr));
        }

        // ---- pass 2: the MAD buckets' values and the candidates ----
        let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
        const CH2: usize = 1 << 16;
        for q in 0..4 {
            if plan[q].is_some() {
                let n = dims[q].0 * dims[q].1;
                let mut s = 0;
                while s < n {
                    tasks.push((q, s, (s + CH2).min(n)));
                    s += CH2;
                }
            }
        }
        let parts: Vec<(usize, Vec<f32>, Vec<u32>)> = tasks
            .par_iter()
            .map_init(
                || (vec![0f32; CH2], vec![0u32; CH2]),
                |(vb, cb), &(q, s, e)| {
                    let (_, b_lo, b_hi, _, cthr) = plan[q].unwrap();
                    let (blo, span) = (b_lo as u32, (b_hi - b_lo) as u32);
                    let sl = &diff_ro[offs[q] + s..offs[q] + e];
                    // branch-free compaction: write every element, advance only on a hit
                    let (mut nv, mut nc) = (0usize, 0usize);
                    for (k, &v) in sl.iter().enumerate() {
                        let b = (v.to_bits() & 0x7FFF_FFFF) >> HP_SHIFT;
                        vb[nv] = v.abs();
                        nv += (b.wrapping_sub(blo) <= span) as usize;
                        cb[nc] = (s + k) as u32;
                        nc += (v > cthr) as usize;
                    }
                    (q, vb[..nv].to_vec(), cb[..nc].to_vec())
                },
            )
            .collect();
        let mut sigma = [0f32; 4];
        let mut vals_q: [Vec<f32>; 4] = Default::default();
        let mut cands: Vec<(usize, u32)> = Vec::new();
        for (q, v, c) in parts {
            vals_q[q].extend_from_slice(&v);
            cands.extend(c.into_iter().map(|k| (q, k)));
        }
        for q in 0..4 {
            if let Some((mid, _, _, below, _)) = plan[q] {
                let n = dims[q].0 * dims[q].1;
                let v = &mut vals_q[q];
                // the same order statistics `median_inplace` read off (select, then the
                // largest of the left part for an even count)
                let r = mid - below as usize;
                let (left, &mut m, _) = v.select_nth_unstable_by(r, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mad = if n % 2 == 1 {
                    m
                } else {
                    let lo = left.iter().fold(f32::NEG_INFINITY, |acc, &x| if x > acc { x } else { acc });
                    0.5 * (lo + m)
                };
                let s = mad * sigma_k;
                if s >= 1e-6 {
                    sigma[q] = s;
                }
            }
        }

        // ---- exact test, star support, replacement value (from the untouched input) ----
        let z = |yy: usize, xx: usize| -> f32 {
            let q = (yy & 1) * 2 + (xx & 1);
            if sigma[q] > 0f32 {
                diff_ro[offs[q] + (yy >> 1) * dims[q].1 + (xx >> 1)] / sigma[q]
            } else {
                0f32
            }
        };
        let fixes: Vec<(usize, f32)> = cands
            .par_iter()
            .filter_map(|&(q, k)| {
                let s = sigma[q];
                let v = diff_ro[offs[q] + k as usize];
                if !(s > 0f32 && v > thr * s) {
                    return None;
                }
                let (hh, ww) = dims[q];
                let (i, j) = (k as usize / ww, k as usize % ww);
                let (py_, px) = (q >> 1, q & 1);
                let (y, x) = (2 * i + py_, 2 * j + px);
                if let Some(sup) = star_support {
                    let mut near = f32::NEG_INFINITY;
                    if y > 0 { near = near.max(z(y - 1, x)); }
                    if y + 1 < h { near = near.max(z(y + 1, x)); }
                    if x > 0 { near = near.max(z(y, x - 1)); }
                    if x + 1 < w { near = near.max(z(y, x + 1)); }
                    if near > sup {
                        return None;
                    }
                }
                Some((y * w + x, bayer_plane_median3_as_filter(src, w, hh, ww, py_, px, i, j)))
            })
            .collect();
        hotpix_give(diff);
        for &(idx, m) in &fixes {
            d[idx] = m;
        }
        fixes.len()
    });
    Ok(n_fixed)
}

/// Mean of the 4th and 5th smallest of 8 (no NaN): Knuth's optimal 19-comparator
/// sorting network, branchless min/max. Identical to sorting and averaging.
#[inline(always)]
pub(crate) fn med8(mut a: [f32; 8]) -> f32 {
    #[inline(always)]
    fn cs(a: &mut [f32; 8], i: usize, j: usize) {
        let (x, y) = (a[i], a[j]);
        a[i] = x.min(y);
        a[j] = x.max(y);
    }
    cs(&mut a, 0, 2); cs(&mut a, 1, 3); cs(&mut a, 4, 6); cs(&mut a, 5, 7);
    cs(&mut a, 0, 4); cs(&mut a, 1, 5); cs(&mut a, 2, 6); cs(&mut a, 3, 7);
    cs(&mut a, 0, 1); cs(&mut a, 2, 3); cs(&mut a, 4, 5); cs(&mut a, 6, 7);
    cs(&mut a, 2, 4); cs(&mut a, 3, 5);
    cs(&mut a, 1, 4); cs(&mut a, 3, 6);
    cs(&mut a, 1, 2); cs(&mut a, 3, 4); cs(&mut a, 5, 6);
    (a[3] + a[4]) * 0.5f32
}

/// Sharp-spike (cosmic ray / residual hot pixel) rejection on a calibrated 2-D
/// mosaic, before debayering spreads a spike over its neighbours. Returns the
/// cleaned copy and the number of pixels replaced.
///
/// Per 2x2 Bayer sub-plane: `m` = median of the 8 same-plane neighbours
/// (reflect boundary on the plane, mean of the 4th/5th sorted values), `sigma`
/// = 1.4826 x median |P - m| and `bg` = the median of P, both over every other
/// row and column of the plane. A pixel is replaced by
/// `m` when all hold:
///   * `v` exceeds all 8 same-plane neighbours -- a star's wing pixel has a
///     brighter one on the side of the core;
///   * `v - m > k * sigma` -- significant;
///   * `v - m > contrast * max(m - bg, sigma)` -- sharp: a PSF-sampled star
///     lifts its same-plane neighbours (2 px away) to a large fraction of its
///     peak, a particle hit does not;
///   * fewer than 3 of the 4 adjacent mosaic pixels (other planes, 1 px away)
///     are above `support_frac * (v - bg)` over their own plane median -- a
///     star, even an undersampled one, lifts nearly all of them.
/// Unlike `hot_pixel_bayer`'s star-support rule, a 2-pixel hit (two adjacent
/// pixels, each lifting the other) is still caught. All f32, numpy's order;
/// bit-identical to `debayer._spike_reject_bayer_numpy`.
#[pyfunction]
#[pyo3(signature = (data, k, contrast, support_frac))]
pub(crate) fn spike_reject_bayer<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    k: f32,
    contrast: f32,
    support_frac: f32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, usize)> {
    let arr = data.as_array();
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    let d: &[f32] = arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("data must be contiguous"))?;

    let (out, n) = py.detach(|| {
        let sigma_k = 1.4826f64 as f32;
        // the 8 same-plane neighbours of mosaic pixel (y, x), reflect boundary on the plane
        let neigh = |y: usize, x: usize| -> [f32; 8] {
            let (py_, px) = (y & 1, x & 1);
            let hh = (h - py_ + 1) / 2;
            let ww = (w - px + 1) / 2;
            let (i, j) = ((y >> 1) as isize, (x >> 1) as isize);
            let mut win = [0f32; 8];
            let mut c = 0;
            for di in -1isize..=1 {
                let ii = reflect_idx(i + di, hh);
                for dj in -1isize..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    win[c] = d[(2 * ii + py_) * w + 2 * reflect_idx(j + dj, ww) + px];
                    c += 1;
                }
            }
            win
        };
        // per plane: (sigma, bg); sigma 0 = leave the plane alone. Statistics from
        // every other row/column of the plane: a quarter of the median work, and
        // they only set the threshold scale.
        let stage: Vec<(f32, f32)> = (0..4usize)
            .into_par_iter()
            .map(|q| {
                let (py_, px) = (q >> 1, q & 1);
                let hh = (h.saturating_sub(py_) + 1) / 2;
                let ww = (w.saturating_sub(px) + 1) / 2;
                if hh < 2 || ww < 2 {
                    return (0f32, 0f32);
                }
                let mut nan = false;
                for i in 0..hh {
                    let r = (2 * i + py_) * w + px;
                    for j in 0..ww {
                        if d[r + 2 * j].is_nan() {
                            nan = true;
                        }
                    }
                }
                if nan {
                    return (0f32, 0f32);
                }
                let mut ad = Vec::with_capacity(((hh + 1) / 2) * ((ww + 1) / 2));
                let mut sub = Vec::with_capacity(ad.capacity());
                for i in (0..hh).step_by(2) {
                    for j in (0..ww).step_by(2) {
                        let (y, x) = (2 * i + py_, 2 * j + px);
                        let v = d[y * w + x];
                        ad.push((v - med8(neigh(y, x))).abs());
                        sub.push(v);
                    }
                }
                let sigma = median_inplace(&mut ad) * sigma_k;
                let bg = median_inplace(&mut sub);
                if !(sigma >= 1e-6) {
                    return (0f32, bg);
                }
                (sigma, bg)
            })
            .collect();
        let mut out = d.to_vec();
        let n: usize = out
            .par_chunks_mut(w)
            .enumerate()
            .map(|(y, row)| {
                let mut cnt = 0usize;
                for x in 0..w {
                    let (sigma, bg) = stage[(y & 1) * 2 + (x & 1)];
                    if sigma <= 0f32 {
                        continue;
                    }
                    let v = d[y * w + x];
                    let win = neigh(y, x);
                    // m >= min of the 8, so v - min <= k*sigma rules the pixel out exactly;
                    // and a hit is the brightest of its plane neighbourhood (a star's wing
                    // pixel has a brighter same-plane neighbour on the side of the core)
                    let mut lo = win[0];
                    let mut hi = win[0];
                    for &u in &win[1..] {
                        if u < lo {
                            lo = u;
                        }
                        if u > hi {
                            hi = u;
                        }
                    }
                    if !(v - lo > k * sigma) || !(v > hi) {
                        continue;
                    }
                    let m = med8(win);
                    let exc = v - m;
                    if !(exc > k * sigma) {
                        continue;
                    }
                    let lift = m - bg;
                    let denom = if lift > sigma { lift } else { sigma };
                    if !(exc > contrast * denom) {
                        continue;
                    }
                    let lim = support_frac * (v - bg);
                    let mut up = 0;
                    let mut nb = |yy: usize, xx: usize| {
                        let bgn = stage[(yy & 1) * 2 + (xx & 1)].1;
                        if d[yy * w + xx] - bgn > lim {
                            up += 1;
                        }
                    };
                    if y > 0 { nb(y - 1, x); }
                    if y + 1 < h { nb(y + 1, x); }
                    if x > 0 { nb(y, x - 1); }
                    if x + 1 < w { nb(y, x + 1); }
                    if up < 3 {
                        row[x] = m;
                        cnt += 1;
                    }
                }
                cnt
            })
            .sum();
        (out, n)
    });
    Ok((
        numpy::ndarray::Array2::from_shape_vec((h, w), out)
            .unwrap()
            .into_pyarray(py),
        n,
    ))
}

/// Everything `hot_pixel_rgb` decides, before any pixel is written: the luminance,
/// and (when something exceeded the cut) each flagged pixel's index and its
/// replacement. f32 arithmetic in numpy's order, box mean in f64.
pub(crate) enum HotRgb {
    /// MAD-based sigma < 1e-6: numpy falls back to `np.std`, which the caller runs.
    Degenerate,
    /// Nothing flagged (or a NaN made the MAD NaN, so nothing passes): the input is the answer.
    Clean(Vec<f32>),
    /// Luminance of the *input*, plus (pixel index, replacement rgb).
    Hot(Vec<f32>, Vec<(usize, [f32; 3])>),
}

pub(crate) fn hot_rgb_core(f: &[f32], h: usize, w: usize, threshold: f64) -> HotRgb {
    let (kr, kg, kb) = (0.299f64 as f32, 0.587f64 as f32, 0.114f64 as f32);
    let mut lum = vec![0f32; h * w];
    par_rows(&mut lum, w, |y, row| {
        for x in 0..w {
            let b = (y * w + x) * 3;
            row[x] = kr * f[b] + kg * f[b + 1] + kb * f[b + 2];
        }
    });
    let med = median_filter_2d_f32(&lum, h, w, 3);
    let mut ad: Vec<f32> = lum.par_iter().zip(med.par_iter()).map(|(a, m)| (a - m).abs()).collect();
    if ad.par_iter().any(|v| v.is_nan()) {
        return HotRgb::Clean(lum);
    }
    let mad = median_inplace(&mut ad);
    drop(ad);
    let sigma = mad as f64 * 1.4826;
    if sigma < 1e-6 {
        return HotRgb::Degenerate;
    }
    let cut = (threshold * sigma) as f32;
    let hot: Vec<usize> = (0..h * w)
        .into_par_iter()
        .filter(|&i| lum[i] - med[i] > cut)
        .collect();
    if hot.is_empty() {
        return HotRgb::Clean(lum);
    }
    let repl: Vec<(usize, [f32; 3])> = hot
        .par_iter()
        .map(|&i| {
            let (y, x) = (i / w, i % w);
            let mut v = [0f32; 3];
            for ch in 0..3usize {
                let mut acc = 0.0f64;
                for dy in -1isize..=1 {
                    let yy = reflect_idx(y as isize + dy, h);
                    for dx in -1isize..=1 {
                        let xx = reflect_idx(x as isize + dx, w);
                        acc += f[(yy * w + xx) * 3 + ch] as f64;
                    }
                }
                v[ch] = (acc / 9.0) as f32;
            }
            (i, v)
        })
        .collect();
    HotRgb::Hot(lum, repl)
}

/// Write the replacements into `buf` and the luminance of each replaced pixel into `lum`.
pub(crate) fn hot_rgb_apply(buf: &mut [f32], lum: &mut [f32], repl: &[(usize, [f32; 3])]) {
    let (kr, kg, kb) = (0.299f64 as f32, 0.587f64 as f32, 0.114f64 as f32);
    for &(i, v) in repl {
        buf[i * 3] = v[0];
        buf[i * 3 + 1] = v[1];
        buf[i * 3 + 2] = v[2];
        lum[i] = kr * v[0] + kg * v[1] + kb * v[2];
    }
}

pub(crate) fn hot_rgb_check(rgb: &numpy::ndarray::ArrayView3<'_, f32>) -> PyResult<(usize, usize)> {
    let s = rgb.shape();
    if s[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("expected 3 channels"));
    }
    Ok((s[0], s[1]))
}

/// RGB hot-pixel removal on luminance, fused: luma, its 3x3 median, the MAD
/// threshold and the sparse 3x3 box-mean replacement (all three channels of a
/// flagged pixel), returning a new array.
///
/// Returns None when the MAD-based sigma is degenerate (< 1e-6: the numpy path
/// then falls back to `np.std`, which the caller runs). Otherwise
/// `(fixed_or_None, lum)`: `fixed` is None when nothing was flagged (the input
/// is already the answer), `lum` is the luminance of the returned image.
/// Bit-identical to `_fix_hot_rgb_impl`.
#[pyfunction]
pub(crate) fn hot_pixel_rgb<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f32>,
    threshold: f64,
) -> PyResult<Option<(Option<Bound<'py, PyArray3<f32>>>, Bound<'py, PyArray2<f32>>)>> {
    let arr = rgb.as_array();
    let (h, w) = hot_rgb_check(&arr)?;
    let f: &[f32] = arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("rgb must be contiguous"))?;
    let res = py.detach(|| match hot_rgb_core(f, h, w, threshold) {
        HotRgb::Degenerate => None,
        HotRgb::Clean(lum) => Some((None, lum)),
        HotRgb::Hot(mut lum, repl) => {
            let mut out = f.to_vec();
            hot_rgb_apply(&mut out, &mut lum, &repl);
            Some((Some(out), lum))
        }
    });
    Ok(res.map(|(out, lum)| {
        (
            out.map(|o| numpy::ndarray::Array3::from_shape_vec((h, w, 3), o).unwrap().into_pyarray(py)),
            numpy::ndarray::Array2::from_shape_vec((h, w), lum).unwrap().into_pyarray(py),
        )
    }))
}

/// `hot_pixel_rgb` that writes the replacements into `rgb` itself: only the few
/// flagged pixels are touched, so no 75 MB output array is allocated and copied.
/// Returns the luminance of the resulting image, or None for a degenerate MAD
/// (nothing is modified then).
#[pyfunction]
pub(crate) fn hot_pixel_rgb_inplace<'py>(
    py: Python<'py>,
    mut rgb: numpy::PyReadwriteArray3<'py, f32>,
    threshold: f64,
) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
    let (h, w) = hot_rgb_check(&rgb.as_array())?;
    let f = rgb
        .as_slice_mut()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("rgb must be contiguous"))?;
    let lum = py.detach(|| match hot_rgb_core(&*f, h, w, threshold) {
        HotRgb::Degenerate => None,
        HotRgb::Clean(lum) => Some(lum),
        HotRgb::Hot(mut lum, repl) => {
            hot_rgb_apply(f, &mut lum, &repl);
            Some(lum)
        }
    });
    Ok(lum.map(|l| numpy::ndarray::Array2::from_shape_vec((h, w), l).unwrap().into_pyarray(py)))
}
