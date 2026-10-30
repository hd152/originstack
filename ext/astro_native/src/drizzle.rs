//! CFA drizzle, Lanczos-3 drizzle accumulate, splat drizzle.

use crate::*;

// ---------------------------------------------------------------------------
// CFA drizzle: splat one frame's measured Bayer samples (src/cfa_drizzle.py)
// ---------------------------------------------------------------------------
//
// One call folds a whole calibrated frame into the running drizzle
// accumulators. Every sensor pixel measured exactly one colour (its Bayer
// parity picks the channel via `chan_table[(y&1)*2 + (x&1)]`); its sample is
// (1) mapped onto the output grid through the frame's affine, (2) tested
// against the reference stack, and (3) deposited as a square drop of half-side
// `h` by exact area overlap into `num` (weighted value), `den` (weight) and
// `cov` (unweighted coverage), each (out_h, out_w, 3) f64.
//
// The numpy path (`_frame_splat_numpy`) walks the same three steps with
// per-channel bincounts -- ~6 full-size bincount passes per neighbour tap per
// channel, which is what made a 148-frame run take 407 s. Here the work is one
// pass over the samples, parallelised over bands of OUTPUT rows: every thread
// owns its rows exclusively (a drop that straddles a band edge is deposited
// by each band only into its own rows), so there are no atomics and no
// per-thread accumulator copies (three full-size f64 planes per thread would
// be gigabytes at sensor size).

#[inline]
pub(crate) fn bilinear_at(img: &[f32], oh: usize, ow: usize, c: usize, y: f64, x: f64) -> f64 {
    let yc = y.max(0.0).min(oh as f64 - 1.0);
    let xc = x.max(0.0).min(ow as f64 - 1.0);
    let y0 = (yc.floor() as usize).min(oh - 2);
    let x0 = (xc.floor() as usize).min(ow - 2);
    let fy = yc - y0 as f64;
    let fx = xc - x0 as f64;
    let at = |yy: usize, xx: usize| img[(yy * ow + xx) * 3 + c] as f64;
    (1.0 - fy) * ((1.0 - fx) * at(y0, x0) + fx * at(y0, x0 + 1))
        + fy * ((1.0 - fx) * at(y0 + 1, x0) + fx * at(y0 + 1, x0 + 1))
}

/// Returns (samples_considered, samples_rejected) for this frame.
#[pyfunction]
#[pyo3(signature = (rgb, reference, chan_table, minv, off, h, sig, sky, reject_sigma, signal_tol, num, den, cov))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cfa_drizzle_frame<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f32>,
    reference: PyReadonlyArray3<'py, f32>,
    chan_table: PyReadonlyArray1<'py, u8>,
    minv: PyReadonlyArray1<'py, f64>,
    off: PyReadonlyArray1<'py, f64>,
    h: f64,
    sig: PyReadonlyArray1<'py, f64>,
    sky: PyReadonlyArray1<'py, f64>,
    reject_sigma: f64,
    signal_tol: f64,
    mut num: PyReadwriteArray3<'py, f64>,
    mut den: PyReadwriteArray3<'py, f64>,
    mut cov: PyReadwriteArray3<'py, f64>,
) -> PyResult<(u64, u64)> {
    let rs = rgb.as_array().shape().to_vec();
    let os = reference.as_array().shape().to_vec();
    if rs[2] != 3 || os[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("rgb and reference must have 3 channels"));
    }
    let (sh, sw) = (rs[0], rs[1]);
    let (oh, ow) = (os[0], os[1]);
    if oh < 2 || ow < 2 || num.as_array().shape() != [oh, ow, 3]
        || den.as_array().shape() != [oh, ow, 3] || cov.as_array().shape() != [oh, ow, 3]
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "accumulators must be (out_h, out_w, 3) matching reference",
        ));
    }
    let rgb_s = rgb.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("rgb must be C-contiguous")
    })?;
    let ref_s = reference.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("reference must be C-contiguous")
    })?;
    let ct = chan_table.as_slice()?;
    let mv = minv.as_slice()?;
    let of = off.as_slice()?;
    let sg = sig.as_slice()?;
    let sk = sky.as_slice()?;
    if ct.len() != 4 || mv.len() != 4 || of.len() != 2 || sg.len() != 3 || sk.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("bad parameter vector length"));
    }
    let num_s = num.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("num must be C-contiguous")
    })?;
    let den_s = den.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("den must be C-contiguous")
    })?;
    let cov_s = cov.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("cov must be C-contiguous")
    })?;

    let (m00, m01, m10, m11) = (mv[0], mv[1], mv[2], mv[3]);
    let (off0, off1) = (of[0], of[1]);
    // Forward affine input->output is o = Minv (i - off); its inverse
    // i = M o + off maps an output band back to the input rows that can land in it.
    let det = m00 * m11 - m01 * m10;
    if det.abs() < 1e-12 {
        return Err(pyo3::exceptions::PyValueError::new_err("singular affine"));
    }
    let (i00, i01) = (m11 / det, -m01 / det);

    let k = (2.0 * h).floor() as i64 + 2;
    let norm = 1.0 / (4.0 * h * h);
    let valid_ch: [bool; 3] = [
        sg[0].is_finite() && sg[0] > 0.0,
        sg[1].is_finite() && sg[1] > 0.0,
        sg[2].is_finite() && sg[2] > 0.0,
    ];
    let fw: [f64; 3] = [
        if valid_ch[0] { 1.0 / (sg[0] * sg[0]) } else { 0.0 },
        if valid_ch[1] { 1.0 / (sg[1] * sg[1]) } else { 0.0 },
        if valid_ch[2] { 1.0 / (sg[2] * sg[2]) } else { 0.0 },
    ];

    const BAND: usize = 16;
    let row_len = ow * 3;
    let considered = std::sync::atomic::AtomicU64::new(0);
    let rejected = std::sync::atomic::AtomicU64::new(0);

    py.detach(|| {
        num_s
            .par_chunks_mut(BAND * row_len)
            .zip(den_s.par_chunks_mut(BAND * row_len))
            .zip(cov_s.par_chunks_mut(BAND * row_len))
            .enumerate()
            .for_each(|(band, ((nb, db), cb))| {
                let r0 = band * BAND;
                let r1 = (r0 + nb.len() / row_len).min(oh);
                // Output rows this band may receive (a drop reaches h beyond its centre).
                let lo_o = r0 as f64 - h - 0.5;
                let hi_o = r1 as f64 + h - 0.5;

                // Bounding input rows: map the band's 4 corners back to the sensor.
                let mut imin = f64::INFINITY;
                let mut imax = f64::NEG_INFINITY;
                for &oy in &[lo_o, hi_o] {
                    for &ox in &[-h - 0.5, ow as f64 + h - 0.5] {
                        let iy = i00 * oy + i01 * ox + off0;
                        if iy < imin { imin = iy; }
                        if iy > imax { imax = iy; }
                    }
                }
                let y_start = (imin.floor() as i64 - 1).max(0) as usize;
                let y_end = ((imax.ceil() as i64 + 2).max(0) as usize).min(sh);

                let mut n_cons = 0u64;
                let mut n_rej = 0u64;
                for iy in y_start..y_end {
                    let dy = iy as f64 - off0;
                    // oy(ix) = m00*dy + m01*(ix - off1): solve for the ix span that
                    // lands inside [lo_o, hi_o]; that keeps this pass O(samples in band).
                    let base = m00 * dy;
                    let (x_lo, x_hi) = if m01.abs() < 1e-12 {
                        if base >= lo_o && base <= hi_o { (0usize, sw) } else { continue; }
                    } else {
                        let a = (lo_o - base) / m01 + off1;
                        let b = (hi_o - base) / m01 + off1;
                        let (u, v) = if a < b { (a, b) } else { (b, a) };
                        (
                            (u.floor() as i64 - 1).max(0) as usize,
                            ((v.ceil() as i64 + 2).max(0) as usize).min(sw),
                        )
                    };
                    for ix in x_lo..x_hi {
                        let dx = ix as f64 - off1;
                        let oy = m00 * dy + m01 * dx;
                        let ox = m10 * dy + m11 * dx;
                        // same admission test as the numpy path
                        if !(oy > -h - 0.5 && oy < oh as f64 + h - 0.5
                            && ox > -h - 0.5 && ox < ow as f64 + h - 0.5)
                        {
                            continue;
                        }
                        // band ownership: skip drops that cannot touch our rows
                        if oy + h <= r0 as f64 - 0.5 || oy - h >= r1 as f64 - 0.5 {
                            continue;
                        }
                        let c = ct[(iy & 1) * 2 + (ix & 1)] as usize;
                        if c > 2 {
                            continue;
                        }
                        let v = rgb_s[(iy * sw + ix) * 3 + c] as f64;
                        let r_at = bilinear_at(ref_s, oh, ow, c, oy, ox);
                        let resid = v - r_at;
                        // Count each sample once: only the band that owns its centre row.
                        let owner = {
                            let cy = oy.round().max(0.0).min(oh as f64 - 1.0) as usize;
                            cy >= r0 && cy < r1
                        };
                        if !valid_ch[c] {
                            continue;
                        }
                        let s = sg[c];
                        let sig_sig = (r_at - sk[c]).max(0.0) * signal_tol;
                        let tol = reject_sigma * (s * s + sig_sig * sig_sig).sqrt();
                        let keep = resid.abs() <= tol;
                        if owner {
                            n_cons += 1;
                            if !keep { n_rej += 1; }
                        }
                        if !keep {
                            continue;
                        }
                        let y0 = (oy - h + 0.5).floor() as i64;
                        let x0 = (ox - h + 0.5).floor() as i64;
                        for a in 0..k {
                            let qy = y0 + a;
                            if qy < r0 as i64 || qy >= r1 as i64 { continue; }
                            let qyf = qy as f64;
                            let ovy = (oy + h).min(qyf + 0.5) - (oy - h).max(qyf - 0.5);
                            if ovy <= 0.0 { continue; }
                            for b in 0..k {
                                let qx = x0 + b;
                                if qx < 0 || qx >= ow as i64 { continue; }
                                let qxf = qx as f64;
                                let ovx = (ox + h).min(qxf + 0.5) - (ox - h).max(qxf - 0.5);
                                if ovx <= 0.0 { continue; }
                                let wgt = ovy * ovx * norm;
                                let idx = ((qy as usize - r0) * ow + qx as usize) * 3 + c;
                                nb[idx] += wgt * v * fw[c];
                                db[idx] += wgt * fw[c];
                                cb[idx] += wgt;
                            }
                        }
                    }
                }
                considered.fetch_add(n_cons, std::sync::atomic::Ordering::Relaxed);
                rejected.fetch_add(n_rej, std::sync::atomic::Ordering::Relaxed);
            });
    });

    Ok((
        considered.load(std::sync::atomic::Ordering::Relaxed),
        rejected.load(std::sync::atomic::Ordering::Relaxed),
    ))
}

// ---------------------------------------------------------------------------
// Drizzle: fused warp + weight + accumulate (src/stacking.py run_stacking_phase)
// ---------------------------------------------------------------------------
//
// The Python drizzle loop warped each frame into a fresh (out_h, out_w, 3) f32
// array, multiplied it by the frame weight, and added it into a shared f64
// accumulator under a lock -- three full-size passes and a serialisation point
// per frame (the add alone was ~150 ms of a ~435 ms rotated frame at 2x scale,
// one thread at a time). This kernel does the same arithmetic per output row and
// adds straight into the accumulator: rows are the parallel unit, so each thread
// owns its accumulator rows and no lock or temporary array exists.
//
// Bit-identical to the old path: the row is produced by `lanczos3_row_flat` (the
// very code `warp_affine_lanczos3` runs), then `v * (w as f32)` in f32 (numpy
// in-place `resampled *= w`), then `acc += f64(v)`. With `pixfrac < 1` the tent
// weight numpy built from seven full-size f64 temporaries is computed per pixel
// with the same float64 operations in the same order, and `wmap += pfw`.
#[pyfunction]
#[pyo3(signature = (acc, data, mat, off, weight, pixfrac, wmap=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn drizzle_accumulate_lanczos3<'py>(
    py: Python<'py>,
    mut acc: PyReadwriteArray3<'py, f64>,
    data: PyReadonlyArray3<'py, f32>,
    mat: [f64; 4],
    off: [f64; 2],
    weight: f64,
    pixfrac: f64,
    wmap: Option<PyReadwriteArray3<'py, f64>>,
) -> PyResult<()> {
    let s = data.as_array().shape().to_vec();
    let (h, w, c) = (s[0], s[1], s[2]);
    let acs = acc.as_array().shape().to_vec();
    if c != 3 || acs.len() != 3 || acs[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("data and acc must have 3 channels"));
    }
    let (out_h, out_w) = (acs[0], acs[1]);
    let img = data.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("data must be C-contiguous")
    })?;
    let acc_s = acc.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("acc must be C-contiguous")
    })?;
    let use_pf = pixfrac < 1.0 - 1e-9;
    let mut wmap = wmap;
    let wm_s: Option<&mut [f64]> = match wmap.as_mut() {
        Some(m) => {
            if m.as_array().shape() != [out_h, out_w, 1] {
                return Err(pyo3::exceptions::PyValueError::new_err("wmap must be (out_h, out_w, 1)"));
            }
            Some(m.as_slice_mut().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("wmap must be C-contiguous")
            })?)
        }
        None => None,
    };
    if use_pf && wm_s.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err("pixfrac < 1 needs a wmap"));
    }
    let (m00, m01, m10, m11) = (mat[0], mat[1], mat[2], mat[3]);
    let (o0, o1) = (off[0], off[1]);
    let col_tab: Option<(Vec<[f64; 6]>, Vec<isize>)> = if m01 == 0.0 && m10 == 0.0 {
        let mut wxs = vec![[0f64; 6]; out_w];
        let mut bxs = vec![0isize; out_w];
        for ox in 0..out_w {
            let ix = m11 * ox as f64 + o1;
            let fx = ix.floor();
            lanczos6_weights(ix - fx, &mut wxs[ox]);
            bxs[ox] = fx as isize - 2;
        }
        Some((wxs, bxs))
    } else {
        None
    };
    let wf32 = weight as f32;
    let half_drop = (pixfrac / 2.0).max(1e-6);

    py.detach(|| {
        let row = |oy: usize, arow: &mut [f64], mrow: Option<&mut [f64]>, buf: &mut Vec<f32>| {
            buf.resize(out_w * 3, 0.0);
            lanczos3_row_flat(img, h, w, 3, oy, buf, [m00, m01, m10, m11], [o0, o1], &col_tab, 0.0, 0);
            if use_pf {
                let mrow = mrow.unwrap();
                for ox in 0..out_w {
                    let raw_y = m00 * oy as f64 + m01 * ox as f64 + o0;
                    let raw_x = m10 * oy as f64 + m11 * ox as f64 + o1;
                    let fy = (raw_y - raw_y.round_ties_even()).abs();
                    let fx = (raw_x - raw_x.round_ties_even()).abs();
                    let wy_ = (1.0 - fy / half_drop).max(0.0);
                    let wx_ = (1.0 - fx / half_drop).max(0.0);
                    let pfw = wy_ * wx_ * weight;
                    for ch in 0..3 {
                        arow[ox * 3 + ch] += buf[ox * 3 + ch] as f64 * pfw;
                    }
                    mrow[ox] += pfw;
                }
            } else {
                for i in 0..out_w * 3 {
                    arow[i] += (buf[i] * wf32) as f64;
                }
            }
        };
        match wm_s {
            Some(wm) if use_pf => {
                acc_s
                    .par_chunks_mut(out_w * 3)
                    .zip(wm.par_chunks_mut(out_w))
                    .enumerate()
                    .for_each_init(Vec::new, |buf, (oy, (arow, mrow))| {
                        row(oy, arow, Some(mrow), buf)
                    });
            }
            _ => {
                acc_s
                    .par_chunks_mut(out_w * 3)
                    .enumerate()
                    .for_each_init(Vec::new, |buf, (oy, arow)| row(oy, arow, None, buf));
            }
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Drizzle by area-overlap splatting (--drizzle-method splat)
// ---------------------------------------------------------------------------
//
// The resampling path evaluates a 6x6-tap Lanczos gather at every OUTPUT pixel
// (4x as many as input pixels at 2x scale) and does not implement drizzle's
// pixfrac at all (it is a weight applied afterwards). This is the algorithm
// itself: every input pixel is a square drop of side `pixfrac * scale` output
// pixels (half-side `h`), mapped through the frame's affine and deposited by
// exact area overlap into `num` (weight * value, per channel) and `den`
// (weight). Work is one pass over the INPUT (~16x fewer multiply-adds than the
// gather) and parallel over bands of output rows so every thread owns its rows.
// Same drop geometry and band scheme as `cfa_drizzle_frame`; all three channels
// share one drop (a debayered frame carries a value in each).
#[pyfunction]
#[pyo3(signature = (rgb, fwd, off, h, weight, num, den))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn drizzle_splat_frame<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f32>,
    fwd: [f64; 4],
    off: [f64; 2],
    h: f64,
    weight: f64,
    mut num: PyReadwriteArray3<'py, f64>,
    mut den: PyReadwriteArray3<'py, f64>,
) -> PyResult<()> {
    let rs = rgb.as_array().shape().to_vec();
    let ns = num.as_array().shape().to_vec();
    if rs[2] != 3 || ns[2] != 3 || den.as_array().shape() != [ns[0], ns[1], 1] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rgb/num must have 3 channels and den must be (out_h, out_w, 1)",
        ));
    }
    if !(h > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err("h must be > 0"));
    }
    let (sh, sw) = (rs[0], rs[1]);
    let (oh, ow) = (ns[0], ns[1]);
    let rgb_s = rgb.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("rgb must be C-contiguous")
    })?;
    let num_s = num.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("num must be C-contiguous")
    })?;
    let den_s = den.as_slice_mut().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("den must be C-contiguous")
    })?;
    let (m00, m01, m10, m11) = (fwd[0], fwd[1], fwd[2], fwd[3]);
    let (off0, off1) = (off[0], off[1]);
    let det = m00 * m11 - m01 * m10;
    if det.abs() < 1e-12 {
        return Err(pyo3::exceptions::PyValueError::new_err("singular affine"));
    }
    let (i00, i01) = (m11 / det, -m01 / det);
    let k = (2.0 * h).floor() as i64 + 2;
    let norm = 1.0 / (4.0 * h * h);

    const BAND: usize = 16;
    py.detach(|| {
        num_s
            .par_chunks_mut(BAND * ow * 3)
            .zip(den_s.par_chunks_mut(BAND * ow))
            .enumerate()
            .for_each(|(band, (nb, db))| {
                let r0 = band * BAND;
                let r1 = (r0 + db.len() / ow).min(oh);
                let lo_o = r0 as f64 - h - 0.5;
                let hi_o = r1 as f64 + h - 0.5;
                let mut imin = f64::INFINITY;
                let mut imax = f64::NEG_INFINITY;
                for &oy in &[lo_o, hi_o] {
                    for &ox in &[-h - 0.5, ow as f64 + h - 0.5] {
                        let iy = i00 * oy + i01 * ox + off0;
                        if iy < imin { imin = iy; }
                        if iy > imax { imax = iy; }
                    }
                }
                let y_start = (imin.floor() as i64 - 1).max(0) as usize;
                let y_end = ((imax.ceil() as i64 + 2).max(0) as usize).min(sh);
                for iy in y_start..y_end {
                    let dy = iy as f64 - off0;
                    let base = m00 * dy;
                    let (x_lo, x_hi) = if m01.abs() < 1e-12 {
                        if base >= lo_o && base <= hi_o { (0usize, sw) } else { continue; }
                    } else {
                        let a = (lo_o - base) / m01 + off1;
                        let b = (hi_o - base) / m01 + off1;
                        let (u, v) = if a < b { (a, b) } else { (b, a) };
                        (
                            (u.floor() as i64 - 1).max(0) as usize,
                            ((v.ceil() as i64 + 2).max(0) as usize).min(sw),
                        )
                    };
                    for ix in x_lo..x_hi {
                        let dx = ix as f64 - off1;
                        let oy = m00 * dy + m01 * dx;
                        let ox = m10 * dy + m11 * dx;
                        if !(oy > -h - 0.5 && oy < oh as f64 + h - 0.5
                            && ox > -h - 0.5 && ox < ow as f64 + h - 0.5)
                        {
                            continue;
                        }
                        if oy + h <= r0 as f64 - 0.5 || oy - h >= r1 as f64 - 0.5 {
                            continue;
                        }
                        let px = &rgb_s[(iy * sw + ix) * 3..(iy * sw + ix) * 3 + 3];
                        if !(px[0].is_finite() && px[1].is_finite() && px[2].is_finite()) {
                            continue;
                        }
                        let (v0, v1, v2) = (px[0] as f64, px[1] as f64, px[2] as f64);
                        let y0 = (oy - h + 0.5).floor() as i64;
                        let x0 = (ox - h + 0.5).floor() as i64;
                        for a in 0..k {
                            let qy = y0 + a;
                            if qy < r0 as i64 || qy >= r1 as i64 { continue; }
                            let qyf = qy as f64;
                            let ovy = (oy + h).min(qyf + 0.5) - (oy - h).max(qyf - 0.5);
                            if ovy <= 0.0 { continue; }
                            for b in 0..k {
                                let qx = x0 + b;
                                if qx < 0 || qx >= ow as i64 { continue; }
                                let qxf = qx as f64;
                                let ovx = (ox + h).min(qxf + 0.5) - (ox - h).max(qxf - 0.5);
                                if ovx <= 0.0 { continue; }
                                let wgt = ovy * ovx * norm * weight;
                                let cell = (qy as usize - r0) * ow + qx as usize;
                                nb[cell * 3] += wgt * v0;
                                nb[cell * 3 + 1] += wgt * v1;
                                nb[cell * 3 + 2] += wgt * v2;
                                db[cell] += wgt;
                            }
                        }
                    }
                }
            });
    });
    Ok(())
}
