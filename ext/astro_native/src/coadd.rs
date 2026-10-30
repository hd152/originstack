//! Proper-coaddition (Zackay-Ofek) accumulate kernels.

use crate::*;

/// Proper coadd (src/proper_coadd.py), per frame: reject each sample against the
/// normal stack on this frame's flux scale and write the sky-subtracted frame into
/// each channel's zero-padded FFT buffer -- one pass instead of ~10 numpy passes
/// over a strided channel view. Same f32 operations in numpy's order, so the
/// buffers are bit-identical to the numpy path:
///   exp = ref_sig * f; resid = (x - sky) - exp; t = (max(exp, 0) * frac)^2 + sig2
///   bad = |resid| > k * sqrt(t);  out = bad ? exp : x - sky
/// `frame` (H, W, C), `ref_sig` (C, H, W) = reference stack minus its sky,
/// `bufs` (C, PH, PW) written at offset (pad, pad) (the rest is left as is).
/// Returns the number of replaced samples.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn proper_coadd_prep<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray3<'py, f32>,
    ref_sig: PyReadonlyArray3<'py, f32>,
    mut bufs: numpy::PyReadwriteArray3<'py, f32>,
    f: f32,
    sky: Vec<f32>,
    sig2: Vec<f32>,
    k: f32,
    frac: f32,
    pad: usize,
) -> PyResult<u64> {
    let fa = frame.as_array();
    let (h, w, c) = (fa.shape()[0], fa.shape()[1], fa.shape()[2]);
    let ra = ref_sig.as_array();
    if ra.shape() != [c, h, w] || sky.len() != c || sig2.len() != c {
        return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
    }
    let bshape = bufs.as_array().shape().to_vec();
    let (ph, pw) = (bshape[1], bshape[2]);
    if bshape[0] != c || ph < h + pad || pw < w + pad {
        return Err(pyo3::exceptions::PyValueError::new_err("buffer too small"));
    }
    let fr = fa.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("frame must be contiguous"))?;
    let rs = ra.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ref_sig must be contiguous"))?;
    let bs = bufs.as_slice_mut().map_err(|_| pyo3::exceptions::PyValueError::new_err("bufs must be contiguous"))?;
    let n = py.detach(|| {
        let mut total = 0u64;
        for (ch, plane) in bs.chunks_mut(ph * pw).enumerate() {
            let (sk, s2) = (sky[ch], sig2[ch]);
            let refc = &rs[ch * h * w..(ch + 1) * h * w];
            total += plane[pad * pw..(pad + h) * pw]
                .par_chunks_mut(pw)
                .enumerate()
                .map(|(y, row)| {
                    let mut cnt = 0u64;
                    let out = &mut row[pad..pad + w];
                    let rr = &refc[y * w..(y + 1) * w];
                    let src = &fr[y * w * c..(y + 1) * w * c];
                    for x in 0..w {
                        let v = src[x * c + ch] - sk;
                        let e = rr[x] * f;
                        let resid = v - e;
                        let mut t = if e > 0.0 { e } else { 0.0 };
                        t *= frac;
                        t = t * t;
                        t += s2;
                        if resid.abs() > k * t.sqrt() {
                            out[x] = e;
                            cnt += 1;
                        } else {
                            out[x] = v;
                        }
                    }
                    cnt
                })
                .sum::<u64>();
        }
        total
    });
    Ok(n)
}

/// Proper coadd accumulation for one frame and channel, fused:
/// `num += wnum * fm * conj(ph)` and `den += wden * |ph|^2`. Complex arrays are passed
/// as their interleaved float views (`.view(float32)` of complex64). Float32
/// accumulators (each producer thread keeps its own and they are summed once at the
/// end; the old shared float64 `proper_coadd_accum` was removed in crate 0.53.0).
/// The per-element arithmetic is f64;
/// only the running sums are stored in f32 -- 531 frames of relative precision 1e-7
/// leave ~1e-5 relative, far below the stack's own noise.
#[pyfunction]
pub(crate) fn proper_coadd_accum32<'py>(
    py: Python<'py>,
    mut num: numpy::PyReadwriteArray2<'py, f32>,
    mut den: numpy::PyReadwriteArray2<'py, f32>,
    fm: PyReadonlyArray2<'py, f32>,
    ph: PyReadonlyArray2<'py, f32>,
    wnum: f64,
    wden: f64,
) -> PyResult<()> {
    let ns = num.as_array().shape().to_vec();
    if den.as_array().shape() != [ns[0], ns[1] / 2] || fm.as_array().shape() != &ns[..]
        || ph.as_array().shape() != &ns[..] || ns[1] % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
    }
    let row = ns[1];
    let nm = num.as_slice_mut().map_err(|_| pyo3::exceptions::PyValueError::new_err("num must be contiguous"))?;
    let dn = den.as_slice_mut().map_err(|_| pyo3::exceptions::PyValueError::new_err("den must be contiguous"))?;
    let f = fm.as_slice()?;
    let p = ph.as_slice()?;
    py.detach(|| {
        nm.par_chunks_mut(row).zip(dn.par_chunks_mut(row / 2)).enumerate().for_each(|(y, (nr, dr))| {
            let fr = &f[y * row..(y + 1) * row];
            let pr = &p[y * row..(y + 1) * row];
            for i in 0..row / 2 {
                let (a, b) = (fr[2 * i] as f64, fr[2 * i + 1] as f64);
                let (c, d) = (pr[2 * i] as f64, pr[2 * i + 1] as f64);
                nr[2 * i] = (nr[2 * i] as f64 + wnum * (a * c + b * d)) as f32;
                nr[2 * i + 1] = (nr[2 * i + 1] as f64 + wnum * (b * c - a * d)) as f32;
                dr[i] = (dr[i] as f64 + wden * (c * c + d * d)) as f32;
            }
        });
    });
    Ok(())
}

/// `proper_coadd_prep` for a band of data rows `y0 .. y0 + nb` of a batch of B frames,
/// into a small buffer `band` (B, C, nb, PW): each row is written whole -- `pad`
/// zeros, the rejected and sky-subtracted samples, zeros to PW -- so the buffer can be
/// reused band after band and FFT'd while it is still in cache (the full padded planes
/// were ~43 MB per frame written once and read back once by the FFT), and the
/// reference row is read once for the whole batch. `f` has one entry per frame, `sky`
/// and `sig2` B*C (frame-major). Same f32 operations as `proper_coadd_prep`, so the
/// rows are bit-identical to that kernel's. Sequential: the combine pass runs one frame
/// batch per thread. Returns the number of replaced samples.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn proper_coadd_prep_rows<'py>(
    py: Python<'py>,
    frames: Vec<PyReadonlyArray3<'py, f32>>,
    ref_sig: PyReadonlyArray3<'py, f32>,
    mut band: numpy::PyReadwriteArray4<'py, f32>,
    y0: usize,
    f: Vec<f32>,
    sky: Vec<f32>,
    sig2: Vec<f32>,
    k: f32,
    frac: f32,
    pad: usize,
) -> PyResult<u64> {
    let ra = ref_sig.as_array();
    let (c, h, w) = (ra.shape()[0], ra.shape()[1], ra.shape()[2]);
    let nbat = frames.len();
    let bshape = band.as_array().shape().to_vec();
    let (nb, pw) = (bshape[2], bshape[3]);
    if bshape[0] != nbat || bshape[1] != c || pw < w + pad || y0 + nb > h
        || f.len() != nbat || sky.len() != nbat * c || sig2.len() != nbat * c {
        return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
    }
    let mut frs: Vec<&[f32]> = Vec::with_capacity(nbat);
    for fr in frames.iter() {
        if fr.as_array().shape() != [h, w, c] {
            return Err(pyo3::exceptions::PyValueError::new_err("frame shape mismatch"));
        }
        frs.push(fr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("frames must be contiguous"))?);
    }
    let rs = ra.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ref_sig must be contiguous"))?;
    let bs = band.as_slice_mut().map_err(|_| pyo3::exceptions::PyValueError::new_err("band must be contiguous"))?;
    let n = py.detach(|| {
        let mut cnt = 0u64;
        for r in 0..nb {
            let y = y0 + r;
            for ch in 0..c {
                let rr = &rs[(ch * h + y) * w..(ch * h + y + 1) * w];
                for (b, fr) in frs.iter().enumerate() {
                    let (fb, sk, s2) = (f[b], sky[b * c + ch], sig2[b * c + ch]);
                    let o = ((b * c + ch) * nb + r) * pw;
                    let row = &mut bs[o..o + pw];
                    row[..pad].fill(0.0);
                    row[pad + w..].fill(0.0);
                    let src = &fr[y * w * c..(y + 1) * w * c];
                    cnt += pc_prep_row(src, rr, &mut row[pad..pad + w], c, ch, fb, sk, s2, k, frac);
                }
            }
        }
        cnt
    });
    Ok(n)
}

/// Proper coadd's column-tiled spectrum layout: copy rows `row0 .. row0 + nr` of the
/// row-FFT'd planes `src` (P, nr, 2 NF) (interleaved complex64) into plane
/// `planes[p]` of `tiles` (Q, NT, PH, 2 CB): tile `t` holds spectrum columns
/// `t*CB .. (t+1)*CB` of every row contiguously, so the column FFT and the
/// accumulation read 144 KB blocks instead of one 128-byte piece per row (those
/// strided reads were the part of the combine pass that did not scale past 2 threads).
/// A pure copy. Sequential.
#[pyfunction]
pub(crate) fn proper_coadd_scatter_tiles<'py>(
    py: Python<'py>,
    src: PyReadonlyArray3<'py, f32>,
    mut tiles: numpy::PyReadwriteArray4<'py, f32>,
    planes: Vec<usize>,
    row0: usize,
) -> PyResult<()> {
    let ss = src.as_array().shape().to_vec();
    let ts = tiles.as_array().shape().to_vec();
    let (np_, nr, nf2) = (ss[0], ss[1], ss[2]);
    let (nq, nt, ph, cb2) = (ts[0], ts[1], ts[2], ts[3]);
    if planes.len() != np_ || planes.iter().any(|&q| q >= nq) || row0 + nr > ph
        || nt * cb2 < nf2 || (nt - 1) * cb2 >= nf2 {
        return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
    }
    let sv = src.as_slice()?;
    let tv = tiles.as_slice_mut().map_err(|_| pyo3::exceptions::PyValueError::new_err("tiles must be contiguous"))?;
    py.detach(|| {
        for (p, &q) in planes.iter().enumerate() {
            for t in 0..nt {
                let c0 = t * cb2;
                let wdt = cb2.min(nf2 - c0);
                for r in 0..nr {
                    let s0 = (p * nr + r) * nf2 + c0;
                    let d0 = ((q * nt + t) * ph + row0 + r) * cb2;
                    tv[d0..d0 + wdt].copy_from_slice(&sv[s0..s0 + wdt]);
                }
            }
        }
    });
    Ok(())
}

/// `proper_coadd_accum32` over one column tile for every channel and a batch of B
/// frames: `bspec` (B, C + 1, PH, 2 nb) is the interleaved float view of the tile's
/// column-FFT'd spectra (per frame channels 0..C-1, the PSF last), `wnum`/`wden` the
/// B*C weights (frame-major), `num` (C, NT, PH, 2 CB) and `den` (C, NT, PH, CB) the
/// float32 accumulators in the same tiled layout (nb <= CB valid columns). Each
/// accumulator element takes the B frames' updates in batch order with
/// `proper_coadd_accum32`'s arithmetic (f64 add, rounded to f32 after every frame), so
/// the sums are bit-identical to accumulating the frames one at a time -- but the
/// accumulators are read and written once per batch instead of once per frame.
/// Sequential: the combine pass runs one frame batch per thread.
#[pyfunction]
pub(crate) fn proper_coadd_accum32_tile<'py>(
    py: Python<'py>,
    mut num: numpy::PyReadwriteArray4<'py, f32>,
    mut den: numpy::PyReadwriteArray4<'py, f32>,
    bspec: numpy::PyReadonlyArray4<'py, f32>,
    t: usize,
    wnum: Vec<f64>,
    wden: Vec<f64>,
) -> PyResult<()> {
    let ns = num.as_array().shape().to_vec();
    let bsh = bspec.as_array().shape().to_vec();
    let (c, nt, ph, cbw) = (ns[0], ns[1], ns[2], ns[3] / 2);
    let (nbat, nb) = (bsh[0], bsh[3] / 2);
    if den.as_array().shape() != [c, nt, ph, cbw] || ns[3] % 2 != 0 || bsh[3] % 2 != 0
        || bsh[1] != c + 1 || bsh[2] != ph || t >= nt || nb > cbw
        || wnum.len() != nbat * c || wden.len() != nbat * c {
        return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
    }
    let nm = num.as_slice_mut().map_err(|_| pyo3::exceptions::PyValueError::new_err("num must be contiguous"))?;
    let dn = den.as_slice_mut().map_err(|_| pyo3::exceptions::PyValueError::new_err("den must be contiguous"))?;
    let bs = bspec.as_slice()?;
    let plane = ph * 2 * nb;
    py.detach(|| {
        for ch in 0..c {
            for y in 0..ph {
                let o = ((ch * nt + t) * ph + y) * cbw;
                let nr = &mut nm[2 * o..2 * (o + nb)];
                let dr = &mut dn[o..o + nb];
                // frames in batch order, each over the whole tile row (in L1): every
                // element still takes its B updates in order, and the inner loop is
                // independent per element, so it vectorises
                for b in 0..nbat {
                    let base = b * (c + 1) * plane + y * 2 * nb;
                    let fr = &bs[base + ch * plane..base + ch * plane + 2 * nb];
                    let pr = &bs[base + c * plane..base + c * plane + 2 * nb];
                    pc_accum_row(nr, dr, fr, pr, wnum[b * c + ch], wden[b * c + ch]);
                }
            }
        }
    });
    Ok(())
}

/// One row of `proper_coadd_prep_rows` (channel `ch` of an interleaved (W, C) source
/// row): `proper_coadd_prep`'s f32 operations. Returns the number of replaced samples.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pc_prep_row(src: &[f32], rr: &[f32], out: &mut [f32], c: usize, ch: usize,
               fb: f32, sk: f32, s2: f32, k: f32, frac: f32) -> u64 {
    let mut cnt = 0u64;
    for x in 0..out.len() {
        let v = src[x * c + ch] - sk;
        let e = rr[x] * fb;
        let resid = v - e;
        let mut t = if e > 0.0 { e } else { 0.0 };
        t *= frac;
        t = t * t;
        t += s2;
        let bad = resid.abs() > k * t.sqrt();
        out[x] = if bad { e } else { v };
        cnt += bad as u64;
    }
    cnt
}

/// One tile row of `proper_coadd_accum32_tile` for one frame: `proper_coadd_accum32`'s
/// per-element arithmetic (f64, rounded to f32 after the add), independent per element.
pub(crate) fn pc_accum_row(nr: &mut [f32], dr: &mut [f32], fr: &[f32], pr: &[f32], wn: f64, wd: f64) {
    for i in 0..dr.len() {
        let (fa, fb) = (fr[2 * i] as f64, fr[2 * i + 1] as f64);
        let (pc, pd) = (pr[2 * i] as f64, pr[2 * i + 1] as f64);
        nr[2 * i] = (nr[2 * i] as f64 + wn * (fa * pc + fb * pd)) as f32;
        nr[2 * i + 1] = (nr[2 * i + 1] as f64 + wn * (fb * pc - fa * pd)) as f32;
        dr[i] = (dr[i] as f64 + wd * (pc * pc + pd * pd)) as f32;
    }
}
