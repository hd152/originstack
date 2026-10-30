//! Radial-bin medians and batched aperture photometry.

use crate::*;

// ---------------------------------------------------------------------------
// Radial-bin median profile (src/denoising.py's `radial_renormalize`,
// comet-mode only)
// ---------------------------------------------------------------------------
//
// The numpy reference rebuilds a boolean mask over the *whole* image once
// per bin (n_bins full-image passes) just to select each bin's pixels
// before taking the median -- wasteful, since bin membership only needs one
// O(H*W) bucket-assignment pass. This does exactly that: one pass buckets
// every pixel's channel value by its radial bin (matching the numpy
// reference's half-open `[edge[b], edge[b+1])` binning), then a native
// quickselect median per bucket, parallel over bins.

#[pyfunction]
pub(crate) fn radial_bin_median<'py>(
    py: Python<'py>,
    radii: PyReadonlyArray2<'py, f64>,
    channel: PyReadonlyArray2<'py, f64>,
    max_radius: f64,
    n_bins: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let radii = radii.as_array();
    let channel = channel.as_array();
    let (h, w) = (radii.shape()[0], radii.shape()[1]);
    let bin_width = (max_radius + 1.0) / n_bins as f64;

    let mut buckets: Vec<Vec<f64>> = (0..n_bins).map(|_| Vec::new()).collect();
    for y in 0..h {
        for x in 0..w {
            let rad = radii[[y, x]];
            let b = ((rad / bin_width) as isize).clamp(0, n_bins as isize - 1) as usize;
            buckets[b].push(channel[[y, x]]);
        }
    }

    let profile: Vec<f64> = buckets
        .into_par_iter()
        .map(|mut v| if v.is_empty() { 0.0 } else { median_inplace(&mut v) })
        .collect();

    Ok(profile.into_pyarray(py))
}

/// Batch circular-aperture photometry with partial-pixel apertures.
///
/// For each of N star centres, integrates an image `(H, W, C)` inside a
/// circle of radius `r_ap` pixels, estimates the local sky from a robust
/// median over the `(r_in, r_out]` annulus, and returns per channel the
/// background-subtracted flux, the sky level, the robust sky sigma
/// (`1.4826 * MAD`), the raw max pixel value inside the aperture (for
/// saturation flags), plus the effective aperture pixel area (shared by
/// all channels).
///
/// Aperture-edge pixels are weighted by the fraction of their unit cell
/// inside the circle, estimated by `subpix`^2 supersampling; a pixel more
/// than `sqrt(0.5)` inside/outside the edge is taken as fully in / fully
/// out with no supersampling. Integer pixel coordinates are pixel centres,
/// matching this project's `_aperture_flux` / star-detection convention. A
/// star whose full `r_out` disk is not inside the frame gets an all-NaN
/// row (the caller decides what a missing point means).
///
/// Hot path for time-series photometry: N stars x M frames aperture
/// measurements, each otherwise a Python-level masked reduction. Parallel
/// over stars (each writes its own output rows). Numpy mirror:
/// `_aperture_photometry_batch_numpy` in `src/photometry.py`.
#[pyfunction]
#[pyo3(signature = (img, xs, ys, r_ap, r_in, r_out, subpix=4))]
pub(crate) fn aperture_photometry_batch<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, f32>,
    xs: PyReadonlyArray1<'py, f64>,
    ys: PyReadonlyArray1<'py, f64>,
    r_ap: f64,
    r_in: f64,
    r_out: f64,
    subpix: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let img_arr = img.as_array();
    let (h, w, c) = (img_arr.shape()[0], img_arr.shape()[1], img_arr.shape()[2]);
    let img_flat = img_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("img must be C-contiguous (H, W, C)")
    })?;
    let xs = xs.as_slice()?;
    let ys = ys.as_slice()?;
    let n = xs.len();
    if ys.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xs and ys must match in length",
        ));
    }
    if subpix == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("subpix must be >= 1"));
    }
    if !(r_ap > 0.0) || r_in < r_ap || r_out <= r_in {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "need 0 < r_ap <= r_in < r_out",
        ));
    }

    let sub = subpix;
    let sub_area = 1.0 / (sub * sub) as f64;
    let sub_off: Vec<f64> = (0..sub)
        .map(|k| (k as f64 + 0.5) / sub as f64 - 0.5)
        .collect();
    const HALF_DIAG: f64 = 0.7071067811865476; // sqrt(0.5)
    let r_ap2 = r_ap * r_ap;
    let r_in2 = r_in * r_in;
    let r_out2 = r_out * r_out;
    let full_in = if r_ap > HALF_DIAG {
        (r_ap - HALF_DIAG).powi(2)
    } else {
        -1.0
    };
    let full_out = (r_ap + HALF_DIAG).powi(2);

    let mut flux = vec![f64::NAN; n * c];
    let mut sky = vec![f64::NAN; n * c];
    let mut sky_sig = vec![f64::NAN; n * c];
    let mut peak = vec![f64::NAN; n * c];
    let mut area = vec![f64::NAN; n];

    py.detach(|| {
        flux.par_chunks_mut(c)
            .zip(sky.par_chunks_mut(c))
            .zip(sky_sig.par_chunks_mut(c))
            .zip(peak.par_chunks_mut(c))
            .zip(area.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((((frow, srow), sgrow), prow), arow))| {
                let cx = xs[i];
                let cy = ys[i];
                if !cx.is_finite()
                    || !cy.is_finite()
                    || cx - r_out < 0.0
                    || cx + r_out >= (w - 1) as f64
                    || cy - r_out < 0.0
                    || cy + r_out >= (h - 1) as f64
                {
                    return;
                }
                let x0 = ((cx - r_out).floor() as isize - 1).max(0) as usize;
                let y0 = ((cy - r_out).floor() as isize - 1).max(0) as usize;
                let x1 = ((cx + r_out).ceil() as isize + 1).min(w as isize - 1) as usize;
                let y1 = ((cy + r_out).ceil() as isize + 1).min(h as isize - 1) as usize;

                let mut ap_sum = vec![0f64; c];
                let mut ap_area = 0f64;
                let mut ann: Vec<Vec<f32>> = vec![Vec::new(); c];

                for iy in y0..=y1 {
                    let dy = iy as f64 - cy;
                    for ix in x0..=x1 {
                        let dx = ix as f64 - cx;
                        let d2 = dx * dx + dy * dy;
                        let base = (iy * w + ix) * c;
                        let frac = if full_in > 0.0 && d2 <= full_in {
                            1.0
                        } else if d2 >= full_out {
                            0.0
                        } else {
                            let mut inside = 0usize;
                            for &oy in &sub_off {
                                let sy = dy + oy;
                                for &ox in &sub_off {
                                    let sx = dx + ox;
                                    if sx * sx + sy * sy <= r_ap2 {
                                        inside += 1;
                                    }
                                }
                            }
                            inside as f64 * sub_area
                        };
                        if frac > 0.0 {
                            ap_area += frac;
                            for ch in 0..c {
                                let val = img_flat[base + ch] as f64;
                                ap_sum[ch] += frac * val;
                                if !(val <= prow[ch]) {
                                    prow[ch] = val;
                                }
                            }
                        }
                        if d2 > r_in2 && d2 <= r_out2 {
                            for ch in 0..c {
                                ann[ch].push(img_flat[base + ch]);
                            }
                        }
                    }
                }

                *arow = ap_area;
                for ch in 0..c {
                    let v = &mut ann[ch];
                    if v.len() < 4 {
                        continue;
                    }
                    let med = median_inplace(v) as f64;
                    let mut dev: Vec<f32> =
                        v.iter().map(|&x| (x as f64 - med).abs() as f32).collect();
                    let mad = median_inplace(&mut dev) as f64;
                    srow[ch] = med;
                    sgrow[ch] = 1.4826 * mad;
                    frow[ch] = ap_sum[ch] - med * ap_area;
                }
            });
    });

    let flux2 = numpy::ndarray::Array2::from_shape_vec((n, c), flux).unwrap();
    let sky2 = numpy::ndarray::Array2::from_shape_vec((n, c), sky).unwrap();
    let sig2 = numpy::ndarray::Array2::from_shape_vec((n, c), sky_sig).unwrap();
    let peak2 = numpy::ndarray::Array2::from_shape_vec((n, c), peak).unwrap();
    let area1 = numpy::ndarray::Array1::from_vec(area);
    Ok((
        flux2.into_pyarray(py),
        sky2.into_pyarray(py),
        sig2.into_pyarray(py),
        peak2.into_pyarray(py),
        area1.into_pyarray(py),
    ))
}
