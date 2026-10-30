//! Native hot-path kernels for OriginStack (`astro_native`).
//!
//! Each module mirrors a numpy/scipy reference in `src/` and is registered in
//! the `#[pymodule]` at the bottom of this file; see `dev-notes/native-kernels.md`
//! for per-kernel design and timings. Modules share one flat namespace: every
//! item is `pub(crate)` and re-exported here, and each module starts with
//! `use crate::*;`, so a helper moved between modules needs no import edits.
//!
//! - `codec`: byte-shuffle + LZ4 codec for the Phase 1 compressed frame store.
//! - `stats`: shared order statistics (medians, mean/std).
//! - `combine`: per-pixel stack combines (sigma-clip, median, ESD, IVW, ...).
//! - `warp`, `drizzle`, `coadd`, `registration`: resampling and alignment.
//! - `calibrate`, `hotpix`, `cfa_equalize`, `color`, `debayer`, `rcd`: Phase 1.
//! - `filters`, `background`, `detect`, `psf`, `pca`, `wavelet`, `nn`: Phase 4 and fitting.
//! - `transient_triage`: tract ONNX inference, behind the `triage` Cargo feature.

pub(crate) use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4, PyReadwriteArray3,
};
pub(crate) use pyo3::prelude::*;
pub(crate) use pyo3::types::PyBytes;
pub(crate) use rayon::prelude::*;

/// Defines `$name` as a run-time AVX2 dispatcher over the portable `$body` with the same
/// signature: when the CPU reports AVX2 it calls `$body` through an inner
/// `#[target_feature(enable = "avx2")]` twin (`$body` and everything inlined into it are
/// then compiled 8-wide), otherwise it calls `$body` directly.
///
/// The crate ships for baseline x86-64, so wider vectors must be a run-time choice, never
/// `-C target-cpu`. Enabling `avx2` does not enable FMA and Rust never contracts `a * b + c`
/// on its own, so the twin is bit-identical to the baseline build; a twin with hand-written
/// intrinsics must keep multiply and add separate for the same reason (see
/// `wavelet::dwt_row_accumulate_avx2`). `$body` should be `#[inline]`, or the twin gains nothing.
macro_rules! avx2_twin {
    ($vis:vis fn $name:ident $(<$($lt:lifetime),+>)? ($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? => $body:ident) => {
        $vis fn $name $(<$($lt),+>)? ($($arg: $ty),*) $(-> $ret)? {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2")]
                unsafe fn twin $(<$($lt),+>)? ($($arg: $ty),*) $(-> $ret)? {
                    $body($($arg),*)
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: the CPU reports AVX2, the only feature `twin` enables.
                    return unsafe { twin($($arg),*) };
                }
            }
            $body($($arg),*)
        }
    };
}

mod background;
mod calibrate;
mod cfa_equalize;
mod coadd;
mod codec;
mod color;
mod combine;
mod debayer;
mod detect;
mod drizzle;
mod filters;
mod hotpix;
mod moving;
mod nn;
mod parallel;
mod pca;
mod photometry;
mod psf;
mod quality;
mod rcd;
mod registration;
mod stats;
mod trails;
#[cfg(feature = "triage")]
mod transient_triage;
mod warp;
mod wavelet;

pub(crate) use background::*;
pub(crate) use calibrate::*;
pub(crate) use cfa_equalize::*;
pub(crate) use coadd::*;
pub(crate) use codec::*;
pub(crate) use color::*;
pub(crate) use combine::*;
pub(crate) use debayer::*;
pub(crate) use detect::*;
pub(crate) use drizzle::*;
pub(crate) use filters::*;
pub(crate) use hotpix::*;
pub(crate) use moving::*;
pub(crate) use nn::*;
pub(crate) use parallel::*;
pub(crate) use pca::*;
pub(crate) use photometry::*;
pub(crate) use psf::*;
pub(crate) use quality::*;
pub(crate) use rcd::*;
pub(crate) use registration::*;
pub(crate) use stats::*;
pub(crate) use trails::*;
pub(crate) use warp::*;
pub(crate) use wavelet::*;

#[pymodule]
fn astro_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_frame, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_frame, m)?)?;
    m.add_function(wrap_pyfunction!(compress_frame_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(compress_frame_chunked_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_frame_file_into, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_frame_into, m)?)?;
    m.add_function(wrap_pyfunction!(vote_velocity, m)?)?;
    #[cfg(feature = "triage")]
    m.add_function(wrap_pyfunction!(transient_triage::transient_triage_score, m)?)?;
    m.add_function(wrap_pyfunction!(sigma_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(online_sigma_clip_seed_burnin, m)?)?;
    m.add_function(wrap_pyfunction!(online_sigma_clip_fold_frame, m)?)?;
    m.add_function(wrap_pyfunction!(patch_weighted_sigma_combine_fast, m)?)?;
    m.add_function(wrap_pyfunction!(median_combine, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(esd_combine, m)?)?;
    m.add_function(wrap_pyfunction!(linear_fit_clip_combine, m)?)?;
    m.add_function(wrap_pyfunction!(ivw_combine, m)?)?;
    m.add_function(wrap_pyfunction!(ivw_combine_with_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(dwt2_native, m)?)?;
    m.add_function(wrap_pyfunction!(idwt2_native, m)?)?;
    m.add_function(wrap_pyfunction!(sigma_clipped_median_native, m)?)?;
    m.add_function(wrap_pyfunction!(hot_pixel_box_replace_native, m)?)?;
    m.add_function(wrap_pyfunction!(blind_match_hypotheses, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine_lanczos3, m)?)?;
    m.add_function(wrap_pyfunction!(ndimage_affine_order1, m)?)?;
    m.add_function(wrap_pyfunction!(ndimage_shift_order1, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine_lanczos3_into, m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_diffusion, m)?)?;
    m.add_function(wrap_pyfunction!(lacosmic_reject_native, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_native, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_filter_native, m)?)?;
    m.add_function(wrap_pyfunction!(dbe_fit_surface, m)?)?;
    m.add_function(wrap_pyfunction!(dbe_sample_patches, m)?)?;
    m.add_function(wrap_pyfunction!(patch_entropy_batch, m)?)?;
    m.add_function(wrap_pyfunction!(detect_stars_matched_filter, m)?)?;
    m.add_function(wrap_pyfunction!(fit_rigid_ransac, m)?)?;
    m.add_function(wrap_pyfunction!(debayer_malvar, m)?)?;
    m.add_function(wrap_pyfunction!(debayer_malvar_into, m)?)?;
    m.add_function(wrap_pyfunction!(debayer_menon2007, m)?)?;
    m.add_function(wrap_pyfunction!(bilateral_filter, m)?)?;
    m.add_function(wrap_pyfunction!(warp_affine_kernel_table, m)?)?;
    m.add_function(wrap_pyfunction!(gram_matrix_wide, m)?)?;
    m.add_function(wrap_pyfunction!(small_times_wide, m)?)?;
    m.add_function(wrap_pyfunction!(robust_pca_pre_svd_input, m)?)?;
    m.add_function(wrap_pyfunction!(robust_pca_iterate, m)?)?;
    m.add_function(wrap_pyfunction!(continuum_scale_moments, m)?)?;
    m.add_function(wrap_pyfunction!(fit_moffat_native, m)?)?;
    m.add_function(wrap_pyfunction!(fit_psf_moffat2d_native, m)?)?;
    m.add_function(wrap_pyfunction!(fit_psf_gauss2d_native, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_median_grid, m)?)?;
    m.add_function(wrap_pyfunction!(local_normalize_grid, m)?)?;
    m.add_function(wrap_pyfunction!(stamp_star_disks, m)?)?;
    m.add_function(wrap_pyfunction!(bresenham_line_native, m)?)?;
    m.add_function(wrap_pyfunction!(hough_accumulator, m)?)?;
    m.add_function(wrap_pyfunction!(radial_bin_median, m)?)?;
    m.add_function(wrap_pyfunction!(aperture_photometry_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cfa_drizzle_frame, m)?)?;
    m.add_function(wrap_pyfunction!(drizzle_accumulate_lanczos3, m)?)?;
    m.add_function(wrap_pyfunction!(drizzle_splat_frame, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_frame_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_frame_from_be16, m)?)?;
    m.add_function(wrap_pyfunction!(patch_brenner_scores, m)?)?;
    m.add_function(wrap_pyfunction!(hot_pixel_bayer, m)?)?;
    m.add_function(wrap_pyfunction!(hot_pixel_bayer_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(spike_reject_bayer, m)?)?;
    m.add_function(wrap_pyfunction!(debayer_rcd_native, m)?)?;
    m.add_function(wrap_pyfunction!(debayer_rcd_native_into, m)?)?;
    m.add_function(wrap_pyfunction!(xcorr_window, m)?)?;
    m.add_function(wrap_pyfunction!(harmonic_fill, m)?)?;
    m.add_function(wrap_pyfunction!(downsample_half_f32, m)?)?;
    m.add_function(wrap_pyfunction!(proper_coadd_prep, m)?)?;
    m.add_function(wrap_pyfunction!(proper_coadd_accum32, m)?)?;
    m.add_function(wrap_pyfunction!(proper_coadd_prep_rows, m)?)?;
    m.add_function(wrap_pyfunction!(proper_coadd_scatter_tiles, m)?)?;
    m.add_function(wrap_pyfunction!(proper_coadd_accum32_tile, m)?)?;
    m.add_function(wrap_pyfunction!(nn_conv3x3, m)?)?;
    m.add_function(wrap_pyfunction!(nn_conv3x3_portable, m)?)?;
    m.add_function(wrap_pyfunction!(frame_sky_noise, m)?)?;
    m.add_function(wrap_pyfunction!(validate_frame_stats, m)?)?;
    m.add_function(wrap_pyfunction!(hot_pixel_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(pre_gradient_apply, m)?)?;
    m.add_function(wrap_pyfunction!(hot_pixel_rgb_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(luminance_native, m)?)?;
    m.add_function(wrap_pyfunction!(strided_sigma_clipped_median, m)?)?;
    m.add_function(wrap_pyfunction!(green_equalize_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(bayer_grid_equalize_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(white_balance_grayworld_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(white_balance_grayworld_lum_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(white_balance_grayworld_lum_into, m)?)?;
    m.add_function(wrap_pyfunction!(white_balance_apply, m)?)?;
    m.add_function(wrap_pyfunction!(white_balance_grayworld, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
