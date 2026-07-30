"""Background extraction and sky normalization."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple, Any

import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation, gaussian_filter, zoom

try:
    import astro_native as _native
    HAS_NATIVE = hasattr(_native, 'dbe_fit_surface')
except Exception:
    _native = None
    HAS_NATIVE = False

# Local dependencies - Added fallbacks for standalone execution or missing Config
try:
    from src.models import Config
except ImportError:
    # Fallback configuration for standalone usage or testing
    class _MockConfig:
        BORDER_FRAC = 0.1
        DBE_PATCH_SIZE = 64
        DBE_MASKED_FRAC_THRESH = 0.5
        DBE_OUTLIER_SIGMA = 3.0
        DBE_OUTLIER_ITERS = 5
        DBE_MIN_SAMPLES = 10
        DBE_MAX_SAMPLES = 1000
        DBE_FIT_SIGMA_PATCHES = 1.25
    Config = _MockConfig

try:
    from src.utils import safe_print
except ImportError:
    # Fallback logging for standalone usage
    def safe_print(*args, **kwargs):
        print(*args, **kwargs)
    safe_print = print

try:
    from astropy.stats import sigma_clipped_stats
except ImportError:
    sigma_clipped_stats = None

# Global import check for scipy functions to keep them at module level
try:
    from scipy.ndimage import gaussian_filter as _gf
except ImportError:
    _gf = None


def gaussian_filter_ds(arr: np.ndarray, sigma: float,
                       ds_threshold: float = 24.0) -> np.ndarray:
    """Gaussian blur evaluated on a block-downsampled copy for large sigmas.

    A blur of tens-to-hundreds of pixels produces a field with no content
    anywhere near the downsampled Nyquist limit, so computing it at 1/ds
    resolution with sigma/ds and bilinear-upsampling back is visually and
    numerically indistinguishable (the block-average is itself a small
    pre-filter absorbed by the huge Gaussian) at ~ds^2 less work. Below
    ``ds_threshold`` the plain full-resolution filter runs. 2-D input only.
    Weighted (numerator/denominator) usages stay consistent as long as both
    halves go through the same path, which sigma alone determines.
    """
    sigma = float(sigma)
    a = np.asarray(arr, dtype=np.float64)
    if sigma < ds_threshold or a.ndim != 2:
        return gaussian_filter(a, sigma=sigma)
    ds = 4 if sigma < 96.0 else 8
    H, W = a.shape
    h2, w2 = (H // ds) * ds, (W // ds) * ds
    if h2 < ds or w2 < ds:
        return gaussian_filter(a, sigma=sigma)
    coarse = a[:h2, :w2].reshape(h2 // ds, ds, w2 // ds, ds).mean(axis=(1, 3))
    sm = gaussian_filter(coarse, sigma=sigma / ds)
    # Centre-aligned upsample: coarse pixel j represents fine pixels
    # [j*ds, (j+1)*ds), whose centre is j*ds + (ds-1)/2. Plain zoom is
    # corner-aligned and would return the whole field shifted by ~ds/2 px,
    # which multiplied by the field's gradient near bright features is the
    # dominant error term. affine_transform with the matching offset samples
    # the coarse grid at the true block centres.
    off = -(ds - 1) / (2.0 * ds)
    out = ndimage.affine_transform(
        sm, np.array([1.0 / ds, 1.0 / ds]), offset=[off, off],
        output_shape=(H, W), order=3, mode='nearest')
    return out


def _estimate_sky_sigma(img: np.ndarray) -> float:
    """Estimate per-pixel sky noise from adjacent-pixel diffs on sky-only pairs."""
    # Ensure float64 for calculations to maintain precision
    G = img[:, :, 1].astype(np.float64)
    img_max = float(img.max())
    
    pos_g = G[G > 0]
    if pos_g.size < 100:
        return max(img_max * 1e-4, 1.0)
    
    p80 = float(np.percentile(pos_g, 80))
    if p80 == 0: return max(img_max * 1e-5, 1.0)

    # Vectorize horizontal and vertical difference calculations
    # Using slicing avoids creating temporary lists, only boolean masks and slices
    lft, rgt = G[:, :-1], G[:, 1:]
    tp,  bot = G[:-1, :], G[1:, :]

    msk_h = (lft > 0) & (rgt > 0) & (lft < p80) & (rgt < p80)
    msk_v = (tp  > 0) & (bot > 0) & (tp  < p80) & (bot  < p80)
    
    # Concatenate diffs
    # Using np.concatenate on flattened arrays is efficient
    h_diffs = (rgt - lft)[msk_h]
    v_diffs = (bot - tp)[msk_v]
    diffs = np.concatenate((h_diffs, v_diffs))

    if diffs.size < 1000:
        return max(img_max * 1e-4, 1.0)

    # Median absolute deviation conversion to Sigma for Gaussian
    # factor 0.6745 is approx 1 / sqrt(2) * erf_inv(0.5) for MAD to Sigma
    raw = float(np.median(np.abs(diffs))) / (0.6745 * np.sqrt(2))
    return max(raw, img_max * 1e-5)


def _border_pixels(lum: np.ndarray, frac: float = None) -> np.ndarray:
    """Return border-strip pixels from a 2-D luminance array."""
    if frac is None:
        frac = Config.BORDER_FRAC
    H, W = lum.shape
    by = max(10, int(H * frac))
    bx = max(10, int(W * frac))
    return np.concatenate([
        lum[:by, :].ravel(), lum[-by:, :].ravel(),
        lum[by:-by, :bx].ravel(), lum[by:-by, -bx:].ravel(),
    ])


def _sigma_sky(arr: np.ndarray, sigma: float = 3.0, maxiters: int = 5) -> Tuple[float, float]:
    """Return (median, std) of sky samples with sigma-clipped stats when available."""
    if sigma_clipped_stats is not None:
        try:
            _, med, std = sigma_clipped_stats(arr, sigma=sigma, maxiters=maxiters)
            return float(med), float(std)
        except Exception:
            pass
    return float(np.median(arr)), float(np.std(arr))


def extract_background(img: np.ndarray, mesh_size: int = 256, filter_size: int = 3,
                       clip_sigma: float = 3.0, clip_iters: int = 5,
                       star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate smooth sky background using mesh-based sigma-clipped statistics."""
    H, W = img.shape
    # Prevent division by zero for tiny images
    if H == 0 or W == 0:
        return np.zeros((H, W), dtype=np.float32)

    ny = max(1, H // mesh_size)
    nx = max(1, W // mesh_size)

    cell_h = H / ny
    cell_w = W / nx

    # Pre-allocate grid with NaN for robust outlier handling
    bg_grid = np.full((ny, nx), np.nan, dtype=np.float64)

    # Pre-compute grid boundaries to avoid repeated float math in loop
    y_edges = np.arange(ny + 1) * cell_h
    x_edges = np.arange(nx + 1) * cell_w

    for iy in range(ny):
        y0 = int(round(y_edges[iy]))
        y1 = min(int(round(y_edges[iy + 1])), H)
        # Optimization: If cell height is 0 (unlikely due to max(1)), skip
        if y1 <= y0: continue

        for ix in range(nx):
            x0 = int(round(x_edges[ix]))
            x1 = min(int(round(x_edges[ix + 1])), W)
            if x1 <= x0: continue

            # Extract cell data once
            cell = img[y0:y1, x0:x1].ravel()
            if cell.size == 0:
                continue

            # Mask out star/galaxy pixels if star_mask provided
            # Ensure mask is sliced and checked efficiently
            if star_mask is not None:
                sm = star_mask[y0:y1, x0:x1].ravel()
                # Check masked fraction first (faster than slicing both)
                masked_count = np.sum(sm >= 0.5)
                if masked_count > 10: # Threshold to avoid overhead for trivial masks
                    bg_pixels = cell[sm < 0.5]
                    if bg_pixels.size > 10:
                        cell = bg_pixels
                    elif masked_count > (sm.size * 0.5):
                        # Cell mostly contaminated - leave as NaN for interpolation
                        continue

            if cell.size == 0:
                continue

            median_val = float('nan')
            if sigma_clipped_stats is not None:
                try:
                    _, median_val, _ = sigma_clipped_stats(
                        cell, sigma=clip_sigma, maxiters=clip_iters)
                    if not np.isnan(median_val):
                        bg_grid[iy, ix] = float(median_val)
                        continue
                except Exception:
                    pass

            # Fallback: Manual sigma-clipping
            clipped = cell.copy()
            for _ in range(clip_iters):
                med = np.median(clipped)
                std = np.std(clipped)
                if std < 1e-12:
                    break
                # Avoid creating mask if empty
                if clipped.size == 0:
                    break
                mask = np.abs(clipped - med) < clip_sigma * std
                if np.any(mask):
                    clipped = clipped[mask]
                else:
                    break
            bg_grid[iy, ix] = float(np.median(clipped))

    # --- Outlier Rejection & Interpolation ---
    nan_mask = np.isnan(bg_grid)
    
    # Determine global statistics for outlier rejection
    if np.any(~nan_mask):
        finite_vals = bg_grid[~nan_mask].ravel()
        if sigma_clipped_stats is not None:
            try:
                _, _gm, _gs = sigma_clipped_stats(finite_vals, sigma=3.0, maxiters=5)
                grid_median, grid_std = float(_gm), float(_gs)
            except Exception:
                grid_median = float(np.nanmedian(bg_grid))
                grid_std = float(np.nanstd(bg_grid))
        else:
            grid_median = float(np.nanmedian(bg_grid))
            grid_std = float(np.nanstd(bg_grid))
    else:
        grid_median, grid_std = 0.0, 1.0

    # Outlier Mask
    outlier_mask = nan_mask.copy()
    if grid_std > 1e-6:
        bright_thresh = grid_median + 2.5 * grid_std
        dim_thresh = grid_median - 2.5 * grid_std
        outlier_mask |= (bg_grid > bright_thresh) | (bg_grid < dim_thresh)

    if np.any(outlier_mask) and not np.all(outlier_mask):
        iy_good, ix_good = np.where(~outlier_mask)
        vals_good = bg_grid[iy_good, ix_good]
        
        # Normalized coordinates for numerical stability
        y_good = (iy_good.astype(float) + 0.5) / ny
        x_good = (ix_good.astype(float) + 0.5) / nx
        
        # Polynomial degree selection
        # Ensure enough points for degree 3 (10 params) or fallback to degree 2 (6 params)
        min_samples_poly3 = 15 
        min_samples_poly2 = 6
        is_poly3 = len(y_good) >= min_samples_poly3

        if is_poly3:
            features = np.column_stack([
                np.ones(len(y_good)), y_good, x_good,
                y_good ** 2, y_good * x_good, x_good ** 2,
                y_good ** 3, y_good ** 2 * x_good, y_good * x_good ** 2, x_good ** 3])
        else:
            features = np.column_stack([
                np.ones(len(y_good)), y_good, x_good,
                y_good ** 2, y_good * x_good, x_good ** 2])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(features, vals_good, rcond=None)
            iy_bad, ix_bad = np.where(outlier_mask)
            y_bad = (iy_bad.astype(float) + 0.5) / ny
            x_bad = (ix_bad.astype(float) + 0.5) / nx
            
            if is_poly3:
                bad_features = np.column_stack([
                    np.ones(len(y_bad)), y_bad, x_bad,
                    y_bad ** 2, y_bad * x_bad, x_bad ** 2,
                    y_bad ** 3, y_bad ** 2 * x_bad, y_bad * x_bad ** 2, x_bad ** 3])
            else:
                bad_features = np.column_stack([
                    np.ones(len(y_bad)), y_bad, x_bad,
                    y_bad ** 2, y_bad * x_bad, x_bad ** 2])
            
            bg_grid[outlier_mask] = bad_features.dot(coeffs)
        except Exception:
            bg_grid[outlier_mask] = grid_median
    elif np.all(outlier_mask):
        bg_grid[:] = grid_median if np.isfinite(grid_median) else 0.0

    # --- Smoothing ---
    # Median filter to remove grid artifacts
    if filter_size > 1 and min(ny, nx) >= max(filter_size, 12):
        # Ensure array is C-contiguous for median_filter if possible
        if not bg_grid.flags['C_CONTIGUOUS']:
            bg_grid = np.ascontiguousarray(bg_grid)
        bg_grid = ndimage.median_filter(bg_grid, size=filter_size)

    # Gaussian smooth
    if min(ny, nx) >= 4:
        bg_grid = ndimage.gaussian_filter(bg_grid.astype(np.float64), sigma=0.8)

    # --- Interpolation to Full Res ---
    grid_y = (np.arange(ny) + 0.5) * cell_h
    grid_x = (np.arange(nx) + 0.5) * cell_w

    # Edge extrapolation using linear logic to avoid spline instability at borders
    if ny >= 2 and nx >= 2:
        ext_grid = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        ext_grid[1:-1, 1:-1] = bg_grid

        # Linear extrapolation for edges (preventing spline overshoot)
        # Top
        dy = grid_y[1] - grid_y[0] if ny > 1 else 1.0
        ext_grid[0, 1:-1] = bg_grid[0, :] + (bg_grid[0, :] - bg_grid[1, :]) * (grid_y[0] / dy)
        # Bottom
        dy = grid_y[-1] - grid_y[-2] if ny > 1 else 1.0
        ext_grid[-1, 1:-1] = bg_grid[-1, :] + (bg_grid[-1, :] - bg_grid[-2, :]) * ((H - 1 - grid_y[-1]) / dy)

        # Left
        dx = grid_x[1] - grid_x[0] if nx > 1 else 1.0
        ext_grid[1:-1, 0] = bg_grid[:, 0] + (bg_grid[:, 0] - bg_grid[:, 1]) * (grid_x[0] / dx)
        # Right
        dx = grid_x[-1] - grid_x[-2] if nx > 1 else 1.0
        ext_grid[1:-1, -1] = bg_grid[:, -1] + (bg_grid[:, -1] - bg_grid[:, -2]) * ((W - 1 - grid_x[-1]) / dx)

        # Corners
        ext_grid[0, 0] = 0.5 * (ext_grid[0, 1] + ext_grid[1, 0])
        ext_grid[0, -1] = 0.5 * (ext_grid[0, -2] + ext_grid[1, -1])
        ext_grid[-1, 0] = 0.5 * (ext_grid[-1, 1] + ext_grid[-2, 0])
        ext_grid[-1, -1] = 0.5 * (ext_grid[-1, -2] + ext_grid[-2, -1])

        ext_y = np.concatenate([[0.0], grid_y, [float(H - 1)]])
        ext_x = np.concatenate([[0.0], grid_x, [float(W - 1)]])

        ky = min(3, ny + 1)
        kx = min(3, nx + 1)
        spline = RectBivariateSpline(ext_y, ext_x, ext_grid, kx=kx, ky=ky)
    else:
        ky = min(3, max(0, ny - 1))
        kx = min(3, max(0, nx - 1))
        if ny == 1 and nx == 1: ky, kx = 1, 1
        spline = RectBivariateSpline(grid_y, grid_x, bg_grid, kx=kx, ky=ky)

    # Evaluate spline
    background = spline(np.arange(H), np.arange(W)).astype(np.float32)

    # Final Gaussian blur to suppress high-frequency mesh ripple
    blur_sigma = cell_h * 0.5
    if blur_sigma > 0:
        background = ndimage.gaussian_filter(background.astype(np.float64), sigma=blur_sigma)

    return np.asarray(background, dtype=np.float32)


def apply_background_extraction(rgb: np.ndarray, mesh_size: int = 256,
                                filter_size: int = 3, clip_sigma: float = 3.0,
                                verbose: bool = False,
                                star_mask: Optional[np.ndarray] = None,
                                exclusion_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply per-channel background subtraction with automatic extended-source masking."""
    H, W = rgb.shape[:2]
    if H == 0 or W == 0:
        return rgb.copy()

    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # --- Auto-detect extended source ---
    combined_mask = star_mask.astype(np.float32) if star_mask is not None else np.zeros((H, W), dtype=np.float32)
    # Merge caller-supplied exclusion mask (e.g. comet coma region)
    if exclusion_mask is not None:
        try:
            excl = np.asarray(exclusion_mask, dtype=np.float32)
            if excl.shape == (H, W):
                np.clip(combined_mask + excl, 0.0, 1.0, out=combined_mask)
        except Exception:
            pass

    try:
        smooth_sigma = max(20.0, min(H, W) / 50.0)
        lum_smooth = gaussian_filter_ds(lum.astype(np.float64), sigma=smooth_sigma)

        border_pix = _border_pixels(lum_smooth)
        sky_med = float(np.median(border_pix))
        sky_std = float(np.std(border_pix))

        peak_y, peak_x = np.unravel_index(np.argmax(lum_smooth), (H, W))
        peak_val = float(lum_smooth[peak_y, peak_x])

        detect_thresh = sky_med + 5.0 * max(sky_std, 1.0)
        frac_bright = float(np.mean(lum_smooth > detect_thresh))
        
        if peak_val > detect_thresh and frac_bright > 0.005:
            excl_radius = int(min(H, W) * 0.30)
            yy, xx = np.mgrid[:H, :W]
            remaining_lum = lum_smooth.copy()
            primary_peak = peak_val
            n_sources = 0
            
            for _ in range(3):
                py, px = np.unravel_index(int(np.argmax(remaining_lum)), (H, W))
                pv = float(remaining_lum[py, px])
                if pv <= detect_thresh:
                    break
                
                # Relative brightness check for secondary sources
                if n_sources > 0:
                    primary_excess = primary_peak - sky_med
                    current_excess = pv - sky_med
                    if primary_excess > 0 and current_excess < 0.5 * primary_excess:
                        break
                
                dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
                galaxy_mask = (dist < excl_radius).astype(np.float32)
                np.clip(combined_mask + galaxy_mask, 0, 1, out=combined_mask)
                
                # Mask processed source
                remaining_lum[dist < excl_radius] = float(np.min(remaining_lum))
                n_sources += 1
                
                if verbose:
                    safe_print(f"    Galaxy mask #{n_sources}: centre=({px},{py}), "
                               f"radius={excl_radius}px")
    except Exception as e:
        safe_print(f"    WARNING: Galaxy detection failed ({e}) — proceeding without exclusion mask")

    # --- Per-channel background subtraction ---
    # Process channels in parallel
    result = np.empty_like(rgb)
    channel_names = ['Red', 'Green', 'Blue']
    
    def _process_channel(c):
        channel_data = rgb[:, :, c]
        bg = extract_background(channel_data, mesh_size=mesh_size,
                                filter_size=filter_size, clip_sigma=clip_sigma,
                                star_mask=combined_mask)
        return c, (channel_data - bg)

    with ThreadPoolExecutor(max_workers=3) as executor:
        for c, ch_result in executor.map(_process_channel, range(3)):
            result[:, :, c] = ch_result

    if verbose:
        for c in range(3):
            bg_median = float(np.median(rgb[:, :, c] - result[:, :, c]))
            safe_print(f"    {channel_names[c]}: bg_median={bg_median:.1f}")

    return result


def remove_sky_residual(img: np.ndarray, mesh_size: int = 128,
                       filter_size: int = 3, clip_sigma: float = 3.0,
                       star_mask: Optional[np.ndarray] = None,
                       verbose: bool = False) -> np.ndarray:
    """Remove smooth sky residuals revealed by denoising."""
    H, W = img.shape[:2]
    if H == 0 or W == 0:
        return img.astype(np.float32)
        
    lum = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])

    # --- Detect extended sources ---
    cell_excluded = np.zeros((max(1, H // mesh_size), max(1, W // mesh_size)), dtype=bool)
    try:
        smooth_sigma = max(20.0, min(H, W) / 50.0)
        lum_smooth = gaussian_filter_ds(lum.astype(np.float64), sigma=smooth_sigma)
        border_pix = _border_pixels(lum_smooth)
        sky_med = float(np.median(border_pix))
        sky_std = float(np.std(border_pix))
        detect_thresh = sky_med + 5.0 * max(sky_std, 1.0)
        frac_bright = float(np.mean(lum_smooth > detect_thresh))
        
        if frac_bright > 0.005:
            ny = max(1, H // mesh_size)
            nx = max(1, W // mesh_size)
            cell_h = H / ny
            cell_w = W / nx
            excl_radius = int(min(H, W) * 0.30)

            remaining_lum = lum_smooth.copy()
            primary_peak = float(np.max(remaining_lum))

            # Pre-compute full-res pixel grid and cell-centre grids once
            yy, xx = np.mgrid[:H, :W]
            cell_cy = (np.arange(ny) + 0.5) * cell_h  # (ny,)
            cell_cx = (np.arange(nx) + 0.5) * cell_w  # (nx,)
            cy_grid, cx_grid = np.meshgrid(cell_cy, cell_cx, indexing='ij')  # (ny, nx)

            for _src_i in range(3):
                py, px = np.unravel_index(int(np.argmax(remaining_lum)), (H, W))
                pv = float(remaining_lum[py, px])
                if pv <= detect_thresh:
                    break

                if _src_i > 0:
                    primary_excess = primary_peak - sky_med
                    current_excess = pv - sky_med
                    if primary_excess > 0 and current_excess < 0.5 * primary_excess:
                        break

                # Vectorised cell exclusion (replaces O(ny*nx) Python loop)
                cell_excluded |= (
                    np.sqrt((cy_grid - py) ** 2 + (cx_grid - px) ** 2) < excl_radius
                )

                # Suppress this source region to find next peak
                dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
                remaining_lum[dist < excl_radius] = float(np.min(remaining_lum))
    except Exception:
        pass

    result = np.empty_like(img, dtype=np.float32)
    
    def _process_channel(c):
        ch = img[:, :, c].astype(np.float64)
        ny, nx = cell_excluded.shape
        
        bg_grid = np.full((ny, nx), np.nan, dtype=np.float64)
        
        # Optimization: Pre-calculate coordinates if grid is regular enough, 
        # but for simplicity and correctness, we loop.
        for iy in range(ny):
            if cell_excluded[iy, :].all():
                bg_grid[iy, :] = np.nan
                continue
                
            y0 = int(round(iy * (H/ny)))
            y1 = min(int(round((iy + 1) * (H/ny))), H)
            
            # Vectorize inner loop where possible
            for ix in range(nx):
                if cell_excluded[iy, ix]:
                    bg_grid[iy, ix] = np.nan
                    continue
                
                x0 = int(round(ix * (W/nx)))
                x1 = min(int(round((ix + 1) * (W/nx))), W)
                
                cell = ch[y0:y1, x0:x1].ravel()
                if cell.size == 0: continue
                
                if star_mask is not None:
                    sm = star_mask[y0:y1, x0:x1].ravel()
                    cell = cell[sm < 0.5]
                    
                if len(cell) > 0:
                    bg_grid[iy, ix] = float(np.median(cell))

        # Interpolation
        if np.all(np.isnan(bg_grid)):
            bg_grid[:] = 0.0
        elif np.any(np.isnan(bg_grid)):
            # Fallback: copy nearest valid cell
            nan_mask = np.isnan(bg_grid)
            try:
                _, nn_idx = ndimage.distance_transform_edt(nan_mask, return_indices=True)
                bg_grid[nan_mask] = bg_grid[nn_idx[0][nan_mask], nn_idx[1][nan_mask]]
            except Exception:
                bg_grid[np.isnan(bg_grid)] = float(np.median(bg_grid))

        if filter_size > 1 and min(ny, nx) >= filter_size:
            bg_grid = ndimage.median_filter(bg_grid, size=filter_size)

        if min(ny, nx) >= 4:
            bg_grid = ndimage.gaussian_filter(bg_grid.astype(np.float64), sigma=0.8)

        grid_y = (np.arange(ny) + 0.5) * (H / ny)
        grid_x = (np.arange(nx) + 0.5) * (W / nx)
        
        if ny >= 2 and nx >= 2:
            ext_grid = np.zeros((ny + 2, nx + 2), dtype=np.float64)
            ext_grid[1:-1, 1:-1] = bg_grid
            # Linear Extrapolation Logic reused for brevity
            dy = grid_y[1] - grid_y[0] if ny > 1 else 1.0
            ext_grid[0, 1:-1] = bg_grid[0, :] + (bg_grid[0, :] - bg_grid[1, :]) * (grid_y[0] / dy)
            dy = grid_y[-1] - grid_y[-2] if ny > 1 else 1.0
            ext_grid[-1, 1:-1] = bg_grid[-1, :] + (bg_grid[-1, :] - bg_grid[-2, :]) * ((H - 1 - grid_y[-1]) / dy)
            dx = grid_x[1] - grid_x[0] if nx > 1 else 1.0
            ext_grid[1:-1, 0] = bg_grid[:, 0] + (bg_grid[:, 0] - bg_grid[:, 1]) * (grid_x[0] / dx)
            dx = grid_x[-1] - grid_x[-2] if nx > 1 else 1.0
            ext_grid[1:-1, -1] = bg_grid[:, -1] + (bg_grid[:, -1] - bg_grid[:, -2]) * ((W - 1 - grid_x[-1]) / dx)
            ext_grid[0, 0] = 0.5 * (ext_grid[0, 1] + ext_grid[1, 0])
            ext_grid[0, -1] = 0.5 * (ext_grid[0, -2] + ext_grid[1, -1])
            ext_grid[-1, 0] = 0.5 * (ext_grid[-1, 1] + ext_grid[-2, 0])
            ext_grid[-1, -1] = 0.5 * (ext_grid[-1, -2] + ext_grid[-2, -1])
            ext_y = np.concatenate([[0.0], grid_y, [float(H - 1)]])
            ext_x = np.concatenate([[0.0], grid_x, [float(W - 1)]])
            spline = RectBivariateSpline(ext_y, ext_x, ext_grid, kx=min(3, nx+1), ky=min(3, ny+1))
        else:
            spline = RectBivariateSpline(grid_y, grid_x, bg_grid, kx=min(3, max(0,nx-1)), ky=min(3, max(0,ny-1)))
            
        background = spline(np.arange(H), np.arange(W)).astype(np.float64)
        blur_sigma = (H/ny) * 0.5
        if blur_sigma > 0:
            background = gaussian_filter_ds(background, sigma=blur_sigma)

        result[:, :, c] = (ch - background).astype(np.float32)
        if verbose:
            safe_print(f"    Channel {c}: residual median={float(np.median(background)):.2f}")

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(_process_channel, range(3)))

    return result


def sky_floor_normalize(rgb: np.ndarray, star_mask: Optional[np.ndarray] = None,
                        verbose: bool = False) -> np.ndarray:
    """Drive the constant sky pedestal to zero."""
    H, W = rgb.shape[:2]
    if H == 0 or W == 0:
        return rgb.copy()
        
    result = rgb.copy()
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # --- Build source mask ---
    src_mask = np.zeros((H, W), dtype=np.float32)
    if star_mask is not None:
        np.clip(src_mask + star_mask, 0, 1, out=src_mask)

    # --- Detect emission in border for sky floor estimation ---
    border_lum = _border_pixels(lum)
    sky_med, sky_std = _sigma_sky(border_lum)

    # --- Mask extended sources ---
    smooth_sigma = max(5.0, min(H, W) / 200.0)
    lum_smooth = gaussian_filter_ds(lum.astype(np.float64), sigma=smooth_sigma)
    src_thresh = sky_med + 2.0 * max(sky_std, 1.0)
    np.clip(src_mask + (lum_smooth > src_thresh).astype(np.float32), 0, 1, out=src_mask)

    sky_pix_mask = src_mask < 0.5     # True where it is pure sky

    if sky_pix_mask.sum() < 1000:
        if verbose:
            safe_print("  Sky floor: too few sky pixels to normalise, skipping")
        return result

    sky_frac = 100.0 * sky_pix_mask.sum() / (H * W)

    # --- Per-channel floor subtraction ---
    floors = []
    for c in range(3):
        ch_sky = rgb[:, :, c][sky_pix_mask]
        if sigma_clipped_stats is not None:
            try:
                _, floor, _ = sigma_clipped_stats(ch_sky, sigma=3.0, maxiters=5)
                floor = float(floor)
            except Exception:
                floor = float(np.median(ch_sky))
        else:
            floor = float(np.median(ch_sky))
        floors.append(floor)
        result[:, :, c] -= floor

    # Do NOT clip negatives to zero here.  Floor subtraction shifts the sky
    # noise distribution to centre on zero, so roughly half the sky pixels
    # will be slightly negative.  Hard-clipping those to zero creates a
    # one-sided truncation that makes faint nebula pixels indistinguishable
    # from over-subtracted sky, producing black mottling in the stretch.
    # The stretch functions handle the black point independently.

    if verbose:
        safe_print(f"  Sky floor: {sky_frac:.1f}% sky pixels used, "
                   f"floors subtracted R={floors[0]:.2f} G={floors[1]:.2f} "
                   f"B={floors[2]:.2f} ADU")

    return result


def _dbe_prepare_emission_mask(rgb: np.ndarray, star_mask: Optional[np.ndarray],
                               exclusion_mask: Optional[np.ndarray],
                               verbose: bool, label: str) -> Tuple[np.ndarray, float, float, bool]:
    """Shared emission-mask setup for DBE-family background extractors
    (dynamic_background_extraction, wavelet_background_extraction).
    Returns (emission_mask, sky_med, sky_std, dense_field)."""
    H, W = rgb.shape[:2]
    lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float64)
    smooth_sigma = max(20.0, min(H, W) / 50.0)
    lum_smooth = gaussian_filter_ds(lum, sigma=smooth_sigma)
    sky_med, sky_std = _sigma_sky(_border_pixels(lum_smooth))

    emission_mask = _build_emission_mask(lum, star_mask, lum_smooth, sky_med, sky_std)
    if exclusion_mask is not None:
        try:
            excl = np.asarray(exclusion_mask, dtype=np.float32)
            if excl.shape == emission_mask.shape:
                np.clip(emission_mask + excl, 0.0, 1.0, out=emission_mask)
        except Exception:
            pass

    masked_pct = float(np.mean(emission_mask >= 0.5))
    dense_field = masked_pct > Config.DBE_DENSE_FIELD_THRESH
    if verbose:
        safe_print(f"    {label}: emission mask covers {masked_pct * 100.0:.1f}% of image"
                   + (" (dense star field — sigma-clip mesh fallback)" if dense_field else ""))
    elif dense_field:
        safe_print(f"    {label}: dense star field detected ({masked_pct * 100.0:.0f}% masked) "
                   f"— using sigma-clip mesh without emission mask")
    return emission_mask, sky_med, sky_std, dense_field


# ============ Dynamic Background Extraction (DBE) =========


def _build_emission_mask(lum: np.ndarray, star_mask: Optional[np.ndarray],
                         lum_smooth: np.ndarray,
                         sky_med: float, sky_std: float) -> np.ndarray:
    """Build a float32 exclusion mask (1 = emission/star, 0 = safe background)."""
    H, W = lum.shape
    emission = np.zeros((H, W), dtype=np.float32)

    if star_mask is not None:
        np.clip(emission + star_mask, 0.0, 1.0, out=emission)

    detect_thresh = sky_med + 5.0 * max(sky_std, 1.0)
    frac_bright = float(np.mean(lum_smooth > detect_thresh))
    
    if frac_bright > 0.005:
        dil_radius = max(15, int(min(H, W) * 0.02))
        r = dil_radius
        y_idx, x_idx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (y_idx ** 2 + x_idx ** 2 <= r ** 2)
        structure = disk.astype(np.uint8) # Binary dilation expects struct array
        
        remaining_lum = lum_smooth.copy()
        primary_peak = float(np.max(remaining_lum))

        for src_i in range(3):
            py, px = np.unravel_index(int(np.argmax(remaining_lum)), (H, W))
            pv = float(remaining_lum[py, px])
            if pv <= detect_thresh:
                break
            
            if src_i > 0:
                if (primary_peak - sky_med) > 0:
                    if (pv - sky_med) < 0.5 * (primary_peak - sky_med):
                        break
                        
            src_thresh = sky_med + 0.5 * (detect_thresh - sky_med)
            src_binary = (lum_smooth > src_thresh).astype(np.uint8)
            
            # Use binary_dilation on the binary mask
            dilated = binary_dilation(src_binary, structure=structure).astype(np.float32)
            
            np.clip(emission + dilated, 0.0, 1.0, out=emission)
            # Blank out processed source
            remaining_lum[emission > 0.5] = float(np.min(remaining_lum))

    # Diffuse extended emission: large, low-contrast nebulosity that never
    # reaches the compact-source 5-sigma bar above (real signal, but under
    # ~1 sigma above sky per pixel) is otherwise invisible to the detector
    # above -- DBE then samples it as "background" and fits/subtracts it,
    # silently erasing the nebula from the output. lum_smooth is already
    # smoothed at a scale (sigma = max(20, min(H,W)/50)) large enough that
    # pixel noise averages down far below sky_std, so a modest absolute
    # threshold against the *unsmoothed* sky_std reliably separates real
    # large-scale signal from residual smoothed noise without needing
    # per-pixel significance. pre_gradient_removal already runs before DBE,
    # so most smooth optical vignetting/gradient is gone by this point --
    # what's left this broad is far more likely real extended emission than
    # gradient residual.
    diffuse_thresh = sky_med + 1.0 * max(sky_std, 1.0)
    diffuse_binary = (lum_smooth > diffuse_thresh).astype(np.float32)
    if diffuse_binary.any():
        np.clip(emission + diffuse_binary, 0.0, 1.0, out=emission)

    return emission


def _patch_entropy(pixels: np.ndarray, n_bins: int = 16) -> float:
    """Shannon entropy of a pixel sample (lower = more uniform = better background).

    Uses a fixed-width histogram with ``n_bins`` bins spanning the pixel
    range.  Returns 0 for constant patches and log2(n_bins) for perfectly
    uniform histograms.  Sky-background patches are nearly Gaussian around
    the sky mean and have low entropy; patches contaminated by stars or
    nebula emission have high entropy and should be down-weighted.
    """
    if len(pixels) < 4:
        return 0.0
    mn, mx = float(pixels.min()), float(pixels.max())
    rng = mx - mn
    if rng < 1e-12:
        return 0.0
    counts, _ = np.histogram(pixels, bins=n_bins, range=(mn, mx))
    probs = counts / float(counts.sum() + 1e-12)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _sample_background_patches(
        channel: np.ndarray, emission_mask: np.ndarray,
        patch_size: int, masked_frac_thresh: float,
        sky_ref: float, sky_std: float,
        use_entropy_weights: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Find clean background patches."""
    H, W = channel.shape
    ny = max(1, H // patch_size)
    nx = max(1, W // patch_size)
    cell_h = H / ny
    cell_w = W / nx

    coords = values = variances = None
    if HAS_NATIVE and hasattr(_native, 'dbe_sample_patches'):
        # Native per-patch loop (rayon-parallel): same rejection cascade,
        # medians are exact order statistics so the f32 cast is lossless for
        # f32-representable pixel data. Variance/entropy filters below stay
        # in Python — they operate on the small per-patch result.
        try:
            coords, values, variances = _native.dbe_sample_patches(
                np.ascontiguousarray(channel, dtype=np.float32),
                np.ascontiguousarray(emission_mask, dtype=np.float32),
                int(patch_size), float(masked_frac_thresh),
                float(sky_ref), float(sky_std))
            if len(values) == 0:
                return np.empty((0, 2)), np.empty((0,))
        except Exception:
            coords = values = variances = None

    if coords is not None:
        coords = np.asarray(coords, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        return _filter_sampled_patches(channel, emission_mask, patch_size,
                                       coords, values, variances,
                                       use_entropy_weights)

    coords_list = []
    values_list = []
    variances = []

    for iy in range(ny):
        y0 = int(round(iy * cell_h))
        y1 = min(int(round((iy + 1) * cell_h)), H)
        
        # Optimization: check center pixel of emission mask first if sparse
        # But for correctness, we iterate.
        
        for ix in range(nx):
            x0 = int(round(ix * cell_w))
            x1 = min(int(round((ix + 1) * cell_w)), W)
            
            # Slice once for both mask and data
            em_patch = emission_mask[y0:y1, x0:x1]
            if np.any(em_patch >= 0.5): # Fast check for any mask overlap
                 # Check exact masked fraction threshold
                 if np.mean(em_patch >= 0.5) > masked_frac_thresh:
                     continue

            px = channel[y0:y1, x0:x1].ravel()
            px = px[em_patch.ravel() < 0.5]
            if px.size < 10:
                continue

            patch_med = float(np.median(px))
            # Brightness check against sky reference
            if patch_med > sky_ref + 2.0 * max(sky_std, 1.0):
                continue

            # Fast single-pass sigma-clip
            mad = float(np.median(np.abs(px - patch_med)))
            sig = 1.4826 * mad
            if sig > 1e-12:
                px = px[np.abs(px - patch_med) <= 3.0 * sig]
            
            med_val = float(np.median(px)) if px.size > 0 else patch_med

            coords_list.append(((y0 + (y1 - y0) * 0.5) / H,
                                (x0 + (x1 - x0) * 0.5) / W))
            values_list.append(med_val)
            variances.append(float(np.var(px)))

    if not coords_list:
        return np.empty((0, 2)), np.empty((0,))

    coords = np.array(coords_list, dtype=np.float64)
    values = np.array(values_list, dtype=np.float64)
    variances = np.array(variances, dtype=np.float64)
    return _filter_sampled_patches(channel, emission_mask, patch_size,
                                   coords, values, variances,
                                   use_entropy_weights)


def _filter_sampled_patches(channel: np.ndarray, emission_mask: np.ndarray,
                            patch_size: int,
                            coords: np.ndarray, values: np.ndarray,
                            variances: np.ndarray,
                            use_entropy_weights: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Variance and entropy outlier filters over sampled patches — shared by
    the native and numpy sampling paths."""
    if len(variances) >= 5:
        med_var = float(np.median(variances))
        mad_var = float(np.median(np.abs(variances - med_var)))
        var_thresh = med_var + 3.0 * 1.4826 * max(mad_var, 1e-12)
        keep = variances <= var_thresh
        coords = coords[keep]
        values = values[keep]

    # Entropy filter: reject patches whose Shannon entropy is unusually high,
    # indicating residual contamination by stars or emission structure that the
    # binary emission mask missed.  Low-entropy patches (nearly uniform ADU
    # histogram) are genuine sky background samples.
    if use_entropy_weights and len(values) >= 8:
        H_ch, W_ch = channel.shape
        ny_g = max(1, H_ch // patch_size)
        nx_g = max(1, W_ch // patch_size)
        cell_h_g = H_ch / ny_g
        cell_w_g = W_ch / nx_g
        entropies = []
        for coord in coords:
            iy = int(np.clip(round(coord[0] * ny_g - 0.5), 0, ny_g - 1))
            ix = int(np.clip(round(coord[1] * nx_g - 0.5), 0, nx_g - 1))
            y0 = int(round(iy * cell_h_g))
            y1 = min(int(round((iy + 1) * cell_h_g)), H_ch)
            x0 = int(round(ix * cell_w_g))
            x1 = min(int(round((ix + 1) * cell_w_g)), W_ch)
            em = emission_mask[y0:y1, x0:x1].ravel()
            px = channel[y0:y1, x0:x1].ravel()
            px = px[em < 0.5]
            entropies.append(_patch_entropy(px))
        entropies = np.array(entropies, dtype=np.float64)
        med_ent = float(np.median(entropies))
        mad_ent = float(np.median(np.abs(entropies - med_ent)))
        ent_thresh = med_ent + 2.5 * 1.4826 * max(mad_ent, 1e-9)
        keep_ent = entropies <= ent_thresh
        if keep_ent.sum() >= 6:
            coords = coords[keep_ent]
            values = values[keep_ent]

    return coords, values


def _dbe_solve_local_linear(acc: np.ndarray) -> np.ndarray:
    """Solve the ridge-regularised 3x3 local-linear normal equations for a
    batch of evaluation points. ``acc`` is (M, 9): sw swy swx swyy swyx swxx
    swv swyv swxv. Returns the fitted value (constant term) per point, with a
    plain weighted-mean fallback where ill-conditioned and NaN where sw~0.
    Mirrors the native kernel's ``dbe_solve`` exactly.
    """
    sw, swy, swx, swyy, swyx, swxx, swv, swyv, swxv = acc.T
    lam = 1e-3 * sw
    a22 = swyy + lam
    a33 = swxx + lam
    det = (sw * (a22 * a33 - swyx * swyx) - swy * (swy * a33 - swyx * swx)
           + swx * (swy * swyx - a22 * swx))
    det1 = (swv * (a22 * a33 - swyx * swyx) - swy * (swyv * a33 - swyx * swxv)
            + swx * (swyv * swyx - a22 * swxv))
    scale = np.maximum(np.abs(sw) * np.abs(a22) * np.abs(a33), 1e-30)
    with np.errstate(divide='ignore', invalid='ignore'):
        nw = swv / sw
        ll = det1 / det
    out = np.where(np.abs(det) < 1e-10 * scale, nw, ll)
    return np.where(sw < 1e-12, np.nan, out)


def _dbe_accum_batch(py_: np.ndarray, px_: np.ndarray,
                     ys: np.ndarray, xs: np.ndarray, vs: np.ndarray,
                     wr: np.ndarray, inv_sigma: float,
                     trunc_d2: float) -> np.ndarray:
    """Gaussian-weighted local-linear accumulators for a batch of evaluation
    points against all samples. Returns (M, 9). Mirrors ``dbe_accum``."""
    dy = (ys[None, :] - py_[:, None]) * inv_sigma
    dx = (xs[None, :] - px_[:, None]) * inv_sigma
    d2 = dy * dy + dx * dx
    w = wr[None, :] * np.exp(-0.5 * d2)
    if np.isfinite(trunc_d2):
        w = np.where(d2 > trunc_d2, 0.0, w)
    acc = np.empty((len(py_), 9), dtype=np.float64)
    acc[:, 0] = w.sum(axis=1)
    acc[:, 1] = (w * dy).sum(axis=1)
    acc[:, 2] = (w * dx).sum(axis=1)
    acc[:, 3] = (w * dy * dy).sum(axis=1)
    acc[:, 4] = (w * dy * dx).sum(axis=1)
    acc[:, 5] = (w * dx * dx).sum(axis=1)
    acc[:, 6] = (w * vs[None, :]).sum(axis=1)
    acc[:, 7] = (w * dy * vs[None, :]).sum(axis=1)
    acc[:, 8] = (w * dx * vs[None, :]).sum(axis=1)
    return acc


def _dbe_fit_surface_numpy(coords: np.ndarray, values: np.ndarray,
                           img_h: float, img_w: float,
                           grid_h: int, grid_w: int, sigma_px: float,
                           tukey_c: float = 4.685,
                           irls_iters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy fallback for the native ``dbe_fit_surface`` kernel — identical
    math (Gaussian-weighted local-linear regression, Tukey-biweight IRLS,
    5-sigma truncation with untruncated retry for empty gaps)."""
    n = len(values)
    ys = coords[:, 0] * img_h
    xs = coords[:, 1] * img_w
    vs = values.astype(np.float64)
    inv_sigma = 1.0 / max(sigma_px, 1e-9)
    wr = np.ones(n, dtype=np.float64)
    chunk = 512

    for _ in range(irls_iters):
        residuals = np.empty(n, dtype=np.float64)
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            acc = _dbe_accum_batch(ys[i0:i1], xs[i0:i1], ys, xs, vs, wr,
                                   inv_sigma, 25.0)
            # leave-one-out: remove self (d2=0 -> sw and swv only)
            idx = np.arange(i0, i1)
            acc[:, 0] -= wr[idx]
            acc[:, 6] -= wr[idx] * vs[idx]
            fit = _dbe_solve_local_linear(acc)
            residuals[i0:i1] = np.where(np.isnan(fit), 0.0, vs[idx] - fit)
        med_r = float(np.median(residuals))
        s = 1.4826 * float(np.median(np.abs(residuals - med_r)))
        if s < 1e-9:
            break
        u = (residuals - med_r) / (tukey_c * s)
        wr = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)

    wsum = float(wr.sum())
    global_mean = (float((wr * vs).sum()) / wsum if wsum > 1e-12
                   else float(np.median(vs)))

    gy = (np.linspace(0.0, 1.0, grid_h) * img_h if grid_h > 1
          else np.zeros(1))
    gx = (np.linspace(0.0, 1.0, grid_w) * img_w if grid_w > 1
          else np.zeros(1))
    gyy, gxx = np.meshgrid(gy, gx, indexing='ij')
    pts_y, pts_x = gyy.ravel(), gxx.ravel()
    m = len(pts_y)
    surface = np.empty(m, dtype=np.float64)
    for i0 in range(0, m, chunk):
        i1 = min(i0 + chunk, m)
        acc = _dbe_accum_batch(pts_y[i0:i1], pts_x[i0:i1], ys, xs, vs, wr,
                               inv_sigma, 25.0)
        empty = acc[:, 0] < 1e-12
        if np.any(empty):
            acc[empty] = _dbe_accum_batch(pts_y[i0:i1][empty],
                                          pts_x[i0:i1][empty],
                                          ys, xs, vs, wr, inv_sigma, np.inf)
        fit = _dbe_solve_local_linear(acc)
        surface[i0:i1] = np.where(np.isnan(fit), global_mean, fit)
    return surface.reshape(grid_h, grid_w), wr


def _dbe_regression_coarse_grid(coords: np.ndarray, values: np.ndarray,
                                H: int, W: int, outlier_sigma: float, max_iter: int,
                                sigma_px: float, Hc: int, Wc: int, verbose: bool,
                                label: str = "DBE") -> Tuple[np.ndarray, float, float]:
    """Shared robust local-regression core (Gaussian-weighted local-linear,
    Tukey-biweight IRLS; native/Rust when available, exact numpy mirror
    otherwise), evaluated onto a (Hc, Wc) coarse grid. Returns
    ``(coarse_grid, surf_lo, surf_hi)`` -- the finishing step (how that grid
    gets turned into a full-res surface: Gaussian tail for DBE, starlet hard
    cutoff for the wavelet background) is left to the caller.

    This replaced the earlier thin-plate-spline RBF + hard outlier-rejection
    loop: the RBF was globally supported and unbounded, so the large sample
    gaps the rejection loop carved out near bright stars let it extrapolate
    far outside the real sky range (observed as a near-black wedge fanning
    out from a star after subtraction). The local fit stays near the
    surrounding sample values by construction and downweights outliers
    continuously instead of removing samples outright.
    """
    tukey_c = 4.685 * (outlier_sigma / 2.5)

    coarse = None
    if HAS_NATIVE:
        try:
            coarse, wrob = _native.dbe_fit_surface(
                np.ascontiguousarray(coords, dtype=np.float64),
                np.ascontiguousarray(values, dtype=np.float64),
                float(H), float(W), Hc, Wc, float(sigma_px),
                float(tukey_c), int(max_iter))
        except Exception as e:
            safe_print(f"    WARNING: native {label} fit failed ({e}) — using numpy")
            coarse = None
    if coarse is None:
        coarse, wrob = _dbe_fit_surface_numpy(
            coords, values, float(H), float(W), Hc, Wc, float(sigma_px),
            tukey_c=tukey_c, irls_iters=max_iter)

    if verbose:
        n_down = int(np.sum(wrob < 0.5))
        if n_down:
            safe_print(f"    {label} robust fit: downweighted {n_down}/{len(wrob)} patches")

    # The local fit is already bounded near the sample values, but zoom
    # (order=3) can overshoot slightly at sharp coarse-grid transitions —
    # keep a cheap clamp to the sample range plus a noise margin.
    res_scale = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    surf_lo = float(np.min(values)) - 3.0 * res_scale
    surf_hi = float(np.max(values)) + 3.0 * res_scale
    coarse = np.clip(coarse, surf_lo, surf_hi)
    return coarse, surf_lo, surf_hi


def _fit_background_surface(coords: np.ndarray, values: np.ndarray,
                            H: int, W: int,
                            outlier_sigma: float, max_iter: int,
                            patch_size: int, verbose: bool) -> np.ndarray:
    """Fit a smooth background surface via robust local regression, finished
    with a Gaussian blur tail (see ``_dbe_regression_coarse_grid``)."""
    if len(coords) < 6:
        return _polynomial_surface(coords, values, H, W, patch_size)

    sigma_px = Config.DBE_FIT_SIGMA_PATCHES * patch_size

    # Use coarser grid for large images to keep evaluation tractable
    min_dim = min(H, W)
    if min_dim > 4000:
        stride = max(8, patch_size // 2)
    else:
        stride = max(4, patch_size // 4)
    Hc = max(4, H // stride)
    Wc = max(4, W // stride)

    coarse, surf_lo, surf_hi = _dbe_regression_coarse_grid(
        coords, values, H, W, outlier_sigma, max_iter, sigma_px, Hc, Wc, verbose)

    surface = zoom(coarse, (H / Hc, W / Wc), order=3)[:H, :W]
    surface = ndimage.gaussian_filter(surface, sigma=patch_size * 0.5)
    return np.clip(surface, surf_lo, surf_hi)


def _polynomial_surface(coords: np.ndarray, values: np.ndarray,
                        H: int, W: int, patch_size: int) -> np.ndarray:
    """Polynomial least-squares fallback."""
    if len(coords) < 3:
        med_val = float(np.median(values)) if len(values) else 0.0
        return np.full((H, W), med_val, dtype=np.float64)

    def poly3(y, x):
        return np.column_stack([np.ones(len(y)), y, x, y ** 2, y * x, x ** 2,
                                y ** 3, y ** 2 * x, y * x ** 2, x ** 3])

    def poly2(y, x):
        return np.column_stack([np.ones(len(y)), y, x, y ** 2, y * x, x ** 2])

    poly = poly3 if len(coords) >= 10 else poly2
    A = poly(coords[:, 0], coords[:, 1])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    except Exception as e:
        safe_print(f"    WARNING: Polynomial fit failed ({e}) — using flat median background")
        med_val = float(np.median(values)) if len(values) else 0.0
        return np.full((H, W), med_val, dtype=np.float64)

    yy = np.linspace(0.0, 1.0, H)
    xx = np.linspace(0.0, 1.0, W)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing='ij')
    surface = poly(grid_y.ravel(), grid_x.ravel()).dot(coeffs).reshape(H, W)
    return ndimage.gaussian_filter(surface, sigma=patch_size * 0.5)


def dynamic_background_extraction(
        rgb: np.ndarray,
        patch_size: int = Config.DBE_PATCH_SIZE,
        masked_frac_thresh: float = Config.DBE_MASKED_FRAC_THRESH,
        clip_sigma: float = 3.0,
        outlier_sigma: float = Config.DBE_OUTLIER_SIGMA,
        outlier_iters: int = Config.DBE_OUTLIER_ITERS,
        star_mask: Optional[np.ndarray] = None,
        verbose: bool = False,
        use_entropy_weights: bool = False,
        exclusion_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Dynamic Background Extraction (DBE): adaptive patch sampling + robust
    local-regression surface fit (see ``_fit_background_surface``)."""
    from concurrent.futures import ThreadPoolExecutor

    H, W = rgb.shape[:2]
    if H == 0 or W == 0:
        return rgb.copy()
        
    emission_mask, sky_med, sky_std, _dense_field = _dbe_prepare_emission_mask(
        rgb, star_mask, exclusion_mask, verbose, "DBE")

    if verbose and HAS_NATIVE:
        safe_print("    [rust] DBE robust local-regression surface fit")

    result = np.empty_like(rgb)
    channel_names = ['Red', 'Green', 'Blue']

    def _process_channel(c):
        channel = rgb[:, :, c].astype(np.float64)
        coords, values = _sample_background_patches(
            channel, emission_mask, patch_size, masked_frac_thresh,
            sky_ref=sky_med, sky_std=sky_std,
            use_entropy_weights=use_entropy_weights)

        n = len(coords)
        if verbose:
            safe_print(f"    DBE {channel_names[c]}: {n} background patches accepted")

        if n < Config.DBE_MIN_SAMPLES:
            if verbose:
                safe_print(f"    DBE {channel_names[c]}: insufficient samples ({n}), "
                           f"falling back to "
                           f"{'sigma-clip mesh (no emission mask)' if _dense_field else 'mesh extraction'}")
            if _dense_field:
                # Ultra-dense field: emission mask would exclude too much.
                # Sigma-clipping within each mesh cell reliably rejects bright
                # stars without needing an explicit exclusion mask.
                background = extract_background(
                    channel, mesh_size=patch_size * 2,
                    clip_sigma=clip_sigma,
                    star_mask=None).astype(np.float64)
            else:
                background = extract_background(channel, mesh_size=patch_size,
                                                clip_sigma=clip_sigma,
                                                star_mask=emission_mask).astype(np.float64)
        else:
            # Subsample if too many for RBF performance
            if n > Config.DBE_MAX_SAMPLES:
                step = max(1, n // Config.DBE_MAX_SAMPLES)
                coords = coords[::step]
                values = values[::step]
                if verbose:
                    safe_print(f"    DBE {channel_names[c]}: subsampled to "
                               f"{len(coords)} patches")

            background = _fit_background_surface(
                coords, values, H, W,
                outlier_sigma=outlier_sigma, max_iter=outlier_iters,
                patch_size=patch_size, verbose=verbose)

        subtracted = channel - background
        if verbose:
            safe_print(f"    {channel_names[c]}: bg_median="
                       f"{float(np.median(background)):.1f}, "
                       f"subtracted_median={float(np.median(subtracted)):.1f}")
        return c, subtracted.astype(np.float32)

    with ThreadPoolExecutor(max_workers=3) as executor:
        for c, ch_result in executor.map(_process_channel, range(3)):
            result[:, :, c] = ch_result

    return result


# ============ Wavelet-band (starlet) Background Extraction =========
#
# DBE's surface fit (_fit_background_surface) is a Gaussian-weighted local
# regression: its scale is set by one blur-radius-like sigma, and Gaussian
# tails mean sample points well outside a patch can still tug on it, which
# is what the final large Gaussian blur (patch_size*0.5) is there to hide.
# A starlet (a trous wavelet) decomposition gives a genuine multiresolution
# low-pass instead: convolving with dilated B3-spline kernels at doubling
# scales, then keeping only the coarsest approximation plane, discards *all*
# structure below a hard pixel-scale cutoff by construction -- no long tails,
# and the cutoff is specified directly in "reject anything smaller than N
# pixels" terms rather than a sigma to be reasoned about indirectly. Same
# sky-patch sampling (emission mask + robust per-patch rejection) as DBE;
# only the samples-to-surface fit differs.

_B3_SPLINE_KERNEL = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _atrous_convolve(img: np.ndarray, step: int) -> np.ndarray:
    """Separable a-trous convolution: the B3-spline kernel with ``step - 1``
    zero-holes inserted between taps (step = 2**scale), applied along both
    axes. step=1 is the plain 5-tap kernel."""
    if step <= 1:
        kernel = _B3_SPLINE_KERNEL
    else:
        kernel = np.zeros((len(_B3_SPLINE_KERNEL) - 1) * step + 1, dtype=np.float64)
        kernel[::step] = _B3_SPLINE_KERNEL
    out = ndimage.convolve1d(img, kernel, axis=0, mode='reflect')
    out = ndimage.convolve1d(out, kernel, axis=1, mode='reflect')
    return out


def _starlet_transform(img: np.ndarray, n_scales: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Isotropic undecimated wavelet (starlet / a-trous B3-spline) decomposition.

    Returns ``(detail_planes, coarse_approximation)`` such that
    ``img == sum(detail_planes) + coarse_approximation``. Detail plane j
    captures structure at spatial scale ~2**(j+1) px; the coarse
    approximation after ``n_scales`` levels has (by construction, not
    approximately) no structure finer than ~2**n_scales px.
    """
    c = img.astype(np.float64)
    details = []
    for j in range(n_scales):
        c_next = _atrous_convolve(c, step=1 << j)
        details.append(c - c_next)
        c = c_next
    return details, c


def _wavelet_background_surface(coords: np.ndarray, values: np.ndarray,
                                H: int, W: int, patch_size: int,
                                n_scales: int, outlier_sigma: float = 2.5,
                                max_iter: int = 5, verbose: bool = False) -> np.ndarray:
    """Background surface via starlet low-pass instead of a Gaussian blur tail.

    Reuses the exact same robust local-linear regression as DBE
    (``_dbe_regression_coarse_grid``) to reconstruct a smooth field from the
    sparse sky-patch samples -- a naive linear interpolation between ~50-200
    sparse patches (the alternative) can't be decomposed into meaningful
    sub-patch-scale wavelet planes, since it has no real information at that
    resolution to begin with. Only the finishing step differs from DBE:
    instead of one Gaussian blur (soft tail, some sample influence leaks
    past its nominal radius), a starlet decomposition of the regression
    grid keeps just the coarsest approximation -- a hard, construction-time
    guarantee that nothing below ~2**n_scales grid cells survives, so a
    compact-ish nebula sampled as "background" by mistake bleeds less into
    neighbouring genuine sky than a Gaussian tail would.
    """
    if len(coords) < 6:
        return _polynomial_surface(coords, values, H, W, patch_size)

    sigma_px = Config.DBE_FIT_SIGMA_PATCHES * patch_size

    # Finer grid than DBE's own (which is coarse by design -- its Gaussian
    # support does the smoothing): a starlet needs enough cells for
    # n_scales dyadic levels to mean something below the image size.
    min_dim = min(H, W)
    target_dim = 768 if min_dim > 3000 else 512
    stride = max(1, int(round(min_dim / float(target_dim))))
    Hc = max(64, H // stride)
    Wc = max(64, W // stride)

    coarse, surf_lo, surf_hi = _dbe_regression_coarse_grid(
        coords, values, H, W, outlier_sigma, max_iter, sigma_px, Hc, Wc,
        verbose, label="Wavelet BG")

    # Clamp so the coarsest plane still spans several grid cells -- otherwise
    # it collapses to one constant value.
    max_scales = max(1, int(np.log2(max(Hc, Wc))) - 2)
    scales = int(np.clip(n_scales, 1, max_scales))

    _, coarse_approx = _starlet_transform(coarse, scales)

    if verbose:
        safe_print(f"    Wavelet BG: {scales} scales on {Hc}x{Wc} regression grid "
                   f"(cutoff ~{(1 << scales) * stride}px)")

    coarse_approx = np.clip(coarse_approx, surf_lo, surf_hi)
    surface = zoom(coarse_approx, (H / Hc, W / Wc), order=3)[:H, :W]
    return np.clip(surface, surf_lo, surf_hi)


def wavelet_background_extraction(
        rgb: np.ndarray,
        patch_size: int = Config.DBE_PATCH_SIZE,
        masked_frac_thresh: float = Config.DBE_MASKED_FRAC_THRESH,
        clip_sigma: float = 3.0,
        n_scales: int = 6,
        star_mask: Optional[np.ndarray] = None,
        verbose: bool = False,
        use_entropy_weights: bool = False,
        exclusion_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Background extraction via starlet (a-trous wavelet) low-pass fit.

    Reuses DBE's sky-patch sampling (star/nebula-safe emission mask, robust
    per-patch sigma-clip and entropy rejection) unchanged -- only the fit
    that turns those samples into a smooth 2-D surface differs from
    ``dynamic_background_extraction``. See ``_wavelet_background_surface``.
    """
    from concurrent.futures import ThreadPoolExecutor

    H, W = rgb.shape[:2]
    if H == 0 or W == 0:
        return rgb.copy()

    emission_mask, sky_med, sky_std, _dense_field = _dbe_prepare_emission_mask(
        rgb, star_mask, exclusion_mask, verbose, "Wavelet BG")

    result = np.empty_like(rgb)
    channel_names = ['Red', 'Green', 'Blue']

    def _process_channel(c):
        channel = rgb[:, :, c].astype(np.float64)
        coords, values = _sample_background_patches(
            channel, emission_mask, patch_size, masked_frac_thresh,
            sky_ref=sky_med, sky_std=sky_std,
            use_entropy_weights=use_entropy_weights)

        n = len(coords)
        if verbose:
            safe_print(f"    Wavelet BG {channel_names[c]}: {n} background patches accepted")

        if n < Config.DBE_MIN_SAMPLES:
            if verbose:
                safe_print(f"    Wavelet BG {channel_names[c]}: insufficient samples ({n}), "
                           f"falling back to "
                           f"{'sigma-clip mesh (no emission mask)' if _dense_field else 'mesh extraction'}")
            if _dense_field:
                background = extract_background(
                    channel, mesh_size=patch_size * 2,
                    clip_sigma=clip_sigma,
                    star_mask=None).astype(np.float64)
            else:
                background = extract_background(channel, mesh_size=patch_size,
                                                clip_sigma=clip_sigma,
                                                star_mask=emission_mask).astype(np.float64)
        else:
            if n > Config.DBE_MAX_SAMPLES:
                step = max(1, n // Config.DBE_MAX_SAMPLES)
                coords = coords[::step]
                values = values[::step]
                if verbose:
                    safe_print(f"    Wavelet BG {channel_names[c]}: subsampled to "
                               f"{len(coords)} patches")

            background = _wavelet_background_surface(
                coords, values, H, W, patch_size=patch_size,
                n_scales=n_scales, verbose=verbose)

        subtracted = channel - background
        if verbose:
            safe_print(f"    {channel_names[c]}: bg_median="
                       f"{float(np.median(background)):.1f}, "
                       f"subtracted_median={float(np.median(subtracted)):.1f}")
        return c, subtracted.astype(np.float32)

    with ThreadPoolExecutor(max_workers=3) as executor:
        for c, ch_result in executor.map(_process_channel, range(3)):
            result[:, :, c] = ch_result

    return result
