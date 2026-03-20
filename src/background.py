"""Background extraction and sky normalization."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

from src.models import Config
from src.utils import safe_print

try:
    from astropy.stats import sigma_clipped_stats
except Exception:
    sigma_clipped_stats = None


def _estimate_sky_sigma(img: np.ndarray) -> float:
    """Estimate per-pixel sky noise from adjacent-pixel diffs on sky-only pairs.

    Uses the green channel, restricts to pixel pairs that are both positive
    and below the 80th percentile (excludes stars, nebula, and the negative
    half of the background).  Returns sigma in the same ADU units as img.
    """
    img_max = float(img.max())
    G = img[:, :, 1].astype(np.float64)
    pos_g = G[G > 0]
    if pos_g.size < 100:
        return max(img_max * 1e-4, 1.0)
    p80 = float(np.percentile(pos_g, 80))
    lft, rgt = G[:, :-1], G[:, 1:]
    tp,  bot = G[:-1, :], G[1:, :]
    msk_h = (lft > 0) & (rgt > 0) & (lft < p80) & (rgt < p80)
    msk_v = (tp  > 0) & (bot > 0) & (tp  < p80) & (bot < p80)
    diffs = np.concatenate([(rgt - lft)[msk_h], (bot - tp)[msk_v]])
    if diffs.size < 1000:
        return max(img_max * 1e-4, 1.0)
    raw = float(np.median(np.abs(diffs))) / (0.6745 * np.sqrt(2))
    return max(raw, img_max * 1e-5)


def extract_background(img: np.ndarray, mesh_size: int = 256, filter_size: int = 3,
                       clip_sigma: float = 3.0, clip_iters: int = 5,
                       star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate smooth sky background using mesh-based sigma-clipped statistics.

    Divides image into a grid, computes sigma-clipped median in each cell
    (rejecting stars via both sigma-clipping and optional star mask), rejects
    cells contaminated by extended bright objects (nebulae), then interpolates
    the clean grid to a smooth background model.

    Args:
        img: 2D image array (single channel).
        mesh_size: Size of each grid cell in pixels.
        filter_size: Median filter size applied to the mesh grid.
        clip_sigma: Sigma threshold for iterative clipping within each cell.
        clip_iters: Maximum iterations for sigma clipping.
        star_mask: Optional float mask (0=bg, 1=star) to exclude star pixels.
    """
    H, W = img.shape
    ny = max(1, H // mesh_size)
    nx = max(1, W // mesh_size)

    cell_h = H / ny
    cell_w = W / nx

    bg_grid = np.zeros((ny, nx), dtype=np.float64)

    for iy in range(ny):
        y0 = int(round(iy * cell_h))
        y1 = min(int(round((iy + 1) * cell_h)), H)
        for ix in range(nx):
            x0 = int(round(ix * cell_w))
            x1 = min(int(round((ix + 1) * cell_w)), W)

            cell = img[y0:y1, x0:x1].ravel()

            # Mask out star/galaxy pixels if star_mask provided
            if star_mask is not None:
                sm = star_mask[y0:y1, x0:x1].ravel()
                masked_frac = float(np.mean(sm >= 0.5))
                bg_pixels = cell[sm < 0.5]
                if bg_pixels.size > 10:
                    cell = bg_pixels
                elif masked_frac > 0.5:
                    # Cell is mostly galaxy/nebula — mark as NaN for later
                    # interpolation rather than using contaminated pixels.
                    bg_grid[iy, ix] = np.nan
                    continue

            if cell.size == 0:
                bg_grid[iy, ix] = 0.0
                continue

            if sigma_clipped_stats is not None:
                try:
                    _, median_val, _ = sigma_clipped_stats(
                        cell, sigma=clip_sigma, maxiters=clip_iters)
                    bg_grid[iy, ix] = float(median_val)
                    continue
                except Exception:
                    pass

            # Manual sigma-clipping fallback
            clipped = cell.copy()
            for _ in range(clip_iters):
                med = np.median(clipped)
                std = np.std(clipped)
                if std < 1e-12:
                    break
                mask = np.abs(clipped - med) < clip_sigma * std
                if not np.any(mask):
                    break
                clipped = clipped[mask]
            bg_grid[iy, ix] = float(np.median(clipped))

    # Reject grid cells contaminated by extended objects (nebulae/galaxies).
    # Outlier cells (both bright AND anomalously negative) are replaced by
    # fitting a 2D polynomial to the clean cells and evaluating it at the
    # outlier positions.  This correctly extrapolates chromatic sky gradients
    # into the masked region.
    # Also reject NaN cells (from mostly-masked galaxy regions).
    nan_mask = np.isnan(bg_grid)
    if sigma_clipped_stats is not None:
        try:
            finite_vals = bg_grid[~nan_mask].ravel() if np.any(~nan_mask) else bg_grid.ravel()
            _, _gm, _gs = sigma_clipped_stats(finite_vals, sigma=3.0, maxiters=5)
            grid_median = float(_gm)
            grid_std = float(_gs)
        except Exception:
            grid_median = float(np.nanmedian(bg_grid))
            grid_std = float(np.nanstd(bg_grid))
    else:
        grid_median = float(np.nanmedian(bg_grid))
        grid_std = float(np.nanstd(bg_grid))
    # Symmetric outlier rejection: reject both bright AND dim cells
    outlier_mask = nan_mask.copy()
    if grid_std > 1e-6:
        bright_thresh = grid_median + 2.5 * grid_std
        dim_thresh = grid_median - 2.5 * grid_std
        outlier_mask |= (bg_grid > bright_thresh) | (bg_grid < dim_thresh)
    if np.any(outlier_mask) and not np.all(outlier_mask):
        iy_good, ix_good = np.where(~outlier_mask)
        vals_good = bg_grid[iy_good, ix_good]
        # Normalised coordinates for numerical stability
        y_good = (iy_good.astype(float) + 0.5) / ny
        x_good = (ix_good.astype(float) + 0.5) / nx
        # Build polynomial design matrix for gap-filling.
        # Degree-3 (10 terms) better captures the steep-then-flat radial
        # falloff of LP gradients and IFN halos than a degree-2 parabola.
        # Fall back to degree-2 (6 terms) when there are too few clean cells
        # to constrain a 10-parameter fit reliably (need >= 30 points).
        def poly3_features(y, x):
            return np.column_stack([
                np.ones(len(y)), y, x,
                y ** 2, y * x, x ** 2,
                y ** 3, y ** 2 * x, y * x ** 2, x ** 3])

        def poly2_features(y, x):
            return np.column_stack([
                np.ones(len(y)), y, x, y ** 2, y * x, x ** 2])

        poly_features = poly3_features if len(y_good) >= 30 else poly2_features
        A_good = poly_features(y_good, x_good)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_good, vals_good, rcond=None)
            iy_bad, ix_bad = np.where(outlier_mask)
            y_bad = (iy_bad.astype(float) + 0.5) / ny
            x_bad = (ix_bad.astype(float) + 0.5) / nx
            bg_grid[outlier_mask] = poly_features(y_bad, x_bad).dot(coeffs)
        except Exception:
            bg_grid[outlier_mask] = grid_median
    elif np.all(outlier_mask):
        # All cells are outliers — fill with zero (no background to extract)
        bg_grid[:] = 0.0

    # Smooth the grid to reject remaining anomalous cells.
    # Skip for small grids (< 12 cells on shortest side): the 3x3 median
    # filter with reflect-mode padding biases edge/corner cells toward
    # interior values, systematically overestimating the background at
    # image edges and creating mottled residuals after subtraction.
    if filter_size > 1 and min(ny, nx) >= max(filter_size, 12):
        bg_grid = ndimage.median_filter(bg_grid, size=filter_size)

    # Gaussian smooth the grid to eliminate abrupt cell-to-cell transitions.
    # Without this, the cubic spline rings around sharp jumps between
    # adjacent grid cells, producing visible wave-like artifacts at the mesh
    # scale.  A sigma of 0.8 cells blends neighbours gently while preserving
    # the large-scale gradient structure.
    if min(ny, nx) >= 4:
        bg_grid = ndimage.gaussian_filter(bg_grid.astype(np.float64), sigma=0.8)

    # Interpolate grid back to full image resolution.
    # Extend grid by one cell on each side using linear extrapolation so the
    # spline only *interpolates* (never extrapolates) across the full image.
    # Without this, cubic spline extrapolation at image edges overshoots,
    # and the subsequent hard clamp creates flat patches -> mottled background.
    grid_y = np.array([(i + 0.5) * cell_h for i in range(ny)])
    grid_x = np.array([(j + 0.5) * cell_w for j in range(nx)])

    if ny >= 2 and nx >= 2:
        ext_grid = np.zeros((ny + 2, nx + 2), dtype=np.float64)
        ext_grid[1:-1, 1:-1] = bg_grid

        # Linearly extrapolate top/bottom rows
        dy = grid_y[1] - grid_y[0]
        ext_grid[0, 1:-1] = bg_grid[0, :] + (bg_grid[0, :] - bg_grid[1, :]) * (grid_y[0] / dy)
        dy = grid_y[-1] - grid_y[-2]
        ext_grid[-1, 1:-1] = bg_grid[-1, :] + (bg_grid[-1, :] - bg_grid[-2, :]) * ((H - 1 - grid_y[-1]) / dy)

        # Linearly extrapolate left/right columns
        dx = grid_x[1] - grid_x[0]
        ext_grid[1:-1, 0] = bg_grid[:, 0] + (bg_grid[:, 0] - bg_grid[:, 1]) * (grid_x[0] / dx)
        dx = grid_x[-1] - grid_x[-2]
        ext_grid[1:-1, -1] = bg_grid[:, -1] + (bg_grid[:, -1] - bg_grid[:, -2]) * ((W - 1 - grid_x[-1]) / dx)

        # Corners: average of adjacent edge values
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
        ky = min(3, ny - 1)
        kx = min(3, nx - 1)
        spline = RectBivariateSpline(grid_y, grid_x, bg_grid, kx=kx, ky=ky)

    background = spline(np.arange(H), np.arange(W)).astype(np.float32)

    # Final Gaussian blur at half-mesh scale suppresses any residual
    # mesh-frequency ripple from spline interpolation.  This is the
    # minimum spatial frequency the mesh can represent, so blurring at
    # this scale cannot remove real sky structure — only grid artifacts.
    blur_sigma = cell_h * 0.5
    background = ndimage.gaussian_filter(background, sigma=blur_sigma)

    return background


def apply_background_extraction(rgb: np.ndarray, mesh_size: int = 256,
                                filter_size: int = 3, clip_sigma: float = 3.0,
                                verbose: bool = False,
                                star_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply per-channel background extraction with automatic extended-source masking.

    Detects any large extended source (galaxy/nebula) in the image by smoothing
    strongly and finding the brightest region, then masks it out so the background
    model is only fit to true sky pixels.  Per-channel subtraction is always used
    so that chromatic sky gradients are fully removed.
    """
    H, W = rgb.shape[:2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # --- Auto-detect extended source (galaxy/nebula) and build exclusion mask ---
    combined_mask = star_mask.copy().astype(np.float32) if star_mask is not None else None

    try:
        # Moderate smoothing: removes stars (PSF ~3px) but preserves galaxy shape
        smooth_sigma = max(20.0, min(H, W) / 50.0)
        lum_smooth = ndimage.gaussian_filter(lum, sigma=smooth_sigma)

        # Sky reference from the image border (avoids galaxy near centre)
        by = max(10, int(H * Config.BORDER_FRAC))
        bx = max(10, int(W * Config.BORDER_FRAC))
        border_pix = np.concatenate([
            lum_smooth[:by, :].ravel(), lum_smooth[-by:, :].ravel(),
            lum_smooth[by:-by, :bx].ravel(), lum_smooth[by:-by, -bx:].ravel(),
        ])
        sky_med = float(np.median(border_pix))
        sky_std = float(np.std(border_pix))

        peak_y, peak_x = np.unravel_index(int(np.argmax(lum_smooth)), (H, W))
        peak_val = float(lum_smooth[peak_y, peak_x])

        # Detect extended source: peak must be > 5-sigma above border sky
        # AND the bright region (> sky + 4sigma) must cover > 0.5% of image.
        # Using strict sigma guard prevents LP-gradient peaks and bright
        # isolated stars from being falsely treated as extended sources.
        detect_thresh = sky_med + 5.0 * max(sky_std, 1.0)
        frac_bright = float(np.mean(lum_smooth > detect_thresh))
        if peak_val > detect_thresh and frac_bright > 0.005:
            # Exclusion radius: 30% of shorter image dimension, centred on peak
            excl_radius = int(min(H, W) * 0.30)
            yy, xx = np.mgrid[:H, :W]

            # Detect up to 3 extended sources (handles galaxy pairs/groups like
            # Markarian's Chain where multiple bright galaxies span the field).
            # Additional sources must be nearly as bright as the primary to
            # avoid false positives from light-pollution gradient peaks.
            remaining_lum = lum_smooth.copy()
            n_sources = 0
            primary_peak = peak_val
            for _ in range(3):
                py, px = np.unravel_index(int(np.argmax(remaining_lum)), (H, W))
                pv = float(remaining_lum[py, px])
                if pv <= detect_thresh:
                    break
                # Secondary/tertiary sources must be at least 50% as bright
                # (above sky) as the primary to avoid gradient false positives
                if n_sources > 0:
                    primary_excess = primary_peak - sky_med
                    current_excess = pv - sky_med
                    if primary_excess > 0 and current_excess < 0.5 * primary_excess:
                        break
                dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
                galaxy_mask = (dist < excl_radius).astype(np.float32)
                if combined_mask is None:
                    combined_mask = galaxy_mask
                else:
                    np.clip(combined_mask + galaxy_mask, 0, 1, out=combined_mask)
                # Blank out this source so next iteration finds a different peak
                remaining_lum[dist < excl_radius] = float(np.min(remaining_lum))
                n_sources += 1
                if verbose:
                    n_masked = int(np.sum(galaxy_mask > 0.5))
                    safe_print(f"    Galaxy mask #{n_sources}: centre=({px},{py}), "
                               f"radius={excl_radius}px, "
                               f"{100. * n_masked / H / W:.1f}% masked")
    except Exception:
        pass

    # --- Per-channel background subtraction (handles chromatic gradients) ---
    # Estimate all three channels in parallel — they are independent.
    result = np.empty_like(rgb)
    channel_names = ['Red', 'Green', 'Blue']
    bg_channels = [None, None, None]

    def _extract_bg_channel(c):
        return c, extract_background(rgb[:, :, c], mesh_size=mesh_size,
                                     filter_size=filter_size, clip_sigma=clip_sigma,
                                     star_mask=combined_mask)

    with ThreadPoolExecutor(max_workers=3) as executor:
        for c, bg in executor.map(_extract_bg_channel, range(3)):
            bg_channels[c] = bg

    for c in range(3):
        bg = bg_channels[c]
        subtracted = rgb[:, :, c] - bg
        # Do NOT clip to 0 here.  Clipping converts the negative half of the
        # Gaussian sky noise into exact zeros, creating large patches of
        # identical zero-valued pixels (40-50 % of sky) that appear as
        # "leopard print" in any linear FITS viewer.  Negative sky values are
        # correct and are handled by PixInsight, Siril, DS9, etc.
        result[:, :, c] = subtracted
        if verbose:
            safe_print(f"    {channel_names[c]}: bg_median="
                       f"{float(np.median(bg)):.1f}, "
                       f"subtracted median={float(np.median(subtracted)):.1f}")

    return result


def remove_sky_residual(img: np.ndarray, mesh_size: int = 128,
                       filter_size: int = 3, clip_sigma: float = 3.0,
                       star_mask: Optional[np.ndarray] = None,
                       verbose: bool = False) -> np.ndarray:
    """Remove smooth sky residuals revealed by denoising.

    Background extraction leaves residuals of ~10-25 ADU at the mesh scale.
    Before denoising, per-pixel noise masks these residuals.  After wavelet
    or other denoising reduces noise by 10-50x, the residuals dominate and
    appear as a mottled "leopard print" pattern in both FITS viewers and
    stretched previews.

    Includes automatic extended-source detection so that nebula/galaxy
    emission is not mistaken for a sky residual and subtracted.
    """
    H, W = img.shape[:2]
    ny = max(1, H // mesh_size)
    nx = max(1, W // mesh_size)
    cell_h = H / ny
    cell_w = W / nx

    # --- Detect extended sources and build a cell-level exclusion mask ---
    # This prevents nebula/galaxy emission from being treated as sky residual.
    lum = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1]
           + 0.114 * img[:, :, 2])
    cell_excluded = np.zeros((ny, nx), dtype=bool)
    try:
        smooth_sigma = max(20.0, min(H, W) / 50.0)
        lum_smooth = ndimage.gaussian_filter(lum, sigma=smooth_sigma)
        by = max(10, int(H * Config.BORDER_FRAC))
        bx = max(10, int(W * Config.BORDER_FRAC))
        border_pix = np.concatenate([
            lum_smooth[:by, :].ravel(), lum_smooth[-by:, :].ravel(),
            lum_smooth[by:-by, :bx].ravel(), lum_smooth[by:-by, -bx:].ravel(),
        ])
        sky_med = float(np.median(border_pix))
        sky_std = float(np.std(border_pix))
        detect_thresh = sky_med + 5.0 * max(sky_std, 1.0)
        frac_bright = float(np.mean(lum_smooth > detect_thresh))
        if frac_bright > 0.005:
            peak_y, peak_x = np.unravel_index(
                int(np.argmax(lum_smooth)), (H, W))
            peak_val = float(lum_smooth[peak_y, peak_x])
            if peak_val > detect_thresh:
                excl_radius = int(min(H, W) * 0.30)
                # Mark grid cells whose centres fall inside the exclusion zone
                remaining_lum = lum_smooth.copy()
                primary_peak = peak_val
                for _src_i in range(3):
                    py, px = np.unravel_index(
                        int(np.argmax(remaining_lum)), (H, W))
                    pv = float(remaining_lum[py, px])
                    if pv <= detect_thresh:
                        break
                    if _src_i > 0:
                        primary_excess = primary_peak - sky_med
                        current_excess = pv - sky_med
                        if primary_excess > 0 and current_excess < 0.5 * primary_excess:
                            break
                    for iy in range(ny):
                        cy = (iy + 0.5) * cell_h
                        for ix in range(nx):
                            cx = (ix + 0.5) * cell_w
                            if np.sqrt((cy - py) ** 2 + (cx - px) ** 2) < excl_radius:
                                cell_excluded[iy, ix] = True
                    remaining_lum[np.sqrt(
                        (np.mgrid[:H, :W][0] - py) ** 2 +
                        (np.mgrid[:H, :W][1] - px) ** 2) < excl_radius] = float(
                        np.min(remaining_lum))
    except Exception:
        pass

    result = np.empty_like(img, dtype=np.float32)
    channel_names = ['Red', 'Green', 'Blue']

    for c in range(3):
        ch = img[:, :, c].astype(np.float64)
        bg_grid = np.zeros((ny, nx), dtype=np.float64)

        for iy in range(ny):
            y0 = int(round(iy * cell_h))
            y1 = min(int(round((iy + 1) * cell_h)), H)
            for ix in range(nx):
                x0 = int(round(ix * cell_w))
                x1 = min(int(round((ix + 1) * cell_w)), W)

                # Skip cells in the extended source exclusion zone —
                # their emission must not be treated as sky residual.
                if cell_excluded[iy, ix]:
                    bg_grid[iy, ix] = np.nan
                    continue

                cell = ch[y0:y1, x0:x1].ravel()

                # Mask out star pixels
                if star_mask is not None:
                    sm = star_mask[y0:y1, x0:x1].ravel()
                    bg_pixels = cell[sm < 0.5]
                    if bg_pixels.size > 10:
                        cell = bg_pixels

                if sigma_clipped_stats is not None:
                    try:
                        _, med_val, _ = sigma_clipped_stats(
                            cell, sigma=clip_sigma, maxiters=5)
                        bg_grid[iy, ix] = float(med_val)
                        continue
                    except Exception:
                        pass
                bg_grid[iy, ix] = float(np.median(cell))

        # Fill excluded cells by copying from their nearest valid sky cell.
        # Setting them to 0 would collapse the background model to zero inside
        # the target, creating a residual halo after subtraction.
        nan_mask = np.isnan(bg_grid)
        if nan_mask.any() and not nan_mask.all():
            _, nn_idx = ndimage.distance_transform_edt(nan_mask, return_indices=True)
            bg_grid[nan_mask] = bg_grid[nn_idx[0][nan_mask], nn_idx[1][nan_mask]]
        elif nan_mask.all():
            bg_grid[:] = 0.0

        # Smooth grid to suppress star contamination between cells
        if filter_size > 1 and min(ny, nx) >= filter_size:
            bg_grid = ndimage.median_filter(bg_grid, size=filter_size)

        # Gaussian smooth to eliminate cell-to-cell discontinuities
        if min(ny, nx) >= 4:
            bg_grid = ndimage.gaussian_filter(bg_grid.astype(np.float64), sigma=0.8)

        # Interpolate to full resolution with edge-extended grid
        grid_y = np.array([(i + 0.5) * cell_h for i in range(ny)])
        grid_x = np.array([(j + 0.5) * cell_w for j in range(nx)])
        if ny >= 2 and nx >= 2:
            ext_grid = np.zeros((ny + 2, nx + 2), dtype=np.float64)
            ext_grid[1:-1, 1:-1] = bg_grid
            dy = grid_y[1] - grid_y[0]
            ext_grid[0, 1:-1] = bg_grid[0, :] + (bg_grid[0, :] - bg_grid[1, :]) * (grid_y[0] / dy)
            dy = grid_y[-1] - grid_y[-2]
            ext_grid[-1, 1:-1] = bg_grid[-1, :] + (bg_grid[-1, :] - bg_grid[-2, :]) * ((H - 1 - grid_y[-1]) / dy)
            dx = grid_x[1] - grid_x[0]
            ext_grid[1:-1, 0] = bg_grid[:, 0] + (bg_grid[:, 0] - bg_grid[:, 1]) * (grid_x[0] / dx)
            dx = grid_x[-1] - grid_x[-2]
            ext_grid[1:-1, -1] = bg_grid[:, -1] + (bg_grid[:, -1] - bg_grid[:, -2]) * ((W - 1 - grid_x[-1]) / dx)
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
            ky = min(3, ny - 1)
            kx = min(3, nx - 1)
            spline = RectBivariateSpline(grid_y, grid_x, bg_grid, kx=kx, ky=ky)
        background = spline(np.arange(H), np.arange(W)).astype(np.float64)

        # Blur at half-mesh scale to suppress spline mesh-frequency ripple
        blur_sigma = cell_h * 0.5
        background = ndimage.gaussian_filter(background, sigma=blur_sigma)

        result[:, :, c] = (ch - background).astype(np.float32)
        if verbose:
            safe_print(f"    {channel_names[c]}: residual median="
                       f"{float(np.median(background)):.2f}, "
                       f"range=[{float(background.min()):.1f}, "
                       f"{float(background.max()):.1f}]")

    return result


def sky_floor_normalize(rgb: np.ndarray,
                        star_mask: np.ndarray = None,
                        verbose: bool = False) -> np.ndarray:
    """Drive the constant sky pedestal to zero without touching sources.

    After background extraction the sky is spatially flat but may sit on a
    non-zero pedestal (residual bias, uncorrected dark, or an extraction that
    only removed gradient variation).  This function:

      1. Builds a comprehensive *source mask* — every pixel that is not pure
         sky: stars (passed in via ``star_mask``), plus anything above the sky
         noise floor estimated from the outermost image border.
      2. Estimates a per-channel sky floor as the sigma-clipped median of the
         unmasked sky pixels.
      3. Subtracts that constant per-channel floor from the entire image.

    Because the floor is a single scalar per channel, it cannot distort the
    galaxy shape or wash out extended emission — it simply shifts all values
    down so that true sky reads zero.

    Parameters
    ----------
    rgb : (H, W, 3) float32 array  — calibrated, background-extracted stack.
    star_mask : (H, W) float32, optional  — 1 where stars detected, 0 elsewhere.
    verbose : bool

    Returns
    -------
    (H, W, 3) float32 with sky floor subtracted; negatives clipped to 0.
    """
    from scipy.ndimage import gaussian_filter as _gf

    H, W = rgb.shape[:2]
    result = rgb.copy()

    lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
           + 0.114 * rgb[:, :, 2])

    # --- Build source mask ---
    # Step 1: start with the star mask if provided.
    src_mask = np.zeros((H, W), dtype=np.float32)
    if star_mask is not None:
        np.clip(src_mask + star_mask, 0, 1, out=src_mask)

    # Step 2: estimate sky noise from the outermost border strip —
    # these pixels are furthest from the target and contain the least emission.
    by = max(10, int(H * Config.BORDER_FRAC))
    bx = max(10, int(W * Config.BORDER_FRAC))
    border_lum = np.concatenate([
        lum[:by, :].ravel(), lum[-by:, :].ravel(),
        lum[by:-by, :bx].ravel(), lum[by:-by, -bx:].ravel(),
    ])
    # Use sigma-clipped stats to reject stars in the border strip
    if sigma_clipped_stats is not None:
        try:
            _, sky_med, sky_std = sigma_clipped_stats(
                border_lum, sigma=3.0, maxiters=5)
            sky_med = float(sky_med)
            sky_std = float(sky_std)
        except Exception:
            sky_med = float(np.median(border_lum))
            sky_std = float(np.std(border_lum))
    else:
        sky_med = float(np.median(border_lum))
        sky_std = float(np.std(border_lum))

    # Step 3: mask every pixel where the *smoothed* luminance exceeds the sky
    # floor by more than 2x the sky noise — this catches the galaxy core,
    # spiral arms, IFN halo, and faint nebulosity.  Using the Gaussian-smoothed
    # luminance prevents individual noisy pixels from leaking through.
    smooth_sigma = max(5.0, min(H, W) / 200.0)
    lum_smooth = _gf(lum, sigma=smooth_sigma)
    src_thresh = sky_med + 2.0 * max(sky_std, 1.0)
    np.clip(src_mask + (lum_smooth > src_thresh).astype(np.float32), 0, 1,
            out=src_mask)

    sky_pix_mask = src_mask < 0.5     # True where it is pure sky

    if sky_pix_mask.sum() < 1000:
        # Extremely crowded field or failed detection — skip safely
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

    # Clip negative sky to zero — we never want the background to go below black.
    np.clip(result, 0, None, out=result)

    if verbose:
        safe_print(f"  Sky floor: {sky_frac:.1f}% sky pixels used, "
                   f"floors subtracted R={floors[0]:.2f} G={floors[1]:.2f} "
                   f"B={floors[2]:.2f} ADU")

    return result
