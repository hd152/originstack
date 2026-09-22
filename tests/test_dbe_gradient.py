"""DBE must sample patches along a strong smooth edge gradient, not reject them."""
import numpy as np
import pytest

import src.background as bg


def _edge_glow_image(H=420, W=640, seed=0):
    """Flat sky + noise, R tinted a little above the luminance sky level with a
    broad glow rising toward the left/top edges (the real case: R +68..+111 ADU
    at the edges and enough baseline tint that most R patches sit above the
    luminance-based 'sky + 2 sigma' cut, G/B about half the glow)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    glow = 120.0 * np.clip(1.0 - xx / 260.0, 0, 1) + 90.0 * np.clip(1.0 - yy / 200.0, 0, 1)
    img = np.full((H, W, 3), 1000.0)
    img += rng.normal(0, 20, img.shape)
    img[:, :, 0] += 45.0 + glow
    img[:, :, 1] += 0.4 * glow
    img[:, :, 2] += 0.4 * glow
    return img.astype(np.float32)


def _edge_offsets(img):
    H, W = img.shape[:2]
    mid = np.median(img[H // 2 - 40:H // 2 + 40, W // 2 - 80:W // 2 + 80], axis=(0, 1))
    left = np.median(img[80:-80, :25], axis=(0, 1)) - mid
    top = np.median(img[:25, 80:-80], axis=(0, 1)) - mid
    return float(np.abs(left).max()), float(np.abs(top).max())


def _run():
    img = _edge_glow_image()
    out = bg.dynamic_background_extraction(img, patch_size=32)
    return img, out


def test_edge_glow_in_the_strongest_channel_is_removed():
    img, out = _run()
    before = _edge_offsets(img)
    after = _edge_offsets(out)
    assert before[0] > 100 and before[1] > 60          # the glow is really there
    assert after[0] < 15 and after[1] < 15


def test_without_gradient_admission_the_strong_channel_is_left_uncorrected(monkeypatch):
    """Guards the test above: with the admission pass disabled the same image
    keeps most of its R edge excess, so the assertion is a real one."""
    monkeypatch.setattr(bg, '_admit_gradient_patches',
                        lambda channel, em, ps, mf, sd, coords, values, ent, **k: (coords, values))
    img, out = _run()
    assert _edge_offsets(out)[0] > 20      # vs < 15 with the pass (about half)


def test_a_bright_object_is_not_admitted_as_gradient():
    img = _edge_glow_image()
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    img += (600.0 * np.exp(-((yy - 210) ** 2 + (xx - 330) ** 2) / (2 * 25.0 ** 2)))[..., None]
    out = bg.dynamic_background_extraction(img, patch_size=32)
    # the bump is nowhere near the edges; the DBE surface must not have absorbed it
    assert float(out[210, 330].mean()) - float(np.median(out[100:140, 250:400].mean(axis=2))) > 300.0


def test_emission_mask_disk_dilation_matches_scipy_binary_dilation():
    """`_build_emission_mask`'s compact-source dilation switched from
    `scipy.ndimage.binary_dilation` with a disk structuring element to a
    distance-transform threshold (same result, much faster on a large
    contiguous bright region -- see background.py's comment at the call
    site). Pin the two as exactly equivalent directly, not just through the
    higher-level DBE tests above, since a single wrong offset-by-one in the
    distance-transform version would still likely pass those."""
    from scipy.ndimage import binary_dilation, distance_transform_edt

    H, W = 300, 400
    yy, xx = np.indices((H, W))
    # a contiguous elliptical blob, not scattered noise -- this is exactly
    # the shape that made scipy's generic binary_dilation pathologically
    # slow (a real bright galaxy/comet core), and is what actually exercises
    # the interior of the dilated region, not just its edge.
    src_binary = (((yy - 150) / 40.0) ** 2 + ((xx - 200) / 60.0) ** 2 <= 1.0).astype(np.uint8)
    r = 25
    y_idx, x_idx = np.ogrid[-r:r + 1, -r:r + 1]
    structure = (y_idx ** 2 + x_idx ** 2 <= r ** 2).astype(np.uint8)

    want = binary_dilation(src_binary, structure=structure)
    got = distance_transform_edt(1 - src_binary) <= r

    np.testing.assert_array_equal(got, want)


def test_gaussian_filter_ds_bilinear_upsample_matches_full_resolution():
    """gaussian_filter_ds's large-sigma path downsamples, blurs, then
    upsamples back -- its own upsample step switched from cubic (order=3) to
    bilinear (order=1) interpolation, ~3x faster, on the reasoning that a
    coarse grid which just came out of a huge Gaussian blur has nothing left
    for cubic's extra curvature term to recover. Check that claim directly:
    the ds-optimized result should stay close to a real full-resolution
    gaussian_filter at the same sigma, which is the property the whole
    function exists to preserve regardless of interpolation order."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(6)
    H, W = 300, 400
    field = rng.normal(1000.0, 20.0, (H, W))
    sigma = 40.0  # > ds_threshold=24.0, exercises the downsampled path

    exact = gaussian_filter(field, sigma=sigma)
    fast = bg.gaussian_filter_ds(field, sigma)

    diff = np.abs(exact - fast)
    assert float(diff.mean()) < 0.5           # field std is ~20 -- this is noise-floor small
    assert float(diff.max()) < 5.0
