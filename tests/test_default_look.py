"""Default-look fixes for the targets beginners shoot first: Orion's clipped core
(io_fits._feather_clipped_core), the Horsehead's IC 434 (background._halpha_excess_mask,
dbe_halpha_mask). Andromeda's star discs: tests/test_phase4_quality.py."""
from __future__ import annotations

import numpy as np
import pytest

from src import auto_settings as a
from src.background import _halpha_excess_mask
from src.io_fits import _feather_clipped_core
from tests._helpers import auto_args as _args


def _plateau_image(H=200, W=300):
    yy, xx = np.mgrid[:H, :H if W is None else W]
    rgb = np.full((H, W, 3), 0.1, np.float32)
    core = (yy - 100) ** 2 + ((xx - 150) / 1.3) ** 2 < 35 ** 2      # ~5000 px plateau
    rgb[core] = 1.0
    for y, x in [(30, 40), (160, 260)]:                             # clipped star cores
        rgb[(yy - y) ** 2 + (xx - x) ** 2 <= 16] = 1.0
    return rgb, core


def test_feather_softens_only_a_large_clipped_core():
    rgb, core = _plateau_image()
    disp = rgb.copy()
    out = _feather_clipped_core(disp, rgb)
    from scipy import ndimage
    edge = ndimage.binary_dilation(core, iterations=2) & ~ndimage.binary_erosion(core, iterations=2)
    assert np.abs(out[edge] - disp[edge]).max() > 0.1            # the outline is blended
    far = ~ndimage.binary_dilation(core, iterations=40)
    np.testing.assert_array_equal(out[far], disp[far])           # stars, sky untouched
    deep = ndimage.binary_erosion(core, iterations=15)
    assert out[deep].min() > 0.99                                # still white inside


def test_feather_leaves_an_image_without_a_large_plateau_alone():
    rgb, core = _plateau_image()
    rgb[core] = 0.3
    disp = rgb.copy()
    assert _feather_clipped_core(disp, rgb) is disp


def _red_band_frame(H=600, W=900, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    sky = 1000.0 + 0.5 * xx                                      # neutral glare gradient
    band = 60.0 * np.exp(-((xx - yy - 150) / 80.0) ** 2)         # H-alpha: red only
    rgb = np.stack([sky * 0.9 + band, sky, sky * 1.05], axis=-1)
    rgb = rgb + rng.normal(0, 5, rgb.shape)
    return rgb.astype(np.float32), band > 30


def test_halpha_mask_finds_a_red_band_not_a_neutral_gradient():
    rgb, band = _red_band_frame()
    m = _halpha_excess_mask(rgb, 20.0, 3.0)
    assert m is not None
    assert m[band].mean() > 0.8
    assert m[~band & (np.abs(np.mgrid[:600, :900][1] - np.mgrid[:600, :900][0] - 150) > 250)].mean() < 0.02


def test_halpha_mask_off_and_on_only_for_emission_nebulae():
    from src.background import dynamic_background_extraction
    rgb, _ = _red_band_frame()
    off = dynamic_background_extraction(rgb.copy(), halpha_sigma=0.0)
    on = dynamic_background_extraction(rgb.copy(), halpha_sigma=3.0)
    _, band = _red_band_frame()
    # the red band keeps more of its light with the colour mask
    assert on[..., 0][band].mean() > off[..., 0][band].mean()

    for ttype, want in [('emission_nebula', True), ('galaxy', False), ('star_field', False)]:
        sig = {**a._TYPE_ANCHORS[ttype], 'fwhm': 3.0}
        args = _args()
        a._apply_dynamic_settings(sig, a._blend_weights(sig), args)
        assert bool(getattr(args, 'dbe_halpha_mask', False)) is want, ttype


def test_halpha_mask_respects_an_explicit_setting():
    sig = dict(a._TYPE_ANCHORS['emission_nebula'])
    args = _args(dbe_halpha_mask=False, _explicit_cli_dests={'dbe_halpha_mask'})
    a._apply_dynamic_settings(sig, a._blend_weights(sig), args)
    assert args.dbe_halpha_mask is False


@pytest.mark.parametrize('name', ['dbe_halpha_mask'])
def test_new_setting_has_a_cli_default(name):
    import sys

    from src.cli import parse_args
    argv = sys.argv
    try:
        sys.argv = ['originstack.py', '-d', '.']
        args = parse_args()
    finally:
        sys.argv = argv
    assert getattr(args, name) is False
