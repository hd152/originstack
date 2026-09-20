"""Layered preview stretch, starless_from_image, FWHM fallback, combined-stack
helpers, and the hierarchical reference choice."""
import argparse

import numpy as np

from src.io_fits import render_preview_layered_uint8, render_preview_uint8, save_preview_rgb
from src.postprocess import _median_fwhm
from src.star_removal import starless_from_image


def _galaxy_with_stars(seed=0, H=160):
    """Faint smooth 'galaxy' disk + a few very bright stars + sky noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :H]
    img = np.full((H, H, 3), 1000.0, np.float32)
    disk = 300.0 * np.exp(-(((yy - 80) ** 2 + (xx - 80) ** 2) / (2 * 30.0 ** 2)))
    img += disk[..., None]
    for cy, cx in ((25, 30), (130, 40), (40, 130), (120, 120), (70, 20)):
        img += (30000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.6 ** 2)))[..., None]
    return (img + rng.normal(0, 20, img.shape)).astype(np.float32)


def test_starless_from_image_removes_stars_and_keeps_shape():
    img = _galaxy_with_stars()
    sl = starless_from_image(img)
    assert sl is not None and sl.shape == img.shape
    assert sl[25, 30].max() < 0.2 * img[25, 30].max()          # a star came out
    np.testing.assert_allclose(sl[80, 80], img[80, 80], rtol=0.05)   # galaxy untouched


def test_starless_from_image_declines_without_stars():
    flat = np.random.default_rng(1).normal(1000, 5, (64, 64, 3)).astype(np.float32)
    assert starless_from_image(flat) is None
    assert starless_from_image(np.zeros((8, 8), np.float32)) is None


def test_layered_stretch_gives_faint_structure_more_of_the_range():
    img = _galaxy_with_stars()
    sl = starless_from_image(img)
    kw = dict(ghs_b=8.0, ghs_sp=0.15, ghs_hp=0.95, black_sigma=0.0)
    plain = render_preview_uint8(img, stretch='ghs', **kw)
    layered = render_preview_layered_uint8(img, sl, **kw)
    assert layered.shape == plain.shape and layered.dtype == np.uint8
    disk = (slice(60, 100), slice(60, 100))
    assert layered[disk].mean() > plain[disk].mean() + 10   # disk brighter, not lost


def test_save_preview_layered_only_with_ghs_and_matching_shape(tmp_path):
    img = _galaxy_with_stars()
    sl = starless_from_image(img)
    a, b, c = (str(tmp_path / n) for n in ('a.jpg', 'b.jpg', 'c.jpg'))
    save_preview_rgb(img, a, stretch='ghs', starless=sl)
    save_preview_rgb(img, b, stretch='ghs')
    save_preview_rgb(img, c, stretch='linear', starless=sl)     # ignored: not ghs
    assert open(a, 'rb').read() != open(b, 'rb').read()
    save_preview_rgb(img, str(tmp_path / 'd.jpg'), stretch='ghs', starless=sl[:10])  # bad shape: no crash


class _F:
    def __init__(self, fwhm):
        self.metrics = {'fwhm': fwhm} if fwhm is not None else None


def test_median_fwhm_default_is_actually_used():
    assert _median_fwhm([]) == 4.0
    assert _median_fwhm(None) == 4.0
    assert _median_fwhm([_F(None), _F(0)]) == 4.0
    assert _median_fwhm([_F(3.0), _F(5.0), _F(0)]) == 4.0       # median of 3 and 5
    assert _median_fwhm([_F(6.5)]) == 6.5
    assert not np.isnan(_median_fwhm([]))                        # the old bug: NaN


def test_postprocess_combined_runs_phase4_with_the_reference_args(monkeypatch):
    import src.cli as cli
    import src.postprocess as pp
    seen = {}

    def fake(stacked, args, final, stats):
        seen.update(output=args.output, diag=args._diagnostic_dir, final=final,
                    same_array=stacked)
        return stacked * 2.0

    monkeypatch.setattr(pp, 'postprocess_stack', fake)
    eff = argparse.Namespace(stack_method='sigma_clip', galaxy_mode=True,
                             output='old', _diagnostic_dir='x')
    combined = np.ones((4, 4, 3), np.float32)
    out = cli._postprocess_combined(combined, eff, 'new.fits')
    assert float(out[0, 0, 0]) == 2.0
    assert seen['output'] == 'new.fits' and seen['diag'] is None and seen['final'] == []
    assert seen['same_array'] is not combined            # linear stack must stay untouched
    assert eff.output == 'old'                            # caller's args not mutated
    assert float(combined[0, 0, 0]) == 1.0
