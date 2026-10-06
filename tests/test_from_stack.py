"""--from-stack: Phase 4 alone on an earlier run's linear main FITS."""
import os

import numpy as np
import pytest
from astropy.io import fits

from src import cli
from src.pipeline import postprocess_from_stack


def _linear_stack(path, h=160, w=200, rawstack=True, seed=0):
    rng = np.random.default_rng(seed)
    img = 500.0 + rng.normal(0, 5.0, (3, h, w))
    yy, xx = np.mgrid[:h, :w]
    for y, x in ((40, 50), (100, 150), (120, 30)):
        img += 3000.0 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 4.0)
    hdu = fits.PrimaryHDU(img.astype(np.float32))
    hdu.header['RAWSTACK'] = rawstack
    hdu.header['NFRAMES'] = 12
    hdu.writeto(path)
    return img


def _args(argv):
    args = cli.parse_args(argv)
    cli.apply_post_parse_setup(args)
    return args


def test_directory_optional_only_with_from_stack():
    with pytest.raises(SystemExit):
        cli.parse_args([])
    assert cli.parse_args(['--from-stack', 'a.fits']).directory is None


def test_runs_phase4_writes_preview_and_leaves_input_alone(tmp_path):
    src = str(tmp_path / 'm51.fits')
    _linear_stack(src)
    before = open(src, 'rb').read()
    args = _args(['--from-stack', src, '--stretch', 'arcsinh'])
    assert args.output == str(tmp_path / 'm51_reprocessed.fits')
    out = postprocess_from_stack(src, args.output, args)
    assert out.shape == (160, 200, 3)
    assert os.path.exists(str(tmp_path / 'm51_reprocessed.jpg'))
    assert open(src, 'rb').read() == before


def test_uses_the_earlier_runs_config_with_cli_overrides(tmp_path):
    src = str(tmp_path / 'm51.fits')
    _linear_stack(src)
    with open(str(tmp_path / 'm51_config.toml'), 'w') as fh:
        fh.write('stretch = "arcsinh"\nlocal_contrast = false\npreview_black_sigma = 2.5\n')
    args = _args(['--from-stack', src, '--stretch', 'ghs'])
    assert args.config.endswith('m51_config.toml')
    assert args.local_contrast is False and args.preview_black_sigma == 2.5
    assert args.stretch == 'ghs'                      # explicit flag wins


def test_refuses_a_post_processed_or_same_path_output(tmp_path):
    src = str(tmp_path / 'done.fits')
    _linear_stack(src, rawstack=False)
    args = _args(['--from-stack', src])
    with pytest.raises(ValueError, match='RAWSTACK'):
        postprocess_from_stack(src, args.output, args)
    src2 = str(tmp_path / 'ok.fits')
    _linear_stack(src2)
    with pytest.raises(ValueError, match='-o must differ'):
        postprocess_from_stack(src2, src2, args)


def test_early_phase4_cache_hit_matches_and_settings_invalidate(tmp_path, capsys):
    src = str(tmp_path / 'm33.fits')
    _linear_stack(src, seed=3)
    outs = []
    for extra in ([], ['--stretch', 'ghs'], ['--bg-method', 'mesh']):
        args = _args(['--from-stack', src] + extra)
        capsys.readouterr()
        outs.append(postprocess_from_stack(src, args.output, args))
        outs[-1] = (outs[-1], 'reused from' in capsys.readouterr().out)
    (first, hit0), (second, hit1), (_third, hit2) = outs
    assert not hit0 and hit1 and not hit2          # stretch is late; bg method is early
    assert os.path.exists(str(tmp_path / 'm33_phase4cache.pkl'))
    # a cache hit reproduces the early steps exactly (only the late ones differ here)
    args = _args(['--from-stack', src])
    again = postprocess_from_stack(src, args.output, args)
    np.testing.assert_array_equal(again, first)


def test_normal_runs_never_touch_the_cache(tmp_path):
    from src import postprocess as pp
    args = cli.parse_args(['-d', str(tmp_path)])
    assert not pp._early_cache_usable(args, None)
