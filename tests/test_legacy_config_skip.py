"""Configs saved before 2.11.9 lack skip_step; a --from-stack re-run must still skip
sky_residual for an extended target (or it fits a galaxy away as sky), and --debug
diagnostic works with --from-stack."""
import argparse

import pytest

from src.cli import load_config_file, parse_args


def _load(tmp_path, body, *flags):
    cfg = tmp_path / 'old_config.toml'
    cfg.write_text(body, encoding='utf-8')
    args = parse_args(['-d', 'x', '-o', 'y.fits', *flags])
    changes = load_config_file(str(cfg), args)
    return args, changes


@pytest.mark.parametrize('body', ['galaxy_mode = true\n', 'dbe_diffuse_threshold_sigma = 0.25\n'])
def test_old_extended_target_config_gets_the_sky_residual_skip(tmp_path, body):
    args, changes = _load(tmp_path, body)
    assert 'sky_residual' in args.skip_step
    assert any('sky_residual' in c for c in changes)


@pytest.mark.parametrize('body', [
    'galaxy_mode = true\nskip_step = []\n',             # 2.11.9+: the skip list is recorded as it was
    'galaxy_mode = false\n',                             # not an extended target
    'dbe_diffuse_threshold_sigma = 1.0\n',
])
def test_left_alone_otherwise(tmp_path, body):
    args, _ = _load(tmp_path, body)
    assert 'sky_residual' not in args.skip_step


def test_explicit_skip_step_wins(tmp_path):
    args, _ = _load(tmp_path, 'galaxy_mode = true\n', '--skip-step', 'edge_bands')
    assert args.skip_step == ['edge_bands']


def test_no_duplicate_skip(tmp_path):
    args, _ = _load(tmp_path, 'galaxy_mode = true\nskip_step = ["sky_residual"]\n')
    assert args.skip_step.count('sky_residual') == 1


def test_diagnostic_dir_is_set_for_from_stack_too(tmp_path):
    from src.pipeline import _set_diagnostic_dir
    args = argparse.Namespace(diagnostic=True, diagnostic_dir=None)
    _set_diagnostic_dir(args, str(tmp_path / 'out.fits'), (10, 12, 3))
    assert args._diagnostic_dir == str(tmp_path / 'out_diagnostic')
    assert (tmp_path / 'out_diagnostic').is_dir()
    off = argparse.Namespace(diagnostic=False)
    _set_diagnostic_dir(off, str(tmp_path / 'o.fits'), (10, 12, 3))
    assert off._diagnostic_dir is None
