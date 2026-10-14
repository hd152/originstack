"""save_effective_config must write TOML that load_config can read back."""
import argparse
import tomllib

import pytest

from src.cli import _toml_str, save_effective_config


@pytest.mark.parametrize('value', [
    r'C:\Users\hansd\AppData\Local\Temp\originstack_desktop_app.log',
    'say "hi"',
    'tab\there',
    'new\nline',
    r'\server\share',
    'unicode ✓ ok',
    '',
])
def test_toml_str_round_trips(value):
    assert tomllib.loads(f'k = {_toml_str(value)}')['k'] == value


def test_saved_config_with_windows_log_path_parses(tmp_path):
    log = r'C:\Users\hansd\AppData\Local\Temp\originstack_desktop_app.log'
    args = argparse.Namespace(log_file=log, stack_method='sigma_clip', auto=True,
                              drizzle_scale=1.0, parallel=0, skip_step=[], note=None)
    save_effective_config(args, str(tmp_path / 'out.fits'))
    text = (tmp_path / 'out_config.toml').read_text(encoding='utf-8')
    cfg = tomllib.loads(text)              # used to raise "Invalid hex value"
    assert cfg['log_file'] == log
    assert cfg['stack_method'] == 'sigma_clip' and cfg['auto'] is True


def test_list_settings_round_trip_and_merge_inputs_are_not_saved(tmp_path):
    """--auto's skip_step additions used to be dropped (lists were not written),
    so a --from-stack or --config re-run ran the steps the original run skipped."""
    from src import cli
    args = argparse.Namespace(skip_step=['sky_residual', 'edge_bands'], merge=['a.fits'],
                              stack_method='sigma_clip')
    save_effective_config(args, str(tmp_path / 'out.fits'))
    path = tmp_path / 'out_config.toml'
    cfg = tomllib.loads(path.read_text(encoding='utf-8'))
    assert cfg['skip_step'] == ['sky_residual', 'edge_bands']
    assert 'merge' not in cfg
    fresh = cli.parse_args(['-d', '.'])
    cli.load_config_file(str(path), fresh)
    assert fresh.skip_step == ['sky_residual', 'edge_bands']
