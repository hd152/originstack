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
