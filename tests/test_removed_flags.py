"""originvision was removed in 2.6; its flags stay accepted so old command
lines and saved configs keep working."""
import pytest

from src.cli import load_config_file, parse_args, save_effective_config


@pytest.mark.parametrize('argv', [
    ['--originvision'], ['--no-originvision'], ['--originvision-score-all'],
    ['--originvision-model', 'm.onnx'], ['--originvision-workers', '4'],
    ['--originvision-dir', 'd'], ['--originvision-checkpoint', 'c.onnx'],
])
def test_old_flags_parse_with_one_notice(argv, capsys):
    args = parse_args(['-d', 'x'] + argv)
    out = capsys.readouterr().out
    assert out.count('originvision scoring was removed') == 1
    assert argv[0] in out
    assert args.originvision is False


def test_no_notice_without_old_flags(capsys):
    parse_args(['-d', 'x'])
    assert 'originvision' not in capsys.readouterr().out


def test_saved_config_with_originvision_still_loads(tmp_path):
    cfg = tmp_path / 'old_config.toml'
    cfg.write_text('originvision = true\noriginvision_workers = 8\nstack_method = "median"\n')
    args = parse_args(['-d', 'x'])
    load_config_file(str(cfg), args)
    assert args.stack_method == 'median'


def test_new_saved_config_carries_no_removed_flag_state(tmp_path):
    args = parse_args(['-d', 'x', '--originvision'])
    out = tmp_path / 'stack.fits'
    save_effective_config(args, str(out))
    text = (tmp_path / 'stack_config.toml').read_text()
    assert '_removed_flags' not in text
