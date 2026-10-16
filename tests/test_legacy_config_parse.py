"""Configs saved before v2.1.0 wrote Windows paths unescaped (invalid TOML) and were
ignored as a whole; they now load with those paths read as literal strings."""
from src.cli import _repair_legacy_toml, load_config_file, parse_args

OLD = r'''# OriginStack effective configuration
astrollm_dir = "C:\source\astrollm"
combine_sessions = true
drizzle_scale = 2.0
stretch = "ghs"
temp_dir = "D:\temp\stack"   # trailing comment
'''


def test_repair_turns_backslash_strings_into_literal_strings():
    fixed = _repair_legacy_toml(OLD)
    assert r"astrollm_dir = 'C:\source\astrollm'" in fixed
    assert r"temp_dir = 'D:\temp\stack'   # trailing comment" in fixed
    assert 'stretch = "ghs"' in fixed           # no backslash: untouched


def test_old_config_loads_its_settings(tmp_path):
    cfg = tmp_path / 'old_config.toml'
    cfg.write_text(OLD, encoding='utf-8')
    args = parse_args(['-d', 'x', '-o', 'y.fits'])
    changes = load_config_file(str(cfg), args)
    assert args.combine_sessions is True and args.drizzle_scale == 2.0
    assert args.temp_dir == r'D:\temp\stack'
    assert changes


def test_valid_config_is_read_as_before(tmp_path):
    cfg = tmp_path / 'new_config.toml'
    cfg.write_text('temp_dir = "D:\\\\temp"\ndrizzle_scale = 1.5\n', encoding='utf-8')
    args = parse_args(['-d', 'x', '-o', 'y.fits'])
    load_config_file(str(cfg), args)
    assert args.temp_dir == r'D:\temp' and args.drizzle_scale == 1.5


def test_unreadable_config_says_nothing_was_applied(tmp_path, capsys):
    cfg = tmp_path / 'broken_config.toml'
    cfg.write_text('drizzle_scale = = 2\n', encoding='utf-8')
    args = parse_args(['-d', 'x', '-o', 'y.fits'])
    assert load_config_file(str(cfg), args) == []
    assert 'NONE of its settings are applied' in capsys.readouterr().out
