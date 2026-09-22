"""The self-update check: src.utils.should_check_for_update (the gate) and its
wiring into the CLI (src.cli.main) and the desktop app (src.desktop_app.App).

src.net_query.check_for_update itself (the actual GitHub API call and version
comparison) is covered by tests/test_net_query.py::TestCheckForUpdate. These tests
are about *whether* it gets called at all, and what each caller does with the
result -- both must stay silent/harmless on any failure, since neither a CLI run
nor the app window opening may depend on network access.
"""
from __future__ import annotations

import os
import threading
from unittest import mock

import pytest

from src import utils


@pytest.fixture(autouse=True)
def _clean_env_and_offline_flag():
    had = os.environ.pop('ORIGINSTACK_NO_UPDATE_CHECK', None)
    from src import net_query
    net_query.set_offline(False)
    yield
    net_query.set_offline(False)
    if had is not None:
        os.environ['ORIGINSTACK_NO_UPDATE_CHECK'] = had


class TestShouldCheckForUpdate:
    def test_true_by_default(self):
        assert utils.should_check_for_update() is True

    def test_false_when_env_var_set(self):
        os.environ['ORIGINSTACK_NO_UPDATE_CHECK'] = '1'
        assert utils.should_check_for_update() is False

    def test_env_var_value_does_not_matter(self):
        """Any truthy string disables it -- '0' included, matching how most tools
        treat a presence-only opt-out env var (unset is the only way to mean 'on')."""
        os.environ['ORIGINSTACK_NO_UPDATE_CHECK'] = '0'
        assert utils.should_check_for_update() is False

    def test_false_when_offline_mode_is_active(self):
        from src import net_query
        net_query.set_offline(True)
        assert utils.should_check_for_update() is False

    def test_true_again_once_offline_mode_clears(self):
        from src import net_query
        net_query.set_offline(True)
        net_query.set_offline(False)
        assert utils.should_check_for_update() is True


class TestCliWiring:
    def test_no_thread_started_when_disabled(self, monkeypatch):
        import sys

        from src import cli
        monkeypatch.setattr(cli, 'should_check_for_update', lambda: False)
        started = []
        monkeypatch.setattr(threading, 'Thread',
                            lambda *a, **k: started.append(1) or mock.MagicMock())
        monkeypatch.setattr(sys, 'argv', ['originstack', '-d', 'nonexistent_dir_xyz'])
        with pytest.raises(SystemExit):
            cli.main()
        assert not started, 'no update-check thread should start when disabled'

    def test_notice_printed_after_a_successful_run(self, monkeypatch, capsys, tmp_path):
        import sys

        from src import cli
        monkeypatch.setattr(cli, 'should_check_for_update', lambda: True)
        monkeypatch.setattr(cli, 'process_directory', lambda *a, **k: None)

        # Run the background check inline (same thread) so the result is
        # deterministically ready by the time main() looks for it, rather than
        # racing a real thread in a test.
        class ImmediateThread:
            def __init__(self, target=None, daemon=None):
                self._target = target

            def start(self):
                self._target()
        monkeypatch.setattr(threading, 'Thread', ImmediateThread)
        monkeypatch.setattr('src.net_query.check_for_update',
                           lambda *a, **k: {'version': '9.9.9', 'url': 'https://example.invalid/new'})
        monkeypatch.setattr(sys, 'argv', ['originstack', '-d', str(tmp_path), '-o', str(tmp_path / 'out.fits')])

        cli.main()
        out = capsys.readouterr().out
        assert 'v9.9.9' in out and 'https://example.invalid/new' in out

    def test_no_notice_when_check_finds_nothing(self, monkeypatch, capsys, tmp_path):
        import sys

        from src import cli
        monkeypatch.setattr(cli, 'should_check_for_update', lambda: True)
        monkeypatch.setattr(cli, 'process_directory', lambda *a, **k: None)

        class ImmediateThread:
            def __init__(self, target=None, daemon=None):
                self._target = target

            def start(self):
                self._target()
        monkeypatch.setattr(threading, 'Thread', ImmediateThread)
        monkeypatch.setattr('src.net_query.check_for_update', lambda *a, **k: None)
        monkeypatch.setattr(sys, 'argv', ['originstack', '-d', str(tmp_path), '-o', str(tmp_path / 'out.fits')])

        cli.main()
        assert 'Update available' not in capsys.readouterr().out and 'newer OriginStack' not in capsys.readouterr().out


class TestDesktopAppWiring:
    def test_no_check_started_when_disabled(self, monkeypatch):
        pytest.importorskip('tkinter')
        import tkinter as tk

        try:
            root = tk.Tk()
        except tk.TclError:
            pytest.skip('no display available for a real Tk root')
        try:
            from src import desktop_app
            monkeypatch.setattr('src.utils.should_check_for_update', lambda: False)
            started = []
            monkeypatch.setattr(threading, 'Thread',
                                lambda *a, **k: started.append(1) or mock.MagicMock())
            app = desktop_app.App(root)
            assert not started
            assert app.update_label.cget('text') == ''
        finally:
            root.destroy()

    def test_label_updates_when_a_newer_version_is_found(self, monkeypatch):
        pytest.importorskip('tkinter')
        import tkinter as tk

        try:
            root = tk.Tk()
        except tk.TclError:
            pytest.skip('no display available for a real Tk root')
        try:
            from src import desktop_app
            monkeypatch.setattr('src.utils.should_check_for_update', lambda: True)

            class ImmediateThread:
                def __init__(self, target=None, daemon=None):
                    self._target = target

                def start(self):
                    self._target()
            monkeypatch.setattr(threading, 'Thread', ImmediateThread)
            monkeypatch.setattr('src.net_query.check_for_update',
                               lambda *a, **k: {'version': '9.9.9', 'url': 'https://example.invalid/new'})
            app = desktop_app.App(root)
            root.update()
            assert '9.9.9' in app.update_label.cget('text')
            assert app._update_url == 'https://example.invalid/new'
        finally:
            root.destroy()
