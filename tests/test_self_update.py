"""In-app update: finding the installer in a release, downloading it verified, and when it applies."""
import hashlib
import io
import json
import threading
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

from src import net_query, self_update

REPO = 'hd152/originstack'
URL = f'https://github.com/{REPO}/releases/download/v9.9.9/OriginStack-9.9.9-setup.exe'
DATA = b'MZ' + bytes(range(256)) * 40
SHA = hashlib.sha256(DATA).hexdigest()


def _asset(**kw):
    a = {'name': 'OriginStack-9.9.9-setup.exe', 'browser_download_url': URL, 'size': len(DATA),
         'digest': 'sha256:' + SHA}
    a.update(kw)
    return a


def _release(assets):
    payload = {'tag_name': 'v9.9.9', 'html_url': 'https://github.com/x', 'assets': assets}
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__.return_value = resp
    return resp


class TestInstallerAsset:
    def test_release_with_a_verifiable_installer_carries_it(self):
        with mock.patch('urllib.request.urlopen', return_value=_release([_asset()])):
            info = net_query.check_for_update('2.21.0')
        assert info['installer_url'] == URL and info['installer_size'] == len(DATA)
        assert info['installer_sha256'] == SHA and info['installer_name'].endswith('-setup.exe')

    @pytest.mark.parametrize('bad', [
        {'digest': None}, {'digest': 'sha1:abc'}, {'digest': 'sha256:xyz'}, {'size': 0},
        {'size': 'big'}, {'browser_download_url': 'https://evil.example/OriginStack-9.9.9-setup.exe'},
        {'browser_download_url': 'https://github.com/other/repo/releases/download/v1/a-setup.exe'},
    ])
    def test_an_asset_that_cannot_be_verified_is_not_offered(self, bad):
        with mock.patch('urllib.request.urlopen', return_value=_release([_asset(**bad)])):
            info = net_query.check_for_update('2.21.0')
        assert info is not None and 'installer_url' not in info

    def test_other_assets_are_ignored(self):
        other = _asset(name='OriginStack-9.9.9-windows-x64.zip')
        with mock.patch('urllib.request.urlopen', return_value=_release([other])):
            assert 'installer_url' not in net_query.check_for_update('2.21.0')


def _fake_opener(data):
    resp = io.BytesIO(data)
    opener = mock.MagicMock()
    opener.open.return_value.__enter__.return_value = resp
    return opener


class TestDownload:
    def _run(self, tmp_path, data=DATA, size=len(DATA), sha=SHA, **kw):
        dest = tmp_path / 'u' / 'setup.exe'
        with mock.patch('urllib.request.build_opener', return_value=_fake_opener(data)):
            return dest, net_query.download_update(URL, dest, size, sha, **kw)

    def test_a_matching_download_is_kept_and_reports_progress(self, tmp_path):
        seen = []
        dest, out = self._run(tmp_path, progress=lambda d, t: seen.append((d, t)))
        assert out == dest and dest.read_bytes() == DATA
        assert seen[-1] == (len(DATA), len(DATA))
        assert not list(dest.parent.glob('*.part'))

    @pytest.mark.parametrize('data,size,sha', [
        (DATA[:-1] + b'!', len(DATA), SHA),          # same length, other content: hash
        (DATA[:-5], len(DATA), SHA),                 # cut short
        (DATA + b'x', len(DATA), SHA),               # longer than announced
    ])
    def test_a_wrong_download_leaves_nothing_behind(self, tmp_path, data, size, sha):
        dest = tmp_path / 'u' / 'setup.exe'
        with mock.patch('urllib.request.build_opener', return_value=_fake_opener(data)):
            with pytest.raises(ValueError):
                net_query.download_update(URL, dest, size, sha)
        assert not dest.exists() and not list(dest.parent.glob('*'))

    @pytest.mark.parametrize('url', ['http://github.com/x', 'https://evil.example/setup.exe',
                                     'https://github.com.evil.example/setup.exe'])
    def test_only_github_https_hosts_are_fetched(self, tmp_path, url):
        with mock.patch('urllib.request.build_opener') as bo:
            with pytest.raises(ValueError):
                net_query.download_update(url, tmp_path / 's.exe', 10, SHA)
        bo.assert_not_called()

    def test_a_redirect_to_another_host_is_refused(self):
        h = net_query._HostCheckedRedirect()
        with pytest.raises(urllib.error.URLError):
            h.redirect_request(mock.MagicMock(), None, 302, 'Found', {}, 'https://evil.example/a.exe')
        with pytest.raises(urllib.error.URLError):
            h.redirect_request(mock.MagicMock(), None, 302, 'Found', {},
                               'http://objects.githubusercontent.com/a.exe')

    def test_cancel_stops_and_cleans_up(self, tmp_path):
        ev = threading.Event()
        ev.set()
        dest = tmp_path / 'u' / 'setup.exe'
        with mock.patch('urllib.request.build_opener', return_value=_fake_opener(DATA)):
            with pytest.raises(InterruptedError):
                net_query.download_update(URL, dest, len(DATA), SHA, cancel=ev)
        assert not list(dest.parent.glob('*'))

    def test_offline_mode_blocks_it(self, tmp_path):
        net_query.set_offline(True)
        try:
            with pytest.raises(Exception):
                net_query.download_update(URL, tmp_path / 's.exe', len(DATA), SHA)
        finally:
            net_query.set_offline(False)


class TestWhenItApplies:
    def test_not_a_frozen_windows_copy(self):
        assert self_update.can_install_in_app() is False      # running from source

    def test_installed_copy_needs_the_uninstaller_and_a_verifiable_release(self, tmp_path, monkeypatch):
        (tmp_path / 'unins000.exe').write_bytes(b'')
        monkeypatch.setattr('sys.platform', 'win32')
        monkeypatch.setattr('sys.frozen', True, raising=False)
        monkeypatch.setattr('sys.executable', str(tmp_path / 'OriginStack.exe'))
        info = {'installer_url': URL, 'installer_size': 5, 'installer_sha256': SHA}
        assert self_update.can_install_in_app(info) is True
        assert self_update.can_install_in_app({'version': '9'}) is False
        (tmp_path / 'unins000.exe').unlink()                  # a portable or zip copy
        assert self_update.can_install_in_app(info) is False

    def test_installer_name_cannot_leave_its_folder(self):
        p = self_update.installer_path({'installer_name': r'..\..\evil.exe'})
        assert p.parent == Path(self_update.tempfile.gettempdir()) / 'OriginStack-update'
