"""Copy diagnostics, remembered folders and the non-Origin camera note (headless)."""
import os
from types import SimpleNamespace

from src import desktop_control as dc


def test_diagnostics_text_has_system_settings_and_log_tail(tmp_path):
    home = os.path.expanduser('~')
    text = dc.diagnostics_text({'directory': os.path.join(home, 'lights'), 'auto': True, 'x': ''},
                               '\n'.join(f'line {i}' for i in range(100)), n_lines=5)
    assert text.startswith('OriginStack ')
    for needle in ('Python', 'GPU:', 'Optional packages:', 'astro_native=', 'auto = True',
                   'line 99', 'Log (last 5 lines)'):
        assert needle in text
    assert 'line 94' not in text and 'x =' not in text
    assert home not in text and 'directory = "~' in text


def test_diagnostics_text_without_settings():
    assert '(none)' in dc.diagnostics_text()


class _Hdr(dict):
    pass


def _lights(**hdr):
    return [SimpleNamespace(header=_Hdr(hdr))]


def test_unprofiled_camera_names_other_cameras():
    assert dc.unprofiled_camera(_lights(INSTRUME='ZWO ASI533MC')) == 'ZWO ASI533MC'
    assert dc.unprofiled_camera(_lights()) is None
    assert dc.unprofiled_camera([]) is None


def test_unprofiled_camera_skips_profiled_origin():
    from src.camera_profile import identify, profile_for
    hdr = _Hdr(CAMERA='Origin178-932f562381e97ce70', ISOSPEED=200, NAXIS1=3096, NAXIS2=2080)
    prof = profile_for(identify(hdr), (2080, 3096))
    expected = None if prof is not None else 'Origin178 #932f56'
    assert dc.unprofiled_camera(_lights(**hdr)) == expected


def test_app_setting_roundtrip_for_folders(tmp_path, monkeypatch):
    import src.i18n as i18n
    monkeypatch.setattr(i18n, '_settings_path', lambda: tmp_path / 'settings.json')
    dc.save_app_setting('last_directory', str(tmp_path))
    assert dc.load_app_setting('last_directory') == str(tmp_path)


def test_setup_form_restore_and_remember(tmp_path, monkeypatch):
    import pytest
    tk = pytest.importorskip('tkinter')
    import src.i18n as i18n
    monkeypatch.setattr(i18n, '_settings_path', lambda: tmp_path / 'settings.json')
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    try:
        from src.desktop_app import SetupForm
        form = SetupForm(root)
        folder = tmp_path / 'lights'
        folder.mkdir()
        form.vars['directory'].set(str(folder))
        form.vars['output'].set(str(tmp_path / 'out' / 'stack.fits'))
        form._remember_folders()
        assert dc.load_app_setting('last_output') == str(tmp_path / 'out')
        form2 = SetupForm(root)
        form2.restore_folders()
        assert form2.vars['directory'].get() == str(folder)
    finally:
        root.destroy()
