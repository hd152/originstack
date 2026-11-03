"""Desktop-app layout polish: the one-line preflight and the Compare box."""
import pytest


@pytest.fixture(scope='module')
def app(tk_root):
    import src.desktop_app as da
    return da.App(tk_root)


def test_preflight_collapses_to_one_line_when_all_pass_and_expands_on_a_problem(app):
    for key, var in app.preflight_vars.items():
        var.set('✓ ' + key)
    app._show_preflight_summary()
    assert app._preflight_ok.winfo_manager() == 'pack'
    assert app._preflight_grid.winfo_manager() == ''
    app.preflight_vars['space'].set('⚠ Low disk space')
    app._show_preflight_summary()
    assert app._preflight_grid.winfo_manager() == 'pack'
    assert app._preflight_ok.winfo_manager() == ''


def test_compare_box_follows_the_compare_checkbox_even_when_set_from_code(app):
    assert str(app.compare_combo.cget('state')) == 'disabled'
    app.compare_var.set(True)
    assert str(app.compare_combo.cget('state')) == 'readonly'
    app.compare_var.set(False)           # how the code switches compare off
    assert str(app.compare_combo.cget('state')) == 'disabled'
