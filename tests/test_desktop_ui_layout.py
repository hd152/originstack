"""Desktop-app layout polish: the one-line preflight and the Compare box.

Run against the real methods on stand-in objects holding real widgets: a whole ``App`` per module
left Tk in a state where the next module's root intermittently failed to open."""
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import src.desktop_app as da


def test_preflight_collapses_to_one_line_when_all_pass_and_expands_on_a_problem(tk_root):
    frame = ttk.Frame(tk_root)
    me = SimpleNamespace(
        preflight_vars={k: tk.StringVar(value='✓ ' + k) for k in ('a', 'b', 'c')},
        _preflight_ok_var=tk.StringVar(), form=SimpleNamespace(_data_summary='35 lights'))
    me._preflight_ok = ttk.Label(frame, textvariable=me._preflight_ok_var)
    me._preflight_grid = ttk.Frame(frame)
    da.App._show_preflight_summary(me)
    assert me._preflight_ok.winfo_manager() == 'pack'
    assert me._preflight_grid.winfo_manager() == ''
    assert '35 lights' in me._preflight_ok_var.get()
    me.preflight_vars['b'].set('⚠ Low disk space')
    da.App._show_preflight_summary(me)
    assert me._preflight_grid.winfo_manager() == 'pack'
    assert me._preflight_ok.winfo_manager() == ''


def test_compare_box_follows_the_compare_checkbox_even_when_set_from_code(tk_root):
    me = SimpleNamespace(compare_var=tk.BooleanVar(tk_root, value=False))
    me.compare_combo = ttk.Combobox(tk_root, state='disabled')
    me.compare_var.trace_add('write', lambda *_a: da.App._sync_compare_box(me))
    me.compare_var.set(True)
    assert str(me.compare_combo.cget('state')) == 'readonly'
    me.compare_var.set(False)           # how the code switches compare off
    assert str(me.compare_combo.cget('state')) == 'disabled'
