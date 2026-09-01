"""Desktop app entry point (``python desktop_app.py``).

A native ``tkinter`` window (stdlib -- no ``pywebview``/WebView2 Runtime
dependency): a Setup form auto-built from ``cli.build_parser()`` via
``src/desktop_control.py``'s schema introspection, a live progress/log panel,
and an interactive preview (zoom/pan, live re-stretch, before/after wipe
compare, per-frame thumbnail ring) fed by ``src/ui_events.py``'s in-process
event sink -- the same state model the old HTTP/SSE dashboard used, just
polled directly by a ``root.after()`` timer instead of pushed over a socket.

Every failure path below routes through ``_fatal()``: a packaged PyInstaller
build runs windowed (no console), so a bare ``print()`` is invisible to a
double-click user -- it must show a native dialog and log to a location
that's writable regardless of install directory (Program Files is often
read-only for a non-admin install).
"""
from __future__ import annotations

import datetime
import io
import os
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk
from typing import Any, Dict, List, Optional


def _log_dir() -> Path:
    return Path(os.environ.get('LOCALAPPDATA', '.')) / 'OriginStack' / 'logs'


def _log_startup_status() -> None:
    """Unconditional (not just on failure) one-line startup record --
    packaging/verify_build.ps1 greps this to confirm astro_native (not the
    numpy fallback) loaded in the packaged build, now that there's no
    ``GET /api/health`` to poll instead."""
    try:
        from src.utils import native_status, read_version
        log_dir = _log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / 'desktop_app.log', 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] "
                    f"OriginStack {read_version()} starting -- {native_status()}\n")
    except OSError:
        pass


def _fatal(title: str, message: str) -> int:
    """Last-resort error surface: log full details, show a native dialog."""
    try:
        _log_dir().mkdir(parents=True, exist_ok=True)
        with open(_log_dir() / 'desktop_app_crash.log', 'a', encoding='utf-8') as f:
            f.write(f"\n[{datetime.datetime.now().isoformat()}] {title}\n{message}\n")
            f.write(traceback.format_exc() + '\n')
    except OSError:
        pass
    from src.native_dialog import show_error
    show_error(title, message)
    return 1


# ── design tokens ────────────────────────────────────────────────────────
# The old browser dashboard's palette, ported whole: this is a telescope
# control panel, and Ha/OIII/SII (the three narrowband channels this
# pipeline's own SHO combiner maps to red/teal/gold, src/channel_combine.py)
# are this app's own signal colors, not an arbitrary accent choice. A full
# ttk theme (not just the log pane) so the rest of the window stops clashing
# with it -- ttk's default Windows theme ('vista') ignores color overrides
# almost entirely, so _apply_theme() below switches to 'clam', the one
# built-in theme that actually respects them.
_BG = '#08090c'          # window background -- true dark-sky black
_PANEL = '#0f1117'       # section backgrounds
_WELL = '#030304'        # recessed fields: log, entries, canvases
_LINE = '#23262f'        # borders
_LINE_SOFT = '#1a1c22'
_TEXT = '#e8e6df'        # warm off-white -- phosphor afterglow, not pure white
_TEXT_DIM = '#7c8090'
_TEXT_FAINT = '#4c505e'
_ACCENT = '#ff6a3d'      # H-alpha -- primary: active state, primary actions
_ACCENT2 = '#3ecbe0'     # O-III -- secondary: compare/info accents
_ACCENT3 = '#ffc857'     # S-II -- tertiary: in-progress / warm highlight
_LOG_FG = '#ffb454'      # amber phosphor -- the log pane's signature color
_LOG_BG = _WELL
_SUCCESS = '#5cd98f'
_BAD = '#ff4757'
_FONT = ('Segoe UI', 9)
_FONT_BOLD = ('Segoe UI', 9, 'bold')
_FONT_MONO = ('Consolas', 9)


def _apply_theme(root: tk.Tk) -> ttk.Style:
    """One dark theme for every ttk widget in the app, built on 'clam' (the
    only built-in ttk theme where background/foreground overrides actually
    render on Windows -- 'vista'/'winnative' draw native chrome and mostly
    ignore them)."""
    root.configure(background=_BG)
    style = ttk.Style(root)
    style.theme_use('clam')

    style.configure('.', background=_BG, foreground=_TEXT, font=_FONT,
                    fieldbackground=_WELL, bordercolor=_LINE,
                    darkcolor=_BG, lightcolor=_BG, troughcolor=_WELL,
                    selectbackground=_ACCENT, selectforeground='#1a0a04')
    style.configure('TFrame', background=_BG)
    style.configure('TLabel', background=_BG, foreground=_TEXT)
    style.configure('Dim.TLabel', background=_BG, foreground=_TEXT_DIM)
    style.configure('Faint.TLabel', background=_BG, foreground=_TEXT_FAINT)
    style.configure('Header.TLabel', background=_BG, foreground=_TEXT,
                    font=('Segoe UI', 14, 'bold'))
    style.configure('Accent.TLabel', background=_BG, foreground=_ACCENT,
                    font=('Segoe UI', 14, 'bold'))

    # Content area matches the window background (_BG), not a separate
    # panel color -- ttk widgets don't composite against a parent's
    # background (no transparency), so every child inside a differently-
    # colored LabelFrame would need its own parallel style just to avoid a
    # visible seam. Grouping instead comes from the border + dim uppercase
    # title alone; "recessed" data widgets (log, entries, canvases, table
    # rows) stay the one deliberately darker tier, _WELL, against this.
    style.configure('TLabelframe', background=_BG, bordercolor=_LINE,
                    relief='solid', borderwidth=1)
    style.configure('TLabelframe.Label', background=_BG,
                    foreground=_TEXT_DIM, font=('Segoe UI', 9, 'bold'))

    style.configure('TButton', background=_PANEL, foreground=_TEXT,
                    bordercolor=_LINE, focuscolor=_ACCENT2, padding=(10, 5))
    style.map('TButton',
             background=[('pressed', _LINE), ('active', _LINE_SOFT)],
             bordercolor=[('focus', _ACCENT2)])
    style.configure('Accent.TButton', background=_ACCENT, foreground='#1a0a04',
                    font=_FONT_BOLD, bordercolor=_ACCENT, padding=(14, 7))
    style.map('Accent.TButton',
             background=[('disabled', _LINE), ('pressed', '#e65a30'), ('active', '#ff7d54')],
             foreground=[('disabled', _TEXT_FAINT)])

    style.configure('TCheckbutton', background=_BG, foreground=_TEXT,
                    focuscolor=_ACCENT2)
    style.map('TCheckbutton', background=[('active', _BG)],
             foreground=[('disabled', _TEXT_FAINT)])

    style.configure('TEntry', fieldbackground=_WELL, foreground=_TEXT,
                    bordercolor=_LINE, insertcolor=_TEXT, padding=4)
    style.map('TEntry', bordercolor=[('focus', _ACCENT2)])

    style.configure('TCombobox', fieldbackground=_WELL, foreground=_TEXT,
                    background=_PANEL, bordercolor=_LINE, arrowcolor=_TEXT_DIM,
                    padding=4)
    style.map('TCombobox',
             fieldbackground=[('readonly', _WELL), ('disabled', _PANEL)],
             foreground=[('readonly', _TEXT)],
             bordercolor=[('focus', _ACCENT2)])
    root.option_add('*TCombobox*Listbox.background', _WELL)
    root.option_add('*TCombobox*Listbox.foreground', _TEXT)
    root.option_add('*TCombobox*Listbox.selectBackground', _ACCENT)
    root.option_add('*TCombobox*Listbox.selectForeground', '#1a0a04')

    style.configure('TNotebook', background=_BG, bordercolor=_LINE)
    style.configure('TNotebook.Tab', background=_PANEL, foreground=_TEXT_DIM,
                    padding=(12, 6), font=_FONT)
    style.map('TNotebook.Tab',
             background=[('selected', _BG)],
             foreground=[('selected', _TEXT)],
             bordercolor=[('selected', _ACCENT)])

    style.configure('Horizontal.TScale', background=_BG, troughcolor=_WELL)
    style.configure('Horizontal.TProgressbar', background=_ACCENT,
                    troughcolor=_WELL, bordercolor=_LINE_SOFT,
                    lightcolor=_ACCENT, darkcolor=_ACCENT)

    style.configure('Treeview', background=_WELL, fieldbackground=_WELL,
                    foreground=_TEXT, bordercolor=_LINE, rowheight=22)
    style.configure('Treeview.Heading', background=_PANEL, foreground=_TEXT_FAINT,
                    font=('Segoe UI', 8, 'bold'), relief='flat')
    style.map('Treeview.Heading', background=[('active', _PANEL)])
    style.map('Treeview', background=[('selected', _LINE)],
             foreground=[('selected', _TEXT)])

    style.configure('TScrollbar', background=_PANEL, troughcolor=_BG,
                    bordercolor=_LINE, arrowcolor=_TEXT_DIM)
    style.map('TScrollbar', background=[('active', _LINE)])

    style.configure('TPanedwindow', background=_BG)
    style.configure('Sash', background=_LINE, sashthickness=6)

    for phase_style, bg, fg, border in (
            ('Phase.TLabel', _WELL, _TEXT_DIM, _LINE),
            ('PhaseActive.TLabel', '#2a1710', _TEXT, _ACCENT),
            ('PhaseDone.TLabel', '#12241a', _TEXT_DIM, _SUCCESS)):
        style.configure(phase_style, background=bg, foreground=fg,
                        font=_FONT, padding=6, relief='solid',
                        borderwidth=1, bordercolor=border)
    return style


class ScrollableFrame(ttk.Frame):
    """A vertically-scrollable container -- some Setup tabs have 20+ fields
    and won't fit the window at once."""

    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=_BG)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind('<Configure>',
                        lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        def _wheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), 'units')
        canvas.bind_all('<MouseWheel>', _wheel, add='+')


#  Shown directly on the main form, in this order -- everything else lives
# behind the "Additional options" toggle. This is the same small set the
# old browser dashboard treated as "quick fields", chosen because they're
# the ones almost every run touches (where the lights are, what to call the
# output, and the handful of settings that most change the result) while
# the other ~110 flags are fine-tuning most runs never need.
_COMMON_DESTS = ['directory', 'output', 'preset', 'auto', 'stack_method',
                 'denoiser', 'deconvolve', 'drizzle_scale', 'trail_reject',
                 'use_gpu', 'parallel', 'astrollm']


class SetupForm(ttk.Frame):
    """Auto-built from ``desktop_control.get_form_schema()``. A fixed set of
    common fields (``_COMMON_DESTS``) sits directly on the form; every other
    field lives behind a collapsed "Additional options" section (a sidebar
    of argparse groups, expanded on demand) so a typical run only has to
    look at ~10 fields, not the full ~120-flag surface. Field kinds
    (``bool_true``/``bool_false``/``select``/``list``/``number``/``text``)
    and path widgets (``dir``/``file-open``/``file-save``) come straight
    from the schema; nothing here needs to know about individual flags."""

    def __init__(self, parent):
        super().__init__(parent)
        from src.desktop_control import get_form_schema
        self.schema = get_form_schema()
        self.vars: Dict[str, tk.Variable] = {}
        # The value each Variable was initialized with -- read_form() only
        # emits a dest whose value has actually changed from this. Without
        # that check, every field (even ones the user never touched) would
        # be submitted, and build_argv_from_form/parse_args would mark all
        # of them "explicit", which makes auto_settings.py's target-aware
        # tuning skip them entirely (_set() never overrides an explicit
        # dest) -- silently freezing ~40 settings, including the whole
        # denoiser family, at raw argparse defaults on every GUI-launched
        # run regardless of target type. Confirmed as the actual cause of a
        # reported "galaxy detail lost to denoising" bug: the auto-advisor's
        # galaxy-specific MMT/ACDNR choice never got a chance to apply.
        self._initial: Dict[str, Any] = {}
        self.dir_count_var = tk.StringVar(value='')
        all_fields = {f['dest']: f for fields in self.schema.values() for f in fields}

        common_frame = ttk.Frame(self)
        common_frame.pack(fill='x')
        row = 0
        for dest in _COMMON_DESTS:
            field = all_fields.get(dest)
            if field is not None:
                row += self._build_field(common_frame, row, field)

        self._advanced_shown = False
        self._toggle_btn = ttk.Button(self, text='▸ Additional options',
                                      command=self._toggle_advanced)
        self._toggle_btn.pack(fill='x', pady=(10, 0), anchor='w')

        # A sidebar of group names, not a Notebook of horizontal tabs: this
        # form has 8 groups with names like "Registration & stacking
        # (Phases 2-3)" -- laid out as tabs they clip to unreadable
        # fragments once more than 3-4 fit a half-window pane. A vertical
        # list has room for the full name and scales to more groups later.
        self._advanced_body = ttk.Frame(self)
        nav = tk.Frame(self._advanced_body, background=_PANEL, width=168)
        nav.pack(side='left', fill='y')
        nav.pack_propagate(False)
        content = ttk.Frame(self._advanced_body)
        content.pack(side='left', fill='both', expand=True, padx=(10, 0))

        self._pages: Dict[str, ScrollableFrame] = {}
        self._nav_labels: Dict[str, tk.Label] = {}
        for group_title in self.schema:
            self._pages[group_title] = ScrollableFrame(content)
            lbl = tk.Label(nav, text=group_title, background=_PANEL, foreground=_TEXT_DIM,
                          font=_FONT, anchor='w', justify='left', wraplength=148,
                          padx=12, pady=9)
            lbl.pack(fill='x')
            lbl.bind('<Button-1>', lambda _e, g=group_title: self._show_group(g))
            self._nav_labels[group_title] = lbl

        for group_title, fields in self.schema.items():
            page = self._pages[group_title]
            # Skip anything already on the main form -- building a second
            # widget for the same dest would create a second Variable, and
            # read_form() would silently drop whichever one the user didn't
            # touch last.
            visible_fields = [f for f in fields if f['dest'] not in _COMMON_DESTS]
            row = 0
            for field in visible_fields:
                row += self._build_field(page.inner, row, field)

        self._show_group(next(iter(self.schema)))

    def _toggle_advanced(self) -> None:
        self._advanced_shown = not self._advanced_shown
        if self._advanced_shown:
            self._advanced_body.pack(fill='both', expand=True, pady=(8, 0))
            self._toggle_btn.configure(text='▾ Additional options')
        else:
            self._advanced_body.pack_forget()
            self._toggle_btn.configure(text='▸ Additional options')

    def _show_group(self, group_title: str) -> None:
        for g, page in self._pages.items():
            selected = g == group_title
            if selected:
                page.pack(fill='both', expand=True)
            else:
                page.pack_forget()
            self._nav_labels[g].configure(
                background=_LINE_SOFT if selected else _PANEL,
                foreground=_TEXT if selected else _TEXT_DIM)

    def _build_field(self, parent, row: int, field: Dict[str, Any]) -> None:
        dest, kind = field['dest'], field['kind']
        # bool_false fields (e.g. dest 'auto', flag '--no-auto') default
        # checked=True -- the checkbox represents args.<dest> directly, so
        # checked means "auto stays on" (the flag is *omitted*). Labeling it
        # with the negating flag text would read backwards (a checked box
        # next to "--no-auto" reads as "disable it"); show the dest name
        # instead so checked-and-labeled-"Auto" means what it looks like.
        label_text = dest.replace('_', ' ').capitalize() if kind == 'bool_false' else field['flag']
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky='w', padx=(4, 8), pady=3)
        if field.get('help'):
            _Tooltip(label, field['help'])

        rows_used = 1
        if kind in ('bool_true', 'bool_false'):
            var = tk.BooleanVar(value=bool(field['default']))
            ttk.Checkbutton(parent, variable=var).grid(row=row, column=1, sticky='w')
        elif kind == 'select':
            var = tk.StringVar(value='' if field['default'] is None else str(field['default']))
            cb = ttk.Combobox(parent, textvariable=var, state='readonly',
                              values=[''] + [str(c) for c in (field['choices'] or [])],
                              width=22)
            cb.grid(row=row, column=1, sticky='w')
        else:
            default = field['default']
            text = ', '.join(default) if isinstance(default, list) else \
                   ('' if default is None else str(default))
            var = tk.StringVar(value=text)
            entry = ttk.Entry(parent, textvariable=var, width=30)
            entry.grid(row=row, column=1, sticky='we')
            widget_hint = field.get('widget')
            if widget_hint:
                ttk.Button(parent, text='Browse…', width=9,
                          command=lambda d=dest, v=var, w=widget_hint:
                              self._browse(d, v, w)).grid(row=row, column=2, padx=4)
            if dest == 'directory':
                var.trace_add('write', lambda *_: self._rescan_directory())
                count_label = ttk.Label(parent, textvariable=self.dir_count_var,
                                        style='Dim.TLabel')
                # A row of its own -- sharing row+1 with the next field
                # (grid allows it, but the two widgets would overlap on
                # screen the moment this label actually has text) --
                # reserved by the caller's row cursor via the returned count.
                count_label.grid(row=row + 1, column=1, columnspan=2, sticky='w')
                rows_used = 2
        parent.grid_columnconfigure(1, weight=1)
        self.vars[dest] = var
        self._initial[dest] = var.get()
        return rows_used

    def _browse(self, dest: str, var: tk.Variable, widget_hint: str) -> None:
        if widget_hint == 'dir':
            path = filedialog.askdirectory()
        elif widget_hint == 'file-save':
            path = filedialog.asksaveasfilename(
                filetypes=[('FITS files', '*.fits'), ('All files', '*.*')])
        else:
            path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _rescan_directory(self) -> None:
        directory = self.vars['directory'].get().strip()
        if not directory or not os.path.isdir(directory):
            self.dir_count_var.set('')
            return
        try:
            from src.frame_discovery import discover_frames
            counts = {k: len(v) for k, v in discover_frames(directory).items()}
            if sum(counts.values()) == 0:
                subdirs = [os.path.join(directory, d) for d in os.listdir(directory)
                          if os.path.isdir(os.path.join(directory, d))]
                for d in subdirs:
                    sub_counts = discover_frames(d)
                    for k, v in sub_counts.items():
                        counts[k] = counts.get(k, 0) + len(v)
            parts = ', '.join(f'{v} {k}' for k, v in counts.items() if v)
            self.dir_count_var.set(parts or 'no frames found')
        except Exception:
            self.dir_count_var.set('')

    def read_form(self) -> Dict[str, Any]:
        """``{dest: value}`` for fields the user actually changed from their
        schema default -- ``build_argv_from_form`` already handles blank
        strings, comma lists, and both boolean kinds, so no conversion is
        needed here. Untouched fields are deliberately omitted (see
        ``self._initial``'s docstring): submitting every field unconditionally
        would mark them all "explicit" and defeat the auto-advisor's
        target-aware tuning on every GUI-launched run."""
        return {dest: var.get() for dest, var in self.vars.items()
               if var.get() != self._initial.get(dest)}


class _Tooltip:
    """Minimal hover tooltip -- the schema's ``help`` text is the only place
    each flag's full description lives; a native app has no ``title=`` like
    the old HTML form's inputs did."""

    def __init__(self, widget, text: str) -> None:
        self.widget, self.text, self.tip = widget, text, None
        widget.bind('<Enter>', self._show)
        widget.bind('<Leave>', self._hide)

    def _show(self, _event) -> None:
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f'+{x}+{y}')
        tk.Label(self.tip, text=self.text, background=_LINE_SOFT, foreground=_TEXT,
                relief='solid', borderwidth=1, wraplength=380,
                font=_FONT, justify='left', padx=6, pady=4).pack()

    def _hide(self, _event) -> None:
        if self.tip:
            self.tip.destroy()
            self.tip = None


class PreviewCanvas(ttk.Frame):
    """Zoom/pan/re-stretch/wipe-compare image viewer -- the native
    equivalent of the old dashboard's Canvas-free CSS-transform viewport.
    Panning moves a stored offset; zooming and re-stretching regenerate the
    displayed ``PhotoImage`` at the new scale/pixels. Compare mode composites
    two milestones with ``PIL.Image.paste`` at the wipe position instead of
    the browser's ``clip-path``."""

    def __init__(self, parent):
        super().__init__(parent)
        self.scale = 1.0
        self.fit_scale = 1.0
        self.tx = 0.0
        self.ty = 0.0
        self._nat_w = 0
        self._nat_h = 0
        self._img_a: Optional[Any] = None   # PIL.Image
        self._img_b: Optional[Any] = None
        self._photo = None                  # keep-alive ref
        self.current_slug = ''
        self.compare_on = False
        self.compare_slug = ''
        self.wipe_frac = 0.5
        self._drag_start = None
        self._drag_orig = None
        self.zoom_var = tk.StringVar(value='—')
        self.caption_var = tk.StringVar(value='Waiting for the first stack…')

        toolbar = ttk.Frame(self)
        toolbar.pack(fill='x', pady=(0, 4))
        ttk.Button(toolbar, text='Fit', width=5, command=self.fit).pack(side='left')
        ttk.Button(toolbar, text='1:1', width=5, command=self.one_to_one).pack(side='left', padx=4)
        ttk.Label(toolbar, textvariable=self.zoom_var, width=6, style='Dim.TLabel').pack(side='left')

        self.canvas = tk.Canvas(self, bg=_WELL, highlightthickness=1,
                                highlightbackground=_LINE)
        self.canvas.pack(fill='both', expand=True)
        ttk.Label(self, textvariable=self.caption_var, style='Dim.TLabel').pack(
            anchor='w', pady=(4, 0))

        self.canvas.bind('<Configure>', lambda e: self.redraw())
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)

    # ── loading images ─────────────────────────────────────────────────

    def load_slot(self, jpeg_bytes: bytes, slug: str, caption: str) -> None:
        from PIL import Image
        img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
        first = self._img_a is None
        self._img_a = img
        self._nat_w, self._nat_h = img.size
        self.current_slug = slug
        self.caption_var.set(caption)
        if first:
            self.fit()
        else:
            self.redraw()

    def set_compare_slot(self, jpeg_bytes: Optional[bytes], slug: str) -> None:
        from PIL import Image
        self.compare_slug = slug
        self._img_b = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB') if jpeg_bytes else None
        self.redraw()

    def replace_pixels(self, jpeg_bytes: bytes) -> None:
        """Swap the currently-viewed image (e.g. after a re-stretch) without
        resetting pan/zoom, unlike ``load_slot``."""
        from PIL import Image
        self._img_a = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
        self.redraw()

    # ── view transform ─────────────────────────────────────────────────

    def fit(self) -> None:
        if not self._nat_w or not self._nat_h:
            self.redraw()
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        self.scale = self.fit_scale = min(cw / self._nat_w, ch / self._nat_h)
        self.tx = (cw - self._nat_w * self.scale) / 2
        self.ty = (ch - self._nat_h * self.scale) / 2
        self.redraw()

    def one_to_one(self) -> None:
        if not self._nat_w:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        cx, cy = cw / 2, ch / 2
        ix, iy = (cx - self.tx) / self.scale, (cy - self.ty) / self.scale
        self.scale = 1.0
        self.tx, self.ty = cx - ix * self.scale, cy - iy * self.scale
        self.redraw()

    def _on_wheel(self, event) -> None:
        if not self._nat_w:
            return
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        ix = (event.x - self.tx) / self.scale
        iy = (event.y - self.ty) / self.scale
        self.scale = max(self.fit_scale * 0.5, min(self.scale * factor, 40.0))
        self.tx = event.x - ix * self.scale
        self.ty = event.y - iy * self.scale
        self.redraw()

    def _on_press(self, event) -> None:
        self._drag_start = (event.x, event.y)
        self._drag_orig = (self.tx, self.ty)

    def _on_drag(self, event) -> None:
        if self._drag_start is None:
            return
        dx, dy = event.x - self._drag_start[0], event.y - self._drag_start[1]
        self.tx = self._drag_orig[0] + dx
        self.ty = self._drag_orig[1] + dy
        self.redraw()

    # ── compare ─────────────────────────────────────────────────────────

    def set_compare(self, on: bool) -> None:
        self.compare_on = on
        self.redraw()

    def set_wipe(self, frac: float) -> None:
        self.wipe_frac = max(0.0, min(1.0, frac))
        self.redraw()

    # ── draw ────────────────────────────────────────────────────────────

    def redraw(self) -> None:
        from PIL import Image, ImageTk
        c = self.canvas
        c.delete('all')
        cw, ch = max(c.winfo_width(), 1), max(c.winfo_height(), 1)
        if self._img_a is None:
            c.create_text(cw / 2, ch / 2, text='◆ ORIGINSTACK\nWaiting for the first stack preview…',
                          fill=_TEXT_FAINT, justify='center', font=('Segoe UI', 11))
            self.zoom_var.set('—')
            return
        disp_w = max(1, int(self._nat_w * self.scale))
        disp_h = max(1, int(self._nat_h * self.scale))
        frame = self._img_a.resize((disp_w, disp_h), Image.BILINEAR)
        if self.compare_on and self._img_b is not None:
            frame_b = self._img_b.resize((disp_w, disp_h), Image.BILINEAR)
            wipe_x_img = int(self.wipe_frac * cw - self.tx)
            if wipe_x_img <= 0:
                frame = frame_b
            elif wipe_x_img < disp_w:
                frame = frame.copy()
                frame.paste(frame_b.crop((wipe_x_img, 0, disp_w, disp_h)), (wipe_x_img, 0))
        self._photo = ImageTk.PhotoImage(frame)
        c.create_image(self.tx, self.ty, anchor='nw', image=self._photo)
        if self.compare_on:
            wipe_x = self.wipe_frac * cw
            c.create_line(wipe_x, 0, wipe_x, ch, fill=_ACCENT2, width=2)
        self.zoom_var.set(f'{round(self.scale / self.fit_scale * 100) if self.fit_scale else 100}%')


class FrameStrip(ttk.Frame):
    """Horizontally-scrollable per-frame thumbnail ring (Phase 1)."""

    def __init__(self, parent, on_click):
        super().__init__(parent)
        self.on_click = on_click
        self.canvas = tk.Canvas(self, height=88, bg=_BG, highlightthickness=0)
        hbar = ttk.Scrollbar(self, orient='horizontal', command=self.canvas.xview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind('<Configure>',
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.canvas.configure(xscrollcommand=hbar.set)
        self.canvas.pack(fill='x', expand=True)
        hbar.pack(fill='x')
        self._shown_ids: set = set()
        self._photos: List[Any] = []

    def add_thumb(self, fid: int, name: str, jpeg_bytes: bytes) -> None:
        if fid in self._shown_ids:
            return
        from PIL import Image, ImageTk
        img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
        img.thumbnail((96, 72))
        photo = ImageTk.PhotoImage(img)
        self._photos.append(photo)
        col = len(self._shown_ids)
        self._shown_ids.add(fid)
        btn = tk.Button(self.inner, image=photo, bd=1, relief='flat',
                        background=_WELL, activebackground=_LINE,
                        highlightthickness=1, highlightbackground=_LINE,
                        command=lambda f=fid: self.on_click(f))
        btn.grid(row=0, column=col, padx=3)
        ttk.Label(self.inner, text=name[:14], style='Faint.TLabel',
                 font=('Consolas', 8)).grid(row=1, column=col)


class App:
    """Wires the Setup form, RunManager, and progress/preview panels
    together, polling ``UIEvents.snapshot()`` on a ``root.after()`` timer --
    the pipeline runs on ``RunManager``'s background thread, and tkinter
    widgets may only be touched from the main thread, so nothing below is
    ever updated from inside a publish call itself."""

    POLL_MS = 150

    def __init__(self, root: tk.Tk) -> None:
        from src.desktop_control import get_run_manager
        from src.ui_events import get_ui_events
        self.root = root
        self.ui = get_ui_events()
        self.ui.attach()
        self.rm = get_run_manager()
        self._last_version = -1
        self._shown_log_lines = 0
        self._last_run_status = 'idle'

        root.title('OriginStack')
        root.geometry('1400x900')
        root.minsize(900, 600)
        icon = Path(__file__).resolve().parent.parent / 'packaging' / 'icon.ico'
        if icon.exists():
            try:
                root.iconbitmap(str(icon))
            except tk.TclError:
                pass

        _apply_theme(root)
        self._build_header(root)

        paned = ttk.PanedWindow(root, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=10, pady=(4, 10))
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=1)

        self._build_left(left)
        self._build_right(right)

        # ttk.PanedWindow only honors `weight` once the user drags the sash
        # by hand -- on first layout each pane gets its content's natural
        # (requested) size instead, which left the Setup form's wide tabs
        # swallowing nearly the whole window. Force an even split explicitly
        # once real geometry is available.
        root.update_idletasks()
        paned.sashpos(0, root.winfo_width() // 2)

        root.protocol('WM_DELETE_WINDOW', self._on_closing)
        root.after(self.POLL_MS, self._poll)

    def _build_header(self, root: tk.Tk) -> None:
        from src.utils import read_version
        header = tk.Frame(root, background=_PANEL, height=44)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)
        inner = tk.Frame(header, background=_PANEL)
        inner.pack(fill='both', expand=True, padx=16)
        tk.Label(inner, text='◆', background=_PANEL, foreground=_ACCENT,
                font=('Segoe UI', 13)).pack(side='left', pady=8)
        tk.Label(inner, text='ORIGINSTACK', background=_PANEL, foreground=_TEXT,
                font=('Segoe UI', 12, 'bold')).pack(side='left', padx=(8, 0), pady=8)
        tk.Label(inner, text=f'v{read_version()}', background=_PANEL, foreground=_TEXT_FAINT,
                font=_FONT_MONO).pack(side='left', padx=(10, 0), pady=8)
        self.header_status_var = tk.StringVar(value='Idle')
        tk.Label(inner, textvariable=self.header_status_var, background=_PANEL,
                foreground=_TEXT_DIM, font=_FONT_MONO).pack(side='right', pady=8)
        tk.Frame(root, background=_LINE, height=1).pack(fill='x', side='top')

    # ── left column: setup, run controls, progress, log ───────────────

    def _build_left(self, parent: ttk.Frame) -> None:
        self.form = SetupForm(parent)
        self.form.pack(fill='both', expand=True)

        run_row = ttk.Frame(parent)
        run_row.pack(fill='x', pady=10)
        self.start_btn = ttk.Button(run_row, text='Start', style='Accent.TButton',
                                    command=self._on_start)
        self.start_btn.pack(side='left')
        self.status_var = tk.StringVar(value='Idle.')
        ttk.Label(run_row, textvariable=self.status_var, style='Dim.TLabel').pack(
            side='left', padx=10)
        self.open_folder_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(run_row, text='Open folder when done',
                        variable=self.open_folder_var).pack(side='left', padx=(10, 0))
        ttk.Button(run_row, text='Open Folder', command=self._open_output_folder).pack(
            side='left', padx=(6, 0))

        phase_frame = ttk.LabelFrame(parent, text='PIPELINE')
        phase_frame.pack(fill='x', pady=6)
        self.phase_labels = []
        titles = ['1 · Quality', '2 · Registration', '3 · Stacking', '4 · Post-process']
        for i, t in enumerate(titles):
            lbl = ttk.Label(phase_frame, text=t, style='Phase.TLabel', anchor='center')
            lbl.grid(row=0, column=i, sticky='we', padx=(4 if i == 0 else 2,
                                                          2 if i < 3 else 4), pady=(8, 4))
            phase_frame.grid_columnconfigure(i, weight=1)
            self.phase_labels.append(lbl)
        self.progress_bar = ttk.Progressbar(phase_frame, mode='determinate')
        self.progress_bar.grid(row=1, column=0, columnspan=4, sticky='we', padx=4, pady=(0, 4))
        self.progress_var = tk.StringVar(value='')
        ttk.Label(phase_frame, textvariable=self.progress_var).grid(
            row=2, column=0, columnspan=4, sticky='w', padx=4, pady=(0, 4))

        frames_frame = ttk.LabelFrame(parent, text='RECENT FRAMES')
        frames_frame.pack(fill='x', pady=6)
        cols = ('frame', 'score', 'snr', 'stars', 'fwhm')
        self.frames_tree = ttk.Treeview(frames_frame, columns=cols, show='headings', height=6)
        for c, w in zip(cols, (160, 60, 60, 50, 60)):
            self.frames_tree.heading(c, text=c.capitalize())
            self.frames_tree.column(c, width=w, anchor='e' if c != 'frame' else 'w')
        self.frames_tree.tag_configure('bad', foreground=_BAD)
        self.frames_tree.pack(fill='x')

        log_frame = ttk.LabelFrame(parent, text='LOG')
        log_frame.pack(fill='both', expand=True, pady=6)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=12, bg=_LOG_BG, fg=_LOG_FG, insertbackground=_LOG_FG,
            font=('Consolas', 9), state='disabled', wrap='word')
        self.log_text.pack(fill='both', expand=True)

    # ── right column: preview, frame strip, stretch, summary ──────────

    def _build_right(self, parent: ttk.Frame) -> None:
        preview_frame = ttk.LabelFrame(parent, text='PREVIEW')
        preview_frame.pack(fill='both', expand=True, pady=(0, 6))

        vtools = ttk.Frame(preview_frame)
        vtools.pack(fill='x', pady=4)
        self.view_var = tk.StringVar()
        self.view_combo = ttk.Combobox(vtools, textvariable=self.view_var, state='readonly', width=24)
        self.view_combo.pack(side='left')
        self.view_combo.bind('<<ComboboxSelected>>', self._on_view_selected)
        self.compare_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(vtools, text='Compare', variable=self.compare_var,
                        command=self._on_compare_toggled).pack(side='left', padx=8)
        self.compare_combo_var = tk.StringVar()
        self.compare_combo = ttk.Combobox(vtools, textvariable=self.compare_combo_var,
                                          state='readonly', width=24)
        self.compare_combo.pack(side='left')
        self.compare_combo.bind('<<ComboboxSelected>>', self._on_compare_selected)

        self.preview = PreviewCanvas(preview_frame)
        self.preview.pack(fill='both', expand=True, padx=4)

        self.wipe_scale = ttk.Scale(preview_frame, from_=0, to=1, orient='horizontal',
                                    command=self._on_wipe_moved)
        self.wipe_scale.set(0.5)

        self._known_slugs: List[str] = []

        strip_frame = ttk.LabelFrame(parent, text='FRAMES')
        strip_frame.pack(fill='x', pady=6)
        self.frame_strip = FrameStrip(strip_frame, on_click=self._on_thumb_click)
        self.frame_strip.pack(fill='x')

        stretch_frame = ttk.LabelFrame(parent, text='STRETCH')
        stretch_frame.pack(fill='x', pady=6)
        self.stretch_vars = {
            'black': tk.DoubleVar(value=0.0), 'b': tk.DoubleVar(value=8.0),
            'sp': tk.DoubleVar(value=0.15), 'hp': tk.DoubleVar(value=0.95),
        }
        specs = [('black', 'Black σ', -1, 4), ('b', 'GHS b', 0, 20),
                 ('sp', 'GHS sp', 0, 1), ('hp', 'GHS hp', 0, 1)]
        for row, (key, label, lo, hi) in enumerate(specs):
            ttk.Label(stretch_frame, text=label).grid(row=row, column=0, sticky='w', padx=4)
            ttk.Scale(stretch_frame, from_=lo, to=hi, orient='horizontal',
                     variable=self.stretch_vars[key]).grid(row=row, column=1, sticky='we', padx=4)
            stretch_frame.grid_columnconfigure(1, weight=1)
        btn_row = ttk.Frame(stretch_frame)
        btn_row.grid(row=len(specs), column=0, columnspan=2, pady=6)
        ttk.Button(btn_row, text='Apply to view', command=self._apply_stretch).pack(side='left')
        ttk.Button(btn_row, text='Reset', command=self._reset_stretch).pack(side='left', padx=6)

        summary_frame = ttk.LabelFrame(parent, text='COMPLETE')
        summary_frame.pack(fill='x', pady=6)
        self.summary_var = tk.StringVar(value='')
        ttk.Label(summary_frame, textvariable=self.summary_var, justify='left').pack(
            anchor='w', padx=4, pady=4)

    # ── run control ─────────────────────────────────────────────────────

    def _on_start(self) -> None:
        form = self.form.read_form()
        result = self.rm.start(form)
        if not result.get('ok'):
            self.status_var.set(f"Error: {result.get('error')}")
            return
        self.start_btn.state(['disabled'])
        self.status_var.set('Running…')
        self._shown_log_lines = 0
        self.frames_tree.delete(*self.frames_tree.get_children())
        self.summary_var.set('')

    def _on_closing(self) -> None:
        if self.rm.is_running():
            from src.native_dialog import ask_yes_no
            if not ask_yes_no('OriginStack',
                              'A stacking run is still in progress. Quit anyway?',
                              default=True):
                return
        self.root.destroy()

    # ── preview controls ────────────────────────────────────────────────

    def _on_view_selected(self, _event=None) -> None:
        slug = self._slug_from_label(self.view_var.get())
        data = self.ui.named_jpeg(slug)
        if data:
            slot = next((s for s in self._named_cache if s['slug'] == slug), None)
            self.preview.load_slot(data, slug, slot['caption'] if slot else slug)

    def _on_compare_toggled(self) -> None:
        self.preview.set_compare(self.compare_var.get())
        self.wipe_scale.pack(fill='x', padx=4, pady=(0, 4)) if self.compare_var.get() \
            else self.wipe_scale.pack_forget()
        if self.compare_var.get():
            self._on_compare_selected()

    def _on_compare_selected(self, _event=None) -> None:
        slug = self._slug_from_label(self.compare_combo_var.get())
        data = self.ui.named_jpeg(slug) if slug else None
        self.preview.set_compare_slot(data, slug)

    def _on_wipe_moved(self, value: str) -> None:
        self.preview.set_wipe(float(value))

    def _on_thumb_click(self, fid: int) -> None:
        data = self.ui.frame_jpeg(fid)
        if data:
            self.preview.load_slot(data, f'frame-{fid}', f'Frame #{fid}')

    def _apply_stretch(self) -> None:
        slug = self.preview.current_slug
        if not slug:
            return
        params = {k: v.get() for k, v in self.stretch_vars.items()}
        params['stretch'] = 'ghs'
        data = self.ui.restretch(slug, params)
        if data:
            self.preview.replace_pixels(data)

    def _reset_stretch(self) -> None:
        self.stretch_vars['black'].set(0.0)
        self.stretch_vars['b'].set(8.0)
        self.stretch_vars['sp'].set(0.15)
        self.stretch_vars['hp'].set(0.95)
        self._apply_stretch()

    @staticmethod
    def _slug_from_label(label: str) -> str:
        return label.rsplit(' [', 1)[-1].rstrip(']') if ' [' in label else label

    # ── poll loop ────────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            self._refresh()
        finally:
            self.root.after(self.POLL_MS, self._poll)

    def _refresh(self) -> None:
        snap = self.ui.snapshot()
        if snap['version'] == self._last_version:
            self._refresh_run_button()
            return
        self._last_version = snap['version']

        # phases / progress
        phase = snap['phase']
        run_done = snap['run_status'] == 'ok'
        for i, lbl in enumerate(self.phase_labels, start=1):
            if run_done or i < phase:
                lbl.configure(style='PhaseDone.TLabel')
            elif i == phase:
                lbl.configure(style='PhaseActive.TLabel')
            else:
                lbl.configure(style='Phase.TLabel')
        prog = snap['progress']
        total = max(prog['total'], 1)
        self.progress_bar['maximum'] = total
        self.progress_bar['value'] = prog['done']
        self.progress_var.set(f"{prog['label']} ({prog['done']}/{prog['total']})"
                              if prog['total'] else prog['label'])

        # log (append-only)
        log_lines = snap['log']
        if len(log_lines) < self._shown_log_lines:
            self._shown_log_lines = 0  # run_started() cleared it
        new_lines = log_lines[self._shown_log_lines:]
        if new_lines:
            self.log_text.configure(state='normal')
            self.log_text.insert('end', '\n'.join(new_lines) + '\n')
            self.log_text.see('end')
            self.log_text.configure(state='disabled')
            self._shown_log_lines = len(log_lines)

        # recent frames
        rows = snap['frames']
        self.frames_tree.delete(*self.frames_tree.get_children())
        for r in rows[-20:]:
            tag = () if r['ok'] else ('bad',)
            self.frames_tree.insert('', 'end', values=(r['name'], r['score'], r['snr'],
                                                        r['stars'], r['fwhm']), tags=tag)

        # named preview slots (view/compare dropdowns)
        self._named_cache = snap['named']
        labels = [f"{n['caption']} [{n['slug']}]" for n in self._named_cache]
        self.view_combo['values'] = labels
        self.compare_combo['values'] = labels
        if labels and not self.view_var.get():
            self.view_var.set(labels[-1])
            self._on_view_selected()
        elif self._named_cache and self._named_cache[-1]['slug'] != self.preview.current_slug \
                and not self.compare_var.get():
            # follow the latest milestone unless the user is mid-compare
            self.view_var.set(labels[-1])
            self._on_view_selected()

        # per-frame thumbnail ring
        for f in snap['frames_img']:
            data = self.ui.frame_jpeg(f['id'])
            if data:
                self.frame_strip.add_thumb(f['id'], f['name'], data)

        # summary
        if snap['summary']:
            lines = [f"{k}: {v}" for k, v in snap['summary'].items()]
            self.summary_var.set('\n'.join(lines))

        self._refresh_run_button(snap)

    def _open_output_folder(self) -> None:
        var = self.form.vars.get('output')
        out_path = var.get().strip() if var is not None else ''
        if not out_path:
            return
        folder = os.path.dirname(out_path) or '.'
        if not os.path.isdir(folder):
            return
        try:
            os.startfile(folder)
        except Exception:
            pass

    def _refresh_run_button(self, snap: Optional[Dict[str, Any]] = None) -> None:
        running = self.rm.is_running()
        if running:
            self.start_btn.state(['disabled'])
            self.header_status_var.set('Running…')
        else:
            self.start_btn.state(['!disabled'])
            if snap is not None and snap['run_status'] in ('ok', 'error'):
                done_ok = snap['run_status'] == 'ok'
                self.status_var.set('Done.' if done_ok else f"Failed: {snap['run_error']}")
                self.header_status_var.set('Complete' if done_ok else 'Failed')
                # Edge-triggered (not every poll tick) so it only pops once
                # per completed run, not repeatedly while the status holds.
                if done_ok and self._last_run_status != 'ok' and self.open_folder_var.get():
                    self._open_output_folder()
            elif snap is None or snap['run_status'] == 'idle':
                self.header_status_var.set('Idle')
        if snap is not None:
            self._last_run_status = snap['run_status']


def _run_headless(cli_argv: List[str]) -> int:
    """``--verify-headless <cli args...>``: skip the GUI/mainloop entirely and
    run a real stack through this exact frozen entry point, forwarding the
    rest of argv straight to ``cli.parse_args`` unchanged (same syntax as
    ``originstack.py``). Exists for packaging/verify_build.ps1, which needs
    to trigger a real multi-worker run through the packaged exe to prove
    ``multiprocessing.freeze_support()`` above still works -- the old
    dashboard's ``POST /api/start`` did this over HTTP; there's no server to
    do it through anymore, so this is a direct in-process replacement."""
    from src.cli import apply_post_parse_setup, parse_args, process_directory
    try:
        args = parse_args(cli_argv)
        apply_post_parse_setup(args)
        process_directory(args.directory, args.output, args)
        return 0
    except (Exception, SystemExit) as e:
        print(f"--verify-headless run failed: {e}")
        return 1


def main() -> int:
    import sys
    _log_startup_status()
    if len(sys.argv) > 1 and sys.argv[1] == '--verify-headless':
        return _run_headless(sys.argv[2:])

    root = tk.Tk()
    try:
        App(root)
    except Exception as e:
        return _fatal("OriginStack", f"Failed to open the app window: {e}")
    try:
        root.mainloop()
    except Exception as e:
        return _fatal("OriginStack", f"Unexpected error: {e}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
