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
import sys
import threading
import tkinter as tk
import traceback
import webbrowser
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk
from typing import Any, Dict, List, Optional, Tuple

from src.i18n import N_, _


def _log_dir() -> Path:
    r"""Per-user log folder: %LOCALAPPDATA%\OriginStack\logs on Windows, ~/Library/Logs
    on macOS, $XDG_STATE_HOME (default ~/.local/state) elsewhere -- never the current
    directory, which for a packaged app is wherever the launcher happened to start it."""
    if sys.platform == 'win32':
        return Path(os.environ.get('LOCALAPPDATA', '.')) / 'OriginStack' / 'logs'
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Logs' / 'OriginStack'
    state = os.environ.get('XDG_STATE_HOME') or str(Path.home() / '.local' / 'state')
    return Path(state) / 'OriginStack' / 'logs'


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
# Segoe UI / Consolas ship with Windows; DejaVu is on essentially every Linux desktop.
_SANS = 'Segoe UI' if sys.platform == 'win32' else 'DejaVu Sans'
_MONO = 'Consolas' if sys.platform == 'win32' else 'DejaVu Sans Mono'
_FONT = (_SANS, 9)
_FONT_BOLD = (_SANS, 9, 'bold')
_FONT_MONO = (_MONO, 9)
_CJK_SANS = {'ja': 'Yu Gothic UI', 'zh_CN': 'Microsoft YaHei UI'}


def _set_fonts(language: str) -> None:
    """Use a face with the language's glyphs on Windows: Segoe UI has no CJK
    characters, and Tk's per-glyph fallback mixes mismatched faces."""
    global _SANS, _FONT, _FONT_BOLD
    if sys.platform == 'win32' and language in _CJK_SANS:
        _SANS = _CJK_SANS[language]
        _FONT = (_SANS, 9)
        _FONT_BOLD = (_SANS, 9, 'bold')


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
    # Setup form: the one-line description under each field, the dot marking
    # a field changed from its default, and that field's "reset" link.
    style.configure('Hint.TLabel', background=_BG, foreground=_TEXT_DIM, font=(_SANS, 8))
    style.configure('Changed.TLabel', background=_BG, foreground=_ACCENT, font=_FONT)
    style.configure('Reset.TLabel', background=_BG, foreground=_ACCENT2, font=(_SANS, 8))
    style.configure('Header.TLabel', background=_BG, foreground=_TEXT,
                    font=(_SANS, 14, 'bold'))
    style.configure('Accent.TLabel', background=_BG, foreground=_ACCENT,
                    font=(_SANS, 14, 'bold'))

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
                    foreground=_TEXT_DIM, font=(_SANS, 9, 'bold'))

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
                    font=(_SANS, 8, 'bold'), relief='flat')
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
    """A vertically-scrollable container -- the Setup form (with a field's
    description under it, and the additional options expanded) is taller
    than the window."""

    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=_BG)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind('<Configure>',
                        lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        window = canvas.create_window((0, 0), window=self.inner, anchor='nw')
        # The inner frame follows the canvas width (it would otherwise keep
        # its requested width), so fields fill it and description lines can
        # wrap to it.
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.canvas = canvas

        def _wheel(event):
            # bind_all sees every wheel event in the app: scroll only when
            # the pointer is over this frame, not the log or the preview.
            if str(event.widget).startswith(str(canvas)):
                canvas.yview_scroll(-1 * (event.delta // 120), 'units')
        canvas.bind_all('<MouseWheel>', _wheel, add='+')

    def scroll_to(self, widget: tk.Widget) -> None:
        """Scroll so *widget* (a descendant of ``inner``) is at the top."""
        self.update_idletasks()
        total = self.inner.winfo_height()
        if total > 0:
            y = widget.winfo_rooty() - self.inner.winfo_rooty()
            self.canvas.yview_moveto(max(0.0, y / total))


#  Shown directly on the main form, in this order -- everything else lives
# behind the "Additional options" toggle. This is the same small set the
# old browser dashboard treated as "quick fields", chosen because they're
# the ones almost every run touches (where the lights are, what to call the
# output, and the handful of settings that most change the result) while
# the other ~110 flags are fine-tuning most runs never need.
_COMMON_DESTS = ['directory', 'output', 'temp_dir', 'auto', 'stack_method',
                 'denoiser', 'deconvolve_mode', 'drizzle_scale', 'trail_reject',
                 'use_gpu', 'parallel']

# Human field labels. Anything not listed falls back to the dest name with
# underscores spaced and the first letter capitalised ("stack_method" ->
# "Stack method"). Only the cases where that reads badly are overridden.
_FIELD_LABELS = {
    'directory': N_('Light frames'),
    'output': N_('Output'),
    'temp_dir': N_('Temp folder'),
    'parallel': N_('Workers'),
    'use_gpu': N_('Use GPU'),
    'trail_reject': N_('Trail rejection'),
    'drizzle_scale': N_('Drizzle scale'),
    'plate_solve': N_('Plate solve'),
    'color_calibrate': N_('Colour calibrate'),
    'remove_stars': N_('Starless sidecar'),
    'deconvolve_mode': N_('Deconvolution'),
    'session_cfa_eq': N_('Session colour equalisation'),
    'ca_correction': N_('Chromatic aberration fix'),
    'cfa_drizzle': N_('Bayer drizzle'),
    'dark_temp_model': N_('Dark temperature model'),
    'super_res_iters': N_('Super-resolution passes'),
    'bg_method': N_('Background method'),
    'scnr': N_('SCNR (green removal)'),
}


# Groups and fields hidden until "Show expert options" is ticked: fine-tuning
# that --auto manages, diagnostics, and features the project's own testing
# found not to help on typical data. The CLI is unaffected.
_EXPERT_GROUP_PREFIXES = ('Advanced', 'Diagnostics')
_EXPERT_DESTS = {'cfa_drizzle', 'nmf_separate', 'starless_process', 'layered_stretch',
                 'bg_method', 'fix_atmospheric_dispersion', 'distortion_model'}


def _is_expert_group(title: str) -> bool:
    return title.startswith(_EXPERT_GROUP_PREFIXES)


def _is_expert_field(field: Dict[str, Any]) -> bool:
    return field['dest'] in _EXPERT_DESTS or 'experimental' in (field.get('help') or '').lower()


# Required paths: being set is not a change worth marking.
_NO_CHANGE_MARK = {'directory', 'output', 'temp_dir'}


def _field_label(field: Dict[str, Any]) -> str:
    """The English label, which is also its translation key."""
    dest = field['dest']
    return _FIELD_LABELS.get(dest, dest.replace('_', ' ').capitalize())


def _field_label_ui(field: Dict[str, Any]) -> str:
    return _(_field_label(field))


# ── Setup: "what did you image?" cards and the run mode ─────────────

_EXAMPLES_DIR = Path(__file__).resolve().parent / 'data' / 'examples'

# key, title, description, target_type value, comet_mode value, example file,
# example caption. A card is only a view over the target_type/comet_mode
# fields (the same Variables the Additional options widgets use), so a pick
# reaches the run through read_form() like any other change. Descriptions
# say only what auto_settings._TARGET_SETTINGS does for that type.
_TARGET_CARDS = (
    ('auto', N_('Auto-detect'),
     N_('Recognised from the session, header or folder name, then the frames.'),
     '', False, None, None),
    ('galaxy', N_('Galaxy'),
     N_('Keeps the faint halo out of background removal; trims star size.'),
     'galaxy', False, 'galaxy.jpg', N_('Example galaxy: Whirlpool Galaxy (M51), Celestron Origin')),
    ('nebula', N_('Nebula'),
     N_('Diffuse emission: a gentler stretch, stars kept full size.'),
     'emission_nebula', False, 'nebula.jpg', N_('Example nebula: Omega Nebula (M17), Celestron Origin')),
    ('cluster', N_('Star cluster'),
     N_('Dense clusters: a dark sky background, stars kept full size.'),
     'globular_cluster', False, None, None),
    ('starfield', N_('Star field'),
     N_('Rich star fields and Milky Way regions; light local contrast.'),
     'star_field', False, 'starfield.jpg',
     N_('Example star field: Sagittarius Star Cloud (M24), Celestron Origin')),
    ('comet', N_('Comet'),
     N_('Also stacks on the moving nucleus (saved as a second _comet image).'),
     '', True, 'comet.jpg', N_('Example comet: C/2025 R2 (SWAN), Celestron Origin')),
)
# target_inference's source codes, as the phrase shown after a target name.
_SOURCE_PHRASES = {'session': N_('from the session file'), 'header': N_('from the FITS header'),
                   'folder': N_('from the folder name'), 'filename': N_('from the file name'),
                   'simbad': N_('from SIMBAD')}


def _from_source(src: Optional[str]) -> str:
    return _(_SOURCE_PHRASES[src]) if src in _SOURCE_PHRASES else ''


def _target_label(target_type: str) -> str:
    from src.auto_settings import TARGET_LABELS
    return _(TARGET_LABELS.get(target_type, target_type))


# Inferred target types -> the card that shows them.
_TYPE_TO_CARD = {'galaxy': 'galaxy', 'emission_nebula': 'nebula', 'reflection_nebula': 'nebula',
                 'planetary_nebula': 'nebula', 'globular_cluster': 'cluster',
                 'star_field': 'starfield', 'wide_field': 'starfield'}

# key, title, description, preset value. Only two: the default settings are
# the measured best, and --preset quality forces per-frame L.A.Cosmic (~2.5
# min for <0.3% change) and deconvolution (no help at typical Origin SNR), so
# offering it as "best quality" would sell a slower run, not a better one.
_RUN_MODES = (
    ('full', N_('Full quality'), N_('Recommended. Every step that measured as an improvement.'), ''),
    ('quick', N_('Quick look'), N_('Faster check: no outlier rejection, lighter processing.'), 'quick'),
)
# Fields the cards and run mode set; not counted as "changed" settings.
_PICKER_DESTS = {'target_type', 'comet_mode', 'preset'}

_THUMB = (132, 84)


def _card_thumbnail(card: tuple):
    """A PhotoImage for a card: its example result scaled down, or a small
    drawing for cards without one. None without Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageTk
    except ImportError:
        return None
    key, example = card[0], card[5]
    w, h = _THUMB
    if example and (_EXAMPLES_DIR / example).exists():
        im = Image.open(_EXAMPLES_DIR / example).convert('RGB')
        # Centre crop to the thumbnail's aspect, a little zoomed in.
        sw, sh = im.size
        cw = int(sw * 0.7)
        ch = int(cw * h / w)
        im = im.crop(((sw - cw) // 2, (sh - ch) // 2, (sw + cw) // 2, (sh + ch) // 2))
        return ImageTk.PhotoImage(im.resize(_THUMB, Image.LANCZOS))

    import random
    im = Image.new('RGB', _THUMB, (6, 7, 10))
    d = ImageDraw.Draw(im)
    rnd = random.Random(7)
    for _i in range(50):
        v = rnd.randrange(60, 190)
        d.point((rnd.randrange(w), rnd.randrange(h)), fill=(v, v, v))
    if key == 'cluster':
        for _i in range(260):
            x, y = rnd.gauss(w / 2, w / 9), rnd.gauss(h / 2, h / 7)
            v = rnd.randrange(150, 255)
            d.point((x, y), fill=(v, v, int(v * 0.9)))
        glow = Image.new('L', _THUMB, 0)
        ImageDraw.Draw(glow).ellipse((w / 2 - 14, h / 2 - 14, w / 2 + 14, h / 2 + 14), fill=90)
        im.paste((255, 240, 210), mask=glow.filter(ImageFilter.GaussianBlur(8)))
    else:  # auto
        d.text((w / 2 - 14, h / 2 - 6), 'AUTO', fill=(232, 230, 223))
    return ImageTk.PhotoImage(im)


class _Choice(tk.Frame):
    """A clickable bordered card: optional picture, title, optional badge,
    description. ``set_state`` switches the border between selected and not."""

    def __init__(self, parent, title: str, text: str, on_click, photo=None):
        super().__init__(parent, background=_LINE, padx=1, pady=1, cursor='hand2')
        self.body = tk.Frame(self, background=_PANEL)
        self.body.pack(fill='both', expand=True)
        self.photo = photo  # keep-alive ref
        if photo is not None:
            tk.Label(self.body, image=photo, background=_PANEL, borderwidth=0).pack(
                padx=6, pady=(6, 3))
        row = tk.Frame(self.body, background=_PANEL)
        row.pack(fill='x', padx=8, pady=(4 if photo is None else 0, 0))
        tk.Label(row, text=title, background=_PANEL, foreground=_TEXT,
                 font=_FONT_BOLD).pack(side='left')
        self.badge = tk.Label(row, text='', background=_PANEL, foreground='#04161a',
                              font=(_SANS, 7, 'bold'), padx=0)
        self.badge.pack(side='right')
        self.text = tk.Label(self.body, text=text, background=_PANEL, foreground=_TEXT_DIM,
                             font=(_SANS, 8), wraplength=140, justify='left', anchor='w')
        self.text.pack(fill='x', padx=8, pady=(0, 7))
        for w in self._descendants(self):
            w.bind('<Button-1>', lambda _e: on_click())

    @staticmethod
    def _descendants(w):
        yield w
        for c in w.winfo_children():
            yield from _Choice._descendants(c)

    def set_state(self, selected: bool) -> None:
        self.configure(background=_ACCENT if selected else _LINE,
                       padx=2 if selected else 1, pady=2 if selected else 1)

    def set_badge(self, text: str) -> None:
        self.badge.configure(text=text, padx=4 if text else 0,
                             background=_ACCENT2 if text else _PANEL)


class GoalPicker(ttk.Frame):
    """The two questions at the top of the Setup form: what was imaged
    (target cards) and how to run (full quality / quick look). Both write
    the existing ``target_type``/``comet_mode``/``preset`` Variables, and
    follow them when they are changed from Additional options instead."""

    def __init__(self, parent, vars_: Dict[str, tk.Variable], on_pick=None):
        super().__init__(parent)
        self.vars = vars_
        self._on_pick = on_pick
        self.suggested: Optional[str] = None

        ttk.Label(self, text=_('WHAT DID YOU IMAGE?'), style='TLabelframe.Label').pack(
            anchor='w', pady=(10, 4))
        grid = ttk.Frame(self)
        grid.pack(fill='x')
        self.cards: Dict[str, _Choice] = {}
        for i, card in enumerate(_TARGET_CARDS):
            c = _Choice(grid, _(card[1]), _(card[2]), lambda c=card: self.pick_target(c[0]),
                        photo=_card_thumbnail(card))
            c.grid(row=i // 3, column=i % 3, padx=3, pady=3, sticky='nsew')
            self.cards[card[0]] = c
        for col in range(3):
            grid.grid_columnconfigure(col, weight=1, uniform='card')

        ttk.Label(self, text=_('HOW SHOULD IT RUN?'), style='TLabelframe.Label').pack(
            anchor='w', pady=(12, 4))
        modes = ttk.Frame(self)
        modes.pack(fill='x')
        self.modes: Dict[str, _Choice] = {}
        for i, (key, title, text, _preset) in enumerate(_RUN_MODES):
            m = _Choice(modes, _(title), _(text), lambda k=key: self.pick_mode(k))
            m.text.configure(wraplength=220)
            m.grid(row=0, column=i, padx=3, sticky='nsew')
            modes.grid_columnconfigure(i, weight=1, uniform='mode')
            self.modes[key] = m

        for dest in _PICKER_DESTS:
            self.vars[dest].trace_add('write', lambda *_: self.refresh())
        self.refresh()

    # ── selection ──────────────────────────────────────────────────────

    def selected_target(self) -> Optional[str]:
        """The card matching the fields, or None for a type no card shows
        (planetary nebula, reflection nebula... chosen in Additional options)."""
        if self.vars['comet_mode'].get():
            return 'comet'
        tt = self.vars['target_type'].get()
        for card in _TARGET_CARDS:
            if card[3] == tt and not card[4]:
                return card[0]
        return None

    def selected_mode(self) -> Optional[str]:
        preset = self.vars['preset'].get()
        return next((m[0] for m in _RUN_MODES if m[3] == preset), None)

    def pick_target(self, key: str) -> None:
        card = next(c for c in _TARGET_CARDS if c[0] == key)
        self.vars['target_type'].set(card[3])
        self.vars['comet_mode'].set(card[4])
        if self._on_pick is not None:
            self._on_pick()

    def pick_mode(self, key: str) -> None:
        self.vars['preset'].set(next(m[3] for m in _RUN_MODES if m[0] == key))
        if self._on_pick is not None:
            self._on_pick()

    def set_suggestion(self, target_type: Optional[str], name: Optional[str]) -> None:
        """Badge the card matching what the session/header says, and tell the
        Auto-detect card what it will use."""
        self.suggested = _TYPE_TO_CARD.get(target_type or '')
        auto_text = _(_TARGET_CARDS[0][2])
        if self.suggested and target_type:
            label = _target_label(target_type)
            auto_text = (_('Will start from: {label} ({name}).', label=label, name=name) if name
                         else _('Will start from: {label}.', label=label))
        self.cards['auto'].text.configure(text=auto_text)
        self.refresh()

    def refresh(self) -> None:
        target, mode = self.selected_target(), self.selected_mode()
        for key, card in self.cards.items():
            card.set_state(key == target)
            card.set_badge(_('SUGGESTED') if key == self.suggested and key != target else '')
        for key, m in self.modes.items():
            m.set_state(key == mode)

    def example(self) -> Tuple[Optional[Path], str]:
        """The example image and caption for the selected card (or the
        suggested one under Auto-detect)."""
        key = self.selected_target()
        if key == 'auto' and self.suggested:
            key = self.suggested
        card = next((c for c in _TARGET_CARDS if c[0] == key), None)
        if card and card[5] and (_EXAMPLES_DIR / card[5]).exists():
            return _EXAMPLES_DIR / card[5], _(card[6])
        return None, ''


class SetupForm(ttk.Frame):
    """Auto-built from ``desktop_control.get_form_schema()``. A fixed set of
    common fields (``_COMMON_DESTS``) sits directly on the form; every other
    field lives behind a collapsed "Additional options" section (a sidebar
    of argparse groups, expanded on demand) so a typical run only has to
    look at ~10 fields, not the full ~120-flag surface. Field kinds
    (``bool_true``/``bool_false``/``select``/``list``/``number``/``text``)
    and path widgets (``dir``/``file-open``/``file-save``) come straight
    from the schema; nothing here needs to know about individual flags.

    Each field shows its one-line ``summary`` under it (the full help stays
    in the label's tooltip), a dot and a "reset" link once it differs from
    its default, and the additional options can be searched by label, flag
    or help text across every group (expert ones included -- someone who
    types "dispersion" is looking for it)."""

    def __init__(self, parent, on_expand=None, on_change=None):
        super().__init__(parent)
        # Called (once per burst of edits, at idle) after any field changes,
        # so the App can refresh its "This run" summary and example image.
        self._on_change = on_change
        self._change_pending = False
        self._labels: Dict[str, str] = {}
        self._data_summary = ''
        # (name, type, source) the session/header/folder suggest for the
        # chosen folder -- local lookups only, no SIMBAD from the form.
        self.inferred: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None)
        from src.desktop_control import get_form_schema
        self.schema = get_form_schema()
        # Called with the toggle bar after "Additional options" opens, so a
        # scrolling container can bring the section into view.
        self._on_expand = on_expand
        # parent -> [(hint label, inline)], rewrapped to the parent's width.
        self._hints: Dict[tk.Widget, List[Any]] = {}
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
        # galaxy-specific denoiser choice never got a chance to apply.
        self._initial: Dict[str, Any] = {}
        # dest -> {'widgets', 'group', 'expert', 'haystack', 'dot', 'reset'}
        # for every field built (common fields have group None).
        self._fields: Dict[str, Dict[str, Any]] = {}
        self.dir_count_var = tk.StringVar(value='')
        all_fields = {f['dest']: f for fields in self.schema.values() for f in fields}

        common_frame = ttk.Frame(self)
        common_frame.pack(fill='x')
        row = 0
        goal_row = 0
        for dest in _COMMON_DESTS:
            field = all_fields.get(dest)
            if field is not None:
                row += self._build_field(common_frame, row, field, group=None)
            if dest == 'temp_dir':
                goal_row, row = row, row + 1  # the target cards and run mode go here

        bar = ttk.Frame(self)
        bar.pack(fill='x', pady=(10, 0))
        self._bar = bar
        self._advanced_shown = False
        self._toggle_btn = ttk.Button(bar, text='', command=self._toggle_advanced)
        self._toggle_btn.pack(side='left', fill='x', expand=True)
        self.search_var = tk.StringVar(value='')
        ttk.Entry(bar, textvariable=self.search_var, width=22).pack(side='right', padx=(8, 0))
        ttk.Label(bar, text=_('Search settings'), style='Dim.TLabel').pack(side='right')
        self.search_var.trace_add('write', lambda *_: self._on_search())

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
        self._no_match = ttk.Label(content, text=_('No settings match.'), style='Dim.TLabel')

        # Plain frames: the whole form scrolls in the App's ScrollableFrame
        # (a scroll area per page nested inside that one fought it for the
        # mouse wheel).
        self._pages: Dict[str, ttk.Frame] = {}
        self._nav_labels: Dict[str, tk.Label] = {}
        self._current_group: Optional[str] = None
        for group_title in self.schema:
            self._pages[group_title] = ttk.Frame(content)
            lbl = tk.Label(nav, text=_(group_title), background=_PANEL, foreground=_TEXT_DIM,
                          font=_FONT, anchor='w', justify='left', wraplength=148,
                          padx=12, pady=9)
            lbl.bind('<Button-1>', lambda _e, g=group_title: self._show_group(g))
            self._nav_labels[group_title] = lbl
        self._nav = nav

        self.expert_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(nav, text=_('Expert options'), variable=self.expert_var,
                        command=self._refresh_expert).pack(side='bottom', anchor='w',
                                                           padx=8, pady=8)

        # One widget per dest: a second widget would create a second
        # Variable, and read_form() would silently drop whichever one the
        # user didn't touch last. Common fields are already built; a dest
        # two flags share (--cosmic-ray-rejection / --no-cosmic-ray-rejection)
        # gets the widget of its *last* action, the one
        # build_argv_from_form's dest map resolves it to.
        last = {f['dest']: f for fields in self.schema.values() for f in fields}
        for group_title, fields in self.schema.items():
            page = self._pages[group_title]
            row = 0
            for field in fields:
                if field['dest'] not in self._fields and last[field['dest']] is field:
                    row += self._build_field(page, row, field, group=group_title)

        # Built last: it drives the target_type/comet_mode/preset Variables
        # created with their Additional options widgets above.
        self.goal = GoalPicker(common_frame, self.vars, on_pick=self._notify_change)
        self.goal.grid(row=goal_row, column=0, columnspan=5, sticky='we', pady=(0, 8))

        self._refresh_toggle_text()
        self._refresh_expert()

    # ── visibility: expert toggle and search ───────────────────────────

    def _refresh_expert(self) -> None:
        """Show or hide expert groups/fields (or, while a search is active,
        the matching fields). Hidden fields keep their Variables, so
        ``read_form`` is unaffected -- only visibility changes."""
        expert = self.expert_var.get()
        terms = self.search_var.get().lower().split()
        matches: Dict[str, int] = {g: 0 for g in self.schema}
        for info in self._fields.values():
            if info['group'] is None:
                continue
            if terms:
                visible = all(t in info['haystack'] for t in terms)
            else:
                visible = expert or not info['expert']
            if visible:
                matches[info['group']] += 1
            for w in info['widgets']:
                if visible:
                    w.grid()
                else:
                    w.grid_remove()

        for lbl in self._nav_labels.values():
            lbl.pack_forget()
        if terms:
            visible_groups = [g for g in self.schema if matches[g]]
        else:
            visible_groups = [g for g in self.schema if expert or not _is_expert_group(g)]
        for g, lbl in self._nav_labels.items():
            if g in visible_groups:
                lbl.configure(text=f'{_(g)}  ({matches[g]})' if terms else _(g))
                lbl.pack(fill='x')

        if not visible_groups:
            self._current_group = None
            for page in self._pages.values():
                page.pack_forget()
            self._no_match.pack(anchor='nw', pady=8)
            return
        self._no_match.pack_forget()
        self._show_group(self._current_group if self._current_group in visible_groups
                         else visible_groups[0])

    def _on_search(self) -> None:
        if self.search_var.get().strip() and not self._advanced_shown:
            self._toggle_advanced()
        self._refresh_expert()

    def _toggle_advanced(self) -> None:
        self._advanced_shown = not self._advanced_shown
        if self._advanced_shown:
            self._advanced_body.pack(fill='both', expand=True, pady=(8, 0))
        else:
            self._advanced_body.pack_forget()
        self._refresh_toggle_text()
        if self._advanced_shown and self._on_expand is not None:
            self._on_expand(self._bar)

    def _refresh_toggle_text(self) -> None:
        """'▸ Additional options · 2 changed' -- a change made behind the
        collapsed section stays visible from the main form."""
        n = sum(1 for d, info in self._fields.items()
                if info['group'] is not None and d not in _PICKER_DESTS and self._is_changed(d))
        arrow = '▾' if self._advanced_shown else '▸'
        suffix = '  ·  ' + _('{n} changed', n=n) if n else ''
        self._toggle_btn.configure(text=f"{arrow} {_('Additional options')}{suffix}")

    def _show_group(self, group_title: str) -> None:
        self._current_group = group_title
        for g, page in self._pages.items():
            selected = g == group_title
            if selected:
                page.pack(fill='both', expand=True)
            else:
                page.pack_forget()
            self._nav_labels[g].configure(
                background=_LINE_SOFT if selected else _PANEL,
                foreground=_TEXT if selected else _TEXT_DIM)

    # ── fields ─────────────────────────────────────────────────────────

    def _build_field(self, parent, row: int, field: Dict[str, Any],
                     group: Optional[str]) -> int:
        """Grid one field into *parent* from *row*: column 0 the changed
        dot, 1 the label, 2 the input, 3 a Browse button (paths), 4 the
        reset link; the description on the next row. Returns rows used."""
        dest, kind = field['dest'], field['kind']
        widgets: List[tk.Widget] = []
        dot = ttk.Label(parent, text='', style='Changed.TLabel', width=2)
        dot.grid(row=row, column=0, sticky='e', pady=(5, 0))
        widgets.append(dot)
        # Human label from the dest, never the raw "--flag" (and never the
        # *negating* flag for bool_false fields like dest 'auto' / flag
        # '--no-auto', which would read backwards next to a checked box).
        label = ttk.Label(parent, text=_field_label_ui(field))
        label.grid(row=row, column=1, sticky='w', padx=(0, 8), pady=(5, 0))
        widgets.append(label)
        # Tooltip keeps the CLI flag name and the full help discoverable.
        _tip = _(field['help']) if field.get('help') else ''
        _flag = field.get('flag')
        if _flag:
            _tip = f"{_flag}\n{_tip}".strip()
        if _tip:
            _Tooltip(label, _tip)

        summary_en = field.get('summary') or ''
        summary = _(summary_en) if summary_en else ''
        inline_hint = None
        if kind in ('bool_true', 'bool_false'):
            var = tk.BooleanVar(value=bool(field['default']))
            # The description sits beside a checkbox, not under it: a row
            # saved per checkbox keeps the form short.
            w = ttk.Frame(parent)
            ttk.Checkbutton(w, variable=var).pack(side='left')
            if summary:
                inline_hint = ttk.Label(w, text=summary, style='Hint.TLabel',
                                        wraplength=360, justify='left')
                inline_hint.pack(side='left', padx=(4, 0))
            w.grid(row=row, column=2, columnspan=2, sticky='w', pady=(5, 0))
        elif kind == 'select':
            var = tk.StringVar(value='' if field['default'] is None else str(field['default']))
            w = ttk.Combobox(parent, textvariable=var, state='readonly',
                             values=[''] + [str(c) for c in (field['choices'] or [])],
                             width=14)
            w.grid(row=row, column=2, sticky='we', pady=(5, 0))
        else:
            default = field['default']
            text = ', '.join(default) if isinstance(default, list) else \
                   ('' if default is None else str(default))
            var = tk.StringVar(value=text)
            # Small char width -- the field grows to fill column 2 (weighted)
            # when there's room, but this keeps the left pane able to shrink
            # so the horizontal sash can hold an even split with the preview.
            w = ttk.Entry(parent, textvariable=var, width=12)
            w.grid(row=row, column=2, sticky='we', pady=(5, 0))
            widget_hint = field.get('widget')
            if widget_hint:
                b = ttk.Button(parent, text=_('Browse…'),
                               command=lambda d=dest, v=var, h=widget_hint:
                                   self._browse(d, v, h))
                b.grid(row=row, column=3, padx=(4, 0), pady=(5, 0))
                widgets.append(b)
        widgets.append(w)

        reset = ttk.Label(parent, text='', style='Reset.TLabel', cursor='hand2')
        reset.grid(row=row, column=4, sticky='w', padx=(6, 0), pady=(5, 0))
        reset.bind('<Button-1>', lambda _e, d=dest: self._reset(d))
        widgets.append(reset)

        rows_used = 1
        hint = inline_hint
        if summary and hint is None:
            hint = ttk.Label(parent, text=summary, style='Hint.TLabel',
                             wraplength=360, justify='left')
            hint.grid(row=row + 1, column=1, columnspan=4, sticky='w')
            widgets.append(hint)
            rows_used += 1
        if hint is not None:
            if parent not in self._hints:
                self._hints[parent] = []
                parent.bind('<Configure>', self._rewrap, add='+')
            self._hints[parent].append((hint, hint is inline_hint))
        if dest == 'directory':
            # The frame count found in the folder replaces the description.
            var.trace_add('write', lambda *_: self._rescan_directory())
            if hint is not None:
                self.dir_count_var.trace_add(
                    'write', lambda *_, h=hint, t=summary:
                        h.configure(text=self.dir_count_var.get() or t))
        parent.grid_columnconfigure(2, weight=1)

        self.vars[dest] = var
        self._initial[dest] = var.get()
        self._labels[dest] = _field_label_ui(field)
        var.trace_add('write', lambda *_: self._notify_change())
        self._fields[dest] = {
            'widgets': widgets, 'group': group, 'dot': dot, 'reset': reset,
            'expert': _is_expert_field(field) or (group is not None and _is_expert_group(group)),
            # Not the full help: it mentions related features ("drizzle"
            # appears in the help of half the registration options), so a
            # search would list everything near a topic, not the setting.
            # English and the translation both, so either finds a setting.
            'haystack': ' '.join([_field_label(field), _field_label_ui(field),
                                  field.get('flag') or '', dest, summary_en, summary,
                                  ' '.join(str(c) for c in field.get('choices') or [])
                                  ]).replace('_', ' ').lower(),
        }
        if dest not in _NO_CHANGE_MARK:
            var.trace_add('write', lambda *_, d=dest: self._mark_changed(d))
        return rows_used

    def _rewrap(self, event) -> None:
        """Wrap a container's description lines to its current width (a
        line beside a checkbox starts further right)."""
        for hint, inline in self._hints.get(event.widget, []):
            hint.configure(wraplength=max(160, event.width - (260 if inline else 60)))

    def _is_changed(self, dest: str) -> bool:
        return self.vars[dest].get() != self._initial.get(dest)

    def _mark_changed(self, dest: str) -> None:
        info = self._fields[dest]
        changed = self._is_changed(dest)
        info['dot'].configure(text='●' if changed else '')
        info['reset'].configure(text=_('reset') if changed else '')
        if info['group'] is not None:
            self._refresh_toggle_text()

    def _reset(self, dest: str) -> None:
        self.vars[dest].set(self._initial[dest])

    def _notify_change(self) -> None:
        if self._on_change is not None and not self._change_pending:
            self._change_pending = True
            self.after_idle(self._fire_change)

    def _fire_change(self) -> None:
        self._change_pending = False
        self._on_change()

    def changed_settings(self) -> List[str]:
        """Labels of the settings changed from their defaults, not counting
        the folder/output or what the cards and run mode set."""
        return [self._labels[d] for d in self._fields
                if d not in _PICKER_DESTS and d not in _NO_CHANGE_MARK and self._is_changed(d)]

    def describe_run(self) -> List[Tuple[str, str]]:
        """(heading, text) rows describing what Start will do with the
        current form -- the App's "This run" panel."""
        v = {d: var.get() for d, var in self.vars.items()}
        directory = str(v.get('directory') or '').strip()
        rows = [(_('Frames'), self._data_summary or
                 (_('No light frames found in that folder.') if directory
                  else _('Choose the folder with your light frames.')))]

        name, inf_type, src = self.inferred
        auto = bool(v.get('auto', True))
        if v.get('comet_mode'):
            target = _('Comet: a second stack aligned on the nucleus')
        elif not auto:
            target = _('Auto advisor off: no tuning for the target')
        elif v.get('target_type'):
            target = _('{label} (your choice)', label=_target_label(v['target_type']))
        elif inf_type and inf_type != 'unknown':
            label = _target_label(inf_type)
            if name and _from_source(src):
                target = _('Auto-detect: starts from {label} ({name}, {source}), then checks the frames',
                           label=label, name=name, source=_from_source(src))
            elif name:
                target = _('Auto-detect: starts from {label} ({name}), then checks the frames',
                           label=label, name=name)
            else:
                target = _('Auto-detect: starts from {label}, then checks the frames', label=label)
        else:
            target = _('Auto-detect from the frames after the first pass')
        rows.append((_('Target'), target))

        mode = self.goal.selected_mode()
        rows.append((_('Mode'), next((_('{mode}. {description}', mode=_(m[1]), description=_(m[2]))
                                      for m in _RUN_MODES if m[0] == mode),
                                     _('Preset: {preset}', preset=v.get('preset')))))

        changed = self.changed_settings()
        if changed:
            shown = (_('{settings} and {n} more', settings=', '.join(changed[:4]), n=len(changed) - 4)
                     if len(changed) > 4 else ', '.join(changed))
            rows.append((_('Changed'), shown))
        else:
            rows.append((_('Changed'), _('Nothing; defaults, tuned by --auto') if auto
                         else _('Nothing; defaults')))

        output = str(v.get('output') or '').strip()
        if output:
            rows.append((_('Output'), output))
        elif directory:
            folder = os.path.abspath(directory.rstrip('/\\'))
            rows.append((_('Output'), os.path.join(os.path.dirname(folder),
                                                   os.path.basename(folder) + '_stacked.fits')
                         + '  ' + _('(next to the folder; never overwrites)')))
        return rows

    def _browse(self, dest: str, var: tk.Variable, widget_hint: str) -> None:
        if widget_hint == 'dir':
            path = filedialog.askdirectory()
        elif widget_hint == 'file-save':
            path = filedialog.asksaveasfilename(
                filetypes=[(_('FITS files'), '*.fits'), (_('All files'), '*.*')])
        else:
            path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _rescan_directory(self) -> None:
        directory = self.vars['directory'].get().strip()
        self._data_summary = ''
        self.inferred = (None, None, None)
        if not directory or not os.path.isdir(directory):
            self.dir_count_var.set('')
            self.goal.set_suggestion(None, None)
            return
        try:
            from src.frame_discovery import discover_frames
            found = discover_frames(directory)
            counts = {k: len(v) for k, v in found.items()}
            lights = list(found.get('light', []))
            if sum(counts.values()) == 0:
                subdirs = [os.path.join(directory, d) for d in os.listdir(directory)
                          if os.path.isdir(os.path.join(directory, d))]
                for d in subdirs:
                    sub_counts = discover_frames(d)
                    for k, v in sub_counts.items():
                        counts[k] = counts.get(k, 0) + len(v)
            parts = ', '.join(_frame_count(k, v) for k, v in counts.items() if v)
            self._data_summary = parts
            text = parts or _('no frames found')
            if lights:
                # The same local lookups the run starts with (info.json name,
                # FITS OBJECT, folder name), but never SIMBAD from here.
                from src.session_info import load_session_info
                from src.target_inference import infer_target_from_metadata
                si = load_session_info(directory)
                name, ttype, _conf, src = infer_target_from_metadata(
                    directory, lights, use_simbad=False,
                    session_name=si.object_name if si else None)
                if ttype == 'unknown':
                    ttype = None
                self.inferred = (name, ttype, src)
                if name:
                    text += f'  ·  {name}' + (f', {_from_source(src)}' if _from_source(src) else '')
            self.dir_count_var.set(text)
        except Exception:
            self.dir_count_var.set('')
        self.goal.set_suggestion(self.inferred[1], self.inferred[0])

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


# discover_frames() kinds, singular and plural.
_FRAME_KINDS = {'light': (N_('{n} light'), N_('{n} lights')),
                'dark': (N_('{n} dark'), N_('{n} darks')),
                'flat': (N_('{n} flat'), N_('{n} flats')),
                'bias': (N_('{n} bias'), N_('{n} bias frames')),
                'darkflat': (N_('{n} dark flat'), N_('{n} dark flats'))}


def _frame_count(kind: str, n: int) -> str:
    one, many = _FRAME_KINDS.get(kind, ('{n} ' + kind, '{n} ' + kind))
    return _(one if n == 1 else many, n=n)


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
        self.caption_var = tk.StringVar(value=_('Waiting for the first stack…'))

        toolbar = ttk.Frame(self)
        toolbar.pack(fill='x', pady=(0, 4))
        ttk.Button(toolbar, text=_('Fit'), command=self.fit).pack(side='left')
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

    def clear(self) -> None:
        """Drop the displayed image(s) -- called when a new run starts so the
        viewer doesn't keep showing the previous run's stack."""
        self._img_a = self._img_b = self._photo = None
        self._nat_w = self._nat_h = 0
        self.current_slug = self.compare_slug = ''
        self.compare_on = False
        self.scale = self.fit_scale = 1.0
        self.tx = self.ty = 0.0
        self.caption_var.set(_('Waiting for the first stack…'))
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
            c.create_text(cw / 2, ch / 2,
                          text='◆ ORIGINSTACK\n' + _('Waiting for the first stack preview…'),
                          fill=_TEXT_FAINT, justify='center', font=(_SANS, 11))
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
                 font=(_MONO, 8)).grid(row=1, column=col)

    def clear(self) -> None:
        """Drop every thumbnail -- called when a new run starts."""
        for child in self.inner.winfo_children():
            child.destroy()
        self._shown_ids.clear()
        self._photos.clear()
        self.canvas.configure(scrollregion=(0, 0, 0, 0))


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
        self._named_cache: List[Dict[str, Any]] = []

        root.title('OriginStack')
        root.geometry('1400x900')
        root.minsize(900, 600)
        root_dir = Path(__file__).resolve().parent.parent
        icon = root_dir / 'packaging' / 'icon.ico'
        if sys.platform == 'win32' and icon.exists():
            try:
                root.iconbitmap(str(icon))
            except tk.TclError:
                pass
        elif (root_dir / 'assets' / 'icon.png').exists():     # Linux/macOS: .ico is not understood
            try:
                self._icon_img = tk.PhotoImage(file=str(root_dir / 'assets' / 'icon.png'))
                root.iconphoto(True, self._icon_img)
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
        # (requested) size instead, which lets the Setup form's wide fields
        # swallow most of the window and squeeze the preview. Pin the sash to
        # an even split once the window has a real mapped width; doing it in
        # __init__ (before the WM sizes the window) reads a stale width and
        # the split never lands. Runs once.
        self._split_pinned = False

        def _pin_split(_evt=None):
            if self._split_pinned:
                return
            w = paned.winfo_width()
            if w > 1:
                paned.sashpos(0, w // 2)
                self._split_pinned = True
        paned.bind('<Configure>', _pin_split, add='+')
        root.after(80, _pin_split)

        # The vertical split on the left: the run controls, pipeline bar and
        # a usable log (~300 px) below, the Setup form gets the rest (at
        # least half).
        self._vsplit_pinned = False

        def _pin_vsplit(_evt=None):
            if self._vsplit_pinned:
                return
            h = self._vpaned.winfo_height()
            if h > 1:
                self._vpaned.sashpos(0, max(h // 2, h - 300))
                self._vsplit_pinned = True
        self._vpaned.bind('<Configure>', _pin_vsplit, add='+')
        root.after(80, _pin_vsplit)

        root.protocol('WM_DELETE_WINDOW', self._on_closing)
        root.after(self.POLL_MS, self._poll)
        self._on_form_change()

    def _build_header(self, root: tk.Tk) -> None:
        from src.utils import read_version
        header = tk.Frame(root, background=_PANEL, height=44)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)
        inner = tk.Frame(header, background=_PANEL)
        inner.pack(fill='both', expand=True, padx=16)
        tk.Label(inner, text='◆', background=_PANEL, foreground=_ACCENT,
                font=(_SANS, 13)).pack(side='left', pady=8)
        tk.Label(inner, text='ORIGINSTACK', background=_PANEL, foreground=_TEXT,
                font=(_SANS, 12, 'bold')).pack(side='left', padx=(8, 0), pady=8)
        tk.Label(inner, text=f'v{read_version()}', background=_PANEL, foreground=_TEXT_FAINT,
                font=_FONT_MONO).pack(side='left', padx=(10, 0), pady=8)
        self.update_label = tk.Label(inner, text='', background=_PANEL, foreground=_ACCENT,
                                     font=_FONT, cursor='hand2')
        self.update_label.pack(side='left', padx=(14, 0), pady=8)
        self.update_label.bind('<Button-1>', lambda _e: self._open_update_url())
        self._update_url = ''
        self._start_update_check()
        self.header_status_var = tk.StringVar(value=_('Idle'))
        tk.Label(inner, textvariable=self.header_status_var, background=_PANEL,
                foreground=_TEXT_DIM, font=_FONT_MONO).pack(side='right', pady=8)
        self._build_language_menu(inner)
        tk.Frame(root, background=_LINE, height=1).pack(fill='x', side='top')

    def _build_language_menu(self, parent: tk.Frame) -> None:
        """Header drop-down: Automatic (the system language) or a fixed
        language. The choice is saved and applies from the next start --
        rebuilding every widget mid-session would also drop a run's state."""
        from src.i18n import LANGUAGES, save_choice, saved_choice, system_language
        auto_label = _('Automatic ({language})', language=LANGUAGES[system_language()])
        options = [('auto', auto_label)] + list(LANGUAGES.items())
        current = saved_choice()
        var = tk.StringVar(value=next(label for code, label in options if code == current))
        combo = ttk.Combobox(parent, textvariable=var, values=[label for _c, label in options],
                             state='readonly', width=22)
        combo.pack(side='right', padx=(0, 12), pady=8)
        note = tk.Label(parent, text='', background=_PANEL, foreground=_ACCENT3, font=_FONT)
        note.pack(side='right', padx=(0, 8), pady=8)

        def _picked(_event=None):
            code = next(code for code, label in options if label == var.get())
            save_choice(code)
            note.configure(text=_('Restart OriginStack to change the language.')
                           if code != current else '')
        combo.bind('<<ComboboxSelected>>', _picked)

    def _start_update_check(self) -> None:
        """Background self-update check, once per launch. Never blocks the window
        opening: the network call runs on a daemon thread, and the result (if any)
        is marshalled back to the header label via root.after -- tkinter widgets
        may only be touched from the main thread."""
        from src.utils import read_version, should_check_for_update
        if not should_check_for_update():
            return

        def _bg():
            from src.net_query import check_for_update
            info = check_for_update(read_version())
            if info:
                self.root.after(0, lambda: self._show_update(info))
        threading.Thread(target=_bg, daemon=True).start()

    def _show_update(self, info: Dict[str, str]) -> None:
        self._update_url = info['url']
        self.update_label.configure(text=_('Update available: v{version}', version=info['version']))

    def _open_update_url(self) -> None:
        if self._update_url:
            webbrowser.open(self._update_url)

    # ── left column: setup, run controls, progress, log ───────────────

    def _build_left(self, parent: ttk.Frame) -> None:
        # The Setup form (with descriptions, and the additional options
        # open) is taller than the window, so it scrolls in its own pane
        # above the run controls and log, which must never be pushed off
        # screen. The sash between them is pinned once the window is mapped
        # (see _pin_vsplit), the same way the left/right split is.
        vpaned = ttk.PanedWindow(parent, orient='vertical')
        vpaned.pack(fill='both', expand=True)
        setup = ScrollableFrame(vpaned)
        self.form = SetupForm(setup.inner, on_expand=setup.scroll_to,
                              on_change=self._on_form_change)
        self.form.pack(fill='both', expand=True, padx=(0, 6))
        from src.desktop_control import load_app_setting
        _tv = self.form.vars.get('temp_dir')
        if _tv is not None and not _tv.get():
            _tv.set(load_app_setting('temp_dir', '') or '')
        parent = ttk.Frame(vpaned)
        vpaned.add(setup, weight=1)
        vpaned.add(parent, weight=1)
        self._vpaned = vpaned

        run_row = ttk.Frame(parent)
        run_row.pack(fill='x', pady=10)
        self.start_btn = ttk.Button(run_row, text=_('Start'), style='Accent.TButton',
                                    command=self._on_start)
        self.start_btn.pack(side='left')
        self.cancel_btn = ttk.Button(run_row, text=_('Cancel'), command=self._on_cancel)
        self.cancel_btn.pack(side='left', padx=(6, 0))
        self.cancel_btn.state(['disabled'])
        self.status_var = tk.StringVar(value=_('Idle'))
        ttk.Label(run_row, textvariable=self.status_var, style='Dim.TLabel').pack(
            side='left', padx=10)
        self.open_folder_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(run_row, text=_('Open folder when done'),
                        variable=self.open_folder_var).pack(side='left', padx=(10, 0))
        ttk.Button(run_row, text=_('Open Folder'), command=self._open_output_folder).pack(
            side='left', padx=(6, 0))

        phase_frame = ttk.LabelFrame(parent, text=_('PIPELINE'))
        phase_frame.pack(fill='x', pady=6)
        self.phase_labels = []
        titles = [f'{i} · {name}' for i, name in
                  enumerate((_('Quality'), _('Registration'), _('Stacking'), _('Post-process')),
                            start=1)]
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

        # RECENT FRAMES moved to the right column; the whole left column
        # below the pipeline bar is the log now.
        log_frame = ttk.LabelFrame(parent, text=_('LOG'))
        log_frame.pack(fill='both', expand=True, pady=6)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=20, bg=_LOG_BG, fg=_LOG_FG, insertbackground=_LOG_FG,
            font=(_MONO, 9), state='disabled', wrap='word')
        self.log_text.pack(fill='both', expand=True)

    # ── right column: preview, frame strip, recent frames, summary ────

    def _build_right(self, parent: ttk.Frame) -> None:
        preview_frame = ttk.LabelFrame(parent, text=_('PREVIEW'))
        preview_frame.pack(fill='both', expand=True, pady=(0, 6))

        vtools = ttk.Frame(preview_frame)
        vtools.pack(fill='x', pady=4)
        self.view_var = tk.StringVar()
        self.view_combo = ttk.Combobox(vtools, textvariable=self.view_var,
                                       state='readonly', width=20)
        self.view_combo.pack(side='left')
        self.view_combo.bind('<<ComboboxSelected>>', self._on_view_selected)
        self.compare_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(vtools, text=_('Compare'), variable=self.compare_var,
                        command=self._on_compare_toggled).pack(side='left', padx=8)
        self.compare_combo_var = tk.StringVar()
        self.compare_combo = ttk.Combobox(vtools, textvariable=self.compare_combo_var,
                                          state='readonly', width=20)
        self.compare_combo.pack(side='left', fill='x', expand=True)
        self.compare_combo.bind('<<ComboboxSelected>>', self._on_compare_selected)

        self.preview = PreviewCanvas(preview_frame)
        self.preview.pack(fill='both', expand=True, padx=4)

        self.wipe_scale = ttk.Scale(preview_frame, from_=0, to=1, orient='horizontal',
                                    command=self._on_wipe_moved)
        self.wipe_scale.set(0.5)

        self._known_slugs: List[str] = []

        strip_frame = ttk.LabelFrame(parent, text=_('FRAMES'))
        strip_frame.pack(fill='x', pady=6)
        self.frame_strip = FrameStrip(strip_frame, on_click=self._on_thumb_click)
        self.frame_strip.pack(fill='x')

        frames_frame = ttk.LabelFrame(parent, text=_('RECENT FRAMES'))
        frames_frame.pack(fill='x', pady=6)
        cols = ('frame', 'score', 'snr', 'stars', 'fwhm')
        self.frames_tree = ttk.Treeview(frames_frame, columns=cols, show='headings', height=5)
        headings = {'frame': _('Frame'), 'score': _('Score'), 'snr': 'SNR',
                    'stars': _('Stars'), 'fwhm': 'FWHM'}
        for c, w in zip(cols, (170, 60, 60, 50, 60)):
            self.frames_tree.heading(c, text=headings[c])
            self.frames_tree.column(c, width=w, anchor='e' if c != 'frame' else 'w')
        self.frames_tree.tag_configure('bad', foreground=_BAD)
        self.frames_tree.pack(fill='x')

        # Before a run (and after any change once one has finished) this is
        # "This run": what Start will do with the form. After a run it is the
        # run's summary.
        self.summary_frame = ttk.LabelFrame(parent, text=_('THIS RUN'))
        # Packed ahead of the preview so it gets its rows first; packed last it
        # was squeezed to two lines below the (empty, before a run) frame lists.
        self.summary_frame.pack(fill='x', pady=6, side='bottom', before=preview_frame)
        self.summary_var = tk.StringVar(value='')
        self.summary_label = ttk.Label(self.summary_frame, textvariable=self.summary_var,
                                       justify='left')
        self.plan_frame = ttk.Frame(self.summary_frame)
        self.plan_frame.pack(fill='x', padx=4, pady=4)
        self.plan_frame.grid_columnconfigure(1, weight=1)
        # Wrap the plan to the panel's width: at a fixed 520 px a longer
        # (translated) line wrapped needlessly, and every extra line was
        # taken from the RECENT FRAMES table above it.
        self.plan_frame.bind('<Configure>', self._rewrap_plan, add='+')
        # Bottom-up: this summary, the frames table, the thumbnail strip; the
        # preview (packed last) takes what is left. Packed top-down, the frames
        # table was packed last and lost its rows whenever the summary grew.
        frames_frame.pack_configure(side='bottom', after=self.summary_frame)
        strip_frame.pack_configure(side='bottom', after=frames_frame)
        self._show_plan = True

    def _plan_wrap(self) -> int:
        return max(300, self.plan_frame.winfo_width() - 120)

    def _rewrap_plan(self, _event=None) -> None:
        for child in self.plan_frame.grid_slaves(column=1):
            child.configure(wraplength=self._plan_wrap())

    # ── run control ─────────────────────────────────────────────────────

    def _on_start(self) -> None:
        form = self.form.read_form()
        # remembered for the next launch: a big session's temp files need a roomy drive
        from src.desktop_control import save_app_setting
        save_app_setting('temp_dir', str(form.get('temp_dir') or '').strip())
        result = self.rm.start(form)
        if not result.get('ok'):
            self.status_var.set(_('Error: {error}', error=result.get('error')))
            return
        self.start_btn.state(['disabled'])
        self.cancel_btn.state(['!disabled'])
        self.status_var.set(_('Running…'))
        self._shown_log_lines = 0
        self.frames_tree.delete(*self.frames_tree.get_children())
        self.summary_var.set('')
        self._show_plan = False
        self.summary_frame.configure(text=_('COMPLETE'))
        self.plan_frame.pack_forget()
        self.summary_label.pack(anchor='w', padx=4, pady=4)

    def _on_form_change(self) -> None:
        """The form changed: show what Start would now do, and the example
        result for the chosen target until a run's own previews arrive."""
        if self.rm.is_running():
            return
        self._show_plan = True
        self.summary_frame.configure(text=_('THIS RUN'))
        self.summary_label.pack_forget()
        self.plan_frame.pack(fill='x', padx=4, pady=4)
        for child in self.plan_frame.winfo_children():
            child.destroy()
        for i, (head, text) in enumerate(self.form.describe_run()):
            ttk.Label(self.plan_frame, text=head, style='Dim.TLabel').grid(
                row=i, column=0, sticky='nw', padx=(0, 12), pady=1)
            ttk.Label(self.plan_frame, text=text, wraplength=self._plan_wrap(), justify='left').grid(
                row=i, column=1, sticky='w', pady=1)
        if self.preview.current_slug in ('', 'example'):
            path, caption = self.form.goal.example()
            if path is not None:
                try:
                    self.preview.load_slot(path.read_bytes(), 'example', caption)
                except Exception:
                    self.preview.clear()
            elif self.preview.current_slug == 'example':
                self.preview.clear()

    def _on_cancel(self) -> None:
        # Cooperative, not instant -- takes effect at the next checkpoint
        # (RunManager/frame_processor._check_cancel's docstrings). Disabling
        # the button immediately is the honest signal: pressing it again
        # wouldn't make the pipeline notice any sooner.
        self.rm.cancel()
        self.cancel_btn.state(['disabled'])
        self.status_var.set(_('Cancelling…'))

    def _on_closing(self) -> None:
        if self.rm.is_running():
            from src.native_dialog import ask_yes_no
            if not ask_yes_no('OriginStack',
                              _('A stacking run is still in progress. Quit anyway?'),
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
            self.preview.load_slot(data, f'frame-{fid}', _('Frame #{n}', n=fid))

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

        # New run just started -- drop the previous run's preview image,
        # milestone slots and thumbnail ring (UIEvents.run_started() already
        # cleared its side; the widgets need clearing too).
        if snap['run_status'] == 'running' and self._last_run_status != 'running':
            self.preview.clear()
            self.frame_strip.clear()
            self.view_var.set('')
            self.compare_var.set(False)
            self.compare_combo_var.set('')
            self.wipe_scale.pack_forget()
            self._named_cache = []
            self.view_combo['values'] = []
            self.compare_combo['values'] = []

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

        # summary (unless the form changed since: then it shows "This run")
        if snap['summary'] and not self._show_plan:
            lines = [f"{k}: {v}" for k, v in snap['summary'].items()]
            self.summary_var.set('\n'.join(lines))

        self._refresh_run_button(snap)

    def _open_output_folder(self) -> None:
        """Show the stack the last run wrote (selected in Explorer), else the folder the
        Output field names. A blank Output used to open nothing: the stack is named
        by the run, beside the light-frames folder."""
        path = self.rm.last_output
        if not path:
            var = self.form.vars.get('output')
            typed = var.get().strip() if var is not None else ''
            path = typed if typed else None
        if not path:
            return
        path = os.path.abspath(path)
        target_file = path if os.path.isfile(path) else None
        folder = path if os.path.isdir(path) else (os.path.dirname(path) or '.')
        if not os.path.isdir(folder):
            return
        try:
            if sys.platform == 'win32':
                if target_file:
                    import subprocess
                    subprocess.Popen(['explorer', '/select,', target_file])
                else:
                    os.startfile(folder)
            elif sys.platform == 'darwin':
                import subprocess
                subprocess.Popen(['open', '-R', target_file] if target_file else ['open', folder])
            else:
                import subprocess
                subprocess.Popen(['xdg-open', folder])
        except Exception:
            pass

    def _refresh_run_button(self, snap: Optional[Dict[str, Any]] = None) -> None:
        running = self.rm.is_running()
        if running:
            self.start_btn.state(['disabled'])
            self.header_status_var.set(_('Cancelling…') if self.rm.is_cancelling() else _('Running…'))
        else:
            self.start_btn.state(['!disabled'])
            self.cancel_btn.state(['disabled'])
            if snap is not None and snap['run_status'] in ('ok', 'error', 'cancelled'):
                done_ok = snap['run_status'] == 'ok'
                if snap['run_status'] == 'cancelled':
                    self.status_var.set(_('Cancelled'))
                    self.header_status_var.set(_('Cancelled'))
                else:
                    self.status_var.set(_('Done') if done_ok
                                        else _('Failed: {error}', error=snap['run_error']))
                    self.header_status_var.set(_('Complete') if done_ok else _('Failed'))
                # Edge-triggered (not every poll tick) so it only pops once
                # per completed run, not repeatedly while the status holds.
                if done_ok and self._last_run_status != 'ok' and self.open_folder_var.get():
                    self._open_output_folder()
            elif snap is None or snap['run_status'] == 'idle':
                self.header_status_var.set(_('Idle'))
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

    from src.i18n import resolve_language, set_language
    _set_fonts(set_language(resolve_language()))
    root = tk.Tk()
    try:
        App(root)
    except Exception as e:
        return _fatal("OriginStack", _('Failed to open the app window: {error}', error=e))
    try:
        root.mainloop()
    except Exception as e:
        return _fatal("OriginStack", _('Unexpected error: {error}', error=e))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
