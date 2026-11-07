"""The main window: header, Setup form, run controls, progress and log on the left; the
preview, post-processing controls, recent frames and run summary on the right.

Same job as the tkinter ``App``: the pipeline runs on ``RunManager``'s thread and publishes
into ``UIEvents``; a timer polls ``UIEvents.snapshot()`` and the widgets are only touched here,
on the GUI thread.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QGuiApplication, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSystemTrayIcon,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.desktop_common import with_export_checkboxes
from src.desktop_startup import _fatal, _shutdown_and_exit
from src.i18n import _
from src.qt_app import theme, window_state
from src.qt_app.form_model import FormModel
from src.qt_app.frame_graph import FramesPanel, names_to_setting, setting_to_names
from src.qt_app.help_dialog import ShortcutsDialog
from src.qt_app.post_panel import PostProcessPanel
from src.qt_app.setup_form import SetupForm
from src.qt_app.stretch_panel import StretchEditor
from src.qt_app.taskbar import ERROR, PAUSED, TaskbarProgress
from src.qt_app.viewer import PreviewPane
from src.qt_app.widgets import (
    ClickableLabel,
    HSlider,
    NoWheelComboBox,
    label,
    repolish,
    set_role,
)

POLL_MS = 150
_FRAME_COLS = ('frame', 'score', 'snr', 'stars', 'fwhm')


class _Bridge(QObject):
    """Results of background threads, delivered on the GUI thread."""
    update_found = Signal(object)
    update_progress = Signal(int, int)
    update_done = Signal(str)
    update_error = Signal(str)
    detail_ready = Signal(int, object, object)     # generation, rgb8 array, real size
    detail_failed = Signal(int, str)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        from src.desktop_control import get_run_manager
        from src.ui_events import get_ui_events
        self.ui = get_ui_events()
        self.ui.attach()
        self.rm = get_run_manager()
        self._last_version = -1
        self._shown_log_lines = 0
        self._last_run_status = 'idle'
        self.taskbar = TaskbarProgress()
        self._last_snap: Optional[Dict[str, Any]] = None
        self._named_cache: List[Dict[str, Any]] = []
        self._pinned_frame: Optional[str] = None
        self._frame_rows_shown: List[tuple] = []
        self._frame_sort: Optional[Tuple[str, bool]] = None
        self._show_plan = True
        self._left_running = False
        self._diagnostics_open = False
        self._update_url = ''
        self._update_info: Optional[Dict[str, Any]] = None
        self._update_busy = False
        self._bridge = _Bridge(self)
        self._bridge.update_found.connect(self._show_update)
        self._bridge.update_progress.connect(self._on_update_progress)
        self._bridge.update_done.connect(self._on_update_done)
        self._bridge.update_error.connect(self._on_update_error)
        self._bridge.detail_ready.connect(self._on_detail_ready)
        self._bridge.detail_failed.connect(self._on_detail_failed)
        # Full-size copy of the final picture, loaded when zoomed past the pixels shown.
        self._graph_rows: List[dict] = []
        self._detail_gen = 0            # changes whenever another picture is shown
        self._detail_done = -1          # the generation whose full-size copy is on screen
        self._detail_loading = False
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.timeout.connect(self._request_detail)

        self.setWindowTitle('OriginStack')
        self.setAcceptDrops(True)
        root_dir = Path(__file__).resolve().parent.parent.parent
        icon = root_dir / 'packaging' / 'icon.ico'
        if not icon.exists():
            icon = root_dir / 'assets' / 'icon.png'
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1400, int(screen.width() * 0.96)), min(900, int(screen.height() * 0.92)))
        self.setMinimumSize(900, 600)

        self.model = FormModel()
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())
        split = self.main_split = QSplitter(Qt.Orientation.Horizontal)
        split.setContentsMargins(10, 4, 10, 10)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([700, 700])
        outer.addWidget(split, 1)
        self.setCentralWidget(central)

        self.model.value_changed.connect(lambda _d: self._form_changed())
        self.model.scanned.connect(self._form_changed)
        self.model.restore_folders()
        from src.desktop_control import load_app_setting
        if not self.model.get('temp_dir'):
            self.model.set('temp_dir', load_app_setting('temp_dir', '') or '')
        self._picture_max = False
        self._tools_collapsed = False
        self._shortcuts()
        self._build_menus()
        self._setup_tray()
        self._restore_state()
        self._start_update_check()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(POLL_MS)
        self._form_changed()

    # ── header ──────────────────────────────────────────────────────────

    def _build_header(self) -> QFrame:
        from src.utils import read_version
        header = QFrame()
        header.setObjectName('header')
        header.setFixedHeight(44)
        row = QHBoxLayout(header)
        row.setContentsMargins(16, 0, 16, 0)
        mark = QLabel('◆')
        mark.setStyleSheet(f'color: {theme.ACCENT}; font-size: 13pt;')
        row.addWidget(mark)
        name = QLabel('ORIGINSTACK')
        name.setStyleSheet('font-weight: bold; font-size: 12pt;')
        row.addWidget(name)
        ver = set_role(QLabel(f'v{read_version()}'), 'mono')
        ver.setStyleSheet(f'color: {theme.TEXT_FAINT};')
        row.addWidget(ver)
        self.update_label = set_role(ClickableLabel(''), 'update')
        self.update_label.clicked.connect(self._open_update_url)
        row.addWidget(self.update_label)
        row.addStretch(1)

        support = set_role(ClickableLabel(_('Support OriginStack')), 'link')
        support.setToolTip(_('OriginStack is free. If it saves you time, a tip on Ko-fi helps '
                             'keep it going. Opens in your browser.'))
        support.clicked.connect(self._open_support)
        row.addWidget(support)
        row.addSpacing(12)
        self.report_label = set_role(ClickableLabel(_('Report a problem')), 'link')
        self.report_label.setToolTip(_(
            'Opens a new GitHub issue in your browser, or copies the same report '
            'for an email or a forum post, with the version, system and the end of '
            'the log filled in. Nothing is sent until you send it yourself.'))
        self.report_label.clicked.connect(self._report_menu)
        row.addWidget(self.report_label)
        row.addSpacing(12)
        self._build_language_menu(row)
        row.addSpacing(12)
        self.header_status = set_role(QLabel(_('Idle')), 'mono')
        self.header_status.setStyleSheet(f'color: {theme.TEXT_DIM};')
        row.addWidget(self.header_status)
        return header

    def _build_language_menu(self, row: QHBoxLayout) -> None:
        """Automatic (the system language) or a fixed language. The choice is saved and applies
        from the next start: rebuilding every widget mid-session would drop a run's state."""
        from src.i18n import LANGUAGES, save_choice, saved_choice, system_language
        options = [('auto', _('Automatic ({language})', language=LANGUAGES[system_language()]))]
        options += list(LANGUAGES.items())
        current = saved_choice()
        self.lang_note = QLabel('')
        self.lang_note.setStyleSheet(f'color: {theme.ACCENT3};')
        row.addWidget(self.lang_note)
        combo = NoWheelComboBox()
        for code, text in options:
            combo.addItem(text, code)
        combo.setCurrentIndex(max(0, next((i for i, (c, _t) in enumerate(options) if c == current), 0)))

        def picked(_i: int) -> None:
            code = combo.currentData()
            save_choice(code)
            self.lang_note.setText(_('Restart OriginStack to change the language.')
                                   if code != current else '')
        combo.activated.connect(picked)
        row.addWidget(combo)

    # ── left column ─────────────────────────────────────────────────────

    def _build_left(self) -> QWidget:
        self.vsplit = QSplitter(Qt.Orientation.Vertical)
        self.form = SetupForm(self.model)
        self.setup_scroll = QScrollArea()
        self.setup_scroll.setWidgetResizable(True)
        self.setup_scroll.setWidget(self.form)
        self.setup_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.form.expanded.connect(lambda w: self.setup_scroll.ensureWidgetVisible(w, 0, 20))
        self.vsplit.addWidget(self.setup_scroll)

        run = QWidget()
        lay = QVBoxLayout(run)
        lay.setContentsMargins(0, 6, 0, 0)

        row = QHBoxLayout()
        self.start_btn = QPushButton(_('Start'))
        set_role(self.start_btn, 'accent')
        self.start_btn.clicked.connect(self._on_start)
        self.cancel_btn = QPushButton(_('Cancel'))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.status_label = set_role(QLabel(_('Idle')), 'dim')
        self.open_when_done = QCheckBox(_('Open folder when done'))
        open_btn = QPushButton(_('Open Folder'))
        open_btn.clicked.connect(self._open_output_folder)
        for w in (self.start_btn, self.cancel_btn, self.status_label, self.open_when_done, open_btn):
            row.addWidget(w)
        row.addStretch(1)
        lay.addLayout(row)

        from src.desktop_control import load_app_setting
        save_row = QHBoxLayout()
        save_row.addWidget(set_role(QLabel(_('Also save:')), 'dim'))
        self.export_boxes: Dict[str, QCheckBox] = {}
        for fmt, text, tip in (
                ('png', _('Lossless PNG'), _('The final image without JPEG compression.')),
                ('png16', _('16-bit PNG for editing'),
                 _('The final image in 16 bits per channel: further curves or levels '
                   'adjustments in another editor do not band.'))):
            cb = QCheckBox(text)
            cb.setToolTip(tip)
            cb.setChecked(bool(load_app_setting(f'export_{fmt}', False)))
            save_row.addWidget(cb)
            self.export_boxes[fmt] = cb
        self.review_box = QCheckBox(_('Review frames before stacking'))
        self.review_box.setToolTip(_('Pause after the quality analysis so you can look at every frame on a '
                                     'graph, leave out the bad ones, and then continue.'))
        self.review_box.setChecked(bool(load_app_setting('review_frames', False)))
        save_row.addWidget(self.review_box)
        save_row.addStretch(1)
        lay.addLayout(save_row)

        self.readiness = set_role(QLabel(_('Choose a light-frame folder to begin.')), 'dim')
        self.readiness.setWordWrap(True)
        lay.addWidget(self.readiness)

        self.preflight_group = QGroupBox(_('PREFLIGHT'))
        pg = QGridLayout(self.preflight_group)
        self.preflight_ok = set_role(QLabel(''), 'dim')
        pg.addWidget(self.preflight_ok, 0, 0, 1, 2)
        self.preflight_labels: Dict[str, QLabel] = {}
        self._preflight_titles = {
            'frames': _('Light frames'), 'calibration': _('Calibration frames'),
            'target': _('Target detected'), 'space': _('Temporary disk space'),
            'output': _('Output location')}
        for i, (key, text) in enumerate(self._preflight_titles.items()):
            lbl = set_role(QLabel(f'○ {text}'), 'dim')
            pg.addWidget(lbl, 1 + i // 2, i % 2)
            self.preflight_labels[key] = lbl
        lay.addWidget(self.preflight_group)

        phase_group = QGroupBox(_('PIPELINE'))
        pl = QGridLayout(phase_group)
        self.phase_labels: List[QLabel] = []
        for i, name in enumerate((_('Quality'), _('Registration'), _('Stacking'),
                                  _('Post-process')), start=1):
            lbl = set_role(QLabel(f'{i} · {name}'), 'phase', state='')
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pl.addWidget(lbl, 0, i - 1)
            self.phase_labels.append(lbl)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        pl.addWidget(self.progress, 1, 0, 1, 4)
        self.progress_text = QLabel('')
        self.eta_text = set_role(QLabel(''), 'dim')
        pl.addWidget(self.progress_text, 2, 0, 1, 3)
        pl.addWidget(self.eta_text, 2, 3, Qt.AlignmentFlag.AlignRight)
        lay.addWidget(phase_group)
        self.phase_group = phase_group

        self.diagnostics_group = QGroupBox(_('DIAGNOSTICS'))
        dl = QVBoxLayout(self.diagnostics_group)
        self.diagnostics_text = label(_('Warnings and rejected frames will appear here after a run.'),
                                      wrap=True)
        dl.addWidget(self.diagnostics_text)
        lay.addWidget(self.diagnostics_group)
        self.diagnostics_group.hide()

        log_group = QGroupBox(_('LOG'))
        ll = QVBoxLayout(log_group)
        self.log = QPlainTextEdit()
        self.log.setObjectName('log')
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        ll.addWidget(self.log)
        lay.addWidget(log_group, 1)

        diag_btn = QPushButton(_('Show diagnostics'))
        diag_btn.setProperty('role', 'flat')
        diag_btn.clicked.connect(self._toggle_diagnostics)
        lay.addWidget(diag_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.vsplit.addWidget(run)
        self.vsplit.setStretchFactor(0, 1)
        self.vsplit.setStretchFactor(1, 1)
        self.vsplit.setSizes([450, 450])
        return self.vsplit

    # ── right column ────────────────────────────────────────────────────

    def _build_right(self) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(6, 0, 0, 0)

        group = QGroupBox(_('PREVIEW'))
        gl = QVBoxLayout(group)
        vt = QHBoxLayout()
        vt.addWidget(QLabel(_('Milestone')))
        self.view_combo = NoWheelComboBox()
        self.view_combo.setSizeAdjustPolicy(NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.view_combo.activated.connect(lambda _i: self._on_view_selected())
        vt.addWidget(self.view_combo, 1)
        full = QPushButton(_('Full screen'))
        full.clicked.connect(self._preview_fullscreen)
        vt.addWidget(full)
        gl.addLayout(vt)
        ct = QHBoxLayout()
        self.compare_box = QCheckBox(_('Compare'))
        self.compare_box.toggled.connect(self._on_compare_toggled)
        ct.addWidget(self.compare_box)
        self.compare_combo = NoWheelComboBox()
        self.compare_combo.setEnabled(False)
        self.compare_combo.activated.connect(lambda _i: self._on_compare_selected())
        ct.addWidget(self.compare_combo, 1)
        self.stars_box = QCheckBox(_('Without stars'))
        self.stars_box.setToolTip(_('Show the picture without its stars (also saves a _starless JPEG)'))
        self.stars_box.setEnabled(False)
        ct.addWidget(self.stars_box)
        gl.addLayout(ct)

        self.preview = PreviewPane()
        gl.addWidget(self.preview, 1)
        self.wipe = HSlider()
        self.wipe.moved.connect(self.preview.set_wipe)
        self.wipe.hide()
        gl.addWidget(self.wipe)

        # Tools under the picture: the stretch editor, the Phase 4 steps and the frames table.
        # Tabs appear as they become useful (frames during a run, the other two once it is done);
        # with none to show the picture takes the whole column.
        self.tools = QTabWidget()
        self.editor = StretchEditor(self.ui, self.rm, self.form.field_choices('stretch'),
                                    self.form.field_choices('stretch_color'))
        self.editor.live_image.connect(self._show_live)
        self.editor.image_ready.connect(self._show_final)
        self.editor.leave_compare.connect(lambda: self.compare_box.setChecked(False))
        self.editor.availability_changed.connect(self._on_editor_availability)
        self.post = PostProcessPanel(self.ui, self.rm)
        self.post.image_ready.connect(self._show_final)
        self.post.image_ready.connect(lambda *_a: self.editor.refresh())
        self.post.leave_compare.connect(lambda: self.compare_box.setChecked(False))
        self.stars_box.toggled.connect(self.post.starless_box.setChecked)
        self.post.starless_box.toggled.connect(self.stars_box.setChecked)
        self.preview.hold_original.connect(self.editor.peek)
        self.preview.detail_wanted.connect(self._on_detail_wanted)
        self.preview.linear_sampler = self._sample_linear
        self.preview.wipe_dragged.connect(self.wipe.set_fraction)
        self.preview.set_empty_hint(_('Drop a folder of light frames on the window, or press Ctrl+O.'))
        self.frames_group = QWidget()
        fl = QVBoxLayout(self.frames_group)
        fl.setContentsMargins(0, 0, 0, 0)
        self.frames_panel = FramesPanel()
        self.frames_panel.frame_clicked.connect(self._on_graph_frame)
        self.frames_panel.excluded_changed.connect(
            lambda names: self.model.set('exclude_frames', names_to_setting(names)))
        self.frames_panel.continue_clicked.connect(self._on_review_continue)
        self.frames_panel.restack_clicked.connect(self._on_start)
        fl.addWidget(self.frames_panel)
        self.frames_table = QTableWidget(0, len(_FRAME_COLS))
        self._frame_headings = {'frame': _('Frame'), 'score': _('Score'), 'snr': 'SNR',
                                'stars': _('Stars'), 'fwhm': 'FWHM'}
        self.frames_table.setHorizontalHeaderLabels([self._frame_headings[c] for c in _FRAME_COLS])
        self.frames_table.verticalHeader().hide()
        self.frames_table.verticalHeader().setDefaultSectionSize(24)
        self.frames_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.frames_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.frames_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.frames_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, len(_FRAME_COLS)):
            hh.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        hh.sectionClicked.connect(lambda i: self._sort_frames(_FRAME_COLS[i]))
        self.frames_table.itemSelectionChanged.connect(self._on_frame_selected)
        self.frames_table.setMinimumHeight(110)
        fl.addWidget(self.frames_table, 1)
        self._reviewing = False
        self.model.value_changed.connect(self._on_exclusions_edited)
        self.frames_panel.set_excluded(setting_to_names(self.model.get('exclude_frames')))
        self._tab_index = {}
        for key, widget, title in (('stretch', self.editor, _('Stretch and levels')),
                                   ('steps', self.post, _('Post-processing steps')),
                                   ('frames', self.frames_group, _('Frames'))):
            self._tab_index[key] = self.tools.addTab(widget, title)
        self._frames_visible = False
        self.collapse_btn = QToolButton()
        self.collapse_btn.setText('▾')
        self.collapse_btn.setToolTip(_('Collapse or expand the tools (Ctrl+T)'))
        self.collapse_btn.clicked.connect(lambda: self.set_tools_collapsed(not self._tools_collapsed))
        self.tools.setCornerWidget(self.collapse_btn, Qt.Corner.TopRightCorner)
        self.tools.hide()

        self.tools_split = QSplitter(Qt.Orientation.Vertical)
        self.tools_split.setChildrenCollapsible(False)
        self.tools_split.addWidget(group)
        self.tools_split.addWidget(self.tools)
        self.tools_split.setStretchFactor(0, 3)
        self.tools_split.setStretchFactor(1, 2)
        lay.addWidget(self.tools_split, 1)

        self.summary_group = QGroupBox(_('THIS RUN'))
        sl = QVBoxLayout(self.summary_group)
        self.plan_grid = QGridLayout()
        self.plan_grid.setColumnStretch(1, 1)
        sl.addLayout(self.plan_grid)
        self.summary_text = label('', wrap=True)
        self.summary_text.hide()
        sl.addWidget(self.summary_text)
        self.result_actions = QWidget()
        rv = QVBoxLayout(self.result_actions)
        rv.setContentsMargins(0, 0, 0, 0)
        ra = QHBoxLayout()
        for text, fn in ((_('Copy CLI command'), self._copy_cli_command),
                         (_('Open output folder'), self._open_output_folder),
                         (_('Start another run'), self._reset_run_view)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            ra.addWidget(b)
        ra.addStretch(1)
        rv.addLayout(ra)
        links = QHBoxLayout()
        support = ClickableLabel(_('Like the result? Support OriginStack'))
        support.setStyleSheet(f'color: {theme.ACCENT};')
        support.clicked.connect(self._open_support)
        links.addWidget(support)
        report = ClickableLabel(_('Result not right? Report it'))
        report.setStyleSheet(f'color: {theme.TEXT_DIM};')
        report.clicked.connect(self._report_menu)
        links.addWidget(report)
        links.addStretch(1)
        rv.addLayout(links)
        sl.addWidget(self.result_actions)
        self.result_actions.hide()
        lay.addWidget(self.summary_group)
        # On a short window the column scrolls instead of squeezing the controls.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    # ── menus, remembered state, tools panel, tray ──────────────────────

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu(_('&File'))
        act = file_menu.addAction(_('Choose light-frames folder…'), self._choose_folder)
        act.setShortcut(QKeySequence('Ctrl+O'))
        self.recent_menu = file_menu.addMenu(_('Recent folders'))
        self._refresh_recent_menu()
        file_menu.addSeparator()
        file_menu.addAction(_('Open output folder'), self._open_output_folder)
        file_menu.addAction(_('Copy CLI command'), self._copy_cli_command)
        file_menu.addSeparator()
        file_menu.addAction(_('Quit'), self.close)

        view = bar.addMenu(_('&View'))
        self.act_tools = QAction(_('Show tools'), self, checkable=True, checked=True)
        self.act_tools.setShortcut(QKeySequence('Ctrl+T'))
        self.act_tools.toggled.connect(lambda on: self.set_tools_collapsed(not on))
        view.addAction(self.act_tools)
        self.act_max = QAction(_('Maximise the picture'), self, checkable=True)
        self.act_max.setShortcut(QKeySequence('Ctrl+M'))
        self.act_max.toggled.connect(self.set_picture_maximised)
        view.addAction(self.act_max)
        self.act_hist = QAction(_('Histogram of the picture'), self, checkable=True)
        self.act_hist.toggled.connect(self.preview.hist_box.setChecked)
        self.preview.hist_box.toggled.connect(self.act_hist.setChecked)
        view.addAction(self.act_hist)
        view.addAction(_('Full screen picture'), self._preview_fullscreen)
        view.addSeparator()
        theme_menu = view.addMenu(_('Theme'))
        from src.desktop_control import load_app_setting
        current = load_app_setting('theme', 'auto')
        group = QActionGroup(self)
        for code, text in (('auto', _('Follow the system')), ('dark', _('Dark')), ('light', _('Light'))):
            a = QAction(text, self, checkable=True, checked=(code == current))
            a.triggered.connect(lambda _c=False, c=code: self._set_theme(c))
            group.addAction(a)
            theme_menu.addAction(a)

        help_menu = bar.addMenu(_('&Help'))
        help_menu.addAction(_('Keyboard shortcuts'), lambda: ShortcutsDialog(self).exec())
        help_menu.addAction(_('Report a problem…'), self._report_menu)
        help_menu.addAction(_('Support OriginStack'), self._open_support)
        help_menu.addSeparator()
        help_menu.addAction(_('About OriginStack'), self._about)

    def _about(self) -> None:
        from src.utils import read_version
        QMessageBox.about(self, _('About OriginStack'),
                          f'OriginStack v{read_version()}\n\n' + _(
                              'Stacking and post-processing for Celestron Origin and other '
                              'camera data.'))

    def _set_theme(self, code: str) -> None:
        from src.desktop_control import save_app_setting
        save_app_setting('theme', code)
        QMessageBox.information(self, _('Theme'), _('Restart OriginStack to change the theme.'))

    def _choose_folder(self) -> None:
        if self.rm.is_running():
            return
        start = str(self.model.get('directory')).strip()
        path = QFileDialog.getExistingDirectory(self, _('Choose light-frames folder'), start)
        if path:
            self.model.set('directory', os.path.normpath(path))

    def _refresh_recent_menu(self) -> None:
        self.recent_menu.clear()
        folders = [d for d in window_state.recent_folders() if os.path.isdir(d)]
        for d in folders:
            self.recent_menu.addAction(d, lambda _c=False, p=d: self.model.set('directory', p)
                                       if not self.rm.is_running() else None)
        self.recent_menu.setEnabled(bool(folders))

    def set_tools_collapsed(self, collapsed: bool) -> None:
        """Collapse the tools under the picture to their tab bar (or expand them)."""
        self._tools_collapsed = collapsed
        bar_h = self.tools.tabBar().sizeHint().height() + 4
        self.tools.setMaximumHeight(bar_h if collapsed else 16777215)
        self.collapse_btn.setText('▴' if collapsed else '▾')
        self.act_tools.blockSignals(True)
        self.act_tools.setChecked(not collapsed)
        self.act_tools.blockSignals(False)

    def set_picture_maximised(self, on: bool) -> None:
        """Hide the tools altogether so the picture takes the column."""
        self._picture_max = on
        self.act_max.blockSignals(True)
        self.act_max.setChecked(on)
        self.act_max.blockSignals(False)
        self._sync_tabs()

    @staticmethod
    def _phase_times(snap: Dict[str, Any]) -> str:
        """'Quality 12 s · Registration 5 s ...' from the phases the run went through."""
        parts = []
        for num in sorted(snap.get('phases') or {}):
            ph = snap['phases'][num]
            if ph.get('elapsed') is not None:
                parts.append(f"{ph['title']} {ph['elapsed']:.0f} s")
        return ' · '.join(parts)

    def _update_taskbar(self, snap: Optional[Dict[str, Any]], running: bool) -> None:
        """Mirror the run on the taskbar button: the progress bar while it goes, yellow while it
        waits for the frame review, red after a failure (cleared when the window is next used)."""
        if not self.taskbar.available:
            return
        self.taskbar.set_window(int(self.winId()))
        if running:
            if self._reviewing:
                self.taskbar.set_state(PAUSED)
            elif self.rm.is_cancelling() or snap is None:
                self.taskbar.set_value(0, 0)
            else:
                prog = snap['progress']
                self.taskbar.set_value(prog['done'], prog['total'])
        elif snap is not None and snap['run_status'] == 'error' and not self.isActiveWindow():
            self.taskbar.set_state(ERROR)
        else:
            self.taskbar.clear()

    def _setup_tray(self) -> None:
        """A tray icon for the finished-run message where the OS has no balloon of its own
        (``UIEvents`` already raises the Windows one)."""
        self.tray: Optional[QSystemTrayIcon] = None
        if sys.platform == 'win32' or not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(self.windowIcon(), self)
        self.tray.setToolTip('OriginStack')
        self.tray.show()

    def _notify_done(self, status: str, error: Optional[str]) -> None:
        if status in ('ok', 'error') and not self.isActiveWindow():
            QApplication.alert(self)             # flash the taskbar button until the window is used
        if self.tray is None or self.isActiveWindow() or status not in ('ok', 'error'):
            return
        self.tray.showMessage('OriginStack', _('Stacking run complete.') if status == 'ok'
                              else _('Run failed: {error}', error=error or ''))

    def _save_state(self) -> None:
        try:
            window_state.save({
                'geometry': window_state.encode(self.saveGeometry()),
                'main_split': window_state.encode(self.main_split.saveState()),
                'left_split': window_state.encode(self.vsplit.saveState()),
                'tools_split': window_state.encode(self.tools_split.saveState()),
                'tab': self.tools.tabText(self.tools.currentIndex()) if self.tools.count() else '',
                'tools_collapsed': self._tools_collapsed,
                'advanced_open': self.form._advanced_shown,
                'goals_open': self.form.goal.cards_open,
            })
        except Exception as exc:                 # never block closing on a settings file
            import logging
            logging.getLogger('originstack').debug('could not save window state: %s', exc)

    def _restore_state(self) -> None:
        st = window_state.load()
        if not st:
            return
        try:
            if st.get('geometry'):
                self.restoreGeometry(window_state.decode(st['geometry']))
            for key, widget in (('main_split', self.main_split),
                                ('left_split', self.vsplit), ('tools_split', self.tools_split)):
                if st.get(key):
                    widget.restoreState(window_state.decode(st[key]))
            if st.get('tools_collapsed'):
                self.set_tools_collapsed(True)
            if st.get('advanced_open') and not self.form._advanced_shown:
                self.form.toggle_advanced()
            self.form.goal.set_cards_open(st.get('goals_open', True))
            self._wanted_tab = st.get('tab', '')
        except Exception as exc:
            import logging
            logging.getLogger('originstack').debug('could not restore window state: %s', exc)

    def _shortcuts(self) -> None:
        ctx = Qt.ShortcutContext.WindowShortcut
        QShortcut(QKeySequence('Ctrl+Return'), self, activated=self._on_start, context=ctx)
        QShortcut(QKeySequence('Ctrl+Enter'), self, activated=self._on_start, context=ctx)
        QShortcut(QKeySequence('Escape'), self,
                  activated=lambda: self._on_cancel() if self.rm.is_running() else None, context=ctx)
        # Single letters only while the picture has the focus, or typing in a field would trigger them.
        pctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
        QShortcut(QKeySequence('F'), self.preview, activated=self.preview.fit, context=pctx)
        QShortcut(QKeySequence('1'), self.preview, activated=self.preview.one_to_one, context=pctx)
        QShortcut(QKeySequence('C'), self.preview,
                  activated=lambda: self.compare_box.setChecked(not self.compare_box.isChecked()),
                  context=pctx)

    # ── header actions: update, report, support ─────────────────────────

    def _start_update_check(self) -> None:
        """Once per launch, on a daemon thread; never blocks the window opening."""
        from src.utils import read_version, should_check_for_update
        if not should_check_for_update():
            return

        def bg() -> None:
            from src.net_query import check_for_update
            info = check_for_update(read_version())
            if info:
                self._bridge.update_found.emit(info)
        threading.Thread(target=bg, daemon=True).start()

    def _show_update(self, info: Dict[str, str]) -> None:
        self._update_url = info['url']
        self._update_info = info
        self.update_label.setText(_('Update available: v{version}', version=info['version']))

    def _open_update_url(self) -> None:
        from src.self_update import can_install_in_app
        info = self._update_info
        if info and can_install_in_app(info) and not self._update_busy:
            self._install_update(info)
        elif self._update_url:
            webbrowser.open(self._update_url)

    def _install_update(self, info: Dict[str, Any]) -> None:
        """Download the installer in the background, then start it and close the app."""
        if self.rm.is_running():
            QMessageBox.information(self, _('Update'),
                                    _('A run is in progress. Update when it has finished.'))
            return
        mb = info['installer_size'] / 1024 ** 2
        ask = QMessageBox.question(
            self, _('Update OriginStack'),
            _('Download and install v{version} ({mb:.0f} MB)?\n\nOriginStack closes, the '
              'installer runs, and you can start the new version from it.',
              version=info['version'], mb=mb))
        if ask != QMessageBox.StandardButton.Yes:
            return
        self._update_busy = True
        from src import net_query
        from src.self_update import installer_path

        def bg() -> None:
            try:
                path = net_query.download_update(
                    info['installer_url'], installer_path(info), info['installer_size'],
                    info['installer_sha256'],
                    progress=lambda done, total: self._bridge.update_progress.emit(done, total))
                self._bridge.update_done.emit(str(path))
            except Exception as exc:
                self._bridge.update_error.emit(str(exc))
        threading.Thread(target=bg, daemon=True).start()

    def _on_update_progress(self, done: int, total: int) -> None:
        self.update_label.setText(_('Downloading update… {pct}%', pct=int(100 * done / max(1, total))))

    def _on_update_error(self, error: str) -> None:
        self._update_busy = False
        if self._update_info:
            self.update_label.setText(_('Update available: v{version}',
                                        version=self._update_info['version']))
        QMessageBox.critical(self, _('Update'), _('The update could not be downloaded: {error}\n\n'
                                                  'The release page opens instead.', error=error))
        if self._update_url:
            webbrowser.open(self._update_url)

    def _on_update_done(self, path: str) -> None:
        self._update_busy = False
        from src.self_update import launch_installer
        try:
            launch_installer(path)
        except OSError as exc:
            QMessageBox.critical(self, _('Update'),
                                 _('The installer could not be started: {error}', error=exc))
            return
        QTimer.singleShot(300, lambda: (self.close(), _shutdown_and_exit(0)))

    def _report_menu(self) -> None:
        """A GitHub issue needs an account, so the same report can also be copied."""
        menu = QMenu(self)
        menu.addAction(_('Open a GitHub issue…'), self._report_problem)
        menu.addAction(_('Copy report (for an email or forum post)'), self._copy_problem_report)
        menu.addAction(_('Copy diagnostics (version, system, settings, log)'), self._copy_diagnostics)
        menu.exec(self.cursor().pos())

    def _report_error(self) -> Optional[str]:
        snap = self.ui.snapshot()
        return snap.get('run_error') if snap.get('run_status') == 'error' else None

    def _log_text(self) -> str:
        return self.log.toPlainText()

    def _report_problem(self) -> None:
        from src.desktop_control import problem_report_url
        webbrowser.open(problem_report_url(self._log_text(), self._report_error()))

    def _copy_problem_report(self) -> None:
        from src.desktop_control import problem_report_text
        QApplication.clipboard().setText(problem_report_text(self._log_text(), self._report_error()))
        self._say(_('Copied the problem report to the clipboard. Check it, then '
                    'paste it into an email or a forum post (Cloudy Nights, the '
                    'GitHub Discussions page).'))

    def _copy_diagnostics(self) -> None:
        from src.desktop_control import diagnostics_text
        QApplication.clipboard().setText(diagnostics_text(self.model.read_form(), self._log_text()))
        self._say(_('Copied the diagnostics to the clipboard. Home folders are '
                    'shown as ~; check it before sharing.'))

    def _say(self, text: str) -> None:
        """A note in the diagnostics box, opened if it was closed."""
        self.diagnostics_text.setText(text)
        if not self._diagnostics_open:
            self._toggle_diagnostics()

    def _open_support(self) -> None:
        from src.desktop_control import SUPPORT_URL
        webbrowser.open(SUPPORT_URL)

    def _toggle_diagnostics(self) -> None:
        self._diagnostics_open = not self._diagnostics_open
        self.diagnostics_group.setVisible(self._diagnostics_open)

    # ── run control ─────────────────────────────────────────────────────

    def _on_start(self) -> None:
        if not self.start_btn.isEnabled():
            return
        form = self.model.read_form()
        from src.desktop_control import save_app_setting
        save_app_setting('temp_dir', str(form.get('temp_dir') or '').strip())    # a big session needs a roomy drive
        self.model.remember_folders()
        if str(self.model.get('directory')).strip():
            window_state.remember_folder(str(self.model.get('directory')))
            self._refresh_recent_menu()
        form = with_export_checkboxes(form, str(self.model.get('export', '')),
                                      {f: b.isChecked() for f, b in self.export_boxes.items()})
        for fmt, box in self.export_boxes.items():
            save_app_setting(f'export_{fmt}', box.isChecked())
        save_app_setting('review_frames', self.review_box.isChecked())
        self.rm.review_frames = self.review_box.isChecked()
        result = self.rm.start(form)
        if not result.get('ok'):
            self.status_label.setText(_('Error: {error}', error=result.get('error')))
            return
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText(_('Running…'))
        self.readiness.setText('')
        self.readiness.hide()
        self._shown_log_lines = 0
        self._clear_frames_table()
        self.summary_text.setText('')
        self._show_plan = False
        self.summary_group.setTitle(_('COMPLETE'))
        self._set_plan_visible(False)
        self.summary_text.show()
        self.result_actions.hide()

    def _on_cancel(self) -> None:
        # Cooperative, not instant: takes effect at the next checkpoint. Disabling the button
        # is the honest signal, pressing it again would not make the pipeline notice sooner.
        self.rm.cancel()
        self.cancel_btn.setEnabled(False)
        self.status_label.setText(_('Cancelling…'))

    def closeEvent(self, event):
        if self.rm.is_running():
            ask = QMessageBox.question(self, 'OriginStack',
                                       _('A stacking run is still in progress. Quit anyway?'))
            if ask != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.post.close_compare()
        self._save_state()
        event.accept()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """A session folder (or a file from it) dropped on the window sets the light frames."""
        from src.drop_files import folder_from_drop
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        folder = folder_from_drop(paths)
        if folder and not self.rm.is_running():
            self.model.set('directory', os.path.normpath(folder))

    # ── preflight, "This run" ───────────────────────────────────────────

    @staticmethod
    def _check_icon(ok: Optional[bool], text: str, warning: str = '') -> str:
        if ok is True:
            return f'✓ {text}'
        if ok is False:
            return f'⚠ {warning or text}'
        return f'○ {text}'

    def _update_preflight(self) -> None:
        directory = str(self.model.get('directory')).strip()
        t = self._preflight_titles
        lab = self.preflight_labels
        if not directory or not os.path.isdir(directory):
            for key, lbl in lab.items():
                lbl.setText(self._check_icon(None, t[key]))
            self._show_preflight_summary()
            return
        counts = self.model.frame_counts
        lights = counts.get('light', 0)
        has_cal = any(counts.get(k, 0) for k in ('dark', 'flat', 'bias', 'darkflat'))
        free = shutil.disk_usage(directory).free
        estimated = max(1, lights) * 170 * 1024 * 1024
        lab['frames'].setText(self._check_icon(lights > 0, t['frames'], _('No light frames found')))
        lab['calibration'].setText(self._check_icon(has_cal, t['calibration'],
                                                    _('No calibration frames found')))
        lab['target'].setText(self._check_icon(bool(self.model.inferred[1]), t['target'],
                                               _('Target will be selected manually')))
        lab['space'].setText(self._check_icon(
            free >= estimated, t['space'],
            _('Low disk space — about {gb:.1f} GB needed').format(gb=estimated / 1024 ** 3)))
        lab['output'].setText(self._check_icon(bool(directory), t['output']))
        self._show_preflight_summary()

    def _show_preflight_summary(self) -> None:
        """One line once every check passes; the full grid as soon as one needs attention."""
        ok = all(lbl.text().startswith('✓') for lbl in self.preflight_labels.values())
        for lbl in self.preflight_labels.values():
            lbl.setVisible(not ok)
        self.preflight_ok.setVisible(ok)
        if ok:
            summary = self.model.data_summary
            self.preflight_ok.setText('✓ ' + (_('Ready to start · {summary}', summary=summary)
                                              if summary else _('All checks passed')))

    def _set_plan_visible(self, on: bool) -> None:
        for i in range(self.plan_grid.count()):
            w = self.plan_grid.itemAt(i).widget()
            if w is not None:
                w.setVisible(on)

    def _form_changed(self) -> None:
        """Show what Start would now do, and the example result for the chosen target until
        a run's own previews arrive."""
        if self.rm.is_running():
            return
        folder = str(self.model.get('directory')).strip()
        if not folder:
            msg = _('Choose a light-frame folder to begin.')
        elif not os.path.isdir(folder):
            msg = _('Folder not found — choose an existing light-frame folder.')
        elif not self.model.data_summary:
            msg = _('No frames found — choose a folder containing light frames.')
        else:
            msg = ''
        self.readiness.setText(msg)
        self.readiness.setVisible(bool(msg))
        self._update_preflight()
        self._show_plan = True
        self.summary_group.setTitle(_('THIS RUN'))
        self.summary_text.hide()
        while self.plan_grid.count():
            w = self.plan_grid.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        for i, (head, text) in enumerate(self.model.describe_run()):
            self.plan_grid.addWidget(set_role(QLabel(head), 'dim'), i, 0, Qt.AlignmentFlag.AlignTop)
            self.plan_grid.addWidget(label(text, wrap=True), i, 1)
        if self.preview.current_slug in ('', 'example'):
            path, caption = self.model.example()
            if path is not None:
                try:
                    self.preview.load_slot(path.read_bytes(), 'example', caption)
                except Exception:
                    self.preview.clear()
            elif self.preview.current_slug == 'example':
                self.preview.clear()

    def _reset_run_view(self) -> None:
        if self.rm.is_running():
            return
        self.preview.clear()
        self._clear_frames_table()
        self.summary_text.setText('')
        self.result_actions.hide()
        self._form_changed()

    def _copy_cli_command(self) -> None:
        """The current form as a reproducible command line."""
        from src.desktop_control import build_argv_from_form
        try:
            argv = build_argv_from_form(self.model.read_form())
        except Exception as exc:
            _fatal(_('Command export failed'), str(exc))
            return
        command = subprocess.list2cmdline([sys.executable, 'originstack.py', *argv])
        QApplication.clipboard().setText(command)
        self._say(_('Copied command to the clipboard:\n{command}', command=command))

    # ── preview controls ────────────────────────────────────────────────

    @staticmethod
    def _combo_slug(combo: NoWheelComboBox) -> str:
        return combo.currentData() or ''

    def _show_picture(self, data: bytes, slug: str, caption: str) -> None:
        """Show a JPEG in the viewer. The final picture (and the live stretch of it) is a reduced
        copy of the real image: tell the viewer the real size so 1:1 means real pixels, and look
        for a full-size copy if the user zooms in."""
        final_family = slug in ('final', 'live')
        self.preview.load_slot(data, slug, caption, self.ui.final_size() if final_family else None)
        self._detail_gen += 1
        self._detail_loading = False
        self.preview.set_detail_status('')

    def _load_named(self, slug: str) -> None:
        data = self.ui.named_jpeg(slug)
        if data:
            slot = next((s for s in self._named_cache if s['slug'] == slug), None)
            self._show_picture(data, slug, slot['caption'] if slot else slug)

    # ── full-size copy for zooming in ───────────────────────────────────

    def _sample_linear(self, x: int, y: int):
        if self.preview.current_slug not in ('final', 'live'):
            return None
        return self.ui.sample_linear(x, y)

    def _on_detail_wanted(self) -> None:
        self._detail_timer.start(250)               # wait for the zoom to settle

    def _request_detail(self) -> None:
        v = self.preview.view
        if (self.preview.current_slug not in ('final', 'live') or self.rm.is_running()
                or self._detail_loading or self._detail_done == self._detail_gen
                or not v.needs_detail() or self.ui.final_size() is None):
            return
        self._detail_loading = True
        gen = self._detail_gen
        self.preview.set_detail_status(_('loading full size…'))

        def bg() -> None:
            try:
                r = self.ui.full_view()
                self._bridge.detail_ready.emit(gen, r['rgb8'], r['size'])
            except Exception as e:
                self._bridge.detail_failed.emit(gen, str(e))
        threading.Thread(target=bg, daemon=True).start()

    def _on_detail_ready(self, gen: int, rgb8, size) -> None:
        self._detail_loading = False
        if gen != self._detail_gen or self.preview.current_slug not in ('final', 'live'):
            self._on_detail_wanted()                # the picture changed meanwhile: try again
            return
        self.preview.load_array(rgb8, size)
        self._detail_done = gen

    def _on_detail_failed(self, gen: int, error: str) -> None:
        self._detail_loading = False
        self._detail_done = gen                     # do not retry the same picture in a loop
        self.preview.set_detail_status(_('full size unavailable'))

    def _on_view_selected(self) -> None:
        self._unpin_frame()
        self._load_named(self._combo_slug(self.view_combo))

    def _on_compare_toggled(self, on: bool) -> None:
        self.compare_combo.setEnabled(on)
        self.preview.set_compare(on)
        self.wipe.setVisible(on)
        if on:
            self._on_compare_selected()

    def _on_compare_selected(self) -> None:
        slug = self._combo_slug(self.compare_combo)
        data = self.ui.named_jpeg(slug) if slug else None
        self.preview.set_compare_slot(data, slug)

    def _show_final(self, data: bytes, slug: str, caption: str) -> None:
        self._show_picture(data, slug, caption)

    def _show_live(self, data: bytes, caption: str) -> None:
        """The stretch editor's reduced render, in place of the final picture."""
        self._show_picture(data, 'live', caption)
        self._on_detail_wanted()                    # zoomed in: sharpen once the editing pauses

    def _on_editor_availability(self, available: bool) -> None:
        if available:
            self.tools.setCurrentIndex(self._tab_index['stretch'])

    def _sync_tabs(self) -> None:
        """Show the tool tabs that have something to show; none at all hides the tool area."""
        shown = {'stretch': self.editor.available,
                 'steps': not self.post.pp_group.isHidden() and self.editor.available,
                 'frames': self._frames_visible or self._reviewing}
        for key, idx in self._tab_index.items():
            self.tools.setTabVisible(idx, shown[key])
        any_tab = any(shown.values())
        self.stars_box.setEnabled(shown['steps'])
        self.tools.setVisible(any_tab and not self._picture_max)
        want = getattr(self, '_wanted_tab', '')
        if want:
            for key, idx in self._tab_index.items():
                if shown[key] and self.tools.tabText(idx) == want:
                    self.tools.setCurrentIndex(idx)
                    self._wanted_tab = ''
        if any_tab and not self.tools.isTabVisible(self.tools.currentIndex()):
            self.tools.setCurrentIndex(next(i for k, i in self._tab_index.items() if shown[k]))

    def _preview_fullscreen(self) -> None:
        """The current picture in a large window, closed with Escape."""
        if not self.preview.has_image():
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(_('Preview'))
        dlg.resize(1200, 800)
        lay = QVBoxLayout(dlg)
        view = PreviewPane()
        lay.addWidget(view)
        view.load_slot(self.preview.source_bytes(), self.preview.current_slug, self.preview.caption())
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.show()

    # ── frames table ────────────────────────────────────────────────────

    def _sorted_frame_rows(self, rows: List[tuple]) -> List[tuple]:
        """The rows in the order the header was last clicked (arrival order until then)."""
        if self._frame_sort is None:
            return rows
        col, desc = self._frame_sort
        i = _FRAME_COLS.index(col)

        def key(row):
            v = row[i]
            if col == 'frame':
                return str(v)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float('-inf')
        return sorted(rows, key=key, reverse=desc)

    def _sort_frames(self, col: str) -> None:
        """Header click: sort by that column, again to reverse, a third time for arrival order."""
        cur = self._frame_sort
        if cur is None or cur[0] != col:
            self._frame_sort = (col, col != 'frame')      # quality columns best first
        elif cur[1] == (col != 'frame'):
            self._frame_sort = (col, not cur[1])
        else:
            self._frame_sort = None
        for i, c in enumerate(_FRAME_COLS):
            arrow = ''
            if self._frame_sort and self._frame_sort[0] == c:
                arrow = ' ▼' if self._frame_sort[1] else ' ▲'
            self.frames_table.horizontalHeaderItem(i).setText(self._frame_headings[c] + arrow)
        self._frame_rows_shown = []            # rebuilt on the next poll

    def _clear_frames_table(self) -> None:
        self._pinned_frame = None
        self._frame_rows_shown = []
        self.frames_table.setRowCount(0)
        self.frames_panel.set_rows([])

    def _fill_frames_table(self, rows: List[tuple]) -> None:
        self.frames_table.blockSignals(True)
        self.frames_table.setRowCount(len(rows))
        for r, (name, score, snr, stars, fwhm, ok) in enumerate(rows):
            for c, value in enumerate((name, score, snr, stars, fwhm)):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment((Qt.AlignmentFlag.AlignLeft if c == 0 else Qt.AlignmentFlag.AlignRight)
                                      | Qt.AlignmentFlag.AlignVCenter)
                if not ok:
                    item.setForeground(QColor(theme.BAD))
                self.frames_table.setItem(r, c, item)
            if name == self._pinned_frame:           # keep the picked row selected
                self.frames_table.selectRow(r)
        self.frames_table.blockSignals(False)

    def _on_frame_selected(self) -> None:
        """A row was picked: show that frame in the preview."""
        picked = self.frames_table.selectedItems()
        if not picked:
            return
        name = self.frames_table.item(picked[0].row(), 0).text()
        slug = f'frame:{name}'
        if self._pinned_frame == name and self.preview.current_slug == slug:
            return                    # the table was rebuilt around the row already shown
        self._pinned_frame = name
        if self._reviewing:
            self.frames_panel.stop_countdown()          # looking at a frame: stay paused
        self.frames_panel.graph.set_focus(name)
        data = self.ui.frame_jpeg_for(name)
        if data:
            self.preview.load_slot(data, slug, name)
        else:
            self.preview.set_caption(_('{name}: no preview was kept for this frame', name=name))

    def _on_graph_frame(self, name: str) -> None:
        """A frame was clicked on the graph: select its row (which shows its picture)."""
        for r in range(self.frames_table.rowCount()):
            if self.frames_table.item(r, 0).text() == name:
                self.frames_table.selectRow(r)
                return
        self._pinned_frame = name                        # not in the table: show it anyway
        data = self.ui.frame_jpeg_for(name)
        if data:
            self.preview.load_slot(data, f'frame:{name}', name)
        else:
            self.preview.set_caption(_('{name}: no preview was kept for this frame', name=name))

    def _on_exclusions_edited(self, dest: str) -> None:
        if dest == 'exclude_frames':
            self.frames_panel.set_excluded(setting_to_names(self.model.get('exclude_frames')))

    def _sync_review(self, waiting: bool) -> None:
        """While the run waits for the frame review: show the graph with its banner."""
        if waiting == self._reviewing:
            return
        self._reviewing = waiting
        self.frames_panel.set_review_mode(waiting)
        if waiting:
            self.set_picture_maximised(False)
            self.set_tools_collapsed(False)
            self._sync_tabs()
            self.tools.setCurrentIndex(self._tab_index['frames'])
            self.status_label.setText(_('Paused: review the frames'))
            self.header_status.setText(_('Waiting for you'))
            if not self.isActiveWindow():
                QApplication.alert(self)
            self.activateWindow()
        else:
            self._sync_tabs()

    def _on_review_continue(self) -> None:
        names = self.frames_panel.excluded()
        self.model.set('exclude_frames', names_to_setting(names))
        self.rm.review_continue(names)
        self.status_label.setText(_('Running…'))

    def _unpin_frame(self) -> None:
        """Back to following the run's milestones."""
        self._pinned_frame = None
        self.frames_table.blockSignals(True)
        self.frames_table.clearSelection()
        self.frames_table.blockSignals(False)

    # ── poll loop ───────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            self._refresh()
        except Exception as exc:                 # a bad snapshot must not stop the timer
            import logging
            logging.getLogger('originstack').debug('refresh failed: %s', exc, exc_info=True)

    def _fill_combo(self, combo: NoWheelComboBox, items: List[Tuple[str, str]]) -> None:
        """Replace the entries when they changed, keeping the selected slug."""
        current = [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]
        if current == items:
            return
        keep = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for caption, slug in items:
            combo.addItem(caption, slug)
        i = combo.findData(keep) if keep else -1
        combo.setCurrentIndex(i)
        combo.blockSignals(False)

    def _refresh(self) -> None:
        snap = self.ui.snapshot()
        self._last_snap = snap
        if snap['version'] == self._last_version:
            self._refresh_run_button()
            return
        self._last_version = snap['version']

        # A new run: drop the previous run's picture, milestones and thumbnails.
        if snap['run_status'] == 'running' and self._last_run_status != 'running':
            self.preview.clear()
            self._clear_frames_table()
            self.compare_box.setChecked(False)
            self._named_cache = []
            for c in (self.view_combo, self.compare_combo):
                c.blockSignals(True)
                c.clear()
                c.blockSignals(False)

        phase = snap['phase']
        run_done = snap['run_status'] == 'ok'
        for i, lbl in enumerate(self.phase_labels, start=1):
            lbl.setProperty('state', 'done' if run_done or i < phase
                            else 'active' if i == phase else '')
            repolish(lbl)
        prog = snap['progress']
        self.progress.setMaximum(max(prog['total'], 1))
        self.progress.setValue(prog['done'])
        self.progress_text.setText(f"{prog['label']} ({prog['done']}/{prog['total']})"
                                   if prog['total'] else prog['label'])
        eta = prog.get('eta')
        if eta is not None and eta > 0:
            seconds = int(eta)
            self.eta_text.setText(
                _('About {minutes} min remaining', minutes=max(1, round(seconds / 60)))
                if seconds >= 60 else _('About {seconds} sec remaining', seconds=seconds))
        else:
            self.eta_text.setText('')

        log_lines = snap['log']
        if len(log_lines) < self._shown_log_lines:
            self._shown_log_lines = 0            # run_started() cleared it
            self.log.clear()
        new_lines = log_lines[self._shown_log_lines:]
        if new_lines:
            self.log.appendPlainText('\n'.join(new_lines))
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
            self._shown_log_lines = len(log_lines)

        rows = [(r['name'], r['score'], r['snr'], r['stars'], r['fwhm'], r['ok'])
                for r in snap['frames']]
        rows = self._sorted_frame_rows(rows)
        self._frames_visible = bool(rows)
        if rows != self._frame_rows_shown:
            self._frame_rows_shown = rows
            self._fill_frames_table(rows)

        review = snap.get('review') or {}
        graph_rows = review['frames'] if review.get('waiting') else snap['frames']
        if graph_rows != self._graph_rows:
            self._graph_rows = [dict(r) for r in graph_rows]
            self.frames_panel.set_rows(self._graph_rows)
        self._sync_review(bool(review.get('waiting')))
        self._named_cache = snap['named']
        items = [(n['caption'], n['slug']) for n in self._named_cache]
        self._fill_combo(self.view_combo, items)
        self._fill_combo(self.compare_combo, items)
        if self._pinned_frame is not None:
            pass                                  # the user is looking at a frame they picked
        elif items and self.view_combo.currentIndex() < 0:
            self.view_combo.setCurrentIndex(len(items) - 1)
            self._on_view_selected()
        elif items and items[-1][1] != self.preview.current_slug and not self.compare_box.isChecked():
            self.view_combo.setCurrentIndex(len(items) - 1)       # follow the latest milestone
            self._on_view_selected()

        if snap['summary'] and not self._show_plan:
            lines = [f'{k}: {v}' for k, v in snap['summary'].items()]
            times = self._phase_times(snap)
            if times:
                lines.append(_('Time per phase: {times}', times=times))
            self.summary_text.setText('\n'.join(lines))
            self.result_actions.show()
        if snap['run_status'] in ('ok', 'error', 'cancelled'):
            rejected = sum(1 for row in snap['frames'] if not row.get('ok', True))
            message = _('No warnings recorded.')
            if rejected:
                message = _('{count} frame(s) were rejected during quality analysis.').format(
                    count=rejected)
            if snap['run_error']:
                message = _('Run failed: {error}').format(error=snap['run_error'])
                message += '\n' + _('Report a problem (top right) opens a GitHub issue or copies '
                                    'the details, with the log, for an email or a forum post.')
            self.diagnostics_text.setText(message)
            self.report_label.setStyleSheet(
                f'color: {theme.ACCENT if snap["run_error"] else theme.TEXT_DIM};')

        self.post.sync(snap)
        self.editor.sync(snap)
        self._sync_tabs()
        self._refresh_run_button(snap)

    def _open_output_folder(self) -> None:
        """Show the stack the last run wrote (selected in Explorer), else the folder the Output
        field names."""
        path = self.rm.last_output
        if not path:
            typed = str(self.model.get('output')).strip()
            path = typed or None
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
                    subprocess.Popen(['explorer', '/select,', target_file])
                else:
                    os.startfile(folder)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', '-R', target_file] if target_file else ['open', folder])
            else:
                subprocess.Popen(['xdg-open', folder])
        except Exception as exc:                  # opening the folder is a convenience
            import logging
            logging.getLogger('originstack').debug('could not open the output folder: %s', exc)

    def _shape_left_for_run(self, running: bool) -> None:
        """While a run goes the log and progress matter and the form does not: give the log the
        height; when it is over, give the form its half back."""
        if running == self._left_running:
            return
        self._left_running = running
        self.preflight_group.setVisible(not running)
        h = self.vsplit.height()
        if h > 1:
            top = int(h * 0.28) if running else max(h // 2, h - 300)
            self.vsplit.setSizes([top, h - top])

    def _refresh_run_button(self, snap: Optional[Dict[str, Any]] = None) -> None:
        # Called every poll, also when nothing changed: use the last snapshot then. The run
        # thread can still be winding down when the finished snapshot arrives, so the final
        # status has to be applied on a later poll, not only on the one that brought it.
        snap = snap or self._last_snap
        running = self.rm.is_running()
        self._shape_left_for_run(running)
        if running:
            self.start_btn.setEnabled(False)
            if self._reviewing:
                self.header_status.setText(_('Waiting for you'))
            else:
                self.header_status.setText(_('Cancelling…') if self.rm.is_cancelling() else _('Running…'))
        else:
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            if snap is not None and snap['run_status'] in ('ok', 'error', 'cancelled'):
                done_ok = snap['run_status'] == 'ok'
                if snap['run_status'] == 'cancelled':
                    self.status_label.setText(_('Cancelled'))
                    self.header_status.setText(_('Cancelled'))
                else:
                    self.status_label.setText(_('Done') if done_ok
                                              else _('Failed: {error}', error=snap['run_error']))
                    self.header_status.setText(_('Complete') if done_ok else _('Failed'))
                # Edge-triggered, so it opens once per completed run, not on every poll.
                if self._last_run_status == 'running':
                    self._notify_done(snap['run_status'], snap['run_error'])
                if done_ok and self._last_run_status != 'ok' and self.open_when_done.isChecked():
                    self._open_output_folder()
            elif snap is None or snap['run_status'] == 'idle':
                self.header_status.setText(_('Idle'))
        self._update_taskbar(snap, running)
        if snap is not None:
            self._last_run_status = snap['run_status']
