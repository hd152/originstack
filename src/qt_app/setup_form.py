"""The Setup form: target cards, run mode, common fields, saved settings and the searchable
"Additional options" sections. All state lives in ``FormModel``; these widgets mirror it."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.desktop_common import (
    _COMMON_DESTS,
    _NO_CHANGE_MARK,
    _PICKER_DESTS,
    _RUN_MODES,
    _TARGET_CARDS,
    _field_label,
    _field_label_ui,
    _is_expert_field,
    _is_expert_group,
)
from src.i18n import _
from src.qt_app.form_model import FormModel
from src.qt_app.widgets import (
    ClickableFrame,
    ClickableLabel,
    NoWheelComboBox,
    PixmapLabel,
    card_pixmap,
    label,
    repolish,
    set_role,
)

RESCAN_DEBOUNCE_MS = 400


class Card(ClickableFrame):
    """A clickable bordered card: optional picture, title, optional badge, description."""

    def __init__(self, title: str, text: str, pixmap=None, wrap_px: int = 150, parent=None):
        super().__init__(parent)
        set_role(self, 'card', selected='false')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(3)
        if pixmap is not None:
            lay.addWidget(PixmapLabel(pixmap))
        row = QHBoxLayout()
        self.title = QLabel(title)
        self.title.setStyleSheet('font-weight: bold;')
        row.addWidget(self.title)
        row.addStretch(1)
        self.badge = set_role(QLabel(''), 'badge')
        self.badge.hide()
        row.addWidget(self.badge)
        lay.addLayout(row)
        self.text = label(text, 'dim', wrap=True)
        self.text.setStyleSheet('font-size: 8pt;')
        lay.addWidget(self.text)

    def set_state(self, selected: bool) -> None:
        self.setProperty('selected', 'true' if selected else 'false')
        repolish(self)

    def set_badge(self, text: str) -> None:
        self.badge.setText(text)
        self.badge.setVisible(bool(text))


class GoalPicker(QWidget):
    """The two questions at the top of the form: what was imaged and how to run. Both write the
    target_type / comet_mode / preset values of the model, and follow them when those are
    changed from Additional options."""

    def __init__(self, model: FormModel, parent=None):
        super().__init__(parent)
        self.model = model
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 0)
        head = QHBoxLayout()
        self.cards_toggle = ClickableLabel('')
        set_role(self.cards_toggle, 'section')
        self.cards_toggle.setToolTip(_('Show or hide the target cards'))
        self.cards_toggle.clicked.connect(lambda: self.set_cards_open(not self.cards_open))
        head.addWidget(self.cards_toggle)
        self.cards_summary = set_role(QLabel(''), 'dim')
        head.addWidget(self.cards_summary)
        head.addStretch(1)
        lay.addLayout(head)
        self.cards_box = QWidget()
        grid = QGridLayout(self.cards_box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self.cards: Dict[str, Card] = {}
        from src.desktop_common import _EXAMPLES_DIR
        for i, card in enumerate(_TARGET_CARDS):
            example = _EXAMPLES_DIR / card[5] if card[5] else None
            c = Card(_(card[1]), _(card[2]), card_pixmap(example, card[0]))
            c.clicked.connect(lambda k=card[0]: model.pick_target(k))
            grid.addWidget(c, i // 3, i % 3)
            self.cards[card[0]] = c
        for col in range(3):
            grid.setColumnStretch(col, 1)
        lay.addWidget(self.cards_box)
        self.cards_open = True
        lay.addSpacing(6)
        lay.addWidget(set_role(QLabel(_('HOW SHOULD IT RUN?')), 'section'))
        modes = QHBoxLayout()
        self.modes: Dict[str, Card] = {}
        for key, title, text, _preset in _RUN_MODES:
            m = Card(_(title), _(text))
            m.clicked.connect(lambda k=key: model.pick_mode(k))
            modes.addWidget(m, 1)
            self.modes[key] = m
        lay.addLayout(modes)
        model.value_changed.connect(self._on_value)
        model.scanned.connect(self.refresh)
        self.set_cards_open(True)
        self.refresh()

    def set_cards_open(self, on: bool) -> None:
        """Show the target cards, or fold them to one line naming the choice (a short window
        has little room for six pictures)."""
        self.cards_open = bool(on)
        self.cards_box.setVisible(self.cards_open)
        self.cards_toggle.setText(('▾ ' if self.cards_open else '▸ ') + _('WHAT DID YOU IMAGE?'))
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        key = self.model.selected_target()
        card = next((c for c in _TARGET_CARDS if c[0] == key), None)
        self.cards_summary.setText('' if self.cards_open else (_(card[1]) if card else ''))

    def _on_value(self, dest: str) -> None:
        if dest in _PICKER_DESTS:
            self.refresh()

    def refresh(self) -> None:
        self._refresh_summary()
        target, mode = self.model.selected_target(), self.model.selected_mode()
        self.cards['auto'].text.setText(self.model.auto_card_text())
        for key, card in self.cards.items():
            card.set_state(key == target)
            card.set_badge(_('SUGGESTED') if key == self.model.suggested and key != target else '')
        for key, m in self.modes.items():
            m.set_state(key == mode)


class FieldView:
    """The widgets of one field, laid out on a grid row: changed dot, label, input, Browse,
    reset link, and the one-line description beneath (or beside a checkbox)."""

    def __init__(self, form: 'SetupForm', parent: QWidget, grid: QGridLayout, row: int,
                 field: Dict[str, Any], group: Optional[str]):
        self.form = form
        self.dest = dest = field['dest']
        self.field = field
        self.group = group
        model = form.model
        kind = field['kind']
        self.widgets: List[QWidget] = []
        self.rows_used = 1
        self._loading = False

        self.dot = set_role(QLabel(''), 'changed')
        self.dot.setFixedWidth(14)
        grid.addWidget(self.dot, row, 0, Qt.AlignmentFlag.AlignRight)
        self.label = QLabel(_field_label_ui(field))
        tip = _(field['help']) if field.get('help') else ''
        if field.get('flag'):
            tip = f"{field['flag']}\n{tip}".strip()
        if tip:
            self.label.setToolTip(tip)
        grid.addWidget(self.label, row, 1)
        self.widgets += [self.dot, self.label]

        summary_en = field.get('summary') or ''
        summary = _(summary_en) if summary_en else ''
        self.hint: Optional[QLabel] = None
        self.edit = None
        if kind in ('bool_true', 'bool_false'):
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            self.edit = QCheckBox()
            self.edit.toggled.connect(lambda on: self._edited(on))
            hl.addWidget(self.edit)
            if summary:
                self.hint = set_role(QLabel(summary), 'hint')
                self.hint.setWordWrap(True)
                hl.addWidget(self.hint, 1)
            else:
                hl.addStretch(1)
            grid.addWidget(holder, row, 2, 1, 2)
            self.widgets.append(holder)
        elif kind == 'select':
            self.edit = NoWheelComboBox()
            self.edit.addItems([''] + [str(c) for c in (field['choices'] or [])])
            self.edit.currentTextChanged.connect(self._edited)
            grid.addWidget(self.edit, row, 2)
            self.widgets.append(self.edit)
        else:
            self.edit = QLineEdit()
            self.edit.textEdited.connect(self._edited)
            grid.addWidget(self.edit, row, 2)
            self.widgets.append(self.edit)
            hint_kind = field.get('widget')
            if hint_kind:
                btn = QPushButton(_('Browse…'))
                btn.clicked.connect(lambda _c=False, h=hint_kind: self._browse(h))
                grid.addWidget(btn, row, 3)
                self.widgets.append(btn)
        if tip and self.edit is not None and kind not in ('bool_true', 'bool_false'):
            self.edit.setToolTip(tip)

        self.reset = set_role(ClickableLabel(''), 'reset')
        self.reset.clicked.connect(lambda: model.reset(dest))
        grid.addWidget(self.reset, row, 4)
        self.widgets.append(self.reset)

        if summary and self.hint is None:
            self.hint = set_role(QLabel(summary), 'hint')
            self.hint.setWordWrap(True)
            grid.addWidget(self.hint, row + 1, 1, 1, 4)
            self.widgets.append(self.hint)
            self.rows_used = 2
        if self.hint is not None and tip:
            self.hint.setToolTip(tip)
        if dest == 'directory' and self.hint is not None:
            self._dir_summary = summary

        self.expert = _is_expert_field(field) or (group is not None and _is_expert_group(group))
        self.haystack = ' '.join([
            _field_label(field), _field_label_ui(field), field.get('flag') or '', dest,
            summary_en, summary, ' '.join(str(c) for c in field.get('choices') or [])
        ]).replace('_', ' ').lower()
        self.load()

    # ── model <-> widget ────────────────────────────────────────────────

    def load(self) -> None:
        """Show the model's value in the input (without echoing it back)."""
        v = self.form.model.get(self.dest)
        self._loading = True
        try:
            if isinstance(self.edit, QCheckBox):
                self.edit.setChecked(bool(v))
            elif isinstance(self.edit, NoWheelComboBox):
                i = self.edit.findText(str(v))
                self.edit.setCurrentIndex(i if i >= 0 else 0)
            elif isinstance(self.edit, QLineEdit) and self.edit.text() != str(v):
                self.edit.setText(str(v))
        finally:
            self._loading = False
        self.update_changed()

    def _edited(self, value) -> None:
        if self._loading:
            return
        if self.dest == 'directory':
            self.form.model.set(self.dest, value, rescan=False)
            self.form.schedule_rescan()
        else:
            self.form.model.set(self.dest, value)

    def update_changed(self) -> None:
        changed = self.form.model.is_changed(self.dest)
        mark = changed and self.dest not in _NO_CHANGE_MARK
        self.dot.setText('●' if mark else '')
        self.reset.setText(_('reset') if mark else '')

    def _browse(self, kind: str) -> None:
        if kind == 'dir':
            path = QFileDialog.getExistingDirectory(self.form, '')
        elif kind == 'file-save':
            path, _f = QFileDialog.getSaveFileName(
                self.form, '', '', f"{_('FITS files')} (*.fits);;{_('All files')} (*)")
        else:
            path, _f = QFileDialog.getOpenFileName(self.form)
        if path:
            self.form.model.set(self.dest, path)

    def set_visible(self, visible: bool) -> None:
        for w in self.widgets:
            w.setVisible(visible)


class SetupForm(QWidget):
    """Fixed common fields on the form; every other field behind a collapsed "Additional
    options" section (a sidebar of groups, searchable). Field kinds and path widgets come
    straight from the schema; nothing here knows individual flags."""

    #: after "Additional options" opened (so a scroll area can bring it into view)
    expanded = Signal(object)

    def __init__(self, model: Optional[FormModel] = None, parent=None):
        super().__init__(parent)
        self.model = model or FormModel()
        self.fields: Dict[str, FieldView] = {}
        self._advanced_shown = False
        self._current_group: Optional[str] = None
        schema = self.model.schema
        all_fields = self.model.fields

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 6, 0)

        # ── common fields, with the cards after the temp folder ──
        common = QWidget()
        cgrid = QGridLayout(common)
        cgrid.setContentsMargins(0, 0, 0, 0)
        cgrid.setColumnStretch(2, 1)
        row = 0
        goal_row = 0
        self.camera_label = set_role(QLabel(''), 'dim')
        self.camera_label.setWordWrap(True)
        self.camera_label.hide()
        self.merge_box = QCheckBox('')
        self.merge_box.hide()
        self.merge_box.toggled.connect(self._merge_toggled)
        for dest in _COMMON_DESTS:
            field = all_fields.get(dest)
            if field is not None:
                fv = FieldView(self, common, cgrid, row, field, None)
                self.fields[dest] = fv
                row += fv.rows_used
            if dest == 'temp_dir':
                goal_row, row = row, row + 1
            if dest == 'directory':
                cgrid.addWidget(self.camera_label, row, 0, 1, 5)
                row += 1
                cgrid.addWidget(self.merge_box, row, 0, 1, 5)
                row += 1
        self.goal = GoalPicker(self.model)
        cgrid.addWidget(self.goal, goal_row, 0, 1, 5)
        root.addWidget(common)

        # ── saved settings ──
        prow = QHBoxLayout()
        prow.setContentsMargins(0, 10, 0, 0)
        prow.addWidget(set_role(QLabel(_('My settings')), 'dim'))
        self.preset_combo = NoWheelComboBox()
        self.preset_combo.setMinimumWidth(180)
        self.preset_combo.activated.connect(lambda _i: self._load_preset())
        prow.addWidget(self.preset_combo)
        save = QPushButton(_('Save as…'))
        save.clicked.connect(self._save_preset_dialog)
        prow.addWidget(save)
        delete = QPushButton(_('Delete'))
        delete.clicked.connect(self._delete_preset)
        prow.addWidget(delete)
        prow.addStretch(1)
        root.addLayout(prow)
        self.refresh_presets()

        # ── additional options: toggle, search, only-changed ──
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 10, 0, 0)
        self.toggle_btn = QPushButton('')
        self.toggle_btn.clicked.connect(self.toggle_advanced)
        bar.addWidget(self.toggle_btn, 1)
        self.changed_only = QCheckBox(_('Only changed'))
        self.changed_only.toggled.connect(self._on_changed_only)
        bar.addWidget(self.changed_only)
        bar.addWidget(set_role(QLabel(_('Search settings')), 'dim'))
        self.search = QLineEdit()
        self.search.setMinimumWidth(160)
        self.search.textChanged.connect(self._on_search)
        bar.addWidget(self.search)
        self.bar = QWidget()
        self.bar.setLayout(bar)
        root.addWidget(self.bar)

        # ── the sections: a sidebar of group names beside the page of the current group ──
        self.advanced = QWidget()
        arow = QHBoxLayout(self.advanced)
        arow.setContentsMargins(0, 8, 0, 0)
        side = QVBoxLayout()
        self.nav = QListWidget()
        self.nav.setFixedWidth(190)
        self.nav.setWordWrap(True)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.currentRowChanged.connect(self._on_nav)
        side.addWidget(self.nav, 1)
        self.expert = QCheckBox(_('Expert options'))
        self.expert.toggled.connect(lambda _on: self.refresh_visibility())
        side.addWidget(self.expert)
        arow.addLayout(side)
        pages_host = QVBoxLayout()
        self.no_match = set_role(QLabel(_('No settings match.')), 'dim')
        pages_host.addWidget(self.no_match)
        self.pages: Dict[str, QWidget] = {}
        self._page_grids: Dict[str, QGridLayout] = {}
        # Hidden pages take no height (a QStackedWidget would size to the tallest one).
        for group in schema:
            page = QWidget()
            grid = QGridLayout(page)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setColumnStretch(2, 1)
            grid.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.pages[group] = page
            self._page_grids[group] = grid
            pages_host.addWidget(page)
            page.hide()
        pages_host.addStretch(1)
        arow.addLayout(pages_host, 1)
        root.addWidget(self.advanced)
        self.advanced.hide()

        rows = {g: 0 for g in schema}
        for group, fields in schema.items():
            for f in fields:
                dest = f['dest']
                if dest not in self.fields and self.model.fields[dest] is f:
                    fv = FieldView(self, self.pages[group], self._page_grids[group],
                                   rows[group], f, group)
                    self.fields[dest] = fv
                    rows[group] += fv.rows_used
        root.addStretch(1)

        self._rescan_timer = QTimer(self)
        self._rescan_timer.setSingleShot(True)
        self._rescan_timer.timeout.connect(self.model.rescan_directory)
        self.model.value_changed.connect(self._on_value)
        self.model.scanned.connect(self._on_scanned)
        self.refresh_toggle_text()
        self.refresh_visibility()

    def schedule_rescan(self) -> None:
        """Scan the folder once typing pauses."""
        self._rescan_timer.start(RESCAN_DEBOUNCE_MS)

    # ── model signals ───────────────────────────────────────────────────

    def _on_value(self, dest: str) -> None:
        fv = self.fields.get(dest)
        if fv is not None:
            fv.load()
        if dest == 'merge':
            self._refresh_merge_box()
        if self.changed_only.isChecked():
            self.refresh_visibility()
        self.refresh_toggle_text()

    def _merge_toggled(self, on: bool) -> None:
        self.model.set_merge_previous(on)

    def _refresh_merge_box(self) -> None:
        prev = self.model.previous_stacks
        self.merge_box.blockSignals(True)
        if prev:
            name = self.model.inferred[0] or ''
            frames = sum(s['frames'] for s in prev)
            self.merge_box.setText(_(
                'Add {n} earlier stack(s) of {name} ({frames} frames) to this one',
                n=len(prev), name=name, frames=frames))
            self.merge_box.setChecked(self.model.merge_previous())
        self.merge_box.setVisible(bool(prev))
        self.merge_box.blockSignals(False)

    def _on_scanned(self) -> None:
        self._refresh_merge_box()
        self.camera_label.setText(self.model.camera_note)
        self.camera_label.setVisible(bool(self.model.camera_note))
        fv = self.fields.get('directory')
        if fv is not None and fv.hint is not None:
            fv.hint.setText(self.model.count_text or getattr(fv, '_dir_summary', ''))

    # ── additional options: visibility, search ──────────────────────────

    def refresh_visibility(self) -> None:
        """Show or hide expert groups/fields (or, while searching, the matching fields).
        Hidden fields keep their values, so ``read_form`` is unaffected."""
        expert = self.expert.isChecked()
        terms = self.search.text().lower().split()
        only_changed = self.changed_only.isChecked()
        matches = {g: 0 for g in self.model.schema}
        for dest, fv in self.fields.items():
            if fv.group is None:
                continue
            if terms:
                visible = all(t in fv.haystack for t in terms)
            else:
                visible = expert or not fv.expert
            if only_changed:
                visible = visible and self.model.is_changed(dest)
            if visible:
                matches[fv.group] += 1
            fv.set_visible(visible)
        filtering = bool(terms) or only_changed
        if filtering:
            groups = [g for g in self.model.schema if matches[g]]
        else:
            groups = [g for g in self.model.schema if expert or not _is_expert_group(g)]
        self.nav.blockSignals(True)
        self.nav.clear()
        for g in groups:
            text = f'{_(g)}  ({matches[g]})' if filtering else _(g)
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, g)
            self.nav.addItem(item)
        self.nav.blockSignals(False)
        self.no_match.setVisible(not groups)
        if not groups:
            self._current_group = None
            for page in self.pages.values():
                page.hide()
            return
        self.show_group(self._current_group if self._current_group in groups else groups[0])

    def show_group(self, group: str) -> None:
        self._current_group = group
        for g, page in self.pages.items():
            page.setVisible(g == group)
        for i in range(self.nav.count()):
            if self.nav.item(i).data(Qt.ItemDataRole.UserRole) == group:
                self.nav.blockSignals(True)
                self.nav.setCurrentRow(i)
                self.nav.blockSignals(False)
                break

    def _on_nav(self, row: int) -> None:
        item = self.nav.item(row)
        if item is not None:
            self.show_group(item.data(Qt.ItemDataRole.UserRole))

    def _on_changed_only(self, on: bool) -> None:
        if on and not self._advanced_shown:
            self.toggle_advanced()
        if on:
            self.expert.setChecked(True)     # a changed expert option must not hide behind the toggle
        self.refresh_visibility()

    def _on_search(self, _text: str) -> None:
        if self.search.text().strip() and not self._advanced_shown:
            self.toggle_advanced()
        self.refresh_visibility()

    def toggle_advanced(self) -> None:
        self._advanced_shown = not self._advanced_shown
        self.advanced.setVisible(self._advanced_shown)
        self.refresh_toggle_text()
        if self._advanced_shown:
            self.expanded.emit(self.bar)

    def refresh_toggle_text(self) -> None:
        """'▸ Additional options · 2 changed': a change made behind the collapsed section stays
        visible from the main form."""
        n = self.model.changed_in_groups()
        arrow = '▾' if self._advanced_shown else '▸'
        suffix = '  ·  ' + _('{n} changed', n=n) if n else ''
        self.toggle_btn.setText(f"{arrow} {_('Additional options')}{suffix}")

    # ── saved settings ──────────────────────────────────────────────────

    def refresh_presets(self, select: str = '') -> None:
        self.preset_combo.clear()
        self.preset_combo.addItems(sorted(self.model.presets()))
        i = self.preset_combo.findText(select) if select else -1
        self.preset_combo.setCurrentIndex(i)

    def _save_preset_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, _('Save settings'), _('Name for these settings:'),
                                        text=self.preset_combo.currentText())
        if ok and self.model.save_preset(name):
            self.refresh_presets(select=name.strip())

    def _load_preset(self) -> None:
        state = self.model.presets().get(self.preset_combo.currentText())
        if state is not None:
            self.model.apply_preset(state)

    def _delete_preset(self) -> None:
        if self.model.delete_preset(self.preset_combo.currentText()):
            self.refresh_presets()

    # ── convenience ─────────────────────────────────────────────────────

    def field_choices(self, dest: str) -> List[str]:
        f = self.model.fields.get(dest)
        return list(f['choices'] or []) if f else []

