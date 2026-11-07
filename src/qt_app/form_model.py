"""The Setup form's state, without any widgets.

``FormModel`` is built from ``desktop_control.get_form_schema()`` (one entry per CLI flag) and
holds one value per ``dest``. It knows which values differ from their defaults, which are
submitted to the run (``read_form``), the folder scan (frame counts, inferred target, camera
note), saved presets and the "This run" description. The widgets in ``setup_form`` only show
and edit it, so everything here is testable without a display.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Signal

from src.desktop_common import (
    _COMMON_DESTS,
    _NO_CHANGE_MARK,
    _PICKER_DESTS,
    _RUN_MODES,
    _TARGET_CARDS,
    _TYPE_TO_CARD,
    _field_label_ui,
    _frame_count,
    _from_source,
    _target_label,
)
from src.i18n import _


class FormModel(QObject):
    """Values of every Setup field, keyed by the argparse ``dest``."""

    #: a value changed (dest)
    value_changed = Signal(str)
    #: the folder scan finished: frame counts, inferred target or camera note may differ
    scanned = Signal()

    def __init__(self, schema: Optional[Dict[str, List[Dict[str, Any]]]] = None, parent=None):
        super().__init__(parent)
        if schema is None:
            from src.desktop_control import get_form_schema
            schema = get_form_schema()
        self.schema = schema
        # One field per dest: a dest two flags share (--cosmic-ray-rejection and its --no-
        # form) takes the field of its last action, which is what build_argv_from_form resolves.
        self.fields: Dict[str, Dict[str, Any]] = {}
        self.group_of: Dict[str, str] = {}
        for group, fields in schema.items():
            for f in fields:
                self.fields[f['dest']] = f
                self.group_of[f['dest']] = group
        self.values: Dict[str, Any] = {}
        # What each value started as. ``read_form`` submits only a dest that differs: submitting
        # every field would mark them all "explicit" and stop --auto tuning them.
        self.initial: Dict[str, Any] = {}
        self.path_dests: set = set()
        for dest, f in self.fields.items():
            v = self._default(f)
            self.values[dest] = v
            self.initial[dest] = v
            if f.get('widget'):
                self.path_dests.add(dest)

        self.data_summary = ''
        self.frame_counts: Dict[str, int] = {}
        # (name, type, source) the session/header/folder suggest -- local lookups only.
        self.inferred: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None)
        self.camera_note = ''
        self.count_text = ''          # shown under the folder field
        self.suggested: Optional[str] = None       # the target card the folder suggests

    @staticmethod
    def _default(field: Dict[str, Any]) -> Any:
        kind, default = field['kind'], field['default']
        if kind in ('bool_true', 'bool_false'):
            return bool(default)
        if isinstance(default, list):
            return ', '.join(str(d) for d in default)
        return '' if default is None else str(default)

    # ── values ──────────────────────────────────────────────────────────

    def get(self, dest: str, default: Any = '') -> Any:
        return self.values.get(dest, default)

    def set(self, dest: str, value: Any, rescan: bool = True) -> None:
        """Set a value. ``rescan=False`` for a folder being typed: the caller rescans once the
        typing pauses (every partial path that exists would otherwise be scanned)."""
        if dest not in self.values:
            return
        if isinstance(self.initial[dest], bool):
            value = bool(value)
        elif value is None:
            value = ''
        else:
            value = str(value)
        if self.values[dest] == value:
            return
        self.values[dest] = value
        self.value_changed.emit(dest)
        if dest == 'directory' and rescan:
            self.rescan_directory()

    def is_changed(self, dest: str) -> bool:
        return self.values[dest] != self.initial.get(dest)

    def reset(self, dest: str) -> None:
        self.set(dest, self.initial[dest])

    def read_form(self) -> Dict[str, Any]:
        """``{dest: value}`` of the fields changed from their defaults (see ``initial``)."""
        return {d: v for d, v in self.values.items() if v != self.initial.get(d)}

    def changed_settings(self) -> List[str]:
        """Labels of the settings changed from their defaults, not counting the folders or what
        the cards and run mode set."""
        return [_field_label_ui(self.fields[d]) for d in self.fields
                if d not in _PICKER_DESTS and d not in _NO_CHANGE_MARK and self.is_changed(d)]

    def changed_in_groups(self) -> int:
        """How many settings behind "Additional options" differ from their defaults."""
        return sum(1 for d in self.fields
                   if d not in _COMMON_DESTS and d not in _PICKER_DESTS and self.is_changed(d))

    # ── target cards and run mode (views over target_type / comet_mode / preset) ─────

    def selected_target(self) -> Optional[str]:
        """The card matching the fields, or None for a type no card shows."""
        if self.get('comet_mode'):
            return 'comet'
        tt = self.get('target_type')
        for card in _TARGET_CARDS:
            if card[3] == tt and not card[4]:
                return card[0]
        return None

    def selected_mode(self) -> Optional[str]:
        preset = self.get('preset')
        return next((m[0] for m in _RUN_MODES if m[3] == preset), None)

    def pick_target(self, key: str) -> None:
        card = next(c for c in _TARGET_CARDS if c[0] == key)
        self.set('target_type', card[3])
        self.set('comet_mode', card[4])

    def pick_mode(self, key: str) -> None:
        self.set('preset', next(m[3] for m in _RUN_MODES if m[0] == key))

    def example(self):
        """The example image path and caption for the selected card (the suggested one under
        Auto-detect), or (None, '')."""
        from src.desktop_common import _EXAMPLES_DIR
        key = self.selected_target()
        if key == 'auto' and self.suggested:
            key = self.suggested
        card = next((c for c in _TARGET_CARDS if c[0] == key), None)
        if card and card[5] and (_EXAMPLES_DIR / card[5]).exists():
            return _EXAMPLES_DIR / card[5], _(card[6])
        return None, ''

    # ── the folder ──────────────────────────────────────────────────────

    def rescan_directory(self) -> None:
        directory = str(self.get('directory')).strip()
        self.data_summary = ''
        self.frame_counts = {}
        self.inferred = (None, None, None)
        self.camera_note = ''
        self.count_text = ''
        self.suggested = None
        if directory and os.path.isdir(directory):
            try:
                self._scan(directory)
            except Exception:
                self.count_text = ''
        self.suggested = _TYPE_TO_CARD.get(self.inferred[1] or '')
        self.scanned.emit()

    def _scan(self, directory: str) -> None:
        from src.frame_discovery import discover_frames
        found = discover_frames(directory)
        counts = {k: len(v) for k, v in found.items()}
        lights = list(found.get('light', []))
        if sum(counts.values()) == 0:
            for d in os.listdir(directory):
                sub = os.path.join(directory, d)
                if os.path.isdir(sub):
                    for k, v in discover_frames(sub).items():
                        counts[k] = counts.get(k, 0) + len(v)
        parts = ', '.join(_frame_count(k, v) for k, v in counts.items() if v)
        self.data_summary = parts
        self.frame_counts = counts
        text = parts or _('no frames found')
        if lights:
            # The same local lookups the run starts with (info.json name, FITS OBJECT, folder
            # name), but never SIMBAD from here.
            from src.desktop_control import unprofiled_camera
            from src.session_info import load_session_info
            from src.target_inference import infer_target_from_metadata
            si = load_session_info(directory)
            name, ttype, _conf, src = infer_target_from_metadata(
                directory, lights, use_simbad=False,
                session_name=si.object_name if si else None)
            if ttype == 'unknown':
                ttype = None
            self.inferred = (name, ttype, src)
            cam_name = unprofiled_camera(lights)
            if cam_name:
                self.camera_note = _(
                    'Camera: {name}. The learned sharpening was trained on Origin data: it '
                    'runs after the star widths are rescaled, and --auto picks it only when '
                    'stars can be measured.', name=cam_name)
            if name:
                text += f'  ·  {name}' + (f', {_from_source(src)}' if _from_source(src) else '')
        self.count_text = text

    def auto_card_text(self) -> str:
        """The Auto-detect card's description: what it will start from, once known."""
        _name, ttype = self.inferred[0], self.inferred[1]
        if self.suggested and ttype:
            label = _target_label(ttype)
            return (_('Will start from: {label} ({name}).', label=label, name=_name) if _name
                    else _('Will start from: {label}.', label=label))
        return _(_TARGET_CARDS[0][2])

    # ── saved settings ("My settings") ──────────────────────────────────

    @staticmethod
    def presets() -> Dict[str, Dict[str, Any]]:
        from src.desktop_control import load_app_setting
        data = load_app_setting('setup_presets', {})
        return data if isinstance(data, dict) else {}

    def save_preset(self, name: str) -> bool:
        from src.desktop_control import save_app_setting
        name = name.strip()
        if not name:
            return False
        presets = self.presets()
        presets[name] = self.preset_state()
        save_app_setting('setup_presets', presets)
        return True

    def delete_preset(self, name: str) -> bool:
        from src.desktop_control import save_app_setting
        presets = self.presets()
        if presets.pop(name, None) is None:
            return False
        save_app_setting('setup_presets', presets)
        return True

    def preset_state(self) -> Dict[str, Any]:
        """The settings as a saved preset: what differs from the defaults, without the folders
        and files, which belong to one session."""
        return {d: v for d, v in self.read_form().items() if d not in self.path_dests}

    def apply_preset(self, state: Dict[str, Any]) -> None:
        """Put a saved preset on the form: every setting not in it returns to its default (a
        preset is the whole of what it describes, not a patch on the previous one); the
        folders stay as they are."""
        for dest in list(self.values):
            if dest in self.path_dests:
                continue
            self.set(dest, state.get(dest, self.initial.get(dest)))

    # ── remembered folders ──────────────────────────────────────────────

    def remember_folders(self) -> None:
        from src.desktop_control import save_app_setting
        for dest in ('directory', 'output'):
            value = str(self.get(dest)).strip()
            if dest == 'output' and value.lower().endswith(('.fits', '.fit')):
                value = os.path.dirname(value)    # the folder, so the next stack never overwrites
            if value:
                save_app_setting(f'last_{dest}', value)

    def restore_folders(self) -> None:
        """Put back the folders of the last run, where nothing has been typed yet."""
        from src.desktop_control import load_app_setting
        for dest in ('directory', 'output'):
            saved = load_app_setting(f'last_{dest}', '')
            if dest in self.values and not self.get(dest) and isinstance(saved, str) and saved \
                    and os.path.isdir(saved):
                self.set(dest, saved)

    # ── "This run" ──────────────────────────────────────────────────────

    def describe_run(self) -> List[Tuple[str, str]]:
        """(heading, text) rows describing what Start will do with the current form."""
        v = self.values
        directory = str(v.get('directory') or '').strip()
        rows = [(_('Frames'), self.data_summary or
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

        mode = self.selected_mode()
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
