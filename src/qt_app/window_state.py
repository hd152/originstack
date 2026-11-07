"""Remember the window between runs: size and position, splitters, the tool tab, whether the
tools are collapsed, and the recent folders. Stored in the app's settings file under
``qt_window``; a missing or damaged entry just means the defaults."""
from __future__ import annotations

import base64
from typing import Any, Dict, List

from PySide6.QtCore import QByteArray

MAX_RECENT = 8


def encode(data: QByteArray) -> str:
    return base64.b64encode(bytes(data)).decode('ascii')


def decode(text: Any) -> QByteArray:
    try:
        return QByteArray(base64.b64decode(str(text)))
    except Exception:
        return QByteArray()


def load() -> Dict[str, Any]:
    from src.desktop_control import load_app_setting
    data = load_app_setting('qt_window', {})
    return data if isinstance(data, dict) else {}


def save(state: Dict[str, Any]) -> None:
    from src.desktop_control import save_app_setting
    save_app_setting('qt_window', state)


def recent_folders() -> List[str]:
    from src.desktop_control import load_app_setting
    data = load_app_setting('recent_folders', [])
    return [d for d in data if isinstance(d, str)] if isinstance(data, list) else []


def remember_folder(path: str) -> List[str]:
    """Put ``path`` first in the recent folders (each once, newest first, at most MAX_RECENT)."""
    from src.desktop_control import save_app_setting
    path = str(path).strip()
    items = [d for d in recent_folders() if d != path]
    if path:
        items.insert(0, path)
    items = items[:MAX_RECENT]
    save_app_setting('recent_folders', items)
    return items
