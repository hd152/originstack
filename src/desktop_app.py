"""Desktop app entry point (``python desktop_app.py``): starts the Qt window in ``src/qt_app``.

The window (a Setup form built from ``cli.build_parser()``, a live progress/log panel and the
interactive preview) lives in ``src/qt_app``; the schema, run control and event sink it shares
with the pipeline are ``src/desktop_common.py``, ``src/desktop_control.py`` and
``src/ui_events.py``. The old tkinter window (``--tk``) is gone.

Every failure path routes through ``_fatal()``: a packaged PyInstaller build runs windowed (no
console), so a bare ``print()`` is invisible to a double-click user -- it must show a native
dialog and log to a location that is writable regardless of install directory.
"""
from __future__ import annotations

import sys

from src.desktop_startup import _fatal
from src.i18n import _


def main() -> int:
    try:
        from src.qt_app import main as qt_main
    except ImportError as exc:                   # PySide6 missing from a source checkout
        return _fatal('OriginStack', _('The Qt interface is not installed: {error}', error=exc))
    return qt_main(sys.argv)


if __name__ == '__main__':
    raise SystemExit(main())
