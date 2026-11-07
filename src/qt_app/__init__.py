"""OriginStack desktop app, Qt (PySide6) edition.

``main()`` is what ``desktop_app.py`` runs. The pipeline, ``RunManager`` and ``UIEvents`` are the
same ones the tkinter window used; this package is only the window.
"""
from __future__ import annotations

import sys

from src.desktop_startup import _fatal, _log_startup_status, _run_headless, _shutdown_and_exit
from src.i18n import _


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    _log_startup_status()
    if len(argv) > 1 and argv[1] == '--verify-headless':
        return _run_headless(argv[2:])

    from src.i18n import resolve_language, set_language
    language = set_language(resolve_language())
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        return _fatal('OriginStack', _('The Qt interface is not installed: {error}', error=exc))
    from src.qt_app import theme
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName('OriginStack')
    from src.desktop_control import load_app_setting
    theme.apply_mode(theme.resolve_mode(str(load_app_setting('theme', 'auto'))))
    app.setStyleSheet(theme.stylesheet(theme.sans_for(language)))
    try:
        from src.qt_app.main_window import MainWindow
        window = MainWindow()
        window.show()
    except Exception as e:
        return _fatal('OriginStack', _('Failed to open the app window: {error}', error=e))
    try:
        app.exec()
    except Exception as e:
        return _fatal('OriginStack', _('Unexpected error: {error}', error=e))
    _shutdown_and_exit(0)
    return 0
