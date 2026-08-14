"""Desktop app entry point (``python desktop_app.py``).

Starts the existing ``src/webview.py`` dashboard server (unchanged: same
``ThreadingHTTPServer``, same ``/events`` SSE stream every pipeline phase
already publishes into) and opens it in a native ``pywebview`` window instead
of a browser tab. The window's only extra capability over the plain browser
page is a native file/folder picker, exposed to the page's JS as
``window.pywebview.api.browse_directory()`` / ``browse_output_path()`` --
everything else (the Setup form, ``POST /api/start``, live progress) works
identically whether the page is opened here or in an ordinary browser.

Every failure path below routes through ``_fatal()``: a packaged PyInstaller
build runs windowed (no console), so a bare ``print()`` is invisible to a
double-click user -- it must show a native dialog and log to a location
that's writable regardless of install directory (Program Files is often
read-only for a non-admin install).
"""
from __future__ import annotations

import os
import datetime
import traceback
import webbrowser
from pathlib import Path


def _fatal(title: str, message: str) -> int:
    """Last-resort error surface: log full details, show a native dialog via
    tkinter (stdlib, works even if pywebview itself is what's broken)."""
    log_dir = Path(os.environ.get('LOCALAPPDATA', '.')) / 'OriginStack' / 'logs'
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / 'desktop_app_crash.log', 'a', encoding='utf-8') as f:
            f.write(f"\n[{datetime.datetime.now().isoformat()}] {title}\n{message}\n")
            f.write(traceback.format_exc() + '\n')
    except OSError:
        pass
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        pass  # tkinter itself failing is the true last resort -- nothing left to try
    return 1


class Api:
    """Exposed to the page as ``window.pywebview.api``. Native pickers are
    only meaningful inside the pywebview window -- the front-end JS hides
    every "Browse…" button entirely when ``window.pywebview`` is undefined,
    so these are never called from a plain-browser session."""

    def browse_directory(self) -> str:
        return self.browse_path('dir')

    def browse_output_path(self) -> str:
        import webview
        result = webview.windows[0].create_file_dialog(
            webview.SAVE_DIALOG, file_types=('FITS files (*.fits)', 'All files (*.*)'))
        return (result[0] if result else '') or ''

    def browse_path(self, kind: str) -> str:
        """Generic picker for the Advanced panel's auto-rendered path fields
        (``--cal-dir``, ``--astap-path``, ``--config``, etc, tagged via
        ``_WIDGET_HINTS`` in src/webview_control.py::get_form_schema()).
        *kind* is one of the ``widget`` values that schema already produces:
        'dir' -> folder picker, 'file-save' -> save dialog, anything else
        ('file-open' or unset) -> open-file dialog."""
        import webview
        if kind == 'dir':
            result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        elif kind == 'file-save':
            result = webview.windows[0].create_file_dialog(webview.SAVE_DIALOG)
        else:
            result = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG)
        return (result[0] if result else '') or ''


def main() -> int:
    from src.webview import get_webview
    from src.webview_control import get_run_manager

    wv = get_webview()
    url = wv.start(port=8765)
    if url is None:
        return _fatal("OriginStack",
                      "Could not start the local dashboard server "
                      "(port 8765 in use?).")

    try:
        import webview
    except ImportError:
        webbrowser.open(url)
        print(f"pywebview is not installed (pip install pywebview). "
              f"The dashboard server is running at {url} -- opened in your "
              f"browser instead. Ctrl+C to stop.")
        # The HTTP server runs on a daemon thread (WebView.start()) -- it
        # dies the instant this process's main thread exits, which would
        # happen immediately after printing the URL above with nothing
        # keeping it alive. Block here instead, same as --live's fallback
        # in cli.py's main(), so the dashboard stays reachable.
        import time
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        wv.stop()
        return 0

    rm = get_run_manager()

    def on_closing() -> bool:
        """Returning False here cancels the close (pywebview's Event.set()
        treats any handler returning False as should_cancel=True)."""
        if not rm.is_running():
            return True
        try:
            import tkinter
            from tkinter import messagebox
            root = tkinter.Tk()
            root.withdraw()
            ok = messagebox.askyesno(
                "OriginStack", "A stacking run is still in progress. Quit anyway?")
            root.destroy()
            return bool(ok)
        except Exception:
            return True  # can't ask -- don't trap the user in an unclosable window

    try:
        window = webview.create_window('OriginStack', url, js_api=Api(),
                                       width=1400, height=900, min_size=(900, 600))
        window.events.closing += on_closing
        webview.start()
    except Exception as e:
        return _fatal("OriginStack",
                      f"Failed to open the app window: {e}\n\n"
                      f"This usually means the Microsoft Edge WebView2 "
                      f"Runtime is not installed.")
    finally:
        wv.stop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
