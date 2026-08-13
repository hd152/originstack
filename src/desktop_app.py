"""Desktop app entry point (``python desktop_app.py``).

Starts the existing ``src/webview.py`` dashboard server (unchanged: same
``ThreadingHTTPServer``, same ``/events`` SSE stream every pipeline phase
already publishes into) and opens it in a native ``pywebview`` window instead
of a browser tab. The window's only extra capability over the plain browser
page is a native file/folder picker, exposed to the page's JS as
``window.pywebview.api.browse_directory()`` / ``browse_output_path()`` --
everything else (the Setup form, ``POST /api/start``, live progress) works
identically whether the page is opened here or in an ordinary browser.
"""
from __future__ import annotations


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

    wv = get_webview()
    url = wv.start(port=8765)
    if url is None:
        print("ERROR: could not start the local dashboard server "
              "(port 8765 in use?)")
        return 1

    try:
        import webview
    except ImportError:
        print(f"pywebview is not installed (pip install pywebview). "
              f"The dashboard server is running at {url} -- open it in a "
              f"browser instead. Ctrl+C to stop.")
        # The HTTP server runs on a daemon thread (WebView.start()) -- it
        # dies the instant this process's main thread exits, which would
        # happen immediately after printing the URL above with nothing
        # keeping it alive. Block here instead, same as --live's fallback
        # in cli.py's main(), so the just-printed URL stays reachable.
        import time
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        wv.stop()
        return 0

    webview.create_window('OriginStack', url, js_api=Api(),
                          width=1400, height=900, min_size=(900, 600))
    webview.start()
    wv.stop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
