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
    the "Browse…" buttons entirely when ``window.pywebview`` is undefined,
    so these are never called from a plain-browser session."""

    def browse_directory(self) -> str:
        import webview
        result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        return (result[0] if result else '') or ''

    def browse_output_path(self) -> str:
        import webview
        result = webview.windows[0].create_file_dialog(
            webview.SAVE_DIALOG, file_types=('FITS files (*.fits)', 'All files (*.*)'))
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
              f"browser instead.")
        return 1

    webview.create_window('OriginStack', url, js_api=Api(),
                          width=1400, height=900, min_size=(900, 600))
    webview.start()
    wv.stop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
