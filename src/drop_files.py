"""Drop files or folders on the desktop window (Windows).

tkinter has no drag-and-drop. Rather than add a dependency (tkdnd's binaries would have to be
bundled into the packaged app), this asks Windows for ``WM_DROPFILES`` on the window and subclasses
its window procedure to read them: ``DragAcceptFiles`` + ``DragQueryFileW`` from shell32, through
ctypes. Other platforms do nothing (``enable_file_drop`` returns False) and the Browse buttons
remain.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Callable, List, Optional

log = logging.getLogger("originstack")

_WM_DROPFILES = 0x0233
_WM_COPYGLOBALDATA = 0x0049        # sent by Explorer to a process at a lower integrity level
_MSGFLT_ALLOW = 1
_GWL_WNDPROC = -4

# Keeps the window-procedure thunk (and the original procedure) alive for the process: a
# collected callback would crash the next message.
_keep: dict = {}


def folder_from_drop(paths: List[str]) -> Optional[str]:
    """The folder a drop means: the first dropped folder, else the folder holding the first file.

    A folder of lights is what the app takes; dropping some of its files (or one light) names that
    folder just as well."""
    for p in paths:
        if os.path.isdir(p):
            return p
    for p in paths:
        if os.path.isfile(p):
            return os.path.dirname(p)
    return None


def enable_file_drop(root, on_drop: Callable[[List[str]], None]) -> bool:
    """Call ``on_drop(paths)`` (on the Tk thread, after the drop returns) for files dropped on
    ``root``. Returns whether drops are now accepted."""
    if sys.platform != 'win32':
        return False
    try:
        import ctypes
        from ctypes import wintypes
        root.update_idletasks()
        user32, shell32 = ctypes.windll.user32, ctypes.windll.shell32
        # Tk's visible top level is the parent of the widget window ``winfo_id`` names.
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()

        lresult = ctypes.c_ssize_t
        proc_t = ctypes.WINFUNCTYPE(lresult, wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                                    wintypes.LPARAM)
        user32.CallWindowProcW.restype = lresult
        user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT,
                                           wintypes.WPARAM, wintypes.LPARAM]
        user32.SetWindowLongPtrW.restype = ctypes.c_void_p
        user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
        shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, wintypes.UINT, wintypes.LPWSTR,
                                           wintypes.UINT]
        shell32.DragFinish.argtypes = [ctypes.c_void_p]

        def read_paths(hdrop) -> List[str]:
            n = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
            out = []
            for i in range(n):
                size = shell32.DragQueryFileW(hdrop, i, None, 0) + 1
                buf = ctypes.create_unicode_buffer(size)
                shell32.DragQueryFileW(hdrop, i, buf, size)
                out.append(buf.value)
            return out

        original = None

        def window_proc(h, msg, wparam, lparam):
            if msg == _WM_DROPFILES:
                try:
                    paths = read_paths(wparam)
                finally:
                    shell32.DragFinish(wparam)
                root.after_idle(lambda: on_drop(paths))
                return 0
            return user32.CallWindowProcW(original, h, msg, wparam, lparam)

        thunk = proc_t(window_proc)
        original = user32.SetWindowLongPtrW(hwnd, _GWL_WNDPROC, ctypes.cast(thunk, ctypes.c_void_p))
        if not original:
            return False
        _keep[hwnd] = (thunk, original)
        shell32.DragAcceptFiles(hwnd, True)
        try:    # an elevated app would otherwise ignore drops from a normal Explorer
            for msg in (_WM_DROPFILES, _WM_COPYGLOBALDATA):
                user32.ChangeWindowMessageFilterEx(hwnd, msg, _MSGFLT_ALLOW, None)
        except (AttributeError, OSError):
            pass
        return True
    except Exception as exc:        # a failed hook must never stop the app opening
        log.warning("file drop not available: %s", exc)
        return False
