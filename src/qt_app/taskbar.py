"""Progress on the Windows taskbar button (``ITaskbarList3``), through ctypes.

Qt 6 dropped ``QWinTaskbarButton`` with the Qt Windows Extras module, and a COM wrapper package
would be one more dependency for the packaged app, so this calls the two methods it needs
straight off the COM vtable (the same choice ``src/notify.py`` and ``src/drop_files.py`` made for
their shell32 calls). Everything is best-effort: off Windows, or if COM refuses, every call is a
silent no-op, so a run never depends on it.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from typing import Optional

log = logging.getLogger("originstack")

NOPROGRESS = 0
INDETERMINATE = 1
NORMAL = 2
ERROR = 4
PAUSED = 8

# vtable slots: IUnknown 0-2, ITaskbarList 3-7 (HrInit first), ITaskbarList2 8, ITaskbarList3 9...
_HR_INIT = 3
_SET_PROGRESS_VALUE = 9
_SET_PROGRESS_STATE = 10

_CLSID_TASKBAR_LIST = '{56FDF344-FD6D-11D0-958A-006097C9A090}'
_IID_TASKBAR_LIST3 = '{EA1AFB91-9E28-4B86-90E9-9E9F8A5EEFAF}'


class _GUID(ctypes.Structure):
    _fields_ = [('Data1', ctypes.c_ulong), ('Data2', ctypes.c_ushort),
                ('Data3', ctypes.c_ushort), ('Data4', ctypes.c_ubyte * 8)]


def _guid(text: str) -> '_GUID':
    g = _GUID()
    ctypes.oledll.ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(g))
    return g


class TaskbarProgress:
    """Set the progress bar drawn on a window's taskbar button.

    ``available`` is False off Windows or when the COM object could not be made. The state is
    remembered, so polling every 150 ms with an unchanged value makes no COM call."""

    def __init__(self, hwnd: Optional[int] = None):
        self.hwnd = hwnd
        self.available = False
        self._ptr = None
        self._vtable = None
        self._state: Optional[int] = None
        self._value: Optional[tuple] = None
        if sys.platform == 'win32':
            self._create()

    def _create(self) -> None:
        try:
            ole32 = ctypes.oledll.ole32
            ole32.CoInitialize(None)         # S_FALSE when the thread already has COM: fine
            clsid, iid = _guid(_CLSID_TASKBAR_LIST), _guid(_IID_TASKBAR_LIST3)
            ptr = ctypes.c_void_p()
            ole32.CoCreateInstance(ctypes.byref(clsid), None, 1, ctypes.byref(iid), ctypes.byref(ptr))
            # The object's first word points at its vtable of function pointers.
            vtable = ctypes.cast(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0],
                                 ctypes.POINTER(ctypes.c_void_p))
            hr_init = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)(vtable[_HR_INIT])
            if hr_init(ptr) != 0:
                return
            self._ptr, self._vtable = ptr, vtable
            self.available = True
        except Exception as exc:             # no COM, no shell: the taskbar just stays plain
            log.debug('taskbar progress unavailable: %s', exc)

    def set_window(self, hwnd: int) -> None:
        if hwnd != self.hwnd:
            self.hwnd, self._state, self._value = hwnd, None, None

    def _call(self, slot: int, argtypes: tuple, *args) -> None:
        proto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, *argtypes)
        proto(self._vtable[slot])(self._ptr, ctypes.c_void_p(self.hwnd), *args)

    def set_state(self, state: int) -> None:
        if not self.available or not self.hwnd or state == self._state:
            return
        try:
            self._call(_SET_PROGRESS_STATE, (ctypes.c_int,), state)
            self._state = state
        except Exception as exc:
            log.debug('taskbar state failed: %s', exc)

    def set_value(self, done: int, total: int) -> None:
        """A determinate bar at done/total (a total of 0 shows the pulsing bar instead)."""
        if total <= 0:
            self.set_state(INDETERMINATE)
            return
        done = max(0, min(int(done), int(total)))
        self.set_state(NORMAL)
        if not self.available or not self.hwnd or (done, total) == self._value:
            return
        try:
            self._call(_SET_PROGRESS_VALUE, (ctypes.c_ulonglong, ctypes.c_ulonglong),
                       ctypes.c_ulonglong(done), ctypes.c_ulonglong(total))
            self._value = (done, total)
        except Exception as exc:
            log.debug('taskbar value failed: %s', exc)

    def clear(self) -> None:
        self.set_state(NOPROGRESS)
        self._value = None


def idle_state(run_status: str) -> int:
    """The taskbar state once no run is going: red after a failure, plain otherwise."""
    return ERROR if run_status == 'error' else NOPROGRESS
