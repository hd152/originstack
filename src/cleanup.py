"""Global temporary-file registry: auto-remove all registered paths on exit or interrupt."""
from __future__ import annotations

import atexit
import gc
import os
import shutil
import threading
from typing import List

_lock: threading.Lock = threading.Lock()
_paths: List[str] = []


def register(path: str) -> None:
    """Register *path* for deletion when the process exits (including Ctrl+C)."""
    with _lock:
        if path not in _paths:
            _paths.append(path)


def deregister(path: str) -> None:
    """Remove *path* from the cleanup registry after a successful manual deletion."""
    with _lock:
        try:
            _paths.remove(path)
        except ValueError:
            pass


def _do_cleanup() -> None:
    """atexit handler: delete every registered path still on disk."""
    gc.collect()
    with _lock:
        remaining = list(_paths)
        _paths.clear()
    for p in remaining:
        try:
            if os.path.isfile(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


atexit.register(_do_cleanup)
