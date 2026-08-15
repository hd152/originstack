"""OriginStack desktop app -- native window over the local dashboard server.

Usage: python desktop_app.py

NOTE: This file is a thin entry-point shim; all implementation lives
      under src/desktop_app.py.
"""
from __future__ import annotations

import multiprocessing

from src.desktop_app import main

if __name__ == '__main__':
    # Required for a frozen (PyInstaller) build on Windows: Phase 1's
    # ProcessPoolExecutor uses the 'spawn' start method, which re-executes
    # this exact exe for every worker. freeze_support() must be the very
    # first thing that runs in __main__ so a spawned worker process detects
    # it's a multiprocessing child and jumps straight to the worker
    # bootstrap -- without it, every worker re-runs this file's own main()
    # from scratch, opening its own full GUI window (observed in the wild:
    # a 12-core machine opened 12 windows the moment Phase 1 started).
    multiprocessing.freeze_support()
    raise SystemExit(main())
