"""OriginStack desktop app -- native window over the local dashboard server.

Usage: python desktop_app.py

NOTE: This file is a thin entry-point shim; all implementation lives
      under src/desktop_app.py.
"""
from __future__ import annotations

from src.desktop_app import main

if __name__ == '__main__':
    raise SystemExit(main())
