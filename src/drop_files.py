"""What a drop on the desktop window means (the Qt window receives the drop itself).

A folder of lights is what the app takes; dropping some of its files (or one light) names that
folder just as well.
"""
from __future__ import annotations

import os
from typing import List, Optional


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
