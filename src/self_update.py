"""Install a new version from inside the desktop app (Windows installed copy).

The update check (``net_query.check_for_update``) already says a newer release exists. For the
installed app this downloads that release's ``*-setup.exe`` (``net_query.download_update``: GitHub's
own host only, exact size and SHA-256 from the release), then starts the installer and lets the app
close. The installer closes any copy still running, replaces it in place (same AppId) and offers to
launch the new one. Portable, zip, Mac and Linux copies cannot replace themselves: they keep the
"open the release page" behaviour.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional


def can_install_in_app(info: Optional[Dict[str, object]] = None) -> bool:
    """True for an *installed* Windows copy (the Inno uninstaller sits beside the exe) when the
    release has a verifiable installer."""
    if sys.platform != 'win32' or not getattr(sys, 'frozen', False):
        return False
    if not (Path(sys.executable).parent / 'unins000.exe').exists():
        return False
    return info is None or all(info.get(k) for k in ('installer_url', 'installer_size', 'installer_sha256'))


def installer_path(info: Dict[str, object]) -> Path:
    """Where the installer is saved: a fixed per-user temp folder, the file name from the release
    (reduced to its base name, so a crafted name cannot leave the folder)."""
    name = os.path.basename(str(info['installer_name'])) or 'OriginStack-setup.exe'
    return Path(tempfile.gettempdir()) / 'OriginStack-update' / name


def launch_installer(path: Path) -> None:
    """Start the installer detached; the caller then closes the app."""
    flags = 0
    if sys.platform == 'win32':
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(path)], close_fds=True, creationflags=flags)  # nosec B603
