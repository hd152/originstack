"""Find earlier linear stacks of the same target, for "add them to tonight's stack".

Toolkit-free (the desktop form and tests both use it). A stack qualifies when it is a main
output FITS of an earlier run (``RAWSTACK=True``, what ``--merge`` accepts) whose ``OBJECT``
matches the target name after normalising case, spaces and punctuation.
"""
import logging
import os
import re
from typing import Dict, Iterable, List, Optional

_log = logging.getLogger("originstack")

_MAX_CANDIDATES = 200


def _norm(name: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _candidate_files(dirs: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for n in names:
            low = n.lower()
            if "_stacked" in low and low.endswith((".fits", ".fit")):
                p = os.path.abspath(os.path.join(d, n))
                if p not in seen:
                    seen.add(p)
                    out.append(p)
    return out[:_MAX_CANDIDATES]


def find_previous_stacks(target: Optional[str], dirs: Iterable[str],
                         exclude: Iterable[str] = ()) -> List[Dict]:
    """Earlier stacks of ``target`` in ``dirs`` (not recursive), oldest first.

    Each item: ``{'path', 'frames', 'date'}``. ``exclude`` holds paths to skip (the output
    this run is about to write). Header-only reads; unreadable files are skipped.
    """
    want = _norm(target)
    if not want:
        return []
    skip = {os.path.abspath(p) for p in exclude if p}
    try:
        from astropy.io import fits
    except Exception:       # astropy is a hard dependency; guard anyway for odd installs
        return []
    found = []
    for path in _candidate_files(dirs):
        if path in skip:
            continue
        try:
            hdr = fits.getheader(path)
        except Exception:
            _log.debug("find_previous_stacks: unreadable %s", path, exc_info=True)
            continue
        if not hdr.get("RAWSTACK", False) or _norm(hdr.get("OBJECT")) != want:
            continue
        found.append({"path": path, "frames": int(hdr.get("NFRAMES", 0) or 0),
                      "date": str(hdr.get("DATE-OBS", "") or "")})
    found.sort(key=lambda s: (s["date"], s["path"]))
    return found
