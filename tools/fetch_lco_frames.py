"""Public calibrated frames from the Las Cumbres Observatory archive, for testing the learned
deconvolver (and anything else) on cameras, pixel scales and PSFs other than Origin's.

LCO runs 0.4 m (``0m4``), 1 m (``1m0``) and 2 m (``2m0``) telescopes at several sites, so one
target is available at pixel scales from ~0.3" to ~0.8" and at very different seeing. The
frames are BANZAI-reduced (``reduction_level`` 91: bias/dark/flat calibrated, WCS solved) mono
images in one filter; ``PIXSCALE`` and ``L1FWHM`` (arcsec) in each header give the grid and
the PSF the pipeline measured.

Only frames whose ``public_date`` has passed are returned (``public=true``); no account or
token is needed. Frames are stored as downloaded (``.fits.fz``; astropy reads them). Use the
data for evaluation and credit LCO (https://lco.global/documentation/data/data-use-policy/);
do not redistribute or train on it without checking the proposal's terms.

One run picks the most-populated (instrument, filter) group among the matches, so the frames
share a camera, pixel grid and filter, and downloads up to ``--n`` of them, preferring the
nights with the most frames (a stack wants a stable field). ``manifest.json`` in the output
folder lists what was fetched.

All HTTP goes through ``net_query._http_get`` (the --offline guard).

Usage:
    python tools/fetch_lco_frames.py --target M13 --telescope 1m0 --filter rp --out C:/nn_work/lco/m13_1m0 --n 30
    python tools/fetch_lco_frames.py --target M51 --telescope 0m4 --list      # what exists, download nothing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional
from urllib.parse import urlencode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

API = "https://archive-api.lco.global/frames/"
PAGE = 100
MAX_PAGES = 30            # 3000 candidate frames is plenty to find a group


def _get(url: str, timeout: float = 300.0) -> bytes:
    from src.net_query import _http_get
    for attempt in range(4):
        try:
            return _http_get(url, timeout=timeout)
        except Exception as exc:            # dropped connections: retry with back-off
            if attempt == 3:
                raise
            print(f"  retrying ({exc.__class__.__name__})", flush=True)
            time.sleep(10 * (attempt + 1))


def query_url(target: str, min_exp: float, max_exp: float, start: Optional[str], end: Optional[str]) -> str:
    q = {"public": "true", "reduction_level": 91, "configuration_type": "EXPOSE",
         "target_name": target, "exposure_time_min": min_exp, "exposure_time_max": max_exp,
         "limit": PAGE, "ordering": "-observation_date"}
    if start:
        q["start"] = start
    if end:
        q["end"] = end
    return f"{API}?{urlencode(q)}"


def search(target: str, min_exp: float, max_exp: float, start: Optional[str] = None,
           end: Optional[str] = None) -> List[dict]:
    """Every public reduced frame of ``target`` in the exposure range (follows the API's pages)."""
    url, out = query_url(target, min_exp, max_exp, start, end), []
    for _ in range(MAX_PAGES):
        page = json.loads(_get(url, 120))
        out.extend(page.get("results", []))
        url = page.get("next")
        if not url:
            break
    return out


def select(frames: List[dict], telescope: Optional[str], filt: Optional[str], n: int) -> List[dict]:
    """Up to ``n`` frames of the most-populated (instrument, filter) group.

    ``telescope`` is a class prefix (``1m0``) of ``telescope_id`` (``1m0a``); ``filt`` is matched
    exactly against the optical element (``rp``). Within the group, nights with the most frames
    come first and a night is taken whole before the next, so the field stays put."""
    keep = [f for f in frames
            if (not telescope or str(f.get("telescope_id", "")).startswith(telescope))
            and (not filt or f.get("primary_optical_element") == filt)]
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for f in keep:
        groups[(f.get("instrument_id"), f.get("primary_optical_element"))].append(f)
    if not groups:
        return []
    best = max(groups.values(), key=len)
    nights = Counter(f.get("observation_day") for f in best)
    best.sort(key=lambda f: (-nights[f.get("observation_day")], f.get("observation_day") or "",
                             f.get("observation_date") or ""))
    return best[:n]


def describe(frames: List[dict]) -> str:
    c = Counter((f.get("telescope_id"), f.get("instrument_id"), f.get("primary_optical_element"))
                for f in frames)
    lines = [f"  {t or '?':6s} {i or '?':8s} {o or '?':6s} {k:5d} frames" for (t, i, o), k in sorted(c.items())]
    return "\n".join(lines) if lines else "  (none)"


def header_summary(path: str) -> dict:
    from astropy.io import fits
    with fits.open(path) as h:
        hd = next((x.header for x in h if x.data is not None), h[0].header)
        return {k: hd.get(k) for k in ("INSTRUME", "TELESCOP", "FILTER", "EXPTIME", "PIXSCALE",
                                       "L1FWHM", "SATURATE", "GAIN", "RDNOISE", "NAXIS1", "NAXIS2")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", required=True, help="LCO target name, e.g. M13, M51, NGC7000")
    ap.add_argument("--telescope", help="telescope class prefix: 0m4, 1m0 or 2m0")
    ap.add_argument("--filter", help="optical element, e.g. gp, rp, ip, V (default: the most common)")
    ap.add_argument("--n", type=int, default=30, help="frames to download (default 30)")
    ap.add_argument("--min-exp", type=float, default=10.0)
    ap.add_argument("--max-exp", type=float, default=600.0)
    ap.add_argument("--start", help="earliest observation date, YYYY-MM-DD")
    ap.add_argument("--end", help="latest observation date, YYYY-MM-DD")
    ap.add_argument("--out", help="output folder (required unless --list)")
    ap.add_argument("--list", action="store_true", help="show what is available and exit")
    a = ap.parse_args(argv)
    if not a.list and not a.out:
        ap.error("--out is required unless --list")

    print(f"Searching LCO for {a.target} ...", flush=True)
    found = search(a.target, a.min_exp, a.max_exp, a.start, a.end)
    print(f"{len(found)} public reduced frames:\n{describe(found)}")
    if a.list:
        return 0
    pick = select(found, a.telescope, a.filter, a.n)
    if not pick:
        print("No frames match --telescope/--filter.")
        return 1

    os.makedirs(a.out, exist_ok=True)
    manifest, t0 = [], time.time()
    print(f"Downloading {len(pick)} frames ({pick[0].get('telescope_id')} {pick[0].get('instrument_id')} "
          f"{pick[0].get('primary_optical_element')}) to {a.out}", flush=True)
    for k, f in enumerate(pick, 1):
        dest = os.path.join(a.out, f["filename"])
        if not os.path.exists(dest):
            tmp = dest + ".part"
            with open(tmp, "wb") as fh:                      # whole file is held in memory by _http_get
                fh.write(_get(f["url"]))
            os.replace(tmp, dest)
        entry = {"file": f["filename"], "observation_date": f.get("observation_date"),
                 "telescope_id": f.get("telescope_id"), "target": f.get("target_name"),
                 "proposal_id": f.get("proposal_id")}
        entry.update(header_summary(dest))
        manifest.append(entry)
        print(f"  [{k}/{len(pick)}] {f['filename']}  pixscale {entry.get('PIXSCALE')}  "
              f"L1FWHM {entry.get('L1FWHM')}", flush=True)
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump({"source": "Las Cumbres Observatory archive (public data)", "frames": manifest}, fh, indent=1)
    print(f"Done in {time.time() - t0:.0f} s. Credit LCO when using these frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
