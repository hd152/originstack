"""Write an AAVSO Extended Format report from a finished --photometry-timeseries run.

The same report ``--aavso-report`` writes during a run (src/aavso_report.py), made afterwards
from the run's ``<stem>_lightcurves.csv`` and the WCS of its linear stack -- no restacking.

Usage:
    python tools/aavso_report.py stacked.fits --star "SS Cyg" --obscode XYZ --bin 5
    python tools/aavso_report.py stacked.fits --target 83.8221,-5.3911 --obscode XYZ
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    from astropy.io import fits
    from astropy.wcs import WCS

    from src.aavso_report import format_summary, write_aavso_report
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stack", help="the run's linear output FITS (next to its _lightcurves.csv)")
    ap.add_argument("--star", help="AAVSO name of the variable star")
    ap.add_argument("--target", metavar="RA,DEC", help="target position in degrees (instead of --star)")
    ap.add_argument("--obscode", help="your AAVSO observer code")
    ap.add_argument("--bin", type=float, default=0.0, help="time bins in minutes (0 = every frame)")
    ap.add_argument("--fov", type=float, default=60.0, help="VSP chart field of view, arcmin")
    ap.add_argument("--maglimit", type=float, default=15.0, help="faintest VSP comparison star")
    a = ap.parse_args()
    if not a.star and not a.target:
        ap.error("give --star NAME or --target RA,DEC")
    hdr = fits.getheader(a.stack)
    wcs_header = WCS(hdr, naxis=2).to_header()
    exptime = float(hdr.get("EXPTIME") or hdr.get("EXPOSURE") or 0.0)
    args = SimpleNamespace(aavso_star=a.star, photometry_target=a.target, aavso_obscode=a.obscode,
                           aavso_bin=a.bin, aavso_fov=a.fov, aavso_maglimit=a.maglimit)
    s = write_aavso_report(args, a.stack, wcs_header, exptime)
    print(format_summary(s, s.get("path")))
    if "error" in s:
        sys.exit(1)


if __name__ == "__main__":
    main()
