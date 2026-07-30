"""Plate solving via ASTAP (local) or astrometry.net (online).

Backend selection:
  --plate-solver astap        Fast local solver (ASTAP binary + star database)
  --plate-solver astrometry   Online nova.astrometry.net (default, requires API key)

ASTAP is preferred for speed (~1 s vs 30–120 s) and offline use.
Download ASTAP from: https://www.hnsky.org/astap.htm
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Optional

import numpy as np
from astropy.io import fits

from src import net_query
from src.utils import safe_print


# ---------------------------------------------------------------------------
# ASTAP helpers
# ---------------------------------------------------------------------------

def _find_astap_binary() -> Optional[str]:
    """Search PATH and common install locations for the ASTAP binary."""
    candidates = [
        "astap",
        # Windows default install locations
        r"C:\Program Files\astap\astap.exe",
        r"C:\Program Files (x86)\astap\astap.exe",
        # macOS
        "/Applications/ASTAP.app/Contents/MacOS/astap",
        "/usr/local/bin/astap",
        # Linux
        "/usr/bin/astap",
    ]
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
        if os.path.isfile(candidate):
            return candidate
    return None


def _solve_astap(lum: np.ndarray, header: fits.Header,
                 verbose: bool = False,
                 astap_path: Optional[str] = None) -> bool:
    """Plate-solve using the local ASTAP binary.

    Writes a temporary FITS file, calls ASTAP, reads the WCS sidecar
    (.wcs), and injects WCS keywords into *header*.

    Returns True on success.
    """
    binary = astap_path or _find_astap_binary()
    if binary is None:
        if verbose:
            print("  [Plate solving] ASTAP binary not found — falling back to astrometry.net")
        return False

    with tempfile.TemporaryDirectory() as td:
        in_fits = os.path.join(td, "astap_in.fits")
        wcs_path = os.path.join(td, "astap_in.wcs")

        # Write luminance FITS
        tmp_hdu = fits.PrimaryHDU(lum.astype(np.float32))
        for key in ("TELESCOP", "INSTRUME", "FOCALLEN", "XPIXSZ", "YPIXSZ",
                    "NAXIS1", "NAXIS2"):
            if key in header:
                tmp_hdu.header[key] = header[key]
        tmp_hdu.writeto(in_fits, overwrite=True)

        # Build ASTAP command
        cmd = [binary, "-f", in_fits, "-wcs", "-update", "-o", in_fits]

        # Optional: pass initial RA/Dec hint to speed up blind solve
        if "OBJCTRA" in header and "OBJCTDEC" in header:
            cmd += ["-ra", str(header["OBJCTRA"]),
                    "-spd", str(float(header["OBJCTDEC"]) + 90.0)]
        elif "CRVAL1" in header and "CRVAL2" in header:
            cmd += ["-ra",  str(header["CRVAL1"]),
                    "-spd", str(float(header["CRVAL2"]) + 90.0)]

        # Plate scale hint
        if "FOCALLEN" in header and "XPIXSZ" in header:
            try:
                fl_mm = float(header["FOCALLEN"])
                px_um = float(header["XPIXSZ"])
                scale_asp = (px_um / 1000.0) / fl_mm * 206265.0
                cmd += ["-s", f"{scale_asp * 0.9:.3f}", "-t", f"{scale_asp * 1.1:.3f}"]
            except (ValueError, ZeroDivisionError):
                pass

        if verbose:
            print(f"  [Plate solving] Running ASTAP: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
        except subprocess.TimeoutExpired:
            if verbose:
                print("  [Plate solving] ASTAP timed out after 120 s")
            return False
        except FileNotFoundError:
            if verbose:
                print(f"  [Plate solving] ASTAP binary not executable: {binary}")
            return False

        if verbose and result.stdout:
            for line in result.stdout.strip().splitlines()[-5:]:
                print(f"    ASTAP: {line}")

        # ASTAP writes a .wcs sidecar alongside the input file
        if not os.path.exists(wcs_path):
            if verbose:
                print("  [Plate solving] ASTAP: no .wcs output produced — solve failed")
            return False

        # Parse WCS keywords from sidecar
        try:
            with fits.open(wcs_path) as wcs_hdul:
                wcs_header = wcs_hdul[0].header
        except Exception as e:
            if verbose:
                print(f"  [Plate solving] ASTAP: could not read .wcs file: {e}")
            return False

        wcs_keys = ["CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2",
                    "CRPIX1", "CRPIX2", "CD1_1", "CD1_2",
                    "CD2_1", "CD2_2", "CDELT1", "CDELT2",
                    "CROTA1", "CROTA2", "EQUINOX", "RADECSYS"]
        copied = 0
        for key in wcs_keys:
            if key in wcs_header:
                header[key] = wcs_header[key]
                copied += 1

        if copied < 4:
            if verbose:
                print("  [Plate solving] ASTAP: insufficient WCS keywords in .wcs file")
            return False

        header["PLTSOLVD"] = (True, "Plate solving successful (ASTAP)")
        header["PLTSOLVR"] = ("ASTAP", "Plate solver used")
        if verbose:
            print(f"  [Plate solving] ASTAP solved: "
                  f"RA={header.get('CRVAL1', '?'):.4f} "
                  f"Dec={header.get('CRVAL2', '?'):.4f}")
        return True


def _try_simbad_identification(header: fits.Header, verbose: bool = False) -> None:
    """Query SIMBAD for the object at the plate-solved coordinates."""
    if "CRVAL1" not in header or "CRVAL2" not in header:
        return
    try:
        ra  = float(header["CRVAL1"])
        dec = float(header["CRVAL2"])
        result = net_query.simbad_cone_search(ra, dec, radius_deg=0.5)
        if result is not None and len(result) > 0:
            obj_name = result["main_id"][0]
            obj_type = result["otype"][0]
            if verbose:
                print(f"  [Plate solving] Identified object: {obj_name} ({obj_type})")
            header["OBJECT"]  = (str(obj_name), "Object identified via SIMBAD")
            header["OBJTYPE"] = (str(obj_type), "Object type from SIMBAD")
    except Exception as e:
        if verbose:
            print(f"  [Plate solving] Object identification failed: {e}")


def solve_plate(image_data: np.ndarray, header: fits.Header, output_path: str,
                verbose: bool = False,
                solver: str = "astrometry",
                astap_path: Optional[str] = None) -> bool:
    """
    Attempt to plate-solve the stacked image and add WCS + object info to header.

    Args:
        image_data: Stacked image data (H, W, 3) or (3, H, W)
        header: FITS header to update with WCS info
        output_path: Path where FITS file is saved
        verbose: Print detailed progress
        solver: 'astap' for local ASTAP binary, 'astrometry' for nova.astrometry.net
        astap_path: Optional explicit path to the ASTAP binary

    Returns:
        True if plate solving succeeded, False otherwise
    """
    # Convert image to luminance
    if image_data.ndim == 3:
        if image_data.shape[0] == 3:
            lum = 0.299 * image_data[0] + 0.587 * image_data[1] + 0.114 * image_data[2]
        else:
            lum = 0.299 * image_data[:, :, 0] + 0.587 * image_data[:, :, 1] + 0.114 * image_data[:, :, 2]
    else:
        lum = image_data

    # Try ASTAP first if requested
    if solver == "astap":
        if verbose:
            print("  [Plate solving] Attempting ASTAP (local solver)...")
        try:
            if _solve_astap(lum, header, verbose=verbose, astap_path=astap_path):
                # SIMBAD identification using the solved coordinates
                _try_simbad_identification(header, verbose=verbose)
                return True
        except Exception as e:
            if verbose:
                print(f"  [Plate solving] ASTAP error: {e} — falling back to astrometry.net")
        if verbose:
            print("  [Plate solving] ASTAP failed — falling back to astrometry.net")

    try:
        # Create temporary FITS file with luminance for submission
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save luminance to temporary file
            tmp_hdu = fits.PrimaryHDU(lum.astype(np.float32))
            # Copy relevant keywords that might help plate solving
            for key in ['TELESCOP', 'INSTRUME', 'FOCALLEN', 'XPIXSZ', 'YPIXSZ', 'APTDIA']:
                if key in header:
                    tmp_hdu.header[key] = header[key]
            tmp_hdu.writeto(tmp_path, overwrite=True)

            if verbose:
                print("  [Plate solving] Submitting to astrometry.net...")

            # Check for API key in environment or config
            api_key = os.environ.get('ASTROMETRY_API_KEY')
            if not api_key:
                if verbose:
                    print("  [Plate solving] No API key found. Set ASTROMETRY_API_KEY environment variable.")
                    print("  [Plate solving] Get a key from: https://nova.astrometry.net/api_help")
                return False

            # Estimate scale from header if available
            scale_units = 'arcsecperpix'
            scale_lower = None
            scale_upper = None

            if 'FOCALLEN' in header and 'XPIXSZ' in header:
                try:
                    focal_length_mm = float(header['FOCALLEN'])
                    pixel_size_um = float(header['XPIXSZ'])
                    # Calculate plate scale: pixel_size / focal_length * 206265 arcsec/radian
                    plate_scale = (pixel_size_um / 1000.0) / focal_length_mm * 206265.0
                    scale_lower = plate_scale * 0.9  # 10% tolerance
                    scale_upper = plate_scale * 1.1
                    if verbose:
                        print(f"  [Plate solving] Estimated scale: {plate_scale:.2f} arcsec/pixel")
                except (ValueError, ZeroDivisionError):
                    pass

            # Submit for solving with timeout protection
            solve_result = [None]
            solve_error = [None]

            def _solve():
                try:
                    session = net_query.astrometry_net_login(api_key)
                    if session is None:
                        solve_error[0] = RuntimeError("astrometry.net login failed")
                        return
                    subid = net_query.astrometry_net_upload(
                        session, tmp_path,
                        scale_units=scale_units,
                        scale_lower=scale_lower,
                        scale_upper=scale_upper)
                    if subid is None:
                        solve_error[0] = RuntimeError("astrometry.net upload failed")
                        return
                    deadline = time.time() + 300  # solve budget
                    job_id = net_query.astrometry_net_poll_submission(subid, deadline)
                    if job_id is None:
                        return  # no job yet -> treated as unsolved below
                    ok = net_query.astrometry_net_poll_job(job_id, deadline)
                    if ok:
                        solve_result[0] = net_query.astrometry_net_fetch_wcs(job_id)
                except Exception as e:
                    solve_error[0] = e

            solver_thread = threading.Thread(target=_solve, daemon=True)
            solver_thread.start()
            solver_thread.join(timeout=360)  # 6 minute hard timeout

            if solver_thread.is_alive():
                if verbose:
                    print("  [Plate solving] Timed out after 360 seconds")
                header['PLTSOLVD'] = (False, 'Plate solving timed out')
                return False

            if solve_error[0] is not None:
                raise solve_error[0]

            wcs_header = solve_result[0]

            if wcs_header:
                if verbose:
                    safe_print("  [Plate solving] ✓ Success! Adding WCS to header...")

                # Copy WCS keywords to main header
                wcs_keywords = ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                               'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CROTA1', 'CROTA2',
                               'EQUINOX', 'RADECSYS', 'CUNIT1', 'CUNIT2']
                for key in wcs_keywords:
                    if key in wcs_header:
                        header[key] = wcs_header[key]

                # Add plate solving success flag
                header['PLTSOLVD'] = (True, 'Plate solving successful')
                header['PLTSOLVR'] = ('astrometry.net', 'Plate solver used')

                _try_simbad_identification(header, verbose=verbose)
                return True
            else:
                if verbose:
                    print("  [Plate solving] Failed to solve plate")
                header['PLTSOLVD'] = (False, 'Plate solving attempted but failed')
                return False

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

    except Exception as e:
        if verbose:
            print(f"  [Plate solving] Error: {e}")
        header['PLTSOLVD'] = (False, f'Plate solving error: {str(e)[:50]}')
        return False
