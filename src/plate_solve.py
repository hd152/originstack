"""Plate solving via astrometry.net."""
from __future__ import annotations

import os
import tempfile
import threading

import numpy as np
from astropy.io import fits

try:
    from astroquery.astrometry_net import AstrometryNet
    HAS_ASTROMETRY_NET = True
except Exception:
    HAS_ASTROMETRY_NET = False


def solve_plate(image_data: np.ndarray, header: fits.Header, output_path: str, verbose: bool = False) -> bool:
    """
    Attempt to plate-solve the stacked image and add WCS + object info to header.

    Args:
        image_data: Stacked image data (H, W, 3) or (3, H, W)
        header: FITS header to update with WCS info
        output_path: Path where FITS file is saved
        verbose: Print detailed progress

    Returns:
        True if plate solving succeeded, False otherwise
    """
    if not HAS_ASTROMETRY_NET:
        if verbose:
            print("  [Plate solving] astroquery not available - skipping")
        return False

    try:
        # Convert to grayscale luminance for plate solving
        if image_data.ndim == 3:
            if image_data.shape[0] == 3:
                # (3, H, W) format
                lum = 0.299 * image_data[0] + 0.587 * image_data[1] + 0.114 * image_data[2]
            else:
                # (H, W, 3) format
                lum = 0.299 * image_data[:, :, 0] + 0.587 * image_data[:, :, 1] + 0.114 * image_data[:, :, 2]
        else:
            lum = image_data

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

            # Initialize astrometry.net client
            ast = AstrometryNet()

            # Check for API key in environment or config
            api_key = os.environ.get('ASTROMETRY_API_KEY')
            if not api_key:
                if verbose:
                    print("  [Plate solving] No API key found. Set ASTROMETRY_API_KEY environment variable.")
                    print("  [Plate solving] Get a key from: https://nova.astrometry.net/api_help")
                return False

            ast.api_key = api_key

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
                    solve_result[0] = ast.solve_from_image(
                        tmp_path,
                        force_image_upload=True,
                        solve_timeout=300,
                        scale_units=scale_units,
                        scale_lower=scale_lower,
                        scale_upper=scale_upper,
                        publicly_visible='n'
                    )
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
                    print("  [Plate solving] ✓ Success! Adding WCS to header...")

                # Copy WCS keywords to main header
                wcs_keywords = ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                               'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CROTA1', 'CROTA2',
                               'EQUINOX', 'RADECSYS', 'CUNIT1', 'CUNIT2']
                for key in wcs_keywords:
                    if key in wcs_header:
                        header[key] = wcs_header[key]

                # Add plate solving success flag
                header['PLTSOLVD'] = (True, 'Plate solving successful')

                # Try to identify object using SIMBAD
                if 'CRVAL1' in header and 'CRVAL2' in header:
                    try:
                        from astroquery.simbad import Simbad
                        ra = float(header['CRVAL1'])
                        dec = float(header['CRVAL2'])

                        # Query SIMBAD for objects near the center
                        custom_simbad = Simbad()
                        custom_simbad.add_votable_fields('otype')
                        result = custom_simbad.query_region(f"{ra} {dec}", radius='0d30m0s', frame='icrs')

                        if result and len(result) > 0:
                            # Get the brightest/most prominent object
                            obj_name = result[0]['MAIN_ID']
                            obj_type = result[0]['OTYPE']
                            if verbose:
                                print(f"  [Plate solving] Identified object: {obj_name} ({obj_type})")
                            header['OBJECT'] = (str(obj_name), 'Object identified via SIMBAD')
                            header['OBJTYPE'] = (str(obj_type), 'Object type from SIMBAD')
                    except Exception as e:
                        if verbose:
                            print(f"  [Plate solving] Object identification failed: {e}")

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
