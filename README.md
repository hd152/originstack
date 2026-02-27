# Astrophotography FITS Stacker

Lightweight, streaming FITS stacker designed to run on resource-limited hardware.

Quick start

Install dependencies (recommended in virtualenv):

```bash
pip install -r requirements.txt
# Optional GPU packages (only on machines with CUDA):
pip install -r requirements-gpu.txt
```

Run (single folder):

```bash
python astro_stack.py -d lights/ -o stacked.fits
```

See `astro_stack.py --help` for full CLI options.

## Diagnostics & Troubleshooting

Run with `-v` (verbose) flag to see detailed output including frame quality metrics and registration shifts:

```bash
python astro_stack.py -d lights/ -o stacked.fits -v
```

This will show:
- **Per-frame quality metrics**: brightness, contrast, star count, SNR for each image
- **Per-frame registration shifts**: X/Y offsets and shift magnitude for each frame
- **Frame rejection reasons** when using `--quality-filter`

**New in this version:**
- Enhanced quality analysis with per-frame metrics reporting
- Shift reporting with magnitude (helps identify tracking/guiding issues)
- Improved shift robustness with better fallback algorithm
- Comprehensive diagnostic guides

**Registration not working?** Use `--debug-registration` to visualize diagnostic data:
```bash
python astro_stack.py -d lights/ -o stacked.fits --debug-registration
```
This creates PNG images and statistics files in `_registration_debug/` folder.

See documentation for detailed analysis:
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - One-page quick start
- [DIAGNOSTICS_IMPROVEMENTS.md](DIAGNOSTICS_IMPROVEMENTS.md) - Feature guide
- [SHIFT_PATTERN_GUIDE.md](SHIFT_PATTERN_GUIDE.md) - Detailed shift pattern analysis
- [REGISTRATION_DEBUG_GUIDE.md](REGISTRATION_DEBUG_GUIDE.md) - Fix registration issues

## Testing

Install `pytest` then run:

```bash
pip install pytest
pytest -q
```

## FITS Header Metadata

The stacker now populates comprehensive FITS headers with:
- **Stacking metadata**: Number of frames, rejection count, stacking method
- **Calibration info**: Which calibration frames were applied (bias/dark/flat)
- **Registration statistics**: Mean/std of shifts, maximum shift magnitude
- **Processing times**: Quality analysis, registration, and stacking times
- **Quality metrics**: Average brightness, contrast, and quality scores
- **Original metadata**: Copied from input frames (telescope, instrument, observer, exposure time, etc.)

## Plate Solving (Astrometry)

Automatic plate solving can be enabled to identify celestial objects and add WCS (World Coordinate System) information to the FITS header:

**Requirements:**
1. Install astroquery: `pip install astroquery`
2. Get a free API key from [nova.astrometry.net](https://nova.astrometry.net/api_help)
3. Set environment variable: `set ASTROMETRY_API_KEY=your_key_here` (Windows) or `export ASTROMETRY_API_KEY=your_key_here` (Linux/Mac)

**Usage:**
```bash
# Plate solving is attempted by default (if astroquery is installed and API key is set)
python astro_stack.py -d lights/ -o stacked.fits -v

# Skip plate solving
python astro_stack.py -d lights/ -o stacked.fits --skip-plate-solve
```

When plate solving succeeds:
- WCS keywords are added (CRVAL1/2, CRPIX1/2, CD matrix, etc.)
- Object name is identified via SIMBAD database (if available)
- Field center coordinates (RA/DEC) are recorded

This enables astronomical software (like DS9, PixInsight) to display coordinate grids and identify objects in your stacked image.

New features included in this scaffold:
- Malvar demosaicing (`--debayer-method malvar`)
- White-balance options: `--white-balance grayworld|whitepatch|none`
- Simple drizzle combining: `--drizzle-scale N`
- Hot-pixel removal and gradient subtraction
- Experimental CuPy hooks (`--use-gpu`) — requires CuPy installed and is best-effort
- **NEW**: Comprehensive FITS header population with all metadata
- **NEW**: Automatic plate solving and object identification

