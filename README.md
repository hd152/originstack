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

See documentation for detailed analysis:
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - One-page quick start
- [DIAGNOSTICS_IMPROVEMENTS.md](DIAGNOSTICS_IMPROVEMENTS.md) - Feature guide
- [SHIFT_PATTERN_GUIDE.md](SHIFT_PATTERN_GUIDE.md) - Detailed shift pattern analysis

## Testing

Install `pytest` then run:

```bash
pip install pytest
pytest -q
```

New features included in this scaffold:
- Malvar demosaicing (`--debayer-method malvar`)
- White-balance options: `--white-balance grayworld|whitepatch|none`
- Simple drizzle combining: `--drizzle-scale N`
- Hot-pixel removal and gradient subtraction
- Experimental CuPy hooks (`--use-gpu`) — requires CuPy installed and is best-effort

