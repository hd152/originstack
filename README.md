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

Testing

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

