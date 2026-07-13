"""Create synthetic datasets for smoke testing the stacker.

Bare ``python tools/create_synthetic.py`` (no args) keeps producing the
original FITS-only ``synthetic_data/`` directory the CI smoke test uses --
unchanged, for backward compatibility. ``--mixed`` additionally builds a
``synthetic_data_mixed/`` directory mixing FITS + TIFF + XISF + SER lights
in one folder, exercising discover_frames/classify_frame/load_frame/
make_master across every supported input format in a single real run.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_star_image(shape=(256, 256), n_stars=30, seed=None, bg=1000.0):
    rng = np.random.default_rng(seed)
    im = np.full(shape, bg, dtype=np.float32)
    yy, xx = np.indices(shape)
    margin = 20
    for _ in range(n_stars):
        y = rng.integers(margin, shape[0] - margin)
        x = rng.integers(margin, shape[1] - margin)
        amp = rng.uniform(2000.0, 8000.0)
        sigma = rng.uniform(1.5, 3.0)
        r2 = (yy - y) ** 2 + (xx - x) ** 2
        im += amp * np.exp(-r2 / (2.0 * sigma ** 2))
    im += rng.normal(scale=5.0, size=shape).astype(np.float32)
    return im


def _write_synthetic_ser(path: str, frames: list) -> None:
    """Hand-rolled SER writer, synthetic-fixture only -- io_ser.py is
    read-only by design (this pipeline never produces SER output)."""
    h, w = frames[0].shape
    frames_u16 = [np.clip(f, 0, 65535).astype('<u2') for f in frames]
    header = struct.pack(
        '<14s7i40s40s40sqq',
        b'LUCAM-RECORDER', 0, 0, 1, w, h, 16, len(frames_u16),
        b'\x00' * 40, b'\x00' * 40, b'\x00' * 40, 0, 0,
    )
    with open(path, 'wb') as fh:
        fh.write(header)
        for fr in frames_u16:
            fh.write(fr.tobytes())


def make_mixed_format_dataset(outdir: str) -> None:
    """One directory mixing FITS + TIFF + XISF + SER lights, plus FITS
    calibration frames, all the same dimensions so calibration matching
    works across formats."""
    os.makedirs(outdir, exist_ok=True)
    shape = (256, 256)

    for i in range(2):
        d = np.random.default_rng(i).normal(50.0, 1.5, shape).astype(np.float32)
        fits.writeto(os.path.join(outdir, f'dark_{i:03d}.fit'), d, overwrite=True)
    yy, xx = np.indices(shape)
    r = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
    fimg = (1000.0 * (1.0 - 0.0003 * r)).astype(np.float32)
    fits.writeto(os.path.join(outdir, 'flat_000.fit'), fimg, overwrite=True)

    # All 5 lights share ONE base star field with small integer dithers (like
    # the single-format generator above) rather than independent random star
    # placements -- otherwise registration has no real dither relationship to
    # find between formats and rejects everything for an uninteresting reason
    # (proving nothing about format interop). Dithers applied via np.roll.
    base = make_star_image(shape, n_stars=20, seed=100, bg=1000.0)
    dithers = [(0, 0), (1, -1), (-2, 1), (1, 2), (-1, -2)]

    def _dithered(i):
        dy, dx = dithers[i]
        return np.roll(np.roll(base, dy, axis=0), dx, axis=1)

    # Light 0: FITS (Bayer mosaic, header-declared pattern)
    hdr = fits.Header()
    hdr['BAYERPAT'] = 'RGGB'
    fits.writeto(os.path.join(outdir, 'light_000.fit'), _dithered(0).astype(np.float32),
                 header=hdr, overwrite=True)

    # Light 1: TIFF (Bayer mosaic; TIFF can't carry BAYERPAT itself, so a
    # .json sidecar supplies it -- same mechanism used for FITS/RAW gaps)
    try:
        import tifffile
        tpath = os.path.join(outdir, 'light_001.tif')
        tifffile.imwrite(tpath, np.clip(_dithered(1), 0, 65535).astype(np.uint16))
        with open(os.path.join(outdir, 'light_001.json'), 'w', encoding='utf-8') as fh:
            json.dump({'bayerPattern': 'RGGB', 'exposureTimeMS': 60000}, fh)
    except ImportError:
        print('  tifffile not installed -- skipping TIFF fixture')

    # Light 2: XISF (already-debayered RGB, no Bayer ambiguity)
    from src.xisf_writer import write_xisf
    img2 = _dithered(2)
    rgb2 = np.stack([img2, img2, img2], axis=2).astype(np.float32)
    write_xisf(rgb2, os.path.join(outdir, 'light_002.xisf'),
              header_meta={'EXPTIME': 60.0})

    # Light 3+4: one SER file holding 2 mono frames -> expands to 2 lights
    _write_synthetic_ser(os.path.join(outdir, 'light_003.ser'),
                         [_dithered(3), _dithered(4)])

    print('Created mixed-format synthetic dataset in', outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixed', action='store_true',
                        help='Also build a mixed-format (FITS+TIFF+XISF+SER) '
                             'synthetic_data_mixed/ directory')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), '..', 'synthetic_data')
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    # Darks — thermal background ~50 ADU
    for i in range(3):
        d = np.random.default_rng(i).normal(loc=50.0, scale=1.5, size=(256, 256)).astype(np.float32)
        fits.writeto(os.path.join(outdir, f'dark_{i:03d}.fit'), d, overwrite=True)

    # Flats — normalised to ~1000 ADU with slight vignetting
    yy, xx = np.indices((256, 256))
    r = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
    for i in range(2):
        fimg = (1000.0 * (1.0 - 0.0003 * r)).astype(np.float32)
        fits.writeto(os.path.join(outdir, f'flat_{i:03d}.fit'), fimg, overwrite=True)

    # Lights — 30 stars, small dither shifts between frames
    shifts = [(0, 0), (2, -2), (-3, 3), (1, 4), (4, 1), (-2, -3)]
    for i, (dy, dx) in enumerate(shifts):
        img = make_star_image((256, 256), n_stars=30, seed=42 + i, bg=1000.0)
        # apply integer shift by rolling
        img = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
        hdr = fits.Header()
        hdr['BAYERPAT'] = 'RGGB'
        fits.writeto(os.path.join(outdir, f'light_{i:03d}.fit'),
                     img.astype(np.float32), header=hdr, overwrite=True)

    print('Created synthetic dataset in', outdir)

    if args.mixed:
        mixed_outdir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'synthetic_data_mixed'))
        make_mixed_format_dataset(mixed_outdir)


if __name__ == '__main__':
    main()
