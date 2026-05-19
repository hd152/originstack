"""Create synthetic FITS dataset for smoke testing the stacker."""
import os
import numpy as np
from astropy.io import fits


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


def main():
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


if __name__ == '__main__':
    main()
