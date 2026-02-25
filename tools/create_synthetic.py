"""Create synthetic FITS dataset for smoke testing the stacker."""
import os
import numpy as np
from astropy.io import fits


def make_star_image(shape=(256, 256), centers=None, amp=5000.0, bg=100.0):
    if centers is None:
        centers = [(shape[0] // 2, shape[1] // 2)]
    im = np.zeros(shape, dtype=np.float32)
    yy, xx = np.indices(shape)
    for y, x in centers:
        r2 = (yy - y) ** 2 + (xx - x) ** 2
        im += amp * np.exp(-r2 / (2.0 * 2.5 ** 2))
    im += bg
    # add small noise
    im += np.random.normal(scale=2.0, size=shape)
    return im


def main():
    outdir = os.path.join(os.path.dirname(__file__), '..', 'synthetic_data')
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    # create darks
    darks = []
    for i in range(3):
        d = np.random.normal(loc=50.0, scale=1.0, size=(256, 256)).astype(np.float32)
        p = os.path.join(outdir, f'dark_{i:03d}.fit')
        fits.writeto(p, d, overwrite=True)
        darks.append(p)
    # create flats
    for i in range(2):
        fimg = np.ones((256, 256), dtype=np.float32) * 10000.0
        # slight vignetting
        yy, xx = np.indices((256, 256))
        r = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
        fimg *= (1.0 - 0.0003 * r)
        p = os.path.join(outdir, f'flat_{i:03d}.fit')
        fits.writeto(p, fimg.astype(np.float32), overwrite=True)
    # create lights with small shifts
    shifts = [(0, 0), (1, -1), (-2, 2), (0, 3), (3, 0), (-1, -2)]
    for i, sh in enumerate(shifts):
        cy = 128 + sh[0]
        cx = 128 + sh[1]
        img = make_star_image((256, 256), centers=[(cy, cx)], amp=3000.0, bg=120.0)
        p = os.path.join(outdir, f'light_{i:03d}.fit')
        hdr = fits.Header()
        hdr['BAYERPAT'] = 'RGGB'
        fits.writeto(p, img.astype(np.float32), header=hdr, overwrite=True)
    print('Created synthetic dataset in', outdir)


if __name__ == '__main__':
    main()
