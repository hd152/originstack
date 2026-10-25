"""Pan-STARRS DR1 stack images on an Origin stack's own pixel grid, as truth for the learned
deconvolver.

For a crop of a linear stack with a celestial WCS (the pipeline's ``_stack_wcs``), each of the
g, r and i stack cutouts covering it (STScI's fitscut, which returns *linear* counts; the CDS
HiPS of r and i keep the PS1 stacks' per-skycell asinh compression and were not linear) is
reprojected onto the crop: every Origin pixel averages a SUB x SUB grid of samples (bilinear
in the PS1 image, 0.25"/px against Origin's ~1.47"), i.e. close to an area average. Pixels the
cutout does not cover (beyond its skycell) are NaN.

Output ``<out>/<name>.npz``: ``ps1`` (h, w, 3) float32 in g, r, i order (PS1 counts per
0.25" pixel, averaged), ``origin`` (h, w, 3) the stack's own crop, ``box`` (y0, x0, h, w),
``center`` (ra, dec).

All HTTP goes through ``net_query._http_get`` (the --offline guard).

Usage:
    python tools/fetch_ps1_truth.py STACK.fits --out C:/nn_work/survey/ps1 [--size 600] [--crops 2]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PS1 = "https://ps1images.stsci.edu/cgi-bin"
PS1_SCALE = 0.25          # arcsec / px
SUB = 6                   # samples per Origin pixel per axis (~ Origin / PS1 pixel ratio)


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


def ps1_files(ra: float, dec: float) -> dict:
    txt = _get(f"{PS1}/ps1filenames.py?ra={ra}&dec={dec}&filters=gri&type=stack", 60).decode()
    rows = [ln.split() for ln in txt.strip().splitlines()[1:]]
    return {r[4]: r[7] for r in rows if len(r) >= 8}


def skycells_for(points) -> list:
    """The distinct PS1 stack skycells containing any of ``points`` ((ra, dec) pairs): a cutout
    holds only its own skycell (~25'), so a crop across a cell edge needs several."""
    seen, out = set(), []
    for ra, dec in points:
        files = ps1_files(ra, dec)
        key = files.get("r")
        if key and key not in seen:
            seen.add(key)
            out.append(files)
    return out


def ps1_cutout(ra: float, dec: float, size: int, filename: str):
    data = _get(f"{PS1}/fitscut.cgi?ra={ra}&dec={dec}&size={size}&format=fits&red={filename}")
    with fits.open(io.BytesIO(data)) as h:
        return np.asarray(h[0].data, np.float32), WCS(h[0].header)


def reproject_onto(src: np.ndarray, src_wcs: WCS, dst_wcs: WCS, shape) -> np.ndarray:
    """Area-like average of ``src`` over each pixel of ``dst`` (SUB x SUB bilinear samples)."""
    h, w = shape
    off = (np.arange(SUB) + 0.5) / SUB - 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    acc = np.zeros((h, w))
    bad = np.zeros((h, w), bool)
    finite = np.isfinite(src)
    s0 = np.where(finite, src, 0.0)
    for dy in off:
        for dx in off:
            sky = dst_wcs.pixel_to_world_values(xx + dx, yy + dy)
            px, py = src_wcs.world_to_pixel_values(*sky)
            v = ndi.map_coordinates(s0, [py, px], order=1, mode="constant", cval=np.nan)
            ok = ndi.map_coordinates(finite.astype(np.float32), [py, px], order=1, mode="constant", cval=0.0)
            bad |= ~np.isfinite(v) | (ok < 0.999)
            acc += np.nan_to_num(v)
    out = (acc / (SUB * SUB)).astype(np.float32)
    out[bad] = np.nan
    return out


def crop_centres(H: int, W: int, size: int, n: int):
    """The frame centre first (usually the target), then points across the field."""
    cs = [(H // 2, W // 2)]
    # appended only: crop k keeps its place, so a larger --crops adds to an existing set
    for fy, fx in ((0.3, 0.25), (0.7, 0.75), (0.3, 0.75), (0.7, 0.25),
                   (0.5, 0.1), (0.5, 0.9), (0.15, 0.5), (0.85, 0.5)):
        cs.append((int(H * fy), int(W * fx)))
    half = size // 2
    return [(min(max(y, half), H - half), min(max(x, half), W - half)) for y, x in cs[:n]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=600, help="crop size in Origin pixels")
    ap.add_argument("--crops", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data, hdr = fits.getdata(args.stack, header=True)
    d = np.asarray(data, np.float32)
    A = np.moveaxis(d, 0, -1) if d.shape[0] == 3 else d
    wcs = WCS(hdr, naxis=2)
    H, W = A.shape[:2]
    scale = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    stem = os.path.splitext(os.path.basename(args.stack))[0]
    for k, (cy, cx) in enumerate(crop_centres(H, W, args.size, args.crops)):
        name = f"{stem}_c{k}"
        dst = os.path.join(args.out, name + ".npz")
        if os.path.exists(dst):
            continue
        y0, x0 = cy - args.size // 2, cx - args.size // 2
        # the crop's WCS: the stack's with CRPIX moved by the crop origin (WCS.slice put the
        # crop centre 0.26 deg off on these headers)
        sub_wcs = wcs.deepcopy()
        sub_wcs.wcs.crpix = sub_wcs.wcs.crpix - np.array([x0, y0], float)
        ra, dec = (float(v) for v in wcs.pixel_to_world_values(cx, cy))
        chk = np.array(sub_wcs.pixel_to_world_values(cx - x0, cy - y0), float)
        assert np.allclose(chk, [ra, dec], atol=1e-6), (chk, ra, dec)
        need = int(np.ceil(args.size * scale / PS1_SCALE * 1.45)) + 16    # any rotation fits
        n = args.size - 1
        corners = [(0, 0), (0, n), (n, 0), (n, n), (n / 2, n / 2), (0, n / 2), (n, n / 2), (n / 2, 0), (n / 2, n)]
        cells = skycells_for([tuple(float(v) for v in sub_wcs.pixel_to_world_values(x, y)) for y, x in corners])
        bands = []
        t = time.time()
        for b in "gri":
            acc = np.zeros((args.size, args.size))
            cnt = np.zeros((args.size, args.size))
            for files in cells:                    # skycells overlap: average where several cover
                img, w = ps1_cutout(ra, dec, need, files[b])
                r = reproject_onto(img, w, sub_wcs, (args.size, args.size))
                ok = np.isfinite(r)
                acc[ok] += r[ok]
                cnt[ok] += 1
            band = np.full((args.size, args.size), np.nan, np.float32)
            band[cnt > 0] = (acc[cnt > 0] / cnt[cnt > 0]).astype(np.float32)
            bands.append(band)
        ps1 = np.stack(bands, -1)
        np.savez(dst, ps1=ps1, origin=A[y0:y0 + args.size, x0:x0 + args.size], box=np.array([y0, x0, args.size, args.size]),
                 center=np.array([ra, dec]), scale=scale)
        print(f"{name}: ra {ra:.4f} dec {dec:+.4f}, PS1 covered {np.isfinite(ps1).all(-1).mean():.3f} "
              f"from {len(cells)} skycell(s) ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
