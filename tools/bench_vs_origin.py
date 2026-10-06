"""Compare OriginStack's stack with the Celestron Origin's own on-board stack.

Every Origin session folder holds the telescope's result, ``FinalStackedMaster.tiff``. This
stacks the same lights with OriginStack (or takes an existing stack) and reports image quality
side by side: star sharpness on the same stars, noise per channel, and noise at matched
sharpness. It also writes a PNG of identical crops given the same stretch.

    python tools/bench_vs_origin.py "D:/astro/Black_Eye_Galaxy_2026-03-14_21-20-36"
    python tools/bench_vs_origin.py SESSION --stack existing_stacked.fits --png compare.png

How the two are made comparable:
  * The Origin TIFF is linear but sits on a large pedestal (sky ~30000 of 65535 in G/B), so
    bright stars and cores clip. Its channels are put into OriginStack's flux units with one
    gain per channel, measured from the brightness above sky in 300-15000 counts, where it is
    linear (the report says how linear). Clipped stars are left out of the width fits.
  * OriginStack's figures are for its linear stack (the main FITS, before post-processing),
    the like-for-like product. Its finished JPG is shown in the PNG for reference only.
  * Star width: one shared set of stars, fitted on each image's own pixel grid
    (common_star_fwhm.py), never resampled.
  * Noise: lag-4 pixel-difference MAD on each image's own grid. A sharper image is noisier per
    pixel, so the matched figure blurs whichever is sharper, per channel, until the shared
    stars are equally wide, then compares noise.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

_CLIP = 65000          # Origin TIFF values at or above this are treated as clipped
_LINEAR_RANGE = (300.0, 15000.0)


def load_origin(session):
    import tifffile
    path = os.path.join(session, "FinalStackedMaster.tiff")
    if not os.path.isfile(path):
        sys.exit(f"no FinalStackedMaster.tiff in {session}")
    return np.moveaxis(tifffile.imread(path).astype(np.float64), -1, 0)


def run_originstack(session, out):
    cmd = [sys.executable, os.path.join(ROOT, "originstack.py"), "-d", session, "-o", out]
    print("  running:", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def _stars(cube):
    from src.star_detect import detect_stars_matched_filter as detect
    s = detect(cube.mean(0) - np.median(cube.mean(0)))
    return s[np.argsort(-s["flux"])]


def register(os_cube, org):
    """Origin image resampled onto OriginStack's grid (bilinear, NaN outside); for the gain fit
    and the PNG only -- widths and noise are measured on each image's own grid."""
    from scipy import ndimage as ndi

    from src.blind_match import match_rigid_unknown_rotation
    m = match_rigid_unknown_rotation(_stars(os_cube), _stars(org), max_stars=60, pixel_tol=2.0)
    if m is None:
        sys.exit("could not match stars between the two stacks")
    p = m.params
    matrix = np.array([[p[1, 1], p[1, 0]], [p[0, 1], p[0, 0]]])
    offset = [p[1, 2], p[0, 2]]
    return np.stack([ndi.affine_transform(org[c], matrix, offset=offset, output_shape=os_cube.shape[1:],
                                          order=1, mode="constant", cval=np.nan) for c in range(3)])


def channel_gains(os_cube, org_on):
    """Per channel: (gain, max departure from it over the linear range, Origin sky, OS sky).
    gain converts Origin counts above sky into OriginStack counts above sky."""
    from scipy import ndimage as ndi
    out = []
    sl = (slice(100, -100), slice(100, -100))
    for c in range(3):
        valid = np.isfinite(org_on[c][sl]).ravel()
        a = ndi.gaussian_filter(np.nan_to_num(org_on[c]), 1.5)[sl].ravel()[valid]
        b = ndi.gaussian_filter(os_cube[c], 1.5)[sl].ravel()[valid]
        sa, sb = float(np.median(a)), float(np.median(b))
        ea, eb = a - sa, b - sb
        edges = np.geomspace(*_LINEAR_RANGE, 9)
        ratios = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (ea >= lo) & (ea < hi) & (a < _CLIP)
            if sel.sum() >= 50:
                ratios.append(float(np.median(eb[sel]) / np.median(ea[sel])))
        if not ratios:
            out.append((float("nan"), float("nan"), sa, sb))
            continue
        g = float(np.median(ratios))
        out.append((g, float(max(abs(r / g - 1) for r in ratios)), sa, sb))
    return out


def to_os_units(org, gains):
    """Origin image in OriginStack's flux units (own grid): gain * (x - sky) + OS sky."""
    return np.stack([g * (org[c] - sa) + sb for c, (g, _, sa, sb) in enumerate(gains)])


def lag4_noise(x):
    d = x[:, 4:] - x[:, :-4]
    return 1.4826 * float(np.nanmedian(np.abs(d - np.nanmedian(d)))) / np.sqrt(2)


def star_scale_noise(cube, fwhm):
    """Per channel: noise at the scale of a star -- background removed with a wide high-pass,
    then smoothed with a Gaussian of the stars' FWHM (a matched filter), robust sigma over
    the central field. This, not per-pixel noise, limits how faint a star can be seen: on
    oversampled stars a blur too small to widen them measurably still halves per-pixel noise."""
    from scipy import ndimage as ndi
    sigma = fwhm / 2.3548
    sl = (slice(200, -200), slice(200, -200))
    out = []
    for c in range(3):
        x = np.nan_to_num(cube[c])
        hp = ndi.gaussian_filter(x, sigma) - ndi.gaussian_filter(x, 10 * sigma)
        v = hp[sl]
        out.append(1.4826 * float(np.median(np.abs(v - np.median(v)))))
    return out


def comparison_png(os_cube, org_on, jpg_path, png_path, crop=600):
    """Three identical crops (Origin, OriginStack linear, OriginStack finished JPG) at the
    brightest smooth region, the first two given one asinh stretch with the same black and
    white points relative to each image's own sky."""
    from PIL import Image

    h, w = os_cube.shape[1:]
    # The Origin centres its target; crop there (1:1), and show the whole frame at 1/4.
    ys, xs = slice(h // 2 - crop // 2, h // 2 + crop // 2), slice(w // 2 - crop // 2, w // 2 + crop // 2)

    def stretch(cube, ys=slice(None), xs=slice(None)):
        out = []
        for c in range(3):
            ch = cube[c][ys, xs]
            sky = np.nanmedian(cube[c][200:-200, 200:-200])
            noise = lag4_noise(cube[c][200:-200, 200:-200])
            v = np.arcsinh(np.clip(ch - sky + 2 * noise, 0, None) / (3 * noise))
            out.append(v)
        v = np.stack(out, -1)
        top = np.nanpercentile(v, 99.8)
        return (np.clip(np.nan_to_num(v) / top, 0, 1) * 255).astype(np.uint8)

    full = [stretch(org_on), stretch(os_cube)]
    titles = ["Celestron Origin (same stretch)", "OriginStack linear (same stretch)"]
    if jpg_path and os.path.isfile(jpg_path):
        jpg = np.asarray(Image.open(jpg_path).convert("RGB"))
        if jpg.shape[:2] == os_cube.shape[1:]:
            full.append(jpg)
            titles.append("OriginStack finished JPG")
    rows = [[Image.fromarray(f).resize((w // 4, h // 4), Image.LANCZOS) for f in full],
            [Image.fromarray(f[ys, xs]) for f in full]]
    gap, head = 8, 24
    width = max(sum(im.width for im in r) + gap * (len(r) - 1) for r in rows)
    height = head + sum(r[0].height for r in rows) + gap
    canvas = Image.new("RGB", (width, height), (8, 9, 12))
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    y0 = head
    for r in rows:
        x0 = 0
        for i, im in enumerate(r):
            canvas.paste(im, (x0, y0))
            if y0 == head:
                d.text((x0 + 6, 5), titles[i], fill=(232, 230, 223))
            x0 += im.width + gap
        y0 += r[0].height + gap
    canvas.save(png_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session", help="Origin session folder (lights + FinalStackedMaster.tiff)")
    ap.add_argument("--stack", help="existing OriginStack linear FITS for this session")
    ap.add_argument("--workdir", default="bench_origin_out")
    ap.add_argument("--png", help="write a side-by-side crop PNG here")
    args = ap.parse_args()

    from astropy.io import fits
    name = os.path.basename(os.path.normpath(args.session))
    stack = args.stack
    if not stack:
        os.makedirs(args.workdir, exist_ok=True)
        stack = os.path.join(args.workdir, f"{name}_stacked.fits")
        if not os.path.isfile(stack):
            run_originstack(args.session, stack)
    os_cube = fits.getdata(stack).astype(np.float64)
    org = load_origin(args.session)
    org_on = register(os_cube, org)
    gains = channel_gains(os_cube, org_on)
    org_lin = to_os_units(org, gains)
    clip_mask = (org >= _CLIP).any(0)
    from scipy import ndimage as ndi
    clip_mask = ndi.binary_dilation(clip_mask, iterations=3)

    from common_star_fwhm import common_star_fwhm
    luma = common_star_fwhm(os_cube, org_lin, exclude_b=clip_mask)
    per_ch = []
    for c in range(3):
        r = common_star_fwhm(np.repeat(os_cube[c][None], 3, 0), np.repeat(org_lin[c][None], 3, 0),
                             exclude_b=clip_mask)
        per_ch.append(float("nan") if r is None else r["ratio_a_over_b"])
    sl = (slice(200, -200), slice(200, -200))
    pix = [lag4_noise(os_cube[c][sl]) / lag4_noise(org_lin[c][sl]) for c in range(3)]
    fwhm = luma["a"] if luma else 4.0
    star_os, star_org = star_scale_noise(os_cube, fwhm), star_scale_noise(org_lin, fwhm)

    print(f"\n{name}")
    print(f"  sizes: OriginStack {os_cube.shape[2]}x{os_cube.shape[1]}, Origin {org.shape[2]}x{org.shape[1]}")
    for c, (g, dev, sa, _sb) in zip("RGB", gains):
        print(f"  Origin {c}: linear within {dev * 100:.1f}% over +{_LINEAR_RANGE[0]:.0f}..+{_LINEAR_RANGE[1]:.0f}; "
              f"sky {sa:.0f}, headroom to clip {65535 - sa:.0f}")
    print(f"  clipped Origin pixels: {(org >= _CLIP).any(0).mean() * 100:.3f}% of the frame")
    if luma:
        print(f"  star width (same {luma['n']} stars): OriginStack {luma['a']:.2f} px, Origin {luma['b']:.2f} px, "
              f"ratio {luma['ratio_a_over_b']:.3f}")
    print("  per channel, OriginStack / Origin (below 1 = OriginStack sharper / quieter):")
    for c in range(3):
        print(f"    {'RGB'[c]}: star width {per_ch[c]:.3f}   pixel noise {pix[c]:.2f}   "
              f"star-scale noise {star_os[c] / star_org[c]:.2f}")
    if args.png:
        jpg = os.path.splitext(stack)[0] + ".jpg"
        comparison_png(os_cube, org_on, jpg, args.png)
        print(f"  wrote {args.png}")


if __name__ == "__main__":
    main()
