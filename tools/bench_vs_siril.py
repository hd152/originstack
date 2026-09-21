"""Reproduce the OriginStack-vs-Siril comparison in the README on your own data.

Stacks one folder of lights (plus any bias/dark/flat files in it) with OriginStack and
with Siril's command-line build, then reports per-stage wall time, star sharpness
(FWHM, measured with one detector on both stacks) and noise at matched resolution.

    python tools/bench_vs_siril.py "D:/astro/Omega/session1" --workdir bench_out
    python tools/bench_vs_siril.py lights/ --siril "C:/Program Files/Siril/bin/siril-cli.exe"

Needs a Siril 1.2+ install (``siril-cli``). Colour (Bayer) FITS lights only. Everything is
written under ``--workdir``; the input folder is never modified.

How the numbers are made comparable (and where they are not):
  * Both tools get the same lights and the same bias/dark/flat, with default-style settings.
    Siril: winsorized sigma clip 3/3, additive+scaling normalisation, 2-pass registration.
    OriginStack: its defaults (``--auto``).
  * OriginStack's time includes Phase 4 post-processing; Siril's does not. The report gives
    OriginStack's total *and* its total without Phase 4, which is the like-for-like figure.
  * Noise is a lag-4 pixel-difference MAD on the linear stacks after registering
    OriginStack's onto Siril's grid and matching the flux scale per channel. It depends on
    resolution (a blurrier stack is smoother per pixel), so it is also given after blurring
    OriginStack's stack to a sharpness no better than Siril's.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SIRIL_CANDIDATES = (
    r"C:\Program Files\Siril\bin\siril-cli.exe",
    "/usr/bin/siril-cli",
    "/usr/local/bin/siril-cli",
    "/Applications/Siril.app/Contents/MacOS/siril-cli",
)


def find_siril(explicit):
    for cand in ([explicit] if explicit else []) + [shutil.which("siril-cli")] + list(SIRIL_CANDIDATES):
        if cand and os.path.exists(cand):
            return cand
    sys.exit("siril-cli not found: install Siril 1.2+ or pass --siril PATH")


def split_inputs(folder):
    # case-insensitive filesystems return the same file for both patterns
    found = {os.path.normcase(p): p for pat in ("Light*.fits", "light*.fits")
             for p in glob.glob(os.path.join(folder, pat))}
    lights = sorted(found.values())
    if not lights:
        sys.exit(f"no Light*.fits frames in {folder}")
    cal = {}
    for kind in ("bias", "dark", "flat"):
        found = sorted(glob.glob(os.path.join(folder, f"{kind}*.fits")))
        cal[kind] = found[0] if found else None
    return lights, cal


def run_originstack(folder, workdir):
    out = os.path.join(workdir, "originstack.fits")
    t0 = time.time()
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "originstack.py"), "-d", folder, "-o", out],
                          capture_output=True, text=True, errors="replace")
    wall = time.time() - t0
    if proc.returncode != 0 or not os.path.exists(out):
        sys.exit("OriginStack failed:\n" + proc.stdout[-2000:] + proc.stderr[-2000:])
    phases = {}
    for name in ("Quality+Load", "Registration", "Stacking", "Post-process", "Other (I/O)"):
        m = re.search(re.escape(name) + r":\s+(?:(\d+)m )?([\d.]+)s", proc.stdout)
        if m:
            phases[name] = int(m.group(1) or 0) * 60 + float(m.group(2))
    return out, wall, phases


def run_siril(siril, lights, cal, workdir):
    root = os.path.join(workdir, "siril")
    shutil.rmtree(root, ignore_errors=True)
    for sub in ("raw", "cal", "process"):
        os.makedirs(os.path.join(root, sub))
    for f in lights:
        shutil.copy(f, os.path.join(root, "raw"))
    flags = []
    for kind, path in cal.items():
        if path:
            shutil.copy(path, os.path.join(root, "cal"))
            flags.append(f"-{kind}=../cal/{os.path.basename(path)}")
    script = "\n".join([
        "requires 1.2.0", "cd raw", "convert light -out=../process", "cd ../process",
        "calibrate light " + " ".join(flags) + " -cfa -equalize_cfa -debayer",
        "register pp_light -2pass", "seqapplyreg pp_light -framing=min",
        "stack r_pp_light rej 3 3 -norm=addscale -output_norm -rgb_equal -out=../siril_stack", "close", ""])
    with open(os.path.join(root, "run.ssf"), "w") as fh:
        fh.write(script)
    t0 = time.time()
    proc = subprocess.run([siril, "-d", root, "-s", "run.ssf"], capture_output=True, text=True, errors="replace")
    wall = time.time() - t0
    out = os.path.join(root, "siril_stack.fit")
    if not os.path.exists(out):
        sys.exit("Siril failed:\n" + proc.stdout[-2000:])
    # one "Execution time" line per command, in script order (register logs two: its two passes)
    secs = []
    for m in re.finditer(r"Execution time: ([\d.]+) (ms|s)\b", proc.stdout):
        secs.append(float(m.group(1)) / (1000 if m.group(2) == "ms" else 1))
    stages = {}
    if len(secs) >= 6:
        stages = {"Load+calibrate+debayer": secs[0] + secs[1], "Registration": secs[2] + secs[3],
                  "Warp": secs[4], "Combine": secs[5]}
    return out, wall, stages


def load_linear(path, siril=False):
    from astropy.io import fits
    data = fits.getdata(path).astype(np.float64)
    return data * 65535.0 if siril else data


def fwhm_and_stars(cube):
    from src.quality import detect_stars_auto, measure_fwhm
    lum = cube.mean(0).astype(np.float32)
    lum -= np.median(lum)
    d = lum[:, 4:] - lum[:, :-4]
    noise = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
    stars = detect_stars_auto(lum, noise)
    return float(measure_fwhm(lum, stars)), 0 if stars is None else len(stars)


def lag4_noise(x):
    d = x[:, 4:] - x[:, :-4]
    return 1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d))) / np.sqrt(2)


def noise_ratios(os_cube, siril_cube, blur_sigmas=(0.0, 0.5, 0.75, 1.0)):
    """OriginStack noise / Siril noise per channel at each blur, on Siril's grid."""
    from scipy import ndimage as ndi

    from src.blind_match import match_rigid_unknown_rotation
    from src.star_detect import detect_stars_matched_filter as detect

    def stars(c):
        s = detect(c.mean(0) - np.median(c.mean(0)))
        return s[np.argsort(-s["flux"])]

    params = match_rigid_unknown_rotation(stars(siril_cube), stars(os_cube), max_stars=60, pixel_tol=2.0)
    if params is None:
        return None
    p = params.params
    matrix = np.array([[p[1, 1], p[1, 0]], [p[0, 1], p[0, 0]]])
    offset = [p[1, 2], p[0, 2]]
    on_siril = np.stack([ndi.affine_transform(os_cube[i], matrix, offset=offset, output_shape=siril_cube.shape[1:],
                                              order=3, mode="nearest") for i in range(3)])
    sl = (slice(200, -200), slice(200, -200))
    rows = {}
    for sigma in blur_sigmas:
        blurred = np.stack([ndi.gaussian_filter(c, sigma) if sigma else c for c in on_siril])
        ratios = []
        for i in range(3):
            a, b = ndi.gaussian_filter(on_siril[i][sl], 4), ndi.gaussian_filter(siril_cube[i][sl], 4)
            gain = np.linalg.lstsq(np.c_[a.ravel(), np.ones(a.size)], b.ravel(), rcond=None)[0][0]
            ratios.append(gain * lag4_noise(blurred[i][sl]) / lag4_noise(siril_cube[i][sl]))
        rows[sigma] = (fwhm_and_stars(blurred)[0], ratios)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="folder with Light*.fits and optional bias*/dark*/flat* FITS files")
    ap.add_argument("--workdir", default="bench_out")
    ap.add_argument("--siril", help="path to siril-cli")
    args = ap.parse_args()

    lights, cal = split_inputs(args.folder)
    os.makedirs(args.workdir, exist_ok=True)
    siril = find_siril(args.siril)
    print(f"{len(lights)} lights; calibration: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in cal.items()))

    print("Running OriginStack ...")
    os_path, os_wall, os_phases = run_originstack(args.folder, args.workdir)
    print("Running Siril ...")
    si_path, si_wall, si_stages = run_siril(siril, lights, cal, args.workdir)

    os_cube, si_cube = load_linear(os_path), load_linear(si_path, siril=True)
    os_fwhm, os_stars = fwhm_and_stars(os_cube)
    si_fwhm, si_stars = fwhm_and_stars(si_cube)
    rows = noise_ratios(os_cube, si_cube)

    no_p4 = sum(v for k, v in os_phases.items() if k != "Post-process") if os_phases else None
    print("\n=== Timing (seconds) ===")
    print(f"OriginStack end to end (incl. Phase 4 post-processing): {os_wall:.1f}")
    if no_p4:
        print(f"OriginStack without post-processing (like-for-like):    {no_p4:.1f}  {os_phases}")
    print(f"Siril script total:                                     {si_wall:.1f}  {si_stages}")
    print("\n=== Stack quality (same detector on both linear stacks) ===")
    print(f"FWHM  OriginStack {os_fwhm:.2f} px   Siril {si_fwhm:.2f} px      stars {os_stars} / {si_stars}")
    if rows:
        print("\nOriginStack noise / Siril noise (R, G, B) after blurring OriginStack's stack:")
        for sigma, (fw, r) in rows.items():
            print(f"  blur sigma {sigma:<4} FWHM {fw:5.2f}   {r[0]:.2f} {r[1]:.2f} {r[2]:.2f}")
        print("  Compare at the row whose FWHM is closest to Siril's without being sharper.")
    with open(os.path.join(args.workdir, "results.json"), "w") as fh:
        json.dump({"lights": len(lights), "originstack": {"wall": os_wall, "phases": os_phases, "fwhm": os_fwhm,
                                                          "stars": os_stars},
                   "siril": {"wall": si_wall, "stages": si_stages, "fwhm": si_fwhm, "stars": si_stars},
                   "noise": {str(k): v[1] for k, v in (rows or {}).items()}}, fh, indent=2)


if __name__ == "__main__":
    main()
