"""Reproduce the OriginStack-vs-Siril comparison in the README on your own data.

Stacks one folder of lights (plus any bias/dark/flat files in it) with OriginStack and
with Siril's command-line build, then reports per-stage wall time, star sharpness
(FWHM, fitted on the same stars in both stacks) and noise.

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
    resolution (a blurrier stack is smoother per pixel), so read it together with the star
    widths; the blurred rows are for reference only.
  * Star width is fitted on ONE shared set of stars, at the same positions in both stacks and
    on each stack's own pixel grid (see common_star_fwhm.py). Comparing each stack's FWHM over
    its own detected star list is not valid: which stars get picked changes the number by more
    than the differences being claimed (an earlier version of this tool did exactly that).
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
sys.path.insert(0, os.path.join(ROOT, "tools"))

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


def run_sampled(cmd, disk_path, **kw):
    """Run *cmd* and sample, every 0.25 s, the private memory of the whole process tree
    (workers included) and how far free space on *disk_path*'s drive has fallen. Private
    bytes, not RSS: memory-mapped temp files and shared pages would otherwise be counted once
    per process. Returns (returncode, stdout, seconds, peak_private_mb, peak_disk_gb)."""
    import threading

    import psutil

    base_free = shutil.disk_usage(disk_path).free
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace", **kw)
    chunks = []
    reader = threading.Thread(target=lambda: chunks.append(proc.stdout.read()), daemon=True)
    reader.start()
    root = psutil.Process(proc.pid)
    peak_mem = peak_disk = 0
    t0 = time.time()
    while proc.poll() is None:
        total = 0
        for p in [root] + root.children(recursive=True):
            try:
                mi = p.memory_info()
                total += getattr(mi, "private", mi.rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak_mem = max(peak_mem, total)
        peak_disk = max(peak_disk, base_free - shutil.disk_usage(disk_path).free)
        time.sleep(0.25)
    reader.join(timeout=10)
    return proc.returncode, "".join(chunks), time.time() - t0, peak_mem / 2**20, peak_disk / 2**30


def run_originstack(folder, workdir):
    out = os.path.join(workdir, "originstack.fits")
    rc, text, wall, mem, disk = run_sampled(
        [sys.executable, os.path.join(ROOT, "originstack.py"), "-d", folder, "-o", out], workdir)
    if rc != 0 or not os.path.exists(out):
        sys.exit("OriginStack failed:\n" + text[-3000:])
    phases = {}
    for name in ("Quality+Load", "Registration", "Stacking", "Post-process", "Other (I/O)"):
        m = re.search(re.escape(name) + r":\s+(?:(\d+)m )?([\d.]+)s", text)
        if m:
            phases[name] = int(m.group(1) or 0) * 60 + float(m.group(2))
    m = re.search(r"Frames stacked:\s+(\d+)", text)
    return out, wall, phases, {"peak_private_mb": mem, "peak_disk_gb": disk,
                               "frames_stacked": int(m.group(1)) if m else None}


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
    rc, text, wall, mem, disk = run_sampled([siril, "-d", root, "-s", "run.ssf"], root)
    out = os.path.join(root, "siril_stack.fit")
    if not os.path.exists(out):
        sys.exit("Siril failed:\n" + text[-3000:])
    # one "Execution time" line per command, in script order (register logs two: its two passes)
    secs = []
    for m in re.finditer(r"Execution time: ([\d.]+) (ms|s)\b", text):
        secs.append(float(m.group(1)) / (1000 if m.group(2) == "ms" else 1))
    stages = {}
    if len(secs) >= 6:
        stages = {"Load+calibrate+debayer": secs[0] + secs[1], "Registration": secs[2] + secs[3],
                  "Warp": secs[4], "Combine": secs[5]}
    m = re.search(r"(\d+) images have been stacked", text)
    return out, wall, stages, {"peak_private_mb": mem, "peak_disk_gb": disk,
                               "frames_stacked": int(m.group(1)) if m else None}


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
    os_path, os_wall, os_phases, os_res = run_originstack(args.folder, args.workdir)
    print("Running Siril ...")
    si_path, si_wall, si_stages, si_res = run_siril(siril, lights, cal, args.workdir)

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
    print("\n=== Resources (private memory summed over all processes; disk = fall in free space) ===")
    for name, r in (("OriginStack", os_res), ("Siril", si_res)):
        print(f"{name:<12} peak memory {r['peak_private_mb'] / 1024:6.1f} GB   peak disk {r['peak_disk_gb']:6.1f} GB   "
              f"frames stacked {r['frames_stacked']} of {len(lights)}")
    from common_star_fwhm import common_star_fwhm
    shared = common_star_fwhm(os_cube, si_cube)
    print("\n=== Star width on the same stars in both linear stacks (smaller is sharper) ===")
    if shared:
        print(f"OriginStack {shared['a']:.2f} px (IQR {shared['a_iqr'][0]:.2f}-{shared['a_iqr'][1]:.2f})   "
              f"Siril {shared['b']:.2f} px (IQR {shared['b_iqr'][0]:.2f}-{shared['b_iqr'][1]:.2f})   "
              f"median per-star ratio {shared['ratio_a_over_b']:.3f}   n={shared['n']}")
    else:
        print("too few matched, unsaturated, isolated stars to compare")
    print(f"(stars detected: OriginStack {os_stars}, Siril {si_stars}; not comparable as a sharpness figure)")
    if rows:
        print("\nOriginStack noise / Siril noise (R, G, B); rows after 0 blur OriginStack's stack, for reference only:")
        for sigma, (fw, r) in rows.items():
            print(f"  blur sigma {sigma:<4} FWHM {fw:5.2f}   {r[0]:.2f} {r[1]:.2f} {r[2]:.2f}")
    with open(os.path.join(args.workdir, "results.json"), "w") as fh:
        json.dump({"lights": len(lights), "originstack": {"wall": os_wall, "phases": os_phases, "fwhm": os_fwhm,
                                                          "stars": os_stars, **os_res},
                   "siril": {"wall": si_wall, "stages": si_stages, "fwhm": si_fwhm, "stars": si_stars, **si_res},
                   "noise": {str(k): v[1] for k, v in (rows or {}).items()}}, fh, indent=2)


if __name__ == "__main__":
    main()
