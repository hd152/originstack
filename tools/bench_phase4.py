"""Phase 4 image-quality harness: scores the image a user actually gets.

``bench_vs_siril.py`` scores the *linear* stack and ``bench_denoise_quality.py``
scores denoisers in isolation, before the stretch. Neither sees the delivered
picture -- the stretched preview, the only place Phase 4's work shows (the main
FITS stays linear). This runs Phase 4 in-process on linear stacks (exactly as
``--from-stack`` does, with each stack's ``<stem>_config.toml``) plus a
synthetic scene with known truth, renders the display stretch as float, and
scores it.

Metrics (per input; ``D`` = display image in [0, 1], ``P`` = Phase 4 output
before the stretch, ``L`` = linear input stack):

  fwhm_ratio     median per-star FWHM of P / L on the same isolated, unsaturated
                 stars (``common_star_fwhm``). > 1 means Phase 4 widened stars.
  colour_slope   Theil-Sen slope of each star's display colour index
                 (-2.5 log10 B/R, aperture photometry on D) against the same
                 star's colour index in L. 1 = colour spread kept; toward 0 =
                 stars rendered toward white.
  colour_rho     Spearman correlation of the same pairs.
  hue_err_deg    median angle between the RGB vector of D and of (P - sky) on
                 bright pixels (luminance excess > 30 sky sigma). What the
                 stretch alone does to hue.
  sky_noise      robust sigma of D's high-passed luminance on sky, 8-bit units.
  faint_cnr      (median D on faint signal - median D on sky) / sky_noise.
                 Faint = 2-10 sigma (3 px smoothed) above sky in L. Means, not medians.
  ring           median over bright isolated stars of the deepest annulus
                 (1.5-5 FWHM) of D's luminance below the 6-10 FWHM background,
                 in sky_noise units. Negative = dark moat.
  blotch         std of D's chroma (R-G, B-G) smoothed at 12 px over sky,
                 8-bit units: large-scale colour mottle.
  flatness       p98 - p2 of D's luminance smoothed at 24 px over sky, 8-bit.
  gaia_rho, gaia_solar_ci
                 with a WCS: Spearman of display colour index vs Gaia BP-RP,
                 and the display colour index a solar-colour star (BP-RP
                 0.82) gets from the robust line fit -- 0 means a G2V star
                 renders white, the usual colour-calibration target. The Gaia
                 query is cached next to the stack (``<stem>_gaia.json``).

The synthetic scene adds ``*_truth`` versions of colour_slope and hue_err_deg,
measured against the noise-free truth instead of L / P.

Usage:
  python tools/bench_phase4.py STACK.fits [...] [--synthetic] \\
      [--variant NAME="--flag value ..."] [--json out.json] [--crops DIR]
A variant's string is extra command-line flags appended after the stack's
config (explicit flags win, as on the real command line); ``@key=value``
tokens set a config-only setting after parsing (``@aniso_kappa_sigma=1.0``).
Linear stacks go through ``pipeline.prepare_linear_for_phase4`` first, as with
``--from-stack``. ``base`` (no extra
flags) is always run.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import sys
import time
from typing import Dict, Optional

import numpy as np
from scipy import ndimage, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

LUMA = np.array([0.299, 0.587, 0.114])
SOLAR_BP_RP = 0.82


# ---------------------------------------------------------------------------
# Running Phase 4
# ---------------------------------------------------------------------------

def run_phase4(stacked: np.ndarray, extra: list, stack_path: Optional[str],
               out_dir: str, name: str, synth_config: Optional[str] = None,
               header=None):
    """Phase 4 + display stretch on ``stacked`` with the CLI's own argument
    handling. Returns (P, D, args)."""
    from src import cli
    from src.io_fits import render_preview_float
    from src.models import ProcessingStats
    from src.postprocess import postprocess_stack

    out = os.path.join(out_dir, f"{name}.fits")
    sets = [t[1:].split("=", 1) for t in extra if t.startswith("@")]
    extra = [t for t in extra if not t.startswith("@")]
    if stack_path:
        argv = ["--from-stack", stack_path, "-o", out, *extra]
    else:
        argv = ["--from-stack", os.path.join(out_dir, "synthetic.fits"), "-o", out,
                *(["--config", synth_config] if synth_config else []), *extra]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        args = cli.parse_args(argv)
        cli.apply_post_parse_setup(args)
        for k, v in sets:
            setattr(args, k, type(getattr(args, k))(v) if getattr(args, k, None) is not None
                    else float(v))
        if stack_path:
            args._pp_early_cache = os.path.splitext(stack_path)[0] + "_phase4cache.pkl"
        stacked = stacked.copy()
        if header is not None:
            from src.pipeline import prepare_linear_for_phase4
            prepare_linear_for_phase4(args, stacked, header)
        P = postprocess_stack(stacked, args, [], ProcessingStats())
        D = render_preview_float(P, stretch=getattr(args, "stretch", "ghs"),
                                 ghs_b=float(getattr(args, "ghs_b", 8.0)),
                                 ghs_sp=float(getattr(args, "ghs_sp", 0.15)),
                                 ghs_hp=float(getattr(args, "ghs_hp", 0.95)),
                                 black_sigma=float(getattr(args, "preview_black_sigma", 0.0)),
                                 color=getattr(args, "stretch_color", "preserve"))
    return P, np.asarray(D, dtype=np.float64), args, buf.getvalue()


# ---------------------------------------------------------------------------
# Masks and star lists from the linear input
# ---------------------------------------------------------------------------

def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else 1.0


def regions(L: np.ndarray) -> dict:
    """Sky / faint / bright masks and a star list, all from the linear input."""
    from src.star_detect import detect_stars_matched_filter

    lum = L @ LUMA
    sm = ndimage.gaussian_filter(lum, 3.0)
    bg = ndimage.median_filter(sm[::8, ::8], size=15, mode="nearest")
    bg = ndimage.zoom(bg, (lum.shape[0] / bg.shape[0], lum.shape[1] / bg.shape[1]), order=1)
    bg = bg[:lum.shape[0], :lum.shape[1]]
    ex = sm - bg
    s_sm = _robust_sigma(ex[::3, ::3])
    s_px = _robust_sigma((lum - sm)[::3, ::3])
    src = detect_stars_matched_filter(lum.astype(np.float32))
    xs = np.asarray(src["xcentroid"], float)
    ys = np.asarray(src["ycentroid"], float)
    flux = np.asarray(src["flux"], float)
    starm = np.zeros(lum.shape, bool)
    ok = (ys >= 0) & (ys < lum.shape[0]) & (xs >= 0) & (xs < lum.shape[1])
    starm[ys[ok].astype(int), xs[ok].astype(int)] = True
    starm = ndimage.binary_dilation(starm, iterations=8)
    sky = (ex < 1.0 * s_sm) & ~starm
    faint = (ex > 2.0 * s_sm) & (ex < 10.0 * s_sm) & ~starm
    bright = (lum - bg) > 30.0 * s_px
    return dict(lum=lum, bg=bg, sky=sky, faint=faint, bright=bright, xs=xs, ys=ys,
                flux=flux, s_px=s_px)


def _isolated(xs, ys, order, min_sep, limit):
    keep = []
    for i in order:
        d2 = (xs - xs[i]) ** 2 + (ys - ys[i]) ** 2
        d2[i] = np.inf
        if d2.min() >= min_sep ** 2:
            keep.append(i)
        if len(keep) >= limit:
            break
    return np.array(keep, int)


def _colour_index(img: np.ndarray, xs, ys, fwhm: float) -> np.ndarray:
    """-2.5 log10(B/R) by aperture photometry; NaN where a flux is <= 0."""
    from src.photometry_core import aperture_photometry_batch
    r = max(2.0, 1.5 * fwhm)
    flux = aperture_photometry_batch(np.ascontiguousarray(img, dtype=np.float32),
                                     xs, ys, r, r + 3, r + 8)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        ci = -2.5 * np.log10(flux[:, 2] / flux[:, 0])
    ci[~np.isfinite(ci)] = np.nan
    return ci


def _slope_rho(a: np.ndarray, b: np.ndarray):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float("nan"), float("nan"), ok.sum()
    sl = stats.theilslopes(b[ok], a[ok])
    rho = stats.spearmanr(a[ok], b[ok]).statistic
    return float(sl[0]), float(rho), int(ok.sum())


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    c = np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# ---------------------------------------------------------------------------
# Gaia
# ---------------------------------------------------------------------------

def gaia_for(header, shape, cache_path: str):
    """(x, y, bp_rp) of Gaia stars in the field, cached as JSON."""
    if os.path.isfile(cache_path):
        with open(cache_path) as fh:
            d = json.load(fh)
        return np.array(d["x"]), np.array(d["y"]), np.array(d["bp_rp"])
    from src import net_query
    from src.photometry_core import _field_centre_and_radius, _pixel_coords
    fc = _field_centre_and_radius(header, shape)
    if fc is None:
        return None
    ra, dec, rad, _ = fc
    try:
        tab = net_query.gaia_cone_search(ra, dec, rad, ["ra", "dec", "phot_g_mean_mag", "bp_rp"],
                                         max_rows=2000, require_not_null=["bp_rp"])
    except Exception:
        return None
    if tab is None or len(tab) == 0:
        return None
    xy = _pixel_coords(tab, header)
    if xy is None:
        return None
    bp = np.asarray(tab["bp_rp"], float)
    with open(cache_path, "w") as fh:
        json.dump({"x": xy[:, 0].tolist(), "y": xy[:, 1].tolist(), "bp_rp": bp.tolist()}, fh)
    return xy[:, 0], xy[:, 1], bp


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(L, P, D, reg, fwhm, gaia=None, truth=None) -> Dict[str, float]:
    from common_star_fwhm import common_star_fwhm

    m: Dict[str, float] = {}
    r = common_star_fwhm(np.transpose(P, (2, 0, 1)), np.transpose(L, (2, 0, 1)))
    m["fwhm_ratio"] = r["ratio_a_over_b"] if r else float("nan")

    dl = D @ LUMA
    hp = dl - ndimage.gaussian_filter(dl, 2.0)
    sky_noise = _robust_sigma(hp[reg["sky"]]) * 255
    m["sky_noise"] = sky_noise
    # Means, not medians: with a positive black point the sky clips to 0 and so
    # can the faint region's median.
    m["faint_cnr"] = (float(np.mean(dl[reg["faint"]]) - np.mean(dl[reg["sky"]])) * 255
                      / max(sky_noise, 1e-9)) if reg["faint"].any() else float("nan")

    # Star colours: bright, isolated, unsaturated-in-L stars
    xs, ys, flux = reg["xs"], reg["ys"], reg["flux"]
    H, W = dl.shape
    inside = (xs > 20) & (xs < W - 20) & (ys > 20) & (ys < H - 20)
    order = np.argsort(-flux)
    order = order[inside[order]]
    sel = _isolated(xs, ys, order, 6 * fwhm, 300)
    ci_L = _colour_index(L, xs[sel], ys[sel], fwhm)
    ci_D = _colour_index(D, xs[sel], ys[sel], fwhm)
    m["colour_slope"], m["colour_rho"], m["n_colour"] = _slope_rho(ci_L, ci_D)

    # Hue on bright pixels: stretch alone
    sky_c = np.array([np.median(P[..., c][reg["sky"]]) for c in range(3)])
    b = reg["bright"]
    lin = np.clip(P[b] - sky_c, 0, None)
    m["hue_err_deg"] = float(np.median(_angle_deg(lin, D[b]))) if b.any() else float("nan")

    # Rings around bright stars
    sel_r = _isolated(xs, ys, order, 12 * fwhm, 40)
    yy, xx = np.mgrid[0:H, 0:W]
    rings = []
    for i in sel_r:
        x0, y0 = xs[i], ys[i]
        R = int(10 * fwhm) + 2
        ya, yb = int(max(y0 - R, 0)), int(min(y0 + R + 1, H))
        xa, xb = int(max(x0 - R, 0)), int(min(x0 + R + 1, W))
        rr = np.hypot(yy[ya:yb, xa:xb] - y0, xx[ya:yb, xa:xb] - x0) / fwhm
        cut = dl[ya:yb, xa:xb]
        outer = cut[(rr >= 6) & (rr < 10)]
        if outer.size < 20:
            continue
        bgv = np.median(outer)
        prof = [np.mean(cut[(rr >= a) & (rr < a + 0.5)]) for a in np.arange(1.5, 5.0, 0.5)]
        rings.append((min(prof) - bgv) * 255 / max(sky_noise, 1e-9))
    m["ring"] = float(np.median(rings)) if rings else float("nan")

    # Large-scale colour mottle and flatness over sky
    w = ndimage.gaussian_filter(reg["sky"].astype(float), 12) + 1e-9
    blot = []
    for a, bb in ((0, 1), (2, 1)):
        ch = ndimage.gaussian_filter((D[..., a] - D[..., bb]) * reg["sky"], 12) / w
        blot.append(np.std(ch[reg["sky"]]))
    m["blotch"] = float(np.hypot(*blot)) * 255
    w24 = ndimage.gaussian_filter(reg["sky"].astype(float), 24) + 1e-9
    fl = ndimage.gaussian_filter(dl * reg["sky"], 24) / w24
    v = fl[reg["sky"]]
    m["flatness"] = float(np.percentile(v, 98) - np.percentile(v, 2)) * 255

    if gaia is not None:
        gx, gy, bp = gaia
        ok = (gx > 20) & (gx < W - 20) & (gy > 20) & (gy < H - 20)
        gx, gy, bp = gx[ok], gy[ok], bp[ok]
        # Keep catalogue stars that coincide with a detection (1.5 px) and are isolated
        from scipy.spatial import cKDTree
        t = cKDTree(np.c_[xs, ys])
        d, j = t.query(np.c_[gx, gy])
        hit = d < 1.5
        gx, gy, bp = xs[j[hit]], ys[j[hit]], bp[hit]
        o = np.argsort(-flux[j[hit]])
        keep = _isolated(gx, gy, o, 6 * fwhm, 300)
        ci = _colour_index(D, gx[keep], gy[keep], fwhm)
        okc = np.isfinite(ci)
        if okc.sum() >= 8:
            sl, ic = stats.theilslopes(ci[okc], bp[keep][okc])[:2]
            m["gaia_rho"] = float(stats.spearmanr(bp[keep][okc], ci[okc]).statistic)
            m["gaia_solar_ci"] = float(ic + sl * SOLAR_BP_RP)
            m["n_gaia"] = int(okc.sum())

    if truth is not None:
        ci_T = _colour_index(truth, xs[sel], ys[sel], fwhm)
        m["colour_slope_truth"], _, _ = _slope_rho(ci_T, ci_D)
        tb = truth[b] - np.median(truth[reg["sky"]], axis=0)
        m["hue_err_truth"] = float(np.median(_angle_deg(np.clip(tb, 0, None), D[b])))
    return m


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def synthetic_case(out_dir: str):
    """A galaxy scene with a sky gradient, through the denoise bench's
    mosaic -> noise -> debayer -> dither-stack model, written as a linear
    RAWSTACK so it goes through exactly the --from-stack path."""
    import bench_denoise_quality as bdq
    from astropy.io import fits
    bdq.SIZE = 768
    case = bdq.make_case("galaxy", sigma_mosaic=20.0, seed=7, scale=60.0)
    h, w = case["noisy"].shape[:2]
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    grad = (40 * xx + 25 * yy)[..., None] * np.array([1.2, 1.0, 0.8])
    noisy = (case["noisy"] + grad).astype(np.float32)
    truth = case["truth"] + grad
    path = os.path.join(out_dir, "synthetic.fits")
    hdr = fits.Header()
    hdr["RAWSTACK"] = True
    hdr["NFRAMES"] = 6
    fits.PrimaryHDU(np.transpose(noisy, (2, 0, 1)), header=hdr).writeto(path, overwrite=True)
    return path, noisy.astype(np.float64), truth


def _fwhm_of(L: np.ndarray, reg) -> float:
    from common_star_fwhm import _gauss_fit
    lum = reg["lum"]
    order = np.argsort(-reg["flux"])[:200]
    f = [r[0] for i in order if (r := _gauss_fit(lum, reg["xs"][i], reg["ys"][i])) is not None
         and 1.0 < r[0] < 14]
    return float(np.median(f)) if f else 4.0


def save_crop(D: np.ndarray, path: str):
    from PIL import Image
    Image.fromarray(np.clip(D * 255, 0, 255).astype(np.uint8)).save(path, quality=95)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stacks", nargs="*")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-config",
                    help="_config.toml the synthetic scene runs with (e.g. a real galaxy "
                         "run's, so it gets --auto's galaxy choices)")
    ap.add_argument("--variant", action="append", default=[],
                    help='NAME="extra flags" (repeatable)')
    ap.add_argument("--work", default="bench_phase4_out")
    ap.add_argument("--json")
    ap.add_argument("--crops", action="store_true", help="write each display image as JPEG")
    a = ap.parse_args()

    from astropy.io import fits

    from src.merge import load_merge_stack
    from src.utils import disable_astropy_network
    disable_astropy_network()
    os.makedirs(a.work, exist_ok=True)

    variants = {"base": []}
    for v in a.variant:
        k, _, flags = v.partition("=")
        variants[k] = shlex.split(flags)

    inputs = [(os.path.splitext(os.path.basename(p))[0], p) for p in a.stacks]
    synth = None
    if a.synthetic:
        sp, sL, sT = synthetic_case(a.work)
        synth = (sL, sT)
        inputs.insert(0, ("synthetic", None))

    results = {}
    for name, path in inputs:
        if path is None:
            L, truth, header = synth[0], synth[1], None
        else:
            L = load_merge_stack(path)[0].astype(np.float64)
            truth, header = None, fits.getheader(path)
        reg = regions(L)
        fwhm = _fwhm_of(L, reg)
        gaia = None
        if header is not None:
            gaia = gaia_for(header, L.shape, os.path.splitext(path)[0] + "_gaia.json")
        for vname, extra in variants.items():
            t0 = time.time()
            P, D, _args, log = run_phase4(L.astype(np.float32), extra, path, a.work,
                                          f"{name}__{vname}", a.synthetic_config, header)
            m = score(L, P.astype(np.float64), D, reg, fwhm, gaia=gaia, truth=truth)
            m["seconds"] = round(time.time() - t0, 1)
            sc = getattr(_args, "_colcal_scales", None)
            if sc:
                m["colcal_r"], m["colcal_b"] = round(sc[0], 3), round(sc[2], 3)
            results.setdefault(name, {})[vname] = m
            if a.crops:
                save_crop(D, os.path.join(a.work, f"{name}__{vname}.jpg"))
            print(f"{name:28s} {vname:14s} " + " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()),
                flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
