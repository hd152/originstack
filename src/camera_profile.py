"""Camera profiles: per-model constants measured once, and a per-unit library.

The Celestron Origin writes its camera into every FITS header as
``CAMERA = 'Origin178-932f562381e97ce70'`` -- a model (``Origin178``) and a unit
ID. Several things this pipeline otherwise re-measures on every session, or takes
from a header that is wrong, are properties of that model or of that one unit:

* **Gain** (model, per ISO). The header's ``EGAIN`` is ~4.7x too high
  (0.0742 e-/ADU at ISO 200 against 0.0158 measured). ``measure_raw_gain`` is the
  photon-transfer two-point estimate on raw lights; the shipped table is its median
  over every session on disk (tools/measure_camera_profile.py), and a run checks it
  on two of its own lights before trusting it (``verify_gain``).
* **CFA equalisation prior** (model, per debayer method) -- see ``cfa_prior``.
* **Colour response** (model) -- see ``colour_slopes``.
* **Bad pixels and vignetting** (unit) -- ``CameraLibrary``, learned from the
  unit's own sessions under ``camera_library_dir()``.

Everything here is a *prior*: a profile that does not match the data is reported
and ignored, never forced. A header without a recognised camera changes nothing.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_log = logging.getLogger("originstack")

_PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'camera_profiles')
_CAMERA_RE = re.compile(r'^\s*([A-Za-z][A-Za-z0-9_]*?)-([0-9a-fA-F]{8,})\s*$')


@dataclass(frozen=True)
class CameraId:
    model: str                    # 'Origin178'
    unit: Optional[str]           # '932f562381e97ce70', or None when only the model is known
    iso: Optional[int]
    firmware: Optional[str]       # 'Origin 1.4.6084'

    @property
    def label(self) -> str:
        return f"{self.model}" + (f" #{self.unit[:6]}" if self.unit else "")


def identify(header) -> Optional[CameraId]:
    """The camera behind a light frame's header, or None for an unknown one."""
    if header is None:
        return None
    try:
        cam = header.get('CAMERA')
    except Exception:
        return None
    if not cam:
        return None
    m = _CAMERA_RE.match(str(cam))
    model, unit = (m.group(1), m.group(2).lower()) if m else (str(cam).strip(), None)
    iso = header.get('ISOSPEED')
    try:
        iso = int(round(float(iso))) if iso is not None else None
    except (TypeError, ValueError):
        iso = None
    fw = header.get('CREATOR')
    return CameraId(model=model, unit=unit, iso=iso, firmware=str(fw).strip() if fw else None)


_PROFILE_CACHE: dict = {}


def load_profile(model: str) -> Optional[dict]:
    """The shipped profile of ``model`` (src/data/camera_profiles/<model>.json)."""
    if model in _PROFILE_CACHE:
        return _PROFILE_CACHE[model]
    prof = None
    path = os.path.join(_PROFILE_DIR, f"{model}.json")
    if re.fullmatch(r'[A-Za-z0-9_]+', model or '') and os.path.isfile(path):
        try:
            with open(path, encoding='utf-8') as f:
                prof = json.load(f)
        except (OSError, ValueError) as e:
            _log.warning("camera profile %s unreadable: %s", path, e)
    _PROFILE_CACHE[model] = prof
    return prof


def profile_for(cam: Optional[CameraId], shape=None) -> Optional[dict]:
    """The profile for ``cam`` when it exists and the frame geometry matches it."""
    if cam is None:
        return None
    prof = load_profile(cam.model)
    if prof is None:
        return None
    if shape is not None and tuple(prof.get('shape', ())) and tuple(shape[:2]) != tuple(prof['shape']):
        return None
    return prof


def _iso_entry(prof: dict, iso: Optional[int]) -> Optional[dict]:
    if prof is None or iso is None:
        return None
    return (prof.get('iso') or {}).get(str(int(iso)))


# ── Colour ───────────────────────────────────────────────────────────────────

def colour_slopes(camera: Optional[dict]):
    """(slope_BR, slope_GR) of instrumental colour against Gaia BP-RP for the camera
    ``resolve_camera`` found (``args._camera``), or None. Used by the solar colour
    fit only when a field has too few stars for a free fit."""
    if not camera:
        return None
    prof = load_profile(camera.get('model', ''))
    col = (prof or {}).get('colour') or {}
    try:
        return float(col['slope_br']), float(col['slope_gr'])
    except (KeyError, TypeError, ValueError):
        return None


# ── Gain ─────────────────────────────────────────────────────────────────────

def _clipped_var(d: np.ndarray, k: float = 4.0, iters: int = 4) -> float:
    """Variance after iterative k-sigma clipping. Not a MAD: Origin raw values come
    in coarse steps (tens of ADU), and a MAD of stepped data jumps between a few
    discrete values, which made a variance-vs-signal slope mostly quantisation."""
    d = d[np.isfinite(d)]
    if d.size < 100:
        return float('nan')
    m, s = float(np.median(d)), float(np.std(d))
    for _ in range(iters):
        sel = np.abs(d - m) < k * s
        if sel.sum() < 100:
            break
        m, s = float(d[sel].mean()), float(d[sel].std())
    # a k-sigma clip of a Gaussian keeps the core: undo its variance loss
    from math import erf, exp, pi, sqrt
    loss = erf(k / sqrt(2)) - 2 * k * exp(-k * k / 2) / sqrt(2 * pi)
    return s * s / loss


def raw_pair_gain(a: np.ndarray, b: np.ndarray, pattern: str, pedestal: float,
                  floor_var=None, border: int = 8,
                  min_signal: float = 1500.0) -> Optional[np.ndarray]:
    """Per-channel (R, G, B) gain in e-/ADU from two consecutive raw lights.

    Two-point photon transfer: in F1 - F2 everything static cancels (sky structure,
    vignetting, the sensor's fixed pattern), so var(F1 - F2) / 2 is the temporal
    noise, and gain = (signal - pedestal) / variance. Read and dark-current noise are
    left in the variance; on Origin skies (thousands of ADU above the pedestal) they
    are < 1% of it, and bins less than ``min_signal`` above the pedestal are not used.
    Sky-end signal bins only (the 5-60th percentile), where stars are rare."""
    from src.noise_model import _BIN_PCTS, _channel_sites
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if a.shape != b.shape or a.ndim != 2:
        return None
    a = a[border:-border, border:-border]
    b = b[border:-border, border:-border]
    out = np.full(3, np.nan)
    for c in range(3):
        sites = _channel_sites(pattern, c)
        if not sites:
            return None
        ds, ss = [], []
        for dy, dx in sites:
            pa = a[dy::2, dx::2].astype(np.float64)
            pb = b[dy::2, dx::2].astype(np.float64)
            d = pa - pb
            ds.append((d - np.median(d[::3, ::3])).ravel())
            ss.append((0.5 * (pa + pb)).ravel())
        d, s = np.concatenate(ds), np.concatenate(ss)
        q = np.percentile(s[::7], _BIN_PCTS)
        g = []
        for lo, hi in zip(q[:-1], q[1:]):
            sel = (s >= lo) & (s < hi)
            if sel.sum() < 1000:
                continue
            sig = float(np.median(s[sel])) - pedestal
            var = _clipped_var(d[sel]) / 2.0
            if floor_var is not None:
                var -= float(floor_var[c])
            if sig > min_signal and var > 0:
                g.append(sig / var)
        if g:
            out[c] = float(np.median(g))
    return out if np.isfinite(out).any() else None


def measure_raw_gain(light_paths: Sequence[str], pattern: str, pedestal: float,
                     n_pairs: int = 6, floor_var=None) -> Optional[np.ndarray]:
    """Median per-channel gain over up to ``n_pairs`` consecutive light pairs spread
    through ``light_paths`` (time order). None when nothing could be measured."""
    from src.io_fits import load_frame
    n = len(light_paths)
    if n < 2:
        return None
    starts = sorted({int((j + 0.5) * (n - 1) / n_pairs) for j in range(min(n_pairs, n - 1))})
    gs = []
    for i in starts:
        try:
            a = load_frame(light_paths[i])[0]
            b = load_frame(light_paths[i + 1])[0]
        except Exception:
            continue
        g = raw_pair_gain(a, b, pattern, pedestal, floor_var=floor_var)
        if g is not None:
            gs.append(g)
    if not gs:
        return None
    return np.nanmedian(np.array(gs), axis=0)


def table_gain(prof: Optional[dict], iso: Optional[int]) -> Optional[float]:
    """The shipped e-/ADU of one raw ADU at ``iso`` (all channels share it)."""
    e = _iso_entry(prof, iso)
    try:
        g = float(e['gain_e_per_adu']) if e else None
    except (KeyError, TypeError, ValueError):
        return None
    return g if g and g > 0 else None


def table_floor(prof: Optional[dict], iso: Optional[int]):
    e = _iso_entry(prof, iso)
    try:
        f = [float(v) for v in e['noise_floor_var_adu2']] if e else None
    except (KeyError, TypeError, ValueError):
        return None
    return f if f and len(f) == 3 else None


def table_pedestal(prof: Optional[dict], iso: Optional[int]) -> Optional[float]:
    e = _iso_entry(prof, iso)
    try:
        return float(e['pedestal_adu']) if e and 'pedestal_adu' in e else None
    except (TypeError, ValueError):
        return None


GAIN_TOLERANCE = 0.15     # a session gain further than this from the table is not trusted


def verify_gain(prof: dict, iso: int, light_paths: Sequence[str], pattern: str,
                pedestal: Optional[float] = None):
    """``(table_gain, measured, ok)``: the table value checked on one consecutive pair
    of this session's own lights (the middle two). ``ok`` is False when the table has
    no entry, or the measurement lands further than ``GAIN_TOLERANCE`` from it -- a
    firmware change or another camera mode would show up that way. ``measured`` is
    None when the pair could not be measured, and the table is then used unchecked."""
    g_tab = table_gain(prof, iso)
    if g_tab is None:
        return None, None, False
    ped = pedestal if pedestal is not None else table_pedestal(prof, iso)
    if ped is None or len(light_paths) < 2:
        return g_tab, None, True
    m = len(light_paths) // 2
    meas = measure_raw_gain(list(light_paths)[m - 1:m + 1], pattern, ped, n_pairs=1,
                            floor_var=table_floor(prof, iso))
    if meas is None or not np.isfinite(meas).any():
        return g_tab, None, True
    g_meas = float(np.nanmedian(meas))
    return g_tab, g_meas, abs(g_meas / g_tab - 1.0) <= GAIN_TOLERANCE


def resolve_camera(lights, args, verify: bool = False) -> Optional[dict]:
    """Identify the camera of a target's lights and, with ``verify``, check the
    profile's gain on two of them. Stores the result on ``args._camera`` (``model``,
    ``unit``, ``iso``, ``gain`` -- None unless the profile has one that the session
    did not contradict -- and ``gain_measured``) and returns it; None for a camera
    without a profile. Prints one line."""
    from src.utils import safe_print
    args._camera = None
    if not lights:
        return None
    hdr = getattr(lights[0], 'header', None)
    cam = identify(hdr)
    try:
        shape = (int(hdr.get('NAXIS2')), int(hdr.get('NAXIS1')))
    except Exception:
        shape = None
    prof = profile_for(cam, shape)
    if prof is None:
        return None
    info = {'model': cam.model, 'unit': cam.unit, 'iso': cam.iso, 'gain': None,
            'gain_measured': None, 'label': cam.label}
    g_tab = table_gain(prof, cam.iso)
    msg = f"  Camera: {cam.label}, ISO {cam.iso}"
    if g_tab is None:
        msg += " -- no profiled gain at this ISO"
    elif not verify:
        info['gain'] = g_tab
        msg += f", gain {g_tab:.4f} e-/ADU (profile)"
    else:
        pattern = str(hdr.get('BAYERPAT') or prof.get('bayer') or 'RGGB').strip().upper()
        _, g_meas, ok = verify_gain(prof, cam.iso, [f.path for f in lights], pattern)
        info['gain_measured'] = g_meas
        if ok:
            info['gain'] = g_tab
            msg += f", gain {g_tab:.4f} e-/ADU (profile" + (
                f"; this session {g_meas:.4f})" if g_meas is not None else ", unchecked)")
        else:
            msg += (f" -- profile gain {g_tab:.4f} e-/ADU does not match this session's "
                    f"{g_meas:.4f}; not used")
    egain = None
    try:
        egain = float(hdr.get('EGAIN')) if hdr.get('EGAIN') is not None else None
    except (TypeError, ValueError):
        pass
    if info['gain'] and egain and abs(egain / info['gain'] - 1.0) > 0.5:
        msg += f"; header EGAIN {egain:.4f} ignored"
    safe_print(msg)
    args._camera = info
    return info
