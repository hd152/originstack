"""Health check report for light-frame consistency and calibration compatibility."""
from __future__ import annotations

from collections import Counter

import numpy as np

from src.models import Config
from src.utils import print_header, safe_print


def run_health_check(frames: dict, masters: dict, directory: str) -> None:
    """Print a health-check report for light-frame consistency and calibration compatibility."""
    lights = frames.get('light', [])
    darks  = frames.get('dark',  [])
    flats  = frames.get('flat',  [])
    biases = frames.get('bias',  [])
    warnings_hc: list = []

    # ── Light Frames ──────────────────────────────────────────────────────────
    print_header("LIGHT FRAMES", char='-')

    if not lights:
        safe_print("  ERROR: No light frames found.")
        print_header("HEALTH CHECK RESULT", char='-')
        safe_print("  STATUS: CANNOT STACK — no light frames found")
        return

    # Dimensions
    dim_counter = Counter(
        (f.header.get('NAXIS2'), f.header.get('NAXIS1')) for f in lights
    )
    if len(dim_counter) == 1:
        (H, W), _ = dim_counter.most_common(1)[0]
        safe_print(f"  Dimensions:    {H}×{W} px  (all {len(lights)} frames consistent)")
    else:
        safe_print(f"  Dimensions:    INCONSISTENT — {len(dim_counter)} different sizes:")
        for (H, W), cnt in dim_counter.most_common():
            safe_print(f"    {H}×{W}: {cnt} frame(s)")
        warnings_hc.append("Light frames have mixed dimensions — cannot stack mixed sizes")
    light_dims = dim_counter.most_common(1)[0][0]  # (H, W) of majority

    # Exposure time
    exptimes = [float(f.header['EXPTIME']) for f in lights if 'EXPTIME' in f.header]
    if exptimes:
        et_counter = Counter(round(e, 1) for e in exptimes)
        if len(et_counter) == 1:
            safe_print(f"  Exposure:      {list(et_counter)[0]:.1f}s  (all frames)")
        else:
            parts = ', '.join(f'{t:.1f}s ×{c}' for t, c in et_counter.most_common())
            safe_print(f"  Exposure:      mixed — {parts}")
            warnings_hc.append("Light frames have inconsistent exposure times")
    light_et = Counter(round(e, 1) for e in exptimes).most_common(1)[0][0] if exptimes else None

    # ISO / gain
    isos = [f.header.get('ISOSPEED') or f.header.get('ISO') or f.header.get('GAIN')
            for f in lights]
    isos = [i for i in isos if i is not None]
    if isos:
        iso_counter = Counter(str(i) for i in isos)
        if len(iso_counter) == 1:
            safe_print(f"  ISO:           {list(iso_counter)[0]}  (all frames)")
        else:
            parts = ', '.join(f'ISO {k} ×{v}' for k, v in iso_counter.most_common())
            safe_print(f"  ISO:           mixed — {parts}")
            warnings_hc.append("Light frames have inconsistent ISO settings")
    light_iso = Counter(str(i) for i in isos).most_common(1)[0][0] if isos else None

    # Bayer pattern
    bayerpats = [f.header.get('BAYERPAT') or f.header.get('COLORTYP') for f in lights]
    bayerpats = [b for b in bayerpats if b is not None]
    if bayerpats:
        bp_counter = Counter(str(b) for b in bayerpats)
        if len(bp_counter) == 1:
            safe_print(f"  Bayer pattern: {list(bp_counter)[0]}  (all frames)")
        else:
            parts = ', '.join(f'{k} ×{v}' for k, v in bp_counter.most_common())
            safe_print(f"  Bayer pattern: mixed — {parts}  ⚠")
            warnings_hc.append("Light frames have mixed Bayer patterns")
    else:
        safe_print("  Bayer pattern: not recorded in headers (mono or unknown)")

    # Binning
    binnings = [(f.header.get('XBINNING', 1), f.header.get('YBINNING', 1)) for f in lights
                if 'XBINNING' in f.header or 'YBINNING' in f.header]
    if binnings:
        bin_counter = Counter(binnings)
        if len(bin_counter) == 1:
            xb, yb = list(bin_counter)[0]
            safe_print(f"  Binning:       {xb}×{yb}  (all frames)")
        else:
            parts = ', '.join(f'{xb}×{yb} ×{c}' for (xb, yb), c in bin_counter.most_common())
            safe_print(f"  Binning:       mixed — {parts}  ⚠")
            warnings_hc.append("Light frames have mixed binning settings")

    # CCD temperature range
    temps = [float(f.header['CCD-TEMP']) for f in lights if 'CCD-TEMP' in f.header]
    if temps:
        t_min, t_max = min(temps), max(temps)
        tf_min = t_min * 9.0 / 5.0 + 32.0
        tf_max = t_max * 9.0 / 5.0 + 32.0
        safe_print(f"  CCD temp:      {t_min:.1f}–{t_max:.1f}°C  ({tf_min:.1f}–{tf_max:.1f}°F)")

    # Date range
    dates = sorted(f.header['DATE-OBS'] for f in lights if 'DATE-OBS' in f.header)
    if dates:
        safe_print(f"  Date range:    {dates[0][:19]}  →  {dates[-1][:19]}")

    if len(lights) < Config.MIN_RECOMMENDED_FRAMES:
        safe_print(f"  ⚠ Frame count: {len(lights)} (recommended: {Config.MIN_RECOMMENDED_FRAMES}+)")
        warnings_hc.append(f"Only {len(lights)} light frame(s) — stack quality may be poor")

    # ── Calibration compatibility ──────────────────────────────────────────────
    print_header("CALIBRATION COMPATIBILITY", char='-')

    # Dark
    if darks:
        dark_hdr    = darks[0].header
        dark_et_val = masters.get('dark_exptime')
        dark_iso_v  = dark_hdr.get('ISOSPEED') or dark_hdr.get('ISO') or dark_hdr.get('GAIN')
        dark_temp_c = dark_hdr.get('CCD-TEMP')
        dark_dims   = (dark_hdr.get('NAXIS2'), dark_hdr.get('NAXIS1'))
        issues = []
        if dark_et_val and light_et and abs(dark_et_val - light_et) > 0.5:
            issues.append(f"exposure {dark_et_val:.1f}s ≠ lights {light_et:.1f}s")
            warnings_hc.append(f"Dark exposure ({dark_et_val:.1f}s) differs from lights ({light_et:.1f}s)")
        if dark_iso_v is not None and light_iso and str(dark_iso_v) != light_iso:
            issues.append(f"ISO {dark_iso_v} ≠ lights ISO {light_iso}")
            # ISO mismatch is already printed by the existing dark analysis
        if None not in dark_dims and dark_dims != light_dims:
            issues.append(f"size {dark_dims[1]}×{dark_dims[0]} ≠ lights {light_dims[1]}×{light_dims[0]}")
            warnings_hc.append("Dark frame dimensions differ from lights")
        temp_note = ''
        if dark_temp_c is not None and temps:
            delta = dark_temp_c - float(np.mean(temps))
            temp_note = f"  (Δ{delta:+.1f}°C vs lights)"
            if abs(delta) > 10:
                issues.append(f"temp delta {delta:+.1f}°C")
                warnings_hc.append(f"Dark sensor temp differs from lights by {abs(delta):.1f}°C — consider re-taking darks")
        status = "ISSUES: " + "; ".join(issues) if issues else "OK"
        safe_print(f"  Darks  ({len(darks)} frame(s)){temp_note}:  {status}")
    else:
        safe_print("  Darks:   none  ⚠")
        warnings_hc.append("No dark frames — hot pixels and thermal noise will not be corrected")

    # Flat
    if flats:
        flat_hdr  = flats[0].header
        flat_dims = (flat_hdr.get('NAXIS2'), flat_hdr.get('NAXIS1'))
        issues = []
        if None not in flat_dims and flat_dims != light_dims:
            issues.append(f"size {flat_dims[1]}×{flat_dims[0]} ≠ lights {light_dims[1]}×{light_dims[0]}")
            warnings_hc.append("Flat frame dimensions differ from lights")
        status = "ISSUES: " + "; ".join(issues) if issues else "OK"
        safe_print(f"  Flats  ({len(flats)} frame(s)):  {status}")
    else:
        safe_print("  Flats:   none  ⚠")
        warnings_hc.append("No flat frames — vignetting and per-channel response will not be corrected")

    # Bias
    if biases:
        safe_print(f"  Bias   ({len(biases)} frame(s)):  OK")
    elif darks:
        safe_print("  Bias:   none  (dark frames correct the bias pedestal)")
    else:
        safe_print("  Bias:   none  ⚠")
        warnings_hc.append("No bias or dark frames — bias pedestal will not be subtracted")

    # ── Overall result ─────────────────────────────────────────────────────────
    print_header("HEALTH CHECK RESULT", char='-')
    if warnings_hc:
        safe_print(f"  Warnings ({len(warnings_hc)}):")
        for w in warnings_hc:
            safe_print(f"    ⚠  {w}")
        safe_print("")
    critical = any(
        keyword in w.lower()
        for w in warnings_hc
        for keyword in ("mixed dimensions", "cannot stack", "differ from lights")
        if "dimensions" in w.lower()
    )
    if "Light frames have mixed dimensions" in warnings_hc or not lights:
        safe_print("  STATUS: CANNOT STACK — critical issues must be resolved first")
    elif len(warnings_hc) == 0:
        safe_print("  STATUS: READY TO STACK")
    elif len(warnings_hc) <= 2 and not any("differ" in w or "mismatch" in w.lower() for w in warnings_hc
                                            if "dimension" in w.lower() or "exposure" in w.lower()):
        safe_print("  STATUS: READY TO STACK (minor warnings — review above)")
    else:
        safe_print("  STATUS: PROCEED WITH CAUTION — review warnings above")
