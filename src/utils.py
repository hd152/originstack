"""Utility functions for printing and formatting."""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import Optional

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


def setup_logging(level: str = 'WARNING', log_file: Optional[str] = None) -> logging.Logger:
    """Configure the 'originstack' logger hierarchy.

    Sets up a named logger so all modules can emit structured log records
    through a single hierarchy rather than using bare ``logging.warning()``.
    The console handler only emits WARNING+ by default; the optional file
    handler captures everything at DEBUG level for post-run diagnostics.

    Args:
        level: Minimum severity shown on the console ('DEBUG', 'INFO',
               'WARNING', 'ERROR').  Does not affect the file handler.
        log_file: Optional path to write a full DEBUG-level log.  Created
                  (or appended to) each run.
    """
    log_level = getattr(logging, level.upper(), logging.WARNING)
    logger = logging.getLogger('originstack')
    logger.setLevel(logging.DEBUG)  # capture everything; handlers filter

    if not logger.handlers:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(log_level)
        ch.setFormatter(logging.Formatter('%(levelname)s [%(module)s]: %(message)s'))
        logger.addHandler(ch)

    if log_file:
        # Remove any existing file handler before adding a new one
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
                h.close()
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-8s [%(name)s.%(module)s]: %(message)s'))
        logger.addHandler(fh)

    return logger


def get_logger() -> logging.Logger:
    """Return the package-level 'originstack' logger."""
    return logging.getLogger('originstack')


def safe_print(text: str):
    """Print text with fallback for unicode characters on Windows."""
    # Tee into the desktop app's log pane (no-op unless attached).
    try:
        from src.ui_events import get_ui_events
        get_ui_events().log(text)
    except Exception:
        pass
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: replace unicode symbols with ASCII
        text = text.replace('✓', '[OK]')
        text = text.replace('✗', '[X]')
        text = text.replace('⚠', '[!]')
        text = text.replace('ℹ', '[i]')
        text = text.replace('─', '-')
        text = text.replace('→', '->')
        text = text.replace('×', 'x')
        text = text.replace('Δ', 'd')
        text = text.replace('≠', '!=')
        text = text.replace('–', '-')
        text = text.replace('—', '--')
        text = text.replace('σ', 'sigma')
        text = text.replace('κ', 'kappa')
        text = text.replace('γ', 'gamma')
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', errors='replace').decode('ascii'))


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    safe_print(f"\n{char * 70}")
    safe_print(text)
    safe_print(f"{char * 70}")


def print_quality_table(frames, show_all: bool = False):
    """Print a formatted table of frame quality metrics."""
    if not frames:
        return

    # Filter to only frames with metrics
    frames_with_metrics = [f for f in frames if f.metrics and 'score' in f.metrics]
    if not frames_with_metrics:
        return

    # Header
    safe_print("\n  Frame Quality Details:")
    safe_print("  " + "─" * 110)
    safe_print(f"  {'Frame':<30} {'Bright':>8} {'Bg':>8} {'Noise':>7} {'SNR':>5} "
               f"{'Stars':>6} {'FWHM':>6} {'Sharp':>8} {'Score':>10} {'St':>3}")
    safe_print("  " + "─" * 110)

    for i, f in enumerate(frames_with_metrics):
        if not show_all and len(frames_with_metrics) > 20 and i == 10:
            safe_print(f"  {'...':<30} {'...':>8} {'...':>8} {'...':>7} {'...':>5} "
                       f"{'...':>6} {'...':>6} {'...':>8} {'...':>10} {'...':>3}")
            continue
        elif not show_all and len(frames_with_metrics) > 20 and 10 < i < len(frames_with_metrics) - 10:
            continue

        name = os.path.basename(f.path)
        if len(name) > 30:
            name = name[:27] + "..."

        m = f.metrics
        brightness  = m.get('brightness', 0)
        background  = m.get('background', 0)
        noise       = m.get('noise', 0)
        snr         = m.get('snr', 0)
        stars       = m.get('star_count', 0)
        fwhm        = m.get('fwhm', 0)
        sharpness   = m.get('sharpness', 0)
        score       = m.get('score', 0)
        status      = "✓" if f.accepted else "✗"

        safe_print(f"  {name:<30} {brightness:8.1f} {background:8.1f} {noise:7.2f} {snr:5.1f} "
                   f"{stars:6} {fwhm:6.1f} {sharpness:8.0f} {score:10.1f} {status:>3}")

    safe_print("  " + "─" * 110)


def print_phase(phase_num: int, title: str):
    """Print a phase header."""
    try:
        from src.ui_events import get_ui_events
        wv = get_ui_events()
        wv.phase(phase_num, title)
        wv.log(f"PHASE {phase_num}: {title.upper()}")
    except Exception:
        pass
    print(f"\n{'=' * 70}")
    print(f"PHASE {phase_num}: {title.upper()}")
    print(f"{'=' * 70}")


def read_version() -> str:
    """Reads the app VERSION file. Checks sys._MEIPASS first (PyInstaller's
    onedir frozen-bundle root, where the spec copies VERSION alongside the
    exe), then the repo root, for a source checkout / `python desktop_app.py`
    dev run. Returns 'dev' if neither exists, so a plain checkout with no
    VERSION file (the pre-packaging state) keeps working unchanged."""
    import sys
    from pathlib import Path
    candidates = []
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        candidates.append(Path(meipass) / 'VERSION')
    candidates.append(Path(__file__).resolve().parent.parent / 'VERSION')
    for p in candidates:
        try:
            return p.read_text(encoding='utf-8').strip()
        except OSError:
            continue
    return 'dev'


def native_status() -> str:
    """One-line status of the optional native (Rust) acceleration."""
    try:
        import astro_native
        ver = getattr(astro_native, '__version__', '?')
        # dir() on the compiled extension module includes non-kernel noise
        # (a self-referential 'astro_native' entry among them) -- count only
        # actual callables, not every non-underscore attribute name.
        n_fns = len([f for f in dir(astro_native)
                    if not f.startswith('_') and callable(getattr(astro_native, f, None))])
        return (f"Native accel: astro_native v{ver} ACTIVE - {n_fns} Rust kernels "
                f"(stacking combine, Lanczos warp, aniso diffusion)")
    except Exception:
        return ("Native accel: not installed - using numpy fallback "
                "(build ext/astro_native for ~5-37x on stacking/registration)")


def should_check_for_update() -> bool:
    """Gate for the self-update check (CLI at startup, desktop app on launch): off
    if $ORIGINSTACK_NO_UPDATE_CHECK is set (any value), or if --offline already put
    net_query into its no-network mode for this process. The check itself
    (src.net_query.check_for_update) is a single unauthenticated GET to GitHub's
    public releases API and fails silent on any error -- this only decides whether
    it's attempted at all, so a truly offline run never even tries the socket."""
    if os.environ.get('ORIGINSTACK_NO_UPDATE_CHECK'):
        return False
    try:
        from src.net_query import is_offline
        return not is_offline()
    except Exception:
        return True


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def embed_to_shape(arr, H: int, W: int):
    """Place ``arr`` in the top-left of an ``(H, W[, C])`` zero canvas (crop
    if larger). Pixel coordinates are preserved, so a transform computed on
    the embedded array applies directly to it.

    Shared by ``merge.py`` (a previous stack's own shape rarely matches the
    current run's) and ``difference_imaging.py`` (two independently-stacked
    sessions of the same target routinely differ in pixel dimensions --
    different dither pattern, different Phase 3 common-crop -- even though
    they're the same field). A no-op (identity, no copy) when the shape
    already matches.
    """
    import numpy as np
    if arr.shape[0] == H and arr.shape[1] == W:
        return arr
    if arr.ndim == 3:
        out = np.zeros((H, W, arr.shape[2]), dtype=arr.dtype)
    else:
        out = np.zeros((H, W), dtype=arr.dtype)
    h = min(H, arr.shape[0])
    w = min(W, arr.shape[1])
    out[:h, :w] = arr[:h, :w]
    return out


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024**2
    return 0.0


def disable_astropy_network() -> None:
    """Stop astropy silently reaching for the network mid-run.

    astropy refreshes its IERS earth-orientation table over HTTP when the
    bundled one looks stale. That turns an offline or firewalled machine into
    a multi-second socket timeout per mirror, inside functions that then fail
    soft and report nothing -- and it fires during a stack, not at startup.
    This project's ephemeris math is closed-form precisely to avoid depending
    on those tables (see ``sky_model``), so there is nothing to refresh.

    No-op when astropy is absent.
    """
    try:
        from astropy.utils import iers
        iers.conf.auto_download = False
    except Exception:
        pass


def parse_timestamp(when: str):
    """Parse an ISO-8601-ish timestamp to a naive UTC ``datetime``, or None.

    **Offsets are converted, never stripped.** A Celestron Origin
    ``info.json`` and its FITS headers stamp local time with a numeric offset
    (``2026-08-31T20:40:32-0700``). Failing to parse that at all silently
    disables every feature keyed on observation time; merely discarding the
    ``-0700`` puts the timestamp seven hours out, which moves the moon most of
    the way across the sky. A naive timestamp is assumed to already be UTC.

    Serves both ``sky_model.julian_date`` and ``observing_geometry``, which
    each carried their own copy of this normalisation.
    """
    import datetime as _dt

    if not when:
        return None
    text = str(when).strip()

    dt = None
    # fromisoformat covers 'Z', '+HH:MM' and (3.11+) '+HHMM' in one shot.
    try:
        dt = _dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        naive = text.replace('Z', '').replace('T', ' ')
        for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d %H:%M', '%Y-%m-%d'):
            try:
                dt = _dt.datetime.strptime(naive, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return dt


# A FITS TIMEZONE value that can be appended to DATE-OBS: '-0700', '+05:30'.
# Anything else ('PDT', 'US/Pacific') would make an unparseable timestamp.
TZ_OFFSET_RE = re.compile(r'[+-]\d{2}:?\d{2}')

_DATE_OBS_KEYS = ('DATE-OBS', 'DATE_OBS', 'DATEOBS')


def obs_time_utc_iso(header=None, fallback=None) -> Optional[str]:
    """Observation time as a naive-UTC ISO string, or None.

    Celestron Origin lights stamp ``DATE-OBS`` in *local* time with the
    offset in a separate ``TIMEZONE`` keyword; ``parse_timestamp`` treats an
    offset-less timestamp as UTC, so reading DATE-OBS alone put every
    time-dependent result seven hours out (a zenith angle of 150 deg -- below
    the horizon -- for a target at 72 deg). A valid TIMEZONE is appended when
    DATE-OBS carries no offset of its own. ``fallback`` (e.g. an info.json
    ``dateTime`` like ``2026-08-31T20:40:32-0700``) is used when the header
    has no DATE-OBS. The result has no offset, so astropy ``Time`` parses it
    too (it rejects ``-0700``).
    """
    when = header_get_first(header, _DATE_OBS_KEYS, cast=str) if header is not None else None
    if when:
        when = when.strip()
        tz = str(header.get('TIMEZONE', '') or '').strip()
        has_time = len(when) > 10          # a bare date takes no offset
        if (has_time and tz and TZ_OFFSET_RE.fullmatch(tz)
                and not TZ_OFFSET_RE.search(when[10:]) and not when.endswith('Z')):
            when += tz
    else:
        when = fallback
    dt = parse_timestamp(when) if when else None
    return dt.isoformat() if dt is not None else None


def header_get_first(header, keys, cast=None, default=None):
    """First present, non-None value among ``keys`` in a FITS-header-like
    mapping (anything with ``.get``). With ``cast`` given, the value is run
    through it and a failing cast is treated as absent. Returns ``default``
    when nothing matches.

    Folds the recurring "try each of these header spellings" pattern
    (``DATE-OBS``/``DATE_OBS``/``DATEOBS``, ``EGAIN``/``GAIN``,
    ``SATURATE``/``DATAMAX``, the ``CCD-TEMP`` family, ...).
    """
    if header is None or not hasattr(header, "get"):
        return default
    for key in keys:
        val = header.get(key)
        if val is None:
            continue
        if cast is None:
            return val
        try:
            return cast(val)
        except (TypeError, ValueError):
            continue
    return default


def mp_context():
    """The multiprocessing context every process pool here must use: ``spawn``.

    Linux's default start method (``fork`` before Python 3.14) copies the parent after
    it has already started native threads -- the Rust kernels' rayon pool, OpenBLAS --
    into children that inherit that pool's state but none of its threads, so the first
    parallel call in a worker waits forever (seen: a packaged Linux build hung in Phase 1
    with four idle workers). Windows only ever had ``spawn``; using it everywhere makes the
    platforms behave alike, at the cost of each worker importing the stack once."""
    import multiprocessing
    return multiprocessing.get_context("spawn")
