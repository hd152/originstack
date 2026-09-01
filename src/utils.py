"""Utility functions for printing and formatting."""
from __future__ import annotations

import logging
import os
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


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024**2
    return 0.0


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
