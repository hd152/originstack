"""Utility functions for printing and formatting."""
from __future__ import annotations

import os
from typing import List

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


def safe_print(text: str):
    """Print text with fallback for unicode characters on Windows."""
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
        print(text)


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
    safe_print("  " + "─" * 100)
    safe_print(f"  {'Frame':<30} {'Bright':>8} {'Contr':>8} {'Stars':>6} {'Score':>10} {'Status':>8}")
    safe_print("  " + "─" * 100)

    # Show first 10, last 10, or all if show_all
    if show_all or len(frames_with_metrics) <= 20:
        frames_to_show = frames_with_metrics
    else:
        frames_to_show = frames_with_metrics[:10] + frames_with_metrics[-10:]
        show_ellipsis = True

    shown_count = 0
    for i, f in enumerate(frames_with_metrics):
        if not show_all and len(frames_with_metrics) > 20 and i == 10:
            print(f"  {'...':<30} {'...':>8} {'...':>8} {'...':>6} {'...':>10} {'...':>8}")
            continue
        elif not show_all and len(frames_with_metrics) > 20 and 10 < i < len(frames_with_metrics) - 10:
            continue

        name = os.path.basename(f.path)
        if len(name) > 30:
            name = name[:27] + "..."

        brightness = f.metrics.get('brightness', 0)
        contrast = f.metrics.get('contrast', 0)
        stars = f.metrics.get('star_count', 0)
        score = f.metrics.get('score', 0)
        status = "✓" if f.accepted else "✗"

        safe_print(f"  {name:<30} {brightness:8.1f} {contrast:8.1f} {stars:6} {score:10.1f} {status:>8}")

    safe_print("  " + "─" * 100)


def print_phase(phase_num: int, title: str):
    """Print a phase header."""
    print(f"\n{'=' * 70}")
    print(f"PHASE {phase_num}: {title.upper()}")
    print(f"{'=' * 70}")


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
