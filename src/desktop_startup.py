"""Start-up and shut-down helpers shared by the Qt and tkinter desktop windows
(no GUI toolkit imported here): the log folder, the fatal-error surface, the headless
verification run and the hard exit after the window closes."""
from __future__ import annotations

import datetime
import os
import sys
import traceback
from pathlib import Path
from typing import List


def _log_dir() -> Path:
    r"""Per-user log folder: %LOCALAPPDATA%\OriginStack\logs on Windows, ~/Library/Logs
    on macOS, $XDG_STATE_HOME (default ~/.local/state) elsewhere -- never the current
    directory, which for a packaged app is wherever the launcher happened to start it."""
    if sys.platform == 'win32':
        return Path(os.environ.get('LOCALAPPDATA', '.')) / 'OriginStack' / 'logs'
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Logs' / 'OriginStack'
    state = os.environ.get('XDG_STATE_HOME') or str(Path.home() / '.local' / 'state')
    return Path(state) / 'OriginStack' / 'logs'


def _log_startup_status() -> None:
    """Unconditional (not just on failure) one-line startup record --
    packaging/verify_build.ps1 greps this to confirm astro_native (not the
    numpy fallback) loaded in the packaged build, now that there's no
    ``GET /api/health`` to poll instead."""
    try:
        from src.utils import native_status, read_version
        log_dir = _log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / 'desktop_app.log', 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] "
                    f"OriginStack {read_version()} starting -- {native_status()}\n")
    except OSError:
        pass


def _fatal(title: str, message: str) -> int:
    """Last-resort error surface: log full details, show a native dialog."""
    try:
        _log_dir().mkdir(parents=True, exist_ok=True)
        with open(_log_dir() / 'desktop_app_crash.log', 'a', encoding='utf-8') as f:
            f.write(f"\n[{datetime.datetime.now().isoformat()}] {title}\n{message}\n")
            f.write(traceback.format_exc() + '\n')
    except OSError:
        pass
    from src.native_dialog import show_error
    show_error(title, message)
    return 1


def _run_headless(cli_argv: List[str]) -> int:
    """``--verify-headless <cli args...>``: skip the GUI/mainloop entirely and
    run a real stack through this exact frozen entry point, forwarding the
    rest of argv straight to ``cli.parse_args`` unchanged (same syntax as
    ``originstack.py``). Exists for packaging/verify_build.ps1, which needs
    to trigger a real multi-worker run through the packaged exe to prove
    ``multiprocessing.freeze_support()`` above still works -- the old
    dashboard's ``POST /api/start`` did this over HTTP; there's no server to
    do it through anymore, so this is a direct in-process replacement."""
    from src.cli import apply_post_parse_setup, parse_args, process_directory
    try:
        args = parse_args(cli_argv)
        apply_post_parse_setup(args)
        process_directory(args.directory, args.output, args)
        return 0
    except (Exception, SystemExit) as e:
        print(f"--verify-headless run failed: {e}")
        return 1


def _shutdown_and_exit(code: int = 0, wait_s: float = 5.0) -> None:
    """Leave the process for certain once the window has closed.

    A normal interpreter exit with a run still going tore Python down while the
    run thread was inside native code -- a crash on exit in one of two tests
    (in the packaged app, a "stopped working" dialog or a process that does not
    go away) -- and it waits for any non-daemon thread and for the stacking
    code's thread pools. Instead: cancel the run and give it ``wait_s`` to stop
    at a checkpoint, stop Phase 1's worker processes, delete this run's temp
    files when nothing is using them any more, then exit at once."""
    import multiprocessing
    try:
        from src.desktop_control import get_run_manager
        rm = get_run_manager()
        if rm.is_running():
            rm.cancel()
            if rm.thread is not None:
                rm.thread.join(timeout=wait_s)
        run_stopped = not rm.is_running()
    except Exception:
        run_stopped = False
    try:
        for p in multiprocessing.active_children():
            p.terminate()
    except Exception:
        pass
    try:
        import psutil
        for child in psutil.Process().children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
    except Exception:
        pass
    if not run_stopped:
        # with its workers gone, a run waiting on them fails at once
        try:
            if rm.thread is not None:
                rm.thread.join(timeout=wait_s)
            run_stopped = not rm.is_running()
        except Exception:
            pass
    if run_stopped:
        # with the run still inside a phase its memory-mapped files are in use; a
        # later run's start-up sweep removes them instead
        try:
            from src.cleanup import cleanup_now
            cleanup_now()
        except Exception:
            pass
    try:
        import logging
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)
