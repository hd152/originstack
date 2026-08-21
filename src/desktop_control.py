"""Desktop-app control layer: turns a submitted form into a real pipeline run.

``get_form_schema()`` introspects ``cli.build_parser()`` at request time
rather than duplicating its ~120 flags into a hand-maintained schema file --
the parser's ``help=``/``choices``/``default`` are already the source of
truth, and a static copy would drift every time a flag is added (this repo
adds them often). ``build_argv_from_form()`` turns a submitted ``{dest:
value}`` form back into a synthetic argv and feeds it through the real
``cli.parse_args()``, so every existing preset/``--config``/denoiser-mapping/
export-string-parsing code path -- and ``_explicit_cli_dests`` -- comes out
exactly as it would from a real command line, with nothing reimplemented
here.
"""
from __future__ import annotations

import argparse
import threading
from typing import Any, Dict, List, Optional

# A handful of path-typed fields the auto-rendered form can't infer a picker
# kind for from argparse metadata alone.
_WIDGET_HINTS: Dict[str, str] = {
    'directory': 'dir',
    'cal_dir': 'dir',
    'export_frames_dir': 'dir',
    'output': 'file-save',
    'vignette_map': 'file-open',
    'config': 'file-open',
    'log_file': 'file-save',
    'quality_report': 'file-save',
    'astap_path': 'file-open',
    'hdr_combine': 'file-open',
}

# Flags the desktop app doesn't drive a run through (they either watch/loop
# forever with no cancel support yet, or exit without stacking).
_UNSUPPORTED_DESTS = {'live', 'stream', 'quality_sweep', 'sweep_undo'}


def _field_for_action(action: argparse.Action) -> Optional[Dict[str, Any]]:
    if action.dest in ('help', argparse.SUPPRESS) or action.dest in _UNSUPPORTED_DESTS:
        return None
    flag = next((s for s in action.option_strings if s.startswith('--')),
                (action.option_strings or [None])[0])
    if flag is None:  # positional -- none exist in this parser today
        return None

    cls = type(action).__name__
    if cls == '_StoreTrueAction':
        kind, default = 'bool_true', bool(action.default)
    elif cls == '_StoreFalseAction':
        kind, default = 'bool_false', bool(action.default)
    elif action.choices:
        kind, default = 'select', action.default
    elif cls == '_AppendAction' or action.nargs in ('+', '*'):
        kind, default = 'list', action.default
    elif action.type is float:
        kind, default = 'number', action.default
    elif action.type is int:
        kind, default = 'number', action.default
    else:
        kind, default = 'text', action.default

    return {
        'dest': action.dest,
        'flag': flag,
        'kind': kind,
        'choices': list(action.choices) if action.choices else None,
        'default': default,
        'help': action.help or '',
        'widget': _WIDGET_HINTS.get(action.dest),
    }


def get_form_schema() -> Dict[str, Any]:
    """``{group_title: [field, ...]}`` for every group in ``cli.build_parser()``."""
    from src.cli import build_parser
    parser = build_parser()
    schema: Dict[str, Any] = {}
    for group in parser._action_groups:
        if not group.title or group.title in ('positional arguments', 'options'):
            continue
        fields = [f for f in (_field_for_action(a) for a in group._group_actions)
                   if f is not None]
        if fields:
            schema[group.title] = fields
    return schema


def _dest_action_map(parser: argparse.ArgumentParser) -> Dict[str, argparse.Action]:
    return {a.dest: a for a in parser._actions if a.dest not in ('help', argparse.SUPPRESS)}


def build_argv_from_form(form: Dict[str, Any]) -> List[str]:
    """Turn a ``{dest: value}`` form dict into a synthetic argv for
    ``cli.parse_args()``. Only keys present (and non-``None``) in *form* are
    emitted -- an untouched field is equivalent to never having passed that
    flag on the CLI, so it keeps its argparse default and stays out of
    ``_explicit_cli_dests``."""
    from src.cli import build_parser
    parser = build_parser()
    actions = _dest_action_map(parser)

    argv: List[str] = []
    for dest, value in form.items():
        if value is None or dest in _UNSUPPORTED_DESTS:
            continue
        action = actions.get(dest)
        if action is None:
            continue  # unknown/stale field name -- ignore rather than fail the whole run
        flag = next((s for s in action.option_strings if s.startswith('--')),
                    (action.option_strings or [None])[0])
        if flag is None:
            continue

        cls = type(action).__name__
        if cls == '_StoreTrueAction':
            if bool(value):
                argv.append(flag)
        elif cls == '_StoreFalseAction':
            if not bool(value):
                argv.append(flag)
        elif cls == '_AppendAction' or action.nargs in ('+', '*'):
            tokens = value if isinstance(value, list) else \
                [t.strip() for t in str(value).split(',') if t.strip()]
            if not tokens:
                continue
            if cls == '_AppendAction':
                for t in tokens:
                    argv.extend([flag, t])
            else:
                argv.append(flag)
                argv.extend(tokens)
        else:
            s = str(value)
            if s == '':
                continue
            argv.extend([flag, s])
    return argv


class RunManager:
    """Runs one pipeline job at a time on a background thread, publishing
    progress through the ``UIEvents`` singleton (``src/ui_events.py``) --
    the desktop app attaches it before entering the tkinter mainloop, so
    it's already active by the time a run starts; only the pipeline work
    itself needs to move off the GUI's own (main) thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = 'idle'
        self.thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self.status == 'running'

    def start(self, form: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self.status == 'running':
                return {'ok': False, 'error': 'a run is already in progress'}
            try:
                argv = build_argv_from_form(form)
            except Exception as e:
                return {'ok': False, 'error': f'invalid form: {e}'}
            self.status = 'running'
        self.thread = threading.Thread(target=self._run, args=(argv,),
                                       name='desktop-run', daemon=True)
        self.thread.start()
        return {'ok': True}

    def _run(self, argv: List[str]) -> None:
        import os
        import tempfile
        from src.cli import parse_args, process_directory, apply_post_parse_setup
        from src.utils import get_logger, safe_print
        from src.ui_events import get_ui_events

        wv = get_ui_events()
        status, error = 'ok', None
        try:
            args = parse_args(argv)

            # Default a durable log file for GUI-triggered runs specifically
            # (not in apply_post_parse_setup, which is also the plain CLI's
            # path and shouldn't start writing files a CLI user never asked
            # for) -- a desktop-app run has no attached console, so without
            # this the only record of a failure is the ephemeral in-memory
            # UIEvents state, lost on the next run or when the window closes.
            if not getattr(args, 'log_file', None):
                args.log_file = os.path.join(tempfile.gettempdir(), 'originstack_desktop_app.log')

            # Same post-parse setup main() applies (output default, --config,
            # --preset, logging, GPU context) -- RunManager calls
            # process_directory() directly, bypassing main() itself.
            apply_post_parse_setup(args)

            wv.run_started()
            process_directory(args.directory, args.output, args)
        except (Exception, SystemExit) as e:
            status = 'error'
            error = str(e) or e.__class__.__name__
            safe_print(f"  ERROR: run failed: {error}")
            get_logger().exception(f"desktop app run failed: {error}")
        finally:
            wv.run_finished(status, error)
            with self._lock:
                self.status = status


# Constructed eagerly at import time, matching get_gpu()/get_ui_events()'s
# module-level singleton pattern -- Start-button clicks are serialized on
# tkinter's own main-thread event loop, so unlike the old HTTP dashboard
# (concurrent POST /api/start requests on separate request-handling
# threads) there's no construction race to guard against here. Kept eager
# anyway: RunManager() has no expensive setup, so it costs nothing.
_run_manager: RunManager = RunManager()


def get_run_manager() -> RunManager:
    return _run_manager
