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
import re
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
    'from_stack': 'file-open',
}

# Flags the desktop app doesn't drive a run through (they either watch/loop
# forever with no cancel support yet, or exit without stacking).
_UNSUPPORTED_DESTS = {'live', 'stream', 'quality_sweep', 'sweep_undo'}


# One-line descriptions shown under a field in the desktop form, written for
# a photographer rather than for the command line. Only fields whose help
# text does not summarise well get one here: the help leads with a bare
# heading ("Processing preset."), describes the *negating* flag of a checkbox
# ("Disable the auto advisor"), or is too technical for the form. Every other
# field's line comes from summarize_help(). Keys must be real dests
# (tests/test_desktop_setup_form.py checks).
_FIELD_SUMMARIES: Dict[str, str] = {
    'directory': 'The folder with your light frames (calibration frames can sit alongside).',
    'output': 'Where to save the stack. Leave blank to save next to the light frames.',
    'preset': 'A starting point for the settings: quick, quality, or tuned for a target type.',
    'auto': 'Recognises the target after the first pass and picks settings for it. '
            'Anything you change here still wins.',
    'stack_method': 'How frames are combined and outliers (satellites, cosmic rays) rejected. '
                    'auto picks for the session.',
    'denoiser': 'Noise reduction applied to the finished stack. auto picks for the target.',
    'deconvolve_mode': 'Sharpen fine detail by undoing the blur of seeing and optics. '
                       'Best on bright, high-SNR targets.',
    'drizzle_scale': 'Upscale the output (e.g. 2 for twice the resolution). '
                     'Needs many dithered frames.',
    'trail_reject': 'Find and erase satellite and aircraft trails in each frame.',
    'use_gpu': 'Use an NVIDIA GPU (CuPy) where it helps. Usually no faster on small cards.',
    'parallel': 'Frames processed at once. 0 = automatic, 1 = one at a time.',
    'originvision': 'Score a few frames with the bundled image classifier '
                    '(advisory; nudges --auto).',
    'debayer_method': 'How colour is rebuilt from the sensor mosaic. rcd is sharpest and quietest.',
    'white_balance': 'How the colour balance of each frame is set before stacking.',
    'verbose': 'Print more detail in the log.',
    'no_registration': 'Stack frames without aligning them (only for already-aligned data).',
    'directional_protect_strength': 'How strongly noise reduction protects filaments and '
                                    'spiral arms (0-1).',
    'skip_step': 'Turn off named post-processing steps (comma-separated).',
    'plate_solve': 'Work out exactly where in the sky the image points '
                   '(needed for labels and photometry).',
    'annotate': 'Save a copy of the preview with stars and named objects labelled '
                '(needs internet).',
    'photometry': 'Measure star brightnesses against Gaia and save a calibrated catalogue (CSV).',
    'export': 'Extra file formats to save alongside the FITS: tiff, xisf.',
    'stretch': 'How the preview JPEG is brightened from the linear stack.',
    'merge': 'Add earlier stacks of the same target to this one (their linear FITS files).',
    'session_cfa_eq': 'Measure sensor colour offsets once per session (faster). '
                      'Untick to re-measure on every frame.',
}

_ABBREV_END = re.compile(r'\b(e\.g|i\.e|vs|etc|approx|cf)$', re.IGNORECASE)
_NEGATION = re.compile(r"^(disable|do not|don't|turn off|skip)\s+", re.IGNORECASE)


def summarize_help(help_text: str, kind: str, limit: int = 120) -> str:
    """First sentence of an argparse help string, trimmed for display under a
    form field. Stops at the first top-level '.', ';' or ':' (not inside
    parentheses, not after "e.g.", and not at a short leading heading such
    as "Bayer-aware drizzle:"), drops a dangling "(default: ..." and caps the
    length. A checkbox for a store_false flag shows the *enabled* state, so
    its "Disable X" help becomes "X."; any other wording is dropped rather
    than shown backwards next to a ticked box."""
    text = ' '.join((help_text or '').split())
    if not text:
        return ''
    end, depth = len(text), 0
    for i, ch in enumerate(text):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        elif depth == 0 and ch in '.;:' and (i + 1 == len(text) or text[i + 1] == ' '):
            head = text[:i]
            if ch == '.' and _ABBREV_END.search(head):
                continue
            if ch == ':' and len(head) < 40:
                continue
            end = i
            break
    s = text[:end].strip()
    # The input already shows the default: drop "(default: x)" and an
    # unbalanced trailing "(default: ..." cut off by the sentence end.
    s = re.sub(r'\s*\(default:[^)]*\)', '', s)
    # Pointers into the source ("(see foo in src/debayer.py)") mean nothing here.
    s = re.sub(r'\s*\([^)]*\.py\b[^)]*\)', '', s)
    s = re.sub(r'\s*\([^)]*$', '', s).strip()
    if kind == 'bool_false':
        m = _NEGATION.match(s)
        if not m:
            return ''
        s = s[m.end():]
        s = s[:1].upper() + s[1:]
    if len(s) > limit:
        s = s[:limit].rsplit(' ', 1)[0].rstrip(',;:-') + '…'
    if s and not s.endswith(('.', '…', '?')):
        s += '.'
    return s


def _field_for_action(action: argparse.Action) -> Optional[Dict[str, Any]]:
    if action.dest in ('help', argparse.SUPPRESS) or action.dest in _UNSUPPORTED_DESTS:
        return None
    if action.help is argparse.SUPPRESS:  # hidden/deprecated flags stay out of the form
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
        'summary': _FIELD_SUMMARIES.get(action.dest) or summarize_help(action.help or '', kind),
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


def _default_output_beside_lights(form: Dict[str, Any]) -> Dict[str, Any]:
    """A blank Output file means "next to the light-frames folder". The CLI's
    default is ``<session>_stacked.fits`` in the *working directory*, which
    for a desktop app is wherever it was launched from (the source checkout,
    or the install folder of the packaged build). Passing the folder's parent
    as ``-o`` uses the CLI's folder-output mode: ``<session>_stacked.fits``
    there, never overwriting an existing stack."""
    import os
    directory = str(form.get('directory') or '').strip()
    if str(form.get('output') or '').strip() or not directory or form.get('from_stack'):
        return form
    parent = os.path.dirname(os.path.abspath(directory.rstrip('/\\')))
    return {**form, 'output': os.path.join(parent, '')}


class RunManager:
    """Runs one pipeline job at a time on a background thread, publishing
    progress through the ``UIEvents`` singleton (``src/ui_events.py``) --
    the desktop app attaches it before entering the tkinter mainloop, so
    it's already active by the time a run starts; only the pipeline work
    itself needs to move off the GUI's own (main) thread.

    Cancellation is cooperative, not a thread kill (CPython threads can't be
    force-stopped, and Phase 1 workers are real OS processes/subprocesses
    mid-computation anyway): ``cancel()`` sets a ``threading.Event`` shared
    onto the run's ``args`` as ``_cancel_event``, and the pipeline notices it
    at a handful of checkpoints (``frame_processor._check_cancel`` between
    Phase 1 frames -- usually the longest phase -- and between targets in
    ``cli.process_directory``). Phases 2-4 of a single target aren't
    interruptible yet: once one starts, it runs to completion."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = 'idle'
        self.thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()

    def is_running(self) -> bool:
        return self.status == 'running'

    def is_cancelling(self) -> bool:
        return self.status == 'running' and self._cancel_event.is_set()

    def cancel(self) -> None:
        """Request a stop. A no-op if nothing is running; safe to call more
        than once. Takes effect at the next checkpoint, not instantly."""
        if self.status == 'running':
            self._cancel_event.set()

    def start(self, form: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self.status == 'running':
                return {'ok': False, 'error': 'a run is already in progress'}
            form = _default_output_beside_lights(form)
            try:
                argv = build_argv_from_form(form)
            except Exception as e:
                return {'ok': False, 'error': f'invalid form: {e}'}
            self.status = 'running'
            self._cancel_event = threading.Event()  # fresh flag per run
        self.thread = threading.Thread(target=self._run, args=(argv,),
                                       name='desktop-run', daemon=True)
        self.thread.start()
        return {'ok': True}

    def _run(self, argv: List[str]) -> None:
        import os
        import tempfile

        from src.cli import apply_post_parse_setup, parse_args, process_directory
        from src.models import RunCancelled
        from src.ui_events import get_ui_events
        from src.utils import get_logger, safe_print

        wv = get_ui_events()
        status, error = 'ok', None
        try:
            args = parse_args(argv)
            args._cancel_event = self._cancel_event

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
            if getattr(args, 'from_stack', None):
                from src.cli import save_effective_config
                from src.pipeline import postprocess_from_stack
                postprocess_from_stack(args.from_stack, args.output, args)
                save_effective_config(args, args.output)
            else:
                process_directory(args.directory, args.output, args)
        except RunCancelled:
            status = 'cancelled'
            safe_print("  Run cancelled.")
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
