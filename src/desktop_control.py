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


def load_app_setting(key: str, default=None):
    """A value saved in the app's settings file (the one that holds the language)."""
    import json

    from src.i18n import _settings_path
    try:
        data = json.loads(_settings_path().read_text(encoding='utf-8'))
        return data.get(key, default) if isinstance(data, dict) else default
    except (OSError, ValueError):
        return default


def save_app_setting(key: str, value) -> None:
    import json

    from src.i18n import _settings_path
    path = _settings_path()
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data[key] = value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    except OSError:
        pass


# A handful of path-typed fields the auto-rendered form can't infer a picker
# kind for from argparse metadata alone.
_WIDGET_HINTS: Dict[str, str] = {
    'directory': 'dir',
    'cal_dir': 'dir',
    'export_frames_dir': 'dir',
    # A folder is what most people want (the stack is named for them and never
    # overwrites); a full .fits path can still be typed in.
    'output': 'dir',
    'temp_dir': 'dir',
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
    'output': 'Folder to save the stack in (or type a full .fits path). Leave blank to save it '
              'beside the light-frames folder, in the folder that contains it.',
    'temp_dir': 'Where temporary files go while stacking (about 170 MB per light frame). '
                'Pick a drive with plenty of free space for long sessions.',
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
    'export': ('Extra file formats to save alongside the FITS: tiff, xisf, png (a lossless '
               'copy of the preview), png16 (the preview in 16 bits, for further editing).'),
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
    """``{dest: action}``; where a hidden alias shares a dest with the visible option
    (``--no-full-field``, ``--no-comet-mode``), the visible one. Last-wins put the
    hidden alias there: the comet card stopped emitting ``--comet-mode``, and an edited
    full-field fraction became ``--no-full-field 0.5``."""
    out: Dict[str, argparse.Action] = {}
    for a in parser._actions:
        if a.dest in ('help', argparse.SUPPRESS):
            continue
        if a.dest in out and a.help is argparse.SUPPRESS:
            continue
        out[a.dest] = a
    return out


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
        # The output file of the last run as the pipeline resolved it (a blank
        # Output, or a folder, becomes <session>_stacked.fits somewhere), for the
        # app's Open Folder button.
        self.last_output: Optional[str] = None

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
        args = None
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
            # Temp files of this run, whatever happened: the app's process lives on,
            # so the at-exit cleanup alone left a failed run's files (tens of GB)
            # behind until the app was closed.
            try:
                from src.cleanup import cleanup_now
                cleanup_now()
            except Exception:
                pass
            if args is not None and getattr(args, 'output', None):
                self.last_output = os.path.abspath(str(args.output))
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


# ── "Report a problem": a pre-filled GitHub issue ──────────────────────
#
# Nothing is sent from the app: the URL opens the issue form in the browser,
# where the user reads (and can edit) the text before choosing to submit it.

ISSUES_URL = 'https://github.com/hd152/originstack/issues/new'
# "Support OriginStack": the Ko-fi page, opened in the browser. The same address is in
# .github/FUNDING.yml (GitHub's Sponsor button), the README and the website;
# tests/test_support_link.py keeps them in step.
SUPPORT_URL = 'https://ko-fi.com/originstack'
# GitHub refuses issue URLs much past 8 KB; keep the whole URL under this.
_REPORT_URL_MAX = 7000


def _redact_home(text: str) -> str:
    """The home folder as ``~`` (it usually holds the account name)."""
    import os
    home = os.path.expanduser('~')
    if not home or home == '~':
        return text
    for form in {home, home.replace('\\', '/'), home.replace('/', '\\')}:
        text = re.sub(re.escape(form), '~', text, flags=re.IGNORECASE)
    return text


def _report_parts(log_text: str, error: Optional[str], n_lines: int):
    """Title, the body's head (what happened, system, error) and the end of the
    log (progress runs folded to their last line), shared by the issue URL and
    the copied report."""
    import platform
    import sys

    from src.utils import native_status, read_version
    version = read_version()
    head = ['**What happened?**', '', '<!-- What you did, what you expected, what you saw. -->', '', '',
            '**System**', '',
            f'- OriginStack {version}' + (' (packaged app)' if getattr(sys, 'frozen', False) else ''),
            f'- {platform.platform()}, Python {platform.python_version()}',
            f'- {native_status()}']
    if error:
        head += ['', '**Error**', '', '```', _redact_home(error.strip())[:1500], '```']
    title = f'[{version}] ' + (_redact_home(error.strip().splitlines()[0])[:80] if error else '')
    lines = [ln.rstrip() for ln in _redact_home(log_text or '').splitlines() if ln.strip()]
    # a progress run ("Combining: 969/1964 rows", ...): only its last line
    stem = [re.sub(r'\d.*', '', ln) for ln in lines]
    lines = [ln for i, ln in enumerate(lines)
             if i + 1 == len(lines) or stem[i + 1] != stem[i] or not re.search(r'\d+/\d+', ln)]
    return title, head, lines[-n_lines:]


def _report_body(head: List[str], lines: List[str], note: str) -> str:
    body = head[:]
    if lines:
        body += ['', f'**Log** (last {len(lines)} lines)', '', '```', *lines, '```']
    return '\n'.join(body + ['', note])


def problem_report_url(log_text: str = '', error: Optional[str] = None) -> str:
    """The issue-form URL: version, OS, Python and native status, the last
    run's error, and as much of the end of the log as fits."""
    from urllib.parse import urlencode

    title, head, lines = _report_parts(log_text, error, 80)
    note = ('<sub>Please check the text above before submitting: this issue is public. '
            'Remove anything you do not want to share.</sub>')

    def url(lines: List[str]) -> str:
        return ISSUES_URL + '?' + urlencode({'title': title, 'body': _report_body(head, lines, note)})

    while lines and len(url(lines)) > _REPORT_URL_MAX:
        lines = lines[max(1, len(lines) // 10):]          # drop the oldest tenth
    return url(lines)


def problem_report_text(log_text: str = '', error: Optional[str] = None) -> str:
    """The same report as plain text for the clipboard, for someone without a
    GitHub account: pasted into an email or a forum post. No URL limit, so a
    longer log tail."""
    title, head, lines = _report_parts(log_text, error, 200)
    head = ['(What you did, what you expected, what you saw.)' if ln.startswith('<!--') else ln
            for ln in head]
    note = f'(Copied from OriginStack. Bug reports: {ISSUES_URL})'
    return (f'OriginStack problem report {title}'.rstrip() + '\n\n'
            + _report_body(head, lines, note))


_OPTIONAL_DEPS = ('astro_native', 'cupy', 'rawpy', 'tifffile', 'PIL', 'psutil', 'tqdm')


def _gpu_summary() -> str:
    """CuPy and a CUDA device, without raising (no GPU is the usual case)."""
    import importlib.util
    if importlib.util.find_spec('cupy') is None:
        return 'cupy not installed'
    try:
        import cupy
        n = int(cupy.cuda.runtime.getDeviceCount())
        if n < 1:
            return 'cupy installed, no CUDA device'
        name = cupy.cuda.runtime.getDeviceProperties(0)['name']
        name = name.decode(errors='replace') if isinstance(name, bytes) else str(name)
        return f'cupy installed, {n} CUDA device(s): {name}'
    except Exception as exc:
        return f'cupy installed, CUDA unavailable ({type(exc).__name__})'


def diagnostics_text(settings: Optional[Dict[str, Any]] = None, log_text: str = '',
                     n_lines: int = 60) -> str:
    """Plain text for the clipboard: version, OS/Python, native and GPU status, which
    optional packages import, the form settings changed from their defaults (as
    ``key = value`` config lines) and the tail of the log. The home folder is
    written ``~`` throughout."""
    import importlib.util
    import platform
    import sys

    from src.utils import native_status, read_version
    deps = [f"{m}={'yes' if importlib.util.find_spec(m) is not None else 'no'}"
            for m in _OPTIONAL_DEPS]
    out = [f"OriginStack {read_version()}" + (' (packaged app)' if getattr(sys, 'frozen', False) else ''),
           f'{platform.platform()}, Python {platform.python_version()}',
           native_status(), f'GPU: {_gpu_summary()}', 'Optional packages: ' + ', '.join(deps),
           '', 'Settings changed from the defaults:']
    items = [(k, v) for k, v in sorted((settings or {}).items()) if v not in (None, '')]
    out += [f'{k} = "{v}"' if isinstance(v, str) else f'{k} = {v}' for k, v in items] or ['(none)']
    lines = [ln.rstrip() for ln in (log_text or '').splitlines() if ln.strip()][-n_lines:]
    out += ['', f'Log (last {len(lines)} lines):', *lines]
    return _redact_home('\n'.join(out))


def unprofiled_camera(lights) -> Optional[str]:
    """Name of the camera of a folder whose lights come from a camera other than a
    profiled one (the Origin178); None for a profiled camera or no readable header."""
    from src.camera_profile import identify, profile_for
    if not lights:
        return None
    hdr = getattr(lights[0], 'header', None)
    if hdr is None:
        return None
    cam = identify(hdr)
    if cam is not None:
        try:
            shape = (int(hdr.get('NAXIS2')), int(hdr.get('NAXIS1')))
        except Exception:
            shape = None
        if profile_for(cam, shape) is not None:
            return None
        name = cam.label
    else:
        name = str(hdr.get('INSTRUME') or '').strip()
        if not name:
            return None
    return name
