"""PyInstaller runtime hook: a windowed (console=False) build has sys.stdout/
sys.stderr set to None, not just a silent stream -- the first ordinary
print()/safe_print() call anywhere in the pipeline (not just the two
documented desktop_app.py error paths) would raise
'NoneType' object has no attribute 'write' deep in a background thread.
Redirect to a discard stream before any app code runs."""
import io
import sys

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()
