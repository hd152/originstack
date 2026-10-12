"""Pipeline progress bars must not fail in the console-less desktop process."""

import io

from tqdm import tqdm

from src.utils import progress_disabled


def test_progress_bar_is_disabled_when_stderr_is_missing(monkeypatch):
    import src.utils as utils

    monkeypatch.setattr(utils.sys, 'stderr', None)
    assert progress_disabled() is True
    assert list(tqdm([1, 2], disable=progress_disabled())) == [1, 2]


def test_progress_bar_remains_enabled_for_a_writable_stderr(monkeypatch):
    import src.utils as utils

    monkeypatch.setattr(utils.sys, 'stderr', io.StringIO())
    assert progress_disabled() is False
    assert progress_disabled(verbose=True) is True


def test_progress_bar_is_disabled_for_a_closed_stderr(monkeypatch):
    import src.utils as utils

    stderr = io.StringIO()
    stderr.close()
    monkeypatch.setattr(utils.sys, 'stderr', stderr)

    assert progress_disabled() is True
