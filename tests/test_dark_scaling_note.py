"""The dark-scaling note is printed once per run, by the main process, not by every Phase 1
worker for every exposure it meets."""
from src.cli import _note_dark_scaling
from src.models import FrameInfo


def _lights(*exps):
    return [FrameInfo(path=f"l{i}.fits", type="light", header={"EXPTIME": e}) for i, e in enumerate(exps)]


def test_one_line_lists_every_scaled_exposure(capsys):
    _note_dark_scaling({"light": _lights(20, 20, 25, 30, 30, 30)}, 10.0)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert "10 s dark" in out[0]
    assert "20 s (2.00x, 2 frames)" in out[0]
    assert "25 s (2.50x, 1 frame)" in out[0]
    assert "30 s (3.00x, 3 frames)" in out[0]


def test_silent_when_exposures_match_or_dark_exposure_unknown(capsys):
    _note_dark_scaling({"light": _lights(10, 10.2)}, 10.0)       # within 5%
    _note_dark_scaling({"light": _lights(20)}, None)
    _note_dark_scaling({"light": [FrameInfo(path="x", type="light", header={})]}, 10.0)
    assert capsys.readouterr().out == ""


def test_workers_no_longer_print_it():
    import inspect

    import src.frame_processor as fp
    assert "NOTE: dark scaling" not in inspect.getsource(fp)
    assert not hasattr(fp, "_warned_dark_scales")
