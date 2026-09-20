"""``-o`` given a folder generates the FITS (and so the JPG) name inside it."""
import os

from src.cli import _name_output_in_folder, _output_is_folder, apply_post_parse_setup, parse_args


def test_folder_detection(tmp_path):
    assert _output_is_folder(str(tmp_path))                       # existing dir
    assert _output_is_folder(str(tmp_path / "new") + os.sep)      # trailing separator
    assert _output_is_folder(str(tmp_path / "newfolder"))         # no extension
    assert not _output_is_folder(str(tmp_path / "x.fits"))
    assert not _output_is_folder("stack.tif")


def test_generated_name_uses_session_and_never_overwrites(tmp_path):
    session = tmp_path / "Fireworks_Galaxy"
    session.mkdir()
    out = tmp_path / "out"
    first = _name_output_in_folder(str(out), str(session))
    assert first == os.path.join(str(out), "Fireworks_Galaxy_stacked.fits") and out.is_dir()
    open(first, "w").close()
    second = _name_output_in_folder(str(out), str(session))
    assert second.endswith("Fireworks_Galaxy_stacked_2.fits")
    open(os.path.join(str(out), "Fireworks_Galaxy_stacked_2.jpg"), "w").close()   # a stray JPG counts too
    assert _name_output_in_folder(str(out), str(session)).endswith("_stacked_3.fits")


def test_post_parse_setup_rewrites_folder_output(tmp_path):
    session = tmp_path / "M31"
    session.mkdir()
    out = tmp_path / "results"
    args = parse_args(["-d", str(session), "-o", str(out)])
    apply_post_parse_setup(args)
    assert args.output == os.path.join(str(out), "M31_stacked.fits")


def test_explicit_file_is_left_alone(tmp_path):
    session = tmp_path / "M31"
    session.mkdir()
    target = str(tmp_path / "mine.fits")
    args = parse_args(["-d", str(session), "-o", target])
    apply_post_parse_setup(args)
    assert args.output == target
