import importlib.util
from pathlib import Path

_PATH = Path(__file__).parents[1] / "tools" / "profile_stack.py"
_SPEC = importlib.util.spec_from_file_location("profile_stack", _PATH)
assert _SPEC and _SPEC.loader
profile_stack = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profile_stack)


def test_scalene_command_profiles_originstack():
    command = profile_stack._scalene_command(
        "scalene", Path("profiling") / "scalene.html", ["-d", "lights", "-j", "4"]
    )
    assert command[:6] == [
        "scalene",
        "run",
        "--profile-all",
        "--outfile",
        str(Path("profiling") / "scalene.html"),
        "---",
    ]
    assert command[6:] == [
        str(profile_stack.ROOT / "originstack.py"),
        "-d",
        "lights",
        "-j",
        "4",
    ]


def test_nsight_command_captures_process_tree_and_cuda():
    command = profile_stack._nsight_command(
        "nsys", Path("profiling") / "originstack-nsight", ["--gpu-coadd"]
    )
    assert command[:8] == [
        "nsys",
        "profile",
        "--trace=cuda,osrt",
        "--sample=process-tree",
        "--force-overwrite=true",
        "--output",
        str(Path("profiling") / "originstack-nsight"),
        "--",
    ]
    assert command[8] == profile_stack.sys.executable
    assert command[9:] == [str(profile_stack.ROOT / "originstack.py"), "--gpu-coadd"]
