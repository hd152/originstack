"""Every CLI setting read by Phase 1-3 code must be in the checkpoint fingerprint.

``checkpoint.stack_fingerprint`` hashes the CLI groups for Phase 1 and Phases 2-3 plus
the ``_P1_EXTRA``/``_P23_EXTRA`` names. A flag that lives in another group (or has no
group) but is read by a Phase 1-3 module is silently left out, so changing it resumes a
stale checkpoint. This test finds such flags by reading the modules' source, the same
way ``postprocess._early_args_read`` does for the early Phase 4 cache.
"""
import ast
import pathlib

from src import checkpoint
from src.cli import build_parser

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

# Modules whose code runs in Phases 1-3.
PHASE123_MODULES = (
    "frame_processor", "registration", "stacking", "proper_coadd", "affine_fit",
    "blind_match", "phase_correlate", "transparency", "trail_reject", "local_normalize",
    "banding", "debayer", "frame_store", "full_field", "cfa_drizzle", "distortion",
    "robust_pca", "dark_scale", "dark_temp_model", "vignette_calib", "star_detect",
    "matched_filter", "quality", "frame_discovery",
)

# Read in Phase 1-3 code but known not to change the stacked pixels.
NOT_PIXEL_AFFECTING = {
    "parallel",         # worker count: output is bit-identical
    "frame_store",      # RAM vs disk placement: bit-identical (tests/test_frame_store.py)
    "verbose",          # logging
    "ui_frame_every",   # progress reporting
    "offline",          # network guard
    "output",           # names side files only
}


def _args_reads(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "args":
            names.add(n.attr)
        if (isinstance(n, ast.Call) and getattr(n.func, "id", None) in ("getattr", "hasattr")
                and len(n.args) >= 2 and isinstance(n.args[0], ast.Name)
                and n.args[0].id == "args" and isinstance(n.args[1], ast.Constant)):
            names.add(n.args[1].value)
    return names


def test_phase123_flags_are_fingerprinted():
    dests = {a.dest for a in build_parser()._actions}
    covered = (set(checkpoint._group_dests(checkpoint._P1_GROUPS))
               | set(checkpoint._P1_EXTRA)
               | (set(checkpoint._group_dests(checkpoint._P23_GROUPS))
                  - set(checkpoint._P23_IGNORE))
               | set(checkpoint._P23_EXTRA))
    missing = {}
    for mod in PHASE123_MODULES:
        for name in _args_reads(SRC / f"{mod}.py"):
            if name in dests and name not in covered and name not in NOT_PIXEL_AFFECTING:
                missing.setdefault(name, []).append(mod)
    assert not missing, (
        "Read by Phase 1-3 code but not in the checkpoint fingerprint -- add to "
        "checkpoint._P1_EXTRA/_P23_EXTRA (or to NOT_PIXEL_AFFECTING here if it cannot "
        f"change the stack): {missing}")


def test_modules_exist():
    for mod in PHASE123_MODULES:
        assert (SRC / f"{mod}.py").exists(), mod
