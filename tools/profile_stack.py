"""Profile an OriginStack run with Scalene and/or NVIDIA Nsight Systems.

Examples:

    python tools/profile_stack.py --profiler scalene -- -d lights -o profiled.fits -j 4
    python tools/profile_stack.py --profiler nsight -- -d lights -o profiled.fits --gpu-coadd
    python tools/profile_stack.py --profiler both --output-dir profiling -- -d lights -o profiled.fits

Scalene is an optional development dependency. Nsight Systems is an external NVIDIA
tool and must be installed separately. Profiling never changes normal pipeline behavior;
the stack command is launched in a separate process.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "profiling"


def _executable(name: str, override: str | None) -> str:
    command = override or shutil.which(name)
    if not command:
        raise SystemExit(
            f"{name} was not found. Install it separately or pass "
            f"--{name.replace(' ', '-').replace('_', '-')} COMMAND."
        )
    return command


def _scalene_command(
    scalene: str, output: Path, stack_args: Sequence[str]
) -> list[str]:
    return [
        scalene,
        "run",
        "--profile-all",
        "--outfile",
        str(output),
        "---",
        str(ROOT / "originstack.py"),
        *stack_args,
    ]


def _nsight_command(
    nsys: str, output: Path, stack_args: Sequence[str]
) -> list[str]:
    return [
        nsys,
        "profile",
        "--trace=cuda,osrt",
        "--sample=process-tree",
        "--force-overwrite=true",
        "--output",
        str(output),
        "--",
        sys.executable,
        str(ROOT / "originstack.py"),
        *stack_args,
    ]


def _metadata(
    profiler: str, command: Sequence[str], output: Path, stack_args: Sequence[str]
) -> dict[str, object]:
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "profiler": profiler,
        "command": list(command),
        "stack_args": list(stack_args),
        "output": str(output),
        "cwd": str(ROOT),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": commit,
    }


def _run_profile(
    profiler: str,
    command: Sequence[str],
    output: Path,
    stack_args: Sequence[str],
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_name(output.name + ".json")
    metadata_path.write_text(
        json.dumps(_metadata(profiler, command, output, stack_args), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Running {profiler}: {' '.join(command)}")
    return subprocess.run(command, cwd=ROOT, env=os.environ.copy()).returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiler",
        choices=("scalene", "nsight", "both"),
        default="scalene",
        help="Profiler to run (default: scalene).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for profiler artifacts and metadata.",
    )
    parser.add_argument(
        "--scalene-command",
        help="Scalene executable override (otherwise found on PATH).",
    )
    parser.add_argument(
        "--nsys-command",
        help="Nsight Systems executable override (otherwise found on PATH).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print profiling commands and write metadata without launching them.",
    )
    parser.add_argument(
        "stack_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to originstack.py; put them after --.",
    )
    args = parser.parse_args(argv)
    stack_args = list(args.stack_args)
    if stack_args[:1] == ["--"]:
        stack_args.pop(0)
    if not stack_args:
        parser.error("provide OriginStack arguments after --")

    profilers = ("scalene", "nsight") if args.profiler == "both" else (args.profiler,)
    for profiler in profilers:
        if profiler == "scalene":
            executable = _executable("scalene", args.scalene_command)
            output = args.output_dir / "scalene.html"
            command = _scalene_command(executable, output, stack_args)
        else:
            executable = _executable("nsys", args.nsys_command)
            output = args.output_dir / "originstack-nsight"
            command = _nsight_command(executable, output, stack_args)
        if args.dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            metadata_path = output.with_name(output.name + ".json")
            metadata_path.write_text(
                json.dumps(_metadata(profiler, command, output, stack_args), indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"{profiler}: {' '.join(command)}")
            result = 0
        else:
            result = _run_profile(profiler, command, output, stack_args)
        if result:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
