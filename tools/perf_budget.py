"""Time and peak-memory check for a stack run.

Runs ``originstack.py`` on a folder, samples the whole process tree (workers included) for
peak resident memory, and compares wall time and peak RSS with a budget file.

    python tools/perf_budget.py                         # synthetic data, budget check (CI)
    python tools/perf_budget.py --record                # store this machine's numbers as budget
    python tools/perf_budget.py -d lights/ --label laptop16gb   # report only, a real session

With ``-d`` there is no budget (real sessions differ per machine): the result is printed and
appended to ``--log`` so runs on different machines can be compared, e.g. the 16 GB laptop
low-memory check. Exit status 1 when a budget is exceeded.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGET = ROOT / "tools" / "data" / "perf_budget.json"
# Budgets are recorded numbers times these margins: CI runners are noisy (the dev laptop moves
# up to ~40% between identical runs), so only a real regression should trip.
TIME_MARGIN = 2.0
RSS_MARGIN = 1.5


def _tree_rss(proc) -> int:
    try:
        procs = [proc] + proc.children(recursive=True)
    except Exception:
        return 0
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except Exception:
            pass
    return total


def run_measured(cmd, cwd):
    import psutil
    t0 = time.perf_counter()
    child = subprocess.Popen(cmd, cwd=cwd)
    ps = psutil.Process(child.pid)
    peak = 0
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, _tree_rss(ps))
            stop.wait(0.2)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    rc = child.wait()
    stop.set()
    th.join()
    return rc, time.perf_counter() - t0, peak


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--directory", help="folder to stack (default: synthetic data)")
    ap.add_argument("--label", default="", help="name for this run in the log")
    ap.add_argument("--record", action="store_true", help="write this run as the budget")
    ap.add_argument("--log", help="append the result to this JSON-lines file")
    ap.add_argument("--time-margin", type=float, default=TIME_MARGIN,
                    help="allowed time over the recorded budget (CI runners have fewer cores "
                         "than the machine that recorded it)")
    ap.add_argument("--keep", action="store_true", help="keep the output stack")
    ap.add_argument("extra", nargs="*", help="extra originstack.py arguments (after --)")
    a = ap.parse_args(argv)

    import psutil  # noqa: F401  (fail early with a clear error)

    work = ROOT
    synthetic = a.directory is None
    if synthetic:
        subprocess.check_call([sys.executable, str(ROOT / "tools" / "create_synthetic.py")], cwd=work)
        folder = "synthetic_data"
        extra = ["--debayer-method", "malvar", "--white-balance", "grayworld",
                 "--stack-method", "median"]
    else:
        folder, extra = a.directory, []
    out = work / "perf_budget_stack.fits"
    cmd = [sys.executable, str(ROOT / "originstack.py"), "-d", folder, "-o", str(out)] + extra + a.extra
    rc, secs, peak = run_measured(cmd, work)
    if not a.keep:
        for p in work.glob("perf_budget_stack*"):
            try:
                p.unlink()
            except OSError:
                pass
    if rc != 0:
        print(f"perf_budget: stack run failed (exit {rc})")
        return rc

    result = {"label": a.label, "seconds": round(secs, 1), "peak_rss_mb": round(peak / 2**20),
              "machine": platform.node(), "cpus": os.cpu_count(),
              "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
              "synthetic": synthetic, "time": time.strftime("%Y-%m-%d %H:%M")}
    print(f"perf_budget: {result['seconds']} s, peak {result['peak_rss_mb']} MB "
          f"(tree RSS) on {result['cpus']} CPUs / {result['ram_gb']} GB")
    if a.log:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    if not synthetic:
        return 0
    if a.record:
        BUDGET.parent.mkdir(exist_ok=True)
        BUDGET.write_text(json.dumps({"seconds": result["seconds"],
                                      "peak_rss_mb": result["peak_rss_mb"]}, indent=2) + "\n")
        print(f"perf_budget: recorded {BUDGET}")
        return 0
    if not BUDGET.exists():
        print("perf_budget: no budget file; run with --record on a quiet machine")
        return 0
    b = json.loads(BUDGET.read_text())
    bad = []
    if result["seconds"] > b["seconds"] * a.time_margin:
        bad.append(f"time {result['seconds']} s > {b['seconds']} s x {a.time_margin}")
    if result["peak_rss_mb"] > b["peak_rss_mb"] * RSS_MARGIN:
        bad.append(f"peak RSS {result['peak_rss_mb']} MB > {b['peak_rss_mb']} MB x {RSS_MARGIN}")
    if bad:
        print("perf_budget: OVER BUDGET: " + "; ".join(bad))
        return 1
    print("perf_budget: within budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
