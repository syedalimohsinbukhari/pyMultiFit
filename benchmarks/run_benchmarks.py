"""Run the whole benchmark protocol in one command (work in progress, built step by step).

Implemented so far
------------------
Pre-flight: a real run is rejected, with the reason and the fix, when

* the working tree has uncommitted changes (commit or discard them),
* a run for this machine and commit already exists (delete that folder yourself to redo it),
* the CPU is set up for noisy timings (governor, boost/turbo, energy preference),
* there is no passing smoke run for this machine yet.

``--smoke``: executes the real ``speed.ipynb`` headless with tiny sizes and 2 repetitions in a temporary directory, then
checks that every output exists and has the right shape and that the environment lock worked inside the kernel. It writes
nothing to ``results/`` or ``plots/`` (only a marker ``results/.smoke_ok_<hostname>``) and does not require a clean tree
or the hardware settings, because it only tests the pipeline.

Nothing is ever changed or deleted outside the temporary directory.

Usage (from ``benchmarks/``)::

    uv run python run_benchmarks.py --smoke
    uv run python run_benchmarks.py
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from bench_env import RESULTS_ROOT, capture, hardware_issues, lock_environment, reject_reasons, run_name

HERE = Path(__file__).resolve().parent
SMOKE_SIZES = 4  # keep in sync with speed.ipynb (BENCH_SMOKE)
N_CSV = 8  # {PDF, CDF} x {multifit, scipy} x {df, variable_df}
N_PLOTS = 50  # 25 distribution cells in speed.ipynb (Chi2 has three parameter sets) x {pdf, cdf}


def smoke_marker(env: dict) -> Path:
    return RESULTS_ROOT / f".smoke_ok_{env['hostname'].lower().replace('-', '_')}"


def execute_notebook(workdir: Path, smoke: bool) -> tuple[int, Path]:
    """Run ``speed.ipynb`` headless in ``workdir``. The executed copy stays in ``workdir`` and is never kept."""
    child_env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")])),
        "BENCH_RESULTS_ROOT": str(workdir / "results"),
    }
    if smoke:
        child_env["BENCH_SMOKE"] = "1"
    log = workdir / "notebook.log"
    # nbconvert runs a notebook with the notebook's own folder as working directory, so run a copy that lives in workdir:
    # relative paths in the notebook (plots/...) then land in workdir and never in benchmarks/
    notebook = workdir / "speed.ipynb"
    shutil.copy(HERE / "speed.ipynb", notebook)
    command = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(notebook),
        "--output", "speed_executed.ipynb", "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=python3",
    ]
    with log.open("w") as handle:
        code = subprocess.run(command, cwd=workdir, env=child_env, stdout=handle, stderr=subprocess.STDOUT).returncode
    return code, log


def verify_smoke(workdir: Path) -> list[str]:
    """Everything a real run relies on, checked on the smoke output. Returns the problems found (empty: all good)."""
    import pandas as pd

    problems = []
    run_dirs = [p for p in (workdir / "results").glob("*") if p.is_dir()]
    if len(run_dirs) != 1:
        return [f"expected exactly one run folder in the temporary results, found {[p.name for p in run_dirs]}"]
    run = run_dirs[0]

    csvs = sorted(run.glob("*.csv"))
    if len(csvs) != N_CSV:
        problems.append(f"expected {N_CSV} CSV files, found {len(csvs)}")
    for csv in csvs:
        frame = pd.read_csv(csv)
        if frame.shape != (SMOKE_SIZES, 12):
            problems.append(f"{csv.name}: shape {frame.shape}, expected {(SMOKE_SIZES, 12)}")
        elif not (frame.notna().all().all() and (frame > 0).all().all()):
            problems.append(f"{csv.name}: contains missing or non-positive timings")
    # the CDF files must hold CDF timings, not a copy of the PDF ones (an old bug)
    for who in ("multifit", "scipy"):
        pdf, cdf = run / f"PDF_{who}_df.csv", run / f"CDF_{who}_df.csv"
        if pdf.exists() and cdf.exists() and pdf.read_bytes() == cdf.read_bytes():
            problems.append(f"PDF_{who}_df.csv and CDF_{who}_df.csv are identical")

    plots = sorted((workdir / "plots" / "speed").glob("*.png"))
    if len(plots) != N_PLOTS:
        problems.append(f"expected {N_PLOTS} speed plots, found {len(plots)}")
    bad_names = [p.name for p in plots if p.name != p.name.lower() or " " in p.name]
    if bad_names:
        problems.append(f"plot names that are not lowercase snake_case: {bad_names[:3]}")

    env_file = run / "env.json"
    if not env_file.exists():
        return problems + ["env.json was not written"]
    env = json.loads(env_file.read_text())
    if any(v != "1" for v in env["threads_env"].values()):
        problems.append(f"thread limits not applied in the kernel: {env['threads_env']}")
    if not env["affinity"] or len(env["affinity"]) != 1:
        problems.append(f"kernel is not pinned to one core: {env['affinity']}")
    if Path(os.path.abspath(env["executable"])).parent != Path(os.path.abspath(sys.executable)).parent:
        problems.append(f"the kernel used {env['executable']}, not this environment's {sys.executable}")
    return problems


def smoke() -> int:
    lock_environment(core=0)
    env = capture()
    print(f"smoke run on {env['hostname']} (commit {(env['git_commit'] or 'none')[:7]}), tiny sizes, temporary directory")
    for problem, fix in hardware_issues(env):
        print(f"  note: {problem} (a real run would be rejected; fix: {fix})")

    before = _snapshot()
    workdir = Path(tempfile.mkdtemp(prefix="bench_smoke_"))
    code, log = execute_notebook(workdir, smoke=True)
    if code != 0:
        print(f"smoke FAILED: the notebook exited with {code}. Last lines of the log ({log}):")
        print("".join(log.read_text().splitlines(keepends=True)[-15:]))
        print(f"temporary directory kept for inspection: {workdir}")
        return 1

    problems = verify_smoke(workdir)
    if _snapshot() != before:
        problems.append("the smoke run changed files under benchmarks/results, plots or variation_plots (it must write only to the temporary directory)")
    if problems:
        print("smoke FAILED:")
        for number, problem in enumerate(problems, 1):
            print(f"  {number}. {problem}")
        print(f"temporary directory kept for inspection: {workdir}")
        return 1

    shutil.rmtree(workdir)
    marker = smoke_marker(env)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"commit": env["git_commit"], "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    print(f"smoke passed: {N_CSV} CSVs, {N_PLOTS} plots, thread limits and core pinning verified inside the kernel.")
    return 0


def _snapshot() -> list[tuple]:
    """(name, size, modification time) of everything under ``benchmarks/`` a run could touch, except the smoke marker."""
    entries = []
    for folder in ("results", "plots", "variation_plots"):
        for path in sorted((HERE / folder).rglob("*")):
            if path.name.startswith(".smoke_ok_"):
                continue
            stat = path.stat()
            entries.append((str(path.relative_to(HERE)), stat.st_size, stat.st_mtime_ns))
    return entries


def preflight() -> int:
    lock_environment(core=0)
    env = capture()
    print(f"{env['hostname']} | {env['cpu']['model']} | commit {(env['git_commit'] or 'none')[:7]}")

    reasons = reject_reasons(env)
    if not smoke_marker(env).exists():
        reasons.append("no passing smoke run for this machine yet.\n    fix: uv run python run_benchmarks.py --smoke")
    if reasons:
        print(f"\nrun rejected ({len(reasons)} problem{'s' * (len(reasons) != 1)}):")
        for number, reason in enumerate(reasons, 1):
            print(f"  {number}. {reason}")
        return 1

    print(f"pre-flight passed; this run would write to {(RESULTS_ROOT / run_name(env)).relative_to(RESULTS_ROOT.parent)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="test the pipeline with tiny sizes in a temporary directory")
    args = parser.parse_args()
    return smoke() if args.smoke else preflight()


if __name__ == "__main__":
    sys.exit(main())
