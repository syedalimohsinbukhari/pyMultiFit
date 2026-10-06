"""Compare two benchmark runs: did the code get faster or slower, beyond the noise?

For every distribution the quantity that is compared is ``ratio = multifit time / scipy time`` (below 1: multifit is
faster), as the median over the largest sizes. The ratio cancels most of the hardware speed, so it is meaningful across
commits on one machine and, with care, across machines.

* ``change``      : new ratio / reference ratio - 1, in %.
* ``scipy drift`` : how much scipy's own time moved between the two runs. scipy did not change, so this is the noise
                    (and, across machines, the hardware difference) of the measurement.
* verdict         : ``faster`` / ``slower`` when ``|change| > max(6 %, 3 x scipy drift)``, otherwise ``~``.

A change is called *consistent* when the two parameter sets of a distribution (``df`` and ``variable_df``) agree on the
direction, and *unproven* when only one does or they disagree.

Usage (from ``benchmarks/``)::

    uv run python compare_runs.py <new run folder> <reference run folder>

The folders are names inside ``results/`` or paths. ``run_benchmarks.py --against <folder>`` calls this after a run.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from bench_env import MUST_MATCH, RESULTS_ROOT

TAIL = 12  # number of largest sizes the median is taken over
FLOOR_PCT = 6.0
DRIFT_FACTOR = 3.0
SETS = (("df", "default parameters"), ("variable_df", "variable parameters"))
FUNCTIONS = ("PDF", "CDF")


def resolve(folder: str | Path) -> Path:
    path = Path(folder)
    if not path.is_dir():
        path = RESULTS_ROOT / str(folder)
    if not path.is_dir():
        raise FileNotFoundError(f"run folder not found: {folder} (also tried {RESULTS_ROOT / str(folder)})")
    return path


def _load(run: Path, function: str, who: str, suffix: str) -> pd.DataFrame:
    return pd.read_csv(run / f"{function}_{who}_{suffix}.csv")


def table(new: Path, ref: Path, function: str, suffix: str) -> pd.DataFrame:
    """One row per distribution: reference ratio, new ratio, change %, scipy drift %, verdict."""
    n_multi, n_scipy = _load(new, function, "multifit", suffix), _load(new, function, "scipy", suffix)
    r_multi, r_scipy = _load(ref, function, "multifit", suffix), _load(ref, function, "scipy", suffix)
    if len(n_multi) != len(r_multi):
        raise ValueError(f"the runs have different numbers of sizes ({len(n_multi)} and {len(r_multi)}), cannot compare")
    columns = [c for c in n_multi.columns if c in r_multi.columns]
    tail = slice(-TAIL, None)
    new_ratio = (n_multi[columns] / n_scipy[columns]).iloc[tail].median()
    ref_ratio = (r_multi[columns] / r_scipy[columns]).iloc[tail].median()
    drift = ((n_scipy[columns] / r_scipy[columns]).iloc[tail].median() - 1) * 100
    change = (new_ratio / ref_ratio - 1) * 100
    limit = (DRIFT_FACTOR * drift.abs()).clip(lower=FLOOR_PCT)
    verdict = pd.Series("~", index=columns)
    verdict[change > limit] = "slower"
    verdict[change < -limit] = "faster"
    return pd.DataFrame({"ref ratio": ref_ratio, "new ratio": new_ratio, "change %": change, "scipy drift %": drift,
                         "verdict": verdict})


def _fmt(frame: pd.DataFrame) -> str:
    lines = ["| distribution | ref ratio | new ratio | change | scipy drift | verdict |", "|---|---:|---:|---:|---:|---|"]
    for name, row in frame.iterrows():
        mark = "**" if row["verdict"] != "~" else ""
        lines.append(f"| {name} | {row['ref ratio']:.2f} | {row['new ratio']:.2f} | {mark}{row['change %']:+.1f} %{mark} "
                     f"| {row['scipy drift %']:+.1f} % | {mark}{row['verdict']}{mark} |")
    return "\n".join(lines)


def _env_lines(new: Path, ref: Path) -> list[str]:
    try:
        a, b = (json.loads((p / "env.json").read_text()) for p in (new, ref))
    except FileNotFoundError:
        return ["_env.json is missing in one of the runs, software match not checked._"]
    def commit(env: dict) -> str:
        package, harness = (env.get("package_commit") or env["git_commit"] or "?")[:7], (env["git_commit"] or "?")[:7]
        return package if package == harness else f"{package} (benchmark code {harness})"

    lines = [f"- new: {a['hostname']} | {a['cpu']['model']} | commit {commit(a)}",
             f"- reference: {b['hostname']} | {b['cpu']['model']} | commit {commit(b)}"]
    bad = [k for k in MUST_MATCH if a.get(k) != b.get(k) and k not in ("git_commit", "git_dirty")]
    bad += [f"numpy_build.{k}" for k in ("blas", "NPY_DISABLE_CPU_FEATURES") if a["numpy_build"].get(k) != b["numpy_build"].get(k)]
    lines.append("- software stacks match" if not bad else f"- **software differs: {', '.join(bad)}**, timings are not directly comparable")
    if a["hostname"] != b["hostname"]:
        lines.append("- **different machines**: the ratios are still comparable, absolute times are not; "
                     "the scipy drift column then also contains the hardware difference")
    return lines


def report(new: Path, ref: Path) -> tuple[str, list[str]]:
    """Return the markdown report and the headline lines (consistent / unproven changes)."""
    tables = {(f, s): table(new, ref, f, s) for f in FUNCTIONS for s, _ in SETS}
    consistent, unproven = [], []
    for function in FUNCTIONS:
        first, second = (tables[(function, s)]["verdict"] for s, _ in SETS)
        change_a, change_b = (tables[(function, s)]["change %"] for s, _ in SETS)
        for name in first.index.intersection(second.index):
            va, vb = first[name], second[name]
            if va == "~" and vb == "~":
                continue
            text = f"{name} {function} ({change_a[name]:+.0f} % / {change_b[name]:+.0f} %)"
            (consistent if va == vb else unproven).append((va if va == vb else "mixed", text))

    headline = []
    for verdict in ("faster", "slower"):
        items = [t for v, t in consistent if v == verdict]
        headline.append(f"consistently {verdict}: " + (", ".join(items) if items else "none"))
    headline.append("unproven (only one parameter set, or opposite): " + (", ".join(t for _, t in unproven) or "none"))

    out = [f"# Comparison: {new.name} vs {ref.name}", "", *_env_lines(new, ref), "",
           f"Ratio = multifit time / scipy time (below 1: multifit faster), median over the {TAIL} largest sizes. "
           f"Verdict when |change| > max({FLOOR_PCT:.0f} %, {DRIFT_FACTOR:.0f} x scipy drift).", "",
           "## Summary", "", *[f"- {line}" for line in headline], "(changes listed as default / variable parameters)", ""]
    for function in FUNCTIONS:
        for suffix, label in SETS:
            out += [f"## {function}, {label}", "", _fmt(tables[(function, suffix)]), ""]
    return "\n".join(out), headline


def find_reference(run: Path) -> Path | None:
    """The latest other run of the same machine that has results; ``None`` when there is none."""
    def stamp(folder: Path) -> str:
        for name, key in (("status.json", "finished_utc"), ("env.json", "captured_utc")):
            try:
                return json.loads((folder / name).read_text())[key]
            except (FileNotFoundError, KeyError, ValueError):
                continue
        return ""

    try:
        host = json.loads((run / "env.json").read_text())["hostname"]
    except (FileNotFoundError, KeyError, ValueError):
        return None
    candidates = []
    for folder in RESULTS_ROOT.iterdir():
        if not folder.is_dir() or folder.resolve() == run.resolve() or len(list(folder.glob("*.csv"))) < 8:
            continue
        try:
            if json.loads((folder / "env.json").read_text())["hostname"] == host:
                candidates.append((stamp(folder), folder))
        except (FileNotFoundError, KeyError, ValueError):
            continue
    return max(candidates)[1] if candidates else None


def write_report(new: Path, ref: Path, folder: Path | None = None) -> tuple[Path, list[str]]:
    text, headline = report(new, ref)
    target = (folder or new) / f"compare_{ref.name}.md"
    target.write_text(text + "\n")
    return target, headline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("new")
    parser.add_argument("reference")
    parser.add_argument("--print", action="store_true", help="print the full report instead of writing it into the new run")
    args = parser.parse_args()
    new, ref = resolve(args.new), resolve(args.reference)
    if args.print:
        print(report(new, ref)[0])
        return 0
    target, headline = write_report(new, ref)
    print("\n".join(headline))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
