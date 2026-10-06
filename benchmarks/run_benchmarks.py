"""Run the whole benchmark protocol in one command (work in progress, built step by step).

Implemented so far: the pre-flight checks. A real run is rejected, with the reason and the fix, when

* the working tree has uncommitted changes (commit or discard them),
* a run for this machine and commit already exists (delete that folder yourself to redo it; nothing is deleted here),
* the CPU is set up for noisy timings (governor, boost/turbo, energy preference).

Nothing is ever changed or deleted by this script.

Usage (from ``benchmarks/``)::

    uv run python run_benchmarks.py
"""

import sys

from bench_env import RESULTS_ROOT, capture, lock_environment, reject_reasons, run_name


def preflight() -> int:
    lock_environment(core=0)
    env = capture()
    print(f"{env['hostname']} | {env['cpu']['model']} | commit {(env['git_commit'] or 'none')[:7]}")

    reasons = reject_reasons(env)
    if reasons:
        print(f"\nrun rejected ({len(reasons)} problem{'s' * (len(reasons) != 1)}):")
        for number, reason in enumerate(reasons, 1):
            print(f"  {number}. {reason}")
        return 1

    print(f"pre-flight passed; this run would write to {(RESULTS_ROOT / run_name(env)).relative_to(RESULTS_ROOT.parent)}")
    return 0


def main() -> int:
    return preflight()


if __name__ == "__main__":
    sys.exit(main())
