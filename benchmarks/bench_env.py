"""Pin and record the benchmark environment, so runs on different machines can be compared.

Usage (from ``benchmarks/``)::

    uv sync --frozen                                  # identical library versions from uv.lock
    uv run python bench_env.py capture                # writes env_<hostname>.json, prints warnings
    uv run python bench_env.py compare env_a.json env_b.json

In a notebook or script, call ``lock_environment()`` *before* importing numpy/scipy::

    from bench_env import lock_environment
    lock_environment(core=2)
    import numpy as np

What is pinned: BLAS/OpenMP threads (1), CPU affinity (one core), ``PYTHONHASHSEED``.
What is recorded: CPU model/cores/caches/SIMD flags, frequency governor and boost, SMT, NumPy's SIMD dispatch,
Python/NumPy/SciPy/pandas versions, ``uv.lock`` hash, git commit and dirty state.
Linux only (reads /proc and /sys).
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
SIMD_OF_INTEREST = ("sse42", "avx", "avx2", "fma", "fma3", "avx512f", "avx512cd", "avx512_skx")

# keys that must be equal on both machines for timings to be comparable / these are expected to differ
MUST_MATCH = ("python", "packages", "uv_lock_sha256", "git_commit", "git_dirty", "threads_env", "numpy_build")


_LIMITS: list = []  # keeps threadpoolctl limits alive


def lock_environment(core: int | None = 0, seed_hash: int = 0) -> dict:
    """Set threads to 1 and pin to one core. Must run before numpy/scipy are imported to take effect on BLAS."""
    for var in THREAD_VARS:
        os.environ[var] = "1"
    os.environ.setdefault("PYTHONHASHSEED", str(seed_hash))  # only effective for child processes
    pinned = None
    if core is not None and hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        pinned = allowed[core % len(allowed)]
        os.sched_setaffinity(0, {pinned})
    late = "numpy" in sys.modules
    if late:
        try:  # works after numpy is loaded, if threadpoolctl is installed
            from threadpoolctl import threadpool_limits

            _LIMITS.append(threadpool_limits(limits=1))
            late = False
        except ImportError:
            print(
                "warning: numpy was imported before lock_environment() and threadpoolctl is missing; BLAS threads are not "
                "limited (pure ufunc timings such as pdf/cdf are unaffected). Restart the kernel, or run as a plain script.",
                file=sys.stderr,
            )
    return {"threads": 1, "pinned_cpu": pinned, "blas_limited": not late}


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cpu() -> dict:
    info = {"model": None, "flags": [], "physical_cores": None, "logical_cpus": os.cpu_count()}
    text = _read("/proc/cpuinfo") or ""
    cores = set()
    phys = None
    for line in text.splitlines():
        key, _, value = (s.strip() for s in line.partition(":"))
        if key == "model name" and info["model"] is None:
            info["model"] = value
        elif key == "flags" and not info["flags"]:
            info["flags"] = sorted(f for f in value.split() if f in SIMD_OF_INTEREST or f.startswith("avx"))
        elif key == "physical id":
            phys = value
        elif key == "core id":
            cores.add((phys, value))
    info["physical_cores"] = len(cores) or None
    info["caches_kb"] = {
        f"L{idx}{kind}": _read(f"/sys/devices/system/cpu/cpu0/cache/index{i}/size")
        for i, (idx, kind) in enumerate((("1", "d"), ("1", "i"), ("2", ""), ("3", "")))
    }
    info["governor"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    info["scaling_driver"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver")
    info["energy_preference"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")
    info["max_mhz"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    # boost: intel_pstate no_turbo (1 = boost off) or acpi-cpufreq/amd boost (1 = boost on)
    info["intel_no_turbo"] = _read("/sys/devices/system/cpu/intel_pstate/no_turbo")
    info["cpufreq_boost"] = _read("/sys/devices/system/cpu/cpufreq/boost")
    info["smt_active"] = _read("/sys/devices/system/cpu/smt/active")
    return info


def _git() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None}


def _packages() -> dict:
    out = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "pymultifit", "statsmodels"):
        try:
            out[name] = __import__("importlib.metadata", fromlist=["version"]).version(name)
        except Exception:
            out[name] = None
    return out


def _numpy_build() -> dict:
    try:
        import numpy as np

        try:
            from numpy._core._multiarray_umath import __cpu_features__ as feats  # numpy >= 2
        except ImportError:
            from numpy.core._multiarray_umath import __cpu_features__ as feats
        return {
            "simd_enabled": sorted(k for k, v in feats.items() if v),
            "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
            "blas": (np.show_config(mode="dicts") or {}).get("Build Dependencies", {}).get("blas", {}).get("name"),
        }
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}


def capture() -> dict:
    lock = ROOT / "uv.lock"
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "cpu": _cpu(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "threads_env": {v: os.environ.get(v) for v in THREAD_VARS},
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "packages": _packages(),
        "numpy_build": _numpy_build(),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        **_git(),
    }


def warnings_for(env: dict) -> list[str]:
    cpu, warn = env["cpu"], []
    if cpu["governor"] not in (None, "performance"):
        warn.append(f"CPU governor is '{cpu['governor']}', use 'performance' (sudo cpupower frequency-set -g performance).")
    if cpu["energy_preference"] not in (None, "performance"):
        warn.append(f"Energy preference is '{cpu['energy_preference']}', set energy_performance_preference to 'performance'.")
    if cpu["intel_no_turbo"] == "0" or cpu["cpufreq_boost"] == "1":
        warn.append("Turbo/boost is on: clocks drift with load and temperature; disable it or accept more noise.")
    if env["git_dirty"]:
        warn.append("Working tree has uncommitted changes to tracked files.")
    if any(v != "1" for v in env["threads_env"].values()):
        warn.append("BLAS/OpenMP threads not all 1; call lock_environment() before importing numpy.")
    if env["affinity"] and len(env["affinity"]) > 1:
        warn.append("Process is not pinned to a single core.")
    return warn


def compare(a: dict, b: dict) -> int:
    bad = 0
    for key in MUST_MATCH:
        if a.get(key) != b.get(key):
            bad += 1
            print(f"MISMATCH {key}:\n  A: {a.get(key)}\n  B: {b.get(key)}")
    for key in ("model", "physical_cores", "flags", "governor", "scaling_driver", "energy_preference", "caches_kb"):
        print(f"differs (expected) cpu.{key}:\n  A: {a['cpu'].get(key)}\n  B: {b['cpu'].get(key)}" if a["cpu"].get(key) != b["cpu"].get(key) else f"same cpu.{key}")
    print("\nOK: software stacks match." if not bad else f"\n{bad} must-match field(s) differ; timings are not directly comparable.")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="record this machine's environment to JSON")
    cap.add_argument("-o", "--output", type=Path, default=None)
    cap.add_argument("--core", type=int, default=0, help="index (in the allowed set) of the core to pin to")
    cmp_ = sub.add_parser("compare", help="compare two captured JSON files")
    cmp_.add_argument("a", type=Path)
    cmp_.add_argument("b", type=Path)
    args = parser.parse_args()

    if args.cmd == "compare":
        return compare(json.loads(args.a.read_text()), json.loads(args.b.read_text()))

    lock_environment(core=args.core)  # numpy is imported lazily in _numpy_build(), after the limits are set
    env = capture()
    out = args.output or Path(__file__).parent / f"env_{env['hostname']}.json"
    out.write_text(json.dumps(env, indent=2) + "\n")
    print(f"{env['cpu']['model']} | {env['cpu']['physical_cores']}c/{env['cpu']['logical_cpus']}t | python {env['python']} "
          f"| numpy {env['packages']['numpy']} | scipy {env['packages']['scipy']}")
    print(f"SIMD used by numpy: {', '.join(env['numpy_build'].get('simd_enabled', []))}")
    for w in warnings_for(env):
        print("warning:", w)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
