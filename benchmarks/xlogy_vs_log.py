"""Does ``XLOGY(1.0, a)`` cost anything compared to a plain log in the log-CDFs that still use it?

Every function is timed end to end (through the distribution class, as in ``speed.ipynb``), with ``utilities_d.XLOGY``
temporarily replaced by a variant. Only calls of the form ``XLOGY(1.0, a)`` are replaced, which is all these functions use.

``uniform_log_cdf_`` and ``q_exponential_log_cdf_`` no longer call ``XLOGY`` (they use ``_log_pos``, the masked variant
below, since this script showed it is 10-33 % faster there), so together with ``laplace`` they are controls: all variants
must give ~1.00x for them. ``beta`` and ``half_normal`` still use ``XLOGY(1.0, ...)``.

Variants
--------
xlogy   : the current code (scipy ``xlogy``).
masked  : ``np.log`` only where ``a`` is not <= 0 (-inf elsewhere, NaN stays NaN): same output, no warning.
log     : plain ``np.log`` under ``errstate``: the speed upper bound, NOT usable as is (warns / differs at 0).

Usage (from ``benchmarks/``)::

    uv run python xlogy_vs_log.py                 # sizes 1e3, 1e5, 1e6; writes results/<run>/xlogy_vs_log.csv
    uv run python xlogy_vs_log.py --repeats 100

Controls (no ``XLOGY`` any more): uniform, q_exponential, laplace. Still using it: beta, half_normal.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path

from bench_env import lock_environment, results_dir

lock_environment(core=0)  # before numpy is imported

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from timeit import default_timer as timer  # noqa: E402

import pymultifit.distributions as p_dist  # noqa: E402
from pymultifit.distributions import utilities_d  # noqa: E402

CASES = {
    "beta": (lambda: p_dist.BetaDistribution.from_scipy_params(a=5, b=80, loc=-3, scale=5.6), (-3.0, 2.6)),
    "uniform": (lambda: p_dist.UniformDistribution.from_scipy_params(loc=-3, scale=2), (-4.0, 0.0)),
    "q_exponential": (lambda: p_dist.QExponentialDistribution(q=1.5, rate=1.0, normalize=True), (-1.0, 8.0)),
    "laplace": (lambda: p_dist.LaplaceDistribution.from_scipy_params(loc=-3, scale=3), (-30.0, 30.0)),
    "half_normal": (lambda: p_dist.HalfNormalDistribution.from_scipy_params(scale=2), (-1.0, 12.0)),
}


def _xlogy_current(a, b):
    return _ORIGINAL(a, b)


def _xlogy_masked(a, b):
    assert a == 1.0
    b = np.asarray(b, dtype=float)
    out = np.full(b.shape, -np.inf)
    return np.log(b, out=out, where=~(b <= 0))


def _xlogy_log(a, b):
    assert a == 1.0
    with np.errstate(all="ignore"):
        return np.log(b)


_ORIGINAL = utilities_d.XLOGY
VARIANTS = {"xlogy": _xlogy_current, "masked": _xlogy_masked, "log": _xlogy_log}


@contextmanager
def patched(variant):
    utilities_d.XLOGY = VARIANTS[variant]
    try:
        yield
    finally:
        utilities_d.XLOGY = _ORIGINAL


def median_time(func, x, repeats, warmup=3):
    for _ in range(warmup):
        func(x)
    times = []
    for _ in range(repeats):
        start = timer()
        func(x)
        times.append(timer() - start)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 100_000, 1_000_000])
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("-o", "--output", type=Path, default=None, help="CSV path (default: results/<run>/xlogy_vs_log.csv)")
    args = parser.parse_args()
    output = args.output or results_dir(write_env=False) / "xlogy_vs_log.csv"  # joins the run folder of this commit, keeps its env.json

    rows = []
    for name, (make, (lo, hi)) in CASES.items():
        dist = make()
        for n in args.sizes:
            x = np.linspace(lo, hi, n)
            with np.errstate(all="ignore"):
                reference = dist.logcdf(x)
            for variant in ("masked", "log"):  # output must match the current code (except ``log`` at -inf)
                with patched(variant), np.errstate(all="ignore"):
                    got = dist.logcdf(x)
                if variant == "masked":
                    assert np.array_equal(got, reference, equal_nan=True), f"{name}: masked output differs"
            row = {"function": f"{name}_log_cdf", "n": n}
            for variant in VARIANTS:
                with patched(variant), np.errstate(all="ignore"):
                    row[variant] = median_time(dist.logcdf, x, args.repeats)
            row["masked/xlogy"] = row["masked"] / row["xlogy"]
            row["log/xlogy"] = row["log"] / row["xlogy"]
            rows.append(row)
            print(f"{row['function']:<22} n={n:>8}  xlogy {row['xlogy'] * 1e3:8.3f} ms  "
                  f"masked {row['masked/xlogy']:.2f}x  log {row['log/xlogy']:.2f}x")

    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
