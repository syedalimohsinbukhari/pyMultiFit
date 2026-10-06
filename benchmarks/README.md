# Benchmarks

Speed and accuracy of the `pymultifit` distributions against `scipy.stats`. Timings are only useful when the
conditions are the same every time, so everything here is built around reproducible runs on a fixed machine setup.

All commands are run from this folder (`benchmarks/`).

## One-time setup per machine

```bash
uv sync --frozen            # identical library versions from uv.lock (the dev group has jupyter)
```

Set the CPU up for stable clocks. A real run is rejected until these are done, and the rejection prints the exact command:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo        # Intel: turbo off
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost                 # AMD: boost off
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference   # amd-pstate-epp only
```

These reset on reboot. Keep a laptop on mains power.

## Running

```bash
uv run python run_benchmarks.py --smoke                 # once per machine: tests the whole pipeline in ~2 min
uv run python run_benchmarks.py --baseline-ref a6019cc  # optional: old pymultifit, current benchmark code
uv run python run_benchmarks.py                         # the real run; compares itself with the latest other run
uv run python run_benchmarks.py --against <folder>      # ... or with a specific run
uv run python xlogy_vs_log.py                           # after the real run: xlogy vs log in the log-CDFs
```

Follow a long run with `tail -f results/<run>/progress.log` (one line per distribution).

### What a run refuses to do

The script only reports and never changes or deletes anything of yours:

| rejected when | what to do |
|---|---|
| the working tree has uncommitted changes | commit or discard them, then re-run |
| `results/<host>_<commit>/` already exists | commit something new, or delete that folder yourself |
| governor, boost/turbo or energy preference are not set for stable clocks | run the printed command |
| no passing `--smoke` run on this machine | run `--smoke` |
| `--against` / `--baseline-ref` point to something that does not exist | pick a real folder / ref |

A failed or interrupted run leaves its folder behind (with `status.json` and `run.log`); delete it before running that
commit again. On a failure of the notebook itself the temporary directory is kept and its path is printed.

## What a run produces

```
results/<host>_<short commit>/          baseline run: <host>_<ref>_on_<current commit>
    PDF_multifit_df.csv  PDF_scipy_df.csv  PDF_multifit_variable_df.csv  PDF_scipy_variable_df.csv
    CDF_multifit_df.csv  CDF_scipy_df.csv  CDF_multifit_variable_df.csv  CDF_scipy_variable_df.csv
    env.json                            CPU, SIMD, governor, boost, library versions, uv.lock hash, commits
    run.log, progress.log, status.json  what happened, one line per distribution, ok / failed / interrupted
    compare_<reference>.md              the comparison with the reference run (when there is one)
    xlogy_vs_log.csv                    only if xlogy_vs_log.py was run
plots/speed/                            plots of the latest run (lowercase snake_case names, overwritten by each run)
plots/accuracy/                         accuracy plots
variation_plots/                        implementation variants (the arcSine_*.py scripts)
```

CSV layout: one row per number of points (`np.logspace(0.3, 6, 50)`), one column per distribution. `*_df` uses the default
parameters, `*_variable_df` a second parameter set.

## Reading a comparison

`ratio = multifit time / scipy time`, as the median over the 12 largest sizes. Below 1 means multifit is faster. The ratio
cancels most of the hardware speed, so it can be compared across commits, and with care across machines.

- `change` is the new ratio over the reference ratio.
- `scipy drift` is how much scipy's own time moved. scipy did not change, so this is the noise of the measurement
  (across machines it also holds the hardware difference).
- A distribution is `faster` / `slower` when `|change| > max(6 %, 3 x scipy drift)`.
- A change is **consistent** when both parameter sets agree, **unproven** when only one does or they disagree. Act on
  consistent ones; re-measure the unproven ones.

Compare any two runs by hand:

```bash
uv run python compare_runs.py <new run> <reference run> [--print]
uv run python bench_env.py compare results/<a>/env.json results/<b>/env.json   # software / hardware differences
```

## The protocol (what makes timings comparable)

- Same library versions (`uv sync --frozen`) and the same Python; `bench_env.py compare` shows any mismatch.
- BLAS/OpenMP threads set to 1 and the process pinned to one core, before numpy is imported (`lock_environment`). The
  smoke and real runs verify this inside the notebook kernel.
- 3 untimed warm-up calls, then the median of the repetitions (the variation plots: the mean of 500 runs per repeat, the
  median across repeats).
- PDF and CDF are timed separately. CSVs produced before the benchmark fix in `cffb5ea` held PDF timings in the CDF
  files, so they must not be compared with newer runs.
- Seeds: the timing inputs are deterministic (`np.linspace`), so no seed affects them. Any random data added later should
  use `np.random.default_rng(<fixed seed>)`.
- Timings differ between CPUs because NumPy picks SIMD loops by instruction set (the Ivy Bridge T530 has AVX but no AVX2/FMA).
  The captured `env.json` records which ones were used.

## Files

| file | purpose |
|---|---|
| `run_benchmarks.py` | the whole protocol in one command (`--smoke`, `--baseline-ref`, `--against`) |
| `speed.ipynb` | the benchmark itself; run headless by `run_benchmarks.py`, or interactively (run its first code cell first) |
| `functions.py` | timing, plotting and `slugify` helpers used by the notebooks |
| `bench_env.py` | `lock_environment`, `capture`, reject checks, `compare` (also a CLI) |
| `compare_runs.py` | the comparison report between two runs |
| `xlogy_vs_log.py` | end-to-end cost of `XLOGY(1.0, a)` against a masked / plain log in the log-CDFs that still use it (beta, half normal; uniform, q-exponential and laplace are controls) |
| `summary.ipynb` | heatmaps and summaries; set `RESULTS` in its first code cell to the run to summarise |
| `accuracy.ipynb` | accuracy against scipy |
| `arcSine_*.py` | implementation variants of the arcsine functions (`variation_plots/`) |
