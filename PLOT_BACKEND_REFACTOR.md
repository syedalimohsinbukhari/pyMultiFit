# Plot backend refactor and 2.0.0 release notes

Status (2026-10-21): the refactor, steps 1-3 of "Proposed design", is **open and deliberately not part of 2.0.0**; do it
before (or as the first step of) splitting the plotting code into a separate `pymultifit-plotter` package. The safety net
exists: `src/tests/test_fitters/test_plot_smoke.py` (plus `test_ci_parity.py`, `test_ci_object.py`, `test_qq_quantiles.py`,
`test_plotter_decoupling.py`). Everything else of 2.0.0 is on `plot-fixes` (PR #141 `distribution-optimization`, PR #152
`ci-fixes`), see "Release checklist" at the end.

## Already done (branch `plot-fixes`)

- `FitResult` / `Component` (`result.py`) and `to_result()` on `BaseFitter` and `MixedDataFitter`; `FitResult` is a read-only
  snapshot (read-only array copies, identity equality/hash) with `as_result()`.
- `ci.py` moved out of `fitters/backend`; `compute_ci_bounds` takes a `FitResult` and returns a `ConfidenceIntervals`.
- `pymultifit.plot`, `ci` and `result` no longer import `pymultifit.fitters`.
- `qq_compare` accepts fitters or `FitResult`s; `statsmodels` replaced by `_normal_quantiles` (scipy + numpy) and moved to
  the `dev` group.
- CI/PI fixes: exact levels in the legends, `0.29` no longer truncated to 28, `plot_prediction_intervals` validates its levels,
  overlays no longer overwrite axis labels/title (`_keep_axis_text`), prediction intervals at 99.5 % and above are finite.
- `_plotter` is initialised in `BaseFitter.__init__`, so `dry_run()` and `plotter` work before `fit()`.
- The CI pipeline (PR #152), see "CI pipeline" below.
- API docs for `plot` / `result` / `ci`, `plotting.rst` guide, `reproducibility.rst`, docstrings.
- Dead code removed: `FitPlotter._unwrap_plotter`, `FitPlotter._validate_fitted`, the `BaseFitter._format_param` copy,
  `utilities_d._eqx` (no caller) and `q_exp_to_gen_pareto` (unreleased; `QExponentialDistribution.to_scipy_params()` replaces
  it).

## What is still tangled

The two modules inside `pymultifit.plot` still depend on each other (line numbers of `_plot_backend.py` at `fdba8de`):

- `_fit_and_residual` (l.105), `_plot` (l.228) and `_resid` (l.440) take `plot_object: "FitPlotter"`, mostly just to read
  `plot_object.result` via `_validate(plot_object)` (l.432).
- `_plot()` calls back into the class: `plot_object._plot_individual_fits(axis=axis)` (l.253). The backend imports
  `FitPlotter` under `TYPE_CHECKING` (l.21) purely for that.
- Inconsistent signatures: `_qq`, `_param_correlation`, `_prediction_interval`, `_ci_plotter` take `result`; the three above
  take `plot_object`.
- `_validate(plot_object)` duplicates `result.require_fit()` (which the other backend functions already call); it goes with
  step 1.

### Import cycle in `pymultifit.plot`: none at runtime
A graphify trial (2026-10-13, code-only graph of `src/pymultifit`, checked with grep afterwards) reported
`FitPlotter.py -> _plot_backend.py -> plot/__init__.py -> FitPlotter.py`. That is a false positive: the backend imports only
`..`, `..ci`, `..exceptions`, `..result`, `..typing` (never `FitPlotter` or `plot/__init__`), and the `FitPlotter` import in the
backend sits under `TYPE_CHECKING` (the coupling listed above). Graphify seems to treat "a module of the package imports a
sibling" as a dependency on the package `__init__`. Step 1 still removes the type-only back-reference, but there is no cycle to
break. The same trial found the two dead functions removed above.

## Behaviour decisions (all closed, 2026-10-10)

- `plot_fit(show_individuals=True)`: the total fit is always drawn, the components go on top (also for one component).
- CI return convention: `confidence_intervals()` only computes and always returns a `ConfidenceIntervals`, the plotter only
  draws it, see "CI pipeline".
- `plot_fit_and_residuals(is_scatter=(bool, bool))` stays a pair `(fit_panel, residual_panel)` on purpose, so the data and the
  residuals can differ; a single bool is accepted for both and anything else raises a clear `ValueError`.

## Proposed design (open)

1. Backend functions take only `result: FitResult` (plus drawing options), never the plotter. Move
   `_plot_individual_fits` / `_plot_component` / `_get_color_cycle` / `_format_param` into `_plot_backend.py` as
   `_plot_components(result, axis)`; `FitPlotter` becomes a thin facade: validate, delegate, return axes.
2. One validation path: `result.require_fit()` (drop `_validate`).
3. One `_format_param`, living with the backend (the `BaseFitter` copy is already gone).

(The earlier steps 4, "one convention for compute and plot", and 5, the single-component `show_individuals` behaviour, are
done, see above.)

### Why it matters for the split

With 1-3 done, `pymultifit.plot` is a self-contained module: `FitPlotter(FitResult)` + backend functions, depending only on
`numpy`, `scipy`, `matplotlib`, `plotez` and `pymultifit.result` / `pymultifit.ci`. A `pymultifit-plotter` package would then
need `FitResult` and `ConfidenceIntervals` / `compute_ci_bounds` from somewhere, so the split decision is really about where
those live (core stays the owner: plotter depends on core, core imports the plotter lazily, or a third tiny shared package).

Graph view of the same question (graphify, 2026-10-13): `FitResult` is one of the three bridge nodes of the package (with
`BaseDistribution` and `BaseFitter`); it is the only thing connecting `_plot_backend`, `FitPlotter` and `ci.py` to the fitters.
That agrees with the plan, and changes to `FitResult` are the riskiest edits for the split. The report also flags
`_plot_backend.py` as a low-cohesion community (0.14), in line with step 1.

### Effort / risk

- Steps 1-3 are mechanical, ~150-200 lines in `FitPlotter.py` and `_plot_backend.py`, no numerical change.
- Verification: the smoke tests, the CI/Q-Q oracle tests, and running every `examples/plot_mechanics/*.py` headless.

---

# CI pipeline (decided and implemented 2026-10-10, merged in PR #152)

Nothing of the CI API exists in a release (v1.0.9 has none), so there is no deprecation path.
- Problems that were found: three places computed CIs (`compute_ci_bounds`, `BaseFitter.confidence_intervals(plot=, axis=)`,
  `FitPlotter.plot_confidence_intervals`, which repeated the ten compute arguments); the return type depended on `plot=` /
  `get_value=`; the result was a stringly-typed dict that did not say which levels / parts it held, so the plotter was told
  `ci_levels`, `overall_ci`, `individual_ci` again (KeyError on a mismatch); whole-percent keys collided (99.5 and 99.7 both
  `overall_ci_100`); the overall quantile loop duplicated `_curves_to_ci_results` and the model was evaluated twice when both
  overall and individual were asked for; the default `x_range` used the first and last element of `x` (wrong for unsorted
  data: [-15, 15] gave [-11.4, -5.5]); `x_range` could give wrong bands, see "x_range" below.
- Decisions, all implemented:
  1. The result is an object: frozen, read-only `ConfidenceIntervals` (`x`, exact `levels` as ascending floats, `n_bootstrap`,
     `overall: {level: Band}`, `individual: {level: tuple[Band, ...]}`, `Band(lower, median, upper)`), with `band(level,
     component=None)` and `as_dict()` for the flat layout (`x_range`, `overall_ci_95`, `overall_ci_99.7`).
  2. The plotter takes the object only: `plot_confidence_intervals(ci, overall=None, individual=None, axis=None) -> Axes`; it
     draws what the object holds, `TypeError` for anything else, `ValueError` for a part that was not computed.
  3. `confidence_intervals(...)` only computes and always returns the object: `plot=`, `axis=`, `get_value` and `seed` are gone.
  4. Naming: CI and PI stay separate concepts (CI is the bootstrap uncertainty of the curve, PI is analytic from the
     residuals); the prefixes stay, `pi_level` became `pi_levels` (it takes a list), `ci_levels` is unchanged. Both share one
     level parser, `normalize_levels`, which replaced `ci_level_labels`, `ci_level_percents` and `_ci_to_percentiles`.
  5. The object keeps the bands only, not the bootstrap curves (40-200 MB): a new level set or `x_range` is a recompute, which
     a seeded `rng_engine` makes reproducible; compute all the levels at once. The overall curve is accumulated inside the
     bootstrap loop, the per-component curves exist only when `individual_ci=True`. `keep_curves` can be added later without
     breaking anything.
  6. Random numbers: one argument, `rng_engine: Generator | int | None`. `None` is fresh entropy (a call without a seed used to
     raise), an integer is the seed of a new `default_rng`, a `Generator` is used as it is and advances. The lesson of
     `SEEDING.md` (another repository) applies: build one generator at the root and pass it on, a new `default_rng(42)` per
     call repeats the draws; its hash-of-filename seed is not needed (a user-side recipe at most, not library behaviour).
     The data generators (`multi_gaussian`, the other `multi_*`, `multiple_models`) got the same `rng_engine`, `None` keeps
     the global `np.random` state so `np.random.seed(...)` still works.
- Result: bit-identical to the old implementation on the default path (all three flag combinations, checked against the old
  `ci.py`, and the `test_ci_parity` oracle unchanged); 5000 bootstraps x 1000 points: overall only 0.92 -> 0.81 s, overall +
  individual 2.30 -> 1.13 s (each component is evaluated once per sample). Tests: `test_ci_object.py` (69 cases, 5 of them fail
  without the x_range fix), migrated `test_plot_smoke.py` / `test_ci_parity.py`; examples 04, 06, 08, 09 and the docs
  (`ci/_ci.rst`, `plotting.rst`, built without warnings) use the new API; examples 04 and 09 thread one `rng`.
- Docs: new `reproducibility.rst` (the one-generator rule, `rng_engine`, bootstrap noise figure, parallel guidance, what
  reproducibility cannot promise), `benchmarks/benchmarks.rst` rewritten with the measured ratios (all 48 combinations faster
  than scipy) and the reproduction protocol, an "Amplitude and the evaluation grid" section in `distributions/_distributions.rst`
  (and a pointer from `results/_results.rst`), `ci/_ci.rst` for `ConfidenceIntervals`, a generators section, index / API index /
  tutorials pointers, and two wrong rows of the distributions table fixed (scaled inverse chi-squared linked to the q-exponential,
  Student's t missing).
- Not done, on purpose: vectorising the bootstrap loop (the component functions take scalar parameters, a bigger change).

## x_range and the peak-normalised amplitude (found 2026-10-10)
- Cause: an unnormalised distribution returns `amplitude * pdf / max(pdf over the evaluated array)` (`_pdf_scaling`), so the
  amplitude is the peak height *of the array evaluated*, and a model value at an `x` depends on the other `x` evaluated with it
  (a lone point gives exactly `amplitude`). The fit evaluates on the data `x`, so the fitted amplitudes mean "peak height on the
  data grid"; everything agrees to ~0.1 % as long as the data sample the peaks.
- Effect on the CI: evaluating a window that leaves out a component's peak rescaled that component's tail to the full amplitude
  (two Gaussians, peak 10, `x_range=[0, 15]`: median off by 9.6; `[8, 15]`: off by 13.2). The default grid (the data range) was
  fine (0.01). The docs example `x_range=[-25, 25]` contains the peaks, which is why it was never noticed.
- Decision B, implemented: a given `x_range` is evaluated together with the data `x` and only its part is kept, so the scale
  comes from the data grid; the error drops to 0.01 for every window; cost about +30 % evaluation for an explicit `x_range`
  only, the default grid is untouched (bit-identical). Documented in `compute_ci_bounds` and `ci/_ci.rst`.
  `test_premise_amplitude_is_the_peak_of_the_evaluated_array` pins the semantic on purpose, it fails the day C lands.
- **Option C, for a future release (not 2.0.0): make the scaling pointwise.** Scale by the analytic peak (the value at the mode)
  instead of the array maximum, so `model(x)` no longer depends on the other `x`; B and the extra data-x evaluation could then
  go. Why it is not now: it changes every fitted number in the library slightly (the amplitude would mean the true peak
  height, not the sampled one: up to ~0.1-0.3 % on typical grids, more on coarse ones), changes what `amplitude` means for
  existing users, breaks the oracle tests, and needs an answer for densities without a finite peak (arcsine, beta with
  alpha < 1, the `inf / inf` branch of `_pdf_scaling`), and for components with no closed-form mode. It also touches
  `FitResult.model(other_x)`, `component_curve` and the distribution classes' `pdf` alike. Plan: decide the semantics for the
  unbounded cases first, add a closed-form `mode` helper for every distribution (`stats()` has most of them), run old and new
  scaling side by side on the benchmark data to quantify the shift, then switch under a major version.

---

# Distribution optimization (branch `distribution-optimization`, merged in PR #141)

## What was done
- `e4b7055`: removed all 44 `@suppress_numpy_warnings()` from `utilities_d.py`, added `test_no_numpy_warnings.py`.
- `6e02284`: removed the dead `FitPlotter._unwrap_plotter` / `_validate_fitted` and the `BaseFitter._format_param` copy.
- `cffb5ea`: benchmark fix: `cdf_pdf_plots` timed the PDF twice, so every "CDF" CSV before this held PDF timings. Timing now
  uses 3 warm-up calls and the median; the placeholder plot title is fixed.
- `bd27449`: `arc_sine_pdf_` / `arc_sine_log_pdf_` evaluate only the support (`out=` / `where=`); plot files are lowercase
  snake_case in `plots/speed/` and `plots/accuracy/`.
- `5050eca` .. `6be103c`: the benchmark tooling, see `benchmarks/README.md`: `results/<host>_<commit>/` run folders with
  `env.json`, `bench_env.py` (environment lock and capture), `run_benchmarks.py` (`--smoke`, `--baseline-ref`, `--against`,
  strict pre-flight, `run.log`, `status.json`), `compare_runs.py` (ratio comparison with a noise-aware verdict),
  `xlogy_vs_log.py`.
- `08ce6d4`: `log_normal_cdf_` (masked `out=` / `where=`), `chi_square_pdf_` (computes only on the in-support values) and
  `uniform_log_cdf_` / `q_exponential_log_cdf_` (new `_log_pos` helper instead of `xlogy(1.0, a)`): all bit-identical to the
  previous output (165 edge / scalar / 0-d / empty / NaN / inf cases) and warning-free as before.
- `f857781`: the constants from `pymultifit/__init__.py` (`LOG`, `EXP`, `LOG1P`, `SQRT`, `XLOGY`, `INF`) are used throughout
  the package. Convention (note in `utilities_d.py`): `LOG1P` where the argument is provably > -1 (its warning is then a
  tripwire), scipy's `log1p` / `xlog1py` where it can reach -1 or below. Also fixed a real leftover from the decorator
  removal: `q_exponential_log_pdf_` warned for `1 < q < 2` and `x < 0`; the warnings test now covers `q` in (1, 2) (the old
  grid had only 0.5, 1, 2, 10) and has a dedicated regression test.
- `023b05e`: `xlogy_vs_log.py` fixed for the `XLOGY` name (it crashed on import).
- `2ad86fa`: `suppress_numpy_warnings` is gone from the package. Its last use, `LogNormalDistribution.stats()`, overflows to
  `inf` for `std` > ~26.6 (the correct value, scipy gives it too, and warns itself) and now uses a local
  `np.errstate(over="ignore")`; `m**2` became `m * m` (Python raised `OverflowError` for `mu` > ~1e154 where scipy returns
  `inf`); the helper and its `functools` import are deleted from `__init__.py`. A test scans the whole package for the name.
- `a9ffeec`: audit of every `stats()` (18 classes) with warnings as errors, over an extreme grid (1e-300..1e308) and a wide
  realistic one (-10..1e6, with 0, tiny and negative values). The shared `btf.stats` helper did not catch any of this: it draws
  500 triples from [-100, 100] (so `mu` <= 100), does not turn warnings into errors, and scipy warns itself in the same call.
  Fixed, all on valid or plainly invalid input:
  - `SkewNormal`: `ZeroDivisionError` for `shape=0` (a normal distribution; scipy gives mean 1, variance 4 for loc 1, scale 2),
    NaN for shapes as small as 5e-324.
  - `ScaledInverseChiSquare`: `ZeroDivisionError` at `df=2` and `df=4`, now `inf` where the moment does not exist.
  - `SymmetricGeneralizedNormal`: warnings for `scale <= 0` / `shape <= 0` (now the NaN dict) and for a huge scale.
  - `QExponential`: `ZeroDivisionError` for `rate=0` (now NaN values).
  - `JohnsonSU`: overflow warnings in `sinh` / `cosh` / `exp` / `expm1`, and `gamma=0` gave NaN (`inf * 0`), now `xi`.
  - `LogNormal`: a huge `mu` with a tiny `std` gave NaN plus a warning (`p - 1` rounded to 0), now `expm1(std**2)`.
  `test_no_numpy_warnings.py` runs `stats()` of all 18 classes over the wide grid (warnings as errors, no exceptions) and a
  test of the degenerate-but-valid cases; both fail on the previous code.

## Q-exponential review (2026-10-20, uncommitted at the time of writing, goes into the next commit)
- `QExponentialDistribution.to_scipy_params()` replaces `utilities_d.q_exp_to_gen_pareto`; the `qexponential` example and
  `benchmarks/accuracy.ipynb` use it (the `literalinclude` line ranges of the class docstring were shifted accordingly).
- `q_exponential_pdf_` / `log_pdf_` / `cdf_` no longer reject `q <= 0` (`reject_x(x, rate, loc=loc)`: `q` is not a positive-only
  shape, every `q < 2` is valid); `q_exponential_cdf_` clips the `log1p` argument at -1, so the CDF is 1 beyond the upper end of
  the support for `q < 1` (it was NaN); `stats()` variance gets the missing `(2 - q)` factor.
- `test_qExponential_d.py` (new) checks the class against `scipy.stats.genpareto`.

## Final Ryzen comparison (`023b05e` against the `a6019cc` baseline, same machine, clean trees, same settings)
Governor `performance`, boost off, energy preference `performance`, threads 1, one core; baseline imported from a worktree.
- Consistently faster: arcsine PDF (-42 % / -38 %, ratio 0.58 -> 0.34, about 3x faster than scipy), laplace CDF (-41 % / -42 %).
- Consistently slower: **none**. The earlier lognorm CDF (+6.5 % / +10 %) is back to +0.1 % / +1.1 % and chi2 PDF
  (+5.4 % / +8.5 %) to -4.6 % / -2.1 %.
- Unproven (one parameter set only): gamma PDF (-7 % / -2 %), arcsine CDF (-14 % / -1 %).
- Everything else within the noise. scipy's own time moved <= 3 % between runs.
- `xlogy_vs_log.py` (Ryzen, 1e5-1e6 points): the controls (uniform, q-exponential, laplace, which no longer call `XLOGY`) give
  1.00x for all variants; `beta_log_cdf_` 0.97-0.99x (the special function dominates); `half_normal_log_cdf_` 0.91-0.92x with
  a masked or plain log. Against the same functions at `6be103c` (before `08ce6d4`): `uniform_log_cdf_` 7.6 -> 4.3 ms (-44 %),
  `q_exponential_log_cdf_` -8 %, `laplace_log_cdf_` unchanged.
- Noise lesson: `beta_log_cdf_` has identical code in both runs (only the `xlogy` -> `XLOGY` rename) yet took 49.3 ms then
  59.2 ms. Absolute times differ up to ~20 % between runs of one function, while the multifit/scipy ratio within a run is
  stable to a few %. Compare ratios, never absolute milliseconds across runs.

## Decisions taken / closed
- `beta` slower than scipy (PDF ratio ~2.3, CDF ~1.1): closed by the support masking of `2eaf7f8` (PDF ~0.6, CDF 0.4-0.8 of
  scipy's time, `benchmarks/SPEED_SUMMARY.md`).
- Stop the speed work here: a bit of slowness is acceptable. The other 12 places with the `ys = np.where(y > 0, y, ...)` pattern
  in `utilities_d.py` are left as they are (the benchmark does not cover them).
- `xlogy` is not worth replacing anywhere else: the remaining `XLOGY(1.0, ...)` sites sit next to `betainc` / `gammainc` and
  similar. Only `half_normal_log_cdf_` (~8 %) would gain; left alone.
- `arc_sine_log_pdf_` with everything inside the support is ~7 % slower than before the decorator removal (all other input
  mixes are faster): acceptable.
- `folded_normal_pdf_` "50 % slower" came from an ad hoc test; foldnorm PDF is within the noise in every run comparison. Closed
  unless it shows up again.

## Still open (none blocks 2.0.0)
- [ ] `stats()` of all 17 exported distributions still raises `OverflowError` (Python `**` / float overflow) for absurd
      parameters (about 1e155 and up) where scipy returns `inf` / `nan`; beta also raises `ZeroDivisionError` for
      `alpha = beta = 1e-300`. Deliberately not fixed (nothing realistic gets near), the new test only covers -10..1e6.
- [ ] `chi_square_pdf_` still warns for `x = +inf` whenever `df != 2` (`inf - inf`), as it did before the decorator
      removal; the warnings test uses finite inputs only.
- [ ] Decide on the `-inf` vs ~-1.6e308 saturation in the `gumbel` / `sym_gen_normal` log functions for extreme tails
      (chosen for speed).
- [ ] The T530 and the Ryzen captures differ in the Python patch version (3.10.12 and 3.10.19); `uv python pin` one of them
      if exactly equal interpreters matter.

---

# Release checklist (v2.0.0)
- [x] `version.py` is `2.0.0`.
- [x] `CHANGELOG.md` 2.0.0 block written from the diffs (2026-10-10, extended 2026-10-21: PR #152, `to_scipy_params`, the
      q-exponential fixes).
- [x] PR #141 (`distribution-optimization`) and PR #152 (`ci-fixes`) merged into `plot-fixes`; CI of #152 passed (tests,
      lint 3.10-3.12, codecov).
- [ ] Commit the q-exponential work (`qExponential_d.py`, `utilities_d.py`, the example, `accuracy.ipynb`,
      `test_qExponential_d.py`) and this documentation pass on `plot-fixes`.
- [ ] Open the PR `plot-fixes` -> `main`, watch the CI after the push: `test_ci_parity.py` compares against an oracle generated
      locally (tolerance now 1e-6) and failed once across platforms.
- [ ] Change `[2.0.0] - Unreleased` in `CHANGELOG.md` to the release date, tag `v2.0.0`, publish.
- [ ] Rebuild the docs with the new `requirements-dev.txt` path (Read the Docs) to confirm it works online.
- [ ] `uv.lock` was rewritten by a newer `uv` (revision 3 -> 5); regenerate with the `uv` you release with if an older
      one cannot read it.
- [ ] `black` would reformat `fitters/backend/baseFitter.py` (pre-existing); decide whether to format it.
- [ ] Untracked files still to decide: `pymultifit-banner.*`, `pymultifit-logo.*` (commit or ignore).
