# Changelog

All notable changes to `pyMultiFit` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Unreleased

**PRs:** [PR #114], [PR #116], [PR #120], [PR #123], [PR #124], [PR #125] · **Issues:** [issue #89], [issue #90], [issue #94], [issue #95], [issue #99], [issue #100], [issue #117], [issue #118], [issue #119], [issue #121]

### Added

- Added `QExponentialDistribution` and `StudentsTDistribution` to `pymultifit.distributions.generalized`.
- Added Gumbel PDF/log-PDF/CDF/log-CDF utility functions (`gumbel_pdf_`, `gumbel_log_pdf_`, `gumbel_cdf_`, `gumbel_log_cdf_`) to `distributions.utilities_d` — note: only the standalone utility functions shipped, not a public `GumbelDistribution` class.
- Added a new `pymultifit.plot` subpackage (`FitPlotter`, `_plot_backend`) providing residual plots, Q-Q plots, confidence-interval bounds, prediction intervals, parameter-correlation plots, and gridlines on plot axes, exposed via a cached `BaseFitter.plotter` property.
- Added a confidence-interval computation backend (`fitters.backend._ci_backend`: `compute_ci_bounds`, `compute_individual_ci_base`, `compute_individual_ci_mixed`).
- Added frozen-parameter fitting support across fitters, with auto-padding of a `pn_par`-length `frozen` mask to `n_par` length (emits a `UserWarning`).
- Added an `is_scatter` option to `dry_run()`/plotting methods to render raw data as a scatter plot instead of a line.
- Added `pymultifit.exceptions` module (`pyMultiFitErrors`, `AxesError`).
- Added `pymultifit.typing` module (`ArrayLike`, `NDArray`, `Params_`, `RaggedParams`).
- `MixedDataFitter` can now infer `model_list` from `model_dictionary` keys when `model_list` is omitted.
- Added extensive new examples: `examples/frozen_basefitter.py`, `examples/frozen_mixedfitter.py`, `examples/residuals_gaussian.py`, `examples/residuals_mixed.py`, and an `examples/plot_mechanics/` series covering fit/residual/CI/QQ/prediction-interval/parameter-correlation plotting.

### Changed

- Unified all library docstrings to NumPy style throughout `distributions`, `fitters`, and `generators`, including a documentation/typing rework of `JohnsonSUDistribution`.
- Overhauled type hints across the codebase to modern `X | None` / `from __future__ import annotations` syntax; consolidated ad-hoc type aliases into `pymultifit.typing`.
- `MixedDataFitter` now subclasses `BaseFitter` instead of duplicating its logic, unifying residual/CI/plotting support across all fitters.
- Swapped the `deprecated`/`Deprecated` (`deprecated.sphinx`) dependency for `deprecation`; internal `mark_deprecated`/`md_scipy_like` helpers reworked accordingly.
- Swapped the plotting dependency `mpyez` for `plotez` (`LinePlotConfig`, `plot_xy`).
- Added `statsmodels` as a runtime dependency; added dev tooling (`pytest-cov`, `nox`, `uv`, `pylint`, `isort`, `tqdm`, `pyqt6`, `docutils<0.21`).
- Relaxed the `numpy<2.1.0` upper version pin to an unconstrained `numpy` dependency.

### Fixed

- Tightened NumPy warning suppression in distribution utility functions to only ignore `invalid`/`divide` warnings instead of all warnings.

### Breaking Changes

- Dropped Python 3.9 support — `requires-python` raised from `>=3.9` to `>=3.10`; added a 3.12 classifier.
- Removed the top-level type aliases `pymultifit.OneDArray`, `ListOrNdArray`, `ParamTuple`, `Params_` from `pymultifit/__init__.py`, replaced by `pymultifit.typing.{ArrayLike, NDArray, Params_, RaggedParams}`. Code importing these names from the package root will break.
- `pymultifit.mark_deprecated`'s companion `md_scipy_like` was renamed to the private `_md_scipy_like`, removing it from the public API.
- Removed `BaseFitter._covariance()`; covariance is now accessed via the `covariance` attribute directly.
- Replaced the `mpyez` plotting dependency with `plotez`; external code importing `mpyez.backend.uPlotting.LinePlot` / `mpyez.ezPlotting.plot_xy` through pyMultiFit's imports will need to switch to `plotez`.

## [1.0.9] - 2025-11-03

**PRs:** [PR #112]

### Added

- Added `JohnsonSUDistribution` (PDF, log-PDF, CDF, log-CDF).
- Exported shared math constants (`PI`, `SQRT_TWO`, `TWO_BY_PI`, etc.) from the top-level `pymultifit` package for reuse across distributions.

### Changed

- Moved shared math constants out of `distributions/utilities_d.py` into the top-level package namespace and updated `BetaPrimeDistribution`, `BetaDistribution`, `ExponentialDistribution`, `FoldedNormalDistribution`, `HalfNormalDistribution`, and `SkewNormalDistribution` to reuse them instead of recomputing inline.
- Minor formatting cleanup (PEP 8 spacing) across several distribution files.

### Breaking Changes

- `ExponentialDistribution.stats()` no longer includes a `mode` key in its returned dict.

## [1.0.8] - 2025-11-02

**PRs:** [PR #111] · **Issues:** [issue #109]

### Added

- Added `BetaPrimeDistribution` with full PDF/CDF/log-PDF/log-CDF support (`beta_prime_pdf_`, `beta_prime_cdf_`, `beta_prime_log_pdf_`, `beta_prime_log_cdf_`), an example script, tests, and a docs stub.
- Added the `OneDArray` type alias (`Annotated[NDArray[np.float64], "1D array"]`) as a precise 1D-array type hint, now used across distribution/fitter/generator public signatures.

### Changed

- Migrated public method/function signatures across all distributions, fitters, and generators from the loose `ListOrNdArray` alias to the new `OneDArray` type hint (`ListOrNdArray` itself remains defined and importable).
- Removed the hand-maintained `utilities_d.pyi` stub file in favor of inline type hints.
- Reformatted code across distributions/fitters/examples (consistent double-quoted strings, condensed multi-line calls) — no behavioral change.

### Fixed

- Corrected `ChiSquareDistribution`'s docstring, which incorrectly referenced a nonexistent `GammaDistributionSR` class and misstated the shape/scale relationship to `GammaDistribution`.

### Breaking Changes

- `BetaDistribution.__init__` no longer raises `NegativeAmplitudeError`, `NegativeAlphaError`, or `NegativeBetaError` for invalid `amplitude`/`alpha`/`beta` values — the validation checks were removed with no replacement, so previously-invalid constructor calls now silently succeed.

## [1.0.7] - 2025-10-05

**PRs:** [PR #92], [PR #104] · **Issues:** [issue #88], [issue #102]

### Added

- Added a unified polynomial distribution/fitter family: `LineFunction`, `QuadraticFunction`, `CubicFunction` backend classes and a public `polynomial_f.py` fitter.
- Added a `generalized` distributions subpackage housing `SymmetricGeneralizedNormalDistribution` and `ScaledInverseChiSquareDistribution`.
- Added `.pyi` type-stub coverage for `distributions/utilities_d.py`, plus package-wide type hints/annotations across all distribution and fitter classes.
- Added `mark_deprecated`/`md_scipy_like` decorators and `INF`/`LOG` shorthand constants to the top-level package.
- Added a `noxfile.py` task runner, `CONTRIBUTING.md`, and `update_requirements_doc.py` developer tooling.
- Added `paper/` example scripts (`distribution_.py`, `mg_fitter.py`, `mixed_paper.py`) supporting a JOSS/pyOpenSci paper submission.

### Changed

- Consolidated build metadata into `pyproject.toml` and removed the legacy `setup.py`.
- Refactored `distributions/utilities_d.py` and `MixedDataFitter` internals (`mixed_f.py`) — the latter split its monolithic `_instantiate_fitter` into `_instantiate_bounds`/`_instantiate_class`/`_instantiate_n_par` — without changing public `fit()`/`plot_fit()` call signatures.
- `BaseFitter.__init__` now runs input validation via a new `sanity_check` helper before storing `x_values`/`y_values`.
- Broad typing/formatting pass across all distributions and fitters (explicit return types, `ArrayLike`/`NDArray` typing) as part of a pyOpenSci review cycle.
- Reworked `generators/generators.py` and its `__init__.py` exports.

### Fixed

- Corrected inconsistent `show_individuals` keyword usage between `plot_fit` and its internal plotting call site in `BaseFitter`.

### Breaking Changes

- Merged `GammaDistributionSR` and `GammaDistributionSS` into a single `GammaDistribution`; the `GAMMA_SR`/`GAMMA_SS` string constants were replaced by one `GAMMA` constant.
- Removed the public `Line` distribution (`distributions/others.py`) in favor of the new `LineFunction` class; removed the free-function `backend/polynomials.py` module (`line`/`quadratic`/`cubic` functions) in favor of class-based `backend/polynomial_d.py`.
- Moved `ScaledInverseChiSquareDistribution` out of `distributions/scaledInvChiSquare_d.py` into `distributions/generalized/`, changing its import path.
- Renamed top-level type aliases: `fArray` removed (replaced by direct `np.ndarray`/`ArrayLike` typing), `listOfTuplesOrArray` → `Params_`, `oFloat` → `OptionalFloat`.

## [1.0.6] - 2025-03-09

**PRs:** [PR #86] · **Issues:** [issue #87]

### Fixed

- Corrected `ExponentialDistribution` PDF/log-PDF/CDF to consistently use the reparameterized scale `θ = 1/λ`, fixing an inconsistency with the underlying GammaSR-based implementation.
- Added `np.errstate(divide='ignore')` guards around `np.log(cdf_)` in `beta_log_cdf_` and `chi_square_log_cdf_` to suppress spurious divide-by-zero warnings at distribution boundaries.

### Changed

- Extracted a shared `_plot_fit()` helper (`fitters/utilities_f.py`) used by both `BaseFitter.plot_fit()` and `MixedDataFitter.plot_fit()`, removing duplicated plotting code.
- `plot_fit()` gained an `axis: Optional[Axes]` parameter on both fitters, allowing the fit to be drawn onto an existing matplotlib axes instead of always creating a new figure.

### Breaking Changes

- `BaseFitter.get_fit_values()` and `MixedDataFitter.get_fit_values()` renamed to `get_fitted_curve()`.
- `BaseFitter.plot_fit()` parameter `show_individual` renamed to `show_individuals` (now consistent with `MixedDataFitter`).
- `MixedDataFitter.plot_fit()` no longer returns a `(fig, plotter)` tuple — now returns just `plotter` (matching `BaseFitter`), and dropped the `figure_size` parameter.

## [1.0.5-experimental] - 2025-03-05

### Changed

- Minor internal type-hint touch-up in `BaseFitter` (no functional change).

## [1.0.4-experimental] - 2025-03-05

**PRs:** [PR #81], [PR #82], [PR #84] · **Issues:** [issue #83]

### Added

- Added `GeneralizedNormalDistribution` (symmetric generalized normal) class.
- Added `ScaledInvChiSquareDistribution` class.
- Added `log_pdf_`/`log_cdf_` functions across nearly all distributions in `utilities_d.py` (ArcSine, Beta, ChiSquare, Exponential, FoldedNormal, GammaSR/SS, Gaussian, HalfNormal, Laplace, LogNormal, ScaledInvChiSquare, SymGenNormal).
- Added benchmark summary reporting and new ArcSine accuracy/speed benchmark scripts.
- Added new test modules: `test_gen_norm.py`, `test_scaledInvChiSquare_d.py`, shared `base_test_functions.py` helper.

### Changed

- Reorganized the docs source tree: distribution/fitter/generator/benchmark `.rst` stubs moved into dedicated `distributions/`, `fitters/`, `benchmarks/` subfolders.

## [1.0.3] - 2025-01-14

**PRs:** [PR #80] · **Issues:** [issue #66], [issue #67]

### Added

- `BaseFitter.fit()` gained a `frozen: List[bool]` parameter to lock specific parameters near their initial guess during fitting, via a new `_fit_preprocessing()` helper.
- Added a `LineFitter` class (`fitters/others.py`) for fitting straight-line models.
- Reworked `MixedDataFitter` to delegate to the existing per-distribution fitter classes (`ChiSquareFitter`, `ExponentialFitter`, `FoldedNormalFitter`, `GammaFitterSR/SS`, `HalfNormalFitter`, `LineFitter`, etc.) via a `fitter_dict`, and added a `fitter_dictionary` constructor param for supplying custom fitters — substantially widens which distributions can be mixed-fit.
- Extended `generators/generators.py` to generate synthetic data for many more distributions (ArcSine, Beta, ChiSquare, Exponential, FoldedNormal, GammaSR, etc.), not just Gaussian/Laplace/LogNormal/SkewNormal/Line.

### Changed

- Moved polynomial helper functions (`line`, `linear`, `quadratic`, `cubic`, `nth_polynomial`) from `distributions/others.py` into `distributions/backend/polynomials.py`.
- `SkewNormalDistribution` no longer requires `shape > 0` (removed the `NegativeShapeError` check) — negative shape values are now valid.

### Breaking Changes

- `BaseFitter.get_parameters()` renamed to `get_model_parameters()`.

## [1.0.2] - 2025-01-09

**PRs:** [PR #79] · **Issues:** [issue #78]

### Fixed

- `BaseDistribution.mean`/`median`/`variance`/`stddev`/`mode` properties previously always returned `None`; now correctly delegate to each distribution's `stats()`.

### Changed

- Consolidated duplicated per-distribution `mean`/`median`/`variance`/`stddev` property implementations into single `stats()` methods across `beta_d.py`, `chiSquare_d.py`, `exponential_d.py`, `gamma_d.py`, `gaussian_d.py`, `laplace_d.py`, `logNormal_d.py`, `skewNormal_d.py`, `uniform_d.py`.

## [1.0.1] - 2025-01-07

**PRs:** [PR #77]

### Fixed

- Type-hint compatibility fix in `ChiSquareDistribution` (`int | float` → `Union[int, float]`, PEP 604 syntax incompatible with then-supported Python versions).
- Doc build / Read the Docs configuration fix.

## [1.0.0] - 2025-01-07

Major stabilization release: full test suite, Sphinx documentation site, benchmarks, and an API cleanup/rename pass.

**PRs:** [PR #45], [PR #46], [PR #49], [PR #51], [PR #52], [PR #53], [PR #54], [PR #60], [PR #65], [PR #72], [PR #74], [PR #76] · **Issues:** [issue #7], [issue #39], [issue #42], [issue #48], [issue #50], [issue #58], [issue #59], [issue #61], [issue #62], [issue #63], [issue #69], [issue #71]

### Added

- Added `ArcSineDistribution` and `UniformDistribution`.
- Added `GammaDistributionSR` / `GammaDistributionSS` and matching `GammaFitterSR` / `GammaFitterSS` (shape-rate/shape-scale gamma parameterizations).
- Added a `scipy_like()` classmethod on distributions for scipy-compatible construction.
- Separated, correctness-focused PDF/CDF generation logic.
- Added parameter constraint validation on distributions.
- Added `errorHandling.py` with dedicated exception classes.
- Added a full pytest suite under `src/tests/test_distributions/`.
- Added a `benchmarks/` module and accuracy/speed benchmark docs.
- Added a full Sphinx documentation site — installation, tutorials, per-distribution/fitter API pages.
- Added an `examples/basic/*.py` example script for every distribution.

### Changed

- Standardized constructor/method signatures across distributions.
- Reworked `fitters/backend/baseFitter.py` and `generators/generators.py` internals.
- Bumped `mpyez` dependency 0.0.9a3 → 0.1.0.

### Breaking Changes

- Renamed modules/classes: `logNorm_d`→`logNormal_d`, `skewNorm_d`→`skewNormal_d`, `foldedHalfNormal_d`→`foldedNormal_d` (`FoldedHalfNormalDistribution`→`FoldedNormalDistribution`), with matching fitter renames (`logNorm_f`→`logNormal_f`, `skewNorm_f`→`skewNormal_f`, `foldedHalfNormal_f`→`foldedNormal_f`).
- `GammaDistribution`/`GammaFitter` split into `GammaDistributionSR`/`GammaDistributionSS` and `GammaFitterSR`/`GammaFitterSS` — old single-class API removed.
- Removed `NorrisDistribution` and `PowerLawDistribution` (and their fitters) from the public API.
- Relocated internal utility modules: `distributions/utilities.py`→`utilities_d.py`, `fitters/utilities.py`→`utilities_f.py`.

## [0.2.1] - 2024-12-05

**PRs:** [PR #37], [PR #41] · **Issues:** [issue #34], [issue #36]

### Added

- Added `ChiSquareDistribution` and `ChiSquareFitter`.
- Added `FoldedHalfNormalDistribution` and `FoldedHalfNormalFitter`.
- Added `HalfNormalDistribution` and `HalfNormalFitter`.
- Added `generate_multi_chi_squared_data`, `generate_multi_fhnd_data`, `generate_multi_hnd_data` generator functions.

### Changed

- Consolidated per-distribution utility/PDF helper functions into a new central `distributions/utilities.py` (replacing the old `distributions/backend/utilities.py`).
- Reworked internals of `exponential_d.py`, `gamma_d.py`, `gaussian_d.py`, `laplace_d.py`, `logNorm_d.py`, `beta_d.py`, `powerLaw_d.py`, `norris_d.py`.
- Moved `fitters/backend/utilities.py` → `fitters/utilities.py`.

## [0.2.0] - 2024-12-03

**PRs:** [PR #32], [PR #33] · **Issues:** [issue #22], [issue #31]

### Added

- Added `ExponentialDistribution` class (+ `exponential_()` function), `ExponentialFitter`, and `generate_multi_exponential_data`.

### Breaking Changes

- Unified distribution constructors (`GaussianDistribution`, `BetaDistribution`, `LaplaceDistribution`, `LogNormalDistribution`, `ArcSineDistribution`, etc.): removed the separate `with_amplitude()` classmethods and merged into each class's `__init__`, which now takes `amplitude` and `normalize` directly.
- Flipped the default normalization behavior: PDFs are now amplitude-scaled by default (`normalize=False`) instead of normalized-by-default; pass `normalize=True` explicitly to integrate to 1.

## [0.1.5a0] - 2024-11-30

**PRs:** [PR #26], [PR #28]

### Added

- Added `Norris2005Distribution`/`Norris2011Distribution` (+ `norris2005()`/`norris2011()` functions) and `Norris2005Fitter`/`Norris2011Fitter`.
- Added `generate_multi_norris2005_data` and `generate_multi_norris2011_data` generator functions.

### Changed

- Reimplemented `gaussian_()`'s PDF in pure NumPy, dropping the compiled `sharedLib.cppModels` C extension to remove the platform-specific binary dependency.

### Breaking Changes

- Renamed the internal `_backend` package to `backend` under both `distributions` and `fitters` (import path change).
- `PowerLawDistribution`'s constructor changed from `PowerLawDistribution(alpha)` + separate `with_amplitude(amplitude, alpha)` classmethod to a single `PowerLawDistribution(amplitude=1.0, alpha=-1)`; `with_amplitude`/`powerLawWA` removed.

## [0.1.4] - 2024-11-24

**PRs:** [PR #15], [PR #25] · **Issues:** [issue #16], [issue #18], [issue #21], [issue #23], [issue #24]

### Added

- Added `BaseFitter.get_parameters()` for extracting parameter values/errors for selected sub-models by index.

### Changed

- Adopted `mpyez` for fitter plotting; consolidated `_plot_individual_fitter`/`format_param`, previously duplicated across `GaussianFitter`, `LaplaceFitter`, `LogNormalFitter`, `PowerLawFitter`, `SkewedNormalFitter`, into `BaseFitter`.
- Routed `MixedDataFitter`'s per-model dispatch through a shared `model_dict` lookup instead of a per-model if/elif chain.
- Deprecated `GaussianFitter.parameter_extractor()` in favor of `get_parameters()`.

### Breaking Changes

- `BaseFitter.plot_fit()` signature changed: removed `auto_label`, `fig_size`, `ax` params in favor of explicit `x_label`, `y_label`, `title`, `data_label`, `axis`.
- `MixedDataFitter.__init__()` renamed the `x_data`/`y_data` params to `x_values`/`y_values`.

## [0.1.4a1] - 2024-11-10

### Added

- Added `PowerLawDistribution` class and `PowerLawFitter`.
- Added a `generate_multi_powerlaw_data` generator function.

## [0.1.4a0] - 2024-11-02

### Added

- Added a compiled `sharedLib.cppModels` C++ backend; `GaussianDistribution`'s PDF computation now routes through it.
- Added a `version.py` module (`__version__`, `__author__`, `__email__`, `__license__`, `__url__`, `__description__`) and exposed these from the package `__init__.py`.
- Added `GaussianFitter.get_parameters()` as the successor to `parameter_extractor()`.

### Changed

- Renamed `fitters/_backend/multiFitter.py` to `fitters/_backend/baseFitter.py`.
- Deprecated `GaussianFitter.parameter_extractor()` in favor of `get_parameters()`.

## [0.1.3] - 2024-09-20

**PRs:** [PR #12] · **Issues:** [issue #10], [issue #13]

### Added

- Added module-level string constants (`GAUSSIAN`, `LAPLACE`, `LOG_NORMAL`, `SKEW_NORMAL`, `GAMMA`, `BETA`, `ARCSINE`, `LINE`, `QUADRATIC`, `CUBIC`, etc.) to `pymultifit/__init__.py`.
- Added Laplace model support to `MixedDataFitter`.
- Added input validation (`sanity_check`) to `MixedDataFitter`.

### Fixed

- Fixed `MixedDataFitter` not accounting for `skew_normal` needing 4 parameters in `_expected_param_count()`/`_get_bounds()`, which caused a parameter-count mismatch.
- Fixed `MixedDataFitter.fit()` not flattening nested `p0` before passing to `curve_fit`.

### Changed

- Moved the `others.py` helper module into the `distributions` subpackage.
- Renamed private distribution PDF functions to public (`_gaussian`→`gaussian_`, `_gamma`→`gamma_`, `_laplace`→`laplace_`, `_log_normal`→`log_normal_`).
- Replaced hardcoded model-name string literals with the new module-level constants throughout `MixedDataFitter`.

## [0.1.2] - 2024-08-19

**PRs:** [PR #5] · **Issues:** [issue #2], [issue #6], [issue #8], [issue #9]

### Added

- Added a `distributions` subpackage: `GaussianDistribution`, `LaplaceDistribution`, `LogNormalDistribution`, `SkewedNormalDistribution`, `ArcSineDistribution`, `BetaDistribution`, `GammaDistribution`.
- Added a `generators` subpackage for synthetic multi-modal data generation.
- Added `MixedDataFitter` for fitting a mixture of different model types (Gaussian, LogNormal, SkewedNormal, line) to a single dataset.
- Added an `others.py` module with `line`, `linear`, `quadratic`, `cubic`, `nth_polynomial` helper functions.

### Changed

- Moved fitters into a `fitters` subpackage with a shared `_backend` module.

### Breaking Changes

- Renamed fitter classes: `Gaussian`→`GaussianFitter`, `Laplace`→`LaplaceFitter`, `LogNormal`→`LogNormalFitter`, `SkewedNormal`→`SkewedNormalFitter`.
- Removed top-level re-exports from `pymultifit/__init__.py` — fitters and `BaseFitter` now live under `pymultifit.fitters`/`pymultifit.distributions`, no longer importable directly from `pymultifit`.
- Removed the `pymultifit.backend` module, superseded by `pymultifit.fitters._backend`.

## [0.1.1] - 2024-08-03

**PRs:** [PR #4] · **Issues:** [issue #3]

### Added

- Added a `Laplace` fitter to the top-level `pymultifit` package exports.
- Added `BaseFitter.parameter_extractor()` for extracting amplitude/mu/sigma from fit results (implemented for `Gaussian`).

### Fixed

- Fixed `BaseFitter.get_fit_values()` calling the single-component `_fitter` instead of the multi-component `_n_fitter`, producing wrong overall fit output.

### Changed

- Reordered `BaseFitter.plot_fit()` keyword arguments (`auto_label` now precedes `fig_size`).

### Breaking Changes

- Renamed the `BaseFitter.n_parameters` attribute to `n_par`.

## [0.1.0] - 2024-07-22

Initial public release.

**PRs:** [PR #1]

### Added

- Added the `MultiFitter` base class (`backend/multiFitter.py`) and Gaussian, Laplace, LogNormal, and Skewed-Normal multi-model fitters (`gaussian_f.py`, `laplace_f.py`, `logNorm_f.py`, `skewNorm_f.py`).

[PR #1]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/1
[PR #4]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/4
[PR #5]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/5
[PR #12]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/12
[PR #15]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/15
[PR #25]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/25
[PR #26]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/26
[PR #28]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/28
[PR #32]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/32
[PR #33]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/33
[PR #37]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/37
[PR #41]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/41
[PR #45]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/45
[PR #46]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/46
[PR #49]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/49
[PR #51]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/51
[PR #52]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/52
[PR #53]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/53
[PR #54]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/54
[PR #60]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/60
[PR #65]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/65
[PR #72]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/72
[PR #74]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/74
[PR #76]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/76
[PR #77]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/77
[PR #79]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/79
[PR #80]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/80
[PR #81]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/81
[PR #82]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/82
[PR #84]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/84
[PR #86]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/86
[PR #92]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/92
[PR #104]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/104
[PR #111]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/111
[PR #112]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/112
[PR #114]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/114
[PR #116]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/116
[PR #120]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/120
[PR #123]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/123
[PR #124]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/124
[PR #125]: https://github.com/syedalimohsinbukhari/pyMultiFit/pull/125
[issue #2]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/2
[issue #3]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/3
[issue #6]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/6
[issue #7]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/7
[issue #8]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/8
[issue #9]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/9
[issue #10]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/10
[issue #13]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/13
[issue #16]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/16
[issue #18]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/18
[issue #21]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/21
[issue #22]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/22
[issue #23]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/23
[issue #24]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/24
[issue #31]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/31
[issue #34]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/34
[issue #36]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/36
[issue #39]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/39
[issue #42]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/42
[issue #48]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/48
[issue #50]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/50
[issue #58]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/58
[issue #59]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/59
[issue #61]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/61
[issue #62]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/62
[issue #63]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/63
[issue #66]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/66
[issue #67]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/67
[issue #69]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/69
[issue #71]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/71
[issue #78]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/78
[issue #83]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/83
[issue #87]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/87
[issue #88]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/88
[issue #89]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/89
[issue #90]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/90
[issue #94]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/94
[issue #95]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/95
[issue #99]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/99
[issue #100]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/100
[issue #102]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/102
[issue #109]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/109
[issue #117]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/117
[issue #118]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/118
[issue #119]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/119
[issue #121]: https://github.com/syedalimohsinbukhari/pyMultiFit/issues/121

[2.0.0]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v1.0.9...HEAD
[1.0.9]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v1.0.8...v1.0.9
[1.0.8]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v1.0.7...v1.0.8
[1.0.7]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v1.0.6...v1.0.7
[1.0.6]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v1.0.5-experimental...v1.0.6
[1.0.5-experimental]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v1.0.4-experimental...v1.0.5-experimental
[1.0.4-experimental]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v1.0.3...v1.0.4-experimental
[1.0.3]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v0.2.1...v1.0.0
[0.2.1]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v0.1.5a0...v0.2.0
[0.1.5a0]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v0.1.4...v0.1.5a0
[0.1.4]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v0.1.4a1...v0.1.4
[0.1.4a1]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v0.1.4a0...v0.1.4a1
[0.1.4a0]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v0.1.3...v0.1.4a0
[0.1.3]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/syedalimohsinbukhari/pyMultiFit/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/syedalimohsinbukhari/pyMultiFit/releases/tag/v0.1.0
