# CHANGELOG

All notable changes to this project will be documented in this file.  
This format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---
## [v0.1.2] — 19/08/2024
**Release:** [v0.1.2](https://pypi.org/project/pymultifit/0.1.2)  
**Pull Reqeust:** [#5 | [MINOR] code base enhancements - II](https://github.com/syedalimohsinbukhari/pyMultiFit/pull/5)  
**Issues:**
---
## [v0.1.1] — 03/08/2024
**Release:** [v0.1.1](https://pypi.org/project/pymultifit/0.1.1)  
**Pull Reqeust:** [#4 | [MINOR] code base enhancements - I](https://github.com/syedalimohsinbukhari/pyMultiFit/pull/4)  
**Issues:**

#### OVERVIEW
This release introduces minor code base enhancements.

#### ADDED
##### Core framework
- Added `self.n_par` for allowing the fitters to be self-aware of number of parameters.
- Added `parameter_extractor` method for extracting parameter values from the fitted model.

---
## [v0.1.0] — 22/07/2024
**Release:** [v0.1.0](https://pypi.org/project/pymultifit/0.1.0)  
**Pull Request:** [#1 | ENH: base multi-fit function](https://github.com/syedalimohsinbukhari/pyMultiFit/pull/1)
**Issues:**
- [#2 | DOC: Needs documentation](https://github.com/syedalimohsinbukhari/pyMultiFit/issues/2)
- [#3 | URG: Extracting specfic parameter values](https://github.com/syedalimohsinbukhari/pyMultiFit/issues/3)

### Overview
This release introduces the **first stable version** of `pyMultiFit`, establishing the base framework for multi-distribution fitting and model generation.

### Added

#### Core Framework
- Introduced the `BaseFitter` class providing a unified API for multi-distribution fitting, plotting, and parameter extraction.  
- Added multi-data generation utilities for synthetic dataset creation.

#### Derived Fitters
- Implemented the following distribution fitters:
  - `Gaussian`
  - `Laplace`
  - `LogNormal`
  - `SkewNormal`

#### Examples
- Added functional examples demonstrating each derived fitter in `examples/`.

#### Configuration & Packaging
- Fully configured Python package structure:
  - `pyproject.toml`, `setup.py`, `setup.cfg`
  - `requirements.txt` with dependency management
- CI/CD setup:
  - `pylint.yml` for linting
  - `publish_pymultifit.yml` for automated publishing
- Ensured **Python 3.9+** compatibility.
---
## [17/07/2024] — Initial Commit
Project repository initialization.
