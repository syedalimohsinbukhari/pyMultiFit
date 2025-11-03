# CHANGELOG

All notable changes to this project will be documented in this file.  
This format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---
## [v1.0.8] - 02/11/2025
**Release:** [v1.0.8](https://pypi.org/project/pymultifit/1.0.8)  
**Pull Request:** https://github.com/syedalimohsinbukhari/pyMultiFit/pull/111  
**Issues:**
- BetaPrime (https://github.com/syedalimohsinbukhari/pyMultiFit/issues/109)  

#### ADDED
- `BetaPrimeDistribution`
- `betaPrime.py` in examples
- updated sphinx documentation.
- added tests to `test_BetaPrimeDistribution`.

---
## [v0.1.2] — 19/08/2024
**Release:** [v0.1.2](https://pypi.org/project/pymultifit/0.1.2)  
**Pull Reqeust:** https://github.com/syedalimohsinbukhari/pyMultiFit/pull/5  
**Issues:**
- Documentation (https://github.com/syedalimohsinbukhari/pyMultiFit/issues/2)
- Amplitude issue (https://github.com/syedalimohsinbukhari/pyMultiFit/issues/6)
- Additional functionalities (https://github.com/syedalimohsinbukhari/pyMultiFit/issues/8)
- Generator functions (https://github.com/syedalimohsinbukhari/pyMultiFit/issues/9)

#### OVERVIEW

---
## [v0.1.1] — 03/08/2024
**Release:** [v0.1.1](https://pypi.org/project/pymultifit/0.1.1)  
**Pull Reqeust:** https://github.com/syedalimohsinbukhari/pyMultiFit/pull/4  
**Issues:**
- Parameter extraction (https://github.com/syedalimohsinbukhari/pyMultiFit/issues/3)

Introduces minor code base enhancements.

#### ADDED
##### Core framework
- Added `self.n_par` for allowing the fitters to be self-aware of number of parameters.
- Added `parameter_extractor` method for extracting parameter values from the fitted model.

---
## [v0.1.0] — 22/07/2024
**Release:** [v0.1.0](https://pypi.org/project/pymultifit/0.1.0)  
**Pull Request:** https://github.com/syedalimohsinbukhari/pyMultiFit/pull/1  
**Issues:**

**First stable version** of `pyMultiFit`, establishing the base framework for multi-distribution fitting and model generation.

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
