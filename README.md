# `pyMultiFit`

- [`pyMultiFit`](#pymultifit)
  - [What is `pymultifit`](#what-is-pymultifit)
  - [How to install](#how-to-install)
  - [Modules](#modules)

A python multi-fit library for fitting the data with multiple `X` fitters.

![GitHub-licence](https://img.shields.io/github/license/syedalimohsinbukhari/pymultifit?style=for-the-badge&color=darkblue)
![GitHub top language](https://img.shields.io/github/languages/top/syedalimohsinbukhari/pymultifit?color=lightgreen&style=for-the-badge)
![GitHub contributors](https://img.shields.io/github/contributors/syedalimohsinbukhari/pymultifit?color=gold&style=for-the-badge)
![Github Issues](https://img.shields.io/github/issues/syedalimohsinbukhari/pymultifit?color=orange&style=for-the-badge)
![GitHub OPEN PRs](https://img.shields.io/github/issues-pr/syedalimohsinbukhari/pymultifit?color=darkred&style=for-the-badge)
![GitHub CLOSED PRs](https://img.shields.io/github/issues-pr-closed/syedalimohsinbukhari/pymultifit?color=darkgreen&style=for-the-badge)

## What is `pymultifit`

`pymultifit` is a library made specifically to tackle one problem, **fit the data with multiple fitters**.

Fitter implementations include,

- `Gaussian` fitter,
- `SkewedNormal` fitter,
- `LogNormal` fitter,
- `Laplace` fitter, and more.

Additionally, it provides capabilities to generated n-modal data as well through its `generators` module.
Along with this, the user can also generate probability distribution data using `distributions` module.

## How to install

Using pip: `pip install pymultifit`

## Modules

The following modules are currently implemented in `pymultifit` library,

1. [`distributions`](https://github.com/syedalimohsinbukhari/pyMultiFit/tree/main/docs/distributions.md)
2. [`fitters`](https://github.com/syedalimohsinbukhari/pyMultiFit/tree/main/docs/fitters.md)
3. [`generators`](https://github.com/syedalimohsinbukhari/pyMultiFit/tree/main/docs/generators.md)