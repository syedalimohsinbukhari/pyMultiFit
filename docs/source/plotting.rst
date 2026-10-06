Plotting a fit
==============

A fit is only as good as it looks: **pyMultiFit** ships the plots needed to visually assess a fit, and every fitter exposes them through its ``plotter`` property.
This page walks through each of them; all figures below are generated from the code shown next to them when the documentation is built.

.. contents::
   :local:
   :depth: 1

Setup
-----

Three fits are used throughout: five Gaussians fitted with :class:`~pymultifit.fitters.gaussian_f.GaussianFitter`, a line plus three peaks fitted with :class:`~pymultifit.fitters.mixed_f.MixedDataFitter`, and a small, noisy two-Gaussian data set (``nf``) where the uncertainty bands are easy to see.

.. plot::
   :context: reset
   :include-source:

   import numpy as np
   from matplotlib import pyplot as plt

   from pymultifit import GAUSSIAN, LAPLACE, LINE
   from pymultifit.fitters import GaussianFitter, LaplaceFitter, MixedDataFitter
   from pymultifit.generators import multi_gaussian, multiple_models

   # five Gaussians
   x = np.linspace(-35, 35, 1500)
   true_g = [(20, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]
   y = multi_gaussian(x, params=true_g, noise_level=0.2)

   gf = GaussianFitter(x, y)
   gf.fit([(10, -18, 1), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)])

   # a line plus three peaks
   models = [LINE, GAUSSIAN, LAPLACE, GAUSSIAN]
   y_mixed = multiple_models(x, params=[(-0.1, 3), (8, -15, 3), (6, 5, 2), (4, 20, 4)], model_list=models, noise_level=0.1)

   mf = MixedDataFitter(x, y_mixed, model_list=models)
   mf.fit([(0, 2), (6, -15, 2), (4, 5, 1), (3, 20, 3)])

   # two noisy Gaussians
   x_n = np.linspace(-15, 15, 500)
   y_n = multi_gaussian(x_n, params=[(10, -5, 2), (8, 5, 3)], noise_level=0.7)

   nf = GaussianFitter(x_n, y_n)
   nf.fit([(8, -4, 1.5), (6, 4, 2)])

Looking at the data before fitting
----------------------------------

:meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.dry_run` plots the raw data, which helps in choosing initial guesses. It works before :meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.fit` is called.

.. plot::
   :context: close-figs
   :include-source:

   GaussianFitter(x, y).dry_run(is_scatter=True)

The fit
-------

:meth:`~pymultifit.plot.FitPlotter.plot_fit` draws the data and the total fit. With ``show_individuals=True`` every component is drawn as a dashed line, labelled with its fitted parameters.

.. plot::
   :context: close-figs
   :include-source:

   fig, ax = plt.subplots(figsize=(10, 5))
   gf.plotter.plot_fit(show_individuals=True, x_label="X data", y_label="Amplitude", axis=ax)
   fig.tight_layout()

The same call works for a :class:`~pymultifit.fitters.mixed_f.MixedDataFitter`.

.. plot::
   :context: close-figs
   :include-source:

   fig, ax = plt.subplots(figsize=(10, 5))
   mf.plotter.plot_fit(show_individuals=True, axis=ax)
   fig.tight_layout()

Residuals
---------

:meth:`~pymultifit.plot.FitPlotter.plot_residuals` shows ``data - model``. The numeric values are available from :meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.get_residuals`.

.. plot::
   :context: close-figs
   :include-source:

   fig, ax = plt.subplots(figsize=(10, 3))
   gf.plotter.plot_residuals(is_scatter=True, axis=ax)
   fig.tight_layout()

:meth:`~pymultifit.plot.FitPlotter.plot_fit_and_residuals` stacks both. Pass your own pair of axes to control the figure; when ``axes`` is omitted a 3:1 two-panel figure is created.

.. plot::
   :context: close-figs
   :include-source:

   fig, (ax_fit, ax_res) = plt.subplots(
       nrows=2, ncols=1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
   )
   gf.plotter.plot_fit_and_residuals(show_individuals=True, axes=(ax_fit, ax_res))
   fig.tight_layout()

Q-Q plot
--------

:meth:`~pymultifit.plot.FitPlotter.plot_qq_plot` compares the residuals with a normal distribution; the Pearson ``r`` in the legend is a quick gauge, and systematic curvature points to a wrong model family.
:func:`~pymultifit.plot.qq_compare` puts the Q-Q plots of two fitters (or their :class:`~pymultifit.result.FitResult`) side by side.

.. plot::
   :context: close-figs
   :include-source:

   from pymultifit.plot import qq_compare

   lf = LaplaceFitter(x, y)  # the wrong model family for Gaussian-shaped data
   lf.fit([(20, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)])

   qq_compare(fitter_left=gf, fitter_right=lf)

Confidence intervals
--------------------

:meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.confidence_intervals` computes bootstrap confidence intervals (see :doc:`/ci/_ci`); with ``plot=True`` it also draws them.
Levels can be given as percentages or decimals (``95``, ``95.0``, ``0.95``, ``[68, 0.95]``) and the bands never overwrite the labels and title already on the axes.
Several levels are drawn as nested bands, and ``individual_ci=True`` adds one band per component.

.. plot::
   :context: close-figs
   :include-source:

   fig, ax = plt.subplots(figsize=(10, 5))
   nf.plotter.plot_fit(axis=ax)
   nf.confidence_intervals(ci_levels=[68, 95], n_bootstrap=300, seed=42, plot=True, axis=ax)
   ax.set_title("68% and 95% bootstrap confidence intervals")
   fig.tight_layout()

The intervals can also be evaluated beyond the fitted range with ``x_range``, and an already computed result can be re-plotted without recomputing it:

.. plot::
   :context: close-figs
   :include-source:

   x_wide = np.linspace(-25, 25, 600)
   results = nf.confidence_intervals(ci_levels=[68, 95], n_bootstrap=300, seed=42, x_range=x_wide)

   fig, ax = plt.subplots(figsize=(10, 5))
   nf.plotter.plot_fit(axis=ax)
   nf.plotter.plot_confidence_intervals(ci_levels=[68, 95], results=results, axis=ax)
   ax.set_title("Confidence intervals evaluated beyond the fitted range")
   fig.tight_layout()

Prediction intervals
--------------------

:meth:`~pymultifit.plot.FitPlotter.plot_prediction_intervals` shows where *new observations* are expected to fall. They are wider than confidence intervals because they include the scatter of individual observations on top of the parameter uncertainty.

.. plot::
   :context: close-figs
   :include-source:

   fig, ax = plt.subplots(figsize=(10, 5))
   nf.plotter.plot_prediction_intervals(pi_level=[68, 95], axis=ax)
   fig.tight_layout()

Parameter correlations
----------------------

:meth:`~pymultifit.plot.FitPlotter.plot_parameter_correlation` is a heatmap of the correlation matrix derived from the covariance of the fit; values near ±1 signal parameters that are hard to tell apart.
Labels are generated automatically (``p1, p2, ...`` for single-model fitters, ``Gaussian_2_p1, ...`` for a mixed fit) or can be supplied.

.. plot::
   :context: close-figs
   :include-source:

   fig, ax = plt.subplots(figsize=(7, 7))
   mf.plotter.plot_parameter_correlation(axis=ax)
   fig.tight_layout()

Saving a figure
---------------

:meth:`~pymultifit.plot.FitPlotter.save_plot` saves the current (or a given) figure; the format is taken from the file extension.

.. code-block:: python

   fig, ax = plt.subplots(figsize=(10, 5))
   fitter.plotter.plot_fit(axis=ax)
   fitter.plotter.save_plot("fit.pdf", figure=fig, dpi=200)

How plotting works
------------------

The plotting code does not look inside the fitters. Instead, a fitter describes its current state as an immutable :class:`~pymultifit.result.FitResult` (via :meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.to_result`), and a :class:`~pymultifit.plot.FitPlotter` is built from that result.
The plotter is cached on the fitter and is rebuilt after every ``fit()``, so it always reflects the latest fit.

.. code-block:: text

   fitter  --to_result()-->  FitResult  -->  FitPlotter(result)
                                  \--------->  compute_ci_bounds(result)

Because :mod:`pymultifit.plot`, :mod:`pymultifit.ci` and :mod:`pymultifit.result` do not import the fitters, they can be used with any object that can produce a ``FitResult``.
