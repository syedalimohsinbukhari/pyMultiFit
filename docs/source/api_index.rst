pyMultiFit API
==============

The Application Programming Interface (API) of `pyMultiFit` provides tools for statistical data generation, model fitting and fit diagnostics, organized into the following layers:

* **Distributions**: Statistical distributions for data modeling, generation, and fitting.
* **Fitters**: Classes implementing fitting algorithms for statistical models.
* **Generators**: Functions for generating synthetic datasets.
* **Plotting**: :class:`~pymultifit.plot.FitPlotter` and friends, for visually assessing a fit (fit, residuals, Q-Q, confidence and prediction intervals, parameter correlations).
* **Results**: :class:`~pymultifit.result.FitResult`, an immutable description of a fit that the plotting and confidence-interval code consume.
* **Confidence intervals**: :func:`~pymultifit.ci.compute_ci_bounds`, bootstrap confidence intervals computed from a :class:`~pymultifit.result.FitResult`.

The documentation first goes through a birdseye view for each module, followed by detailed documentation for each class and function.

.. toctree::
   :maxdepth: 2
   :hidden:

   distributions/_distributions
   fitters/_fitters
   generators/_generators
   plot/_plot
   results/_results
   ci/_ci
   others/_error_handling
   others/_constants
