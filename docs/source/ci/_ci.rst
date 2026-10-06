``ci``
======

.. py:currentmodule:: pymultifit.ci

Bootstrap confidence intervals are computed by sampling parameter vectors from the multivariate normal distribution defined by the fitted parameters and their covariance, evaluating the model for every sample, and taking quantiles pointwise.
:func:`compute_ci_bounds` works on a :class:`~pymultifit.result.FitResult`; for everyday use call :meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.confidence_intervals` on a fitter, which wraps it.

Result format
-------------

``compute_ci_bounds`` returns a dictionary, where ``<level>`` is the integer percentage (e.g. ``95``):

.. list-table::
   :align: center
   :header-rows: 1

   * - Key
     - Value
   * - ``"x_range"``
     - The x-values at which the intervals were evaluated.
   * - ``"overall_ci_<level>"``
     - ``{"lower": ..., "median": ..., "upper": ...}`` for the total model (if ``overall_ci=True``).
   * - ``"individual_ci_<level>"``
     - A list with one ``{"lower", "median", "upper"}`` dictionary per component (if ``individual_ci=True``).

.. autofunction:: compute_ci_bounds

Recommended Import
------------------

.. code-block:: python

   from pymultifit.ci import compute_ci_bounds
