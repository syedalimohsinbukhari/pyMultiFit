``ci``
======

.. py:currentmodule:: pymultifit.ci

Bootstrap confidence intervals are computed by sampling parameter vectors from the multivariate normal distribution defined by the fitted parameters and their covariance, evaluating the model for every sample, and taking quantiles pointwise.
:func:`compute_ci_bounds` works on a :class:`~pymultifit.result.FitResult`; for everyday use call :meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.confidence_intervals` on a fitter, which wraps it.

Result format
-------------

:func:`compute_ci_bounds` returns a :class:`ConfidenceIntervals` object. It describes itself: the exact ``levels`` in percent (ascending, e.g. ``(68.0, 95.0, 99.7)``), the x-values ``x`` the bands were evaluated at, the number of bootstrap samples ``n_bootstrap``, and the bands of the parts that were computed:

.. list-table::
   :align: center
   :header-rows: 1

   * - Attribute
     - Value
   * - ``overall``
     - ``{level: Band}`` for the total model, ``None`` if ``overall_ci=False``.
   * - ``individual``
     - ``{level: (Band, ...)}`` with one :class:`Band` per component, ``None`` if ``individual_ci=False``.

A :class:`Band` holds the pointwise ``lower``, ``median`` and ``upper`` arrays, so a result is read with attributes and float levels instead of string keys:

.. code-block:: python

   ci = fitter.confidence_intervals(ci_levels=[68, 95, 99.7], rng_engine=42, individual_ci=True)

   ci.levels                      # (68.0, 95.0, 99.7)
   ci.overall[95.0].upper         # the upper edge of the overall 95 % band
   ci.band(0.95)                  # the same band, the level in percentage or decimal form
   ci.band(95, component=1)       # the 95 % band of the second component
   fitter.plotter.plot_confidence_intervals(ci)

Only the bands are kept, not the bootstrap curves. New levels or a new ``x_range`` mean computing again, which a seeded ``rng_engine`` makes reproducible; ask for all the levels at once, the extra ones cost almost nothing.

The flat dictionary layout is available as :meth:`ConfidenceIntervals.as_dict`:

.. list-table::
   :align: center
   :header-rows: 1

   * - Key
     - Value
   * - ``"x_range"``
     - The x-values at which the intervals were evaluated.
   * - ``"overall_ci_<level>"``
     - ``{"lower": ..., "median": ..., "upper": ...}`` for the total model, ``<level>`` being the level in percent without a trailing zero (``95``, ``99.7``).
   * - ``"individual_ci_<level>"``
     - A list with one ``{"lower", "median", "upper"}`` dictionary per component.

Random numbers
--------------

``rng_engine`` is a NumPy :class:`~numpy.random.Generator`, or an integer seed for a new one; ``None`` draws fresh entropy.
To get reproducible and still independent results over several calls, create one generator and pass the same object to all of them: it advances with every call, whereas a new generator from the same integer repeats the draws.

The evaluation window
---------------------

The amplitude of an unnormalised distribution is the height of its peak *on the array it is evaluated on*, so a component's value at an ``x`` depends on the other ``x`` evaluated with it.
Evaluating a narrow ``x_range`` on its own would therefore rescale a component whose peak lies outside the window, and its tail would be drawn as a full peak.
A given ``x_range`` is evaluated together with the fitted data's ``x`` and only the ``x_range`` part is kept, so the scale always comes from the grid the amplitudes were fitted on and the bands follow the fitted curve for any window.
The default grid spans the data range and needs no help.

.. autofunction:: compute_ci_bounds

.. autofunction:: normalize_levels

.. autoclass:: ConfidenceIntervals
   :members: band, as_dict

.. autoclass:: Band

Recommended Import
------------------

.. code-block:: python

   from pymultifit.ci import ConfidenceIntervals, compute_ci_bounds
