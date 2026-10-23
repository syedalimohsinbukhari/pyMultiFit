``plot``
========

The :mod:`pymultifit.plot` subpackage holds the plotting machinery for fitted models.
Every fitter exposes it through its cached :attr:`~pymultifit.fitters.backend.baseFitter.BaseFitter.plotter` property, and the most common plots are also available directly on the fitter (:meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.plot_fit`, :meth:`~pymultifit.fitters.backend.baseFitter.BaseFitter.dry_run`).
See :doc:`/plotting` for a guided tour with figures.

Plotter
-------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :class:`~pymultifit.plot.FitPlotter.FitPlotter`
     - Plotting class built from a :class:`~pymultifit.result.FitResult`.

Functions
---------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :func:`~pymultifit.plot.qq_compare`
     - Side-by-side Q-Q plots of the residuals of two fitters.

.. toctree::
   :hidden:

   FitPlotter <FitPlotter>
   qq_compare <qq_compare>
