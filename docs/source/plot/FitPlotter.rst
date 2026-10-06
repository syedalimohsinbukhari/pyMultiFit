FitPlotter
==========

.. autoclass:: pymultifit.plot.FitPlotter.FitPlotter
   :members:
   :class-doc-from: class

Recommended Import
------------------

.. code-block:: python

   from pymultifit.plot import FitPlotter

In day-to-day use the plotter is obtained from a fitter rather than constructed by hand:

.. code-block:: python

   fitter.fit(p0)
   fitter.plotter.plot_fit(show_individuals=True)
