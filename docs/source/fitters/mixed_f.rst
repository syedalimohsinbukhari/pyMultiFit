Mixed Data Fitter
==================

.. autoclass:: pymultifit.fitters.mixed_f.MixedDataFitter
   :members:
   :show-inheritance:
   :private-members:
   :undoc-members:
   :class-doc-from: class


Currently the MixedDataFitter supports fitting the following models together,

* :class:`~pymultifit.distributions.gaussian_d.GaussianDistribution`
* :class:`~pymultifit.distributions.laplace_d.LaplaceDistribution`
* :class:`~pymultifit.distributions.logNormal_d.LogNormalDistribution`
* :class:`~pymultifit.distributions.skewNormal_d.SkewNormalDistribution`

Recommended Import
------------------

.. code-block:: python

   from pymultifit.fitters import MixedDataFitter

Full Import
-----------

.. code-block:: python

   from pymultifit.fitters.mixed_f import MixedDataFitter
