``fitters``
===========

Base Class
----------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :class:`~pymultifit.fitters.backend.baseFitter.BaseFitter`
     - The base class for multi-fitting functionality.

Standalone Fitter
-----------------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :class:`~pymultifit.fitters.mixed_f.MixedDataFitter`
     - Mixed model fitting class.


Derived Fitters
---------------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :class:`~pymultifit.fitters.chiSquare_f.ChiSquareFitter`
     - ChiSquare fitter.
   * - :class:`~pymultifit.fitters.exponential_f.ExponentialFitter`
     - Exponential fitter.
   * - :class:`~pymultifit.fitters.foldedNormal_f.FoldedNormalFitter`
     - Folded Normal fitter.
   * - :class:`~pymultifit.fitters.gamma_f.GammaFitterSR`
     - Gamma fitter with shape and rate parametrization.
   * - :class:`~pymultifit.fitters.gamma_f.GammaFitterSS`
     - Gamma fitter with shape and scale parametrization.
   * - :class:`~pymultifit.fitters.gaussian_f.GaussianFitter`
     - Gaussian fitter.
   * - :class:`~pymultifit.fitters.halfNormal_f.HalfNormalFitter`
     - Half-Normal fitter.
   * - :class:`~pymultifit.fitters.laplace_f.LaplaceFitter`
     - Laplace fitter.
   * - :class:`~pymultifit.fitters.logNormal_f.LogNormalFitter`
     - Log-Normal fitter.
   * - :class:`~pymultifit.fitters.skewNormal_f.SkewNormalFitter`
     - Skew-Normal fitter.

.. toctree::
   :hidden:

   BaseFitter         <fitters/baseFitter>
   ChiSquareFitter    <fitters/chiSquare_f>
   ExponentialFitter  <fitters/exponential_f>
   FoldedNormalFitter <fitters/foldedNormal_f>
   GammaFitterSR      <fitters/gamma_sr_f>
   GammaFitterSS      <fitters/gamma_ss_f>
   GaussianFitter     <fitters/gaussian_f>
   HalfNormalFitter   <fitters/halfNormal_f>
   LaplaceFitter      <fitters/laplace_f>
   LogNormalFitter    <fitters/logNormal_f>
   MixedDataFitter    <fitters/mixed_f>
   SkewNormalFitter   <fitters/skewNormal_f>
