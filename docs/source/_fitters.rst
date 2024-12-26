``fitters``
===========

Base Class
----------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :class:`BaseFitter <pymultifit.fitters.backend.baseFitter.BaseFitter>`
     - The base class for multi-fitting functionality.
   * - :class:`MixedFitter <pymultifit.fitters.mixed_f.MixedDataFitter>`
     - Mixed model fitting class.

Derived Fitters
---------------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :class:`ChiSquaredFitter <pymultifit.fitters.chiSquare_f.ChiSquareFitter>`
     - ChiSquare fitter.
   * - :class:`ExponentialFitter <pymultifit.fitters.exponential_f.ExponentialFitter>`
     - Exponential fitter.
   * - :class:`FoldedNormalFitter <pymultifit.fitters.foldedNormal_f.FoldedNormalFitter>`
     - Folded Normal fitter.
   * - :class:`GaussianFitter <pymultifit.fitters.gaussian_f.GaussianFitter>`
     - Gaussian fitter.
   * - :class:`HalfNormalFitter <pymultifit.fitters.halfNormal_f.HalfNormalFitter>`
     - Half-Normal fitter.
   * - :class:`LaplaceFitter <pymultifit.fitters.laplace_f.LaplaceFitter>`
     - Laplace fitter.
   * - :class:`LogNormalFitter <pymultifit.fitters.logNormal_f.LogNormalFitter>`
     - Log-Normal fitter.
   * - :class:`SkewNormalFitter <pymultifit.fitters.skewNormal_f.SkewNormalFitter>`
     - Skew-Normal fitter.

.. toctree::
   :hidden:

   ChiSquareFitter    <fitters/chiSquare_f>
   ExponentialFitter  <fitters/exponential_f>
   FoldedNormalFitter <fitters/foldedNormal_f>
   GammaFitterSR      <fitters/gamma_sr_f>
   GammaFitterSS      <fitters/gamma_ss_f>
   GaussianFitter     <fitters/gaussian_f>
   HalfNormalFitter   <fitters/halfNormal_f>
   LaplaceFitter      <fitters/laplace_f>
   LogNormalFitter    <fitters/logNormal_f>
   SkewNormalFitter   <fitters/skewNormal_f>
