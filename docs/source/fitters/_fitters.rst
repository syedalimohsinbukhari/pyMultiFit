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
   * - :class:`~pymultifit.fitters.gamma_f.GammaFitter`
     - Gamma fitter with shape and rate parametrization.
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

   BaseFitter         <baseFitter>
   ChiSquareFitter    <chiSquare_f>
   ExponentialFitter  <exponential_f>
   FoldedNormalFitter <foldedNormal_f>
   GammaFitter        <gamma_f>
   GaussianFitter     <gaussian_f>
   HalfNormalFitter   <halfNormal_f>
   LaplaceFitter      <laplace_f>
   LogNormalFitter    <logNormal_f>
   MixedDataFitter    <mixed_f>
   SkewNormalFitter   <skewNormal_f>
   FitterUtilities    <utilities_f>
