.. _api:

pyMultiFit API
==============

The Application Programming Interface (API) of `pyMultiFit` provides tools for statistical data generation and model fitting, organized into three primary layers:

* **Distributions**: Statistical distributions for data modeling, generation, and fitting.
* **Fitters**: Classes implementing fitting algorithms for statistical models.
* **Generators**: Functions for generating synthetic datasets.

--------------------------------------------

distributions
-------------

The `distributions` module provides tools for defining, generating, and using statistical distributions.
It includes a base distributions and several specific implementations.

Base Class
^^^^^^^^^^
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :class:`BaseDistribution <pymultifit.distributions.backend.baseDistribution.BaseDistribution>`
     - Bare-bones class for statistical distributions to provide consistent methods.

Derived Distributions
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :class:`ArcSineDistribution <pymultifit.distributions.arcSine_d.ArcSineDistribution>`
     - ArcSine distribution.
   * - :class:`BetaDistribution <pymultifit.distributions.beta_d.BetaDistribution>`
     - Beta distribution.
   * - :class:`ChiSquareDistribution <pymultifit.distributions.chiSquare_d.ChiSquareDistribution>`
     - ChiSquare distribution.
   * - :class:`ExponentialDistribution <pymultifit.distributions.exponential_d.ExponentialDistribution>`
     - Exponential distribution.
   * - :class:`FoldedNormalDistribution <pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution>`
     - Folded Normal distribution.
   * - :class:`GammaDistributionSR <pymultifit.distributions.gamma_d.GammaDistributionSR>`
     - Gamma distribution with shape-rate parameterization.
   * - :class:`GammaDistributionSS <pymultifit.distributions.gamma_d.GammaDistributionSS>`
     - Gamma distribution with shape-scale parameterization.
   * - :class:`GaussianDistribution <pymultifit.distributions.gaussian_d.GaussianDistribution>`
     - Gaussian distribution.
   * - :class:`HalfNormalDistribution <pymultifit.distributions.halfNormal_d.HalfNormalDistribution>`
     - Half-Normal distribution.
   * - :class:`LaplaceDistribution <pymultifit.distributions.laplace_d.LaplaceDistribution>`
     - Laplace distribution.
   * - :class:`LogNormalDistribution <pymultifit.distributions.logNorm_.LogNormalDistribution>`
     - Log-Normal distribution.
   * - :class:`SkewNormalDistribution <pymultifit.distributions.skewNorm_d.SkewNormalDistribution>`
     - Skew-Normal distribution.
   * - :class:`UniformDistribution <pymultifit.distributions.uniform_d.UniformDistribution>`
     - Uniform distribution.

--------------------------------------------

fitters
-------

The `fitters` module provides tools for fitting single, multiple and mixed models.
It includes a base fitter method for user-defined fitting algorithms and several specific implementations.

Base Class
^^^^^^^^^^
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :class:`BaseFitter <pymultifit.fitters.backend.baseFitter.BaseFitter>`
     - The base class for multi-fitting functionality.
   * - :class:`MixedFitter <pymultifit.fitters.mixed_f.MixedFitter>`
     - Mixed model fitting class.

Derived Fitters
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 75
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
   * - :class:`LogNormalFitter <pymultifit.fitters.logNorm_f.LogNormalFitter>`
     - Log-Normal fitter.
   * - :class:`SkewNormalFitter <pymultifit.fitters.skewNorm_f.SkewNormalFitter>`
     - Skew-Normal fitter.

--------------------------------------------

generators
----------

The `generators` module provides functions for generating single or multimodal data for specific distributions or mixed models.

Base function
^^^^^^^^^^^^^
.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - :class:`BaseFitter <pymultifit.fitters.backend.baseFitter.BaseFitter>`
     - The base class for multi-fitting functionality.
   * - :class:`MixedFitter <pymultifit.fitters.mixed_f.MixedFitter>`
     - Mixed model fitting class.

Derived Fitters
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 75
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
   * - :class:`LogNormalFitter <pymultifit.fitters.logNorm_f.LogNormalFitter>`
     - Log-Normal fitter.
   * - :class:`SkewNormalFitter <pymultifit.fitters.skewNorm_f.SkewNormalFitter>`
     - Skew-Normal fitter.
