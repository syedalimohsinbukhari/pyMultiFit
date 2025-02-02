``distributions``
=================

Base Class
----------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :class:`~pymultifit.distributions.backend.baseDistribution.BaseDistribution`
     - Bare-bones class for statistical distributions to provide consistent methods.

Derived Distributions
----------------------

.. py:currentmodule:: pymultifit.distributions

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
     - PDF function
   * - :class:`~arcSine_d.ArcSineDistribution`
     - ArcSine distribution.
     - :math:`\dfrac{1}{\pi\sqrt{x(1-x)}}`
   * - :class:`~beta_d.BetaDistribution`
     - Beta distribution.
     - :math:`\dfrac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}`
   * - :class:`~chiSquare_d.ChiSquareDistribution`
     - ChiSquare distribution.
     - :math:`\dfrac{1}{2^{k/2}\Gamma(k/2)}x^{\frac{k}{2}-1}\exp\left[-\dfrac{x}{2}\right]`
   * - :class:`~exponential_d.ExponentialDistribution`
     - Exponential distribution.
     - :math:`\lambda\exp\left[-\lambda x\right]`
   * - :class:`~foldedNormal_d.FoldedNormalDistribution`
     - Folded Normal distribution.
     - :math:`\dfrac{1}{\sqrt{2\pi}\sigma}\exp\left[-\dfrac{(x-\mu)^2}{2\sigma^2}\right] + \dfrac{1}{\sqrt{2\pi}\sigma}\exp\left[-\dfrac{(x+\mu)^2}{2\sigma^2}\right]`
   * - :class:`~gamma_d.GammaDistributionSS`
     - Gamma distribution with shape-scale parameterization.
     - :math:`\dfrac{1}{\Gamma(\alpha)\theta^\alpha}x^{\alpha - 1}\exp\left[-\dfrac{x}{\theta}\right]`
   * - :class:`~gamma_d.GammaDistributionSR`
     - Gamma distribution with shape-rate parameterization.
     - :math:`\dfrac{1}{\Gamma(\alpha)}\lambda^\alpha x^{\alpha - 1}\exp\left[-\lambda x\right]`
   * - :class:`~generalized.genNorm_d.SymmetricGeneralizedNormalDistribution`
     - Symmetric Generalized Normal distribution.
     - :math:`\dfrac{\beta}{2\alpha\Gamma(1/\beta)}\exp\left[-\dfrac{|x-\mu|^\beta}{\alpha^\beta}\right]`
   * - :class:`~gaussian_d.GaussianDistribution`
     - Gaussian distribution.
     - :math:`\dfrac{1}{\sqrt{2\pi}\sigma}\exp\left[-\dfrac{(x-\mu)^2}{2\sigma^2}\right]`
   * - :class:`~halfNormal_d.HalfNormalDistribution`
     - Half-Normal distribution.
     - :math:`\dfrac{1}{\sigma}\sqrt{\dfrac{2}{\pi}}\exp\left[-\dfrac{x^2}{2\sigma^2}\right]`
   * - :class:`~laplace_d.LaplaceDistribution`
     - Laplace distribution.
     - :math:`\dfrac{1}{2b}\exp\left[-\dfrac{|x-\mu|}{b}\right]`
   * - :class:`~logNormal_d.LogNormalDistribution`
     - Log-Normal distribution.
     - :math:`\dfrac{1}{x\sigma\sqrt{2\pi}}\exp\left[-\dfrac{(\ln x-\mu)^2}{2\sigma^2}\right]`
   * - :class:`~skewNormal_d.SkewNormalDistribution`
     - Skew-Normal distribution.
     - :math:`\dfrac{2}{\sigma}\phi\left[\dfrac{x-\mu}{\sigma}\right]\Phi\left[\alpha\left(\dfrac{x-\mu}{\sigma}\right)\right]`
   * - :class:`~uniform_d.UniformDistribution`
     - Uniform distribution.
     - :math:`\dfrac{1}{b-a}\ \forall\ x\in[a,b]\ \text{else}\ 0`

Other functions
---------------

.. py:currentmodule:: pymultifit.distributions.backend.polynomials

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :func:`line`
     - Line function.
   * - :func:`linear`
     - Similar to :func:`line`.
   * - :func:`quadratic`
     - Quadratic function
   * - :func:`cubic`
     - Cubic function
   * - :func:`nth_polynomial`
     - Polynomial function


.. toctree::
   :hidden:

   BaseDistribution         <baseDistribution>
   ArcSineDistribution      <arcSine_d>
   BetaDistribution         <beta_d>
   ChiSquareDistribution    <chiSquare_d>
   ExponentialDistribution  <exponential_d>
   FoldedNormalDistribution <foldedNormal_d>
   GammaDistributionSR      <gamma_sr_d>
   GammaDistributionSS      <gamma_ss_d>
   GaussianDistribution     <gaussian_d>
   HalfNormalDistribution   <halfNormal_d>
   LaplaceDistribution      <laplace_d>
   LogNormalDistribution    <logNormal_d>
   ScaledInverseChiSquareDistribution <scaledInvChiSquare_d>
   SkewNormalDistribution   <skewNormal_d>
   SymGeneralizedNormalDistribution <genNorm_d>
   UniformDistribution      <uniform_d>
   DistributionUtilities    <utilities_d>
   OtherFunctions           <others>
