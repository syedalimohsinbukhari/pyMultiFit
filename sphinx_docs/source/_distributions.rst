``distributions``
=================

Base Class
----------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
   * - :class:`BaseDistribution <pymultifit.distributions.backend.baseDistribution.BaseDistribution>`
     - Bare-bones class for statistical distributions to provide consistent methods.

Derived Distributions
----------------------

.. list-table::
   :align: center
   :header-rows: 1

   * - Name
     - Description
     - Parameters
   * - :class:`ArcSineDistribution <pymultifit.distributions.arcSine_d.ArcSineDistribution>`
     - ArcSine distribution.
     - :math:`\dfrac{1}{\pi\sqrt{x(1-x)}}`
   * - :class:`BetaDistribution <pymultifit.distributions.beta_d.BetaDistribution>`
     - Beta distribution.
     - :math:`\dfrac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}`
   * - :class:`ChiSquareDistribution <pymultifit.distributions.chiSquare_d.ChiSquareDistribution>`
     - ChiSquare distribution.
     - :math:`\dfrac{1}{2^{k/2}\Gamma(k/2)}x^{\frac{k}{2}-1}\exp\left[-\dfrac{x}{2}\right]`
   * - :class:`ExponentialDistribution <pymultifit.distributions.exponential_d.ExponentialDistribution>`
     - Exponential distribution.
     - :math:`\lambda\exp\left[-\lambda x\right]`
   * - :class:`FoldedNormalDistribution <pymultifit.distributions.foldedNormal_d.FoldedNormalDistribution>`
     - Folded Normal distribution.
     - :math:`\dfrac{1}{\sqrt{2\pi}\sigma}\exp\left[-\dfrac{(x-\mu)^2}{2\sigma^2}\right] + \dfrac{1}{\sqrt{2\pi}\sigma}\exp\left[-\dfrac{(x+\mu)^2}{2\sigma^2}\right]`
   * - :class:`GammaDistributionSS <pymultifit.distributions.gamma_d.GammaDistributionSS>`
     - Gamma distribution with shape-scale parameterization.
     - :math:`\dfrac{1}{\Gamma(\alpha)\theta^\alpha}x^{\alpha - 1}\exp\left[-\dfrac{x}{\theta}\right]`
   * - :class:`GammaDistributionSR <pymultifit.distributions.gamma_d.GammaDistributionSR>`
     - Gamma distribution with shape-rate parameterization.
     - :math:`\dfrac{1}{\Gamma(\alpha)}\lambda^\alpha x^{\alpha - 1}\exp\left[-\lambda x\right]`
   * - :class:`GaussianDistribution <pymultifit.distributions.gaussian_d.GaussianDistribution>`
     - Gaussian distribution.
     - :math:`\dfrac{1}{\sqrt{2\pi}\sigma}\exp\left[-\dfrac{(x-\mu)^2}{2\sigma^2}\right]`
   * - :class:`HalfNormalDistribution <pymultifit.distributions.halfNormal_d.HalfNormalDistribution>`
     - Half-Normal distribution.
     - :math:`\dfrac{1}{\sigma}\sqrt{\dfrac{2}{\pi}}\exp\left[-\dfrac{x^2}{2\sigma^2}\right]`
   * - :class:`LaplaceDistribution <pymultifit.distributions.laplace_d.LaplaceDistribution>`
     - Laplace distribution.
     - :math:`\dfrac{1}{2b}\exp\left[-\dfrac{|x-\mu|}{b}\right]`
   * - :class:`LogNormalDistribution <pymultifit.distributions.logNorm_.LogNormalDistribution>`
     - Log-Normal distribution.
     - :math:`\dfrac{1}{x\sigma\sqrt{2\pi}}\exp\left[-\dfrac{(\ln x-\mu)^2}{2\sigma^2}\right]`
   * - :class:`SkewNormalDistribution <pymultifit.distributions.skewNorm_d.SkewNormalDistribution>`
     - Skew-Normal distribution.
     - :math:`\dfrac{2}{\sigma}\phi\left[\dfrac{x-\mu}{\sigma}\right]\Phi\left[\alpha\left(\dfrac{x-\mu}{\sigma}\right)\right]`
   * - :class:`UniformDistribution <pymultifit.distributions.uniform_d.UniformDistribution>`
     - Uniform distribution.
     - :math:`\dfrac{1}{b-a}\ \forall\ x\in[a,b]\ \text{else}\ 0`
