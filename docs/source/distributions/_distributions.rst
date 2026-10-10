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
   * - :class:`~betaPrime_d.BetaPrimeDistribution`
     - Beta prime distribution.
     - :math:`\dfrac{1}{B(\alpha,\beta)}x^{\alpha-1}(1+x)^{-\alpha-\beta}`
   * - :class:`~chiSquare_d.ChiSquareDistribution`
     - ChiSquare distribution.
     - :math:`\dfrac{1}{2^{k/2}\Gamma(k/2)}x^{\frac{k}{2}-1}\exp\left[-\dfrac{x}{2}\right]`
   * - :class:`~exponential_d.ExponentialDistribution`
     - Exponential distribution.
     - :math:`\lambda\exp\left[-\lambda x\right]`
   * - :class:`~foldedNormal_d.FoldedNormalDistribution`
     - Folded Normal distribution.
     - :math:`\dfrac{1}{\sqrt{2\pi}\sigma}\exp\left[-\dfrac{(x-\mu)^2}{2\sigma^2}\right] + \dfrac{1}{\sqrt{2\pi}\sigma}\exp\left[-\dfrac{(x+\mu)^2}{2\sigma^2}\right]`
   * - :class:`~gamma_d.GammaDistribution`
     - Gamma distribution.
     - :math:`\dfrac{1}{\Gamma(\alpha)\theta^\alpha}x^{\alpha - 1}\exp\left[-\dfrac{x}{\theta}\right]`
   * - :class:`~gaussian_d.GaussianDistribution`
     - Gaussian distribution.
     - :math:`\dfrac{1}{\sqrt{2\pi}\sigma}\exp\left[-\dfrac{(x-\mu)^2}{2\sigma^2}\right]`
   * - :class:`~halfNormal_d.HalfNormalDistribution`
     - Half-Normal distribution.
     - :math:`\dfrac{1}{\sigma}\sqrt{\dfrac{2}{\pi}}\exp\left[-\dfrac{x^2}{2\sigma^2}\right]`
   * - :class:`~johnsonSU_d.JohnsonSUDistribution`
     - Johnson SU distribution.
     - :math:`\dfrac{\delta}{\lambda\sqrt{2\pi}\sqrt{1+z^2}}\exp\left[-\dfrac{1}{2}\left(\gamma+\delta\sinh^{-1}(z)\right)^2\right]`
   * - :class:`~laplace_d.LaplaceDistribution`
     - Laplace distribution.
     - :math:`\dfrac{1}{2b}\exp\left[-\dfrac{|x-\mu|}{b}\right]`
   * - :class:`~logNormal_d.LogNormalDistribution`
     - Log-Normal distribution.
     - :math:`\dfrac{1}{x\sigma\sqrt{2\pi}}\exp\left[-\dfrac{(\ln x-\mu)^2}{2\sigma^2}\right]`
   * - :class:`~generalized.qExponential_d.QExponentialDistribution`
     - q-Exponential distribution.
     - :math:`(2-q)\lambda\exp_q(-\lambda x)`
   * - :class:`~generalized.scaledInvChiSquare_d.ScaledInverseChiSquareDistribution`
     - Scaled inverse chi-squared distribution.
     - :math:`\dfrac{\tau^2(\nu/2)}{\Gamma(\nu/2)}\dfrac{1}{x^{1+(\nu/2)}}\exp\left[-\dfrac{\nu\tau^2}{2x}\right]`
   * - :class:`~skewNormal_d.SkewNormalDistribution`
     - Skew-Normal distribution.
     - :math:`\dfrac{2}{\sigma}\phi\left[\dfrac{x-\mu}{\sigma}\right]\Phi\left[\alpha\left(\dfrac{x-\mu}{\sigma}\right)\right]`
   * - :class:`~generalized.studentsT_d.StudentsTDistribution`
     - Student's t distribution.
     - :math:`\dfrac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)}\left(1+\dfrac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}`
   * - :class:`~generalized.genNorm_d.SymmetricGeneralizedNormalDistribution`
     - Symmetric generalized Normal distribution.
     - :math:`\dfrac{\beta}{2\Gamma(1/\beta)}\exp\left(-\left|\dfrac{x - \mu}{\alpha}\right|^\beta\right)`
   * - :class:`~uniform_d.UniformDistribution`
     - Uniform distribution.
     - :math:`\dfrac{1}{b-a}\ \forall\ x\in[a,b]\ \text{else}\ 0`


Non-distribution function
-------------------------

.. py:currentmodule:: pymultifit.distributions

.. list-table::
   :align: center
   :header-rows: 1


   * - Name
     - Description
     - Function
   * - :class:`~utilities_d.line`
     - First order polynomial.
     - :math:`mx+c`
   * - :class:`~utilities_d.quadratic`
     - Quadratic polynomial.
     - :math:`ax^2+bx+c`
   * - :class:`~utilities_d.cubic`
     - Cubic polynomial.
     - :math:`ax^3+bx^2+cx+d`


.. _amplitude-grid:

Amplitude and the evaluation grid
---------------------------------

With the default ``normalize=False`` a distribution is scaled so that its **peak on the evaluated array** equals ``amplitude``: the values are ``amplitude * pdf(x) / max(pdf(x))``.
The amplitude is therefore the height of the peak *as sampled by the array*, and the value at an ``x`` depends on the other ``x`` evaluated with it:

.. plot::
   :include-source:

   import numpy as np
   from matplotlib import pyplot as plt

   from pymultifit.distributions import GaussianDistribution

   gaussian = GaussianDistribution(amplitude=10, mu=-5, std=2)
   x_wide, x_window = np.linspace(-15, 15, 600), np.linspace(0, 15, 300)

   fig, ax = plt.subplots(figsize=(10, 5))
   ax.plot(x_wide, gaussian.pdf(x_wide), label="evaluated on [-15, 15]")
   ax.plot(x_window, gaussian.pdf(x_window), label="evaluated on [0, 15], the peak is not in the array")
   ax.set_xlabel("x"), ax.set_ylabel("value"), ax.legend(), fig.tight_layout()

The second curve is the tail of the same Gaussian, but it is rescaled so that its largest value in the array, the one at ``x = 0``, equals the amplitude of 10 (a lone point gives exactly ``amplitude``).

This is what makes fits consistent, since a fitter always evaluates on the data it fits, and it is harmless as long as the array contains the peaks. Take care when a model is evaluated on another grid (``fitter.to_result().model(x_new)``, ``component_curve``, a distribution's ``pdf``): include the peaks in it.
The bootstrap confidence intervals do this for you, they evaluate a given ``x_range`` together with the fitted data (see :doc:`/ci/_ci`).

With ``normalize=True`` the values are the true probability density, with an area of one, and do not depend on the array they are evaluated on: ``GaussianDistribution(mu=-5, std=2, normalize=True)``. Use it when you need pointwise values.


.. toctree::
   :hidden:

   BaseDistribution         <baseDistribution>
   ArcSineDistribution      <arcSine_d>
   BetaDistribution         <beta_d>
   BetaPrimeDistribution    <betaPrime_d>
   ChiSquareDistribution    <chiSquare_d>
   ExponentialDistribution  <exponential_d>
   FoldedNormalDistribution <foldedNormal_d>
   GammaDistribution        <gamma_d>
   GaussianDistribution     <gaussian_d>
   HalfNormalDistribution   <halfNormal_d>
   JohnsonSUDistribution    <johnsonSU_d>
   LaplaceDistribution      <laplace_d>
   LogNormalDistribution    <logNormal_d>
   Non-distribution functions <polynomial_d>
   qExponentialDistribution <qExponential_d>
   ScaledInverseChiSquareDistribution <scaledInvChiSquare_d>
   SkewNormalDistribution   <skewNormal_d>
   StudentsTDistribution   <studentsT_d>
   SymGeneralizedNormalDistribution <genNorm_d>
   UniformDistribution      <uniform_d>
   DistributionUtilities    <utilities_d>
