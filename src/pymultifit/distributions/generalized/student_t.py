"""Created on July 28 11:15:23 2026"""

from scipy.stats import cauchy as scipy_cauchy
from scipy.stats import norm as scipy_norm
from scipy.stats import t as scipy_t

from ..backend import BaseDistribution, errorHandling as erH
from ...typing import ArrayLike, NDArray



class StudentsTDistribution(BaseDistribution):
    r"""
    Class for :class:`~.StudentsTDistribution`.

    Parameters
    ----------
    v :
        Degrees of freedom parameter, :math:`v`. Must be strictly positive (:math:`v > 0`). Defaults to 1.0.
    loc :
        The location parameter, :math:`\mu`. Defaults to 0.0.
    scale :
        The scale parameter, :math:`\sigma`. Defaults to 1.0. Must be strictly positive.

    Notes
    -----
    **Analytical Log-PDF Derivation:**

    To prevent numerical instability or overflow when evaluating Gamma functions at large degrees of freedom,
    the log-PDF is computed using log-gamma functions:

    .. math::

        \log f(x \mid v) = \ln \Gamma\left(\frac{v+1}{2}\right) - \ln \Gamma\left(\frac{v}{2}\right)
        - \frac{1}{2}\ln(v\pi) - \frac{v+1}{2} \ln\left(1 + \frac{x^2}{v}\right)

    **Limiting Case Delegation:**

    - **Lorentzian/Cauchy Limit (:math:`v = 1`)**: When :math:`v \approx 1`, the distribution reduces
      analytically to the standard Lorentzian/Cauchy distribution and delegates to :class:`scipy.stats.cauchy`.
    - **Gaussian Limit (:math:`v \to \infty`)**: For very large degrees of freedom (:math:`v > 10^5`),
      the distribution converges to the Gaussian limit and delegates directly to :class:`scipy.stats.norm`.
    """

    def __init__(self, v: float = 1.0, loc: float = 0.0, scale: float = 1.0):
        if v <= 0:
            raise ValueError(f"Degrees of freedom v must be > 0, got {v}")
        if scale <= 0:
            raise ValueError(f"Scale must be > 0, got {scale}")

        self.v = v
        self.loc = loc
        self.scale = scale

    def pdf(self, x: ArrayLike) -> NDArray:
        if self.v == 1 or abs(self.v - 1.0) < 1e-8:
            return scipy_cauchy.pdf(x, loc=self.loc, scale=self.scale)
        elif self.v > 1e5:
            return scipy_norm.pdf(x, loc=self.loc, scale=self.scale)
        return scipy_t.pdf(x, df=self.v, loc=self.loc, scale=self.scale)

    def logpdf(self, x: ArrayLike) -> NDArray:
        if self.v == 1 or abs(self.v - 1.0) < 1e-8:
            return scipy_cauchy.logpdf(x, loc=self.loc, scale=self.scale)
        elif self.v > 1e5:
            return scipy_norm.logpdf(x, loc=self.loc, scale=self.scale)
        return scipy_t.logpdf(x, df=self.v, loc=self.loc, scale=self.scale)

    def cdf(self, x: ArrayLike) -> NDArray:
        if self.v == 1 or abs(self.v - 1.0) < 1e-8:
            return scipy_cauchy.cdf(x, loc=self.loc, scale=self.scale)
        elif self.v > 1e5:
            return scipy_norm.cdf(x, loc=self.loc, scale=self.scale)
        return scipy_t.cdf(x, df=self.v, loc=self.loc, scale=self.scale)

    def logcdf(self, x: ArrayLike) -> NDArray:
        if self.v == 1 or abs(self.v - 1.0) < 1e-8:
            return scipy_cauchy.logcdf(x, loc=self.loc, scale=self.scale)
        elif self.v > 1e5:
            return scipy_norm.logcdf(x, loc=self.loc, scale=self.scale)
        return scipy_t.logcdf(x, df=self.v, loc=self.loc, scale=self.scale)