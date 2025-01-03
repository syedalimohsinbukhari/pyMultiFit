"""Created on Aug 14 00:45:37 2024"""

from typing import Dict

import numpy as np
from scipy.special import betaincinv

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import beta_cdf_, beta_pdf_


class BetaDistribution(BaseDistribution):
    r"""
    Class for Beta distribution.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if ``normalize`` is ``True``.
    :type amplitude: float, optional

    :param alpha: The :math:`\alpha` parameter. Defaults to 1.0.
    :type alpha: float, optional

    :param beta: The :math:`\beta` parameter. Defaults to 1.0.
    :type beta: float, optional

    :param loc: float, optional The location parameter, for shifting. Defaults to 0.0.
    :type loc: float, optional

    :param scale: float, optional The scale parameter, for scaling. Defaults to 1.0.
    :type scale: float, optional

    :param normalize: bool, optional If ``True``, the distribution is normalized so that the total area under the PDF equals 1. Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeAlphaError: If the provided value of :math:`\alpha` is negative.
    :raise NegativeBetaError: If the provided value of :math:`\beta` is negative.

    Examples
    --------
    Importing libraries

    .. literalinclude:: ../../../examples/basic/beta1.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard :math:`\text{Beta}(2, 30)` distribution with ``pyMultiFit`` and ``scipy``.

    .. literalinclude:: ../../../examples/basic/beta1.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/beta1.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/beta_example1.png
       :alt: Beta distribution (5, 30)
       :align: center

    Generating a shifted and translated :math:`\text{Beta}(2, 30)` distribution.

    .. literalinclude:: ../../../examples/basic/beta2.py
       :language: python
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/beta2.py
       :language: python
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/beta_example2.png
       :alt: Beta distribution (shifted and translated)
       :align: center
    """

    def __init__(self,
                 amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
                 loc: float = 0.0, scale: float = 1.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif alpha <= 0:
            raise erH.NegativeAlphaError()
        elif beta <= 0:
            raise erH.NegativeBetaError()
        self.amplitude = 1. if normalize else amplitude
        self.alpha = alpha
        self.beta = beta
        self.loc = loc
        self.scale = scale

        self.norm = normalize

    @classmethod
    def scipy_like(cls, a: float, b: float, loc: float = 0.0, scale: float = 1.0):
        r"""
        Instantiate BetaDistribution with scipy parameterization.

        Parameters
        ----------
        a: float
            The shape parameter, :math:`\alpha`.
        b: float
            The shape parameter, :math:`\beta`.
        loc: float, optional
            The location parameter. Defaults to 0.0.
        scale: float, optional
            The scale parameter,. Defaults to 1.0.

        Returns
        -------
        BetaDistribution
            An instance of normalized BetaDistribution.
        """
        return cls(alpha=a, beta=b, loc=loc, scale=scale, normalize=True)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if self.scale > 0:
            return beta_pdf_(x=x, amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, loc=self.loc,
                             scale=self.scale, normalize=self.norm)
        else:
            return np.full(shape=x.shape, fill_value=np.nan)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if self.scale > 0:
            return beta_cdf_(x=x, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale)
        else:
            return np.full(shape=x.shape, fill_value=np.nan)

    # def logpdf(self, x: np.array) -> np.array:
    #     return beta_logpdf_(x=x, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale, normalize=self.norm)

    @property
    def mean(self) -> float:
        mean_ = self.alpha / (self.alpha + self.beta)
        return (self.scale * mean_) + self.loc

    @property
    def median(self) -> float:
        median_ = betaincinv(self.alpha, self.beta, 0.5)
        return (self.scale * median_) + self.loc

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        num_ = a * b
        den_ = (a + b)**2 * (a + b + 1)

        return self.scale**2 * (num_ / den_)

    @property
    def stddev(self) -> float:
        return np.sqrt(self.variance)

    def stats(self) -> Dict[str, float]:
        return {'mean': self.mean,
                'median': self.median,
                'variance': self.variance,
                'std': self.stddev}
