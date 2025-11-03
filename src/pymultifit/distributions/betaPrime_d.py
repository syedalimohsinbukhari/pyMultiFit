"""Created on Oct 31 18:28:47 2025"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution
from .utilities_d import beta_prime_pdf_, beta_prime_log_pdf_, beta_prime_cdf_, beta_prime_log_cdf_
from .. import OneDArray


class BetaPrimeDistribution(BaseDistribution):
    r"""
    Class for Beta Prime distribution.

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

    Examples
    --------
    Importing libraries

    .. literalinclude:: ../../../examples/basic/betaPrime.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard :math:`\text{BetaPrime}(2, 30)` distribution with ``pyMultiFit`` and ``scipy``.

    .. literalinclude:: ../../../examples/basic/betaPrime.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/betaPrime.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/beta_prime1.png
       :alt: BetaPrime distribution (5, 30)
       :align: center

    Generating a shifted and translated :math:`\text{BetaPrime}(2, 30, 5, 3)` distribution.

    .. literalinclude:: ../../../examples/basic/betaPrime.py
       :language: python
       :linenos:
       :lineno-start: 33
       :lines: 33-34

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/betaPrime.py
       :language: python
       :linenos:
       :lineno-start: 36
       :lines: 36-51

    .. image:: ../../../images/beta_prime2.png
       :alt: Beta Prime distribution (shifted and translated)
       :align: center
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        loc: float = 0.0,
        scale: float = 1.0,
        normalize: bool = False,
    ):
        self.amplitude = 1.0 if normalize else amplitude
        self.alpha = alpha
        self.beta = beta
        self.loc = loc
        self.scale = scale

        self.norm = normalize

    @classmethod
    def from_scipy_params(cls, a: float, b: float, loc: float = 0.0, scale: float = 1.0) -> "BetaPrimeDistribution":
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

    def pdf(self, x: OneDArray) -> OneDArray:
        return beta_prime_pdf_(
            x,
            amplitude=self.amplitude,
            alpha=self.alpha,
            beta_=self.beta,
            loc=self.loc,
            scale=self.scale,
            normalize=self.norm,
        )

    def logpdf(self, x: OneDArray) -> OneDArray:
        return beta_prime_log_pdf_(
            x,
            amplitude=self.amplitude,
            alpha=self.alpha,
            beta_=self.beta,
            loc=self.loc,
            scale=self.scale,
            normalize=self.norm,
        )

    def cdf(self, x: OneDArray) -> OneDArray:
        return beta_prime_cdf_(
            x,
            amplitude=self.amplitude,
            alpha=self.alpha,
            beta_=self.beta,
            loc=self.loc,
            scale=self.scale,
            normalize=self.norm,
        )

    def logcdf(self, x: OneDArray) -> OneDArray:
        return beta_prime_log_cdf_(
            x,
            amplitude=self.amplitude,
            alpha=self.alpha,
            beta_=self.beta,
            loc=self.loc,
            scale=self.scale,
            normalize=self.norm,
        )

    def stats(self) -> Dict[str, float]:
        a, b = self.alpha, self.beta
        s, _l = self.scale, self.loc

        mean_ = a / (b - 1) if b > 1 else np.inf
        mean_ = (s * mean_) + _l

        num_ = a * (a + b - 1)
        den_ = (b - 2) * (b - 1) ** 2
        variance_ = num_ / den_ if b > 2 else np.inf
        variance_ = variance_ * s**2

        return {"mean": mean_, "variance": variance_, "std": np.sqrt(variance_)}
