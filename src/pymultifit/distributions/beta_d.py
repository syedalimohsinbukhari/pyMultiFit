"""Created on Aug 14 00:45:37 2024"""

from __future__ import annotations

from numpy import sqrt
from scipy.special import betaincinv

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import beta_cdf_, beta_log_cdf_, beta_log_pdf_, beta_pdf_
from .. import md_scipy_like
from ..typing import ArrayLike, NDArray


class BetaDistribution(BaseDistribution):
    r"""
    Class for Beta distribution.
    
    Parameters
    ----------
    amplitude
        The amplitude of the PDF. Defaults to 1.0. Ignored if ``normalize`` is ``True``.
    alpha
        The :math:`\alpha` parameter. Defaults to 1.0.
    beta
        The :math:`\beta` parameter. Defaults to 1.0.
    loc
        The location parameter, for shifting. Defaults to 0.0.
    scale
        The scale parameter, for scaling. Defaults to 1.0.
    normalize
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.
        
    Raises
    ------
    NegativeAmplitudeError
        If the provided value of amplitude is negative.

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
       :lines: 13-28

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

    def __init__(
        self,
        amplitude: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        loc: float = 0.0,
        scale: float = 1.0,
        normalize: bool = False,
    ):
        if amplitude < 0:
            raise erH.NegativeAmplitudeError()

        self.amplitude = 1.0 if normalize else amplitude
        self.alpha = alpha
        self.beta = beta
        self.loc = loc
        self.scale = scale

        self.norm = normalize

    @classmethod
    @md_scipy_like("v1.0.7")
    def scipy_like(cls, a: float, b: float, loc: float = 0.0, scale: float = 1.0) -> "BetaDistribution":
        r"""
        Instantiate `BetaDistribution` with scipy parameterization.
        
        Parameters
        ----------
        a
            The shape parameter, :math:`\alpha`.
        b
            The shape parameter, :math:`\beta`.
        loc
            The location parameter. Defaults to 0.0.
        scale
            The scale parameter,. Defaults to 1.0.
            
        Returns
        -------
        BetaDistribution
            An instance of normalized BetaDistribution.
        """
        return cls(alpha=a, beta=b, loc=loc, scale=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, a: float, b: float, loc: float = 0.0, scale: float = 1.0) -> "BetaDistribution":
        r"""
        Instantiate `BetaDistribution` with scipy parameterization.

        Parameters
        ----------
        a
            The shape parameter, :math:`\alpha`.
        b
            The shape parameter, :math:`\beta`.
        loc
            The location parameter. Defaults to 0.0.
        scale
            The scale parameter,. Defaults to 1.0.

        Returns
        -------
        BetaDistribution
            An instance of normalized BetaDistribution.
        """
        return cls(alpha=a, beta=b, loc=loc, scale=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return beta_pdf_(
            x,
            amplitude=self.amplitude,
            alpha=self.alpha,
            beta_=self.beta,
            loc=self.loc,
            scale=self.scale,
            normalize=self.norm,
        )

    def logpdf(self, x: ArrayLike) -> NDArray:
        return beta_log_pdf_(
            x,
            amplitude=self.amplitude,
            alpha=self.alpha,
            beta_=self.beta,
            loc=self.loc,
            scale=self.scale,
            normalize=self.norm,
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return beta_cdf_(
            x,
            amplitude=self.amplitude,
            alpha=self.alpha,
            beta_=self.beta,
            loc=self.loc,
            scale=self.scale,
            normalize=self.norm,
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return beta_log_cdf_(
            x,
            amplitude=self.amplitude,
            alpha=self.alpha,
            beta_=self.beta,
            loc=self.loc,
            scale=self.scale,
            normalize=self.norm,
        )

    def stats(self) -> dict[str, float]:
        a, b = self.alpha, self.beta
        s, _l = self.scale, self.loc

        mean_ = a / (a + b)
        mean_ = (s * mean_) + _l

        median_ = betaincinv(a, b, 0.5)
        median_ = (s * median_) + _l

        num_ = a * b
        den_ = (a + b) ** 2 * (a + b + 1)

        variance_ = s ** 2 * (num_ / den_)

        return {"mean": mean_, "median": median_.astype(float), "variance": variance_, "std": sqrt(variance_)}
