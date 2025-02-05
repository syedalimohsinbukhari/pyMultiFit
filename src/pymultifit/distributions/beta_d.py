"""Created on Aug 14 00:45:37 2024"""

from scipy.special import betaincinv

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import beta_cdf_, beta_pdf_, beta_log_pdf_, beta_log_cdf_


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

    def pdf(self, x):
        return beta_pdf_(x,
                         amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale,
                         normalize=self.norm)

    def logpdf(self, x):
        return beta_log_pdf_(x,
                             amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale,
                             normalize=self.norm)

    def cdf(self, x):
        return beta_cdf_(x,
                         amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale,
                         normalize=self.norm)

    def logcdf(self, x):
        return beta_log_cdf_(x,
                             amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale,
                             normalize=self.norm)

    def stats(self):
        a, b = self.alpha, self.beta
        s, _l = self.scale, self.loc

        mean_ = a / (a + b)
        mean_ = (s * mean_) + _l

        median_ = betaincinv(a, b, 0.5)
        median_ = (s * median_) + _l

        num_ = a * b
        den_ = (a + b)**2 * (a + b + 1)

        variance_ = s**2 * (num_ / den_)

        return {'mean': mean_,
                'median': median_,
                'mode': None,
                'variance': variance_,
                'std': variance_**0.5}
