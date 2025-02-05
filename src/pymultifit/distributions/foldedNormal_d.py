"""Created on Dec 04 03:42:42 2024"""

from math import erf, sqrt, exp, pi

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import folded_normal_cdf_, folded_normal_pdf_, folded_normal_log_pdf_, folded_normal_log_cdf_


class FoldedNormalDistribution(BaseDistribution):
    r"""
    Class for FoldedNormal distribution.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param mu: The mean parameter, :math:`\mu`. Defaults to 0.0.
    :type mu: float, optional

    :param sigma: The standard deviation parameter, :math:`\sigma`. Defaults to 1.0.
    :type sigma: float, optional

    :param loc: The location parameter, for shifting. Defaults to 0.0.
    :type loc: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeStandardDeviationError: If the provided value of standard deviation is negative.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/foldednorm.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Folded Normal(:math:`\mu=0, \sigma = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/foldednorm.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/foldednorm.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/folded_normal_example1.png
       :alt: Gaussian(0, 1)
       :align: center

    Generating a translated Gaussian(:math:`\mu=2, \sigma=3`) distribution with :math:`\text{loc}=3`:

    .. literalinclude:: ../../../examples/basic/foldednorm.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/foldednorm.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/folded_normal_example2.png
       :alt: Gaussian(3, 2)
       :align: center
    """

    def __init__(self, amplitude: float = 1.0, mu: float = 0.0, sigma: float = 1., loc: float = 0.0,
                 normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        self.amplitude = 1. if normalize else amplitude
        self.mu = mu
        self.sigma = sigma
        self.loc = loc

        self.norm = normalize

    @classmethod
    def scipy_like(cls, c, loc: float = 0.0, scale: float = 1.0):
        r"""
        Instantiate FoldedNormalDistribution with scipy parametrization.

        Parameters
        ----------
        c: float
            The shape parameter.
        loc: float, optional
            The location parameter. Defaults to 0.0.
        scale: float, optional
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        FoldedNormalDistribution
            An instance of normalized FoldedNormalDistribution.
        """
        return cls(mu=c, sigma=scale, loc=loc, normalize=True)

    def pdf(self, x):
        return folded_normal_pdf_(x,
                                  amplitude=self.amplitude, mean=self.mu, sigma=self.sigma, loc=self.loc,
                                  normalize=self.norm)

    def logpdf(self, x):
        return folded_normal_log_pdf_(x,
                                      amplitude=self.amplitude, mean=self.mu, sigma=self.sigma, loc=self.loc,
                                      normalize=self.norm)

    def cdf(self, x):
        return folded_normal_cdf_(x,
                                  amplitude=self.amplitude, mean=self.mu, sigma=self.sigma, loc=self.loc,
                                  normalize=self.norm)

    def logcdf(self, x):
        return folded_normal_log_cdf_(x,
                                      amplitude=self.amplitude, mean=self.mu, sigma=self.sigma, loc=self.loc,
                                      normalize=self.norm)

    def stats(self):
        mean_, std_ = self.mu, self.sigma

        sqrt_ = (2 / pi)**0.5

        f1 = sqrt_ * exp(-0.5 * mean_**2)
        f2 = mean_ * erf(mean_ / sqrt(2))

        mu_y = f1 + f2
        var_y = mean_**2 + 1 - mu_y**2

        return {'mean': (std_ * mu_y) + self.loc,
                'variance': var_y * std_**2,
                'std': sqrt(var_y * std_**2)}
