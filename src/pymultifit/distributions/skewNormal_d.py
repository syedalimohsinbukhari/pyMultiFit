"""Created on Aug 03 21:35:28 2024"""

from math import sqrt, pi, exp

import numpy as np

from .backend import BaseDistribution
from .backend.errorHandling import NegativeAmplitudeError, NegativeScaleError
from .utilities_d import skew_normal_cdf_, skew_normal_pdf_


class SkewNormalDistribution(BaseDistribution):
    r"""
    Class for SkewNormal distribution.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param shape: The mean parameter, :math:`\mu`. Defaults to 0.0.
    :type shape: float, optional

    :param scale: The scale parameter, for scaling. Defaults to 1.0,
    :type scale: float, optional

    :param location: The location parameter, for shifting. Defaults to 0.0.
    :type location: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1. Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeStandardDeviationError: If the provided value of standard deviation is negative.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/skewnormal.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Skew Normal(:math:`\xi=1, \mu = 0, \sigma = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/skewnormal.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/skewnormal.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/skew_norm_example1.png
       :alt: SkewNormal(1, 0, 1)
       :align: center

    Generating a translated Skew Normal(:math:`\xi=3, \mu=-3, \sigma=3`) distribution:

    .. literalinclude:: ../../../examples/basic/skewnormal.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/skewnormal.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/skew_norm_example2.png
       :alt: Skew Normal(3, -3, 3)
       :align: center
    """

    def __init__(self, amplitude: float = 1.0, shape: float = 1., location: float = 0., scale: float = 1.,
                 normalize: bool = False):
        if not normalize and amplitude < 0.:
            raise NegativeAmplitudeError()
        if scale <= 0.:
            raise NegativeScaleError()

        self.amplitude = 1 if normalize else amplitude
        self.shape = shape
        self.location = location
        self.scale = scale

        self.norm = normalize

    @classmethod
    def scipy_like(cls, a: float, loc: float = 0.0, scale: float = 1.0):
        """
        Instantiate SkewNormalDistribution with scipy parametrization.

        Parameters
        ----------
        a : float
            The skewness parameter.
        loc : float, optional
            The location parameter. Defaults to 0.0.
        scale : float, optional
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        SkewNormalDistribution
            An instance of normalized SkewNormalDistribution.
        """
        return cls(shape=a, location=loc, scale=scale, normalize=True)

    def pdf(self, x):
        return skew_normal_pdf_(x,
                                amplitude=self.amplitude, shape=self.shape, loc=self.location, scale=self.scale,
                                normalize=self.norm)

    def cdf(self, x):
        return skew_normal_cdf_(x,
                                amplitude=self.amplitude, shape=self.shape, loc=self.location, scale=self.scale,
                                normalize=self.norm)

    def stats(self):
        alpha, omega, epsilon = self.shape, self.scale, self.location
        delta = alpha / sqrt(1 + alpha**2)
        delta_sqrt_2_pi = sqrt(2 / pi) * delta

        def _m0(alpha_):
            term2 = (1 - pi / 4) * delta_sqrt_2_pi**3 / (1 - (2 / pi) * delta**2)
            term3 = (2 * pi / abs(alpha_)) * exp(-(2 * pi / abs(alpha_))) * np.sign(alpha_)
            return delta_sqrt_2_pi - term2 - term3

        # Calculating mean, mode, variance, and std
        mean_ = epsilon + omega * delta_sqrt_2_pi
        mode_ = epsilon + omega * _m0(alpha)
        variance_ = omega**2 * (1 - (2 * delta**2 / pi))
        std_ = sqrt(variance_)

        return {'mean': mean_,
                'mode': mode_,
                'median': None,
                'variance': variance_,
                'std': std_}
