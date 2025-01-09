"""Created on Aug 03 21:02:45 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import log_normal_cdf_, log_normal_pdf_


class LogNormalDistribution(BaseDistribution):
    r"""
    Class for LogNormal distribution.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param mu: The mean parameter, :math:`\mu`. Defaults to 0.0.
    :type mu: float, optional

    :param std: The standard deviation parameter, :math:`\sigma`. Defaults to 1.0.
    :type std: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1. Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeStandardDeviationError: If the provided value of standard deviation is negative.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Gaussian(:math:`\mu=0, \sigma = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/gaussian_example1.png
       :alt: Gaussian(0, 1)
       :align: center

    Generating a translated Gaussian(:math:`\mu=3, \sigma=2`) distribution:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/gaussian_example2.png
       :alt: Gaussian(3, 2)
       :align: center
    """

    def __init__(self, amplitude: float = 1., mu: float = 0.0, std: float = 1.0, loc: float = 0.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif std <= 0:
            raise erH.NegativeStandardDeviationError()
        self.amplitude = 1. if normalize else amplitude
        self.mu = np.log(mu)
        self.std = std
        self.loc = loc

        self.norm = normalize

    @classmethod
    def scipy_like(cls, s, loc: float = 0.0, scale: float = 1.0):
        """
        Instantiate LogNormalDistribution with scipy parametrization.

        Parameters
        ----------
        s: float
            The shape parameter.
        loc: float, optional
            The location parameter. Defaults to 0.0.
        scale: float, optional
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        LogNormalDistribution
            An instance of normalized LogNormalDistribution.
        """
        return cls(std=s, mu=scale, loc=loc, normalize=True)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return log_normal_pdf_(x, amplitude=self.amplitude, mean=self.mu, std=self.std, loc=self.loc, normalize=self.norm)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return log_normal_cdf_(x, amplitude=self.amplitude, mean=self.mu, std=self.std, loc=self.loc, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        m, s, l_ = self.mu, self.std, self.loc

        mean_ = np.exp(m + (s**2 / 2)) + l_
        median_ = np.exp(m) + l_
        mode_ = np.exp(m - s**2) + l_
        variance_ = (np.exp(s**2) - 1) * np.exp(2 * m + s**2)

        return {'mean': mean_,
                'median': median_,
                'mode': mode_,
                'variance': variance_,
                'std': np.sqrt(variance_)}
