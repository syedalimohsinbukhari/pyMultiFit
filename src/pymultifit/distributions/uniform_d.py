"""Created on Dec 11 20:40:15 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import uniform_cdf_, uniform_pdf_


class UniformDistribution(BaseDistribution):
    r"""
    Class for Uniform Distribution.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param low: Lower bound of distribution.
    :type low: float, optional

    :param high: Upper bound of distribution.
    :type high: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1. Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/uniform.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Uniform(0, 1) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/uniform.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/uniform.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/uniform_example1.png
       :alt: Uniform(0, 1)
       :align: center

    Generating a translated Uniform(3, 5) distribution:

    .. literalinclude:: ../../../examples/basic/uniform.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/uniform.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/uniform_example2.png
       :alt: Uniform(3, 5)
       :align: center
    """

    def __init__(self, amplitude: float = 1.0, low: float = 0.0, high: float = 1.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        self.amplitude = 1 if normalize else amplitude
        self.low = low
        self.high = high

        self.norm = normalize

    @classmethod
    def scipy_like(cls, loc: float = 0.0, scale: float = 1.0):
        """
        Instantiate UniformDistribution with scipy parametrization.

        Parameters
        ----------
        loc: float, optional
            The location parameter. Defaults to 0.0.
        scale: float, optional
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        UniformDistribution
            An instance of normalized UniformDistribution.
        """
        return cls(low=loc, high=scale, normalize=True)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return uniform_pdf_(x=x, amplitude=self.amplitude, low=self.low, high=self.high, normalize=self.norm)

    def cdf(self, x: np.array) -> np.array:
        return uniform_cdf_(x=x, amplitude=self.amplitude, low=self.low, high=self.high, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        low, high = self.low, self.low + self.high

        if low == high:
            return {'mean': np.nan,
                    'median': np.nan,
                    'variance': np.nan,
                    'std': np.nan}

        mean_ = 0.5 * (low + high)
        median_ = mean_
        variance_ = (1 / 12.) * (high - low)**2

        return {'mean': mean_,
                'median': median_,
                'variance': variance_,
                'std': np.sqrt(variance_)}
