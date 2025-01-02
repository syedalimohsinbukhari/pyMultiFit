"""Created on Aug 03 20:07:50 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import gaussian_cdf_, gaussian_pdf_


class GaussianDistribution(BaseDistribution):
    r"""
    Class for Gaussian distribution.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param mean: The mean parameter, :math:`\mu`. Defaults to 0.0.
    :type mean: float, optional

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

    def __init__(self, amplitude: float = 1.0, mean: float = 0., std: float = 1., normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif std <= 0:
            raise erH.NegativeStandardDeviationError()

        self.amplitude = 1. if normalize else amplitude
        self.mean = mean
        self.std_ = std
        self.norm = normalize

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return gaussian_pdf_(x, amplitude=self.amplitude, mean=self.mean, std=self.std_, normalize=self.norm)

    def cdf(self, x: np.array) -> np.array:
        return gaussian_cdf_(x, amplitude=self.amplitude, mean=self.mean, std=self.std_, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        mean_, std_ = self.mean, self.std_
        return {'mean': mean_,
                'median': mean_,
                'mode': mean_,
                'variance': std_**2}
