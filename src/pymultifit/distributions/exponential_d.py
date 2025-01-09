"""Created on Nov 30 10:49:49 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import exponential_cdf_, exponential_pdf_


class ExponentialDistribution(BaseDistribution):
    r"""
    Class for Exponential distribution.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param scale: The scale parameter, :math:`\lambda`. Defaults to 1.0.
    :type scale: float, optional

    :param loc: The location parameter, for shifting. Defaults to 0.0.
    :type loc: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
                      Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeScaleError: If the provided value of scale is negative.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Exponential(:math:`\lambda =1.5`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/expon_example1.png
       :alt: Expon(1.5)
       :align: center

    Generating a translated Exponential(:math:`\lambda=1.5`) distribution with :math:`\text{loc} = 3`:

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/expon_example2.png
       :alt: Expon(1.5, 3)
       :align: center
    """

    def __init__(self, amplitude: float = 1.0, scale: float = 1.0, loc: float = 0.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif scale <= 0:
            raise erH.NegativeScaleError()
        self.amplitude = 1 if normalize else amplitude
        self.scale = scale
        self.loc = loc

        self.norm = normalize

    @classmethod
    def scipy_like(cls, loc: float = 0.0, scale: float = 1.0):
        r"""
        Instantiate ExponentialDistribution with scipy parameterization.

        Parameters
        ----------
        loc: float, optional
            The location parameter. Defaults to 0.0.
        scale: float, optional
            The rate parameter. Defaults to 1.0.

        Returns
        -------
        ExponentialDistribution
            A instance of normalized ExponentialDistribution.
        """
        return cls(loc=loc, scale=1 / scale, normalize=True)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return exponential_pdf_(x=x, amplitude=self.amplitude, lambda_=self.scale, loc=self.loc, normalize=self.norm)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return exponential_cdf_(x=x, amplitude=self.amplitude, scale=self.scale, loc=self.loc, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        s, l_ = self.scale, self.loc

        mean_ = (1 / s) + l_
        median_ = (np.log(2) / s) + l_
        mode_ = 0
        variance_ = 1 / s**2
        return {'mean': mean_,
                'median': median_,
                'mode': mode_,
                'variance': variance_,
                'std': np.sqrt(variance_)}
