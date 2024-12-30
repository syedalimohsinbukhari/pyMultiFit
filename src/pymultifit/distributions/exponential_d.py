"""Created on Nov 30 10:49:49 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import exponential_cdf_, exponential_pdf_


class ExponentialDistribution(BaseDistribution):
    r"""
    Class for Exponential distribution.

    Parameters
    ----------
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    scale: float, optional
        The scale parameter, :math:`\lambda`.
        Defaults to 1.0.
    loc : float, optional
        The location parameter, for shifting.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
    Importing libraries

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Exponential(:math:`\lambda =1.5`) distribution with ``pyMultiFit`` and ``scipy``.

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/expon_example1.png
       :alt: Expon(1.5)
       :align: center

    Generating a translated Exponential(:math:`\lambda=1.5`) distribution with :math:`\text{loc} = 3`.

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/expon_example2.png
       :alt: Expon(1.5, 3)
       :align: center

    Raises
    ------
    :class:`~pymultifit.distributions.backend.errorHandling.NegativeAmplitudeError`
        If the provided value of amplitude is negative.
    :class:`~pymultifit.distributions.backend.errorHandling.NegativeScaleError`
        If the provided value of scale is negative.
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

    def pdf(self, x: np.array) -> np.array:
        return exponential_pdf_(x=x, amplitude=self.amplitude, lambda_=self.scale, loc=self.loc, normalize=self.norm)

    def cdf(self, x: np.array) -> np.array:
        return exponential_cdf_(x=x, amplitude=self.amplitude, scale=self.scale, loc=self.loc, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        lambda_ = self.scale

        return {'mean': 1 / lambda_ + self.loc,
                'median': np.log(2) / lambda_ + self.loc,
                'mode': 0 + self.loc,
                'variance': 1 / lambda_**2}
