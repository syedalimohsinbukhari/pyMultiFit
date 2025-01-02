"""Created on Aug 14 01:28:13 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import gamma_sr_cdf_, gamma_sr_pdf_


class GammaDistributionSR(BaseDistribution):
    r"""
    Class for Gamma distribution with shape and rate parameters.

    :param amplitude: The amplitude of the PDF. Default is 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param shape: The shape parameter, :math:`\alpha`. Defaults to 1.0.
    :type shape: float, optional

    :param rate: The rate parameter, :math:`\lambda`. Defaults to 1.0.
    :type rate: float, optional

    :param loc: The location parameter, for shifting. Defaults to 0.0.
    :type loc: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1. Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeShapeError: If the provided value of shape is negative.
    :raise NegativeRateError: If the provided value of rate is negative.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/gammaSR_.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard GammaSR(:math:`\alpha =1.5, \lambda = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/gammaSR_.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gammaSR_.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/gammaSR_example1.png
       :alt: GammaSR(1.5, 1)
       :align: center

    Generating a translated Gamma(:math:`\alpha=1.5, \lambda=0.2`) distribution with :math:`\text{loc} = 3`:

    .. literalinclude:: ../../../examples/basic/gammaSR_.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gammaSR_.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/gammaSR_example2.png
       :alt: GammaSR(1.5, 0.2, 3)
       :align: center
    """

    def __init__(self,
                 amplitude: float = 1.0, shape: float = 1.0, rate: float = 1.0,
                 loc: float = 0.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif shape <= 0:
            raise erH.NegativeShapeError()
        elif rate <= 0:
            raise erH.NegativeRateError()
        self.amplitude = 1. if normalize else amplitude
        self.shape = shape
        self.rate = rate
        self.loc = loc

        self.norm = normalize

    def pdf(self, x: np.array) -> np.array:
        return gamma_sr_pdf_(x,
                             amplitude=self.amplitude, alpha=self.shape, lambda_=self.rate, loc=self.loc,
                             normalize=self.norm)

    def cdf(self, x: np.array) -> np.array:
        return gamma_sr_cdf_(x,
                             amplitude=self.amplitude, alpha=self.shape, lambda_=self.rate, loc=self.loc,
                             normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        a, b, loc = self.shape, self.rate, self.loc

        mean_ = (a / b) + loc
        mode_ = (a - 1) / b + loc if a >= 1 else 0
        variance_ = a / b**2

        return {'mean': mean_,
                'mode': mode_,
                'variance': variance_}


class GammaDistributionSS(GammaDistributionSR):
    r"""
    Class for Gamma distribution with shape and scale parameters.

    :param amplitude: The amplitude of the PDF. Default is 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param shape: The shape parameter, :math:`\alpha`. Defaults to 1.0.
    :type shape: float, optional

    :param scale: The rate parameter, :math:`\theta`. Defaults to 1.0.
    :type scale: float, optional

    :param loc: The location parameter, for shifting. Defaults to 0.0.
    :type loc: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1. Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeShapeError: If the provided value of shape is negative.
    :raise NegativeScaleError: If the provided value of scale is negative.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/gammaSS_.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard GammaSS(:math:`\alpha =1.5, \theta = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/gammaSS_.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gammaSS_.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/gammaSS_example1.png
       :alt: GammaSS(1.5, 1)
       :align: center

    Generating a translated Gamma(:math:`\alpha=1.5, \theta=0.2`) distribution with :math:`\text{loc} = 3`:

    .. literalinclude:: ../../../examples/basic/gammaSS_.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gammaSS_.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/gammaSS_example2.png
       :alt: GammaSS(1.5, 0.2, 3)
       :align: center
    """

    def __init__(self, amplitude: float = 1.0, shape: float = 1.0, scale: float = 1.0, loc: float = 0.0,
                 normalize: bool = False):
        self.scale = scale
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif shape <= 0:
            raise erH.NegativeShapeError()
        elif scale <= 0:
            raise erH.NegativeScaleError()
        super().__init__(amplitude=amplitude, shape=shape, rate=1 / self.scale, loc=loc, normalize=normalize)
