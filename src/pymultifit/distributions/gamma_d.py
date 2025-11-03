"""Created on Aug 14 01:28:13 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import gamma_pdf_, gamma_log_pdf_, gamma_cdf_, gamma_log_cdf_
from .. import md_scipy_like, OneDArray


class GammaDistribution(BaseDistribution):
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

    .. literalinclude:: ../../../examples/basic/gamma_.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard GammaSS(:math:`\alpha =1.5, \lambda = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/gamma_.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gamma_.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/gammaSS_example1.png
       :alt: GammaSS(1.5, 1)
       :align: center

    Generating a translated Gamma(:math:`\alpha=1.5, \lambda=0.2`) distribution with :math:`\text{loc} = 3`:

    .. literalinclude:: ../../../examples/basic/gamma_.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gamma_.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/gammaSS_example2.png
       :alt: GammaSS(1.5, 0.2, 3)
       :align: center
    """

    def __init__(
        self, amplitude: float = 1.0, shape: float = 1.0, scale: float = 1.0, loc: float = 0.0, normalize: bool = False
    ):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        if shape <= 0:
            raise erH.NegativeShapeError()
        if scale <= 0:
            raise erH.NegativeScaleError()
        self.amplitude = 1.0 if normalize else amplitude
        self.shape = shape
        self.scale = scale
        self.loc = loc

        self.norm = normalize

    @classmethod
    @md_scipy_like("1.0.7")
    def scipy_like(cls, a: float, loc: float = 0.0, scale: float = 1.0) -> "GammaDistribution":
        r"""
        Instantiate GammaDistributionSS with scipy parametrization.

        Parameters
        ----------
        a: float
            The shape parameter.
        loc: float, optional
            The location parameter. Defaults to 0.0.
        scale: float, optional
            The scaling parameter. Defaults to 1.0.

        Returns
        -------
        GammaDistribution
            An instance of normalized GammaDistributionSS.
        """
        return cls(shape=a, loc=loc, scale=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, a: float, loc: float = 0.0, scale: float = 1.0) -> "GammaDistribution":
        r"""
        Instantiate GammaDistributionSS with scipy parametrization.

        Parameters
        ----------
        a: float
            The shape parameter.
        loc: float, optional
            The location parameter. Defaults to 0.0.
        scale: float, optional
            The scaling parameter. Defaults to 1.0.

        Returns
        -------
        GammaDistribution
            An instance of normalized GammaDistributionSS.
        """
        return cls(shape=a, loc=loc, scale=scale, normalize=True)

    def pdf(self, x: OneDArray) -> OneDArray:
        return gamma_pdf_(
            x, amplitude=self.amplitude, alpha=self.shape, theta=self.scale, loc=self.loc, normalize=self.norm
        )

    def logpdf(self, x: OneDArray) -> OneDArray:
        return gamma_log_pdf_(
            x, amplitude=self.amplitude, alpha=self.shape, theta=self.scale, loc=self.loc, normalize=self.norm
        )

    def cdf(self, x: OneDArray) -> OneDArray:
        return gamma_cdf_(
            x, amplitude=self.amplitude, alpha=self.shape, theta=self.scale, loc=self.loc, normalize=self.norm
        )

    def logcdf(self, x: OneDArray) -> OneDArray:
        return gamma_log_cdf_(
            x, amplitude=self.amplitude, alpha=self.shape, theta=self.scale, loc=self.loc, normalize=self.norm
        )

    def stats(self) -> Dict[str, float]:
        s, r, l_ = self.shape, self.scale, self.loc

        mean_ = (s * r) + l_
        variance_ = s * r**2
        mode_ = (s - 1) * r + l_ if s >= 1 else 0

        return {"mean": mean_, "mode": mode_, "variance": variance_, "std": np.sqrt(variance_)}
