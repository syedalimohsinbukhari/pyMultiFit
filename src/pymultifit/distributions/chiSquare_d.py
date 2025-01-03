"""Created on Dec 03 17:37:05 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import chi_square_cdf_, chi_square_pdf_


class ChiSquareDistribution(BaseDistribution):
    r"""Class for :class:`ChiSquareDistribution` distribution.

    .. note::
        The :class:`ChiSquareDistribution` is a special case of the :class:`~pymultifit.distributions.gamma_d.GammaDistributionSR`,

        * :math:`\alpha_\text{gammaSR} = \text{dof} / 2`,
        * :math:`\lambda_\text{gammaSR} = 0.5`.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param degree_of_freedom: The degree of freedom for the chi-square distribution. Default is 1.0.
    :type degree_of_freedom: int or float, optional

    :param loc: The location parameter, for shifting. Defaults to 0.0.
    :type loc: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1. Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise DegreeOfFreedomError: If the provided value of degree of freedom is either less than or equal to 0 or not an integer.

    :examples:

    Importing libraries

    .. literalinclude:: ../../../examples/basic/chisquare.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard :math:`\chi^2(1)` distribution with ``pyMultiFit`` and ``scipy``.

    .. literalinclude:: ../../../examples/basic/chisquare.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/chisquare.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/chi2_example1.png
       :alt: Beta distribution (5, 30)
       :align: center

    Generating a translated :math:`\chi^2(1)` distribution with :math:`\text{loc} = 3`.

    .. literalinclude:: ../../../examples/basic/chisquare.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/chisquare.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/chi2_example2.png
       :alt: Beta distribution (shifted and translated)
       :align: center
    """

    def __init__(self,
                 amplitude: float = 1.0, degree_of_freedom: int | float = 1,
                 loc: float = 0.0, scale: float = 1.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        self.amplitude = 1 if normalize else amplitude
        self.dof = degree_of_freedom
        self.loc = loc
        self.scale = scale

        self.norm = normalize

    @classmethod
    def scipy_like(cls, df: int | float, loc: float = 0.0, scale: float = 1.0):
        """
        Instantiate ChiSquareDistribution with scipy parameterization.

        Parameters
        ----------
        df: int or float
            The degree of freedom for the ChiSquare distribution.
        loc: float, optional
            The location parameter. Defaults to 0.0.
        scale: float, optional
            The scale parameter. Defaults to 1.0

        Returns
        -------
        ChiSquareDistribution
            An instance of normalized ChiSquareDistribution.
        """
        return cls(degree_of_freedom=df, loc=loc, scale=scale, normalize=True)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return chi_square_pdf_(x, amplitude=self.amplitude, degree_of_freedom=self.dof, loc=self.loc, scale=self.scale,
                               normalize=self.norm)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return chi_square_cdf_(x, amplitude=self.amplitude, degree_of_freedom=self.dof, loc=self.loc, scale=self.scale,
                               normalize=self.norm)

    @property
    def mean(self) -> float:
        return (self.scale * self.dof) + self.loc

    @property
    def variance(self) -> float:
        return 2 * self.dof * self.scale**2

    @property
    def stddev(self) -> float:
        return np.sqrt(self.variance)

    @property
    def mode(self):
        return max(self.dof - 2, 0)

    def stats(self) -> Dict[str, float]:
        return {'mean': self.mean,
                'median': self.median,
                'mode': self.mode,
                'variance': self.variance,
                'std': self.stddev}
