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
    :type degree_of_freedom: int, optional

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
                 amplitude: float = 1.0, degree_of_freedom: int = 1,
                 loc: float = 0.0, scale: float = 1.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        self.amplitude = 1 if normalize else amplitude
        self.dof = degree_of_freedom
        self.loc = loc
        self.scale = scale

        self.norm = normalize

    def pdf(self, x: np.array) -> np.array:
        return chi_square_pdf_(x, amplitude=self.amplitude, degree_of_freedom=self.dof, loc=self.loc, scale=self.scale, normalize=self.norm)

    def cdf(self, x: np.array) -> np.array:
        return chi_square_cdf_(x, amplitude=self.amplitude, degree_of_freedom=self.dof, loc=self.loc, scale=self.scale, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        stat_ = super().stats()
        f1 = 9 * self.dof
        f1 = 1 - (2 / f1)
        f1 = self.dof * f1**3

        stat_['median'] = f1

        return stat_
