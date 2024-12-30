"""Created on Dec 03 17:37:05 2024"""

from typing import Dict

from .backend import errorHandling as erH
from .gamma_d import GammaDistributionSR


class ChiSquareDistribution(GammaDistributionSR):
    r"""Class for ChiSquare distribution.

    Parameters
    ----------
    amplitude : float, optional
        The amplitude of the PDF. Defaults to 1.0.
        Ignored if **normalize** is ``True``.
    degree_of_freedom : int, optional
        The degree of freedom for the chi-square distribution.
        Default is 1.0.
    loc : float, optional
        The location parameter, for shifting.
        Defaults to 0.0.
    normalize : bool, optional
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
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

    Raises
    ------
    :class:`~pymultifit.distributions.backend.errorHandling.NegativeAmplitudeError`
        If the provided value of amplitude is negative.
    :class:`~pymultifit.distributions.backend.errorHandling.DegreeOfFreedomError`
        If the provided value of degree of freedom is either less than or equal to 0 or not an integer.
    """

    def __init__(self,
                 amplitude: float = 1.0, degree_of_freedom: int = 1, loc: float = 0.0,
                 normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif not isinstance(degree_of_freedom, int) or degree_of_freedom <= 0:
            raise erH.DegreeOfFreedomError()
        self.dof = degree_of_freedom
        self.loc = loc
        super().__init__(amplitude=amplitude, shape=degree_of_freedom / 2., rate=0.5, loc=loc, normalize=normalize)

    def stats(self) -> Dict[str, float]:
        stat_ = super().stats()
        f1 = 9 * self.dof
        f1 = 1 - (2 / f1)
        f1 = self.dof * f1**3

        stat_['median'] = f1

        return stat_
