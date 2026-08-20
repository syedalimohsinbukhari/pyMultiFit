"""Created on Nov 02 18:49:12 2025"""

from __future__ import annotations

from numpy import cosh, expm1, sinh

from .. import EXP, NAN_DICT, SQRT
from ..typing import ArrayLike, NDArray
from .backend import BaseDistribution
from .utilities_d import johnsonSU_cdf_, johnsonSU_log_cdf_, johnsonSU_log_pdf_, johnsonSU_pdf_


class JohnsonSUDistribution(BaseDistribution):
    r"""
    Class for :class:`~.JohnsonSUDistribution`.

    Parameters
    ----------
    amplitude :
        The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    gamma :
        The first shape parameter :math:`\gamma`. Controls skewness. Defaults to 0.0.
    delta :
        The second shape parameter :math:`\delta` (:math:`\delta > 0`). Controls tail weight. Defaults to 1.0.
    xi :
        The location parameter :math:`\xi` for shifting along the x-axis. Defaults to 0.0.
    lambda_ :
        The scale parameter :math:`\lambda` (:math:`\lambda > 0`) for stretching along the x-axis. Defaults to 1.0.
    normalize :
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
    Importing libraries

    .. literalinclude:: ../../../examples/basic/johnsonsu.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard normalized :class:`~.JohnsonSUDistribution` with ``pyMultiFit`` and ``scipy``.

    .. literalinclude:: ../../../examples/basic/johnsonsu.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/johnsonsu.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/johnsonsu_example1.png
       :alt: Johnson SU distribution (standard)
       :align: center

    Generating an unnormalized scaled :class:`~.JohnsonSUDistribution` with ``amplitude = 5.0``.

    .. literalinclude:: ../../../examples/basic/johnsonsu.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**

    .. literalinclude:: ../../../examples/basic/johnsonsu.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/johnsonsu_example2.png
       :alt: Johnson SU distribution (scaled amplitude)
       :align: center
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        gamma: float = 0.0,
        delta: float = 1.0,
        xi: float = 0.0,
        lambda_: float = 1.0,
        normalize: bool = False,
    ):
        self.amplitude = 1.0 if normalize else amplitude
        self.gamma = gamma
        self.delta = delta
        self.xi = xi
        self.lambda_ = lambda_

        self.norm = normalize

    @classmethod
    def from_scipy_params(cls, a: float, b: float, loc: float = 0.0, scale: float = 1.0) -> "JohnsonSUDistribution":
        return cls(gamma=a, delta=b, xi=loc, lambda_=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return johnsonSU_pdf_(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def logpdf(self, x: ArrayLike) -> NDArray:
        return johnsonSU_log_pdf_(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return johnsonSU_cdf_(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return johnsonSU_log_cdf_(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def stats(self) -> dict[str, float]:
        a, b = self.gamma, self.delta
        s, l_ = self.lambda_, self.xi

        if any(param <= 0 for param in (b, s)):
            return NAN_DICT

        mean_ = l_ - s * EXP(1 / (2 * b**2)) * sinh(a / b)

        median_ = l_ + s * sinh(-a / b)

        v1 = EXP(b**-2) * cosh(2 * a / b) + 1
        v2 = expm1(b**-2)
        variance_ = s**2 / 2 * v1 * v2

        return {"mean": mean_, "median": median_, "variance": variance_, "std": SQRT(variance_)}
