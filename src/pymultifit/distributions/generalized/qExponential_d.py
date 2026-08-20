"""Created on August 12 11:26:00 2026"""

import numpy as np

from ... import INF, NAN, NAN_DICT, SQRT
from ...typing import ArrayLike, NDArray
from ..backend import BaseDistribution
from ..utilities_d import q_exponential_cdf_, q_exponential_log_cdf_, q_exponential_log_pdf_, q_exponential_pdf_


class QExponentialDistribution(BaseDistribution):
    r"""
    Class for :class:`~.QExponentialDistribution`.

    .. note::
        The :class:`~QExponentialDistribution` reduces to the standard :class:`~pymultifit.distributions.exponential_d.ExponentialDistribution`
        when :math:`q = 1`.

    Parameters
    ----------
    amplitude :
        The amplitude or scaling factor of the distribution. Defaults to 1.0.
    q :
        The entropic index parameter, :math:`q`.
        Must satisfy :math:`q < 2`.
    rate :
        The rate parameter, :math:`\lambda`.
        Defaults to 1.0.
        Must be strictly positive (:math:`\lambda > 0`).
    loc :
        The location parameter, :math:`\mu`.
        Defaults to 0.0.
    normalize :
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/qexponential.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-8

    Generating a standard qExponential(:math:`q=1, \lambda=1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/qexponential.py
       :language: python
       :linenos:
       :lineno-start: 10
       :lines: 10-13

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/qexponential.py
       :language: python
       :linenos:
       :lineno-start: 15
       :lines: 15-30

    .. image:: ../../../images/q_exponential_example1.png
       :alt: QExponential(1, 1, 0)
       :align: center

    Generating a translated qExponential(:math:`q=1.5, \lambda=1.3, \text{loc}=-3.3`) distribution:

    .. literalinclude:: ../../../examples/basic/qexponential.py
       :language: python
       :lineno-start: 33
       :lines: 33

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/qexponential.py
       :language: python
       :lineno-start: 35
       :lines: 35-52

    .. image:: ../../../images/q_exponential_example2.png
       :alt: QExponential(1.5, 1.3, -3.3)
       :align: center
    """

    def __init__(
        self, amplitude: float = 1.0, q: float = 1.0, rate: float = 1.0, loc: float = 0.0, normalize: bool = False
    ):
        self.amplitude = amplitude
        self.q = q
        self.rate = rate
        self.loc = loc
        self.normalize = normalize

    def logpdf(self, x: ArrayLike) -> NDArray:
        return q_exponential_log_pdf_(
            x, amplitude=self.amplitude, q=self.q, rate=self.rate, loc=self.loc, normalize=self.normalize
        )

    def pdf(self, x: ArrayLike) -> NDArray:
        return q_exponential_pdf_(
            x, amplitude=self.amplitude, q=self.q, rate=self.rate, loc=self.loc, normalize=self.normalize
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return q_exponential_cdf_(
            x, amplitude=self.amplitude, q=self.q, rate=self.rate, loc=self.loc, normalize=self.normalize
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return q_exponential_log_cdf_(
            x, amplitude=self.amplitude, q=self.q, rate=self.rate, loc=self.loc, normalize=self.normalize
        )

    def stats(self) -> dict[str, float]:
        q, rate, loc = self.q, self.rate, self.loc
        mode_ = loc

        # Mean exists for q < 1.5 (3 - 2q > 0)
        if q < 1.5:
            mean_ = loc + 1.0 / (rate * (3.0 - 2.0 * q))
        else:
            mean_ = INF

        # Variance exists for q < 4/3 (4 - 3q > 0)
        if q < (4.0 / 3.0):
            variance_ = 1.0 / ((rate**2) * ((3.0 - 2.0 * q) ** 2) * (4.0 - 3.0 * q))
        elif q < 1.5:
            variance_ = INF
        else:
            variance_ = NAN

        return {"mean": mean_, "mode": mode_, "variance": variance_, "std": SQRT(variance_) if q < (4.0 / 3.0) else NAN}
