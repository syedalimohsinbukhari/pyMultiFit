"""Created on August 12 11:26:00 2026"""

from ..backend import BaseDistribution
from ... import INF, NAN, NAN_DICT, SQRT
from ...typing import ArrayLike, NDArray
from ..utilities_d import (
    q_exponential_cdf_,
    q_exponential_log_cdf_,
    q_exponential_log_pdf_,
    q_exponential_pdf_,
)


class QExponentialDistribution(BaseDistribution):
    r"""
    Class for :class:`~.QExponentialDistribution`.

    Parameters
    ----------
    amplitude : float, default=1.0
        The amplitude or scaling factor of the distribution. Defaults to 1.0.
    q : float, default=1.0
        The entropic index parameter, :math:`q`. Must satisfy :math:`q < 2`.
        When :math:`q \to 1`, the distribution reduces to the standard exponential distribution.
    rate : float, default=1.0
        The rate parameter, :math:`\lambda`. Defaults to 1.0. Must be strictly positive (:math:`\lambda > 0`).
    loc : float, default=0.0
        The location parameter, :math:`\mu`. Defaults to 0.0.
    normalize : bool, default=False
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a heavy-tailed q-exponential distribution (:math:`q=1.5, \mu=0, \lambda=1.0`)
    with ``pyMultiFit``:

    .. literalinclude:: ../../../examples/basic/q_exponential_d.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/q_exponential_d.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/q_exponential_example1.png
       :alt: QExponential(q=1.5, loc=0, rate=1.0)
       :align: center
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        q: float = 1.0,
        rate: float = 1.0,
        loc: float = 0.0,
        normalize: bool = False,
    ):
        if q >= 2.0:
            raise ValueError(f"Entropic index q must be < 2, got {q}")
        if rate <= 0:
            raise ValueError(f"Rate must be > 0, got {rate}")

        self.amplitude = float(amplitude)
        self.q = float(q)
        self.rate = float(rate)
        self.loc = float(loc)
        self.normalize = bool(normalize)

    def logpdf(self, x: ArrayLike) -> NDArray:
        return q_exponential_log_pdf_(
            x,
            amplitude=self.amplitude,
            q=self.q,
            rate=self.rate,
            loc=self.loc,
            normalize=self.normalize,
        )

    def pdf(self, x: ArrayLike) -> NDArray:
        return q_exponential_pdf_(
            x,
            amplitude=self.amplitude,
            q=self.q,
            rate=self.rate,
            loc=self.loc,
            normalize=self.normalize,
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return q_exponential_cdf_(
            x,
            amplitude=self.amplitude,
            q=self.q,
            rate=self.rate,
            loc=self.loc,
            normalize=self.normalize,
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return q_exponential_log_cdf_(
            x,
            amplitude=self.amplitude,
            q=self.q,
            rate=self.rate,
            loc=self.loc,
            normalize=self.normalize,
        )

    def stats(self) -> dict[str, float]:
        q, rate, loc = self.q, self.rate, self.loc

        if q >= 2.0 or rate <= 0:
            return NAN_DICT

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

        return {
            "mean": mean_,
            "mode": mode_,
            "variance": variance_,
            "std": SQRT(variance_) if q < (4.0 / 3.0) else NAN,
        }