"""Created on Aug 03 21:35:28 2024"""

from __future__ import annotations

from numpy import sign

from .. import EXP, LOG, NAN_DICT, PI, SQRT, SQRT_TWO_BY_PI, TWO_BY_PI, TWO_PI, _md_scipy_like
from ..typing import ArrayLike, NDArray
from .backend import BaseDistribution
from .backend import errorHandling as erH
from .utilities_d import skew_normal_cdf_, skew_normal_log_pdf_, skew_normal_pdf_


class SkewNormalDistribution(BaseDistribution):
    r"""
    Class for :class:`~.SkewNormalDistribution`.

    Parameters
    ----------
    amplitude :
        The amplitude of the PDF. Defaults to 1.0. Ignored if ``normalize`` is ``True``.
    shape :
        The mean parameter, :math:`\mu`. Defaults to 0.0.
    scale :
        The scale parameter, for scaling. Defaults to 1.0.
    location :
        The location parameter, for shifting. Defaults to 0.0.
    normalize :
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/skewnormal.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Skew Normal(:math:`\xi=1, \mu = 0, \sigma = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/skewnormal.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/skewnormal.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/skew_norm_example1.png
       :alt: SkewNormal(1, 0, 1)
       :align: center

    Generating a translated Skew Normal(:math:`\xi=3, \mu=-3, \sigma=3`) distribution:

    .. literalinclude:: ../../../examples/basic/skewnormal.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/skewnormal.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/skew_norm_example2.png
       :alt: Skew Normal(3, -3, 3)
       :align: center
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        shape: float = 1.0,
        location: float = 0.0,
        scale: float = 1.0,
        normalize: bool = False,
    ):
        self.amplitude = 1 if normalize else amplitude
        self.shape = shape
        self.location = location
        self.scale = scale

        self.norm = normalize

    @classmethod
    @_md_scipy_like("1.0.7")
    def scipy_like(cls, a: float, loc: float = 0.0, scale: float = 1.0) -> "SkewNormalDistribution":
        r"""
        Instantiate :class:`~.SkewNormalDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        a :
            The skewness parameter.
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.SkewNormalDistribution`
            An instance of normalized :class:`~.SkewNormalDistribution`.
        """
        return cls(shape=a, location=loc, scale=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, a: float, loc: float = 0.0, scale: float = 1.0) -> "SkewNormalDistribution":
        r"""
        Instantiate :class:`~.SkewNormalDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        a :
            The skewness parameter.
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.SkewNormalDistribution`
            An instance of normalized :class:`~.SkewNormalDistribution`.
        """
        return cls(shape=a, location=loc, scale=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return skew_normal_pdf_(
            x, amplitude=self.amplitude, shape=self.shape, loc=self.location, scale=self.scale, normalize=self.norm
        )

    def logpdf(self, x: ArrayLike) -> NDArray:
        return skew_normal_log_pdf_(
            x, amplitude=self.amplitude, shape=self.shape, loc=self.location, scale=self.scale, normalize=self.norm
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return skew_normal_cdf_(
            x, amplitude=self.amplitude, shape=self.shape, loc=self.location, scale=self.scale, normalize=self.norm
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return LOG(self.cdf(x))

    def stats(self) -> dict[str, float]:
        alpha, omega, epsilon = self.shape, self.scale, self.location

        if omega <= 0:
            return NAN_DICT

        delta = alpha / SQRT(1 + alpha**2)
        sqrt_2_pi_delta = omega * SQRT_TWO_BY_PI * delta

        def _m0(alpha_):
            term2 = (1 - PI / 4) * sqrt_2_pi_delta**3 / (1 - TWO_BY_PI * delta**2)
            term3 = (TWO_PI / abs(alpha_)) * EXP(-TWO_PI / abs(alpha_)) * sign(alpha_)
            return sqrt_2_pi_delta - term2 - term3

        # Calculating mean, mode, variance, and std
        mean_ = epsilon + sqrt_2_pi_delta
        mode_ = epsilon + omega * _m0(alpha)
        variance_ = omega**2 * (1 - (2 * delta**2 / PI))
        std_ = SQRT(variance_)

        return {"mean": mean_, "mode": mode_, "variance": variance_, "std": std_}
