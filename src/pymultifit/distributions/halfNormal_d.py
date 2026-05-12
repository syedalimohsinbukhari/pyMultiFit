"""Created on Dec 04 03:57:18 2024"""

from __future__ import annotations

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import half_normal_cdf_, half_normal_log_cdf_, half_normal_log_pdf_, half_normal_pdf_
from .. import SQRT, SQRT_TWO_BY_PI, TWO_BY_PI, md_scipy_like
from ..typing import ArrayLike, NDArray


class HalfNormalDistribution(BaseDistribution):
    r"""
    Class for halfnormal distribution.

    Parameters
    ----------
    amplitude :
        The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    scale :
        The standard deviation parameter, :math:`\sigma`. Defaults to 1.0.
    loc :
        The location parameter, for shifting. Defaults to 0.0.
    normalize :
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Raises
    ------
    NegativeAmplitudeError
        If the provided value of amplitude is negative.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/halfnorm.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Half Normal(:math:`\sigma = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/halfnorm.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/halfnorm.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/half_normal_example1.png
       :alt: HN(1)
       :align: center

    Generating a translated Gaussian(:math:`\sigma=3`) distribution with :math:`\text{loc}=3`:

    .. literalinclude:: ../../../examples/basic/halfnorm.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/halfnorm.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/half_normal_example2.png
       :alt: HN(2, 3)
       :align: center
    """

    def __init__(self, amplitude: float = 1.0, scale: float = 1.0, loc: float = 0.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()

        self.amplitude = 1 if normalize else amplitude
        self.scale = scale
        self.loc = loc

        self.norm = normalize

    @classmethod
    @md_scipy_like("1.0.7")
    def scipy_like(cls, loc: float = 0.0, scale: float = 1.0) -> "HalfNormalDistribution":
        """
        Instantiate HalfNormalDistribution with scipy parametrization.

        Parameters
        ----------
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        HalfNormalDistribution
            An instance of normalized HalfNormalDistribution.
        """
        return cls(loc=loc, scale=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, loc: float = 0.0, scale: float = 1.0) -> "HalfNormalDistribution":
        """
        Instantiate HalfNormalDistribution with scipy parametrization.

        Parameters
        ----------
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        HalfNormalDistribution
            An instance of normalized HalfNormalDistribution.
        """
        return cls(loc=loc, scale=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return half_normal_pdf_(x, amplitude=self.amplitude, sigma=self.scale, loc=self.loc, normalize=self.norm)

    def logpdf(self, x: ArrayLike) -> NDArray:
        return half_normal_log_pdf_(x, amplitude=self.amplitude, sigma=self.scale, loc=self.loc, normalize=self.norm)

    def cdf(self, x: ArrayLike) -> NDArray:
        return half_normal_cdf_(x, amplitude=self.amplitude, sigma=self.scale, loc=self.loc, normalize=self.norm)

    def logcdf(self, x: ArrayLike) -> NDArray:
        return half_normal_log_cdf_(x, amplitude=self.amplitude, sigma=self.scale, loc=self.loc, normalize=self.norm)

    def stats(self) -> dict[str, float]:
        s_, l_ = self.scale, self.loc

        mean_ = SQRT_TWO_BY_PI
        mode_ = 0

        variance_ = 1 - TWO_BY_PI
        variance_ *= s_ ** 2

        return {"mean": (s_ * mean_) + l_, "mode": mode_, "variance": variance_, "std": SQRT(variance_)}
