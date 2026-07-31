"""Created on Dec 11 20:40:15 2024"""

from __future__ import annotations

from .backend import BaseDistribution
from .utilities_d import uniform_cdf_, uniform_log_cdf_, uniform_log_pdf_, uniform_pdf_
from .. import NAN_DICT, SQRT, _md_scipy_like
from ..typing import ArrayLike, NDArray


class UniformDistribution(BaseDistribution):
    r"""
    Class for :class:`~.UniformDistribution`.

    Parameters
    ----------
    amplitude :
        The amplitude of the PDF. Defaults to 1.0. Ignored if ``normalize`` is ``True``.
    low :
        Lower bound of distribution. Defaults to 0.0.
    high :
        Upper bound of distribution. Defaults to 1.0.
    normalize :
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/uniform.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Uniform(0, 1) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/uniform.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/uniform.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/uniform_example1.png
       :alt: Uniform(0, 1)
       :align: center

    Generating a translated Uniform(3, 5) distribution:

    .. literalinclude:: ../../../examples/basic/uniform.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/uniform.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/uniform_example2.png
       :alt: Uniform(3, 5)
       :align: center
    """

    def __init__(self, amplitude: float = 1.0, low: float = 0.0, high: float = 1.0, normalize: bool = False):
        self.amplitude = 1 if normalize else amplitude
        self.low = low
        self.high = high

        self.norm = normalize

    @classmethod
    @_md_scipy_like("1.0.7")
    def scipy_like(cls, loc: float = 0.0, scale: float = 1.0) -> "UniformDistribution":
        r"""
        Instantiate :class:`~.UniformDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.UniformDistribution`
            An instance of normalized :class:`~.UniformDistribution`.
        """
        return cls(low=loc, high=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, loc: float = 0.0, scale: float = 1.0) -> "UniformDistribution":
        r"""
        Instantiate :class:`~.UniformDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.UniformDistribution`
            An instance of normalized :class:`~.UniformDistribution`.
        """
        return cls(low=loc, high=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return uniform_pdf_(x, amplitude=self.amplitude, low=self.low, high=self.high, normalize=self.norm)

    def logpdf(self, x: ArrayLike) -> NDArray:
        return uniform_log_pdf_(x, amplitude=self.amplitude, low=self.low, high=self.high, normalize=self.norm)

    def cdf(self, x: ArrayLike) -> NDArray:
        return uniform_cdf_(x, amplitude=self.amplitude, low=self.low, high=self.high, normalize=self.norm)

    def logcdf(self, x: ArrayLike) -> NDArray:
        return uniform_log_cdf_(x, amplitude=self.amplitude, low=self.low, high=self.high, normalize=self.norm)

    def stats(self) -> dict[str, float]:
        low, high = self.low, self.low + self.high

        if low >= high:
            return NAN_DICT

        mean_ = 0.5 * (low + high)
        median_ = mean_
        variance_ = (1 / 12.0) * (high - low) ** 2

        return {"mean": mean_, "median": median_, "variance": variance_, "std": SQRT(variance_)}