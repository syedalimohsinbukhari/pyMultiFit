"""Created on Aug 03 20:07:50 2024"""

from __future__ import annotations

from .backend import BaseDistribution
from .utilities_d import gaussian_cdf_, gaussian_log_cdf_, gaussian_log_pdf_, gaussian_pdf_
from .. import NAN_DICT, _md_scipy_like
from ..typing import ArrayLike, NDArray


class GaussianDistribution(BaseDistribution):
    r"""
    Class for :class:`~.GaussianDistribution`.

    Parameters
    ----------
    amplitude :
        The amplitude of the PDF. Defaults to 1.0. Ignored if ``normalize`` is ``True``.
    mu :
        The mean parameter, :math:`\mu`. Defaults to 0.0.
    std :
        The standard deviation parameter, :math:`\sigma`. Defaults to 1.0.
    normalize :
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

    Generating a standard Gaussian(:math:`\mu=0, \sigma = 1`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/gaussian_example1.png
       :alt: Gaussian(0, 1)
       :align: center

    Generating a translated Gaussian(:math:`\mu=3, \sigma=2`) distribution:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/gaussian.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/gaussian_example2.png
       :alt: Gaussian(3, 2)
       :align: center
    """

    def __init__(self, amplitude: float = 1.0, mu: float = 0.0, std: float = 1.0, normalize: bool = False):
        self.amplitude = 1.0 if normalize else amplitude
        self.mu = mu
        self.std_ = std
        self.norm = normalize

    @classmethod
    @_md_scipy_like("1.0.7")
    def scipy_like(cls, loc: float = 0.0, scale: float = 1.0) -> "GaussianDistribution":
        r"""
        Instantiate :class:`~.GaussianDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        loc :
            The mean parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.GaussianDistribution`
            An instance of normalized :class:`~.GaussianDistribution`.
        """
        return cls(mu=loc, std=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, loc: float = 0.0, scale: float = 1.0) -> "GaussianDistribution":
        r"""
        Instantiate :class:`~.GaussianDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        loc :
            The mean parameter. Defaults to 0.0.
        scale :
            The scale parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.GaussianDistribution`
            An instance of normalized :class:`~.GaussianDistribution`.
        """
        return cls(mu=loc, std=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return gaussian_pdf_(x, amplitude=self.amplitude, mean=self.mu, std=self.std_, normalize=self.norm)

    def logpdf(self, x: ArrayLike) -> NDArray:
        return gaussian_log_pdf_(x, amplitude=self.amplitude, mean=self.mu, std=self.std_, normalize=self.norm)

    def cdf(self, x: ArrayLike) -> NDArray:
        return gaussian_cdf_(x, amplitude=self.amplitude, mean=self.mu, std=self.std_, normalize=self.norm)

    def logcdf(self, x: ArrayLike) -> NDArray:
        return gaussian_log_cdf_(x, amplitude=self.amplitude, mean=self.mu, std=self.std_, normalize=self.norm)

    def stats(self) -> dict[str, float]:
        m, s = self.mu, self.std_

        if s <= 0:
            return NAN_DICT

        return {"mean": m, "median": m, "mode": m, "variance": s**2, "std": s}