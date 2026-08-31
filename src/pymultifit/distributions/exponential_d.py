"""Created on Nov 30 10:49:49 2024"""

from __future__ import annotations

from .. import LOG_TWO, NAN_DICT, SQRT, _md_scipy_like
from ..typing import ArrayLike, NDArray
from .backend import BaseDistribution
from .utilities_d import exponential_cdf_, exponential_log_cdf_, exponential_log_pdf_, exponential_pdf_


class ExponentialDistribution(BaseDistribution):
    r"""
    Class for :class:`~.ExponentialDistribution`.

    .. note::
        The :class:`~.ExponentialDistribution` is a special case of
        the :class:`~pymultifit.distributions.gamma_d.GammaDistribution`,

        * :math:`\alpha_\text{gammaSR} = 1`,
        * :math:`\lambda_\text{gammaSR} = \lambda_\text{expon}`.

    Parameters
    ----------
    amplitude :
        The amplitude of the PDF, defaults to 1.0. Ignored if ``normalize`` is ``True``.
    scale :
        The scale parameter, :math:`\lambda`. Defaults to 1.0.
    loc :
        The location parameter, for shifting. Defaults to 0.0.
    normalize :
        If ``True``, the distribution is normalized so that the total area under the PDF equals 1.
        Defaults to ``False``.

    Examples
    --------
    Importing libraries:

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :linenos:
       :lineno-start: 3
       :lines: 3-7

    Generating a standard Exponential(:math:`\lambda = 1.5`) distribution with ``pyMultiFit`` and ``scipy``:

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :linenos:
       :lineno-start: 9
       :lines: 9-12

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :linenos:
       :lineno-start: 14
       :lines: 14-29

    .. image:: ../../../images/expon_example1.png
       :alt: Expon(1.5)
       :align: center

    Generating a translated Exponential(:math:`\lambda = 1.5`) distribution with :math:`\text{loc} = 3`:

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :lineno-start: 32
       :lines: 32

    Plotting **PDF** and **CDF**:

    .. literalinclude:: ../../../examples/basic/expon.py
       :language: python
       :lineno-start: 34
       :lines: 34-49

    .. image:: ../../../images/expon_example2.png
       :alt: Expon(1.5, 3)
       :align: center
    """

    def __init__(self, amplitude: float = 1.0, scale: float = 1.0, loc: float = 0.0, normalize: bool = False):
        self.amplitude = 1 if normalize else amplitude
        self.scale = scale
        self.loc = loc

        self.norm = normalize

    @classmethod
    @_md_scipy_like("1.0.7")
    def scipy_like(cls, loc: float = 0.0, scale: float = 1.0) -> "ExponentialDistribution":
        r"""
        Instantiate :class:`~.ExponentialDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The rate parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.ExponentialDistribution`
            An instance of normalized :class:`~.ExponentialDistribution`.
        """
        return cls(loc=loc, scale=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, loc: float = 0.0, scale: float = 1.0) -> "ExponentialDistribution":
        r"""
        Instantiate :class:`~.ExponentialDistribution` with ``scipy`` parameterization.

        Parameters
        ----------
        loc :
            The location parameter. Defaults to 0.0.
        scale :
            The rate parameter. Defaults to 1.0.

        Returns
        -------
        :class:`~.ExponentialDistribution`
            An instance of normalized :class:`~.ExponentialDistribution`.
        """
        return cls(loc=loc, scale=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return exponential_pdf_(x, amplitude=self.amplitude, lambda_=self.scale, loc=self.loc, normalize=self.norm)

    def logpdf(self, x: ArrayLike) -> NDArray:
        return exponential_log_pdf_(x, amplitude=self.amplitude, lambda_=self.scale, loc=self.loc, normalize=self.norm)

    def cdf(self, x: ArrayLike) -> NDArray:
        return exponential_cdf_(x, amplitude=self.amplitude, lambda_=self.scale, loc=self.loc, normalize=self.norm)

    def logcdf(self, x: ArrayLike) -> NDArray:
        return exponential_log_cdf_(x, amplitude=self.amplitude, lambda_=self.scale, loc=self.loc, normalize=self.norm)

    def stats(self) -> dict[str, float]:
        s, l_ = self.scale, self.loc

        if s <= 0:
            return NAN_DICT

        mean_ = (1 / s) + l_
        median_ = (LOG_TWO / s) + l_
        variance_ = 1 / s**2

        return {"mean": mean_, "median": median_, "variance": variance_, "std": SQRT(variance_)}
